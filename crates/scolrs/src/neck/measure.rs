use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};

use crate::neck::cli::MeasureArgs;
use anyhow::{Context, Result};
use indexmap::IndexMap;
use labelme_rs::{LabelMeData, LabelMeDataLine};
use log::warn;
use scolrs::head_neck::{self, LateralPoints, NeckMeasureComponent};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct MeasureLine<V>
where
    V: std::hash::Hash + Eq + std::cmp::Ord,
{
    pub filename: String,
    pub content: IndexMap<V, Vec<f32>>,
}

fn measure_all(
    measures: Vec<Box<dyn NeckMeasureComponent + '_>>,
) -> Result<IndexMap<&'static str, Vec<f32>>> {
    let mut results: IndexMap<&str, Vec<f32>> = Default::default();
    for measure in measures {
        let result = measure.measure();
        match result {
            Ok(result) => {
                results.insert(measure.name(), result);
            }
            Err(e) => match e {
                scolrs::MeasureError::InvalidNumberOfPoints(err) => {
                    warn!("Skip point count error for {}: {:?}", measure.name(), err);
                }
                e => return Err(e.into()),
            },
        }
    }
    Ok(results)
}

fn process_data(data: &LabelMeData) -> Result<IndexMap<&str, Vec<f32>>> {
    let lateral_points = LateralPoints::try_from(data)?;
    let measures: Vec<Box<dyn NeckMeasureComponent + '_>> = vec![
        Box::new(head_neck::Adi(&lateral_points)),
        Box::new(head_neck::WedgeAngle(&lateral_points)),
        Box::new(head_neck::Sacs(&lateral_points)),
        Box::new(head_neck::OC2(&lateral_points)),
    ];
    measure_all(measures)
}

fn process_json(args: MeasureArgs) -> Result<()> {
    let data = LabelMeData::try_from(args.input.as_path())
        .with_context(|| format!("Loading {:?}", args.input))?;
    let results = process_data(&data)?;
    if let Some(output) = args.output {
        std::fs::write(output, serde_json::to_string_pretty(&results)?)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&results)?);
    }
    Ok(())
}

fn process_ndjson(args: MeasureArgs) -> Result<()> {
    let reader: Box<dyn BufRead> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::new(std::io::stdin()))
    } else {
        Box::new(BufReader::new(File::open(&args.input)?))
    };
    let write: Box<dyn Write> = if let Some(output) = args.output {
        Box::new(BufWriter::new(File::create(output)?))
    } else {
        Box::new(BufWriter::new(std::io::stdout()))
    };
    let mut writer = std::io::BufWriter::new(write);

    for line in reader.lines() {
        let line = line?;
        let data: LabelMeDataLine = serde_json::from_str(&line)?;
        let results = process_data(&data.content);
        match results {
            Ok(results) => {
                let results_line = MeasureLine {
                    filename: data.filename,
                    content: results,
                };
                writeln!(writer, "{}", serde_json::to_string(&results_line)?)?;
            }
            Err(e) => {
                warn!("Skip {:?}: {:?}", data.filename, e);
            }
        }
    }
    Ok(())
}

pub fn cmd(args: MeasureArgs) -> Result<()> {
    if args.input.extension().unwrap_or_default() == "json" {
        process_json(args)
    } else if args.input.as_os_str() == "-"
        || args.input.extension().unwrap_or_default() == "ndjson"
    {
        process_ndjson(args)
    } else {
        Err(anyhow::anyhow!("Unsupported file format"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_measure_cmd_case1() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");

        let input = data_dir.join("neck_case1/lateral.json");
        let args = MeasureArgs {
            input,
            output: None,
            measures: vec![],
        };
        cmd(args)
    }

    #[test]
    fn test_measure_cmd_case2() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");

        let input = data_dir.join("neck_case2/extension_lateral.json");
        let args = MeasureArgs {
            input,
            output: None,
            measures: vec![],
        };
        cmd(args)?;

        let input = data_dir.join("neck_case2/flexion_lateral.json");
        let args = MeasureArgs {
            input,
            output: None,
            measures: vec![],
        };
        cmd(args)
    }
}
