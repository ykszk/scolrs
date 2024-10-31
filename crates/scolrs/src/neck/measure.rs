use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};

use crate::neck::cli::MeasureArgs;
use anyhow::{Context, Result};
use indexmap::IndexMap;
use labelme_rs::{LabelMeData, LabelMeDataLine};
use log::warn;
use scolrs::{
    head_neck::{LateralPoints, NeckLateralMeasure, NeckMeasureComponent},
    parse_measures,
};
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
                results.insert(measure.id(), result);
            }
            Err(e) => match e {
                scolrs::MeasureError::InvalidNumberOfPoints(err) => {
                    warn!("Skip point count error for {}: {:?}", measure.id(), err);
                }
                e => return Err(e.into()),
            },
        }
    }
    Ok(results)
}

fn process_data<'a>(
    data: &'a LabelMeData,
    measures: &'a [NeckLateralMeasure],
) -> Result<IndexMap<&'a str, Vec<f32>>> {
    let lateral_points = LateralPoints::try_from(data)?;
    let measures: Vec<Box<dyn NeckMeasureComponent>> = measures
        .iter()
        .map(|m| (m, &lateral_points).into())
        .collect();
    measure_all(measures)
}

fn process_json(args: MeasureArgs) -> Result<()> {
    let data = LabelMeData::try_from(args.input.as_path())
        .with_context(|| format!("Loading {:?}", args.input))?;
    let measures = handle_measures_arg(&args.measures)?;
    let results = process_data(&data, &measures)?;
    if let Some(output) = args.output {
        std::fs::write(output, serde_json::to_string_pretty(&results)?)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&results)?);
    }
    Ok(())
}

fn handle_measures_arg(measure_strs: &[String]) -> Result<Vec<NeckLateralMeasure>, anyhow::Error> {
    let measures: Vec<NeckLateralMeasure> = if measure_strs.is_empty() {
        NeckLateralMeasure::all()
    } else {
        parse_measures(measure_strs).map_err(|e| anyhow::anyhow!(e))?
    };
    Ok(measures)
}

fn process_ndjson(args: MeasureArgs) -> Result<()> {
    let measures = handle_measures_arg(&args.measures)?;

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
        let results = process_data(&data.content, &measures);
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
            ..Default::default()
        };
        cmd(args)
    }

    #[test]
    fn test_measure_cmd_case2() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");

        let input = data_dir.join("neck_case2/extension_lateral.json");
        let args = MeasureArgs {
            input,
            ..Default::default()
        };
        cmd(args)?;

        let input = data_dir.join("neck_case2/flexion_lateral.json");
        let args = MeasureArgs {
            input,
            ..Default::default()
        };
        cmd(args)
    }
}
