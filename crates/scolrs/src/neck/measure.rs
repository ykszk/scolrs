use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};

use crate::neck::cli::MeasureArgs;
use anyhow::{Context, Result};
use indexmap::IndexMap;
use log::warn;
use scolrs::{
    head_neck::{
        LateralPoints, LateralPointsIR, LateralPointsIRLine, NeckLateralMeasure,
        NeckMeasureComponent,
    },
    parse_measures,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Measurements {
    pub measurements: IndexMap<String, Vec<f32>>,
    pub unit: String,
}

#[derive(Serialize, Deserialize)]
pub struct MeasurementsLine {
    pub filename: String,
    pub content: Measurements,
}

fn measure_all(
    measures: Vec<Box<dyn NeckMeasureComponent + '_>>,
) -> Result<IndexMap<String, Vec<f32>>> {
    let mut results: IndexMap<String, Vec<f32>> = Default::default();
    for measure in measures {
        let result = measure.measure();
        match result {
            Ok(result) => {
                results.insert(measure.id().to_string(), result);
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

fn process_data(
    lateral_points: LateralPoints,
    measures: &[NeckLateralMeasure],
) -> Result<Measurements> {
    let mut lateral_points = lateral_points;
    if lateral_points.image_data.spacing_xy.0 != 1.0
        || lateral_points.image_data.spacing_xy.1 != 1.0
    {
        lateral_points.scale();
    }
    let measures: Vec<Box<dyn NeckMeasureComponent>> = measures
        .iter()
        .map(|m| (m, &lateral_points).into())
        .collect();
    let measures = measure_all(measures)?;
    Ok(Measurements {
        measurements: measures,
        unit: lateral_points.image_data.unit.clone(),
    })
}

fn process_json(args: MeasureArgs) -> Result<()> {
    let data_ir = LateralPointsIR::try_from(std::fs::read_to_string(&args.input)?.as_str())
        .with_context(|| format!("Loading {:?}", args.input))?;
    let data = LateralPoints::try_from(&data_ir)?;
    let measures = handle_measures_arg(&args.measures)?;
    let results = process_data(data, &measures)?;
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
        let data_line: LateralPointsIRLine = serde_json::from_str(&line)?;
        let data = LateralPoints::try_from(&data_line.content)?;
        let results = process_data(data, &measures);
        match results {
            Ok(results) => {
                let results_line = MeasurementsLine {
                    filename: data_line.filename,
                    content: results,
                };
                writeln!(writer, "{}", serde_json::to_string(&results_line)?)?;
            }
            Err(e) => {
                warn!("Skip {:?}: {:?}", data_line.filename, e);
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
    use pretty_assertions::assert_eq;
    use std::path::PathBuf;

    #[test]
    fn test_measure_cmd_case1() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");

        let input = data_dir.join("neck_case1/lateral_lateral_points.json");
        let args = MeasureArgs {
            input,
            ..Default::default()
        };
        cmd(args)
    }

    #[test]
    fn test_measure_cmd_case2() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");

        let input = data_dir.join("neck_case2/extension_lateral_lateral_points.json");
        let args = MeasureArgs {
            input,
            ..Default::default()
        };
        cmd(args)?;

        let input = data_dir.join("neck_case2/flexion_lateral_lateral_points.json");
        let args = MeasureArgs {
            input,
            ..Default::default()
        };
        cmd(args)
    }

    #[test]
    fn test_scaled_measurements() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");

        let input = data_dir.join("neck_case1/lateral_lateral_points.json");
        let mut lateral_points = LateralPoints::try_from(&LateralPointsIR::try_from(
            std::fs::read_to_string(&input)?.as_str(),
        )?)?;
        lateral_points.image_data.spacing_xy = (0.5, 0.5);
        let mut non_scaled = lateral_points.clone();
        non_scaled.image_data.spacing_xy = (1.0, 1.0);
        let measures = NeckLateralMeasure::all();
        let measurements = process_data(lateral_points, &measures)?.measurements;
        let measurements_non_scaled = process_data(non_scaled, &measures)?.measurements;

        for distance_measure in ["Adi", "Sacs", "ModifiedRenawatIndex"] {
            let distance_measurements = measurements.get(distance_measure).unwrap();
            let distance_measurements_non_scaled =
                measurements_non_scaled.get(distance_measure).unwrap();

            for (scaled, non_scaled) in distance_measurements
                .iter()
                .zip(distance_measurements_non_scaled.iter())
            {
                assert_eq!(
                    scaled,
                    &(non_scaled * 0.5),
                    "Scaled measurements for {}",
                    distance_measure
                );
            }
        }

        Ok(())
    }
}
