use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
};

use crate::neck::cli::MeasureArgs;
use anyhow::{Context, Result};
use indexmap::IndexMap;
use log::warn;
use scolrs::{draw::MeasureError, TryFromJson};
use scolrs::{
    head_neck::{LateralPoints, LateralPointsIRLine, NeckLateralMeasure, NeckMeasureComponent},
    Scalable,
};
use serde::{Deserialize, Serialize};

type MeasurementVec = std::result::Result<Vec<f64>, MeasureError>;

#[derive(Serialize, Deserialize)]
pub struct Measurements {
    pub measurements: IndexMap<String, MeasurementVec>,
    pub unit_of_length: String,
}

#[derive(Serialize, Deserialize)]
pub struct MeasurementsLine {
    pub filename: String,
    pub content: Measurements,
}

fn measure_all(
    measures: Vec<Box<dyn NeckMeasureComponent + '_>>,
) -> Result<IndexMap<String, MeasurementVec>> {
    let measures: Vec<_> = measures
        .into_iter()
        .map(|m| {
            let result = m.measure();
            (m.id().to_string(), result)
        })
        .collect();
    let map = IndexMap::from_iter(measures);
    Ok(map)
}

fn process_data(
    lateral_points: LateralPoints,
    measures: &[NeckLateralMeasure],
) -> Result<Measurements> {
    let mut lateral_points = lateral_points;
    if lateral_points.image_metadata.spacing_xy.0 != 1.0
        || lateral_points.image_metadata.spacing_xy.1 != 1.0
    {
        lateral_points.scale()?;
    }
    let measures: Vec<Box<dyn NeckMeasureComponent>> = measures
        .iter()
        .map(|m| (m, &lateral_points).into())
        .collect();
    let measures = measure_all(measures)?;
    Ok(Measurements {
        measurements: measures,
        unit_of_length: lateral_points.image_metadata.unit.clone(),
    })
}

fn process_json(args: MeasureArgs) -> Result<()> {
    let data = LateralPoints::try_from_native_json(
        std::fs::read_to_string(&args.input)
            .with_context(|| format!("Loading {:?}", args.input))?
            .as_str(),
    )?;
    let measures = args.measures.unwrap_or_else(NeckLateralMeasure::all);
    let results = process_data(data, &measures)?;
    if let Some(output) = args.output {
        std::fs::write(output, serde_json::to_string_pretty(&results)?)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&results)?);
    }
    Ok(())
}

fn process_ndjson(args: MeasureArgs) -> Result<()> {
    let measures = args.measures.unwrap_or_else(NeckLateralMeasure::all);

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
        let mut lateral_points =
            LateralPoints::try_from_native_json(std::fs::read_to_string(&input)?.as_str())?;
        lateral_points.image_metadata.spacing_xy = (0.5, 0.5);
        let mut non_scaled = lateral_points.clone();
        non_scaled.image_metadata.spacing_xy = (1.0, 1.0);
        let measures = NeckLateralMeasure::all();
        let mut measurements = process_data(lateral_points, &measures)?.measurements;
        let mut measurements_non_scaled = process_data(non_scaled, &measures)?.measurements;

        for distance_measure in ["Adi", "Sacs", "ModifiedRenawatIndex"] {
            let distance_measurements =
                measurements.swap_remove(distance_measure).unwrap().unwrap();
            let distance_measurements_non_scaled = measurements_non_scaled
                .swap_remove(distance_measure)
                .unwrap()
                .unwrap();

            for (scaled, non_scaled) in distance_measurements
                .iter()
                .zip(distance_measurements_non_scaled.iter())
            {
                assert!(
                    float_cmp::approx_eq!(f64, *scaled, non_scaled * 0.5, epsilon = 1e-6),
                    "Scaled measurements for {} are not correct: {} != {}",
                    distance_measure,
                    scaled,
                    non_scaled * 0.5
                );
            }
        }

        Ok(())
    }
}
