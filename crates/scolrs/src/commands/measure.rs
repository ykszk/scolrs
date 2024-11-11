use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::cli::{MeasureArgs, Plane};
use anyhow::{Context, Result};
use indexmap::IndexMap;
use labelme_rs::{serde_json, LabelMeData, LabelMeDataLine};
use log::debug;
use scolrs::{
    parse_measures, CoronalComponent, CoronalMeasure, CoronalPoints, CoronalPointsIR,
    CoronalPointsIRLine, MeasureError, SagittalComponent, SagittalMeasure, SagittalPoints,
    SagittalPointsIR, SagittalPointsIRLine, ScolDesc, ScolError,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

type MeasureResult = std::result::Result<f64, MeasureError>;

#[derive(Serialize, Deserialize)]
pub struct MeasureLine<V>
where
    V: std::hash::Hash + Eq + std::cmp::Ord,
{
    pub filename: String,
    pub content: IndexMap<V, MeasureResult>,
}

type SagittalMeasureLine = MeasureLine<SagittalMeasure>;
type CoronalMeasureLine = MeasureLine<CoronalMeasure>;

fn process_ndjson(args: MeasureArgs) -> Result<()> {
    let reader: Box<dyn BufRead> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::new(std::io::stdin()))
    } else {
        Box::new(BufReader::new(File::open(&args.input)?))
    };
    for line in reader.lines() {
        match &args.direction {
            Plane::Coronal => {
                let data_line: CoronalPointsIRLine =
                    load_labelme_or_native::<CoronalPointsIRLine, LabelMeDataLine>(&args, &line?)?;
                let results = measure_coronal(data_line.content.try_into()?, &args)?;
                let line = CoronalMeasureLine {
                    filename: data_line.filename,
                    content: results,
                };
                println!("{}", serde_json::to_string(&line)?);
            }
            Plane::Sagittal => {
                let data_line: SagittalPointsIRLine =
                    load_labelme_or_native::<SagittalPointsIRLine, LabelMeDataLine>(&args, &line?)?;
                let results = measure_sagittal(data_line.content.try_into()?, &args)?;
                let line = SagittalMeasureLine {
                    filename: data_line.filename,
                    content: results,
                };
                println!("{}", serde_json::to_string(&line)?);
            }
        }
    }
    Ok(())
}

/// Return IR from native or LabelMe data
///
/// if `args.labelme` is true, load LM then convert to IR
/// otherwise, load IR directly
fn load_labelme_or_native<IR, LM>(args: &MeasureArgs, json_str: &str) -> Result<IR>
where
    IR: DeserializeOwned,
    IR: TryFrom<LM, Error = ScolError>,
    LM: for<'a> TryFrom<&'a str, Error = serde_json::Error>,
{
    if args.labelme {
        let data: LM = json_str
            .try_into()
            .with_context(|| format!("Load LabelMeData from {:?}", &args.input))?;
        let data = IR::try_from(data)?;
        Ok(data)
    } else {
        let data: IR = serde_json::from_str(json_str)?;
        Ok(data)
    }
}

fn process_json(args: MeasureArgs) -> Result<()> {
    debug!("Loading {:?}", args.input);
    let data_str = std::fs::read_to_string(&args.input)?;

    match args.direction {
        Plane::Coronal => {
            let data = load_labelme_or_native::<CoronalPointsIR, LabelMeData>(&args, &data_str)?;
            let results = measure_coronal(data.try_into()?, &args)?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        Plane::Sagittal => {
            let data = load_labelme_or_native::<SagittalPointsIR, LabelMeData>(&args, &data_str)?;
            let results = measure_sagittal(data.try_into()?, &args)?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
    };

    Ok(())
}

fn measure_sagittal(
    sagittal_points: SagittalPoints,
    args: &MeasureArgs,
) -> Result<IndexMap<SagittalMeasure, MeasureResult>, anyhow::Error> {
    let measures: Vec<SagittalMeasure> = if args.measures.is_empty() {
        SagittalMeasure::all_measures()
    } else {
        parse_measures(&args.measures).map_err(|e| anyhow::anyhow!(e))?
    };
    let mut results: IndexMap<SagittalMeasure, MeasureResult> = Default::default();
    for measure in measures {
        let spinal_measure: Box<dyn SagittalComponent> = (measure, &sagittal_points).into();
        results.insert(measure, spinal_measure.measure());
    }
    Ok(results)
}

fn measure_coronal(
    coronal_points: CoronalPoints,
    args: &MeasureArgs,
) -> Result<IndexMap<CoronalMeasure, MeasureResult>, anyhow::Error> {
    let measures: Vec<CoronalMeasure> = if args.measures.is_empty() {
        CoronalMeasure::all_measures()
    } else {
        parse_measures(&args.measures).map_err(|e| anyhow::anyhow!(e))?
    };
    let (curve_set, apex_set) = if let Some(curve_set) = args.curve_set.as_ref() {
        let reader = std::fs::File::open(curve_set)
            .with_context(|| format!("Load curve set {:?}", curve_set))?;
        let cs: ScolDesc = serde_json::from_reader(reader)?;
        (cs.curves, cs.apices)
    } else {
        let (cs, apexes, _major_curve) = coronal_points.identify_curves();
        (cs, apexes)
    };
    let mut results: IndexMap<CoronalMeasure, MeasureResult> = Default::default();
    for measure in measures {
        let spinal_measure: Box<dyn CoronalComponent> =
            (measure, &coronal_points, &curve_set, &apex_set).into();
        results.insert(measure, spinal_measure.measure());
    }
    Ok(results)
}

pub fn cmd(args: MeasureArgs) -> Result<()> {
    if args.input.as_os_str() == "-"
        || args.input.extension().unwrap_or_default() == "ndjson"
        || args.input.extension().unwrap_or_default() == "jsonl"
    {
        process_ndjson(args)
    } else {
        process_json(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data_directory() -> std::path::PathBuf {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../../tests/data");
        path
    }

    fn _test_case(case_dir: &str, labelme: bool) -> Result<()> {
        // Just test if the command runs without errors
        let data_dir = test_data_directory();
        let input = if labelme {
            data_dir.join(case_dir).join("lateral.json")
        } else {
            data_dir.join(case_dir).join("lateral_native.json")
        };
        let curve_set = None;
        let direction = Plane::Sagittal;
        let measures = Default::default();
        let args = MeasureArgs {
            input,
            curve_set,
            direction,
            measures,
            labelme,
        };
        cmd(args)?;

        // TODO: Test correct measurements

        let input = if labelme {
            data_dir.join(case_dir).join("frontal.json")
        } else {
            data_dir.join(case_dir).join("frontal_native.json")
        };
        let curve_set = None;
        let direction = Plane::Coronal;
        let measures = Default::default();
        let args = MeasureArgs {
            input,
            curve_set,
            direction,
            measures,
            labelme,
        };
        cmd(args)?;

        // Test if the command fails with invalid measures
        let input = if labelme {
            data_dir.join(case_dir).join("lateral.json")
        } else {
            data_dir.join(case_dir).join("lateral_native.json")
        };
        let curve_set = None;
        let direction = Plane::Sagittal;
        let measures = vec!["CobbPT".to_string()];
        let args = MeasureArgs {
            input,
            curve_set,
            direction,
            measures,
            labelme,
        };
        assert!(cmd(args).is_err());

        let input = if labelme {
            data_dir.join(case_dir).join("frontal.json")
        } else {
            data_dir.join(case_dir).join("frontal_coronal_points.json")
        };
        let curve_set = None;
        let direction = Plane::Coronal;
        let measures = vec!["ThoracicKyphosis".to_string()];
        let args = MeasureArgs {
            input,
            curve_set,
            direction,
            measures,
            labelme,
        };
        assert!(cmd(args).is_err());
        Ok(())
    }

    #[test]
    fn test_case1() -> Result<()> {
        _test_case("case1", true)
    }
    #[test]
    fn test_case2() -> Result<()> {
        _test_case("case2", true)?;
        _test_case("case2", false)
    }
    #[test]
    fn test_case3() -> Result<()> {
        _test_case("case3", true)
    }
    #[test]
    fn test_case4() -> Result<()> {
        _test_case("case4", true)
    }
}
