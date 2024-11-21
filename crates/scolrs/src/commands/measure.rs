use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use crate::cli::{MeasureArgs, MeasureSubCommands, MeasureSubCoronalArgs, MeasureSubSagittallArgs};
use anyhow::{Context, Result};
use indexmap::IndexMap;
use labelme_rs::{serde_json, LabelMeData, LabelMeDataLine};
use log::debug;
use scolrs::{
    draw::{MeasureComponent, MeasureError},
    CoronalMeasure, CoronalPoints, CoronalPointsIR, CoronalPointsIRLine, CurveDesc, MeasureAndDraw,
    SagittalMeasure, SagittalPoints, SagittalPointsIR, SagittalPointsIRLine, ScolError,
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
        match args.subcommand.clone() {
            MeasureSubCommands::Coronal(subcommand) => {
                let data_line: CoronalPointsIRLine = load_labelme_or_native::<
                    CoronalPointsIRLine,
                    LabelMeDataLine,
                >(
                    args.labelme, &args.input, &line?
                )?;
                let results =
                    measure_coronal(data_line.content.try_into()?, subcommand, &args.curve_set)?;
                let line = CoronalMeasureLine {
                    filename: data_line.filename,
                    content: results,
                };
                println!("{}", serde_json::to_string(&line)?);
            }
            MeasureSubCommands::Sagittal(subcommand) => {
                let data_line: SagittalPointsIRLine = load_labelme_or_native::<
                    SagittalPointsIRLine,
                    LabelMeDataLine,
                >(
                    args.labelme, &args.input, &line?
                )?;
                let results = measure_sagittal(data_line.content.try_into()?, subcommand)?;
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
fn load_labelme_or_native<IR, LM>(labelme: bool, input: &Path, json_str: &str) -> Result<IR>
where
    IR: DeserializeOwned,
    IR: TryFrom<LM, Error = ScolError>,
    LM: for<'a> TryFrom<&'a str, Error = serde_json::Error>,
{
    if labelme {
        let data: LM = json_str
            .try_into()
            .with_context(|| format!("Load LabelMeData from {:?}", input))?;
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

    match args.subcommand {
        MeasureSubCommands::Coronal(subcommand) => {
            let data = load_labelme_or_native::<CoronalPointsIR, LabelMeData>(
                args.labelme,
                &args.input,
                &data_str,
            )?;
            let results = measure_coronal(data.try_into()?, subcommand, &args.curve_set)?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        MeasureSubCommands::Sagittal(subcommand) => {
            let data = load_labelme_or_native::<SagittalPointsIR, LabelMeData>(
                args.labelme,
                &args.input,
                &data_str,
            )?;
            let results = measure_sagittal(data.try_into()?, subcommand)?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
    };

    Ok(())
}

fn measure_sagittal(
    sagittal_points: SagittalPoints,
    subcommand: MeasureSubSagittallArgs,
) -> Result<IndexMap<SagittalMeasure, MeasureResult>, anyhow::Error> {
    let measures = subcommand
        .measures
        .unwrap_or_else(SagittalMeasure::all_measures);
    let mut results: IndexMap<SagittalMeasure, MeasureResult> = Default::default();
    for measure in measures {
        let spinal_measure: Box<dyn MeasureComponent> = (&measure, &sagittal_points).into();
        results.insert(measure, spinal_measure.measure());
    }
    Ok(results)
}

fn measure_coronal(
    coronal_points: CoronalPoints,
    subcommand: MeasureSubCoronalArgs,
    curve_set_path: &Option<PathBuf>,
) -> Result<IndexMap<CoronalMeasure, MeasureResult>, anyhow::Error> {
    let measures = subcommand
        .measures
        .unwrap_or_else(CoronalMeasure::all_measures);
    let (curve_set, apex_set) = if let Some(curve_set) = curve_set_path.as_ref() {
        let reader = std::fs::File::open(curve_set)
            .with_context(|| format!("Load curve set {:?}", curve_set))?;
        let cs: CurveDesc = serde_json::from_reader(reader)?;
        (cs.curves, cs.apices)
    } else {
        let curve_desc = coronal_points.identify_curves();
        (curve_desc.curves, curve_desc.apices)
    };
    let mut results: IndexMap<CoronalMeasure, MeasureResult> = Default::default();
    let data = (&coronal_points, &curve_set, &apex_set);
    for measure in measures {
        let spinal_measure: Box<dyn MeasureComponent> = (&measure, &data).into();
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
        let subcommand = MeasureSubCommands::Sagittal(MeasureSubSagittallArgs::default());
        let args = MeasureArgs {
            input,
            curve_set,
            labelme,
            subcommand,
        };
        cmd(args)?;

        // TODO: Test correct measurements

        let input = if labelme {
            data_dir.join(case_dir).join("frontal.json")
        } else {
            data_dir.join(case_dir).join("frontal_native.json")
        };
        let curve_set = None;
        let subcommand = MeasureSubCommands::Coronal(MeasureSubCoronalArgs::default());
        let args = MeasureArgs {
            input,
            curve_set,
            labelme,
            subcommand,
        };
        cmd(args)?;

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
