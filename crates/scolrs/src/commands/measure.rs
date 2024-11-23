use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use crate::cli::{MeasureArgs, MeasureSubCommands, MeasureSubCoronalArgs, MeasureSubSagittallArgs};
use anyhow::{Context, Result};
use indexmap::IndexMap;
use labelme_rs::{serde_json, LabelMeData, LabelMeDataLine};
use log::debug;
use scolrs::{
    draw::{MeasureComponent, MeasureError},
    CoronalMeasure, CoronalPointsAndCurve, CoronalPointsAndCurveLine, HasImageMetadata,
    MeasureAndDraw, SagittalMeasure, SagittalPoints, SagittalPointsLine,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

type Measurement = std::result::Result<f64, MeasureError>;

#[derive(Serialize, Deserialize)]
pub struct MeasureResult<V>
where
    V: std::hash::Hash + Eq + std::cmp::Ord,
{
    pub measurements: IndexMap<V, Measurement>,
    pub unit_of_length: String,
}

#[derive(Serialize, Deserialize)]
pub struct MeasureLine<V>
where
    V: std::hash::Hash + Eq + std::cmp::Ord,
{
    pub filename: String,
    pub content: MeasureResult<V>,
}

type SagittalMeasureLine = MeasureLine<SagittalMeasure>;
type CoronalMeasureLine = MeasureLine<CoronalMeasure>;

fn process_ndjson(args: MeasureArgs) -> Result<()> {
    let reader: Box<dyn BufRead> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::new(std::io::stdin()))
    } else {
        Box::new(BufReader::new(File::open(&args.input)?))
    };
    let mut writer: Box<dyn Write> = if let Some(output) = args.output.as_ref() {
        Box::new(BufWriter::new(File::create(output)?))
    } else {
        Box::new(std::io::stdout())
    };
    for line in reader.lines() {
        match args.subcommand.clone() {
            MeasureSubCommands::Coronal(subcommand) => {
                let data_line = load_labelme_or_native::<CoronalPointsAndCurveLine, LabelMeDataLine>(
                    args.labelme,
                    &args.input,
                    &line?,
                )?;
                let results = measure_coronal(data_line.content, subcommand)?;
                let line = CoronalMeasureLine {
                    filename: data_line.filename,
                    content: results,
                };
                writeln!(writer, "{}", serde_json::to_string(&line)?)?;
            }
            MeasureSubCommands::Sagittal(subcommand) => {
                let data_line = load_labelme_or_native::<SagittalPointsLine, LabelMeDataLine>(
                    args.labelme,
                    &args.input,
                    &line?,
                )?;
                let results = measure_sagittal(data_line.content, subcommand)?;
                let line = SagittalMeasureLine {
                    filename: data_line.filename,
                    content: results,
                };
                writeln!(writer, "{}", serde_json::to_string(&line)?)?;
            }
        }
    }
    Ok(())
}

/// Return T from native or LabelMe data
///
/// if `args.labelme` is true, load LM then convert to T
/// otherwise, load T directly
fn load_labelme_or_native<T, LM>(labelme: bool, input: &Path, json_str: &str) -> Result<T>
where
    T: DeserializeOwned,
    T: TryFrom<LM>,
    <T as std::convert::TryFrom<LM>>::Error:
        std::error::Error + std::marker::Send + std::marker::Sync + 'static,
    LM: for<'a> TryFrom<&'a str, Error = serde_json::Error>,
{
    if labelme {
        let data: LM = json_str
            .try_into()
            .with_context(|| format!("Load LabelMeData from {:?}", input))?;
        let data = T::try_from(data)?;
        Ok(data)
    } else {
        let data: T = serde_json::from_str(json_str)?;
        Ok(data)
    }
}

fn process_json(args: MeasureArgs) -> Result<()> {
    debug!("Loading {:?}", args.input);
    let data_str = std::fs::read_to_string(&args.input)?;

    let mut writer: Box<dyn Write> = if let Some(output) = args.output.as_ref() {
        Box::new(BufWriter::new(File::create(output)?))
    } else {
        Box::new(std::io::stdout())
    };
    match args.subcommand {
        MeasureSubCommands::Coronal(subcommand) => {
            let data = load_labelme_or_native::<CoronalPointsAndCurve, LabelMeData>(
                args.labelme,
                &args.input,
                &data_str,
            )?;
            let results = measure_coronal(data, subcommand)?;
            writeln!(writer, "{}", serde_json::to_string_pretty(&results)?)?;
        }
        MeasureSubCommands::Sagittal(subcommand) => {
            let data = load_labelme_or_native::<SagittalPoints, LabelMeData>(
                args.labelme,
                &args.input,
                &data_str,
            )?;
            let results = measure_sagittal(data, subcommand)?;
            writeln!(writer, "{}", serde_json::to_string_pretty(&results)?)?;
        }
    };

    Ok(())
}

fn measure_x<T, U>(data: T, measures: Vec<U>) -> Result<MeasureResult<U>>
where
    T: HasImageMetadata,
    U: std::hash::Hash + Eq + std::cmp::Ord,
    for<'a, 'b> (&'a U, &'b T): Into<Box<dyn MeasureComponent + 'b>>,
{
    let mut measurements: IndexMap<U, Measurement> = Default::default();
    for measure in measures {
        let spinal_measure: Box<dyn MeasureComponent> = (&measure, &data).into();
        measurements.insert(measure, spinal_measure.measure());
    }
    let result = MeasureResult {
        measurements,
        unit_of_length: data.image_metadata().unit.clone(),
    };
    Ok(result)
}

fn measure_sagittal(
    sagittal_points: SagittalPoints,
    subcommand: MeasureSubSagittallArgs,
) -> Result<MeasureResult<SagittalMeasure>> {
    let measures = subcommand
        .measures
        .unwrap_or_else(SagittalMeasure::all_measures);
    measure_x(sagittal_points, measures)
}

fn measure_coronal(
    data: CoronalPointsAndCurve,
    subcommand: MeasureSubCoronalArgs,
) -> Result<MeasureResult<CoronalMeasure>, anyhow::Error> {
    let measures = subcommand
        .measures
        .unwrap_or_else(CoronalMeasure::all_measures);
    measure_x(data, measures)
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
    use std::path::PathBuf;

    fn test_data_directory() -> std::path::PathBuf {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../../tests/data");
        path
    }

    fn output_path(name: &str) -> Result<PathBuf> {
        if let Ok(dir) = std::env::var("TEST_OUTPUT_DIR") {
            let path = PathBuf::from(dir).join("scol").join(name);
            std::fs::create_dir_all(path.parent().unwrap())?;
            return Ok(path);
        }
        let devnull = PathBuf::from("/dev/null");
        if devnull.exists() {
            Ok(devnull)
        } else {
            Ok(PathBuf::from("NUL".to_string()))
        }
    }

    fn _test_case(case_dir: &str, labelme: bool) -> Result<()> {
        // Just test if the command runs without errors
        let data_dir = test_data_directory();
        let (input, output) = if labelme {
            (
                data_dir.join(case_dir).join("lateral.json"),
                output_path(&format!("{}_lateral.json", case_dir))?,
            )
        } else {
            (
                data_dir.join(case_dir).join("lateral_native.json"),
                output_path(&format!("{}_lateral_native.json", case_dir))?,
            )
        };
        let subcommand = MeasureSubCommands::Sagittal(MeasureSubSagittallArgs::default());
        let args = MeasureArgs {
            input,
            output: Some(output),
            labelme,
            subcommand,
        };
        cmd(args)?;

        // TODO: Test correct measurements

        let (input, output) = if labelme {
            (
                data_dir.join(case_dir).join("frontal.json"),
                output_path(&format!("{}_frontal.json", case_dir))?,
            )
        } else {
            (
                data_dir.join(case_dir).join("frontal_native.json"),
                output_path(&format!("{}_frontal_native.json", case_dir))?,
            )
        };
        let subcommand = MeasureSubCommands::Coronal(MeasureSubCoronalArgs::default());
        let args = MeasureArgs {
            input,
            output: Some(output),
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
