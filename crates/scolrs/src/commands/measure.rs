use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use crate::cli::{
    MeasureArgs, MeasureSubCommands, MeasureSubCoronalArgs, MeasureSubNeckArgs,
    MeasureSubSagittallArgs,
};
use anyhow::{Context, Result};
use indexmap::IndexMap;
use labelme_rs::{serde_json, LabelMeData, LabelMeDataLine};
use log::debug;
use scolrs::{
    draw::{MeasureComponent, MeasureError},
    head_neck::{LateralPoints, LateralPointsLine, NeckLateralMeasure, NeckMeasureComponent},
    CoronalMeasure, CoronalPointsAndCurve, CoronalPointsAndCurveLine, HasImageMetadata,
    MeasureAndDraw, SagittalMeasure, SagittalPoints, SagittalPointsLine, Scalable,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

type Measurement = std::result::Result<f64, MeasureError>;
type MeasurementVec = std::result::Result<Vec<f64>, MeasureError>;

#[derive(Serialize, Deserialize)]
pub struct MeasureResult<V, T>
where
    V: std::hash::Hash + Eq + std::cmp::Ord,
{
    pub measurements: IndexMap<V, std::result::Result<T, MeasureError>>,
    pub unit_of_length: String,
}

#[derive(Serialize, Deserialize)]
pub struct MeasureLine<V, T>
where
    V: std::hash::Hash + Eq + std::cmp::Ord,
{
    pub filename: String,
    pub content: MeasureResult<V, T>,
}

type SagittalMeasureLine = MeasureLine<SagittalMeasure, f64>;
type CoronalMeasureLine = MeasureLine<CoronalMeasure, f64>;

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
            MeasureSubCommands::Neck(measure_sub_neck_args) => {
                let data_line = load_labelme_or_native::<LateralPointsLine, LabelMeDataLine>(
                    args.labelme,
                    &args.input,
                    &line?,
                )?;
                let results = measure_neck(data_line.content, measure_sub_neck_args)?;
                let line = MeasureLine {
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
        MeasureSubCommands::Neck(measure_sub_neck_args) => {
            let data = load_labelme_or_native::<LateralPoints, LabelMeData>(
                args.labelme,
                &args.input,
                &data_str,
            )?;
            let results = measure_neck(data, measure_sub_neck_args)?;
            writeln!(writer, "{}", serde_json::to_string_pretty(&results)?)?;
        }
    };

    Ok(())
}

fn measure_x<T, U>(data: T, measures: Vec<U>) -> Result<MeasureResult<U, f64>>
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
) -> Result<MeasureResult<SagittalMeasure, f64>> {
    let measures = subcommand
        .measures
        .unwrap_or_else(SagittalMeasure::all_measures);
    measure_x(sagittal_points, measures)
}

fn measure_coronal(
    data: CoronalPointsAndCurve,
    subcommand: MeasureSubCoronalArgs,
) -> Result<MeasureResult<CoronalMeasure, f64>, anyhow::Error> {
    let measures = subcommand
        .measures
        .unwrap_or_else(CoronalMeasure::all_measures);
    measure_x(data, measures)
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

type NeckMeasurements = MeasureResult<String, Vec<f64>>;

fn process_neck(
    lateral_points: LateralPoints,
    measures: &[NeckLateralMeasure],
) -> Result<NeckMeasurements> {
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
    Ok(NeckMeasurements {
        measurements: measures,
        unit_of_length: lateral_points.image_metadata.unit.clone(),
    })
}

fn measure_neck(
    lateral_points: LateralPoints,
    subcommand: MeasureSubNeckArgs,
) -> Result<NeckMeasurements> {
    let measures = subcommand.measures.unwrap_or_else(NeckLateralMeasure::all);
    process_neck(lateral_points, &measures)
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

    fn null_device() -> PathBuf {
        if cfg!(windows) {
            PathBuf::from("NUL")
        } else {
            PathBuf::from("/dev/null")
        }
    }

    fn output_path(name: &str) -> Result<PathBuf> {
        if let Ok(dir) = std::env::var("TEST_OUTPUT_DIR") {
            let path = PathBuf::from(dir).join("scol").join(name);
            std::fs::create_dir_all(path.parent().unwrap())?;
            return Ok(path);
        }
        let devnull = null_device();
        Ok(devnull)
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

    #[test]
    fn test_measure_cmd_neck_case1() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");

        let input = data_dir.join("neck_case1/lateral_lateral_points.json");
        let subcommand = MeasureSubCommands::Neck(MeasureSubNeckArgs::default());
        let args = MeasureArgs {
            input,
            output: Some(null_device()),
            subcommand,
            ..Default::default()
        };
        cmd(args)
    }

    #[test]
    fn test_measure_cmd_neck_case2() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");

        let input = data_dir.join("neck_case2/extension_lateral_lateral_points.json");
        let subcommand = MeasureSubCommands::Neck(MeasureSubNeckArgs::default());
        let args = MeasureArgs {
            input,
            output: Some(null_device()),
            subcommand,
            ..Default::default()
        };
        cmd(args)?;

        let input = data_dir.join("neck_case2/flexion_lateral_lateral_points.json");
        let subcommand = MeasureSubCommands::Neck(MeasureSubNeckArgs::default());
        let args = MeasureArgs {
            input,
            output: Some(null_device()),
            subcommand,
            ..Default::default()
        };

        cmd(args)
    }

    #[test]
    fn _test_neck_scaled_measurements() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");

        let input = data_dir.join("neck_case1/lateral_lateral_points.json");
        let mut lateral_points: LateralPoints =
            serde_json::from_str(std::fs::read_to_string(&input)?.as_str())?;
        lateral_points.image_metadata.spacing_xy = (0.5, 0.5);
        let mut non_scaled = lateral_points.clone();
        non_scaled.image_metadata.spacing_xy = (1.0, 1.0);
        let measures = NeckLateralMeasure::all();
        let mut measurements = process_neck(lateral_points, &measures)?.measurements;
        let mut measurements_non_scaled = process_neck(non_scaled, &measures)?.measurements;

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
