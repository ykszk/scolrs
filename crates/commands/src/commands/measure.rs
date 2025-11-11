use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use crate::cli::{
    MeasureArgs, MeasureSubCommands, MeasureSubCoronalArgs, MeasureSubNeckArgs,
    MeasureSubSagittallArgs,
};
use crate::utils::Ndjson;
use anyhow::{Context, Result};
use indexmap::IndexMap;
use labelme_rs::{serde_json, LabelMeData, LabelMeDataLine};
use log::debug;
use scolrs::{
    draw::{ConfidenceComponent, MeasureComponent, MeasureError},
    head_neck::{LateralPoints, LateralPointsLine, NeckLateralMeasure, NeckMeasureComponent},
    CoronalMeasure, CoronalPointsAndCurve, CoronalPointsAndCurveLine, HasImageMetadata,
    MeasureAndDraw, PointConfidence, SagittalMeasure, SagittalPoints, SagittalPointsLine, Scalable,
    ScaledType,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

type Measurement = std::result::Result<f64, MeasureError>;
type MeasurementVec = std::result::Result<Vec<f64>, MeasureError>;

#[derive(Serialize, Deserialize, Default)]
pub struct MeasureResult<V, T>
where
    V: std::hash::Hash + Eq + std::cmp::Ord,
{
    pub measurements: IndexMap<V, std::result::Result<T, MeasureError>>,
    pub confidences: Option<IndexMap<V, std::result::Result<f64, MeasureError>>>,
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
type FlattenedNeckMeasurements = MeasureLine<String, f64>;

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
                let flatten = measure_sub_neck_args.flatten;
                let results = measure_neck(data_line.content, measure_sub_neck_args)?;
                let line = MeasureLine {
                    filename: data_line.filename,
                    content: results,
                };
                if flatten {
                    let mut flattened = FlattenedNeckMeasurements {
                        filename: line.filename,
                        content: Default::default(),
                    };
                    flattened.content.unit_of_length = line.content.unit_of_length;
                    for (k, v) in line.content.measurements {
                        if let Ok(v) = v {
                            if v.len() == 1 {
                                flattened.content.measurements.insert(k, Ok(v[0]));
                            } else {
                                for (i, value) in v.into_iter().enumerate() {
                                    let key = format!("{}_{}", k, i + 1);
                                    flattened.content.measurements.insert(key, Ok(value));
                                }
                            }
                        }
                    }
                    writeln!(writer, "{}", serde_json::to_string(&flattened)?)?;
                } else {
                    writeln!(writer, "{}", serde_json::to_string(&line)?)?;
                }
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

fn measure_x<T, U>(data: ScaledType<T>, measures: Vec<U>) -> Result<MeasureResult<U, f64>>
where
    T: HasImageMetadata + Scalable + PointConfidence,
    U: std::hash::Hash + Eq + std::cmp::Ord + Clone + std::fmt::Debug,
    for<'a, 'b> (&'a U, &'b ScaledType<T>):
        Into<Box<dyn MeasureComponent + 'b>> + Into<Box<dyn ConfidenceComponent + 'b>>,
{
    let confidences = if data.0.get_confidence().is_some() {
        let confs = confidence_x(&data, &measures)?;
        Some(confs)
    } else {
        None
    };
    let mut measurements: IndexMap<U, Measurement> = Default::default();
    for measure in measures {
        let spinal_measure: Box<dyn MeasureComponent> = (&measure, &data).into();
        measurements.insert(measure.clone(), spinal_measure.measure());
    }
    let result = MeasureResult {
        measurements,
        confidences,
        unit_of_length: data.0.image_metadata().unit.clone(),
    };
    Ok(result)
}

type ConfResult<U> = IndexMap<U, std::result::Result<f64, MeasureError>>;

fn confidence_x<T, U>(data: &ScaledType<T>, measures: &[U]) -> Result<ConfResult<U>>
where
    T: Scalable,
    U: std::hash::Hash + Eq + std::cmp::Ord + Clone + std::fmt::Debug,
    for<'a, 'b> (&'a U, &'b ScaledType<T>): Into<Box<dyn ConfidenceComponent + 'b>>,
{
    let mut measurements: IndexMap<U, std::result::Result<f64, MeasureError>> = Default::default();
    for measure in measures {
        log::debug!("Calculating confidence for measure {:?}", measure);
        let spinal_measure: Box<dyn ConfidenceComponent> = (measure, data).into();
        let conf = spinal_measure.confidence();
        if let Some(conf) = conf {
            measurements.insert(measure.clone(), conf);
        }
    }
    Ok(measurements)
}

fn measure_sagittal(
    sagittal_points: SagittalPoints,
    subcommand: MeasureSubSagittallArgs,
) -> Result<MeasureResult<SagittalMeasure, f64>> {
    let scaled_data = sagittal_points.into_scaled()?;
    let measures = subcommand.measures.unwrap_or_else(SagittalMeasure::all);
    measure_x(scaled_data, measures)
}

fn measure_coronal(
    data: CoronalPointsAndCurve,
    subcommand: MeasureSubCoronalArgs,
) -> Result<MeasureResult<CoronalMeasure, f64>, anyhow::Error> {
    let scaled_data = data.into_scaled()?;
    let measures = subcommand.measures.unwrap_or_else(CoronalMeasure::all);
    measure_x(scaled_data, measures)
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
    let scaled_data = lateral_points.into_scaled()?;
    let measures: Vec<Box<dyn NeckMeasureComponent>> =
        measures.iter().map(|m| (m, &scaled_data).into()).collect();
    let measures = measure_all(measures)?;
    Ok(NeckMeasurements {
        measurements: measures,
        confidences: None,
        unit_of_length: scaled_data.0.image_metadata.unit.clone(),
    })
}

fn measure_neck(
    lateral_points: LateralPoints,
    subcommand: MeasureSubNeckArgs,
) -> Result<NeckMeasurements> {
    // perform scaling in [process_neck]
    let measures = subcommand.measures.unwrap_or_else(NeckLateralMeasure::all);
    process_neck(lateral_points, &measures)
}

pub fn cmd(args: MeasureArgs) -> Result<()> {
    if args.input.as_os_str() == "-" || args.input.is_ndjson() {
        process_ndjson(args)
    } else {
        process_json(args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
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

    #[rstest]
    #[case::c1("case1", true)]
    #[case::c2("case2", true)]
    #[case::c2_native("case2", false)]
    #[case::c3("case3", true)]
    #[case::c4("case4", true)]
    fn measure_case(#[case] case_dir: &str, #[case] labelme: bool) -> Result<()> {
        // Just test if the command runs without errors
        let data_dir = test_data_directory();
        let suffix = if labelme { "" } else { "_native" };

        let input = data_dir
            .join(case_dir)
            .join(format!("lateral{suffix}.json"));
        let output = output_path(&format!("{case_dir}_lateral{suffix}.json"))?;
        let subcommand = MeasureSubCommands::Sagittal(MeasureSubSagittallArgs::default());
        let args = MeasureArgs {
            input,
            output: Some(output),
            labelme,
            subcommand,
        };
        cmd(args)?;

        // TODO: Test correct measurements
        let input = data_dir
            .join(case_dir)
            .join(format!("frontal{suffix}.json"));
        let output = output_path(&format!("{case_dir}_frontal{suffix}.json"))?;
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

    #[rstest]
    fn neck(
        #[values(
            "neck_case1/lateral_lateral_points.json",
            "neck_case2/extension_lateral_lateral_points.json",
            "neck_case2/flexion_lateral_lateral_points.json"
        )]
        filename: &str,
    ) -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");

        let input = data_dir.join(filename);
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
    fn neck_scaled_measurements() -> Result<()> {
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
