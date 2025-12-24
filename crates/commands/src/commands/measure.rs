use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use crate::cli::{
    CurveSetAlgorithm, MeasureArgs, MeasureSubCommands, MeasureSubCoronalArgs, MeasureSubNeckArgs,
    MeasureSubSagittallArgs,
};
use crate::utils::Ndjson;
use anyhow::{Context, Result};
use labelme_rs::{serde_json, LabelMeData, LabelMeDataLine};
use log::debug;
use scolrs::measure::{measure_x, FlattenResult, MeasureLine, MeasureResult, NeckMeasurements};
use scolrs::{
    draw::ReductionMethod,
    head_neck::{LateralPoints, LateralPointsLine, NeckLateralMeasure},
    CoronalMeasure, CoronalPointsAndCurve, CoronalPointsAndCurveLine, MeasureAndDraw,
    SagittalMeasure, SagittalPoints, SagittalPointsLine, Scalable,
};
use serde::de::DeserializeOwned;

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
                let mut data_line = load_labelme_or_native::<
                    CoronalPointsAndCurveLine,
                    LabelMeDataLine,
                >(args.labelme, &args.input, &line?)?;
                if subcommand.algorithm != CurveSetAlgorithm::Score {
                    let algorithm = scolrs::CurveSetAlgorithm::Apex;
                    let curves = data_line
                        .content
                        .coronal_points
                        .identify_curves_with_algorithm(&algorithm);
                    data_line.content.curves = curves;
                }
                let results = measure_coronal(data_line.content, subcommand, args.reduce)?;
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
                let results = measure_sagittal(data_line.content, subcommand, args.reduce)?;
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
                let results = measure_neck(data_line.content, measure_sub_neck_args, args.reduce)?;
                let line = MeasureLine {
                    filename: data_line.filename,
                    content: results,
                };
                if flatten {
                    let mut flattened = FlattenedNeckMeasurements {
                        filename: line.filename,
                        content: Default::default(),
                    };
                    flattened.content.unit_of_length = line.content.unit_of_length.clone();
                    flattened.content = line.content.into_flat();
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
    match args.subcommand.clone() {
        MeasureSubCommands::Coronal(subcommand) => {
            let mut data = load_labelme_or_native::<CoronalPointsAndCurve, LabelMeData>(
                args.labelme,
                &args.input,
                &data_str,
            )?;
            if subcommand.algorithm != CurveSetAlgorithm::Score {
                let algorithm = scolrs::CurveSetAlgorithm::Apex;
                let curves = data
                    .coronal_points
                    .identify_curves_with_algorithm(&algorithm);
                data.curves = curves;
            }
            let results = measure_coronal(data, subcommand, args.reduce)?;
            writeln!(writer, "{}", serde_json::to_string_pretty(&results)?)?;
        }
        MeasureSubCommands::Sagittal(subcommand) => {
            let data = load_labelme_or_native::<SagittalPoints, LabelMeData>(
                args.labelme,
                &args.input,
                &data_str,
            )?;
            let results = measure_sagittal(data, subcommand, args.reduce)?;
            writeln!(writer, "{}", serde_json::to_string_pretty(&results)?)?;
        }
        MeasureSubCommands::Neck(measure_sub_neck_args) => {
            let data = load_labelme_or_native::<LateralPoints, LabelMeData>(
                args.labelme,
                &args.input,
                &data_str,
            )?;
            let results = measure_neck(data, measure_sub_neck_args, args.reduce)?;
            writeln!(writer, "{}", serde_json::to_string_pretty(&results)?)?;
        }
    };

    Ok(())
}

fn convert_reduce(method: crate::cli::ReductionMethod) -> ReductionMethod {
    match method {
        crate::cli::ReductionMethod::Arithmetic => ReductionMethod::ArithmeticMean,
        crate::cli::ReductionMethod::Geometric => ReductionMethod::GeometricMean,
        crate::cli::ReductionMethod::Harmonic => ReductionMethod::HarmonicMean,
    }
}

fn measure_sagittal(
    sagittal_points: SagittalPoints,
    subcommand: MeasureSubSagittallArgs,
    reduce: crate::cli::ReductionMethod,
) -> Result<MeasureResult<SagittalMeasure, f64>> {
    let scaled_data = sagittal_points.into_scaled()?;
    let measures = subcommand.measures.unwrap_or_else(SagittalMeasure::all);
    let reduce = convert_reduce(reduce);
    Ok(measure_x(scaled_data, measures, reduce))
}

fn measure_coronal(
    data: CoronalPointsAndCurve,
    subcommand: MeasureSubCoronalArgs,
    reduce: crate::cli::ReductionMethod,
) -> Result<MeasureResult<CoronalMeasure, f64>> {
    let scaled_data = data.into_scaled()?;
    let measures = subcommand.measures.unwrap_or_else(CoronalMeasure::all);
    let reduce = convert_reduce(reduce);
    Ok(measure_x(scaled_data, measures, reduce))
}

fn measure_neck(
    lateral_points: LateralPoints,
    subcommand: MeasureSubNeckArgs,
    reduce: crate::cli::ReductionMethod,
) -> Result<NeckMeasurements> {
    // perform scaling in [process_neck]
    let measures = subcommand.measures.unwrap_or_else(NeckLateralMeasure::all);
    let scaled_data = lateral_points.into_scaled()?;
    let reduce = convert_reduce(reduce);
    let measures = measure_x(scaled_data, measures, reduce);
    let measures = measures.into_string_map();
    Ok(measures)
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
        let reduce = crate::cli::ReductionMethod::Geometric;

        let input = data_dir
            .join(case_dir)
            .join(format!("lateral{suffix}.json"));
        let output = output_path(&format!("{case_dir}_lateral{suffix}.json"))?;
        let subcommand = MeasureSubCommands::Sagittal(MeasureSubSagittallArgs::default());
        let args = MeasureArgs {
            input,
            output: Some(output),
            labelme,
            reduce,
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
            reduce,
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
        let subcommand = MeasureSubNeckArgs {
            measures: Some(measures.clone()),
            flatten: false,
        };
        let reduce = crate::cli::ReductionMethod::Geometric;
        let mut measurements =
            measure_neck(lateral_points, subcommand.clone(), reduce)?.measurements;
        let mut measurements_non_scaled =
            measure_neck(non_scaled, subcommand, reduce)?.measurements;

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
