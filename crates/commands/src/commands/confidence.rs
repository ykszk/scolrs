use crate::utils::{read_heatmaps, CreateReaderWriter};
use crate::{cli::ConfidenceArgs, utils::Ndjson};
use anyhow::{bail, Context, Result};
use scolrs::{
    ContentFilename, CoronalPoints, CoronalPointsLine, HasImageMetadata, PointConfidence,
    SagittalPoints, SagittalPointsLine,
};
use std::{
    io::BufRead,
    path::{Path, PathBuf},
};

fn process<PointsType: PointConfidence + serde::Serialize + for<'de> serde::Deserialize<'de>>(
    input_path: &Path,
    heatmaps: &ndarray::Array3<f32>,
    args: &ConfidenceArgs,
) -> Result<()> {
    let file =
        std::fs::File::open(input_path).with_context(|| format!("Opening {:?}", input_path))?;
    let reader = std::io::BufReader::new(file);
    let points: PointsType = serde_json::from_reader(reader)?;
    if points.get_confidence().is_some() {
        log::warn!("Overwriting existing confidence values");
    }
    let confidence = points.extract_point_confidence(heatmaps.view());
    let mut output_points = points;
    *output_points.get_confidence_mut() = Some(confidence);
    let output_file = std::fs::File::create(&args.output)
        .with_context(|| format!("Creating {:?}", args.output))?;
    let writer = std::io::BufWriter::new(output_file);
    serde_json::to_writer_pretty(writer, &output_points)?;
    Ok(())
}

fn process_line<LineType: ContentFilename + serde::Serialize + for<'de> serde::Deserialize<'de>>(
    line: &str,
    args: &ConfidenceArgs,
    writer: &mut dyn std::io::Write,
) -> Result<()>
where
    LineType::ContentType: PointConfidence + HasImageMetadata,
{
    let points_line: LineType = serde_json::from_str(line)?;
    let line_filename = points_line.filename().to_string();
    log::info!("Processing line with filename {}", line_filename);
    let points = points_line.content_filename().0;
    if points.get_confidence().is_some() {
        log::warn!("Overwriting existing confidence values");
    }
    let image_path = PathBuf::from(&points.image_metadata().path);
    let heatmap_path = args
        .confidence_map
        .join(image_path.with_extension("mha").file_name().unwrap());
    let heatmaps = read_heatmaps(&heatmap_path)?;
    let confidence = points.extract_point_confidence(heatmaps.view());
    let mut output_points = points;
    *output_points.get_confidence_mut() = Some(confidence);
    let output_points_line = LineType::new(output_points, line_filename);
    let output_line = serde_json::to_string(&output_points_line)?;
    writeln!(writer, "{}", output_line)?;
    Ok(())
}

fn process_ndjson(args: ConfidenceArgs) -> Result<()> {
    if !args.confidence_map.exists() {
        bail!(
            "Confidence map directory {:?} does not exist",
            args.confidence_map
        );
    }

    let reader = args.input.create_reader()?;
    let mut writer = args.output.create_writer()?;
    for line in reader.lines() {
        let line = line?;
        match args.input_type {
            crate::cli::Plane::Coronal => {
                process_line::<CoronalPointsLine>(&line, &args, &mut writer)?;
            }
            crate::cli::Plane::Sagittal => {
                process_line::<SagittalPointsLine>(&line, &args, &mut writer)?;
            }
        }
    }

    Ok(())
}

pub fn cmd(args: ConfidenceArgs) -> Result<()> {
    if args.input.as_os_str() == "-" || args.input.is_ndjson() {
        return process_ndjson(args);
    }

    let input_path = Path::new(&args.input);
    let heatmaps = match args.confidence_map.extension() {
        Some(ext) if ext == "mha" || ext == "mhd" => read_heatmaps(&args.confidence_map)?,
        _ => {
            anyhow::bail!(
                "Confidence map file must have .mha or .mhd extension, got {:?}",
                args.confidence_map
            );
        }
    };
    match args.input_type {
        crate::cli::Plane::Coronal => {
            process::<CoronalPoints>(input_path, &heatmaps, &args)?;
        }
        crate::cli::Plane::Sagittal => {
            process::<SagittalPoints>(input_path, &heatmaps, &args)?;
        }
    }
    Ok(())
}
