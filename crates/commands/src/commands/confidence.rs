use crate::cli::ConfidenceArgs;
use anyhow::{Context, Result};
use scolrs::{CoronalPoints, PointConfidence, SagittalPoints};
use std::path::Path;

fn process<PointsType: PointConfidence + serde::Serialize + for<'de> serde::Deserialize<'de>>(
    input_path: &Path,
    heatmaps: &ndarray::Array3<f64>,
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
        .with_context(|| format!("Creating {:?}", &args.output))?;
    let writer = std::io::BufWriter::new(output_file);
    serde_json::to_writer_pretty(writer, &output_points)?;
    Ok(())
}

pub fn cmd(args: ConfidenceArgs) -> Result<()> {
    let input_path = Path::new(&args.input);
    let mut npz = ndarray_npz::NpzReader::new(
        std::fs::File::open(&args.confidence_map)
            .with_context(|| format!("Opening {:?}", &args.confidence_map))?,
    )?;
    let heatmaps: ndarray::Array3<f64> = npz
        .by_name(&args.key)
        .with_context(|| format!("Reading heatmaps from npz for key {}", &args.key))?;
    match args.plane {
        crate::cli::Plane::Coronal => {
            process::<CoronalPoints>(input_path, &heatmaps, &args)?;
        }
        crate::cli::Plane::Sagittal => {
            process::<SagittalPoints>(input_path, &heatmaps, &args)?;
        }
    }
    Ok(())
}
