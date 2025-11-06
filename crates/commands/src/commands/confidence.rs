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

fn read_npz(path: &Path, key: &str) -> Result<ndarray::Array3<f64>> {
    let mut npz = ndarray_npz::NpzReader::new(
        std::fs::File::open(path).with_context(|| format!("Opening {:?}", path))?,
    )?;
    let array: ndarray::Array3<f64> = npz.by_name(key).with_context(|| match npz.names() {
        Ok(names) => {
            if names.is_empty() {
                "The provided npz file contains no arrays.".to_string()
            } else {
                format!(
                    "The provided npz file does not contain the key {}. Available keys are {}",
                    key,
                    names.join(", ")
                )
            }
        }
        Err(e) => format!("Failed to read keys from npz file: {}", e),
    })?;
    Ok(array)
}

pub fn cmd(args: ConfidenceArgs) -> Result<()> {
    let input_path = Path::new(&args.input);
    let heatmaps = match args.confidence_map.extension() {
        Some(ext) if ext == "npz" => read_npz(&args.confidence_map, &args.key)?,
        Some(ext) if ext == "npy" => {
            let array: ndarray::Array3<f64> =
                ndarray_npz::ndarray_npy::read_npy(&args.confidence_map).with_context(|| {
                    format!("Reading numpy array from {:?}", &args.confidence_map)
                })?;
            array
        }
        _ => {
            anyhow::bail!(
                "Confidence map file must have .npz or .npy extension, got {:?}",
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
