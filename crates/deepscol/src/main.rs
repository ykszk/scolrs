use std::vec;

use anyhow::Result;
use ndarray::{s, Array3, Axis, CowArray, NewAxis};
// use ndarray_npz::ndarray::{s, Array3, Axis, CowArray, NewAxis};
use clap::Parser;
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Path to the ONNX model file
    model: PathBuf,
    /// Path to the input image file
    input_image: PathBuf,
    /// Path to save the output heatmaps or image
    output_image: PathBuf,
    /// Save LabelMe format points
    #[arg(long)]
    labelme: Option<PathBuf>,
    /// Text file of labels for the points in LabelMe format
    #[arg(long)]
    labels: Option<PathBuf>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let model_path = &args.model;
    let image_path = &args.input_image;
    let output_path = &args.output_image;

    // Load the model
    let environment = Environment::builder().build()?.into_arc();
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .with_model_from_file(model_path)?;
    log::info!("Model loaded successfully");
    log::debug!("model input dimensions {:?}", session.inputs[0].dimensions);
    // Load the image
    let image = image::open(image_path).expect("Failed to open image");
    let original_image_width = image.width();
    let original_image_height = image.height();
    log::info!("Image loaded successfully");
    let model_input_height = session.inputs[0].dimensions[2].unwrap();
    let image = image.thumbnail(10000, model_input_height);
    // Convert the image to a tensor
    let image = Array3::from_shape_vec(
        (image.height() as usize, image.width() as usize, 1),
        image.to_luma8().into_raw(),
    )?;
    // convert to float32
    let image = image.mapv(|x| x as f32 / 255.0);
    log::debug!("Image converted to tensor shape: {:?}", image.shape());
    // pad image width to be divisible by 256
    let padded_width = ((image.shape()[1] + 255) / 256) * 256;
    let mut padded_image: Array3<f32> =
        Array3::zeros((image.shape()[0], padded_width, image.shape()[2]));
    padded_image
        .slice_mut(s![.., ..image.shape()[1], ..])
        .assign(&image);
    let image = padded_image;
    // to channel first
    let image = image.permuted_axes([2, 0, 1]);

    let input = CowArray::from(image.slice(s![NewAxis, .., .., ..]).into_dyn());
    log::debug!(
        "Image converted to tensor successfully with shape: {:?}",
        input.shape()
    );
    // Run the model
    let inputs = vec![Value::from_array(session.allocator(), &input)?];
    log::info!("Running model");
    let mut outputs: Vec<Value> = session.run(inputs)?;
    log::info!("Model run completed");

    // convert to ndarray
    let output = outputs.pop().unwrap();
    let output_array = output.try_extract::<f32>()?.view().to_owned();
    let output3 = output_array
        .index_axis_move(Axis(0), 0)
        .into_dimensionality::<ndarray::Ix3>()?;
    log::debug!("Output tensor shape: {:?}", output3.shape());
    // apply sigmoid
    let output3 = output3.mapv(|x| 1.0 / (1.0 + (-x).exp()));
    if output_path.ends_with(".npz") {
        let mut npz = ndarray_npz::NpzWriter::new_compressed(std::fs::File::create(output_path)?);
        //  convert ort's ndarray (v0.15.6) to ndarray_npz's ndarray (v0.16.1). Can be removed when these crates are updated.
        let ndarray_output3 = ndarray_npz::ndarray::Array3::from_shape_vec(
            (output3.shape()[0], output3.shape()[1], output3.shape()[2]),
            output3.clone().into_raw_vec(),
        )?;
        let ndarray_output3 = ndarray_output3.mapv(|x| (x * 1000.0) as u16);
        // save the output
        npz.add_array("heatmaps", &ndarray_output3)?;
        npz.finish()?;
    } else if output_path.ends_with(".png") {
        let rgb = output3.slice(s![..3, .., ..]).to_owned();
        let first_channel_image =
            image::ImageBuffer::from_fn(rgb.shape()[2] as u32, rgb.shape()[1] as u32, |x, y| {
                let r = rgb[[0, y as usize, x as usize]];
                let r = (r * 255.0) as u8;
                let g = rgb[[1, y as usize, x as usize]];
                let g = (g * 255.0) as u8;
                let b = rgb[[2, y as usize, x as usize]];
                let b = (b * 255.0) as u8;

                image::Rgb([r, g, b])
            });
        first_channel_image
            .save(output_path)
            .expect("Failed to save output image");
    }

    if let Some(labelme_path) = &args.labelme {
        let points = deepscol::extract_points(&output3, 0.1).unwrap();
        // Save the points to LabelMe format
        let mut lm_data = labelme_rs::LabelMeData {
            imagePath: std::path::absolute(image_path)?
                .to_string_lossy()
                .to_string(),
            imageHeight: original_image_height as usize,
            imageWidth: original_image_width as usize,
            ..Default::default()
        };
        let labels = if let Some(labels_path) = &args.labels {
            let labels = std::fs::read_to_string(labels_path)
                .expect("Failed to read labels file")
                .lines()
                .map(|line| line.trim().to_string())
                .collect::<Vec<String>>();
            if labels.len() != points.len() {
                return Err(anyhow::anyhow!(
                    "Number of labels does not match number of channels. Expected {} labels, got {}",
                    points.len(),
                    labels.len()
                ));
            }
            labels
        } else {
            points
                .iter()
                .enumerate()
                .map(|(i, _)| format!("label{}", i + 1))
                .collect::<Vec<String>>()
        };

        for (i_point, v_point) in points.into_iter().enumerate() {
            if v_point.is_empty() {
                log::warn!("No points found for {}", labels[i_point]);
                continue; // Skip empty points
            }
            for point in v_point {
                let points = vec![(point.0 as f64, point.1 as f64)];
                let label = labels[i_point].clone();
                let shape_type = "point".to_string();
                let shape = labelme_rs::Shape {
                    label,
                    points,
                    shape_type,
                    ..Default::default()
                };
                lm_data.shapes.push(shape);
            }
        }
        let scale = original_image_height as f64 / model_input_height as f64;
        lm_data.scale(scale);
        // Save the LabelMe data to a file
        let labelme_json = serde_json::to_string_pretty(&lm_data)?;

        std::fs::write(labelme_path, labelme_json).expect("Failed to write LabelMe data to file");
    }
    Ok(())
}
