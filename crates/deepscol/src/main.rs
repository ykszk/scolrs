use std::path::PathBuf;
use std::vec;

use anyhow::Result;
use clap::Parser;
use labelme_rs::LabelMeDataWImage;
use ndarray::s;

use scolrs::{CoronalPointsAndCurve, HasImageMetadata};

#[derive(Parser)]
#[command(version, about)]
struct Args {
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
    /// Output html
    #[arg(long)]
    html: Option<PathBuf>,
    /// Path to the onnx model file
    #[arg(long)]
    model: Option<PathBuf>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let model_path = &args.model;
    let image_path = &args.input_image;
    let output_path = &args.output_image;
    // ort::set_api(ort_tract::api());
    let builder = ort::session::Session::builder()
        .expect("Cannot create Session builder.")
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Disable)
        .expect("Cannot optimize graph.")
        .with_parallel_execution(true)
        .expect("Cannot activate parallel execution.")
        .with_intra_threads(2)
        .expect("Cannot set intra thread count.")
        .with_inter_threads(1)
        .expect("Cannot set inter thread count.");
    let mut session = if let Some(model_path) = model_path {
        builder
            .commit_from_file(model_path)
            .expect("Cannot load model from file.")
    } else {
        builder
            .commit_from_memory(include_bytes!("../models/spine_mobileone_s1.onnx"))
            .expect("Cannot load model from memory.")
    };
    log::info!("Model loaded successfully");
    log::debug!(
        "model input dimensions {:?}",
        session.inputs[0].input_type.tensor_shape().unwrap()
    );

    let (image, metadata) = deepscol::load_image(&std::fs::read(image_path)?)?;
    let original_image_width = image.width();
    let original_image_height = image.height();
    log::info!(
        "Image with shape w x h {:?} loaded successfully",
        (original_image_width, original_image_height)
    );
    log::debug!("Expected model input {:?}", session);
    let model_input_height = session.inputs[0].input_type.tensor_shape().unwrap()[2];
    let arr4 = deepscol::to_model_input(image, model_input_height)?;
    let input = ort::value::Tensor::from_array(arr4)?;
    log::debug!(
        "Image converted to tensor successfully with shape: {:?}",
        input.shape()
    );
    // Run the model
    // let inputs = vec![Value::from_array(session.allocator(), &input)?];
    let inputs = ort::inputs!["modelInput" => input];
    log::info!("Running model");
    let outputs: ort::session::SessionOutputs = session.run(inputs)?;
    log::info!("Model run completed");

    // convert to ndarray
    let output3 = deepscol::extract_array_from_output(outputs);
    if output_path.ends_with(".npz") {
        let mut npz = ndarray_npz::NpzWriter::new_compressed(std::fs::File::create(output_path)?);
        //  convert ort's ndarray (v0.15.6) to ndarray_npz's ndarray (v0.16.1). Can be removed when these crates are updated.
        let ndarray_output3 = ndarray_npz::ndarray::Array3::from_shape_vec(
            (output3.shape()[0], output3.shape()[1], output3.shape()[2]),
            output3.clone().into_raw_vec_and_offset().0,
        )?;
        let ndarray_output3 = ndarray_output3.mapv(|x| (x * 1000.0) as u16);
        // save the output
        npz.add_array("heatmaps", &ndarray_output3)?;
        npz.finish()?;
    } else {
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
    let lm_scale = original_image_height as f64 / model_input_height as f64;

    let mut extracted_points: Option<Vec<Vec<(f32, f32)>>> = None;
    let mut extracted_lm: Option<labelme_rs::LabelMeData> = None;

    let abs_path = image_path
        .canonicalize()
        .expect("Failed to get absolute path of image")
        .to_string_lossy()
        .to_string();
    if let Some(labelme_path) = &args.labelme {
        let points = deepscol::extract_points(&output3, 0.1).unwrap();
        // Save the points to LabelMe format
        let lm_data = deepscol::create_lm(
            &deepscol::LABELS,
            abs_path.clone(),
            original_image_width,
            original_image_height,
            lm_scale,
            points.as_slice(),
        );
        // Save the LabelMe data to a file
        let labelme_json = serde_json::to_string_pretty(&lm_data)?;

        std::fs::write(labelme_path, labelme_json).expect("Failed to write LabelMe data to file");
        extracted_points = Some(points);
        extracted_lm = Some(lm_data);
    }
    if let Some(html_path) = &args.html {
        let points = extracted_points.unwrap_or_else(|| {
            deepscol::extract_points(&output3, 0.1).expect("Failed to extract points from heatmaps")
        });
        let lm_data = extracted_lm.unwrap_or_else(|| {
            deepscol::create_lm(
                &deepscol::LABELS,
                abs_path,
                original_image_width,
                original_image_height,
                lm_scale,
                points.as_slice(),
            )
        });
        let mut cp = CoronalPointsAndCurve::try_from(lm_data.clone())?;
        *cp.image_metadata_mut() = metadata;
        let lm_data_with_image = LabelMeDataWImage::try_from(lm_data)?;
        let html = deepscol::create_coronal_html(cp, lm_data_with_image)?;
        // Save the HTML to a file
        std::fs::write(html_path, html).expect("Failed to write HTML file");
    }
    Ok(())
}
