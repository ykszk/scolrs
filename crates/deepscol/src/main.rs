use std::vec;

use anyhow::Result;
use ndarray::{s, Array3, Axis, CowArray, NewAxis};
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    // get arguments: first is the model path, second is the image path, and third is the output path
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: {} <model_path> <input_image_path> <output_image_path> [labelme_path]",
            args[0]
        );
        std::process::exit(1);
    }
    let model_path = &args[1];
    let image_path = &args[2];
    let output_path = &args[3];
    log::info!("Model path: {}", model_path);
    log::info!("Image path: {}", image_path);
    log::info!("Output path: {}", output_path);

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
    let original_image_height = image.height();
    log::info!("Image loaded successfully");
    let model_input_height = session.inputs[0].dimensions[2].unwrap() as u32;
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
    if args.len() > 4 {
        let labelme_path = &args[4];
        log::info!(
            "Extracting points and saving to LabelMe format at {}",
            labelme_path
        );
        let points = deepscol::extract_points(&output3, 0.1).unwrap();
        // Save the points to LabelMe format
        let mut lm_data = labelme_rs::LabelMeData {
            imagePath: image_path.to_string(),
            ..Default::default()
        };

        for (i_point, v_point) in points.into_iter().enumerate() {
            if v_point.is_empty() {
                log::warn!("No points found for channel {}", i_point + 1);
                continue; // Skip empty points
            }
            for point in v_point {
                let points = vec![(point.0 as f64, point.1 as f64)];
                let shape = labelme_rs::Shape {
                    label: format!("label{}", i_point + 1),
                    points,
                    shape_type: "point".to_string(),
                    ..Default::default()
                };
                lm_data.shapes.push(shape);
            }
        }
        let scale = original_image_height as f64 / model_input_height as f64;
        lm_data.scale(scale);
        // Save the LabelMe data to a file
        let labelme_json = serde_json::to_string(&lm_data)?;

        std::fs::write(labelme_path, labelme_json).expect("Failed to write LabelMe data to file");
    }
    Ok(())
}
