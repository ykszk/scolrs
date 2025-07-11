use std::{ops::Deref, vec};

use anyhow::Result;
use clap::Parser;
use image::DynamicImage;
use labelme_rs::LabelMeDataWImage;
use ndarray::{s, Array3, Axis, CowArray, NewAxis};
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use scolrs::{
    draw::{self, draw_coronal, wrap_in_html},
    CoronalDraw, CoronalPointsAndCurve, HasImageMetadata, MeasureAndDraw, PointDataWithImage,
};
use std::path::PathBuf;

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

fn resize_height(image: &DynamicImage, target_height: u32) -> DynamicImage {
    let width = image.width();
    let height = image.height();
    let resized_width: u32 = (width as f32 * (target_height as f32 / height as f32)) as u32;
    image.thumbnail(resized_width + 1, target_height)
}

fn pad_width(image: &Array3<f32>, multiple_of: u32) -> Array3<f32> {
    let padded_width = image.shape()[1].div_ceil(multiple_of as usize) * multiple_of as usize;
    let mut padded_image: Array3<f32> =
        Array3::zeros((image.shape()[0], padded_width, image.shape()[2]));
    padded_image
        .slice_mut(s![.., ..image.shape()[1], ..])
        .assign(image);
    padded_image
}

enum MySession<'a> {
    Session(ort::Session),
    InMemorySession(ort::InMemorySession<'a>),
}

impl MySession<'_> {
    fn get(&self) -> &ort::Session {
        match self {
            MySession::Session(sess) => sess,
            MySession::InMemorySession(sess) => sess.deref(),
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let model_path = &args.model;
    let image_path = &args.input_image;
    let output_path = &args.output_image;

    // Load the model
    let environment = Environment::builder().build()?.into_arc();
    let session: MySession = if let Some(model_path) = model_path {
        let sess = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_file(model_path)?;
        MySession::Session(sess)
    } else {
        let sess = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_model_from_memory(include_bytes!(
                "../../../tests/data/models/spine_mobileone_s1.onnx"
            ))?;
        MySession::InMemorySession(sess)
    };
    log::info!("Model loaded successfully");
    log::debug!(
        "model input dimensions {:?}",
        session.get().inputs[0].dimensions
    );

    let (image, metadata) = deepscol::load_image(&std::fs::read(image_path)?)?;
    let original_image_width = image.width();
    let original_image_height = image.height();
    log::info!(
        "Image with shape w x h {:?} loaded successfully",
        (original_image_width, original_image_height)
    );
    let model_input_height = session.get().inputs[0].dimensions[2].unwrap();
    let image = resize_height(&image, model_input_height);
    // Convert the image to a tensor
    let image = Array3::from_shape_vec(
        (image.height() as usize, image.width() as usize, 1),
        image.to_luma8().into_raw(),
    )?;
    // convert to float32
    let image = image.mapv(|x| x as f32 / 255.0);
    log::debug!("Image converted to tensor shape: {:?}", image.shape());
    // pad image width to be divisible by 256
    let image = pad_width(&image, 256);
    // to channel first
    let image = image.permuted_axes([2, 0, 1]);

    let input = CowArray::from(image.slice(s![NewAxis, .., .., ..]).into_dyn());
    log::debug!(
        "Image converted to tensor successfully with shape: {:?}",
        input.shape()
    );
    // Run the model
    let inputs = vec![Value::from_array(session.get().allocator(), &input)?];
    log::info!("Running model");
    let mut outputs: Vec<Value> = session.get().run(inputs)?;
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
    let n_channels = output3.shape()[0];
    let labels = if let Some(labels_path) = &args.labels {
        let labels = std::fs::read_to_string(labels_path)
            .expect("Failed to read labels file")
            .lines()
            .map(|line| line.trim().to_string())
            .collect::<Vec<String>>();
        if labels.len() != n_channels {
            return Err(anyhow::anyhow!(
                "Number of labels does not match number of channels. Expected {} labels, got {}",
                n_channels,
                labels.len()
            ));
        }
        labels
    } else {
        (0..n_channels)
            .map(|i| format!("channel{}", i + 1))
            .collect::<Vec<String>>()
    };
    if let Some(labelme_path) = &args.labelme {
        let points = deepscol::extract_points(&output3, 0.1).unwrap();
        // Save the points to LabelMe format
        let lm_data = create_lm(
            &labels,
            image_path,
            original_image_width,
            original_image_height,
            lm_scale,
            points.as_slice(),
        )?;
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
            create_lm(
                &labels,
                image_path,
                original_image_width,
                original_image_height,
                lm_scale,
                points.as_slice(),
            )
            .expect("Failed to create LabelMe data")
        });
        let mut cp = CoronalPointsAndCurve::try_from(lm_data.clone())?;
        *cp.image_metadata_mut() = metadata;
        let lm_data_with_image = LabelMeDataWImage::try_from(lm_data)?;
        let points_with_image = PointDataWithImage::new(cp, lm_data_with_image);
        let draws = CoronalDraw::all();
        let draw_param = scolrs::DrawParam::default();
        let resize_param = labelme_rs::ResizeParam::Size(800, 800);
        let svg_size = None;
        let palettes = scolrs::draw::ColorPalettes::default();
        let non_hide = [
            CoronalDraw::VertebralLabels,
            CoronalDraw::VertebralPoints,
            CoronalDraw::CobbPT,
            CoronalDraw::CobbMT,
            CoronalDraw::CobbTLL,
        ];
        let hide = CoronalDraw::all()
            .into_iter()
            .filter(|d| !non_hide.contains(d))
            .collect::<Vec<_>>();
        let draw_args = draw::CoronalDrawArguments {
            image: points_with_image.data_image.image,
            data: points_with_image.data,
            draw_param,
            resize_param: Some(resize_param),
            svg_size,
            palettes,
            draw: &draws,
            hide: &hide,
        };
        // CoronalPointsAndCurve::draw()
        let document = draw_coronal(draw_args).expect("Failed to draw coronal points and curve");
        let html = wrap_in_html(
            document.to_string(),
            &["g.Component".to_string()],
            "deepscol result".to_string(),
        )?;
        // Save the HTML to a file
        std::fs::write(html_path, html).expect("Failed to write HTML file");
    }
    Ok(())
}

fn create_lm(
    labels: &[String],
    image_path: &PathBuf,
    image_width: u32,
    image_height: u32,
    scale: f64,
    points: &[Vec<(f32, f32)>],
) -> Result<labelme_rs::LabelMeData> {
    let mut lm_data = labelme_rs::LabelMeData {
        imagePath: std::path::absolute(image_path)?
            .to_string_lossy()
            .to_string(),
        imageHeight: image_height as usize,
        imageWidth: image_width as usize,
        ..Default::default()
    };

    for (i_point, v_point) in points.iter().enumerate() {
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
    lm_data.scale(scale);
    Ok(lm_data)
}
