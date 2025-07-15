use dicom_pixeldata::{ConvertOptions, PixelDecoder, VoiLutOption};
use image::DynamicImage;
use imageproc::region_labelling::{connected_components, Connectivity};
use log::debug;
use ndarray::{s, Array3, Array4, Axis, NewAxis};
use scolrs::{HasImageMetadata, ImageMetadata};
use std::collections::HashMap;
pub type Point = (f32, f32);

/// Extract landmark point out of the input heatmaps
pub fn extract_points(arr: &ndarray::Array3<f32>, thresh: f32) -> Result<Vec<Vec<Point>>, String> {
    let height = arr.shape()[1];
    let width = arr.shape()[2];
    let ch_axis = Axis(0);
    let mut all_points: Vec<Vec<Point>> = Vec::new();
    for img_ch in arr.axis_iter(ch_axis) {
        let bin_arr = img_ch.mapv(|v| if v > thresh { 1u8 } else { 0u8 });
        let bin_img = image::GrayImage::from_raw(
            width as _,
            height as _,
            bin_arr.into_raw_vec_and_offset().0,
        )
        .unwrap();
        let cced = connected_components(&bin_img, Connectivity::Four, image::Luma([0u8]));
        let n_cc = *cced.iter().max().unwrap();
        debug!("# of CC is {}", n_cc);

        let mut local_maximas: HashMap<u32, (f32, usize, usize)> = HashMap::new();
        // traverse each pixel in cced with the coordinates
        for (x, y, &cc_val) in cced.enumerate_pixels() {
            let cc_val = cc_val.0[0];
            if cc_val == 0 {
                continue; // Skip background
            }
            let (y, x) = (y as usize, x as usize);

            let val = img_ch[[y, x]];
            let local_maxima = local_maximas.get(&cc_val);
            if let Some((max_val, _max_y, _max_x)) = local_maxima {
                if val > *max_val {
                    // Update local maxima if current pixel is greater
                    local_maximas.insert(cc_val, (val, y, x));
                } else {
                    // If current pixel is not greater, skip it
                    continue;
                }
            } else {
                // If no local maxima exists for this cc_val, set current pixel as local maxima
                local_maximas.insert(cc_val, (val, y, x));
            }
        }
        let mut points_for_ch: Vec<Point> = local_maximas
            .into_iter()
            .map(|(_, (_val, y, x))| {
                // Convert to (x, y) coordinates
                (x as f32, y as f32)
            })
            .collect::<Vec<Point>>();
        // sort by y and then by x
        points_for_ch.sort_by(|a, b| {
            if a.1 == b.1 {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        all_points.push(points_for_ch);
    }

    Ok(all_points)
}

pub fn load_dicom_from_u8(
    bytes: &[u8],
) -> Result<(image::DynamicImage, ImageMetadata), Box<dyn std::error::Error>> {
    assert!(bytes.len() > 128);
    let wo_preamble = &bytes[128..]; // skip preamble
    let obj = dicom_object::from_reader(wo_preamble)?;
    let options = ConvertOptions::new()
        .with_voi_lut(VoiLutOption::Normalize)
        .force_8bit();
    let dynamic_image = obj
        .decode_pixel_data()?
        .to_dynamic_image_with_options(0, &options)?;
    let metadata = ImageMetadata::try_from_dicom(&obj, "".to_string())?;
    Ok((dynamic_image, metadata))
}

/// load dicom or png/jpeg image
pub fn load_image(
    raw_bytes: &[u8],
) -> Result<(image::DynamicImage, ImageMetadata), image::ImageError> {
    let dcm_img = load_dicom_from_u8(raw_bytes);
    match dcm_img {
        Ok(img) => {
            log::debug!("Dicom image has been loaded successfully.");
            Ok(img)
        }
        Err(e) => {
            debug!("Tried to read the file as dicom but failed: {}", e);
            let img = image::load_from_memory(raw_bytes)?;
            Ok((
                DynamicImage::ImageLuma8(img.to_luma8()),
                ImageMetadata::default(),
            ))
        }
    }
}

pub fn resize_height(image: &DynamicImage, target_height: u32) -> DynamicImage {
    let width = image.width();
    let height = image.height();
    let resized_width: u32 = (width as f32 * (target_height as f32 / height as f32)) as u32;
    image.thumbnail(resized_width + 1, target_height)
}

pub fn pad_width(image: &Array3<f32>, multiple_of: u32) -> Array3<f32> {
    let padded_width = image.shape()[1].div_ceil(multiple_of as usize) * multiple_of as usize;
    let mut padded_image: Array3<f32> =
        Array3::zeros((image.shape()[0], padded_width, image.shape()[2]));
    padded_image
        .slice_mut(s![.., ..image.shape()[1], ..])
        .assign(image);
    padded_image
}

pub fn to_model_input(
    image: image::DynamicImage,
    model_input_height: i64,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>, anyhow::Error> {
    let image = resize_height(&image, model_input_height as u32);
    let image = Array3::from_shape_vec(
        (image.height() as usize, image.width() as usize, 1),
        image.to_luma8().into_raw(),
    )?;
    let image: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>> =
        image.mapv(|x| x as f32 / 255.0);
    log::debug!("Image converted to tensor shape: {:?}", image.shape());
    let image = pad_width(&image, 256);
    let image: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>> =
        image.permuted_axes([2, 0, 1]);
    let arr4: Array4<f32> = image.slice(s![NewAxis, .., .., ..]).to_owned();
    Ok(arr4)
}

pub fn create_lm(
    labels: &[&str],
    image_path: String,
    image_width: u32,
    image_height: u32,
    scale: f64,
    points: &[Vec<(f32, f32)>],
) -> labelme_rs::LabelMeData {
    let mut lm_data = labelme_rs::LabelMeData {
        imagePath: image_path,
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
            let label = labels[i_point].to_string();
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
    lm_data
}

pub fn process_output(
    outputs: ort::session::SessionOutputs<'_>,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>> {
    // convert to ndarray
    let output = &outputs[0];
    let output_array = output.try_extract_array::<f32>().unwrap().view().to_owned();
    let output3 = output_array
        .index_axis_move(Axis(0), 0)
        .into_dimensionality::<ndarray::Ix3>()
        .unwrap();
    log::debug!("Output tensor shape: {:?}", output3.shape());
    // apply sigmoid

    output3.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

use labelme_rs::LabelMeDataWImage;
use scolrs::{
    draw::{self, draw_coronal, wrap_in_html},
    CoronalDraw, CoronalPointsAndCurve, MeasureAndDraw, PointDataWithImage,
};

pub const LABELS: [&str; 13] = [
    "TL",
    "TR",
    "BL",
    "BR",
    "Shoulder",
    "Clavicle",
    "Pelvis",
    "Iliac",
    "FemoralHead",
    "C7-TL",
    "C7-TR",
    "S-TL",
    "S-TR",
];

pub fn create_html(
    cp: CoronalPointsAndCurve,
    lm_data_with_image: LabelMeDataWImage,
) -> Result<String, anyhow::Error> {
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
    let document = draw_coronal(draw_args).expect("Failed to draw coronal points and curve");
    let html = wrap_in_html(
        document.to_string(),
        &["g.Component".to_string()],
        "deepscol result".to_string(),
    )?;
    Ok(html)
}

#[wasm_bindgen]
pub fn run(bytes: &[u8], shape: &[u32]) -> String {
    log::debug!("Create session");
    let builder = ort::session::Session::builder()
        .expect("Cannot create Session builder.")
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .expect("Cannot optimize graph.")
        .with_parallel_execution(true)
        .expect("Cannot activate parallel execution.")
        .with_intra_threads(2)
        .expect("Cannot set intra thread count.")
        .with_inter_threads(1)
        .expect("Cannot set inter thread count.");
    let mut session = builder
        .commit_from_memory(include_bytes!("../models/spine_mobileone_s1.onnx"))
        .expect("Cannot load model from memory.");

    log::debug!("Create array from bytes with shape {:?}", shape);
    let (image, metadata) = load_image(bytes).unwrap();
    let original_image_width = image.width();
    let original_image_height = image.height();
    let model_input_height = session.inputs[0].input_type.tensor_shape().unwrap()[2];
    let arr4 = to_model_input(image, model_input_height).unwrap();
    let input = ort::value::Tensor::from_array(arr4).unwrap();
    log::debug!(
        "Image converted to tensor successfully with shape: {:?}",
        input.shape()
    );
    // Run the model
    let inputs = ort::inputs!["modelInput" => input];
    log::info!("Running model");
    let outputs: ort::session::SessionOutputs = session.run(inputs).expect("Model run failed");
    log::info!("Model run completed");
    let output3 = process_output(outputs);
    let points = extract_points(&output3, 0.1).unwrap();
    let lm_scale = original_image_height as f64 / model_input_height as f64;
    let lm_data = create_lm(
        &LABELS,
        "".to_string(),
        original_image_width,
        original_image_height,
        lm_scale,
        points.as_slice(),
    );
    let mut cp = CoronalPointsAndCurve::try_from(lm_data.clone()).unwrap();
    *cp.image_metadata_mut() = metadata;
    let lm_data_with_image = LabelMeDataWImage::try_from(lm_data).unwrap();
    create_html(cp, lm_data_with_image).unwrap()
}

extern crate console_error_panic_hook;
extern crate wasm_bindgen;
use std::panic;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn start() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    wasm_logger::init(wasm_logger::Config::default());
    debug!("logger initialized");
    ort::set_api(ort_tract::api());
    debug!("ort api set to tract");
}
