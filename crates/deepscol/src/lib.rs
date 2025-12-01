extern crate wasm_bindgen;
use dicom_pixeldata::{ConvertOptions, PixelDecoder, VoiLutOption};
use image::DynamicImage;
use imageproc::region_labelling::{connected_components, Connectivity};
use log::debug;
use ndarray::{s, Array3, Array4, Axis, Dim, NewAxis};
use scolrs::{
    draw::{draw_sagittal, ImageOverlay},
    HasImageMetadata, ImageMetadata, PointConfidence, SagittalDraw,
};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
pub type Point = (f32, f32);

extern crate wee_alloc;
// Use `wee_alloc` as the global allocator.
// Default allocator somehow panics in `wrap_in_html`, which can be fixed in the future.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
#[derive(Default, Debug, Clone, Copy)]
pub enum ScanDirection {
    #[default]
    Coronal,
    Sagittal,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(getter_with_clone)]
#[derive(Default, Debug)]
pub struct Settings {
    pub flip_image: bool,
    pub scan_direction: ScanDirection,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl Settings {
    #[wasm_bindgen(constructor)]
    pub fn new(flip_image: bool, scan_direction: ScanDirection) -> Settings {
        Settings {
            flip_image,
            scan_direction,
        }
    }
}

/// Extract landmark point out of the input heatmaps
pub fn extract_points(arr: &ndarray::Array3<f32>, thresh: f32) -> Result<Vec<Vec<Point>>, String> {
    let height = arr.shape()[1];
    let width = arr.shape()[2];
    let ch_axis = Axis(0);
    let mut all_points: Vec<Vec<Point>> = Vec::new();
    for (img_ch, max_point_count) in arr.axis_iter(ch_axis).zip(MAX_POINT_COUNTS.iter()) {
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

        if points_for_ch.len() > *max_point_count {
            log::debug!(
                "Channel has more points than max_point_count: {} > {}",
                points_for_ch.len(),
                max_point_count
            );
            let mut heatmap_values = points_for_ch
                .into_iter()
                .map(|(x, y)| {
                    let val = img_ch[[y as usize, x as usize]];
                    (x, y, val)
                })
                .collect::<Vec<(f32, f32, f32)>>();
            // sort by heatmap value
            heatmap_values
                .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            // take the top `max_point_count` points
            points_for_ch = heatmap_values
                .into_iter()
                .take(*max_point_count)
                .map(|(x, y, _val)| (x, y))
                .collect::<Vec<Point>>();
        }

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

#[wasm_bindgen(getter_with_clone)]
pub struct DecodedImage {
    pub image: String,
    pub width: u32,
    pub height: u32,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn decode_image(encoded: &[u8], settings: Settings) -> Result<DecodedImage, JsValue> {
    let (mut image, metadata) =
        load_image(encoded).map_err(|e| JsValue::from_str(&e.to_string()))?;
    log::debug!("Image metadata: {:?}", metadata);
    if settings.flip_image {
        log::debug!("Flipping image horizontally");
        image = image.fliph();
    }
    let b64_image = labelme_rs::img2base64(&image, labelme_rs::image::ImageFormat::Png)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(DecodedImage {
        image: b64_image,
        width: metadata.width as u32,
        height: metadata.height as u32,
    })
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
            let metadata = ImageMetadata {
                width: img.width() as usize,
                height: img.height() as usize,
                ..ImageMetadata::default()
            };
            Ok((DynamicImage::ImageLuma8(img.to_luma8()), metadata))
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
    model_input_height: u32,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>, anyhow::Error> {
    let image = resize_height(&image, model_input_height);
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
    lm_data.imageWidth = image_width as usize;
    lm_data.imageHeight = image_height as usize;
    // set flag L4, L5, or L6
    let tl_label_index = labels.iter().position(|&l| l == "TL");
    if let Some(tl_index) = tl_label_index {
        let tl_count = points[tl_index].len();
        match tl_count {
            18 => {
                lm_data.flags.insert("L4".to_string(), true);
            }
            19 => {
                lm_data.flags.insert("L5".to_string(), true);
            }
            20 => {
                lm_data.flags.insert("L6".to_string(), true);
            }
            _ => log::warn!("Unexpected TL point count: {}", tl_count),
        };
    }

    lm_data
}

pub fn extract_array_from_output(
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

// '{"TL":19, "TR":19, "BL":18, "BR":18, "Shoulder": 2, "Clavicle": 2, "Pelvis": 2, "Iliac": 2, "FemoralHead": 2, "C7-TL": 1, "C7-TR": 1, "S-TL": 1, "S-TR": 1}'
pub const MAX_POINT_COUNTS: [usize; 13] = [
    20, // TL
    20, // TR
    19, // BL
    19, // BR
    2,  // Shoulder
    2,  // Clavicle
    2,  // Pelvis
    2,  // Iliac
    2,  // FemoralHead
    1,  // C7-TL
    1,  // C7-TR
    1,  // S-TL
    1,  // S-TR
];

pub const RESIZE_PARAM_SIZE: u32 = 1200;

/// Returns the bounding box (min_x, min_y, max_x, max_y) of the true values in a 2D boolean ndarray.
/// Returns None if no true values are found.
fn bounding_box(arr: &ndarray::Array2<bool>) -> Option<(usize, usize, usize, usize)> {
    let mut min_x = arr.shape()[1];
    let mut min_y = arr.shape()[0];
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;

    for ((y, x), &val) in arr.indexed_iter() {
        if val {
            found = true;
            if x < min_x {
                min_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if x > max_x {
                max_x = x;
            }
            if y > max_y {
                max_y = y;
            }
        }
    }
    if found {
        Some((min_x, min_y, max_x, max_y))
    } else {
        None
    }
}

trait MaxAxis {
    type D;
    fn axis_max(&self, axis: Axis) -> ndarray::Array<f32, Dim<Self::D>>;
}

impl MaxAxis for ndarray::ArrayView3<'_, f32> {
    type D = [usize; 2];
    fn axis_max(&self, axis: Axis) -> ndarray::Array<f32, Dim<Self::D>> {
        self.fold_axis(axis, 0.0f32, |acc, &x| acc.max(x))
    }
}

pub fn output3_to_heatmap(
    output3: &ndarray::Array3<f32>,
    input_image_wh: (u32, u32),
) -> DynamicImage {
    let aspect_ratio = input_image_wh.0 as f32 / input_image_wh.1 as f32;
    let heatmap_width = (output3.shape()[1] as f32 * aspect_ratio).round() as usize;
    let output3 = output3.slice(s![.., .., ..heatmap_width]);
    let r = output3.slice(s![0..2, .., ..]).axis_max(Axis(0)); // TL and TR
    let g = output3.slice(s![2..4, .., ..]).axis_max(Axis(0)); // BL and BR
    let b = output3.slice(s![4..9, .., ..]).axis_max(Axis(0)); // Other points
    let heatmap = image::ImageBuffer::from_fn(r.shape()[1] as u32, r.shape()[0] as u32, |x, y| {
        let r_val = (r[[y as usize, x as usize]] * 255.0) as u8;
        let g_val = (g[[y as usize, x as usize]] * 255.0) as u8;
        let b_val = (b[[y as usize, x as usize]] * 255.0) as u8;
        let a_val = r_val.max(g_val).max(b_val);
        image::Rgba([r_val, g_val, b_val, a_val])
    });
    DynamicImage::ImageRgba8(heatmap)
}

pub fn calculate_crop_parameters(
    output3: &ndarray::Array3<f32>,
    thresh: f32,
    original_image_height: u32,
    original_image_width: u32,
    model_input_height: u32,
) -> Option<(usize, usize, usize, usize)> {
    let heatmap2 = output3.view().axis_max(Axis(0));
    let heatmap2_bin = heatmap2.mapv(|x| x > thresh);
    let bbox = bounding_box(&heatmap2_bin);
    if let Some((min_x, min_y, max_x, max_y)) = bbox {
        if (max_x - min_x) < (0.9 * original_image_width as f64) as usize
            || (max_y - min_y) < (0.9 * original_image_height as f64) as usize
        {
            let margin_rate = 0.1;
            let scale = original_image_height as f64 / model_input_height as f64;
            let max_x = scale * max_x as f64;
            let max_y = scale * max_y as f64;
            let min_x = scale * min_x as f64;
            let min_y = scale * min_y as f64;
            let margin_x = original_image_width as f64 * margin_rate;
            let margin_y = original_image_height as f64 * margin_rate;
            let min_x = (min_x - margin_x).max(0.0) as usize;
            let min_y = (min_y - margin_y).max(0.0) as usize;
            let max_x = (max_x + margin_x).min(original_image_width as f64) as usize;
            let max_y = (max_y + margin_y).min(original_image_height as f64) as usize;
            return Some((min_x, min_y, max_x, max_y));
        }
    }
    None
}

// readonly CROP_LABELS="TL TR BL BR Shoulder Clavicle Pelvis Iliac FemoralHead"

pub fn create_coronal_html(
    cp: CoronalPointsAndCurve,
    lm_data_with_image: LabelMeDataWImage,
    overlays: Vec<ImageOverlay>,
) -> Result<String, anyhow::Error> {
    let points_with_image = PointDataWithImage::new(cp, lm_data_with_image);
    let non_draws = [CoronalDraw::SpinalLine, CoronalDraw::Centroids];
    let draws = CoronalDraw::all()
        .into_iter()
        .filter(|d| !non_draws.contains(d))
        .collect::<Vec<_>>();

    let draw_param = scolrs::DrawParam::default();
    let resize_param = labelme_rs::ResizeParam::Size(RESIZE_PARAM_SIZE, RESIZE_PARAM_SIZE);
    let svg_size = None;
    let palettes = scolrs::draw::ColorPalettes::default();
    let non_hide = [
        CoronalDraw::AllPoints,
        CoronalDraw::VertebralLabels,
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
        overlays,
    };
    let document = draw_coronal(draw_args).expect("Failed to draw coronal points and curve");
    let html = wrap_in_html(
        document.to_string(),
        &["g.Component".to_string()],
        "deepscol result".to_string(),
    )?;
    Ok(html)
}

pub fn create_sagittal_html(
    cp: scolrs::SagittalPoints,
    lm_data_with_image: LabelMeDataWImage,
    overlays: Vec<ImageOverlay>,
) -> Result<String, anyhow::Error> {
    let points_with_image = PointDataWithImage::new(cp, lm_data_with_image);
    let draws = scolrs::SagittalDraw::all();
    let draw_param = scolrs::DrawParam::default();
    let resize_param = labelme_rs::ResizeParam::Size(RESIZE_PARAM_SIZE, RESIZE_PARAM_SIZE);
    let svg_size = None;
    let palettes = scolrs::draw::ColorPalettes::default();
    let non_hide = [
        SagittalDraw::AllPoints,
        SagittalDraw::VertebralLabels,
        SagittalDraw::ThoracicKyphosis,
        SagittalDraw::LumbarLordosis,
    ];
    let hide = scolrs::SagittalDraw::all()
        .into_iter()
        .filter(|d| !non_hide.contains(d))
        .collect::<Vec<_>>();
    let draw_args = draw::SagittalDrawArguments {
        image: points_with_image.data_image.image,
        data: points_with_image.data,
        draw_param,
        resize_param: Some(resize_param),
        svg_size,
        palettes,
        draw: &draws,
        hide: &hide,
        overlays,
    };
    let document = draw_sagittal(draw_args).expect("Failed to draw sagittal points and curve");
    let html = wrap_in_html(
        document.to_string(),
        &["g.Component".to_string()],
        "deepscol result".to_string(),
    )?;
    Ok(html)
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct CroppingParams {
    pub min_x: usize,
    pub min_y: usize,
    pub max_x: usize,
    pub max_y: usize,
}

#[wasm_bindgen]
impl CroppingParams {
    pub fn copy(&self) -> CroppingParams {
        *self
    }
}

pub struct ResultHtmlArguments {
    pub points: Vec<Vec<(f32, f32)>>,
    pub model_input_height: u32,
    pub image: DynamicImage,
    pub metadata: ImageMetadata,
    pub scan_direction: ScanDirection,
    pub cropping_params: Option<CroppingParams>,
    pub heatmap: DynamicImage,
    pub model_output: Array3<f32>,
}

pub fn calc_overlay_params(
    original_image_width: u32,
    original_image_height: u32,
    cropping_params: Option<(usize, usize, usize, usize)>,
    ol_img_width_height: (u32, u32),
    model_input_height: u32,
) -> ((f64, f64), (f64, f64)) {
    let resize_param = labelme_rs::ResizeParam::Size(RESIZE_PARAM_SIZE, RESIZE_PARAM_SIZE);
    let orig_svg_scale = resize_param.scale(original_image_width, original_image_height);
    if let Some((min_x, min_y, _max_x, max_y)) = cropping_params {
        let crop_height = max_y - min_y;
        let crop_to_input_scale = model_input_height as f64 / crop_height as f64;
        let scale = orig_svg_scale / crop_to_input_scale;
        let x_y = (min_x as f64 * orig_svg_scale, min_y as f64 * orig_svg_scale);
        let width_height = (
            ol_img_width_height.0 as f64 * scale,
            ol_img_width_height.1 as f64 * scale,
        );
        (x_y, width_height)
    } else {
        (
            (0.0, 0.0),
            (
                original_image_width as f64 * orig_svg_scale,
                original_image_height as f64 * orig_svg_scale,
            ),
        )
    }
}

pub fn create_result_html(args: ResultHtmlArguments) -> Result<String, anyhow::Error> {
    let ResultHtmlArguments {
        points,
        model_input_height,
        image,
        metadata,
        scan_direction,
        cropping_params,
        heatmap,
        model_output,
    } = args;
    log::debug!("Extracted points: {:?}", points);
    let original_image_width = image.width();
    let original_image_height = image.height();
    let lm_scale = if let Some(cropping_params) = &cropping_params {
        let crop_height = cropping_params.max_y - cropping_params.min_y;
        log::debug!(
            "Cropped image height: {}, original image height: {}",
            crop_height,
            original_image_height
        );
        crop_height as f64 / model_input_height as f64
    } else {
        original_image_height as f64 / model_input_height as f64
    };
    log::debug!("LM scale: {}", lm_scale);
    let mut lm_data = create_lm(
        &LABELS,
        "".to_string(),
        original_image_width,
        original_image_height,
        lm_scale,
        points.as_slice(),
    );
    if let Some(crop_params) = &cropping_params {
        log::debug!(
            "Shifting LabelMeData by min_x: {}, min_y: {}",
            crop_params.min_x,
            crop_params.min_y
        );
        lm_data.shift(crop_params.min_x as f64, crop_params.min_y as f64);
    }

    let (x_y, width_height) = calc_overlay_params(
        original_image_width,
        original_image_height,
        cropping_params.map(|cp| (cp.min_x, cp.min_y, cp.max_x, cp.max_y)),
        (heatmap.width(), heatmap.height()),
        model_input_height,
    );
    let overlays = vec![ImageOverlay::new(
        "Heatmap".to_string(),
        "Heatmap".to_string(),
        Some(
            "Red: Top left and top right. Green: Bottom left and bottom right. Blue: Other points"
                .to_string(),
        ),
        heatmap,
        x_y,
        width_height,
    )];

    let crop_adjusted_lm_data = if let Some(cropping_params) = &cropping_params {
        // shift
        log::debug!(
            "Adjusting LabelMeData for cropping params: {:?}",
            cropping_params
        );
        let mut lm_data_adjusted = lm_data.clone();
        lm_data_adjusted.shift(
            -(cropping_params.min_x as f64),
            -(cropping_params.min_y as f64),
        );
        // scale
        let crop_height = cropping_params.max_y - cropping_params.min_y;
        let scale = model_input_height as f64 / crop_height as f64;
        log::debug!("Scaling LabelMeData by scale: {}", scale);
        lm_data_adjusted.scale(scale);
        lm_data_adjusted
    } else {
        let scale = model_input_height as f64 / original_image_height as f64;
        log::debug!("Scaling LabelMeData by scale: {}", scale);
        let mut lm_data_adjusted = lm_data.clone();
        lm_data_adjusted.scale(scale);
        lm_data_adjusted
    };

    let model_output_f64 = model_output.mapv(|x| x as f64);
    let ch_last_output = model_output_f64.permuted_axes([1, 2, 0]);

    match scan_direction {
        ScanDirection::Coronal => {
            log::debug!("Creating CoronalPointsAndCurve");
            let mut cp = CoronalPointsAndCurve::try_from(lm_data.clone())?;
            let crop_adjusted_cp = CoronalPointsAndCurve::try_from(crop_adjusted_lm_data)?;
            *cp.image_metadata_mut() = metadata;
            *cp.coronal_points.get_confidence_mut() = Some(
                crop_adjusted_cp
                    .coronal_points
                    .extract_point_confidence(ch_last_output.view()),
            );
            let lm_data_with_image = LabelMeDataWImage::new(lm_data, image);
            create_coronal_html(cp, lm_data_with_image, overlays)
        }
        ScanDirection::Sagittal => {
            log::debug!("Creating SagittalPoints");
            let mut cp: scolrs::SagittalPoints = scolrs::SagittalPoints::try_from(lm_data.clone())?;
            let crop_adjusted_cp = scolrs::SagittalPoints::try_from(crop_adjusted_lm_data)?;
            *cp.image_metadata_mut() = metadata;
            *cp.get_confidence_mut() =
                Some(crop_adjusted_cp.extract_point_confidence(ch_last_output.view()));
            let lm_data_with_image = LabelMeDataWImage::new(lm_data, image);
            create_sagittal_html(cp, lm_data_with_image, overlays)
        }
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen(getter_with_clone)]
pub struct Arr3 {
    pub arr: js_sys::Float32Array,
    pub d1: usize,
    pub d2: usize,
    pub d3: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn create_input_array(
    bytes: &[u8],
    model_input_height: u32,
    settings: Settings,
    cropping_params: Option<CroppingParams>,
) -> Result<Arr3, JsValue> {
    let (image, _metadata) = load_image(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut image = if settings.flip_image {
        log::debug!("Flipping image horizontally");
        image.fliph()
    } else {
        image
    };
    if let Some(cp) = cropping_params {
        let cropped_image = image.crop(
            cp.min_x as u32,
            cp.min_y as u32,
            (cp.max_x - cp.min_x) as u32,
            (cp.max_y - cp.min_y) as u32,
        );
        log::debug!(
            "Cropped image size: {}x{}",
            cropped_image.width(),
            cropped_image.height()
        );
        image = cropped_image;
    }
    let arr4 =
        to_model_input(image, model_input_height).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(Arr3 {
        arr: js_sys::Float32Array::from(arr4.as_slice().unwrap()),
        d1: arr4.shape()[1] as usize,
        d2: arr4.shape()[2] as usize,
        d3: arr4.shape()[3] as usize,
    })
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn calculate_crop_parameters_wasm(
    raw_output: &[f32],
    tensor_dims: js_sys::Uint32Array,
    original_image_width: u32,
    original_image_height: u32,
) -> Result<Option<CroppingParams>, JsValue> {
    let output3 = Array3::from_shape_vec(
        (
            tensor_dims.get_index(1) as usize,
            tensor_dims.get_index(2) as usize,
            tensor_dims.get_index(3) as usize,
        ),
        raw_output.to_vec(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // apply sigmoid
    let output3 = output3.mapv(|x| 1.0 / (1.0 + (-x).exp()));

    let params = calculate_crop_parameters(
        &output3,
        0.05,
        original_image_height,
        original_image_width,
        tensor_dims.get_index(2),
    )
    .map(|(min_x, min_y, max_x, max_y)| CroppingParams {
        min_x,
        min_y,
        max_x,
        max_y,
    });
    Ok(params)
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn process_output(
    encoded: &[u8],
    raw_output: &[f32],
    tensor_dims: js_sys::Uint32Array,
    settings: Settings,
    cropping_params: Option<CroppingParams>,
) -> Result<String, JsValue> {
    let (image, metadata) = load_image(encoded).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let image = if settings.flip_image {
        log::debug!("Flipping image horizontally");
        image.fliph()
    } else {
        image
    };

    let output3 = Array3::from_shape_vec(
        (
            tensor_dims.get_index(1) as usize,
            tensor_dims.get_index(2) as usize,
            tensor_dims.get_index(3) as usize,
        ),
        raw_output.to_vec(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // apply sigmoid
    let output3 = output3.mapv(|x| 1.0 / (1.0 + (-x).exp()));

    let mut input_image_wh = (image.width(), image.height());
    if let Some(cp) = &cropping_params {
        input_image_wh = ((cp.max_x - cp.min_x) as u32, (cp.max_y - cp.min_y) as u32);
    }

    // create rgb heatmap using output3
    let heatmap = output3_to_heatmap(&output3, input_image_wh);

    log::debug!("Output tensor shape: {:?}", output3.shape());
    log::debug!("Cropping parameters: {:?}", cropping_params);

    let model_input_height = output3.shape()[1] as u32;
    let points = extract_points(&output3, 0.1)
        .map_err(|e| JsValue::from_str(&format!("Failed to extract points: {}", e)))?;
    let result_args = ResultHtmlArguments {
        points,
        model_input_height,
        image,
        metadata,
        scan_direction: settings.scan_direction,
        cropping_params,
        heatmap,
        model_output: output3,
    };
    let html = create_result_html(result_args)
        .map_err(|e| JsValue::from_str(&format!("Failed to create HTML: {}", e)))?;
    Ok(html)
}

#[cfg(feature = "wasm")]
extern crate console_error_panic_hook;
#[cfg(feature = "wasm")]
use std::panic;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn start() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    wasm_logger::init(wasm_logger::Config::default());
    debug!("logger initialized");
}
