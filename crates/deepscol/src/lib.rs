extern crate wasm_bindgen;
use dicom_pixeldata::{ConvertOptions, PixelDecoder, VoiLutOption};
use image::DynamicImage;
use imageproc::region_labelling::{connected_components, Connectivity};
use log::debug;
use ndarray::{s, Array3, Array4, Axis, NewAxis};
use scolrs::{
    draw::{
        draw_generic, draw_sagittal,
        generic::{GenericDraw, GenericPoints},
        AxisMax, ImageOverlay,
    },
    HasImageMetadata, ImageMetadata, PointConfidence, SagittalDraw, SagittalPoints,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
pub type Point = (f32, f32);

#[wasm_bindgen]
#[derive(Default, Debug, Clone, Copy)]
pub enum ScanDirection {
    #[default]
    Coronal,
    Sagittal,
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
    points: &[Vec<(f32, f32)>],
) -> labelme_rs::LabelMeData {
    let mut lm_data = labelme_rs::LabelMeData {
        version: scolrs::VERSION.to_string(),
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
            _ => {
                log::warn!("Unexpected TL point count: {}", tl_count);
                lm_data.flags.insert("L?".to_string(), true);
            }
        };
    }

    lm_data
}

pub fn create_scaled_lm(
    labels: &[&str],
    image_path: String,
    image_width: u32,
    image_height: u32,
    scale: f64,
    points: &[Vec<(f32, f32)>],
) -> LabelMeData {
    let mut lm_data = create_lm(labels, image_path, image_width, image_height, points);
    lm_data.scale(scale);
    lm_data.imageWidth = image_width as usize;
    lm_data.imageHeight = image_height as usize;

    lm_data
}

pub fn create_lm_from_cropped(
    labels: &[&str],
    image_path: String,
    image_width: u32,
    image_height: u32,
    cropping_params: &CroppingParams,
    model_input_height: u32,
    points: &[Vec<(f32, f32)>],
) -> LabelMeData {
    let CroppingParams {
        min_x,
        min_y,
        max_x: _,
        max_y,
    } = cropping_params;
    let lm_scale = (max_y - min_y) as f64 / model_input_height as f64;
    let mut lm_data = create_scaled_lm(
        labels,
        image_path,
        image_width,
        image_height,
        lm_scale,
        points,
    );
    lm_data.shift(*min_x as f64, *min_y as f64);
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

use labelme_rs::{svg::node::element::SVG, LabelMeData, LabelMeDataWImage};
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CropConfig {
    /// Threshold for heatmap binarization
    pub thresh: f32,
    pub crop_min_coverage: f64,
    pub margin_x_rate: f64,
    pub margin_y_rate: f64,
}

impl Default for CropConfig {
    fn default() -> Self {
        Self {
            thresh: 0.05,
            crop_min_coverage: 0.9,
            margin_x_rate: 0.1,
            margin_y_rate: 0.1,
        }
    }
}

pub fn calculate_crop_parameters(
    output3: &ndarray::Array3<f32>,
    config: &CropConfig,
    original_image_height: u32,
    original_image_width: u32,
    model_input_height: u32,
) -> Option<CroppingParams> {
    let heatmap2 = output3.view().axis_max(Axis(0));
    let heatmap2_bin = heatmap2.mapv(|x| x > config.thresh);
    let bbox = bounding_box(&heatmap2_bin);
    if let Some((min_x, min_y, max_x, max_y)) = bbox {
        if (max_x - min_x) < (config.crop_min_coverage * original_image_width as f64) as usize
            || (max_y - min_y) < (config.crop_min_coverage * original_image_height as f64) as usize
        {
            let scale = original_image_height as f64 / model_input_height as f64;
            let max_x = scale * max_x as f64;
            let max_y = scale * max_y as f64;
            let min_x = scale * min_x as f64;
            let min_y = scale * min_y as f64;
            let margin_x = original_image_width as f64 * config.margin_x_rate;
            let margin_y = original_image_height as f64 * config.margin_y_rate;
            let min_x = (min_x - margin_x).max(0.0) as usize;
            let min_y = (min_y - margin_y).max(0.0) as usize;
            let max_x = (max_x + margin_x).min(original_image_width as f64) as usize;
            let max_y = (max_y + margin_y).min(original_image_height as f64) as usize;
            return Some(CroppingParams {
                min_x,
                min_y,
                max_x,
                max_y,
            });
        }
    }
    None
}

pub fn create_generic_svg(
    gp: GenericPoints,
    lm_data_with_image: LabelMeDataWImage,
    overlays: Vec<ImageOverlay>,
) -> Result<SVG, anyhow::Error> {
    let points_with_image = PointDataWithImage::new(gp, lm_data_with_image);
    let draws = vec![GenericDraw::AllPoints];

    let draw_param = scolrs::draw::DrawParam::default();
    let resize_param = labelme_rs::ResizeParam::Size(RESIZE_PARAM_SIZE, RESIZE_PARAM_SIZE);
    let svg_size = None;
    let palettes = scolrs::draw::ColorPalettes::default();
    let hide = vec![GenericDraw::AllPoints];
    let draw_args = draw::GenericDrawArguments {
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
    let document = draw_generic(draw_args).expect("Failed to draw generic points and curve");
    Ok(document)
}

pub fn create_coronal_svg(
    cp: CoronalPointsAndCurve,
    lm_data_with_image: LabelMeDataWImage,
    overlays: Vec<ImageOverlay>,
) -> Result<SVG, anyhow::Error> {
    let points_with_image = PointDataWithImage::new(cp, lm_data_with_image);
    let non_draws = [CoronalDraw::SpinalLine, CoronalDraw::Centroids];
    let draws = CoronalDraw::all()
        .into_iter()
        .filter(|d| !non_draws.contains(d))
        .collect::<Vec<_>>();

    let draw_param = scolrs::draw::DrawParam::default();
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
    Ok(document)
}

pub fn create_sagittal_svg(
    cp: SagittalPoints,
    lm_data_with_image: LabelMeDataWImage,
    overlays: Vec<ImageOverlay>,
) -> Result<SVG, anyhow::Error> {
    let points_with_image = PointDataWithImage::new(cp, lm_data_with_image);
    let draws = scolrs::SagittalDraw::all();
    let draw_param = scolrs::draw::DrawParam::default();
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
    Ok(document)
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
    pub image: DynamicImage,
    pub metadata: ImageMetadata,
    pub scan_direction: ScanDirection,
    pub cropping_params: Option<CroppingParams>,
    pub heatmap: DynamicImage,
    pub model_output: Array3<f32>,
    pub title: String,
}

#[deprecated(note = "Use ModelIO::overlay_params instead")]
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

pub struct OriginalIO {
    output3: Array3<f32>,
    /// LabelMeData in the original image coordinate system
    /// image width and height correspond to original image
    lm_data: LabelMeData,
}

impl OriginalIO {
    pub fn new(
        original_image_wh: (u32, u32),
        output3: Array3<f32>,
        points: Vec<Vec<(f32, f32)>>,
    ) -> Self {
        let input_height = output3.shape()[1] as u32;
        let lm_scale = original_image_wh.1 as f64 / input_height as f64;
        let lm_data = create_scaled_lm(
            &LABELS,
            "".to_string(),
            original_image_wh.0,
            original_image_wh.1,
            lm_scale,
            points.as_slice(),
        );
        Self { output3, lm_data }
    }
}

pub struct CroppedIO {
    input_height: u32,
    output3: Array3<f32>,
    cropping_params: CroppingParams,
    /// LabelMeData in the original image coordinate system
    lm_data: LabelMeData,
    /// LabelMeData in the cropped (heatmap) image coordinate system
    /// image width and height correspond to output3 shape
    cropped_lm_data: LabelMeData,
}

impl CroppedIO {
    pub fn new(
        original_image_wh: (u32, u32),
        output3: Array3<f32>,
        points: Vec<Vec<(f32, f32)>>,
        cropping_params: CroppingParams,
    ) -> Self {
        let CroppingParams {
            min_x,
            min_y,
            max_x: _,
            max_y,
        } = cropping_params;

        let cropped_lm_data = create_lm(
            &LABELS,
            "".to_string(),
            output3.shape()[2] as u32,
            output3.shape()[1] as u32,
            points.as_slice(),
        );

        let mut lm_data = cropped_lm_data.clone();
        let input_height = output3.shape()[1] as u32;
        let lm_scale = (max_y - min_y) as f64 / input_height as f64;
        lm_data.scale(lm_scale);
        lm_data.shift(min_x as f64, min_y as f64);
        lm_data.imageWidth = original_image_wh.0 as usize;
        lm_data.imageHeight = original_image_wh.1 as usize;

        Self {
            input_height,
            output3,
            cropping_params,
            lm_data,
            cropped_lm_data,
        }
    }
}

pub enum ModelIO {
    Original(Box<OriginalIO>),
    Cropped(Box<CroppedIO>),
}

impl ModelIO {
    pub fn output3(&self) -> &Array3<f32> {
        match self {
            ModelIO::Original(original) => &original.output3,
            ModelIO::Cropped(cropped) => &cropped.output3,
        }
    }
    pub fn lm_data(&self) -> &LabelMeData {
        match self {
            ModelIO::Original(original) => &original.lm_data,
            ModelIO::Cropped(cropped) => &cropped.lm_data,
        }
    }
    pub fn heatmap_lm_data(&self) -> &LabelMeData {
        match self {
            ModelIO::Original(original) => &original.lm_data,
            ModelIO::Cropped(cropped) => &cropped.cropped_lm_data,
        }
    }
    pub fn set_heatmap_lm_data(&mut self, lm_data: LabelMeData) {
        match self {
            ModelIO::Original(original) => {
                original.lm_data = lm_data;
            }
            ModelIO::Cropped(cropped) => {
                cropped.cropped_lm_data = lm_data.clone();
                // scaled lm data as well
                let CroppingParams {
                    min_x,
                    min_y,
                    max_x: _,
                    max_y,
                } = cropped.cropping_params;
                let lm_scale = (max_y - min_y) as f64 / cropped.input_height as f64;
                let image_width = cropped.lm_data.imageWidth;
                let image_height = cropped.lm_data.imageHeight;
                cropped.lm_data = lm_data;
                cropped.lm_data.scale(lm_scale);
                cropped.lm_data.shift(min_x as f64, min_y as f64);
                cropped.lm_data.imageWidth = image_width;
                cropped.lm_data.imageHeight = image_height;
            }
        }
    }
    pub fn overlay_params(
        &self,
        original_image_width: u32,
        original_image_height: u32,
        ol_img_width_height: (u32, u32),
    ) -> ((f64, f64), (f64, f64)) {
        let resize_param = labelme_rs::ResizeParam::Size(RESIZE_PARAM_SIZE, RESIZE_PARAM_SIZE);
        let orig_svg_scale = resize_param.scale(original_image_width, original_image_height);

        match self {
            ModelIO::Original(_) => (
                (0.0, 0.0),
                (
                    original_image_width as f64 * orig_svg_scale,
                    original_image_height as f64 * orig_svg_scale,
                ),
            ),
            ModelIO::Cropped(cropped) => {
                let CroppingParams {
                    min_x,
                    min_y,
                    max_x: _,
                    max_y,
                } = cropped.cropping_params;
                let crop_height = max_y - min_y;
                let crop_to_input_scale = cropped.input_height as f64 / crop_height as f64;
                let scale = orig_svg_scale / crop_to_input_scale;
                let x_y = (min_x as f64 * orig_svg_scale, min_y as f64 * orig_svg_scale);
                let width_height = (
                    ol_img_width_height.0 as f64 * scale,
                    ol_img_width_height.1 as f64 * scale,
                );
                (x_y, width_height)
            }
        }
    }
}
pub struct ResultHtmlLmArgs {
    pub model_io: ModelIO,
    pub image: DynamicImage,
    pub metadata: ImageMetadata,
    pub scan_direction: ScanDirection,
    pub heatmap: DynamicImage,
    pub title: String,
}

pub fn create_result_html_from_lm(args: ResultHtmlLmArgs) -> Result<String, anyhow::Error> {
    let ResultHtmlLmArgs {
        model_io,
        image,
        metadata,
        scan_direction,
        heatmap,
        title,
    } = args;

    let original_image_width = image.width();
    let original_image_height = image.height();
    let (x_y, width_height) = model_io.overlay_params(
        original_image_width,
        original_image_height,
        (heatmap.width(), heatmap.height()),
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

    let model_output_f64 = model_io.output3().mapv(|x| x as f64);
    let ch_last_output = model_output_f64.permuted_axes([1, 2, 0]);

    if let Err(e) = scolrs::C7TLS::check_counts(model_io.lm_data()) {
        log::warn!(
            "Incorrect number of points for Spine falling back to heatmap display: {}",
            e
        );
        let mut gp = GenericPoints::from(model_io.lm_data().clone());
        *gp.image_metadata_mut() = metadata;
        let html = create_generic_svg(
            gp,
            LabelMeDataWImage::new(model_io.lm_data().clone(), image),
            overlays,
        )?;
        let html = wrap_in_html(html.to_string(), &["g.Component".to_string()], title)?;
        return Ok(html);
    }
    let document = match scan_direction {
        ScanDirection::Coronal => {
            log::debug!("Creating CoronalPointsAndCurve");
            let mut cp = CoronalPointsAndCurve::try_from(model_io.lm_data())?;
            let crop_adjusted_cp = CoronalPointsAndCurve::try_from(model_io.heatmap_lm_data())?;
            *cp.image_metadata_mut() = metadata;
            let confidence = crop_adjusted_cp
                .coronal_points
                .extract_point_confidence(ch_last_output.view());
            log::trace!("Extracted confidence: {:?}", confidence);
            *cp.coronal_points.get_confidence_mut() = Some(confidence);
            let lm_data_with_image: LabelMeDataWImage =
                LabelMeDataWImage::new(model_io.lm_data().clone(), image);
            create_coronal_svg(cp, lm_data_with_image, overlays)
        }
        ScanDirection::Sagittal => {
            log::debug!("Creating SagittalPoints");
            let mut cp: SagittalPoints = SagittalPoints::try_from(model_io.lm_data())?;
            let crop_adjusted_cp = SagittalPoints::try_from(model_io.heatmap_lm_data())?;
            *cp.image_metadata_mut() = metadata;
            let confidence = crop_adjusted_cp.extract_point_confidence(ch_last_output.view());
            log::trace!("Extracted confidence: {:?}", confidence);
            *cp.get_confidence_mut() = Some(confidence);
            let lm_data_with_image: LabelMeDataWImage =
                LabelMeDataWImage::new(model_io.lm_data().clone(), image);
            create_sagittal_svg(cp, lm_data_with_image, overlays)
        }
    }?;
    let html = wrap_in_html(document.to_string(), &["g.Component".to_string()], title)?;
    Ok(html)
}

pub fn create_result_html(args: ResultHtmlArguments) -> Result<String, anyhow::Error> {
    let ResultHtmlArguments {
        points,
        image,
        metadata,
        scan_direction,
        cropping_params,
        heatmap,
        model_output,
        title,
    } = args;
    let model_io = if let Some(cropping_params) = cropping_params {
        ModelIO::Cropped(Box::new(CroppedIO::new(
            (image.width(), image.height()),
            model_output,
            points,
            cropping_params,
        )))
    } else {
        ModelIO::Original(Box::new(OriginalIO::new(
            (image.width(), image.height()),
            model_output,
            points,
        )))
    };
    let args = ResultHtmlLmArgs {
        model_io,
        image,
        metadata,
        scan_direction,
        heatmap,
        title,
    };
    create_result_html_from_lm(args)
}
