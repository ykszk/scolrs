extern crate wasm_bindgen;
use core::fmt;
use dicom_pixeldata::{ConvertOptions, PixelDecoder, VoiLutOption};
pub use image;
use image::{DynamicImage, GenericImageView};
use imageproc::region_labelling::{connected_components, Connectivity};
use log::debug;
use ndarray::{s, Array3, Array4, ArrayView3, Axis, CowArray, NewAxis};
use ndarray_ndimage as ndi;
use scolrs::{
    asm::{model::ActiveShapeModel, AsmConfig, AsmError},
    draw::{
        generic::{GenericDraw, GenericPoints},
        output3_to_heatmap, AsMeasure, AxisMax, ConfidenceComponent, ConfidenceDisplay,
        DrawComponent, EmbeddedData, ImageOverlay, MapKey, MeasureComponent,
    },
    head_neck::{self},
    measure::{measure_x, FlattenResult, MeasureResult},
    HasImageMetadata, ImageMetadata, PointConfidence, SagittalDraw, SagittalPoints, Scalable,
    ScaledType, ScolError,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, ops::Range};
use wasm_bindgen::prelude::*;
pub type Point = (f32, f32);

#[wasm_bindgen]
#[derive(Default, Debug, Clone, Copy)]
pub enum ScanDirection {
    #[default]
    Coronal,
    Sagittal,
    NeckLateral,
}

impl ScanDirection {
    pub fn check_counts(&self, lm_data: &LabelMeData) -> Result<(), ScolError> {
        match self {
            ScanDirection::Coronal => scolrs::C7TLS::check_counts(lm_data),
            ScanDirection::Sagittal => scolrs::C7TLS::check_counts(lm_data),
            ScanDirection::NeckLateral => head_neck::VertebralCornerPoints::check_counts(lm_data),
        }
    }

    fn svg_and_measurements(
        &self,
        model_io: ModelIO,
        metadata: ImageMetadata,
        overlays: Vec<ImageOverlay>,
        size_config: &SizeConfig,
        ch_last_output: ArrayView3<f32>,
        reduce: scolrs::draw::ReductionMethod,
    ) -> Result<(SVG, MeasureResult<String, f64>), anyhow::Error> {
        fn _svg_and_measurements<PointType, Draw, ValueType>(
            model_io: ModelIO,
            metadata: ImageMetadata,
            overlays: Vec<ImageOverlay>,
            size_config: &SizeConfig,
            ch_last_output: ArrayView3<f32>,
            reduce: scolrs::draw::ReductionMethod,
        ) -> Result<(SVG, MeasureResult<String, f64>), anyhow::Error>
        where
            PointType: HasImageMetadata + Scalable + PointConfidence + Clone,
            PointType: for<'a> TryFrom<&'a LabelMeData, Error = ScolError>,
            <PointType as scolrs::Scalable>::Error:
                std::error::Error + std::marker::Send + std::marker::Sync + fmt::Debug + 'static,
            for<'b> (&'b Draw, &'b ScaledType<PointType>): Into<Box<dyn DrawComponent + 'b>>,
            for<'a, 'b> (&'a Draw::MeasureType, &'b ScaledType<PointType>): Into<Box<dyn MeasureComponent<ValueType = ValueType> + 'b>>
                + Into<Box<dyn ConfidenceComponent<ValueType = ValueType> + 'b>>,
            Draw: MapKey + AsMeasure + DefaultDraws + std::fmt::Display,
            <Draw as AsMeasure>::MeasureType:
                MapKey + MeasureAndDraw + std::fmt::Display + std::fmt::Debug,
            ValueType: ConfidenceDisplay,
            MeasureResult<String, ValueType>: FlattenResult<FlatType = MeasureResult<String, f64>>,
        {
            log::debug!("Creating {:}", std::any::type_name::<PointType>());
            let mut cp = PointType::try_from(model_io.lm_data())?;
            let crop_adjusted_cp = PointType::try_from(model_io.heatmap_lm_data())?;
            *cp.image_metadata_mut() = metadata;
            let confidence = crop_adjusted_cp.extract_point_confidence(ch_last_output.view());
            *cp.get_confidence_mut() = Some(confidence);
            let lm_data_with_image = model_io.into_lm_data_w_image();

            let measures = <Draw as AsMeasure>::MeasureType::all();
            let scaled_data = cp.clone().into_scaled()?;
            let measurements = measure_x(scaled_data, measures, reduce).into_string_map();
            let measurements = measurements.into_flat();
            let svg = create_svg(cp, lm_data_with_image, overlays, size_config)?;
            Ok((svg, measurements))
        }
        match self {
            ScanDirection::Coronal => {
                _svg_and_measurements::<CoronalPointsAndCurve, CoronalDraw, f64>(
                    model_io,
                    metadata,
                    overlays,
                    size_config,
                    ch_last_output,
                    reduce,
                )
            }
            ScanDirection::Sagittal => _svg_and_measurements::<SagittalPoints, SagittalDraw, f64>(
                model_io,
                metadata,
                overlays,
                size_config,
                ch_last_output,
                reduce,
            ),
            ScanDirection::NeckLateral => _svg_and_measurements::<
                head_neck::LateralPoints,
                head_neck::NeckLateralDraw,
                Vec<f64>,
            >(
                model_io,
                metadata,
                overlays,
                size_config,
                ch_last_output,
                reduce,
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeConfig {
    #[serde(rename = "image_resize")]
    pub resize: (u32, u32),
    pub svg_size: Option<(usize, usize)>,
}

impl Default for SizeConfig {
    fn default() -> Self {
        Self {
            resize: (1200, 1200),
            svg_size: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HeatmapConfig {
    /// Threshold for heatmap binarization for point extraction
    pub thresh: ThresholdConfig,
    /// Sigma for gaussian blur applied to heatmap before point extraction.
    pub blur_sigma: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdMinMax {
    pub min: Option<f32>,
    pub max: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridThreshold {
    pub absolute: ThresholdMinMax,
    pub relative: ThresholdMinMax,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdConfig {
    Absolute(ThresholdMinMax),
    Relative(ThresholdMinMax),
    Hybrid(HybridThreshold),
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        let absolute = ThresholdMinMax {
            min: Some(0.01),
            max: None,
        };
        let relative = ThresholdMinMax {
            min: Some(0.1),
            max: None,
        };
        Self::Hybrid(HybridThreshold { absolute, relative })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdMethod {
    Absolute,
    /// Threshold value is relative to the maximum value in the heatmap
    Relative,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    pub parallel_execution: bool,
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            parallel_execution: false,
            intra_threads: Some(4),
            inter_threads: Some(1),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeepscolConfig {
    pub heatmap: HeatmapConfig,
    /// SVG size and resize parameters
    pub size: SizeConfig,
    pub crop: CropConfig,
    pub session: SessionConfig,
}

/// Extract landmark point out of the input heatmaps
pub fn extract_points(
    arr: &ndarray::Array3<f32>,
    thresh: &ThresholdConfig,
    blur_sigma: Option<f32>,
    max_point_counts: &[usize],
) -> Result<Vec<Vec<Point>>, String> {
    let height = arr.shape()[1];
    let width = arr.shape()[2];
    let ch_axis = Axis(0);
    let mut all_points: Vec<Vec<Point>> = Vec::new();
    for (i_ch, (img_ch, max_point_count)) in arr
        .axis_iter(ch_axis)
        .zip(max_point_counts.iter())
        .enumerate()
    {
        let (min_thresh, max_thresh) = match thresh {
            ThresholdConfig::Absolute(thresh_minmax) => {
                let min_thresh = thresh_minmax.min.unwrap_or(0.0);
                let max_thresh = thresh_minmax.max.unwrap_or(f32::MAX);
                (min_thresh, max_thresh)
            }
            ThresholdConfig::Relative(thresh_minmax) => {
                let max_val = img_ch.iter().cloned().fold(f32::MIN, f32::max);
                let min_thresh = thresh_minmax.min.map(|v| v * max_val).unwrap_or(0.0);
                let max_thresh = thresh_minmax.max.map(|v| v * max_val).unwrap_or(f32::MAX);
                log::debug!(
                    "Channel {} max value: {}, relative computed min: {}, max: {}",
                    i_ch,
                    max_val,
                    min_thresh,
                    max_thresh
                );
                (min_thresh, max_thresh)
            }
            ThresholdConfig::Hybrid(hybrid) => {
                let max_val = img_ch.iter().cloned().fold(f32::MIN, f32::max);
                let min_thresh_abs = hybrid.absolute.min.unwrap_or(0.0);
                let max_thresh_abs = hybrid.absolute.max.unwrap_or(f32::MAX);
                let min_thresh_rel = hybrid.relative.min.map(|v| v * max_val).unwrap_or(0.0);
                let max_thresh_rel = hybrid.relative.max.map(|v| v * max_val).unwrap_or(f32::MAX);
                let min_thresh = min_thresh_abs.max(min_thresh_rel);
                let max_thresh = max_thresh_abs.min(max_thresh_rel);
                log::debug!(
                    "Channel {} max value: {}, hybrid threshold absolute min: {:?}, max: {:?}, relative min: {:?}, max: {:?}, computed min: {}, max: {}",
                    i_ch,
                    max_val,
                    hybrid.absolute.min,
                    hybrid.absolute.max,
                    min_thresh_rel,
                    max_thresh_rel,
                    min_thresh,
                    max_thresh
                );
                (min_thresh, max_thresh)
            }
        };
        let bin_arr = img_ch.mapv(|v| {
            if v > min_thresh && v < max_thresh {
                1u8
            } else {
                0u8
            }
        });
        let bin_img = image::GrayImage::from_raw(
            width as _,
            height as _,
            bin_arr.into_raw_vec_and_offset().0,
        )
        .unwrap();

        let img_ch: CowArray<'_, f32, _> = if let Some(sigma) = blur_sigma {
            CowArray::from(ndi::gaussian_filter(
                &img_ch,
                sigma,
                0,
                ndi::BorderMode::Nearest,
                3,
            ))
        } else {
            CowArray::from(img_ch)
        };

        let cced = connected_components(&bin_img, Connectivity::Four, image::Luma([0u8]));
        let n_cc = *cced.iter().max().unwrap();
        log::trace!("# of CC in channel {} is {}", i_ch, n_cc);

        let mut local_maxima: HashMap<u32, (f32, usize, usize)> = HashMap::new();
        // traverse each pixel in cced with the coordinates
        for (x, y, &cc_val) in cced.enumerate_pixels() {
            let cc_val = cc_val.0[0];
            if cc_val == 0 {
                continue; // Skip background
            }
            let (y, x) = (y as usize, x as usize);

            let val = img_ch[[y, x]];
            let local_maximum = local_maxima.get(&cc_val);
            if let Some((max_val, _max_y, _max_x)) = local_maximum {
                if val > *max_val {
                    // Update local maxima if current pixel is greater
                    local_maxima.insert(cc_val, (val, y, x));
                } else {
                    // If current pixel is not greater, skip it
                    continue;
                }
            } else {
                // If no local maxima exists for this cc_val, set current pixel as local maxima
                local_maxima.insert(cc_val, (val, y, x));
            }
        }
        let mut points_for_ch: Vec<Point> = local_maxima
            .into_iter()
            .map(|(_, (_val, y, x))| {
                // Convert to (x, y) coordinates
                (x as f32, y as f32)
            })
            .collect::<Vec<Point>>();

        if points_for_ch.len() > *max_point_count {
            log::debug!(
                "Channel {} has more points than max_point_count: {} > {}",
                i_ch,
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

pub fn load_image_from_path(
    path: &std::path::Path,
) -> Result<(image::DynamicImage, ImageMetadata), image::ImageError> {
    let raw_bytes = std::fs::read(path).map_err(image::ImageError::IoError)?;
    let (image, mut metadata) = load_image(&raw_bytes)?;
    metadata.path = path.to_string_lossy().to_string();
    Ok((image, metadata))
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
    image: &image::DynamicImage,
    model_input_height: u32,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>>, anyhow::Error> {
    let image = resize_height(image, model_input_height);
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
    labels: &[String],
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
            log::debug!("No points found for {}", labels[i_point]);
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
    lm_data
}

pub fn create_scaled_lm(
    labels: &[String],
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
    labels: &[String],
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
    output3
}

use labelme_rs::{
    svg::node::element::{self, SVG},
    LabelMeData, LabelMeDataWImage,
};
use scolrs::{
    draw::{self, wrap_in_html},
    CoronalDraw, CoronalPointsAndCurve, MeasureAndDraw,
};

pub use scolrs::asm::point_config::{self, PointSetConfig};

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

pub trait DefaultDraws {
    fn draws() -> Vec<Self>
    where
        Self: Sized;
    fn hides() -> Vec<Self>
    where
        Self: Sized;
}

impl DefaultDraws for GenericDraw {
    fn draws() -> Vec<Self> {
        vec![GenericDraw::AllPoints]
    }
    fn hides() -> Vec<Self> {
        vec![GenericDraw::AllPoints]
    }
}

impl DefaultDraws for CoronalDraw {
    fn draws() -> Vec<Self> {
        let non_draws = [CoronalDraw::SpinalLine, CoronalDraw::Centroids];
        CoronalDraw::all()
            .into_iter()
            .filter(|d| !non_draws.contains(d))
            .collect::<Vec<_>>()
    }
    fn hides() -> Vec<Self> {
        let non_hide = [
            CoronalDraw::AllPoints,
            CoronalDraw::VertebralLabels,
            CoronalDraw::CobbPT,
            CoronalDraw::CobbMT,
            CoronalDraw::CobbTLL,
        ];
        CoronalDraw::all()
            .into_iter()
            .filter(|d| !non_hide.contains(d))
            .collect::<Vec<_>>()
    }
}

impl DefaultDraws for scolrs::SagittalDraw {
    fn draws() -> Vec<Self> {
        scolrs::SagittalDraw::all()
    }
    fn hides() -> Vec<Self> {
        let non_hide = [
            SagittalDraw::AllPoints,
            SagittalDraw::VertebralLabels,
            SagittalDraw::ThoracicKyphosis,
            SagittalDraw::LumbarLordosis,
        ];
        scolrs::SagittalDraw::all()
            .into_iter()
            .filter(|d| !non_hide.contains(d))
            .collect::<Vec<_>>()
    }
}

impl DefaultDraws for head_neck::NeckLateralDraw {
    fn draws() -> Vec<Self> {
        head_neck::NeckLateralDraw::all()
    }
    fn hides() -> Vec<Self> {
        use head_neck::NeckLateralDraw as ND;
        let non_hides = vec![
            ND::CervicalPoints,
            ND::VertebralLabels,
            ND::OC2,
            ND::WedgeAngle,
            ND::ThoracicInletAngle,
            ND::NeckTilt,
            ND::SpinoCranialAngle,
            ND::OccipitocervicalInclination,
            ND::CranialSlope,
        ];
        head_neck::NeckLateralDraw::all()
            .into_iter()
            .filter(|d| !non_hides.contains(d))
            .collect::<Vec<_>>()
    }
}

pub fn create_svg<PointType, Draw, ValueType>(
    gp: PointType,
    lm_data_with_image: LabelMeDataWImage,
    overlays: Vec<ImageOverlay>,
    size_config: &SizeConfig,
) -> Result<SVG, anyhow::Error>
where
    PointType: HasImageMetadata + Scalable,
    <PointType as Scalable>::Error: std::fmt::Debug,
    for<'b> (&'b Draw, &'b ScaledType<PointType>): Into<Box<dyn DrawComponent + 'b>>,
    for<'b> (&'b Draw::MeasureType, &'b ScaledType<PointType>):
        Into<Box<dyn ConfidenceComponent<ValueType = ValueType> + 'b>>,
    Draw: Clone + PartialEq + AsMeasure + DefaultDraws,
    ValueType: ConfidenceDisplay,
{
    let draws = Draw::draws();

    let draw_param = scolrs::draw::DrawParam::default();
    let resize_param = labelme_rs::ResizeParam::Size(size_config.resize.0, size_config.resize.1);
    let svg_size = size_config.svg_size;
    let palettes = scolrs::draw::ColorPalettes::default();
    let hides = Draw::hides();
    let draw_args = draw::DrawArguments {
        image: lm_data_with_image.image,
        data: gp,
        draw_param,
        resize_param: Some(resize_param),
        svg_size,
        palettes,
        draw: &draws,
        hide: &hides,
        overlays,
    };
    let document = draw::draw_on_image(draw_args).expect("Failed to draw generic points and curve");
    Ok(document)
}

pub fn create_generic_svg(
    gp: GenericPoints,
    lm_data_with_image: LabelMeDataWImage,
    overlays: Vec<ImageOverlay>,
    size_config: &SizeConfig,
) -> Result<SVG, anyhow::Error> {
    create_svg::<GenericPoints, GenericDraw, f64>(gp, lm_data_with_image, overlays, size_config)
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
    pub model_output: Array3<f32>,
    pub size_config: SizeConfig,
    pub title: String,
}

pub struct OriginalIO {
    original_image: DynamicImage,
    output3: Array3<f32>,
    /// LabelMeData in the original image coordinate system
    /// image width and height correspond to original image
    lm_data: LabelMeData,
}

impl OriginalIO {
    pub fn new(
        original_image: DynamicImage,
        image_path: String,
        output3: Array3<f32>,
        points: Vec<Vec<(f32, f32)>>,
        point_set_config: &PointSetConfig,
    ) -> Self {
        let original_image_wh = original_image.dimensions();
        let input_height = output3.shape()[1] as u32;
        let lm_scale = original_image_wh.1 as f64 / input_height as f64;
        let lm_data = create_scaled_lm(
            &point_set_config.labels,
            image_path,
            original_image_wh.0,
            original_image_wh.1,
            lm_scale,
            points.as_slice(),
        );
        Self {
            original_image,
            output3,
            lm_data,
        }
    }
}

pub struct CroppedIO {
    original_image: DynamicImage,
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
        original_image: DynamicImage,
        image_path: String,
        output3: Array3<f32>,
        points: Vec<Vec<(f32, f32)>>,
        cropping_params: CroppingParams,
        point_set_config: &PointSetConfig,
    ) -> Self {
        let CroppingParams {
            min_x,
            min_y,
            max_x: _,
            max_y,
        } = cropping_params;

        let cropped_lm_data = create_lm(
            &point_set_config.labels,
            image_path,
            output3.shape()[2] as u32,
            output3.shape()[1] as u32,
            points.as_slice(),
        );

        let original_image_wh = original_image.dimensions();
        let mut lm_data = cropped_lm_data.clone();
        let input_height = output3.shape()[1] as u32;
        let lm_scale = (max_y - min_y) as f64 / input_height as f64;
        lm_data.scale(lm_scale);
        lm_data.shift(min_x as f64, min_y as f64);
        lm_data.imageWidth = original_image_wh.0 as usize;
        lm_data.imageHeight = original_image_wh.1 as usize;

        Self {
            original_image,
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
    pub fn original_image(&self) -> &DynamicImage {
        match self {
            ModelIO::Original(original) => &original.original_image,
            ModelIO::Cropped(cropped) => &cropped.original_image,
        }
    }
    pub fn into_lm_data_w_image(self) -> LabelMeDataWImage {
        match self {
            ModelIO::Original(original) => LabelMeDataWImage {
                data: original.lm_data,
                image: original.original_image,
            },
            ModelIO::Cropped(cropped) => LabelMeDataWImage {
                data: cropped.lm_data,
                image: cropped.original_image,
            },
        }
    }
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

    /// set heatmap lm data and update original lm data accordingly
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
        ol_img_width_height: (u32, u32),
        size_config: &SizeConfig,
    ) -> ((f64, f64), (f64, f64)) {
        let original_image_width = self.original_image().width();
        let original_image_height = self.original_image().height();
        let resize_param =
            labelme_rs::ResizeParam::Size(size_config.resize.0, size_config.resize.1);
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

    pub fn output3_to_heatmap(
        &self,
        ch_range_rgb: (Range<usize>, Range<usize>, Range<usize>),
    ) -> DynamicImage {
        let input_image_wh = match self {
            ModelIO::Original(original) => (
                original.lm_data.imageWidth as u32,
                original.lm_data.imageHeight as u32,
            ),
            ModelIO::Cropped(cropped) => (
                (cropped.cropping_params.max_x - cropped.cropping_params.min_x) as u32,
                (cropped.cropping_params.max_y - cropped.cropping_params.min_y) as u32,
            ),
        };
        output3_to_heatmap(self.output3().view(), input_image_wh, ch_range_rgb)
    }
}
pub struct ResultHtmlLmArgs {
    pub model_io: ModelIO,
    pub metadata: ImageMetadata,
    pub scan_direction: ScanDirection,
    pub size_config: SizeConfig,
    pub title: String,
}

pub fn create_result_html_from_lm(args: ResultHtmlLmArgs) -> Result<String, anyhow::Error> {
    let ResultHtmlLmArgs {
        model_io,
        metadata,
        scan_direction,
        size_config,
        title,
    } = args;
    let ch_range_rgb = match scan_direction {
        ScanDirection::Coronal => (0..2, 2..4, 4..9),
        ScanDirection::Sagittal => (0..2, 2..4, 4..9),
        ScanDirection::NeckLateral => (0..2, 2..4, 4..(model_io.output3().shape()[0] - 2)),
    };
    let heatmap = model_io.output3_to_heatmap(ch_range_rgb);
    let (x_y, width_height) =
        model_io.overlay_params((heatmap.width(), heatmap.height()), &size_config);

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

    let ch_last_output = model_io.output3().to_owned().permuted_axes([1, 2, 0]);

    if let Err(e) = scan_direction.check_counts(model_io.lm_data()) {
        log::warn!(
            "Incorrect number of points for {:?} falling back to heatmap display: {}",
            scan_direction,
            e
        );
        let mut gp = GenericPoints::from(model_io.lm_data().clone());
        *gp.image_metadata_mut() = metadata;
        let html = create_generic_svg(gp, model_io.into_lm_data_w_image(), overlays, &size_config)?;
        let html = wrap_in_html(html.to_string(), &["g.Component".to_string()], title)?;
        return Ok(html);
    }
    let embedded_lm_data = EmbeddedData::new(
        "labelme.json".to_string(),
        serde_json::to_string_pretty(model_io.lm_data())?,
        "application/json".to_string(),
    );

    let reduce = scolrs::draw::ReductionMethod::GeometricMean;

    let (mut document, measurements) = scan_direction.svg_and_measurements(
        model_io,
        metadata,
        overlays,
        &size_config,
        ch_last_output.view(),
        reduce,
    )?;

    let measure_json = serde_json::to_string_pretty(&measurements)?;
    let transposed_result = measurements.into_transposed();
    let mut wtr = csv::Writer::from_writer(vec![]);
    wtr.serialize(("Measurement", "Value", "Confidence"))?;
    for entry in transposed_result.entries {
        let measure = entry.1.measurement.ok();
        // Option<Result<f64, _>> to Option<f64>
        let confidence = entry.1.confidence.and_then(|r| r.ok());
        wtr.serialize((entry.0, measure, confidence))?;
    }
    wtr.serialize((
        "unit of length",
        transposed_result.unit_of_length,
        None::<f64>,
    ))?;
    let csv_data = String::from_utf8(wtr.into_inner()?)?;

    let embedded_data = [
        EmbeddedData::new(
            "measurements.json".to_string(),
            measure_json,
            "application/json".to_string(),
        ),
        EmbeddedData::new(
            "measurements.csv".to_string(),
            csv_data,
            "text/csv".to_string(),
        ),
        embedded_lm_data,
    ];
    let mut scripts = element::Definitions::new().set("id", "downloads");
    for ed in embedded_data.iter() {
        let tag = ed.to_script_tag(true);
        scripts = scripts.add(tag);
    }
    document = document.add(scripts);
    let html = wrap_in_html(document.to_string(), &["g.Component".to_string()], title)?;
    Ok(html)
}

#[cfg(feature = "embed_asm")]
pub fn embedded_asm(direction: ScanDirection) -> Result<Vec<ActiveShapeModel>, serde_json::Error> {
    match direction {
        ScanDirection::Coronal => {
            let json = include_str!("../models/coronal/asms.json");
            serde_json::from_str(json)
        }
        ScanDirection::Sagittal => {
            let json = include_str!("../models/sagittal/asms.json");
            serde_json::from_str(json)
        }
        ScanDirection::NeckLateral => Ok(Vec::new()), // TODO: add lateral neck ASMs
    }
}

fn _apply_asms(
    model_io: &ModelIO,
    output3: ArrayView3<f64>,
    point_config: &PointSetConfig,
    asms: Vec<(ActiveShapeModel, AsmConfig)>,
) -> Result<Vec<(String, labelme_rs::LabelMeData, f64)>, AsmError> {
    if asms.is_empty() {
        log::info!("No ASM models provided for fitting.");
        return Ok(Vec::new());
    }
    let first_asm = &asms[0].0;
    let heatmaps = point_config.select_channels(output3, &first_asm.labels);
    let mut cached_heatmaps = scolrs::asm::CachedHeatmaps::new(heatmaps.view());
    let mut asm_results = Vec::new();
    for (asm, asm_config) in asms {
        let mut heatmap_lm = model_io.heatmap_lm_data().to_owned();

        let asm_labels = asm.labels.clone();
        let model_name = asm.model_name.clone();
        log::info!("Starting ASM model fitting for model: {}", model_name);
        let (_fitted_lm_data, histories, _optimal_params, fitted_shape) =
            scolrs::asm::apply_asm(asm, asm_config, &mut cached_heatmaps, &heatmap_lm)?;
        // remove old points
        for label in asm_labels.iter() {
            heatmap_lm.shapes.retain(|shape| &shape.label != label);
        }
        // update points with fitted points
        for (label, points) in asm_labels.iter().zip(fitted_shape.iter()) {
            for point in points.axis_iter(Axis(0)) {
                let shape = labelme_rs::Shape {
                    label: label.clone(),
                    points: vec![(point[0], point[1])],
                    group_id: None,
                    shape_type: "point".to_string(),
                    flags: Default::default(),
                };
                heatmap_lm.shapes.push(shape);
            }
        }

        let best_loss = histories
            .last()
            .unwrap()
            .data_objectives
            .iter()
            .fold(f64::INFINITY, |a, b| a.min(*b));
        asm_results.push((model_name, heatmap_lm, best_loss));
        log::info!("ASM model fitting completed.");
    }
    Ok(asm_results)
}

pub fn apply_asms(
    model_io: &ModelIO,
    output3: ArrayView3<f64>,
    point_config: &PointSetConfig,
    asms: Vec<(ActiveShapeModel, AsmConfig)>,
) -> Result<Option<LabelMeData>, AsmError> {
    let mut asm_results = _apply_asms(model_io, output3, point_config, asms)?;
    // Choose the model with minimum loss
    asm_results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    asm_results
        .first()
        .map(|(best_model_name, best_lm_data, best_loss)| {
            log::info!("Best ASM model {} w/ loss: {}", best_model_name, best_loss);
            Ok(best_lm_data.clone())
        })
        .transpose()
}
