use dicom_pixeldata::{ConvertOptions, PixelDecoder, VoiLutOption};
use image::DynamicImage;
use imageproc::region_labelling::{connected_components, Connectivity};
use log::debug;
use ndarray::Axis;
use scolrs::ImageMetadata;
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

extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;
extern crate console_error_panic_hook;
use std::panic;

#[wasm_bindgen(start)]
pub fn start() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    wasm_logger::init(wasm_logger::Config::default());
    debug!("logger initialized");
}
