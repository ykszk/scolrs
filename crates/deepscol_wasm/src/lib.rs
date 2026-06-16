extern crate wasm_bindgen;
extern crate wee_alloc;
use ndarray::Array3;
use wasm_bindgen::prelude::*;

use deepscol::{
    calculate_crop_parameters, extract_points, load_image, to_model_input, CropConfig, CroppedIO,
    ModelIO, OriginalIO, ResultHtmlLmArgs,
};

pub use deepscol::{CroppingParams, ScanDirection};

// Use `wee_alloc` as the global allocator.
// Default allocator somehow panics in `wrap_in_html`, which can be fixed in the future.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(getter_with_clone)]
pub struct DecodedImage {
    pub image_bytes: Vec<u8>,
    pub format: String,
    pub width: u32,
    pub height: u32,
}

#[wasm_bindgen]
pub fn decode_image(encoded: &[u8]) -> Result<DecodedImage, JsValue> {
    let (image, metadata) = load_image(encoded).map_err(|e| JsValue::from_str(&e.to_string()))?;
    log::debug!("Image metadata: {:?}", metadata);
    let mut cursor = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut cursor, deepscol::image::ImageFormat::Jpeg)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(DecodedImage {
        image_bytes: cursor.into_inner(),
        format: "jpeg".to_string(),
        width: metadata.width as u32,
        height: metadata.height as u32,
    })
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Default, Debug)]
pub struct Settings {
    pub flip_image: bool,
    pub scan_direction: ScanDirection,
}

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

#[wasm_bindgen(getter_with_clone)]
pub struct Arr3 {
    pub arr: js_sys::Float32Array,
    pub d1: usize,
    pub d2: usize,
    pub d3: usize,
}

/// Note: this function takes the ownership of the cropping_params
#[wasm_bindgen]
pub fn create_input_array(
    bytes: &[u8],
    model_input_height: u32,
    settings: &Settings,
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
    let arr4 = to_model_input(&image, model_input_height)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(Arr3 {
        arr: js_sys::Float32Array::from(arr4.as_slice().unwrap()),
        d1: arr4.shape()[1] as usize,
        d2: arr4.shape()[2] as usize,
        d3: arr4.shape()[3] as usize,
    })
}

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
        &CropConfig::default(),
        original_image_height,
        original_image_width,
        tensor_dims.get_index(2),
    );
    Ok(params)
}

/// Note: this function takes the ownership of the cropping_params
#[wasm_bindgen]
pub fn process_output(
    image_filename: &str,
    encoded: &[u8],
    raw_output: &[f32],
    tensor_dims: js_sys::Uint32Array,
    settings: &Settings,
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

    let point_set_config = deepscol::point_config::PointSetConfig::spine();

    let blur_sigma = Some(3.0);
    let points = extract_points(&output3, 0.1, blur_sigma, &point_set_config.max_counts)
        .map_err(|e| JsValue::from_str(&format!("Failed to extract points: {}", e)))?;

    let mut model_io = if let Some(cp) = cropping_params {
        ModelIO::Cropped(Box::new(CroppedIO::new(
            image,
            image_filename.to_string(),
            output3,
            points,
            cp,
            &point_set_config,
        )))
    } else {
        log::info!("No cropping applied");
        ModelIO::Original(Box::new(OriginalIO::new(
            image,
            image_filename.to_string(),
            output3,
            points,
            &point_set_config,
        )))
    };

    log::debug!("Output tensor shape: {:?}", model_io.output3().shape());
    log::debug!("Cropping parameters: {:?}", cropping_params);

    if let Err(e) = scolrs::C7TLS::check_counts(model_io.lm_data()) {
        log::info!("Point counts are invalid: {}", e);
        let asms = deepscol::embedded_asm(settings.scan_direction).map_err(|e| {
            JsValue::from_str(&format!("Failed to load embedded ASM models: {}", e))
        })?;
        let asm_config = scolrs::asm::AsmConfig::default();
        let asms: Vec<_> = asms
            .into_iter()
            .map(|asm| (asm, asm_config.clone()))
            .collect();
        let output_f64 = model_io.output3().mapv(|x| x as f64);
        let result_best_asm_lm_data =
            deepscol::apply_asms(&model_io, output_f64.view(), &point_set_config, asms);
        match result_best_asm_lm_data {
            Err(e) => {
                log::warn!("Failed to apply ASM models: {}", e);
            }
            Ok(best_asm_lm_data) => {
                if let Some(best_lm_data) = best_asm_lm_data {
                    model_io.set_heatmap_lm_data(best_lm_data);
                } else {
                    log::warn!("No ASM model provided, skipping ASM fitting.");
                }
            }
        }
    }

    let result_args = ResultHtmlLmArgs {
        model_io,
        metadata,
        scan_direction: settings.scan_direction,
        size_config: deepscol::SizeConfig::default(),
        title: format!("{} - deepscol result", image_filename),
    };
    let html = deepscol::create_result_html_from_lm(result_args)
        .map_err(|e| JsValue::from_str(&format!("Failed to create HTML: {}", e)))?;
    Ok(html)
}

extern crate console_error_panic_hook;
use std::panic;

#[wasm_bindgen]
pub fn start() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    wasm_logger::init(wasm_logger::Config::default());
    log::debug!("logger initialized");
}
