extern crate wasm_bindgen;
extern crate wee_alloc;
use ndarray::Array3;
use wasm_bindgen::prelude::*;

use deepscol::{
    calculate_crop_parameters, extract_points, load_image, to_model_input, CropConfig, CroppedIO,
    ModelIO, OriginalIO, ResultHtmlLmArgs, ThresholdConfig,
};
use labelme_rs::LabelMeData;

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

/// A `#[wasm_bindgen]`-friendly stand-in for `Option<CroppingParams>`.
///
/// wasm-bindgen doesn't support `Option<&T>` for custom struct types, so functions taking
/// `Option<CroppingParams>` by value consume (and invalidate) the JS-side object. This type is
/// taken by reference instead (`&OptionalCroppingParams`), which only borrows on the JS side and
/// never invalidates the passed object.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Default)]
pub struct OptionalCroppingParams {
    pub has_value: bool,
    pub params: CroppingParams,
}

#[wasm_bindgen]
impl OptionalCroppingParams {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn some(params: CroppingParams) -> Self {
        Self {
            has_value: true,
            params,
        }
    }
}

impl From<CroppingParams> for OptionalCroppingParams {
    fn from(cp: CroppingParams) -> Self {
        Self {
            has_value: true,
            params: cp,
        }
    }
}

impl From<Option<CroppingParams>> for OptionalCroppingParams {
    fn from(cp: Option<CroppingParams>) -> Self {
        match cp {
            Some(cp) => cp.into(),
            None => Self::default(),
        }
    }
}

impl From<&OptionalCroppingParams> for Option<CroppingParams> {
    fn from(o: &OptionalCroppingParams) -> Self {
        o.has_value.then_some(o.params)
    }
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

#[wasm_bindgen]
pub fn create_input_array(
    bytes: &[u8],
    model_input_height: u32,
    settings: &Settings,
    cropping_params: &OptionalCroppingParams,
) -> Result<Arr3, JsValue> {
    let (image, _metadata) = load_image(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let image = if settings.flip_image {
        log::debug!("Flipping image horizontally");
        image.fliph()
    } else {
        image
    };
    let arr4 = to_model_input(&image, model_input_height, cropping_params.into())
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
) -> Result<OptionalCroppingParams, JsValue> {
    let output3 = Array3::from_shape_vec(
        (
            tensor_dims.get_index(1) as usize,
            tensor_dims.get_index(2) as usize,
            tensor_dims.get_index(3) as usize,
        ),
        raw_output.to_vec(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let params = calculate_crop_parameters(
        &output3,
        &CropConfig::default(),
        original_image_height,
        original_image_width,
        tensor_dims.get_index(2),
    );
    Ok(params.into())
}

fn build_model_io(
    image_filename: &str,
    image: deepscol::image::DynamicImage,
    raw_output: &[f32],
    tensor_dims: &js_sys::Uint32Array,
    cropping_params: Option<CroppingParams>,
    settings: &Settings,
) -> Result<ModelIO, JsValue> {
    let output3 = Array3::from_shape_vec(
        (
            tensor_dims.get_index(1) as usize,
            tensor_dims.get_index(2) as usize,
            tensor_dims.get_index(3) as usize,
        ),
        raw_output.to_vec(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let point_set_config = match settings.scan_direction {
        ScanDirection::Coronal => deepscol::point_config::PointSetConfig::spine(),
        ScanDirection::Sagittal => deepscol::point_config::PointSetConfig::spine(),
        ScanDirection::NeckLateral => deepscol::point_config::PointSetConfig::neck_lateral(),
    };

    let points = extract_points(
        &output3,
        &ThresholdConfig::default(),
        Some(3.0),
        &point_set_config.max_counts,
    )
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

    if let Err(e) = settings.scan_direction.check_counts(model_io.lm_data()) {
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

    Ok(model_io)
}

#[wasm_bindgen(getter_with_clone)]
pub struct HtmlJson {
    pub html: String,
    pub lm_json: String,
}

fn remove_anchors(lm_data: &mut LabelMeData, scan_direction: ScanDirection) {
    let anchor_labels = match scan_direction {
        ScanDirection::Coronal | ScanDirection::Sagittal => vec!["C7-TL", "C7-TR", "S-TL", "S-TR"],
        ScanDirection::NeckLateral => vec!["Lamina1", "C3_BL", "C3_BR"],
    };
    for shape in &mut lm_data.shapes {
        if anchor_labels.contains(&shape.label.as_str()) {
            shape.points.clear();
        }
    }
}

#[wasm_bindgen]
pub fn process_output(
    image_filename: &str,
    encoded: &[u8],
    raw_output: &[f32],
    tensor_dims: js_sys::Uint32Array,
    settings: &Settings,
    cropping_params: &OptionalCroppingParams,
) -> Result<HtmlJson, JsValue> {
    let (image, metadata) = load_image(encoded).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let image = if settings.flip_image {
        log::debug!("Flipping image horizontally");
        image.fliph()
    } else {
        image
    };

    let model_io = build_model_io(
        image_filename,
        image,
        raw_output,
        &tensor_dims,
        cropping_params.into(),
        settings,
    )?;

    log::debug!("Output tensor shape: {:?}", model_io.output3().shape());
    log::debug!("Cropping parameters: {:?}", cropping_params);

    let mut lm_data = model_io.lm_data().clone();
    remove_anchors(&mut lm_data, settings.scan_direction);
    let lm_json = serde_json::to_string_pretty(&lm_data)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize landmark data: {}", e)))?;
    let result_args = ResultHtmlLmArgs {
        model_io,
        metadata,
        scan_direction: settings.scan_direction,
        size_config: deepscol::SizeConfig::default(),
        title: format!("{} - deepscol result", image_filename),
    };
    let html = deepscol::create_result_html_from_lm(result_args)
        .map_err(|e| JsValue::from_str(&format!("Failed to create HTML: {}", e)))?;
    Ok(HtmlJson { html, lm_json })
}

#[wasm_bindgen]
pub fn process_output_with_labelme(
    image_filename: &str,
    encoded: &[u8],
    raw_output: &[f32],
    tensor_dims: js_sys::Uint32Array,
    settings: &Settings,
    cropping_params: &OptionalCroppingParams,
    edited_labelme_json: &str,
) -> Result<String, JsValue> {
    let (image, metadata) = load_image(encoded).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let image = if settings.flip_image {
        log::debug!("Flipping image horizontally");
        image.fliph()
    } else {
        image
    };

    let mut model_io = build_model_io(
        image_filename,
        image,
        raw_output,
        &tensor_dims,
        cropping_params.into(),
        settings,
    )?;

    let edited_lm_data: LabelMeData = serde_json::from_str(edited_labelme_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse edited landmark data: {}", e)))?;
    model_io.set_lm_data(edited_lm_data);

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
