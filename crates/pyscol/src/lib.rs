use std::collections::HashMap;

use devscol::{trimming_box_with_resample, BoundingBox, PredicateSource};
use labelme_rs::image::{self, DynamicImage, GrayImage};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use scolrs::{
    draw_components, ColorPalette, ColorPalettes, CoronalMeasure, CoronalPointsAndCurve,
    DrawComponent, DrawError, DrawParam, HasImageMetadata, MeasureError, Painter, SagittalMeasure,
    SagittalPoints, Scalable, TryFromJson,
};
use svg::node::element;

#[pyfunction]
#[pyo3(name = "trimming_box_with_resample")]
fn py_trimming_box_with_resample(
    arr2d: PyReadonlyArray2<'_, i16>,
    predicate_sources_in_json: Vec<String>,
    resample_step: usize,
) -> PyResult<BoundingBox> {
    let arr2d = arr2d.as_array();
    let predicate_sources: Vec<PredicateSource> = predicate_sources_in_json
        .iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    trimming_box_with_resample(arr2d, predicate_sources, resample_step)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Error in trimming: {}", e)))
}

macro_rules! clahe_impl {
    ($fn_name:ident, $input_type:ty, $output_type:ty) => {
        #[pyfunction]
        fn $fn_name<'py>(
            py: Python<'py>,
            arr2d: PyReadonlyArray2<'py, $input_type>,
            grid_width: u32,
            grid_height: u32,
            clip_limit: u32,
            tile_sample: f64,
        ) -> PyResult<Bound<'py, PyArray2<$output_type>>> {
            let arr2d = arr2d.as_array();
            if tile_sample == 0.0 {
                clahe::clahe_wo_interpolation(
                    arr2d,
                    arr2d.ncols() as u32 / grid_width,
                    arr2d.nrows() as u32 / grid_height,
                    clip_limit,
                )
            } else {
                clahe::clahe_ndarray(arr2d, grid_width, grid_height, clip_limit, tile_sample)
            }
            .map(|a| a.into_pyarray_bound(py))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Clahe error: {}", e)))
        }
    };
}

clahe_impl!(clahe_u8_u8, u8, u8);
clahe_impl!(clahe_u16_u8, u16, u8);
clahe_impl!(clahe_u16_u16, u16, u16);

// #[pyfunction]
// fn ada_minmax_u16_u16<'py>(
//     py: Python<'py>,
//     arr2d: PyReadonlyArray2<'py, u16>,
//     grid_width: u32,
//     grid_height: u32,
//     tile_sample: f64,
// ) -> PyResult<&'py PyArray2<u16>> {
//     let arr2d = arr2d.as_array();
//     clahe::ada_minmax(arr2d, grid_width, grid_height, tile_sample)
//         .map(|a| a.into_pyarray(py))
//         .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Clahe error: {}", e)))
// }

// #[pyfunction]
// fn ada_minmax_u8_u8<'py>(
//     py: Python<'py>,
//     arr2d: PyReadonlyArray2<'py, u8>,
//     grid_width: u32,
//     grid_height: u32,
//     tile_sample: f64,
// ) -> PyResult<&'py PyArray2<u8>> {
//     let arr2d = arr2d.as_array();
//     clahe::ada_minmax(arr2d, grid_width, grid_height, tile_sample)
//         .map(|a| a.into_pyarray(py))
//         .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Clahe error: {}", e)))
// }

#[derive(thiserror::Error, Debug)]
pub enum PyScolError {
    #[error("Error in scolrs: {0}")]
    ScolError(#[from] scolrs::ScolError),
    #[error("Error in labelme_rs: {0}")]
    LabelMeError(#[from] labelme_rs::LabelMeDataError),
    #[error("Error in drawing: {0}")]
    DrawError(#[from] DrawError),
    #[error("Error in measuring: {0}")]
    MeasureError(#[from] MeasureError),
    #[error("Error in json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Image must be 2D or 3D: {0}")]
    ImageShape(String),
    #[error("Channel has to be 3 for 3D array")]
    ChannelMismatch,
}

impl From<PyScolError> for PyErr {
    fn from(e: PyScolError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e))
    }
}

fn ndarray_to_dynamic_image(arr: PyReadonlyArrayDyn<'_, u8>) -> Result<DynamicImage, PyScolError> {
    let shape = arr.shape();
    match shape.len() {
        2 => {
            let arr2d = arr.as_array().as_standard_layout().to_owned();
            Ok(DynamicImage::ImageLuma8(
                GrayImage::from_raw(shape[1] as u32, shape[0] as u32, arr2d.into_raw_vec())
                    .unwrap(),
            ))
        }
        3 => {
            if shape[2] != 3 {
                return Err(PyScolError::ImageShape(format!("{:?}", shape)));
            }
            let arr3d = arr.as_array().as_standard_layout().to_owned();
            Ok(DynamicImage::ImageRgb8(
                image::RgbImage::from_raw(shape[1] as u32, shape[0] as u32, arr3d.into_raw_vec())
                    .unwrap(),
            ))
        }
        _ => Err(PyScolError::ImageShape(format!("{:?}", shape))),
    }
}

trait DeserializeAll<T> {
    fn deserialize_all(&self) -> Result<Vec<T>, serde_json::Error>
    where
        T: for<'a> serde::de::Deserialize<'a>;
}

impl<T> DeserializeAll<T> for Vec<String> {
    fn deserialize_all(&self) -> Result<Vec<T>, serde_json::Error>
    where
        T: for<'a> serde::de::Deserialize<'a>,
    {
        self.iter().map(|s| serde_json::from_str(s)).collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_on_image<'a, T, S>(
    image: PyReadonlyArrayDyn<'_, u8>,
    data: T,
    draws: Vec<S>,
    hide: Vec<S>,
    draw_param_json: &str,
    svg_size: (usize, usize),
    label_colors: HashMap<String, String>,
    line_colors: HashMap<String, String>,
    overlay: Option<PyReadonlyArrayDyn<'_, u8>>,
) -> Result<String, PyScolError>
where
    for<'b> (&'b S, &'b T): Into<Box<dyn DrawComponent + 'b>>,
    S: Clone + Copy + PartialEq,
    T: HasImageMetadata + Scalable,
    <T as Scalable>::Error: std::fmt::Debug,
{
    let draw_param: DrawParam = serde_json::from_str(draw_param_json)?;

    let style = element::Style::new(draw_param.style());
    let painter = Painter::new(draw_param, svg_size);

    let mut line_colors = ColorPalette::new(line_colors);
    let mut label_colors = ColorPalette::new(label_colors);

    let dynamic_image = ndarray_to_dynamic_image(image)?;

    let mut document = painter.doc_w_background(&dynamic_image)?;

    document = document.add(style);

    if let Some(overlay) = overlay {
        let overlay_image = ndarray_to_dynamic_image(overlay)?;
        let g = scolrs::ImageOverlay::new(
            "heatmap".to_string(),
            "heatmap".to_string(),
            None,
            overlay_image,
        )
        .draw(&painter, &mut label_colors, &mut line_colors)?;
        document = document.add(g);
    }
    let palettes = ColorPalettes {
        line_colors,
        label_colors,
    };

    let groups = draw_components(data, &draws, &hide, &painter, palettes)?;

    for g in groups {
        document = document.add(g);
    }

    let svg = document.to_string();

    Ok(svg)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_draw_coronal(
    image: PyReadonlyArrayDyn<'_, u8>,
    coronal_points_json: &str,
    draws: Vec<String>,
    hide: Vec<String>,
    draw_param_json: &str,
    svg_size: (usize, usize),
    label_colors: HashMap<String, String>,
    line_colors: HashMap<String, String>,
    overlay: Option<PyReadonlyArrayDyn<'_, u8>>,
) -> Result<String, PyScolError> {
    let coronal_set = CoronalPointsAndCurve::try_from_ir_json(coronal_points_json)?;

    let draws = if draws.is_empty() {
        CoronalMeasure::all_draws()
    } else {
        draws.deserialize_all()?
    };
    let hide: Vec<CoronalMeasure> = hide.deserialize_all()?;
    draw_on_image(
        image,
        coronal_set,
        draws,
        hide,
        draw_param_json,
        svg_size,
        label_colors,
        line_colors,
        overlay,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_draw_sagittal(
    image: PyReadonlyArrayDyn<'_, u8>,
    coronal_points_json: &str,
    draws: Vec<String>,
    hide: Vec<String>,
    draw_param_json: &str,
    svg_size: (usize, usize),
    label_colors: HashMap<String, String>,
    line_colors: HashMap<String, String>,
    overlay: Option<PyReadonlyArrayDyn<'_, u8>>,
) -> Result<String, PyScolError> {
    let sagittal_points = SagittalPoints::try_from_ir_json(coronal_points_json)?;

    let draws = if draws.is_empty() {
        SagittalMeasure::all_draws()
    } else {
        draws.deserialize_all()?
    };
    let hide: Vec<SagittalMeasure> = hide.deserialize_all()?;
    draw_on_image(
        image,
        sagittal_points,
        draws,
        hide,
        draw_param_json,
        svg_size,
        label_colors,
        line_colors,
        overlay,
    )
}

#[pymodule]
fn pyscol(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::init();
    m.add_function(wrap_pyfunction!(py_trimming_box_with_resample, m)?)?;
    m.add_function(wrap_pyfunction!(clahe_u8_u8, m)?)?;
    m.add_function(wrap_pyfunction!(clahe_u16_u8, m)?)?;
    m.add_function(wrap_pyfunction!(clahe_u16_u16, m)?)?;
    m.add_function(wrap_pyfunction!(py_draw_coronal, m)?)?;
    m.add_function(wrap_pyfunction!(py_draw_sagittal, m)?)?;
    // m.add_function(wrap_pyfunction!(ada_minmax_u8_u8, m)?)?;
    // m.add_function(wrap_pyfunction!(ada_minmax_u16_u16, m)?)?;
    Ok(())
}
