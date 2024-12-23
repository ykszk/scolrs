use std::{collections::HashMap, error::Error, path::Path, str::FromStr};

use devscol::{trimming_box_with_resample, BoundingBox, PredicateSource};
use labelme_rs::{
    image::{self, DynamicImage, GrayImage},
    LabelMeData, LabelMeDataWImage, ResizeParam,
};
use numpy::{
    ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use scolrs::{
    draw::{
        draw_components, ColorPalette, ColorPalettes, DrawComponent, DrawError, MeasureError,
        Painter,
    },
    CoronalMeasure, CoronalPointsAndCurve, DrawParam, HasImageMetadata, MeasureAndDraw,
    PointDataWithImage, SagittalMeasure, SagittalPoints, Scalable, UpdatePoints,
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
    #[error("Channel has to be 3 or 4 for 3D array. Found {0}")]
    ChannelMismatch(usize),
    #[error("Error in html: {0}")]
    Html(#[from] scolrs::draw::HtmlWrapError),
    #[error("Error in image: {0}")]
    Image(#[from] labelme_rs::ImageError),
    #[error("Uncategorized error: {0}")]
    Uncategorized(String),
}

impl From<PyScolError> for PyErr {
    fn from(e: PyScolError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}: {:?}", e, e.source()))
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
        3 => match shape[2] {
            3 => {
                let arr3d = arr.as_array().as_standard_layout().to_owned();
                Ok(DynamicImage::ImageRgb8(
                    image::RgbImage::from_raw(
                        shape[1] as u32,
                        shape[0] as u32,
                        arr3d.into_raw_vec(),
                    )
                    .unwrap(),
                ))
            }
            4 => {
                let arr3d = arr.as_array().as_standard_layout().to_owned();
                Ok(DynamicImage::ImageRgba8(
                    image::RgbaImage::from_raw(
                        shape[1] as u32,
                        shape[0] as u32,
                        arr3d.into_raw_vec(),
                    )
                    .unwrap(),
                ))
            }
            _ => Err(PyScolError::ChannelMismatch(shape[2])),
        },
        _ => Err(PyScolError::ImageShape(format!("{:?}", shape))),
    }
}

trait DeserializeAll<T> {
    fn deserialize_all(&self) -> Result<Vec<T>, PyScolError>
    where
        T: MeasureAndDraw + FromStr,
        <T as std::str::FromStr>::Err: std::fmt::Display;
}

impl<T> DeserializeAll<T> for Vec<String> {
    fn deserialize_all(&self) -> Result<Vec<T>, PyScolError>
    where
        T: MeasureAndDraw + FromStr,
        <T as std::str::FromStr>::Err: std::fmt::Display,
    {
        self.iter()
            .map(|s| {
                s.parse()
                    .map_err(|e| PyScolError::Uncategorized(format!("{}", e)))
            })
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_on_image<'a, T, S>(
    image: DynamicImage,
    data: T,
    draws: Vec<S>,
    hide: Vec<S>,
    draw_param_json: &str,
    svg_size: (usize, usize),
    label_colors: HashMap<String, String>,
    line_colors: HashMap<String, String>,
    overlay: Option<PyReadonlyArrayDyn<'_, u8>>,
    point_sets: Option<Vec<(String, Array2<f64>)>>,
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

    let mut document = painter.doc_w_background(&image)?;

    document = document.add(style);

    if let Some(overlay) = overlay {
        let overlay_image = ndarray_to_dynamic_image(overlay)?;
        let g = scolrs::draw::ImageOverlay::new(
            "heatmap".to_string(),
            "heatmap".to_string(),
            None,
            overlay_image,
        )
        .draw(&painter, &mut label_colors, &mut line_colors)?;
        let g = g.set("visibility", "hidden");
        document = document.add(g);
    }
    if let Some(point_sets) = point_sets {
        for (label, point_set) in point_sets {
            // let point_set = py_point_set.as_array().to_owned();
            let g = scolrs::draw::DrawPointSet::new(label.clone(), label, None, point_set).draw(
                &painter,
                &mut label_colors,
                &mut line_colors,
            )?;
            let g = g.set("visibility", "hidden");
            document = document.add(g);
        }
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

#[allow(clippy::too_many_arguments)]
fn draw_generic<T, S>(
    coronal_points_json: &str,
    json_path: &str,
    draws: Vec<String>,
    hide: Vec<String>,
    draw_param_json: &str,
    label_colors: HashMap<String, String>,
    line_colors: HashMap<String, String>,
    resize: Option<String>,
    overlay: Option<PyReadonlyArrayDyn<'_, u8>>,
    point_sets: Option<Vec<(String, PyReadonlyArray2<'_, f64>)>>,
) -> Result<String, PyScolError>
where
    T: Clone + UpdatePoints,
    for<'de> T: serde::Deserialize<'de>,
    LabelMeData: From<T>,
    S: MeasureAndDraw + FromStr,
    <S as std::str::FromStr>::Err: std::fmt::Display,
    for<'b> (&'b S, &'b T): Into<Box<dyn DrawComponent + 'b>>,
    S: Clone + Copy + PartialEq,
    T: HasImageMetadata + Scalable,
    <T as Scalable>::Error: std::fmt::Debug,
{
    let coronal_set: T = serde_json::from_str(coronal_points_json)?;
    let lm_data = LabelMeData::from(coronal_set.clone());
    let data_w_image = LabelMeDataWImage::try_from_data_and_path(lm_data, Path::new(json_path))?;

    let mut point_with_image = PointDataWithImage::<T>::new(coronal_set, data_w_image);

    let mut point_sets = point_sets.map(|point_sets| {
        point_sets
            .into_iter()
            .map(|(label, py_point_set)| {
                let point_set = py_point_set.as_array().to_owned();
                (label, point_set)
            })
            .collect::<Vec<_>>()
    });

    if let Some(resize) = resize {
        let resize = ResizeParam::try_from(resize.as_str())
            .map_err(|e| PyScolError::Uncategorized(format!("Error in resize: {}", e)))?;
        if let Some(point_sets) = point_sets.as_deref_mut() {
            let scale = resize.scale(
                point_with_image.data_image.image.width(),
                point_with_image.data_image.image.height(),
            );
            for (_, point_set) in point_sets.iter_mut() {
                point_set.mapv_inplace(|x| x * scale);
            }
        }

        point_with_image.resize(&resize);
    }

    let svg_size = (
        point_with_image.data_image.image.width() as usize,
        point_with_image.data_image.image.height() as usize,
    );

    let draws = if draws.is_empty() {
        S::all_draws()
    } else {
        draws.deserialize_all()?
    };
    let hide: Vec<S> = hide.deserialize_all()?;
    draw_on_image(
        point_with_image.data_image.image,
        point_with_image.data,
        draws,
        hide,
        draw_param_json,
        svg_size,
        label_colors,
        line_colors,
        overlay,
        point_sets,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_draw_coronal(
    coronal_points_json: &str,
    json_path: &str,
    draws: Vec<String>,
    hide: Vec<String>,
    draw_param_json: &str,
    label_colors: HashMap<String, String>,
    line_colors: HashMap<String, String>,
    resize: Option<String>,
    overlay: Option<PyReadonlyArrayDyn<'_, u8>>,
    point_sets: Option<Vec<(String, PyReadonlyArray2<'_, f64>)>>,
) -> Result<String, PyScolError> {
    draw_generic::<CoronalPointsAndCurve, CoronalMeasure>(
        coronal_points_json,
        json_path,
        draws,
        hide,
        draw_param_json,
        label_colors,
        line_colors,
        resize,
        overlay,
        point_sets,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_draw_sagittal(
    coronal_points_json: &str,
    json_path: &str,
    draws: Vec<String>,
    hide: Vec<String>,
    draw_param_json: &str,
    label_colors: HashMap<String, String>,
    line_colors: HashMap<String, String>,
    resize: Option<String>,
    overlay: Option<PyReadonlyArrayDyn<'_, u8>>,
    point_sets: Option<Vec<(String, PyReadonlyArray2<'_, f64>)>>,
) -> Result<String, PyScolError> {
    draw_generic::<SagittalPoints, SagittalMeasure>(
        coronal_points_json,
        json_path,
        draws,
        hide,
        draw_param_json,
        label_colors,
        line_colors,
        resize,
        overlay,
        point_sets,
    )
}

#[pyfunction]
fn py_wrap_in_html(
    svg: String,
    title: String,
    selector: Option<Vec<String>>,
) -> Result<String, PyScolError> {
    let selector = selector.unwrap_or_else(|| vec!["g.Component".to_string()]);
    Ok(scolrs::draw::wrap_in_html(svg, selector, title)?)
}

#[pyfunction]
fn py_calc_resize(
    image_shape_xy: (u32, u32),
    resize_param: &str,
) -> Result<(u32, u32), PyScolError> {
    let resize_param = ResizeParam::try_from(resize_param)
        .map_err(|e| PyScolError::Uncategorized(format!("Error in resize: {}", e)))?;
    Ok(resize_param.size(image_shape_xy.0, image_shape_xy.1))
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
    m.add_function(wrap_pyfunction!(py_wrap_in_html, m)?)?;
    m.add_function(wrap_pyfunction!(py_calc_resize, m)?)?;
    // m.add_function(wrap_pyfunction!(ada_minmax_u8_u8, m)?)?;
    // m.add_function(wrap_pyfunction!(ada_minmax_u16_u16, m)?)?;
    Ok(())
}
