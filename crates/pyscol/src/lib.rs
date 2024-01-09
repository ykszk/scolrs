use devscol::{trimming_box_with_resample, BoundingBox};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "trimming_box_with_resample")]
fn py_trimming_box_with_resample(
    arr2d: PyReadonlyArray2<'_, i16>,
    thresh_quantile: f64,
    resample_step: usize,
) -> PyResult<BoundingBox> {
    let arr2d = arr2d.as_array();
    trimming_box_with_resample(arr2d, thresh_quantile, resample_step)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Error in trimming: {}", e)))
}

#[pyfunction]
fn clahe_u16_u8<'py>(
    py: Python<'py>,
    arr2d: PyReadonlyArray2<'py, u16>,
    grid_width: u32,
    grid_height: u32,
    clip_limit: u32,
    tile_sample: f64,
) -> PyResult<&'py PyArray2<u8>> {
    let arr2d = arr2d.as_array();
    clahe::clahe_ndarray(arr2d, grid_width, grid_height, clip_limit, tile_sample)
        .map(|a| a.into_pyarray(py))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Clahe error: {}", e)))
}

#[pyfunction]
fn clahe_u8_u8<'py>(
    py: Python<'py>,
    arr2d: PyReadonlyArray2<'py, u8>,
    grid_width: u32,
    grid_height: u32,
    clip_limit: u32,
    tile_sample: f64,
) -> PyResult<&'py PyArray2<u8>> {
    let arr2d = arr2d.as_array();
    clahe::clahe_ndarray(arr2d, grid_width, grid_height, clip_limit, tile_sample)
        .map(|a| a.into_pyarray(py))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Clahe error: {}", e)))
}

#[pymodule]
fn pyscol(_py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::init();
    m.add_function(wrap_pyfunction!(py_trimming_box_with_resample, m)?)?;
    m.add_function(wrap_pyfunction!(clahe_u8_u8, m)?)?;
    m.add_function(wrap_pyfunction!(clahe_u16_u8, m)?)?;
    Ok(())
}
