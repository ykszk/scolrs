use devscol::{trimming_box_with_resample, BoundingBox};
use numpy::PyReadonlyArray2;
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

#[pymodule]
fn pyscol(_py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::init();
    m.add_function(wrap_pyfunction!(py_trimming_box_with_resample, m)?)?;
    Ok(())
}
