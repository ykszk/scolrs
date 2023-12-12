use devscol::{trimming_box_with_resample, BoundingBox};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "trimming_param_with_resample")]
fn py_trimming_param_with_resample(
    arr2d: PyReadonlyArray2<'_, i16>,
    resample_step: usize,
) -> PyResult<BoundingBox> {
    let arr2d = arr2d.as_array();
    Ok(trimming_box_with_resample(arr2d, resample_step))
}

#[pymodule]
fn pyscol(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_trimming_param_with_resample, m)?)?;
    Ok(())
}
