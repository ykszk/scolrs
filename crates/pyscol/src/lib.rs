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
        ) -> PyResult<&'py PyArray2<$output_type>> {
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
            .map(|a| a.into_pyarray(py))
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

#[pymodule]
fn pyscol(_py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::init();
    m.add_function(wrap_pyfunction!(py_trimming_box_with_resample, m)?)?;
    m.add_function(wrap_pyfunction!(clahe_u8_u8, m)?)?;
    m.add_function(wrap_pyfunction!(clahe_u16_u8, m)?)?;
    m.add_function(wrap_pyfunction!(clahe_u16_u16, m)?)?;
    // m.add_function(wrap_pyfunction!(ada_minmax_u8_u8, m)?)?;
    // m.add_function(wrap_pyfunction!(ada_minmax_u16_u16, m)?)?;
    Ok(())
}
