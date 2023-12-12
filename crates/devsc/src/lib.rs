use ndarray::{
    array, s, Array, Array1, ArrayView, ArrayView2, Axis, Dim, Dimension, Slice, SliceInfo,
    SliceInfoElem,
};
use ndarray_ndimage::{convolve, BorderMode};

use log::debug;
use ndarray_stats::QuantileExt;

pub trait Normalize<D> {
    fn minmax_normalize(&self) -> Result<Array<u8, D>, ndarray_stats::errors::MinMaxError>
    where
        D: Dimension;
}

impl<A, D> Normalize<D> for Array<A, D>
where
    A: PartialOrd + std::ops::Sub + Copy + Clone + Into<f64>,
    f64: From<A>,
    f64: From<<A as std::ops::Sub>::Output>,
    D: Dimension,
{
    fn minmax_normalize(&self) -> Result<Array<u8, D>, ndarray_stats::errors::MinMaxError>
    where
        D: Dimension,
    {
        let min_max = (self.min()?, self.max()?);
        let s = 255.0 / f64::from(*min_max.1 - *min_max.0);
        let sub = min_max.0.to_owned();
        Ok(self.mapv(|e| f64::round(s * f64::from(e - sub)) as u8))
    }
}

pub struct CDF(Array1<usize>);

impl<'a, D: Dimension> From<ArrayView<'a, u8, D>> for CDF {
    fn from(arr: ArrayView<u8, D>) -> Self {
        let hist = histogram(arr);
        Self(cumsum(hist))
    }
}

impl CDF {
    pub fn quantile(&self, q: f64) -> u8 {
        let thresh = (q * self.0[self.0.len() - 1] as f64) as usize;
        for (i, count) in self.0.indexed_iter() {
            if *count >= thresh {
                return i as u8;
            }
        }
        255
    }
}

fn histogram<D>(arr: ArrayView<u8, D>) -> Array1<usize>
where
    D: Dimension,
{
    let mut hist = Array1::zeros(256);
    arr.mapv(|e| hist[e as usize] += 1);
    hist
}

fn cumsum(mut hist: Array1<usize>) -> Array1<usize> {
    hist.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
    hist
}

pub type Index2d = [usize; 2];
pub type BoundingBox = (Index2d, Index2d);

pub fn create_slice(
    bb: &BoundingBox,
    original_shape: &(usize, usize),
) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> {
    let (bmin, bmax) = bb;
    let sy = Slice {
        start: bmin[0] as isize,
        end: if bmax[0] > original_shape.0 {
            None
        } else {
            Some(bmax[0] as isize)
        },
        step: 1,
    };
    let sx = Slice {
        start: bmin[1] as isize,
        end: if bmax[1] > original_shape.1 {
            None
        } else {
            Some(bmax[1] as isize)
        },
        step: 1,
    };
    s![sy, sx]
}

pub trait ElementWise {
    fn add(&self, other: &Self) -> Self;
    fn maximum(&self, other: &Self) -> Self;
    fn minimum(&self, other: &Self) -> Self;
    fn multiply(&self, other: &Self) -> Self;
}
impl ElementWise for Index2d {
    fn maximum(&self, other: &Self) -> Self {
        [self[0].max(other[0]), self[1].max(other[1])]
    }
    fn minimum(&self, other: &Self) -> Self {
        [self[0].min(other[0]), self[1].min(other[1])]
    }

    fn add(&self, other: &Self) -> Self {
        [self[0] + other[0], self[1] + other[1]]
    }

    fn multiply(&self, other: &Self) -> Self {
        [self[0] * other[0], self[1] * other[1]]
    }
}

/// Calculate bounding box of `predicate(pixel)==true`.
/// Return None when predicate is never true
pub fn bounding_box<F>(img: ArrayView2<u8>, predicate: F) -> Option<BoundingBox>
where
    F: Fn(u8) -> bool,
{
    let mut bmin = [img.nrows(), img.ncols()];
    let mut bmax = [0usize, 0usize];
    img.indexed_iter().for_each(|(index, p)| {
        if predicate(*p) {
            let index = [index.0, index.1];
            bmin = bmin.minimum(&index);
            bmax = bmax.maximum(&index);
        }
    });

    if bmin == [img.nrows(), img.ncols()] && bmax == [0usize, 0usize] {
        // bmin and bmax were not updated. i.e. empty bounding box
        None
    } else {
        Some((bmin, [bmax[0] + 1, bmax[1] + 1]))
    }
}

/// Calculate trimming parameter (bounding box)
pub fn trimming_box(
    img: ArrayView2<i16>,
) -> Result<BoundingBox, ndarray_stats::errors::MinMaxError> {
    let border_mode = BorderMode::Nearest;
    let weights = array![[0, 1, 0], [1, -4, 1], [0, 1, 0]];
    let laplacian = convolve(&img.view(), &weights.view(), border_mode, 0);
    let thresh_quantile = 0.1;
    let original_shape = (img.nrows(), img.ncols());

    let filtered = [
        img.to_owned().minmax_normalize()?,
        // sobel(&img, Axis(0), border_mode),
        // sobel(&img, Axis(1), border_mode),
        laplacian.minmax_normalize()?,
    ];

    let mut bmin = [0usize, 0usize];
    let mut bmax = [original_shape.0 + 1, original_shape.1 + 1];
    let max_iter = 10;
    for i_iter in 0..max_iter {
        let prev = (bmin, bmax);
        for (_i_filter, u8arr) in filtered.iter().enumerate() {
            // manually create slices because ndarray::slice doesn't accept 0..len
            let bboxed = u8arr.slice(create_slice(&(bmin, bmax), &original_shape));
            let cdf = CDF::from(bboxed);
            let (t_min, t_max) = (
                cdf.quantile(thresh_quantile),
                cdf.quantile(1.0 - thresh_quantile),
            );
            let predicate = |p| t_min < p && p < t_max;
            let bbox = bounding_box(bboxed, predicate);
            if let Some((local_bmin, local_bmax)) = bbox {
                bmax = local_bmax.add(&bmin); // update bmax first
                bmin = local_bmin.add(&bmin);
            }
        }
        if prev == (bmin, bmax) {
            debug!("Break trimming loop at {}", i_iter);
            break;
        } else {
            debug!("Updated bounding box: {:?} vs. {:?}", prev, (bmin, bmax));
        }
    }
    Ok((bmin, bmax))
}

/// Call `trimming_box` with resampled input for faster calculation
pub fn trimming_box_with_resample(
    img: ArrayView2<i16>,
    resample_step: usize,
) -> Result<BoundingBox, ndarray_stats::errors::MinMaxError> {
    let img = img.slice(s![..; resample_step, ..; resample_step]);
    let (bmin, bmax) = trimming_box(img)?;
    Ok((
        bmin.multiply(&[resample_step, resample_step]),
        bmax.multiply(&[resample_step, resample_step]),
    ))
}
