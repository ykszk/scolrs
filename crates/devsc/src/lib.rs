use ndarray::{
    array, s, Array, Array1, ArrayView, ArrayView2, Axis, Dim, Dimension, Slice, SliceInfo,
    SliceInfoElem,
};
use ndarray_ndimage::{convolve, sobel, BorderMode};

use log::debug;
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};

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

impl<D: Dimension> From<ArrayView<'_, u8, D>> for CDF {
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
    arr.for_each(|e| hist[*e as usize] += 1);
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFilter {
    Original,
    SobelX,
    SobelY,
    Laplacian,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PredicateSource {
    pub filter: ImageFilter,
    pub quantile_min: Option<f64>,
    pub quantile_max: Option<f64>,
    /// Use specified value as threshold instead of quantile value calculated from cdf
    pub raw: bool,
}

/// Calculate trimming parameter (bounding box)
///
/// # Arguments
///
/// - `thresh_quantile` - Pixels below this value will be considered as noises
pub fn trimming_box(
    img: ArrayView2<i16>,
    predicate_sources: Vec<PredicateSource>,
) -> Result<BoundingBox, ndarray_stats::errors::MinMaxError> {
    let border_mode = BorderMode::Nearest;
    let weights = array![[0, 1, 0], [1, -4, 1], [0, 1, 0]];
    let original_shape = (img.nrows(), img.ncols());

    // at least one of quantile_min and quantile_max should be set
    // TODO: implement error handling
    assert!(predicate_sources
        .iter()
        .all(|source| { source.quantile_min.is_some() || source.quantile_max.is_some() }));

    let sources = predicate_sources
        .iter()
        .map(|source| {
            let filtered = match source.filter {
                ImageFilter::Original => img.to_owned().minmax_normalize()?,
                ImageFilter::SobelX => sobel(&img, Axis(1), border_mode)
                    .mapv(i16::abs)
                    .minmax_normalize()?,
                ImageFilter::SobelY => sobel(&img, Axis(0), border_mode)
                    .mapv(i16::abs)
                    .minmax_normalize()?,
                ImageFilter::Laplacian => convolve(&img.view(), &weights.view(), border_mode, 0)
                    .mapv(i16::abs)
                    .minmax_normalize()?,
            };
            Ok((source, filtered))
        })
        .collect::<Result<Vec<_>, ndarray_stats::errors::MinMaxError>>()?;

    let mut bmin = [0usize, 0usize];
    let mut bmax = [original_shape.0 + 1, original_shape.1 + 1];
    let max_iter = 10;
    for i_iter in 0..max_iter {
        let prev = (bmin, bmax);
        for (i_filter, source) in sources.iter().enumerate() {
            // manually create slices because ndarray::slice doesn't accept 0..len
            let u8arr = source.1.view();
            let bboxed = u8arr.slice(create_slice(&(bmin, bmax), &original_shape));
            let cdf = if source.0.raw {
                None
            } else {
                Some(CDF::from(bboxed))
            };
            let t_min = source.0.quantile_min.map(|min| {
                if source.0.raw {
                    (min * 255.0) as u8
                } else {
                    cdf.as_ref().unwrap().quantile(min)
                }
            });
            let t_max = source.0.quantile_max.map(|max| {
                if source.0.raw {
                    (max * 255.0) as u8
                } else {
                    cdf.unwrap().quantile(max)
                }
            });
            debug!(
                "Quantile for {}: {:?} -> ({:?}, {:?})",
                i_filter, source.0, t_min, t_max,
            );
            let bb = match (t_min, t_max) {
                (None, None) => unreachable!(),
                (Some(t_min), None) => bounding_box(bboxed, |p| p >= t_min),
                (None, Some(t_max)) => bounding_box(bboxed, |p| p < t_max),
                (Some(t_min), Some(t_max)) => bounding_box(bboxed, |p| p < t_max && p >= t_min),
            };
            if let Some((local_bmin, local_bmax)) = bb {
                if local_bmax != [0usize, 0usize]
                    || local_bmax != [original_shape.0 + 1, original_shape.1 + 1]
                {
                    bmax = local_bmax.add(&bmin); // update bmax first
                    bmin = local_bmin.add(&bmin);
                    debug!(
                        "Updated bounding box {:?}: {:?} vs. {:?}",
                        source.0.filter, bmin, bmax
                    );
                }
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
    predicate_sources: Vec<PredicateSource>,
    resample_step: usize,
) -> Result<BoundingBox, ndarray_stats::errors::MinMaxError> {
    let img = img.slice(s![..; resample_step, ..; resample_step]);
    let (bmin, bmax) = trimming_box(img, predicate_sources)?;
    Ok((
        bmin.multiply(&[resample_step, resample_step]),
        bmax.multiply(&[resample_step, resample_step]),
    ))
}
