use clap::ValueEnum;
use labelme_rs::LabelMeData;
use log::{debug, error};
use ndarray::{
    concatenate, s, stack, Array1, Array2, Array3, ArrayBase, ArrayView1, ArrayView2, ArrayView3,
    Axis, Data,
};
use ndarray_stats::QuantileExt;
pub use serde;
use serde::{Deserialize, Serialize};
use std::cmp::Ord;
use std::fmt::Display;
use std::iter::zip;
use std::ops::AddAssign;
use std::result::Result;
use strum::VariantArray;
use thiserror::Error;
mod defs;
pub use defs::*;
mod draw;
pub use draw::*;
pub mod head_neck;

#[derive(Error, Debug)]
pub enum ScolError {
    #[error("Invalid point count for shape {0}: {1} != {2}")]
    InvalidShape(String, usize, usize),
    #[error("Invalid point count for {0}: {1}")]
    InvalidPointCount(String, usize),
    #[error("Invalid combination of the numbeer of points for {0} and {1}: {2} vs. {3}")]
    InvalidPointCombo(String, String, usize, usize),
    #[error("Linalg error")]
    Linalg(#[from] rulinalg::error::Error),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ImageMetadata {
    pub path: String,
    pub height: usize,
    pub width: usize,
    pub spacing_xy: (f64, f64),
    pub unit: String,
}

impl Default for ImageMetadata {
    fn default() -> Self {
        let spacing_xy = (1.0, 1.0);
        let unit = "px".to_string();
        Self {
            path: String::default(),
            height: 0,
            width: 0,
            spacing_xy,
            unit,
        }
    }
}

impl From<&LabelMeData> for ImageMetadata {
    fn from(data: &LabelMeData) -> Self {
        let path = data.imagePath.clone();
        let height = data.imageHeight;
        let width = data.imageWidth;
        Self {
            path,
            height,
            width,
            ..Default::default()
        }
    }
}

fn extract_points(data: &LabelMeData, label: &str) -> Result<Array2<f64>, ScolError> {
    let tuples: Result<Vec<_>, _> = data
        .shapes
        .iter()
        .filter_map(|shape| {
            if shape.shape_type == "point" && shape.label == label {
                Some(&shape.points)
            } else {
                None
            }
        })
        .map(|points| {
            if points.len() != 1 {
                return Err(ScolError::InvalidShape(label.to_string(), 1, points.len()));
            }
            Ok(points[0])
        })
        .collect();
    let tuples = tuples?;
    let mut vec = Vec::with_capacity(tuples.len() * 2);
    for t in tuples.iter() {
        vec.push(t.0);
        vec.push(t.1);
    }
    let arr = Array2::from_shape_vec((tuples.len(), 2), vec).unwrap();
    Ok(arr)
}

#[derive(Debug, Clone)]
pub struct AtMost2<T>(pub T);

trait ValidateLength {
    fn validate_length(&self, expected_len: usize) -> Result<(), MeasureError>;
    fn validate_length_more_than(&self, min_len: usize) -> Result<(), MeasureError>;
}

impl<S, I> ValidateLength for ndarray::ArrayBase<S, I>
where
    S: ndarray::Data<Elem = f64>,
    I: ndarray::Dimension,
{
    fn validate_length(&self, expected_len: usize) -> Result<(), MeasureError> {
        if self.len_of(Axis(0)) != expected_len {
            return Err(MeasureError::InvalidNumberOfPoints(
                crate::InvalidNumberOfPoints::IncorrectNumberOfPoints(expected_len, self.len()),
            ));
        }
        Ok(())
    }
    fn validate_length_more_than(&self, min_len: usize) -> Result<(), MeasureError> {
        if self.len_of(Axis(0)) < min_len {
            return Err(MeasureError::InvalidNumberOfPoints(
                crate::InvalidNumberOfPoints::TooFewPoints(min_len, self.len()),
            ));
        }
        Ok(())
    }
}

#[allow(dead_code)] // TODO: maybe remove?
trait LeftFirst {
    /// sort array by x
    fn left_first(self) -> Self;
}

impl LeftFirst for Array2<f64> {
    fn left_first(mut self) -> Self {
        if self.len_of(Axis(0)) == 2 && self[[0, 0]] > self[[1, 0]] {
            self.swap([0, 0], [1, 0]);
            self.swap([0, 1], [1, 1]);
        }
        if self.len_of(Axis(0)) > 2 {
            error!("Array length is greater than 2");
        }
        self
    }
}

trait HasCornerPoints {
    fn top_left(&self) -> ArrayView2<f64>;
    fn top_right(&self) -> ArrayView2<f64>;
    fn bottom_left(&self) -> ArrayView2<f64>;
    fn bottom_right(&self) -> ArrayView2<f64>;
}

/// Polynomial fitting of `deg` degrees
pub fn polyfit<S>(
    xs: ndarray::ArrayBase<S, ndarray::Ix1>,
    ys: ndarray::ArrayBase<S, ndarray::Ix1>,
    deg: usize,
) -> Result<ndarray::Array1<f64>, rulinalg::error::Error>
where
    S: ndarray::Data<Elem = f64>,
{
    let mut vander = Array2::zeros([xs.len(), deg + 1]);

    // f64 is required for higher degrees
    let xs = xs.mapv(|x| x);
    let ys = ys.mapv(|x| x);

    for d in 0..=deg {
        vander
            .slice_mut(s![.., d])
            .assign(&xs.mapv(|x| x.powi(d as i32)));
    }
    use rulinalg::matrix::{BaseMatrix, Matrix};
    use rulinalg::vector::Vector;

    let vander = Matrix::new(
        vander.nrows(),
        vander.ncols(),
        vander.into_raw_vec_and_offset().0,
    );
    let ys = Vector::new(ys.into_raw_vec_and_offset().0);
    let a = vander.transpose() * &vander;

    let b = &vander.transpose() * &ys;
    a.solve(b)
        .map(|c| ndarray::Array1::from_iter(c).mapv(|e| e))
}

/// Calculate polynomial curve points
pub fn polynomial<S, T>(
    xs: ndarray::ArrayBase<S, ndarray::Ix1>,
    coef: ndarray::ArrayBase<T, ndarray::Ix1>,
) -> Array1<f64>
where
    S: ndarray::Data<Elem = f64>,
    T: ndarray::Data<Elem = f64>,
{
    let mut ys: Array1<f64> = ndarray::Array::zeros(xs.len());
    for (i, c) in coef.iter().enumerate() {
        ys.add_assign(&xs.mapv(|x| c * x.powi(i as i32)));
    }
    ys
}

/// Point sets representing a spine
#[derive(Debug, Clone)]
pub struct Spine {
    pub c7tls: C7TLS,

    /// Corner points of C7, thoracic, and lumbar vertebrae
    /// The number of points/vertebrae can vary because some spine have 4 or 6 lumbar vertebrae
    pub v_c7tl: VertebraeC7TL,
    pub c_c7tl: Centroids,
    /// Coefficients of the polynomial curve of the spine
    pub c_coefs: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct CoronalPoints {
    pub spine: Spine,
    pub clavicle: AtMost2<Array2<f64>>,
    pub shoulder: AtMost2<Array2<f64>>,
    pub pelvis: AtMost2<Array2<f64>>,
    pub femoral_head: AtMost2<Array2<f64>>,

    pub image_data: ImageMetadata,
}

#[derive(Debug, Clone)]
pub struct SagittalPoints {
    pub spine: Spine,
    pub femoral_head: AtMost2<Array2<f64>>,

    pub image_data: ImageMetadata,
}

/// Spinal curve represented by superior and inferior indices of vertebrae
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Curve {
    pub sup: usize,
    pub inf: usize,
}

pub const T2T5_CURVE: Curve = Curve {
    sup: VertebralIndex::T2 as usize,
    inf: VertebralIndex::T5 as usize,
};

pub const T5T12_CURVE: Curve = Curve {
    sup: VertebralIndex::T5 as usize,
    inf: VertebralIndex::T12 as usize,
};

pub const T10L2_CURVE: Curve = Curve {
    sup: VertebralIndex::T10 as usize,
    inf: VertebralIndex::L2 as usize,
};

/// Curve position for determining if PT is structural
pub const KYOPHOSIS_CURVE_PT: Curve = T2T5_CURVE;
/// Curve position for determining if MT is structural
pub const KYOPHOSIS_CURVE_MT: Curve = T10L2_CURVE;
/// Curve position for determining if TLL is structural
pub const KYOPHOSIS_CURVE_TLL: Curve = T10L2_CURVE;

impl Display for Curve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sup = VertebralIndex::from(self.sup as u8);
        let inf = VertebralIndex::from(self.inf as u8);
        write!(f, "{sup}{inf}")
    }
}

/// Set of PT, MT, and TLL curves
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
#[serde(deny_unknown_fields, rename_all = "UPPERCASE")]
pub struct CurveSet {
    pub pt: Option<(Curve, f64)>,
    pub mt: Option<(Curve, f64)>,
    pub tll: Option<(Curve, f64)>,
}

impl CurveSet {
    fn apices(&self, scol: &Spine) -> ApexSet {
        ApexSet {
            pt: if let Some((c, _)) = self.pt.as_ref() {
                Some(scol.id_apex(c))
            } else {
                None
            },
            mt: if let Some((c, _)) = self.mt.as_ref() {
                Some(scol.id_apex(c))
            } else {
                None
            },
            tll: if let Some((c, _)) = self.tll.as_ref() {
                Some(scol.id_apex(c))
            } else {
                None
            },
        }
    }
}

/// Set of PT, MT, and TLL apices
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
#[serde(deny_unknown_fields, rename_all = "UPPERCASE")]
pub struct ApexSet {
    pub pt: Option<VertebraDiscIndex>,
    pub mt: Option<VertebraDiscIndex>,
    pub tll: Option<VertebraDiscIndex>,
}

/// Major curve in Lenke classification
/// PT can't be the major curve
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum MajorCurve {
    MT,
    TLL,
}

/// Descriptor for scoliosis consisting of curves, apices, and major-curve-kind (MT or TLL)
#[derive(Serialize, Deserialize, Debug)]
pub struct ScolDesc {
    pub curves: CurveSet,
    pub apices: ApexSet,
    pub major_curve: Option<MajorCurve>,
}

impl ScolDesc {
    pub fn new(curves: CurveSet, apices: ApexSet, major_curve: Option<MajorCurve>) -> Self {
        Self {
            curves,
            apices,
            major_curve,
        }
    }
}

impl TryFrom<&LabelMeData> for ScolDesc {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let spine = Spine::try_from(data)?;
        let (curves, apex_set, major_curve) = spine.identify_curves();
        Ok(ScolDesc::new(curves, apex_set, major_curve))
    }
}

/// Trait that enables `arr.l2norm()` to a vector (Array1 or ArrayView1)
pub trait L2Norm<T> {
    fn l2norm(&self) -> T;
}

impl<S> L2Norm<f64> for ndarray::ArrayBase<S, ndarray::Ix1>
where
    S: ndarray::Data<Elem = f64>,
{
    fn l2norm(&self) -> f64 {
        self.mapv(|a| a * a).sum().sqrt()
    }
}

/// Check monotonicity of arrays
pub trait LineCharacteristic<T> {
    /// Check if the array is not-strictly increasing or decreasing
    fn is_monotonic(&self) -> bool;

    /// Check if the array has the same sign
    fn have_same_signs(&self) -> bool;
}

impl<S> LineCharacteristic<f64> for ndarray::ArrayBase<S, ndarray::Ix1>
where
    S: ndarray::Data<Elem = f64>,
{
    fn is_monotonic(&self) -> bool {
        if self.len() < 2 {
            return true;
        }

        let mut signs = Vec::new();
        for i in 1..self.len() {
            signs.push(if self[i] > self[i - 1] {
                1
            } else if self[i] < self[i - 1] {
                -1
            } else {
                0
            });
        }

        let first_sign = signs[0];
        signs.iter().all(|&sign| sign == first_sign)
    }

    fn have_same_signs(&self) -> bool {
        let mut non_zeros = self.iter().filter(|&&x| x != 0.0);
        if let Some(first) = non_zeros.clone().next() {
            non_zeros.all(|&x| x.signum() == first.signum())
        } else {
            true
        }
    }
}

/// Angle between two lines in degrees
/// TODO: Check the difference from [`angle_between`]?
pub fn angle_from_lines(line1: ArrayView2<f64>, line2: ArrayView2<f64>) -> Option<f64> {
    let v_sup = &line1.index_axis(Axis(0), 1) - &line1.index_axis(Axis(0), 0);
    let v_inf = &line2.index_axis(Axis(0), 1) - &line2.index_axis(Axis(0), 0);
    let len_sup = v_sup.l2norm();
    let len_inf = v_inf.l2norm();
    if len_sup == 0.0 || len_inf == 0.0 {
        return None;
    }
    let v_sup = &v_sup / len_sup;
    let v_inf = &v_inf / len_inf;
    let cos = v_sup.dot(&v_inf);
    let cos = cos.clamp(-1.0, 1.0);
    let deg = cos.acos().to_degrees();
    if v_sup[1] < v_inf[1] {
        Some(-deg)
    } else {
        Some(deg)
    }
}

impl Spine {
    /// Corner points of thoracic and lumbar vertebrae
    pub fn tl_corners(&self) -> Corners<ndarray::ViewRepr<&f64>> {
        let tl = self.v_c7tl.0.slice(s![1.., .., ..]);
        Corners(tl)
    }

    pub fn tl_vert_disc_corners(&self) -> Corners<ndarray::OwnedRepr<f64>> {
        let vert_corners = self.tl_corners();
        let disc_corners = vert_corners.between();
        let mut vert_disc_corners: Array3<f64> = ndarray::Array::zeros((
            vert_corners.0.len_of(Axis(0)) + disc_corners.len_of(Axis(0)),
            4,
            2,
        ));
        for (i, vc) in vert_corners.0.axis_iter(Axis(0)).enumerate() {
            vert_disc_corners.slice_mut(s![2 * i, .., ..]).assign(&vc);
        }
        for (i, vc) in disc_corners.axis_iter(Axis(0)).enumerate() {
            vert_disc_corners
                .slice_mut(s![2 * i + 1, .., ..])
                .assign(&vc);
        }
        Corners(vert_disc_corners)
    }
    pub fn tl_sup_lines(&self) -> ArrayView3<'_, f64> {
        self.v_c7tl.0.slice(s![1.., ..2, ..])
    }
    pub fn tl_inf_lines(&self) -> ArrayView3<'_, f64> {
        self.v_c7tl.0.slice(s![1.., 2.., ..])
    }
    pub fn tl_centroids(&self) -> ArrayView2<'_, f64> {
        self.c_c7tl.slice(s![1.., ..])
    }

    pub fn sup_plate(&self, index: usize) -> ArrayView2<'_, f64> {
        self.c7tls.0.slice(s![index + 1, ..2, ..])
    }
    pub fn inf_plate(&self, index: usize) -> ArrayView2<'_, f64> {
        self.c7tls.0.slice(s![index + 1, 2.., ..])
    }

    pub fn sacral_sup_plate(&self) -> ArrayView2<f64> {
        self.c7tls
            .0
            .slice(s![self.c7tls.0.len_of(Axis(0)) - 1, 0..2, ..])
    }

    // mean point of sacral TL and TR
    pub fn sacral_center(&self) -> Array1<f64> {
        self.sacral_sup_plate().mean_axis(Axis(0)).unwrap()
    }

    pub fn lumbar_modifier(&self, apex: VertebraDiscIndex) -> LumbarModifier {
        let x_scvl = self.sacral_center()[0];
        // vertebral index -> vertebral index or disc index -> vertebral index right above the disc
        let index = (apex as u8 / 2) as usize;
        let vertebra = self.v_c7tl.0.index_axis(Axis(0), index + 1);
        let v_xs = vertebra.index_axis(Axis(1), 0);
        let x_min = v_xs.min().unwrap();
        let x_max = v_xs.max().unwrap();
        if x_scvl < *x_min || *x_max < x_scvl {
            LumbarModifier::C
        } else {
            // TODO: implement pedicle checking
            LumbarModifier::AorB
        }
    }

    pub fn spinal_poly(&self, xs: ArrayView1<f64>) -> Array1<f64> {
        polynomial(xs, self.c_coefs.view())
    }

    fn find_largest_curve(&self) -> Option<(Curve, f64)> {
        let n = self.tl_corners().0.len_of(ndarray::Axis(0));
        let mut curves = Vec::new();
        for sup in 0..n - 2 {
            for inf in sup..n {
                if self.is_valid_curve(sup, inf) {
                    curves.push(Curve { sup, inf });
                } else {
                    break;
                }
            }
        }
        self._find_largest_curve(curves)
    }

    fn find_largest_up(&self, inf: usize) -> Option<(Curve, f64)> {
        let mut curves = Vec::new();
        if inf <= 1 {
            return None;
        }
        let end = inf - 2;
        // search sup from bottom to top (by .rev()) so that we can break early from the loop
        for sup in (0..end).rev() {
            if self.is_valid_curve(sup, inf) {
                curves.push(Curve { sup, inf });
            } else {
                break;
            }
        }
        self._find_largest_curve(curves)
    }

    fn find_largest_down(&self, sup: usize) -> Option<(Curve, f64)> {
        let n = self.tl_corners().0.len_of(ndarray::Axis(0));
        let mut curves = Vec::new();
        let start = (sup + 2).min(n);
        for inf in start..n {
            if self.is_valid_curve(sup, inf) {
                curves.push(Curve { sup, inf });
            } else {
                break;
            }
        }
        self._find_largest_curve(curves)
    }

    /// Curve is invalid when it contains S curve, i.e. checking if the curve is convex
    fn is_valid_curve(&self, sup: usize, inf: usize) -> bool {
        // inf - sup <= 2
        if inf <= sup + 2 {
            return true;
        }
        let centroids = self.tl_centroids();
        let ys = centroids.slice(s![sup + 1..inf, 1]);
        let xs = polynomial(ys, self.c_coefs.view());

        // coefficients of the second derivative
        let coefs2: Array1<f64> = self
            .c_coefs
            .slice(s![2..])
            .iter()
            .enumerate()
            .map(|(i, c)| c * ((i + 1) * (i + 2)) as f64)
            .collect();

        let ddxs = polynomial(ys, coefs2.view());
        debug!("sup: {}, inf: {}", sup, inf);
        debug!("xs: {:?}", xs);
        debug!("ys: {:?}", ys);
        debug!("ddxs: {:?}", ddxs);

        debug!("ddxs.have_same_signs(): {}", ddxs.have_same_signs());
        ddxs.have_same_signs()
    }

    fn _find_largest_curve(&self, curves: Vec<Curve>) -> Option<(Curve, f64)> {
        let angles: Vec<_> = curves
            .iter()
            .filter_map(|c| self.angle(c).map(|a| (c, a)))
            .collect();
        if angles.is_empty() {
            return None;
        }
        let (i_max, _max_value) = angles.iter().map(|e| e.1.abs()).enumerate().fold(
            (0, angles[0].1),
            |(i_max, max_value), (i, value)| {
                if value > max_value {
                    (i, value)
                } else {
                    (i_max, max_value)
                }
            },
        );
        Some((angles[i_max].0.clone(), angles[i_max].1))
    }

    /// Identify the apex of the curve
    pub fn id_apex(&self, curve: &Curve) -> VertebraDiscIndex {
        let vert_disc_corners = self.tl_vert_disc_corners();
        let vd_centroids: Centroids = vert_disc_corners.into();
        let sup = VertebraDiscIndex::from(VertebralIndex::from(curve.sup as u8)) as usize;
        let inf = VertebraDiscIndex::from(VertebralIndex::from(curve.inf as u8)) as usize;
        let xs = polynomial(vd_centroids.slice(s![sup..=inf, 1]), self.c_coefs.view());
        if xs.is_monotonic() {
            VertebraDiscIndex::from((sup + inf) as u8 / 2)
        } else {
            let ts = (2.0 * &xs - xs[0] - xs[xs.len() - 1]).mapv(|e| e.abs());
            let i_max = ts.argmax().unwrap();
            VertebraDiscIndex::from((i_max + sup) as u8)
        }
    }

    fn find_all_down(&self, mut sup: usize) -> Vec<(Curve, f64)> {
        let mut curves = Vec::new();
        while let Some(largest_curve) = self.find_largest_down(sup) {
            sup = largest_curve.0.inf;
            curves.push(largest_curve);
        }
        curves
    }

    fn find_all_up(&self, mut inf: usize) -> Vec<(Curve, f64)> {
        let mut curves = Vec::new();
        while let Some(largest_curve) = self.find_largest_up(inf) {
            inf = largest_curve.0.sup;
            curves.push(largest_curve);
        }
        curves
    }

    pub fn find_all_curves(&self) -> Vec<(Curve, f64)> {
        if let Some(largest_curve) = self.find_largest_curve() {
            let mut downs = self.find_all_down(largest_curve.0.inf);
            let ups = self.find_all_up(largest_curve.0.sup);
            downs.extend(vec![largest_curve]);
            downs.extend(ups);
            downs
        } else {
            Vec::default()
        }
    }

    pub fn identify_curves(&self) -> (CurveSet, ApexSet, Option<MajorCurve>) {
        let mut curves = CurveSet::default();
        let mut major_curve = None;
        if let Some(largest_curve) = self.find_largest_curve() {
            let major_apex = self.id_apex(&largest_curve.0);
            major_curve = if major_apex <= VertebraDiscIndex::T5 {
                // largest curve is PT
                debug!("PT is the largest curve");
                if let Some(mt) = self.find_largest_down(largest_curve.0.inf) {
                    curves.tll = self.find_largest_down(mt.0.inf);
                    curves.mt = Some(mt);
                }
                curves.pt = Some(largest_curve);
                Some(MajorCurve::MT) // PT is never major
            } else if major_apex <= VertebraDiscIndex::DiscT11T12 {
                // largest curve is MT
                debug!("MT is the largest curve");
                curves.pt = self.find_largest_up(largest_curve.0.sup);
                curves.tll = self.find_largest_down(largest_curve.0.inf);
                curves.mt = Some(largest_curve);
                Some(MajorCurve::MT)
            } else {
                // largest curve is TLL
                debug!("TLL is the largest curve");
                if let Some(mt) = self.find_largest_up(largest_curve.0.sup) {
                    curves.pt = self.find_largest_up(mt.0.sup);
                    curves.mt = Some(mt);
                }
                curves.tll = Some(largest_curve);
                Some(MajorCurve::TLL)
            };
        }
        let apices = curves.apices(self);
        (curves, apices, major_curve)
    }

    /// Calculate Cobb angle in degrees
    pub fn angle(&self, curve: &Curve) -> Option<f64> {
        let sup_line = self.sup_plate(curve.sup);
        let inf_line = self.inf_plate(curve.inf);
        angle_from_lines(sup_line, inf_line)
    }
}

impl TryFrom<&LabelMeData> for Spine {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let c7tls = C7TLS::try_from(data)?;
        let v_c7tl = VertebraeC7TL::try_from(data)?;
        let c_c7tl: Centroids = Corners(v_c7tl.0.view()).into();
        let c_coefs = polyfit(c_c7tl.slice(s![1.., 1]), c_c7tl.slice(s![1.., 0]), 6)?;

        Ok(Spine {
            c7tls,
            v_c7tl,
            c_c7tl,
            c_coefs,
        })
    }
}

fn _new_at_most2(data: &LabelMeData, label: &str) -> Result<AtMost2<Array2<f64>>, ScolError> {
    let points = extract_points(data, label)?;
    if points.len_of(Axis(0)) > 2 {
        return Err(ScolError::InvalidPointCount(
            label.to_string(),
            points.len_of(Axis(0)),
        ));
    }
    Ok(AtMost2(points))
}

impl TryFrom<&LabelMeData> for CoronalPoints {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let spine = Spine::try_from(data)?;
        // TODO: accept invalid number of points and let later functions handle it
        let clavicle = _new_at_most2(data, "Clavicle")?;
        let shoulder = _new_at_most2(data, "Shoulder")?;
        let pelvis = _new_at_most2(data, "Pelvis")?;
        let femoral_head = _new_at_most2(data, "FemoralHead")?;

        let image_data = ImageMetadata::from(data);

        Ok(CoronalPoints {
            spine,
            clavicle,
            pelvis,
            shoulder,
            femoral_head,
            image_data,
        })
    }
}

impl TryFrom<&LabelMeData> for SagittalPoints {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let spine = Spine::try_from(data)?;
        let femoral_head = _new_at_most2(data, "FemoralHead")?;

        let image_data = ImageMetadata::from(data);

        Ok(SagittalPoints {
            spine,
            femoral_head,
            image_data,
        })
    }
}

/// Corner points of all C7, thoracic, and lumbar vertebrae and sacrum top plate.
///
/// Note: sacrum corners = (TL, TR, copy of TL, copy of TR)
#[derive(Debug, Clone)]
pub struct C7TLS(pub Array3<f64>);

/// C7, thoracic and lumber vertebrae
#[derive(Debug, Clone)]
pub struct VertebraeC7TL(pub Array3<f64>);

/// Thoracic and lumber vertebrae
pub struct VertebraeTL<'a>(pub ArrayView3<'a, f64>);

impl<'a> From<&'a VertebraeC7TL> for VertebraeTL<'a> {
    fn from(c7tl: &'a VertebraeC7TL) -> Self {
        let tl = c7tl.0.slice(s![1.., .., ..]);
        Self(tl)
    }
}

/// Corner points of structures.
///
/// Points are in [tl, tr, bl, br] order
pub struct Corners<S: Data<Elem = f64>>(pub ArrayBase<S, ndarray::Ix3>);

impl<S: Data<Elem = f64>> Corners<S> {
    pub fn between(&self) -> Array3<f64> {
        let bottom = self.0.slice(s![1.., ..2, ..]);
        let top = self.0.slice(s![..(self.0.shape()[0] - 1), 2.., ..]);
        let between = ndarray::concatenate![Axis(1), top, bottom];
        between
    }
}

type Centroids = Array2<f64>;
impl<S: Data<Elem = f64>> From<Corners<S>> for Centroids {
    /// Calculate centroids from list of four corners.
    /// Centroids are not geometric centers but the intersections of mid-lines
    fn from(corners: Corners<S>) -> Self {
        let top = corners.0.slice(s![.., ..2, ..]).mean_axis(Axis(1)).unwrap();
        let bottom = corners.0.slice(s![.., 2.., ..]).mean_axis(Axis(1)).unwrap();
        let left = corners
            .0
            .slice(s![.., 0..;2, ..])
            .mean_axis(Axis(1))
            .unwrap();
        let mut right = corners
            .0
            .slice(s![.., 1..;2, ..])
            .mean_axis(Axis(1))
            .unwrap();
        // reuse `right` to store centroids
        for (t, (b, (l, mut r))) in zip(
            top.axis_iter(Axis(0)),
            zip(
                bottom.axis_iter(Axis(0)),
                zip(left.axis_iter(Axis(0)), right.axis_iter_mut(Axis(0))),
            ),
        ) {
            // [python - Numpy and line intersections - Stack Overflow](https://stackoverflow.com/questions/3252194/numpy-and-line-intersections/57821199#57821199)
            let (a1, a2) = (&t, &b);
            let (b1, b2) = (&l, &r);
            let da = a2 - a1;
            let db = b2 - b1;
            let dp = a1 - b1;
            let mut dap = da.clone();
            dap[[0]] = -da[[1]];
            dap[[1]] = da[[0]];
            let denom = dap.dot(&db);
            if denom == 0.0 {
                unreachable!("Parallel mid-lines of a vertebrae thus no centroid");
            } else {
                let num = dap.dot(&dp);
                let c = num / denom * db + b1;
                r.assign(&c);
            }
        }
        right
    }
}

const FRONTAL_ANGLE_THRESH: f64 = 25.0_f64;
const BEND_ANGLE_THRESH: f64 = 25.0_f64;
const LATERAL_ANGLE_THRESH: f64 = 20.0_f64;

/// Set of `Spine`s required for Lenke classification
pub struct Study {
    pub coronal: Spine,
    pub left_bend: Option<Spine>,
    pub right_bend: Option<Spine>,
    pub sagittal: Option<Spine>,
}

impl Study {
    pub fn new(
        coronal: Spine,
        left_bend: Option<Spine>,
        right_bend: Option<Spine>,
        sagittal: Option<Spine>,
    ) -> Study {
        Study {
            coronal,
            left_bend,
            right_bend,
            sagittal,
        }
    }

    pub fn full(coronal: Spine, left_bend: Spine, right_bend: Spine, sagittal: Spine) -> Study {
        let left_bend = Some(left_bend);
        let right_bend = Some(right_bend);
        let sagittal = Some(sagittal);
        Study {
            coronal,
            left_bend,
            right_bend,
            sagittal,
        }
    }

    pub fn minor_param(
        &self,
        coronal: f64,
        coronal_curve: &Curve,
        sagittal_curve: &Curve,
    ) -> MinorStructuralParam {
        let right_bend = self
            .right_bend
            .as_ref()
            .map(|scol| scol.angle(coronal_curve).unwrap());
        let left_bend = self
            .left_bend
            .as_ref()
            .map(|scol| scol.angle(coronal_curve).unwrap());
        let sagittal = self
            .sagittal
            .as_ref()
            .map(|scol| (sagittal_curve.clone(), scol.angle(sagittal_curve).unwrap()));
        MinorStructuralParam {
            coronal,
            right_bend,
            left_bend,
            sagittal,
        }
    }

    pub fn chart(&self, curve_set: &CurveSet, major_curve: MajorCurve) -> Chart {
        let pt = curve_set.pt.as_ref().map(|pt| {
            self.minor_param(pt.1, &pt.0, &KYOPHOSIS_CURVE_PT)
                .curve_type()
        });
        let mt = if major_curve == MajorCurve::MT {
            Some(RegionalCurveType::Structural(StructuralReason::Major()))
        } else {
            curve_set.mt.as_ref().map(|mt| {
                self.minor_param(mt.1, &mt.0, &KYOPHOSIS_CURVE_MT)
                    .curve_type()
            })
        };
        let tll = if major_curve == MajorCurve::TLL {
            Some(RegionalCurveType::Structural(StructuralReason::Major()))
        } else {
            curve_set.tll.as_ref().map(|tll| {
                self.minor_param(tll.1, &tll.0, &KYOPHOSIS_CURVE_TLL)
                    .curve_type()
            })
        };
        Chart { pt, mt, tll }
    }
}

impl TryFrom<&LabelMeData> for C7TLS {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let mut corners = CORNER_LABELS
            .iter()
            .map(|label| extract_points(data, label))
            .collect::<Result<Vec<_>, _>>()?;
        if corners[0].shape()[0] != corners[1].shape()[0] {
            return Err(ScolError::InvalidPointCombo(
                "TL".into(),
                "TR".into(),
                corners[0].shape()[0],
                corners[1].shape()[0],
            ));
        }
        if corners[2].shape()[0] != corners[3].shape()[0] {
            return Err(ScolError::InvalidPointCombo(
                "BL".into(),
                "BR".into(),
                corners[2].shape()[0],
                corners[3].shape()[0],
            ));
        }
        if corners[0].shape()[0] - 1 != corners[2].shape()[0] {
            return Err(ScolError::InvalidPointCombo(
                "TL-1".into(),
                "BL".into(),
                corners[0].shape()[0] - 1,
                corners[2].shape()[0],
            ));
        }
        let last = corners[0]
            .index_axis(Axis(0), corners[0].len_of(Axis(0)) - 1)
            .insert_axis(Axis(0));
        corners[2] = concatenate(Axis(0), &[corners[2].view(), last]).unwrap();
        let last = corners[1]
            .index_axis(Axis(0), corners[1].len_of(Axis(0)) - 1)
            .insert_axis(Axis(0));
        corners[3] = concatenate(Axis(0), &[corners[3].view(), last]).unwrap();
        let verts = stack![Axis(1), corners[0], corners[1], corners[2], corners[3]];
        Ok(C7TLS(verts))
    }
}

impl TryFrom<&LabelMeData> for VertebraeC7TL {
    type Error = ScolError;

    /// From [C7[TL, TR, BL, BR] - Sacral[TL, TR]] points
    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let vertebrae: C7TLS = data.try_into()?;
        let verts_c7_t_l = vertebrae
            .0
            .slice_axis(
                Axis(0),
                ndarray::Slice::from(0..vertebrae.0.len_of(Axis(0)) - 1),
            )
            .to_owned();
        Ok(VertebraeC7TL(verts_c7_t_l))
    }
}

/// Curve types in Lenke classification
#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CurveType {
    /// Main Thoracic
    Type1,
    /// Double Thoracic
    Type2,
    /// Double Major
    Type3,
    /// Triple Major
    Type4,
    /// Thoracolumbar/Lumbar
    Type5,
    /// Thoracolumbar/Lumbar - Main Thoracic
    Type6,
}

/// Modifier based on T5-T12 sagittal angle
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum SagittalModifier {
    /// angle < 10
    Hypokyphosis,
    /// 10 <= angle < 40
    Normokyphosis,
    /// 40 <= angle
    Hyperkyphosis,
}

impl From<f64> for SagittalModifier {
    fn from(angle: f64) -> Self {
        if angle.abs() < 10.0 {
            SagittalModifier::Hypokyphosis
        } else if angle.abs() < 40.0 {
            SagittalModifier::Normokyphosis
        } else {
            SagittalModifier::Hyperkyphosis
        }
    }
}

impl Display for SagittalModifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SagittalModifier::Hypokyphosis => write!(f, "-"),
            SagittalModifier::Normokyphosis => write!(f, "N"),
            SagittalModifier::Hyperkyphosis => write!(f, "+"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LumbarModifier {
    /// CSVL between pedicles
    A,
    /// CSVL touches apical predicle
    B,
    // A or B
    AorB,
    /// The apical vertebral bodies are completely lateral to the CSVL
    C,
}

#[derive(Debug, PartialEq)]
pub enum RegionalCurveType {
    Structural(StructuralReason),
    NonStructural(MinorReason),
    /// Uncertain due to the lack of some images
    Uncertain(MinorReason),
}

impl Display for RegionalCurveType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegionalCurveType::Structural(r) => {
                write!(f, "Structural({})", r)
            }
            RegionalCurveType::NonStructural(r) => {
                write!(f, "NonStructural({})", r.to_non_structural_string())
            }
            RegionalCurveType::Uncertain(r) => {
                // TODO: implement
                write!(f, "Uncertain({r:?})")
            }
        }
    }
}

/// Chart for Lenke classification
#[derive(Debug)]
pub struct Chart {
    pub pt: Option<RegionalCurveType>,
    pub mt: Option<RegionalCurveType>,
    pub tll: Option<RegionalCurveType>,
}

impl Display for Chart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PT: {}\nMT: {}\nTLL: {}",
            self.pt.as_ref().map_or("N/A".into(), |e| format!("{}", e)),
            self.mt.as_ref().map_or("N/A".into(), |e| format!("{}", e)),
            self.tll.as_ref().map_or("N/A".into(), |e| format!("{}", e)),
        )
    }
}

impl Chart {
    pub fn classify(&self) -> Result<CurveType, Vec<CurveType>> {
        let mut types = [
            None,
            Some(CurveType::Type1),
            Some(CurveType::Type2),
            Some(CurveType::Type3),
            Some(CurveType::Type4),
            Some(CurveType::Type5),
            Some(CurveType::Type6),
        ];
        if let Some(mt) = self.mt.as_ref() {
            match mt {
                RegionalCurveType::Structural(reason) => match reason {
                    StructuralReason::Major() => {
                        types[5] = None; // Type5
                        types[6] = None; // Type6
                    }
                    StructuralReason::Minor(_) => return Ok(CurveType::Type6),
                },
                RegionalCurveType::NonStructural(_) => return Ok(CurveType::Type5),
                RegionalCurveType::Uncertain(_) => {}
            }
        }
        if let Some(tll) = self.tll.as_ref() {
            match tll {
                RegionalCurveType::Structural(_) => {
                    types[1] = None; // Type1
                    types[2] = None; // Type2
                }
                RegionalCurveType::NonStructural(_) => {
                    types[3] = None; // Type3
                    types[4] = None; // Type4
                }
                RegionalCurveType::Uncertain(_) => {}
            }
        }
        if let Some(pt) = self.pt.as_ref() {
            match pt {
                RegionalCurveType::Structural(_) => {
                    types[1] = None; // Type1
                    types[3] = None; // Type3
                }
                RegionalCurveType::NonStructural(_) => {
                    types[2] = None; // Type2
                    types[4] = None; // Type4
                }
                RegionalCurveType::Uncertain(_) => {}
            }
        }
        let types: Vec<_> = types.into_iter().flatten().collect();
        if types.len() == 1 {
            Ok(types[0])
        } else {
            Err(types)
        }
    }
}

/// Angles descriving bending criteria
#[derive(Debug, PartialEq)]
pub struct BendReasonAngles {
    /// angle in normal coronal image
    coronal: f64,
    /// angle in bending image
    bend: f64,
}

impl BendReasonAngles {
    pub fn new(coronal: f64, bend: f64) -> BendReasonAngles {
        BendReasonAngles { coronal, bend }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IsStructural {
    // True
    T,
    // False
    F,
}

impl From<IsStructural> for bool {
    fn from(s: IsStructural) -> Self {
        match s {
            IsStructural::T => true,
            IsStructural::F => false,
        }
    }
}

impl From<bool> for IsStructural {
    fn from(is_structural: bool) -> Self {
        if is_structural {
            IsStructural::T
        } else {
            IsStructural::F
        }
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct BendReason {
    pub left: Option<(IsStructural, BendReasonAngles)>,
    pub right: Option<(IsStructural, BendReasonAngles)>,
}

#[derive(Debug, PartialEq)]
pub struct MinorReason {
    pub coronal: (IsStructural, f64),
    pub bend: BendReason,
    pub sagittal: Option<(IsStructural, (Curve, f64))>,
}

trait StructuralValue {
    fn is_structural(&self) -> bool;
    fn is_non_structural(&self) -> bool;
}

impl<T> StructuralValue for (IsStructural, T) {
    fn is_structural(&self) -> bool {
        bool::from(self.0)
    }
    fn is_non_structural(&self) -> bool {
        !self.is_structural()
    }
}

impl MinorReason {
    pub fn with_coronal(coronal: (IsStructural, f64)) -> MinorReason {
        MinorReason {
            coronal,
            bend: BendReason::default(),
            sagittal: None,
        }
    }
    fn to_structural_string(&self) -> String {
        let coronal_reason = if self.coronal.is_structural() {
            // using two `unwrap()`s below is safe
            // because a non-structural curve with `self.coronal.is_structural()`always has bend params
            let left_reason = format!(
                "|LeftBendAngle|={:.1} >= {BEND_ANGLE_THRESH}",
                self.bend.left.as_ref().unwrap().1.bend.abs()
            );
            let right_reason = format!(
                "|RightBendAngle|={:.1} >= {BEND_ANGLE_THRESH}",
                self.bend.right.as_ref().unwrap().1.bend.abs()
            );
            let bend_reason = format!("Coronal({left_reason} & {right_reason})");
            bend_reason.to_string()
        } else {
            // should be unreachable but return dummy value anyway
            let err =
            format!(
                "MinorReason::to_structural_string should not be called for {:?}, where coronal is structural",
                self
            );
            error!("{}", err);
            err
        };
        if let Some(sagittal) = self.sagittal.as_ref() {
            let sagittal_reason = format!(
                "Sagittal(|{}Kyphosis|={:.1} < {LATERAL_ANGLE_THRESH})",
                sagittal.1 .0,
                sagittal.1 .1.abs()
            );
            format!("{coronal_reason} or {sagittal_reason}")
        } else {
            coronal_reason
        }
    }
    fn to_non_structural_string(&self) -> String {
        let coronal_reason = if self.coronal.is_structural() {
            // using two `unwrap()`s below is safe
            // because a non-structural curve with `self.coronal.is_structural()`always has bend params
            let left_reason = format!(
                "|LeftBendAngle|={:.1} < {BEND_ANGLE_THRESH}",
                self.bend.left.as_ref().unwrap().1.bend.abs()
            );
            let right_reason = format!(
                "|RightBendAngle|={:.1} < {BEND_ANGLE_THRESH}",
                self.bend.right.as_ref().unwrap().1.bend.abs()
            );
            let bend_reason = format!("Coronal({left_reason} or {right_reason})");
            bend_reason.to_string()
        } else {
            format!(
                "|CoronalAngle|={:.1} < {}",
                self.coronal.1.abs(),
                FRONTAL_ANGLE_THRESH
            )
        };
        // safe to unwrap because sagittal is required for a curve to be non-structural
        let sagittal = self.sagittal.as_ref().unwrap();
        if sagittal.is_non_structural() {
            let sagittal_reason = format!(
                "Sagittal(|{}Kyphosis|={:.1} < {LATERAL_ANGLE_THRESH})",
                sagittal.1 .0,
                sagittal.1 .1.abs()
            );
            format!("{coronal_reason} & {sagittal_reason}")
        } else {
            coronal_reason
        }
    }
}

/// Parameters that explain why it's structural or non-structural
#[derive(Debug, PartialEq)]
pub struct MinorStructuralParam {
    coronal: f64,
    right_bend: Option<f64>,
    left_bend: Option<f64>,
    sagittal: Option<(Curve, f64)>,
}

trait CertainlyStructural {
    fn is_some_structural(&self) -> bool;
    fn is_some_non_structural(&self) -> bool;
}

impl<T> CertainlyStructural for Option<(IsStructural, T)> {
    /// Some(is_structural)
    fn is_some_structural(&self) -> bool {
        self.as_ref().is_some_and(|v| bool::from(v.0))
    }

    /// Some(is_non_structural)
    fn is_some_non_structural(&self) -> bool {
        self.as_ref().is_some_and(|v| !bool::from(v.0))
    }
}

impl MinorStructuralParam {
    pub fn curve_type(&self) -> RegionalCurveType {
        let mut reason = if self.coronal.abs() < FRONTAL_ANGLE_THRESH {
            MinorReason::with_coronal((IsStructural::F, self.coronal))
        } else {
            MinorReason::with_coronal((IsStructural::T, self.coronal))
        };

        for (bend, is_right) in [(self.right_bend, true), (self.left_bend, false)] {
            if let Some(angle) = bend {
                let angle = angle.abs();

                let is_structural = (angle.abs() >= BEND_ANGLE_THRESH).into();
                if is_right {
                    reason.bend.right =
                        Some((is_structural, BendReasonAngles::new(self.coronal, angle)));
                } else {
                    reason.bend.left =
                        Some((is_structural, BendReasonAngles::new(self.coronal, angle)));
                }
            }
        }
        if let Some(sagittal) = self.sagittal.as_ref() {
            let is_structural = (sagittal.1.abs() >= LATERAL_ANGLE_THRESH).into();
            reason.sagittal = Some((is_structural, sagittal.clone()));
        }

        if (reason.bend.left.is_some_structural() && reason.bend.right.is_some_structural())
            || reason.sagittal.is_some_structural()
        {
            return RegionalCurveType::Structural(StructuralReason::Minor(reason));
        }
        if reason.coronal.is_non_structural() && reason.sagittal.is_some_non_structural() {
            return RegionalCurveType::NonStructural(reason);
        }
        if (reason.bend.right.is_some_non_structural() || reason.bend.left.is_some_non_structural())
            && reason.sagittal.is_some_non_structural()
        {
            return RegionalCurveType::NonStructural(reason);
        }
        RegionalCurveType::Uncertain(reason)
    }
}

#[derive(Debug, PartialEq)]
pub enum StructuralReason {
    Major(),
    Minor(MinorReason),
}

impl Display for StructuralReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StructuralReason::Major() => write!(f, "{:?}", self),
            StructuralReason::Minor(r) => write!(f, "Minor({})", r.to_structural_string()),
        }
    }
}

#[derive(
    strum::EnumString,
    strum::Display,
    strum::VariantArray,
    ValueEnum,
    Serialize,
    Deserialize,
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
)]
#[serde(rename_all = "PascalCase")]
#[clap(rename_all = "PascalCase")]
pub enum CoronalMeasure {
    CobbPT,
    CobbMT,
    CobbTLL,
    CurveApex,
    CSVL,
    T1TiltAngle,
    CoronalBalance,
    ClavicleAngle,
    ShoulderHeight,
    PelvicObliquity,
    SacralObliquity,
    LegLengthDiscrepancy,
}

impl CoronalMeasure {
    pub fn all_draws() -> Vec<Self> {
        CoronalMeasure::VARIANTS.to_vec()
    }
    pub fn all_measures() -> Vec<Self> {
        use CoronalMeasure::*;
        CoronalMeasure::VARIANTS
            .iter()
            .filter(|v| !matches!(v, CurveApex | CSVL))
            .copied()
            .collect()
    }
}

#[derive(
    strum::EnumString,
    strum::Display,
    strum::VariantArray,
    ValueEnum,
    Serialize,
    Deserialize,
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
)]
#[clap(rename_all = "PascalCase")]
#[strum(serialize_all = "PascalCase")]
pub enum SagittalMeasure {
    ThoracicKyphosis,
    ProximalThoracicKyphosis,
    MidLowerThoracicKyphosis,
    ThoracolumbarSagittalAlignment,
    LumbarLordosis,
    SagittalBalance,
    PelvicIncidence,
    PelvicTilt,
    SacralSlope,
    L5IncidenceAngle,
    PelvicRadiusAngle,
    LumbosacralAngle,
}

impl SagittalMeasure {
    pub fn all_draws() -> Vec<Self> {
        SagittalMeasure::VARIANTS.to_vec()
    }
    pub fn all_measures() -> Vec<Self> {
        Self::all_draws()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{Corners, LineCharacteristic};
    use anyhow::Result;
    use ndarray::{arr3, array, Array3};
    #[test]
    fn test_corners() -> Result<()> {
        let arr: Array3<f64> = arr3(&[
            [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]],
            [[1.0, 3.0], [2.0, 3.0], [1.0, 4.0], [2.0, 4.0]],
        ]);
        let corners = Corners(arr);
        let bet = corners.between();

        let expected: Array3<f64> =
            ndarray::arr3(&[[[1.0, 2.0], [2.0, 2.0], [1.0, 3.0], [2.0, 3.0]]]);
        assert_eq!(bet, expected);
        Ok(())
    }

    #[test]
    fn test_is_monotonic_increasing() {
        let arr = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(arr.is_monotonic());
    }

    #[test]
    fn test_is_monotonic_decreasing() {
        let arr = array![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!(arr.is_monotonic());
    }

    #[test]
    fn test_is_monotonic_constant() {
        let arr = array![2.0, 2.0, 2.0, 2.0, 2.0];
        assert!(arr.is_monotonic());
    }

    #[test]
    fn test_is_not_monotonic() {
        let arr = array![1.0, 2.0, 3.0, 2.0, 1.0];
        assert!(!arr.is_monotonic());
    }
}
