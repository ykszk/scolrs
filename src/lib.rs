use labelme_rs::LabelMeData;
use ndarray::{
    s, stack, Array1, Array2, Array3, ArrayBase, ArrayView1, ArrayView2, ArrayView3, Axis, Data,
};
use ndarray_linalg::Solve;
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::iter::zip;
use std::ops::AddAssign;
use std::result::Result;
use thiserror::Error;

fn default_radius() -> usize {
    2
}
fn default_line_width() -> usize {
    1
}
fn default_text_stroke() -> String {
    "black".into()
}
fn default_text_stroke_width() -> usize {
    1
}
fn default_text_fill() -> String {
    "white".into()
}

mod defs;
pub use defs::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RenderParam {
    /// Point radius
    #[serde(default = "default_radius")]
    pub radius: usize,
    /// Line width
    #[serde(default = "default_line_width")]
    pub line_width: usize,

    /// `stroke` for texts
    #[serde(default = "default_text_stroke")]
    pub text_stroke: String,
    /// `stroke-width` for texts
    #[serde(default = "default_text_stroke_width")]
    pub text_stroke_width: usize,
    /// `fill` for texts
    #[serde(default = "default_text_fill")]
    pub text_fill: String,
}

impl Default for RenderParam {
    fn default() -> Self {
        Self {
            radius: default_radius(),
            line_width: default_line_width(),
            text_stroke: default_text_stroke(),
            text_stroke_width: default_text_stroke_width(),
            text_fill: default_text_fill(),
        }
    }
}

#[derive(Error, Debug)]
pub enum ScolError {
    #[error("Invalid points for label: {0}")]
    InvalidPointCount(String),
    #[error("Invalid combination points for {0} and {1}: {2} vs. {3}")]
    InvalidPointCombo(String, String, usize, usize),
}

fn extract_points(data: &LabelMeData, label: &str) -> Result<Array2<f32>, ScolError> {
    let tuples: Option<Vec<_>> = data
        .shapes
        .iter()
        .filter_map(|shape| {
            if shape.label == label {
                Some(shape.points.first())
            } else {
                None
            }
        })
        .collect();
    let tuples = tuples.ok_or_else(|| ScolError::InvalidPointCount(label.to_string()))?;
    let mut vec = Vec::with_capacity(tuples.len() * 2);
    for t in tuples.iter() {
        vec.push(t.0);
        vec.push(t.1);
    }
    let arr = Array2::from_shape_vec((tuples.len(), 2), vec).unwrap();
    Ok(arr)
}

pub fn polyfit<S>(
    xs: ndarray::ArrayBase<S, ndarray::Ix1>,
    ys: ndarray::ArrayBase<S, ndarray::Ix1>,
    deg: usize,
) -> ndarray_linalg::error::Result<ndarray::Array1<f32>>
where
    S: ndarray::Data<Elem = f32>,
{
    let mut vander = Array2::zeros([xs.len(), deg + 1]);

    // f64 is required for higher degrees
    let xs = xs.mapv(|x| x as f64);
    let ys = ys.mapv(|x| x as f64);

    for d in 0..=deg {
        vander
            .slice_mut(s![.., d])
            .assign(&xs.mapv(|x| x.powi(d as i32)));
    }

    vander
        .t()
        .dot(&vander)
        .solve(&vander.t().dot(&ys))
        .map(|arr| arr.mapv(|x| x as f32))
}

pub fn polynomial<S, T>(
    xs: ndarray::ArrayBase<S, ndarray::Ix1>,
    coef: ndarray::ArrayBase<T, ndarray::Ix1>,
) -> Array1<f32>
where
    S: ndarray::Data<Elem = f32>,
    T: ndarray::Data<Elem = f32>,
{
    let xs = xs.mapv(|x| x as f64);
    let coef = coef.mapv(|x| x as f64);
    let mut ys: Array1<f64> = ndarray::Array::zeros(xs.len());
    for (i, c) in coef.iter().enumerate() {
        ys.add_assign(&xs.mapv(|x| c * x.powi(i as i32)));
    }
    ys.mapv(|e| e as f32)
}

pub struct Scoliosis {
    v_c7tl: VertebraeC7TL,
    c_c7tl: Centroids,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Curve {
    pub sup: usize,
    pub inf: usize,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct CurveSet {
    pub pt: Option<(Curve, f32)>,
    pub mt: Option<(Curve, f32)>,
    pub tll: Option<(Curve, f32)>,
}

#[derive(Debug, Default)]
pub struct ApexSet {
    pub pt: Option<VertebraDiscIndex>,
    pub mt: Option<VertebraDiscIndex>,
    pub tll: Option<VertebraDiscIndex>,
}

pub struct LineFactory(lyon_geom::Line<f32>);

impl LineFactory {
    pub fn create<T: Into<Self>>(value: T) -> lyon_geom::Line<f32> {
        let f: Self = value.into();
        f.0
    }
}

impl From<ArrayView2<'_, f32>> for LineFactory {
    fn from(start_end: ArrayView2<'_, f32>) -> Self {
        let start = start_end.index_axis(Axis(1), 0);
        let end = start_end.index_axis(Axis(1), 1);
        (start, end).into()
    }
}

impl From<(ArrayView1<'_, f32>, ArrayView1<'_, f32>)> for LineFactory {
    fn from(start_end: (ArrayView1<'_, f32>, ArrayView1<'_, f32>)) -> Self {
        let point = lyon_geom::Point::new(start_end.0[0], start_end.0[1]);
        let p2 = lyon_geom::Point::new(start_end.1[0], start_end.1[1]);
        let vector = p2 - point;
        Self(lyon_geom::Line { point, vector })
    }
}

pub struct LineSegmentFactory(lyon_geom::LineSegment<f32>);
impl LineSegmentFactory {
    pub fn create<T: Into<Self>>(value: T) -> lyon_geom::LineSegment<f32> {
        let f: Self = value.into();
        f.0
    }
}

impl From<ArrayView2<'_, f32>> for LineSegmentFactory {
    fn from(start_end: ArrayView2<'_, f32>) -> Self {
        let from = start_end.index_axis(Axis(1), 0);
        let to = start_end.index_axis(Axis(1), 1);
        (from, to).into()
    }
}

impl From<(ArrayView1<'_, f32>, ArrayView1<'_, f32>)> for LineSegmentFactory {
    fn from(start_end: (ArrayView1<'_, f32>, ArrayView1<'_, f32>)) -> Self {
        let from = lyon_geom::Point::new(start_end.0[0], start_end.0[1]);
        let to = lyon_geom::Point::new(start_end.1[0], start_end.1[1]);
        Self(lyon_geom::LineSegment { from, to })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MajorCurve {
    PT,
    MT,
    TLL,
}

impl Scoliosis {
    pub fn tl_corners(&self) -> Corners<ndarray::ViewRepr<&f32>> {
        let tl = self.v_c7tl.0.slice(s![1.., .., ..]);
        Corners(tl)
    }

    pub fn tl_vert_disc_corners(&self) -> Corners<ndarray::OwnedRepr<f32>> {
        let vert_corners = self.tl_corners();
        let disc_corners = vert_corners.between();
        let mut vert_disc_corners: Array3<f32> = ndarray::Array::zeros((
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
    pub fn tl_sup_lines(&self) -> ArrayView3<'_, f32> {
        self.v_c7tl.0.slice(s![1.., ..2, ..])
    }
    pub fn tl_inf_lines(&self) -> ArrayView3<'_, f32> {
        self.v_c7tl.0.slice(s![1.., 2.., ..])
    }
    pub fn tl_centroids(&self) -> ArrayView2<'_, f32> {
        self.c_c7tl.0.slice(s![1.., ..])
    }

    pub fn tl_sup_plate(&self, index: usize) -> ArrayView2<'_, f32> {
        self.v_c7tl.0.slice(s![index + 1, ..2, ..])
    }
    pub fn tl_inf_plate(&self, index: usize) -> ArrayView2<'_, f32> {
        self.v_c7tl.0.slice(s![index + 1, 2.., ..])
    }

    pub fn spinal_poly(&self) -> ndarray_linalg::error::Result<ndarray::Array1<f32>> {
        let centroids = self.tl_centroids();
        let coefs = polyfit(centroids.slice(s![.., 1]), centroids.slice(s![.., 0]), 6);
        coefs
    }

    fn find_largest_curve(&self) -> Option<(Curve, f32)> {
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

    fn find_largest_up(&self, inf: usize) -> Option<(Curve, f32)> {
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

    fn find_largest_down(&self, sup: usize) -> Option<(Curve, f32)> {
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
        let xs = centroids.slice(s![sup..inf, 0]);

        // check if the second derivatives have the same sign
        let dx2s = -&xs.slice(s![..xs.len() - 2]) + 2.0 * &xs.slice(s![1..xs.len() - 1])
            - xs.slice(s![2..]);
        // implement num.sign to get signs as integers
        let signs = dx2s.mapv(|e| {
            if e > 0.0 {
                1
            } else if e < 0.0 {
                -1
            } else {
                0
            }
        });
        let sign = signs[0_usize];
        let mut sing_changes = false;
        for i in 1..signs.len() {
            let diff = sign - signs[i];
            if diff != 0 {
                sing_changes = true;
                break;
            }
        }
        !sing_changes
    }

    fn _find_largest_curve(&self, curves: Vec<Curve>) -> Option<(Curve, f32)> {
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
        Some((angles[i_max].0.clone(), _max_value))
    }

    fn find_apex(&self, curve: &Curve) -> ndarray_linalg::error::Result<VertebraDiscIndex> {
        let coefs = self.spinal_poly()?;
        let vert_disc_corners = self.tl_vert_disc_corners();
        let vd_centroids: Centroids = vert_disc_corners.into();
        let sup = VertebraDiscIndex::from(VertebralIndex::from(curve.sup as u8)) as usize;
        let inf = VertebraDiscIndex::from(VertebralIndex::from(curve.inf as u8)) as usize;
        let xs = polynomial(vd_centroids.0.slice(s![sup..=inf, 1]), coefs);
        let ts = (2.0 * &xs - xs[0] - xs[xs.len() - 1]).mapv(|e| e.abs());
        let i_max = ts.argmax().unwrap();
        let i_apex = VertebraDiscIndex::from((i_max + sup) as u8);
        Ok(i_apex)
    }

    pub fn identify_curves(&self) -> (CurveSet, ApexSet, Option<MajorCurve>) {
        let mut curves = CurveSet::default();
        let mut apexes = ApexSet::default();
        let mut major_curve = None;
        if let Some(largest_curve) = self.find_largest_curve() {
            let major_apex = self.find_apex(&largest_curve.0).unwrap();
            major_curve = if major_apex <= VertebraDiscIndex::T5 {
                // largest curve is PT
                if let Some(mt) = self.find_largest_down(largest_curve.0.inf) {
                    curves.tll = self.find_largest_down(mt.0.inf);
                    apexes.tll = curves.tll.as_ref().map(|pt| self.find_apex(&pt.0).unwrap());
                    apexes.mt = Some(self.find_apex(&mt.0).unwrap());
                    curves.mt = Some(mt);
                }
                curves.pt = Some(largest_curve);
                apexes.pt = Some(major_apex);
                Some(MajorCurve::PT)
            } else if major_apex <= VertebraDiscIndex::DiscT11T12 {
                // largest curve is MT
                curves.pt = self.find_largest_up(largest_curve.0.sup);
                apexes.pt = curves.pt.as_ref().map(|pt| self.find_apex(&pt.0).unwrap());
                curves.tll = self.find_largest_down(largest_curve.0.inf);
                apexes.tll = curves.tll.as_ref().map(|pt| self.find_apex(&pt.0).unwrap());
                curves.mt = Some(largest_curve);
                apexes.mt = Some(major_apex);
                Some(MajorCurve::MT)
            } else {
                // largest curve is TLL
                if let Some(mt) = self.find_largest_up(largest_curve.0.sup) {
                    curves.pt = self.find_largest_up(mt.0.sup);
                    apexes.pt = curves.pt.as_ref().map(|pt| self.find_apex(&pt.0).unwrap());
                    apexes.mt = Some(self.find_apex(&mt.0).unwrap());
                    curves.mt = Some(mt);
                }
                curves.tll = Some(largest_curve);
                apexes.tll = Some(major_apex);
                Some(MajorCurve::TLL)
            };
        }
        (curves, apexes, major_curve)
    }

    pub fn angle(&self, curve: &Curve) -> Option<f32> {
        let sup_line = self.tl_sup_plate(curve.sup);
        let inf_line = self.tl_inf_plate(curve.inf);
        let v_sup = &sup_line.index_axis(Axis(0), 1) - &sup_line.index_axis(Axis(0), 0);
        let v_inf = &inf_line.index_axis(Axis(0), 1) - &inf_line.index_axis(Axis(0), 0);
        let len_sup = v_sup.mapv(|a| a * a).sum().sqrt();
        let len_inf = v_inf.mapv(|a| a * a).sum().sqrt();
        if len_sup == 0.0 || len_inf == 0.0 {
            return None;
        }
        let v_sup = &v_sup / len_sup;
        let v_inf = &v_inf / len_inf;
        let cos = v_sup.dot(&v_inf);
        let cos = cos.max(-1.0).min(1.0);
        let rad = cos.acos();
        if v_sup[1] < v_inf[1] {
            Some(-rad)
        } else {
            Some(rad)
        }
    }
}

impl TryFrom<&LabelMeData> for Scoliosis {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let v_c7tl = VertebraeC7TL::try_from(data)?;
        let c_c7tl = Corners(v_c7tl.0.view()).into();
        Ok(Self { v_c7tl, c_c7tl })
    }
}

/// C7, thoracic and lumber vertebrae
pub struct VertebraeC7TL(pub Array3<f32>);

/// Thoracic and lumber vertebrae
pub struct VertebraeTL<'a>(ArrayView3<'a, f32>);

impl<'a> From<&'a VertebraeC7TL> for VertebraeTL<'a> {
    fn from(c7tl: &'a VertebraeC7TL) -> Self {
        let tl = c7tl.0.slice(s![1.., .., ..]);
        Self(tl)
    }
}

/// Corner points of structures.
/// Points are in [tl, tr, bl, br] order
pub struct Corners<S: Data<Elem = f32>>(pub ArrayBase<S, ndarray::Ix3>);

impl<S: Data<Elem = f32>> Corners<S> {
    pub fn between(&self) -> Array3<f32> {
        let bottom = self.0.slice(s![1.., ..2, ..]);
        let top = self.0.slice(s![..(self.0.shape()[0] - 1), 2.., ..]);
        let between = ndarray::concatenate![Axis(1), top, bottom];
        between
    }
}

pub struct Centroids(pub Array2<f32>);

impl<S: Data<Elem = f32>> From<Corners<S>> for Centroids {
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
                unreachable!("No centroid found for a corner");
            } else {
                let num = dap.dot(&dp);
                let c = num / denom * db + b1;
                r.assign(&c);
            }
        }
        Centroids(right)
    }
}

impl TryFrom<&LabelMeData> for VertebraeC7TL {
    type Error = ScolError;

    /// From [C7[TL, TR, BL, BR] - Sacral[TL, TR]] points
    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let corners = CORNER_LABELS
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
        let verts_c7_t_l = stack![
            Axis(1),
            corners[0].slice(s![0..(corners[0].shape()[0] - 1), ..]),
            corners[1].slice(s![0..(corners[1].shape()[0] - 1), ..]),
            corners[2],
            corners[3]
        ];
        Ok(VertebraeC7TL(verts_c7_t_l))
    }
}

pub type LineColors = HashMap<String, String>;

#[derive(Debug, serde::Deserialize)]
struct LineColor {
    label: String,
    color: String,
}

pub fn load_line_colors<S: Read>(reader: S) -> Result<LineColors, csv::Error> {
    let mut rdr = csv::Reader::from_reader(reader);
    let mut colors = LineColors::new();
    for result in rdr.deserialize() {
        let lc: LineColor = result?;
        colors.insert(lc.label, lc.color);
    }
    Ok(colors)
}

#[cfg(test)]
mod tests {
    use crate::MajorCurve;

    use super::{Corners, Scoliosis, VertebraDiscIndex, VertebralIndex};
    use anyhow::{Context, Result};
    use labelme_rs::LabelMeData;
    use ndarray::{arr3, Array3};
    use pretty_assertions::assert_eq;
    use std::path::PathBuf;

    #[test]
    fn test_case1() -> Result<()> {
        let mut tests = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        tests.push("tests");
        let json_filename = tests.join("case1/frontal.json");
        let s = std::fs::read_to_string(&json_filename)
            .with_context(|| format!("Opening {:?}", &json_filename))?;
        let data: LabelMeData = s.as_str().try_into()?;
        let scol = Scoliosis::try_from(&data)?;

        let (curve_set, apex_set, major_curve) = scol.identify_curves();
        assert_eq!(major_curve.unwrap(), MajorCurve::MT);
        let (mt_curve, _angle) = curve_set.mt.unwrap();
        assert_eq!(mt_curve.sup, VertebralIndex::T5 as usize);
        assert_eq!(mt_curve.inf, VertebralIndex::T12 as usize);
        let (pt_curve, _angle) = curve_set.pt.unwrap();
        assert_eq!(pt_curve.inf, mt_curve.sup);
        assert_eq!(pt_curve.sup, VertebralIndex::T1 as usize);
        let (tll_curve, _angle) = curve_set.tll.unwrap();
        assert_eq!(tll_curve.sup, mt_curve.inf);
        assert_eq!(tll_curve.inf, VertebralIndex::L5 as usize);

        assert!(apex_set.pt.is_some());
        assert!(apex_set.tll.is_some());
        assert_eq!(apex_set.mt.unwrap(), VertebraDiscIndex::DiscT7T8);

        Ok(())
    }

    #[test]
    fn test_case2() -> Result<()> {
        let mut tests = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        tests.push("tests");
        let json_filename = tests.join("case2/frontal.json");
        let s = std::fs::read_to_string(&json_filename)
            .with_context(|| format!("Opening {:?}", &json_filename))?;
        let data: LabelMeData = s.as_str().try_into()?;
        let scol = Scoliosis::try_from(&data)?;

        let (curve_set, apex_set, major_curve) = scol.identify_curves();
        // no strict testing of curve positions because case 2 is hard to determine curve with some certainty.
        assert!(curve_set.pt.is_some());
        assert!(curve_set.mt.is_some());
        assert!(curve_set.tll.is_some());

        // largest curve is tll though.
        assert_eq!(major_curve.unwrap(), MajorCurve::TLL);
        assert!(apex_set.pt.is_some());
        assert!(apex_set.mt.is_some());
        assert!(apex_set.tll.is_some());

        Ok(())
    }

    #[test]
    fn test_case3() -> Result<()> {
        let mut tests = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        tests.push("tests");
        let json_filename = tests.join("case3/frontal.json");
        let s = std::fs::read_to_string(&json_filename)
            .with_context(|| format!("Opening {:?}", &json_filename))?;
        let data: LabelMeData = s.as_str().try_into()?;
        let scol = Scoliosis::try_from(&data)?;

        let (curve_set, apex_set, major_curve) = scol.identify_curves();
        assert_eq!(major_curve.unwrap(), MajorCurve::PT);
        let (mt_curve, _angle) = curve_set.mt.unwrap();
        assert_eq!(mt_curve.sup, VertebralIndex::T7 as usize);
        assert_eq!(mt_curve.inf, VertebralIndex::T12 as usize);
        let (pt_curve, _angle) = curve_set.pt.unwrap();
        assert_eq!(pt_curve.inf, mt_curve.sup);
        assert_eq!(pt_curve.sup, VertebralIndex::T2 as usize);
        let (tll_curve, _angle) = curve_set.tll.unwrap();
        assert_eq!(tll_curve.sup, mt_curve.inf);
        assert_eq!(tll_curve.inf, VertebralIndex::L4 as usize);

        assert!(apex_set.pt.is_some());
        assert!(apex_set.mt.is_some());
        assert!(apex_set.tll.is_some());

        Ok(())
    }

    #[test]
    fn test_lenke() -> Result<()> {
        let mut tests = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        tests.push("tests");
        let json_filename = tests.join("case1/frontal.json");
        let s = std::fs::read_to_string(&json_filename)
            .with_context(|| format!("Opening {:?}", &json_filename))?;
        let data: LabelMeData = s.as_str().try_into()?;
        let scol = Scoliosis::try_from(&data)?;

        let (curve_set, apex_set, major_curve) = scol.identify_curves();
        println!("apex_set: {:?}", apex_set);
        let major_curve = major_curve.unwrap();
        // match major_curve {
        //     crate::MajorCurve::PT => {}
        //     crate::MajorCurve::MT => todo!(),
        //     crate::MajorCurve::TLL => todo!(),
        // }
        // let angles = [curve_set.pt, curve_set.mt, curve_set.tll]
        //     .map(|c| c.map_or(0.0, |c| scol.angle(&c).unwrap_or(0.0)));

        Ok(())
    }

    #[test]
    fn test_corners() -> Result<()> {
        let arr: Array3<f32> = arr3(&[
            [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]],
            [[1.0, 3.0], [2.0, 3.0], [1.0, 4.0], [2.0, 4.0]],
        ]);
        let corners = Corners(arr);
        let bet = corners.between();

        let expected: Array3<f32> =
            ndarray::arr3(&[[[1.0, 2.0], [2.0, 2.0], [1.0, 3.0], [2.0, 3.0]]]);
        assert_eq!(bet, expected);
        Ok(())
    }
}
