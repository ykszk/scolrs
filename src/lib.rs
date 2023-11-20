use labelme_rs::LabelMeData;
use ndarray::{
    s, stack, Array2, Array3, ArrayBase, ArrayView1, ArrayView2, ArrayView3, Axis, Data,
};
use serde::{Deserialize, Serialize};
use std::iter::zip;
use std::result::Result;
use thiserror::Error;

fn default_radius() -> usize {
    2
}
fn default_line_width() -> usize {
    2
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

pub struct Scoliosis {
    v_c7tl: VertebraeC7TL,
    c_c7tl: Centroids,
}

#[derive(Debug, Clone)]
pub struct Curve {
    pub sup: usize,
    pub inf: usize,
}

#[derive(Debug, Default)]
pub struct CurveSet {
    pub pt: Option<Curve>,
    pub mt: Option<Curve>,
    pub tll: Option<Curve>,
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

impl Scoliosis {
    pub fn tl_corners(&self) -> Corners<ndarray::ViewRepr<&f32>> {
        let tl = self.v_c7tl.0.slice(s![1.., .., ..]);
        Corners(tl)
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

    fn find_largest_curve(&self) -> Option<Curve> {
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

    fn find_largest_up(&self, inf: usize) -> Option<Curve> {
        let mut curves = Vec::new();
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

    fn find_largest_down(&self, sup: usize) -> Option<Curve> {
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

    /// Curve is invalid when it contains S curve
    fn is_valid_curve(&self, sup: usize, inf: usize) -> bool {
        // inf - sup <= 3
        if inf <= sup + 3 {
            return true;
        }
        let centroids = self.tl_centroids();
        let xs = centroids.slice(s![sup..inf, 0]);

        // check second derivatives
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

    fn _find_largest_curve(&self, curves: Vec<Curve>) -> Option<Curve> {
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
        Some(angles[i_max].0.clone())
    }

    pub fn find_curve_set(&self) -> CurveSet {
        let mut curves = CurveSet::default();
        if let Some(largest_curve) = self.find_largest_curve() {
            curves.pt = self.find_largest_up(largest_curve.sup);
            curves.tll = self.find_largest_down(largest_curve.inf);
            curves.mt = Some(largest_curve);
        }
        curves
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

pub struct Corners<S: Data<Elem = f32>>(pub ArrayBase<S, ndarray::Ix3>);

impl<S: Data<Elem = f32>> Corners<S> {
    pub fn between(&self) -> Array3<f32> {
        let bottom = self.0.slice(s![..(self.0.shape()[0] - 1), ..2, ..]);
        let top = self.0.slice(s![1.., 2.., ..]);
        let between = ndarray::concatenate![Axis(1), top, bottom];
        between
    }
}

pub struct Centroids(pub Array2<f32>);

impl<S: Data<Elem = f32>> From<Corners<S>> for Centroids {
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

#[test]
fn test_check_json() -> anyhow::Result<()> {
    use anyhow::Context;
    use std::path::PathBuf;

    let mut tests = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    tests.push("tests");
    let json_filename = tests.join("case1/frontal.json");
    let s = std::fs::read_to_string(&json_filename)
        .with_context(|| format!("Opening {:?}", &json_filename))?;
    let data: LabelMeData = s.as_str().try_into()?;
    let scol = Scoliosis::try_from(&data)?;

    let curve_set = scol.find_curve_set();
    let largest_curve = curve_set.mt.unwrap();
    assert_eq!(largest_curve.sup, 4);
    assert_eq!(largest_curve.inf, 11);
    let pt_curve = curve_set.pt.unwrap();
    assert_eq!(pt_curve.inf, largest_curve.sup);
    assert_eq!(pt_curve.sup, 0);
    let tll_curve = curve_set.tll.unwrap();
    assert_eq!(tll_curve.sup, largest_curve.inf);
    assert_eq!(tll_curve.inf, 16);

    Ok(())
}
