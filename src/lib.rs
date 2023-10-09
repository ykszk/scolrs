use anyhow::ensure;
use labelme_rs::LabelMeData;
use ndarray::{s, stack, Array2, Array3, Axis};
use std::iter::zip;
use std::result::Result;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ShapeError {
    #[error("Invalid points for label: {0}")]
    InvalidPointCount(String),
}

fn extract_points(data: &LabelMeData, label: &str) -> Result<Array2<f32>, ShapeError> {
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
    let tuples = tuples.ok_or_else(|| ShapeError::InvalidPointCount(label.to_string()))?;
    let mut vec = Vec::with_capacity(tuples.len() * 2);
    for t in tuples.iter() {
        vec.push(t.0);
        vec.push(t.1);
    }
    let arr = Array2::from_shape_vec((tuples.len(), 2), vec).unwrap();
    Ok(arr)
}

pub const CORNER_LABELS: [&str; 4] = ["TL", "TR", "BL", "BR"];

/// C7, thoracic and lumber vertebrae
pub struct VertebraeC7TL(pub Array3<f32>);

pub struct Corners(pub Array3<f32>);
impl From<VertebraeTL> for Corners {
    fn from(value: VertebraeTL) -> Self {
        Corners(value.corners)
    }
}

impl Corners {
    pub fn between(&self) -> Self {
        let bottom = self.0.slice(s![..(self.0.shape()[0] - 1), ..2, ..]);
        let top = self.0.slice(s![1.., 2.., ..]);
        let between = ndarray::concatenate![Axis(1), top, bottom];
        Corners(between)
    }
}

pub struct Centroids(pub Array2<f32>);

impl From<&Corners> for Centroids {
    fn from(corners: &Corners) -> Self {
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
    type Error = anyhow::Error;

    /// From [C7[TL, TR, BL, BR] - Sacral[TL, TR]] points
    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let corners = CORNER_LABELS
            .iter()
            .map(|label| extract_points(data, label))
            .collect::<Result<Vec<_>, _>>()?;
        ensure!(
            corners[0].shape()[0] == corners[1].shape()[0],
            "The number of TL and TR points does not match: {} and {}",
            corners[0].shape()[0],
            corners[1].shape()[0]
        );
        ensure!(
            corners[2].shape()[0] == corners[3].shape()[0],
            "The number of TL and TR points does not match: {} and {}",
            corners[2].shape()[0],
            corners[3].shape()[0]
        );
        ensure!(
            corners[0].shape()[0] - 1 == corners[2].shape()[0],
            "The number of TL and BL points does not satisfy |TL|-1==|BL|: {} and {}",
            corners[0].shape()[0],
            corners[2].shape()[0],
        );
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

/// Thoracic and lumber vertebrae
pub struct VertebraeTL {
    corners: Array3<f32>,
}
impl From<VertebraeC7TL> for VertebraeTL {
    fn from(c7tl: VertebraeC7TL) -> Self {
        let corners = c7tl.0.slice(s![1.., .., ..]).to_owned();
        Self { corners }
    }
}

impl TryFrom<&LabelMeData> for VertebraeTL {
    type Error = anyhow::Error;

    fn try_from(data: &LabelMeData) -> std::result::Result<Self, Self::Error> {
        let c7tl: VertebraeC7TL = data.try_into()?;
        Ok(c7tl.into())
    }
}
