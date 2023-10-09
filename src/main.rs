use anyhow::{ensure, Result};
use clap::Parser;
use core::panic;
use labelme_rs::LabelMeData;
use ndarray::{s, stack, Array2, Array3, Axis};
use std::iter::zip;
use std::path::PathBuf;

/// LabelMeData to Vertebrae
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input labelme json filename
    filename: PathBuf,
}

fn extract_points(data: &LabelMeData, label: &str) -> Result<Array2<f32>> {
    let tuples: Vec<_> = data
        .shapes
        .iter()
        .filter_map(|shape| {
            if shape.label == label {
                Some(shape.points[0])
            } else {
                None
            }
        })
        .collect();
    let mut vec = Vec::with_capacity(tuples.len() * 2);
    for t in tuples.iter() {
        vec.push(t.0);
        vec.push(t.1);
    }
    let arr = Array2::from_shape_vec((tuples.len(), 2), vec)?;
    Ok(arr)
}

/// C7, thoracic and lumber vertebrae
struct VertebraeC7TL(Array3<f32>);

struct Corners(Array3<f32>);
impl From<VertebraeTL> for Corners {
    fn from(value: VertebraeTL) -> Self {
        Corners(value.corners)
    }
}

impl Corners {
    fn between(&self) -> Self {
        let bottom = self.0.slice(s![..(self.0.shape()[0]-1), ..2, ..]);
        let top = self.0.slice(s![1.., 2.., ..]);
        let between = ndarray::concatenate![Axis(1), top, bottom];
        Corners(between)
    }
}

struct Centroids(Array2<f32>);

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
        for (t, (b, (l, mut r))) in zip(
            top.axis_iter(Axis(1)),
            zip(
                bottom.axis_iter(Axis(1)),
                zip(left.axis_iter(Axis(1)), right.axis_iter_mut(Axis(1))),
            ),
        ) {
            // [python - Numpy and line intersections - Stack Overflow](https://stackoverflow.com/questions/3252194/numpy-and-line-intersections/57821199#57821199)
            let (a1, a2) = (&t, &b);
            let (b1, b2) = (&l, &r);
            let da = a2 - a1;
            let db = b2 - b1;
            let dp = a1 - b1;
            let mut dap = da.clone();
            dap[[0]] = -da[[0]];
            let denom = dap.dot(&db);
            if denom == 0.0 {
                panic!("parallel line"); // should not reach here
            } else {
                let num = dap.dot(&dp);
                let c = num / denom * db + b1;
                r[[0]] = c[[0]];
            }
        }
        Centroids(right)
    }
}

impl TryFrom<&LabelMeData> for VertebraeC7TL {
    type Error = anyhow::Error;

    /// From [C7[TL, TR, BL, BR] - Sacral[TL, TR]] points
    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let corners = ["TL", "TR", "BL", "BR"]
            .iter()
            .map(|label| extract_points(&data, label))
            .collect::<Result<Vec<_>>>()?;
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
struct VertebraeTL {
    corners: Array3<f32>,
}
impl From<VertebraeC7TL> for VertebraeTL {
    fn from(c7tl: VertebraeC7TL) -> Self {
        let corners = c7tl.0.slice(s![1.., .., ..]).to_owned();
        Self { corners }
    }
}

impl TryFrom<LabelMeData> for VertebraeTL {
    type Error = anyhow::Error;

    fn try_from(data: LabelMeData) -> std::result::Result<Self, Self::Error> {
        let c7tl: VertebraeC7TL = (&data).try_into()?;
        Ok(c7tl.into())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let s = std::fs::read_to_string(args.filename)?;
    let data: LabelMeData = s.as_str().try_into()?;
    let v: VertebraeTL = data.try_into()?;
    let corners: Corners = v.into();
    let centroids: Centroids = (&corners).into();
    println!("{:?}", centroids.0.shape());
    let discs = corners.between();
    let disc_centroids: Centroids = (&discs).into();
    println!("{:?}", disc_centroids.0.shape());

    Ok(())
}
