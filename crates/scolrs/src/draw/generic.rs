use std::convert::Infallible;

use crate::{HasImageMetadata, Scalable, ScaledType};
use clap::ValueEnum;
use labelme_rs::LabelMeData;
use ndarray::{Array2, ArrayView2};
use serde;
use serde::{Deserialize, Serialize};

use crate::draw::{self, AsMeasure, ConfidenceComponent, DrawComponent};
use crate::ImageMetadata;

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
)]
#[serde(rename_all = "PascalCase")]
#[clap(rename_all = "PascalCase")]
pub enum GenericDraw {
    AllPoints,
}

pub enum GenericMeasure {}

#[derive(Debug, Clone, PartialEq, HasImageMetadata)]
pub struct GenericPoints {
    pub all_points: Vec<(String, Array2<f64>)>,
    pub image_metadata: ImageMetadata,
}

impl From<LabelMeData> for GenericPoints {
    fn from(lm_data: LabelMeData) -> Self {
        let shape_map = lm_data.to_shape_map();
        let lm_points = shape_map.get("point");
        let mut all_points: Vec<(String, Array2<f64>)> = Vec::new();
        if let Some(lm_points) = lm_points {
            for (label, points) in lm_points {
                if points.is_empty() {
                    continue;
                }
                let mut points_arr: Array2<f64> = Array2::zeros((points.len(), 2));
                for (i, point) in points.iter().enumerate() {
                    points_arr[[i, 0]] = point[0].0;
                    points_arr[[i, 1]] = point[0].1;
                }
                all_points.push((label.to_string(), points_arr));
            }
        }
        let image_metadata = ImageMetadata::from(lm_data.clone());
        GenericPoints {
            all_points,
            image_metadata,
        }
    }
}

impl GenericPoints {
    pub fn view(&self) -> Vec<(String, ArrayView2<'_, f64>)> {
        self.all_points
            .iter()
            .map(|(label, points)| (label.clone(), points.view()))
            .collect()
    }
}

impl Scalable for GenericPoints {
    type Error = Infallible;
    fn _impl_scale(&mut self) -> Result<(), Self::Error> {
        let scale_xy = ndarray::array![
            self.image_metadata.spacing_xy.0,
            self.image_metadata.spacing_xy.1
        ];
        for (_label, points) in &mut self.all_points {
            for mut point in points.outer_iter_mut() {
                point *= &scale_xy;
            }
        }
        Ok(())
    }
}

impl<'a, 'b> From<(&'b GenericDraw, &'a ScaledType<GenericPoints>)>
    for Box<dyn DrawComponent + 'a>
{
    fn from((_draw_type, gp): (&'b GenericDraw, &'a ScaledType<GenericPoints>)) -> Self {
        let all_points = gp.0.view();
        Box::new(draw::AllPoints(all_points))
    }
}

impl<'a, 'b> From<(&'b GenericMeasure, &'a ScaledType<GenericPoints>)>
    for Box<dyn ConfidenceComponent + 'a>
{
    fn from(_: (&'b GenericMeasure, &'a ScaledType<GenericPoints>)) -> Self {
        unreachable!("GenericDraw does not have confidence measures");
    }
}

impl AsMeasure for GenericDraw {
    type MeasureType = GenericMeasure;

    fn as_measure(&self) -> Option<Self::MeasureType> {
        None
    }
}
