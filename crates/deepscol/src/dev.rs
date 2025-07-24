use std::fs::File;

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use serde::Deserialize;

#[derive(Debug)]
pub struct ActiveShapeModel {
    pub model_name: String,
    pub mean: Array1<f32>,
    pub components: Array2<f32>,
    pub explained_variance: Array1<f32>,
    pub explained_variance_ratio: Array1<f32>,
    pub template_points: Array2<f32>,
    pub template_reference: Vec<Option<Array1<f32>>>,
    pub labels: Vec<String>,
    pub point_counts: Vec<usize>,
}

impl<'de> Deserialize<'de> for ActiveShapeModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data: ActiveShapeModelData = Deserialize::deserialize(deserializer)?;
        let mean = Array1::from(data.mean);
        let components = Array2::from_shape_vec(
            (data.components.len(), data.components[0].len()),
            data.components.into_iter().flatten().collect(),
        )
        .map_err(serde::de::Error::custom)?;
        let explained_variance = Array1::from(data.explained_variance);
        let explained_variance_ratio = Array1::from(data.explained_variance_ratio);
        let template_points = Array2::from_shape_vec(
            (data.template_points.len(), data.template_points[0].len()),
            data.template_points.into_iter().flatten().collect(),
        )
        .map_err(serde::de::Error::custom)?;
        let template_reference = data
            .template_reference
            .into_iter()
            .map(|opt| opt.map(Array1::from))
            .collect();
        Ok(Self {
            model_name: data.model_name,
            mean,
            components,
            explained_variance,
            explained_variance_ratio,
            template_points,
            template_reference,
            labels: data.labels,
            point_counts: data.point_counts,
        })
    }
}

#[derive(Deserialize)]
struct ActiveShapeModelData {
    model_name: String,
    mean: Vec<f32>,
    components: Vec<Vec<f32>>,
    explained_variance: Vec<f32>,
    explained_variance_ratio: Vec<f32>,
    template_points: Vec<Vec<f32>>,
    template_reference: Vec<Option<Vec<f32>>>,
    labels: Vec<String>,
    point_counts: Vec<usize>,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <pca_json_file_path>", args[0]);
        std::process::exit(1);
    }
    let pca_json_file_path = &args[1];
    // load the pca json file
    let pca_json = File::open(pca_json_file_path)?;
    let pca_model: ActiveShapeModel =
        serde_json::from_reader(pca_json).context("Failed to deserialize PCA model from JSON")?;
    println!("Model: {:?}", pca_model);
    Ok(())
}
