pub mod adam;
pub mod alignment;
pub mod fit;
pub mod model;

use crate::asm::{
    self,
    adam::{Stopper, StopperConfig},
    alignment::AlignmentError,
    fit::{fit_asm_to_heatmap, FitConfig, History},
    model::{ActiveShapeModel, ModeConfig},
};
use labelme_rs::{LabelMeData, Shape};
use ndarray::{s, Array1, Array2, Array3, ArrayView3, Axis};
use ndarray_ndimage as ndi;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsmConfig {
    pub fit: FitConfig,
    pub stopper: StopperConfig,
    pub sigmas: Vec<f64>,
    pub mode: ModeConfig,
}

impl Default for AsmConfig {
    fn default() -> Self {
        let sigmas = vec![10.0, 5.0, 0.1];
        Self {
            sigmas,
            fit: FitConfig::default(),
            stopper: StopperConfig::default(),
            mode: ModeConfig::default(),
        }
    }
}

pub fn extract_reference_points(
    lm_data: &LabelMeData,
    reference_labels: &[String],
) -> Vec<Option<(f64, f64)>> {
    let mut points = Vec::new();
    let shape_dict = lm_data.to_shape_map();
    let Some(point_map) = shape_dict.get("point") else {
        log::warn!("No 'point' shape found in labelme data.");
        return vec![None; reference_labels.len()];
    };
    for label in reference_labels {
        match point_map.get(label.as_str()) {
            Some(p) => {
                if p.len() > 1 {
                    log::warn!(
                        "Reference point '{}' has more than one point ({}), using the first one.",
                        label,
                        p.len()
                    );
                }
                points.push(Some((p[0][0].0, p[0][0].1)));
            }
            None => {
                points.push(None);
            }
        }
    }
    points
}

pub fn create_shapes_from_fitted_points(
    fitted_shape: &[Array2<f64>],
    labels: &[String],
) -> Vec<Shape> {
    let mut shapes = Vec::new();
    for (points, label) in fitted_shape.iter().zip(labels.iter()) {
        for point in points.axis_iter(Axis(0)) {
            let shape = Shape {
                label: label.clone(),
                points: vec![(point[0], point[1])],
                group_id: None,
                shape_type: "point".to_string(),
                flags: Default::default(),
            };
            shapes.push(shape);
        }
    }
    shapes
}

fn smooth_heatmaps(heatmaps: ArrayView3<f64>, sigma: f64) -> Array3<f64> {
    let mut smoothed_heatmaps = heatmaps.to_owned();
    for (ch_idx, channel) in heatmaps.axis_iter(Axis(2)).enumerate() {
        let smoothed = ndi::gaussian_filter(&channel, sigma, 0, ndi::BorderMode::Mirror, 3);
        smoothed_heatmaps
            .index_axis_mut(ndarray::Axis(2), ch_idx)
            .assign(&smoothed);
    }
    smoothed_heatmaps
}

fn fit_asm(
    asm: &ActiveShapeModel,
    asm_config: AsmConfig,
    heatmaps: ArrayView3<f64>,
) -> (Vec<asm::fit::History>, Array1<f64>, Vec<Array2<f64>>) {
    let n_mode = asm.calculate_mode(asm_config.mode);
    log::info!("Using {} modes for fitting.", n_mode);

    let mut initial_params = Array1::zeros(n_mode);
    if asm_config.sigmas.is_empty() {
        unreachable!("At least one sigma value must be provided for heatmap smoothing.")
    };
    let mut histories = Vec::new();
    for sigma in asm_config.sigmas {
        let mut stopper = Stopper::from_config(asm_config.stopper.clone());
        log::info!("Fitting ASM with heatmap smoothing sigma = {}", sigma);
        let smoothed_heatmaps = if sigma > 0.0 {
            let smoothed = smooth_heatmaps(heatmaps, sigma);
            log::debug!("Smoothed heatmaps with sigma {}", sigma);
            smoothed
        } else {
            heatmaps.to_owned()
        };
        let (fitted_params, obj_history) = fit_asm_to_heatmap(
            asm,
            initial_params.view(),
            &mut stopper,
            smoothed_heatmaps.view(),
            asm_config.fit.clone(),
        );
        log::info!(
            "Fitting with sigma {} completed in {} iterations.",
            sigma,
            obj_history.data_objectives.len()
        );
        // Update initial_params for next sigma
        initial_params.assign(&fitted_params);
        histories.push(obj_history);
    }
    let optimal_params = initial_params;

    // Get final fitted landmark positions
    let fitted_shape = asm.pad_deform(optimal_params.view());
    (histories, optimal_params, fitted_shape)
}

/// Errors that can occur during alignment operations
#[derive(Debug, thiserror::Error, Clone)]
pub enum AsmError {
    #[error("Alignment error: {0}")]
    AlignmentError(#[from] AlignmentError),
    #[error("Not enough heatmap channels: {0}, but ASM requires {1} channels.")]
    HeatmapError(usize, usize),
}

type AsmReturn = (LabelMeData, Vec<History>, Array1<f64>, Vec<Array2<f64>>);

pub fn apply_asm(
    asm: ActiveShapeModel,
    asm_config: AsmConfig,
    heatmaps: ArrayView3<f64>,
    ref_lm: &LabelMeData,
) -> Result<AsmReturn, AsmError> {
    let n_required_channels = asm.labels.len();
    let n_actual_channels = heatmaps.len_of(Axis(2));
    if n_actual_channels < n_required_channels {
        return Err(AsmError::HeatmapError(
            n_actual_channels,
            n_required_channels,
        ));
    }
    let heatmaps = if n_actual_channels > n_required_channels {
        log::info!(
            "Heatmaps have {0} channels, but ASM requires only {1} channels. Using the first {1} channels.",
            n_actual_channels,
            n_required_channels,
        );
        heatmaps.slice_move(s![.., .., 0..n_required_channels])
    } else {
        heatmaps
    };
    let scale_hm_to_ref = ref_lm.imageHeight as f64 / heatmaps.len_of(Axis(0)) as f64;
    let mut scaled_ref_lm = ref_lm.clone();
    log::debug!(
        "Scaling reference labelme data by factor {} to match heatmap size",
        scale_hm_to_ref
    );
    scaled_ref_lm.scale(1.0 / scale_hm_to_ref);
    let ref_points = extract_reference_points(&scaled_ref_lm, &asm.reference_labels);
    let asm_movable_points = asm.to_movable_points();
    log::debug!(
        "Calculating transform to align ASM {:?} to reference points {:?}",
        asm.template_reference,
        ref_points
    );
    let tr_asm_to_ref = asm_movable_points.calculate_transform_with_missing(&ref_points, true)?;
    log::debug!("Transform ASM to reference: {:?}", tr_asm_to_ref);
    let mut asm = asm;
    asm.global_transform(&tr_asm_to_ref);
    let (histories, optimal_params, fitted_shape) = fit_asm(&asm, asm_config, heatmaps);

    let mut output_lm = ref_lm.clone();
    output_lm.shapes = create_shapes_from_fitted_points(&fitted_shape, &asm.labels);
    output_lm.scale(scale_hm_to_ref);
    output_lm.imageWidth = ref_lm.imageWidth;
    output_lm.imageHeight = ref_lm.imageHeight;
    output_lm.version = crate::VERSION.to_string();

    Ok((output_lm, histories, optimal_params, fitted_shape))
}
