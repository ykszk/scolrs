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
use ndarray::{Array1, Array2, Array3, ArrayView3, Axis};
use ndarray_ndimage as ndi;
use serde::{Deserialize, Serialize};

pub fn add_env_config(
    builder: config::ConfigBuilder<config::builder::DefaultState>,
) -> config::ConfigBuilder<config::builder::DefaultState> {
    builder.add_source(
        config::Environment::with_prefix("ASM")
            .separator("__")
            .list_separator(",")
            .try_parsing(true),
    )
}

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
    for (ch_idx, channel) in heatmaps.axis_iter(Axis(0)).enumerate() {
        let smoothed = ndi::gaussian_filter(&channel, sigma, 0, ndi::BorderMode::Nearest, 3);
        smoothed_heatmaps
            .index_axis_mut(ndarray::Axis(0), ch_idx)
            .assign(&smoothed);
    }
    smoothed_heatmaps
}

/// Cache smoothed heatmaps for different sigma values
pub struct CachedHeatmaps<'a> {
    pub original: ArrayView3<'a, f64>,
    pub smoothed: Vec<(f64, Array3<f64>)>,
}

impl<'a> CachedHeatmaps<'a> {
    pub fn new(heatmaps: ArrayView3<'a, f64>) -> Self {
        Self {
            original: heatmaps,
            smoothed: Vec::new(),
        }
    }

    pub fn get_smoothed(&'_ mut self, sigma: f64) -> ArrayView3<'_, f64> {
        if sigma == 0.0 {
            log::debug!("Using original heatmaps without smoothing for sigma 0.0.");
            return self.original.view();
        }
        if let Some(index) = self.smoothed.iter().position(|(s, _)| *s == sigma) {
            log::debug!("Using cached smoothed heatmaps for sigma {}", sigma,);
            self.smoothed[index].1.view()
        } else {
            log::debug!("Smoothing heatmaps with sigma {}", sigma,);
            let smoothed = smooth_heatmaps(self.original, sigma);
            self.smoothed.push((sigma, smoothed));
            self.smoothed.last().unwrap().1.view()
        }
    }
}

fn fit_asm(
    asm: &ActiveShapeModel,
    asm_config: AsmConfig,
    cached_heatmaps: &mut CachedHeatmaps,
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

        let smoothed_heatmaps = cached_heatmaps.get_smoothed(sigma);
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

/// Apply Active Shape Model to the given heatmaps and reference labelme data
///
/// heatmaps is in channel-first format (channels, height, width)
pub fn apply_asm(
    asm: ActiveShapeModel,
    asm_config: AsmConfig,
    cached_heatmaps: &mut CachedHeatmaps,
    ref_lm: &LabelMeData,
) -> Result<AsmReturn, AsmError> {
    let scale_hm_to_ref =
        ref_lm.imageHeight as f64 / cached_heatmaps.original.len_of(Axis(1)) as f64;
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
    let (histories, optimal_params, fitted_shape) = fit_asm(&asm, asm_config, cached_heatmaps);

    let mut output_lm = ref_lm.clone();
    output_lm.shapes = create_shapes_from_fitted_points(&fitted_shape, &asm.labels);
    output_lm.scale(scale_hm_to_ref);
    output_lm.imageWidth = ref_lm.imageWidth;
    output_lm.imageHeight = ref_lm.imageHeight;
    output_lm.version = crate::VERSION.to_string();

    Ok((output_lm, histories, optimal_params, fitted_shape))
}
