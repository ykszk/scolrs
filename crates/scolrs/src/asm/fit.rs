use crate::asm::{adam, adam::EarlyTermination, model::ActiveShapeModel};
use ndarray::{Array1, ArrayView1, ArrayView2, ArrayView3, Axis};
use serde::{Deserialize, Serialize};

fn bilinear_interpolate_with_gradient(
    heatmap: &ArrayView2<f64>,
    x: f64,
    y: f64,
) -> (f64, f64, f64) {
    let (height, width) = heatmap.dim();

    // Clamp coordinates to valid range
    let x = x.max(0.0).min(width as f64 - 1.001);
    let y = y.max(0.0).min(height as f64 - 1.001);

    // Get integer and fractional parts
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let alpha = x - x0 as f64;
    let beta = y - y0 as f64;

    // Get four corner values
    let h00 = heatmap[[y0, x0]];
    let h10 = heatmap[[y0, x1]];
    let h01 = heatmap[[y1, x0]];
    let h11 = heatmap[[y1, x1]];

    // Bilinear interpolation
    let value = (1.0 - alpha) * (1.0 - beta) * h00
        + alpha * (1.0 - beta) * h10
        + (1.0 - alpha) * beta * h01
        + alpha * beta * h11;

    // Gradient with respect to x
    let grad_x = (1.0 - beta) * (h10 - h00) + beta * (h11 - h01);

    // Gradient with respect to y
    let grad_y = (1.0 - alpha) * (h01 - h00) + alpha * (h11 - h10);

    (value, grad_x, grad_y)
}

fn compute_objective_and_gradient(
    asm: &ActiveShapeModel,
    heatmaps: ArrayView3<f64>,
    shape_params: ArrayView1<f64>,
    // lambda: f64,
) -> (f64, Array1<f64>) {
    // Compute current shape: x = mean_shape + P * b
    let nested_points = asm.pad_deform(shape_params);
    let mut objective = 0.0;
    let mut gradient = Array1::zeros(shape_params.len());

    for (channel_idx, current_shape) in nested_points.iter().enumerate() {
        let heatmap = heatmaps.index_axis(ndarray::Axis(2), channel_idx);

        // Compute objective and gradient for this channel
        for (i, point) in current_shape.axis_iter(Axis(0)).enumerate() {
            let x_i = point[0];
            let y_i = point[1];

            // Get heatmap value and spatial gradients at landmark position
            let (h_value, grad_x, grad_y) =
                bilinear_interpolate_with_gradient(&heatmap.view(), x_i, y_i);
            log::trace!(
                "Point {} as channel {}: position=({:.2}, {:.2}), heatmap value={:.4}, grad=({:.4}, {:.4})",
                i,
                channel_idx,
                x_i,
                y_i,
                h_value,
                grad_x,
                grad_y
            );

            // Accumulate objective (negative because we want to maximize heatmap response)
            objective -= h_value;

            // Compute gradient for each shape parameter b_m
            for m in 0..gradient.len() {
                // P[m, 2i-1] corresponds to x component
                // P[m, 2i] corresponds to y component
                let p_x = asm.scaled_components[[m, 2 * i]];
                let p_y = asm.scaled_components[[m, 2 * i + 1]];

                // Chain rule: dJ/db_m = -sum_i (dH/dx_i * P[m, 2i] + dH/dy_i * P[m, 2i+1])
                gradient[m] -= grad_x * p_x + grad_y * p_y;
            }
        }
    }

    (objective, gradient)
}

pub struct HistoryItem {
    pub data_objective: f64,
    pub reg_objective: f64,
    pub shape_params: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct History {
    pub data_objectives: Vec<f64>,
    pub reg_objectives: Vec<f64>,
    pub shape_params: Vec<Vec<f64>>,
}

impl History {
    pub fn from_items(items: Vec<HistoryItem>) -> Self {
        let data_objectives = items.iter().map(|item| item.data_objective).collect();
        let reg_objectives = items.iter().map(|item| item.reg_objective).collect();
        let shape_params = items.into_iter().map(|item| item.shape_params).collect();
        History {
            data_objectives,
            reg_objectives,
            shape_params,
        }
    }
}

pub fn fit_asm_to_heatmap(
    asm: &ActiveShapeModel,
    initial_params: ArrayView1<f64>,
    termination: &mut EarlyTermination,
    heatmaps: ArrayView3<f64>,
    lambda: f64,
    learning_rate: f64,
) -> (Array1<f64>, History) {
    // Initialize shape parameters to zero (mean shape)
    let mut shape_params = initial_params.to_owned();
    let mut best_params = initial_params.to_owned();
    let mut best_objective = f64::INFINITY;

    // Initialize Adam optimizer
    let mut optimizer = adam::Adam::new(initial_params.len(), learning_rate);

    let mut obj_history = Vec::new();

    loop {
        // Compute objective and gradient
        let (data_objective, data_gradient) =
            compute_objective_and_gradient(asm, heatmaps, shape_params.view());
        let reg_objective = shape_params.dot(&shape_params);
        let objective = data_objective + lambda * reg_objective;
        let reg_gradient = 2.0 * lambda * &shape_params;
        let gradient = &data_gradient + &reg_gradient;
        obj_history.push(HistoryItem {
            data_objective,
            reg_objective,
            shape_params: shape_params.to_vec(),
        });
        log::debug!(
            "Iteration {}: Data Obj = {:.6}, Reg Obj = {:.6}, Total Obj = {:.6}",
            optimizer.timestep(),
            data_objective,
            reg_objective,
            objective
        );

        if objective < best_objective {
            log::info!(
                "New best objective at {}: {}",
                optimizer.timestep(),
                objective
            );
            best_objective = objective;
            best_params = shape_params.clone();
        }

        if termination.should_stop(objective) {
            log::info!(
                "Early stopping at iteration {} with objective {}",
                optimizer.timestep(),
                best_objective
            );
            break;
        }

        // Update parameters using Adam
        optimizer.step(&mut shape_params, gradient.view());
    }

    (best_params, History::from_items(obj_history))
}
