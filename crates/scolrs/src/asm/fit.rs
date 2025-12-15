use crate::asm::{
    adam, adam::EarlyTermination, adam::TerminationCriterion, alignment::SimilarityTransform,
};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};

struct ActiveShapeModel {
    mean_shape: Array1<f64>,   // 2n vector: [x1, y1, x2, y2, ..., xn, yn]
    eigenvectors: Array2<f64>, // 2n x k matrix
    n_landmarks: usize,        // number of landmarks
    n_modes: usize,            // number of shape modes (k)
}

impl ActiveShapeModel {
    fn shape(&self, shape_params: &Array1<f64>) -> Array1<f64> {
        &self.mean_shape + self.eigenvectors.dot(shape_params)
    }

    /// Pre-apply global transform to the active shape model
    fn global_transform(&mut self, tr: &SimilarityTransform) {
        // mean
        let mean_2d = self
            .mean_shape
            .to_owned()
            .into_shape_with_order((self.mean_shape.len() / 2, 2))
            .unwrap();
        let transformed_mean: Array2<f64> = tr.transform(&mean_2d);
        self.mean_shape = transformed_mean
            .into_shape_with_order(self.mean_shape.len())
            .unwrap();
        // // eigenvectors
        for i in 0..self.eigenvectors.len_of(Axis(0)) {
            let ev_2d = self
                .eigenvectors
                .slice(s![i, ..])
                .to_owned()
                .into_shape_with_order((self.eigenvectors.len_of(Axis(1)) / 2, 2))
                .unwrap();
            let transformed_ev: Array2<f64> = tr.transform(&ev_2d);
            let ev_flat = transformed_ev
                .into_shape_with_order(self.eigenvectors.len_of(Axis(1)))
                .unwrap();
            self.eigenvectors.slice_mut(s![i, ..]).assign(&ev_flat);
        }
    }
}

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
    heatmap: &ArrayView2<f64>,
    shape_params: &Array1<f64>,
    // lambda: f64,
) -> (f64, Array1<f64>) {
    // Compute current shape: x = mean_shape + P * b
    let current_shape = asm.shape(shape_params);

    let mut objective = 0.0;
    let mut gradient = Array1::zeros(asm.n_modes);

    // Loop over all landmarks
    for i in 0..asm.n_landmarks {
        let x_i = current_shape[2 * i];
        let y_i = current_shape[2 * i + 1];

        // Get heatmap value and spatial gradients at landmark position
        let (h_value, grad_x, grad_y) = bilinear_interpolate_with_gradient(heatmap, x_i, y_i);

        // Accumulate objective (negative because we want to maximize heatmap response)
        objective -= h_value;

        // Compute gradient for each shape parameter b_m
        for m in 0..asm.n_modes {
            // P[2i-1, m] corresponds to x component
            // P[2i, m] corresponds to y component
            let p_x = asm.eigenvectors[[2 * i, m]];
            let p_y = asm.eigenvectors[[2 * i + 1, m]];

            // Chain rule: dJ/db_m = -sum_i (dH/dx_i * P[2i-1,m] + dH/dy_i * P[2i,m])
            gradient[m] -= grad_x * p_x + grad_y * p_y;
        }
    }

    (objective, gradient)
}

fn fit_asm_to_heatmap(
    asm: &ActiveShapeModel,
    termination: &mut EarlyTermination,
    heatmap: &Array2<f64>,
    lambda: f64,
    learning_rate: f64,
) -> (Array1<f64>, Vec<(f64, f64)>) {
    // Initialize shape parameters to zero (mean shape)
    let mut shape_params = Array1::zeros(asm.n_modes);
    let mut best_params = shape_params.clone();
    let mut best_objective = f64::INFINITY;

    // Initialize Adam optimizer
    let mut optimizer = adam::Adam::new(asm.n_modes, learning_rate);

    let mut obj_history = Vec::new();

    // for iteration in 0..max_iterations {
    loop {
        // Compute objective and gradient
        let (data_objective, data_gradient) =
            compute_objective_and_gradient(asm, &heatmap.view(), &shape_params);
        let reg_objective = lambda * shape_params.dot(&shape_params);
        let objective = data_objective + reg_objective;
        let reg_gradient = 2.0 * lambda * &shape_params;
        let gradient = &data_gradient + &reg_gradient;
        obj_history.push((data_objective, reg_objective));

        if objective < best_objective {
            log::debug!("New best objective: {}", objective);
            best_objective = objective;
            best_params = shape_params.clone();
        }

        if termination.should_stop(objective) {
            log::info!(
                "Early stopping at iteration {} with objective {}",
                optimizer.timestep(),
                objective
            );
            break;
        }

        // Update parameters using Adam
        optimizer.step(&mut shape_params, &gradient);
    }

    (best_params, obj_history)
}

fn get_fitted_shape(asm: &ActiveShapeModel, shape_params: &Array1<f64>) -> Array1<f64> {
    // Return the actual landmark positions: x = mean_shape + P * b
    &asm.mean_shape + asm.eigenvectors.dot(shape_params)
}

// Example usage
fn main() {
    // Assume we have loaded:
    // - asm: ActiveShapeModel with mean_shape and eigenvectors
    // - heatmap: Array2<f64> with shape (height, width)

    let n_landmarks = 68; // example: 68 facial landmarks
    let n_modes = 10; // use first 10 principal components

    // Create mock ASM (in practice, load from training data)
    let asm = ActiveShapeModel {
        mean_shape: Array1::zeros(2 * n_landmarks),
        eigenvectors: Array2::zeros((2 * n_landmarks, n_modes)),
        n_landmarks,
        n_modes,
    };

    // Create mock heatmap (in practice, load from neural network output)
    let heatmap = Array2::zeros((256, 256));

    // Fit ASM to heatmap
    let lambda = 0.01; // regularization weight
    let learning_rate = 0.001; // Adam learning rate
    let max_iterations = 1000;
    let patience = 8;
    let min_delta = 0.0;
    let mut termination = adam::EarlyTermination::new(TerminationCriterion::Any(vec![
        TerminationCriterion::NoImprovement {
            patience,
            min_delta,
        },
        TerminationCriterion::MaxIterations(max_iterations),
    ]));

    let (optimal_params, obj_history) =
        fit_asm_to_heatmap(&asm, &mut termination, &heatmap, lambda, learning_rate);

    // Get final fitted landmark positions
    let fitted_shape = get_fitted_shape(&asm, &optimal_params);

    println!("Fitted shape parameters: {:?}", optimal_params);
    println!("Objective history: {:?}", obj_history);

    // Extract individual landmarks
    for i in 0..n_landmarks {
        let x = fitted_shape[2 * i];
        let y = fitted_shape[2 * i + 1];
        println!("Landmark {}: ({:.2}, {:.2})", i, x, y);
    }
}
