use crate::asm::model::ActiveShapeModel;
use ndarray::{array, s, Array1, Array2, Axis};
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};

use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;
// use std::collections::HashMap;

/// Result of ICP optimization
#[derive(Debug, Clone)]
pub struct IcpResult {
    pub b: Array1<f64>,
    pub energy: f64,
    pub n_iter: usize,
    pub converged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcpConfig {
    /// Weight for model-to-target term
    pub alpha: f64,
    /// Weight for target-to-model term
    pub beta: f64,
    /// Weight for regularization term
    pub lambda: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance (relative energy change)
    pub tol: f64,
}

impl Default for IcpConfig {
    fn default() -> Self {
        IcpConfig {
            alpha: 1.0,
            beta: 1.0,
            lambda: 0.1,
            max_iter: 100,
            tol: 1e-6,
        }
    }
}

/// Calculate all pairwise squared distances between two sets of points
fn cdist(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let na = a.len_of(Axis(0));
    let nb = b.len_of(Axis(0));
    let mut dists = Array2::zeros((na, nb));
    for i in 0..na {
        for j in 0..nb {
            let diff = &a.row(i) - &b.row(j);
            dists[[i, j]] = diff.dot(&diff)
        }
    }
    dists
}

impl ActiveShapeModel {
    /// Bidirectional ICP-ASM optimization
    ///
    /// # Arguments
    /// * `target_points` - (N, 2) array of target points
    /// * `alpha`, `beta`, `lambda` - weights for model-to-target, target-to-model, and regularization
    /// * `max_iter` - maximum number of iterations
    /// * `tol` - convergence tolerance (relative energy change)
    /// * `b0` - initial shape parameters (optional)
    pub fn icp_optimize(
        &self,
        target_points: &Array2<f64>,
        config: &IcpConfig,
        b0: Option<Array1<f64>>,
    ) -> Result<IcpResult, rulinalg::error::Error> {
        let IcpConfig {
            alpha,
            beta,
            lambda,
            max_iter,
            tol,
        } = *config;

        let m = self.mean.len() / 2; // number of model points
        let k = self.components.nrows();
        let d = 2;

        // Initial b
        let mut b = b0.unwrap_or_else(|| Array1::zeros(k));
        let mut prev_energy = f64::MAX;
        let mut converged = false;
        let mut n_iter = 0;

        // Precompute P_i blocks
        let mut p_blocks: Vec<Array2<f64>> = Vec::with_capacity(m);
        for i in 0..m {
            let p_i = self
                .scaled_components
                .slice(s![.., i * d..(i + 1) * d])
                .t()
                .to_owned(); // (2, k)
            p_blocks.push(p_i);
        }

        // Precompute mean_i blocks
        let mut mean_blocks: Vec<Array1<f64>> = Vec::with_capacity(m);
        for i in 0..m {
            let mean_i = self.mean.slice(s![i * d..(i + 1) * d]).to_owned();
            mean_blocks.push(mean_i);
        }

        // Precompute P^T P
        let ptp = {
            let p = self.scaled_components.view(); // (k, 2m)
            let pt = p.t();
            pt.dot(&p)
        };

        for iter in 0..max_iter {
            // 1. Compute current model points
            let x_b = self.inverse_transform(b.view()); // (2m,)

            let dists = cdist(
                &Array2::from_shape_vec((m, 2), x_b.to_vec()).unwrap(),
                target_points,
            );

            // 2. Forward correspondence: for each model point, find closest target point
            let c_x2y = dists
                .axis_iter(Axis(0))
                .map(|row| row.argmin().unwrap())
                .collect::<Vec<usize>>();

            // 3. Backward correspondence: for each target point, find closest model point
            let c_y2x = dists
                .axis_iter(Axis(1))
                .map(|col| col.argmin().unwrap())
                .collect::<Vec<usize>>();

            // 4. Build system matrix A and vector g
            // A = alpha P^T P + beta sum_j P_{c_y2x(j)}^T P_{c_y2x(j)} + lambda I
            // g = alpha P^T r^{x2y} + beta sum_j P_{c_y2x(j)}^T r_j^{y2x}

            // Count for each model point how many target points map to it
            let mut n_i = vec![0usize; m];
            for &i in &c_y2x {
                n_i[i] += 1;
            }

            // sum_j P_{c_y2x(j)}^T P_{c_y2x(j)}
            let mut sum_p_i_t = Array2::<f64>::zeros((k, k));
            for i in 0..m {
                if n_i[i] > 0 {
                    let p_i = &p_blocks[i]; // (2, k)
                    let p_i_t = p_i.t(); // (k, 2)
                    let prod = p_i_t.dot(p_i); // (k, k)
                    sum_p_i_t = &sum_p_i_t + &(prod * n_i[i] as f64);
                }
            }

            // System matrix A
            let mut a = alpha * &ptp + beta * &sum_p_i_t;
            a.diag_mut().iter_mut().for_each(|x| *x += lambda);

            // r^{x2y} = y_c^{x2y} - mean
            let mut y_c_x2y = Array1::<f64>::zeros(m * d);
            for i in 0..m {
                let j = c_x2y[i];
                y_c_x2y[i * d] = target_points[[j, 0]];
                y_c_x2y[i * d + 1] = target_points[[j, 1]];
            }
            let r_x2y = &y_c_x2y - &self.mean;

            // P^T r^{x2y}
            let p = self.scaled_components.view();
            let pt_r = p.t().dot(&r_x2y);

            // sum_j P_{c_y2x(j)}^T r_j^{y2x}
            let mut sum_pit_r = Array1::<f64>::zeros(k);
            for (j, &i) in c_y2x.iter().enumerate() {
                let p_i = &p_blocks[i]; // (2, k)
                let p_i_t = p_i.t(); // (k, 2)
                let yj = target_points.row(j);
                let mean_i = &mean_blocks[i];
                let r_j = array![yj[0] - mean_i[0], yj[1] - mean_i[1]];
                let contrib = p_i_t.dot(&r_j);
                sum_pit_r = &sum_pit_r + &contrib;
            }

            let g = alpha * &pt_r + beta * &sum_pit_r;

            // Solve A b = g
            // Convert to rulinalg for solving
            let a_mat = Matrix::from_fn(k, k, |i, j| a[(i, j)]);
            let g_vec = Vector::from_fn(k, |i| g[i]);
            let b_new = a_mat.solve(g_vec)?;
            let b_new_arr = Array1::from(b_new.into_vec());

            // Compute energy
            let energy_m2t = c_x2y
                .iter()
                .enumerate()
                .map(|(i, &j)| dists[[i, j]])
                .sum::<f64>();

            // Target-to-model
            let energy_t2m = c_y2x
                .iter()
                .enumerate()
                .map(|(j, &i)| dists[[i, j]])
                .sum::<f64>();

            // Regularization
            let energy_reg = b_new_arr.dot(&b_new_arr);

            let energy = alpha * energy_m2t + beta * energy_t2m + lambda * energy_reg;

            // Check convergence
            let rel_change = ((energy - prev_energy).abs() / prev_energy.abs()).min(1.0);
            if rel_change < tol {
                log::info!(
                    "ICP converged at iter {}: energy={:.6}, rel_change={:.6}",
                    iter,
                    energy,
                    rel_change
                );
                converged = true;
                n_iter = iter + 1;
                b = b_new_arr;
                prev_energy = energy;
                break;
            }
            b = b_new_arr;
            prev_energy = energy;
            n_iter = iter + 1;
            log::trace!(
                "ICP iter {}: energy={:.6}, rel_change={:.6}",
                iter,
                energy,
                rel_change
            );
        }

        Ok(IcpResult {
            b,
            energy: prev_energy,
            n_iter,
            converged,
        })
    }
}
