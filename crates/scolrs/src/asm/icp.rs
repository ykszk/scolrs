use crate::asm::model::ActiveShapeModel;
use ndarray::{array, s, Array1, Array2};
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;
// use std::collections::HashMap;

/// Result of ICP optimization
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
    ) -> IcpResult {
        let IcpConfig {
            alpha,
            beta,
            lambda,
            max_iter,
            tol,
        } = *config;

        let m = self.mean.len() / 2; // number of model points
        let n = target_points.nrows();
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

        for iter in 0..max_iter {
            // 1. Compute current model points
            let x_b = self.inverse_transform(b.view()); // (2m,)
            let x_b_points = x_b
                .as_slice()
                .unwrap()
                .chunks(2)
                .map(|xy| [xy[0], xy[1]])
                .collect::<Vec<_>>(); // Vec<[f64;2]>

            // 2. Forward correspondence: for each model point, find closest target point
            let mut c_x2y = vec![0; m];
            for i in 0..m {
                let xi = &x_b_points[i];
                let mut min_dist = f64::MAX;
                let mut min_j = 0;
                for j in 0..n {
                    let yj = target_points.row(j);
                    let dist = (xi[0] - yj[0]).powi(2) + (xi[1] - yj[1]).powi(2);
                    if dist < min_dist {
                        min_dist = dist;
                        min_j = j;
                    }
                }
                c_x2y[i] = min_j;
            }

            // 3. Backward correspondence: for each target point, find closest model point
            let mut c_y2x = vec![0; n];
            for j in 0..n {
                let yj = target_points.row(j);
                let mut min_dist = f64::MAX;
                let mut min_i = 0;
                for i in 0..m {
                    let xi = &x_b_points[i];
                    let dist = (xi[0] - yj[0]).powi(2) + (xi[1] - yj[1]).powi(2);
                    if dist < min_dist {
                        min_dist = dist;
                        min_i = i;
                    }
                }
                c_y2x[j] = min_i;
            }

            // 4. Build system matrix A and vector g
            // A = alpha P^T P + beta sum_j P_{c_y2x(j)}^T P_{c_y2x(j)} + lambda I
            // g = alpha P^T r^{x2y} + beta sum_j P_{c_y2x(j)}^T r_j^{y2x}

            // P^T P
            let ptp = {
                let p = self.scaled_components.view(); // (k, 2m)
                let pt = p.t();
                let ptp = pt.dot(&p);
                ptp
            };

            // Count for each model point how many target points map to it
            let mut n_i = vec![0usize; m];
            for &i in &c_y2x {
                n_i[i] += 1;
            }

            // beta sum_j P_{c_y2x(j)}^T P_{c_y2x(j)}
            let mut beta_sum = Array2::<f64>::zeros((k, k));
            for i in 0..m {
                if n_i[i] > 0 {
                    let p_i = &p_blocks[i]; // (2, k)
                    let p_i_t = p_i.t(); // (k, 2)
                    let prod = p_i_t.dot(p_i); // (k, k)
                    beta_sum = &beta_sum + &(prod * n_i[i] as f64);
                }
            }

            // System matrix A
            let mut a = ptp.mapv(|v| alpha * v) + beta_sum.mapv(|v| beta * v);
            for i in 0..k {
                a[(i, i)] += lambda;
            }

            // r^{x2y} = y_c^{x2y} - mean
            let mut y_c_x2y = Array1::<f64>::zeros(m * d);
            for i in 0..m {
                let j = c_x2y[i];
                y_c_x2y[i * d] = target_points[[j, 0]];
                y_c_x2y[i * d + 1] = target_points[[j, 1]];
            }
            let r_x2y = &y_c_x2y - &self.mean;

            // alpha P^T r^{x2y}
            let p = self.scaled_components.view();
            let alpha_ptr = p.t().dot(&r_x2y) * alpha;

            // beta sum_j P_{c_y2x(j)}^T r_j^{y2x}
            let mut beta_sum_vec = Array1::<f64>::zeros(k);
            for j in 0..n {
                let i = c_y2x[j];
                let p_i = &p_blocks[i]; // (2, k)
                let p_i_t = p_i.t(); // (k, 2)
                let yj = target_points.row(j);
                let mean_i = &mean_blocks[i];
                let r_j = array![yj[0] - mean_i[0], yj[1] - mean_i[1]];
                let contrib = p_i_t.dot(&r_j) * beta;
                beta_sum_vec = &beta_sum_vec + &contrib;
            }

            let g = &alpha_ptr + &beta_sum_vec;

            // Solve A b = g
            // Convert to rulinalg for solving
            let a_mat = Matrix::from_fn(k, k, |i, j| a[(i, j)]);
            let g_vec = Vector::from_fn(k, |i| g[i]);
            let b_new = a_mat.solve(g_vec).unwrap();
            let b_new_arr = Array1::from(b_new.data().clone());

            // Compute energy
            let mut energy = 0.0;
            // Model-to-target
            for i in 0..m {
                let xi = &x_b_points[i];
                let j = c_x2y[i];
                let yj = target_points.row(j);
                energy += alpha * ((xi[0] - yj[0]).powi(2) + (xi[1] - yj[1]).powi(2));
            }
            // Target-to-model
            for j in 0..n {
                let i = c_y2x[j];
                let xi = &x_b_points[i];
                let yj = target_points.row(j);
                energy += beta * ((xi[0] - yj[0]).powi(2) + (xi[1] - yj[1]).powi(2));
            }
            // Regularization
            energy += lambda * b_new_arr.dot(&b_new_arr);

            // Check convergence
            let rel_change = ((energy - prev_energy).abs() / prev_energy.abs()).min(1.0);
            if rel_change < tol {
                converged = true;
                n_iter = iter + 1;
                b = b_new_arr;
                prev_energy = energy;
                break;
            }
            b = b_new_arr;
            prev_energy = energy;
            n_iter = iter + 1;
        }

        IcpResult {
            b,
            energy: prev_energy,
            n_iter,
            converged,
        }
    }
}
