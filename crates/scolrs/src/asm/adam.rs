use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Adam (Adaptive Moment Estimation) optimizer with early termination
#[derive(Debug, Clone)]
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
    m: Array1<f64>, // First moment estimate
    v: Array1<f64>, // Second moment estimate
}

impl Adam {
    /// Create a new Adam optimizer
    pub fn new(param_count: usize, learning_rate: f64) -> Self {
        Self::with_params(param_count, learning_rate, 0.9, 0.999, 1e-8)
    }

    /// Create Adam optimizer with custom hyperparameters
    pub fn with_params(
        param_count: usize,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: Array1::zeros(param_count),
            v: Array1::zeros(param_count),
        }
    }

    /// Perform one optimization step
    pub fn step(&mut self, params: &mut Array1<f64>, gradients: ArrayView1<f64>) {
        assert_eq!(params.len(), gradients.len());
        assert_eq!(params.len(), self.m.len());

        self.t += 1;

        // Bias correction terms
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        // Update biased first moment estimate
        self.m = &self.m * self.beta1 + &gradients * (1.0 - self.beta1);

        // Update biased second moment estimate
        self.v = &self.v * self.beta2 + &(&gradients * &gradients) * (1.0 - self.beta2);

        // Compute bias-corrected estimates and update parameters
        let m_hat = &self.m / bias_correction1;
        let v_hat = &self.v / bias_correction2;

        *params -= &(m_hat * self.learning_rate / (v_hat.mapv(f64::sqrt) + self.epsilon));
    }

    /// Perform one optimization step with array views
    pub fn step_view(&mut self, mut params: ArrayViewMut1<f64>, gradients: ArrayView1<f64>) {
        assert_eq!(params.len(), gradients.len());
        assert_eq!(params.len(), self.m.len());

        self.t += 1;

        // Bias correction terms
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        // Update moments and parameters
        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradients[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gradients[i].powi(2);

            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            params[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.t = 0;
        self.m.fill(0.0);
        self.v.fill(0.0);
    }

    /// Get current timestep
    pub fn timestep(&self) -> usize {
        self.t
    }
}

#[derive(Debug, Clone)]
pub enum ObjectiveType {
    Minimize,
    Maximize,
}

/// Early termination criteria
#[derive(Debug, Clone)]
pub enum TerminationCriterion {
    /// Stop when loss is below threshold
    LossThreshold(f64),
    /// Stop when loss improvement over N iterations is below threshold
    NoImprovement {
        patience: usize,
        min_delta: f64,
        objective: ObjectiveType,
    },
    /// Stop after maximum iterations
    MaxIterations(usize),
    /// Combine multiple criteria (stops when any is met)
    Any(Vec<TerminationCriterion>),
    /// Combine multiple criteria (stops when all are met)
    All(Vec<TerminationCriterion>),
}

/// Early termination tracker
pub struct EarlyTermination {
    criterion: TerminationCriterion,
    best_loss: f64,
    iterations_without_improvement: usize,
    iteration: usize,
}

impl EarlyTermination {
    pub fn new(criterion: TerminationCriterion) -> Self {
        Self {
            criterion,
            best_loss: f64::INFINITY,
            iterations_without_improvement: 0,
            iteration: 0,
        }
    }

    /// Check if training should terminate
    pub fn should_stop(&mut self, loss: f64) -> bool {
        self.iteration += 1;
        self.check_criterion(&self.criterion.clone(), loss)
    }

    fn check_criterion(&mut self, criterion: &TerminationCriterion, loss: f64) -> bool {
        match criterion {
            TerminationCriterion::LossThreshold(threshold) => loss < *threshold,

            TerminationCriterion::NoImprovement {
                patience,
                min_delta,
                objective,
            } => {
                let is_improvement = match objective {
                    ObjectiveType::Minimize => loss < self.best_loss - min_delta,
                    ObjectiveType::Maximize => loss > self.best_loss + min_delta,
                };

                if is_improvement {
                    self.best_loss = loss;
                    self.iterations_without_improvement = 0;
                    false
                } else {
                    self.iterations_without_improvement += 1;
                    self.iterations_without_improvement >= *patience
                }
            }

            TerminationCriterion::MaxIterations(max_iter) => self.iteration >= *max_iter,

            TerminationCriterion::Any(criteria) => {
                criteria.iter().any(|c| self.check_criterion(c, loss))
            }

            TerminationCriterion::All(criteria) => {
                criteria.iter().all(|c| self.check_criterion(c, loss))
            }
        }
    }

    pub fn reset(&mut self) {
        self.best_loss = f64::INFINITY;
        self.iterations_without_improvement = 0;
        self.iteration = 0;
    }

    pub fn current_iteration(&self) -> usize {
        self.iteration
    }
}
