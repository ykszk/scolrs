use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use serde::Deserialize;
use thiserror::Error;

use crate::asm::alignment::SimilarityTransform;

/// Errors that can occur during Active Shape Model operations
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Deformation dimension {0} does not match PCA components dimension {1}")]
    DeformationDimensionMismatch(usize, usize),

    #[error("Points dimension {0} does not match PCA mean dimension {1}")]
    PointsDimensionMismatch(usize, usize),

    #[error("Failed to load model from file: {0}")]
    LoadError(String),

    #[error("Invalid model data: {0}")]
    InvalidData(String),
}

/// Active Shape Model (ASM) for statistical shape modeling and fitting
///
/// This struct contains the PCA decomposition of a set of aligned shapes,
/// allowing for shape generation, deformation, and inverse transformation.
#[derive(Debug, Clone)]
pub struct ActiveShapeModel {
    /// PCA mean vector (flattened 2D points)
    pub pca_mean: Array1<f64>,

    /// PCA components matrix (eigenvectors)
    pub pca_components: Array2<f64>,

    /// PCA components scaled by sqrt(explained_variance)
    pub pca_scaled_components: Array2<f64>,

    /// Explained variance for each component
    pub explained_variance: Array1<f64>,

    /// Ratio of explained variance for each component
    pub explained_variance_ratio: Array1<f64>,

    /// Template points (reference shape)
    pub template_points: Array2<f64>,

    /// Template reference points (potentially None if missing)
    pub template_reference: Vec<Option<(f64, f64)>>,

    /// Labels for each landmark
    pub labels: Vec<String>,

    /// Number of points per landmark group
    pub point_counts: Vec<usize>,

    /// Split indices for unraveling flattened points into groups
    split: Vec<usize>,
}

/// Parameters for creating an ActiveShapeModel
#[derive(Debug, Clone)]
pub struct ModelParams {
    /// PCA mean vector
    pub mean: Array1<f64>,
    /// PCA components matrix
    pub components: Array2<f64>,
    /// Explained variance for each component
    pub explained_variance: Array1<f64>,
    /// Ratio of explained variance
    pub explained_variance_ratio: Array1<f64>,
    /// Template reference shape
    pub template_points: Array2<f64>,
    /// Template reference points for alignment
    pub template_reference: Vec<Option<(f64, f64)>>,
    /// Labels for landmarks
    pub labels: Vec<String>,
    /// Number of points per landmark group
    pub point_counts: Vec<usize>,
}

impl ActiveShapeModel {
    /// Create a new ActiveShapeModel
    ///
    /// # Arguments
    /// * `params` - Model parameters structure containing all required data
    pub fn new(params: ModelParams) -> Self {
        let ModelParams {
            mean,
            components,
            explained_variance,
            explained_variance_ratio,
            template_points,
            template_reference,
            labels,
            point_counts,
        } = params;
        // Calculate scaled components
        let pca_scaled_components =
            &components * &explained_variance.mapv(|v| v.sqrt()).insert_axis(Axis(1));

        // Calculate split indices: [0, 2*count[0], 2*count[0]+2*count[1], ...]
        let mut split = vec![0];
        let mut cumsum = 0;
        for &count in &point_counts {
            cumsum += 2 * count;
            split.push(cumsum);
        }

        Self {
            pca_mean: mean,
            pca_components: components,
            pca_scaled_components,
            explained_variance,
            explained_variance_ratio,
            template_points,
            template_reference,
            labels,
            point_counts,
            split,
        }
    }

    /// Unravel flattened points into a list of arrays per landmark group
    ///
    /// # Arguments
    /// * `points` - Flattened 1D array of points
    ///
    /// # Returns
    /// * Vector of 2D point arrays, one per landmark group
    pub fn unravel(&self, points: &Array1<f64>) -> Vec<Array2<f64>> {
        let mut unraveled = Vec::with_capacity(self.split.len() - 1);

        for i in 1..self.split.len() {
            let start = self.split[i - 1];
            let end = self.split[i];
            let slice = points.slice(ndarray::s![start..end]);
            let n_points = (end - start) / 2;
            let reshaped = slice
                .to_owned()
                .into_shape_with_order((n_points, 2))
                .unwrap();
            unraveled.push(reshaped);
        }

        unraveled
    }

    /// Transform PCA deformation parameters to point coordinates
    ///
    /// # Arguments
    /// * `deformation` - PCA parameter vector
    ///
    /// # Returns
    /// * Result containing flattened points or ModelError
    pub fn inverse_transform(&self, deformation: ArrayView1<f64>) -> Array1<f64> {
        if deformation.len() == self.pca_components.nrows() {
            // tfed = mean + deformation @ scaled_components
            return &self.pca_mean + deformation.dot(&self.pca_scaled_components);
        }
        let deformation = if deformation.len() < self.pca_components.nrows() {
            // Handle case where deformation has fewer dimensions than components
            log::debug!(
                "Deformation dimension {} is less than PCA components dimension {}, padding with zeros",
                deformation.len(),
                self.pca_components.nrows()
            );
            let mut full_deformation = Array1::zeros(self.pca_components.nrows());
            full_deformation
                .slice_mut(s![..deformation.len()])
                .assign(&deformation);
            full_deformation
        } else {
            // truncate
            log::debug!(
                "Deformation dimension {} is greater than PCA components dimension {}, truncating",
                deformation.len(),
                self.pca_components.nrows()
            );
            let truncated_deformation = deformation.slice(s![..self.pca_components.nrows()]);
            truncated_deformation.to_owned()
        };
        &self.pca_mean + deformation.dot(&self.pca_scaled_components)
    }

    /// Apply deformation and return unraveled point groups
    ///
    /// # Arguments
    /// * `deformation` - PCA parameter vector
    ///
    /// # Returns
    /// * Result containing vector of 2D point arrays or ModelError
    pub fn deform(&self, deformation: ArrayView1<f64>) -> Vec<Array2<f64>> {
        let transformed = self.inverse_transform(deformation);
        self.unravel(&transformed)
    }

    /// Transform point coordinates to PCA parameters
    ///
    /// # Arguments
    /// * `points` - Flattened 1D array of points
    ///
    /// # Returns
    /// * Result containing PCA parameters or ModelError
    pub fn transform(&self, points: &Array1<f64>) -> Result<Array1<f64>, ModelError> {
        if points.len() != self.pca_mean.len() {
            return Err(ModelError::PointsDimensionMismatch(
                points.len(),
                self.pca_mean.len(),
            ));
        }

        // (points - mean) @ components.T / sqrt(explained_variance)
        let centered = points - &self.pca_mean;
        let projected = centered.dot(&self.pca_components.t());
        let scaled = &projected / &self.explained_variance.mapv(|v| v.sqrt());

        Ok(scaled)
    }

    /// Transform deformed point groups back to PCA parameters
    ///
    /// # Arguments
    /// * `deformed_points` - Vector of 2D point arrays
    ///
    /// # Returns
    /// * Result containing PCA parameters or ModelError
    pub fn undo_deform(&self, deformed_points: &[Array2<f64>]) -> Result<Array1<f64>, ModelError> {
        // Concatenate all point arrays and flatten
        let mut flattened = Vec::new();
        for points in deformed_points {
            for row in points.outer_iter() {
                flattened.push(row[0]);
                flattened.push(row[1]);
            }
        }

        let points_array = Array1::from_vec(flattened);
        self.transform(&points_array)
    }

    /// Get the number of PCA components
    pub fn n_components(&self) -> usize {
        self.pca_components.nrows()
    }

    /// Get the dimensionality of the mean shape (number of coordinates)
    pub fn n_dims(&self) -> usize {
        self.pca_mean.len()
    }

    /// Get the number of landmark groups
    pub fn n_groups(&self) -> usize {
        self.point_counts.len()
    }

    /// Pre-apply global transform to the active shape model
    pub fn global_transform(&mut self, tr: &SimilarityTransform) {
        // mean
        let mean_2d = self
            .pca_mean
            .to_owned()
            .into_shape_with_order((self.pca_mean.len() / 2, 2))
            .unwrap();
        let transformed_mean: Array2<f64> = tr.transform(&mean_2d);
        self.pca_mean = transformed_mean
            .into_shape_with_order(self.pca_mean.len())
            .unwrap();
        // // eigenvectors
        for i in 0..self.pca_components.len_of(Axis(0)) {
            let ev_2d = self
                .pca_components
                .slice(s![i, ..])
                .to_owned()
                .into_shape_with_order((self.pca_components.len_of(Axis(1)) / 2, 2))
                .unwrap();
            let transformed_ev: Array2<f64> = tr.transform(&ev_2d);
            let ev_flat = transformed_ev
                .into_shape_with_order(self.pca_components.len_of(Axis(1)))
                .unwrap();
            self.pca_components.slice_mut(s![i, ..]).assign(&ev_flat);
        }
    }
}

// Struct to fascilitate deserialization of ActiveShapeModel
#[derive(Debug, Clone, Deserialize)]
struct ActiveShapeModelDe {
    // Use `Vec`s instead of `Array`s for deserialization
    pca_mean: Vec<f64>,
    pca_components: Vec<Vec<f64>>,
    explained_variance: Vec<f64>,
    explained_variance_ratio: Vec<f64>,
    template_points: Vec<Vec<f64>>,
    template_reference: Vec<Option<(f64, f64)>>,
    labels: Vec<String>,
    point_counts: Vec<usize>,
}

// Implement conversion from deserialized struct to ActiveShapeModel
impl From<ActiveShapeModelDe> for ActiveShapeModel {
    fn from(de: ActiveShapeModelDe) -> Self {
        let mean = Array1::from(de.pca_mean);
        let components = Array2::from_shape_vec(
            (de.pca_components.len(), de.pca_components[0].len()),
            de.pca_components.into_iter().flatten().collect(),
        )
        .unwrap();
        let explained_variance = Array1::from(de.explained_variance);
        let explained_variance_ratio = Array1::from(de.explained_variance_ratio);
        let template_points = Array2::from_shape_vec(
            (de.template_points.len(), de.template_points[0].len()),
            de.template_points.into_iter().flatten().collect(),
        )
        .unwrap();
        Self::new(ModelParams {
            mean,
            components,
            explained_variance,
            explained_variance_ratio,
            template_points,
            template_reference: de.template_reference,
            labels: de.labels,
            point_counts: de.point_counts,
        })
    }
}

impl<'de> Deserialize<'de> for ActiveShapeModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let de = ActiveShapeModelDe::deserialize(deserializer)?;
        Ok(Self::from(de))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_active_shape_model_creation() {
        let mean = array![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let components = array![
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        ];
        let explained_variance = array![1.0, 0.5];
        let explained_variance_ratio = array![0.67, 0.33];
        let template_points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let template_reference = vec![Some((0.0, 0.0)), Some((1.0, 0.0)), Some((0.0, 1.0))];
        let labels = vec!["p1".to_string(), "p2".to_string(), "p3".to_string()];
        let point_counts = vec![3];

        let model = ActiveShapeModel::new(ModelParams {
            mean,
            components,
            explained_variance,
            explained_variance_ratio,
            template_points,
            template_reference,
            labels,
            point_counts,
        });

        assert_eq!(model.n_components(), 2);
        assert_eq!(model.n_dims(), 6);
        assert_eq!(model.n_groups(), 1);
    }

    #[test]
    fn test_unravel() {
        let mean = array![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0];
        let components = array![[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let explained_variance = array![1.0];
        let explained_variance_ratio = array![1.0];
        let template_points = array![[0.0, 0.0], [1.0, 0.0]];
        let template_reference = vec![];
        let labels = vec![];
        let point_counts = vec![3, 1]; // First group has 3 points, second has 1

        let model = ActiveShapeModel::new(ModelParams {
            mean,
            components,
            explained_variance,
            explained_variance_ratio,
            template_points,
            template_reference,
            labels,
            point_counts,
        });

        let points = array![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0];
        let unraveled = model.unravel(&points);

        assert_eq!(unraveled.len(), 2);
        assert_eq!(unraveled[0].shape(), &[3, 2]);
        assert_eq!(unraveled[1].shape(), &[1, 2]);
    }

    #[test]
    fn test_inverse_transform() {
        let mean = array![0.0, 0.0, 1.0, 0.0];
        let components = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let explained_variance = array![1.0, 1.0];
        let explained_variance_ratio = array![0.5, 0.5];
        let template_points = array![[0.0, 0.0], [1.0, 0.0]];
        let template_reference = vec![];
        let labels = vec![];
        let point_counts = vec![2];

        let model = ActiveShapeModel::new(ModelParams {
            mean,
            components,
            explained_variance,
            explained_variance_ratio,
            template_points,
            template_reference,
            labels,
            point_counts,
        });

        let deformation = array![1.0, 0.0];
        let result = model.inverse_transform(deformation.view());

        // Expected: mean + [1.0, 0.0] @ [[1.0, 0, 0, 0], [0, 1.0, 0, 0]]
        // = [0, 0, 1, 0] + [1, 0, 0, 0] = [1, 0, 1, 0]
        assert_eq!(result, array![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_transform_and_inverse() {
        let mean = array![0.0, 0.0, 1.0, 0.0];
        let components = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let explained_variance = array![1.0, 1.0];
        let explained_variance_ratio = array![0.5, 0.5];
        let template_points = array![[0.0, 0.0], [1.0, 0.0]];
        let template_reference = vec![];
        let labels = vec![];
        let point_counts = vec![2];

        let model = ActiveShapeModel::new(ModelParams {
            mean,
            components,
            explained_variance,
            explained_variance_ratio,
            template_points,
            template_reference,
            labels,
            point_counts,
        });

        let points = array![1.0, 0.0, 1.0, 0.0];
        let params = model.transform(&points).unwrap();
        let reconstructed = model.inverse_transform(params.view());

        // Should approximately reconstruct the original points
        for (a, b) in reconstructed.iter().zip(points.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
