use ndarray::Array2;
use rulinalg::matrix::{BaseMatrix, Matrix};
use rulinalg::vector::Vector;
use std::fmt;
use thiserror::Error;

/// Errors that can occur during alignment operations
#[derive(Debug, Error, Clone)]
pub enum AlignmentError {
    #[error(
        "Source and destination must have same number of points, got source: {0}, destination: {1}"
    )]
    PointCountMismatch(usize, usize),

    #[error("Need at least {0} points to estimate similarity transform, got {1}")]
    InsufficientPoints(usize, usize),

    #[error("Failed to solve linear system: {0}")]
    LinearSystemSolverFailed(String),
}

/// Similarity transformation for 2D point transformations
/// Preserves angles and uniform scaling (rotation + uniform scale + translation)
#[derive(Debug, Clone)]
pub struct SimilarityTransform {
    /// 3x3 transformation matrix
    matrix: Matrix<f64>,
}

impl SimilarityTransform {
    /// Create a new identity similarity transform
    pub fn new() -> Self {
        Self {
            matrix: Matrix::identity(3),
        }
    }

    /// Estimate similarity transform from source to destination points
    /// Uses least squares to solve for 4 parameters: scale, rotation, tx, ty
    ///
    /// # Arguments
    /// * `src` - Source points as slice of (x, y) tuples
    /// * `dst` - Destination points as slice of (x, y) tuples
    ///
    /// # Returns
    /// * `Result<(), AlignmentError>` - Ok if successful, Err with error type otherwise
    pub fn estimate(
        &mut self,
        src: &[(f64, f64)],
        dst: &[(f64, f64)],
    ) -> Result<(), AlignmentError> {
        if src.len() != dst.len() {
            return Err(AlignmentError::PointCountMismatch(src.len(), dst.len()));
        }
        if src.len() < 2 {
            return Err(AlignmentError::InsufficientPoints(2, src.len()));
        }

        let n = src.len();

        // Build least squares system for similarity transform
        // Transform: x' = s*cos(θ)*x - s*sin(θ)*y + tx
        //           y' = s*sin(θ)*x + s*cos(θ)*y + ty
        // Let a = s*cos(θ), b = s*sin(θ)
        // Then: x' = a*x - b*y + tx
        //       y' = b*x + a*y + ty
        // Solve for [a, b, tx, ty]

        let mut a_data = Vec::with_capacity(n * 2 * 4);
        let mut b_data = Vec::with_capacity(n * 2);

        for i in 0..n {
            let (x, y) = src[i];
            let (u, v) = dst[i];

            // Row for x-coordinate: u = a*x - b*y + tx
            a_data.extend_from_slice(&[x, -y, 1.0, 0.0]);
            b_data.push(u);

            // Row for y-coordinate: v = b*x + a*y + ty
            a_data.extend_from_slice(&[y, x, 0.0, 1.0]);
            b_data.push(v);
        }

        let a = Matrix::new(n * 2, 4, a_data);
        let b = Vector::new(b_data);

        // Solve least squares: A^T * A * x = A^T * b
        let at = a.transpose();
        let ata = &at * &a;

        // Convert b to matrix for multiplication, then back to vector
        let b_mat = Matrix::new(b.size(), 1, b.data().to_vec());
        let atb_mat = &at * &b_mat;
        let atb = Vector::new(atb_mat.data().to_vec());

        // Solve the system using LU decomposition
        match ata.solve(atb) {
            Ok(params) => {
                // Build the 3x3 transformation matrix from [a, b, tx, ty]
                let p = params.data();
                let a = p[0];
                let b = p[1];
                let tx = p[2];
                let ty = p[3];

                self.matrix = Matrix::new(
                    3,
                    3,
                    vec![
                        a, -b, tx, // First row
                        b, a, ty, // Second row
                        0.0, 0.0, 1.0, // Third row (homogeneous)
                    ],
                );
                Ok(())
            }
            Err(e) => Err(AlignmentError::LinearSystemSolverFailed(e.to_string())),
        }
    }

    /// Transform points using the similarity transformation
    ///
    /// # Arguments
    /// * `points` - Nx2 array of points to transform
    ///
    /// # Returns
    /// * Transformed points as Nx2 array
    pub fn transform(&self, points: &Array2<f64>) -> Array2<f64> {
        let n_points = points.nrows();
        let mut result = Array2::zeros((n_points, 2));

        for i in 0..n_points {
            let x = points[[i, 0]];
            let y = points[[i, 1]];

            // Apply transformation: [x', y', 1]^T = M * [x, y, 1]^T
            let m = &self.matrix;
            let x_new = m[[0, 0]] * x + m[[0, 1]] * y + m[[0, 2]];
            let y_new = m[[1, 0]] * x + m[[1, 1]] * y + m[[1, 2]];

            result[[i, 0]] = x_new;
            result[[i, 1]] = y_new;
        }

        result
    }

    /// Get the transformation matrix
    pub fn matrix(&self) -> &Matrix<f64> {
        &self.matrix
    }
}

impl Default for SimilarityTransform {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SimilarityTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SimilarityTransform:\n{}", self.matrix)
    }
}

/// Structure representing movable points with reference points for alignment
#[derive(Debug, Clone)]
pub struct MovablePoints {
    /// All points in the shape (Nx2 array)
    pub points: Array2<f64>,
    /// Reference points used for calculating transform (potentially None if missing)
    pub reference_points: Vec<Option<(f64, f64)>>,
    /// Count of points per landmark group
    pub point_counts: Vec<usize>,
}

impl MovablePoints {
    /// Create a new MovablePoints instance
    ///
    /// # Arguments
    /// * `points` - All points in the shape (Nx2 array)
    /// * `reference_points` - Reference points for alignment (potentially None if missing)
    /// * `point_counts` - Count of points per landmark group
    pub fn new(
        points: Array2<f64>,
        reference_points: Vec<Option<(f64, f64)>>,
        point_counts: Vec<usize>,
    ) -> Self {
        // Log warning if any reference points are missing
        if reference_points.iter().any(|p| p.is_none()) {
            log::warn!("Some reference points are None");
        }

        Self {
            points,
            reference_points,
            point_counts,
        }
    }

    /// Calculate the transform to align src_points to self (template)
    ///
    /// # Arguments
    /// * `src_points` - Source points to align
    /// * `inverse` - If true, calculate inverse transform
    ///
    /// # Returns
    /// * Result containing the SimilarityTransform
    pub fn calculate_transform(
        &self,
        src_points: &MovablePoints,
        inverse: bool,
    ) -> Result<SimilarityTransform, AlignmentError> {
        let src: Vec<(f64, f64)> = src_points
            .reference_points
            .iter()
            .filter_map(|&p| p)
            .collect();
        let dst: Vec<(f64, f64)> = self.reference_points.iter().filter_map(|&p| p).collect();

        let mut tf = SimilarityTransform::new();
        if inverse {
            tf.estimate(&dst, &src)?;
        } else {
            tf.estimate(&src, &dst)?;
        }
        Ok(tf)
    }

    /// Calculate transform with missing reference points handled
    ///
    /// # Arguments
    /// * `src_ref_points` - Source reference points to align
    /// * `inverse` - If true, calculate inverse transform
    ///
    /// # Returns
    /// * Result containing the SimilarityTransform
    pub fn calculate_transform_with_missing(
        &self,
        src_ref_points: &[Option<(f64, f64)>],
        inverse: bool,
    ) -> Result<SimilarityTransform, AlignmentError> {
        // Find valid indices where both src and dst have non-None values
        let valid_indices: Vec<usize> = (0..src_ref_points.len())
            .filter(|&i| {
                i < self.reference_points.len()
                    && src_ref_points[i].is_some()
                    && self.reference_points[i].is_some()
            })
            .collect();

        if valid_indices.len() < 2 {
            return Err(AlignmentError::InsufficientPoints(2, valid_indices.len()));
        }

        let src: Vec<(f64, f64)> = valid_indices
            .iter()
            .filter_map(|&i| src_ref_points[i])
            .collect();
        let dst: Vec<(f64, f64)> = valid_indices
            .iter()
            .filter_map(|&i| self.reference_points[i])
            .collect();

        let mut tf = SimilarityTransform::new();
        if inverse {
            tf.estimate(&dst, &src)?;
        } else {
            tf.estimate(&src, &dst)?;
        }
        Ok(tf)
    }

    /// Align src_points to template (self) using calculated transform
    ///
    /// # Arguments
    /// * `src_points` - Source points to align
    ///
    /// # Returns
    /// * Tuple of (transformed_points, transform)
    pub fn align(
        &self,
        src_points: &MovablePoints,
    ) -> Result<(Array2<f64>, SimilarityTransform), AlignmentError> {
        let tf = self.calculate_transform(src_points, false)?;
        let transformed_points = tf.transform(&src_points.points);
        Ok((transformed_points, tf))
    }

    /// Align src_points to template (self) handling missing reference points
    ///
    /// # Arguments
    /// * `src_points` - Source points to align
    ///
    /// # Returns
    /// * Tuple of (transformed_points, transform)
    pub fn align_with_missing(
        &self,
        src_points: &MovablePoints,
    ) -> Result<(Array2<f64>, SimilarityTransform), AlignmentError> {
        let tf = self.calculate_transform_with_missing(&src_points.reference_points, false)?;
        let transformed_points = tf.transform(&src_points.points);
        Ok((transformed_points, tf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_similarity_transform_identity() {
        let tf = SimilarityTransform::new();
        let points = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let transformed = tf.transform(&points);

        // Identity transform should not change points
        assert!((transformed - &points).mapv(|x| x.abs()).sum() < 1e-10);
    }

    #[test]
    fn test_similarity_transform_estimation() {
        let src = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let dst = vec![(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]; // Translation by (1, 1)

        let mut tf = SimilarityTransform::new();
        assert!(tf.estimate(&src, &dst).is_ok());

        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let transformed = tf.transform(&points);

        // Check if transformation is approximately correct
        assert!((transformed[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((transformed[[0, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_movable_points_align() {
        let src_points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let src_ref = vec![Some((0.0, 0.0)), Some((1.0, 0.0)), Some((0.0, 1.0))];
        let src = MovablePoints::new(src_points, src_ref, vec![4]);

        let dst_ref = vec![Some((1.0, 1.0)), Some((2.0, 1.0)), Some((1.0, 2.0))];
        let dst_points = array![[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]];
        let dst = MovablePoints::new(dst_points, dst_ref, vec![4]);

        let result = dst.align(&src);
        assert!(result.is_ok());

        let (transformed, _tf) = result.unwrap();
        // First reference point should be transformed close to (1, 1)
        assert!((transformed[[0, 0]] - 1.0).abs() < 0.1);
        assert!((transformed[[0, 1]] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_movable_points_with_missing() {
        let src_points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let src_ref = vec![Some((0.0, 0.0)), None, Some((0.0, 1.0)), Some((1.0, 1.0))];
        let src = MovablePoints::new(src_points, src_ref, vec![4]);

        let dst_points = array![[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]];
        let dst_ref = vec![Some((1.0, 1.0)), None, Some((1.0, 2.0)), Some((2.0, 2.0))];
        let dst = MovablePoints::new(dst_points, dst_ref, vec![4]);

        let result = dst.align_with_missing(&src);
        assert!(result.is_ok());
    }

    #[test]
    fn test_not_enough_points() {
        // Test with only 1 point - should fail
        let src_points = array![[0.0, 0.0]];
        let src_ref = vec![Some((0.0, 0.0))];
        let src = MovablePoints::new(src_points, src_ref, vec![1]);

        let dst_points = array![[1.0, 1.0]];
        let dst_ref = vec![Some((1.0, 1.0))];
        let dst = MovablePoints::new(dst_points, dst_ref, vec![1]);

        let result = dst.align_with_missing(&src);
        assert!(result.is_err());
    }

    #[test]
    fn test_minimum_points_similarity() {
        // Test with 2 points - should succeed for similarity transform
        let src_points = array![[0.0, 0.0], [1.0, 0.0]];
        let src_ref = vec![Some((0.0, 0.0)), Some((1.0, 0.0))];
        let src = MovablePoints::new(src_points, src_ref, vec![2]);

        let dst_points = array![[1.0, 1.0], [2.0, 1.0]];
        let dst_ref = vec![Some((1.0, 1.0)), Some((2.0, 1.0))];
        let dst = MovablePoints::new(dst_points, dst_ref, vec![2]);

        let result = dst.align_with_missing(&src);
        assert!(result.is_ok());
    }
}
