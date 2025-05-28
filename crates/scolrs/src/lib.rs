use clap::ValueEnum;
use draw::MeasureError;
use head_neck::TryConvertContentFilename;
use labelme_rs::{LabelMeData, LabelMeDataLine, LabelMeDataWImage};
use lenke::{LumbarModifier, MajorCurve};
use log::{debug, error, warn};
pub use named_derive::{ContentFilename, HasImageMetadata};
use ndarray::{
    concatenate, s, stack, Array, Array1, Array2, Array3, ArrayBase, ArrayView1, ArrayView2,
    ArrayView3, Axis, Data,
};
use ndarray_stats::QuantileExt;
pub use serde;
use serde::{Deserialize, Serialize};
use std::cmp::Ord;
use std::convert::Infallible;
use std::fmt::Display;
use std::iter::zip;
use std::ops::AddAssign;
use std::path::Path;
use std::result::Result;
use strum::VariantArray;
use thiserror::Error;
mod defs;
pub use defs::*;
pub mod draw;
pub mod head_neck;
pub mod implant;
pub mod lenke;
use head_neck::Scale2DPoints;
use shadow_rs::shadow;

shadow!(build);
pub const VERSION: &str = shadow_rs::concatcp!(
    build::PKG_VERSION,
    " ",
    build::SHORT_COMMIT,
    " ",
    build::BUILD_TIME,
    if build::GIT_CLEAN { "" } else { " (dirty)" }
);

pub type Point2d = (f64, f64);

/// Trait for converting between different content filename types
pub trait ContentFilename {
    type ContentType;
    fn content_filename(self) -> (Self::ContentType, String);
    fn filename(&self) -> &str;
    fn new(content: Self::ContentType, filename: String) -> Self;
}

#[derive(Error, Debug)]
pub enum ScolError {
    #[error("Invalid point count for shape {0}: {1} != {2}")]
    InvalidShape(String, usize, usize),
    #[error("Invalid point count for {0}: {1}")]
    InvalidPointCount(String, usize),
    #[error("Invalid combination of the numbers of points for {0} and {1}: {2} vs. {3}")]
    InvalidPointCombo(String, String, usize, usize),
    #[error("Linalg error")]
    Linalg(#[from] rulinalg::error::Error),
    #[error("Json error")]
    Json(#[from] serde_json::Error),
    #[error("Array shape error")]
    ArrayShape(#[from] ndarray::ShapeError),
    #[error("Value error")]
    Value(String),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ImageMetadata {
    pub path: String,
    pub height: usize,
    pub width: usize,
    pub spacing_xy: (f64, f64),
    pub unit: String,
}

pub trait HasImageMetadata {
    fn image_metadata(&self) -> &ImageMetadata;
    fn image_metadata_mut(&mut self) -> &mut ImageMetadata;
}

/// Marker type for scaled types
pub struct ScaledType<T: Scalable>(pub T);

/// Scale the points by internal spacing data
pub trait Scalable: HasImageMetadata {
    type Error;
    /// Scale the points by the spacing data
    ///
    /// If the spacing is already 1.0, do nothing
    fn _scale(&mut self) -> Result<(), Self::Error> {
        if self.image_metadata().spacing_xy == (1.0, 1.0) {
            return Ok(());
        }
        self._impl_scale()
    }

    /// Implement the scaling
    /// Do not call this directly
    fn _impl_scale(&mut self) -> Result<(), Self::Error>;

    fn into_scaled(self) -> Result<ScaledType<Self>, Self::Error>
    where
        Self: Sized,
    {
        let mut this = self;
        this._scale()?;
        Ok(ScaledType(this))
    }
}

impl Default for ImageMetadata {
    fn default() -> Self {
        let spacing_xy = (1.0, 1.0);
        let unit = "px".to_string();
        Self {
            path: String::default(),
            height: 0,
            width: 0,
            spacing_xy,
            unit,
        }
    }
}

impl From<LabelMeData> for ImageMetadata {
    fn from(data: LabelMeData) -> Self {
        let path = data.imagePath.clone();
        let height = data.imageHeight;
        let width = data.imageWidth;
        Self {
            path,
            height,
            width,
            ..Default::default()
        }
    }
}

#[derive(Error, Debug)]
pub enum DicomError {
    #[error("Read error")]
    Read(#[from] dicom_object::ReadError),
    #[error("Tag not found: {0}")]
    MissingTag(String),
    #[error("Convert error: {0}")]
    Convert(String),
}

pub fn get_pixel_spacing(
    obj: &dicom_object::DefaultDicomObject,
) -> Result<Option<(f64, f64)>, DicomError> {
    use dicom_dictionary_std::tags;
    let spacing = obj.get(tags::PIXEL_SPACING);
    if let Some(spacing) = spacing {
        let spacing = spacing
            .to_multi_float64()
            .map_err(|e| DicomError::Convert(e.to_string()))?;
        return Ok(Some((spacing[0], spacing[1])));
    } else {
        let spacing = obj.get(tags::IMAGER_PIXEL_SPACING);
        if let Some(spacing) = spacing {
            let spacing = spacing
                .to_multi_float64()
                .map_err(|e| DicomError::Convert(e.to_string()))?;
            warn!("Using Imager Pixel Spacing (0018,1164) instead of Pixel Spacing (0028,0030)");
            return Ok(Some((spacing[0], spacing[1])));
        }
    }
    Ok(None)
}

impl TryFrom<&Path> for ImageMetadata {
    type Error = DicomError;

    fn try_from(path: &Path) -> Result<Self, DicomError> {
        use dicom_dictionary_std::tags;
        let obj = dicom_object::open_file(path)?;
        let spacing = get_pixel_spacing(&obj)?;
        let (spacing_xy, unit) = if let Some(spacing) = spacing {
            (spacing, "mm".to_string())
        } else {
            warn!("Spacing not found. Using default spacing (1.0, 1.0)");
            ((1.0, 1.0), "px".to_string())
        };
        let height = obj
            .get(tags::ROWS)
            .ok_or(DicomError::MissingTag("Rows (0028,0010)".to_string()))?
            .to_int()
            .map_err(|e| DicomError::Convert(e.to_string()))?;
        let width = obj
            .get(tags::COLUMNS)
            .ok_or(DicomError::MissingTag("Columns (0028,0011)".to_string()))?
            .to_int()
            .map_err(|e| DicomError::Convert(e.to_string()))?;
        Ok(Self {
            width,
            height,
            path: path.to_string_lossy().to_string(),
            spacing_xy,
            unit,
        })
    }
}

pub trait PullImageMetadata {
    fn pull_image_metadata(&mut self) -> Result<(), DicomError>;
}

impl<T> PullImageMetadata for T
where
    T: HasImageMetadata,
{
    fn pull_image_metadata(&mut self) -> Result<(), DicomError> {
        let metadata = self.image_metadata_mut();
        if metadata.path.ends_with(".dcm")
            || metadata.path.ends_with(".DCM")
            || metadata.path.ends_with(".dicom")
            || metadata.path.ends_with(".DICOM")
        {
            let dicom_file = dicom_object::OpenFileOptions::new()
                .read_until(dicom_dictionary_std::tags::PIXEL_DATA)
                .open_file(&metadata.path)?;
            if let Some(spacing) = get_pixel_spacing(&dicom_file)? {
                metadata.spacing_xy = spacing;
                metadata.unit = "mm".to_string();
            } else {
                warn!("No pixel spacing found in dicom: {:?}", metadata.path);
            }
        } else {
            warn!("No dicom: {:?}", metadata.path);
        }
        Ok(())
    }
}

fn extract_points(data: &LabelMeData, label: &str) -> Result<Array2<f64>, ScolError> {
    let tuples: Result<Vec<_>, _> = data
        .shapes
        .iter()
        .filter_map(|shape| {
            if shape.shape_type == "point" && shape.label == label {
                Some(&shape.points)
            } else {
                None
            }
        })
        .map(|points| {
            if points.len() != 1 {
                return Err(ScolError::InvalidShape(label.to_string(), 1, points.len()));
            }
            Ok(points[0])
        })
        .collect();
    let tuples = tuples?;
    let mut vec = Vec::with_capacity(tuples.len() * 2);
    for t in tuples.iter() {
        vec.push(t.0);
        vec.push(t.1);
    }
    let arr = Array2::from_shape_vec((tuples.len(), 2), vec).unwrap();
    Ok(arr)
}

#[derive(Debug, Clone, PartialEq)]
pub struct AtMost2<T>(pub T);

trait ValidateLength {
    fn validate_length(&self, expected_len: usize) -> Result<(), draw::InvalidNumberOfPoints>;
    fn validate_length_more_than(&self, min_len: usize) -> Result<(), draw::InvalidNumberOfPoints>;
    fn validate_label_length(&self, label: &str, expected_len: usize) -> Result<(), MeasureError>;
    fn validate_label_length_more_than(
        &self,
        label: &str,
        min_len: usize,
    ) -> Result<(), MeasureError>;
}

impl<S, I> ValidateLength for ndarray::ArrayBase<S, I>
where
    S: ndarray::Data<Elem = f64>,
    I: ndarray::Dimension,
{
    fn validate_length(&self, expected_len: usize) -> Result<(), draw::InvalidNumberOfPoints> {
        if self.len_of(Axis(0)) != expected_len {
            return Err(draw::InvalidNumberOfPoints::IncorrectNumberOfPoints(
                expected_len,
                self.len_of(Axis(0)),
            ));
        }
        Ok(())
    }
    fn validate_length_more_than(&self, min_len: usize) -> Result<(), draw::InvalidNumberOfPoints> {
        if self.len_of(Axis(0)) < min_len {
            return Err(draw::InvalidNumberOfPoints::TooFewPoints(
                min_len,
                self.len_of(Axis(0)),
            ));
        }
        Ok(())
    }

    fn validate_label_length(&self, label: &str, expected_len: usize) -> Result<(), MeasureError> {
        self.validate_length(expected_len)
            .map_err(|e| MeasureError::InvalidNumberOfPoints(label.to_string(), e))
    }

    fn validate_label_length_more_than(
        &self,
        label: &str,
        min_len: usize,
    ) -> Result<(), MeasureError> {
        self.validate_length_more_than(min_len)
            .map_err(|e| MeasureError::InvalidNumberOfPoints(label.to_string(), e))
    }
}

#[allow(dead_code)] // TODO: maybe remove?
trait LeftFirst {
    /// sort array by x
    fn left_first(self) -> Self;
}

impl LeftFirst for Array2<f64> {
    fn left_first(mut self) -> Self {
        if self.len_of(Axis(0)) == 2 && self[[0, 0]] > self[[1, 0]] {
            self.swap([0, 0], [1, 0]);
            self.swap([0, 1], [1, 1]);
        }
        if self.len_of(Axis(0)) > 2 {
            error!("Array length is greater than 2");
        }
        self
    }
}

trait HasCornerPoints {
    fn top_left(&self) -> ArrayView2<f64>;
    fn top_right(&self) -> ArrayView2<f64>;
    fn bottom_left(&self) -> ArrayView2<f64>;
    fn bottom_right(&self) -> ArrayView2<f64>;
}

fn vec_points_to_array2(nested_vec: &[Point2d]) -> Result<Array2<f64>, ndarray::ShapeError> {
    if nested_vec.is_empty() {
        return Ok(Array::zeros((0, 0)));
    }
    let flattened = nested_vec.iter().flat_map(|p| vec![p.0, p.1]).collect();
    Array::from_shape_vec((nested_vec.len(), 2), flattened)
}

fn nested_vec_to_array3(nested_vec: &[Vec<Point2d>]) -> Result<Array3<f64>, ndarray::ShapeError> {
    if nested_vec.is_empty() {
        return Ok(Array::zeros((0, 0, 0)));
    }
    if nested_vec[0].is_empty() {
        return Ok(Array::zeros((nested_vec.len(), 0, 0)));
    }
    let shape = (nested_vec.len(), nested_vec[0].len(), 2);
    let flattened = nested_vec
        .concat()
        .iter()
        .flat_map(|p| vec![p.0, p.1])
        .collect();
    Array::from_shape_vec(shape, flattened)
}

fn array2_to_vec_points(array: Array2<f64>) -> Vec<Point2d> {
    array.axis_iter(Axis(0)).map(|a| (a[0], a[1])).collect()
}

fn array3_to_nested_vec(array: Array3<f64>) -> Vec<Vec<Point2d>> {
    array
        .axis_iter(Axis(0))
        .map(|a| array2_to_vec_points(a.to_owned()))
        .collect()
}

/// Polynomial fitting of `deg` degrees
pub fn polyfit<S>(
    xs: ndarray::ArrayBase<S, ndarray::Ix1>,
    ys: ndarray::ArrayBase<S, ndarray::Ix1>,
    deg: usize,
) -> Result<ndarray::Array1<f64>, ScolError>
where
    S: ndarray::Data<Elem = f64>,
{
    let mut vander = Array2::zeros([xs.len(), deg + 1]);

    for d in 0..=deg {
        vander
            .slice_mut(s![.., d])
            .assign(&xs.mapv(|x| x.powi(d as i32)));
    }
    use rulinalg::matrix::{BaseMatrix, Matrix};
    use rulinalg::vector::Vector;

    let vander = Matrix::new(vander.nrows(), vander.ncols(), vander.into_raw_vec());
    let ys = Vector::new(ys.to_owned().into_raw_vec());
    let a = vander.transpose() * &vander;

    let b = &vander.transpose() * &ys;
    Ok(a.solve(b)
        .map(|c| ndarray::Array1::from_iter(c).mapv(|e| e))?)
}

/// Calculate polynomial curve points
pub fn polynomial<S, T>(
    xs: ndarray::ArrayBase<S, ndarray::Ix1>,
    coef: ndarray::ArrayBase<T, ndarray::Ix1>,
) -> Array1<f64>
where
    S: ndarray::Data<Elem = f64>,
    T: ndarray::Data<Elem = f64>,
{
    let mut ys: Array1<f64> = ndarray::Array::zeros(xs.len());
    for (i, c) in coef.iter().enumerate() {
        ys.add_assign(&xs.mapv(|x| c * x.powi(i as i32)));
    }
    ys
}

/// Point sets representing a spine
#[derive(Debug, Clone, PartialEq)]
pub struct Spine {
    pub c7tls: C7TLS,

    /// Corner points of C7, thoracic, and lumbar vertebrae
    /// The number of points/vertebrae can vary because some spine have 4 or 6 lumbar vertebrae
    pub v_c7tl: VertebraeC7TL,
    pub c_c7tl: Centroids,
}

impl Spine {
    /// Corner points of T1 to Sacral vertebrae
    pub fn t1_to_sac_vertebrae(&self) -> ArrayView3<f64> {
        self.v_c7tl.0.slice(s![1.., .., ..])
    }

    pub fn t1_to_sac_centroids(&self) -> ArrayView2<f64> {
        self.c_c7tl.slice(s![1.., ..])
    }

    pub fn scale(&mut self, scale_xy: ArrayView1<f64>) {
        self.c7tls.0.scale(scale_xy);
        self.v_c7tl.0.scale(scale_xy);
        self.c_c7tl.scale(scale_xy);
    }

    pub fn fit_poly(&self) -> Result<Array1<f64>, ScolError> {
        polyfit(
            self.c_c7tl.slice(s![1.., 1]),
            self.c_c7tl.slice(s![1.., 0]),
            6,
        )
    }

    /// Rotate the spine so that the spine is vertical
    ///
    /// The rotation is done by rotating the spine so that the line connecting the top and bottom
    /// centroids of the spine is vertical.
    pub fn verticalize(&mut self) {
        let centroids = self.c_c7tl.clone();
        let top = centroids.index_axis(Axis(0), 0);
        let bottom = centroids.index_axis(Axis(0), centroids.len_of(Axis(0)) - 1);
        let v = &bottom - &top;
        if v[0].abs() < 1e-6 {
            debug!("Spine is already vertical");
            return;
        }
        let angle = v[0].atan2(v[1]);
        let rot_mat = ndarray::arr2(&[[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]]);

        self.c7tls.0 = offsetted_rotate_array3(self.c7tls.0.view(), top, rot_mat.view());
        self.v_c7tl.0 = offsetted_rotate_array3(self.v_c7tl.0.view(), top, rot_mat.view());

        self.c_c7tl = offsetted_rotate_array2(self.c_c7tl.view(), top, rot_mat.view());
    }
}

/// Point sets extracted from a coronal radiograph
#[derive(Debug, Clone, PartialEq, HasImageMetadata)]
pub struct CoronalPoints {
    pub spine: Spine,
    pub clavicle: AtMost2<Array2<f64>>,
    pub shoulder: AtMost2<Array2<f64>>,
    pub pelvis: AtMost2<Array2<f64>>,
    pub femoral_head: AtMost2<Array2<f64>>,
    /// Coefficients of the polynomial curve of the spine
    pub c_coefs: Array1<f64>,

    pub image_metadata: ImageMetadata,
}

/// Intermediary representation for [CoronalPoints]
#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq, HasImageMetadata)]
struct CoronalPointsIR {
    pub spine: Vec<Vec<Point2d>>,
    pub clavicle: Vec<Point2d>,
    pub shoulder: Vec<Point2d>,
    pub pelvis: Vec<Point2d>,
    pub femoral_head: Vec<Point2d>,

    pub image_metadata: ImageMetadata,
}

impl Serialize for CoronalPoints {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        CoronalPointsIR::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CoronalPoints {
    fn deserialize<D>(deserializer: D) -> Result<CoronalPoints, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let ir = CoronalPointsIR::deserialize(deserializer)?;
        CoronalPoints::try_from(ir).map_err(serde::de::Error::custom)
    }
}

impl Scalable for CoronalPoints {
    type Error = ScolError;
    /// Scale all points by `scale_xy` and refit the polynomial curve
    fn _impl_scale(&mut self) -> Result<(), ScolError> {
        let scale_xy = ndarray::array![
            self.image_metadata.spacing_xy.0,
            self.image_metadata.spacing_xy.1
        ];

        self.spine.scale(scale_xy.view());
        self.clavicle.0.scale(scale_xy.view());
        self.shoulder.0.scale(scale_xy.view());
        self.pelvis.0.scale(scale_xy.view());
        self.femoral_head.0.scale(scale_xy.view());

        self.c_coefs = self.spine.fit_poly()?;
        Ok(())
    }
}

/// [`CoronalPoints`] in [`ContentFilename`] struct
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ContentFilename)]
pub struct CoronalPointsLine {
    pub content: CoronalPoints,
    pub filename: String,
}

impl TryFrom<LabelMeDataLine> for CoronalPointsLine {
    type Error = <CoronalPointsLine as TryConvertContentFilename<LabelMeDataLine>>::Error;

    fn try_from(data: LabelMeDataLine) -> Result<Self, Self::Error> {
        CoronalPointsLine::try_convert_from(data)
    }
}

type CornerPoints = (
    Vec<(f64, f64)>,
    Vec<(f64, f64)>,
    Vec<(f64, f64)>,
    Vec<(f64, f64)>,
);

fn split_corner_points(spine: &Vec<Vec<Point2d>>) -> CornerPoints {
    let (mut tl, mut tr, mut bl, mut br) = (
        Vec::with_capacity(spine.len()),
        Vec::with_capacity(spine.len()),
        Vec::with_capacity(spine.len()),
        Vec::with_capacity(spine.len()),
    );
    for points in spine {
        tl.push(points[0]);
        tr.push(points[1]);
        bl.push(points[2]);
        br.push(points[3]);
    }
    // remove last points from bl and br
    bl.pop();
    br.pop();
    (tl, tr, bl, br)
}

fn create_shapes(label_points: &[(&str, Vec<Point2d>)]) -> Vec<labelme_rs::Shape> {
    let mut shapes: Vec<labelme_rs::Shape> = Vec::new();
    for (label, points) in label_points {
        for point in points {
            let shape = labelme_rs::Shape {
                label: label.to_string(),
                points: vec![*point],
                shape_type: "point".to_string(),
                ..Default::default()
            };
            shapes.push(shape);
        }
    }
    shapes
}

impl From<CoronalPoints> for LabelMeData {
    fn from(cp: CoronalPoints) -> Self {
        let ir = CoronalPointsIR::from(&cp);
        LabelMeData::from(ir)
    }
}

impl From<CoronalPointsIR> for LabelMeData {
    fn from(ir: CoronalPointsIR) -> Self {
        let mut data = LabelMeData {
            imagePath: ir.image_metadata.path,
            imageHeight: ir.image_metadata.height,
            imageWidth: ir.image_metadata.width,
            ..Default::default()
        };

        let (tl, tr, bl, br) = split_corner_points(&ir.spine);

        data.shapes = create_shapes(&[
            ("TL", tl),
            ("TR", tr),
            ("BL", bl),
            ("BR", br),
            ("Clavicle", ir.clavicle),
            ("Shoulder", ir.shoulder),
            ("Pelvis", ir.pelvis),
            ("FemoralHead", ir.femoral_head),
        ]);
        data
    }
}

impl TryFrom<CoronalPointsIR> for CoronalPoints {
    type Error = ScolError;

    fn try_from(ir: CoronalPointsIR) -> Result<Self, Self::Error> {
        let image_data = ir.image_metadata.clone();
        let data = LabelMeData::from(ir);
        let mut cp = CoronalPoints::try_from(data)?;
        cp.image_metadata = image_data;
        Ok(cp)
    }
}

impl From<&CoronalPoints> for CoronalPointsIR {
    fn from(cp: &CoronalPoints) -> Self {
        let spine = array3_to_nested_vec(cp.spine.c7tls.0.clone());
        let clavicle = array2_to_vec_points(cp.clavicle.0.clone());
        let shoulder = array2_to_vec_points(cp.shoulder.0.clone());
        let pelvis = array2_to_vec_points(cp.pelvis.0.clone());
        let femoral_head = array2_to_vec_points(cp.femoral_head.0.clone());
        Self {
            spine,
            clavicle,
            shoulder,
            pelvis,
            femoral_head,
            image_metadata: cp.image_metadata.clone(),
        }
    }
}

const MAX_NUM_VERTS_IN_CURVE: usize = 10;

fn offsetted_rotate_array2(
    points: ArrayView2<f64>,
    offset: ArrayView1<f64>,
    rot_mat: ArrayView2<f64>,
) -> Array2<f64> {
    let offsetted = &points - &offset;
    let rotated = offsetted.dot(&rot_mat.t());
    rotated + offset
}

fn offsetted_rotate_array3(
    points: ArrayView3<f64>,
    offset: ArrayView1<f64>,
    rot_mat: ArrayView2<f64>,
) -> Array3<f64> {
    let offsetted = &points - &offset;
    // reshape Array3 to Array2
    let shape = points.shape();
    let offsetted = offsetted
        .as_standard_layout()
        .into_owned()
        .into_shape((shape[0] * shape[1], 2))
        .unwrap();
    let rotated = offsetted.dot(&rot_mat.t());

    rotated.into_shape((shape[0], shape[1], 2)).unwrap() + offset
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CurveScore {
    range: f64,
    apex: f64,
    angle: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct CurveScoreSet {
    pt: Option<CurveScore>,
    mt: Option<CurveScore>,
    tll: Option<CurveScore>,
}

#[derive(Debug, Clone, Copy)]
pub struct CurveWeights {
    pt: CurveScore,
    mt: CurveScore,
    tll: CurveScore,
}

impl Default for CurveWeights {
    fn default() -> Self {
        Self {
            pt: CurveScore {
                range: 9.4,
                apex: 0.0,
                angle: 4.2,
            },
            mt: CurveScore {
                range: 7.3,
                apex: 0.7,
                angle: 0.35,
            },
            tll: CurveScore {
                range: 9.2,
                apex: 1.7,
                angle: 3.9,
            },
        }
    }
}

impl From<[f64; 9]> for CurveWeights {
    fn from(weights: [f64; 9]) -> Self {
        Self {
            pt: CurveScore {
                range: weights[0],
                apex: weights[1],
                angle: weights[2],
            },
            mt: CurveScore {
                range: weights[3],
                apex: weights[4],
                angle: weights[5],
            },
            tll: CurveScore {
                range: weights[6],
                apex: weights[7],
                angle: weights[8],
            },
        }
    }
}

impl CurveScoreSet {
    fn total(&self, weights: &CurveWeights) -> i64 {
        let mut total = 0.0;
        if let Some(pt) = self.pt {
            total += pt.range * weights.pt.range;
            total += pt.apex * weights.pt.apex;
            total += pt.angle * weights.pt.angle;
        }
        if let Some(mt) = self.mt {
            total += mt.range * weights.mt.range;
            total += mt.apex * weights.mt.apex;
            total += mt.angle * weights.mt.angle;
        }
        if let Some(tll) = self.tll {
            total += tll.range * weights.tll.range;
            total += tll.apex * weights.tll.apex;
            total += tll.angle * weights.tll.angle;
        }
        (total * 100.0).round() as i64
    }
}

pub enum CurveSetAlgorithm {
    Score(CurveWeights),
    Apex,
}

/// Assess the curve set
///
/// For mt curve, add number of thoracic vertebrae and deduct number of lumbar vertebrae
/// For tll curve, deduct number of thoracic vertebrae and add number of lumbar vertebrae
/// For pt curve, add one point if the curve is above T12
fn assess_curve_set(curve_set: &CurveSet, apex_set: ApexSet) -> CurveScoreSet {
    let mut curve_score_set = CurveScoreSet::default();
    if let Some((curve, angle)) = &curve_set.pt {
        let mut range_score = 0.0;
        for i in curve.sup..=curve.inf {
            let vertebra = VertebralIndex::from(i as u8);
            if vertebra <= VertebralIndex::T3 {
                range_score += 1.0;
            }
            if vertebra > VertebralIndex::T5 {
                range_score -= 1.0;
            }
        }
        let apex = apex_set.pt.unwrap() as i8;
        let apex = (apex + (curve.inf + curve.sup) as i8 / 2) / 2;
        let apex_score = -(VertebraDiscIndex::T2 as i8 - apex) as f64;

        let angle_score = angle.abs();
        curve_score_set.pt = Some(CurveScore {
            range: range_score,
            apex: apex_score,
            angle: angle_score,
        });
    };
    if let Some((curve, angle)) = &curve_set.mt {
        let mut range_score = 0.0;
        for i in curve.sup..=curve.inf {
            let vertebra = VertebralIndex::from(i as u8);
            // add number of thoracic vertebrae
            if vertebra <= VertebralIndex::T12 {
                range_score += 1.0;
            }
            // deduct number of non-thoracolumbar vertebrae
            if vertebra >= VertebralIndex::L2 {
                range_score -= 1.0;
            }
        }
        let apex = apex_set.mt.unwrap() as i8;
        let apex = (apex + (curve.inf + curve.sup) as i8 / 2) / 2;
        let apex_score = (-(VertebraDiscIndex::T9 as i8 - apex) as f64).max(-5.0);
        let angle_score = if let Some((_curve, pt_angle)) = &curve_set.pt {
            (pt_angle - angle).abs()
        } else {
            angle.abs()
        };
        curve_score_set.mt = Some(CurveScore {
            range: range_score,
            apex: apex_score,
            angle: angle_score,
        });
    };
    if let Some((curve, angle)) = &curve_set.tll {
        let mut range_score = 0.0;
        for i in curve.sup..=curve.inf {
            let vertebra = VertebralIndex::from(i as u8);
            // add number of thoracolumbar/lumbar vertebrae
            if vertebra >= VertebralIndex::T10 {
                range_score += 1.0;
            }
            // deduct number of non-thoracolumbar vertebrae
            if vertebra <= VertebralIndex::T9 {
                range_score -= 1.0;
            }
        }
        let apex = apex_set.tll.unwrap() as i8;
        let apex = (apex + (curve.inf + curve.sup) as i8 / 2) / 2;
        let apex_score = (-(VertebraDiscIndex::L2 as i8 - apex) as f64).max(-5.0);
        let angle_score = if let Some((_curve, mt_angle)) = &curve_set.mt {
            (mt_angle - angle).abs()
        } else {
            angle.abs()
        };
        curve_score_set.tll = Some(CurveScore {
            range: range_score,
            apex: apex_score,
            angle: angle_score,
        });
    };
    curve_score_set
}

impl CoronalPoints {
    pub fn spinal_poly(&self, xs: ArrayView1<f64>) -> Array1<f64> {
        polynomial(xs, self.c_coefs.view())
    }

    pub fn verticalize_spine(&mut self) -> Result<(), ScolError> {
        self.spine.verticalize();
        self.c_coefs = self.spine.fit_poly()?;
        Ok(())
    }

    fn find_largest_curve(&self) -> Option<(Curve, f64)> {
        let n = self.spine.tl_corners().0.len_of(ndarray::Axis(0));
        let mut curves = Vec::new();
        for sup in 0..n - 2 {
            for inf in sup..n {
                if inf - sup > MAX_NUM_VERTS_IN_CURVE {
                    break;
                }
                if self.is_valid_curve(sup, inf) {
                    curves.push(Curve { sup, inf });
                }
            }
        }
        self._find_largest_curve(curves)
    }

    fn find_largest_up(&self, inf: usize) -> Option<(Curve, f64)> {
        let mut curves = Vec::new();
        if inf <= 1 {
            return None;
        }
        let end = inf - 1;
        // search sup from bottom to top (by .rev()) so that we can break early from the loop
        for sup in (0..end).rev() {
            if inf - sup > MAX_NUM_VERTS_IN_CURVE {
                break;
            }
            if self.is_valid_curve(sup, inf) {
                curves.push(Curve { sup, inf });
            }
        }
        self._find_largest_curve(curves)
    }

    fn find_largest_down(&self, sup: usize) -> Option<(Curve, f64)> {
        let n = self.spine.tl_corners().0.len_of(ndarray::Axis(0));
        let mut curves = Vec::new();
        let start = (sup + 2).min(n);
        for inf in start..n {
            if inf - sup > MAX_NUM_VERTS_IN_CURVE {
                break;
            }
            if self.is_valid_curve(sup, inf) {
                curves.push(Curve { sup, inf });
            }
        }
        self._find_largest_curve(curves)
    }

    /// Curve is invalid when it contains S curve, i.e. checking if the curve is convex
    fn is_valid_curve(&self, sup: usize, inf: usize) -> bool {
        // inf - sup <= 2
        if inf <= sup + 2 {
            return true;
        }
        let centroids = self.spine.tl_centroids();
        let sub_centroids = centroids.slice(s![sup..=inf, ..]);
        let top = sub_centroids.index_axis(Axis(0), 0);
        let bottom = sub_centroids.index_axis(Axis(0), sub_centroids.len_of(Axis(0)) - 1);
        // check if all points are on the same side of the line
        let top2bottom = &bottom - &top;
        let cross_products = sub_centroids.map_axis(Axis(1), |p| {
            let v = &p - &top;
            top2bottom[0] * v[1] - top2bottom[1] * v[0]
        });
        cross_products.have_same_signs()
    }

    fn _find_largest_curve(&self, curves: Vec<Curve>) -> Option<(Curve, f64)> {
        let angles: Vec<_> = curves
            .iter()
            .filter_map(|c| self.spine.angle(c).map(|a| (c, a)))
            .collect();
        if angles.is_empty() {
            return None;
        }
        let (i_max, _max_value) = angles.iter().map(|e| e.1.abs()).enumerate().fold(
            (0, angles[0].1),
            |(i_max, max_value), (i, value)| {
                if value > max_value {
                    (i, value)
                } else {
                    (i_max, max_value)
                }
            },
        );
        Some((angles[i_max].0.clone(), angles[i_max].1))
    }

    /// Identify the apex of the curve
    pub fn id_apex(&self, curve: &Curve) -> VertebraDiscIndex {
        let vert_disc_corners = self.spine.tl_vert_disc_corners();
        let vd_centroids: Centroids = vert_disc_corners.into();
        let sup = VertebraDiscIndex::from(VertebralIndex::from(curve.sup as u8)) as usize;
        let inf = VertebraDiscIndex::from(VertebralIndex::from(curve.inf as u8)) as usize;
        let xs = polynomial(vd_centroids.slice(s![sup..=inf, 1]), self.c_coefs.view());
        if xs.is_monotonic() {
            VertebraDiscIndex::from((sup + inf) as u8 / 2)
        } else {
            let ts = (2.0 * &xs - xs[0] - xs[xs.len() - 1]).mapv(|e| e.abs());
            let i_max = ts.argmax().unwrap();
            VertebraDiscIndex::from((i_max + sup) as u8)
        }
    }

    /// Find all curves going down from the given sup
    ///
    /// The returned curves are sorted from the highest to the lowest
    fn find_all_down(&self, mut sup: usize) -> Vec<(Curve, f64)> {
        let mut curves = Vec::new();
        while let Some(largest_curve) = self.find_largest_down(sup) {
            sup = largest_curve.0.inf;
            curves.push(largest_curve);
        }
        curves
    }

    /// Find all curves going up from the given inf
    ///
    /// The returned curves are sorted from the lowest to the highest
    fn find_all_up(&self, mut inf: usize) -> Vec<(Curve, f64)> {
        let mut curves = Vec::new();
        while let Some(largest_curve) = self.find_largest_up(inf) {
            inf = largest_curve.0.sup;
            curves.push(largest_curve);
        }
        curves
    }

    pub fn find_all_curves(&self) -> Vec<(Curve, f64)> {
        if let Some(largest_curve) = self.find_largest_curve() {
            let mut downs = self.find_all_down(largest_curve.0.inf);
            let ups = self.find_all_up(largest_curve.0.sup);
            downs.extend(vec![largest_curve]);
            downs.extend(ups);
            downs
        } else {
            Vec::default()
        }
    }

    pub fn find_all_curve_sets(&self) -> Vec<CurveSet> {
        if let Some(largest_curve) = self.find_largest_curve() {
            let ups = self.find_all_up(largest_curve.0.sup);
            let downs = self.find_all_down(largest_curve.0.inf);
            // create vec of last two elements of downs, and largest_curve and first two elements of ups
            let mut curves_vec: Vec<Vec<_>> = Vec::with_capacity(5);
            if ups.len() >= 2 {
                curves_vec.push(vec![ups[1].clone(), ups[0].clone(), largest_curve.clone()]);
            }
            if !ups.is_empty() {
                curves_vec.push(vec![ups[0].clone(), largest_curve.clone()]);
                if !downs.is_empty() {
                    curves_vec.push(vec![
                        ups[0].clone(),
                        largest_curve.clone(),
                        downs[0].clone(),
                    ]);
                }
            }
            if downs.len() >= 2 {
                curves_vec.push(vec![
                    largest_curve.clone(),
                    downs[0].clone(),
                    downs[1].clone(),
                ]);
            }
            if !downs.is_empty() {
                curves_vec.push(vec![largest_curve.clone(), downs[0].clone()]);
            }
            let mut curveset_candidates = Vec::new();
            for curves in curves_vec {
                if curves.len() == 3 {
                    curveset_candidates.push(CurveSet {
                        pt: Some(curves[0].clone()),
                        mt: Some(curves[1].clone()),
                        tll: Some(curves[2].clone()),
                    });
                } else {
                    curveset_candidates.push(CurveSet {
                        pt: None,
                        mt: Some(curves[0].clone()),
                        tll: Some(curves[1].clone()),
                    });
                    curveset_candidates.push(CurveSet {
                        pt: Some(curves[0].clone()),
                        mt: Some(curves[1].clone()),
                        tll: None,
                    });
                }
            }
            curveset_candidates
        } else {
            Vec::default()
        }
    }

    /// Identify the curves of the spine
    ///
    /// The curves is optimized by the [assess_curve_set] function
    fn identify_curves_score(&self, weights: &CurveWeights) -> CurveDesc {
        let mut curves = CurveSet::default();
        let mut major_curve = None;
        let curveset_candidates = self.find_all_curve_sets();

        if !curveset_candidates.is_empty() {
            let points = curveset_candidates
                .iter()
                .map(|cs| {
                    let apex_set = cs.apices(self);
                    (cs, assess_curve_set(cs, apex_set).total(weights))
                })
                .collect::<Vec<_>>();
            let (curve_set, _) = points.iter().max_by_key(|(_, points)| *points).unwrap();
            curves = (*curve_set).clone();
            major_curve = curves.major_curve();
        }
        let apices = curves.apices(self);
        CurveDesc::new(curves, apices, major_curve)
    }

    fn identify_curves_apex(&self) -> CurveDesc {
        let mut curves = CurveSet::default();
        let mut major_curve = None;
        if let Some(largest_curve) = self.find_largest_curve() {
            let major_apex = self.id_apex(&largest_curve.0);
            major_curve = if major_apex <= VertebraDiscIndex::T5 {
                // largest curve is PT
                debug!("PT is the largest curve");
                if let Some(mt) = self.find_largest_down(largest_curve.0.inf) {
                    curves.tll = self.find_largest_down(mt.0.inf);
                    curves.mt = Some(mt);
                }
                curves.pt = Some(largest_curve);
                Some(MajorCurve::MT) // PT is never major
            } else if major_apex <= VertebraDiscIndex::DiscT11T12 {
                // largest curve is MT
                debug!("MT is the largest curve");
                curves.pt = self.find_largest_up(largest_curve.0.sup);
                curves.tll = self.find_largest_down(largest_curve.0.inf);
                curves.mt = Some(largest_curve);
                Some(MajorCurve::MT)
            } else {
                // largest curve is TLL
                debug!("TLL is the largest curve");
                if let Some(mt) = self.find_largest_up(largest_curve.0.sup) {
                    curves.pt = self.find_largest_up(mt.0.sup);
                    curves.mt = Some(mt);
                }
                curves.tll = Some(largest_curve);
                Some(MajorCurve::TLL)
            };
        }
        let apices = curves.apices(self);
        CurveDesc::new(curves, apices, major_curve)
    }

    pub fn identify_curves_with_algorithm(&self, algorithm: &CurveSetAlgorithm) -> CurveDesc {
        let mut cloned = self.clone();
        cloned.verticalize_spine().unwrap();

        match algorithm {
            CurveSetAlgorithm::Score(weights) => cloned.identify_curves_score(weights),
            CurveSetAlgorithm::Apex => cloned.identify_curves_apex(),
        }
    }

    pub fn identify_curves(&self) -> CurveDesc {
        let algorithm = CurveSetAlgorithm::Score(CurveWeights::default());
        self.identify_curves_with_algorithm(&algorithm)
    }

    pub fn lumbar_modifier(&self, apex: VertebraDiscIndex) -> LumbarModifier {
        let x_scvl = self.spine.sacral_center()[0];
        // vertebral index -> vertebral index or disc index -> vertebral index right above the disc
        let index = (apex as u8 / 2) as usize;
        let vertebra = self.spine.v_c7tl.0.index_axis(Axis(0), index + 1);
        let v_xs = vertebra.index_axis(Axis(1), 0);
        let x_min = v_xs.min().unwrap();
        let x_max = v_xs.max().unwrap();
        if x_scvl < *x_min || *x_max < x_scvl {
            LumbarModifier::C
        } else {
            // TODO: implement pedicle checking
            LumbarModifier::AorB
        }
    }
}

/// Point sets extracted from a sagittal radiograph
#[derive(Debug, Clone, PartialEq, HasImageMetadata)]
pub struct SagittalPoints {
    pub spine: Spine,
    pub femoral_head: AtMost2<Array2<f64>>,

    pub image_metadata: ImageMetadata,
}

impl Serialize for SagittalPoints {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        SagittalPointsIR::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SagittalPoints {
    fn deserialize<D>(deserializer: D) -> Result<SagittalPoints, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let ir = SagittalPointsIR::deserialize(deserializer)?;
        SagittalPoints::try_from(ir).map_err(serde::de::Error::custom)
    }
}

impl Scalable for SagittalPoints {
    type Error = Infallible;
    fn _impl_scale(&mut self) -> Result<(), Self::Error> {
        let scale_xy = ndarray::array![
            self.image_metadata.spacing_xy.0,
            self.image_metadata.spacing_xy.1
        ];

        self.spine.scale(scale_xy.view());
        self.femoral_head.0.scale(scale_xy.view());
        Ok(())
    }
}

/// Intermediate representation of [`SagittalPoints`] for serde
#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq, HasImageMetadata)]
struct SagittalPointsIR {
    pub spine: Vec<Vec<Point2d>>,
    pub femoral_head: Vec<Point2d>,

    pub image_metadata: ImageMetadata,
}

impl From<SagittalPointsIR> for LabelMeData {
    fn from(ir: SagittalPointsIR) -> Self {
        let mut data = LabelMeData {
            imagePath: ir.image_metadata.path,
            imageHeight: ir.image_metadata.height,
            imageWidth: ir.image_metadata.width,
            ..Default::default()
        };

        let (tl, tr, bl, br) = split_corner_points(&ir.spine);

        data.shapes = create_shapes(&[
            ("TL", tl),
            ("TR", tr),
            ("BL", bl),
            ("BR", br),
            ("FemoralHead", ir.femoral_head),
        ]);
        data
    }
}

impl From<SagittalPoints> for LabelMeData {
    fn from(sp: SagittalPoints) -> Self {
        let ir = SagittalPointsIR::from(&sp);
        LabelMeData::from(ir)
    }
}

impl From<&SagittalPoints> for SagittalPointsIR {
    fn from(cp: &SagittalPoints) -> Self {
        let spine = array3_to_nested_vec(cp.spine.c7tls.0.clone());
        let femoral_head = array2_to_vec_points(cp.femoral_head.0.clone());
        Self {
            spine,
            femoral_head,
            image_metadata: cp.image_metadata.clone(),
        }
    }
}

impl TryFrom<LabelMeData> for SagittalPointsIR {
    type Error = ScolError;

    fn try_from(data: LabelMeData) -> Result<Self, Self::Error> {
        let coronal_points = SagittalPoints::try_from(data)?;
        let coronal_points_ir = SagittalPointsIR::from(&coronal_points);
        Ok(coronal_points_ir)
    }
}

impl TryFrom<SagittalPointsIR> for SagittalPoints {
    type Error = ScolError;

    fn try_from(ir: SagittalPointsIR) -> Result<Self, Self::Error> {
        let image_data = ir.image_metadata.clone();
        let data = LabelMeData::from(ir);
        let mut sp = SagittalPoints::try_from(data)?;
        sp.image_metadata = image_data;
        Ok(sp)
    }
}

/// [`SagittalPoints`] in [`ContentFilename`] struct
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, ContentFilename)]
pub struct SagittalPointsLine {
    pub content: SagittalPoints,
    pub filename: String,
}

impl TryFrom<LabelMeDataLine> for SagittalPointsLine {
    type Error = <SagittalPointsLine as TryConvertContentFilename<LabelMeDataLine>>::Error;

    fn try_from(data: LabelMeDataLine) -> Result<Self, Self::Error> {
        SagittalPointsLine::try_convert_from(data)
    }
}

/// Spinal curve represented by superior and inferior indices of vertebrae
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Curve {
    pub sup: usize,
    pub inf: usize,
}

impl Display for Curve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sup = VertebralIndex::from(self.sup as u8);
        let inf = VertebralIndex::from(self.inf as u8);
        write!(f, "{sup}{inf}")
    }
}

/// Set of PT, MT, and TLL curves
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "UPPERCASE")]
pub struct CurveSet {
    pub pt: Option<(Curve, f64)>,
    pub mt: Option<(Curve, f64)>,
    pub tll: Option<(Curve, f64)>,
}

/// [`CurveSet`] with optional angles used for [`CoronalPointsAndCurveIR`]
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "UPPERCASE")]
pub struct CurveSetOptionalAngles {
    pub pt: Option<(Curve, Option<f64>)>,
    pub mt: Option<(Curve, Option<f64>)>,
    pub tll: Option<(Curve, Option<f64>)>,
}

impl CurveSet {
    fn apices(&self, coronal_points: &CoronalPoints) -> ApexSet {
        ApexSet {
            pt: self.pt.as_ref().map(|(c, _)| coronal_points.id_apex(c)),
            mt: self.mt.as_ref().map(|(c, _)| coronal_points.id_apex(c)),
            tll: self.tll.as_ref().map(|(c, _)| coronal_points.id_apex(c)),
        }
    }

    /// Determine the major curve
    ///
    /// The major curve is determined by the curve with the largest absolute angle
    /// If PT is the largest, then the major curve is MT
    fn major_curve(&self) -> Option<MajorCurve> {
        if self.pt.is_none() && self.mt.is_none() && self.tll.is_none() {
            return None;
        }
        // absolute angles
        let pt = self.pt.as_ref().map(|(_, a)| a.abs()).unwrap_or(0.0);
        let mt = self.mt.as_ref().map(|(_, a)| a.abs()).unwrap_or(0.0);
        let tll = self.tll.as_ref().map(|(_, a)| a.abs()).unwrap_or(0.0);
        if (pt >= mt && pt >= tll) || (mt >= pt && mt >= tll) {
            Some(MajorCurve::MT)
        } else {
            Some(MajorCurve::TLL)
        }
    }

    pub fn score(&self, coronal_points: &CoronalPoints) -> CurveScoreSet {
        assess_curve_set(self, self.apices(coronal_points))
    }
}

/// Set of PT, MT, and TLL apices
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "UPPERCASE")]
pub struct ApexSet {
    pub pt: Option<VertebraDiscIndex>,
    pub mt: Option<VertebraDiscIndex>,
    pub tll: Option<VertebraDiscIndex>,
}

/// Descriptor for curves of scoliosis consisting of curves, apices, and major-curve-kind (MT or TLL)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CurveDesc {
    pub curves: CurveSet,
    pub apices: ApexSet,
    pub major_curve: Option<MajorCurve>,
}

impl CurveDesc {
    pub fn new(curves: CurveSet, apices: ApexSet, major_curve: Option<MajorCurve>) -> Self {
        Self {
            curves,
            apices,
            major_curve,
        }
    }
}

/// [`CurveDesc`] with optional angles used for [`CoronalPointsAndCurveIR`]
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CurveDescOptionalAngles {
    pub curves: CurveSetOptionalAngles,
    pub apices: ApexSet,
    pub major_curve: Option<MajorCurve>,
}

impl CurveDescOptionalAngles {
    pub fn resolve_angles(self, spine: &Spine) -> CurveDesc {
        let curves = CurveSet {
            pt: self.curves.pt.map(|(c, a)| {
                let angle = a.unwrap_or_else(|| spine.angle(&c).unwrap());
                (c.clone(), angle)
            }),
            mt: self.curves.mt.map(|(c, a)| {
                let angle = a.unwrap_or_else(|| spine.angle(&c).unwrap());
                (c, angle)
            }),
            tll: self.curves.tll.map(|(c, a)| {
                let angle = a.unwrap_or_else(|| spine.angle(&c).unwrap());
                (c, angle)
            }),
        };
        CurveDesc::new(curves, self.apices, self.major_curve)
    }
}

impl TryFrom<(&LabelMeData, &CurveSetAlgorithm)> for CurveDesc {
    type Error = ScolError;

    fn try_from(
        (data, algorithm): (&LabelMeData, &CurveSetAlgorithm),
    ) -> Result<Self, Self::Error> {
        let coronal_points = CoronalPoints::try_from(data.clone())?;
        Ok(coronal_points.identify_curves_with_algorithm(algorithm))
    }
}

/// [`CurveDesc`] in [`ContentFilename`] struct
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, ContentFilename)]
pub struct ScolDescLine {
    pub content: CurveDesc,
    pub filename: String,
}

/// Data required for calculating scoliosis drawing
#[derive(Debug, Clone, Serialize)]
pub struct CoronalPointsAndCurve {
    #[serde(flatten)]
    pub coronal_points: CoronalPoints,
    pub curves: CurveDesc,
}

impl<'de> Deserialize<'de> for CoronalPointsAndCurve {
    fn deserialize<D>(deserializer: D) -> Result<CoronalPointsAndCurve, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let ir: CoronalPointsAndCurveIR = CoronalPointsAndCurveIR::deserialize(deserializer)?;
        if let Some(curves) = ir.curves {
            log::debug!("Using provided curves");
            let curves = curves.resolve_angles(&ir.coronal_points.spine);
            Ok(CoronalPointsAndCurve::new(ir.coronal_points, curves))
        } else {
            log::debug!("Identifying curves from coronal points");
            let curves = ir.coronal_points.identify_curves();
            Ok(CoronalPointsAndCurve::new(ir.coronal_points, curves))
        }
    }
}

/// Data required for calculating scoliosis drawing
#[derive(Debug, Clone, Deserialize)]
pub struct CoronalPointsAndCurveIR {
    #[serde(flatten)]
    pub coronal_points: CoronalPoints,
    #[serde(flatten)]
    pub curves: Option<CurveDescOptionalAngles>,
}

/// Maintain redundant point data in sync with scaling and resizing
pub struct PointDataWithImage<T> {
    pub data: T,
    pub data_image: LabelMeDataWImage,
}

impl<T> PointDataWithImage<T> {
    pub fn new(data: T, data_image: LabelMeDataWImage) -> Self {
        Self { data, data_image }
    }
}

/// [`CoronalPointsAndCurve`] in [`ContentFilename`] struct
#[derive(Clone, Debug, Serialize, Deserialize, ContentFilename)]
pub struct CoronalPointsAndCurveLine {
    pub content: CoronalPointsAndCurve,
    pub filename: String,
}

impl TryFrom<LabelMeDataLine> for CoronalPointsAndCurveLine {
    type Error = <CoronalPointsAndCurveLine as TryConvertContentFilename<LabelMeDataLine>>::Error;

    fn try_from(data: LabelMeDataLine) -> Result<Self, Self::Error> {
        let converted = TryConvertContentFilename::try_convert_from(data)?;
        Ok(converted)
    }
}

impl CoronalPointsAndCurve {
    pub fn new(coronal_points: CoronalPoints, curves: CurveDesc) -> Self {
        Self {
            coronal_points,
            curves,
        }
    }

    // pub fn update_curve(&mut self) {
    //     self.curves = self.coronal_points.identify_curves();
    // }
}

impl Scalable for CoronalPointsAndCurve {
    type Error = ScolError;
    fn _impl_scale(&mut self) -> Result<(), Self::Error> {
        self.coronal_points._impl_scale()
    }
}

impl HasImageMetadata for CoronalPointsAndCurve {
    fn image_metadata(&self) -> &ImageMetadata {
        &self.coronal_points.image_metadata
    }
    fn image_metadata_mut(&mut self) -> &mut ImageMetadata {
        &mut self.coronal_points.image_metadata
    }
}

impl TryFrom<&LabelMeData> for CoronalPointsAndCurve {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let coronal_points = CoronalPoints::try_from(data.clone())?;
        let curves = coronal_points.identify_curves();
        Ok(CoronalPointsAndCurve::new(coronal_points, curves))
    }
}

impl TryFrom<LabelMeData> for CoronalPointsAndCurve {
    type Error = ScolError;

    fn try_from(data: LabelMeData) -> Result<Self, Self::Error> {
        CoronalPointsAndCurve::try_from(&data)
    }
}

impl From<CoronalPointsAndCurve> for LabelMeData {
    fn from(cp: CoronalPointsAndCurve) -> Self {
        LabelMeData::from(cp.coronal_points)
    }
}

/// Trait that enables `arr.l2norm()` to a vector (Array1 or ArrayView1)
pub trait L2Norm<T> {
    fn l2norm(&self) -> T;
}

impl<S> L2Norm<f64> for ndarray::ArrayBase<S, ndarray::Ix1>
where
    S: ndarray::Data<Elem = f64>,
{
    fn l2norm(&self) -> f64 {
        self.mapv(|a| a * a).sum().sqrt()
    }
}

/// Check monotonicity of arrays
pub trait LineCharacteristic<T> {
    /// Check if the array is not-strictly increasing or decreasing
    fn is_monotonic(&self) -> bool;

    /// Check if the array has the same sign
    fn have_same_signs(&self) -> bool;
}

impl<S> LineCharacteristic<f64> for ndarray::ArrayBase<S, ndarray::Ix1>
where
    S: ndarray::Data<Elem = f64>,
{
    fn is_monotonic(&self) -> bool {
        if self.len() < 2 {
            return true;
        }

        let mut signs = Vec::new();
        for i in 1..self.len() {
            signs.push(if self[i] > self[i - 1] {
                1
            } else if self[i] < self[i - 1] {
                -1
            } else {
                0
            });
        }

        let first_sign = signs[0];
        signs.iter().all(|&sign| sign == first_sign)
    }

    fn have_same_signs(&self) -> bool {
        let mut non_zeros = self.iter().filter(|&&x| x != 0.0);
        if let Some(first) = non_zeros.clone().next() {
            non_zeros.all(|&x| x.signum() == first.signum())
        } else {
            true
        }
    }
}

/// Angle between two lines in degrees
/// TODO: Check the difference from `angle_between`?
pub fn angle_from_lines(line1: ArrayView2<f64>, line2: ArrayView2<f64>) -> Option<f64> {
    let v_sup = &line1.index_axis(Axis(0), 1) - &line1.index_axis(Axis(0), 0);
    let v_inf = &line2.index_axis(Axis(0), 1) - &line2.index_axis(Axis(0), 0);
    let len_sup = v_sup.l2norm();
    let len_inf = v_inf.l2norm();
    if len_sup == 0.0 || len_inf == 0.0 {
        return None;
    }
    let v_sup = &v_sup / len_sup;
    let v_inf = &v_inf / len_inf;
    let cos = v_sup.dot(&v_inf);
    let cos = cos.clamp(-1.0, 1.0);
    let deg = cos.acos().to_degrees();
    if v_sup[1] < v_inf[1] {
        Some(-deg)
    } else {
        Some(deg)
    }
}

impl Spine {
    /// Corner points of thoracic and lumbar vertebrae
    pub fn tl_corners(&self) -> Corners<ndarray::ViewRepr<&f64>> {
        let tl = self.v_c7tl.0.slice(s![1.., .., ..]);
        Corners(tl)
    }

    pub fn tl_vert_disc_corners(&self) -> Corners<ndarray::OwnedRepr<f64>> {
        let vert_corners = self.tl_corners();
        let disc_corners = vert_corners.between();
        let mut vert_disc_corners: Array3<f64> = ndarray::Array::zeros((
            vert_corners.0.len_of(Axis(0)) + disc_corners.len_of(Axis(0)),
            4,
            2,
        ));
        for (i, vc) in vert_corners.0.axis_iter(Axis(0)).enumerate() {
            vert_disc_corners.slice_mut(s![2 * i, .., ..]).assign(&vc);
        }
        for (i, vc) in disc_corners.axis_iter(Axis(0)).enumerate() {
            vert_disc_corners
                .slice_mut(s![2 * i + 1, .., ..])
                .assign(&vc);
        }
        Corners(vert_disc_corners)
    }
    pub fn tl_sup_lines(&self) -> ArrayView3<'_, f64> {
        self.v_c7tl.0.slice(s![1.., ..2, ..])
    }
    pub fn tl_inf_lines(&self) -> ArrayView3<'_, f64> {
        self.v_c7tl.0.slice(s![1.., 2.., ..])
    }
    pub fn tl_centroids(&self) -> ArrayView2<'_, f64> {
        self.c_c7tl.slice(s![1.., ..])
    }

    pub fn sup_plate(&self, index: usize) -> ArrayView2<'_, f64> {
        self.c7tls.0.slice(s![index + 1, ..2, ..])
    }
    pub fn inf_plate(&self, index: usize) -> ArrayView2<'_, f64> {
        self.c7tls.0.slice(s![index + 1, 2.., ..])
    }

    pub fn sacral_sup_plate(&self) -> ArrayView2<f64> {
        self.c7tls
            .0
            .slice(s![self.c7tls.0.len_of(Axis(0)) - 1, 0..2, ..])
    }

    // mean point of sacral TL and TR
    pub fn sacral_center(&self) -> Array1<f64> {
        self.sacral_sup_plate().mean_axis(Axis(0)).unwrap()
    }

    /// Calculate Cobb angle in degrees
    pub fn angle(&self, curve: &Curve) -> Option<f64> {
        let sup_line = self.sup_plate(curve.sup);
        let inf_line = self.inf_plate(curve.inf);
        angle_from_lines(sup_line, inf_line)
    }
}

impl TryFrom<&LabelMeData> for Spine {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let c7tls = C7TLS::try_from(data)?;
        let v_c7tl = VertebraeC7TL::try_from(data)?;
        let c_c7tl: Centroids = Corners(v_c7tl.0.view()).into();

        Ok(Spine {
            c7tls,
            v_c7tl,
            c_c7tl,
        })
    }
}

fn _new_at_most2(data: &LabelMeData, label: &str) -> Result<AtMost2<Array2<f64>>, ScolError> {
    let points = extract_points(data, label)?;
    if points.len_of(Axis(0)) > 2 {
        return Err(ScolError::InvalidPointCount(
            label.to_string(),
            points.len_of(Axis(0)),
        ));
    }
    Ok(AtMost2(points))
}

impl TryFrom<LabelMeData> for CoronalPoints {
    type Error = ScolError;

    fn try_from(data: LabelMeData) -> Result<Self, Self::Error> {
        let spine = Spine::try_from(&data)?;
        // TODO: accept invalid number of points and let later functions handle it
        let clavicle = _new_at_most2(&data, "Clavicle")?;
        let shoulder = _new_at_most2(&data, "Shoulder")?;
        let pelvis = _new_at_most2(&data, "Pelvis")?;
        let femoral_head = _new_at_most2(&data, "FemoralHead")?;

        let c_coefs = spine.fit_poly()?;

        let image_data = ImageMetadata::from(data);

        Ok(CoronalPoints {
            spine,
            clavicle,
            pelvis,
            shoulder,
            femoral_head,
            c_coefs,
            image_metadata: image_data,
        })
    }
}

impl TryFrom<&LabelMeData> for CoronalPoints {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        CoronalPoints::try_from(data.clone())
    }
}

impl TryFrom<LabelMeData> for SagittalPoints {
    type Error = ScolError;

    fn try_from(data: LabelMeData) -> Result<Self, Self::Error> {
        let spine = Spine::try_from(&data)?;
        let femoral_head = _new_at_most2(&data, "FemoralHead")?;

        let image_data = ImageMetadata::from(data);

        Ok(SagittalPoints {
            spine,
            femoral_head,
            image_metadata: image_data,
        })
    }
}

impl TryFrom<&LabelMeData> for SagittalPoints {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        SagittalPoints::try_from(data.clone())
    }
}

/// Corner points of all C7, thoracic, and lumbar vertebrae and sacrum top plate.
///
/// Note: sacrum corners = (TL, TR, copy of TL, copy of TR)
#[derive(Debug, Clone, PartialEq)]
pub struct C7TLS(pub Array3<f64>);

/// C7, thoracic and lumber vertebrae
#[derive(Debug, Clone, PartialEq)]
pub struct VertebraeC7TL(pub Array3<f64>);

/// Thoracic and lumber vertebrae
pub struct VertebraeTL<'a>(pub ArrayView3<'a, f64>);

impl<'a> From<&'a VertebraeC7TL> for VertebraeTL<'a> {
    fn from(c7tl: &'a VertebraeC7TL) -> Self {
        let tl = c7tl.0.slice(s![1.., .., ..]);
        Self(tl)
    }
}

/// Corner points of structures.
///
/// Points are in [tl, tr, bl, br] order
pub struct Corners<S: Data<Elem = f64>>(pub ArrayBase<S, ndarray::Ix3>);

impl<S: Data<Elem = f64>> Corners<S> {
    pub fn between(&self) -> Array3<f64> {
        let bottom = self.0.slice(s![1.., ..2, ..]);
        let top = self.0.slice(s![..(self.0.shape()[0] - 1), 2.., ..]);
        let between = ndarray::concatenate![Axis(1), top, bottom];
        between
    }
}

type Centroids = Array2<f64>;
impl<S: Data<Elem = f64>> From<Corners<S>> for Centroids {
    /// Calculate centroids from list of four corners.
    /// Centroids are not geometric centers but the intersections of mid-lines
    fn from(corners: Corners<S>) -> Self {
        let top = corners.0.slice(s![.., ..2, ..]).mean_axis(Axis(1)).unwrap();
        let bottom = corners.0.slice(s![.., 2.., ..]).mean_axis(Axis(1)).unwrap();
        let left = corners
            .0
            .slice(s![.., 0..;2, ..])
            .mean_axis(Axis(1))
            .unwrap();
        let mut right = corners
            .0
            .slice(s![.., 1..;2, ..])
            .mean_axis(Axis(1))
            .unwrap();
        // reuse `right` to store centroids
        for (t, (b, (l, mut r))) in zip(
            top.axis_iter(Axis(0)),
            zip(
                bottom.axis_iter(Axis(0)),
                zip(left.axis_iter(Axis(0)), right.axis_iter_mut(Axis(0))),
            ),
        ) {
            // [python - Numpy and line intersections - Stack Overflow](https://stackoverflow.com/questions/3252194/numpy-and-line-intersections/57821199#57821199)
            let (a1, a2) = (&t, &b);
            let (b1, b2) = (&l, &r);
            let da = a2 - a1;
            let db = b2 - b1;
            let dp = a1 - b1;
            let mut dap = da.clone();
            dap[[0]] = -da[[1]];
            dap[[1]] = da[[0]];
            let denom = dap.dot(&db);
            if denom == 0.0 {
                warn!("Parallel mid-lines of a vertebrae thus no centroid. Using center of gravity instead.");
                let c = (&t + &b + l + &r) / 4.0;
                r.assign(&c);
            } else {
                let num = dap.dot(&dp);
                let c = num / denom * db + b1;
                r.assign(&c);
            }
        }
        right
    }
}

impl TryFrom<&LabelMeData> for C7TLS {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let mut corners = CORNER_LABELS
            .iter()
            .map(|label| extract_points(data, label))
            .collect::<Result<Vec<_>, _>>()?;
        if corners[0].shape()[0] != corners[1].shape()[0] {
            return Err(ScolError::InvalidPointCombo(
                "TL".into(),
                "TR".into(),
                corners[0].shape()[0],
                corners[1].shape()[0],
            ));
        }
        if corners[2].shape()[0] != corners[3].shape()[0] {
            return Err(ScolError::InvalidPointCombo(
                "BL".into(),
                "BR".into(),
                corners[2].shape()[0],
                corners[3].shape()[0],
            ));
        }
        if corners[0].shape()[0] == 0 {
            return Err(ScolError::InvalidPointCount(
                "TL".into(),
                corners[0].shape()[0],
            ));
        }
        if corners[0].shape()[0] - 1 != corners[2].shape()[0] {
            return Err(ScolError::InvalidPointCombo(
                "TL-1".into(),
                "BL".into(),
                corners[0].shape()[0] - 1,
                corners[2].shape()[0],
            ));
        }

        // Add the last point of TL to BL and TR to BR
        let last = corners[0]
            .index_axis(Axis(0), corners[0].len_of(Axis(0)) - 1)
            .insert_axis(Axis(0));
        corners[2] = concatenate(Axis(0), &[corners[2].view(), last]).unwrap();
        let last = corners[1]
            .index_axis(Axis(0), corners[1].len_of(Axis(0)) - 1)
            .insert_axis(Axis(0));
        corners[3] = concatenate(Axis(0), &[corners[3].view(), last]).unwrap();

        let verts = stack![Axis(1), corners[0], corners[1], corners[2], corners[3]];
        Ok(C7TLS(verts))
    }
}

impl TryFrom<&LabelMeData> for VertebraeC7TL {
    type Error = ScolError;

    /// From [C7[TL, TR, BL, BR] - Sacral[TL, TR]] points
    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let vertebrae: C7TLS = data.try_into()?;
        let verts_c7_t_l = vertebrae
            .0
            .slice_axis(
                Axis(0),
                ndarray::Slice::from(0..vertebrae.0.len_of(Axis(0)) - 1),
            )
            .to_owned();
        Ok(VertebraeC7TL(verts_c7_t_l))
    }
}

#[derive(
    strum::EnumString,
    strum::Display,
    strum::VariantArray,
    ValueEnum,
    Serialize,
    Deserialize,
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
)]
#[serde(rename_all = "PascalCase")]
#[clap(rename_all = "PascalCase")]
pub enum CoronalDraw {
    VertebralLabels,
    VertebralPoints,
    Centroids,
    SpinalLine,
    CurveApex,
    CSVL,
    CobbPT,
    CobbMT,
    CobbTLL,
    T1TiltAngle,
    CoronalBalance,
    ClavicleAngle,
    ShoulderHeight,
    PelvicObliquity,
    SacralObliquity,
    LegLengthDiscrepancy,
}

#[derive(
    strum::EnumString,
    strum::Display,
    strum::VariantArray,
    ValueEnum,
    Serialize,
    Deserialize,
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
)]
#[serde(rename_all = "PascalCase")]
#[clap(rename_all = "PascalCase")]
pub enum CoronalMeasure {
    CobbPT,
    CobbMT,
    CobbTLL,
    T1TiltAngle,
    CoronalBalance,
    ClavicleAngle,
    ShoulderHeight,
    PelvicObliquity,
    SacralObliquity,
    LegLengthDiscrepancy,
}

#[derive(
    strum::EnumString,
    strum::Display,
    strum::VariantArray,
    ValueEnum,
    Serialize,
    Deserialize,
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
)]
#[serde(rename_all = "PascalCase")]
#[clap(rename_all = "PascalCase")]
pub enum ImplantDraw {
    VertebralLabels,
    Screws,
}

pub trait MeasureAndDraw {
    fn all() -> Vec<Self>
    where
        Self: std::marker::Sized;
}

impl<T> MeasureAndDraw for T
where
    T: ValueEnum + VariantArray,
{
    fn all() -> Vec<Self> {
        T::VARIANTS.to_vec()
    }
}

#[derive(
    strum::EnumString,
    strum::Display,
    strum::VariantArray,
    ValueEnum,
    Serialize,
    Deserialize,
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
)]
#[clap(rename_all = "PascalCase")]
#[strum(serialize_all = "PascalCase")]
pub enum SagittalDraw {
    VertebralLabels,
    VertebralPoints,
    ThoracicKyphosis,
    ThoracicKyphosisT1,
    ProximalThoracicKyphosis,
    MidLowerThoracicKyphosis,
    ThoracolumbarSagittalAlignment,
    LumbarLordosis,
    T1Slope,
    SagittalBalance,
    PelvicIncidence,
    PelvicTilt,
    SacralSlope,
    L5IncidenceAngle,
    PelvicRadiusAngle,
    LumbosacralAngle,
}

#[derive(
    strum::EnumString,
    strum::Display,
    strum::VariantArray,
    ValueEnum,
    Serialize,
    Deserialize,
    Debug,
    Copy,
    Clone,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
)]
#[clap(rename_all = "PascalCase")]
#[strum(serialize_all = "PascalCase")]
pub enum SagittalMeasure {
    ThoracicKyphosis,
    T1ThoracicKyphosis,
    ProximalThoracicKyphosis,
    MidLowerThoracicKyphosis,
    ThoracolumbarSagittalAlignment,
    LumbarLordosis,
    T1Slope,
    SagittalBalance,
    PelvicIncidence,
    PelvicTilt,
    SacralSlope,
    L5IncidenceAngle,
    PelvicRadiusAngle,
    LumbosacralAngle,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{Corners, LineCharacteristic};
    use anyhow::Result;
    use ndarray::{arr3, array, Array3};
    #[test]
    fn test_corners() -> Result<()> {
        let arr: Array3<f64> = arr3(&[
            [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]],
            [[1.0, 3.0], [2.0, 3.0], [1.0, 4.0], [2.0, 4.0]],
        ]);
        let corners = Corners(arr);
        let bet = corners.between();

        let expected: Array3<f64> =
            ndarray::arr3(&[[[1.0, 2.0], [2.0, 2.0], [1.0, 3.0], [2.0, 3.0]]]);
        assert_eq!(bet, expected);
        Ok(())
    }

    #[test]
    fn test_is_monotonic_increasing() {
        let arr = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(arr.is_monotonic());
    }

    #[test]
    fn test_is_monotonic_decreasing() {
        let arr = array![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!(arr.is_monotonic());
    }

    #[test]
    fn test_is_monotonic_constant() {
        let arr = array![2.0, 2.0, 2.0, 2.0, 2.0];
        assert!(arr.is_monotonic());
    }

    #[test]
    fn test_is_not_monotonic() {
        let arr = array![1.0, 2.0, 3.0, 2.0, 1.0];
        assert!(!arr.is_monotonic());
    }
}
