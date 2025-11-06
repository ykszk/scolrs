use std::{convert::Infallible, ops::MulAssign};

use crate::{
    angle_from_lines, array2_to_vec_points, array3_to_nested_vec_points, create_shapes,
    draw::{
        angle_between, distanced_pair3, draw_incidence_angle, draw_tilt_angle,
        femoral_incidence_angle, points2line, tilt_angle, CobbAux, ColorPalette, DrawArguments,
        DrawComponent, DrawCorners, DrawError, MeasureError, Named, Painter, CLASS_ANGLE,
        CLASS_ANNOTATION, CLASS_DISTANCE, CLASS_LINE, CLASS_MEASURE, CLASS_POINT, CLASS_RATIO,
        CLASS_TEXT,
    },
    extract_points, nested_vec_to_array3, vec_points_to_array2, Centroids, ContentFilename,
    Corners, HasCornerPoints, HasImageMetadata, ImageMetadata, L2Norm, Point2d, Scalable,
    ScaledType, ScolError, ValidateLength, CORNER_LABELS,
};
use clap::{self, ValueEnum};
use lyon_geom::point;

use labelme_rs::{LabelMeData, LabelMeDataLine};
use ndarray::{concatenate, s, stack, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use ndarray_stats::DeviationExt;
use serde::{Deserialize, Serialize};
use strum::VariantArray;
use svg::node::element;

/// Vertebral corner points in the order: TL, TR, BL, BR
/// The first point of TL and TR is a dummy point.
#[derive(Debug, Clone, PartialEq)]
pub struct VertebralCornerPoints(pub Array3<f64>);

impl TryFrom<&LabelMeData> for VertebralCornerPoints {
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
        // check BL count before subtracting 1 (potential underflow)
        if corners[2].shape()[0] == 0 {
            return Err(ScolError::InvalidPointCount(
                "BL should be at least 1".into(),
                corners[2].shape()[0],
            ));
        }

        if corners[0].shape()[0] != corners[2].shape()[0] - 1 {
            return Err(ScolError::InvalidPointCombo(
                "TL".into(),
                "BL - 1".into(),
                corners[0].shape()[0],
                corners[2].shape()[0] - 1,
            ));
        }
        if corners[0].shape()[0] != 6 {
            return Err(ScolError::InvalidPointCount(
                "TL should be 6".into(),
                corners[0].shape()[0],
            ));
        }
        // prepend first points of TL and TR
        let first = corners[0].index_axis(Axis(0), 0).insert_axis(Axis(0));
        corners[0] = concatenate(Axis(0), &[first, corners[0].view()]).unwrap();
        let first = corners[1].index_axis(Axis(0), 0).insert_axis(Axis(0));
        corners[1] = concatenate(Axis(0), &[first, corners[1].view()]).unwrap();
        let verts = stack![Axis(1), corners[0], corners[1], corners[2], corners[3]];
        Ok(VertebralCornerPoints(verts))
    }
}

#[derive(Debug, Clone, PartialEq, HasImageMetadata)]
pub struct LateralPoints {
    pub corners: VertebralCornerPoints,
    pub lamina: Array2<f64>,

    pub brow: Array2<f64>,
    pub sella: Array2<f64>,
    pub orbit: Array2<f64>,
    pub external_auditory_canal: Array2<f64>,
    pub occipital: Array2<f64>,
    pub anterior_c1_arch: Array2<f64>,
    pub anterior_dens: Array2<f64>,
    pub posterior_dens: Array2<f64>,
    pub posterior_hard_palate: Array2<f64>,
    pub chin: Array2<f64>,
    pub manubrium: Array2<f64>,

    pub image_metadata: ImageMetadata,
}

impl Serialize for LateralPoints {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        LateralPointsIR::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LateralPoints {
    fn deserialize<D>(deserializer: D) -> Result<LateralPoints, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let lateral_points_ir = LateralPointsIR::deserialize(deserializer)?;
        LateralPoints::try_from(&lateral_points_ir).map_err(serde::de::Error::custom)
    }
}

impl Scalable for LateralPoints {
    type Error = Infallible;
    fn _impl_scale(&mut self) -> Result<(), Self::Error> {
        let scale_xy = ndarray::array![
            self.image_metadata.spacing_xy.0,
            self.image_metadata.spacing_xy.1
        ];
        self.corners.0.scale(scale_xy.view());
        self.lamina.scale(scale_xy.view());
        self.brow.scale(scale_xy.view());
        self.sella.scale(scale_xy.view());
        self.orbit.scale(scale_xy.view());
        self.external_auditory_canal.scale(scale_xy.view());
        self.occipital.scale(scale_xy.view());
        self.anterior_c1_arch.scale(scale_xy.view());
        self.anterior_dens.scale(scale_xy.view());
        self.posterior_dens.scale(scale_xy.view());
        self.posterior_hard_palate.scale(scale_xy.view());
        self.chin.scale(scale_xy.view());
        self.manubrium.scale(scale_xy.view());
        Ok(())
    }
}

impl TryFrom<LabelMeData> for LateralPoints {
    type Error = ScolError;

    fn try_from(data: LabelMeData) -> Result<Self, Self::Error> {
        let corners = VertebralCornerPoints::try_from(&data)?;
        let lamina = extract_points(&data, "Lamina")?;
        let brow = extract_points(&data, "Brow")?;
        let sella = extract_points(&data, "Sella")?;
        let orbit = extract_points(&data, "Orbit")?;
        let external_auditory_canal = extract_points(&data, "ExternalAuditoryCanal")?;
        let occipital = extract_points(&data, "Occipital")?;
        let anterior_c1_arch = extract_points(&data, "AnteriorC1Arch")?;
        let anterior_dens = extract_points(&data, "AnteriorDens")?;
        let posterior_dens = extract_points(&data, "PosteriorDens")?;
        let posterior_hard_palate = extract_points(&data, "PosteriorHardPalate")?;
        let chin = extract_points(&data, "Chin")?;
        let manubrium = extract_points(&data, "Manubrium")?;

        let image_data = ImageMetadata::from(data);

        Ok(LateralPoints {
            corners,
            lamina,
            brow,
            sella,
            orbit,
            external_auditory_canal,
            occipital,
            anterior_c1_arch,
            anterior_dens,
            posterior_dens,
            posterior_hard_palate,
            chin,
            manubrium,
            image_metadata: image_data,
        })
    }
}

impl TryFrom<&LabelMeData> for LateralPoints {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        LateralPoints::try_from(data.clone())
    }
}

impl TryFrom<&LateralPointsIR> for LateralPoints {
    type Error = ndarray::ShapeError;

    fn try_from(ir: &LateralPointsIR) -> Result<Self, Self::Error> {
        Ok(LateralPoints {
            corners: VertebralCornerPoints(nested_vec_to_array3(&ir.corners)?),
            lamina: vec_points_to_array2(&ir.lamina)?,
            brow: vec_points_to_array2(&ir.brow)?,
            sella: vec_points_to_array2(&ir.sella)?,
            orbit: vec_points_to_array2(&ir.orbit)?,
            external_auditory_canal: vec_points_to_array2(&ir.external_auditory_canal)?,
            occipital: vec_points_to_array2(&ir.occipital)?,
            anterior_c1_arch: vec_points_to_array2(&ir.anterior_c1_arch)?,
            anterior_dens: vec_points_to_array2(&ir.anterior_dens)?,
            posterior_dens: vec_points_to_array2(&ir.posterior_dens)?,
            posterior_hard_palate: vec_points_to_array2(&ir.posterior_hard_palate)?,
            chin: vec_points_to_array2(&ir.chin)?,
            manubrium: vec_points_to_array2(&ir.manubrium)?,
            image_metadata: ir.image_metadata.clone(),
        })
    }
}

/// Intermediate representation for lateral neck points for serialization and deserialization
#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq, HasImageMetadata)]
struct LateralPointsIR {
    pub corners: Vec<Vec<Point2d>>,
    pub lamina: Vec<Point2d>,
    pub brow: Vec<Point2d>,
    pub sella: Vec<Point2d>,
    pub orbit: Vec<Point2d>,
    pub external_auditory_canal: Vec<Point2d>,
    pub occipital: Vec<Point2d>,
    pub anterior_c1_arch: Vec<Point2d>,
    pub anterior_dens: Vec<Point2d>,
    pub posterior_dens: Vec<Point2d>,
    pub posterior_hard_palate: Vec<Point2d>,
    pub chin: Vec<Point2d>,
    pub manubrium: Vec<Point2d>,

    pub image_metadata: ImageMetadata,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, ContentFilename)]
pub struct LateralPointsLine {
    pub content: LateralPoints,
    pub filename: String,
}

impl TryFrom<LabelMeDataLine> for LateralPointsLine {
    type Error = <LateralPointsLine as TryConvertContentFilename<LabelMeDataLine>>::Error;

    fn try_from(data: LabelMeDataLine) -> Result<Self, Self::Error> {
        LateralPointsLine::try_convert_from(data)
    }
}

/// Trait facilitating conversion between different content filename types
///
/// To use blanket implementation, the target type must implement `TryFrom` for the source type.
/// For example, to convert from `DataLineA` to `DataLineB`, `DataLineB::ContentType` must implement `TryFrom<DataLineA::ContentType>`.
pub trait TryConvertContentFilename<T>: ContentFilename
where
    T: ContentFilename,
{
    type Error;

    fn try_convert_from(other: T) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized;
}

impl<T, U> TryConvertContentFilename<U> for T
where
    T: ContentFilename,
    U: ContentFilename,
    T::ContentType: TryFrom<U::ContentType>,
    <T::ContentType as TryFrom<U::ContentType>>::Error: std::fmt::Debug,
{
    type Error = <T::ContentType as TryFrom<U::ContentType>>::Error;

    fn try_convert_from(other: U) -> Result<Self, Self::Error> {
        let (from_content, filename) = other.content_filename();
        let content = T::ContentType::try_from(from_content)?;
        Ok(Self::new(content, filename))
    }
}

impl ContentFilename for labelme_rs::LabelMeDataLine {
    type ContentType = labelme_rs::LabelMeData;

    fn content_filename(self) -> (Self::ContentType, String) {
        (self.content, self.filename)
    }

    fn filename(&self) -> &str {
        &self.filename
    }

    fn new(content: Self::ContentType, filename: String) -> Self {
        Self { content, filename }
    }
}

pub trait Scale2DPoints {
    fn scale(&mut self, scale_xy: ArrayView1<f64>);
}

impl Scale2DPoints for Array2<f64> {
    fn scale(&mut self, scale_xy: ArrayView1<f64>) {
        if self.len_of(Axis(0)) == 0 {
            return;
        }
        let scale_xy = scale_xy.insert_axis(Axis(0)).to_owned();
        self.mul_assign(&scale_xy);
    }
}

impl Scale2DPoints for Array3<f64> {
    fn scale(&mut self, scale_xy: ArrayView1<f64>) {
        if self.len_of(Axis(0)) == 0 {
            return;
        }
        if self.len_of(Axis(1)) == 0 {
            return;
        }

        let scale_xy = scale_xy
            .insert_axis(Axis(0))
            .insert_axis(Axis(0))
            .to_owned();
        self.mul_assign(&scale_xy);
    }
}

impl From<&LateralPoints> for LateralPointsIR {
    fn from(lateral_points: &LateralPoints) -> Self {
        LateralPointsIR {
            corners: array3_to_nested_vec_points(lateral_points.corners.0.to_owned()),
            lamina: array2_to_vec_points(lateral_points.lamina.to_owned()),
            brow: array2_to_vec_points(lateral_points.brow.to_owned()),
            sella: array2_to_vec_points(lateral_points.sella.to_owned()),
            orbit: array2_to_vec_points(lateral_points.orbit.to_owned()),
            external_auditory_canal: array2_to_vec_points(
                lateral_points.external_auditory_canal.to_owned(),
            ),
            occipital: array2_to_vec_points(lateral_points.occipital.to_owned()),
            anterior_c1_arch: array2_to_vec_points(lateral_points.anterior_c1_arch.to_owned()),
            anterior_dens: array2_to_vec_points(lateral_points.anterior_dens.to_owned()),
            posterior_dens: array2_to_vec_points(lateral_points.posterior_dens.to_owned()),
            posterior_hard_palate: array2_to_vec_points(
                lateral_points.posterior_hard_palate.to_owned(),
            ),
            chin: array2_to_vec_points(lateral_points.chin.to_owned()),
            manubrium: array2_to_vec_points(lateral_points.manubrium.to_owned()),
            image_metadata: lateral_points.image_metadata.clone(),
        }
    }
}

impl From<LateralPoints> for LabelMeData {
    fn from(lateral_points: LateralPoints) -> Self {
        // let lateral_points = LateralPoints::try_from(&lateral_points_ir)?;
        let mut data = LabelMeData {
            imagePath: lateral_points.image_metadata.path,
            imageHeight: lateral_points.image_metadata.height,
            imageWidth: lateral_points.image_metadata.width,
            ..Default::default()
        };
        // First poitns of TL and TR are discarded because they are dummy points
        let tl_points = array2_to_vec_points(
            lateral_points
                .corners
                .0
                .index_axis(Axis(1), 0)
                .slice(s![1.., ..])
                .to_owned(),
        );
        let tr_points = array2_to_vec_points(
            lateral_points
                .corners
                .0
                .index_axis(Axis(1), 1)
                .slice(s![1.., ..])
                .to_owned(),
        );
        let bl_points =
            array2_to_vec_points(lateral_points.corners.0.index_axis(Axis(1), 2).to_owned());
        let br_points =
            array2_to_vec_points(lateral_points.corners.0.index_axis(Axis(1), 3).to_owned());

        data.shapes = create_shapes(&[
            ("TL", tl_points),
            ("TR", tr_points),
            ("BL", bl_points),
            ("BR", br_points),
            ("Lamina", array2_to_vec_points(lateral_points.lamina)),
            ("Brow", array2_to_vec_points(lateral_points.brow)),
            ("Sella", array2_to_vec_points(lateral_points.sella)),
            ("Orbit", array2_to_vec_points(lateral_points.orbit)),
            (
                "ExternalAuditoryCanal",
                array2_to_vec_points(lateral_points.external_auditory_canal),
            ),
            ("Occipital", array2_to_vec_points(lateral_points.occipital)),
            (
                "AnteriorC1Arch",
                array2_to_vec_points(lateral_points.anterior_c1_arch),
            ),
            (
                "AnteriorDens",
                array2_to_vec_points(lateral_points.anterior_dens),
            ),
            (
                "PosteriorDens",
                array2_to_vec_points(lateral_points.posterior_dens),
            ),
            (
                "PosteriorHardPalate",
                array2_to_vec_points(lateral_points.posterior_hard_palate),
            ),
            ("Chin", array2_to_vec_points(lateral_points.chin)),
            ("Manubrium", array2_to_vec_points(lateral_points.manubrium)),
        ]);

        data
    }
}

impl From<&LateralPoints> for LabelMeData {
    fn from(lateral_points: &LateralPoints) -> Self {
        LabelMeData::from(lateral_points.clone())
    }
}

const NEKC_SAGITTAL_COMPONENT_CLASS: &str = "NeckSagittalComponent";
pub trait NeckSagittalComponent: DrawComponent {
    fn default_group(&self) -> element::Group {
        self.default_group_w_classes(&["Component", NEKC_SAGITTAL_COMPONENT_CLASS])
    }
}

pub trait NeckMeasureComponent: Named {
    fn measure(&self) -> Result<Vec<f64>, MeasureError>;
}

/// Sacral Avaialble Spaces
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_LINE, CLASS_DISTANCE])]
#[label("SACs")]
pub struct Sacs<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for Sacs<'_> {}
impl DrawComponent for Sacs<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        self.0.posterior_dens.validate_label_length("Dens", 1)?;
        self.0.lamina.validate_label_length("Lamina", 8)?;
        let color = line_colors.get_or_new(self.id());
        let mut group = self.default_group().set("stroke", color);
        // C1SAC
        let lamina = self.0.lamina.index_axis(Axis(0), 0);
        let points = stack![
            Axis(0),
            lamina.view(),
            self.0.posterior_dens.index_axis(Axis(0), 0).view()
        ];
        let line = painter.line(points.view());
        group = group.add(line);
        let length = (&points.index_axis(Axis(0), 0) - &points.index_axis(Axis(0), 1)).l2norm();
        let text = painter.text(
            &format!("{:.1}", length),
            points.index_axis(Axis(0), 0),
            Some("C1SAC"),
            Some(&self.0.image_metadata.unit),
        );
        group = group.add(text);

        // C2SAC to T1SAC
        let mut trs = self.0.corners.0.index_axis(Axis(1), 1).to_owned();
        trs.index_axis_mut(Axis(0), 0)
            .assign(&self.0.posterior_dens.index_axis(Axis(0), 0));
        let brs = self.0.corners.0.index_axis(Axis(1), 3);
        let tr_brs = stack![Axis(0), trs, brs];
        let lamina_below_c2 = self.0.lamina.slice(s![1.., ..]);
        for (i, (tr_br, lamina)) in tr_brs
            .axis_iter(Axis(1))
            .zip(lamina_below_c2.axis_iter(Axis(0)))
            .enumerate()
        {
            let posterior_line = points2line(tr_br);
            let l_lamina = lyon_geom::Point::new(lamina[0], lamina[1]);

            let line_equation = posterior_line.equation();
            let projected_point = line_equation.project_point(&l_lamina);
            let projected_point = Array::from(vec![projected_point.x, projected_point.y]);

            let points = stack![Axis(0), lamina, projected_point.view()];
            let length = (&points.index_axis(Axis(0), 0) - &points.index_axis(Axis(0), 1)).l2norm();
            let line = painter.line(points.view());
            group = group.add(line);
            let (line_p1, line_p2) = distanced_pair3(
                tr_br.index_axis(Axis(0), 0),
                tr_br.index_axis(Axis(0), 1),
                projected_point.view(),
            );
            let line = painter.line(stack![Axis(0), line_p1, line_p2].view());
            group = group.add(line);
            let title = if i < 6 {
                format!("C{}SAC", 2 + i)
            } else {
                "T1SAC".to_string()
            };
            let text = painter.text(
                &format!("{:.1}", length),
                lamina,
                Some(title.as_str()),
                Some(&self.0.image_metadata.unit),
            );
            group = group.add(text);
        }
        Ok(group)
    }
}

impl NeckMeasureComponent for Sacs<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        self.0.posterior_dens.validate_label_length("Dens", 1)?;
        self.0.lamina.validate_label_length("Lamina", 8)?;
        let mut lengths: Vec<f64> = Vec::new();
        // C1SAC
        let lamina = self.0.lamina.index_axis(Axis(0), 0);
        let points = stack![
            Axis(0),
            lamina.view(),
            self.0.posterior_dens.index_axis(Axis(0), 0).view()
        ];
        lengths.push((&points.index_axis(Axis(0), 0) - &points.index_axis(Axis(0), 1)).l2norm());

        // C2SAC to T1SAC
        let mut trs = self.0.corners.0.index_axis(Axis(1), 1).to_owned();
        trs.index_axis_mut(Axis(0), 0)
            .assign(&self.0.posterior_dens.index_axis(Axis(0), 0));
        let brs = self.0.corners.0.index_axis(Axis(1), 3);
        let tr_brs = stack![Axis(0), trs, brs];
        let lamina_below_c2 = self.0.lamina.slice(s![1.., ..]);
        for (tr_br, lamina) in tr_brs
            .axis_iter(Axis(1))
            .zip(lamina_below_c2.axis_iter(Axis(0)))
        {
            let posterior_line = points2line(tr_br);
            let l_lamina = lyon_geom::Point::new(lamina[0], lamina[1]);

            let line_equation = posterior_line.equation();
            let projected_point = line_equation.project_point(&l_lamina);
            let projected_point = Array::from(vec![projected_point.x, projected_point.y]);

            let points = stack![Axis(0), lamina, projected_point.view()];
            lengths
                .push((&points.index_axis(Axis(0), 0) - &points.index_axis(Axis(0), 1)).l2norm());
        }
        Ok(lengths)
    }
}

/// Atlanto-dental interval
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_LINE, CLASS_DISTANCE])]
#[label("ADI")]
pub struct Adi<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for Adi<'_> {}
impl Adi<'_> {
    fn prep(&self) -> Result<(), MeasureError> {
        self.0.anterior_dens.validate_label_length("Dens", 1)?;
        self.0
            .anterior_c1_arch
            .validate_label_length("AnteriorC1Arch", 1)?;
        Ok(())
    }
}
impl DrawComponent for Adi<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let mut group = self.default_group().set("stroke", color);
        self.prep()?;
        let points = stack![
            Axis(0),
            self.0.anterior_dens.index_axis(Axis(0), 0).view(),
            self.0.anterior_c1_arch.index_axis(Axis(0), 0).view()
        ];
        let line = painter.line(points.view());
        group = group.add(line);
        let length = (&points.index_axis(Axis(0), 0) - &points.index_axis(Axis(0), 1)).l2norm();
        let text = painter.text(
            &format!("{:.1}", length,),
            points.index_axis(Axis(0), 0),
            Some(self.id()),
            Some(&self.0.image_metadata.unit),
        );
        group = group.add(text);
        Ok(group)
    }
}
impl NeckMeasureComponent for Adi<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        self.prep()?;
        let diff = &self.0.anterior_dens.index_axis(Axis(0), 0)
            - &self.0.anterior_c1_arch.index_axis(Axis(0), 0);
        Ok(vec![diff.l2norm()])
    }
}

trait NeckExt {
    /// McGregor's line
    /// Line between the posterior hard palate and the occipital point
    fn mcgregor_line(&self) -> Result<Array2<f64>, MeasureError>;
}

impl NeckExt for LateralPoints {
    fn mcgregor_line(&self) -> Result<Array2<f64>, MeasureError> {
        self.posterior_hard_palate
            .validate_label_length("PosteriorHardPlate", 1)?;
        self.occipital.validate_label_length("Occipital", 1)?;
        let mcgregor_points = stack![
            Axis(0),
            self.posterior_hard_palate.index_axis(Axis(0), 0),
            self.occipital.index_axis(Axis(0), 0),
        ];
        Ok(mcgregor_points)
    }
}

/// Occiput-C2 Angle
/// Angle between McGregor's line and C2 lower endplate
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct OC2<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for OC2<'_> {}
impl OC2<'_> {
    fn prep(&self) -> Result<(Array2<f64>, Array2<f64>), MeasureError> {
        let mcgregor_points = self.0.mcgregor_line()?;
        let c2 = self.0.corners.0.index_axis(Axis(0), 0);
        let c2_lower_endplate = c2.slice(s![2.., ..]).to_owned();
        Ok((mcgregor_points, c2_lower_endplate))
    }
}
impl DrawComponent for OC2<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let mut group = self.default_group().set("stroke", color);
        let (mcgregor_points, c2_lower_endplate) = self.prep()?;
        let line = painter.line(mcgregor_points.view());
        group = group.add(line);
        let line = painter.line(c2_lower_endplate.view());
        let c2_length = c2_lower_endplate
            .index_axis(Axis(0), 0)
            .l2_dist(&c2_lower_endplate.index_axis(Axis(0), 1))
            .unwrap();
        group = painter.cobb_from_plates(
            group,
            (mcgregor_points, c2_lower_endplate.to_owned()),
            &CobbAux {
                plate_scale: 10.0,
                flip_sign: false,
                ..Default::default()
            },
            c2_length,
            Some(self.id()),
        );
        group = group.add(line);
        Ok(group)
    }
}
impl NeckMeasureComponent for OC2<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        let (mcgregor_points, c2_lower_endplate) = self.prep()?;
        let angle =
            angle_from_lines(mcgregor_points.view(), c2_lower_endplate.view()).unwrap_or_default();
        Ok(vec![angle])
    }
}

/// Wedge (C2-C3 to C7-T1 intervertebral) angles
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct WedgeAngle<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for WedgeAngle<'_> {}
impl DrawComponent for WedgeAngle<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let mut group = self.default_group();
        group = group.set("stroke", line_colors.get_or_new(self.id()));
        let wedge_lengths: Vec<_> = self
            .0
            .corners
            .0
            .axis_iter(Axis(0))
            .map(|c| {
                c.index_axis(Axis(0), 0)
                    .l2_dist(&c.index_axis(Axis(0), 1))
                    .unwrap()
            })
            .collect();
        let mean_wedge_length = wedge_lengths.iter().sum::<f64>() / wedge_lengths.len() as f64;
        for i in 0..6 {
            let wedge_upper = self.0.corners.0.index_axis(Axis(0), i);
            let wedge_upper = wedge_upper.slice(s![2.., ..]);
            let wedge_lower = self.0.corners.0.index_axis(Axis(0), i + 1);
            let wedge_lower = wedge_lower.slice(s![..2, ..]);
            group = painter.cobb_from_plates(
                group,
                (wedge_upper.to_owned(), wedge_lower.to_owned()),
                &CobbAux {
                    plate_scale: 1.5,
                    flip_sign: false,
                    ..Default::default()
                },
                mean_wedge_length,
                Some(self.id()),
            );
        }
        Ok(group)
    }
}
impl NeckMeasureComponent for WedgeAngle<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        let mut angles = Vec::new();
        for i in 0..6 {
            let wedge_upper = self.0.corners.0.index_axis(Axis(0), i);
            let wedge_upper = wedge_upper.slice(s![2.., ..]);
            let wedge_lower = self.0.corners.0.index_axis(Axis(0), i + 1);
            let wedge_lower = wedge_lower.slice(s![..2, ..]);
            let angle = angle_from_lines(wedge_upper, wedge_lower).unwrap_or_default();
            angles.push(angle);
        }
        Ok(angles)
    }
}

/// Modified Renawat Index
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct ModifiedRenawatIndex<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for ModifiedRenawatIndex<'_> {}
impl ModifiedRenawatIndex<'_> {
    /// Array2 of [intersection, c2_lower_middle]
    fn prep(&self) -> Result<Option<Array2<f64>>, MeasureError> {
        self.0
            .anterior_c1_arch
            .validate_label_length("AnteriorC1Arch", 1)?;
        // posterior_c2_arch?
        self.0.posterior_dens.validate_label_length("Dens", 1)?;
        let c1_line = points2line(stack![
            Axis(0),
            self.0.anterior_c1_arch.index_axis(Axis(0), 0).view(),
            self.0.posterior_dens.index_axis(Axis(0), 0).view()
        ]);
        let c2 = self.0.corners.0.index_axis(Axis(0), 0);
        let c2_lower_endplate = c2.slice(s![2.., ..]);
        let c2_lower_middle = c2_lower_endplate.mean_axis(Axis(0)).unwrap();
        // perpendicular direction to the line
        let c2_normal = points2line(c2_lower_endplate).equation().normal();
        let c2_perpendicular_line = lyon_geom::Line {
            point: point(c2_lower_middle[0], c2_lower_middle[1]),
            vector: c2_normal,
        };
        // intersection point
        let intersection = c1_line.intersection(&c2_perpendicular_line);
        if let Some(intersection) = intersection {
            let intersection = Array::from(vec![intersection.x, intersection.y]);
            Ok(Some(stack![Axis(0), intersection, c2_lower_middle]))
        } else {
            Ok(None)
        }
    }
}
impl DrawComponent for ModifiedRenawatIndex<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let mut group = self.default_group().set("stroke", color);
        let intersection_c2_lower_middle = self.prep()?;
        if let Some(intersection_c2_lower_middle) = intersection_c2_lower_middle {
            let line = painter.line(intersection_c2_lower_middle.view());
            group = group.add(line);
            let (p1, p2) = distanced_pair3(
                intersection_c2_lower_middle.index_axis(Axis(0), 0),
                self.0.anterior_c1_arch.index_axis(Axis(0), 0),
                self.0.posterior_dens.index_axis(Axis(0), 0),
            );
            let line = painter.line(stack![Axis(0), p1, p2].view());
            group = group.add(line);
            let length = (&intersection_c2_lower_middle.index_axis(Axis(0), 0)
                - &intersection_c2_lower_middle.index_axis(Axis(0), 1))
                .l2norm();
            let text = painter.text(
                &format!("{:.1}", length),
                intersection_c2_lower_middle.index_axis(Axis(0), 0),
                Some(self.id()),
                Some(&self.0.image_metadata.unit),
            );
            group = group.add(text);
        }

        Ok(group)
    }
}
impl NeckMeasureComponent for ModifiedRenawatIndex<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        let intersection = self.prep()?;
        if let Some(intersection) = intersection {
            let diff = &intersection.index_axis(Axis(0), 0) - &intersection.index_axis(Axis(0), 1);
            Ok(vec![diff.l2norm()])
        } else {
            Ok(vec![0.0])
        }
    }
}

/// Thoracic Inlet Angle
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct ThoracicInletAngle<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for ThoracicInletAngle<'_> {}
impl ThoracicInletAngle<'_> {
    fn prep(&self) -> Result<Array2<f64>, MeasureError> {
        self.0.manubrium.validate_label_length("Manubrium", 1)?;
        self.0.corners.0.validate_label_length("Vertebra", 7)?;
        let t1 = self.0.corners.0.index_axis(Axis(0), 6);
        let t1_top_plate = t1.slice(s![..2, ..]);
        Ok(t1_top_plate.to_owned())
    }
}
impl DrawComponent for ThoracicInletAngle<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let group = self.default_group().set("stroke", color).set("fill", color);
        let t1_top_plate = self.prep()?;
        let group = draw_incidence_angle(
            group,
            self.id(),
            self.0.manubrium.view(),
            t1_top_plate.view(),
            painter,
        );
        Ok(group)
    }
}
impl NeckMeasureComponent for ThoracicInletAngle<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        let t1_top_plate = self.prep()?;
        let angle = femoral_incidence_angle(
            t1_top_plate.view(),
            &crate::AtMost2(self.0.manubrium.to_owned()),
        )?;
        Ok(vec![angle])
    }
}

/// Neck Tilt Angle
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct NeckTilt<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for NeckTilt<'_> {}
impl NeckTilt<'_> {
    fn prep(&self) -> Result<(Array2<f64>, Array2<f64>), MeasureError> {
        self.0.manubrium.validate_label_length("Manubrium", 1)?;
        self.0.corners.0.validate_label_length("Vertebra", 7)?;
        let t1 = self.0.corners.0.index_axis(Axis(0), 6);
        let t1_top_plate = t1.slice(s![..2, ..]);
        let t1_top_middle = t1_top_plate.mean_axis(Axis(0)).unwrap();
        let manubrium_to_t1 = stack![
            Axis(0),
            self.0.manubrium.index_axis(Axis(0), 0).view(),
            t1_top_middle
        ];
        let mut v_line_from_manubrium = stack![
            Axis(0),
            self.0.manubrium.index_axis(Axis(0), 0),
            self.0.manubrium.index_axis(Axis(0), 0),
        ];
        let v_line_length = 0.5
            * manubrium_to_t1
                .index_axis(Axis(0), 0)
                .l2_dist(&manubrium_to_t1.index_axis(Axis(0), 1))
                .unwrap();
        v_line_from_manubrium[[1, 1]] -= v_line_length;
        Ok((v_line_from_manubrium, manubrium_to_t1))
    }
}
impl DrawComponent for NeckTilt<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let mut group = self.default_group().set("stroke", color);
        let (v_line_from_manubrium, manubrium_to_t1) = self.prep()?;
        group = painter
            .angle_between(
                group,
                v_line_from_manubrium.view(),
                manubrium_to_t1.view(),
                v_line_from_manubrium.index_axis(Axis(0), 0),
                0.8 * manubrium_to_t1
                    .index_axis(Axis(0), 0)
                    .l2_dist(&v_line_from_manubrium.index_axis(Axis(0), 1))
                    .unwrap(),
                Some(self.id()),
            )
            .0;
        Ok(group)
    }
}
impl NeckMeasureComponent for NeckTilt<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        let (line1, line2) = self.prep()?;
        let angle = angle_between(line1.view(), line2.view()).to_degrees();
        Ok(vec![angle])
    }
}

/// Spino-Cranial Angle
/// Angle between C7 upper endplate and the line connecting sella and the middle of C7 upper endplate
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct SpinoCranialAngle<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for SpinoCranialAngle<'_> {}
impl SpinoCranialAngle<'_> {
    fn prep(&self) -> Result<(Array2<f64>, Array2<f64>), MeasureError> {
        self.0.sella.validate_label_length("Sella", 1)?;
        self.0.corners.0.validate_label_length("Vertebra", 7)?;
        let c7 = self.0.corners.0.index_axis(Axis(0), 6);
        let c7_upper_endplate = c7.slice(s![..2, ..]);
        let c7_upper_middle = c7_upper_endplate.mean_axis(Axis(0)).unwrap();
        let sella = self.0.sella.index_axis(Axis(0), 0);
        // let sella_to_c7 = stack![Axis(0), sella, c7_upper_middle.view()];
        let c7_to_sella = stack![Axis(0), c7_upper_middle.view(), sella.view()];
        let c7_mid_to_posterior = stack![
            Axis(0),
            c7_upper_middle.view(),
            c7_upper_endplate.index_axis(Axis(0), 1).view()
        ];
        Ok((c7_to_sella, c7_mid_to_posterior))
    }
}
impl DrawComponent for SpinoCranialAngle<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let mut group = self.default_group().set("stroke", color);
        let (c7_to_sella, c7_mid_to_posterior) = self.prep()?;

        let arc_radius = 0.8
            * c7_to_sella
                .index_axis(Axis(0), 0)
                .l2_dist(&c7_mid_to_posterior.index_axis(Axis(0), 1))
                .unwrap();
        group = painter
            .angle_between(
                group,
                c7_to_sella.view(),
                c7_mid_to_posterior.view(),
                c7_mid_to_posterior.index_axis(Axis(0), 0),
                arc_radius,
                Some(self.id()),
            )
            .0;
        Ok(group)
    }
}
impl NeckMeasureComponent for SpinoCranialAngle<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        let (c7_to_sella, c7_mid_to_posterior) = self.prep()?;
        let angle = angle_between(c7_to_sella.view(), c7_mid_to_posterior.view()).to_degrees();
        Ok(vec![angle])
    }
}

/// Occipitocervical inclination
/// The angle formed by the line connecting McGregor’s line and the posterior border of the C4 vertebral body
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct OccipitocervicalInclination<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for OccipitocervicalInclination<'_> {}
type OccipitocervicalInclinationPrepResult = Option<(Array2<f64>, Array2<f64>, Array1<f64>)>;
impl OccipitocervicalInclination<'_> {
    fn prep(&self) -> Result<OccipitocervicalInclinationPrepResult, MeasureError> {
        self.0.mcgregor_line()?;
        self.0.corners.0.validate_label_length("Vertebra", 7)?;
        let c4 = self.0.corners.0.index_axis(Axis(0), 3);
        let c4_posterior = c4.slice(s![1..;2, ..]);
        let mcgregor_points = self.0.mcgregor_line()?;
        let c4_line = points2line(c4_posterior);
        let mcgregor_line = points2line(mcgregor_points.view());
        let intersection = mcgregor_line.intersection(&c4_line);
        if let Some(intersection) = intersection {
            let intersection = Array::from(vec![intersection.x, intersection.y]);
            Ok(Some((
                c4_posterior.to_owned(),
                mcgregor_points,
                intersection,
            )))
        } else {
            Ok(None)
        }
    }
}
impl DrawComponent for OccipitocervicalInclination<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let mut group = self.default_group().set("stroke", color);
        let prep_data = self.prep()?;
        let (c4_posterior, mcgregor_points, intersection) = if let Some(data) = prep_data {
            data
        } else {
            // TODO??: draw parallel lines
            return Ok(group); // No intersection found, nothing to draw
        };
        let extended_c4_line = stack![Axis(0), intersection, c4_posterior.index_axis(Axis(0), 1)];
        let arc_radius = 0.8
            * c4_posterior
                .index_axis(Axis(0), 0)
                .l2_dist(&c4_posterior.index_axis(Axis(0), 1))
                .unwrap();
        group = painter
            .angle_between(
                group,
                mcgregor_points.view(),
                extended_c4_line.view(),
                intersection.view(),
                arc_radius,
                Some(self.id()),
            )
            .0;
        Ok(group)
    }
}
impl NeckMeasureComponent for OccipitocervicalInclination<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        let prep_data = self.prep()?;
        let (c4_posterior, mcgregor_points, _intersection) = if let Some(data) = prep_data {
            data
        } else {
            return Ok(vec![0.0]); // No intersection found, return 0
        };
        let angle = angle_between(mcgregor_points.view(), c4_posterior.view()).to_degrees();
        Ok(vec![angle])
    }
}

/// Cranial Slope
/// Angle between McGregor's line and the horizontal line
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct CranialSlope<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for CranialSlope<'_> {}
impl DrawComponent for CranialSlope<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let mut group = self.default_group().set("stroke", color).set("fill", color);
        let mcgregor_points = self.0.mcgregor_line()?;
        group = draw_tilt_angle(group, painter, &mcgregor_points, Some(self.id()));
        Ok(group)
    }
}
impl NeckMeasureComponent for CranialSlope<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        let mcgregor_points = self.0.mcgregor_line()?;
        tilt_angle(self.id(), mcgregor_points.view()).map(|angle| vec![angle])
    }
}

/// T1 tilt angle (T1 slope)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct T1Slope<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for T1Slope<'_> {}
impl T1Slope<'_> {
    fn prep(&self) -> Array2<f64> {
        let t1 = self.0.corners.0.index_axis(Axis(0), 6);
        let t1_top_plate = t1.slice(s![..2, ..]);
        t1_top_plate.to_owned()
    }
}
impl DrawComponent for T1Slope<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let group = self.default_group().set("stroke", color).set("fill", color);
        let t1_top_plate = self.prep();
        let group = draw_tilt_angle(group, painter, &t1_top_plate, Some(self.id()));
        Ok(group)
    }
}
impl NeckMeasureComponent for T1Slope<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        let t1_top_plate = self.prep();
        tilt_angle(self.id(), t1_top_plate.view()).map(|angle| vec![angle])
    }
}

/// Torg-Pavlov Ratio
/// canal-to-body ratio: https://radiopaedia.org/articles/canal-to-body-ratio-of-torg-and-pavlov
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_RATIO])]
pub struct TPR<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for TPR<'_> {}
impl DrawComponent for TPR<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let mut group = self.default_group().set("stroke", color);
        self.0.corners.0.validate_label_length("Vertebra", 7)?;
        let lamina_below_c2 = self.0.lamina.slice(s![2.., ..]);
        let vertebra_below_c2 = self.0.corners.0.slice(s![1.., .., ..]);
        for (i, (lamina, vertebra)) in lamina_below_c2
            .axis_iter(Axis(0))
            .zip(vertebra_below_c2.axis_iter(Axis(0)))
            .enumerate()
        {
            let anterior_mid_point = stack![
                Axis(0),
                vertebra.index_axis(Axis(0), 0),
                vertebra.index_axis(Axis(0), 2)
            ]
            .mean_axis(Axis(0))
            .unwrap();
            let posterior_mid_point = stack![
                Axis(0),
                vertebra.index_axis(Axis(0), 1),
                vertebra.index_axis(Axis(0), 3)
            ]
            .mean_axis(Axis(0))
            .unwrap();
            let canal_diameter = lamina.l2_dist(&posterior_mid_point).unwrap();
            let body_diameter = anterior_mid_point.l2_dist(&posterior_mid_point).unwrap();
            let ratio = canal_diameter / body_diameter;

            let poly = painter.polyline(stack![
                Axis(0),
                anterior_mid_point.view(),
                posterior_mid_point.view(),
                lamina.view()
            ]);
            group = group.add(poly);
            let title = if i < 6 {
                format!("C{}-TPR", i + 2)
            } else {
                format!("T{}-TPR", i - 5)
            };
            let text = painter.text(
                &format!("{:.2}", ratio),
                anterior_mid_point.view(),
                Some(&title),
                None,
            );
            group = group.add(text);
        }
        Ok(group)
    }
}
impl NeckMeasureComponent for TPR<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        self.0.corners.0.validate_label_length("Vertebra", 7)?;
        let mut ratios = Vec::new();
        let lamina_below_c2 = self.0.lamina.slice(s![2.., ..]);
        let vertebra_below_c2 = self.0.corners.0.slice(s![1.., .., ..]);
        for (lamina, vertebra) in lamina_below_c2
            .axis_iter(Axis(0))
            .zip(vertebra_below_c2.axis_iter(Axis(0)))
        {
            let anterior_mid_point = stack![
                Axis(0),
                vertebra.index_axis(Axis(0), 0),
                vertebra.index_axis(Axis(0), 2)
            ]
            .mean_axis(Axis(0))
            .unwrap();
            let posterior_mid_point = stack![
                Axis(0),
                vertebra.index_axis(Axis(0), 1),
                vertebra.index_axis(Axis(0), 3)
            ]
            .mean_axis(Axis(0))
            .unwrap();
            let canal_diameter = lamina.l2_dist(&posterior_mid_point).unwrap();
            let body_diameter = anterior_mid_point.l2_dist(&posterior_mid_point).unwrap();
            ratios.push(canal_diameter / body_diameter);
        }
        Ok(ratios)
    }
}

fn sagittal_vertical_axis(
    painter: &mut Painter,
    mut group: element::Group,
    c2_lower_middle: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>,
    c7_tr: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>,
    label: &str,
    unit: &str,
) -> element::Group {
    // plumb line point from c2
    let mut c2_plumb_point = c2_lower_middle.clone();
    c2_plumb_point[1] = c7_tr[1];
    // same y
    let polyline_points = stack![
        Axis(0),
        c2_lower_middle.view(),
        c2_plumb_point.view(),
        c7_tr.view()
    ];
    let line = painter.polyline(polyline_points.view());
    group = group.add(line);
    let distance = c7_tr[0] - c2_lower_middle[0];
    let text_position = polyline_points
        .slice(s![1.., ..])
        .mean_axis(Axis(0))
        .unwrap();
    let text = painter.text(
        &format!("{:.1}", distance),
        text_position.view(),
        Some(label),
        Some(unit),
    );
    group.add(text)
}

/// C2-7 Sagittal Vertical Axis
/// The horizontal distance from the center of the lower end plate of C2 to the posterior-superior corner of C7
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_DISTANCE])]
pub struct C2C7SVA<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for C2C7SVA<'_> {}
impl C2C7SVA<'_> {
    fn prep(&self) -> Result<(Array1<f64>, Array1<f64>), MeasureError> {
        let c2 = self.0.corners.0.index_axis(Axis(0), 0);
        let c2_lower_endplate = c2.slice(s![2.., ..]);
        let c2_lower_middle = c2_lower_endplate.mean_axis(Axis(0)).unwrap();
        let c7 = self.0.corners.0.index_axis(Axis(0), 5);
        let c7_tr = c7.index_axis(Axis(0), 1);
        Ok((c2_lower_middle.to_owned(), c7_tr.to_owned()))
    }
}
impl DrawComponent for C2C7SVA<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let mut group = self.default_group().set("stroke", color);
        let (c2_lower_middle, c7_tr) = self.prep()?;
        let label = self.id();
        let unit = &self.0.image_metadata.unit;
        group = sagittal_vertical_axis(painter, group, c2_lower_middle, c7_tr, label, unit);
        Ok(group)
    }
}

impl NeckMeasureComponent for C2C7SVA<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        let (c2_lower_middle, c7_tr) = self.prep()?;
        let distance = c7_tr[0] - c2_lower_middle[0];
        Ok(vec![distance])
    }
}

/// External Auditory Canal Sagittal Vertical Axis
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_DISTANCE])]
pub struct EACSVA<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for EACSVA<'_> {}
impl EACSVA<'_> {
    fn prep(&self) -> Result<(Array1<f64>, Array1<f64>), MeasureError> {
        self.0
            .external_auditory_canal
            .validate_label_length_more_than("ExternalAuditoryCanal", 1)?;
        let c7 = self.0.corners.0.index_axis(Axis(0), 5);
        let c7_tr = c7.index_axis(Axis(0), 1);
        let eac = if self.0.external_auditory_canal.shape()[0] == 1 {
            // if there is one point, use it
            self.0
                .external_auditory_canal
                .index_axis(Axis(0), 0)
                .to_owned()
        } else {
            // otherwise, use the mean of the first two points
            self.0
                .external_auditory_canal
                .slice(s![..2, ..])
                .mean_axis(Axis(0))
                .unwrap()
        };
        Ok((eac, c7_tr.to_owned()))
    }
}
impl DrawComponent for EACSVA<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let mut group = self.default_group().set("stroke", color);
        let (eac, c7_tr) = self.prep()?;
        let label = self.id();
        let unit = &self.0.image_metadata.unit;
        group = sagittal_vertical_axis(painter, group, eac, c7_tr, label, unit);
        Ok(group)
    }
}
impl NeckMeasureComponent for EACSVA<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        let (eac, c7_tr) = self.prep()?;
        let distance = c7_tr[0] - eac[0];
        Ok(vec![distance])
    }
}

/// End plate angle
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct EndPlateAngle<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for EndPlateAngle<'_> {}
impl EndPlateAngle<'_> {
    fn prep(&self) -> Result<Array3<f64>, MeasureError> {
        // (7, 4, 2) -> (14, 2, 2)
        let endplates = self
            .0
            .corners
            .0
            .to_owned()
            .into_shape_with_order((14, 2, 2))
            .unwrap();
        // remove the first end plate because it is dummy
        let endplates = endplates.slice(s![1.., .., ..]).to_owned();
        Ok(endplates)
    }
}
impl DrawComponent for EndPlateAngle<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let mut group = self.default_group().set("stroke", color);
        let endplates = self.prep()?;
        for plate in endplates.axis_iter(Axis(0)) {
            let angle = -crate::draw::tilt_angle(self.id(), plate.view()).unwrap_or_default();
            let line = painter.line(plate);
            group = group.add(line);
            let mid_point = plate.mean_axis(Axis(0)).unwrap();
            let text = painter.text(
                &format!("{:.1}°", angle),
                mid_point.view(),
                Some(self.id()),
                None,
            );
            group = group.add(text);
        }
        Ok(group)
    }
}
impl NeckMeasureComponent for EndPlateAngle<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        let endplates = self.prep()?;
        let mut angles = Vec::new();
        for plate in endplates.axis_iter(Axis(0)) {
            let angle = -tilt_angle(self.id(), plate.view())?;
            angles.push(angle);
        }
        Ok(angles)
    }
}

/// Four corner points of each vertebra
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POINT])]
pub struct CervicalPoints<'a>(pub &'a LateralPoints);
impl HasCornerPoints for CervicalPoints<'_> {
    fn top_left(&'_ self) -> ArrayView2<'_, f64> {
        self.0.corners.0.slice(s![1.., 0, ..])
    }
    fn top_right(&'_ self) -> ArrayView2<'_, f64> {
        self.0.corners.0.slice(s![1.., 1, ..])
    }
    fn bottom_left(&'_ self) -> ArrayView2<'_, f64> {
        self.0.corners.0.slice(s![.., 2, ..])
    }
    fn bottom_right(&'_ self) -> ArrayView2<'_, f64> {
        self.0.corners.0.slice(s![.., 3, ..])
    }
}
impl NeckSagittalComponent for CervicalPoints<'_> {}
impl DrawComponent for CervicalPoints<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let mut group = self.default_group();
        group = self.draw_corners(group, painter, label_colors)?;

        for (label, points) in [
            ("Lamina", &self.0.lamina),
            ("Brow", &self.0.brow),
            ("Sella", &self.0.sella),
            ("Orbit", &self.0.orbit),
            ("ExternalAuditoryCanal", &self.0.external_auditory_canal),
            ("Occipital", &self.0.occipital),
            ("AnteriorC1Arch", &self.0.anterior_c1_arch),
            ("AnteriorDens", &self.0.anterior_dens),
            ("PosteriorDens", &self.0.posterior_dens),
            ("PosteriorHardPalate", &self.0.posterior_hard_palate),
            ("Chin", &self.0.chin),
            ("Manubrium", &self.0.manubrium),
        ] {
            let color = label_colors.get_or_new(label);
            for point in points.rows() {
                let p = painter.point(point);
                let p = p.add(painter.title(label));
                group = group.add(p.set("stroke", color).set("fill", color));
            }
        }
        Ok(group)
    }
}

/// Label text for each vertebra
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_TEXT])]
pub struct VertebralLabels<'a>(pub &'a LateralPoints);
impl NeckSagittalComponent for VertebralLabels<'_> {}
impl DrawComponent for VertebralLabels<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let mut group = self.default_group();
        let mut c2t1_corners = self.0.corners.0.clone();
        let dens = concatenate(
            Axis(0),
            &[self.0.anterior_dens.view(), self.0.posterior_dens.view()],
        )
        .unwrap();
        let c2 = c2t1_corners.index_axis(Axis(0), 0);
        let c2_bl_br = c2.slice(s![2.., ..]);

        // Use the middle point of the dens and c2 lower endplate as the pseudo c2 top endplate
        let c2_tl_tr = 2.0 * &dens / 3.0 + &c2_bl_br / 3.0;
        c2t1_corners
            .index_axis_mut(Axis(0), 0)
            .slice_mut(s![..2, ..])
            .assign(&c2_tl_tr);
        let c1_centroid = dens.mean_axis(Axis(0)).unwrap();
        let c2t1_centroids = Centroids::from(Corners(c2t1_corners));
        let c1t1_centroids =
            concatenate![Axis(0), c1_centroid.insert_axis(Axis(0)), c2t1_centroids];
        for (i, centroid) in c1t1_centroids.axis_iter(Axis(0)).enumerate() {
            let label = if i <= 6 {
                // C1 to C7
                format!("C{}", i + 1)
            } else {
                "T1".to_string()
            };
            let text = painter.text(&label, centroid, None, None);
            group = group.add(text);
        }
        Ok(group)
    }
}

#[derive(strum::EnumString, strum::Display, strum::VariantArray, ValueEnum, Debug, Copy, Clone)]
#[clap(rename_all = "PascalCase")]
pub enum NeckLateralMeasure {
    Adi,
    OC2,
    Sacs,
    WedgeAngle,
    ModifiedRenawatIndex,
    ThoracicInletAngle,
    NeckTilt,
    SpinoCranialAngle,
    OccipitocervicalInclination,
    CranialSlope,
    T1Slope,
    TPR,
    C2C7SVA,
    EACSVA,
    EndPlateAngle,
}

impl NeckLateralMeasure {
    pub fn all() -> Vec<Self> {
        NeckLateralMeasure::VARIANTS.to_vec()
    }
}

impl<'a> From<(&NeckLateralMeasure, &'a ScaledType<LateralPoints>)>
    for Box<dyn NeckMeasureComponent + 'a>
{
    fn from(value: (&NeckLateralMeasure, &'a ScaledType<LateralPoints>)) -> Self {
        let (measure, lateral_points) = value;
        let lateral_points = &lateral_points.0;
        match measure {
            NeckLateralMeasure::Adi => Box::new(Adi(lateral_points)),
            NeckLateralMeasure::OC2 => Box::new(OC2(lateral_points)),
            NeckLateralMeasure::Sacs => Box::new(Sacs(lateral_points)),
            NeckLateralMeasure::WedgeAngle => Box::new(WedgeAngle(lateral_points)),
            NeckLateralMeasure::ModifiedRenawatIndex => {
                Box::new(ModifiedRenawatIndex(lateral_points))
            }
            NeckLateralMeasure::ThoracicInletAngle => Box::new(ThoracicInletAngle(lateral_points)),
            NeckLateralMeasure::NeckTilt => Box::new(NeckTilt(lateral_points)),
            NeckLateralMeasure::SpinoCranialAngle => Box::new(SpinoCranialAngle(lateral_points)),
            NeckLateralMeasure::OccipitocervicalInclination => {
                Box::new(OccipitocervicalInclination(lateral_points))
            }
            NeckLateralMeasure::CranialSlope => Box::new(CranialSlope(lateral_points)),
            NeckLateralMeasure::T1Slope => Box::new(T1Slope(lateral_points)),
            NeckLateralMeasure::TPR => Box::new(TPR(lateral_points)),
            NeckLateralMeasure::C2C7SVA => Box::new(C2C7SVA(lateral_points)),
            NeckLateralMeasure::EACSVA => Box::new(EACSVA(lateral_points)),
            NeckLateralMeasure::EndPlateAngle => Box::new(EndPlateAngle(lateral_points)),
        }
    }
}

#[derive(
    strum::EnumString, strum::Display, strum::VariantArray, ValueEnum, Debug, Copy, Clone, PartialEq,
)]
#[clap(rename_all = "PascalCase")]
pub enum NeckLateralDraw {
    CervicalPoints,
    VertebralLabels,
    Adi,
    OC2,
    Sacs,
    WedgeAngle,
    ModifiedRenawatIndex,
    ThoracicInletAngle,
    NeckTilt,
    SpinoCranialAngle,
    OccipitocervicalInclination,
    CranialSlope,
    T1Tilt,
    TPR,
    C2C7SVA,
    EACSVA,
    EndPlateAngle,
}

impl NeckLateralDraw {
    pub fn all() -> Vec<Self> {
        NeckLateralDraw::VARIANTS.to_vec()
    }
}

impl<'a> From<(&NeckLateralDraw, &'a ScaledType<LateralPoints>)> for Box<dyn DrawComponent + 'a> {
    fn from(value: (&NeckLateralDraw, &'a ScaledType<LateralPoints>)) -> Self {
        let (draw, lateral_points) = value;
        let lateral_points = &lateral_points.0;
        match draw {
            NeckLateralDraw::CervicalPoints => Box::new(CervicalPoints(lateral_points)),
            NeckLateralDraw::VertebralLabels => Box::new(VertebralLabels(lateral_points)),
            NeckLateralDraw::Adi => Box::new(Adi(lateral_points)),
            NeckLateralDraw::OC2 => Box::new(OC2(lateral_points)),
            NeckLateralDraw::Sacs => Box::new(Sacs(lateral_points)),
            NeckLateralDraw::WedgeAngle => Box::new(WedgeAngle(lateral_points)),
            NeckLateralDraw::ModifiedRenawatIndex => Box::new(ModifiedRenawatIndex(lateral_points)),
            NeckLateralDraw::ThoracicInletAngle => Box::new(ThoracicInletAngle(lateral_points)),
            NeckLateralDraw::NeckTilt => Box::new(NeckTilt(lateral_points)),
            NeckLateralDraw::SpinoCranialAngle => Box::new(SpinoCranialAngle(lateral_points)),
            NeckLateralDraw::OccipitocervicalInclination => {
                Box::new(OccipitocervicalInclination(lateral_points))
            }
            NeckLateralDraw::CranialSlope => Box::new(CranialSlope(lateral_points)),
            NeckLateralDraw::T1Tilt => Box::new(T1Slope(lateral_points)),
            NeckLateralDraw::TPR => Box::new(TPR(lateral_points)),
            NeckLateralDraw::C2C7SVA => Box::new(C2C7SVA(lateral_points)),
            NeckLateralDraw::EACSVA => Box::new(EACSVA(lateral_points)),
            NeckLateralDraw::EndPlateAngle => Box::new(EndPlateAngle(lateral_points)),
        }
    }
}

pub type NeckLateralDrawArguments<'a> = DrawArguments<'a, LateralPoints, NeckLateralDraw>;

pub fn draw_neck(
    args: DrawArguments<LateralPoints, NeckLateralDraw>,
) -> Result<element::SVG, DrawError> {
    crate::draw::draw_on_image(args)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use anyhow::Result;
    use labelme_rs::LabelMeDataLine;
    use pretty_assertions::assert_eq;
    use std::path::{Path, PathBuf};

    use crate::draw::{CLASS_LINE, CLASS_MEASURE};
    use crate::{CoronalPointsLine, SagittalPointsLine};

    /// Test struct for Named derive
    #[derive(Named)]
    #[draw_type([CLASS_MEASURE, CLASS_LINE])]
    #[label("Label")]
    struct TestNamed<'a>(&'a str);

    #[test]
    fn test_derive_name() {
        let test_named = TestNamed("Test");
        assert_eq!(test_named.0, "Test");
        assert_eq!(test_named.id(), "TestNamed");
        assert_eq!(test_named.label(), "Label");
        assert_eq!(test_named.draw_type(), &[CLASS_MEASURE, CLASS_LINE]);
        assert_eq!(
            test_named.description().unwrap(),
            "Test struct for Named derive"
        );
    }

    /// Test round trip conversion between LabelMeDataLine and T
    fn test_conversion<T>(data_dir: &Path, filename: &str) -> Result<()>
    where
        T: TryConvertContentFilename<LabelMeDataLine> + Clone + PartialEq + std::fmt::Debug,
        <T as TryConvertContentFilename<labelme_rs::LabelMeDataLine>>::Error:
            std::error::Error + std::marker::Send + std::marker::Sync + std::fmt::Debug + 'static,
        LabelMeData: std::convert::From<<T as ContentFilename>::ContentType>,
    {
        let json = std::fs::read_to_string(data_dir.join(filename))?;
        let original_data = LabelMeData::try_from(json.as_str())?;
        let original_data_line = LabelMeDataLine::new(original_data.clone(), filename.to_string());
        let lateral_points_line = T::try_convert_from(original_data_line.clone())?;
        let data_line2 = LabelMeDataLine::try_convert_from(lateral_points_line.clone())?;
        let lateral_point_line2 = T::try_convert_from(data_line2.clone())?;
        assert_eq!(lateral_points_line, lateral_point_line2);
        let data_line3 = LabelMeDataLine::try_convert_from(lateral_point_line2.clone())?;
        assert_eq!(data_line2, data_line3);
        Ok(())
    }

    #[test]
    fn conversion_neck() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");
        for filename in [
            "neck_case1/lateral.json",
            "neck_case2/extension_lateral.json",
            "neck_case2/flexion_lateral.json",
        ]
        .iter()
        {
            test_conversion::<LateralPointsLine>(&data_dir, filename)?;
        }
        Ok(())
    }

    #[test]
    fn conversion_scoliosis_frontal() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");
        for filename in [
            "case1/frontal.json",
            "case2/frontal.json",
            "case3/frontal.json",
            "case4/frontal.json",
        ]
        .iter()
        {
            test_conversion::<CoronalPointsLine>(&data_dir, filename)?;
        }
        Ok(())
    }

    #[test]
    fn conversion_scoliosis_sagittal() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");
        for filename in [
            "case1/lateral.json",
            "case2/lateral.json",
            "case3/lateral.json",
            "case4/lateral.json",
        ]
        .iter()
        {
            test_conversion::<SagittalPointsLine>(&data_dir, filename)?;
        }
        Ok(())
    }
}
