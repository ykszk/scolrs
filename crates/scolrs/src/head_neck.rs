use std::ops::MulAssign;

use crate::{
    angle_between, angle_from_lines, array2_to_vec_points, array3_to_nested_vec, create_shapes,
    distanced_pair3, draw_incidence_angle, extract_points, femoral_incidence_angle,
    nested_vec_to_array3, points2line, vec_points_to_array2, Centroids, CobbAux, ColorPalette,
    ContentFilename, Corners, DrawComponent, DrawCorners, DrawError, HasCornerPoints,
    HasImageMetadata, ImageMetadata, L2Norm, MeasureError, Named, Painter, Point2d, ScolError,
    TryFromJson, ValidateLength, CLASS_ANGLE, CLASS_ANNOTATION, CLASS_DISTANCE, CLASS_LINE,
    CLASS_MEASURE, CLASS_POINT, CLASS_TEXT, CORNER_LABELS,
};
use clap::{self, ValueEnum};
use lyon_geom::point;
use named_derive::{Named, TryFromJsonStr};

use labelme_rs::LabelMeData;
use ndarray::{concatenate, s, stack, Array, Array2, Array3, ArrayView1, ArrayView2, Axis};
use ndarray_stats::DeviationExt;
use serde::{Deserialize, Serialize};
use strum::VariantArray;
use svg::node::element;

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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

impl LateralPoints {
    // Scale point coordinates using image_data.spacing_xy
    pub fn scale(&mut self) {
        if self.image_metadata.spacing_xy == (1.0, 1.0) {
            return;
        }
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
    }
}

impl TryFromJson for LateralPoints {
    type Error = ScolError;

    fn try_from_ir_json(json: &str) -> Result<Self, Self::Error> {
        let lateral_points_ir: LateralPointsIR = serde_json::from_str(json)?;
        let lp = LateralPoints::try_from(&lateral_points_ir)?;
        Ok(lp)
    }

    fn try_from_labelme_json(json: &str) -> Result<Self, Self::Error> {
        let data: LabelMeData = serde_json::from_str(json)?;
        let lp = LateralPoints::try_from(&data)?;
        Ok(lp)
    }
}

impl TryFrom<&LabelMeData> for LateralPoints {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let corners = VertebralCornerPoints::try_from(data)?;
        let lamina = extract_points(data, "Lamina")?;
        let brow = extract_points(data, "Brow")?;
        let sella = extract_points(data, "Sella")?;
        let orbit = extract_points(data, "Orbit")?;
        let external_auditory_canal = extract_points(data, "ExternalAuditoryCanal")?;
        let occipital = extract_points(data, "Occipital")?;
        let anterior_c1_arch = extract_points(data, "AnteriorC1Arch")?;
        let anterior_dens = extract_points(data, "AnteriorDens")?;
        let posterior_dens = extract_points(data, "PosteriorDens")?;
        let posterior_hard_palate = extract_points(data, "PosteriorHardPalate")?;
        let chin = extract_points(data, "Chin")?;
        let manubrium = extract_points(data, "Manubrium")?;

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

impl TryFrom<&LateralPoints> for LabelMeData {
    type Error = ndarray::ShapeError;

    fn try_from(lateral_points: &LateralPoints) -> Result<Self, Self::Error> {
        let lateral_points_ir = LateralPointsIR::from(lateral_points);
        lateral_points_ir.try_into()
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
#[derive(
    Serialize, Deserialize, Default, Clone, Debug, PartialEq, TryFromJsonStr, HasImageMetadata,
)]
pub struct LateralPointsIR {
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

#[derive(
    Serialize,
    Deserialize,
    Default,
    Clone,
    Debug,
    PartialEq,
    named_derive::ContentFilename,
    TryFromJsonStr,
)]
pub struct LateralPointsIRLine {
    pub content: LateralPointsIR,
    pub filename: String,
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
            corners: array3_to_nested_vec(lateral_points.corners.0.to_owned()),
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

impl TryFrom<LabelMeData> for LateralPointsIR {
    type Error = ScolError;

    fn try_from(data: LabelMeData) -> Result<Self, Self::Error> {
        let lateral_points = LateralPoints::try_from(&data)?;
        let lateral_points_ir = LateralPointsIR::from(&lateral_points);
        Ok(lateral_points_ir)
    }
}

impl TryFrom<LateralPointsIR> for LabelMeData {
    type Error = ndarray::ShapeError;

    fn try_from(lateral_points_ir: LateralPointsIR) -> Result<Self, Self::Error> {
        let lateral_points = LateralPoints::try_from(&lateral_points_ir)?;
        let mut data = LabelMeData {
            imagePath: lateral_points_ir.image_metadata.path,
            imageHeight: lateral_points_ir.image_metadata.height,
            imageWidth: lateral_points_ir.image_metadata.width,
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
            ("Lamina", lateral_points_ir.lamina.clone()),
            ("Brow", lateral_points_ir.brow.clone()),
            ("Sella", lateral_points_ir.sella.clone()),
            ("Orbit", lateral_points_ir.orbit.clone()),
            (
                "ExternalAuditoryCanal",
                lateral_points_ir.external_auditory_canal.clone(),
            ),
            ("Occipital", lateral_points_ir.occipital.clone()),
            ("AnteriorC1Arch", lateral_points_ir.anterior_c1_arch.clone()),
            ("AnteriorDens", lateral_points_ir.anterior_dens.clone()),
            ("PosteriorDens", lateral_points_ir.posterior_dens.clone()),
            (
                "PosteriorHardPalate",
                lateral_points_ir.posterior_hard_palate.clone(),
            ),
            ("Chin", lateral_points_ir.chin.clone()),
            ("Manubrium", lateral_points_ir.manubrium.clone()),
        ]);

        Ok(data)
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
impl<'a> NeckSagittalComponent for Sacs<'a> {}
impl<'a> DrawComponent for Sacs<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        self.0.posterior_dens.validate_length(1)?;
        self.0.lamina.validate_length(8)?;
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

impl<'a> NeckMeasureComponent for Sacs<'a> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        self.0.posterior_dens.validate_length(1)?;
        self.0.lamina.validate_length(8)?;
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
impl<'a> NeckSagittalComponent for Adi<'a> {}
impl Adi<'_> {
    fn prep(&self) -> Result<(), MeasureError> {
        self.0.anterior_dens.validate_length(1)?;
        self.0.anterior_c1_arch.validate_length(1)?;
        Ok(())
    }
}
impl<'a> DrawComponent for Adi<'a> {
    fn draw(
        &self,
        painter: &Painter,
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
impl<'a> NeckMeasureComponent for Adi<'a> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        self.prep()?;
        let diff = &self.0.anterior_dens.index_axis(Axis(0), 0)
            - &self.0.anterior_c1_arch.index_axis(Axis(0), 0);
        Ok(vec![diff.l2norm()])
    }
}

/// Occiput-C2 Angle
/// Angle between McGregor's line and C2 lower endplate
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct OC2<'a>(pub &'a LateralPoints);
impl<'a> NeckSagittalComponent for OC2<'a> {}
impl OC2<'_> {
    fn prep(&self) -> Result<(Array2<f64>, Array2<f64>), MeasureError> {
        self.0.occipital.validate_length(1)?;
        self.0.posterior_hard_palate.validate_length(1)?;
        let mcgregor_points = stack![
            Axis(0),
            self.0.posterior_hard_palate.index_axis(Axis(0), 0).view(),
            self.0.occipital.index_axis(Axis(0), 0).view(),
        ];
        let c2 = self.0.corners.0.index_axis(Axis(0), 0);
        let c2_lower_endplate = c2.slice(s![2.., ..]).to_owned();
        Ok((mcgregor_points, c2_lower_endplate))
    }
}
impl<'a> DrawComponent for OC2<'a> {
    fn draw(
        &self,
        painter: &Painter,
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
            mcgregor_points,
            c2_lower_endplate.to_owned(),
            &CobbAux {
                plate_scale: 5.0,
                ..Default::default()
            },
            c2_length,
            Some(self.id()),
        );
        group = group.add(line);
        Ok(group)
    }
}
impl<'a> NeckMeasureComponent for OC2<'a> {
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
impl<'a> NeckSagittalComponent for WedgeAngle<'a> {}
impl<'a> DrawComponent for WedgeAngle<'a> {
    fn draw(
        &self,
        painter: &Painter,
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
                wedge_upper.to_owned(),
                wedge_lower.to_owned(),
                &CobbAux {
                    plate_scale: 1.5,
                    ..Default::default()
                },
                mean_wedge_length,
                Some(self.id()),
            );
        }
        Ok(group)
    }
}
impl<'a> NeckMeasureComponent for WedgeAngle<'a> {
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
impl<'a> NeckSagittalComponent for ModifiedRenawatIndex<'a> {}
impl<'a> ModifiedRenawatIndex<'a> {
    /// Array2 of [intersection, c2_lower_middle]
    fn prep(&self) -> Result<Option<Array2<f64>>, MeasureError> {
        self.0.anterior_c1_arch.validate_length(1)?;
        // posterior_c2_arch?
        self.0.posterior_dens.validate_length(1)?;
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
impl<'a> DrawComponent for ModifiedRenawatIndex<'a> {
    fn draw(
        &self,
        painter: &Painter,
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
impl<'a> NeckMeasureComponent for ModifiedRenawatIndex<'a> {
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
impl<'a> NeckSagittalComponent for ThoracicInletAngle<'a> {}
impl<'a> ThoracicInletAngle<'a> {
    fn prep(&self) -> Result<Array2<f64>, MeasureError> {
        self.0.manubrium.validate_length(1)?;
        self.0.corners.0.validate_length(7)?;
        let t1 = self.0.corners.0.index_axis(Axis(0), 6);
        let t1_top_plate = t1.slice(s![..2, ..]);
        Ok(t1_top_plate.to_owned())
    }
}
impl<'a> DrawComponent for ThoracicInletAngle<'a> {
    fn draw(
        &self,
        painter: &Painter,
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
impl<'a> NeckMeasureComponent for ThoracicInletAngle<'a> {
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
impl<'a> NeckSagittalComponent for NeckTilt<'a> {}
impl<'a> NeckTilt<'a> {
    fn prep(&self) -> Result<(Array2<f64>, Array2<f64>, f64), MeasureError> {
        self.0.manubrium.validate_length(1)?;
        self.0.corners.0.validate_length(7)?;
        let t1 = self.0.corners.0.index_axis(Axis(0), 6);
        let t1_top_plate = t1.slice(s![..2, ..]);
        let t1_top_middle = t1_top_plate.mean_axis(Axis(0)).unwrap();
        let manubrium_to_t1 = stack![
            Axis(0),
            self.0.manubrium.index_axis(Axis(0), 0).view(),
            t1_top_middle
        ];
        let mut v_line_from_manubrium = manubrium_to_t1.clone();
        v_line_from_manubrium[[1, 0]] = manubrium_to_t1[[0, 0]];
        let angle =
            angle_between(v_line_from_manubrium.view(), manubrium_to_t1.view()).to_degrees();
        Ok((manubrium_to_t1, v_line_from_manubrium, angle))
    }
}
impl<'a> DrawComponent for NeckTilt<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let mut group = self.default_group().set("stroke", color);
        let (manubrium_to_t1, v_line_from_manubrium, angle) = self.prep()?;
        let line = painter.line(manubrium_to_t1.view());
        group = group.add(line);
        let line = painter.line(v_line_from_manubrium.view());
        group = group.add(line);

        let text = format!("{:.1}°", angle);
        let text = painter.text(
            &text,
            self.0.manubrium.index_axis(Axis(0), 0),
            Some(self.id()),
            None,
        );
        group = group.add(text);
        Ok(group)
    }
}
impl NeckMeasureComponent for NeckTilt<'_> {
    fn measure(&self) -> Result<Vec<f64>, MeasureError> {
        let (_, _, angle) = self.prep()?;
        Ok(vec![angle])
    }
}

/// Four corner points of each vertebra
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POINT])]
pub struct CervicalPoints<'a>(pub &'a LateralPoints);
impl HasCornerPoints for CervicalPoints<'_> {
    fn top_left(&self) -> ArrayView2<f64> {
        self.0.corners.0.slice(s![1.., 0, ..])
    }
    fn top_right(&self) -> ArrayView2<f64> {
        self.0.corners.0.slice(s![1.., 1, ..])
    }
    fn bottom_left(&self) -> ArrayView2<f64> {
        self.0.corners.0.slice(s![.., 2, ..])
    }
    fn bottom_right(&self) -> ArrayView2<f64> {
        self.0.corners.0.slice(s![.., 3, ..])
    }
}
impl<'a> NeckSagittalComponent for CervicalPoints<'a> {}
impl<'a> DrawComponent for CervicalPoints<'a> {
    fn draw(
        &self,
        painter: &Painter,
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
impl<'a> NeckSagittalComponent for VertebralLabels<'a> {}
impl<'a> DrawComponent for VertebralLabels<'a> {
    fn draw(
        &self,
        painter: &Painter,
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
}

impl NeckLateralMeasure {
    pub fn all() -> Vec<Self> {
        NeckLateralMeasure::VARIANTS.to_vec()
    }
}

impl<'a> From<(&NeckLateralMeasure, &'a LateralPoints)> for Box<dyn NeckMeasureComponent + 'a> {
    fn from(value: (&NeckLateralMeasure, &'a LateralPoints)) -> Self {
        let (measure, lateral_points) = value;
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
}

impl NeckLateralDraw {
    pub fn all() -> Vec<Self> {
        NeckLateralDraw::VARIANTS.to_vec()
    }
}

impl<'a> From<(&NeckLateralDraw, &'a LateralPoints)> for Box<dyn DrawComponent + 'a> {
    fn from(value: (&NeckLateralDraw, &'a LateralPoints)) -> Self {
        let (draw, lateral_points) = value;
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
        }
    }
}

/// Test struct for Named derive
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_LINE])]
#[label("Label")]
pub struct TestNamed<'a>(pub &'a str);

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use anyhow::Result;
    use labelme_rs::LabelMeDataLine;
    use pretty_assertions::assert_eq;
    use std::path::PathBuf;

    use crate::{CoronalPointsIRLine, SagittalPointsIRLine, CLASS_LINE, CLASS_MEASURE};

    #[test]
    fn test_derive_name() {
        let test_named = super::TestNamed("Test");
        assert_eq!(test_named.id(), "TestNamed");
        assert_eq!(test_named.label(), "Label");
        assert_eq!(test_named.draw_type(), &[CLASS_MEASURE, CLASS_LINE]);
        assert_eq!(
            test_named.description().unwrap(),
            "Test struct for Named derive"
        );
    }

    #[test]
    fn test_conversion_neck() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");
        for filename in [
            "neck_case1/lateral.json",
            "neck_case2/extension_lateral.json",
            "neck_case2/flexion_lateral.json",
        ]
        .iter()
        {
            let json = std::fs::read_to_string(data_dir.join(filename))?;
            let original_data = LabelMeData::try_from(json.as_str())?;
            let original_data_line =
                LabelMeDataLine::new(original_data.clone(), filename.to_string());
            let lateral_points_line =
                LateralPointsIRLine::try_convert_from(original_data_line.clone())?;
            let data_line2 = LabelMeDataLine::try_convert_from(lateral_points_line.clone())?;
            let lateral_point_line2 = LateralPointsIRLine::try_convert_from(data_line2.clone())?;
            assert_eq!(lateral_points_line, lateral_point_line2);
            let data_line3 = LabelMeDataLine::try_convert_from(lateral_point_line2.clone())?;
            assert_eq!(data_line2, data_line3);
        }
        Ok(())
    }

    #[test]
    fn test_conversion_scoliosis_frontal() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");
        for filename in [
            "case1/frontal.json",
            "case2/frontal.json",
            "case3/frontal.json",
            "case4/frontal.json",
        ]
        .iter()
        {
            let json = std::fs::read_to_string(data_dir.join(filename))?;
            let original_data = LabelMeData::try_from(json.as_str())?;
            let original_data_line =
                LabelMeDataLine::new(original_data.clone(), filename.to_string());
            let coronal_points_line =
                CoronalPointsIRLine::try_convert_from(original_data_line.clone())?;
            let data_line2 = LabelMeDataLine::try_convert_from(coronal_points_line.clone())?;
            let coronal_poins_line2 = CoronalPointsIRLine::try_convert_from(data_line2.clone())?;
            assert_eq!(coronal_points_line, coronal_poins_line2);
            let data_line3 = LabelMeDataLine::try_convert_from(coronal_poins_line2.clone())?;
            assert_eq!(data_line2, data_line3);
        }
        Ok(())
    }

    #[test]
    fn test_conversion_scoliosis_sagittal() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");
        for filename in [
            "case1/lateral.json",
            "case2/lateral.json",
            "case3/lateral.json",
            "case4/lateral.json",
        ]
        .iter()
        {
            let json = std::fs::read_to_string(data_dir.join(filename))?;
            let original_data = LabelMeData::try_from(json.as_str())?;
            let original_data_line =
                LabelMeDataLine::new(original_data.clone(), filename.to_string());
            let lateral_points_line =
                SagittalPointsIRLine::try_convert_from(original_data_line.clone())?;
            let data_line2 = LabelMeDataLine::try_convert_from(lateral_points_line.clone())?;
            let lateral_poins_line2 = SagittalPointsIRLine::try_convert_from(data_line2.clone())?;
            assert_eq!(lateral_points_line, lateral_poins_line2);
            let data_line3 = LabelMeDataLine::try_convert_from(lateral_poins_line2.clone())?;
            assert_eq!(data_line2, data_line3);
        }
        Ok(())
    }
}
