use crate::{
    angle_from_lines, distanced_pair3, extract_points, points2line, Centroids, CobbAux,
    ColorPalette, CommonComponent, Corners, DrawComponent, DrawCorners, HasCornerPoints, L2Norm,
    MeasureError, Named, Painter, ScolError, ValidateLength, CLASS_ANGLE, CLASS_ANNOTATION,
    CLASS_DISTANCE, CLASS_LINE, CLASS_MEASURE, CLASS_POINT, CLASS_TEXT, CORNER_LABELS,
};
use clap::{self, ValueEnum};
use lyon_geom::point;
use named_derive::Named;
use serde::{Deserialize, Serialize};

use labelme_rs::LabelMeData;
use ndarray::{concatenate, s, stack, Array, Array2, Array3, ArrayView2, Axis};
use strum::VariantArray;
use svg::node::element;

#[derive(Debug, Clone)]
pub struct VertebralCornerPoints(pub Array3<f32>);

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
    pub lamina: Array2<f32>,

    pub brow: Array2<f32>,
    pub sella: Array2<f32>,
    pub orbit: Array2<f32>,
    pub external_auditory_canal: Array2<f32>,
    pub occipital: Array2<f32>,
    pub anterior_c1_arch: Array2<f32>,
    pub anterior_dens: Array2<f32>,
    pub posterior_dens: Array2<f32>,
    pub posterior_hard_palate: Array2<f32>,
    pub chin: Array2<f32>,
    pub manubrium: Array2<f32>,
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
        })
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
pub enum NeckSagittalMeasure {
    C1Sac,
    C2Sac,
}

const NEKC_SAGITTAL_COMPONENT_CLASS: &str = "NeckSagittalComponent";
pub trait NeckSagittalComponent: DrawComponent {
    fn default_group(&self) -> element::Group {
        self.default_group_w_classes(&[NEKC_SAGITTAL_COMPONENT_CLASS])
    }
}

pub trait NeckMeasureComponent: Named {
    fn measure(&self) -> Result<Vec<f32>, MeasureError>;
}

/// All sacral available spaces
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_LINE, CLASS_DISTANCE])]
pub struct Sacs<'a>(pub &'a LateralPoints);
impl<'a> NeckSagittalComponent for Sacs<'a> {}
impl<'a> DrawComponent for Sacs<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        self.0.posterior_dens.validate_length(1)?;
        self.0.lamina.validate_length(8)?;
        let color = line_colors.get_or_new(self.name());
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
            let line = painter.line(points.view());
            group = group.add(line);
            let (line_p1, line_p2) = distanced_pair3(
                tr_br.index_axis(Axis(0), 0),
                tr_br.index_axis(Axis(0), 1),
                projected_point.view(),
            );
            let line = painter.line(stack![Axis(0), line_p1, line_p2].view());
            group = group.add(line);
        }
        Ok(group)
    }
}

impl<'a> NeckMeasureComponent for Sacs<'a> {
    fn measure(&self) -> Result<Vec<f32>, MeasureError> {
        self.0.posterior_dens.validate_length(1)?;
        self.0.lamina.validate_length(8)?;
        let mut lengths: Vec<f32> = Vec::new();
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
    ) -> Result<element::Group, MeasureError> {
        let color = line_colors.get_or_new(self.name());
        let mut group = self.default_group().set("stroke", color);
        self.prep()?;
        let points = stack![
            Axis(0),
            self.0.anterior_dens.index_axis(Axis(0), 0).view(),
            self.0.anterior_c1_arch.index_axis(Axis(0), 0).view()
        ];
        let line = painter.line(points.view());
        group = group.add(line);
        Ok(group)
    }
}
impl<'a> NeckMeasureComponent for Adi<'a> {
    fn measure(&self) -> Result<Vec<f32>, MeasureError> {
        self.prep()?;
        let diff = &self.0.anterior_dens.index_axis(Axis(0), 0)
            - &self.0.anterior_c1_arch.index_axis(Axis(0), 0);
        Ok(vec![diff.l2norm()])
    }
}

/// O-C2 Angle
/// Angle between McGregor's line and C2 lower endplate
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct OC2<'a>(pub &'a LateralPoints);
impl<'a> NeckSagittalComponent for OC2<'a> {}
impl OC2<'_> {
    fn prep(&self) -> Result<(Array2<f32>, Array2<f32>), MeasureError> {
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
    ) -> Result<element::Group, MeasureError> {
        let color = line_colors.get_or_new(self.name());
        let mut group = self.default_group().set("stroke", color);
        let (mcgregor_points, c2_lower_endplate) = self.prep()?;
        let line = painter.line(mcgregor_points.view());
        group = group.add(line);
        let line = painter.line(c2_lower_endplate.view());
        let c2_length = c2_lower_endplate.index_axis(Axis(0), 0).l2norm();
        group = painter.cobb_from_plates(
            group,
            mcgregor_points,
            c2_lower_endplate.to_owned(),
            &CobbAux {
                plate_scale: 0.5,
                ..Default::default()
            },
            c2_length,
            Some(self.name()),
        );
        group = group.add(line);
        Ok(group)
    }
}
impl<'a> NeckMeasureComponent for OC2<'a> {
    fn measure(&self) -> Result<Vec<f32>, MeasureError> {
        let (mcgregor_points, c2_lower_endplate) = self.prep()?;
        let angle =
            angle_from_lines(mcgregor_points.view(), c2_lower_endplate.view()).unwrap_or_default();
        Ok(vec![angle])
    }
}

/// Wedge (intervertebral) angles
/// C2-C3 to C7-T1
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
    ) -> Result<element::Group, MeasureError> {
        let mut group = self.default_group();
        group = group.set("stroke", line_colors.get_or_new(self.name()));
        for i in 0..6 {
            let wedge_upper = self.0.corners.0.index_axis(Axis(0), i);
            let wedge_upper = wedge_upper.slice(s![2.., ..]);
            let wedge_lower = self.0.corners.0.index_axis(Axis(0), i + 1);
            let wedge_lower = wedge_lower.slice(s![..2, ..]);
            let wedge_length = wedge_upper.index_axis(Axis(0), 0).l2norm();
            group = painter.cobb_from_plates(
                group,
                wedge_upper.to_owned(),
                wedge_lower.to_owned(),
                &CobbAux {
                    plate_scale: 0.25,
                    ..Default::default()
                },
                wedge_length,
                Some(self.name()),
            );
        }
        Ok(group)
    }
}
impl<'a> NeckMeasureComponent for WedgeAngle<'a> {
    fn measure(&self) -> Result<Vec<f32>, MeasureError> {
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
    fn prep(&self) -> Result<Option<Array2<f32>>, MeasureError> {
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
    ) -> Result<element::Group, MeasureError> {
        let color = line_colors.get_or_new(self.name());
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
        }

        Ok(group)
    }
}
impl<'a> NeckMeasureComponent for ModifiedRenawatIndex<'a> {
    fn measure(&self) -> Result<Vec<f32>, MeasureError> {
        let intersection = self.prep()?;
        if let Some(intersection) = intersection {
            let diff = &intersection.index_axis(Axis(0), 0) - &intersection.index_axis(Axis(0), 1);
            Ok(vec![diff.l2norm()])
        } else {
            Ok(vec![0.0])
        }
    }
}

/// Four corner points of each vertebra
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POINT])]
pub struct CervicalPoints<'a>(pub &'a LateralPoints);
impl HasCornerPoints for CervicalPoints<'_> {
    fn top_left(&self) -> ArrayView2<f32> {
        self.0.corners.0.slice(s![1.., 0, ..])
    }
    fn top_right(&self) -> ArrayView2<f32> {
        self.0.corners.0.slice(s![1.., 1, ..])
    }
    fn bottom_left(&self) -> ArrayView2<f32> {
        self.0.corners.0.slice(s![.., 2, ..])
    }
    fn bottom_right(&self) -> ArrayView2<f32> {
        self.0.corners.0.slice(s![.., 3, ..])
    }
}
impl<'a> CommonComponent for CervicalPoints<'a> {}
impl<'a> DrawComponent for CervicalPoints<'a> {
    fn draw(
        &self,
        painter: &Painter,
        label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        self.draw_corners(painter, label_colors)
    }
}

#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POINT])]
pub struct OptionalPoints<'a>(pub &'a LateralPoints);
impl<'a> NeckSagittalComponent for OptionalPoints<'a> {}
impl<'a> DrawComponent for OptionalPoints<'a> {
    fn draw(
        &self,
        painter: &Painter,
        label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let mut group = self.default_group();
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
    ) -> Result<element::Group, MeasureError> {
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
            let text = painter.text(&label, centroid, None);
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
        }
    }
}

#[derive(strum::EnumString, strum::Display, strum::VariantArray, ValueEnum, Debug, Copy, Clone)]
#[clap(rename_all = "PascalCase")]
pub enum NeckLateralDraw {
    OptionalPoints,
    VertebralLabels,
    Adi,
    OC2,
    Sacs,
    WedgeAngle,
    ModifiedRenawatIndex,
}

impl NeckLateralDraw {
    pub fn all() -> Vec<Self> {
        NeckLateralDraw::VARIANTS.to_vec()
    }
}

impl<'a> From<(&NeckLateralDraw, &'a LateralPoints)> for Box<dyn NeckSagittalComponent + 'a> {
    fn from(value: (&NeckLateralDraw, &'a LateralPoints)) -> Self {
        let (draw, lateral_points) = value;
        match draw {
            NeckLateralDraw::OptionalPoints => Box::new(OptionalPoints(lateral_points)),
            NeckLateralDraw::VertebralLabels => Box::new(VertebralLabels(lateral_points)),
            NeckLateralDraw::Adi => Box::new(Adi(lateral_points)),
            NeckLateralDraw::OC2 => Box::new(OC2(lateral_points)),
            NeckLateralDraw::Sacs => Box::new(Sacs(lateral_points)),
            NeckLateralDraw::WedgeAngle => Box::new(WedgeAngle(lateral_points)),
            NeckLateralDraw::ModifiedRenawatIndex => Box::new(ModifiedRenawatIndex(lateral_points)),
        }
    }
}
