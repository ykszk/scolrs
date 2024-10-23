use crate::{
    distanced_pair3, extract_points, points2line, Centroids, CobbAux, ColorPalette,
    CommonComponent, Corners, DrawComponent, HasCornerPoints, L2Norm, MeasureComponent,
    MeasureError, Named, Painter, ScolError, CLASS_ANGLE, CLASS_ANNOTATION, CLASS_DISTANCE,
    CLASS_LINE, CLASS_MEASURE, CLASS_POINT, CLASS_TEXT, CORNER_LABELS,
};
use clap::{self, ValueEnum};
use named_derive::Named;
use serde::{Deserialize, Serialize};

use labelme_rs::LabelMeData;
use ndarray::{concatenate, s, stack, Array, Array2, Array3, ArrayView2, Axis};
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

// impl LateralPoints {
//     fn extract_optional_point(data: &LabelMeData, label: &str) -> Result<Array2<f32>, ScolError> {
//         let points = extract_points(data, label)?;
//         if points.is_empty() {
//             Ok(Array1::zeros(0))
//         } else {
//             Ok(points.index_axis(Axis(0), 0).to_owned())
//         }
//     }
// }

impl TryFrom<&LabelMeData> for LateralPoints {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let corners = VertebralCornerPoints::try_from(data)?;
        let lamina = extract_points(data, "Lamina")?;
        if lamina.shape()[0] != 8 {
            return Err(ScolError::InvalidPointCount(
                "Lamina should have 8 points".into(),
                lamina.shape()[0],
            ));
        }
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

trait ValidateLength {
    fn validate_length(&self, expected_len: usize) -> Result<(), MeasureError>;
}

impl ValidateLength for Array2<f32> {
    fn validate_length(&self, expected_len: usize) -> Result<(), MeasureError> {
        if self.len_of(Axis(0)) != expected_len {
            return Err(MeasureError::InvalidNumberOfPoints(
                crate::InvalidNumberOfPoints::IncorrectNumberOfPoints(expected_len, self.len()),
            ));
        }
        Ok(())
    }
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

impl<'a> MeasureComponent for Sacs<'a> {
    fn measure(&self) -> Result<f32, MeasureError> {
        // let lamina = self.0.lamina.index_axis(Axis(0), 0);
        // let posterior_dens = self.0.posterior_dens.index_axis(Axis(0), 0);
        // let distance = painter::distance(&lamina, &posterior_dens);
        Ok(0.0)
    }
}

/// Atlanto-dental interval
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_LINE, CLASS_DISTANCE])]
pub struct Adi<'a>(pub &'a LateralPoints);
impl<'a> NeckSagittalComponent for Adi<'a> {}
impl<'a> DrawComponent for Adi<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        self.0.anterior_dens.validate_length(1)?;
        self.0.anterior_c1_arch.validate_length(1)?;
        let color = line_colors.get_or_new(self.name());
        let mut group = self.default_group().set("stroke", color);
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

/// O-C2 Angle
/// Angle between McGregor's line and C2 lower endplate
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct OC2<'a>(pub &'a LateralPoints);
impl<'a> NeckSagittalComponent for OC2<'a> {}
impl<'a> DrawComponent for OC2<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        self.0.occipital.validate_length(1)?;
        self.0.posterior_hard_palate.validate_length(1)?;
        let color = line_colors.get_or_new(self.name());
        let mut group = self.default_group().set("stroke", color);
        let mcgregor_points = stack![
            Axis(0),
            self.0.posterior_hard_palate.index_axis(Axis(0), 0).view(),
            self.0.occipital.index_axis(Axis(0), 0).view(),
        ];
        let line = painter.line(mcgregor_points.view());
        group = group.add(line);
        let c2 = self.0.corners.0.index_axis(Axis(0), 0);
        let c2_lower_endplate = c2.slice(s![2.., ..]);
        let line = painter.line(c2_lower_endplate);
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

#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POINT])]
pub struct LaminalPoints<'a>(pub &'a Array2<f32>);
impl<'a> NeckSagittalComponent for LaminalPoints<'a> {}
impl<'a> DrawComponent for LaminalPoints<'a> {
    fn draw(
        &self,
        painter: &Painter,
        label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let color = label_colors.get_or_new("Lamina");
        let mut group = self.default_group().set("stroke", color).set("fill", color);
        for point in self.0.rows() {
            let p = painter.point(point);
            group = group.add(p);
        }
        Ok(group)
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
            if !points.is_empty() {
                let color = label_colors.get_or_new(label);
                for point in points.rows() {
                    let p = painter.point(point);
                    group = group.add(p.set("stroke", color).set("fill", color));
                }
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
            let text = painter.text(&format!("{}", i + 1), centroid, None);
            group = group.add(text);
        }
        Ok(group)
    }
}
