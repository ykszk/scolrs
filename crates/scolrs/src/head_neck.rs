use crate::{
    extract_points, impl_named_w_lifetime_for, ColorPalette, CommonComponent, DrawComponent,
    HasCornerPoints, MeasureComponent, MeasureError, Named, Painter, ScolError, CLASS_ANNOTATION,
    CLASS_LINE, CLASS_POINT, CORNER_LABELS,
};
use clap::{self, ValueEnum};
use serde::{Deserialize, Serialize};

use labelme_rs::LabelMeData;
use ndarray::{concatenate, s, stack, Array1, Array2, Array3, ArrayView2, Axis};
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

    pub brow: Array1<f32>,
    pub sella: Array1<f32>,
    pub orbit: Array1<f32>,
    pub external_auditory_canal: Array1<f32>,
    pub occipital: Array1<f32>,
    pub anterior_c1_arch: Array1<f32>,
    pub anterior_dens: Array1<f32>,
    pub posterior_dens: Array1<f32>,
    pub posterior_hard_palate: Array1<f32>,
    pub chin: Array1<f32>,
    pub manubrium: Array1<f32>,
}

impl LateralPoints {
    fn extract_optional_point(data: &LabelMeData, label: &str) -> Result<Array1<f32>, ScolError> {
        let points = extract_points(data, label)?;
        if points.is_empty() {
            Ok(Array1::zeros(0))
        } else {
            Ok(points.index_axis(Axis(0), 0).to_owned())
        }
    }
}

impl TryFrom<&LabelMeData> for LateralPoints {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let corners = VertebralCornerPoints::try_from(data)?;
        let lamina = extract_points(data, "Lamina")?;
        let brow = Self::extract_optional_point(data, "Brow")?;
        let sella = Self::extract_optional_point(data, "Sella")?;
        let orbit = Self::extract_optional_point(data, "Orbit")?;
        let external_auditory_canal = Self::extract_optional_point(data, "ExternalAuditoryCanal")?;
        let occipital = Self::extract_optional_point(data, "Occipital")?;
        let anterior_c1_arch = Self::extract_optional_point(data, "AnteriorC1Arch")?;
        let anterior_dens = Self::extract_optional_point(data, "AnteriorDens")?;
        let posterior_dens = Self::extract_optional_point(data, "PosteriorDens")?;
        let posterior_hard_palate = Self::extract_optional_point(data, "PosteriorHardPalate")?;
        let chin = Self::extract_optional_point(data, "Chin")?;
        let manubrium = Self::extract_optional_point(data, "Manubrium")?;

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

pub struct C1Sac<'a>(pub &'a LateralPoints);
impl_named_w_lifetime_for!(C1Sac, &[CLASS_ANNOTATION, CLASS_LINE]);
impl<'a> NeckSagittalComponent for C1Sac<'a> {}
impl<'a> DrawComponent for C1Sac<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let mut group = self.default_group();
        let lamina = self.0.lamina.index_axis(Axis(0), 0);
        let c1 = self.0.corners.0.index_axis(Axis(0), 0);
        let c1_tr_br = c1.slice(s![1..;2, ..]);
        let c1_posterior_center = c1_tr_br.mean_axis(Axis(0)).unwrap();
        let points = stack![Axis(0), lamina, c1_posterior_center];
        let color = line_colors.get_or_new("C1Sac");
        let line = painter.polyline(points.view());
        group = group.add(line.set("stroke", color));
        Ok(group)
    }
}

impl<'a> MeasureComponent for C1Sac<'a> {
    fn measure(&self) -> Result<f32, MeasureError> {
        let points = &self.0.anterior_c1_arch;
        // let len = points.len();
        // if len < 2 {
        //     return Err(MeasureError::InsufficientPoints(len));
        // }
        Ok(points.len() as f32)
    }
}

/// Four corner points of each vertebra
pub struct CervicalPoints<'a>(pub &'a LateralPoints);
impl_named_w_lifetime_for!(CervicalPoints, &[CLASS_ANNOTATION, CLASS_POINT]);
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

pub struct LaminalPoints<'a>(pub &'a Array2<f32>);
impl_named_w_lifetime_for!(LaminalPoints, &[CLASS_ANNOTATION, CLASS_POINT]);
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

pub struct OptionalPoints<'a>(pub &'a LateralPoints);
impl_named_w_lifetime_for!(OptionalPoints, &[CLASS_ANNOTATION, CLASS_POINT]);
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
                let p = painter.point(points.view());
                group = group.add(p.set("stroke", color).set("fill", color));
            }
        }
        Ok(group)
    }
}
