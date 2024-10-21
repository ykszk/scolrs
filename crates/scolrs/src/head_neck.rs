use labelme_rs::LabelMeData;
use ndarray::{concatenate, stack, Array1, Array2, Array3, Axis};

use crate::{extract_points, ScolError, CORNER_LABELS};

#[derive(Debug, Clone)]
pub struct CornerPoints(pub Array3<f32>);

impl TryFrom<&LabelMeData> for CornerPoints {
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
        Ok(CornerPoints(verts))
    }
}

#[derive(Debug, Clone)]
pub struct LateralPoints {
    pub corners: CornerPoints,
    pub lamina: Array2<f32>,

    pub brow: Array1<f32>,
    pub sella: Array1<f32>,
    pub orbit: Array1<f32>,
    pub external_auditory_canal: Array1<f32>,
    pub occipital: Array1<f32>,
    pub anterior_c1_arch: Array1<f32>,
    pub anterior_dense: Array1<f32>,
    pub posterior_dense: Array1<f32>,
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
        let corners = CornerPoints::try_from(data)?;
        let lamina = extract_points(data, "Lamina")?;
        let brow = Self::extract_optional_point(data, "Brow")?;
        let sella = Self::extract_optional_point(data, "Sella")?;
        let orbit = Self::extract_optional_point(data, "Orbit")?;
        let external_auditory_canal = Self::extract_optional_point(data, "ExternalAuditoryCanal")?;
        let occipital = Self::extract_optional_point(data, "Occipital")?;
        let anterior_c1_arch = Self::extract_optional_point(data, "AnteriorC1Arch")?;
        let anterior_dense = Self::extract_optional_point(data, "AnteriorDense")?;
        let posterior_dense = Self::extract_optional_point(data, "PosteriorDense")?;
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
            anterior_dense,
            posterior_dense,
            posterior_hard_palate,
            chin,
            manubrium,
        })
    }
}
