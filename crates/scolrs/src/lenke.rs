use std::fmt::Display;

use serde::{Deserialize, Serialize};

use crate::VertebralIndex;

use super::{CoronalPoints, Curve, CurveSet, Spine};

const FRONTAL_ANGLE_THRESH: f64 = 25.0_f64;
const BEND_ANGLE_THRESH: f64 = 25.0_f64;
const LATERAL_ANGLE_THRESH: f64 = 20.0_f64;

pub const T2T5_CURVE: Curve = Curve {
    sup: VertebralIndex::T2 as usize,
    inf: VertebralIndex::T5 as usize,
};

pub const T5T12_CURVE: Curve = Curve {
    sup: VertebralIndex::T5 as usize,
    inf: VertebralIndex::T12 as usize,
};

pub const T10L2_CURVE: Curve = Curve {
    sup: VertebralIndex::T10 as usize,
    inf: VertebralIndex::L2 as usize,
};

/// Curve position for determining if PT is structural
pub const KYOPHOSIS_CURVE_PT: Curve = T2T5_CURVE;
/// Curve position for determining if MT is structural
pub const KYOPHOSIS_CURVE_MT: Curve = T10L2_CURVE;
/// Curve position for determining if TLL is structural
pub const KYOPHOSIS_CURVE_TLL: Curve = T10L2_CURVE;

/// Major curve in Lenke classification
/// PT can't be the major curve
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum MajorCurve {
    MT,
    TLL,
}

/// Set of [`Spine`]s required for Lenke classification
pub struct Study {
    pub coronal: CoronalPoints,
    pub left_bend: Option<Spine>,
    pub right_bend: Option<Spine>,
    pub sagittal: Option<Spine>,
}

impl Study {
    pub fn new(
        coronal: CoronalPoints,
        left_bend: Option<Spine>,
        right_bend: Option<Spine>,
        sagittal: Option<Spine>,
    ) -> Study {
        Study {
            coronal,
            left_bend,
            right_bend,
            sagittal,
        }
    }

    pub fn full(
        coronal: CoronalPoints,
        left_bend: Spine,
        right_bend: Spine,
        sagittal: Spine,
    ) -> Study {
        let left_bend = Some(left_bend);
        let right_bend = Some(right_bend);
        let sagittal = Some(sagittal);
        Study {
            coronal,
            left_bend,
            right_bend,
            sagittal,
        }
    }

    pub fn minor_param(
        &self,
        coronal: f64,
        coronal_curve: &Curve,
        sagittal_curve: &Curve,
    ) -> MinorStructuralParam {
        let right_bend = self
            .right_bend
            .as_ref()
            .map(|scol| scol.angle(coronal_curve).unwrap());
        let left_bend = self
            .left_bend
            .as_ref()
            .map(|scol| scol.angle(coronal_curve).unwrap());
        let sagittal = self
            .sagittal
            .as_ref()
            .map(|scol| (sagittal_curve.clone(), scol.angle(sagittal_curve).unwrap()));
        MinorStructuralParam {
            coronal,
            right_bend,
            left_bend,
            sagittal,
        }
    }

    pub fn chart(&self, curve_set: &CurveSet, major_curve: MajorCurve) -> Chart {
        let pt = curve_set.pt.as_ref().map(|pt| {
            self.minor_param(pt.1, &pt.0, &KYOPHOSIS_CURVE_PT)
                .curve_type()
        });
        let mt = if major_curve == MajorCurve::MT {
            Some(RegionalCurveType::Structural(StructuralReason::Major()))
        } else {
            curve_set.mt.as_ref().map(|mt| {
                self.minor_param(mt.1, &mt.0, &KYOPHOSIS_CURVE_MT)
                    .curve_type()
            })
        };
        let tll = if major_curve == MajorCurve::TLL {
            Some(RegionalCurveType::Structural(StructuralReason::Major()))
        } else {
            curve_set.tll.as_ref().map(|tll| {
                self.minor_param(tll.1, &tll.0, &KYOPHOSIS_CURVE_TLL)
                    .curve_type()
            })
        };
        Chart { pt, mt, tll }
    }
}

/// Curve types in Lenke classification
#[repr(u8)]
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CurveType {
    /// Main Thoracic
    Type1,
    /// Double Thoracic
    Type2,
    /// Double Major
    Type3,
    /// Triple Major
    Type4,
    /// Thoracolumbar/Lumbar
    Type5,
    /// Thoracolumbar/Lumbar - Main Thoracic
    Type6,
}

/// Modifier based on T5-T12 sagittal angle
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum SagittalModifier {
    /// angle < 10
    Hypokyphosis,
    /// 10 <= angle < 40
    Normokyphosis,
    /// 40 <= angle
    Hyperkyphosis,
}

impl From<f64> for SagittalModifier {
    fn from(angle: f64) -> Self {
        if angle.abs() < 10.0 {
            SagittalModifier::Hypokyphosis
        } else if angle.abs() < 40.0 {
            SagittalModifier::Normokyphosis
        } else {
            SagittalModifier::Hyperkyphosis
        }
    }
}

impl Display for SagittalModifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SagittalModifier::Hypokyphosis => write!(f, "-"),
            SagittalModifier::Normokyphosis => write!(f, "N"),
            SagittalModifier::Hyperkyphosis => write!(f, "+"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LumbarModifier {
    /// CSVL between pedicles
    A,
    /// CSVL touches apical predicle
    B,
    // A or B
    AorB,
    /// The apical vertebral bodies are completely lateral to the CSVL
    C,
}

#[derive(Debug, PartialEq)]
pub enum RegionalCurveType {
    Structural(StructuralReason),
    NonStructural(MinorReason),
    /// Uncertain due to the lack of some images
    Uncertain(MinorReason),
}

impl Display for RegionalCurveType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegionalCurveType::Structural(r) => {
                write!(f, "Structural({})", r)
            }
            RegionalCurveType::NonStructural(r) => {
                write!(f, "NonStructural({})", r.to_non_structural_string())
            }
            RegionalCurveType::Uncertain(r) => {
                // TODO: implement
                write!(f, "Uncertain({r:?})")
            }
        }
    }
}

/// Chart for Lenke classification
#[derive(Debug)]
pub struct Chart {
    pub pt: Option<RegionalCurveType>,
    pub mt: Option<RegionalCurveType>,
    pub tll: Option<RegionalCurveType>,
}

impl Display for Chart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PT: {}\nMT: {}\nTLL: {}",
            self.pt.as_ref().map_or("N/A".into(), |e| format!("{}", e)),
            self.mt.as_ref().map_or("N/A".into(), |e| format!("{}", e)),
            self.tll.as_ref().map_or("N/A".into(), |e| format!("{}", e)),
        )
    }
}

impl Chart {
    pub fn classify(&self) -> Result<CurveType, Vec<CurveType>> {
        let mut types = [
            None,
            Some(CurveType::Type1),
            Some(CurveType::Type2),
            Some(CurveType::Type3),
            Some(CurveType::Type4),
            Some(CurveType::Type5),
            Some(CurveType::Type6),
        ];
        if let Some(mt) = self.mt.as_ref() {
            match mt {
                RegionalCurveType::Structural(reason) => match reason {
                    StructuralReason::Major() => {
                        types[5] = None; // Type5
                        types[6] = None; // Type6
                    }
                    StructuralReason::Minor(_) => return Ok(CurveType::Type6),
                },
                RegionalCurveType::NonStructural(_) => return Ok(CurveType::Type5),
                RegionalCurveType::Uncertain(_) => {}
            }
        }
        if let Some(tll) = self.tll.as_ref() {
            match tll {
                RegionalCurveType::Structural(_) => {
                    types[1] = None; // Type1
                    types[2] = None; // Type2
                }
                RegionalCurveType::NonStructural(_) => {
                    types[3] = None; // Type3
                    types[4] = None; // Type4
                }
                RegionalCurveType::Uncertain(_) => {}
            }
        }
        if let Some(pt) = self.pt.as_ref() {
            match pt {
                RegionalCurveType::Structural(_) => {
                    types[1] = None; // Type1
                    types[3] = None; // Type3
                }
                RegionalCurveType::NonStructural(_) => {
                    types[2] = None; // Type2
                    types[4] = None; // Type4
                }
                RegionalCurveType::Uncertain(_) => {}
            }
        }
        let types: Vec<_> = types.into_iter().flatten().collect();
        if types.len() == 1 {
            Ok(types[0])
        } else {
            Err(types)
        }
    }
}

/// Angles descriving bending criteria
#[derive(Debug, PartialEq)]
pub struct BendReasonAngles {
    /// angle in normal coronal image
    coronal: f64,
    /// angle in bending image
    bend: f64,
}

impl BendReasonAngles {
    pub fn new(coronal: f64, bend: f64) -> BendReasonAngles {
        BendReasonAngles { coronal, bend }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IsStructural {
    // True
    T,
    // False
    F,
}

impl From<IsStructural> for bool {
    fn from(s: IsStructural) -> Self {
        match s {
            IsStructural::T => true,
            IsStructural::F => false,
        }
    }
}

impl From<bool> for IsStructural {
    fn from(is_structural: bool) -> Self {
        if is_structural {
            IsStructural::T
        } else {
            IsStructural::F
        }
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct BendReason {
    pub left: Option<(IsStructural, BendReasonAngles)>,
    pub right: Option<(IsStructural, BendReasonAngles)>,
}

#[derive(Debug, PartialEq)]
pub struct MinorReason {
    pub coronal: (IsStructural, f64),
    pub bend: BendReason,
    pub sagittal: Option<(IsStructural, (Curve, f64))>,
}

trait StructuralValue {
    fn is_structural(&self) -> bool;
    fn is_non_structural(&self) -> bool;
}

impl<T> StructuralValue for (IsStructural, T) {
    fn is_structural(&self) -> bool {
        bool::from(self.0)
    }
    fn is_non_structural(&self) -> bool {
        !self.is_structural()
    }
}

impl MinorReason {
    pub fn with_coronal(coronal: (IsStructural, f64)) -> MinorReason {
        MinorReason {
            coronal,
            bend: BendReason::default(),
            sagittal: None,
        }
    }
    fn to_structural_string(&self) -> String {
        let coronal_reason = if self.coronal.is_structural() {
            // using two `unwrap()`s below is safe
            // because a non-structural curve with `self.coronal.is_structural()`always has bend params
            let left_reason = format!(
                "|LeftBendAngle|={:.1} >= {BEND_ANGLE_THRESH}",
                self.bend.left.as_ref().unwrap().1.bend.abs()
            );
            let right_reason = format!(
                "|RightBendAngle|={:.1} >= {BEND_ANGLE_THRESH}",
                self.bend.right.as_ref().unwrap().1.bend.abs()
            );
            let bend_reason = format!("Coronal({left_reason} & {right_reason})");
            bend_reason.to_string()
        } else {
            // should be unreachable but return dummy value anyway
            let err =
            format!(
                "MinorReason::to_structural_string should not be called for {:?}, where coronal is structural",
                self
            );
            log::error!("{}", err);
            err
        };
        if let Some(sagittal) = self.sagittal.as_ref() {
            let sagittal_reason = format!(
                "Sagittal(|{}Kyphosis|={:.1} < {LATERAL_ANGLE_THRESH})",
                sagittal.1 .0,
                sagittal.1 .1.abs()
            );
            format!("{coronal_reason} or {sagittal_reason}")
        } else {
            coronal_reason
        }
    }
    fn to_non_structural_string(&self) -> String {
        let coronal_reason = if self.coronal.is_structural() {
            // using two `unwrap()`s below is safe
            // because a non-structural curve with `self.coronal.is_structural()`always has bend params
            let left_reason = format!(
                "|LeftBendAngle|={:.1} < {BEND_ANGLE_THRESH}",
                self.bend.left.as_ref().unwrap().1.bend.abs()
            );
            let right_reason = format!(
                "|RightBendAngle|={:.1} < {BEND_ANGLE_THRESH}",
                self.bend.right.as_ref().unwrap().1.bend.abs()
            );
            let bend_reason = format!("Coronal({left_reason} or {right_reason})");
            bend_reason.to_string()
        } else {
            format!(
                "|CoronalAngle|={:.1} < {}",
                self.coronal.1.abs(),
                FRONTAL_ANGLE_THRESH
            )
        };
        // safe to unwrap because sagittal is required for a curve to be non-structural
        let sagittal = self.sagittal.as_ref().unwrap();
        if sagittal.is_non_structural() {
            let sagittal_reason = format!(
                "Sagittal(|{}Kyphosis|={:.1} < {LATERAL_ANGLE_THRESH})",
                sagittal.1 .0,
                sagittal.1 .1.abs()
            );
            format!("{coronal_reason} & {sagittal_reason}")
        } else {
            coronal_reason
        }
    }
}

/// Parameters that explain why it's structural or non-structural
#[derive(Debug, PartialEq)]
pub struct MinorStructuralParam {
    coronal: f64,
    right_bend: Option<f64>,
    left_bend: Option<f64>,
    sagittal: Option<(Curve, f64)>,
}

trait CertainlyStructural {
    fn is_some_structural(&self) -> bool;
    fn is_some_non_structural(&self) -> bool;
}

impl<T> CertainlyStructural for Option<(IsStructural, T)> {
    /// Some(is_structural)
    fn is_some_structural(&self) -> bool {
        self.as_ref().is_some_and(|v| bool::from(v.0))
    }

    /// Some(is_non_structural)
    fn is_some_non_structural(&self) -> bool {
        self.as_ref().is_some_and(|v| !bool::from(v.0))
    }
}

impl MinorStructuralParam {
    pub fn curve_type(&self) -> RegionalCurveType {
        let mut reason = if self.coronal.abs() < FRONTAL_ANGLE_THRESH {
            MinorReason::with_coronal((IsStructural::F, self.coronal))
        } else {
            MinorReason::with_coronal((IsStructural::T, self.coronal))
        };

        for (bend, is_right) in [(self.right_bend, true), (self.left_bend, false)] {
            if let Some(angle) = bend {
                let angle = angle.abs();

                let is_structural = (angle.abs() >= BEND_ANGLE_THRESH).into();
                if is_right {
                    reason.bend.right =
                        Some((is_structural, BendReasonAngles::new(self.coronal, angle)));
                } else {
                    reason.bend.left =
                        Some((is_structural, BendReasonAngles::new(self.coronal, angle)));
                }
            }
        }
        if let Some(sagittal) = self.sagittal.as_ref() {
            let is_structural = (sagittal.1.abs() >= LATERAL_ANGLE_THRESH).into();
            reason.sagittal = Some((is_structural, sagittal.clone()));
        }

        if (reason.bend.left.is_some_structural() && reason.bend.right.is_some_structural())
            || reason.sagittal.is_some_structural()
        {
            return RegionalCurveType::Structural(StructuralReason::Minor(reason));
        }
        if reason.coronal.is_non_structural() && reason.sagittal.is_some_non_structural() {
            return RegionalCurveType::NonStructural(reason);
        }
        if (reason.bend.right.is_some_non_structural() || reason.bend.left.is_some_non_structural())
            && reason.sagittal.is_some_non_structural()
        {
            return RegionalCurveType::NonStructural(reason);
        }
        RegionalCurveType::Uncertain(reason)
    }
}

#[derive(Debug, PartialEq)]
pub enum StructuralReason {
    Major(),
    Minor(MinorReason),
}

impl Display for StructuralReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StructuralReason::Major() => write!(f, "{:?}", self),
            StructuralReason::Minor(r) => write!(f, "Minor({})", r.to_structural_string()),
        }
    }
}
