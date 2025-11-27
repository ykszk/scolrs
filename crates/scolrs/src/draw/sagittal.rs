use crate::draw::{AllPoints, AsMeasure, Named, ReductionMethod};
use crate::{
    Curve, SagittalDraw, SagittalMeasure, SagittalPointConfidence, SagittalPoints, ScaledType,
    Spine, ValidateLength, VertebralIndex,
};
use ndarray::{concatenate, s, stack, Array2, ArrayView2, Axis};
use ndarray_stats::DeviationExt;
use svg::node::element;

use super::{
    angle_between, draw_difference_in_x, draw_incidence_angle, draw_t1_angle,
    femoral_incidence_angle, mean_plate_length, reduce_confidence, tilt_angle, CobbAux,
    ColorPalette, ConfidenceComponent, DrawComponent, DrawError, MeasureComponent, MeasureError,
    Painter, VertebralLabels, VertebralPoints, CLASS_ANGLE, CLASS_DISTANCE, CLASS_MEASURE,
};

const SAGITTAL_COMPONENT_CLASS: &str = "SagittalComponent";
pub trait SagittalComponent: DrawComponent + MeasureComponent {
    fn default_group(&self) -> element::Group {
        self.default_group_w_classes(&["Component", SAGITTAL_COMPONENT_CLASS])
    }
}

fn calc_conf_from_sup_and_inf(
    confidences: &SagittalPointConfidence,
    sup: usize,
    inf: usize,
    reduction_method: ReductionMethod,
) -> f64 {
    let sup_confs = confidences.c7tls.slice(s![sup + 1, ..2]);
    let inf_confs = confidences.c7tls.slice(s![inf + 1, 2..]);
    let confs = concatenate(Axis(0), &[sup_confs, inf_confs]).unwrap();
    log::debug!("Confidences for sup and inf: {:?}", confs);
    let conf = reduce_confidence(confs.view(), reduction_method).unwrap();
    conf
}

macro_rules! impl_kyophosis {
    ($name:ident, $opposite:expr) => {
        impl<'a> DrawComponent for $name<'a> {
            fn draw(
                &self,
                painter: &mut Painter,
                _label_colors: &mut ColorPalette,
                line_colors: &mut ColorPalette,
            ) -> Result<element::Group, DrawError> {
                let label = self.id();
                let mut aux_param = if $opposite {
                    CobbAux::opposite_default()
                } else {
                    CobbAux::default()
                };
                aux_param.flip_sign = true;
                let curve = Curve {
                    sup: Self::SUP,
                    inf: Self::INF,
                };
                let group = self
                    .default_group()
                    .set("stroke", line_colors.get_or_new(label));
                let mean_plate_length = mean_plate_length(&self.0.spine);
                let g = painter.cobb(
                    group,
                    &self.0.spine,
                    &curve,
                    &aux_param,
                    mean_plate_length,
                    Some(label),
                );
                Ok(g)
            }
        }
        impl<'a> MeasureComponent for $name<'a> {
            fn measure(&self) -> Result<f64, MeasureError> {
                let angle = self
                    .0
                    .spine
                    .angle(&Curve {
                        sup: Self::SUP,
                        inf: Self::INF,
                    })
                    .unwrap();
                Ok(angle)
            }
        }
        impl ConfidenceComponent for $name<'_> {
            fn confidence(&self, reduction: ReductionMethod) -> Option<Result<f64, MeasureError>> {
                let confidence = self.0.confidences.as_ref()?;
                let conf = calc_conf_from_sup_and_inf(confidence, Self::SUP, Self::INF, reduction);
                Some(Ok(conf))
            }
        }
    };
}

/// Proximal thoracic (T2-T5) kyphosis (p.65)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct ProximalThoracicKyphosis<'a>(&'a SagittalPoints);
impl ProximalThoracicKyphosis<'_> {
    const SUP: usize = VertebralIndex::T2 as usize;
    const INF: usize = VertebralIndex::T5 as usize;
}
impl SagittalComponent for ProximalThoracicKyphosis<'_> {}
impl_kyophosis!(ProximalThoracicKyphosis, false);

/// Thoracic (T2-T12) kyphosis (p.65)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct ThoracicKyphosis<'a>(&'a SagittalPoints);
impl ThoracicKyphosis<'_> {
    const SUP: usize = VertebralIndex::T2 as usize;
    const INF: usize = VertebralIndex::T12 as usize;
}
impl SagittalComponent for ThoracicKyphosis<'_> {}
impl_kyophosis!(ThoracicKyphosis, true);

/// Thoracic (T1-T12) kyphosis with T1 as the upper endplate.
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct T1ThoracicKyphosis<'a>(&'a SagittalPoints);
impl T1ThoracicKyphosis<'_> {
    const SUP: usize = VertebralIndex::T1 as usize;
    const INF: usize = VertebralIndex::T12 as usize;
}
impl SagittalComponent for T1ThoracicKyphosis<'_> {}
impl_kyophosis!(T1ThoracicKyphosis, true);

/// Mid/Lower thoracic (T5-T12) kyphosis (p.65).
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
#[label("Mid/LowerThoracicKyphosis")]
pub struct MidLowerThoracicKyphosis<'a>(&'a SagittalPoints);
impl SagittalComponent for MidLowerThoracicKyphosis<'_> {}
impl MidLowerThoracicKyphosis<'_> {
    const SUP: usize = VertebralIndex::T5 as usize;
    const INF: usize = VertebralIndex::T12 as usize;
}
impl_kyophosis!(MidLowerThoracicKyphosis, false);

/// Thoracolumbar (T10/L2) sagittal alignment (p.66)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct ThoracolumbarSagittalAlignment<'a>(&'a SagittalPoints);
impl ThoracolumbarSagittalAlignment<'_> {
    const SUP: usize = VertebralIndex::T10 as usize;
    const INF: usize = VertebralIndex::L2 as usize;
}
impl SagittalComponent for ThoracolumbarSagittalAlignment<'_> {}
impl_kyophosis!(ThoracolumbarSagittalAlignment, false);

/// Lumbar sagittal (T12/S1) alignment (p.66)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct LumbarLordosis<'a>(&'a SagittalPoints);
impl LumbarLordosis<'_> {
    fn prep(spine: &Spine) -> (usize, usize) {
        let sup = VertebralIndex::T12 as usize;
        let inf = spine.v_c7tl.0.len_of(Axis(0)) - 1;
        (sup, inf)
    }
}
impl SagittalComponent for LumbarLordosis<'_> {}
impl DrawComponent for LumbarLordosis<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let spine = &self.0.spine;
        let aux_param = CobbAux {
            flip_sign: true,
            ..CobbAux::default()
        };
        let label = self.id();
        let group = self
            .default_group()
            .set("stroke", line_colors.get_or_new(label));
        let mean_plate_length = mean_plate_length(spine);
        let (sup, inf) = Self::prep(spine);
        let g = painter.cobb(
            group,
            spine,
            &Curve { sup, inf },
            &aux_param,
            mean_plate_length,
            Some(label),
        );
        Ok(g)
    }
}
impl MeasureComponent for LumbarLordosis<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let spine = &self.0.spine;
        let (sup, inf) = Self::prep(spine);
        let angle = spine.angle(&Curve { sup, inf }).unwrap();
        Ok(angle)
    }
}
impl ConfidenceComponent for LumbarLordosis<'_> {
    fn confidence(&self, reduction: ReductionMethod) -> Option<Result<f64, MeasureError>> {
        let confidence = self.0.confidences.as_ref()?;
        let (sup, inf) = Self::prep(&self.0.spine);
        let conf = calc_conf_from_sup_and_inf(confidence, sup, inf, reduction);
        Some(Ok(conf))
    }
}

/// T1 slope angle
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct T1Slope<'a>(&'a SagittalPoints);
impl SagittalComponent for T1Slope<'_> {}
impl DrawComponent for T1Slope<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        draw_t1_angle(
            painter,
            line_colors,
            &self.0.spine,
            self,
            true,
            self.default_group(),
        )
    }
}
impl MeasureComponent for T1Slope<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let tl_sup_lines = self.0.spine.tl_sup_lines();
        let t1sup = tl_sup_lines.index_axis(Axis(0), 0);
        tilt_angle("Vertebra", t1sup).map(|a| -a)
    }
}
impl ConfidenceComponent for T1Slope<'_> {
    fn confidence(&self, reduction: ReductionMethod) -> Option<Result<f64, MeasureError>> {
        let confidence = self.0.confidences.as_ref()?;
        let confs = confidence.c7tls.slice(s![1, ..2]);
        let conf = reduce_confidence(confs.view(), reduction).unwrap();
        Some(Ok(conf))
    }
}

/// Sagittal balance (p.67)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_DISTANCE])]
pub struct SagittalBalance<'a>(&'a SagittalPoints);
impl SagittalBalance<'_> {
    fn prep(sagittal_points: &SagittalPoints) -> Array2<f64> {
        let c_c7 = sagittal_points.spine.c_c7tl.index_axis(Axis(0), 0);
        let sac_sup = sagittal_points.spine.sacral_sup_plate();
        let pos_sac = sac_sup.index_axis(Axis(0), 1);
        stack![Axis(0), pos_sac, c_c7]
    }
}
impl SagittalComponent for SagittalBalance<'_> {}
impl DrawComponent for SagittalBalance<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let points = Self::prep(self.0);
        let label = self.id();
        let color = line_colors.get_or_new(label);
        let g = self.default_group().set("stroke", color).set("fill", color);
        let g = draw_difference_in_x(
            g,
            "C7andSacrum",
            label,
            points.view(),
            painter,
            self.0.image_metadata.unit.as_str(),
        );
        g
    }
}
impl MeasureComponent for SagittalBalance<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let points = Self::prep(self.0);
        let p1 = points.index_axis(Axis(0), 0);
        let p2 = points.index_axis(Axis(0), 1);
        let dx = p1[0] - p2[0];
        Ok(dx)
    }
}
impl ConfidenceComponent for SagittalBalance<'_> {
    fn confidence(&self, reduction: ReductionMethod) -> Option<Result<f64, MeasureError>> {
        let confidence = self.0.confidences.as_ref()?;
        let sac_sup_confs = confidence
            .c7tls
            .slice(s![confidence.c7tls.len_of(Axis(0)) - 1, ..2]);
        let c7_confs = confidence.c7tls.index_axis(Axis(0), 0);
        let confs = concatenate(Axis(0), &[sac_sup_confs, c7_confs]).unwrap();
        let conf = reduce_confidence(confs.view(), reduction).unwrap();
        Some(Ok(conf))
    }
}

/// Lumbosacral angle (p.105)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct LumbosacralAngle<'a>(&'a SagittalPoints);
impl LumbosacralAngle<'_> {
    fn prep(spine: &Spine) -> (Array2<f64>, Array2<f64>) {
        let sup = spine
            .inf_plate(spine.v_c7tl.0.len_of(Axis(0)) - 2)
            .to_owned();
        let inf = spine
            .inf_plate(spine.v_c7tl.0.len_of(Axis(0)) - 1)
            .to_owned();
        (sup, inf)
    }
}
impl SagittalComponent for LumbosacralAngle<'_> {}
impl DrawComponent for LumbosacralAngle<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let spine = &self.0.spine;
        let (sup, inf) = Self::prep(spine);
        let label = self.id();
        let mean_plate_length = mean_plate_length(spine);
        let group = self
            .default_group()
            .set("stroke", line_colors.get_or_new(label));
        let aux_param = CobbAux {
            flip_sign: true,
            ..CobbAux::default()
        };
        let g = painter.cobb_from_plates(
            group,
            (sup, inf),
            &aux_param,
            mean_plate_length,
            Some(label),
        );
        Ok(g)
    }
}
impl MeasureComponent for LumbosacralAngle<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let (sup, inf) = Self::prep(&self.0.spine);
        let angle = angle_between(sup.view(), inf.view()).to_degrees();
        Ok(angle)
    }
}
impl ConfidenceComponent for LumbosacralAngle<'_> {
    fn confidence(&self, reduction: ReductionMethod) -> Option<Result<f64, MeasureError>> {
        let confidence = self.0.confidences.as_ref()?;
        let sac_sup_confs = confidence
            .c7tls
            .slice(s![confidence.c7tls.len_of(Axis(0)) - 1, ..2]);
        let l5_inf_confs = confidence
            .c7tls
            .slice(s![confidence.c7tls.len_of(Axis(0)) - 2, 2..]);
        let confs = concatenate(Axis(0), &[sac_sup_confs, l5_inf_confs]).unwrap();
        let conf = reduce_confidence(confs.view(), reduction).unwrap();
        Some(Ok(conf))
    }
}

/// Pelvic Incidence (p.97)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct PelvicIncidence<'a>(&'a SagittalPoints);
impl SagittalComponent for PelvicIncidence<'_> {}
impl DrawComponent for PelvicIncidence<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        self.0
            .femoral_head
            .0
            .validate_label_length_more_than("FemoralHead", 1)?;
        let label = self.id();
        let sac_sup = self.0.spine.sacral_sup_plate();
        let color = line_colors.get_or_new(label);
        let g = self.default_group().set("fill", color).set("stroke", color);
        let g = draw_incidence_angle(g, label, self.0.femoral_head.0.view(), sac_sup, painter);
        Ok(g)
    }
}
impl MeasureComponent for PelvicIncidence<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let plate = self.0.spine.sacral_sup_plate();
        femoral_incidence_angle(plate, &self.0.femoral_head)
    }
}
impl ConfidenceComponent for PelvicIncidence<'_> {
    fn confidence(&self, reduction: ReductionMethod) -> Option<Result<f64, MeasureError>> {
        let confidence = self.0.confidences.as_ref()?;
        confidence
            .femoral_head
            .validate_label_length_more_than("FemoralHead", 1)
            .ok()?;
        let femoral_confs = confidence.femoral_head.view();
        let sac_sup_confs = confidence
            .c7tls
            .slice(s![confidence.c7tls.len_of(Axis(0)) - 1, ..2]);
        let confs = concatenate(Axis(0), &[femoral_confs, sac_sup_confs]).unwrap();
        let conf = reduce_confidence(confs.view(), reduction).unwrap();
        Some(Ok(conf))
    }
}

// Pelvic Tilt (p.98)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct PelvicTilt<'a>(&'a SagittalPoints);
impl PelvicTilt<'_> {
    fn prep(sagittal_points: &SagittalPoints) -> Result<(Array2<f64>, Array2<f64>), MeasureError> {
        sagittal_points
            .femoral_head
            .0
            .validate_label_length_more_than("FemoralHead", 1)?;
        let sac_sup = sagittal_points.spine.sacral_sup_plate();
        let mid_sac = sac_sup.mean_axis(Axis(0)).unwrap();
        let femoral_head = sagittal_points.femoral_head.0.mean_axis(Axis(0)).unwrap();
        let fem2sac = stack![Axis(0), femoral_head, mid_sac];
        let mut v_line_from_fem = fem2sac.clone();
        v_line_from_fem[[1, 0]] = v_line_from_fem[[0, 0]];
        Ok((fem2sac, v_line_from_fem))
    }
}
impl SagittalComponent for PelvicTilt<'_> {}
impl DrawComponent for PelvicTilt<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let sagittal_points = self.0;
        let (fem2sac, v_line) = Self::prep(sagittal_points)?;
        let label = self.id();
        let color = line_colors.get_or_new(label);
        let mut g = self.default_group().set("fill", color).set("stroke", color);
        g = draw_femoral_center(g, sagittal_points.femoral_head.0.view(), painter);
        g = g.add(painter.point(fem2sac.index_axis(Axis(0), 1)));
        let sac_sup = sagittal_points.spine.sacral_sup_plate();
        for p in sac_sup.axis_iter(Axis(0)) {
            g = g.add(painter.point(p));
        }
        g = g.add(painter.line(sac_sup.view()));

        let arc_radius = 0.5
            * fem2sac
                .index_axis(Axis(0), 0)
                .l2_dist(&fem2sac.index_axis(Axis(0), 1))
                .unwrap();
        let (g, _) = painter.angle_between(
            g,
            v_line.view(),
            fem2sac.view(),
            fem2sac.index_axis(Axis(0), 0),
            arc_radius,
            Some(label),
        );
        Ok(g)
    }
}
impl MeasureComponent for PelvicTilt<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let sagittal_points = self.0;
        let (fem2sac, v_line) = Self::prep(sagittal_points)?;
        let angle = angle_between(v_line.view(), fem2sac.view()).to_degrees();
        Ok(angle)
    }
}
impl ConfidenceComponent for PelvicTilt<'_> {
    fn confidence(&self, reduction: ReductionMethod) -> Option<Result<f64, MeasureError>> {
        // Same as Pelvic Incidence
        PelvicIncidence(self.0).confidence(reduction)
    }
}

// Sacral Slope (p.99)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct SacralSlope<'a>(&'a SagittalPoints);
impl SagittalComponent for SacralSlope<'_> {}
impl SacralSlope<'_> {
    fn prep(sagittal_points: &SagittalPoints) -> Result<(Array2<f64>, Array2<f64>), MeasureError> {
        let sac_sup = sagittal_points.spine.sacral_sup_plate().to_owned();
        let h_len = sac_sup
            .index_axis(Axis(0), 0)
            .l2_dist(&sac_sup.index_axis(Axis(0), 1))
            .unwrap();
        let mut h_line_sac = sac_sup.clone();
        h_line_sac[[1, 1]] = h_line_sac[[0, 1]];
        h_line_sac[[1, 0]] = h_line_sac[[0, 0]] + h_len;
        Ok((sac_sup, h_line_sac))
    }
}

impl DrawComponent for SacralSlope<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let sagittal_points = self.0;
        let (sac_sup, h_line_sac) = Self::prep(sagittal_points)?;
        let label = self.id();
        let color = line_colors.get_or_new(label);
        let mut g = self.default_group().set("stroke", color);
        // let angle = angle_between(h_line_sac.view(), sac_sup.view()).to_degrees();
        let arc_radius = 0.5
            * sac_sup
                .index_axis(Axis(0), 0)
                .l2_dist(&sac_sup.index_axis(Axis(0), 1))
                .unwrap();
        g = painter
            .angle_between(
                g,
                sac_sup.view(),
                h_line_sac.view(),
                h_line_sac.index_axis(Axis(0), 0),
                arc_radius,
                Some(label),
            )
            .0;
        Ok(g)
    }
}
impl MeasureComponent for SacralSlope<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let sagittal_points = self.0;
        let (sac_sup, h_line_sac) = Self::prep(sagittal_points)?;
        let angle = angle_between(sac_sup.view(), h_line_sac.view()).to_degrees();
        Ok(angle)
    }
}
impl ConfidenceComponent for SacralSlope<'_> {
    fn confidence(&self, reduction: ReductionMethod) -> Option<Result<f64, MeasureError>> {
        let confidence = self.0.confidences.as_ref()?;
        let sac_sup_confs = confidence
            .c7tls
            .slice(s![confidence.c7tls.len_of(Axis(0)) - 1, ..2]);
        let conf = reduce_confidence(sac_sup_confs.view(), reduction).unwrap();
        Some(Ok(conf))
    }
}

/// L5 Incidence Angle (p.102)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct L5IncidenceAngle<'a>(&'a SagittalPoints);
impl SagittalComponent for L5IncidenceAngle<'_> {}
impl DrawComponent for L5IncidenceAngle<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        self.0
            .femoral_head
            .0
            .validate_label_length_more_than("FemoralHead", 1)?;
        let label = self.id();
        let l5_sup = self
            .0
            .spine
            .sup_plate(self.0.spine.v_c7tl.0.len_of(Axis(0)) - 2)
            .to_owned();
        let color = line_colors.get_or_new(label);
        let g = self.default_group().set("fill", color).set("stroke", color);
        let g = draw_incidence_angle(
            g,
            label,
            self.0.femoral_head.0.view(),
            l5_sup.view(),
            painter,
        );
        Ok(g)
    }
}
impl MeasureComponent for L5IncidenceAngle<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let sagittal_points = self.0;
        let plate = sagittal_points
            .spine
            .sup_plate(sagittal_points.spine.v_c7tl.0.len_of(Axis(0)) - 2);
        femoral_incidence_angle(plate, &sagittal_points.femoral_head)
    }
}
impl ConfidenceComponent for L5IncidenceAngle<'_> {
    fn confidence(&self, reduction: ReductionMethod) -> Option<Result<f64, MeasureError>> {
        let confidence = self.0.confidences.as_ref()?;
        confidence
            .femoral_head
            .validate_label_length_more_than("FemoralHead", 1)
            .ok()?;
        let femoral_confs = confidence.femoral_head.view();
        let l5_sup_confs = confidence
            .c7tls
            .slice(s![confidence.c7tls.len_of(Axis(0)) - 2, ..2]);
        let confs = concatenate(Axis(0), &[femoral_confs, l5_sup_confs]).unwrap();
        let conf = reduce_confidence(confs.view(), reduction).unwrap();
        Some(Ok(conf))
    }
}

fn draw_femoral_center(
    group: element::Group,
    femoral_head: ArrayView2<f64>,
    painter: &mut Painter,
) -> element::Group {
    let mut g = group;
    for p in femoral_head.axis_iter(Axis(0)) {
        g = g.add(painter.point(p));
    }
    if femoral_head.len_of(Axis(0)) == 2 {
        let mid_femoral_heads = femoral_head.mean_axis(Axis(0)).unwrap();
        g = g.add(painter.point(mid_femoral_heads.view()));
        g = g.add(painter.line(femoral_head.view()));
    }
    g
}

/// Pelvic Radius Angle (p.101)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct PelvicRadiusAngle<'a>(&'a SagittalPoints);
impl PelvicRadiusAngle<'_> {
    fn prep(sagittal_points: &SagittalPoints) -> Result<(Array2<f64>, Array2<f64>), MeasureError> {
        sagittal_points
            .femoral_head
            .0
            .validate_label_length_more_than("FemoralHead", 1)?;
        let mid_femoral_heads = sagittal_points.femoral_head.0.mean_axis(Axis(0)).unwrap();
        let sac_sup = sagittal_points.spine.sacral_sup_plate();
        let post_sac = sac_sup.index_axis(Axis(0), 1);
        let sac_sup_post2ante = stack![
            Axis(0),
            sac_sup.index_axis(Axis(0), 1),
            sac_sup.index_axis(Axis(0), 0)
        ];
        let post_sac2fem = stack![Axis(0), post_sac, mid_femoral_heads];
        Ok((sac_sup_post2ante, post_sac2fem))
    }
}
impl SagittalComponent for PelvicRadiusAngle<'_> {}
impl DrawComponent for PelvicRadiusAngle<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let sagittal_points = self.0;
        let (sac_sup_post2ante, post_sac2fem) = Self::prep(sagittal_points)?;
        let post_sac = sac_sup_post2ante.index_axis(Axis(0), 0);
        let label = self.id();
        let color = line_colors.get_or_new(label);
        let mut g = self.default_group().set("fill", color).set("stroke", color);
        g = draw_femoral_center(g, sagittal_points.femoral_head.0.view(), painter);
        let arc_radius = sac_sup_post2ante
            .index_axis(Axis(0), 0)
            .l2_dist(&sac_sup_post2ante.index_axis(Axis(0), 1))
            .unwrap();
        let g = painter
            .angle_between(
                g,
                post_sac2fem.view(),
                sac_sup_post2ante.view(),
                post_sac,
                arc_radius,
                Some(label),
            )
            .0;
        Ok(g)
    }
}
impl MeasureComponent for PelvicRadiusAngle<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let (sac_sup_post2ante, post_sac2fem) = Self::prep(self.0)?;
        let angle = angle_between(post_sac2fem.view(), sac_sup_post2ante.view()).to_degrees();
        Ok(angle)
    }
}
impl ConfidenceComponent for PelvicRadiusAngle<'_> {
    fn confidence(&self, reduction: ReductionMethod) -> Option<Result<f64, MeasureError>> {
        // Same as Pelvic Incidence
        PelvicIncidence(self.0).confidence(reduction)
    }
}

impl<'a, 'b> From<(&'b SagittalDraw, &'a ScaledType<SagittalPoints>)>
    for Box<dyn DrawComponent + 'a>
{
    fn from(
        (measure, sagittal_points): (&'b SagittalDraw, &'a ScaledType<SagittalPoints>),
    ) -> Self {
        let sagittal_points = &sagittal_points.0;
        match measure {
            SagittalDraw::ThoracicKyphosis => Box::new(ThoracicKyphosis(sagittal_points)),
            SagittalDraw::ThoracicKyphosisT1 => Box::new(T1ThoracicKyphosis(sagittal_points)),
            SagittalDraw::ProximalThoracicKyphosis => {
                Box::new(ProximalThoracicKyphosis(sagittal_points))
            }
            SagittalDraw::MidLowerThoracicKyphosis => {
                Box::new(MidLowerThoracicKyphosis(sagittal_points))
            }
            SagittalDraw::ThoracolumbarSagittalAlignment => {
                Box::new(ThoracolumbarSagittalAlignment(sagittal_points))
            }
            SagittalDraw::LumbarLordosis => Box::new(LumbarLordosis(sagittal_points)),
            SagittalDraw::T1Slope => Box::new(T1Slope(sagittal_points)),
            SagittalDraw::SagittalBalance => Box::new(SagittalBalance(sagittal_points)),
            SagittalDraw::LumbosacralAngle => Box::new(LumbosacralAngle(sagittal_points)),
            SagittalDraw::PelvicIncidence => Box::new(PelvicIncidence(sagittal_points)),
            SagittalDraw::PelvicTilt => Box::new(PelvicTilt(sagittal_points)),
            SagittalDraw::SacralSlope => Box::new(SacralSlope(sagittal_points)),
            SagittalDraw::L5IncidenceAngle => Box::new(L5IncidenceAngle(sagittal_points)),
            SagittalDraw::PelvicRadiusAngle => Box::new(PelvicRadiusAngle(sagittal_points)),

            SagittalDraw::AllPoints => Box::new(AllPoints(sagittal_points.to_points())),
            SagittalDraw::VertebralLabels => Box::new(VertebralLabels(&sagittal_points.spine)),
            SagittalDraw::VertebralPoints => Box::new(VertebralPoints(&sagittal_points.spine)),
        }
    }
}

impl<'a, 'b> From<(&'b SagittalMeasure, &'a ScaledType<SagittalPoints>)>
    for Box<dyn MeasureComponent + 'a>
{
    fn from(
        (measure, sagittal_points): (&'b SagittalMeasure, &'a ScaledType<SagittalPoints>),
    ) -> Self {
        let sagittal_points = &sagittal_points.0;
        match measure {
            SagittalMeasure::ThoracicKyphosis => Box::new(ThoracicKyphosis(sagittal_points)),
            SagittalMeasure::T1ThoracicKyphosis => Box::new(T1ThoracicKyphosis(sagittal_points)),
            SagittalMeasure::ProximalThoracicKyphosis => {
                Box::new(ProximalThoracicKyphosis(sagittal_points))
            }
            SagittalMeasure::MidLowerThoracicKyphosis => {
                Box::new(MidLowerThoracicKyphosis(sagittal_points))
            }
            SagittalMeasure::ThoracolumbarSagittalAlignment => {
                Box::new(ThoracolumbarSagittalAlignment(sagittal_points))
            }
            SagittalMeasure::LumbarLordosis => Box::new(LumbarLordosis(sagittal_points)),
            SagittalMeasure::SagittalBalance => Box::new(SagittalBalance(sagittal_points)),
            SagittalMeasure::LumbosacralAngle => Box::new(LumbosacralAngle(sagittal_points)),
            SagittalMeasure::T1Slope => Box::new(T1Slope(sagittal_points)),
            SagittalMeasure::PelvicIncidence => Box::new(PelvicIncidence(sagittal_points)),
            SagittalMeasure::PelvicTilt => Box::new(PelvicTilt(sagittal_points)),
            SagittalMeasure::SacralSlope => Box::new(SacralSlope(sagittal_points)),
            SagittalMeasure::L5IncidenceAngle => Box::new(L5IncidenceAngle(sagittal_points)),
            SagittalMeasure::PelvicRadiusAngle => Box::new(PelvicRadiusAngle(sagittal_points)),
        }
    }
}

impl<'a, 'b> From<(&'b SagittalMeasure, &'a ScaledType<SagittalPoints>)>
    for Box<dyn ConfidenceComponent + 'a>
{
    fn from(
        (measure, sagittal_points): (&'b SagittalMeasure, &'a ScaledType<SagittalPoints>),
    ) -> Self {
        let sagittal_points = &sagittal_points.0;
        match measure {
            SagittalMeasure::ThoracicKyphosis => Box::new(ThoracicKyphosis(sagittal_points)),
            SagittalMeasure::T1ThoracicKyphosis => Box::new(T1ThoracicKyphosis(sagittal_points)),
            SagittalMeasure::ProximalThoracicKyphosis => {
                Box::new(ProximalThoracicKyphosis(sagittal_points))
            }
            SagittalMeasure::MidLowerThoracicKyphosis => {
                Box::new(MidLowerThoracicKyphosis(sagittal_points))
            }
            SagittalMeasure::ThoracolumbarSagittalAlignment => {
                Box::new(ThoracolumbarSagittalAlignment(sagittal_points))
            }
            SagittalMeasure::LumbarLordosis => Box::new(LumbarLordosis(sagittal_points)),
            SagittalMeasure::SagittalBalance => Box::new(SagittalBalance(sagittal_points)),
            SagittalMeasure::LumbosacralAngle => Box::new(LumbosacralAngle(sagittal_points)),
            SagittalMeasure::T1Slope => Box::new(T1Slope(sagittal_points)),
            SagittalMeasure::PelvicIncidence => Box::new(PelvicIncidence(sagittal_points)),
            SagittalMeasure::PelvicTilt => Box::new(PelvicTilt(sagittal_points)),
            SagittalMeasure::SacralSlope => Box::new(SacralSlope(sagittal_points)),
            SagittalMeasure::L5IncidenceAngle => Box::new(L5IncidenceAngle(sagittal_points)),
            SagittalMeasure::PelvicRadiusAngle => Box::new(PelvicRadiusAngle(sagittal_points)),
        }
    }
}

impl AsMeasure for SagittalDraw {
    type MeasureType = SagittalMeasure;
    fn as_measure(&self) -> Option<Self::MeasureType> {
        match self {
            SagittalDraw::ThoracicKyphosis => Some(SagittalMeasure::ThoracicKyphosis),
            SagittalDraw::ThoracicKyphosisT1 => Some(SagittalMeasure::T1ThoracicKyphosis),
            SagittalDraw::ProximalThoracicKyphosis => {
                Some(SagittalMeasure::ProximalThoracicKyphosis)
            }
            SagittalDraw::MidLowerThoracicKyphosis => {
                Some(SagittalMeasure::MidLowerThoracicKyphosis)
            }
            SagittalDraw::ThoracolumbarSagittalAlignment => {
                Some(SagittalMeasure::ThoracolumbarSagittalAlignment)
            }
            SagittalDraw::LumbarLordosis => Some(SagittalMeasure::LumbarLordosis),
            SagittalDraw::T1Slope => Some(SagittalMeasure::T1Slope),
            SagittalDraw::SagittalBalance => Some(SagittalMeasure::SagittalBalance),
            SagittalDraw::LumbosacralAngle => Some(SagittalMeasure::LumbosacralAngle),
            SagittalDraw::PelvicIncidence => Some(SagittalMeasure::PelvicIncidence),
            SagittalDraw::PelvicTilt => Some(SagittalMeasure::PelvicTilt),
            SagittalDraw::SacralSlope => Some(SagittalMeasure::SacralSlope),
            SagittalDraw::L5IncidenceAngle => Some(SagittalMeasure::L5IncidenceAngle),
            SagittalDraw::PelvicRadiusAngle => Some(SagittalMeasure::PelvicRadiusAngle),

            SagittalDraw::AllPoints
            | SagittalDraw::VertebralLabels
            | SagittalDraw::VertebralPoints => None,
        }
    }
}
