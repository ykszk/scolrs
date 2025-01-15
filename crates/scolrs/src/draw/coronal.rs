use crate::draw::Named;
use crate::{ApexSet, Curve, Spine};
use crate::{CoronalDraw, CoronalMeasure, CoronalPoints, CoronalPointsAndCurve, ValidateLength};
use ndarray::{s, stack, Array2, Axis};
use ndarray_stats::DeviationExt;
use svg::node::element;

use super::{
    draw_difference_in_x, draw_difference_in_y, draw_t1_angle, draw_tilt_angle, mean_plate_length,
    points2line, tilt_angle, Centroids, CobbAux, ColorPalette, CommonComponent, DrawComponent,
    DrawError, MeasureComponent, MeasureError, Painter, VertebralLabels, VertebralPoints,
    CLASS_ANGLE, CLASS_ANNOTATION, CLASS_DISTANCE, CLASS_LINE, CLASS_MEASURE, CLASS_POLYGON,
};

const CORONAL_COMPONENT_CLASS: &str = "CoronalComponent";
pub trait CoronalComponent: DrawComponent {
    fn default_group(&self) -> element::Group {
        self.default_group_w_classes(&["Component", CORONAL_COMPONENT_CLASS])
    }
}

macro_rules! impl_cobb_angle {
    ($name:ident) => {
        impl<'a> DrawComponent for $name<'a> {
            fn draw(
                &self,
                painter: &mut Painter,
                _label_colors: &mut ColorPalette,
                line_colors: &mut ColorPalette,
            ) -> Result<element::Group, DrawError> {
                if self.1.is_none() {
                    return Err(DrawError::MeasureError(MeasureError::NoCurveFound));
                }
                let coronal_points = self.0;
                let (curve, _angle) = self.1.as_ref().unwrap();
                let color = line_colors.get_or_new(self.id());
                let g = self.default_group().set("stroke", color);
                let aux_param = CobbAux::default();
                let mean_plate_length = mean_plate_length(&coronal_points.spine);

                let group = painter.cobb(
                    g,
                    &coronal_points.spine,
                    curve,
                    &aux_param,
                    mean_plate_length,
                    Some(self.id()),
                );
                Ok(group)
            }
        }
        impl<'a> MeasureComponent for $name<'a> {
            fn measure(&self) -> Result<f64, MeasureError> {
                if let Some((_curve, angle)) = self.1.as_ref() {
                    Ok(*angle)
                } else {
                    Err(MeasureError::NoCurveFound)
                }
            }
        }
    };
}

/// Cobb angle for PT curve
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
struct CobbPT<'a>(&'a CoronalPoints, Option<(Curve, f64)>);
impl CoronalComponent for CobbPT<'_> {}
impl_cobb_angle!(CobbPT);

/// Cobb angle for MT curve
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
struct CobbMT<'a>(&'a CoronalPoints, Option<(Curve, f64)>);
impl CoronalComponent for CobbMT<'_> {}
impl_cobb_angle!(CobbMT);

/// Cobb angle for TLL curve
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
struct CobbTLL<'a>(&'a CoronalPoints, Option<(Curve, f64)>);
impl CoronalComponent for CobbTLL<'_> {}
impl_cobb_angle!(CobbTLL);

/// Curve apices for each curve
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POLYGON])]
struct CurveApex<'a>(&'a CoronalPoints, &'a ApexSet);
impl CoronalComponent for CurveApex<'_> {}
impl DrawComponent for CurveApex<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let coronal_points = self.0;
        let apex_set = &self.1;
        let vert_discs = coronal_points.spine.tl_vert_disc_corners().0;

        let label = self.id();
        let mut g = self.default_group();

        for apex in [apex_set.pt, apex_set.mt, apex_set.tll]
            .into_iter()
            .flatten()
        {
            let mut corners = vert_discs.index_axis(Axis(0), apex as usize).to_owned();
            // Change point-order from (tl, tr, bl, br) to (tl, tr, br, bl)
            corners.swap((2, 0), (3, 0)); // bl.x <-> br.x
            corners.swap((2, 1), (3, 1)); // bl.y <-> br.y
            let polygon = painter
                .polygon(corners)
                .set("stroke", line_colors.get_or_new(label));
            g = g.add(polygon);
        }
        Ok(g)
    }
}

/// Spinal center line
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_LINE])]
struct SpinalLine<'a>(&'a Spine);
impl CommonComponent for SpinalLine<'_> {}
impl DrawComponent for SpinalLine<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let label = self.id();
        let g = self.default_group();
        let centroids = self.0.tl_centroids();
        let coefs =
            crate::polyfit(centroids.slice(s![.., 1]), centroids.slice(s![.., 0]), 6).unwrap();
        let ys = ndarray::Array::linspace(
            centroids[[0, 1]],
            centroids[[centroids.len_of(Axis(0)) - 1, 1]],
            50,
        );
        let xs = crate::polynomial(ys.view(), coefs);
        let spinal_line = painter
            .polyline(ndarray::stack![Axis(1), xs, ys])
            .set("stroke", line_colors.get_or_new(label));

        Ok(g.add(spinal_line))
    }
}

/// center sacral vertical line (CSVL) (p. 54)
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_LINE])]
pub struct Csvl<'a>(&'a CoronalPoints, &'a ApexSet);
impl CoronalComponent for Csvl<'_> {}
impl DrawComponent for Csvl<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let coronal_points = self.0;
        let spine = &coronal_points.spine;
        let label = self.id();
        let line_color = line_colors.get_or_new(label);
        let mut g = self.default_group().set("stroke", line_color);
        let sup_plate = spine.sacral_sup_plate();
        let sacral_line = painter.line(sup_plate.view());
        g = g.add(sacral_line);
        if let Some(tll) = self.1.tll {
            let v_idx = ((tll as u8) / 2 - 1) as usize; // one level above the tll apex
            let mid = sup_plate.mean_axis(Axis(0)).unwrap();
            let mut vl = ndarray::stack![Axis(0), mid, mid];
            let y = spine.c7tls.0[[v_idx + 1, 0, 1]]; // v_idx+1 because vertebrae include c7
            vl[[0, 1]] = y;
            g = g.add(painter.line(vl));
        }
        Ok(g)
    }
}

/// T1 Tilt Angle (p.55)
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_ANGLE])]
pub struct T1TiltAngle<'a>(&'a CoronalPoints);
impl CoronalComponent for T1TiltAngle<'_> {}
impl DrawComponent for T1TiltAngle<'_> {
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
            self.default_group(),
        )
    }
}
impl MeasureComponent for T1TiltAngle<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let tl_sup_lines = self.0.spine.tl_sup_lines();
        let t1sup = tl_sup_lines.index_axis(Axis(0), 0);
        tilt_angle("Vertebra", t1sup)
    }
}

/// Coronal balance (p. 54)
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_DISTANCE])]
pub struct CoronalBalance<'a>(&'a CoronalPoints);
impl CoronalComponent for CoronalBalance<'_> {}
impl CoronalBalance<'_> {
    fn prep(&self, spine: &Spine) -> Array2<f64> {
        let c_c7 = spine.c_c7tl.index_axis(Axis(0), 0);
        let sac_sup = spine.sacral_sup_plate();
        let mid_sac = sac_sup.mean_axis(Axis(0)).unwrap();
        let points = stack![Axis(0), c_c7, mid_sac];
        points
    }
}
impl DrawComponent for CoronalBalance<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let spine = &self.0.spine;
        let label = self.id();
        let points = self.prep(spine);
        let color = line_colors.get_or_new(label);
        let g = self.default_group().set("fill", color).set("stroke", color);
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
impl MeasureComponent for CoronalBalance<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let points = self.prep(&self.0.spine);
        let dx = points.index_axis(Axis(0), 0)[0] - points.index_axis(Axis(0), 1)[0];
        Ok(dx)
    }
}

/// Clavicle angle (p. 56)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct ClavicleAngle<'a>(&'a CoronalPoints);
impl CoronalComponent for ClavicleAngle<'_> {}
impl DrawComponent for ClavicleAngle<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let coronal_points = self.0;
        coronal_points
            .clavicle
            .0
            .validate_label_length("Clavicle", 2)?;
        let label = self.id();
        let color = line_colors.get_or_new(label);
        let mut g = self.default_group().set("fill", color).set("stroke", color);
        let clavicle = &coronal_points.clavicle.0;
        g = draw_tilt_angle(g, painter, clavicle, Some(label));
        Ok(g)
    }
}
impl MeasureComponent for ClavicleAngle<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        tilt_angle("Clavicle", self.0.clavicle.0.view())
    }
}

/// Radiographic shoulder height (p.57)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_DISTANCE])]
pub struct ShoulderHeight<'a>(&'a CoronalPoints);
impl CoronalComponent for ShoulderHeight<'_> {}
impl DrawComponent for ShoulderHeight<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let g = self.default_group().set("fill", color).set("stroke", color);

        draw_difference_in_y(
            g,
            "Shoulder",
            self.id(),
            self.0.shoulder.0.view(),
            painter,
            self.0.image_metadata.unit.as_str(),
        )
    }
}
impl MeasureComponent for ShoulderHeight<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let coronal_points = self.0;
        coronal_points
            .shoulder
            .0
            .validate_label_length("Shoulder", 2)?;
        let points = coronal_points.shoulder.0.view();
        let dy = points.index_axis(Axis(0), 0)[1] - points.index_axis(Axis(0), 1)[1];
        Ok(dy)
    }
}

/// Pelvic Obliquity (p.69)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct PelvicObliquity<'a>(&'a CoronalPoints);
impl CoronalComponent for PelvicObliquity<'_> {}
impl DrawComponent for PelvicObliquity<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let coronal_points = self.0;
        coronal_points.pelvis.0.validate_label_length("Pelvis", 2)?;
        let label = self.id();
        let color = line_colors.get_or_new(label);
        let mut g = self.default_group().set("fill", color).set("stroke", color);
        let pelvis = &coronal_points.pelvis.0;
        g = draw_tilt_angle(g, painter, pelvis, Some(label));
        Ok(g)
    }
}
impl MeasureComponent for PelvicObliquity<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        tilt_angle("Pelvis", self.0.pelvis.0.view())
    }
}

/// Sacral Obliquity (p.70)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct SacralObliquity<'a>(&'a CoronalPoints);
impl CoronalComponent for SacralObliquity<'_> {}
impl DrawComponent for SacralObliquity<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let coronal_points = self.0;
        coronal_points
            .femoral_head
            .0
            .validate_label_length("FemoralHead", 2)?;
        let label = self.id();
        let color = line_colors.get_or_new(label);
        let mut g = self.default_group().set("fill", color).set("stroke", color);
        let femoral_head = &coronal_points.femoral_head.0;
        for c in femoral_head.axis_iter(Axis(0)) {
            g = g.add(painter.point(c));
        }
        g = g.add(painter.line(femoral_head.view()));
        let sac_line = points2line(coronal_points.spine.sacral_sup_plate());
        let line_eqn = sac_line.equation();
        let mut sac_seg = femoral_head.clone();
        sac_seg[[0, 1]] = line_eqn.solve_y_for_x(femoral_head[[0, 0]]).unwrap();
        sac_seg[[1, 1]] = line_eqn.solve_y_for_x(femoral_head[[1, 0]]).unwrap();
        g = g.add(painter.line(sac_seg.view()));
        let mut hor_line = sac_seg.clone();
        hor_line[[1, 1]] = hor_line[[0, 1]];

        g = painter
            .angle_between(
                g,
                hor_line.view(),
                sac_seg.view(),
                hor_line.index_axis(Axis(0), 0),
                0.8 * sac_seg
                    .index_axis(Axis(0), 0)
                    .l2_dist(&sac_seg.index_axis(Axis(0), 1))
                    .unwrap(),
                Some(label),
            )
            .0;
        Ok(g)
    }
}
impl MeasureComponent for SacralObliquity<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        tilt_angle("FemoralHead", self.0.femoral_head.0.view())
    }
}

/// Leg Length Discrepancy (p.69)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_DISTANCE])]
pub struct LegLengthDiscrepancy<'a>(&'a CoronalPoints);
impl CoronalComponent for LegLengthDiscrepancy<'_> {}
impl DrawComponent for LegLengthDiscrepancy<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = line_colors.get_or_new(self.id());
        let g = self.default_group().set("fill", color).set("stroke", color);
        draw_difference_in_y(
            g,
            "FemoralHead",
            self.id(),
            self.0.femoral_head.0.view(),
            painter,
            self.0.image_metadata.unit.as_str(),
        )
    }
}
impl MeasureComponent for LegLengthDiscrepancy<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let coronal_points = self.0;
        coronal_points
            .femoral_head
            .0
            .validate_label_length("FemoralHead", 2)?;
        let points = coronal_points.femoral_head.0.view();
        let dy = points.index_axis(Axis(0), 0)[1] - points.index_axis(Axis(0), 1)[1];
        Ok(dy)
    }
}

impl<'a, 'b> From<(&'b CoronalDraw, &'a CoronalPointsAndCurve)> for Box<dyn DrawComponent + 'a> {
    fn from(value: (&'b CoronalDraw, &'a CoronalPointsAndCurve)) -> Self {
        let (measure, coronal_set) = value;
        let coronal_points = &coronal_set.coronal_points;
        let curve_set = &coronal_set.curves.curves;
        let apex_set = &coronal_set.curves.apices;
        match measure {
            CoronalDraw::CobbPT => Box::new(CobbPT(coronal_points, curve_set.pt.clone())),
            CoronalDraw::CobbMT => Box::new(CobbMT(coronal_points, curve_set.mt.clone())),
            CoronalDraw::CobbTLL => Box::new(CobbTLL(coronal_points, curve_set.tll.clone())),

            CoronalDraw::CurveApex => Box::new(CurveApex(coronal_points, apex_set)),
            CoronalDraw::CSVL => Box::new(Csvl(coronal_points, apex_set)),
            CoronalDraw::T1TiltAngle => Box::new(T1TiltAngle(coronal_points)),
            CoronalDraw::CoronalBalance => Box::new(CoronalBalance(coronal_points)),
            CoronalDraw::ClavicleAngle => Box::new(ClavicleAngle(coronal_points)),
            CoronalDraw::ShoulderHeight => Box::new(ShoulderHeight(coronal_points)),
            CoronalDraw::PelvicObliquity => Box::new(PelvicObliquity(coronal_points)),
            CoronalDraw::SacralObliquity => Box::new(SacralObliquity(coronal_points)),
            CoronalDraw::LegLengthDiscrepancy => Box::new(LegLengthDiscrepancy(coronal_points)),

            CoronalDraw::VertebralLabels => Box::new(VertebralLabels(&coronal_points.spine)),
            CoronalDraw::VertebralPoints => Box::new(VertebralPoints(&coronal_points.spine)),
            CoronalDraw::Centroids => Box::new(Centroids(&coronal_points.spine)),
            CoronalDraw::SpinalLine => Box::new(SpinalLine(&coronal_points.spine)),
        }
    }
}

impl<'a, 'b> From<(&'b CoronalMeasure, &'a CoronalPointsAndCurve)>
    for Box<dyn MeasureComponent + 'a>
{
    fn from(value: (&'b CoronalMeasure, &'a CoronalPointsAndCurve)) -> Self {
        let (measure, coronal_points_and_curve) = value;
        let coronal_points = &coronal_points_and_curve.coronal_points;
        let curve_set = &coronal_points_and_curve.curves.curves;
        match measure {
            CoronalMeasure::CobbPT => Box::new(CobbPT(coronal_points, curve_set.pt.clone())),
            CoronalMeasure::CobbMT => Box::new(CobbMT(coronal_points, curve_set.mt.clone())),
            CoronalMeasure::CobbTLL => Box::new(CobbTLL(coronal_points, curve_set.tll.clone())),

            CoronalMeasure::T1TiltAngle => Box::new(T1TiltAngle(coronal_points)),
            CoronalMeasure::CoronalBalance => Box::new(CoronalBalance(coronal_points)),
            CoronalMeasure::ClavicleAngle => Box::new(ClavicleAngle(coronal_points)),
            CoronalMeasure::ShoulderHeight => Box::new(ShoulderHeight(coronal_points)),
            CoronalMeasure::PelvicObliquity => Box::new(PelvicObliquity(coronal_points)),
            CoronalMeasure::SacralObliquity => Box::new(SacralObliquity(coronal_points)),
            CoronalMeasure::LegLengthDiscrepancy => Box::new(LegLengthDiscrepancy(coronal_points)),
        }
    }
}
