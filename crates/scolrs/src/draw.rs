use crate::{angle_from_lines, CoronalPoints, L2Norm, SagittalMeasure, SagittalPoints};
use crate::{
    ApexSet, Corners, Curve, CurveSet, DrawParam, Spine, VertebralIndex, VERTEBRAL_LABELS,
};
use labelme_rs::LabelMeDataWImage;
use log::{debug, warn};
use ndarray::{s, stack, Array2, ArrayBase, ArrayView2, Axis, Ix1, Ix2};
use ndarray_stats::DeviationExt;
use std::collections::HashMap;
use std::io::Read;
use std::ops::{AddAssign, SubAssign};
use svg::node::element;
pub type LineColors = HashMap<String, String>;

#[derive(Debug, serde::Deserialize)]
struct LineColor {
    label: String,
    color: String,
}

pub fn load_line_colors<S: Read>(reader: S) -> Result<LineColors, csv::Error> {
    let mut rdr = csv::Reader::from_reader(reader);
    let mut colors = LineColors::new();
    for result in rdr.deserialize() {
        let lc: LineColor = result?;
        colors.insert(lc.label, lc.color);
    }
    Ok(colors)
}

pub struct Painter {
    pub param: DrawParam,
    pub size: (usize, usize),
}

/// Squared distance between vectors. i.e. `(p1 - p2)^2`
fn squared_distance<S>(p1: ArrayBase<S, Ix1>, p2: ArrayBase<S, Ix1>) -> f32
where
    S: ndarray::Data<Elem = f32>,
{
    (&p1 - &p2).mapv(|a| a * a).sum()
}

/// Signed angle from line1 to line2 in radians
/// TODO: Check the difference from [`angle_from_lines`]?
pub fn angle_between<S>(line1: ArrayBase<S, Ix2>, line2: ArrayBase<S, Ix2>) -> f32
where
    S: ndarray::Data<Elem = f32>,
{
    let v = &line1.index_axis(Axis(0), 1) - &line1.index_axis(Axis(0), 0);
    let w = &line2.index_axis(Axis(0), 1) - &line2.index_axis(Axis(0), 0);
    (w[1] * v[0] - w[0] * v[1]).atan2(w[0] * v[0] + w[1] * v[1])
}

/// Return the pair of vectors that has maximum distance
fn distanced_pair3<S>(
    p1: ArrayBase<S, Ix1>,
    p2: ArrayBase<S, Ix1>,
    p3: ArrayBase<S, Ix1>,
) -> (ArrayBase<S, Ix1>, ArrayBase<S, Ix1>)
where
    S: ndarray::Data<Elem = f32>,
{
    let d1 = squared_distance(p1.view(), p2.view());
    let d2 = squared_distance(p2.view(), p3.view());
    let d3 = squared_distance(p1.view(), p3.view());
    if d1 > d3 && d1 > d2 {
        (p1, p2)
    } else if d2 > d3 && d2 > d1 {
        (p2, p3)
    } else {
        (p1, p3)
    }
}

trait JoinWith {
    fn join(&self, delim: &str) -> String;
}

impl<S, D> JoinWith for ArrayBase<S, D>
where
    S: ndarray::Data<Elem = f32>,
    D: ndarray::Dimension,
{
    fn join(&self, delim: &str) -> String {
        self.into_iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(delim)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CobbAux {
    plate_scale: f32,
    perpendicular_scale: f32,
}

impl Default for CobbAux {
    fn default() -> Self {
        Self {
            plate_scale: 3.0,
            perpendicular_scale: 1.4,
        }
    }
}

impl CobbAux {
    fn opposite_default() -> Self {
        let mut dft = Self::default();
        dft.plate_scale *= -1.0;
        dft
    }
}

fn rotate_around<S, T>(
    point: ArrayBase<S, Ix1>,
    origin: ArrayBase<T, Ix1>,
    rad_angle: f32,
) -> ndarray::Array1<f32>
where
    S: ndarray::Data<Elem = f32>,
    T: ndarray::Data<Elem = f32>,
{
    let point = &point - &origin;
    let s = rad_angle.sin();
    let c = rad_angle.cos();
    let rot = ndarray::arr2(&[[c, -s], [s, c]]);
    let mut rotated = point.dot(&rot);
    rotated.add_assign(&origin);
    rotated
}

impl Painter {
    pub fn new(param: DrawParam, size: (usize, usize)) -> Self {
        Self { param, size }
    }

    pub fn text<S>(
        &self,
        text: &str,
        coords: ArrayBase<S, Ix1>,
        title: Option<&str>,
    ) -> element::Text
    where
        S: ndarray::Data<Elem = f32>,
    {
        let t = element::Text::new()
            .set("x", coords[0])
            .set("y", coords[1])
            .add(svg::node::Text::new(text));
        if let Some(title) = title {
            t.add(self.title(title))
        } else {
            t
        }
    }

    pub fn title(&self, text: &str) -> element::Title {
        element::Title::new().add(svg::node::Text::new(text))
    }

    pub fn point<S>(&self, point: ArrayBase<S, Ix1>) -> element::Circle
    where
        S: ndarray::Data<Elem = f32>,
    {
        element::Circle::new()
            .set("cx", point[0])
            .set("cy", point[1])
            .set("r", self.param.radius)
    }

    pub fn line<S>(&self, start_end: ArrayBase<S, Ix2>) -> element::Line
    where
        S: ndarray::Data<Elem = f32>,
    {
        element::Line::new()
            .set("x1", start_end[[0, 0]])
            .set("y1", start_end[[0, 1]])
            .set("x2", start_end[[1, 0]])
            .set("y2", start_end[[1, 1]])
    }

    pub fn horizontal_line(&self, y: f32) -> element::Line {
        element::Line::new()
            .set("x1", 0)
            .set("y1", y)
            .set("x2", self.size.0)
            .set("y2", y)
    }

    #[allow(dead_code)]
    pub fn vertical_line(&self, x: f32) -> element::Line {
        element::Line::new()
            .set("x1", x)
            .set("y1", 0)
            .set("x2", x)
            .set("y2", self.size.1)
    }

    pub fn polyline<S>(&self, points: ArrayBase<S, Ix2>) -> element::Polyline
    where
        S: ndarray::Data<Elem = f32>,
    {
        let s = points.join(" ");
        element::Polyline::new().set("points", s)
    }

    pub fn polygon<S>(&self, points: ArrayBase<S, Ix2>) -> element::Polygon
    where
        S: ndarray::Data<Elem = f32>,
    {
        let s = points.join(" ");
        element::Polygon::new().set("points", s)
    }

    pub fn angle_between<S>(
        &self,
        mut group: element::Group,
        line1: ArrayBase<S, Ix2>,
        line2: ArrayBase<S, Ix2>,
        cross: ArrayBase<S, Ix1>,
        arc_radius: f32,
        title: Option<&str>,
    ) -> (element::Group, f32)
    where
        S: ndarray::Data<Elem = f32>,
    {
        group = group.add(self.line(line1.view()));
        group = group.add(self.line(line2.view()));
        let v1 = &line1.index_axis(Axis(0), 1) - &line1.index_axis(Axis(0), 0);
        let v1_unit = &v1 / v1.l2norm();
        let v2 = &line2.index_axis(Axis(0), 1) - &line2.index_axis(Axis(0), 0);
        let v2_unit = &v2 / v2.l2norm();
        let arc_start = &cross + arc_radius * &v1_unit;
        let arc_end = &cross + arc_radius * &v2_unit;
        let large_arc_flag = 0;
        let angle_rad = angle_between(line1, line2);
        let sweep_flag = if angle_rad < 0.0 { 1 } else { 0 };
        let angle_deg = angle_rad.to_degrees();
        let data = element::path::Data::new()
            .move_to((arc_start[0], arc_start[1]))
            .elliptical_arc_to((
                arc_radius,
                arc_radius,
                0,
                large_arc_flag,
                sweep_flag,
                arc_end[0],
                arc_end[1],
            ));
        let arc = element::Path::new().set('d', data).set("fill", "none");
        group = group.add(arc);
        let text = self.text(
            format!("{:.1}°", angle_deg).as_str(),
            rotate_around(arc_start.view(), cross.view(), angle_rad / 2.0),
            title,
        );
        (group.add(text), angle_deg)
    }

    pub fn plate_end<S, T>(plate: ArrayBase<S, Ix2>, point: ArrayBase<T, Ix1>) -> usize
    where
        S: ndarray::Data<Elem = f32>,
        T: ndarray::Data<Elem = f32>,
    {
        let p0_p1: ndarray::Array1<_> = &plate.slice(s![1, ..]) - &plate.slice(s![0, ..]);
        let p0_pts: ndarray::Array1<_> = &point - &plate.slice(s![0, ..]);
        let dot_prod = p0_p1.dot(&p0_pts);
        if dot_prod > 0.0 {
            0
        } else {
            1
        }
    }

    pub fn cobb_from_plates(
        &self,
        mut group: element::Group,
        sup_plate: Array2<f32>,
        inf_plate: Array2<f32>,
        aux_param: &CobbAux,
        base_length: f32,
        title: Option<&str>,
    ) -> element::Group {
        let sup_line = points2line(sup_plate.view());
        let inf_line = points2line(inf_plate.view());

        let linter = sup_line.intersection(&inf_line);
        if let Some(intersection) = linter {
            let is_inside = intersection.x > 0.0
                && intersection.x < self.size.0 as f32
                && intersection.y > 0.0
                && intersection.y < self.size.1 as f32;
            let angle = angle_from_lines(sup_plate.view(), inf_plate.view()).unwrap(); // lines can't be parallel if there is an intersection point

            let arr_int = ndarray::arr1(&[intersection.x, intersection.y]);
            let plate_origin = Self::plate_end(sup_plate.view(), arr_int.view());
            let dir_plate =
                &sup_plate.slice(s![1 - plate_origin, ..]) - &sup_plate.slice(s![plate_origin, ..]);
            let unit_dir = &dir_plate / dir_plate.l2norm();
            let aux_on_sup: ndarray::Array1<_> = &sup_plate.slice(s![plate_origin, ..])
                + aux_param.plate_scale * base_length * &unit_dir;
            let aux_cross =
                rotate_around(aux_on_sup.view(), arr_int.view(), angle.to_radians() / 2.0);

            let d_btw_aux2p = aux_cross
                .l2_dist(&sup_plate.slice(s![plate_origin, ..]))
                .unwrap();
            let arr_int = ndarray::arr1(&[intersection.x, intersection.y]);
            let d_btw_int2p = arr_int
                .l2_dist(&sup_plate.slice(s![plate_origin, ..]))
                .unwrap();
            if is_inside && d_btw_aux2p > d_btw_int2p {
                // draw intersection point
                for plate in [sup_plate.view(), inf_plate.view()] {
                    let i = Self::plate_end(plate, arr_int.view());
                    let line = self.line(ndarray::arr2(&[
                        [plate[[i, 0]], plate[[i, 1]]],
                        [intersection.x, intersection.y],
                    ]));
                    group = group.add(line);
                }
                let text = self.text(
                    format!("{:.1}°", angle).as_str(),
                    ndarray::arr1(&[intersection.x, intersection.y]),
                    title,
                );

                group = group.add(text);
            } else {
                // draw aux lines and its intersection

                for plate in [sup_plate.view(), inf_plate.view()] {
                    let plate_origin = Self::plate_end(plate, arr_int.view());
                    let dir_plate =
                        &plate.slice(s![1 - plate_origin, ..]) - &plate.slice(s![plate_origin, ..]);
                    let d_aux = &aux_cross - &plate.slice(s![plate_origin, ..]);
                    let t = d_aux.dot(&dir_plate) / dir_plate.mapv(|a| a * a).sum();
                    let projed_aux = &plate.slice(s![plate_origin, ..]) + t * &dir_plate;
                    let (p1, p2) = distanced_pair3(
                        plate.slice(s![0, ..]),
                        plate.slice(s![1, ..]),
                        projed_aux.view(),
                    );
                    let line = self.line(ndarray::stack![Axis(0), p1, p2]);
                    group = group.add(line);
                    let pa_a = &aux_cross - &projed_aux;
                    let line = self.line(ndarray::stack![
                        Axis(0),
                        aux_param.perpendicular_scale * pa_a + &projed_aux,
                        projed_aux
                    ]);
                    group = group.add(line);
                }
                let text = self.text(format!("{:.1}°", angle).as_str(), aux_cross, title);
                group = group.add(text);
            }
        } else {
            // parallel lines
            for plate in [sup_plate, inf_plate] {
                let mut line = plate.to_owned();
                let d = &plate.slice(s![1, ..]) - &plate.slice(s![0, ..]);
                line.slice_mut(s![0, ..]).sub_assign(&d);
                line.slice_mut(s![1, ..]).add_assign(&d);
                let line = self.line(line);
                group = group.add(line);
                // arrow
                let unit_d = &d / d.l2norm();
                let len = self.param.line_width * 8.0;
                let p1 = plate.mean_axis(Axis(0)).unwrap();
                let p2 = rotate_around(&p1 - &unit_d * len, p1.view(), 30.0_f32.to_radians());
                let p3 = rotate_around(&p1 - &unit_d * len, p1.view(), -30.0_f32.to_radians());
                let arrow = ndarray::stack![Axis(0), p1, p2, p3];
                let arrow = self.polygon(arrow);

                group = group.add(arrow);
            }
        }
        group
    }

    pub fn cobb(
        &self,
        group: element::Group,
        spine: &Spine,
        curve: &Curve,
        aux_param: &CobbAux,
        base_length: f32,
        title: Option<&str>,
    ) -> element::Group {
        let sup_plate = spine.sup_plate(curve.sup);
        let inf_plate = spine.inf_plate(curve.inf);
        self.cobb_from_plates(
            group,
            sup_plate.to_owned(),
            inf_plate.to_owned(),
            aux_param,
            base_length,
            title,
        )
    }

    pub fn doc_w_background(&self, image: &labelme_rs::image::DynamicImage) -> svg::Document {
        let (w, h) = self.size;
        let mut document = svg::Document::new()
            .set("width", w)
            .set("height", h)
            .set("viewBox", (0i64, 0i64, w, h))
            .set("xmlns:xlink", "http://www.w3.org/1999/xlink");
        let b64 = format!(
            "data:image/jpeg;base64,{}",
            labelme_rs::img2base64(image, labelme_rs::image::ImageOutputFormat::Jpeg(75))
        );
        let bg = element::Image::new()
            .set("x", 0i64)
            .set("y", 0i64)
            .set("width", w)
            .set("height", h)
            .set("xlink:href", b64);
        document = document.add(bg);
        document
    }
}

fn points2line<S>(plate: ArrayBase<S, Ix2>) -> lyon_geom::Line<f32>
where
    S: ndarray::Data<Elem = f32>,
{
    let point = lyon_geom::Point::new(plate[[0, 0]], plate[[0, 1]]);
    let p2 = lyon_geom::Point::new(plate[[1, 0]], plate[[1, 1]]);
    let vector = p2 - point;
    lyon_geom::Line { point, vector }
}

type ColorMap = HashMap<String, String>;

pub struct ColorPalette {
    color_map: HashMap<String, String>,
    color_cycler: labelme_rs::ColorCycler,
}

impl ColorPalette {
    pub fn new(color_map: ColorMap) -> ColorPalette {
        let color_cycler = labelme_rs::ColorCycler::default();
        Self {
            color_map,
            color_cycler,
        }
    }

    pub fn get_or_new<Q>(&mut self, key: &Q) -> &str
    where
        String: std::borrow::Borrow<Q>,
        Q: ?Sized + core::hash::Hash + std::cmp::Eq + std::fmt::Display,
    {
        self.color_map.get(key).map_or_else(
            || {
                debug!("New color generated for {}", key);
                self.color_cycler.cycle()
            },
            |s| s.as_str(),
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub enum InvalidNumberOfPoints {
    /// Too few points, expected and actual
    #[error("Too few points, expected: {0}, actual: {1}")]
    TooFewPoints(usize, usize),
    /// Too many points, expected and actual
    #[error("Too many points, expected: {0}, actual: {1}")]
    TooManyPoints(usize, usize),
    /// Incorrect number of points, expected and actual
    #[error("Incorrect number of points, expected: {0}, actual: {1}")]
    IncorrectNumberOfPoints(usize, usize),
}

#[derive(Debug, thiserror::Error)]
pub enum MeasureError {
    // Invalid number of points
    #[error("Invalid number of points")]
    InvalidNumberOfPoints(#[from] InvalidNumberOfPoints),

    // Zero length line
    #[error("Zero length line")]
    ZeroLengthLine,
}

pub trait Named {
    fn name(&self) -> &'static str;
    fn draw_type(&self) -> &[&'static str];
    fn ided_group(&self) -> element::Group {
        element::Group::new().set("id", self.name())
    }
    fn default_group_w_classes(&self, classes: &[&str]) -> element::Group {
        let mut classes = Vec::from(classes);
        classes.extend_from_slice(self.draw_type());
        self.ided_group().set("class", classes)
    }
}

const COMMON_COMPONENT_CLASS: &str = "CommonComponent";
pub trait CommonComponent: Named {
    fn draw(
        &self,
        spine: &Spine,
        painter: &Painter,
        label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> element::Group;
    fn default_group(&self) -> element::Group {
        self.default_group_w_classes(&[COMMON_COMPONENT_CLASS])
    }
    // Create a group with extra classes
    // fn group_with_extra_class(&self, extra_classes: &[&str]) -> element::Group {
    //     let mut classes = vec![COMMON_COMPONENT_CLASS];
    //     classes.extend_from_slice(extra_classes);
    //     self.ided_group().set("class", classes.join(" "))
    // }
}

pub trait CoronalComponent {
    fn draw(
        &self,
        coronal_points: &CoronalPoints,
        painter: &Painter,
        label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Option<element::Group>;
}

pub struct VertebralLabels;
impl Named for VertebralLabels {
    fn name(&self) -> &'static str {
        "Centroids"
    }

    fn draw_type(&self) -> &[&'static str] {
        &["Annotation", "Text"]
    }
}

impl CommonComponent for VertebralLabels {
    fn draw(
        &self,
        spine: &Spine,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> element::Group {
        let mut g_vert_labels = self.default_group();
        let centroids = spine.tl_centroids();
        for (coords, label) in
            std::iter::zip(centroids.axis_iter(Axis(0)), VERTEBRAL_LABELS.into_iter())
        {
            let t = painter.text(label, coords, None);
            g_vert_labels = g_vert_labels.add(t);
        }
        g_vert_labels
    }
}

pub struct VertebralPoints;
impl Named for VertebralPoints {
    fn name(&self) -> &'static str {
        "VertebralPoints"
    }

    fn draw_type(&self) -> &[&'static str] {
        &["Annotation", "Point"]
    }
}
impl CommonComponent for VertebralPoints {
    fn draw(
        &self,
        spine: &Spine,
        painter: &Painter,
        label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> element::Group {
        let mut g_corners = element::Group::new();
        for (i_label, &label) in crate::CORNER_LABELS.iter().enumerate() {
            let color = label_colors.get_or_new(label);
            let mut sub_group = self.default_group().set("stroke", color).set("fill", color);
            let points = spine.c7tls.0.index_axis(Axis(1), i_label);
            let n_points = if i_label < 2 {
                points.len_of(Axis(0))
            } else {
                points.len_of(Axis(0)) - 1 // BL and BR points of sacrum are dummies
            };
            for point in points.axis_iter(Axis(0)).take(n_points) {
                let p = painter.point(point);
                sub_group = sub_group.add(p);
            }
            g_corners = g_corners.add(sub_group);
        }
        g_corners
    }
}

const ID_CENTROID: &str = "Centroid";
const ID_COB_ANGLES: &str = "CobbAngles";
const ID_CURVE_APEX: &str = "CurveApex";
const ID_SPINAL_LINE: &str = "SpinalLine";

const ID_CSVL: &str = "CSVL";
const ID_T1_TILT_ANGLE: &str = "T1Tilt";
const ID_CORONAL_BALANCE: &str = "CoronalBalance";
const ID_CLAVICLE_ANGLE: &str = "ClavicleAngle";
const ID_SHOULDER_HEIGHT: &str = "ShoulderHeight";
const ID_PELVIC_OBLIQUITY: &str = "PelvicObliquity";
const ID_SACRAL_OBLIQUITY: &str = "SacralObliquity";
const ID_LEG_LEN_DISCREPANCY: &str = "LegLengthDiscrepancy";

static CORONAL_COMPONENTS: [&str; 12] = [
    ID_CENTROID,
    ID_COB_ANGLES,
    ID_CURVE_APEX,
    ID_SPINAL_LINE,
    ID_CSVL,
    ID_T1_TILT_ANGLE,
    ID_CORONAL_BALANCE,
    ID_CLAVICLE_ANGLE,
    ID_SHOULDER_HEIGHT,
    ID_PELVIC_OBLIQUITY,
    ID_SACRAL_OBLIQUITY,
    ID_LEG_LEN_DISCREPANCY,
];

struct Centroids;
impl Named for Centroids {
    fn name(&self) -> &'static str {
        "Centroids"
    }

    fn draw_type(&self) -> &[&'static str] {
        &["Annotation", "Point"]
    }
}
impl CommonComponent for Centroids {
    fn draw(
        &self,
        spine: &Spine,
        painter: &Painter,
        label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> element::Group {
        let label = ID_CENTROID;
        let color = label_colors.get_or_new(label);
        let mut g_centroids = self.default_group().set("stroke", color).set("fill", color);
        let centroids = spine.tl_centroids();
        for point in centroids.axis_iter(Axis(0)) {
            let p = painter.point(point);
            g_centroids = g_centroids.add(p);
        }
        g_centroids
    }
}

fn mean_plate_length(scol: &Spine) -> f32 {
    let corners = scol.tl_corners().0;
    let sup_inf_shape = (corners.len_of(Axis(0)) * 2, 2, 2); // [n * sup_inf, lr, xy]

    let plate_lr = corners.to_shape(sup_inf_shape).unwrap();
    let diff = &plate_lr.index_axis(Axis(1), 0) - &plate_lr.index_axis(Axis(1), 1);
    diff.map_axis(Axis(1), |a| a.l2norm()).mean().unwrap()
}

struct CobbAngles<'a>(&'a CurveSet);
impl<'a> CoronalComponent for CobbAngles<'a> {
    fn draw(
        &self,
        coronal_points: &CoronalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Option<element::Group> {
        let spine = &coronal_points.spine;
        let curve_set = self.0;
        debug!("Curve set:{:?}", curve_set);
        let mut g_angles = element::Group::new().set("class", "CobbAngles");
        let aux_param = CobbAux::default();

        let mean_plate_length = mean_plate_length(spine);

        if let Some((mt_curve, _angle)) = &curve_set.mt {
            let g_mt = element::Group::new()
                .set("class", "MT")
                .set("stroke", line_colors.get_or_new("MT"));
            let group = painter.cobb(
                g_mt,
                spine,
                mt_curve,
                &aux_param,
                mean_plate_length,
                Some("MT"),
            );
            g_angles = g_angles.add(group);
        };

        if let Some((pt_curve, _angle)) = &curve_set.pt {
            let g_pt = element::Group::new()
                .set("class", "PT")
                .set("stroke", line_colors.get_or_new("PT"));
            let group = painter.cobb(
                g_pt,
                spine,
                pt_curve,
                &aux_param,
                mean_plate_length,
                Some("PT"),
            );
            g_angles = g_angles.add(group);
        }

        if let Some((tll_curve, _angle)) = &curve_set.tll {
            let g_tll = element::Group::new()
                .set("class", "TLL")
                .set("stroke", line_colors.get_or_new("TLL"));
            let group = painter.cobb(
                g_tll,
                spine,
                tll_curve,
                &aux_param,
                mean_plate_length,
                Some("TLL"),
            );
            g_angles = g_angles.add(group);
        }
        Some(g_angles)
    }
}

struct CurveApex<'a>(&'a ApexSet, &'a Corners<ndarray::OwnedRepr<f32>>);
impl<'a> CoronalComponent for CurveApex<'a> {
    fn draw(
        &self,
        _coronal_points: &CoronalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Option<element::Group> {
        let apex_set = &self.0;
        let vert_discs = &self.1 .0;
        let label = ID_CURVE_APEX;
        let mut g = element::Group::new().set("class", label);

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
        Some(g)
    }
}

/// Spinal center line
struct SpinalLine;
impl Named for SpinalLine {
    fn name(&self) -> &'static str {
        "SpinalLine"
    }

    fn draw_type(&self) -> &[&'static str] {
        &["Annotation", "Line"]
    }
}
impl CommonComponent for SpinalLine {
    fn draw(
        &self,
        spine: &Spine,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> element::Group {
        let label = ID_SPINAL_LINE;
        let g = element::Group::new().set("class", label);
        let centroids = spine.tl_centroids();
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

        g.add(spinal_line)
    }
}

/// center sacral vertical line (CSVL)
struct Csvl<'a>(&'a ApexSet);
impl<'a> CoronalComponent for Csvl<'a> {
    fn draw(
        &self,
        coronal_points: &CoronalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Option<element::Group> {
        let spine = &coronal_points.spine;
        let label = ID_CSVL;
        let line_color = line_colors.get_or_new(label);
        let mut g = element::Group::new()
            .set("class", label)
            .set("stroke", line_color);
        let sup_plate = spine.sacral_sup_plate();
        let sacral_line = painter.line(sup_plate.view());
        g = g.add(sacral_line);
        if let Some(tll) = self.0.tll {
            let v_idx = ((tll as u8) / 2 - 1).max(0) as usize; // one level above the tll apex
            let mid = sup_plate.mean_axis(Axis(0)).unwrap();
            let mut vl = ndarray::stack![Axis(0), mid, mid];
            let y = spine.c7tls.0[[v_idx + 1, 0, 1]]; // v_idx+1 because vertebrae include c7
            vl[[0, 1]] = y;
            g = g.add(painter.line(vl));
        }
        Some(g)
    }
}

/// center sacral vertical line (CSVL)
struct T1TiltAngle;
impl CoronalComponent for T1TiltAngle {
    fn draw(
        &self,
        coronal_points: &CoronalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Option<element::Group> {
        let spine = &coronal_points.spine;
        let label = ID_T1_TILT_ANGLE;
        let mut g = element::Group::new().set("class", label);
        g = g
            .set("stroke", line_colors.get_or_new(label))
            .set("fill", "none");
        let tl_sup_lines = spine.tl_sup_lines();
        let t1sup = tl_sup_lines.index_axis(Axis(0), 0);
        let mid = t1sup.mean_axis(Axis(0)).unwrap();
        let (mult_left, mult_right, mult_arc) = (1.0, 4.0, 3.0);
        let l2r = &t1sup.index_axis(Axis(0), 1) - mult_left * &t1sup.index_axis(Axis(0), 0);
        let sup_line = stack![Axis(0), &mid - &l2r, &mid + mult_right * &l2r];
        g = g.add(painter.line(sup_line.view()));
        if l2r[0] == 0.0 {
            // T1 is vertical, which is highly unlikely
            debug!("T1 VERTICAL LINE!!!"); // TODO: implement
        } else {
            let mut arc_start = mid.to_owned();
            let arc_radius = mult_arc * l2r.l2norm();
            arc_start[[0]] += arc_radius;

            if l2r[1] != 0.0 {
                // draw tilted T1 line
                let mut hor_line = stack![Axis(0), mid.view(), mid.view()];
                hor_line[[0, 0]] -= mult_left * l2r.l2norm();
                hor_line[[1, 0]] += mult_right * l2r.l2norm();
                g = painter
                    .angle_between(
                        g,
                        sup_line.view(),
                        hor_line.view(),
                        mid.view(),
                        arc_radius,
                        Some(label),
                    )
                    .0;
            }
        };
        Some(g)
    }
}

struct CoronalBalance;
impl CoronalComponent for CoronalBalance {
    fn draw(
        &self,
        coronal_points: &CoronalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Option<element::Group> {
        let spine = &coronal_points.spine;
        let label = ID_CORONAL_BALANCE;
        let c_c7 = spine.c_c7tl.index_axis(Axis(0), 0);
        let sac_sup = spine.sacral_sup_plate();
        let mid_sac = sac_sup.mean_axis(Axis(0)).unwrap();
        let points = stack![Axis(0), c_c7, mid_sac];
        let color = line_colors.get_or_new(label);
        let g = element::Group::new()
            .set("class", label)
            .set("fill", color)
            .set("stroke", color);
        let g = difference_in_x(g, label, points.view(), painter);
        Some(g)
    }
}

struct ClavicleAngle;
impl CoronalComponent for ClavicleAngle {
    fn draw(
        &self,
        coronal_points: &CoronalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Option<element::Group> {
        coronal_points.clavicle.as_ref()?;
        let label = ID_CLAVICLE_ANGLE;
        let color = line_colors.get_or_new(label);
        let mut g = element::Group::new()
            .set("class", label)
            .set("fill", color)
            .set("stroke", color);
        let clavicle = coronal_points.clavicle.as_ref().unwrap();
        g = add_tilt_angle(g, painter, clavicle, Some(label));
        Some(g)
    }
}

fn difference_in_x(
    group: element::Group,
    label: &str,
    points: ArrayView2<f32>,
    painter: &Painter,
) -> element::Group {
    let mut g = group;
    for c in points.axis_iter(Axis(0)) {
        g = g.add(painter.point(c));
    }
    let p1 = points.index_axis(Axis(0), 0);
    let p2 = points.index_axis(Axis(0), 1);
    let len = 0.4 * (p2[1] - p1[1]);

    let mut v_line = points.to_owned();
    v_line[[1, 0]] = v_line[[0, 0]];
    v_line[[1, 1]] -= len;
    g = g.add(painter.line(v_line.view()));

    let mut v_line = points.to_owned();
    v_line[[0, 0]] = v_line[[1, 0]];
    v_line[[0, 1]] += len;
    g = g.add(painter.line(v_line.view()));

    let mut h_line = points.to_owned();
    h_line[[0, 1]] = (p1[1] + p2[1]) / 2.0;
    h_line[[1, 1]] = h_line[[0, 1]];
    g = g.add(painter.line(h_line.view()));
    let dx = p1[0] - p2[0];
    let text = painter.text(
        &format!("{:.1} {}", dx, painter.param.len_unit),
        h_line.index_axis(Axis(0), 1),
        Some(label),
    );

    g = g.add(text);
    g
}

fn difference_in_y(
    label: &str,
    points: &Option<Array2<f32>>,
    painter: &Painter,
    line_colors: &mut ColorPalette,
) -> Option<element::Group> {
    let points = points.as_ref()?;
    let color = line_colors.get_or_new(label);
    let mut g = element::Group::new()
        .set("class", label)
        .set("fill", color)
        .set("stroke", color);
    for c in points.axis_iter(Axis(0)) {
        g = g.add(painter.point(c));
    }
    g = g.add(painter.horizontal_line(points[[0, 1]]));
    g = g.add(painter.horizontal_line(points[[1, 1]]));
    let mut vline = points.clone();
    vline[[1, 0]] = vline[[0, 0]];
    g = g.add(painter.line(vline.view()));
    let text_pos = vline.mean_axis(Axis(0)).unwrap();
    let text = format!(
        "{:.1} {}",
        vline[[0, 1]] - vline[[1, 1]],
        painter.param.len_unit
    );
    let text = painter.text(&text, text_pos, Some(label));

    g = g.add(text);
    Some(g)
}

struct ShoulderHeight;
impl CoronalComponent for ShoulderHeight {
    fn draw(
        &self,
        coronal_points: &CoronalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Option<element::Group> {
        difference_in_y(
            ID_SHOULDER_HEIGHT,
            &coronal_points.shoulder,
            painter,
            line_colors,
        )
    }
}

fn add_tilt_angle(
    group: element::Group,
    painter: &Painter,
    points: &ndarray::Array2<f32>,
    title: Option<&str>,
) -> element::Group {
    let mut g = group;
    for p in points.axis_iter(Axis(0)) {
        g = g.add(painter.point(p));
    }
    for c in points.axis_iter(Axis(0)) {
        g = g.add(painter.point(c));
        let l2r = &points.index_axis(Axis(0), 1) - &points.index_axis(Axis(0), 0);
        let mut hor_line = points.clone();
        hor_line[[1, 1]] = points[[0, 1]];
        let arc_radius = l2r.l2norm() * 0.8;
        g = painter
            .angle_between(
                g,
                hor_line.view(),
                points.view(),
                points.index_axis(Axis(0), 0),
                arc_radius,
                title,
            )
            .0
    }
    g
}

struct PelvicObliquity;
impl CoronalComponent for PelvicObliquity {
    fn draw(
        &self,
        coronal_points: &CoronalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Option<element::Group> {
        coronal_points.pelvis.as_ref()?;
        let label = ID_PELVIC_OBLIQUITY;
        let color = line_colors.get_or_new(label);
        let mut g = element::Group::new()
            .set("class", label)
            .set("fill", color)
            .set("stroke", color);
        let pelvis = coronal_points.pelvis.as_ref().unwrap();
        g = add_tilt_angle(g, painter, pelvis, Some(label));
        Some(g)
    }
}

struct SacralObliquity;
impl CoronalComponent for SacralObliquity {
    fn draw(
        &self,
        coronal_points: &CoronalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Option<element::Group> {
        coronal_points.femoral_head.as_ref()?;
        let label = ID_SACRAL_OBLIQUITY;
        let color = line_colors.get_or_new(label);
        let mut g = element::Group::new()
            .set("class", label)
            .set("fill", color)
            .set("stroke", color);
        let femoral_head = coronal_points.femoral_head.as_ref().unwrap();
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
                    .unwrap() as f32,
                Some(label),
            )
            .0;
        Some(g)
    }
}

struct LegLengthDiscrepancy;
impl CoronalComponent for LegLengthDiscrepancy {
    fn draw(
        &self,
        coronal_points: &CoronalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Option<element::Group> {
        difference_in_y(
            ID_LEG_LEN_DISCREPANCY,
            &coronal_points.femoral_head,
            painter,
            line_colors,
        )
    }
}

fn draw_incidence_angle(
    group: element::Group,
    label: &str,
    femoral_heads: Array2<f32>,
    plate: Array2<f32>,
    painter: &Painter,
) -> element::Group {
    let mut g = group;
    let mid_femoral_heads = femoral_heads.mean_axis(Axis(0)).unwrap();
    for p in femoral_heads.axis_iter(Axis(0)) {
        g = g.add(painter.point(p));
    }
    if femoral_heads.len_of(Axis(0)) == 2 {
        g = g.add(painter.point(mid_femoral_heads.view()));
        g = g.add(painter.line(femoral_heads.view()));
    }
    g = g.add(painter.line(plate.view()));
    let sac_sup_mid = plate.mean_axis(Axis(0)).unwrap();
    g = g.add(painter.point(sac_sup_mid.view()));
    let line_sac2fem = stack![Axis(0), sac_sup_mid, mid_femoral_heads];
    g = g.add(painter.line(line_sac2fem.view()));

    let sac_p2a = &plate.index_axis(Axis(0), 1) - &plate.index_axis(Axis(0), 0);
    let mut perp_sac = ndarray::Array::from_vec(vec![-sac_p2a[1], sac_p2a[0]]);
    perp_sac /= perp_sac.l2norm();
    perp_sac = 0.25 * sac_sup_mid.l2_dist(&mid_femoral_heads).unwrap() as f32 * perp_sac;
    perp_sac += &sac_sup_mid;
    let perp_line = stack![Axis(0), sac_sup_mid, perp_sac];
    g = g.add(painter.line(perp_line.view()));
    let angle = angle_between(line_sac2fem.view(), perp_line.view()).to_degrees();
    let text = format!("{:.1}°", angle);
    let text = painter.text(&text, sac_sup_mid, Some(label));

    g = g.add(text);
    g
}

const SAGITTAL_COMPONENT_CLASS: &str = "SagittalComponent";
pub trait SagittalComponent: Named {
    fn draw(
        &self,
        sagittal_points: &SagittalPoints,
        painter: &Painter,
        label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError>;

    fn default_group(&self) -> element::Group {
        self.default_group_w_classes(&[SAGITTAL_COMPONENT_CLASS])
    }

    fn measure(&self, sagittal_points: &SagittalPoints) -> Result<f32, MeasureError>;
}

macro_rules! _impl_kyophosis {
    ($name:ident, $sup:expr, $inf:expr, $opposite:expr) => {
        impl SagittalComponent for $name {
            fn draw(
                &self,
                sagittal_points: &SagittalPoints,
                painter: &Painter,
                _label_colors: &mut ColorPalette,
                line_colors: &mut ColorPalette,
            ) -> Result<element::Group, MeasureError> {
                let label = self.name();
                let aux_param = if $opposite {
                    CobbAux::opposite_default()
                } else {
                    CobbAux::default()
                };
                let curve = Curve {
                    sup: Self::SUP,
                    inf: Self::INF,
                };
                let group = self
                    .default_group()
                    .set("stroke", line_colors.get_or_new(label));
                let mean_plate_length = mean_plate_length(&sagittal_points.spine); // TODO: remove redundant calculation
                let g = painter.cobb(
                    group,
                    &sagittal_points.spine,
                    &curve,
                    &aux_param,
                    mean_plate_length,
                    Some(label),
                );
                Ok(g)
            }

            fn measure(&self, sagittal_points: &SagittalPoints) -> Result<f32, MeasureError> {
                let angle = sagittal_points
                    .spine
                    .angle(&Curve {
                        sup: Self::SUP,
                        inf: Self::INF,
                    })
                    .unwrap();
                Ok(angle)
            }
        }
    };
}

/// Implement Kyphosis for each kyphosis type
///
/// Optionally, a display name can be provided as the second argument
macro_rules! impl_kyphosis {
    ($name:ident, $sup:expr, $inf:expr, $opposite:expr) => {
        struct $name;

        impl Named for $name {
            fn name(&self) -> &'static str {
                stringify!($name)
            }
            fn draw_type(&self) -> &[&'static str] {
                &["Measure", "Angle"]
            }
        }

        impl $name {
            const SUP: usize = $sup;
            const INF: usize = $inf;
        }
        _impl_kyophosis!($name, $sup, $inf, $opposite);
    };

    ($name:ident, $disp_name:expr, $sup:expr, $inf:expr, $opposite:expr) => {
        struct $name;

        impl Named for $name {
            fn name(&self) -> &'static str {
                $disp_name
            }
            fn draw_type(&self) -> &[&'static str] {
                &["Measure", "Angle"]
            }
        }

        impl $name {
            const SUP: usize = $sup;
            const INF: usize = $inf;
        }
        _impl_kyophosis!($name, $sup, $inf, $opposite);
    };
}

impl_kyphosis!(
    ProximalThoracicKyphosis,
    VertebralIndex::T2 as usize,
    VertebralIndex::T5 as usize,
    false
);

impl_kyphosis!(
    ThoracicKyphosis,
    VertebralIndex::T2 as usize,
    VertebralIndex::T12 as usize,
    true
);

impl_kyphosis!(
    MidLowerThoracicKyphosis,
    "Mid/LowerThoracicKyphosis",
    VertebralIndex::T5 as usize,
    VertebralIndex::T12 as usize,
    false
);

impl_kyphosis!(
    ThoracoLumbarSagittalAlignment,
    VertebralIndex::T10 as usize,
    VertebralIndex::L2 as usize,
    false
);

struct LumbarLordosis;
impl LumbarLordosis {
    const NAME: &'static str = "LumbarLordosis";
    fn prep(sagittal_points: &SagittalPoints) -> (usize, usize) {
        let sup = VertebralIndex::T12 as usize;
        let inf = sagittal_points.spine.v_c7tl.0.len_of(Axis(0)) - 1;
        (sup, inf)
    }
}
impl Named for LumbarLordosis {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn draw_type(&self) -> &[&'static str] {
        &["Measure", "Angle"]
    }
}
impl SagittalComponent for LumbarLordosis {
    fn draw(
        &self,
        sagittal_points: &SagittalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let aux_param = CobbAux::default();
        let label = self.name();
        let group = self
            .default_group()
            .set("stroke", line_colors.get_or_new(label));
        let mean_plate_length = mean_plate_length(&sagittal_points.spine); // TODO: remove redundant calculation
        let (sup, inf) = Self::prep(sagittal_points);
        let g = painter.cobb(
            group,
            &sagittal_points.spine,
            &Curve { sup, inf },
            &aux_param,
            mean_plate_length,
            Some(label),
        );
        Ok(g)
    }

    fn measure(&self, sagittal_points: &SagittalPoints) -> Result<f32, MeasureError> {
        let (sup, inf) = Self::prep(sagittal_points);
        let angle = sagittal_points.spine.angle(&Curve { sup, inf }).unwrap();
        Ok(angle)
    }
}

struct SagittalBalance;
impl SagittalBalance {
    const NAME: &'static str = "SagittalBalance";
    fn prep(sagittal_points: &SagittalPoints) -> Array2<f32> {
        let c_c7 = sagittal_points.spine.c_c7tl.index_axis(Axis(0), 0);
        let sac_sup = sagittal_points.spine.sacral_sup_plate();
        let pos_sac = sac_sup.index_axis(Axis(0), 1);
        stack![Axis(0), c_c7, pos_sac]
    }
}
impl Named for SagittalBalance {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn draw_type(&self) -> &[&'static str] {
        &["Measure", "Distance"]
    }
}
impl SagittalComponent for SagittalBalance {
    fn draw(
        &self,
        sagittal_points: &SagittalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let points = Self::prep(sagittal_points);
        let color = line_colors.get_or_new(Self::NAME);
        let g = element::Group::new()
            .set("stroke", color)
            .set("fill", color);
        let g = difference_in_x(g, Self::NAME, points.view(), painter);
        Ok(g)
    }

    fn measure(&self, sagittal_points: &SagittalPoints) -> Result<f32, MeasureError> {
        let points = Self::prep(sagittal_points);
        let p1 = points.index_axis(Axis(0), 0);
        let p2 = points.index_axis(Axis(0), 1);
        let dx = p1[0] - p2[0];
        Ok(dx)
    }
}

struct LumbosacralAngle;
impl LumbosacralAngle {
    const NAME: &'static str = "LumbosacralAngle";
    fn prep(sagittal_points: &SagittalPoints) -> (Array2<f32>, Array2<f32>) {
        let spine = &sagittal_points.spine;
        let sup = spine
            .inf_plate(spine.v_c7tl.0.len_of(Axis(0)) - 2)
            .to_owned();
        let inf = spine
            .inf_plate(spine.v_c7tl.0.len_of(Axis(0)) - 1)
            .to_owned();
        (sup, inf)
    }
}
impl Named for LumbosacralAngle {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn draw_type(&self) -> &[&'static str] {
        &["Measure", "Angle"]
    }
}
impl SagittalComponent for LumbosacralAngle {
    fn draw(
        &self,
        sagittal_points: &SagittalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let label = Self::NAME;
        let mean_plate_length = mean_plate_length(&sagittal_points.spine); // TODO:
        let group = self
            .default_group()
            .set("stroke", line_colors.get_or_new(label));
        let (sup, inf) = Self::prep(sagittal_points);
        let aux_param = CobbAux::default();
        let g =
            painter.cobb_from_plates(group, sup, inf, &aux_param, mean_plate_length, Some(label));
        Ok(g)
    }

    fn measure(&self, sagittal_points: &SagittalPoints) -> Result<f32, MeasureError> {
        let (sup, inf) = Self::prep(sagittal_points);
        let angle = angle_between(sup.view(), inf.view()).to_degrees();
        Ok(angle)
    }
}

struct PelvicIncidence;
impl PelvicIncidence {
    const NAME: &'static str = "PelvicIncidence";
}
impl Named for PelvicIncidence {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn draw_type(&self) -> &[&'static str] {
        &["Measure", "Angle"]
    }
}
impl SagittalComponent for PelvicIncidence {
    fn draw(
        &self,
        sagittal_points: &SagittalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        if sagittal_points.femoral_head.0.is_empty() {
            return Err(MeasureError::InvalidNumberOfPoints(
                InvalidNumberOfPoints::TooFewPoints(1, 0),
            ));
        }
        let label = Self::NAME;
        let sac_sup = sagittal_points.spine.sacral_sup_plate();
        let color = line_colors.get_or_new(label);
        let g = self.default_group().set("fill", color).set("stroke", color);
        let g = draw_incidence_angle(
            g,
            label,
            sagittal_points.femoral_head.0.to_owned(),
            sac_sup.to_owned(),
            painter,
        );
        Ok(g)
    }

    fn measure(&self, sagittal_points: &SagittalPoints) -> Result<f32, MeasureError> {
        let plate = sagittal_points.spine.sacral_sup_plate();
        femoral_incidence_angle(plate, &sagittal_points.femoral_head)
    }
}

struct L5IncidenceAngle;
impl L5IncidenceAngle {
    const NAME: &'static str = "L5IncidenceAngle";
}
impl Named for L5IncidenceAngle {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn draw_type(&self) -> &[&'static str] {
        &["Measure", "Angle"]
    }
}
impl SagittalComponent for L5IncidenceAngle {
    fn draw(
        &self,
        sagittal_points: &SagittalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        if sagittal_points.femoral_head.0.is_empty() {
            return Err(MeasureError::InvalidNumberOfPoints(
                InvalidNumberOfPoints::TooFewPoints(1, 0),
            ));
        }
        let label = Self::NAME;
        let l5_sup = sagittal_points
            .spine
            .sup_plate(sagittal_points.spine.v_c7tl.0.len_of(Axis(0)) - 2)
            .to_owned();
        let color = line_colors.get_or_new(label);
        let g = self.default_group().set("fill", color).set("stroke", color);
        let g = draw_incidence_angle(
            g,
            label,
            sagittal_points.femoral_head.0.to_owned(),
            l5_sup.to_owned(),
            painter,
        );
        Ok(g)
    }

    fn measure(&self, sagittal_points: &SagittalPoints) -> Result<f32, MeasureError> {
        let plate = sagittal_points
            .spine
            .sup_plate(sagittal_points.spine.v_c7tl.0.len_of(Axis(0)) - 2);
        femoral_incidence_angle(plate, &sagittal_points.femoral_head)
    }
}

struct PelvicRadiusAngle;
impl PelvicRadiusAngle {
    const NAME: &'static str = "PelvicRadiusAngle";

    fn prep(sagittal_points: &SagittalPoints) -> Result<(Array2<f32>, Array2<f32>), MeasureError> {
        if sagittal_points.femoral_head.0.is_empty() {
            return Err(MeasureError::InvalidNumberOfPoints(
                InvalidNumberOfPoints::TooFewPoints(1, 0),
            ));
        }
        let mid_femoral_heads = sagittal_points.femoral_head.0.mean_axis(Axis(0)).unwrap();
        let sac_sup = sagittal_points.spine.sacral_sup_plate();
        let post_sac = sac_sup.index_axis(Axis(0), 1);
        let line_fem2post_sac = stack![Axis(0), mid_femoral_heads, post_sac];
        Ok((sac_sup.to_owned(), line_fem2post_sac))
    }
}
impl Named for PelvicRadiusAngle {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn draw_type(&self) -> &[&'static str] {
        &["Measure", "Angle"]
    }
}
impl SagittalComponent for PelvicRadiusAngle {
    fn draw(
        &self,
        sagittal_points: &SagittalPoints,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        if sagittal_points.femoral_head.0.is_empty() {
            return Err(MeasureError::InvalidNumberOfPoints(
                InvalidNumberOfPoints::TooFewPoints(1, 0),
            ));
        }
        let (sac_sup, line_fem2post_sac) = Self::prep(sagittal_points)?;
        let post_sac = sac_sup.index_axis(Axis(0), 1);
        let mid_femoral_heads = sagittal_points.femoral_head.0.mean_axis(Axis(0)).unwrap();
        let label = Self::NAME;
        let color = line_colors.get_or_new(label);
        let mut g = self.default_group().set("fill", color).set("stroke", color);
        for p in sagittal_points.femoral_head.0.axis_iter(Axis(0)) {
            g = g.add(painter.point(p));
        }
        if sagittal_points.femoral_head.0.len_of(Axis(0)) == 2 {
            g = g.add(painter.point(mid_femoral_heads.view()));
            g = g.add(painter.line(sagittal_points.femoral_head.0.view()));
        }
        g = g.add(painter.line(sac_sup.view()));
        g = g.add(painter.line(line_fem2post_sac.view()));
        let angle = angle_between(line_fem2post_sac.view(), sac_sup.view()).to_degrees();
        let text = format!("{:.1}°", angle);
        let text = painter.text(&text, post_sac, Some(label));
        g = g.add(text);
        Ok(g)
    }

    fn measure(&self, sagittal_points: &SagittalPoints) -> Result<f32, MeasureError> {
        let (sac_sup, line_fem2post_sac) = Self::prep(sagittal_points)?;
        let angle = angle_between(line_fem2post_sac.view(), sac_sup.view()).to_degrees();
        Ok(angle)
    }
}

fn femoral_incidence_angle(
    plate: ArrayView2<f32>,
    femoral_head: &crate::AtMost2<Array2<f32>>,
) -> Result<f32, MeasureError> {
    if femoral_head.0.is_empty() {
        return Err(MeasureError::InvalidNumberOfPoints(
            InvalidNumberOfPoints::TooFewPoints(1, 0),
        ));
    }
    let mid_femoral_heads = femoral_head.0.mean_axis(Axis(0)).unwrap();
    let sac_sup_mid = plate.mean_axis(Axis(0)).unwrap();
    let line_sac2fem = stack![Axis(0), sac_sup_mid, mid_femoral_heads];
    let sac_p2a = &plate.index_axis(Axis(0), 1) - &plate.index_axis(Axis(0), 0);
    let mut perp_sac = ndarray::Array::from_vec(vec![-sac_p2a[1], sac_p2a[0]]);
    perp_sac /= perp_sac.l2norm();
    perp_sac = 0.25 * sac_sup_mid.l2_dist(&mid_femoral_heads).unwrap() as f32 * perp_sac;
    perp_sac += &sac_sup_mid;
    let perp_line = stack![Axis(0), sac_sup_mid, perp_sac];
    let angle_deg = angle_between(line_sac2fem.view(), perp_line.view()).to_degrees();
    Ok(angle_deg)
}

impl From<SagittalMeasure> for Box<dyn SagittalComponent> {
    fn from(measure: SagittalMeasure) -> Self {
        match measure {
            SagittalMeasure::ThoracicKyphosis => Box::new(ThoracicKyphosis {}),
            SagittalMeasure::ProximalThoracicKyphosis => Box::new(ProximalThoracicKyphosis {}),
            SagittalMeasure::MidLowerThoracicKyphosis => Box::new(MidLowerThoracicKyphosis {}),
            SagittalMeasure::ThoracoLumbarSagittalAlignment => {
                Box::new(ThoracoLumbarSagittalAlignment {})
            }
            SagittalMeasure::LumbarLordosis => Box::new(LumbarLordosis {}),
            SagittalMeasure::SagittalBalance => Box::new(SagittalBalance {}),
            SagittalMeasure::LumbosacralAngle => Box::new(LumbosacralAngle {}),
            SagittalMeasure::PelvicIncidence => Box::new(PelvicIncidence {}),
            SagittalMeasure::L5IncidenceAngle => Box::new(L5IncidenceAngle {}),
            SagittalMeasure::PelvicRadiusAngle => Box::new(PelvicRadiusAngle {}),
        }
    }
}

pub struct ColorPalettes {
    pub label_colors: ColorPalette,
    pub line_colors: ColorPalette,
}

pub fn draw_sagittal(
    data: LabelMeDataWImage,
    sagittal_points: SagittalPoints,
    measures: Vec<SagittalMeasure>,
    hide: Vec<SagittalMeasure>,
    draw_param: DrawParam,
    svg_size: (usize, usize),
    palettes: ColorPalettes,
) -> element::SVG {
    let ColorPalettes {
        mut label_colors,
        mut line_colors,
    } = palettes;
    let spine = &sagittal_points.spine;
    let painter = Painter::new(draw_param.clone(), svg_size);
    let mut document = painter.doc_w_background(&data.image);
    let style = element::Style::new(draw_param.style());
    document = document.add(style);

    // common components
    let common_components: Vec<Box<dyn CommonComponent>> =
        vec![Box::new(VertebralLabels {}), Box::new(VertebralPoints {})];
    for component in common_components {
        let g = component.draw(spine, &painter, &mut label_colors, &mut line_colors);
        document = document.add(g);
    }

    for measure in measures {
        let spinal_measure: Box<dyn SagittalComponent> = measure.into();
        match spinal_measure.draw(
            &sagittal_points,
            &painter,
            &mut label_colors,
            &mut line_colors,
        ) {
            Ok(g) => {
                let visibility = if hide.contains(&measure) {
                    "hidden"
                } else {
                    "visible"
                };
                let g = g.set("visibility", visibility);
                document = document.add(g)
            }
            Err(err) => warn!("Failed to draw {}: {:?}", spinal_measure.name(), err),
        }
    }

    document
}

pub fn draw_coronal(
    data: LabelMeDataWImage,
    coronal_points: CoronalPoints,
    draw_param: DrawParam,
    svg_size: (usize, usize),
    mut label_colors: ColorPalette,
    mut line_colors: ColorPalette,
    curve_apex_set: Option<(CurveSet, ApexSet)>,
) -> element::SVG {
    let spine = &coronal_points.spine;
    let painter = Painter::new(draw_param.clone(), svg_size);
    let mut document = painter.doc_w_background(&data.image);
    let style = element::Style::new(draw_param.style());

    document = document.add(style);

    // common components
    for component in ["VertebralLabels", "VertebralPoints"] {
        let group = match component {
            "VertebralLabels" => {
                VertebralLabels {}.draw(spine, &painter, &mut label_colors, &mut line_colors)
            }
            "VertebralPoints" => {
                VertebralPoints {}.draw(spine, &painter, &mut label_colors, &mut line_colors)
            }
            _ => panic!("Unknown component"),
        };
        document = document.add(group);
    }

    let (curve_set, apex_set) = curve_apex_set.unwrap_or_else(|| {
        let (cs, apexes, _major_curve) = spine.identify_curves();
        (cs, apexes)
    });

    // reverse order to draw important components above others
    for component in CORONAL_COMPONENTS.into_iter().rev() {
        let group = match component {
            ID_CENTROID => {
                Some(Centroids {}.draw(spine, &painter, &mut label_colors, &mut line_colors))
            }
            ID_COB_ANGLES => CobbAngles(&curve_set).draw(
                &coronal_points,
                &painter,
                &mut label_colors,
                &mut line_colors,
            ),
            ID_CURVE_APEX => {
                let vert_discs = spine.tl_vert_disc_corners();
                CurveApex(&apex_set, &vert_discs).draw(
                    &coronal_points,
                    &painter,
                    &mut label_colors,
                    &mut line_colors,
                )
            }
            ID_SPINAL_LINE => {
                Some(SpinalLine {}.draw(spine, &painter, &mut label_colors, &mut line_colors))
            }
            ID_CSVL => Csvl(&apex_set).draw(
                &coronal_points,
                &painter,
                &mut label_colors,
                &mut line_colors,
            ),
            ID_T1_TILT_ANGLE => T1TiltAngle {}.draw(
                &coronal_points,
                &painter,
                &mut label_colors,
                &mut line_colors,
            ),
            ID_CORONAL_BALANCE => CoronalBalance {}.draw(
                &coronal_points,
                &painter,
                &mut label_colors,
                &mut line_colors,
            ),
            ID_CLAVICLE_ANGLE => ClavicleAngle {}.draw(
                &coronal_points,
                &painter,
                &mut label_colors,
                &mut line_colors,
            ),
            ID_SHOULDER_HEIGHT => ShoulderHeight {}.draw(
                &coronal_points,
                &painter,
                &mut label_colors,
                &mut line_colors,
            ),
            ID_PELVIC_OBLIQUITY => PelvicObliquity {}.draw(
                &coronal_points,
                &painter,
                &mut label_colors,
                &mut line_colors,
            ),
            ID_SACRAL_OBLIQUITY => SacralObliquity {}.draw(
                &coronal_points,
                &painter,
                &mut label_colors,
                &mut line_colors,
            ),
            ID_LEG_LEN_DISCREPANCY => LegLengthDiscrepancy {}.draw(
                &coronal_points,
                &painter,
                &mut label_colors,
                &mut line_colors,
            ),
            c => panic!("Unknown component: {}", c),
        };
        if let Some(group) = group {
            document = document.add(group);
        }
    }

    document
}
#[cfg(test)]
pub(crate) mod tests {
    use anyhow::Result;
    #[test]
    fn test_test() -> Result<()> {
        Ok(())
    }
}
