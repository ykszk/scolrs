use crate::{
    angle_from_lines, CoronalMeasure, CoronalPoints, HasCornerPoints, L2Norm, SagittalMeasure,
    SagittalPoints, ValidateLength, CORNER_LABELS,
};
use crate::{ApexSet, Curve, CurveSet, DrawParam, Spine, VertebralIndex, VERTEBRAL_LABELS};
use labelme_rs::LabelMeDataWImage;
use log::{debug, warn};
use named_derive::Named;
use ndarray::{s, stack, Array2, ArrayBase, ArrayView2, Axis, Ix1, Ix2};
use ndarray_stats::DeviationExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::ops::{AddAssign, SubAssign};
use svg::node::element;
use svg::Node;
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

static X_ATTRS: [&str; 4] = ["x", "cx", "x1", "x2"];
static Y_ATTRS: [&str; 4] = ["y", "cy", "y1", "y2"];

/// Scale the coordinates of the SVG node recursively
fn _scale_coordinates(scale: (f64, f64), node: &mut Box<dyn Node>) {
    // TODO: Implemented only for `arc` at the moment. Need to implement for other elements
    if node.get_name() == "path" {
        let attrs = node.get_attributes_mut().unwrap();
        let d = attrs.get_mut("d").unwrap(); // e.g. "M295.8,73.5 A65.42201,65.42201,0,0,1,296.42203,64.5"
        let re = regex::Regex::new(r"([MA])|(-?\d+\.?\d*)").unwrap();

        let matches = re.find_iter(d).map(|m| m.as_str()).collect::<Vec<_>>();

        if matches.len() != 11 || matches[0] != "M" || matches[3] != "A" {
            println!("{:?}", matches);
            panic!("Invalid path data: {}", d);
        }
        let m_scaled_numbers = matches[1..3]
            .iter()
            .map(|s| {
                if let Ok(x) = s.parse::<f64>() {
                    format!("{}", x * scale.0)
                } else {
                    s.to_string()
                }
            })
            .collect::<Vec<_>>();

        let mut a_scaled_numbers: Vec<_> = matches[4..]
            .iter()
            .map(|s| s.parse::<f64>().unwrap())
            .collect();

        a_scaled_numbers[5] *= scale.0;
        a_scaled_numbers[6] *= scale.1;

        let new_d = format!(
            "M{},{} A{}",
            m_scaled_numbers[0],
            m_scaled_numbers[1],
            a_scaled_numbers
                .iter()
                .map(|a| a.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        *d = new_d.into();
    }

    if let Some(attrs) = node.get_attributes_mut() {
        for (attr, value) in attrs.iter_mut() {
            if X_ATTRS.contains(&attr.as_str()) {
                if let Ok(x) = value.parse::<f64>() {
                    *value = format!("{}", x * scale.0).into();
                }
            } else if Y_ATTRS.contains(&attr.as_str()) {
                if let Ok(y) = value.parse::<f64>() {
                    *value = format!("{}", y * scale.1).into();
                }
            } else if attr == "points" {
                // polygon and polyline
                let points: Vec<_> = value.split_whitespace().collect();
                let mut points = Array2::from_shape_vec((points.len() / 2, 2), points)
                    .unwrap()
                    .mapv(|a| a.parse::<f64>().unwrap());
                points
                    .index_axis_mut(Axis(1), 0)
                    .mapv_inplace(|a| a * scale.0);
                points
                    .index_axis_mut(Axis(1), 1)
                    .mapv_inplace(|a| a * scale.1);
                *value = points
                    .iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
                    .into();
            }
        }
    }
    for child in node.get_children_mut().unwrap_or(&mut vec![]) {
        _scale_coordinates(scale, child);
    }
}

/// Scale the coordinates of the SVG nodes
///
/// If the scale is (1.0, 1.0), the function do nothing and returns the original nodes
pub fn scale_coordinates(scale: (f64, f64), groups: Vec<Box<dyn Node>>) -> Vec<Box<dyn Node>> {
    if scale == (1.0, 1.0) {
        return groups;
    }
    groups
        .into_iter()
        .map(|mut g| {
            _scale_coordinates(scale, &mut g);
            g
        })
        .collect()
}

/// Squared distance between vectors. i.e. `(p1 - p2)^2`
fn squared_distance<S>(p1: ArrayBase<S, Ix1>, p2: ArrayBase<S, Ix1>) -> f64
where
    S: ndarray::Data<Elem = f64>,
{
    (&p1 - &p2).mapv(|a| a * a).sum()
}

/// Signed angle from line1 to line2 in radians
/// TODO: Check the difference from [`angle_from_lines`]?
pub fn angle_between<S, T>(line1: ArrayBase<S, Ix2>, line2: ArrayBase<T, Ix2>) -> f64
where
    S: ndarray::Data<Elem = f64>,
    T: ndarray::Data<Elem = f64>,
{
    let v = &line1.index_axis(Axis(0), 1) - &line1.index_axis(Axis(0), 0);
    let w = &line2.index_axis(Axis(0), 1) - &line2.index_axis(Axis(0), 0);
    (w[1] * v[0] - w[0] * v[1]).atan2(w[0] * v[0] + w[1] * v[1])
}

/// Return the pair of vectors that has maximum distance
pub fn distanced_pair3<S>(
    p1: ArrayBase<S, Ix1>,
    p2: ArrayBase<S, Ix1>,
    p3: ArrayBase<S, Ix1>,
) -> (ArrayBase<S, Ix1>, ArrayBase<S, Ix1>)
where
    S: ndarray::Data<Elem = f64>,
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
    S: ndarray::Data<Elem = f64>,
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
    pub plate_scale: f64,
    pub perpendicular_scale: f64,
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
    rad_angle: f64,
) -> ndarray::Array1<f64>
where
    S: ndarray::Data<Elem = f64>,
    T: ndarray::Data<Elem = f64>,
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
        unit: Option<&str>,
    ) -> element::Text
    where
        S: ndarray::Data<Elem = f64>,
    {
        let mut t = element::Text::new(text)
            .set("x", coords[0])
            .set("y", coords[1]);
        if let Some(unit) = unit {
            t = t.add(element::TSpan::new(unit).set("class", "unit"));
        }
        if let Some(title) = title {
            t = t.add(self.title(title))
        }
        t
    }

    pub fn title(&self, text: &str) -> element::Title {
        element::Title::new(text)
    }

    pub fn point<S>(&self, point: ArrayBase<S, Ix1>) -> element::Circle
    where
        S: ndarray::Data<Elem = f64>,
    {
        element::Circle::new()
            .set("cx", point[0])
            .set("cy", point[1])
            .set("r", self.param.radius)
    }

    pub fn line<S>(&self, start_end: ArrayBase<S, Ix2>) -> element::Line
    where
        S: ndarray::Data<Elem = f64>,
    {
        element::Line::new()
            .set("x1", start_end[[0, 0]])
            .set("y1", start_end[[0, 1]])
            .set("x2", start_end[[1, 0]])
            .set("y2", start_end[[1, 1]])
    }

    pub fn horizontal_line(&self, y: f64) -> element::Line {
        element::Line::new()
            .set("x1", 0)
            .set("y1", y)
            .set("x2", self.size.0)
            .set("y2", y)
    }

    #[allow(dead_code)]
    pub fn vertical_line(&self, x: f64) -> element::Line {
        element::Line::new()
            .set("x1", x)
            .set("y1", 0)
            .set("x2", x)
            .set("y2", self.size.1)
    }

    pub fn polyline<S>(&self, points: ArrayBase<S, Ix2>) -> element::Polyline
    where
        S: ndarray::Data<Elem = f64>,
    {
        let s = points.join(" ");
        element::Polyline::new().set("points", s)
    }

    pub fn polygon<S>(&self, points: ArrayBase<S, Ix2>) -> element::Polygon
    where
        S: ndarray::Data<Elem = f64>,
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
        arc_radius: f64,
        title: Option<&str>,
    ) -> (element::Group, f64)
    where
        S: ndarray::Data<Elem = f64>,
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
            None,
        );
        (group.add(text), angle_deg)
    }

    pub fn plate_end<S, T>(plate: ArrayBase<S, Ix2>, point: ArrayBase<T, Ix1>) -> usize
    where
        S: ndarray::Data<Elem = f64>,
        T: ndarray::Data<Elem = f64>,
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
        sup_plate: Array2<f64>,
        inf_plate: Array2<f64>,
        aux_param: &CobbAux,
        base_length: f64,
        title: Option<&str>,
    ) -> element::Group {
        let sup_line = points2line(sup_plate.view());
        let inf_line = points2line(inf_plate.view());

        let linter = sup_line.intersection(&inf_line);
        if let Some(intersection) = linter {
            let is_inside = intersection.x > 0.0
                && intersection.x < self.size.0 as f64
                && intersection.y > 0.0
                && intersection.y < self.size.1 as f64;
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
                    None,
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
                let text = self.text(format!("{:.1}°", angle).as_str(), aux_cross, title, None);
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
                // disable arrow for now because scaling is not handled properly
                // let unit_d = &d / d.l2norm();
                // let len = self.param.line_width * 8.0;
                // let p1 = plate.mean_axis(Axis(0)).unwrap();
                // let p2 = rotate_around(&p1 - &unit_d * len, p1.view(), 30.0_f64.to_radians());
                // let p3 = rotate_around(&p1 - &unit_d * len, p1.view(), -30.0_f64.to_radians());
                // let arrow = ndarray::stack![Axis(0), p1, p2, p3];
                // let arrow = self.polygon(arrow);

                // group = group.add(arrow);
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
        base_length: f64,
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

    pub fn doc_w_background(
        &self,
        image: &labelme_rs::image::DynamicImage,
    ) -> Result<svg::Document, labelme_rs::LabelMeDataError> {
        let (w, h) = self.size;
        let mut document = svg::Document::new()
            .set("width", w)
            .set("height", h)
            .set("viewBox", (0, 0, w, h))
            .set("xmlns:xlink", "http://www.w3.org/1999/xlink");
        let b64 = format!(
            "data:image/jpeg;base64,{}",
            labelme_rs::img2base64(image, labelme_rs::image::ImageFormat::Jpeg)?
        );
        let bg = element::Image::new()
            .set("x", 0i64)
            .set("y", 0i64)
            .set("width", w)
            .set("height", h)
            .set("xlink:href", b64);
        document = document.add(bg);
        Ok(document)
    }
}

pub fn points2line<S>(plate: ArrayBase<S, Ix2>) -> lyon_geom::Line<f64>
where
    S: ndarray::Data<Elem = f64>,
{
    let point = lyon_geom::Point::new(plate[[0, 0]], plate[[0, 1]]);
    let p2 = lyon_geom::Point::new(plate[[1, 0]], plate[[1, 1]]);
    let vector = p2 - point;
    lyon_geom::Line { point, vector }
}

type ColorMap = HashMap<String, String>;

#[derive(Debug, Clone)]
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

#[derive(thiserror::Error, Debug, Serialize, Deserialize)]
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

#[derive(thiserror::Error, Debug, Serialize, Deserialize)]
pub enum MeasureError {
    // Invalid number of points
    #[error("Invalid number of points")]
    InvalidNumberOfPoints(#[from] InvalidNumberOfPoints),

    // Zero length line
    #[error("Zero length line")]
    ZeroLengthLine,

    // Unable to measure
    #[error("Unable to measure: {0}")]
    UnableToMeasure(String),

    // No curve is found
    #[error("No curve is found")]
    NoCurveFound,
}

#[derive(thiserror::Error, Debug)]
pub enum DrawError {
    #[error("Measure error: {0}")]
    MeasureError(#[from] MeasureError),

    #[error("LabelMe data error: {0}")]
    LabelMeDataError(#[from] labelme_rs::LabelMeDataError),
}

/// Trait for drawing and measuring components
/// Use Named macro to implement this trait
pub trait Named {
    /// Unique identifier for the component
    fn id(&self) -> &'static str;
    /// Labels for the component
    fn label(&self) -> &'static str;
    /// Short description for the component
    fn description(&self) -> Option<&'static str>;
    fn draw_type(&self) -> &[&'static str];
    fn ided_group(&self) -> element::Group {
        let g = element::Group::new()
            .set("id", self.id())
            .set("data-label", self.label());
        if let Some(desc) = self.description() {
            g.set("data-description", desc)
        } else {
            g
        }
    }
    fn default_group_w_classes(&self, classes: &[&str]) -> element::Group {
        let mut classes = Vec::from(classes);
        classes.extend_from_slice(self.draw_type());
        self.ided_group().set("class", classes)
    }
}

pub trait DrawComponent: Named {
    fn draw(
        &self,
        painter: &Painter,
        label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError>;
}
pub trait MeasureComponent: Named {
    fn measure(&self) -> Result<f64, MeasureError>;
}

const COMMON_COMPONENT_CLASS: &str = "CommonComponent";
pub trait CommonComponent: DrawComponent {
    fn default_group(&self) -> element::Group {
        self.default_group_w_classes(&["Component", COMMON_COMPONENT_CLASS])
    }
}

const CORONAL_COMPONENT_CLASS: &str = "CoronalComponent";
pub trait CoronalComponent: DrawComponent {
    fn default_group(&self) -> element::Group {
        self.default_group_w_classes(&["Component", CORONAL_COMPONENT_CLASS])
    }
}

const SAGITTAL_COMPONENT_CLASS: &str = "SagittalComponent";
pub trait SagittalComponent: DrawComponent + MeasureComponent {
    fn default_group(&self) -> element::Group {
        self.default_group_w_classes(&["Component", SAGITTAL_COMPONENT_CLASS])
    }
}

/// Label text for each vertebra
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_TEXT])]
pub struct VertebralLabels<'a>(&'a Spine);
impl<'a> CommonComponent for VertebralLabels<'a> {}
impl<'a> DrawComponent for VertebralLabels<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let mut g_vert_labels = self.default_group();
        let centroids = self.0.tl_centroids();
        for (coords, label) in
            std::iter::zip(centroids.axis_iter(Axis(0)), VERTEBRAL_LABELS.into_iter())
        {
            let t = painter.text(label, coords, None, None);
            g_vert_labels = g_vert_labels.add(t);
        }
        Ok(g_vert_labels)
    }
}

/// Four corner points of each vertebra
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POINT])]
pub struct VertebralPoints<'a>(&'a Spine);
impl HasCornerPoints for VertebralPoints<'_> {
    fn top_left(&self) -> ArrayView2<f64> {
        self.0.c7tls.0.slice(s![.., 0, ..])
    }
    fn top_right(&self) -> ArrayView2<f64> {
        self.0.c7tls.0.slice(s![.., 1, ..])
    }
    fn bottom_left(&self) -> ArrayView2<f64> {
        self.0.c7tls.0.slice(s![..-1, 2, ..])
    }
    fn bottom_right(&self) -> ArrayView2<f64> {
        self.0.c7tls.0.slice(s![..-1, 3, ..])
    }
}
impl<'a> CommonComponent for VertebralPoints<'a> {}
impl<'a> DrawComponent for VertebralPoints<'a> {
    fn draw(
        &self,
        painter: &Painter,
        label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let g = self.default_group();
        self.draw_corners(g, painter, label_colors)
    }
}

pub trait DrawCorners {
    fn draw_corners(
        &self,
        group: element::Group,
        painter: &Painter,
        label_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError>;
}

///  Helper trait to provide default implementation for drawing corners
impl<T> DrawCorners for T
where
    T: HasCornerPoints + Named,
{
    fn draw_corners(
        &self,
        group: element::Group,
        painter: &Painter,
        label_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let mut g_corners = group;
        for (points, label) in [
            self.top_left(),
            self.top_right(),
            self.bottom_left(),
            self.bottom_right(),
        ]
        .into_iter()
        .zip(CORNER_LABELS)
        {
            let color = label_colors.get_or_new(label);
            let mut sub_group: element::Group = element::Group::new()
                .set("stroke", color)
                .set("fill", color);
            for point in points.axis_iter(Axis(0)) {
                let p = painter.point(point);
                sub_group = sub_group.add(p);
            }
            g_corners = g_corners.add(sub_group);
        }
        Ok(g_corners)
    }
}

// annotation draw types
pub const CLASS_ANNOTATION: &str = "Annotation";
pub const CLASS_POINT: &str = "Point";
pub const CLASS_POLYGON: &str = "Polygon";
pub const CLASS_LINE: &str = "Line";
pub const CLASS_TEXT: &str = "Text";
// measure draw types
pub const CLASS_MEASURE: &str = "Measure";
pub const CLASS_ANGLE: &str = "Angle";
pub const CLASS_DISTANCE: &str = "Distance";

/// Centroids of each vertebra
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POINT])]
struct Centroids<'a>(&'a Spine);
impl<'a> CommonComponent for Centroids<'a> {}
impl<'a> DrawComponent for Centroids<'a> {
    fn draw(
        &self,
        painter: &Painter,
        label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let label = self.id();
        let color = label_colors.get_or_new(label);
        let mut g_centroids = self.default_group().set("stroke", color).set("fill", color);
        let centroids = self.0.tl_centroids();
        for point in centroids.axis_iter(Axis(0)) {
            let p = painter.point(point);
            g_centroids = g_centroids.add(p);
        }
        Ok(g_centroids)
    }
}

fn mean_plate_length(scol: &Spine) -> f64 {
    let corners = scol.tl_corners().0;
    let sup_inf_shape = (corners.len_of(Axis(0)) * 2, 2, 2); // [n * sup_inf, lr, xy]

    let plate_lr = corners.to_shape(sup_inf_shape).unwrap();
    let diff = &plate_lr.index_axis(Axis(1), 0) - &plate_lr.index_axis(Axis(1), 1);
    diff.map_axis(Axis(1), |a| a.l2norm()).mean().unwrap()
}

macro_rules! impl_cobb_angle {
    ($name:ident) => {
        impl<'a> DrawComponent for $name<'a> {
            fn draw(
                &self,
                painter: &Painter,
                _label_colors: &mut ColorPalette,
                line_colors: &mut ColorPalette,
            ) -> Result<element::Group, MeasureError> {
                if self.1.is_none() {
                    return Err(MeasureError::NoCurveFound);
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
impl<'a> CoronalComponent for CobbPT<'a> {}
impl_cobb_angle!(CobbPT);

/// Cobb angle for MT curve
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
struct CobbMT<'a>(&'a CoronalPoints, Option<(Curve, f64)>);
impl<'a> CoronalComponent for CobbMT<'a> {}
impl_cobb_angle!(CobbMT);

/// Cobb angle for TLL curve
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
struct CobbTLL<'a>(&'a CoronalPoints, Option<(Curve, f64)>);
impl<'a> CoronalComponent for CobbTLL<'a> {}
impl_cobb_angle!(CobbTLL);

/// Curve apices for each curve
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POLYGON])]
struct CurveApex<'a>(&'a CoronalPoints, &'a ApexSet);
impl<'a> CoronalComponent for CurveApex<'a> {}
impl<'a> DrawComponent for CurveApex<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
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
impl<'a> CommonComponent for SpinalLine<'a> {}
impl<'a> DrawComponent for SpinalLine<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
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
impl<'a> CoronalComponent for Csvl<'a> {}
impl<'a> DrawComponent for Csvl<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
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
impl<'a> CoronalComponent for T1TiltAngle<'a> {}
impl DrawComponent for T1TiltAngle<'_> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let coronal_points = self.0;
        let spine = &coronal_points.spine;
        let label = self.id();
        let mut g = self
            .default_group()
            .set("stroke", line_colors.get_or_new(label))
            .set("fill", "none");
        let tl_sup_lines = spine.tl_sup_lines();
        let t1sup = tl_sup_lines.index_axis(Axis(0), 0);
        let mid = t1sup.mean_axis(Axis(0)).unwrap();
        let (mult_left, mult_right, mult_arc) = (1.0, 4.0, 3.0);
        let l2r = &t1sup.index_axis(Axis(0), 1) - mult_left * &t1sup.index_axis(Axis(0), 0);
        if l2r.l2norm() == 0.0 {
            return Err(MeasureError::ZeroLengthLine);
        }
        let sup_line = stack![Axis(0), &mid - &l2r, &mid + mult_right * &l2r];
        g = g.add(painter.line(sup_line.view()));
        if l2r[0] == 0.0 {
            // T1 is vertical, which is highly unlikely
            debug!("T1 is vertical");
            g = g.add(painter.line(t1sup.view()));
            g = g.add(painter.text("90°", mid, Some(label), None));
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
        Ok(g)
    }
}
impl MeasureComponent for T1TiltAngle<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let tl_sup_lines = self.0.spine.tl_sup_lines();
        let t1sup = tl_sup_lines.index_axis(Axis(0), 0);
        tilt_angle(t1sup)
    }
}

/// Coronal balance (p. 54)
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_DISTANCE])]
pub struct CoronalBalance<'a>(&'a CoronalPoints);
impl<'a> CoronalComponent for CoronalBalance<'a> {}
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
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let spine = &self.0.spine;
        let label = self.id();
        let points = self.prep(spine);
        let color = line_colors.get_or_new(label);
        let g = self.default_group().set("fill", color).set("stroke", color);
        let g = draw_difference_in_x(
            g,
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
impl<'a> CoronalComponent for ClavicleAngle<'a> {}
impl DrawComponent for ClavicleAngle<'_> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let coronal_points = self.0;
        coronal_points.clavicle.0.validate_length(2)?;
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
        tilt_angle(self.0.clavicle.0.view())
    }
}

fn difference_in_index(points: ArrayView2<f64>, index: usize) -> Result<f64, MeasureError> {
    points.validate_length(2)?;
    Ok(points.index_axis(Axis(0), 0)[index] - points.index_axis(Axis(0), 1)[index])
}

fn difference_in_x(points: ArrayView2<f64>) -> Result<f64, MeasureError> {
    difference_in_index(points, 0)
}

fn difference_in_y(points: ArrayView2<f64>) -> Result<f64, MeasureError> {
    difference_in_index(points, 1)
}

fn draw_difference_in_x(
    group: element::Group,
    label: &str,
    points: ArrayView2<f64>,
    painter: &Painter,
    unit: &str,
) -> Result<element::Group, MeasureError> {
    let dx = difference_in_x(points)?;
    let mut g = group;
    for c in points.axis_iter(Axis(0)) {
        g = g.add(painter.point(c));
    }
    let p1 = points.index_axis(Axis(0), 0);
    let p2 = points.index_axis(Axis(0), 1);
    let len = 0.4 * dx;

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
        &format!("{:.1}", dx),
        h_line.index_axis(Axis(0), 1),
        Some(label),
        Some(unit),
    );

    g = g.add(text);
    Ok(g)
}

fn draw_difference_in_y(
    label: &str,
    group: element::Group,
    points: ArrayView2<f64>,
    painter: &Painter,
    unit: &str,
) -> Result<element::Group, MeasureError> {
    let dy = difference_in_y(points)?;
    let mut g = group;
    for c in points.axis_iter(Axis(0)) {
        g = g.add(painter.point(c));
    }
    g = g.add(painter.horizontal_line(points[[0, 1]]));
    g = g.add(painter.horizontal_line(points[[1, 1]]));
    let mut vline = points.to_owned();
    vline[[1, 0]] = vline[[0, 0]];
    g = g.add(painter.line(vline.view()));
    let text_pos = vline.mean_axis(Axis(0)).unwrap();
    let text = format!("{:.1}", dy);
    let text = painter.text(&text, text_pos, Some(label), Some(unit));

    g = g.add(text);
    Ok(g)
}

/// Radiographic shoulder height (p.57)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_DISTANCE])]
pub struct ShoulderHeight<'a>(&'a CoronalPoints);
impl<'a> CoronalComponent for ShoulderHeight<'a> {}
impl DrawComponent for ShoulderHeight<'_> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let color = line_colors.get_or_new(self.id());
        let g = self.default_group().set("fill", color).set("stroke", color);

        draw_difference_in_y(
            self.id(),
            g,
            self.0.shoulder.0.view(),
            painter,
            self.0.image_metadata.unit.as_str(),
        )
    }
}
impl MeasureComponent for ShoulderHeight<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let coronal_points = self.0;
        coronal_points.shoulder.0.validate_length(2)?;
        let points = coronal_points.shoulder.0.view();
        let dy = points.index_axis(Axis(0), 0)[1] - points.index_axis(Axis(0), 1)[1];
        Ok(dy)
    }
}

fn tilt_angle(points: ArrayView2<f64>) -> Result<f64, MeasureError> {
    points.validate_length(2)?;
    let mut hor_line = points.to_owned();
    hor_line[[1, 1]] = points[[0, 1]];
    let angle = angle_between(hor_line.view(), points);
    Ok(angle.to_degrees())
}

fn draw_tilt_angle(
    group: element::Group,
    painter: &Painter,
    points: &ndarray::Array2<f64>,
    title: Option<&str>,
) -> element::Group {
    let mut g = group;
    for p in points.axis_iter(Axis(0)) {
        g = g.add(painter.point(p));
    }
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
        .0;
    g
}

/// Pelvic Obliquity (p.69)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct PelvicObliquity<'a>(&'a CoronalPoints);
impl<'a> CoronalComponent for PelvicObliquity<'a> {}
impl DrawComponent for PelvicObliquity<'_> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let coronal_points = self.0;
        coronal_points.pelvis.0.validate_length(2)?;
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
        tilt_angle(self.0.pelvis.0.view())
    }
}

/// Sacral Obliquity (p.70)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct SacralObliquity<'a>(&'a CoronalPoints);
impl<'a> CoronalComponent for SacralObliquity<'a> {}
impl DrawComponent for SacralObliquity<'_> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let coronal_points = self.0;
        coronal_points.femoral_head.0.validate_length(2)?;
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
        tilt_angle(self.0.femoral_head.0.view())
    }
}

/// Leg Length Discrepancy (p.69)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_DISTANCE])]
pub struct LegLengthDiscrepancy<'a>(&'a CoronalPoints);
impl<'a> CoronalComponent for LegLengthDiscrepancy<'a> {}
impl DrawComponent for LegLengthDiscrepancy<'_> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let color = line_colors.get_or_new(self.id());
        let g = self.default_group().set("fill", color).set("stroke", color);
        draw_difference_in_y(
            self.id(),
            g,
            self.0.femoral_head.0.view(),
            painter,
            self.0.image_metadata.unit.as_str(),
        )
    }
}
impl MeasureComponent for LegLengthDiscrepancy<'_> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let coronal_points = self.0;
        coronal_points.femoral_head.0.validate_length(2)?;
        let points = coronal_points.femoral_head.0.view();
        let dy = points.index_axis(Axis(0), 0)[1] - points.index_axis(Axis(0), 1)[1];
        Ok(dy)
    }
}

pub fn draw_incidence_angle(
    group: element::Group,
    label: &str,
    femoral_heads: ArrayView2<f64>,
    plate: ArrayView2<f64>,
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
    perp_sac = 0.25 * sac_sup_mid.l2_dist(&mid_femoral_heads).unwrap() * perp_sac;
    perp_sac += &sac_sup_mid;
    let perp_line = stack![Axis(0), sac_sup_mid, perp_sac];
    g = g.add(painter.line(perp_line.view()));
    let angle = angle_between(line_sac2fem.view(), perp_line.view()).to_degrees();
    let text = format!("{:.1}°", angle);
    let text = painter.text(&text, sac_sup_mid, Some(label), None);

    g = g.add(text);
    g
}

macro_rules! impl_kyophosis {
    ($name:ident, $opposite:expr) => {
        impl<'a> DrawComponent for $name<'a> {
            fn draw(
                &self,
                painter: &Painter,
                _label_colors: &mut ColorPalette,
                line_colors: &mut ColorPalette,
            ) -> Result<element::Group, MeasureError> {
                let label = self.id();
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
    };
}

/// Proximal thoracic kyphosis (p.65)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct ProximalThoracicKyphosis<'a>(&'a SagittalPoints);
impl<'a> ProximalThoracicKyphosis<'a> {
    const SUP: usize = VertebralIndex::T2 as usize;
    const INF: usize = VertebralIndex::T5 as usize;
}
impl<'a> SagittalComponent for ProximalThoracicKyphosis<'a> {}
impl_kyophosis!(ProximalThoracicKyphosis, false);

/// Thoracic kyphosis (p.65)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct ThoracicKyphosis<'a>(&'a SagittalPoints);
impl<'a> ThoracicKyphosis<'a> {
    const SUP: usize = VertebralIndex::T2 as usize;
    const INF: usize = VertebralIndex::T12 as usize;
}
impl<'a> SagittalComponent for ThoracicKyphosis<'a> {}
impl_kyophosis!(ThoracicKyphosis, true);

/// Mid/Lower thoracic kyphosis (p.65)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
#[label("Mid/LowerThoracicKyphosis")]
pub struct MidLowerThoracicKyphosis<'a>(&'a SagittalPoints);
impl<'a> SagittalComponent for MidLowerThoracicKyphosis<'a> {}
impl MidLowerThoracicKyphosis<'_> {
    const SUP: usize = VertebralIndex::T5 as usize;
    const INF: usize = VertebralIndex::T12 as usize;
}
impl_kyophosis!(MidLowerThoracicKyphosis, false);

/// Thoracolumbar(T10/L2) sagittal alignment (p.66)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct ThoracolumbarSagittalAlignment<'a>(&'a SagittalPoints);
impl<'a> ThoracolumbarSagittalAlignment<'a> {
    const SUP: usize = VertebralIndex::T10 as usize;
    const INF: usize = VertebralIndex::L2 as usize;
}
impl<'a> SagittalComponent for ThoracolumbarSagittalAlignment<'a> {}
impl_kyophosis!(ThoracolumbarSagittalAlignment, false);

/// Lumbar sagittal alignment (T12/S1) (p.66)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct LumbarLordosis<'a>(&'a SagittalPoints);
impl<'a> LumbarLordosis<'a> {
    fn prep(spine: &Spine) -> (usize, usize) {
        let sup = VertebralIndex::T12 as usize;
        let inf = spine.v_c7tl.0.len_of(Axis(0)) - 1;
        (sup, inf)
    }
}
impl<'a> SagittalComponent for LumbarLordosis<'a> {}
impl<'a> DrawComponent for LumbarLordosis<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let spine = &self.0.spine;
        let aux_param = CobbAux::default();
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
impl<'a> MeasureComponent for LumbarLordosis<'a> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let spine = &self.0.spine;
        let (sup, inf) = Self::prep(spine);
        let angle = spine.angle(&Curve { sup, inf }).unwrap();
        Ok(angle)
    }
}

/// Sagittal balance (p.67)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_DISTANCE])]
pub struct SagittalBalance<'a>(&'a SagittalPoints);
impl<'a> SagittalBalance<'a> {
    fn prep(sagittal_points: &SagittalPoints) -> Array2<f64> {
        let c_c7 = sagittal_points.spine.c_c7tl.index_axis(Axis(0), 0);
        let sac_sup = sagittal_points.spine.sacral_sup_plate();
        let pos_sac = sac_sup.index_axis(Axis(0), 1);
        stack![Axis(0), c_c7, pos_sac]
    }
}
impl<'a> SagittalComponent for SagittalBalance<'a> {}
impl<'a> DrawComponent for SagittalBalance<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let points = Self::prep(self.0);
        let label = self.id();
        let color = line_colors.get_or_new(label);
        let g = self.default_group().set("stroke", color).set("fill", color);
        let g = draw_difference_in_x(
            g,
            label,
            points.view(),
            painter,
            self.0.image_metadata.unit.as_str(),
        );
        g
    }
}
impl<'a> MeasureComponent for SagittalBalance<'a> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let points = Self::prep(self.0);
        let p1 = points.index_axis(Axis(0), 0);
        let p2 = points.index_axis(Axis(0), 1);
        let dx = p1[0] - p2[0];
        Ok(dx)
    }
}

/// Lumbosacral angle (p.105)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct LumbosacralAngle<'a>(&'a SagittalPoints);
impl<'a> LumbosacralAngle<'a> {
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
impl<'a> SagittalComponent for LumbosacralAngle<'a> {}
impl<'a> DrawComponent for LumbosacralAngle<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let spine = &self.0.spine;
        let (sup, inf) = Self::prep(spine);
        let label = self.id();
        let mean_plate_length = mean_plate_length(spine);
        let group = self
            .default_group()
            .set("stroke", line_colors.get_or_new(label));
        let aux_param = CobbAux::default();
        let g =
            painter.cobb_from_plates(group, sup, inf, &aux_param, mean_plate_length, Some(label));
        Ok(g)
    }
}
impl<'a> MeasureComponent for LumbosacralAngle<'a> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let (sup, inf) = Self::prep(&self.0.spine);
        let angle = angle_between(sup.view(), inf.view()).to_degrees();
        Ok(angle)
    }
}

/// Pelvic Incidence (p.97)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct PelvicIncidence<'a>(&'a SagittalPoints);
impl<'a> SagittalComponent for PelvicIncidence<'a> {}
impl<'a> DrawComponent for PelvicIncidence<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        self.0.femoral_head.0.validate_length_more_than(1)?;
        let label = self.id();
        let sac_sup = self.0.spine.sacral_sup_plate();
        let color = line_colors.get_or_new(label);
        let g = self.default_group().set("fill", color).set("stroke", color);
        let g = draw_incidence_angle(g, label, self.0.femoral_head.0.view(), sac_sup, painter);
        Ok(g)
    }
}
impl<'a> MeasureComponent for PelvicIncidence<'a> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let plate = self.0.spine.sacral_sup_plate();
        femoral_incidence_angle(plate, &self.0.femoral_head)
    }
}

// Pelvic Tilt (p.98)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct PelvicTilt<'a>(&'a SagittalPoints);
impl<'a> PelvicTilt<'a> {
    fn prep(sagittal_points: &SagittalPoints) -> Result<(Array2<f64>, Array2<f64>), MeasureError> {
        sagittal_points
            .femoral_head
            .0
            .validate_length_more_than(1)?;
        let sac_sup = sagittal_points.spine.sacral_sup_plate();
        let mid_sac = sac_sup.mean_axis(Axis(0)).unwrap();
        let femoral_head = sagittal_points.femoral_head.0.mean_axis(Axis(0)).unwrap();
        let sac2fem = stack![Axis(0), mid_sac, femoral_head];
        let mut v_line_from_fem = sac2fem.clone();
        v_line_from_fem[[0, 0]] = v_line_from_fem[[1, 0]];
        Ok((sac2fem, v_line_from_fem))
    }
}
impl<'a> SagittalComponent for PelvicTilt<'a> {}
impl<'a> DrawComponent for PelvicTilt<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let sagittal_points = self.0;
        let (sac2fem, v_line) = Self::prep(sagittal_points)?;
        let label = self.id();
        let color = line_colors.get_or_new(label);
        let mut g = self.default_group().set("fill", color).set("stroke", color);
        g = draw_femoral_center(g, sagittal_points.femoral_head.0.view(), painter);
        g = g.add(painter.point(sac2fem.index_axis(Axis(0), 0)));
        // TODO: use painter.angle_between
        g = g.add(painter.line(sac2fem.view()));
        let sac_sup = sagittal_points.spine.sacral_sup_plate();
        g = g.add(painter.line(sac_sup.view()));
        g = g.add(painter.line(v_line.view()));
        let angle = angle_between(v_line.view(), sac2fem.view()).to_degrees();
        let text = format!("{:.1}°", angle);
        let text = painter.text(&text, sac2fem.index_axis(Axis(0), 1), Some(label), None);
        g = g.add(text);
        Ok(g)
    }
}
impl<'a> MeasureComponent for PelvicTilt<'a> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let sagittal_points = self.0;
        let (sac2fem, v_line) = Self::prep(sagittal_points)?;
        let angle = angle_between(v_line.view(), sac2fem.view()).to_degrees();
        Ok(angle)
    }
}

// Sacral Slope (p.99)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct SacralSlope<'a>(&'a SagittalPoints);
impl<'a> SagittalComponent for SacralSlope<'a> {}
impl<'a> SacralSlope<'a> {
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

impl<'a> DrawComponent for SacralSlope<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
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
impl<'a> MeasureComponent for SacralSlope<'a> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let sagittal_points = self.0;
        let (sac_sup, h_line_sac) = Self::prep(sagittal_points)?;
        let angle = angle_between(sac_sup.view(), h_line_sac.view()).to_degrees();
        Ok(angle)
    }
}

/// L5 Incidence Angle (p.102)
#[derive(Named)]
#[draw_type([CLASS_MEASURE, CLASS_ANGLE])]
pub struct L5IncidenceAngle<'a>(&'a SagittalPoints);
impl<'a> SagittalComponent for L5IncidenceAngle<'a> {}
impl<'a> DrawComponent for L5IncidenceAngle<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        self.0.femoral_head.0.validate_length_more_than(1)?;
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
impl<'a> MeasureComponent for L5IncidenceAngle<'a> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let sagittal_points = self.0;
        let plate = sagittal_points
            .spine
            .sup_plate(sagittal_points.spine.v_c7tl.0.len_of(Axis(0)) - 2);
        femoral_incidence_angle(plate, &sagittal_points.femoral_head)
    }
}

fn draw_femoral_center(
    group: element::Group,
    femoral_head: ArrayView2<f64>,
    painter: &Painter,
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
impl<'a> PelvicRadiusAngle<'a> {
    fn prep(sagittal_points: &SagittalPoints) -> Result<(Array2<f64>, Array2<f64>), MeasureError> {
        sagittal_points
            .femoral_head
            .0
            .validate_length_more_than(1)?;
        let mid_femoral_heads = sagittal_points.femoral_head.0.mean_axis(Axis(0)).unwrap();
        let sac_sup = sagittal_points.spine.sacral_sup_plate();
        let post_sac = sac_sup.index_axis(Axis(0), 1);
        let line_fem2post_sac = stack![Axis(0), mid_femoral_heads, post_sac];
        Ok((sac_sup.to_owned(), line_fem2post_sac))
    }
}
impl<'a> SagittalComponent for PelvicRadiusAngle<'a> {}
impl<'a> DrawComponent for PelvicRadiusAngle<'a> {
    fn draw(
        &self,
        painter: &Painter,
        _label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, MeasureError> {
        let sagittal_points = self.0;
        let (sac_sup, line_fem2post_sac) = Self::prep(sagittal_points)?;
        let post_sac = sac_sup.index_axis(Axis(0), 1);
        let label = self.id();
        let color = line_colors.get_or_new(label);
        let mut g = self.default_group().set("fill", color).set("stroke", color);
        g = draw_femoral_center(g, sagittal_points.femoral_head.0.view(), painter);
        g = g.add(painter.line(sac_sup.view()));
        g = g.add(painter.line(line_fem2post_sac.view()));
        let angle = angle_between(line_fem2post_sac.view(), sac_sup.view()).to_degrees();
        let text = format!("{:.1}°", angle);
        let text = painter.text(&text, post_sac, Some(label), None);
        g = g.add(text);
        Ok(g)
    }
}

impl<'a> MeasureComponent for PelvicRadiusAngle<'a> {
    fn measure(&self) -> Result<f64, MeasureError> {
        let (sac_sup, line_fem2post_sac) = Self::prep(self.0)?;
        let angle = angle_between(line_fem2post_sac.view(), sac_sup.view()).to_degrees();
        Ok(angle)
    }
}

pub fn femoral_incidence_angle(
    plate: ArrayView2<f64>,
    femoral_head: &crate::AtMost2<Array2<f64>>,
) -> Result<f64, MeasureError> {
    femoral_head.0.validate_length_more_than(1)?;
    let mid_femoral_heads = femoral_head.0.mean_axis(Axis(0)).unwrap();
    let sac_sup_mid = plate.mean_axis(Axis(0)).unwrap();
    let line_sac2fem = stack![Axis(0), sac_sup_mid, mid_femoral_heads];
    let sac_p2a = &plate.index_axis(Axis(0), 1) - &plate.index_axis(Axis(0), 0);
    let mut perp_sac = ndarray::Array::from_vec(vec![-sac_p2a[1], sac_p2a[0]]);
    perp_sac /= perp_sac.l2norm();
    perp_sac = 0.25 * sac_sup_mid.l2_dist(&mid_femoral_heads).unwrap() * perp_sac;
    perp_sac += &sac_sup_mid;
    let perp_line = stack![Axis(0), sac_sup_mid, perp_sac];
    let angle_deg = angle_between(line_sac2fem.view(), perp_line.view()).to_degrees();
    Ok(angle_deg)
}

impl<'a> From<(SagittalMeasure, &'a SagittalPoints)> for Box<dyn DrawComponent + 'a> {
    fn from((measure, sagittal_points): (SagittalMeasure, &'a SagittalPoints)) -> Self {
        match measure {
            SagittalMeasure::ThoracicKyphosis => Box::new(ThoracicKyphosis(sagittal_points)),
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
            SagittalMeasure::PelvicIncidence => Box::new(PelvicIncidence(sagittal_points)),
            SagittalMeasure::PelvicTilt => Box::new(PelvicTilt(sagittal_points)),
            SagittalMeasure::SacralSlope => Box::new(SacralSlope(sagittal_points)),
            SagittalMeasure::L5IncidenceAngle => Box::new(L5IncidenceAngle(sagittal_points)),
            SagittalMeasure::PelvicRadiusAngle => Box::new(PelvicRadiusAngle(sagittal_points)),

            SagittalMeasure::VertebralLabels => Box::new(VertebralLabels(&sagittal_points.spine)),
            SagittalMeasure::VertebralPoints => Box::new(VertebralPoints(&sagittal_points.spine)),
        }
    }
}

impl<'a> From<(SagittalMeasure, &'a SagittalPoints)> for Box<dyn MeasureComponent + 'a> {
    fn from((measure, sagittal_points): (SagittalMeasure, &'a SagittalPoints)) -> Self {
        match measure {
            SagittalMeasure::ThoracicKyphosis => Box::new(ThoracicKyphosis(sagittal_points)),
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
            SagittalMeasure::PelvicIncidence => Box::new(PelvicIncidence(sagittal_points)),
            SagittalMeasure::PelvicTilt => Box::new(PelvicTilt(sagittal_points)),
            SagittalMeasure::SacralSlope => Box::new(SacralSlope(sagittal_points)),
            SagittalMeasure::L5IncidenceAngle => Box::new(L5IncidenceAngle(sagittal_points)),
            SagittalMeasure::PelvicRadiusAngle => Box::new(PelvicRadiusAngle(sagittal_points)),

            _ => panic!("Invalid conversion: {:?}", measure),
        }
    }
}

impl<'a> From<(CoronalMeasure, &'a CoronalPoints, &'a CurveSet, &'a ApexSet)>
    for Box<dyn DrawComponent + 'a>
{
    fn from(value: (CoronalMeasure, &'a CoronalPoints, &'a CurveSet, &'a ApexSet)) -> Self {
        let (measure, coronal_points, curve_set, apex_set) = value;
        match measure {
            CoronalMeasure::CobbPT => Box::new(CobbPT(coronal_points, curve_set.pt.clone())),
            CoronalMeasure::CobbMT => Box::new(CobbMT(coronal_points, curve_set.mt.clone())),
            CoronalMeasure::CobbTLL => Box::new(CobbTLL(coronal_points, curve_set.tll.clone())),

            CoronalMeasure::CurveApex => Box::new(CurveApex(coronal_points, apex_set)),
            CoronalMeasure::CSVL => Box::new(Csvl(coronal_points, apex_set)),
            CoronalMeasure::T1TiltAngle => Box::new(T1TiltAngle(coronal_points)),
            CoronalMeasure::CoronalBalance => Box::new(CoronalBalance(coronal_points)),
            CoronalMeasure::ClavicleAngle => Box::new(ClavicleAngle(coronal_points)),
            CoronalMeasure::ShoulderHeight => Box::new(ShoulderHeight(coronal_points)),
            CoronalMeasure::PelvicObliquity => Box::new(PelvicObliquity(coronal_points)),
            CoronalMeasure::SacralObliquity => Box::new(SacralObliquity(coronal_points)),
            CoronalMeasure::LegLengthDiscrepancy => Box::new(LegLengthDiscrepancy(coronal_points)),

            CoronalMeasure::VertebralLabels => Box::new(VertebralLabels(&coronal_points.spine)),
            CoronalMeasure::VertebralPoints => Box::new(VertebralPoints(&coronal_points.spine)),
            CoronalMeasure::Centroids => Box::new(Centroids(&coronal_points.spine)),
            CoronalMeasure::SpinalLine => Box::new(SpinalLine(&coronal_points.spine)),
        }
    }
}

impl<'a> From<(CoronalMeasure, &'a CoronalPoints, &'a CurveSet, &'a ApexSet)>
    for Box<dyn MeasureComponent + 'a>
{
    fn from(value: (CoronalMeasure, &'a CoronalPoints, &'a CurveSet, &'a ApexSet)) -> Self {
        let (measure, coronal_points, curve_set, _apex_set) = value;
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

            _ => panic!("Invalid conversion: {:?}", measure),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ColorPalettes {
    pub label_colors: ColorPalette,
    pub line_colors: ColorPalette,
}
const VISIBILITY_HIDDEN: &str = "hidden";
const VISIBILITY_VISIBLE: &str = "visible";

pub fn draw_sagittal(
    data: LabelMeDataWImage,
    sagittal_points: SagittalPoints,
    draws: Vec<SagittalMeasure>,
    hide: Vec<SagittalMeasure>,
    draw_param: DrawParam,
    svg_size: (usize, usize),
    palettes: ColorPalettes,
) -> Result<element::SVG, DrawError> {
    let ColorPalettes {
        mut label_colors,
        mut line_colors,
    } = palettes;
    let painter = Painter::new(draw_param.clone(), svg_size);
    let mut document = painter.doc_w_background(&data.image)?;
    let style = element::Style::new(draw_param.style());
    document = document.add(style);

    let mut groups = Vec::with_capacity(draws.len());
    for measure in draws {
        let spinal_measure: Box<dyn DrawComponent> = (measure, &sagittal_points).into();
        match spinal_measure.draw(&painter, &mut label_colors, &mut line_colors) {
            Ok(g) => {
                let visibility = if hide.contains(&measure) {
                    VISIBILITY_HIDDEN
                } else {
                    VISIBILITY_VISIBLE
                };
                let g = g.set("visibility", visibility);
                groups.push(g.into());
            }
            Err(err) => warn!("Failed to draw {}: {:?}", spinal_measure.id(), err),
        }
    }

    let spacing = sagittal_points.image_metadata.spacing_xy;
    groups = scale_coordinates((1.0 / spacing.0, 1.0 / spacing.1), groups);
    for g in groups {
        document = document.add(g);
    }

    Ok(document)
}

pub fn draw_coronal(
    data: LabelMeDataWImage,
    coronal_points: CoronalPoints,
    draws_hide: (Vec<CoronalMeasure>, Vec<CoronalMeasure>),
    draw_param: DrawParam,
    svg_size: (usize, usize),
    palettes: ColorPalettes,
    curve_apex_set: Option<(CurveSet, ApexSet)>,
) -> Result<element::SVG, DrawError> {
    let ColorPalettes {
        mut label_colors,
        mut line_colors,
    } = palettes;
    let (draws, hide) = draws_hide;

    let painter = Painter::new(draw_param.clone(), svg_size);
    let mut document = painter.doc_w_background(&data.image)?;
    let style = element::Style::new(draw_param.style());

    document = document.add(style);

    let (curve_set, apex_set) = curve_apex_set.unwrap_or_else(|| {
        let (cs, apexes, _major_curve) = coronal_points.identify_curves();
        debug!("CurveSet: {:?}", cs);
        (cs, apexes)
    });

    let mut groups = Vec::with_capacity(draws.len());
    for measure in draws {
        let spinal_measure: Box<dyn DrawComponent> =
            (measure, &coronal_points, &curve_set, &apex_set).into();

        match spinal_measure.draw(&painter, &mut label_colors, &mut line_colors) {
            Ok(g) => {
                let visibility = if hide.contains(&measure) {
                    VISIBILITY_HIDDEN
                } else {
                    VISIBILITY_VISIBLE
                };
                let g = g.set("visibility", visibility);
                groups.push(g.into());
            }
            Err(err) => warn!("Failed to draw {}: {:?}", spinal_measure.id(), err),
        }
    }

    let spacing = coronal_points.image_metadata.spacing_xy;
    groups = scale_coordinates((1.0 / spacing.0, 1.0 / spacing.1), groups);
    for g in groups {
        document = document.add(g);
    }

    Ok(document)
}
