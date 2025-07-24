use crate::implant::ScrewSpine;
use crate::{
    angle_from_lines, CoronalDraw, CoronalPointsAndCurve, HasCornerPoints, HasImageMetadata,
    ImplantDraw, L2Norm, SagittalDraw, SagittalPoints, Scalable, ScaledType, ValidateLength,
    CORNER_LABELS,
};
use crate::{Curve, DrawParam, Spine, VERTEBRAL_LABELS};
use labelme_rs::image::DynamicImage;
use labelme_rs::ResizeParam;
use log::{debug, warn};
pub use named_derive::Named;
use ndarray::{s, stack, Array2, ArrayBase, ArrayView2, Axis, Ix1, Ix2};
use ndarray_stats::DeviationExt;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Read;
use std::ops::{AddAssign, SubAssign};
use svg::node::element;
use svg::Node;
pub type LineColors = HashMap<String, String>;

mod coronal;
mod sagittal;

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

type ColorMap = HashMap<String, String>;

#[derive(Debug, Clone, Default)]
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

/// Tableau 10 color palette without grayish colors
pub const TAB10_NEW_TAB10: [&str; 18] = [
    "#1f77b4", "#ff7f0f", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
    "#16becf", "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14e", "#edc949", "#af7aa1",
    "#ff9da7", "#9c755f",
];

pub struct Painter {
    pub param: DrawParam,
    pub size: (usize, usize),
    pub bbox: BoundingBox,
}

const X_ATTRS: [&str; 4] = ["x", "cx", "x1", "x2"];
const Y_ATTRS: [&str; 4] = ["y", "cy", "y1", "y2"];

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

        a_scaled_numbers[0] *= scale.0;
        a_scaled_numbers[1] *= scale.1;
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
            } else if attr == "width" {
                if let Ok(size) = value.parse::<f64>() {
                    *value = format!("{}", size * scale.0).into();
                }
            } else if attr == "height" {
                if let Ok(size) = value.parse::<f64>() {
                    *value = format!("{}", size * scale.1).into();
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
    pub flip_sign: bool,
}

impl Default for CobbAux {
    fn default() -> Self {
        Self {
            plate_scale: 3.0,
            perpendicular_scale: 1.4,
            flip_sign: false,
        }
    }
}

impl CobbAux {
    pub fn opposite_default() -> Self {
        let mut dft = Self::default();
        dft.plate_scale *= -1.0;
        dft
    }
    pub fn flip_default() -> Self {
        Self {
            flip_sign: true,
            ..Default::default()
        }
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
        Self {
            param,
            size,
            bbox: BoundingBox::default(),
        }
    }

    pub fn text<S>(
        &mut self,
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
        self.bbox.update((coords[0], coords[1]));
        t
    }

    pub fn title(&self, text: &str) -> element::Title {
        element::Title::new(text)
    }

    pub fn point<S>(&mut self, point: ArrayBase<S, Ix1>) -> element::Circle
    where
        S: ndarray::Data<Elem = f64>,
    {
        self.bbox.update((point[0], point[1]));
        element::Circle::new()
            .set("cx", point[0])
            .set("cy", point[1])
            .set("r", self.param.radius)
    }

    pub fn line<S>(&mut self, start_end: ArrayBase<S, Ix2>) -> element::Line
    where
        S: ndarray::Data<Elem = f64>,
    {
        self.bbox.update((start_end[[0, 0]], start_end[[0, 1]]));
        self.bbox.update((start_end[[1, 0]], start_end[[1, 1]]));
        element::Line::new()
            .set("x1", start_end[[0, 0]])
            .set("y1", start_end[[0, 1]])
            .set("x2", start_end[[1, 0]])
            .set("y2", start_end[[1, 1]])
    }

    pub fn horizontal_line(&mut self, y: f64) -> element::Line {
        // Not updating bbox because current implementation does not respect scaled coordinates
        // self.bbox.update((0.0, y));
        // self.bbox.update((self.size.0 as f64, y));
        element::Line::new()
            .set("x1", 0)
            .set("y1", y)
            .set("x2", "100%")
            .set("y2", y)
    }

    pub fn vertical_line(&mut self, x: f64) -> element::Line {
        // Not updating bbox because current implementation does not respect scaled coordinates
        // self.bbox.update((x, 0.0));
        // self.bbox.update((x, self.size.1 as f64));
        element::Line::new()
            .set("x1", x)
            .set("y1", 0)
            .set("x2", x)
            .set("y2", "100%")
    }

    pub fn polyline<S>(&mut self, points: ArrayBase<S, Ix2>) -> element::Polyline
    where
        S: ndarray::Data<Elem = f64>,
    {
        for point in points.axis_iter(Axis(0)) {
            self.bbox.update((point[0], point[1]));
        }
        let s = points.join(" ");
        element::Polyline::new().set("points", s)
    }

    pub fn rectangle<S>(&mut self, rect: ArrayBase<S, Ix2>) -> element::Rectangle
    where
        S: ndarray::Data<Elem = f64>,
    {
        self.bbox.update((rect[[0, 0]], rect[[0, 1]]));
        self.bbox.update((rect[[1, 0]], rect[[1, 1]]));
        element::Rectangle::new()
            .set("x", rect[[0, 0]])
            .set("y", rect[[0, 1]])
            .set("width", rect[[1, 0]] - rect[[0, 0]])
            .set("height", rect[[1, 1]] - rect[[0, 1]])
    }

    pub fn polygon<S>(&mut self, points: ArrayBase<S, Ix2>) -> element::Polygon
    where
        S: ndarray::Data<Elem = f64>,
    {
        for point in points.axis_iter(Axis(0)) {
            self.bbox.update((point[0], point[1]));
        }
        let s = points.join(" ");
        element::Polygon::new().set("points", s)
    }

    pub fn angle_between<S>(
        &mut self,
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
        let x_axis_rotation = 0;
        let large_arc_flag = 0;
        let angle_rad = angle_between(line1, line2);
        let sweep_flag = if angle_rad < 0.0 { 0 } else { 1 };
        let angle_deg = angle_rad.to_degrees();
        let data = element::path::Data::new()
            .move_to((arc_start[0], arc_start[1]))
            .elliptical_arc_to((
                arc_radius,
                arc_radius,
                x_axis_rotation,
                large_arc_flag,
                sweep_flag,
                arc_end[0],
                arc_end[1],
            ));
        let arc = element::Path::new().set('d', data).set("fill", "none");
        group = group.add(arc);
        self.bbox.update((arc_start[0], arc_start[1]));
        self.bbox.update((arc_end[0], arc_end[1]));
        let text = self.text(
            format!("{:.1}°", angle_deg).as_str(),
            rotate_around(arc_start.view(), cross.view(), -angle_rad / 2.0),
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

    /// Draw two arrows (parallel sign) at the middle of the line
    pub fn parallel_sign(
        &mut self,
        group: element::Group,
        line: ArrayView2<f64>,
        length: f64,
    ) -> element::Group {
        let d = &line.slice(s![1, ..]) - &line.slice(s![0, ..]);
        let unit_d = &d / d.l2norm();
        let p1 = line.mean_axis(Axis(0)).unwrap();
        let p2 = rotate_around(&p1 - &unit_d * length, p1.view(), 30.0_f64.to_radians());
        let p3 = rotate_around(&p1 - &unit_d * length, p1.view(), -30.0_f64.to_radians());
        let arrow = ndarray::stack![Axis(0), p2, p1, p3];
        let arrow1 = &arrow - &unit_d * length * 0.5;
        let arrow2 = arrow + &unit_d * length * 0.5;
        let group = group.add(self.polyline(arrow1));
        group.add(self.polyline(arrow2))
    }

    pub fn cobb_from_plates(
        &mut self,
        mut group: element::Group,
        sup_plate_inf_plate: (Array2<f64>, Array2<f64>),
        aux_param: &CobbAux,
        base_length: f64,
        title: Option<&str>,
    ) -> element::Group {
        let sup_plate = sup_plate_inf_plate.0;
        let inf_plate = sup_plate_inf_plate.1;
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
                let angle = if aux_param.flip_sign { -angle } else { angle };
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
                let angle = if aux_param.flip_sign { -angle } else { angle };
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
        &mut self,
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
            (sup_plate.to_owned(), inf_plate.to_owned()),
            aux_param,
            base_length,
            title,
        )
    }

    pub fn doc_w_background(
        &self,
        image: &labelme_rs::image::DynamicImage,
    ) -> Result<svg::Document, labelme_rs::LabelMeDataError> {
        let b64 = format!(
            "data:image/jpeg;base64,{}",
            labelme_rs::img2base64(image, labelme_rs::image::ImageFormat::Jpeg)?
        );

        let bg = element::Image::new()
            .set("x", 0i64)
            .set("y", 0i64)
            .set("width", self.size.0)
            .set("height", self.size.1)
            .set("xlink:href", b64);

        let mut document = svg::Document::new()
            .set("width", self.size.0)
            .set("height", self.size.1)
            .set("viewBox", (0, 0, self.size.0, self.size.1))
            .set("xmlns:xlink", "http://www.w3.org/1999/xlink");
        document = document.add(bg);
        Ok(document)
    }

    pub fn doc_w_background_and_bbox(
        &self,
        image: &labelme_rs::image::DynamicImage,
        bbox: BoundingBox,
    ) -> Result<svg::Document, labelme_rs::LabelMeDataError> {
        let b64 = format!(
            "data:image/jpeg;base64,{}",
            labelme_rs::img2base64(image, labelme_rs::image::ImageFormat::Jpeg)?
        );

        let bg = element::Image::new()
            .set("x", 0i64)
            .set("y", 0i64)
            .set("width", self.size.0)
            .set("height", self.size.1)
            .set("xlink:href", b64);

        let mut view_box = (0, 0, self.size.0 as i64, self.size.1 as i64);

        if bbox.0 .0 < 0.0 {
            view_box.0 = bbox.0 .0 as i64;
        }
        if bbox.0 .1 < 0.0 {
            view_box.1 = bbox.0 .1 as i64;
        }
        if bbox.1 .0 > self.size.0 as f64 {
            view_box.2 = bbox.1 .0 as i64;
        }
        if bbox.1 .1 > self.size.1 as f64 {
            view_box.3 = bbox.1 .1 as i64;
        }

        let mut document = svg::Document::new()
            .set("width", view_box.2 - view_box.0)
            .set("height", view_box.3 - view_box.1)
            .set("viewBox", view_box)
            .set("xmlns:xlink", "http://www.w3.org/1999/xlink");
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
    InvalidNumberOfPoints(String, InvalidNumberOfPoints),

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
    fn id(&self) -> &str;
    /// Labels for the component
    fn label(&self) -> &str;
    /// Short description for the component
    fn description(&self) -> Option<&str>;
    fn draw_type(&self) -> &[&str];
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
        classes.push(self.id());
        self.ided_group().set("class", classes)
    }
}

/// Two dimensional bounding box with top-left and bottom-right corners
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox((f64, f64), (f64, f64));

impl Default for BoundingBox {
    fn default() -> Self {
        Self((f64::MAX, f64::MAX), (f64::MIN, f64::MIN))
    }
}

impl BoundingBox {
    pub fn update(&mut self, point: (f64, f64)) {
        let (x, y) = point;
        if x < self.0 .0 {
            self.0 .0 = x;
        }
        if y < self.0 .1 {
            self.0 .1 = y;
        }
        if x > self.1 .0 {
            self.1 .0 = x;
        }
        if y > self.1 .1 {
            self.1 .1 = y;
        }
    }
    pub fn scale(&self, sx: f64, sy: f64) -> Self {
        Self(
            (self.0 .0 * sx, self.0 .1 * sy),
            (self.1 .0 * sx, self.1 .1 * sy),
        )
    }
}

pub trait DrawComponent: Named {
    fn draw(
        &self,
        painter: &mut Painter,
        label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError>;
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

pub struct ImageOverlay {
    id: String,
    label: String,
    description: Option<String>,
    image: DynamicImage,
    image_size: (usize, usize),
}

impl ImageOverlay {
    pub fn new(
        id: String,
        label: String,
        description: Option<String>,
        image: DynamicImage,
        image_size: (usize, usize),
    ) -> Self {
        Self {
            id,
            label,
            description,
            image,
            image_size,
        }
    }
}

#[cfg(not(feature = "webp"))]
fn encode_image(image: &DynamicImage) -> Result<String, labelme_rs::LabelMeDataError> {
    let b64 = labelme_rs::img2base64(image, labelme_rs::image::ImageFormat::Png)?;
    Ok(format!("data:image/webp;base64,{}", b64))
}

#[cfg(feature = "webp")]
fn encode_image(image: &DynamicImage) -> Result<String, labelme_rs::LabelMeDataError> {
    use base64::Engine;

    let webp_encoder = webp::Encoder::from_image(image).unwrap();
    let webp_memory = webp_encoder.encode(90.0);
    let b64 = base64::engine::general_purpose::STANDARD.encode(&*webp_memory);
    Ok(format!("data:image/webp;base64,{}", b64))
}

impl Named for ImageOverlay {
    fn id(&self) -> &str {
        &self.id
    }
    fn label(&self) -> &str {
        &self.label
    }
    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
    fn draw_type(&self) -> &[&str] {
        &["ImageOverlay"]
    }
}
impl CommonComponent for ImageOverlay {}
impl DrawComponent for ImageOverlay {
    fn draw(
        &self,
        _painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let group = self.default_group();

        let base64_image_data = encode_image(&self.image).map_err(DrawError::LabelMeDataError)?;
        let layer = element::Image::new()
            .set("x", 0i64)
            .set("y", 0i64)
            .set("width", self.image_size.0)
            .set("height", self.image_size.1)
            .set("xlink:href", base64_image_data);
        Ok(group.add(layer))
    }
}

/// Draw a set of points
pub struct DrawPointSet {
    id: String,
    label: String,
    description: Option<String>,
    points: Array2<f64>,
}

impl DrawPointSet {
    pub fn new(
        id: String,
        label: String,
        description: Option<String>,
        points: Array2<f64>,
    ) -> Self {
        Self {
            id,
            label,
            description,
            points,
        }
    }
}

impl Named for DrawPointSet {
    fn id(&self) -> &str {
        &self.id
    }
    fn label(&self) -> &str {
        &self.label
    }
    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
    fn draw_type(&self) -> &[&str] {
        &["PointSet"]
    }
}
impl CommonComponent for DrawPointSet {}
impl DrawComponent for DrawPointSet {
    fn draw(
        &self,
        painter: &mut Painter,
        label_colors: &mut ColorPalette,
        line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let color = label_colors.get_or_new(&self.label);
        let stroke_color = line_colors.get_or_new(&self.label);
        let mut group: element::Group = self
            .default_group()
            .set("stroke", stroke_color)
            .set("fill", color);
        for point in self.points.axis_iter(Axis(0)) {
            let p = painter.point(point);
            group = group.add(p);
        }
        Ok(group)
    }
}

/// Label text for each vertebra
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_TEXT])]
pub struct VertebralLabels<'a>(pub &'a Spine);
impl CommonComponent for VertebralLabels<'_> {}
impl DrawComponent for VertebralLabels<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
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

/// All detected points
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POINT])]
pub struct AllPoints<'a>(Vec<(String, ArrayView2<'a, f64>)>);
impl CommonComponent for AllPoints<'_> {}
impl DrawComponent for AllPoints<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let mut g_points = self.default_group();
        for (label, points) in self.0.iter() {
            let color = label_colors.get_or_new(label);
            let mut sub_group: element::Group = element::Group::new()
                .set("stroke", color)
                .set("fill", color);
            sub_group = sub_group.add(painter.title(label.as_str()));
            for point in points.axis_iter(Axis(0)) {
                let p = painter.point(point);
                sub_group = sub_group.add(p);
            }
            g_points = g_points.add(sub_group);
        }
        Ok(g_points)
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
impl CommonComponent for VertebralPoints<'_> {}
impl DrawComponent for VertebralPoints<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let g = self.default_group();
        self.draw_corners(g, painter, label_colors)
    }
}

pub trait DrawCorners {
    fn draw_corners(
        &self,
        group: element::Group,
        painter: &mut Painter,
        label_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError>;
}

///  Helper trait to provide default implementation for drawing corners
impl<T> DrawCorners for T
where
    T: HasCornerPoints + Named,
{
    fn draw_corners(
        &self,
        group: element::Group,
        painter: &mut Painter,
        label_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
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
pub const CLASS_RATIO: &str = "Ratio";

/// Centroids of each vertebra
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POINT])]
struct Centroids<'a>(&'a Spine);
impl CommonComponent for Centroids<'_> {}
impl DrawComponent for Centroids<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        label_colors: &mut ColorPalette,
        _line_colors: &mut ColorPalette,
    ) -> Result<element::Group, DrawError> {
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

pub fn mean_plate_length(scol: &Spine) -> f64 {
    let corners = scol.tl_corners().0;
    let sup_inf_shape = (corners.len_of(Axis(0)) * 2, 2, 2); // [n * sup_inf, lr, xy]

    let plate_lr = corners.to_shape(sup_inf_shape).unwrap();
    let diff = &plate_lr.index_axis(Axis(1), 0) - &plate_lr.index_axis(Axis(1), 1);
    diff.map_axis(Axis(1), |a| a.l2norm()).mean().unwrap()
}

fn difference_in_index(
    label: &str,
    points: ArrayView2<f64>,
    index: usize,
) -> Result<f64, MeasureError> {
    points.validate_label_length(label, 2)?;
    Ok(points.index_axis(Axis(0), 0)[index] - points.index_axis(Axis(0), 1)[index])
}

fn difference_in_x(label: &str, points: ArrayView2<f64>) -> Result<f64, MeasureError> {
    difference_in_index(label, points, 0)
}

/// Difference in y-coordinates of two points
/// Return positive value if the first point is above the second point
fn difference_in_y(label: &str, points: ArrayView2<f64>) -> Result<f64, MeasureError> {
    difference_in_index(label, points, 1).map(|dy| -dy)
}

fn draw_difference_in_x(
    group: element::Group,
    point_label: &str,
    draw_label: &str,
    points: ArrayView2<f64>,
    painter: &mut Painter,
    unit: &str,
) -> Result<element::Group, DrawError> {
    let dx = difference_in_x(point_label, points)?;
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
        Some(draw_label),
        Some(unit),
    );

    g = g.add(text);
    Ok(g)
}

fn draw_difference_in_y(
    group: element::Group,
    point_label: &str,
    draw_label: &str,
    points: ArrayView2<f64>,
    painter: &mut Painter,
    unit: &str,
) -> Result<element::Group, DrawError> {
    let dy = difference_in_y(point_label, points)?;
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
    let text = painter.text(&text, text_pos, Some(draw_label), Some(unit));

    g = g.add(text);
    Ok(g)
}

/// Calculate the angle (degree) between two lines defined by points
pub fn tilt_angle(label: &str, points: ArrayView2<f64>) -> Result<f64, MeasureError> {
    points.validate_label_length(label, 2)?;
    let mut hor_line = points.to_owned();
    hor_line[[1, 1]] = points[[0, 1]];
    let angle = angle_between(hor_line.view(), points);
    Ok(angle.to_degrees())
}

pub fn draw_tilt_angle(
    group: element::Group,
    painter: &mut Painter,
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

pub fn draw_incidence_angle(
    group: element::Group,
    label: &str,
    femoral_heads: ArrayView2<f64>,
    plate: ArrayView2<f64>,
    painter: &mut Painter,
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
    perp_sac = 0.85 * sac_sup_mid.l2_dist(&mid_femoral_heads).unwrap() * perp_sac;
    perp_sac += &sac_sup_mid;
    let perp_line = stack![Axis(0), sac_sup_mid, perp_sac];
    let arc_radius = 0.5 * sac_sup_mid.l2_dist(&mid_femoral_heads).unwrap();
    g = painter
        .angle_between(
            g,
            perp_line.view(),
            line_sac2fem.view(),
            sac_sup_mid.view(),
            arc_radius,
            Some(label),
        )
        .0;

    g
}

pub fn femoral_incidence_angle(
    plate: ArrayView2<f64>,
    femoral_head: &crate::AtMost2<Array2<f64>>,
) -> Result<f64, MeasureError> {
    femoral_head
        .0
        .validate_label_length_more_than("FemoralHead", 1)?;
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
    Ok(angle_deg.abs())
}

/// Shared function for drawing T1 tilt angle in coronal view and T1 slope in sagittal view
fn draw_t1_angle<T>(
    painter: &mut Painter,
    line_colors: &mut ColorPalette,
    spine: &Spine,
    component: &T,
    flip_sign: bool,
    default_group: element::Group,
) -> Result<element::Group, DrawError>
where
    T: DrawComponent,
{
    // let spine = &component.0.spine;
    let label = component.id();
    let mut g = default_group
        .set("stroke", line_colors.get_or_new(label))
        .set("fill", "none");
    let tl_sup_lines = spine.tl_sup_lines();
    let t1sup = tl_sup_lines.index_axis(Axis(0), 0);
    let mid = t1sup.mean_axis(Axis(0)).unwrap();
    let (mult_left, mult_right, mult_arc) = (1.0, 4.0, 3.0);
    let l2r = &t1sup.index_axis(Axis(0), 1) - mult_left * &t1sup.index_axis(Axis(0), 0);
    if l2r.l2norm() == 0.0 {
        return Err(DrawError::MeasureError(MeasureError::ZeroLengthLine));
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

        // draw tilted T1 line
        let mut hor_line = stack![Axis(0), mid.view(), mid.view()];
        hor_line[[0, 0]] -= mult_left * l2r.l2norm();
        hor_line[[1, 0]] += mult_right * l2r.l2norm();
        g = if flip_sign {
            painter
                .angle_between(
                    g,
                    sup_line.view(),
                    hor_line.view(),
                    mid.view(),
                    arc_radius,
                    Some(label),
                )
                .0
        } else {
            painter
                .angle_between(
                    g,
                    hor_line.view(),
                    sup_line.view(),
                    mid.view(),
                    arc_radius,
                    Some(label),
                )
                .0
        };
    };
    Ok(g)
}

#[derive(Debug, Clone)]
pub struct ColorPalettes {
    pub label_colors: ColorPalette,
    pub line_colors: ColorPalette,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LabelColors {
    label_colors: HashMap<String, labelme_rs::Color>,
}

impl Default for ColorPalettes {
    fn default() -> Self {
        // embed the content in tests/data/colors.yaml
        let config: LabelColors =
            serde_yaml::from_str(include_str!("../../../tests/data/colors.yaml"))
                .expect("Failed to parse colors.yaml");
        let label_colors = labelme_rs::LabelColorsHex::from_iter(
            config.label_colors.into_iter().map(|(k, v)| (k, v.into())),
        );
        let label_colors = ColorPalette::new(label_colors);
        // embed the content in tests/data/line_colors.csv
        let line_colors =
            load_line_colors(&include_bytes!("../../../tests/data/line_colors.csv")[..]).unwrap();
        let line_colors = ColorPalette::new(line_colors);
        Self {
            label_colors,
            line_colors,
        }
    }
}

const VISIBILITY_HIDDEN: &str = "hidden";
const VISIBILITY_VISIBLE: &str = "visible";

/// Draw the given components
///
/// Pass data in pixel coordinates becase scaling based on image_metadata is handled inside this function.
pub fn draw_components<'a, T, S>(
    data: T,
    draws: &[S],
    hide: &[S],
    painter: &mut Painter,
    palettes: ColorPalettes,
) -> Result<Vec<Box<dyn Node>>, DrawError>
where
    for<'b> (&'b S, &'b ScaledType<T>): Into<Box<dyn DrawComponent + 'b>>,
    S: Clone + Copy + PartialEq,
    T: HasImageMetadata + Scalable,
    <T as Scalable>::Error: std::fmt::Debug,
{
    let scaled_data = data
        .into_scaled()
        .map_err(|e| MeasureError::UnableToMeasure(format!("Failed to scale: {:?}", e)))?;
    let ColorPalettes {
        mut label_colors,
        mut line_colors,
    } = palettes;
    let mut groups = Vec::with_capacity(draws.len());
    for measure in draws {
        let draw_component: Box<dyn DrawComponent> = (measure, &scaled_data).into();
        match draw_component.draw(painter, &mut label_colors, &mut line_colors) {
            Ok(g) => {
                let visibility = if hide.contains(measure) {
                    VISIBILITY_HIDDEN
                } else {
                    VISIBILITY_VISIBLE
                };
                let g = g.set("visibility", visibility);
                groups.push(g.into());
            }
            Err(err) => warn!("Failed to draw {}: {:?}", draw_component.id(), err),
        }
    }

    let image_metadata = scaled_data.0.image_metadata();

    // revert the scaling to the original and rescale to the svg size
    let spacing = image_metadata.spacing_xy;
    let svg_to_image_ratio_x = painter.size.0 as f64 / image_metadata.width as f64;
    let svg_to_image_ratio_y = painter.size.1 as f64 / image_metadata.height as f64;
    let scale = (
        1.0 / spacing.0 * svg_to_image_ratio_x,
        1.0 / spacing.1 * svg_to_image_ratio_y,
    );
    let groups = scale_coordinates(scale, groups);
    // adjust bbox
    painter.bbox = painter.bbox.scale(scale.0, scale.1);

    Ok(groups)
}

pub struct DrawArguments<'a, T, S> {
    pub image: DynamicImage,
    pub data: T,
    pub draw: &'a [S],
    pub hide: &'a [S],
    pub draw_param: DrawParam,
    pub resize_param: Option<ResizeParam>,
    pub svg_size: Option<(usize, usize)>,
    pub palettes: ColorPalettes,
}

/// Draw the given components on the image.
///
/// Pass data in pixel coordinates becase scaling based on image_metadata is handled inside this function.
pub fn draw_on_image<'a, T, S>(args: DrawArguments<'a, T, S>) -> Result<element::SVG, DrawError>
where
    for<'b> (&'b S, &'b ScaledType<T>): Into<Box<dyn DrawComponent + 'b>>,
    S: Clone + Copy + PartialEq,
    T: HasImageMetadata + Scalable,
    <T as Scalable>::Error: std::fmt::Debug,
{
    let DrawArguments {
        image,
        data,
        draw,
        hide,
        draw_param,
        resize_param,
        svg_size,
        palettes,
    } = args;
    let style = element::Style::new(draw_param.style());
    let image: Cow<DynamicImage> = match resize_param {
        Some(resize_param) => Cow::Owned(resize_param.resize(&image)),
        None => Cow::Borrowed(&image),
    };
    let svg_size = svg_size.unwrap_or_else(|| (image.width() as usize, image.height() as usize));
    let mut painter = Painter::new(draw_param, svg_size);

    // draw first to update the internal bounding box
    let groups = draw_components(data, draw, hide, &mut painter, palettes)?;

    let bbox = painter.bbox;
    let mut document = painter.doc_w_background_and_bbox(&image, bbox)?;
    document = document.add(style);

    for g in groups {
        document = document.add(g);
    }
    Ok(document)
}

pub type SagittalDrawArguments<'a> = DrawArguments<'a, SagittalPoints, SagittalDraw>;
pub type CoronalDrawArguments<'a> = DrawArguments<'a, CoronalPointsAndCurve, CoronalDraw>;
pub type ImplantDrawArguments<'a> = DrawArguments<'a, ScrewSpine, ImplantDraw>;

pub fn draw_sagittal(args: SagittalDrawArguments) -> Result<element::SVG, DrawError> {
    draw_on_image(args)
}

pub fn draw_coronal(args: CoronalDrawArguments) -> Result<element::SVG, DrawError> {
    draw_on_image(args)
}

pub fn draw_implant(args: ImplantDrawArguments) -> Result<element::SVG, DrawError> {
    draw_on_image(args)
}

#[derive(thiserror::Error, Debug)]
pub enum HtmlWrapError {
    #[error("Failed to find `id` attribute")]
    IdNotFound,
    #[error("Failed to render template: {0}")]
    Tera(#[from] tera::Error),
}

/// Wrap the SVG in HTML with visibility toggles
pub fn wrap_in_html(
    svg: String,
    selector: &[String],
    title: String,
) -> Result<String, HtmlWrapError> {
    let document = scraper::Html::parse_document(&svg);

    let mut templates = tera::Tera::default();
    templates.autoescape_on(vec![]);
    templates
        .add_raw_templates(vec![
            ("html.jinja", include_str!("templates/html.jinja")),
            ("checkbox.jinja", include_str!("templates/checkbox.jinja")),
            (
                "draw_setting.jinja",
                include_str!("templates/draw_setting.jinja"),
            ),
            (
                "draw_setting.css",
                include_str!("templates/draw_setting.css"),
            ),
            ("draw_setting.js", include_str!("templates/draw_setting.js")),
        ])
        .unwrap();
    let javascript = include_str!("templates/capture.js");
    let mut elements: Vec<_> = Vec::new();
    for selector in selector {
        let selector = scraper::Selector::parse(selector).unwrap_or_else(|_| {
            panic!("Failed to parse selector: {}", &selector);
        });
        elements.extend(document.select(&selector));
    }

    let mut checkboxes = vec![];
    for element in elements {
        let mut context = tera::Context::new();
        let id = element
            .value()
            .attr("id")
            .ok_or(HtmlWrapError::IdNotFound)?;
        context.insert("id", &id);
        context.insert("label", &element.value().attr("data-label").unwrap_or(id));
        context.insert(
            "description",
            &element.value().attr("data-description").unwrap_or(""),
        );
        let visibility = element.value().attr("visibility").unwrap_or("visible");
        let checked = if visibility == "visible" {
            "checked"
        } else {
            ""
        };
        context.insert("checked", checked);
        checkboxes.push(templates.render("checkbox.jinja", &context)?);
    }

    let mut context = tera::Context::new();
    context.insert("svg", &svg);
    context.insert("checkboxes", &checkboxes.join("\n"));
    context.insert("title", &title);
    context.insert("javascript", &javascript);
    context.insert("save_module", &include_str!("templates/save_module.html"));

    let html = templates.render("html.jinja", &context)?;
    Ok(html)
}
