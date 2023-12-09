use std::ops::{AddAssign, SubAssign};

use anyhow::{Context, Result};
use labelme_rs::{image::GenericImageView, LabelColorsHex, LabelMeData, LabelMeDataWImage};
use log::debug;
use ndarray::{s, ArrayBase, Axis, Ix1, Ix2};
use ndarray_stats::DeviationExt;
use scolrs::{
    ApexSet, Curve, CurveSet, LineColors, RenderParam, Scoliosis, VertebralIndex, VERTEBRAL_LABELS,
};
use scolrs::{CurveInfo, L2Norm};
use svg::node::element;

use crate::cli::{Direction, RenderArgs};

struct Renderer {
    pub param: RenderParam,
    pub size: (usize, usize),
}

fn squared_distance<S>(p1: ArrayBase<S, Ix1>, p2: ArrayBase<S, Ix1>) -> f32
where
    S: ndarray::Data<Elem = f32>,
{
    (&p1 - &p2).mapv(|a| a * a).sum()
}

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
struct CobbAux {
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

// enum CobbParam {
//     Aux(CobbAux),
// }

impl Renderer {
    fn new(param: RenderParam, size: (usize, usize)) -> Self {
        Self { param, size }
    }

    fn point<S>(&self, point: ArrayBase<S, Ix1>) -> element::Circle
    where
        S: ndarray::Data<Elem = f32>,
    {
        element::Circle::new()
            .set("cx", point[0])
            .set("cy", point[1])
            .set("r", self.param.radius)
    }

    fn line<S>(&self, start_end: ArrayBase<S, Ix2>) -> element::Line
    where
        S: ndarray::Data<Elem = f32>,
    {
        element::Line::new()
            .set("x1", start_end[[0, 0]])
            .set("y1", start_end[[0, 1]])
            .set("x2", start_end[[1, 0]])
            .set("y2", start_end[[1, 1]])
    }

    fn polyline<S>(&self, points: ArrayBase<S, Ix2>) -> element::Polyline
    where
        S: ndarray::Data<Elem = f32>,
    {
        let s = points.join(" ");
        element::Polyline::new().set("points", s)
    }

    fn polygon<S>(&self, points: ArrayBase<S, Ix2>) -> element::Polygon
    where
        S: ndarray::Data<Elem = f32>,
    {
        let s = points.join(" ");
        element::Polygon::new().set("points", s)
    }

    fn text<S>(&self, text: &str, coords: ArrayBase<S, Ix1>) -> element::Text
    where
        S: ndarray::Data<Elem = f32>,
    {
        element::Text::new()
            .set("x", coords[0])
            .set("y", coords[1])
            .add(svg::node::Text::new(text))
    }

    fn plate_end<S, T>(plate: ArrayBase<S, Ix2>, point: ArrayBase<T, Ix1>) -> usize
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

    fn cobb(
        &self,
        mut group: element::Group,
        scol: &Scoliosis,
        curve: &Curve,
        aux_param: &CobbAux,
        base_length: f32,
    ) -> element::Group {
        let sup_plate = scol.tl_sup_plate(curve.sup);
        let inf_plate = scol.tl_inf_plate(curve.inf);
        let sup_line = plate2line(sup_plate);
        let inf_line = plate2line(inf_plate);

        let linter = sup_line.intersection(&inf_line);
        if let Some(intersection) = linter {
            let is_inside = intersection.x > 0.0
                && intersection.x < self.size.0 as f32
                && intersection.y > 0.0
                && intersection.y < self.size.1 as f32;
            let angle = scol.angle(curve).unwrap(); // lines can't be parallel if there is an intersection point

            let arr_int = ndarray::arr1(&[intersection.x, intersection.y]);
            let i = Self::plate_end(sup_plate, arr_int.view());
            let d = &sup_plate.slice(s![1 - i, ..]) - &sup_plate.slice(s![i, ..]);
            let unit_d = &d / d.l2norm();
            let aux_on_sup: ndarray::Array1<_> =
                &sup_plate.slice(s![i, ..]) + aux_param.plate_scale * base_length * &unit_d;
            let aux_cross =
                Self::rotate_around(aux_on_sup.view(), arr_int.view(), angle.to_radians() / 2.0);

            let d_btw_aux2p = aux_cross.l2_dist(&sup_plate.slice(s![i, ..])).unwrap();
            let arr_int = ndarray::arr1(&[intersection.x, intersection.y]);
            let d_btw_int2p = arr_int.l2_dist(&sup_plate.slice(s![i, ..])).unwrap();
            if is_inside && d_btw_aux2p > d_btw_int2p {
                // draw intersection point
                for plate in [sup_plate, inf_plate] {
                    let i = Self::plate_end(plate, arr_int.view());
                    let line = self.line(ndarray::arr2(&[
                        [plate[[i, 0]], plate[[i, 1]]],
                        [intersection.x, intersection.y],
                    ]));
                    group = group.add(line);
                }
                let text = self
                    .text(
                        format!("{:.1}°", angle).as_str(),
                        ndarray::arr1(&[intersection.x, intersection.y]),
                    )
                    .set("stroke", self.param.text_stroke.as_str())
                    .set("stroke-width", self.param.text_stroke_width)
                    .set("fill", self.param.text_fill.as_str());

                group = group.add(text);
            } else {
                // draw aux lines and its intersection

                for plate in [sup_plate, inf_plate] {
                    let i = Self::plate_end(sup_plate, arr_int.view());
                    let d = &plate.slice(s![1 - i, ..]) - &plate.slice(s![i, ..]);
                    let d_aux = &aux_cross - &plate.slice(s![i, ..]);
                    let t = d_aux.dot(&d) / d.mapv(|a| a * a).sum();
                    let projed_aux = &plate.slice(s![i, ..]) + t * &d;
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
                let text = self
                    .text(format!("{:.1}°", angle).as_str(), aux_cross)
                    .set("stroke", self.param.text_stroke.as_str())
                    .set("stroke-width", self.param.text_stroke_width)
                    .set("fill", self.param.text_fill.as_str());
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
                let len = self.param.line_width as f32 * 8.0;
                let p1 = plate.mean_axis(Axis(0)).unwrap();
                let p2 = Self::rotate_around(&p1 - &unit_d * len, p1.view(), 30.0_f32.to_radians());
                let p3 =
                    Self::rotate_around(&p1 - &unit_d * len, p1.view(), -30.0_f32.to_radians());
                let arrow = ndarray::stack![Axis(0), p1, p2, p3];
                let arrow = self.polygon(arrow);

                group = group.add(arrow);
            }
        }
        group
    }

    fn doc_w_background(&self, image: &labelme_rs::image::DynamicImage) -> svg::Document {
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

const CLS_POINT: &str = "Points";

fn plate2line<S>(plate: ArrayBase<S, Ix2>) -> lyon_geom::Line<f32>
where
    S: ndarray::Data<Elem = f32>,
{
    let point = lyon_geom::Point::new(plate[[0, 0]], plate[[0, 1]]);
    let p2 = lyon_geom::Point::new(plate[[1, 0]], plate[[1, 1]]);
    let vector = p2 - point;
    lyon_geom::Line { point, vector }
}

/// Trait to provide common interface from HashMap and IndexMap
trait TryGet<K, V> {
    fn try_get<Q>(&self, key: &Q) -> Option<&V>
    where
        String: std::borrow::Borrow<Q>,
        Q: ?Sized + core::hash::Hash + std::cmp::Eq;
}

impl TryGet<String, String> for LabelColorsHex {
    fn try_get<Q>(&self, key: &Q) -> Option<&String>
    where
        String: std::borrow::Borrow<Q>,
        Q: ?Sized + core::hash::Hash + std::cmp::Eq,
    {
        self.get(key)
    }
}

impl TryGet<String, String> for LineColors {
    fn try_get<Q>(&self, key: &Q) -> Option<&String>
    where
        String: std::borrow::Borrow<Q>,
        Q: ?Sized + core::hash::Hash + std::cmp::Eq,
    {
        self.get(key)
    }
}

struct ColorPaletts<T>
where
    T: TryGet<String, String>,
{
    color_map: T,
    color_cycler: labelme_rs::ColorCycler,
}

impl<T> ColorPaletts<T>
where
    T: TryGet<String, String>,
{
    fn new(color_map: T) -> ColorPaletts<T> {
        let color_cycler = labelme_rs::ColorCycler::new();
        Self {
            color_map,
            color_cycler,
        }
    }

    fn get_or_new<Q>(&mut self, key: &Q) -> &str
    where
        String: std::borrow::Borrow<Q>,
        Q: ?Sized + core::hash::Hash + std::cmp::Eq + std::fmt::Display,
    {
        self.color_map.try_get(key).map_or_else(
            || {
                debug!("New color generated for {}", key);
                self.color_cycler.cycle()
            },
            |s| s.as_str(),
        )
    }
}

trait Component {
    fn render(
        &self,
        scol: &Scoliosis,
        renderer: &Renderer,
        label_colors: &mut ColorPaletts<LabelColorsHex>,
        line_colors: &mut ColorPaletts<LineColors>,
    ) -> element::Group;
}

struct VertebralLabels;
impl Component for VertebralLabels {
    fn render(
        &self,
        scol: &Scoliosis,
        renderer: &Renderer,
        _label_colors: &mut ColorPaletts<LabelColorsHex>,
        _line_colors: &mut ColorPaletts<LineColors>,
    ) -> element::Group {
        let render_param = &renderer.param;
        let mut g_vert_labels = element::Group::new()
            .set("text-anchor", "middle")
            .set("dominant-baseline", "central")
            .set("stroke", render_param.text_stroke.as_str())
            .set("stroke-width", render_param.text_stroke_width)
            .set("fill", render_param.text_fill.as_str());
        let centroids = scol.tl_centroids();
        for (coords, label) in
            std::iter::zip(centroids.axis_iter(Axis(0)), VERTEBRAL_LABELS.into_iter())
        {
            let t = renderer.text(label, coords);
            g_vert_labels = g_vert_labels.add(t);
        }
        g_vert_labels
    }
}

struct VertebralPoints;
impl Component for VertebralPoints {
    fn render(
        &self,
        scol: &Scoliosis,
        renderer: &Renderer,
        label_colors: &mut ColorPaletts<LabelColorsHex>,
        _line_colors: &mut ColorPaletts<LineColors>,
    ) -> element::Group {
        let mut g_corners = element::Group::new();
        for (i_label, &label) in scolrs::CORNER_LABELS.iter().enumerate() {
            let color = label_colors.get_or_new(label);
            let mut sub_group = element::Group::new()
                .set("class", format!("{CLS_POINT} {label}"))
                .set("fill", color);
            let points = scol.vertebrae.0.index_axis(Axis(1), i_label);
            let n_points = if i_label < 2 {
                points.len_of(Axis(0))
            } else {
                points.len_of(Axis(0)) - 1 // BL and BR points of sacrum are dummies
            };
            for point in points.axis_iter(Axis(0)).take(n_points) {
                let p = renderer.point(point);
                sub_group = sub_group.add(p);
            }
            g_corners = g_corners.add(sub_group);
        }
        g_corners
    }
}

struct Centroids;
impl Component for Centroids {
    fn render(
        &self,
        scol: &Scoliosis,
        renderer: &Renderer,
        label_colors: &mut ColorPaletts<LabelColorsHex>,
        _line_colors: &mut ColorPaletts<LineColors>,
    ) -> element::Group {
        let label = "Centroid";
        let color = label_colors.get_or_new(label);
        let mut g_centroids = element::Group::new().set("class", label).set("fill", color);
        let centroids = scol.tl_centroids();
        for point in centroids.axis_iter(Axis(0)) {
            let p = renderer.point(point);
            g_centroids = g_centroids.add(p);
        }
        g_centroids
    }
}

fn mean_plate_length(scol: &Scoliosis) -> f32 {
    let corners = scol.tl_corners().0;
    let sup_inf_shape = (corners.len_of(Axis(0)) * 2, 2, 2); // [n * sup_inf, lr, xy]

    let plate_lr = corners.to_shape(sup_inf_shape).unwrap();
    let diff = &plate_lr.index_axis(Axis(1), 0) - &plate_lr.index_axis(Axis(1), 1);
    diff.map_axis(Axis(1), |a| a.l2norm()).mean().unwrap()
}

struct FrontalCobbAngles<'a>(&'a CurveSet);
impl<'a> Component for FrontalCobbAngles<'a> {
    fn render(
        &self,
        scol: &Scoliosis,
        renderer: &Renderer,
        _label_colors: &mut ColorPaletts<LabelColorsHex>,
        line_colors: &mut ColorPaletts<LineColors>,
    ) -> element::Group {
        let curve_set = self.0;
        debug!("Curve set:{:?}", curve_set);
        let mut g_angles = element::Group::new().set("class", "CobbAngles");
        let aux_param = CobbAux::default();

        let mean_plate_length = mean_plate_length(scol);

        if let Some((mt_curve, _angle)) = &curve_set.mt {
            let g_mt = element::Group::new()
                .set("class", "MT")
                .set("stroke", line_colors.get_or_new("MT"));
            let group = renderer.cobb(g_mt, scol, mt_curve, &aux_param, mean_plate_length);
            g_angles = g_angles.add(group);
        };

        if let Some((pt_curve, _angle)) = &curve_set.pt {
            let g_pt = element::Group::new()
                .set("class", "PT")
                .set("stroke", line_colors.get_or_new("PT"));
            let group = renderer.cobb(g_pt, scol, pt_curve, &aux_param, mean_plate_length);
            g_angles = g_angles.add(group);
        }

        if let Some((tll_curve, _angle)) = &curve_set.tll {
            let g_tll = element::Group::new()
                .set("class", "TLL")
                .set("stroke", line_colors.get_or_new("TLL"));
            let group = renderer.cobb(g_tll, scol, tll_curve, &aux_param, mean_plate_length);
            g_angles = g_angles.add(group);
        }
        g_angles
    }
}

struct CurveApex<'a>(&'a ApexSet, &'a scolrs::Corners<ndarray::OwnedRepr<f32>>);
impl<'a> Component for CurveApex<'a> {
    fn render(
        &self,
        _scol: &Scoliosis,
        renderer: &Renderer,
        _label_colors: &mut ColorPaletts<LabelColorsHex>,
        line_colors: &mut ColorPaletts<LineColors>,
    ) -> element::Group {
        let apex_set = &self.0;
        let vert_discs = &self.1 .0;
        let label = "CurveApex";
        let mut g = element::Group::new().set("class", label);

        for apex in [apex_set.pt, apex_set.mt, apex_set.tll]
            .into_iter()
            .flatten()
        {
            let mut corners = vert_discs.index_axis(Axis(0), apex as usize).to_owned();
            // from (tl, tr, bl, br) order to (tl, tr, br, bl)
            corners.swap((2, 0), (3, 0)); // bl.x <-> br.x
            corners.swap((2, 1), (3, 1)); // bl.y <-> br.y
            let polygon = renderer
                .polygon(corners)
                .set("stroke", line_colors.get_or_new(label));
            g = g.add(polygon);
        }
        g
    }
}

struct SpinalLine;
impl Component for SpinalLine {
    fn render(
        &self,
        scol: &Scoliosis,
        renderer: &Renderer,
        _label_colors: &mut ColorPaletts<LabelColorsHex>,
        line_colors: &mut ColorPaletts<LineColors>,
    ) -> element::Group {
        let label = "SpinalLine";
        let g = element::Group::new().set("class", label);
        let centroids = scol.tl_centroids();
        let coefs =
            scolrs::polyfit(centroids.slice(s![.., 1]), centroids.slice(s![.., 0]), 6).unwrap();
        let ys = ndarray::Array::linspace(
            centroids[[0, 1]],
            centroids[[centroids.len_of(Axis(0)) - 1, 1]],
            50,
        );
        let xs = scolrs::polynomial(ys.view(), coefs);
        let spinal_line = renderer
            .polyline(ndarray::stack![Axis(1), xs, ys])
            .set("stroke", line_colors.get_or_new(label));

        g.add(spinal_line)
    }
}

/// center sacral vertical line (CSVL)
struct CSVL<'a>(&'a ApexSet);
impl<'a> Component for CSVL<'a> {
    fn render(
        &self,
        scol: &Scoliosis,
        renderer: &Renderer,
        _label_colors: &mut ColorPaletts<LabelColorsHex>,
        line_colors: &mut ColorPaletts<LineColors>,
    ) -> element::Group {
        let label = "CSVL";
        let line_color = line_colors.get_or_new(label);
        let mut g = element::Group::new()
            .set("class", label)
            .set("stroke", line_color);
        let sacral_corners = scol
            .vertebrae
            .0
            .index_axis(Axis(0), scol.vertebrae.0.len_of(Axis(0)) - 1)
            .to_owned(); // required for reshaping?;
        let sacral_line = renderer.line(sacral_corners.view());
        g = g.add(sacral_line);
        if let Some(tll) = self.0.tll {
            let v_idx = ((tll as u8) / 2 - 1).max(0) as usize; // one level above the tll apex
            let mut vl = sacral_corners
                .into_shape((2, 2, 2))
                .unwrap()
                .to_owned()
                .mean_axis(Axis(1))
                .unwrap();
            let y = scol.vertebrae.0[[v_idx + 1, 0, 1]]; // v_idx+1 because vertebrae include c7
            vl[[0, 1]] = y;
            g = g.add(renderer.line(vl));
        }
        g
    }
}

pub fn cmd(args: RenderArgs) -> Result<()> {
    let render_param = if let Some(filename) = args.config {
        let s = std::fs::read_to_string(&filename)
            .with_context(|| format!("reading file {:?}", filename))?;
        toml::from_str(&s)?
    } else {
        RenderParam::default()
    };

    debug!("Loading {:?}", args.input);
    let s = std::fs::read_to_string(&args.input)
        .with_context(|| format!("reading file {:?}", &args.input))?;
    let data: LabelMeData = s.as_str().try_into()?;
    let orig_wd = std::env::current_dir()?;
    if let Some(parent) = args.input.parent() {
        std::env::set_current_dir(parent)?;
    }
    let mut data = LabelMeDataWImage::try_from(data)?;
    if args.input.parent().is_some() {
        std::env::set_current_dir(orig_wd)?;
    }
    if let Some(resize) = args.resize {
        let resize_param = labelme_rs::ResizeParam::try_from(resize.as_str())?;
        data.resize(&resize_param);
    }
    let svg_size = if let Some(size) = args.size {
        let size_param = labelme_rs::ResizeParam::try_from(size.as_str())?;
        data.data
            .scale(size_param.scale(data.image.width(), data.image.height()));
        size_param.size(data.image.width(), data.image.height())
    } else {
        data.image.dimensions()
    };

    let scol = Scoliosis::try_from(&data.data)?;

    let mut label_colors = if let Some(filename) = args.label_colors {
        ColorPaletts::new(
            labelme_rs::load_label_colors(&filename)
                .with_context(|| format!("reading file {:?}", filename))?,
        )
    } else {
        ColorPaletts::new(labelme_rs::LabelColorsHex::default())
    };
    let mut line_colors = if let Some(filename) = args.line_colors {
        let reader = std::fs::File::open(&filename)
            .with_context(|| format!("reading file {:?}", filename))?;
        ColorPaletts::new(scolrs::load_line_colors(reader)?)
    } else {
        ColorPaletts::new(scolrs::LineColors::new())
    };

    let renderer = Renderer::new(
        render_param.clone(),
        (svg_size.0 as usize, svg_size.1 as usize),
    );
    let mut document = renderer.doc_w_background(&data.image);
    let text_style = "text {font-size: 24px; font-family:sans-serif;}";
    let line_style = format!(
        "line, polyline, polygon {{stroke-width: {}; fill: none}}",
        render_param.line_width
    );
    let style = element::Style::new([text_style, line_style.as_str()].join("\n"));
    document = document.add(style);

    // common components
    for component in ["VertebralLabels", "VertebralPoints"] {
        let group = match component {
            "VertebralLabels" => {
                VertebralLabels {}.render(&scol, &renderer, &mut label_colors, &mut line_colors)
            }
            "VertebralPoints" => {
                VertebralPoints {}.render(&scol, &renderer, &mut label_colors, &mut line_colors)
            }
            _ => panic!("Unknown component"),
        };
        document = document.add(group);
    }

    if let Direction::Frontal = args.direction {
        let (curve_set, apex_set) = if let Some(filename) = args.curve_set {
            let reader = std::fs::File::open(&filename)
                .with_context(|| format!("reading file {:?}", filename))?;
            let cs: CurveInfo = labelme_rs::serde_json::from_reader(reader)?;
            (cs.curves, cs.apices)
        } else {
            let (cs, apexes, _major_curve) = scol.identify_curves();
            (cs, apexes)
        };
        for component in ["Centroids", "CobbAngles", "CurveApex", "SpinalLine", "CSVL"] {
            let group = match component {
                "Centroids" => {
                    Centroids {}.render(&scol, &renderer, &mut label_colors, &mut line_colors)
                }
                "CobbAngles" => FrontalCobbAngles(&curve_set).render(
                    &scol,
                    &renderer,
                    &mut label_colors,
                    &mut line_colors,
                ),
                "CurveApex" => {
                    let vert_discs = scol.tl_vert_disc_corners();
                    CurveApex(&apex_set, &vert_discs).render(
                        &scol,
                        &renderer,
                        &mut label_colors,
                        &mut line_colors,
                    )
                }
                "SpinalLine" => {
                    SpinalLine {}.render(&scol, &renderer, &mut label_colors, &mut line_colors)
                }
                "CSVL" => {
                    CSVL(&apex_set).render(&scol, &renderer, &mut label_colors, &mut line_colors)
                }
                _ => panic!("Unknown component"),
            };
            document = document.add(group);
        }
    } else {
        let aux_param = CobbAux::default();
        let mean_plate_length = mean_plate_length(&scol);
        {
            let mut opposite_param = CobbAux::opposite_default();
            opposite_param.plate_scale = -3.0;
            let label = "ThoracicKyphosis";
            let group = element::Group::new()
                .set("class", label)
                .set("stroke", line_colors.get_or_new(label));
            let sup = VertebralIndex::T2 as usize;
            let inf = VertebralIndex::T12 as usize;
            document = document.add(renderer.cobb(
                group,
                &scol,
                &Curve { sup, inf },
                &opposite_param,
                mean_plate_length,
            ));
        }
        {
            let label = "Mid/LowerThoracicKyphosis";
            let group = element::Group::new()
                .set("class", label)
                .set("stroke", line_colors.get_or_new(label));
            let sup = VertebralIndex::T5 as usize;
            let inf = VertebralIndex::T12 as usize;
            document = document.add(renderer.cobb(
                group,
                &scol,
                &Curve { sup, inf },
                &aux_param,
                mean_plate_length,
            ));
        }
        {
            let label = "ProximalThoracicKyphosis";
            let group = element::Group::new()
                .set("class", label)
                .set("stroke", line_colors.get_or_new(label));
            let sup = VertebralIndex::T2 as usize;
            let inf = VertebralIndex::T5 as usize;
            document = document.add(renderer.cobb(
                group,
                &scol,
                &Curve { sup, inf },
                &aux_param,
                mean_plate_length,
            ));
        }
        {
            // required for structural/non-structural analysis for thoracic and tl/l curves
            let label = "T10L2";
            let group = element::Group::new()
                .set("class", label)
                .set("stroke", line_colors.get_or_new(label));
            let sup = VertebralIndex::T10 as usize;
            let inf = VertebralIndex::L2 as usize;
            document = document.add(renderer.cobb(
                group,
                &scol,
                &Curve { sup, inf },
                &aux_param,
                mean_plate_length,
            ));
        }
    };

    std::fs::write(args.output, document.to_string())?;
    Ok(())
}
