use std::ops::{AddAssign, SubAssign};

use anyhow::{Context, Result};
use labelme_rs::{image::GenericImageView, LabelColorsHex, LabelMeData, LabelMeDataWImage};
use ndarray::{s, Axis};
// use scolrs::{Centroids, Corners, VertebraeTL};
use log::debug;
use scolrs::{LineColors, RenderParam, VERTEBRAL_LABELS};
use svg::node::element;

use crate::cli::RenderArgs;

struct Renderer {
    param: RenderParam,
    size: (usize, usize),
}

fn squared_distance<S>(
    p1: ndarray::ArrayBase<S, ndarray::Ix1>,
    p2: ndarray::ArrayBase<S, ndarray::Ix1>,
) -> f32
where
    S: ndarray::Data<Elem = f32>,
{
    (&p1 - &p2).mapv(|a| a * a).sum()
}

fn distanced_pair3<S>(
    p1: ndarray::ArrayBase<S, ndarray::Ix1>,
    p2: ndarray::ArrayBase<S, ndarray::Ix1>,
    p3: ndarray::ArrayBase<S, ndarray::Ix1>,
) -> (
    ndarray::ArrayBase<S, ndarray::Ix1>,
    ndarray::ArrayBase<S, ndarray::Ix1>,
)
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

impl<S, D> JoinWith for ndarray::ArrayBase<S, D>
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

impl Renderer {
    fn new(param: RenderParam, size: (usize, usize)) -> Self {
        Self { param, size }
    }

    fn point<S>(&self, point: ndarray::ArrayBase<S, ndarray::Ix1>) -> svg::node::element::Circle
    where
        S: ndarray::Data<Elem = f32>,
    {
        element::Circle::new()
            .set("cx", point[0])
            .set("cy", point[1])
            .set("r", self.param.radius)
    }

    fn line<S>(&self, start_end: ndarray::ArrayBase<S, ndarray::Ix2>) -> svg::node::element::Line
    where
        S: ndarray::Data<Elem = f32>,
    {
        element::Line::new()
            .set("x1", start_end[[0, 0]])
            .set("y1", start_end[[0, 1]])
            .set("x2", start_end[[1, 0]])
            .set("y2", start_end[[1, 1]])
    }

    fn polyline<S>(
        &self,
        points: ndarray::ArrayBase<S, ndarray::Ix2>,
    ) -> svg::node::element::Polyline
    where
        S: ndarray::Data<Elem = f32>,
    {
        let s = points.join(" ");
        element::Polyline::new().set("points", s)
    }

    fn polygon<S>(&self, points: ndarray::ArrayBase<S, ndarray::Ix2>) -> svg::node::element::Polygon
    where
        S: ndarray::Data<Elem = f32>,
    {
        let s = points.join(" ");
        element::Polygon::new().set("points", s)
    }

    fn text<S>(
        &self,
        text: &str,
        coords: ndarray::ArrayBase<S, ndarray::Ix1>,
    ) -> svg::node::element::Text
    where
        S: ndarray::Data<Elem = f32>,
    {
        element::Text::new()
            .set("x", coords[0])
            .set("y", coords[1])
            .add(svg::node::Text::new(text))
    }

    fn plate_end<S, T>(
        plate: ndarray::ArrayBase<S, ndarray::Ix2>,
        point: ndarray::ArrayBase<T, ndarray::Ix1>,
    ) -> usize
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
        point: ndarray::ArrayBase<S, ndarray::Ix1>,
        origin: ndarray::ArrayBase<T, ndarray::Ix1>,
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
        scol: &scolrs::Scoliosis,
        curve: &scolrs::Curve,
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
            if is_inside {
                // draw intersection point
                for plate in [sup_plate, inf_plate] {
                    let arr_int = ndarray::arr1(&[intersection.x, intersection.y]);
                    let i = Self::plate_end(plate, arr_int);
                    let line = self.line(ndarray::arr2(&[
                        [plate[[i, 0]], plate[[i, 1]]],
                        [intersection.x, intersection.y],
                    ]));
                    group = group.add(line);
                }
                let text = self
                    .text(
                        format!("{:.1}°", angle.to_degrees()).as_str(),
                        ndarray::arr1(&[intersection.x, intersection.y]),
                    )
                    .set("stroke", self.param.text_stroke.as_str())
                    .set("stroke-width", self.param.text_stroke_width)
                    .set("fill", self.param.text_fill.as_str());

                group = group.add(text);
            } else {
                // draw aux lines and its intersection
                let arr_int = ndarray::arr1(&[intersection.x, intersection.y]);
                let i = Self::plate_end(sup_plate, arr_int.view());
                let d = &sup_plate.slice(s![1 - i, ..]) - &sup_plate.slice(s![i, ..]);
                let aux_scale = 4.0;
                let aux_on_sup: ndarray::Array1<_> = &sup_plate.slice(s![i, ..]) + aux_scale * &d;
                let aux_point = Self::rotate_around(aux_on_sup.view(), arr_int.view(), angle / 2.0);

                for plate in [sup_plate, inf_plate] {
                    let i = Self::plate_end(sup_plate, arr_int.view());
                    let d = &plate.slice(s![1 - i, ..]) - &plate.slice(s![i, ..]);
                    let d_aux = &aux_point - &plate.slice(s![i, ..]);
                    let t = d_aux.dot(&d) / d.mapv(|a| a * a).sum();
                    let projed_aux = &plate.slice(s![i, ..]) + t * &d;
                    let (p1, p2) = distanced_pair3(
                        plate.slice(s![0, ..]),
                        plate.slice(s![1, ..]),
                        projed_aux.view(),
                    );
                    let line = self.line(ndarray::stack![Axis(0), p1, p2]);
                    group = group.add(line);
                    let pa_a = &aux_point - &projed_aux;
                    let line = self.line(ndarray::stack![
                        Axis(0),
                        1.4 * pa_a + &projed_aux,
                        projed_aux
                    ]);
                    group = group.add(line);
                }
                let text = self
                    .text(format!("{:.1}°", angle.to_degrees()).as_str(), aux_point)
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
                line.slice_mut(s![1, ..]).sub_assign(&-d);
                let line = self.line(plate);
                group = group.add(line);
            }
        }
        group
    }

    fn doc_w_background(&self, image: &labelme_rs::image::DynamicImage) -> svg::Document {
        let (image_width, image_height) = image.dimensions();
        let mut document = svg::Document::new()
            .set("width", image_width)
            .set("height", image_height)
            .set("viewBox", (0i64, 0i64, image_width, image_height))
            .set("xmlns:xlink", "http://www.w3.org/1999/xlink");
        let b64 = format!(
            "data:image/jpeg;base64,{}",
            labelme_rs::img2base64(image, labelme_rs::image::ImageOutputFormat::Jpeg(75))
        );
        let bg = element::Image::new()
            .set("x", 0i64)
            .set("y", 0i64)
            .set("width", image_width)
            .set("height", image_height)
            .set("xlink:href", b64);
        document = document.add(bg);
        document
    }
}

const CLS_POINT: &str = "Points";

fn plate2line<S>(plate: ndarray::ArrayBase<S, ndarray::Ix2>) -> lyon_geom::Line<f32>
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
        self.color_map.try_get(&key).map_or_else(
            || {
                debug!("New color generated for {}", key);
                self.color_cycler.cycle()
            },
            |s| s.as_str(),
        )
    }
}

pub fn cmd(args: RenderArgs) -> Result<()> {
    let render_param = if let Some(filename) = args.config {
        let s = std::fs::read_to_string(filename)?;
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
    let data = LabelMeDataWImage::try_from(data)?;
    if args.input.parent().is_some() {
        std::env::set_current_dir(&orig_wd)?;
    }
    let scol = scolrs::Scoliosis::try_from(&data.data)?;
    let corners = scol.tl_corners();
    let centroids = scol.tl_centroids();

    let mut label_colors = if let Some(filename) = args.label_colors {
        ColorPaletts::new(labelme_rs::load_label_colors(&filename)?)
    } else {
        ColorPaletts::new(labelme_rs::LabelColorsHex::default())
    };
    let mut line_colors = if let Some(filename) = args.line_colors {
        let reader = std::fs::File::open(filename)?;
        ColorPaletts::new(scolrs::load_line_colors(reader)?)
    } else {
        ColorPaletts::new(scolrs::LineColors::new())
    };

    let renderer = Renderer::new(
        render_param.clone(),
        (data.image.width() as usize, data.image.height() as usize),
    );
    let mut document = renderer.doc_w_background(&data.image);
    let text_style = "text {font-size: 24px; font-family:sans-serif;}";
    let line_style = format!(
        "line, polyline, polygon {{stroke-width: {}; fill: none}}",
        render_param.line_width
    );
    let style = element::Style::new([text_style, line_style.as_str()].join("\n"));
    document = document.add(style);
    let mut g_corners = element::Group::new();
    for (i_label, &label) in scolrs::CORNER_LABELS.iter().enumerate() {
        let color = label_colors.get_or_new(label);
        let mut sub_group = element::Group::new()
            .set("class", format!("{CLS_POINT} {label}"))
            .set("fill", color);
        let points = corners.0.index_axis(Axis(1), i_label);
        for point in points.axis_iter(Axis(0)) {
            let p = renderer.point(point);
            sub_group = sub_group.add(p);
        }
        g_corners = g_corners.add(sub_group);
    }
    document = document.add(g_corners);

    let mut g_vert_labels = element::Group::new()
        .set("text-anchor", "middle")
        .set("dominant-baseline", "central")
        .set("stroke", render_param.text_stroke.as_str())
        .set("stroke-width", render_param.text_stroke_width)
        .set("fill", render_param.text_fill.as_str());
    for (coords, label) in
        std::iter::zip(centroids.axis_iter(Axis(0)), VERTEBRAL_LABELS.into_iter())
    {
        let t = renderer.text(label, coords);
        g_vert_labels = g_vert_labels.add(t);
    }
    document = document.add(g_vert_labels);

    let label = "Centroid";
    let color = label_colors.get_or_new(label);
    let mut g_centroids = element::Group::new().set("class", label).set("fill", color);
    for point in centroids.axis_iter(Axis(0)) {
        let p = renderer.point(point);
        g_centroids = g_centroids.add(p);
    }

    let (curve_set, apex_set) = scol.find_curve_set();
    debug!("Curve set:{:?}", curve_set);
    if let Some(largest_curve) = curve_set.mt {
        let g_mt = element::Group::new()
            .set("class", "MT")
            .set("stroke", line_colors.get_or_new("MT"));
        let group = renderer.cobb(g_mt, &scol, &largest_curve);
        document = document.add(group);
    }

    if let Some(pt_curve) = curve_set.pt {
        let g_pt = element::Group::new()
            .set("class", "PT")
            .set("stroke", line_colors.get_or_new("PT"));
        let group = renderer.cobb(g_pt, &scol, &pt_curve);
        document = document.add(group);
    }

    if let Some(tll_curve) = curve_set.tll {
        let g_tll = element::Group::new()
            .set("class", "TLL")
            .set("stroke", line_colors.get_or_new("TLL"));
        let group = renderer.cobb(g_tll, &scol, &tll_curve);
        document = document.add(group);
    }
    let coefs = scolrs::polyfit(centroids.slice(s![.., 1]), centroids.slice(s![.., 0]), 6).unwrap();
    let ys = ndarray::Array::linspace(
        centroids[[0, 1]],
        centroids[[centroids.len_of(Axis(0)) - 1, 1]],
        50,
    );
    let xs = scolrs::polynomial(ys.view(), coefs);
    let spinal_line = renderer
        .polyline(ndarray::stack![Axis(1), xs, ys])
        .set("stroke", "orange");
    let vert_discs = scol.tl_vert_disc_corners().0;
    for apex in [apex_set.pt, apex_set.mt, apex_set.tll]
        .into_iter()
        .flatten()
    {
        let mut corners = vert_discs.index_axis(Axis(0), apex as usize).to_owned();
        corners.swap((2, 0), (3, 0)); // bl.x <-> br.x
        corners.swap((2, 1), (3, 1)); // bl.y <-> br.y
        let polygon = renderer.polygon(corners).set("stroke", "orange");
        g_centroids = g_centroids.add(polygon);
    }

    g_centroids = g_centroids.add(spinal_line);
    document = document.add(g_centroids);

    std::fs::write(args.output, document.to_string())?;
    Ok(())
}
