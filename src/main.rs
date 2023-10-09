use anyhow::Result;
use clap::Parser;
use labelme_rs::{
    image::{DynamicImage, GenericImageView},
    LabelMeData, LabelMeDataWImage,
};
use scolrs::{Centroids, Corners, VertebraeTL};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use svg::node::element;

/// LabelMeData to Vertebrae
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input labelme json filename
    input: PathBuf,
    /// Output svg filename
    output: PathBuf,
    /// Config file in toml
    #[clap(long)]
    config: Option<PathBuf>,
    /// Label colors in yaml
    #[clap(long)]
    label_colors: Option<PathBuf>,
    #[command(flatten)]
    render_param: RenderParam,
}

fn default_radius() -> usize {
    2
}
fn default_line_width() -> usize {
    2
}

fn default_text_stroke() -> String {
    "black".into()
}

fn default_text_fill() -> String {
    "white".into()
}

#[derive(Serialize, Deserialize, Parser, Debug, Clone)]
pub struct RenderParam {
    /// Point radius
    #[clap(long, default_value = "2")]
    #[serde(default = "default_radius")]
    radius: usize,
    /// Line width
    #[clap(long, default_value = "2")]
    #[serde(default = "default_line_width")]
    line_width: usize,

    /// `stroke` for texts
    #[clap(long, default_value = "black")]
    #[serde(default = "default_text_stroke")]
    text_stroke: String,
    /// `fill` for texts
    #[clap(long, default_value = "white")]
    #[serde(default = "default_text_fill")]
    text_fill: String,
}

struct Renderer {
    param: RenderParam,
}

impl Renderer {
    fn new(param: RenderParam) -> Self {
        Self { param }
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
const VERTEBRAL_LABELS: [&str; 18] = [
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3",
    "L4", "L5", "L6",
];

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(filename) = args.config {
        let s = std::fs::read_to_string(filename)?;
        let config: RenderParam = toml::from_str(&s)?;
        println!("{:?}", config);
    }
    let s = std::fs::read_to_string(&args.input)?;
    let data: LabelMeData = s.as_str().try_into()?;
    let orig_wd = std::env::current_dir()?;
    if let Some(parent) = args.input.parent() {
        std::env::set_current_dir(parent)?;
    }
    let data = LabelMeDataWImage::try_from(data)?;
    if args.input.parent().is_some() {
        std::env::set_current_dir(&orig_wd)?;
    }
    let v = VertebraeTL::try_from(&data.data)?;
    let corners = Corners::from(v);
    let centroids = Centroids::try_from(&corners)?;
    let discs = corners.between();
    let disc_centroids = Centroids::try_from(&discs)?;

    let label_colors = if let Some(filename) = args.label_colors {
        labelme_rs::load_label_colors(&filename)?
    } else {
        labelme_rs::LabelColorsHex::default()
    };
    let mut color_cycler = labelme_rs::ColorCycler::new();

    let render_param = args.render_param;
    let renderer = Renderer::new(render_param.clone());
    let mut document = renderer.doc_w_background(&data.image);
    let mut g_corners = element::Group::new();
    for (i_label, label) in scolrs::CORNER_LABELS.iter().enumerate() {
        let color = label_colors
            .get(*label)
            .map_or_else(|| color_cycler.cycle(), |s| s.as_str());
        let mut sub_group = element::Group::new()
            .set("class", format!("{} {}", CLS_POINT, label))
            .set("fill", color);
        let points = corners.0.index_axis(ndarray::Axis(1), i_label);
        for point in points.axis_iter(ndarray::Axis(0)) {
            let p = renderer.point(point);
            sub_group = sub_group.add(p);
        }
        g_corners = g_corners.add(sub_group);
    }
    document = document.add(g_corners);

    let mut g_vert_labels = element::Group::new()
        .set("text-anchor", "middle")
        .set("dominant-baseline", "central")
        .set("stroke", render_param.text_stroke)
        .set("stroke_width", "1px")
        .set("fill", render_param.text_fill)
        .set("style", "font-size: 24px; font-family:sans-serif");
    for (coords, label) in std::iter::zip(
        centroids.0.axis_iter(ndarray::Axis(0)),
        VERTEBRAL_LABELS.into_iter(),
    ) {
        let t = renderer.text(label, coords);
        g_vert_labels = g_vert_labels.add(t);
    }
    document = document.add(g_vert_labels);

    let color = label_colors
        .get("centroid")
        .map_or_else(|| color_cycler.cycle(), |s| s.as_str());
    let mut g_centroids = element::Group::new()
        .set("class", "centroids")
        .set("fill", color);
    for point in centroids.0.axis_iter(ndarray::Axis(0)) {
        let p = renderer.point(point);
        g_centroids = g_centroids.add(p);
    }
    document = document.add(g_centroids);

    std::fs::write(args.output, document.to_string())?;
    Ok(())
}
