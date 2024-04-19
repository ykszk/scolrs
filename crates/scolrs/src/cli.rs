use clap::{Parser, Subcommand, ValueEnum, ValueHint};
use scolrs::SagittalMeasure;
use std::path::PathBuf;

#[derive(Parser)]
#[clap(name=env!("CARGO_CRATE_NAME"), author, version, about, long_about = None)]
pub struct Cli {
    #[clap(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Create SVG
    Svg(SvgArgs),
    /// Measure scoliotic parameters
    Measure(MeasureArgs),
    /// Determine curves
    Curve(CurveArgs),
    /// Lenke classification
    Lenke(LenkeArgs),
}

#[derive(ValueEnum, Debug, Copy, Clone)]
pub enum Plane {
    Coronal,
    Sagittal,
}

#[derive(Parser, Debug)]
pub struct SvgArgs {
    /// Input labelme json filename
    #[arg(value_hint = ValueHint::FilePath)]
    pub input: PathBuf,
    /// Output svg filename
    #[arg(value_hint = ValueHint::FilePath)]
    pub output: PathBuf,
    /// Use specified curves instaed of calculating from the points
    #[clap(long)]
    pub curve_set: Option<PathBuf>,
    /// Config file in toml
    #[clap(long)]
    pub config: Option<PathBuf>,
    /// Label colors in yaml
    #[clap(long)]
    pub label_colors: Option<PathBuf>,
    /// Line colors in csv with `label` and `color` columns
    #[clap(long)]
    pub line_colors: Option<PathBuf>,
    /// Scan direction
    #[clap(short, long, default_value = "coronal")]
    pub direction: Plane,
    /// Resize x-ray image. Specify in imagemagick's `-resize`-like format
    #[clap(long)]
    pub resize: Option<String>,
    /// Output image size. Aspect ratio will be adjusted based on x-ray image size. Specify in imagemagick's `-resize`-like format
    #[clap(long)]
    pub size: Option<String>,
    /// Measurements to draw. By default, all measurements are drawn
    #[clap(long)]
    pub measures: Vec<SagittalMeasure>,
    /// Hide measurements
    #[clap(long)]
    pub hide: Vec<SagittalMeasure>,
}

#[derive(Parser, Debug)]
pub struct MeasureArgs {
    /// Input labelme json filename
    #[arg(value_hint = ValueHint::FilePath)]
    pub input: PathBuf,
    /// Use specified curves instaed of calculating from the points
    #[clap(long)]
    pub curve_set: Option<PathBuf>,
    /// Scan direction
    #[clap(short, long, default_value = "coronal")]
    pub direction: Plane,
    /// Measurements to draw. By default, all measurements are drawn
    #[clap(long)]
    pub measures: Vec<SagittalMeasure>,
}

#[derive(Parser, Debug)]
pub struct CurveArgs {
    /// Input labelme json/ndjson filename. Specify '-' for stdin with ndjson format
    pub input: PathBuf,
    /// Output all curves
    #[clap(short, long)]
    pub all: bool,
}

#[derive(Parser, Debug)]
pub struct LenkeArgs {
    /// coronal
    #[clap(short, long)]
    pub coronal: PathBuf,
    /// sagittal
    #[clap(short, long)]
    pub sagittal: Option<PathBuf>,
    /// right bend
    #[clap(short, long)]
    pub right: Option<PathBuf>,
    /// left bend
    #[clap(short, long)]
    pub left: Option<PathBuf>,
}
