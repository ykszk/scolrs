use clap::{Parser, Subcommand, ValueEnum};
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
    /// Determine curves
    Curve(CurveArgs),
    /// Lenke classification
    Lenke(LenkeArgs),
}

#[derive(ValueEnum, Debug, Copy, Clone)]
pub enum Direction {
    Frontal,
    Lateral,
}

#[derive(Parser, Debug)]
pub struct SvgArgs {
    /// Input labelme json filename
    pub input: PathBuf,
    /// Output svg filename
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
    #[clap(short, long, default_value = "frontal")]
    pub direction: Direction,
    /// Resize x-ray image. Specify in imagemagick's `-resize`-like format
    #[clap(long)]
    pub resize: Option<String>,
    /// Output image size. Aspect ratio will be adjusted based on x-ray image size. Specify in imagemagick's `-resize`-like format
    #[clap(long)]
    pub size: Option<String>,
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
