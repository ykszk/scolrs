use clap::{Args, Parser, Subcommand, ValueEnum, ValueHint};
use clap_complete::Shell;
use std::path::PathBuf;

#[derive(Parser)]
#[clap(name=env!("CARGO_BIN_NAME"), author, version, about, long_about = None)]
pub struct Cli {
    #[clap(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Generate shell completions
    Complete(CompleteArgs),
    /// Create SVG
    Svg(SvgArgs),
    /// Create SVGs from ndjson
    SvgNdjson(SvgNdjsonArgs),
    /// Measure scoliotic parameters
    Measure(MeasureArgs),
    /// Determine curves
    Curve(CurveArgs),
    /// Lenke classification
    Lenke(LenkeArgs),
    /// List available measurements
    List(ListArgs),
}

#[derive(Parser)]
pub struct CompleteArgs {
    /// Shell to generate completions for
    pub shell: Shell,
}

#[derive(ValueEnum, Debug, Copy, Clone, Default)]
pub enum Plane {
    /// AP, PA and frontal view
    #[default]
    Coronal,
    /// Lateral view
    Sagittal,
}

#[derive(Args, Debug, Clone, Default)]
pub struct SvgArgsCommon {
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
    /// Measurements to draw. By default, all measurements are drawn. Use `--list` to see all measurements. Comma separated list
    #[clap(short, long, value_delimiter = ',', value_hint = ValueHint::Other)]
    pub measures: Vec<String>,
    /// Hide measurements. Use `--list` to see all measurements. Comma separated list
    #[clap(long, value_delimiter = ',', value_hint = ValueHint::Other)]
    pub hide: Vec<String>,
    /// Input data format is labelme instead of native format
    #[clap(long)]
    pub labelme: bool,
}

#[derive(Parser, Debug, Clone, Default)]
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
    #[clap(flatten)]
    pub svg_args: SvgArgsCommon,
}

#[derive(Parser, Debug, Clone, Default)]
pub struct SvgNdjsonArgs {
    /// Input labelme json or ndjson filename
    #[arg(value_hint = ValueHint::FilePath)]
    pub input: PathBuf,
    /// Output directory
    #[arg(value_hint = ValueHint::DirPath)]
    pub output: PathBuf,
    /// Use specified curves from ndjson instaed of calculating from the points
    #[clap(long)]
    pub curve_set: Option<PathBuf>,
    #[clap(flatten)]
    pub svg_args: SvgArgsCommon,
}

#[derive(Parser, Debug)]
pub struct MeasureArgs {
    /// Input labelme json filename or ndjson. Specify '-' for stdin with ndjson format
    #[arg(value_hint = ValueHint::FilePath)]
    pub input: PathBuf,
    /// Use specified curves instaed of calculating from the points
    #[clap(long)]
    pub curve_set: Option<PathBuf>,
    /// Scan direction
    #[clap(short, long, default_value = "coronal")]
    pub direction: Plane,
    /// Measurements to draw. By default, all measurements are drawn
    #[clap(short, long, value_delimiter = ',', value_hint = ValueHint::Other)]
    pub measures: Vec<String>,
    /// Input data format is labelme instead of native format
    #[clap(long)]
    pub labelme: bool,
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

#[derive(Parser, Debug)]
pub struct ListArgs {
    /// List all drawable components instead of measurements
    #[clap(long)]
    pub drawable: bool,
}
