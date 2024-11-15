use clap::{Args, Parser, Subcommand, ValueEnum, ValueHint};
use clap_complete::Shell;
use scolrs::{CoronalMeasure, SagittalMeasure};
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

#[derive(Subcommand, Debug, Clone)]
pub enum SvgSubCommands {
    /// Coronal view
    Coronal(SvgSubCoronalArgs),
    /// Sagittal view
    Sagittal(SvgSubSagittallArgs),
}

impl Default for SvgSubCommands {
    fn default() -> Self {
        SvgSubCommands::Coronal(SvgSubCoronalArgs::default())
    }
}

#[derive(Args, Debug, Clone, Default)]
pub struct SvgSubCoronalArgs {
    /// Measurements to draw. By default, all measurements are drawn. Comma separated list
    #[clap(short, long, value_delimiter = ',')]
    pub measures: Option<Vec<CoronalMeasure>>,
    /// Hide measurements. Comma separated list
    #[clap(long, value_delimiter = ',')]
    pub hide: Vec<CoronalMeasure>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct SvgSubSagittallArgs {
    /// Measurements to draw. By default, all measurements are drawn. Comma separated list
    #[clap(short, long, value_delimiter = ',')]
    pub measures: Option<Vec<SagittalMeasure>>,
    /// Hide measurements. Comma separated list
    #[clap(long, value_delimiter = ',')]
    pub hide: Vec<SagittalMeasure>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct SvgArgsCommon {
    /// Config file in toml
    #[clap(long, value_hint = ValueHint::FilePath)]
    pub config: Option<PathBuf>,
    /// Label colors in yaml
    #[clap(long, value_hint = ValueHint::FilePath)]
    pub label_colors: Option<PathBuf>,
    /// Line colors in csv with `label` and `color` columns
    #[clap(long, value_hint = ValueHint::FilePath)]
    pub line_colors: Option<PathBuf>,
    /// Resize x-ray image. Specify in imagemagick's `-resize`-like format
    #[clap(long)]
    pub resize: Option<String>,
    /// Output image size. Aspect ratio will be adjusted based on x-ray image size. Specify in imagemagick's `-resize`-like format
    #[clap(long)]
    pub size: Option<String>,
    /// Input data format is labelme instead of native format
    #[clap(long)]
    pub labelme: bool,
    #[clap(subcommand)]
    pub subcommand: SvgSubCommands,
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
    #[clap(long, value_hint = ValueHint::FilePath)]
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
    #[clap(long, value_hint = ValueHint::FilePath)]
    pub curve_set: Option<PathBuf>,
    #[clap(flatten)]
    pub svg_args: SvgArgsCommon,
}

#[derive(Subcommand, Debug, Clone)]
pub enum MeasureSubCommands {
    /// Coronal view
    Coronal(MeasureSubCoronalArgs),
    /// Sagittal view
    Sagittal(MeasureSubSagittallArgs),
}

impl Default for MeasureSubCommands {
    fn default() -> Self {
        MeasureSubCommands::Coronal(MeasureSubCoronalArgs::default())
    }
}

#[derive(Args, Debug, Clone, Default)]
pub struct MeasureSubCoronalArgs {
    /// Measurements to draw. By default, all measurements are drawn. Comma separated list
    #[clap(short, long, value_delimiter = ',')]
    pub measures: Option<Vec<CoronalMeasure>>,
    /// Hide measurements. Comma separated list
    #[clap(long, value_delimiter = ',')]
    pub hide: Vec<CoronalMeasure>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct MeasureSubSagittallArgs {
    /// Measurements to draw. By default, all measurements are drawn. Comma separated list
    #[clap(short, long, value_delimiter = ',')]
    pub measures: Option<Vec<SagittalMeasure>>,
    /// Hide measurements. Comma separated list
    #[clap(long, value_delimiter = ',')]
    pub hide: Vec<SagittalMeasure>,
}

#[derive(Parser, Debug)]
pub struct MeasureArgs {
    /// Input labelme json filename or ndjson. Specify '-' for stdin with ndjson format
    #[arg(value_hint = ValueHint::FilePath)]
    pub input: PathBuf,
    /// Use specified curves instaed of calculating from the points
    #[clap(long, value_hint = ValueHint::FilePath)]
    pub curve_set: Option<PathBuf>,
    /// Input data format is labelme instead of native format
    #[clap(long)]
    pub labelme: bool,
    #[clap(subcommand)]
    pub subcommand: MeasureSubCommands,
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
    #[clap(short, long, value_hint = ValueHint::FilePath)]
    pub coronal: PathBuf,
    /// sagittal
    #[clap(short, long, value_hint = ValueHint::FilePath)]
    pub sagittal: Option<PathBuf>,
    /// right bend
    #[clap(short, long, value_hint = ValueHint::FilePath)]
    pub right: Option<PathBuf>,
    /// left bend
    #[clap(short, long, value_hint = ValueHint::FilePath)]
    pub left: Option<PathBuf>,
}

#[derive(Parser, Debug)]
pub struct ListArgs {
    /// List all drawable components instead of measurements
    #[clap(long)]
    pub drawable: bool,
}
