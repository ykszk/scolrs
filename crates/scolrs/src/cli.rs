use clap::{Args, Parser, Subcommand, ValueEnum, ValueHint};
use clap_complete::Shell;
use scolrs::{
    head_neck::{NeckLateralDraw, NeckLateralMeasure},
    CoronalDraw, CoronalMeasure, ImplantDraw, SagittalDraw, SagittalMeasure,
};
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
    /// Compile SVGs into a catalog HTML
    Catalog(CatalogArgs),
    /// Convert SVGs to HTML
    Html(HtmlArgs),
    /// Convert data format
    Conv(ConvArgs),
    /// Evaluate implant
    Implant(ImplantArgs),
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
    /// Coronal view with implants
    CoronalImplant(SvgSubImplantArgs),
    /// Sagittal view
    Sagittal(SvgSubSagittallArgs),
    /// Neck lateral view
    Neck(SvgSubNeckArgs),
}

impl Default for SvgSubCommands {
    fn default() -> Self {
        SvgSubCommands::Coronal(SvgSubCoronalArgs::default())
    }
}

#[derive(ValueEnum, Debug, Copy, Clone)]
pub enum CoronalDrawGroup {
    CobbExtra,
    NonCobb,
}

impl From<&CoronalDrawGroup> for Vec<CoronalDraw> {
    fn from(group: &CoronalDrawGroup) -> Self {
        use CoronalDraw::*;
        match group {
            CoronalDrawGroup::CobbExtra => vec![Centroids, SpinalLine, CurveApex],
            CoronalDrawGroup::NonCobb => vec![
                CSVL,
                T1TiltAngle,
                CoronalBalance,
                ClavicleAngle,
                ShoulderHeight,
                PelvicObliquity,
                SacralObliquity,
                LegLengthDiscrepancy,
            ],
        }
    }
}

#[derive(Args, Debug, Clone, Default)]
pub struct SvgSubCoronalArgs {
    /// Measurements to draw. By default, all measurements are drawn. Comma separated list
    #[clap(short, long, value_delimiter = ',')]
    pub measures: Option<Vec<CoronalDraw>>,
    /// Hide measurements. Comma separated list
    #[clap(long, value_delimiter = ',')]
    pub hide: Vec<CoronalDraw>,
    /// Hide group of measurements. Comma separated list
    #[clap(long, value_delimiter = ',')]
    pub hide_group: Vec<CoronalDrawGroup>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct SvgSubImplantArgs {
    /// Measurements to draw. By default, all measurements are drawn. Comma separated list
    #[clap(short, long, value_delimiter = ',')]
    pub measures: Option<Vec<ImplantDraw>>,
    /// Hide measurements. Comma separated list
    #[clap(long, value_delimiter = ',')]
    pub hide: Vec<ImplantDraw>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct SvgSubSagittallArgs {
    /// Measurements to draw. By default, all measurements are drawn. Comma separated list
    #[clap(short, long, value_delimiter = ',')]
    pub measures: Option<Vec<SagittalDraw>>,
    /// Hide measurements. Comma separated list
    #[clap(long, value_delimiter = ',')]
    pub hide: Vec<SagittalDraw>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct SvgSubNeckArgs {
    /// Measurements to draw. By default, all measurements are drawn. Comma separated list
    #[clap(short, long, value_delimiter = ',')]
    pub measures: Option<Vec<NeckLateralDraw>>,
    /// Hide measurements. Comma separated list
    #[clap(long, value_delimiter = ',')]
    pub hide: Vec<NeckLateralDraw>,
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
    /// Load spacing from dicom
    #[clap(long)]
    pub pull_spacing: bool,
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
    /// Maximum number of jobs to run in parallel
    #[clap(short, long)]
    pub jobs: Option<usize>,
    #[clap(flatten)]
    pub svg_args: SvgArgsCommon,
}

#[derive(Subcommand, Debug, Clone)]
pub enum MeasureSubCommands {
    /// Coronal view
    Coronal(MeasureSubCoronalArgs),
    /// Sagittal view
    Sagittal(MeasureSubSagittallArgs),
    /// Neck lateral view
    Neck(MeasureSubNeckArgs),
}

impl Default for MeasureSubCommands {
    fn default() -> Self {
        MeasureSubCommands::Coronal(MeasureSubCoronalArgs::default())
    }
}

#[derive(Args, Debug, Clone, Default)]
pub struct MeasureSubCoronalArgs {
    /// Measurements. By default, all measurements are drawn. Comma separated list
    #[clap(short, long, value_delimiter = ',')]
    pub measures: Option<Vec<CoronalMeasure>>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct MeasureSubSagittallArgs {
    /// Measurements. By default, all measurements are drawn. Comma separated list
    #[clap(short, long, value_delimiter = ',')]
    pub measures: Option<Vec<SagittalMeasure>>,
}

#[derive(Args, Debug, Clone, Default)]
pub struct MeasureSubNeckArgs {
    /// Measurements. By default, all measurements are drawn. Comma separated list
    #[clap(short, long, value_delimiter = ',')]
    pub measures: Option<Vec<NeckLateralMeasure>>,
}

#[derive(Parser, Debug, Default)]
pub struct MeasureArgs {
    /// Input labelme json filename or ndjson. Specify '-' for stdin with ndjson format
    #[arg(value_hint = ValueHint::FilePath)]
    pub input: PathBuf,
    /// Output json/ndjson filename
    #[arg(value_hint = ValueHint::FilePath)]
    pub output: Option<PathBuf>,
    /// Input data format is labelme instead of native format
    #[clap(long)]
    pub labelme: bool,
    #[clap(subcommand)]
    pub subcommand: MeasureSubCommands,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq)]
pub enum CurveSetAlgorithm {
    /// Weighted sum of the scores. Use --weights to specify the weights
    Score,
    /// Algorithm based on the position of the curve apex
    Apex,
}

#[derive(Parser, Debug)]
pub struct CurveArgs {
    /// Input labelme json/ndjson filename. Specify '-' for stdin with ndjson format
    pub input: PathBuf,
    /// Output all curves
    #[clap(short, long)]
    pub all: bool,
    /// Input data format is labelme instead of native format
    #[clap(long)]
    pub labelme: bool,
    /// Curve set selection algorithm
    #[clap(long, default_value = "score")]
    pub algorithm: CurveSetAlgorithm,
    /// Weights for the score algorithm
    #[clap(
        short,
        long,
        allow_hyphen_values = true,
        num_args = 9,
        value_name = "x"
    )]
    pub weights: Option<Vec<f64>>,
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

#[derive(Parser, Debug)]
pub struct CatalogArgs {
    /// Input svg (or html) file path or svg containing directory
    #[arg(value_hint = ValueHint::AnyPath, required = true)]
    pub input: Vec<PathBuf>,
    /// Output html file
    #[arg(value_hint = ValueHint::FilePath)]
    pub output: PathBuf,
    /// `title` tag in html
    #[clap(short, long)]
    pub title: Option<String>,
    /// Selector(s) for the svg elements
    #[clap(long, value_delimiter = ',', default_value = "g.Component", value_hint = ValueHint::Other)]
    pub selector: Vec<String>,
    /// Maximum number of jobs to run in parallel
    #[clap(short, long)]
    pub jobs: Option<usize>,
}

#[derive(Parser, Debug, Clone)]
pub struct HtmlArgs {
    /// Input svg file
    #[arg(value_hint = ValueHint::FilePath)]
    pub input: PathBuf,
    /// Output html file
    #[arg(value_hint = ValueHint::FilePath)]
    pub output: Option<PathBuf>,
    /// Selector(s) for the svg elements
    #[clap(long, value_delimiter = ',', default_value = "g.Component", value_hint = ValueHint::Other)]
    pub selector: Vec<String>,
    /// Title of the html
    #[clap(short, long)]
    pub title: Option<String>,
}

#[derive(ValueEnum, Debug, Copy, Clone, PartialEq)]
#[clap(rename_all = "PascalCase")]
pub enum ConvFormat {
    Labelme,
    LateralPoints,
    ScoliosisCoronal,
    ScoliosisSagittal,
}

#[derive(Parser, Debug)]
pub struct ConvArgs {
    /// Input file
    #[arg(value_hint = ValueHint::FilePath)]
    pub input: Option<PathBuf>,
    /// Output file
    #[arg(value_hint = ValueHint::FilePath)]
    pub output: Option<PathBuf>,
    /// Input and output in ndjson instead of json
    #[clap(long)]
    pub ndjson: bool,
    /// From format
    #[clap(short, long, default_value = "Labelme")]
    pub from: ConvFormat,
    /// To format
    #[clap(short, long, default_value = "LateralPoints")]
    pub to: ConvFormat,
    /// Pull pixel spacing from the input file if available
    ///
    /// Try Pixel Spacing (0028,0030) first, then Imager Pixel Spacing (0018,1164)
    #[clap(long = "spacing")]
    pub pull_spacing: bool,
}

#[derive(Parser, Debug, Clone)]
pub struct ImplantArgs {
    /// Input labelme and detectron2 joined json filename
    #[arg(value_hint = ValueHint::FilePath)]
    pub input: PathBuf,
}
