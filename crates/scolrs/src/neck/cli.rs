use clap::{Parser, Subcommand, ValueEnum, ValueHint};
use clap_complete::Shell;
use scolrs::head_neck::{NeckLateralDraw, NeckLateralMeasure};
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
    /// Measure neck parameters
    Measure(MeasureArgs),
    /// Compile SVGs into a catalog HTML
    Catalog(CatalogArgs),
    /// List available measurements
    List(ListArgs),
    /// Convert SVGs to HTML
    Html(HtmlArgs),
    /// Convert data format
    Conv(ConvArgs),
}

#[derive(Parser)]
pub struct CompleteArgs {
    /// Shell to generate completions for
    pub shell: Shell,
}

#[derive(Parser, Debug, Default, Clone)]
pub struct SvgArgs {
    /// Input labelme json/ndjson filename
    #[arg(value_hint = ValueHint::FilePath)]
    pub input: PathBuf,
    /// Output svg filename for json input or output directory for ndjson input
    #[arg(value_hint = ValueHint::AnyPath)]
    pub output: PathBuf,
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
    /// Measurements to draw. By default, all measurements are drawn.measurements
    #[clap(short, long, value_delimiter = ',', value_hint = ValueHint::Other)]
    pub measures: Option<Vec<NeckLateralDraw>>,
    /// Hide measurements.
    #[clap(long)]
    pub hide: Vec<NeckLateralDraw>,
    /// Maximum number of jobs to run in parallel
    #[clap(short, long)]
    pub jobs: Option<usize>,
}

#[derive(Parser, Debug, Default)]
pub struct MeasureArgs {
    /// Input json/ndjson file
    #[arg(value_hint = ValueHint::FilePath)]
    pub input: PathBuf,
    /// Output json/ndjson file
    #[arg(value_hint = ValueHint::FilePath)]
    pub output: Option<PathBuf>,
    /// Measurements to draw. By default, all measurements are drawn.
    #[clap(short, long, value_delimiter = ',', value_hint = ValueHint::Other)]
    pub measures: Option<Vec<NeckLateralMeasure>>,
}

#[derive(Parser, Debug)]
pub struct CatalogArgs {
    /// Input svg containing directory
    #[arg(value_hint = ValueHint::DirPath)]
    pub input: PathBuf,
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

#[derive(Parser, Debug)]
pub struct ListArgs {
    /// List drawings instead of measurements
    #[clap(long)]
    pub drawings: bool,
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
