use clap::{Parser, Subcommand, ValueEnum, ValueHint};
use clap_complete::Shell;
use scolrs::head_neck::NeckLateralMeasure;
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
    /// Measure neck parameters
    Measure(MeasureArgs),
    /// List available measurements
    List(ListArgs),
    /// Convert data format
    Conv(ConvArgs),
}

#[derive(Parser)]
pub struct CompleteArgs {
    /// Shell to generate completions for
    pub shell: Shell,
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
pub struct ListArgs {
    /// List drawings instead of measurements
    #[clap(long)]
    pub drawings: bool,
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
