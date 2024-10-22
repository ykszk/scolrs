use clap::{Parser, Subcommand, ValueEnum, ValueHint};
use std::path::PathBuf;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
pub struct Cli {
    #[clap(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Create SVG
    Svg(SvgArgs),
    /// Measure neck parameters
    Measure(MeasureArgs),
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
    /// Resize x-ray image. Specify in imagemagick's `-resize`-like format
    #[clap(long)]
    pub resize: Option<String>,
    /// Output image size. Aspect ratio will be adjusted based on x-ray image size. Specify in imagemagick's `-resize`-like format
    #[clap(long)]
    pub size: Option<String>,
    /// Measurements to draw. By default, all measurements are drawn. Use `--list` to see all measurements
    #[clap(short, long, value_delimiter = ',', value_hint = ValueHint::Other)]
    pub measures: Vec<String>,
    /// Hide measurements. Use `--list` to see all measurements
    #[clap(long)]
    pub hide: Vec<String>,
}

#[derive(Parser, Debug)]
pub struct MeasureArgs {
    /// Input json file
    pub input: PathBuf,
}
