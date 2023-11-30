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
    Render(RenderArgs),
    /// Determine curves
    Curve(CurveArgs),
}

#[derive(ValueEnum, Debug, Copy, Clone)]
pub enum Direction {
    Frontal,
    Lateral,
}

#[derive(Parser, Debug)]
pub struct RenderArgs {
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
    /// Resize image. Specify in imagemagick's `-resize`-like format
    #[clap(long)]
    pub resize: Option<String>,
}

#[derive(Parser, Debug)]
pub struct CurveArgs {
    /// Input labelme json filename
    pub input: PathBuf,
}
