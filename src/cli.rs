use clap::{Args, Parser, Subcommand, ValueEnum};
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

#[derive(Parser, Debug)]
pub struct RenderArgs {
    /// Input labelme json filename
    pub input: PathBuf,
    /// Output svg filename
    pub output: PathBuf,
    /// Config file in toml
    #[clap(long)]
    pub config: Option<PathBuf>,
    /// Label colors in yaml
    #[clap(long)]
    pub label_colors: Option<PathBuf>,
}

#[derive(Parser, Debug)]
pub struct CurveArgs {
    /// Input labelme json filename
    pub input: PathBuf,
    /// Output json filename
    pub output: PathBuf,
}
