use anyhow::Result;
use clap::Parser;

mod neck;
use neck::cli::{Cli, Command};
use neck::{measure, svg};

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    match cli.command {
        Command::Svg(args) => svg::cmd(args),
        Command::Measure(args) => measure::cmd(args),
    }
}
