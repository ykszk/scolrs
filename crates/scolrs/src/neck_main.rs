use anyhow::Result;
use clap::Parser;
use env_logger::Env;

mod neck;
use neck::cli::{Cli, Command};
use neck::{measure, svg};

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("warn")).init();

    let cli = Cli::parse();
    match cli.command {
        Command::Svg(args) => svg::cmd(args),
        Command::Measure(args) => measure::cmd(args),
    }
}
