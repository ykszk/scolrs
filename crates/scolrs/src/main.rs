use anyhow::Result;
use clap::Parser;

mod cli;
use cli::Cli;
use cli::Command;
mod commands;
use commands::{curve, lenke, list, measure, svg};

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    match cli.command {
        Command::Svg(args) => svg::cmd(args),
        Command::Measure(args) => measure::cmd(args),
        Command::Curve(args) => curve::cmd(args),
        Command::Lenke(args) => lenke::cmd(args),
        Command::List(args) => list::cmd(args),
    }
}
