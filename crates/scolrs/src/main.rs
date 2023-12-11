use anyhow::Result;
use clap::Parser;

mod cli;
use cli::Cli;
use cli::Command;
mod commands;
use commands::{curve, lenke, svg};

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    match cli.command {
        Command::Svg(args) => svg::cmd(args),
        Command::Curve(args) => curve::cmd(args),
        Command::Lenke(args) => lenke::cmd(args),
    }
}
