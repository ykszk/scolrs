use anyhow::Result;
use clap::Parser;

mod cli;
use cli::Cli;
use cli::Command;
mod commands;
use commands::{curve, render};

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    // Ok(())
    match cli.command {
        Command::Render(args) => render::cmd(args),
        Command::Curve(args) => curve::cmd(args),
    }
}
