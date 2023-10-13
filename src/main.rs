use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
use clap::Parser;
#[macro_use]
extern crate log;
use anyhow::Result;

mod cli;
mod curve;
mod render;

use cli::Cli;
use cli::Command;

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Render(args) => render::cmd(args),
        Command::Curve(args) => curve::cmd(args),
    }
}
