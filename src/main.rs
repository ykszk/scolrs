use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
use anyhow::Result;
use clap::Parser;

mod cli;
mod curve;
mod render;

use cli::Cli;
use cli::Command;

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    match cli.command {
        Command::Render(args) => render::cmd(args),
        Command::Curve(args) => curve::cmd(args),
    }
}
