use anyhow::Result;
use clap::CommandFactory;
use clap::Parser;
use clap_complete::{generate, Generator};

mod cli;
use cli::Cli;
use cli::Command;
mod commands;
use commands::{curve, lenke, measure, svg};
mod utils;

fn print_completions<G: Generator>(gen: G, cmd: &mut clap::Command) {
    generate(gen, cmd, cmd.get_name().to_string(), &mut std::io::stdout());
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    match cli.command {
        Command::Complete(args) => {
            let mut cmd = Cli::command();
            print_completions(args.shell, &mut cmd);
            Ok(())
        }
        Command::Svg(args) => svg::cmd(args),
        Command::SvgNdjson(args) => commands::svg::cmd_ndjson(args),
        Command::Measure(args) => measure::cmd(args),
        Command::Curve(args) => curve::cmd(args),
        Command::Lenke(args) => lenke::cmd(args),
        Command::Catalog(args) => commands::catalog::cmd(args),
        Command::Html(args) => commands::html::cmd(args),
        Command::Conv(args) => commands::conv::cmd(args),
        Command::Implant(args) => commands::implant::cmd(args),
        Command::Confidence(args) => commands::confidence::cmd(args),
        Command::Asm(args) => commands::asm::cmd(args),
    }
}
