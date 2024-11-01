use anyhow::Result;
use clap::{CommandFactory, Parser};
use clap_complete::{generate, Generator};
use env_logger::Env;

mod neck;
use neck::cli::{Cli, Command};
use neck::{catalog, list, measure, ndconv, svg};

fn print_completions<G: Generator>(gen: G, cmd: &mut clap::Command) {
    generate(gen, cmd, cmd.get_name().to_string(), &mut std::io::stdout());
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("warn")).init();

    let cli = Cli::parse();
    match cli.command {
        Command::Complete(args) => {
            let mut cmd = Cli::command();
            print_completions(args.shell, &mut cmd);
            Ok(())
        }
        Command::Svg(args) => svg::cmd(args),
        Command::Measure(args) => measure::cmd(args),
        Command::Catalog(args) => catalog::cmd(args),
        Command::Html(args) => neck::html::cmd(args),
        Command::List(args) => list::cmd(args),
        Command::Ndconv(args) => ndconv::cmd(args),
    }
}
