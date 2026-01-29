use anyhow::Result;
use clap::CommandFactory;
use std::path::PathBuf;

pub fn cmd(args: crate::cli::ManArgs) -> Result<()> {
    let outdir: PathBuf = args.output;
    if outdir.is_file() {
        panic!("output must be a directory");
    }
    if !outdir.exists() {
        println!("Creating output directory: {:?}", outdir);
        std::fs::create_dir_all(&outdir).unwrap();
    }

    let cmd = crate::cli::Cli::command();
    clap_mangen::generate_to(cmd, &outdir).unwrap();
    Ok(())
}
