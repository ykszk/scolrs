use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use labelme_rs::LabelMeData;
use scolrs::head_neck::LateralPoints;

#[derive(Parser, Debug)]
pub struct Args {
    /// Input json file
    pub input: PathBuf,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let data = LabelMeData::try_from(args.input.as_path())?;
    let lateral_points = LateralPoints::try_from(&data);
    println!("{:?}", lateral_points);
    Ok(())
}
