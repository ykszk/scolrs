use crate::cli::ImplantArgs;
use anyhow::{Context, Result};

pub fn cmd(args: ImplantArgs) -> Result<()> {
    let json_str = std::fs::read_to_string(args.input.as_path())
        .with_context(|| format!("Failed to read file: {:?}", args.input))?;
    let data: labelme_rs::LabelMeData = serde_json::from_str(&json_str)?;
    let implant = scolrs::implant::ImplantSpine::try_from(&data)?;
    let pairs = implant.pair_screw();
    println!("{:?}", pairs);

    Ok(())
}
