use anyhow::Result;
use labelme_rs::{serde_json, LabelMeData};

use crate::cli::CurveArgs;

pub fn cmd(args: CurveArgs) -> Result<()> {
    let s = std::fs::read_to_string(args.input)?;
    let data: LabelMeData = s.as_str().try_into()?;
    let scol = scolrs::Scoliosis::try_from(&data)?;
    let (curves, _apexes) = scol.find_curve_set();
    println!("{}", serde_json::to_string(&curves)?);
    Ok(())
}
