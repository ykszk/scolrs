use crate::neck::cli::MeasureArgs;
use anyhow::Result;
use labelme_rs::LabelMeData;
use scolrs::head_neck::LateralPoints;

pub fn cmd(args: MeasureArgs) -> Result<()> {
    let data = LabelMeData::try_from(args.input.as_path())?;
    let lateral_points = LateralPoints::try_from(&data);
    println!("{:?}", lateral_points);
    Ok(())
}
