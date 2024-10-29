use crate::neck::cli::ListArgs;
use anyhow::Result;

pub fn cmd(args: ListArgs) -> Result<()> {
    if args.drawings {
        for drawing in scolrs::head_neck::NeckLateralDraw::all() {
            println!("{}", drawing);
        }
    } else {
        for measure in scolrs::head_neck::NeckLateralMeasure::all() {
            println!("{}", measure);
        }
    }
    Ok(())
}
