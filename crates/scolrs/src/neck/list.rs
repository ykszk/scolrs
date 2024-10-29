use crate::neck::cli::ListArgs;
use anyhow::Result;
pub fn cmd(_args: ListArgs) -> Result<()> {
    for measure in scolrs::head_neck::NeckLateralMeasure::all() {
        println!("{}", measure);
    }
    Ok(())
}
