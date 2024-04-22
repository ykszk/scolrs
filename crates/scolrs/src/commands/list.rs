use anyhow::Result;
use scolrs::{
    CoronalMeasure, SagittalMeasure,
};
use strum::VariantArray;

use crate::cli::ListArgs;

pub fn cmd(_args: ListArgs) -> Result<()> {
    println!("Available coronal measurements:");
    for measure in CoronalMeasure::VARIANTS {
        print!(" {}", measure);
    }
    println!();
    println!("Available sagittal measurements:");
    for measure in SagittalMeasure::VARIANTS {
        print!(" {}", measure);
    }
    println!();
    Ok(())
}
