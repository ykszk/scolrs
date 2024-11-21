use std::fmt::Display;

use crate::cli::ListArgs;
use anyhow::Result;
use scolrs::{CoronalMeasure, MeasureAndDraw, SagittalMeasure};

fn print_measures<T: Display>(title: &str, measures: Vec<T>) {
    println!("{}", title);
    for measure in measures {
        print!(" {}", measure);
    }
    println!();
}

pub fn cmd(args: ListArgs) -> Result<()> {
    if args.drawable {
        print_measures(
            "Available drawable components:",
            CoronalMeasure::all_draws(),
        );
        print_measures(
            "Available sagittal measurements:",
            SagittalMeasure::all_draws(),
        );
    } else {
        print_measures(
            "Available coronal measurements:",
            CoronalMeasure::all_measures(),
        );
        print_measures(
            "Available sagittal measurements:",
            SagittalMeasure::all_measures(),
        );
    }
    println!();
    Ok(())
}
