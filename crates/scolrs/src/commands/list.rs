use std::fmt::Display;

use crate::cli::ListArgs;
use anyhow::Result;
use scolrs::{
    head_neck::{NeckLateralDraw, NeckLateralMeasure},
    CoronalMeasure, MeasureAndDraw, SagittalMeasure,
};

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
        print_measures(
            "Available neck lateral measurements:",
            NeckLateralDraw::all(),
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
        print_measures(
            "Available neck lateral measurements:",
            NeckLateralMeasure::all(),
        );
    }
    println!();
    Ok(())
}
