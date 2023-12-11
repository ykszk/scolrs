use anyhow::{Context, Result};
use labelme_rs::LabelMeData;
use scolrs::{SagittalModifier, Spine, Study};
use std::path::Path;

use crate::cli::LenkeArgs;

fn load_spine(filename: &Path) -> Result<Spine> {
    let s = std::fs::read_to_string(filename).with_context(|| format!("Opening {:?}", filename))?;
    let data: LabelMeData = s.as_str().try_into()?;
    Ok(Spine::try_from(&data)?)
}

pub fn cmd(args: LenkeArgs) -> Result<()> {
    let coronal = load_spine(&args.coronal)?;

    let (curve_set, apex_set, major_curve) = coronal.identify_curves();
    let left = args
        .left
        .map(|filename| load_spine(filename.as_path()))
        .transpose()?;
    let right = args
        .right
        .map(|filename| load_spine(filename.as_path()))
        .transpose()?;
    let sagittal = args
        .sagittal
        .map(|filename| load_spine(filename.as_path()))
        .transpose()?;

    let study = Study::new(coronal, left, right, sagittal);

    let chart = study.chart(&curve_set, major_curve.unwrap());
    println!("chart:\n{}", chart);
    match chart.classify() {
        Ok(cls) => println!("Curve type: {:?}", cls),
        Err(clses) => {
            if clses.is_empty() {
                println!("No curve type found.")
            } else {
                println!("Potential curve types: {:?}", clses)
            }
        }
    }
    if let Some(apex) = apex_set.tll {
        println!("Lumbar modifier:{:?}", study.coronal.lumbar_modifier(apex));
    }
    if let Some(sagittal) = study.sagittal {
        let angle = sagittal.angle(&scolrs::T5T12).unwrap();
        let s_mod: SagittalModifier = angle.into();
        println!("Sagittal modifier:{}", s_mod);
    }

    Ok(())
}
