use crate::neck::cli::MeasureArgs;
use anyhow::{Context, Result};
use indexmap::IndexMap;
use labelme_rs::LabelMeData;
use log::warn;
use scolrs::head_neck::{self, LateralPoints, NeckMeasureComponent};

pub fn cmd(args: MeasureArgs) -> Result<()> {
    let data = LabelMeData::try_from(args.input.as_path())
        .with_context(|| format!("Loading {:?}", args.input))?;
    let lateral_points = LateralPoints::try_from(&data)?;
    let mut results: IndexMap<&str, Vec<f32>> = Default::default();
    let measures: Vec<Box<dyn NeckMeasureComponent>> = vec![
        Box::new(head_neck::Adi(&lateral_points)),
        Box::new(head_neck::WedgeAngle(&lateral_points)),
        Box::new(head_neck::Sacs(&lateral_points)),
        Box::new(head_neck::OC2(&lateral_points)),
    ];
    for measure in measures {
        let result = measure.measure();
        match result {
            Ok(result) => {
                results.insert(measure.name(), result);
            }
            Err(e) => match e {
                scolrs::MeasureError::InvalidNumberOfPoints(err) => {
                    warn!("Skip point count error for {}: {:?}", measure.name(), err);
                }
                e => return Err(e.into()),
            },
        }
    }

    println!("{}", serde_json::to_string_pretty(&results)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_measure_cmd_case1() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");

        let input = data_dir.join("neck_case1/lateral.json");
        let args = MeasureArgs {
            input,
            measures: vec![],
        };
        cmd(args)
    }

    #[test]
    fn test_measure_cmd_case2() -> Result<()> {
        let data_dir = PathBuf::from("../../tests/data/");

        let input = data_dir.join("neck_case2/extension_lateral.json");
        let args = MeasureArgs {
            input,
            measures: vec![],
        };
        cmd(args)?;

        let input = data_dir.join("neck_case2/flexion_lateral.json");
        let args = MeasureArgs {
            input,
            measures: vec![],
        };
        cmd(args)
    }
}
