use crate::cli::{MeasureArgs, Plane};
use anyhow::{Context, Result};
use indexmap::IndexMap;
use labelme_rs::{serde_json, LabelMeData};
use log::{debug, warn};
use scolrs::{SagittalComponent, SagittalMeasure};

pub fn cmd(args: MeasureArgs) -> Result<()> {
    debug!("Loading {:?}", args.input);

    let data: LabelMeData = args
        .input
        .as_path()
        .try_into()
        .with_context(|| format!("Load LabelMeData from {:?}", &args.input))?;

    let results = match args.direction {
        Plane::Coronal => {
            warn!("Coronal plane is not implemented");
            Default::default()
            // let coronal_points = scolrs::CoronalPoints::try_from(&data)?;
            // let curve_apex_set = if let Some(filename) = args.curve_set {
            //     let reader = std::fs::File::open(&filename)
            //         .with_context(|| format!("Load curve set {:?}", filename))?;
            //     let cs: ScolDesc = labelme_rs::serde_json::from_reader(reader)?;
            //     Some((cs.curves, cs.apices))
            // } else {
            //     None
            // };
            // draw_coronal(
            //     data,
            //     coronal_points,
            //     draw_param,
            //     svg_size,
            //     label_colors,
            //     line_colors,
            //     curve_apex_set,
            // )
        }
        Plane::Sagittal => {
            let sagittal_points = scolrs::SagittalPoints::try_from(&data)?;
            let measures: Vec<SagittalMeasure> = if args.measures.is_empty() {
                SagittalMeasure::all()
            } else {
                args.measures
            };
            let mut results: IndexMap<SagittalMeasure, f32> = Default::default();
            for measure in measures {
                let spinal_measure: Box<dyn SagittalComponent> = measure.into();
                match spinal_measure.measure(&sagittal_points) {
                    Ok(m) => {
                        results.insert(measure, m);
                    }
                    Err(err) => warn!("Failed to draw {}: {:?}", spinal_measure.name(), err),
                }
            }
            results
        }
    };

    // serde_json::to_writer_pretty(std::io::stdout(), &results)?;
    // println!();
    // Use to_string so that tests can capture the output
    println!("{}", serde_json::to_string_pretty(&results)?);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data_directory() -> std::path::PathBuf {
        let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../../tests/data");
        path
    }

    fn _test_case(case_dir: &str) -> Result<()> {
        let data_dir = test_data_directory();
        let input = data_dir.join(case_dir).join("lateral.json");
        let curve_set = None;
        let direction = Plane::Sagittal;
        let measures = Default::default();
        let args = MeasureArgs {
            input,
            curve_set,
            direction,
            measures,
        };
        cmd(args)
    }

    #[test]
    fn test_case1() -> Result<()> {
        _test_case("case1")
    }
    #[test]
    fn test_case2() -> Result<()> {
        _test_case("case2")
    }
    #[test]
    fn test_case3() -> Result<()> {
        _test_case("case3")
    }
    #[test]
    fn test_case4() -> Result<()> {
        _test_case("case4")
    }
}
