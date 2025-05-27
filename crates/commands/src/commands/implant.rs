use std::io::{BufRead, BufReader};

use crate::cli::ImplantArgs;
use crate::utils::Ndjson;
use anyhow::{Context, Result};
use indexmap::IndexMap;
use labelme_rs::LabelMeDataLine;
use scolrs::implant::{LabelMeOptionalDetectron2, LabelMeOptionalDetectron2Line};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct ScrewCount {
    pub content: IndexMap<String, usize>,
    pub filename: String,
}

fn vertebra_label(index: usize) -> String {
    if index < 12 {
        format!("T{}", index + 1)
    } else {
        format!("L{}", index - 11)
    }
}

fn label_counts(counts: Vec<usize>) -> IndexMap<String, usize> {
    counts
        .into_iter()
        .enumerate()
        .map(|(i, count)| (vertebra_label(i), count))
        .collect()
}

pub fn cmd(args: ImplantArgs) -> Result<()> {
    if args.input.as_os_str() == "-" || args.input.is_ndjson() {
        let reader: Box<dyn BufRead> = if args.input.as_os_str() != "-" {
            Box::new(BufReader::new(std::fs::File::open(args.input)?))
        } else {
            Box::new(BufReader::new(std::io::stdin()))
        };
        for line in reader.lines() {
            let line = line?;
            let data_line = serde_json::from_str::<LabelMeOptionalDetectron2Line>(&line)?;
            log::debug!("Processing line: {:?}", data_line.filename);
            let screw_spine = data_line.content.screw_spine()?;
            match args.task {
                crate::cli::ImplantTask::Group => {
                    let labelme =
                        screw_spine.to_labelme(data_line.content.labelme.flags, args.vertebrae);
                    let lm_line = LabelMeDataLine {
                        content: labelme,
                        filename: data_line.filename,
                    };
                    println!("{}", serde_json::to_string(&lm_line)?);
                }
                crate::cli::ImplantTask::Count => {
                    let count = label_counts(screw_spine.count_screws());
                    let count_line = ScrewCount {
                        content: count,
                        filename: data_line.filename,
                    };
                    println!("{}", serde_json::to_string(&count_line)?);
                }
                crate::cli::ImplantTask::Split => todo!(),
            }
        }
    } else {
        let json_str = std::fs::read_to_string(args.input.as_path())
            .with_context(|| format!("Failed to read file: {:?}", args.input))?;
        let data: LabelMeOptionalDetectron2 = serde_json::from_str(&json_str)?;
        let screw_spine = data.screw_spine()?;
        match args.task {
            crate::cli::ImplantTask::Group => {
                let labelme = screw_spine.to_labelme(data.labelme.flags, args.vertebrae);
                println!("{}", serde_json::to_string_pretty(&labelme)?);
            }
            crate::cli::ImplantTask::Count => {
                let count = label_counts(screw_spine.count_screws());
                println!("{}", serde_json::to_string_pretty(&count)?);
            }
            crate::cli::ImplantTask::Split => {
                for screw in screw_spine.screws {
                    if screw.left.is_none() {
                        todo!();
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_cmd() {
        let args = ImplantArgs {
            input: PathBuf::from("../../tests/data/case5/frontal_postop.json"),
            vertebrae: false,
            task: crate::cli::ImplantTask::Group,
        };
        cmd(args).unwrap();
    }
}
