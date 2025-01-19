use std::io::{BufRead, BufReader};

use crate::cli::ImplantArgs;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ScrewCount {
    pub content: Vec<usize>,
    pub filename: String,
}

pub fn cmd(args: ImplantArgs) -> Result<()> {
    if args.input.extension().unwrap_or_default() == "ndjson" || args.input.as_os_str() == "-" {
        let reader: Box<dyn BufRead> = if args.input.as_os_str() != "-" {
            Box::new(BufReader::new(std::fs::File::open(args.input)?))
        } else {
            Box::new(BufReader::new(std::io::stdin()))
        };
        for line in reader.lines() {
            let line = line?;
            let labelme_detectron2: scolrs::implant::LabelMeDetectron2Line =
                serde_json::from_str(&line)?;
            let screw_spine = labelme_detectron2.content.pair_screw()?;
            let screw_counts = screw_spine.count_screws();
            let screw_count = ScrewCount {
                content: screw_counts,
                filename: labelme_detectron2.filename,
            };
            let json_str = serde_json::to_string(&screw_count)?;
            println!("{}", json_str);
        }
    } else {
        let json_str = std::fs::read_to_string(args.input.as_path())
            .with_context(|| format!("Failed to read file: {:?}", args.input))?;
        let labelme_detectron2: scolrs::implant::LabelMeDetectron2 =
            serde_json::from_str(&json_str)?;
        let screw_spine = labelme_detectron2.pair_screw()?;
        let screw_counts = screw_spine.count_screws();
        println!("Screw counts: {:?}", screw_counts);
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
        };
        cmd(args).unwrap();
    }
}
