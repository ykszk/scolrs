use std::io::{BufRead, BufReader};

use crate::cli::ImplantArgs;
use anyhow::{Context, Result};
use scolrs::implant::{
    LabelMeDetectron2, LabelMeOptionalDetectron2, LabelMeOptionalDetectron2Line,
};
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
            let data_line = serde_json::from_str::<LabelMeOptionalDetectron2Line>(&line)?;
            if let Some(detectron2) = data_line.content.detectron2 {
                let data = LabelMeDetectron2 {
                    labelme: data_line.content.labelme,
                    detectron2,
                };
                let screw_spine = data.pair_screw()?;
                let screw_counts = screw_spine.count_screws();
                println!("{} {:?}", data_line.filename, screw_counts);
            } else {
                let implant = scolrs::implant::ImplantSpine::try_from(&data_line.content.labelme)?;
                let pairs = implant.pair_screw();
                println!("{} {:?}", data_line.filename, pairs);
            }
        }
    } else {
        let json_str = std::fs::read_to_string(args.input.as_path())
            .with_context(|| format!("Failed to read file: {:?}", args.input))?;
        let data: LabelMeOptionalDetectron2 = serde_json::from_str(&json_str)?;
        if let Some(detectron2) = data.detectron2 {
            let data = LabelMeDetectron2 {
                labelme: data.labelme,
                detectron2,
            };
            let screw_spine = data.pair_screw()?;
            let screw_counts = screw_spine.count_screws();
            println!("{} {:?}", args.input.display(), screw_counts);
        } else {
            let implant = scolrs::implant::ImplantSpine::try_from(&data.labelme)?;
            let pairs = implant.pair_screw();
            println!("{} {:?}", args.input.display(), pairs);
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
        };
        cmd(args).unwrap();
    }
}
