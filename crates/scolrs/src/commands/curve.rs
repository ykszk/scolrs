use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::Result;
use labelme_rs::{serde_json, LabelMeDataLine};
use scolrs::{Curve, CurveInfo};
use serde::{Deserialize, Serialize};

use crate::cli::CurveArgs;

#[derive(Serialize, Deserialize, Debug)]
pub struct OutputInfo {
    #[serde(flatten)]
    pub info: CurveInfo,
    pub all_curves: Vec<(Curve, f32)>,
    pub filename: String,
}

fn print_single(line: &str) -> Result<()> {
    let data: LabelMeDataLine = line.try_into()?;
    let scol = scolrs::Scoliosis::try_from(&data.data)?;
    let (curves, apex_set, major_curve) = scol.identify_curves();
    let info = CurveInfo::new(curves, apex_set, major_curve);
    let all_curves = scol.find_all_curves();
    let info_line = OutputInfo {
        info,
        all_curves,
        filename: data.filename,
    };
    println!("{}", serde_json::to_string(&info_line)?);
    Ok(())
}

pub fn cmd(args: CurveArgs) -> Result<()> {
    if args.input.as_os_str() == "-"
        || args
            .input
            .extension()
            .map_or(false, |e| e == "ndjson" || e == "jsonl")
    {
        // ndjson IO
        let reader: Box<dyn BufRead> = if args.input.as_os_str() == "-" {
            Box::new(BufReader::new(std::io::stdin()))
        } else {
            Box::new(BufReader::new(File::open(&args.input)?))
        };
        for line in reader.lines() {
            let line = line?;
            print_single(line.as_str())?;
        }
    } else {
        // single json IO
        let s = std::fs::read_to_string(args.input)?;
        print_single(s.as_str())?;
    }
    Ok(())
}
