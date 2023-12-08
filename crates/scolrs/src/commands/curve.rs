use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::Result;
use labelme_rs::{serde_json, LabelMeDataLine};
use scolrs::{Curve, CurveInfo, VertebraDiscIndex};
use serde::{Deserialize, Serialize};

use crate::cli::CurveArgs;

#[derive(Serialize, Deserialize, Debug)]
pub struct CurveSetLine {
    #[serde(flatten)]
    pub info: CurveInfo,
    pub filename: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CurveSetLineAll {
    #[serde(flatten)]
    pub info: CurveInfo,
    pub all_curves: Vec<(Curve, f32, VertebraDiscIndex)>,
    pub filename: String,
}

fn print_single(line: &str, all: bool) -> Result<()> {
    let data: LabelMeDataLine = line.try_into()?;
    let scol = scolrs::Scoliosis::try_from(&data.data)?;
    let (curves, apex_set, major_curve) = scol.identify_curves();
    let info = CurveInfo::new(curves, apex_set, major_curve);
    if all {
        let all_curves = scol.find_all_curves();
        let coefs = scol.spinal_poly()?;
        let all_curves: Vec<_> = all_curves
            .into_iter()
            .map(|(curve, angle)| {
                let apex = scol.find_apex(&curve, coefs.view());
                (curve, angle, apex)
            })
            .collect();
        let info_line = CurveSetLineAll {
            info,
            all_curves,
            filename: data.filename,
        };
        println!("{}", serde_json::to_string(&info_line)?);
    } else {
        let info_line = CurveSetLine {
            info,
            filename: data.filename,
        };
        println!("{}", serde_json::to_string(&info_line)?);
    }
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
            print_single(line.as_str(), args.all)?;
        }
    } else {
        // single json IO
        let s = std::fs::read_to_string(args.input)?;
        print_single(s.as_str(), args.all)?;
    }
    Ok(())
}
