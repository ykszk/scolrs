use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::Result;
use labelme_rs::{serde_json, LabelMeData, LabelMeDataLine};
use scolrs::{ApexSet, CurveInfo, CurveInfoLine, CurveSet};
use serde::{Deserialize, Serialize};

use crate::cli::CurveArgs;

type CurvePair = (usize, usize);
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[allow(non_snake_case)]
struct TSet<T> {
    PT: Option<T>,
    MT: Option<T>,
    TLL: Option<T>,
}

type Indices = TSet<CurvePair>;
type Apices = TSet<usize>;
type Angles = TSet<f32>;

impl From<&CurveSet> for Indices {
    fn from(cs: &CurveSet) -> Self {
        Self {
            PT: cs.pt.as_ref().map(|e| (e.0.sup, e.0.inf)),
            MT: cs.mt.as_ref().map(|e| (e.0.sup, e.0.inf)),
            TLL: cs.tll.as_ref().map(|e| (e.0.sup, e.0.inf)),
        }
    }
}

impl From<&ApexSet> for Apices {
    fn from(set: &ApexSet) -> Self {
        Self {
            PT: set.pt.map(|e| e as usize),
            MT: set.mt.map(|e| e as usize),
            TLL: set.tll.map(|e| e as usize),
        }
    }
}

impl From<&CurveSet> for Angles {
    fn from(cs: &CurveSet) -> Self {
        Self {
            PT: cs.pt.as_ref().map(|e| e.1),
            MT: cs.mt.as_ref().map(|e| e.1),
            TLL: cs.tll.as_ref().map(|e| e.1),
        }
    }
}

pub fn cmd(args: CurveArgs) -> Result<()> {
    if args.input.as_os_str() == "-"
        || args
            .input
            .extension()
            .map_or(false, |e| e == "ndjson" || e == "jsonl")
    {
        let reader: Box<dyn BufRead> = if args.input.as_os_str() == "-" {
            Box::new(BufReader::new(std::io::stdin()))
        } else {
            Box::new(BufReader::new(File::open(&args.input)?))
        };
        for line in reader.lines() {
            let line = line?;
            let data: LabelMeDataLine = line.as_str().try_into()?;
            let scol = scolrs::Scoliosis::try_from(&data.data)?;
            let (curves, apex_set, major_curve) = scol.identify_curves();
            let info = CurveInfo::new(curves, apex_set, major_curve);
            let info_line = CurveInfoLine {
                info,
                filename: data.filename,
            };
            println!("{}", serde_json::to_string(&info_line)?);
        }
    } else {
        let s = std::fs::read_to_string(args.input)?;
        let data: LabelMeData = s.as_str().try_into()?;
        let scol = scolrs::Scoliosis::try_from(&data)?;
        let (curves, apex_set, major_curve) = scol.identify_curves();
        let curve_info = CurveInfo::new(curves, apex_set, major_curve);
        println!("{}", serde_json::to_string(&curve_info)?);
    }
    Ok(())
}
