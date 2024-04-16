use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::Result;
use labelme_rs::{serde_json, LabelMeData, LabelMeDataLine};
use scolrs::{Curve, ScolDesc, Spine, VertebraDiscIndex};
use serde::{Deserialize, Serialize};

use crate::cli::CurveArgs;

#[derive(Serialize, Deserialize, Debug)]
pub struct CurveInfoLine {
    #[serde(flatten)]
    pub info: ScolDesc,
    pub filename: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CurveInfoAll {
    #[serde(flatten)]
    pub info: ScolDesc,
    pub all_curves: Vec<(Curve, f32, VertebraDiscIndex)>,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct CurveInfoAllLine {
    #[serde(flatten)]
    pub info: ScolDesc,
    pub all_curves: Vec<(Curve, f32, VertebraDiscIndex)>,
    pub filename: String,
}

impl TryFrom<&LabelMeData> for CurveInfoAll {
    type Error = anyhow::Error;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let scol = Spine::try_from(data)?;
        let (curves, apex_set, major_curve) = scol.identify_curves();
        let info = ScolDesc::new(curves, apex_set, major_curve);
        let all_curves = scol.find_all_curves();
        let coefs = scol.spinal_poly()?;
        let all_curves: Vec<_> = all_curves
            .into_iter()
            .map(|(curve, angle)| {
                let apex = scol.id_apex(&curve, coefs.view());
                (curve, angle, apex)
            })
            .collect();
        Ok(CurveInfoAll { info, all_curves })
    }
}

impl TryFrom<&LabelMeDataLine> for CurveInfoLine {
    type Error = anyhow::Error;

    fn try_from(data: &LabelMeDataLine) -> Result<Self, Self::Error> {
        let info: ScolDesc = (&data.content).try_into()?;
        Ok(CurveInfoLine {
            info,
            filename: data.filename.clone(),
        })
    }
}

impl TryFrom<&LabelMeDataLine> for CurveInfoAllLine {
    type Error = anyhow::Error;

    fn try_from(data: &LabelMeDataLine) -> Result<Self, Self::Error> {
        let info: CurveInfoAll = (&data.content).try_into()?;
        Ok(CurveInfoAllLine {
            info: info.info,
            all_curves: info.all_curves,
            filename: data.filename.clone(),
        })
    }
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
            let data: LabelMeDataLine = line.as_str().try_into()?;
            if args.all {
                let info: CurveInfoAllLine = (&data).try_into()?;
                println!("{}", serde_json::to_string(&info)?);
            } else {
                let info: CurveInfoLine = (&data).try_into()?;
                println!("{}", serde_json::to_string(&info)?);
            }
        }
    } else {
        // single json IO
        let s = std::fs::read_to_string(args.input)?;
        let data: LabelMeData = s.try_into()?;
        if args.all {
            let info: CurveInfoAll = (&data).try_into()?;
            println!("{}", serde_json::to_string(&info)?);
        } else {
            let info: ScolDesc = (&data).try_into()?;
            println!("{}", serde_json::to_string(&info)?);
        }
    }
    Ok(())
}
