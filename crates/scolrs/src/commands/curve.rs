use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::Result;
use labelme_rs::{serde_json, LabelMeData, LabelMeDataLine};
use scolrs::{CoronalPoints, Curve, CurveDesc, VertebraDiscIndex};
use serde::{Deserialize, Serialize};

use crate::cli::CurveArgs;

#[derive(Serialize, Deserialize, Debug)]
pub struct CurveInfoLine {
    pub content: CurveDesc,
    pub filename: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CurveInfoAll {
    #[serde(flatten)]
    pub info: CurveDesc,
    pub all_curves: Vec<(Curve, f64, VertebraDiscIndex)>,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct CurveInfoAllLine {
    pub content: CurveInfoAll,
    pub filename: String,
}

impl TryFrom<&LabelMeData> for CurveInfoAll {
    type Error = anyhow::Error;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let mut coronal_points = CoronalPoints::try_from(data)?;
        let info = coronal_points.identify_curves();
        coronal_points.spine.verticalize();
        let all_curves = coronal_points.find_all_curves();
        let all_curves: Vec<_> = all_curves
            .into_iter()
            .map(|(curve, angle)| {
                let apex = coronal_points.id_apex(&curve);
                (curve, angle, apex)
            })
            .collect();
        Ok(CurveInfoAll { info, all_curves })
    }
}

impl TryFrom<&LabelMeDataLine> for CurveInfoLine {
    type Error = anyhow::Error;

    fn try_from(data: &LabelMeDataLine) -> Result<Self, Self::Error> {
        let content: CurveDesc = (&data.content).try_into()?;
        Ok(CurveInfoLine {
            content,
            filename: data.filename.clone(),
        })
    }
}

impl TryFrom<&LabelMeDataLine> for CurveInfoAllLine {
    type Error = anyhow::Error;

    fn try_from(data: &LabelMeDataLine) -> Result<Self, Self::Error> {
        let content: CurveInfoAll = (&data.content).try_into()?;
        Ok(CurveInfoAllLine {
            content,
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
        let data: LabelMeData = if args.labelme {
            s.try_into()?
        } else {
            let cp: CoronalPoints = serde_json::from_str(&s)?;
            cp.into()
        };
        if args.all {
            let info: CurveInfoAll = (&data).try_into()?;
            println!("{}", serde_json::to_string(&info)?);
        } else {
            let info: CurveDesc = (&data).try_into()?;
            println!("{}", serde_json::to_string(&info)?);
        }
    }
    Ok(())
}
