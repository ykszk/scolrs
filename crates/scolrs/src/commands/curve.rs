use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result};
use labelme_rs::{serde_json, LabelMeData, LabelMeDataLine};
use scolrs::{
    CoronalPoints, Curve, CurveDesc, CurveScoreSet, CurveSet, CurveSetAlgorithm, VertebraDiscIndex,
};
use serde::{Deserialize, Serialize};

use crate::cli::{self, CurveArgs};

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
    pub all_curve_scores: Vec<(CurveSet, CurveScoreSet)>,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct CurveInfoAllLine {
    pub content: CurveInfoAll,
    pub filename: String,
}

impl TryFrom<(&LabelMeData, &CurveSetAlgorithm)> for CurveInfoAll {
    type Error = anyhow::Error;

    fn try_from(
        (data, algorithm): (&LabelMeData, &CurveSetAlgorithm),
    ) -> Result<Self, Self::Error> {
        let mut coronal_points = CoronalPoints::try_from(data)?;
        coronal_points.spine.verticalize();
        coronal_points.c_coefs = coronal_points.spine.fit_poly().unwrap();

        // let info = coronal_points.identify_curves();
        let info = coronal_points.identify_curves_with_algorithm(algorithm);
        let all_curves = coronal_points.find_all_curves();
        let all_curves: Vec<_> = all_curves
            .into_iter()
            .map(|(curve, angle)| {
                let apex = coronal_points.id_apex(&curve);
                (curve, angle, apex)
            })
            .collect();
        let all_curve_sets = coronal_points.find_all_curve_sets();
        let all_curve_scores = all_curve_sets
            .into_iter()
            .map(|set| {
                let score = set.score(&coronal_points);
                (set, score)
            })
            .collect();
        Ok(CurveInfoAll {
            info,
            all_curves,
            all_curve_scores,
        })
    }
}

impl TryFrom<(&LabelMeDataLine, &CurveSetAlgorithm)> for CurveInfoLine {
    type Error = anyhow::Error;

    fn try_from(
        (data, algorithm): (&LabelMeDataLine, &CurveSetAlgorithm),
    ) -> Result<Self, Self::Error> {
        let content: CurveDesc = (&data.content, algorithm).try_into()?;
        Ok(CurveInfoLine {
            content,
            filename: data.filename.clone(),
        })
    }
}

impl TryFrom<(&LabelMeDataLine, &CurveSetAlgorithm)> for CurveInfoAllLine {
    type Error = anyhow::Error;

    fn try_from(
        (data, algorithm): (&LabelMeDataLine, &CurveSetAlgorithm),
    ) -> Result<Self, Self::Error> {
        let content: CurveInfoAll = (&data.content, algorithm).try_into()?;
        Ok(CurveInfoAllLine {
            content,
            filename: data.filename.clone(),
        })
    }
}

pub fn cmd(args: CurveArgs) -> Result<()> {
    if args.algorithm == cli::CurveSetAlgorithm::Apex && args.weights.is_some() {
        log::warn!("Ignoring weights for Apex algorithm");
    }
    let algorithm = match args.algorithm {
        cli::CurveSetAlgorithm::Apex => CurveSetAlgorithm::Apex,
        cli::CurveSetAlgorithm::Score => {
            let weights = args
                .weights
                .map_or_else(scolrs::CurveWeights::default, |v_ws| {
                    let a_ws: [f64; 9] = v_ws.try_into().unwrap();
                    scolrs::CurveWeights::from(a_ws)
                });
            CurveSetAlgorithm::Score(weights)
        }
    };
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
            Box::new(BufReader::new(
                File::open(&args.input).with_context(|| format!("Open file {:?}", &args.input))?,
            ))
        };
        for line in reader.lines() {
            let line = line?;
            let data: LabelMeDataLine = line.as_str().try_into()?;
            if args.all {
                let info: CurveInfoAllLine = (&data, &algorithm).try_into()?;
                println!("{}", serde_json::to_string(&info)?);
            } else {
                let info: CurveInfoLine = (&data, &algorithm).try_into()?;
                println!("{}", serde_json::to_string(&info)?);
            }
        }
    } else {
        // single json IO
        let s = std::fs::read_to_string(&args.input)
            .with_context(|| format!("Read string from {:?}", &args.input))?;
        let data: LabelMeData = if args.labelme {
            s.try_into()?
        } else {
            let cp: CoronalPoints = serde_json::from_str(&s)?;
            cp.into()
        };
        if args.all {
            let info: CurveInfoAll = (&data, &algorithm).try_into()?;
            println!("{}", serde_json::to_string(&info)?);
        } else {
            let info: CurveDesc = (&data, &algorithm).try_into()?;
            println!("{}", serde_json::to_string(&info)?);
        }
    }
    Ok(())
}
