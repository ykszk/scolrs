use std::io::BufRead;

use crate::cli::{self, CurveArgs};
use crate::utils::{CreateReaderWriter, Ndjson};
use anyhow::{Context, Result};
use labelme_rs::{serde_json, LabelMeData, LabelMeDataLine};
use scolrs::{
    CoronalPoints, CoronalPointsAndCurve, CoronalPointsAndCurveLine, CoronalPointsLine, Curve,
    CurveDesc, CurveDescOptionalAngles, CurveScoreSet, CurveSet, CurveSetAlgorithm,
    VertebraDiscIndex,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct CurveDescLine {
    pub content: CurveDesc,
    pub filename: String,
}

#[derive(Serialize, Debug)]
pub struct CurveDescOptionalAnglesLine {
    pub content: CurveDescOptionalAngles,
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
        coronal_points.verticalize_spine()?;

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

impl TryFrom<(&LabelMeDataLine, &CurveSetAlgorithm)> for CurveDescLine {
    type Error = anyhow::Error;

    fn try_from(
        (data, algorithm): (&LabelMeDataLine, &CurveSetAlgorithm),
    ) -> Result<Self, Self::Error> {
        let content: CurveDesc = (&data.content, algorithm).try_into()?;
        Ok(CurveDescLine {
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
    if args.input.as_os_str() == "-" || args.input.is_ndjson() {
        // ndjson IO
        let reader = args.input.create_reader()?;
        for line in reader.lines() {
            let line = line?;
            // let data: LabelMeDataLine = line.as_str().try_into()?;
            let (data, coronal_points): (LabelMeDataLine, Option<CoronalPointsLine>) =
                if args.labelme {
                    (line.as_str().try_into()?, None)
                } else {
                    let cp: CoronalPointsLine = serde_json::from_str(&line)?;
                    let content: LabelMeData = cp.content.clone().into();
                    (
                        LabelMeDataLine {
                            filename: cp.filename.clone(),
                            content,
                        },
                        Some(cp),
                    )
                };
            match args.format {
                cli::CurveSetOutput::Curve => {
                    let info: CurveDescLine = (&data, &algorithm).try_into()?;
                    if args.omit_angles {
                        let info_no_angles = CurveDescOptionalAnglesLine {
                            content: info.content.strip_angles(),
                            filename: info.filename,
                        };
                        println!("{}", serde_json::to_string(&info_no_angles)?);
                    } else {
                        println!("{}", serde_json::to_string(&info)?);
                    }
                }
                cli::CurveSetOutput::All => {
                    if args.omit_angles {
                        log::warn!("Omit angles option is ignored for 'all' output format");
                    }
                    let info: CurveInfoAllLine = (&data, &algorithm).try_into()?;
                    println!("{}", serde_json::to_string(&info)?);
                }
                cli::CurveSetOutput::Points => {
                    if args.omit_angles {
                        log::warn!("Omit angles option is ignored for 'points' output format");
                    }
                    let info: CurveDescLine = (&data, &algorithm).try_into()?;

                    if let Some(cp) = coronal_points {
                        let pts_and_curve: CoronalPointsAndCurveLine = CoronalPointsAndCurveLine {
                            filename: data.filename,
                            content: CoronalPointsAndCurve {
                                coronal_points: cp.content,
                                curves: info.content,
                            },
                        };
                        println!("{}", serde_json::to_string(&pts_and_curve)?);
                    } else {
                        log::warn!(
                            "No coronal points found for file {:?}, skipping points output",
                            data.filename
                        );
                    }
                }
            }
        }
    } else {
        // single json IO
        let s = std::fs::read_to_string(&args.input)
            .with_context(|| format!("Read string from {:?}", args.input))?;
        let (data, coronal_points): (LabelMeData, Option<CoronalPoints>) = if args.labelme {
            (s.try_into()?, None)
        } else {
            let cp: CoronalPoints = serde_json::from_str(&s)?;
            let lm: LabelMeData = cp.clone().into();
            (lm, Some(cp))
        };
        match args.format {
            cli::CurveSetOutput::Curve => {
                let info: CurveDesc = (&data, &algorithm).try_into()?;
                if args.omit_angles {
                    let no_angles = info.strip_angles();
                    println!("{}", serde_json::to_string(&no_angles)?);
                } else {
                    println!("{}", serde_json::to_string(&info)?);
                }
            }
            cli::CurveSetOutput::All => {
                if args.omit_angles {
                    log::warn!("Omit angles option is ignored for 'all' output format");
                }
                let info: CurveInfoAll = (&data, &algorithm).try_into()?;
                println!("{}", serde_json::to_string(&info)?);
            }
            cli::CurveSetOutput::Points => {
                if args.omit_angles {
                    log::warn!("Omit angles option is ignored for 'points' output format");
                }
                let info: CurveDesc = (&data, &algorithm).try_into()?;

                if let Some(cp) = coronal_points {
                    let pts_and_curve = CoronalPointsAndCurve {
                        coronal_points: cp,
                        curves: info,
                    };
                    println!("{}", serde_json::to_string(&pts_and_curve)?);
                } else {
                    log::warn!(
                        "No coronal points found for file {:?}, skipping points output",
                        args.input
                    );
                }
            }
        }
    }
    Ok(())
}
