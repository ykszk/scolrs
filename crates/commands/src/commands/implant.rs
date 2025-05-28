use std::io::{BufRead, BufReader};

use crate::cli::ImplantArgs;
use crate::utils::Ndjson;
use anyhow::{Context, Result};
use indexmap::IndexMap;
use labelme_rs::LabelMeDataLine;
use ndarray::Axis;
use scolrs::implant::{
    detect_one_sided_configuration, split_unsorted_screws, LabelMeOptionalDetectron2,
    LabelMeOptionalDetectron2Line, ScrewSpine, ScrewVertebra,
};
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
                    let labelme = screw_spine.to_labelme(
                        data_line.content.labelme.flags,
                        args.vertebrae,
                        false,
                    );
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
                crate::cli::ImplantTask::Split => {
                    let screw_spine = split_screws(screw_spine);
                    let labelme = screw_spine.to_labelme(
                        data_line.content.labelme.flags,
                        args.vertebrae,
                        true,
                    );
                    let lm_line = LabelMeDataLine {
                        content: labelme,
                        filename: data_line.filename,
                    };
                    println!("{}", serde_json::to_string(&lm_line)?);
                }
            }
        }
    } else {
        let json_str = std::fs::read_to_string(args.input.as_path())
            .with_context(|| format!("Failed to read file: {:?}", args.input))?;
        let data: LabelMeOptionalDetectron2 = serde_json::from_str(&json_str)?;
        let screw_spine = data.screw_spine()?;
        match args.task {
            crate::cli::ImplantTask::Group => {
                let labelme = screw_spine.to_labelme(data.labelme.flags, args.vertebrae, false);
                println!("{}", serde_json::to_string_pretty(&labelme)?);
            }
            crate::cli::ImplantTask::Count => {
                let count = label_counts(screw_spine.count_screws());
                println!("{}", serde_json::to_string_pretty(&count)?);
            }
            crate::cli::ImplantTask::Split => {
                let screw_spine = split_screws(screw_spine);
                let labelme = screw_spine.to_labelme(data.labelme.flags, args.vertebrae, true);
                println!("{}", serde_json::to_string_pretty(&labelme)?);
            }
        }
    }

    Ok(())
}

/// Splits screws into left and right based on their positions in the spine.
/// If screws are already labeled as left or right, they are kept as is.
fn split_screws(mut screw_spine: ScrewSpine) -> ScrewSpine {
    // Compute centroids for each screw's bounding box
    let screw_centroids: Vec<_> = screw_spine
        .screws
        .iter()
        .map(|s| {
            let rect = scolrs::implant::Rectangle {
                tl: s.bb.tl,
                br: s.bb.br,
            };
            scolrs::implant::CalculateCentroid::centroid(&rect)
        })
        .collect();

    let vertebrae = screw_spine.spine.t1_to_sac_vertebrae();
    let vertebra_centroids = screw_spine.spine.t1_to_sac_centroids();

    // Build ScrewVertebra list
    let screw_vertebrae: Vec<ScrewVertebra> = screw_spine
        .screws
        .iter()
        .enumerate()
        .map(|(i_screw, screw)| {
            let i_vert = screw.vertebra;
            let v_centroid = vertebra_centroids.index_axis(Axis(0), i_vert);
            let c_rect = screw_centroids[i_screw];
            let vertebra = vertebrae.index_axis(Axis(0), i_vert);
            let dx = c_rect.0 - v_centroid[0];
            ScrewVertebra::create(i_screw, c_rect, i_vert, dx, vertebra)
        })
        .collect();

    if detect_one_sided_configuration(&screw_vertebrae) {
        let mean_dx =
            screw_vertebrae.iter().map(|sv| sv.dx).sum::<f64>() / screw_vertebrae.len() as f64;
        let left = mean_dx < 0.0;
        log::debug!(
            "Detected one-sided configuration with {} screws",
            if left { "left" } else { "right" }
        );
        let left = Some(left);

        screw_spine.screws.iter_mut().for_each(|s| s.left = left);
        return screw_spine;
    }

    log::debug!("Detected two-sided configuration");
    // Sort by y-coordinate
    let mut screw_vertebrae = screw_vertebrae;
    use scolrs::implant::ScrewVertebraUtils;
    screw_vertebrae.sort_by_y();

    let (mut hard_left, mut hard_right) = (Vec::new(), Vec::new());

    // Assign screws with explicit left/right or by x-coordinate
    for sv1 in &screw_vertebrae {
        if hard_left.contains(&sv1.i_screw) || hard_right.contains(&sv1.i_screw) {
            continue;
        }
        let screw = &screw_spine.screws[sv1.i_screw];
        if let Some(left) = screw.left {
            if left {
                hard_left.push(sv1.i_screw);
            } else {
                hard_right.push(sv1.i_screw);
            }
            continue;
        }
        for sv2 in &screw_vertebrae {
            if sv1.i_screw == sv2.i_screw
                || hard_left.contains(&sv2.i_screw)
                || hard_right.contains(&sv2.i_screw)
            {
                continue;
            }
            // If two screws are on the same vertebra, assign them based on their x-coordinates
            if sv1.i_vert == sv2.i_vert {
                if sv1.c_rect.0 < sv2.c_rect.0 {
                    hard_left.push(sv1.i_screw);
                    hard_right.push(sv2.i_screw);
                } else {
                    hard_left.push(sv2.i_screw);
                    hard_right.push(sv1.i_screw);
                }
            }
        }
    }

    let unsorted_indices: Vec<_> = (0..screw_vertebrae.len())
        .filter(|i| !hard_left.contains(i) && !hard_right.contains(i))
        .collect();
    log::debug!("Splitting {} unsorted screws", unsorted_indices.len());

    let (left_pairs, right_pairs) =
        split_unsorted_screws(screw_vertebrae, hard_left, unsorted_indices);

    // Assign left/right labels to screws
    for (pairs, left) in [(left_pairs, true), (right_pairs, false)] {
        for sv in pairs {
            if let Some(screw) = screw_spine.screws.get_mut(sv.i_screw) {
                screw.left = Some(left);
            } else {
                log::warn!("Screw index {} out of bounds", sv.i_screw);
            }
        }
    }
    screw_spine
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
