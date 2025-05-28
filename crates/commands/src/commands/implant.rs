use std::io::{BufRead, BufReader};

use crate::cli::ImplantArgs;
use crate::utils::Ndjson;
use anyhow::{Context, Result};
use indexmap::IndexMap;
use labelme_rs::LabelMeDataLine;
use ndarray::{Axis, Slice};
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
fn split_screws(screw_spine: ScrewSpine) -> ScrewSpine {
    let rectangle_centroid: Vec<_> = screw_spine
        .screws
        .iter()
        .map(|s| {
            let tl = s.bb.tl;
            let br = s.bb.br;
            let rect = scolrs::implant::Rectangle { tl, br };
            scolrs::implant::CalculateCentroid::centroid(&rect)
        })
        .collect();
    let vertebrae = screw_spine
        .spine
        .v_c7tl
        .0
        .slice_axis(Axis(0), Slice::from(1..));
    let vertebra_centroids = screw_spine
        .spine
        .c_c7tl
        .slice_axis(Axis(0), Slice::from(1..));
    let mut screw_vertebrae: Vec<ScrewVertebra> = Vec::new();
    for (i, screw) in screw_spine.screws.iter().enumerate() {
        let i_vert = screw.vertebra;
        let v_centroid = vertebra_centroids.index_axis(Axis(0), i_vert);
        let i_rect = i;
        let c_rect = (rectangle_centroid[i_rect].0, rectangle_centroid[i_rect].1);
        let vertebra = vertebrae.index_axis(Axis(0), i_vert);
        let distances = vertebra.map_axis(Axis(1), |xy| {
            ((c_rect.0 - xy[0]).powi(2) + (c_rect.1 - xy[1]).powi(2)).sqrt()
        });
        let dist = distances.iter().cloned().fold(f64::INFINITY, f64::min);
        let dx = c_rect.0 - v_centroid[0];
        let screw_vertebra = ScrewVertebra {
            i_rect,
            c_rect,
            i_vert,
            dist,
            dx,
        };
        screw_vertebrae.push(screw_vertebra);
    }
    let screws_one_sided = detect_one_sided_configuration(&screw_vertebrae);
    if screws_one_sided {
        let mean_dx =
            screw_vertebrae.iter().map(|sv| sv.dx).sum::<f64>() / screw_vertebrae.len() as f64;

        let left = if mean_dx < 0.0 {
            log::info!("Detected one-sided configuration with left screws");
            Some(true)
        } else {
            log::info!("Detected one-sided configuration with right screws");
            Some(false)
        };
        let mut screw_spine = screw_spine;
        screw_spine.screws.iter_mut().for_each(|s| s.left = left);
        screw_spine
    } else {
        log::info!("Detected two-sided configuration");
        // sort screw_vertebrae by y-coordinate of c_rect
        screw_vertebrae.sort_by(|a, b| a.c_rect.1.partial_cmp(&b.c_rect.1).unwrap());
        let (mut hard_left, mut hard_right) = (Vec::new(), Vec::new());
        // if two screws are on the same vertebra, split them into left and right based on their x-coordinates
        for sv1 in screw_vertebrae.iter() {
            let screw = &screw_spine.screws[sv1.i_rect];
            if let Some(left) = screw.left {
                if left {
                    hard_left.push(sv1.i_rect);
                } else {
                    hard_right.push(sv1.i_rect);
                }
                continue;
            }
            if hard_left.contains(&sv1.i_rect) || hard_right.contains(&sv1.i_rect) {
                continue;
            }
            for sv2 in screw_vertebrae.iter() {
                if sv1.i_rect == sv2.i_rect
                    || hard_left.contains(&sv2.i_rect)
                    || hard_right.contains(&sv2.i_rect)
                {
                    continue;
                }
                if sv1.i_vert == sv2.i_vert {
                    if sv1.c_rect.0 < sv2.c_rect.0 {
                        hard_left.push(sv1.i_rect);
                        hard_right.push(sv2.i_rect);
                    } else {
                        hard_left.push(sv2.i_rect);
                        hard_right.push(sv1.i_rect);
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
        let mut screws = Vec::new();
        for left_pair in left_pairs {
            let screw = &screw_spine.screws[left_pair.i_rect];
            let mut new_screw = screw.clone();
            new_screw.left = Some(true);
            screws.push(new_screw);
        }
        for right_pair in right_pairs {
            let screw = &screw_spine.screws[right_pair.i_rect];
            let mut new_screw = screw.clone();
            new_screw.left = Some(false);
            screws.push(new_screw);
        }
        let mut screw_spine = screw_spine;
        screw_spine.screws = screws;
        screw_spine
    }
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
