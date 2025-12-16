use anyhow::Context;
use ndarray::Axis;

use crate::cli::AsmArgs;
use labelme_rs::{LabelMeData, Shape};
use ndarray_ndimage as ndi;
use scolrs::asm::{
    adam::EarlyTermination, adam::TerminationCriterion, alignment::MovablePoints,
    fit::fit_asm_to_heatmap, model::ActiveShapeModel,
};
use serde_json;

fn extract_reference_points(
    lm_data: &LabelMeData,
    reference_labels: &[String],
) -> Vec<Option<(f64, f64)>> {
    let mut points = Vec::new();
    let shape_dict = lm_data.to_shape_map();
    let Some(point_map) = shape_dict.get("point") else {
        log::warn!("No 'point' shape found in labelme data.");
        return vec![None; reference_labels.len()];
    };
    for label in reference_labels {
        match point_map.get(label.as_str()) {
            Some(p) => {
                if p.len() > 1 {
                    log::warn!(
                        "Reference point '{}' has more than one point ({}), using the first one.",
                        label,
                        p.len()
                    );
                }
                points.push(Some((p[0][0].0, p[0][0].1)));
            }
            None => {
                points.push(None);
            }
        }
    }
    points
}

pub fn cmd(args: AsmArgs) -> anyhow::Result<()> {
    // Create ASM from json file
    let reader = std::fs::File::open(&args.asm_model)
        .with_context(|| format!("Opening ASM model file {:?}", args.asm_model))?;
    let mut asm: ActiveShapeModel = serde_json::from_reader(reader)
        .with_context(|| format!("Loading ASM model from {:?}", args.asm_model))?;

    let mut npz = ndarray_npz::NpzReader::new(
        std::fs::File::open(&args.heatmaps)
            .with_context(|| format!("Opening {:?}", args.heatmaps))?,
    )?;
    let heatmaps: ndarray::Array3<f64> = npz
        .by_name(&args.key)
        .with_context(|| format!("Reading array with key '{}' from npz", args.key))?;
    log::debug!(
        "Loaded heatmaps with shape {:?} from {:?}",
        heatmaps.dim(),
        args.heatmaps
    );
    let heatmaps = match args.channel_order {
        crate::cli::ChannelOrder::Last => {
            // log::debug!("Permuting heatmap axes from (H, W, C) to (C, H, W)");
            // heatmaps.permuted_axes([2, 0, 1])
            heatmaps
        }
        crate::cli::ChannelOrder::First => {
            log::debug!("Permuting heatmap axes from (C, H, W) to (H, W, C)");
            heatmaps.permuted_axes([1, 2, 0])
        }
    };
    let heatmaps = if let Some(sigma) = args.sigma {
        log::debug!(
            "Applying Gaussian smoothing to heatmaps with sigma={}",
            sigma
        );
        let mut smoothed_heatmaps = heatmaps.clone();
        for (ch_idx, channel) in heatmaps.axis_iter(Axis(2)).enumerate() {
            let smoothed = ndi::gaussian_filter(&channel, sigma, 0, ndi::BorderMode::Mirror, 5);
            smoothed_heatmaps
                .index_axis_mut(ndarray::Axis(2), ch_idx)
                .assign(&smoothed);
        }
        log::debug!("Finished Gaussian smoothing of heatmaps");
        smoothed_heatmaps
    } else {
        heatmaps
    };

    let ref_lm: LabelMeData = serde_json::from_reader(
        std::fs::File::open(&args.lm_in)
            .with_context(|| format!("Opening labelme file {:?}", args.lm_in))?,
    )?;

    let scale_hm_to_ref = ref_lm.imageHeight as f64 / heatmaps.len_of(Axis(0)) as f64;
    let mut scaled_ref_lm = ref_lm.clone();
    log::debug!(
        "Scaling reference labelme data by factor {} to match heatmap size",
        scale_hm_to_ref
    );
    scaled_ref_lm.scale(1.0 / scale_hm_to_ref);
    let ref_points = extract_reference_points(&scaled_ref_lm, &asm.reference_labels);
    let asm_movable_points = MovablePoints::new(
        asm.template_points.clone(),
        asm.template_reference.clone(),
        asm.point_counts.clone(),
    );

    log::debug!(
        "Calculating transform to align ASM {:?} to reference points {:?}",
        asm.template_reference,
        ref_points
    );
    let tr_asm_to_ref = asm_movable_points.calculate_transform_with_missing(&ref_points, true)?;
    log::debug!("Transform ASM to reference: {:?}", tr_asm_to_ref);
    asm.global_transform(&tr_asm_to_ref);

    // Fit ASM to heatmap
    let patience = args.patience;
    let min_delta = 0.0;
    let mut termination = EarlyTermination::new(TerminationCriterion::Any(vec![
        TerminationCriterion::NoImprovement {
            patience,
            min_delta,
            objective: scolrs::asm::adam::ObjectiveType::Minimize,
        },
        TerminationCriterion::MaxIterations(args.max_iterations),
    ]));

    let (optimal_params, obj_history) = fit_asm_to_heatmap(
        &asm,
        args.n_mode,
        &mut termination,
        heatmaps.view(),
        args.lambda,
        args.learning_rate,
    );

    // Get final fitted landmark positions
    let fitted_shape = asm.pad_deform(optimal_params.view());

    if let Some(output_path) = args.output.params {
        // save as json
        let output_file = std::fs::File::create(&output_path)
            .with_context(|| format!("Creating output file {:?}", output_path))?;
        let param_vec = optimal_params.to_vec();
        serde_json::to_writer_pretty(output_file, &param_vec)
            .with_context(|| format!("Writing parameters to {:?}", output_path))?;
        println!("Saved fitted parameters to {:?}", output_path);
    }
    if let Some(output_path) = args.output.lm_out {
        // save fitted points to labelme json
        let mut output_lm = ref_lm.clone();
        let mut shapes = Vec::new();
        for (points, label) in fitted_shape.iter().zip(asm.labels.iter()) {
            for point in points.axis_iter(Axis(0)) {
                let shape = Shape {
                    label: label.clone(),
                    points: vec![(point[0], point[1])],
                    group_id: None,
                    shape_type: "point".to_string(),
                    flags: Default::default(),
                };
                shapes.push(shape);
            }
        }
        output_lm.shapes = shapes;
        output_lm.scale(scale_hm_to_ref);
        output_lm.imageWidth = ref_lm.imageWidth;
        output_lm.imageHeight = ref_lm.imageHeight;
        let output_file = std::fs::File::create(&output_path)
            .with_context(|| format!("Creating output file {:?}", output_path))?;
        serde_json::to_writer_pretty(output_file, &output_lm)
            .with_context(|| format!("Writing labelme data to {:?}", output_path))?;
        println!("Saved fitted labelme data to {:?}", output_path);
    }
    if let Some(history_path) = args.output.history {
        let output_file = std::fs::File::create(&history_path)
            .with_context(|| format!("Creating output file {:?}", history_path))?;
        serde_json::to_writer_pretty(output_file, &obj_history)
            .with_context(|| format!("Writing objective history to {:?}", history_path))?;
        println!("Saved objective history to {:?}", history_path);
    }
    Ok(())
}
