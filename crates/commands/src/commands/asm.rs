use anyhow::Context;

use crate::cli::AsmArgs;
use scolrs::asm::{
    adam::EarlyTermination, adam::TerminationCriterion, fit::fit_asm_to_heatmap,
    model::ActiveShapeModel,
};
use serde_json;

pub fn cmd(args: AsmArgs) -> anyhow::Result<()> {
    // Create ASM from json file
    let reader = std::fs::File::open(&args.asm_model)
        .with_context(|| format!("Opening ASM model file {:?}", args.asm_model))?;
    let asm: ActiveShapeModel = serde_json::from_reader(reader)
        .with_context(|| format!("Loading ASM model from {:?}", args.asm_model))?;

    let mut npz = ndarray_npz::NpzReader::new(
        std::fs::File::open(&args.heatmaps)
            .with_context(|| format!("Opening {:?}", args.heatmaps))?,
    )?;
    let heatmaps: ndarray::Array3<f64> = npz
        .by_name(&args.key)
        .with_context(|| format!("Reading array with key '{}' from npz", args.key))?;

    // Fit ASM to heatmap
    let patience = args.patience;
    let min_delta = 0.0;
    let mut termination = EarlyTermination::new(TerminationCriterion::Any(vec![
        TerminationCriterion::NoImprovement {
            patience,
            min_delta,
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
    let fitted_shape = asm.deform(optimal_params.view());

    println!("Fitted shape points: {:?}", fitted_shape);
    println!("Fitted shape parameters: {:?}", optimal_params);
    println!("Objective history: {:?}", obj_history);
    Ok(())
}
