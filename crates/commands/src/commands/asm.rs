use std::path::Path;

use anyhow::Context;
use ndarray::{Array2, Axis};

use crate::cli::{
    AsmArgs, AsmConfigArgs, AsmFitArgs, AsmIcpArgs, AsmProjectArgs, AsmReconstructArgs,
    AsmSubCommands,
};
use labelme_rs::LabelMeData;

use scolrs::asm::{
    self, add_env_config, apply_asm, create_shapes_from_fitted_points, extract_points,
    extract_reference_points, icp, model::ActiveShapeModel, AsmConfig,
};
use serde_json;

pub fn cmd(args: AsmArgs) -> anyhow::Result<()> {
    match args.command {
        AsmSubCommands::Fit(args) => cmd_fit(args),
        AsmSubCommands::Project(args) => cmd_project(args),
        AsmSubCommands::Recon(args) => cmd_recon(args),
        AsmSubCommands::Icp(args) => cmd_icp(args),
        AsmSubCommands::Config(args) => cmd_config(args),
    }
}

pub fn cmd_fit(args: AsmFitArgs) -> anyhow::Result<()> {
    // Create ASM from json file
    let asm = read_asm(&args.asm_model)?;

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
            log::debug!(
                "Permuting heatmap foramt from channel-last (H, W, C) to channel-first (C, H, W)"
            );
            heatmaps.permuted_axes([2, 0, 1])
        }
        crate::cli::ChannelOrder::First => heatmaps,
    };
    let ref_lm: LabelMeData = serde_json::from_reader(
        std::fs::File::open(&args.lm_in)
            .with_context(|| format!("Opening labelme file {:?}", args.lm_in))?,
    )?;

    let mut config_builder =
        config::Config::builder().add_source(config::Config::try_from(&AsmConfig::default())?);
    if let Some(config_path) = args.config {
        log::debug!("Loading ASM fitting configuration from {:?}", config_path);
        config_builder = config_builder.add_source(config::File::from(config_path.as_path()))
    }
    let config_builder = add_env_config(config_builder).build()?;

    let asm_config: AsmConfig = config_builder.try_deserialize()?;
    log::debug!("ASM fitting configuration: {:?}", asm_config);

    let mut cached_heatmaps = asm::CachedHeatmaps::new(heatmaps.view());
    let (output_lm, histories, optimal_params, _fitted_shape) =
        apply_asm(asm, asm_config, &mut cached_heatmaps, &ref_lm)?;

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

        let output_file = std::fs::File::create(&output_path)
            .with_context(|| format!("Creating output file {:?}", output_path))?;
        serde_json::to_writer_pretty(output_file, &output_lm)
            .with_context(|| format!("Writing labelme data to {:?}", output_path))?;
        println!("Saved fitted labelme data to {:?}", output_path);
    }
    if let Some(history_path) = args.output.history {
        let obj_history = asm::fit::History::concat(histories);
        // save objective history to json
        let output_file = std::fs::File::create(&history_path)
            .with_context(|| format!("Creating output file {:?}", history_path))?;
        serde_json::to_writer_pretty(output_file, &obj_history)
            .with_context(|| format!("Writing objective history to {:?}", history_path))?;
        println!("Saved objective history to {:?}", history_path);
    }
    Ok(())
}

enum InteractiveInput {
    PosValue(usize, f64), // e.g. "0 1.5" sets parameter[0] = 1.5
    Print,
    Clear,
    Quit,
}

impl TryFrom<&str> for InteractiveInput {
    type Error = &'static str;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let trimmed = value.trim();
        if trimmed.eq_ignore_ascii_case("clear") || trimmed.eq_ignore_ascii_case("c") {
            Ok(InteractiveInput::Clear)
        } else if trimmed.eq_ignore_ascii_case("print") || trimmed.eq_ignore_ascii_case("p") {
            Ok(InteractiveInput::Print)
        } else if trimmed.is_empty()
            || trimmed.eq_ignore_ascii_case("quit")
            || trimmed.eq_ignore_ascii_case("q")
        {
            Ok(InteractiveInput::Quit)
        } else {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() != 2 {
                return Err("Invalid input format. Expected 'index value', 'clear', or 'quit'.");
            }
            let index: usize = parts[0]
                .parse()
                .map_err(|_| "Failed to parse index as usize")?;
            let value: f64 = parts[1]
                .parse()
                .map_err(|_| "Failed to parse value as f64")?;
            Ok(InteractiveInput::PosValue(index, value))
        }
    }
}

fn cmd_recon(args: AsmReconstructArgs) -> anyhow::Result<()> {
    let mut asm = read_asm(&args.asm_model)?;
    let ref_lm: LabelMeData = serde_json::from_reader(
        std::fs::File::open(&args.lm_in)
            .with_context(|| format!("Opening labelme file {:?}", args.lm_in))?,
    )?;
    // Calculate transform to align ASM template to reference points
    let ref_points = extract_reference_points(&ref_lm, &asm.reference_labels);
    let asm_movable_points = asm.to_movable_points();

    log::debug!(
        "Calculating transform to align ASM {:?} to reference points {:?}",
        asm.template_reference,
        ref_points
    );

    let tr_asm_to_ref = asm_movable_points.calculate_transform_with_missing(&ref_points, true)?;
    log::debug!("Transform ASM to reference: {:?}", tr_asm_to_ref);
    asm.global_transform(&tr_asm_to_ref);

    fn recon_and_save(
        asm: &ActiveShapeModel,
        shape_params: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>>,
        ref_lm: &LabelMeData,
        args: &AsmReconstructArgs,
    ) -> anyhow::Result<()> {
        let reconstructed_points = asm.pad_deform(shape_params.view());
        // save reconstructed points to labelme json
        let mut output_lm = ref_lm.clone();
        output_lm.shapes = create_shapes_from_fitted_points(&reconstructed_points, &asm.labels);
        output_lm.imageWidth = ref_lm.imageWidth;
        output_lm.imageHeight = ref_lm.imageHeight;
        output_lm.version = scolrs::VERSION.to_string();
        // not using `serde_json::to_writer_pretty` to write at once so that inotify can detect file write completion
        let output_str = serde_json::to_string_pretty(&output_lm)
            .with_context(|| "Serializing reconstructed labelme data")?;
        std::fs::write(&args.output, output_str)
            .with_context(|| format!("Writing labelme data to {:?}", args.output))?;
        Ok(())
    }

    if args.params.interactive {
        log::info!(
            "Entering interactive mode for shape parameter input. Watch {} for output.",
            args.output.display()
        );
        let mut shape_params = vec![0.0; asm.n_components()];
        loop {
            println!("Enter shape parameter index and value separated by space (e.g., '0 1.5'), 'clear' to reset parameters, or 'quit' to exit:");
            let mut input = String::new();
            std::io::stdin()
                .read_line(&mut input)
                .expect("Failed to read line");
            fn print_shape_params(params: &[f64]) {
                print!("Current shape parameters:");
                let last_non_zero = params.iter().rposition(|&x| x.abs() > f64::EPSILON);
                if last_non_zero.is_none() {
                    println!("  (all parameters are zero)");
                    return;
                }
                let last_non_zero = last_non_zero.unwrap();
                for &val in params.iter().take(last_non_zero + 1) {
                    print!("  {}", val);
                }
                println!();
            }
            match InteractiveInput::try_from(input.as_str()) {
                Ok(InteractiveInput::PosValue(index, value)) => {
                    if index >= shape_params.len() {
                        println!(
                            "Index out of bounds. Please enter an index between 0 and {}.",
                            shape_params.len() - 1
                        );
                    } else {
                        shape_params[index] = value;
                        log::info!("Set parameter[{}] = {}", index, value);
                        recon_and_save(
                            &asm,
                            &ndarray::Array1::from_vec(shape_params.clone()),
                            &ref_lm,
                            &args,
                        )?;
                        print_shape_params(&shape_params);
                    }
                }
                Ok(InteractiveInput::Print) => {
                    print_shape_params(&shape_params);
                }
                Ok(InteractiveInput::Clear) => {
                    shape_params.fill(0.0);
                    log::info!("Cleared all shape parameters.");
                    recon_and_save(
                        &asm,
                        &ndarray::Array1::from_vec(shape_params.clone()),
                        &ref_lm,
                        &args,
                    )?;
                }
                Ok(InteractiveInput::Quit) => {
                    print!("Exiting interactive mode.");
                    return Ok(());
                }
                Err(e) => {
                    println!("Invalid input: {}", e);
                }
            }
        }
    }

    let shape_params: Vec<f64> = if let Some(param_file) = &args.params.param_file {
        // load from file
        let file = std::fs::File::open(param_file)
            .with_context(|| format!("Opening shape parameters file {:?}", param_file))?;
        serde_json::from_reader(file)
            .with_context(|| format!("Loading shape parameters from {:?}", param_file))?
    } else if let Some(param_list) = &args.params.list {
        param_list.clone()
    } else {
        anyhow::bail!("No shape parameters provided. Please provide either a parameter file or a list of parameters.");
    };
    log::info!("Reconstructing shape with parameters: {:?}", shape_params);

    let shape_params: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =
        ndarray::Array1::from_vec(shape_params);
    recon_and_save(&asm, &shape_params, &ref_lm, &args)?;
    println!("Saved reconstructed labelme data to {:?}", args.output);
    Ok(())
}

fn cmd_icp(args: AsmIcpArgs) -> anyhow::Result<()> {
    let mut asm = read_asm(&args.asm_model)?;
    log::info!("Loaded ASM model from {:?}", asm.components.shape());
    let mut config_builder =
        config::Config::builder().add_source(config::Config::try_from(&icp::IcpConfig::default())?);
    if let Some(config_path) = args.config {
        log::debug!("Configuration from {:?}", config_path);
        config_builder = config_builder.add_source(config::File::from(config_path.as_path()))
    }
    let config_builder = add_env_config(config_builder).build()?;
    let icp_config: icp::IcpConfig = config_builder.try_deserialize()?;
    log::debug!("ICP configuration: {:?}", icp_config);

    let tgt_lm: LabelMeData = serde_json::from_reader(
        std::fs::File::open(&args.target_lm)
            .with_context(|| format!("Opening labelme file {:?}", args.target_lm))?,
    )?;
    let tgt_ref_points = extract_reference_points(&tgt_lm, &asm.reference_labels);
    let asm_movable_points = asm.to_movable_points();

    let points = extract_points(&tgt_lm, &asm.labels);
    for (label, pts) in asm.labels.iter().zip(points.iter()) {
        log::debug!("Target points for label '{}': {}", label, pts.len());
    }
    let target_point_sets: Vec<Array2<f64>> = points
        .iter()
        .map(|pts| {
            let flat: Vec<f64> = pts.iter().flat_map(|&(x, y)| vec![x, y]).collect();
            Array2::from_shape_vec((flat.len() / 2, 2), flat).unwrap()
        })
        .collect();

    let tr_asm_to_ref =
        asm_movable_points.calculate_transform_with_missing(&tgt_ref_points, true)?;
    log::debug!(
        "Transform ASM to target reference: scale={}, rotation={:?}, translation={:?}",
        tr_asm_to_ref.scale(),
        tr_asm_to_ref.rotation(),
        tr_asm_to_ref.t()
    );
    asm.global_transform(&tr_asm_to_ref);

    let icp_result = asm.icp_optimize(&target_point_sets, &icp_config, None)?;
    log::debug!("ICP optimization result: {:?}", icp_result);

    let params = icp_result.b.to_owned();
    let reconstructed_points = asm.pad_deform(params.view());
    // save reconstructed points to labelme json
    let mut output_lm = tgt_lm.clone();
    output_lm.shapes = create_shapes_from_fitted_points(&reconstructed_points, &asm.labels);
    output_lm.version = scolrs::VERSION.to_string();
    // not using `serde_json::to_writer_pretty` to write at once so that inotify can detect file write completion
    let output_str = serde_json::to_string_pretty(&output_lm)
        .with_context(|| "Serializing reconstructed labelme data")?;
    std::fs::write(&args.output, output_str)
        .with_context(|| format!("Writing labelme data to {:?}", args.output))?;

    Ok(())
}

fn read_asm(path: &Path) -> Result<ActiveShapeModel, anyhow::Error> {
    let reader =
        std::fs::File::open(path).with_context(|| format!("Opening ASM model file {:?}", path))?;
    let asm: ActiveShapeModel = serde_json::from_reader(reader)
        .with_context(|| format!("Loading ASM model from {:?}", path))?;
    Ok(asm)
}

fn cmd_config(args: AsmConfigArgs) -> anyhow::Result<()> {
    let mut config_builder =
        config::Config::builder().add_source(config::Config::try_from(&AsmConfig::default())?);
    if let Some(config_path) = args.config {
        log::debug!("Configuration from {:?}", config_path);
        config_builder = config_builder.add_source(config::File::from(config_path.as_path()))
    }
    if args.env {
        log::debug!("Configuration from environment variables with prefix 'ASM'");
        config_builder = add_env_config(config_builder);
    }
    let config_builder = config_builder.build()?;
    let asm_config: AsmConfig = config_builder.try_deserialize()?;
    let output_file = std::fs::File::create(&args.output)
        .with_context(|| format!("Creating configuration file {:?}", args.output))?;
    let toml_str = toml::to_string_pretty(&asm_config)?;
    std::io::Write::write_all(
        &mut std::io::BufWriter::new(output_file),
        toml_str.as_bytes(),
    )
    .with_context(|| format!("Writing configuration to {:?}", args.output))?;
    println!("Saved configuration file to {:?}", args.output);
    Ok(())
}

fn cmd_project(args: AsmProjectArgs) -> anyhow::Result<()> {
    let reader = std::fs::File::open(&args.asm_model)
        .with_context(|| format!("Opening ASM model file {:?}", args.asm_model))?;
    let asm: ActiveShapeModel = serde_json::from_reader(reader)
        .with_context(|| format!("Loading ASM model from {:?}", args.asm_model))?;
    let ref_lm: LabelMeData = serde_json::from_reader(
        std::fs::File::open(&args.lm_in)
            .with_context(|| format!("Opening labelme file {:?}", args.lm_in))?,
    )?;
    // Calculate transform to align ASM template to reference points
    let ref_points = extract_reference_points(&ref_lm, &asm.reference_labels);
    let asm_movable_points = asm.to_movable_points();

    let points = extract_points(&ref_lm, &asm.labels);
    // flatten Vec<Vec<(f64, f64)>> to Vec<f64> to Array1<f64>
    let flat_points: Vec<f64> = points
        .iter()
        .flat_map(|v| v.iter().flat_map(|&(x, y)| vec![x, y]))
        .collect();
    let points_arr2 =
        ndarray::Array2::from_shape_vec((flat_points.len() / 2, 2), flat_points).unwrap();

    let tgt_movable_points = scolrs::asm::alignment::MovablePoints::new(
        points_arr2,
        ref_points.clone(),
        asm_movable_points.point_counts.clone(),
    );

    log::debug!(
        "Calculating transform to align ASM {:?} to reference points {:?}",
        asm.template_reference,
        ref_points
    );

    // let tr_asm_to_ref = asm_movable_points.calculate_transform_with_missing(&ref_points, true)?;
    // let tr_ref_to_asm = asm_movable_points.calculate_transform_with_missing(&ref_points, false)?;
    // log::debug!("Transform ASM to reference: {:?}", tr_asm_to_ref);
    // asm.global_transform(&tr_asm_to_ref);
    // vec to array
    // let points = extract_points(&ref_lm, &asm.labels);
    let (tgt_aligned, _tf) = asm_movable_points.align_with_missing(&tgt_movable_points)?;
    log::debug!("Aligned target points: {:?}", tgt_aligned);
    log::debug!("Mean shape: {:?}", asm.mean);
    // flatten Vec<Vec<(f64, f64)>> to Vec<f64> to Array1<f64>
    // let flat_points: Vec<f64> = points
    //     .iter()
    //     .flat_map(|v| v.iter().flat_map(|&(x, y)| vec![x, y]))
    //     .collect();
    // let points_array = ndarray::Array1::from(flat_points);
    // Flatten Array2 to Array1
    let points_array = tgt_aligned
        .to_shape((tgt_aligned.len_of(Axis(0)) * tgt_aligned.len_of(Axis(1)),))
        .unwrap()
        .to_owned();

    let projected_params = asm.transform(&points_array)?.to_vec();
    log::debug!(
        "Projected parameters for given points: {:?}",
        projected_params
    );
    // save projected parameters to file
    let output_file = std::fs::File::create(&args.output)
        .with_context(|| format!("Creating output file {:?}", args.output))?;
    serde_json::to_writer_pretty(output_file, &projected_params)
        .with_context(|| format!("Writing projected parameters to {:?}", args.output))?;
    println!("Saved projected parameters to {:?}", args.output);
    Ok(())
}
