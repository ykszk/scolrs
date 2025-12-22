use anyhow::Context;

use crate::cli::{AsmArgs, AsmConfigArgs, AsmFitArgs, AsmReconstructArgs, AsmSubCommands};
use labelme_rs::LabelMeData;

use scolrs::asm::{
    self, apply_asm, create_shapes_from_fitted_points, extract_reference_points,
    model::ActiveShapeModel, AsmConfig,
};
use serde_json;

pub fn cmd(args: AsmArgs) -> anyhow::Result<()> {
    match args.command {
        AsmSubCommands::Fit(args) => cmd_fit(args),
        AsmSubCommands::Recon(args) => cmd_recon(args),
        AsmSubCommands::Config(args) => cmd_config(args),
    }
}

pub fn cmd_fit(args: AsmFitArgs) -> anyhow::Result<()> {
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

fn cmd_recon(args: AsmReconstructArgs) -> anyhow::Result<()> {
    // Create ASM from json file
    let reader = std::fs::File::open(&args.asm_model)
        .with_context(|| format!("Opening ASM model file {:?}", args.asm_model))?;
    let mut asm: ActiveShapeModel = serde_json::from_reader(reader)
        .with_context(|| format!("Loading ASM model from {:?}", args.asm_model))?;
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
    let reconstructed_points = asm.pad_deform(shape_params.view());
    // save reconstructed points to labelme json
    let mut output_lm = ref_lm.clone();
    output_lm.shapes = create_shapes_from_fitted_points(&reconstructed_points, &asm.labels);
    output_lm.imageWidth = ref_lm.imageWidth;
    output_lm.imageHeight = ref_lm.imageHeight;
    output_lm.version = scolrs::VERSION.to_string();
    let output_file = std::fs::File::create(&args.output)
        .with_context(|| format!("Creating output file {:?}", args.output))?;
    serde_json::to_writer_pretty(output_file, &output_lm)
        .with_context(|| format!("Writing labelme data to {:?}", args.output))?;
    println!("Saved reconstructed labelme data to {:?}", args.output);
    Ok(())
}

fn add_env_config(
    builder: config::ConfigBuilder<config::builder::DefaultState>,
) -> config::ConfigBuilder<config::builder::DefaultState> {
    builder.add_source(
        config::Environment::with_prefix("ASM")
            .separator("__")
            .list_separator(",")
            .try_parsing(true),
    )
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
