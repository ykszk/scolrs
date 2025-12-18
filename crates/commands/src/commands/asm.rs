use anyhow::Context;
use ndarray::{s, Array1, Array2, Axis};

use crate::cli::{AsmArgs, AsmConfigArgs, AsmFitArgs, AsmReconstructArgs, AsmSubCommands};
use labelme_rs::{LabelMeData, Shape};
use ndarray_ndimage as ndi;
use scolrs::asm::{
    self,
    adam::{Stopper, StopperConfig},
    fit::{fit_asm_to_heatmap, FitConfig},
    model::ActiveShapeModel,
};
use serde::{Deserialize, Serialize};
use serde_json;

pub fn cmd(args: AsmArgs) -> anyhow::Result<()> {
    match args.command {
        AsmSubCommands::Fit(args) => cmd_fit(args),
        AsmSubCommands::Recon(args) => cmd_recon(args),
        AsmSubCommands::Config(args) => cmd_config(args),
    }
}

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

fn create_shapes_from_fitted_points(fitted_shape: &[Array2<f64>], labels: &[String]) -> Vec<Shape> {
    let mut shapes = Vec::new();
    for (points, label) in fitted_shape.iter().zip(labels.iter()) {
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
    shapes
}

fn smooth_heatmaps(
    heatmaps: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 3]>>,
    sigma: f64,
) -> ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 3]>> {
    let mut smoothed_heatmaps = heatmaps.clone();
    for (ch_idx, channel) in heatmaps.axis_iter(Axis(2)).enumerate() {
        let smoothed = ndi::gaussian_filter(&channel, sigma, 0, ndi::BorderMode::Mirror, 3);
        smoothed_heatmaps
            .index_axis_mut(ndarray::Axis(2), ch_idx)
            .assign(&smoothed);
    }
    smoothed_heatmaps
}

pub fn cmd_fit(args: AsmFitArgs) -> anyhow::Result<()> {
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
        crate::cli::ChannelOrder::Last => heatmaps,
        crate::cli::ChannelOrder::First => {
            log::debug!("Permuting heatmap axes from (C, H, W) to (H, W, C)");
            heatmaps.permuted_axes([1, 2, 0])
        }
    };
    let n_required_channels = asm.labels.len();
    let heatmaps = if heatmaps.len_of(Axis(2)) > n_required_channels {
        log::info!(
            "Heatmaps have {0} channels, but ASM requires only {1} channels. Using the first {1} channels.",
            heatmaps.len_of(Axis(2)),
            n_required_channels,
        );
        heatmaps.slice_move(s![.., .., 0..n_required_channels])
    } else if heatmaps.len_of(Axis(2)) < n_required_channels {
        anyhow::bail!(
            "Heatmaps have {} channels, but ASM requires {} channels.",
            heatmaps.len_of(Axis(2)),
            n_required_channels
        );
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
    let asm_movable_points = asm.to_movable_points();

    log::debug!(
        "Calculating transform to align ASM {:?} to reference points {:?}",
        asm.template_reference,
        ref_points
    );
    let tr_asm_to_ref = asm_movable_points.calculate_transform_with_missing(&ref_points, true)?;
    log::debug!("Transform ASM to reference: {:?}", tr_asm_to_ref);
    asm.global_transform(&tr_asm_to_ref);

    let mut config_builder =
        config::Config::builder().add_source(config::Config::try_from(&AsmConfig::default())?);
    if let Some(config_path) = args.config {
        log::debug!("Loading ASM fitting configuration from {:?}", config_path);
        config_builder = config_builder.add_source(config::File::from(config_path.as_path()))
    }
    let config_builder = add_env_config(config_builder).build()?;

    let asm_config: AsmConfig = config_builder.try_deserialize()?;
    log::debug!("ASM fitting configuration: {:?}", asm_config);

    let n_mode = asm.calculate_mode(asm_config.mode);

    let mut initial_params = Array1::zeros(n_mode);
    if asm_config.sigmas.is_empty() {
        unreachable!("At least one sigma value must be provided for heatmap smoothing.")
    };
    let mut histories = Vec::new();
    for sigma in asm_config.sigmas {
        let mut stopper = Stopper::from_config(asm_config.stopper.clone());
        log::info!("Fitting ASM with heatmap smoothing sigma = {}", sigma);
        let smoothed_heatmaps = if sigma > 0.0 {
            let smoothed = smooth_heatmaps(&heatmaps, sigma);
            log::debug!("Smoothed heatmaps with sigma {}", sigma);
            smoothed
        } else {
            heatmaps.clone()
        };
        let (fitted_params, obj_history) = fit_asm_to_heatmap(
            &asm,
            initial_params.view(),
            &mut stopper,
            smoothed_heatmaps.view(),
            asm_config.fit.clone(),
        );
        log::info!(
            "Fitting with sigma {} completed in {} iterations.",
            sigma,
            obj_history.data_objectives.len()
        );
        // Update initial_params for next sigma
        initial_params.assign(&fitted_params);
        histories.push(obj_history);
    }
    let optimal_params = initial_params;

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
        output_lm.shapes = create_shapes_from_fitted_points(&fitted_shape, &asm.labels);
        output_lm.scale(scale_hm_to_ref);
        output_lm.imageWidth = ref_lm.imageWidth;
        output_lm.imageHeight = ref_lm.imageHeight;
        output_lm.version = scolrs::VERSION.to_string();
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsmConfig {
    pub fit: FitConfig,
    pub stopper: StopperConfig,
    pub sigmas: Vec<f64>,
    pub mode: asm::model::ModeConfig,
}

impl Default for AsmConfig {
    fn default() -> Self {
        let sigmas = vec![10.0, 5.0, 0.1];
        Self {
            sigmas,
            fit: FitConfig::default(),
            stopper: StopperConfig::default(),
            mode: asm::model::ModeConfig::default(),
        }
    }
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
