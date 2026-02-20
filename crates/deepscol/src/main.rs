use std::ffi::OsStr;
use std::path::PathBuf;
use std::vec;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use deepscol::{self as ds, SessionConfig};
use deepscol::{CroppedIO, CroppingParams, ModelIO, OriginalIO};
use metaimage::WriteMhd;
use ort::execution_providers::ExecutionProvider;
use ort::session::builder::SessionBuilder;
use ort::session::Session;
use scolrs::asm::AddEnvConfig;
use scolrs::asm::{model::ActiveShapeModel, AsmConfig};

#[derive(Debug, Clone, Default, ValueEnum)]
enum Direction {
    #[default]
    #[clap(alias = "ap")]
    Coronal,
    #[clap(alias = "lateral")]
    Sagittal,
    #[clap(alias = "neck")]
    NeckLateral,
}

#[derive(clap::Args)]
#[group(required = true, multiple = true)]
struct OutputGroup {
    /// Path to save the output heatmaps or image
    #[arg(long)]
    heatmap: Option<Option<PathBuf>>,
    /// Save LabelMe format points
    #[arg(long)]
    labelme: Option<Option<PathBuf>>,
    /// Output html
    #[arg(long)]
    html: Option<Option<PathBuf>>,
}

fn parse_path_optional_pair(s: &str) -> Result<(PathBuf, Option<PathBuf>), String> {
    let parts: Vec<&str> = s.split(',').collect();
    match parts.len() {
        1 => Ok((PathBuf::from(parts[0]), None)),
        2 => Ok((PathBuf::from(parts[0]), Some(PathBuf::from(parts[1])))),
        _ => Err(format!(
            "Expected 1 or 2 paths separated by comma, got {} with {}",
            parts.len(),
            s
        )),
    }
}

#[derive(Parser)]
#[clap(name=env!("CARGO_BIN_NAME"), author, version = scolrs::VERSION, about, long_about = None)]
struct CmdArgs {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process a single image
    Single(SingleCmdArgs),
    /// Process a batch of images specified in a text file
    Batch(BatchCmdArgs),
    /// Print default configuration
    Config(ConfigArgs),
}

#[derive(ValueEnum, Debug, Clone)]
enum Accelerator {
    Cuda,
}

impl Accelerator {
    fn to_ep(&self) -> Result<Box<dyn ExecutionProvider>> {
        match self {
            Accelerator::Cuda => Ok(Box::new(
                ort::execution_providers::cuda::CUDAExecutionProvider::default(),
            )),
        }
    }
}

#[derive(Parser, Clone)]
struct CommonArgs {
    /// Path to the onnx model file
    #[arg(long)]
    model: Option<PathBuf>,
    /// Direction of the image
    #[arg(short, long, value_enum, default_value_t = Direction::Coronal)]
    direction: Direction,
    /// Path to the ASM model(s) file for fitting (and configuration optionally)
    #[arg(long, value_parser = parse_path_optional_pair)]
    asm: Vec<(PathBuf, Option<PathBuf>)>,
    /// Configuration file (TOML or JSON). See `config` subcommand for the default configuration.
    /// Additionally, use environment variables with prefix `DS__` (e.g. `DS__HEATMAP__THRESH=0.5`).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Accelerators
    #[arg(short, long)]
    accelerators: Vec<Accelerator>,
}

#[derive(Parser)]
struct SingleCmdArgs {
    /// Path to the input image file
    input_image: PathBuf,
    #[command(flatten)]
    common: CommonArgs,
    #[command(flatten)]
    output: OutputGroup,
}

#[derive(clap::Args)]
#[group(required = true, multiple = true)]
struct BatchOutputGroup {
    /// Path to save the output heatmaps or image
    #[arg(long)]
    heatmap: bool,
    /// Save LabelMe format points
    #[arg(long)]
    labelme: bool,
    /// Output html
    #[arg(long)]
    html: bool,
}

#[derive(Parser)]
struct BatchCmdArgs {
    /// Text file containing paths to the input image files, one per line
    input_images_list: PathBuf,
    #[command(flatten)]
    common: CommonArgs,
    /// Output directory to save the results
    output_dir: PathBuf,
    #[command(flatten)]
    output: BatchOutputGroup,
}

#[derive(ValueEnum, Debug, Clone)]
enum ConfigFormat {
    Json,
    Toml,
}

#[derive(Parser)]
struct ConfigArgs {
    /// Output format for the default configuration (json or yaml)
    #[arg(default_value = "toml")]
    format: ConfigFormat,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = CmdArgs::parse();
    match args.command {
        Commands::Single(single_args) => single_cmd(single_args),
        Commands::Batch(batch_args) => batch_cmd(batch_args),
        Commands::Config(config_args) => config_cmd(config_args),
    }
}

fn single_cmd(args: SingleCmdArgs) -> Result<()> {
    let ds_config = load_config(&args.common.config)?;
    let session = load_model(
        &args.common.model,
        &args.common.accelerators,
        &ds_config.session,
    )?;
    process_image(args, session, &ds_config)?;
    Ok(())
}

fn load_config(path: &Option<PathBuf>) -> Result<ds::DeepscolConfig> {
    let mut builder = config::Config::builder()
        .add_source(config::Config::try_from(&ds::DeepscolConfig::default())?);
    if let Some(config_path) = path.as_ref() {
        log::info!("Loading configuration from file: {:?}", config_path);
        builder = builder.add_source(config::File::from(config_path.as_path()));
    } else {
        log::debug!("No configuration file provided, using default configuration and environment variables.");
    };
    let config = builder
        .add_source(
            config::Environment::with_prefix("DS")
                .separator("__")
                .list_separator(",")
                .try_parsing(true),
        )
        .build()?
        .try_deserialize()?;
    log::debug!("Deepscol configuration: {:?}", config);
    Ok(config)
}

fn process_image(
    args: SingleCmdArgs,
    mut session: Session,
    ds_config: &ds::DeepscolConfig,
) -> Result<Session> {
    let image_path = &args.input_image;

    let (image, metadata) = ds::load_image_from_path(image_path)
        .with_context(|| format!("Failed to load image from path {:?}", image_path))?;
    let original_image_width = image.width();
    let original_image_height = image.height();
    log::info!(
        "Image with shape w x h {:?} loaded successfully",
        (original_image_width, original_image_height)
    );
    log::debug!("Expected model input {:?}", session);
    let model_input_height = session.inputs[0].input_type.tensor_shape().unwrap()[2] as u32;
    let arr4 = ds::to_model_input(&image, model_input_height)?;
    let input = ort::value::Tensor::from_array(arr4)?;
    log::debug!(
        "Image converted to tensor successfully with shape: {:?}",
        input.shape()
    );
    // Run the model
    // let inputs = vec![Value::from_array(session.allocator(), &input)?];
    let inputs = ort::inputs!["modelInput" => input];
    log::info!("Running model");
    let outputs: ort::session::SessionOutputs = session.run(inputs)?;
    log::info!("Model run completed");

    // convert to ndarray
    let mut output3 = ds::extract_array_from_output(outputs);

    let cropping_params = ds::calculate_crop_parameters(
        &output3,
        &ds_config.crop,
        original_image_height,
        original_image_width,
        model_input_height,
    );

    let point_thresh = ds_config.heatmap.thresh;

    let point_set_config = match args.common.direction {
        Direction::Coronal => ds::point_config::PointSetConfig::spine(),
        Direction::Sagittal => ds::point_config::PointSetConfig::spine(),
        Direction::NeckLateral => ds::point_config::PointSetConfig::neck_lateral(),
    };

    let mut model_io = if let Some(cropping_params) = cropping_params {
        log::info!("cropping image to bounding box: {:?}", cropping_params);
        let CroppingParams {
            min_x,
            min_y,
            max_x,
            max_y,
        } = cropping_params;
        // crop the image
        let cropped_image = image.crop_imm(
            min_x as u32,
            min_y as u32,
            (max_x - min_x) as u32,
            (max_y - min_y) as u32,
        );
        let arr4 = ds::to_model_input(&cropped_image, model_input_height)?;

        // update the input tensor
        let input3 = ort::value::Tensor::from_array(arr4)?;
        // update the inputs
        let inputs = ort::inputs!["modelInput" => input3];
        // run the model again
        log::info!("Running model on cropped input");
        let outputs: ort::session::SessionOutputs = session.run(inputs)?;
        log::info!("Model run completed on cropped input");
        // convert to ndarray
        output3 = ds::extract_array_from_output(outputs);
        let points = ds::extract_points(&output3, point_thresh, &point_set_config.max_counts)
            .map_err(|e| anyhow::anyhow!("Failed to extract points from cropped output: {}", e))?;

        ModelIO::Cropped(Box::new(CroppedIO::new(
            image,
            metadata.path.clone(),
            output3,
            points,
            cropping_params,
            &point_set_config,
        )))
    } else {
        log::info!("No cropping applied");
        let points = ds::extract_points(&output3, point_thresh, &point_set_config.max_counts)
            .map_err(|e| anyhow::anyhow!("Failed to extract points from cropped output: {}", e))?;
        ModelIO::Original(Box::new(OriginalIO::new(
            image,
            metadata.path.clone(),
            output3,
            points,
            &point_set_config,
        )))
    };

    if let Some(output_path) = args.output.heatmap {
        let output_path = output_path.unwrap_or_else(|| {
            let mut p = image_path.clone();
            log::debug!("Generating heatmap path from image path: {:?}", p);
            p.set_extension(".mha");
            if p == *image_path {
                p.set_extension("heatmap.mha");
            }
            p
        });
        let file_ext = output_path
            .extension()
            .and_then(OsStr::to_str)
            .unwrap_or("")
            .to_lowercase();
        if file_ext == "npz" {
            let mut npz =
                ndarray_npz::NpzWriter::new_compressed(std::fs::File::create(output_path)?);
            let ndarray_output3 = model_io.output3().mapv(|x| (x * 1000.0) as u16);
            // save the output
            npz.add_array("heatmaps", &ndarray_output3)?;
            npz.finish()?;
        } else if file_ext == "mha" || file_ext == "mhd" {
            let arr = model_io.output3();
            // convert to metaimage's ndarray (v0.17). Can be removed the version conflict is resolved.
            let ndarray_output3 = metaimage::ndarray::Array3::from_shape_vec(
                (arr.shape()[2], arr.shape()[1], arr.shape()[0]),
                arr.clone().into_raw_vec_and_offset().0,
            )?;
            metaimage::MetaImage::write_mhd(ndarray_output3.view(), &output_path)?
        } else {
            anyhow::bail!("Unsupported heatmap output format: {:?}", file_ext);
        }
    }
    let scan_direction = match args.common.direction {
        Direction::Coronal => ds::ScanDirection::Coronal,
        Direction::Sagittal => ds::ScanDirection::Sagittal,
        Direction::NeckLateral => ds::ScanDirection::NeckLateral,
    };
    // check spine point counts and it's spine (i.e. its_spine_not_neck == true)
    if let Err(e) = scan_direction.check_counts(model_io.lm_data()) {
        log::info!("Point counts are invalid: {}", e);
        let output3_f64 = model_io.output3().mapv(|x| x as f64);
        let mut asms = Vec::new();
        for (asm_path, asm_config) in &args.common.asm {
            let reader = std::io::BufReader::new(
                std::fs::File::open(asm_path)
                    .with_context(|| format!("Opening ASM model file {:?}", asm_path))?,
            );
            let asm: ActiveShapeModel = serde_json::from_reader(reader)
                .with_context(|| format!("Loading ASM model from {:?}", asm_path))?;

            let asm_config: AsmConfig = if let Some(asm_config_path) = asm_config.as_ref() {
                let config_builder = config::Config::builder()
                    .add_source(config::File::from(asm_config_path.as_path()))
                    .add_asm_env_source()
                    .build()?;
                config_builder.try_deserialize()?
            } else {
                log::debug!("No ASM config associated to the model provided, using default configuration and environment variables.");
                let config_builder = config::Config::builder()
                    .add_source(config::Config::try_from(&AsmConfig::default())?);
                let config_builder = config_builder.add_asm_env_source().build()?;
                config_builder.try_deserialize()?
            };
            log::debug!("ASM fitting configuration: {:?}", asm_config);
            asms.push((asm, asm_config));
        }
        let result_best_asm_lm_data =
            ds::apply_asms(&model_io, output3_f64.view(), &point_set_config, asms);
        match result_best_asm_lm_data {
            Err(e) => {
                log::warn!("Failed to apply ASM models: {}", e);
            }
            Ok(best_asm_lm_data) => {
                if let Some(best_lm_data) = best_asm_lm_data {
                    model_io.set_heatmap_lm_data(best_lm_data);
                } else {
                    log::warn!("No ASM model provided, skipping ASM fitting.");
                }
            }
        }
    }

    if let Some(labelme_path) = args.output.labelme {
        let labelme_path = labelme_path.unwrap_or_else(|| {
            let mut p = image_path.clone();
            log::debug!("Generating LabelMe path from image path: {:?}", p);
            p.set_extension("json");
            p
        });
        // Save the LabelMe data to a file
        let mut lm_data = model_io.lm_data().to_owned();
        let abs_path = image_path
            .canonicalize()
            .expect("Failed to get absolute path of image")
            .to_string_lossy()
            .to_string();
        lm_data.imagePath = abs_path;
        let labelme_json = serde_json::to_string_pretty(&lm_data)?;

        std::fs::write(&labelme_path, labelme_json).with_context(|| {
            format!(
                "Failed to write LabelMe data to file: {}",
                labelme_path.display()
            )
        })?;
    }
    if let Some(html_path) = args.output.html {
        let html_path = html_path.unwrap_or_else(|| {
            let mut p = image_path.clone();
            log::debug!("Generating HTML path from image path: {:?}", p);
            p.set_extension("html");
            p
        });

        let title = format!(
            "{} - Deepscol",
            image_path.file_stem().unwrap_or_default().to_string_lossy()
        );
        let html_args = ds::ResultHtmlLmArgs {
            model_io,
            metadata,
            scan_direction,
            size_config: ds_config.size.clone(),
            title,
        };
        let html = ds::create_result_html_from_lm(html_args)?;
        // Save the HTML to a file
        std::fs::write(html_path, html).expect("Failed to write HTML file");
    }
    Ok(session)
}

fn batch_cmd(args: BatchCmdArgs) -> Result<()> {
    let model_path = &args.common.model;
    let ds_config = load_config(&args.common.config)?;
    let mut session = load_model(model_path, &args.common.accelerators, &ds_config.session)?;

    let input_images_list =
        std::fs::read_to_string(&args.input_images_list).with_context(|| {
            format!(
                "Failed to read input images list from {:?}",
                args.input_images_list
            )
        })?;
    let image_paths: Vec<PathBuf> = input_images_list
        .lines()
        .map(|line| PathBuf::from(line.trim()))
        .collect();
    log::info!("Found {} images to process", image_paths.len());

    for image_path in image_paths {
        log::info!("Processing image: {:?}", image_path);
        let file_stem = image_path.file_stem().unwrap_or_default().to_string_lossy();
        let heatmap = if args.output.heatmap {
            let path = args.output_dir.join(format!("{}.mha", file_stem));
            Some(Some(path))
        } else {
            None
        };
        let labelme = if args.output.labelme {
            let path = args.output_dir.join(format!("{}.json", file_stem));
            Some(Some(path))
        } else {
            None
        };
        let html = if args.output.html {
            let path = args.output_dir.join(format!("{}.html", file_stem));
            Some(Some(path))
        } else {
            None
        };
        let single_args = SingleCmdArgs {
            input_image: image_path,
            common: args.common.clone(),
            output: OutputGroup {
                heatmap,
                labelme,
                html,
            },
        };
        session = process_image(single_args, session, &ds_config)?;
    }
    Ok(())
}

fn set_builder_config(builder: SessionBuilder, config: &SessionConfig) -> Result<SessionBuilder> {
    let builder = builder.with_parallel_execution(config.parallel_execution)?;
    let builder = if let Some(intra_threads) = config.intra_threads {
        builder.with_intra_threads(intra_threads)?
    } else {
        builder
    };
    let builder = if let Some(inter_threads) = config.inter_threads {
        builder.with_inter_threads(inter_threads)?
    } else {
        builder
    };
    Ok(builder)
}

fn load_model(
    model_path: &Option<PathBuf>,
    accelerators: &[Accelerator],
    session_config: &SessionConfig,
) -> Result<Session> {
    let builder = Session::builder()
        .expect("Cannot create Session builder.")
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Disable)
        .expect("Cannot optimize graph.");
    let mut builder = set_builder_config(builder, session_config)?;
    let eps = accelerators
        .iter()
        .map(|acc| {
            log::info!("Adding execution provider for accelerator: {:?}", acc);
            acc.to_ep()
        })
        .collect::<Result<Vec<Box<dyn ExecutionProvider>>>>()?;
    log::info!("Using execution providers: {:?}", accelerators);
    for ep in eps {
        let reg_result = ep.register(&mut builder);
        if let Err(e) = reg_result {
            log::warn!("Failed to register execution provider: {}", e);
        }
    }
    let session = if let Some(model_path) = model_path {
        log::info!("Loading model from file: {:?}", model_path);
        builder
            .commit_from_file(model_path)
            .expect("Cannot load model from file.")
    } else {
        log::info!("Loading bundled model from memory");
        builder
            .commit_from_memory(include_bytes!("../models/spine_mobileone_s1.onnx"))
            .expect("Cannot load model from memory.")
    };
    log::info!("Model loaded successfully");
    log::debug!(
        "model input dimensions {:?}",
        session.inputs[0].input_type.tensor_shape().unwrap()
    );
    Ok(session)
}

fn config_cmd(args: ConfigArgs) -> Result<()> {
    let ds_config = ds::DeepscolConfig::default();
    let output = match args.format {
        ConfigFormat::Json => serde_json::to_string_pretty(&ds_config)?,
        ConfigFormat::Toml => toml::to_string_pretty(&ds_config)?,
    };
    println!("{}", output);
    Ok(())
}
