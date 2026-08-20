use std::ffi::OsStr;
use std::path::PathBuf;
use std::vec;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use deepscol::CroppingParams;
use deepscol::{self as ds, SessionConfig};
use metaimage::WriteMhd;
use ndarray::{Array3, Array4};
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
    /// Multiplier for heatmap values when saving as image (e.g. 10000 to convert from [0,1] to [0,10000] range). Ignored if saving as .npy or .mha.
    #[arg(long, default_value_t = 10000)]
    heatmap_multiplier: u16,
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
    /// Path to save the output heatmaps or image. Optionally, specify a multiplier
    #[arg(long, num_args=0..=1, default_missing_value = "10000")]
    heatmap: Option<u16>,
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
    /// Number of concurrent inference workers
    #[arg(long, default_value_t = 1)]
    inference_workers: usize,
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

/// An image loaded from disk and converted to the model input tensor. Produced
/// by the preprocessing stage, which touches no ONNX session and so can run in
/// parallel across images.
struct Preprocessed {
    image_path: PathBuf,
    image: image::DynamicImage,
    metadata: scolrs::ImageMetadata,
    model_input_height: u32,
    input: Array4<f32>,
}

/// The raw heatmap output for an image. Produced by the inference stage, which
/// owns the (single) ONNX session. Everything downstream is session-free.
struct Inferred {
    image_path: PathBuf,
    image: image::DynamicImage,
    metadata: scolrs::ImageMetadata,
    output3: Array3<f32>,
    cropping_params: Option<CroppingParams>,
}

/// Stage 1: load the image from disk and build the model input tensor. No
/// session involved, so this runs in parallel across images.
fn preprocess_image(image_path: PathBuf, model_input_height: u32) -> Result<Preprocessed> {
    let (image, metadata) = ds::load_image_from_path(&image_path)
        .with_context(|| format!("Failed to load image from path {:?}", image_path))?;
    log::info!(
        "Image with shape w x h {:?} loaded successfully",
        (image.width(), image.height())
    );
    let input = ds::to_model_input(&image, model_input_height, None)?;
    log::debug!(
        "Image converted to input tensor with shape: {:?}",
        input.shape()
    );
    Ok(Preprocessed {
        image_path,
        image,
        metadata,
        model_input_height,
        input,
    })
}

/// Stage 2: run the model (detection pass, then an optional crop-and-rerun).
/// This is the only stage that needs the session, so it runs single-threaded.
fn infer_image(
    session: &mut Session,
    pre: Preprocessed,
    ds_config: &ds::DeepscolConfig,
) -> Result<Inferred> {
    let Preprocessed {
        image_path,
        image,
        metadata,
        model_input_height,
        input,
    } = pre;
    let original_image_width = image.width();
    let original_image_height = image.height();

    let input = ort::value::Tensor::from_array(input)?;
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

    if let Some(cropping_params) = cropping_params {
        let arr4 = ds::to_model_input(&image, model_input_height, Some(cropping_params))?;
        let input3 = ort::value::Tensor::from_array(arr4)?;
        let inputs = ort::inputs!["modelInput" => input3];
        // run the model again
        log::info!("Running model on cropped input");
        let outputs: ort::session::SessionOutputs = session.run(inputs)?;
        log::info!("Model run completed on cropped input");
        output3 = ds::extract_array_from_output(outputs);
    } else {
        log::info!("No cropping applied");
    }

    Ok(Inferred {
        image_path,
        image,
        metadata,
        output3,
        cropping_params,
    })
}

fn load_asms(
    paths: &Vec<(PathBuf, Option<PathBuf>)>,
) -> Result<Vec<(ActiveShapeModel, AsmConfig)>> {
    let mut asms = Vec::new();
    for (asm_path, asm_config) in paths {
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
    Ok(asms)
}

/// Stage 3: turn the raw model output into points (with optional ASM fitting)
/// and write the requested outputs (heatmap / LabelMe / HTML). Session-free, so
/// this runs in parallel across images.
fn finish_image(
    inf: Inferred,
    output: OutputGroup,
    common: &CommonArgs,
    ds_config: &ds::DeepscolConfig,
) -> Result<()> {
    let Inferred {
        image_path,
        image,
        metadata,
        output3,
        cropping_params,
    } = inf;

    let scan_direction = match common.direction {
        Direction::Coronal => ds::ScanDirection::Coronal,
        Direction::Sagittal => ds::ScanDirection::Sagittal,
        Direction::NeckLateral => ds::ScanDirection::NeckLateral,
    };
    let model_io = deepscol::build_model_io(
        &metadata.path,
        image,
        output3,
        cropping_params,
        &ds_config.heatmap,
        &|| load_asms(&common.asm).map_err(|e| format!("Failed to load ASM models: {}", e)),
        scan_direction,
    )
    .map_err(|e| anyhow::anyhow!("Failed to build model IO: {}", e))?;

    if let Some(output_path) = output.heatmap {
        let output_path = output_path.unwrap_or_else(|| {
            let mut p = image_path.clone();
            log::debug!("Generating heatmap path from image path: {:?}", p);
            p.set_extension(".mha");
            if p == image_path {
                p.set_extension("heatmap.mha");
            }
            p
        });
        let file_ext = output_path
            .extension()
            .and_then(OsStr::to_str)
            .unwrap_or("")
            .to_lowercase();
        // apply multiplier and convert to u16
        let multiplier = output.heatmap_multiplier;
        let ndarray_output3 = model_io.output3().mapv(|x| (x * multiplier as f32) as u16);
        if file_ext == "mha" || file_ext == "mhd" {
            metaimage::MetaImage::write_mhd(ndarray_output3.view(), &output_path)?
        } else {
            anyhow::bail!("Unsupported heatmap output format: {:?}", file_ext);
        }
    }

    if let Some(labelme_path) = output.labelme {
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
    if let Some(html_path) = output.html {
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
    Ok(())
}

/// Process a single image end to end through all three pipeline stages. Used by
/// the `single` command; the `batch` command drives the same stages as a
/// pipeline (see [`batch_cmd`]).
fn process_image(
    args: SingleCmdArgs,
    mut session: Session,
    ds_config: &ds::DeepscolConfig,
) -> Result<Session> {
    let model_input_height = session.inputs[0].input_type.tensor_shape().unwrap()[2] as u32;
    let pre = preprocess_image(args.input_image, model_input_height)?;
    let inf = infer_image(&mut session, pre, ds_config)?;
    finish_image(inf, args.output, &args.common, ds_config)?;
    Ok(session)
}

/// Resolve the per-image output paths for the batch command from the requested
/// output flags and the output directory.
fn batch_output_paths(
    flags: &BatchOutputGroup,
    output_dir: &std::path::Path,
    file_stem: &str,
) -> OutputGroup {
    OutputGroup {
        heatmap: flags
            .heatmap
            .map(|_multiplier| Some(output_dir.join(format!("{}.mha", file_stem)))),
        heatmap_multiplier: flags.heatmap.unwrap_or(10000),
        labelme: flags
            .labelme
            .then(|| Some(output_dir.join(format!("{}.json", file_stem)))),
        html: flags
            .html
            .then(|| Some(output_dir.join(format!("{}.html", file_stem)))),
    }
}

fn batch_cmd(args: BatchCmdArgs) -> Result<()> {
    let ds_config = load_config(&args.common.config)?;

    // One independent session per inference worker; each loads its own copy of
    // the model weights so the workers never contend for a single session.
    let inference_workers = args.inference_workers.max(1);
    let mut sessions = Vec::with_capacity(inference_workers);
    for _ in 0..inference_workers {
        sessions.push(load_model(
            &args.common.model,
            &args.common.accelerators,
            &ds_config.session,
        )?);
    }
    let model_input_height = sessions[0].inputs[0].input_type.tensor_shape().unwrap()[2] as u32;

    let input_images_list =
        std::fs::read_to_string(&args.input_images_list).with_context(|| {
            format!(
                "Failed to read input images list from {:?}",
                args.input_images_list
            )
        })?;
    let image_paths: Vec<PathBuf> = input_images_list
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(PathBuf::from)
        .collect();
    log::info!("Found {} images to process", image_paths.len());

    // Three-stage pipeline: parallel preprocessing -> inference -> parallel
    // postprocessing + output writing. Inference is the only stage that needs a
    // (mutably borrowed) ONNX session; it runs on `inference_workers` threads,
    // each owning its own session. The surrounding disk I/O, image decoding and
    // HTML/ASM work overlaps inference instead of blocking it.
    let num_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let channel_cap = num_workers * 2;

    // Paths are tiny, so an unbounded channel is fine; the Preprocessed/Inferred
    // channels are bounded to bound memory (each item holds a full image).
    let (path_tx, path_rx) = std::sync::mpsc::channel::<PathBuf>();
    let (pre_tx, pre_rx) = std::sync::mpsc::sync_channel::<Preprocessed>(channel_cap);
    let (inf_tx, inf_rx) = std::sync::mpsc::sync_channel::<Inferred>(channel_cap);
    let path_rx = std::sync::Arc::new(std::sync::Mutex::new(path_rx));
    let pre_rx = std::sync::Arc::new(std::sync::Mutex::new(pre_rx));
    let inf_rx = std::sync::Arc::new(std::sync::Mutex::new(inf_rx));

    for image_path in image_paths {
        path_tx.send(image_path).expect("path channel closed");
    }
    drop(path_tx);

    let ds_config = &ds_config;
    let common = &args.common;
    let output_dir = &args.output_dir;
    let output_flags = &args.output;

    std::thread::scope(|scope| {
        // Stage 1: preprocessing workers (image load + tensor build).
        for _ in 0..num_workers {
            let path_rx = std::sync::Arc::clone(&path_rx);
            let pre_tx = pre_tx.clone();
            scope.spawn(move || loop {
                let Ok(image_path) = ({ path_rx.lock().unwrap().recv() }) else {
                    break;
                };
                log::info!("Processing image: {:?}", image_path);
                match preprocess_image(image_path, model_input_height) {
                    Ok(pre) => {
                        if pre_tx.send(pre).is_err() {
                            break;
                        }
                    }
                    Err(e) => log::error!("Failed to preprocess image: {:#}", e),
                }
            });
        }
        drop(pre_tx);

        // Stage 2: inference workers, each owning its own session.
        for mut session in sessions {
            let pre_rx = std::sync::Arc::clone(&pre_rx);
            let inf_tx = inf_tx.clone();
            scope.spawn(move || loop {
                let Ok(pre) = ({ pre_rx.lock().unwrap().recv() }) else {
                    break;
                };
                match infer_image(&mut session, pre, ds_config) {
                    Ok(inf) => {
                        if inf_tx.send(inf).is_err() {
                            break;
                        }
                    }
                    Err(e) => log::error!("Inference failed: {:#}", e),
                }
            });
        }
        drop(inf_tx);

        // Stage 3: postprocessing + output-writing workers.
        for _ in 0..num_workers {
            let inf_rx = std::sync::Arc::clone(&inf_rx);
            scope.spawn(move || loop {
                let Ok(inf) = ({ inf_rx.lock().unwrap().recv() }) else {
                    break;
                };
                let file_stem = inf
                    .image_path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                let output = batch_output_paths(output_flags, output_dir, &file_stem);
                if let Err(e) = finish_image(inf, output, common, ds_config) {
                    log::error!("Failed to process {}: {:#}", file_stem, e);
                }
            });
        }
    });

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
