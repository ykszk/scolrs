use std::path::PathBuf;
use std::vec;

use anyhow::{Context, Result};
use clap::{Args, Parser, ValueEnum};
use deepscol::CroppingParams;
use image::GenericImageView;
use labelme_rs::LabelMeDataWImage;
use ndarray::Axis;
use scolrs::{asm::model::ActiveShapeModel, draw::output3_to_heatmap};

#[derive(Debug, Clone, Default, ValueEnum)]
enum Direction {
    #[default]
    #[clap(alias = "ap")]
    Coronal,
    #[clap(alias = "lateral")]
    Sagittal,
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

#[derive(Args, Debug, Clone, Default)]
pub struct AsmArgs {
    /// Model in json
    #[clap(long)]
    asm: PathBuf,
    /// Config file for ASM fitting
    #[clap(long)]
    config: PathBuf,
}

#[derive(Parser)]
#[clap(name=env!("CARGO_BIN_NAME"), author, version = scolrs::VERSION, about, long_about = None)]
struct CmdArgs {
    /// Path to the input image file
    input_image: PathBuf,
    /// Text file of labels for the points in LabelMe format
    #[arg(long)]
    labels: Option<PathBuf>,
    /// Path to the onnx model file
    #[arg(long)]
    model: Option<PathBuf>,
    /// Direction of the image
    #[arg(short, long, value_enum, default_value_t = Direction::Coronal)]
    direction: Direction,
    /// Active shape model file
    #[clap(flatten)]
    asm: Option<AsmArgs>,
    #[command(flatten)]
    output: OutputGroup,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = CmdArgs::parse();
    let model_path = &args.model;
    let image_path = &args.input_image;
    // ort::set_api(ort_tract::api());
    let builder = ort::session::Session::builder()
        .expect("Cannot create Session builder.")
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Disable)
        .expect("Cannot optimize graph.")
        .with_parallel_execution(true)
        .expect("Cannot activate parallel execution.")
        .with_intra_threads(2)
        .expect("Cannot set intra thread count.")
        .with_inter_threads(1)
        .expect("Cannot set inter thread count.");
    let mut session = if let Some(model_path) = model_path {
        builder
            .commit_from_file(model_path)
            .expect("Cannot load model from file.")
    } else {
        builder
            .commit_from_memory(include_bytes!("../models/spine_mobileone_s1.onnx"))
            .expect("Cannot load model from memory.")
    };
    log::info!("Model loaded successfully");
    log::debug!(
        "model input dimensions {:?}",
        session.inputs[0].input_type.tensor_shape().unwrap()
    );

    let (image, metadata) = deepscol::load_image(&std::fs::read(image_path)?)?;
    let original_image_width = image.width();
    let original_image_height = image.height();
    log::info!(
        "Image with shape w x h {:?} loaded successfully",
        (original_image_width, original_image_height)
    );
    log::debug!("Expected model input {:?}", session);
    let model_input_height = session.inputs[0].input_type.tensor_shape().unwrap()[2] as u32;
    let mut lm_scale = original_image_height as f64 / model_input_height as f64;
    let arr4 = deepscol::to_model_input(image.clone(), model_input_height)?;
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
    let mut output3 = deepscol::extract_array_from_output(outputs);

    let cropping_params = deepscol::calculate_crop_parameters(
        &output3,
        0.05,
        original_image_height,
        original_image_width,
        model_input_height,
    );

    let mut crop_min_xy = None;
    let mut input_image_wh = (original_image_width, original_image_height);

    if let Some(cropping_params) = cropping_params {
        let CroppingParams {
            min_x,
            min_y,
            max_x,
            max_y,
        } = cropping_params;
        log::info!(
            "cropping image to bounding box: ({}, {}, {}, {})",
            min_x,
            min_y,
            max_x,
            max_y
        );
        crop_min_xy = Some((min_x, min_y));
        // crop the image
        let cropped_image = image.crop_imm(
            min_x as u32,
            min_y as u32,
            (max_x - min_x) as u32,
            (max_y - min_y) as u32,
        );
        input_image_wh = (cropped_image.width(), cropped_image.height());
        let arr4 = deepscol::to_model_input(cropped_image, model_input_height)?;
        lm_scale = (max_y - min_y) as f64 / model_input_height as f64;

        // update the input tensor
        let input3 = ort::value::Tensor::from_array(arr4)?;
        // update the inputs
        let inputs = ort::inputs!["modelInput" => input3];
        // run the model again
        log::info!("Running model on cropped input");
        let outputs: ort::session::SessionOutputs = session.run(inputs)?;
        log::info!("Model run completed on cropped input");
        // convert to ndarray
        output3 = deepscol::extract_array_from_output(outputs);
    } else {
        log::info!("No cropping applied");
    }

    if let Some(output_path) = args.output.heatmap {
        let output_path = output_path.unwrap_or_else(|| {
            let mut p = image_path.clone();
            log::debug!("Generating heatmap path from image path: {:?}", p);
            p.set_extension("png");
            if p == *image_path {
                p.set_extension("heatmap.png");
            }
            p
        });
        if output_path.extension().and_then(|s| s.to_str()) == Some("npz") {
            let mut npz =
                ndarray_npz::NpzWriter::new_compressed(std::fs::File::create(output_path)?);
            //  convert ort's ndarray (v0.15.6) to ndarray_npz's ndarray (v0.16.1). Can be removed when these crates are updated.
            let ndarray_output3 = ndarray_npz::ndarray::Array3::from_shape_vec(
                (output3.shape()[0], output3.shape()[1], output3.shape()[2]),
                output3.clone().into_raw_vec_and_offset().0,
            )?;
            let ndarray_output3 = ndarray_output3.mapv(|x| (x * 1000.0) as u16);
            // save the output
            npz.add_array("heatmaps", &ndarray_output3)?;
            npz.finish()?;
        } else {
            let heatmap = output3_to_heatmap(output3.view(), input_image_wh);
            heatmap
                .save(output_path)
                .expect("Failed to save output image");
        }
    }
    let abs_path = image_path
        .canonicalize()
        .expect("Failed to get absolute path of image")
        .to_string_lossy()
        .to_string();
    let thresh = 0.1;

    let mut points = deepscol::extract_points(&output3, thresh).unwrap();
    // Save the points to LabelMe format
    let mut lm_data = deepscol::create_scaled_lm(
        &deepscol::LABELS,
        abs_path.clone(),
        original_image_width,
        original_image_height,
        lm_scale,
        points.as_slice(),
    );

    // check spine point counts
    if let Err(e) = scolrs::C7TLS::check_counts(&lm_data) {
        if let Some(asm_args) = &args.asm {
            log::warn!(
                "Point counts are invalid: {}. Attempting to fit ASM model.",
                e
            );
            let reader = std::fs::File::open(&asm_args.asm)
                .with_context(|| format!("Opening ASM model file {:?}", asm_args.asm))?;
            let asm: ActiveShapeModel = serde_json::from_reader(reader)
                .with_context(|| format!("Loading ASM model from {:?}", asm_args.asm))?;
            let config_builder = config::Config::builder()
                .add_source(config::Config::try_from(&scolrs::asm::AsmConfig::default())?)
                .add_source(config::File::from(asm_args.config.as_path()))
                .build()?;
            let asm_config: scolrs::asm::AsmConfig = config_builder.try_deserialize()?;
            log::debug!("ASM fitting configuration: {:?}", asm_config);

            let heatmap_lm = deepscol::create_lm(
                &deepscol::LABELS,
                abs_path.clone(),
                output3.shape()[2] as u32,
                output3.shape()[1] as u32,
                points.as_slice(),
            );

            let output3_channel_last = output3.mapv(|x| x as f64).permuted_axes((1, 2, 0));
            let (_fitted_lm_data, _histories, _optimal_params, fitted_shape) =
                scolrs::asm::apply_asm(asm, asm_config, output3_channel_last.view(), &heatmap_lm)?;
            let fitted_points = fitted_shape
                .iter()
                .map(|arr2| {
                    arr2.axis_iter(Axis(0))
                        .map(|pt| (pt[0] as f32, pt[1] as f32))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            // update points with fitted points
            for (i, fitted_shape) in fitted_points.into_iter().enumerate() {
                points[i] = fitted_shape;
            }
            log::debug!("Fitted points from ASM: {:?}", points);
            let fitted_lm_data = deepscol::create_scaled_lm(
                &deepscol::LABELS,
                abs_path,
                original_image_width,
                original_image_height,
                lm_scale,
                points.as_slice(),
            );
            lm_data = fitted_lm_data;
            log::info!("ASM model fitting completed.");
        } else {
            log::error!(
                "Point counts are invalid: {}. No ASM model provided, skipping ASM fitting.",
                e
            );
        }
    }

    if let Some(crop_min_xy) = crop_min_xy {
        // Update the points to be relative to the cropped image
        log::debug!("Shifting points by ({}, {})", crop_min_xy.0, crop_min_xy.1);
        lm_data.shift(crop_min_xy.0 as f64, crop_min_xy.1 as f64);
    }

    if let Some(labelme_path) = args.output.labelme {
        let labelme_path = labelme_path.unwrap_or_else(|| {
            let mut p = image_path.clone();
            log::debug!("Generating LabelMe path from image path: {:?}", p);
            p.set_extension("json");
            p
        });
        // Save the LabelMe data to a file
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
        let heatmap = output3_to_heatmap(output3.view(), input_image_wh);
        log::debug!("Heatmap image shape: {:?}", heatmap.dimensions());
        let scan_direction = match args.direction {
            Direction::Coronal => deepscol::ScanDirection::Coronal,
            Direction::Sagittal => deepscol::ScanDirection::Sagittal,
        };
        let title = format!(
            "{} - Deepscol",
            image_path.file_stem().unwrap_or_default().to_string_lossy()
        );
        let lm_data_w_image = LabelMeDataWImage::new(lm_data, image);
        let html_args = deepscol::ResultHtmlLmArgs {
            lm_data_w_image,
            model_input_height,
            metadata,
            scan_direction,
            cropping_params,
            heatmap,
            model_output: output3,
            title,
        };
        let html = deepscol::create_result_html_from_lm(html_args)?;
        // Save the HTML to a file
        std::fs::write(html_path, html).expect("Failed to write HTML file");
    }
    Ok(())
}
