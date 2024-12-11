use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::neck::cli::SvgArgs;
use anyhow::{Context, Result};
use labelme_rs::{image::GenericImageView, LabelMeData, LabelMeDataWImage};
use log::warn;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;
use scolrs::head_neck::{LateralPoints, NeckLateralDraw};
use scolrs::{
    draw::{draw_on_image, ColorPalette, ColorPalettes},
    DrawParam,
};
use svg::node::element::SVG;

fn process_data(
    mut lateral_points: LateralPoints,
    args: &SvgArgs,
    draw_param: &DrawParam,
    label_colors: ColorPalette,
    line_colors: ColorPalette,
    neck_sagittal_draw: &[NeckLateralDraw],
) -> Result<SVG> {
    // use LabelMeDataWImage for resizing
    let mut data =
        LabelMeDataWImage::try_from_data_and_path(LabelMeData::from(&lateral_points), &args.input)
            .with_context(|| format!("Failed to read {}", lateral_points.image_metadata.path))?;
    if let Some(resize) = args.resize.as_ref() {
        let resize_param = labelme_rs::ResizeParam::try_from(resize.as_str())?;
        data.resize(&resize_param);
    }
    // Return scaled points to lateral_points while keeping the original image data
    let original_image_data = lateral_points.image_metadata.clone();
    lateral_points = LateralPoints::try_from(&data.data)?;
    lateral_points.image_metadata = original_image_data;

    let svg_size = if let Some(size) = args.size.as_ref() {
        let size_param = labelme_rs::ResizeParam::try_from(size.as_str())?;
        data.data
            .scale(size_param.scale(data.image.width(), data.image.height()));
        size_param.size(data.image.width(), data.image.height())
    } else {
        data.image.dimensions()
    };
    let svg_size = (svg_size.0 as usize, svg_size.1 as usize);

    let draw_param = draw_param.clone();

    let palettes = ColorPalettes {
        label_colors,
        line_colors,
    };

    let svg = draw_on_image(
        data.image,
        lateral_points,
        (neck_sagittal_draw, &args.hide),
        draw_param,
        svg_size,
        palettes,
    )?;
    Ok(svg)
}

pub fn cmd(args: SvgArgs) -> Result<()> {
    if let Some(jobs) = args.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()
            .context("Failed to build thread pool")?;
    }
    let draw_param = if let Some(filename) = args.config.as_ref() {
        let s = std::fs::read_to_string(filename)
            .with_context(|| format!("Load config file {:?}", filename))?;
        toml::from_str(&s)?
    } else {
        DrawParam::default()
    };

    let label_colors = if let Some(filename) = args.label_colors.as_ref() {
        ColorPalette::new(
            labelme_rs::load_label_colors(filename)
                .with_context(|| format!("Load label color {:?}", filename))?,
        )
    } else {
        ColorPalette::new(labelme_rs::LabelColorsHex::default())
    };
    let line_colors = if let Some(filename) = args.line_colors.as_ref() {
        let reader = std::fs::File::open(filename)
            .with_context(|| format!("Load line color {:?}", filename))?;
        ColorPalette::new(scolrs::draw::load_line_colors(reader)?)
    } else {
        ColorPalette::new(scolrs::draw::LineColors::default())
    };

    let neck_sagittal_draw = args.measures.clone().unwrap_or_else(NeckLateralDraw::all);

    if args.input.extension().unwrap_or_default() == "json" {
        let lateral_points: scolrs::head_neck::LateralPoints =
            serde_json::from_str(&std::fs::read_to_string(&args.input)?)?;
        let document = process_data(
            lateral_points,
            &args,
            &draw_param,
            label_colors,
            line_colors,
            &neck_sagittal_draw,
        )?;
        std::fs::write(&args.output, document.to_string())
            .with_context(|| format!("Saving to {:?}", args.output))?;
        return Ok(());
    } else if args.input.as_os_str() == "-"
        || args.input.extension().unwrap_or_default() == "ndjson"
    {
        let reader: Box<dyn BufRead> = if args.input.as_os_str() == "-" {
            Box::new(BufReader::new(std::io::stdin()))
        } else {
            Box::new(BufReader::new(File::open(&args.input)?))
        };
        let lines = reader.lines().collect::<Result<Vec<_>, _>>()?;
        lines.into_par_iter().try_for_each(|line| -> Result<()> {
            let lateral_points_ir_line: scolrs::head_neck::LateralPointsLine =
                serde_json::from_str(&line)?;
            let lateral_points = lateral_points_ir_line.content;
            let result = process_data(
                lateral_points,
                &args,
                &draw_param,
                label_colors.clone(),
                line_colors.clone(),
                &neck_sagittal_draw,
            );
            let document = match result {
                Ok(document) => document,
                Err(e) => {
                    warn!("Skip {:?}: {:?}", lateral_points_ir_line.filename, e);
                    return Ok(());
                }
            };
            let output = args
                .output
                .join(&lateral_points_ir_line.filename)
                .with_extension("svg");
            std::fs::write(&output, document.to_string())
                .with_context(|| format!("Saving to {:?}", args.output))?;
            Ok(())
        })?;
    } else {
        return Err(anyhow::anyhow!("Unsupported file format"));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use std::path::PathBuf;

    fn output_path(name: &str) -> Result<PathBuf> {
        if let Ok(dir) = std::env::var("TEST_OUTPUT_DIR") {
            let path = PathBuf::from(dir).join(name);
            std::fs::create_dir_all(path.parent().unwrap())?;
            return Ok(path);
        }
        let devnull = PathBuf::from("/dev/null");
        if devnull.exists() {
            Ok(devnull)
        } else {
            Ok(PathBuf::from("NUL".to_string()))
        }
    }

    fn gen_svg_args() -> SvgArgs {
        let data_dir = PathBuf::from("../../tests/data/");
        let config = Some(data_dir.join("config.toml"));
        let label_colors = Some(data_dir.join("colors.yaml"));
        let line_colors = Some(data_dir.join("line_colors.csv"));
        SvgArgs {
            input: Default::default(),
            output: Default::default(),
            config,
            label_colors,
            line_colors,
            ..Default::default()
        }
    }

    /// Entry point for debugging
    #[test]
    fn test_svg_cmd_neck_case1() -> Result<()> {
        let mut svg_args = gen_svg_args();

        let data_dir = PathBuf::from("../../tests/data/");
        svg_args.input = data_dir.join("neck_case1/lateral_lateral_points.json");
        svg_args.output = output_path("neck_case1/neck_case1_lateral.svg")?;
        cmd(svg_args.clone())?;

        if std::env::var("TEST_OUTPUT_DIR").is_err() {
            return Ok(());
        }

        // test html command
        let html_args = crate::neck::cli::HtmlArgs {
            input: svg_args.output,
            output: None,
            selector: vec!["g.Component".to_string()],
            title: None,
        };
        crate::neck::html::cmd(html_args)?;

        Ok(())
    }

    #[test]
    fn test_svg_cmd_neck_case2() -> Result<()> {
        let mut svg_args = gen_svg_args();
        svg_args.resize = Some("768x768".to_string());

        let data_dir = PathBuf::from("../../tests/data/");

        // extension
        svg_args.input = data_dir.join("neck_case2/extension_lateral_lateral_points.json");
        svg_args.output = output_path("neck_case2/extension_lateral.svg")?;
        cmd(svg_args.clone())?;

        // flexion
        svg_args.input = data_dir.join("neck_case2/flexion_lateral_lateral_points.json");
        svg_args.output = output_path("neck_case2/flexion_lateral.svg")?;
        cmd(svg_args.clone())?;

        if std::env::var("TEST_OUTPUT_DIR").is_err() {
            return Ok(());
        }

        // test html command
        let mut html_args = crate::neck::cli::HtmlArgs {
            input: output_path("neck_case2/extension_lateral.svg")?,
            output: None,
            selector: vec!["g.Component".to_string()],
            title: None,
        };
        crate::neck::html::cmd(html_args.clone())?;

        html_args.input = output_path("neck_case2/flexion_lateral.svg")?;
        crate::neck::html::cmd(html_args)?;

        // test catalog command with directory input
        let catalog_args = crate::neck::cli::CatalogArgs {
            input: vec![output_path("neck_case2")?],
            output: output_path("neck_case2_catalog.html")?,
            title: Some("Neck Case 2".to_string()),
            selector: vec!["g.Component".to_string()],
            jobs: None,
        };
        crate::neck::catalog::cmd(catalog_args)?;

        // test catalog command with multiple directories
        let catalog_args = crate::neck::cli::CatalogArgs {
            input: vec![
                output_path("neck_case1/neck_case1_lateral.svg")?,
                output_path("neck_case2/flexion_lateral.svg")?,
            ],
            output: output_path("neck_cases_catalog.html")?,
            title: Some("Neck Cases".to_string()),
            selector: vec!["g.Component".to_string()],
            jobs: None,
        };
        crate::neck::catalog::cmd(catalog_args)?;

        // test catalog command with html input
        let catalog_args = crate::neck::cli::CatalogArgs {
            input: vec![
                output_path("neck_case2/flexion_lateral.html")?,
                output_path("neck_case2/extension_lateral.html")?,
            ],
            output: output_path("neck_catalog_from_html.html")?,
            title: Some("Neck Case 2 from HTML".to_string()),
            selector: vec!["g.Component".to_string()],
            jobs: None,
        };
        crate::neck::catalog::cmd(catalog_args)?;

        Ok(())
    }
}
