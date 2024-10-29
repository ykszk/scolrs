use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::neck::cli::SvgArgs;
use anyhow::{Context, Result};
use labelme_rs::LabelMeDataLine;
use labelme_rs::{image::GenericImageView, LabelMeDataWImage};
use log::{debug, warn};
use scolrs::head_neck::{CervicalPoints, NeckLateralDraw, NeckSagittalComponent};
use scolrs::{parse_measures, ColorPalette, CommonComponent, DrawParam, Painter};
use svg::node::element::{self, SVG};

fn process_data(
    mut data: LabelMeDataWImage,
    args: &SvgArgs,
    draw_param: &DrawParam,
    label_colors: &mut ColorPalette,
    line_colors: &mut ColorPalette,
    neck_sagittal_draw: &[NeckLateralDraw],
) -> Result<SVG> {
    if let Some(resize) = args.resize.as_ref() {
        let resize_param = labelme_rs::ResizeParam::try_from(resize.as_str())?;
        data.resize(&resize_param);
    }
    let svg_size = if let Some(size) = args.size.as_ref() {
        let size_param = labelme_rs::ResizeParam::try_from(size.as_str())?;
        data.data
            .scale(size_param.scale(data.image.width(), data.image.height()));
        size_param.size(data.image.width(), data.image.height())
    } else {
        data.image.dimensions()
    };
    let svg_size = (svg_size.0 as usize, svg_size.1 as usize);

    let painter = Painter::new(draw_param.clone(), svg_size);
    let mut document = painter.doc_w_background(&data.image);
    let style = element::Style::new(draw_param.style());
    document = document.add(style);

    let cervical_points = scolrs::head_neck::LateralPoints::try_from(&data.data)?;

    debug!("Drawing");

    let common_components: Vec<Box<dyn CommonComponent>> =
        vec![Box::new(CervicalPoints(&cervical_points))];

    for component in common_components {
        debug!("Draw {:?}", component.name());
        let g = component.draw(&painter, label_colors, line_colors)?;
        document = document.add(g);
    }

    let neck_sagittal_components: Vec<Box<dyn NeckSagittalComponent>> = neck_sagittal_draw
        .iter()
        .map(|m| (m, &cervical_points).into())
        .collect();

    for component in neck_sagittal_components {
        debug!("Draw {:?}", component.name());
        match component.draw(&painter, label_colors, line_colors) {
            Ok(g) => document = document.add(g),
            Err(e) => match e {
                scolrs::MeasureError::InvalidNumberOfPoints(err) => {
                    warn!("Skip point count error:{:?}", err);
                }
                e => return Err(e.into()),
            },
        }
    }
    Ok(document)
}

pub fn cmd(args: SvgArgs) -> Result<()> {
    let draw_param = if let Some(filename) = args.config.as_ref() {
        let s = std::fs::read_to_string(filename)
            .with_context(|| format!("Load config file {:?}", filename))?;
        toml::from_str(&s)?
    } else {
        DrawParam::default()
    };

    let mut label_colors = if let Some(filename) = args.label_colors.as_ref() {
        ColorPalette::new(
            labelme_rs::load_label_colors(filename)
                .with_context(|| format!("Load label color {:?}", filename))?,
        )
    } else {
        ColorPalette::new(labelme_rs::LabelColorsHex::default())
    };
    let mut line_colors = if let Some(filename) = args.line_colors.as_ref() {
        let reader = std::fs::File::open(filename)
            .with_context(|| format!("Load line color {:?}", filename))?;
        ColorPalette::new(scolrs::load_line_colors(reader)?)
    } else {
        ColorPalette::new(scolrs::LineColors::default())
    };

    let neck_sagittal_draw: Vec<NeckLateralDraw> = if args.measures.is_empty() {
        NeckLateralDraw::all()
    } else {
        parse_measures(&args.measures).map_err(|e| anyhow::anyhow!(e))?
    };

    if args.input.extension().unwrap_or_default() == "json" {
        let data: LabelMeDataWImage = args
            .input
            .as_path()
            .try_into()
            .with_context(|| format!("Load LabelMeData from {:?}", &args.input))?;
        let document = process_data(
            data,
            &args,
            &draw_param,
            &mut label_colors,
            &mut line_colors,
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
        for line in reader.lines() {
            let line = line?;
            let data_line: LabelMeDataLine = serde_json::from_str(&line)?;
            let data = LabelMeDataWImage::try_from(data_line.content)?;
            let result = process_data(
                data,
                &args,
                &draw_param,
                &mut label_colors,
                &mut line_colors,
                &neck_sagittal_draw,
            );
            let document = match result {
                Ok(document) => document,
                Err(e) => {
                    warn!("Skip {:?}: {:?}", data_line.filename, e);
                    continue;
                }
            };
            let output = args.output.join(&data_line.filename).with_extension("svg");
            std::fs::write(&output, document.to_string())
                .with_context(|| format!("Saving to {:?}", args.output))?;
        }
    } else {
        return Err(anyhow::anyhow!("Unsupported file format"));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn output_path(name: &str) -> PathBuf {
        if let Ok(dir) = std::env::var("TEST_OUTPUT_DIR") {
            return PathBuf::from(dir).join(name);
        }
        let devnull = PathBuf::from("/dev/null");
        if devnull.exists() {
            devnull
        } else {
            PathBuf::from("NUL".to_string())
        }
    }

    fn test_vars() -> (PathBuf, Option<PathBuf>, Option<PathBuf>, Option<PathBuf>) {
        let data_dir = PathBuf::from("../../tests/data/");
        let config = Some(data_dir.join("config.toml"));
        let label_colors = Some(data_dir.join("colors.yaml"));
        let line_colors = Some(data_dir.join("line_colors.csv"));
        (data_dir, config, label_colors, line_colors)
    }

    /// Entry point for debugging
    #[test]
    fn test_svg_cmd_neck_case1() -> Result<()> {
        let (data_dir, config, label_colors, line_colors) = test_vars();

        let input = data_dir.join("neck_case1/lateral.json");
        let output = output_path("neck_case1_lateral.svg");
        let args = SvgArgs {
            input,
            output,
            config: config.clone(),
            label_colors: label_colors.clone(),
            line_colors: line_colors.clone(),
            ..Default::default()
        };
        cmd(args)
    }

    #[test]
    fn test_svg_cmd_neck_case2() -> Result<()> {
        let (data_dir, config, label_colors, line_colors) = test_vars();

        let input = data_dir.join("neck_case2/extension_lateral.json");
        let output = output_path("neck_case2_extension_lateral.svg");
        let args = SvgArgs {
            input,
            output,
            config: config.clone(),
            label_colors: label_colors.clone(),
            line_colors: line_colors.clone(),
            resize: Some("768x768".to_string()),
            ..Default::default()
        };
        cmd(args)?;

        let input = data_dir.join("neck_case2/flexion_lateral.json");
        let output = output_path("neck_case2_flexion_lateral.svg");
        let args = SvgArgs {
            input,
            output,
            config: config.clone(),
            label_colors: label_colors.clone(),
            line_colors: line_colors.clone(),
            resize: Some("768x768".to_string()),
            ..Default::default()
        };
        cmd(args)
    }
}
