use crate::neck::cli::SvgArgs;
use anyhow::{Context, Result};
use labelme_rs::{image::GenericImageView, LabelMeDataWImage};
use log::{debug, warn};
use scolrs::head_neck::{
    Adi, CervicalPoints, LaminalPoints, NeckSagittalComponent, OptionalPoints, Sacs, WedgeAngle,
    OC2,
};
use scolrs::{ColorPalette, CommonComponent, DrawParam, Painter};
use svg::node::element;

pub fn cmd(args: SvgArgs) -> Result<()> {
    let draw_param = if let Some(filename) = args.config {
        let s = std::fs::read_to_string(&filename)
            .with_context(|| format!("Load config file {:?}", filename))?;
        toml::from_str(&s)?
    } else {
        DrawParam::default()
    };

    debug!("Loading {:?}", args.input);

    let mut data: LabelMeDataWImage = args
        .input
        .as_path()
        .try_into()
        .with_context(|| format!("Load LabelMeData from {:?}", &args.input))?;

    if let Some(resize) = args.resize {
        let resize_param = labelme_rs::ResizeParam::try_from(resize.as_str())?;
        data.resize(&resize_param);
    }
    let svg_size = if let Some(size) = args.size {
        let size_param = labelme_rs::ResizeParam::try_from(size.as_str())?;
        data.data
            .scale(size_param.scale(data.image.width(), data.image.height()));
        size_param.size(data.image.width(), data.image.height())
    } else {
        data.image.dimensions()
    };
    let svg_size = (svg_size.0 as usize, svg_size.1 as usize);

    let mut label_colors = if let Some(filename) = args.label_colors {
        ColorPalette::new(
            labelme_rs::load_label_colors(&filename)
                .with_context(|| format!("Load label color {:?}", filename))?,
        )
    } else {
        ColorPalette::new(labelme_rs::LabelColorsHex::default())
    };
    let mut line_colors = if let Some(filename) = args.line_colors {
        let reader = std::fs::File::open(&filename)
            .with_context(|| format!("Load line color {:?}", filename))?;
        ColorPalette::new(scolrs::load_line_colors(reader)?)
    } else {
        ColorPalette::new(scolrs::LineColors::default())
    };

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
        let g = component.draw(&painter, &mut label_colors, &mut line_colors)?;
        document = document.add(g);
    }

    let neck_sagittal_components: Vec<Box<dyn NeckSagittalComponent>> = vec![
        Box::new(LaminalPoints(&cervical_points.lamina)),
        Box::new(OptionalPoints(&cervical_points)),
        Box::new(Sacs(&cervical_points)),
        Box::new(Adi(&cervical_points)),
        Box::new(OC2(&cervical_points)),
        Box::new(WedgeAngle(&cervical_points)),
    ];

    for component in neck_sagittal_components {
        debug!("Draw {:?}", component.name());
        match component.draw(&painter, &mut label_colors, &mut line_colors) {
            Ok(g) => document = document.add(g),
            Err(e) => match e {
                scolrs::MeasureError::InvalidNumberOfPoints(err) => {
                    warn!("Skip point count error:{:?}", err);
                }
                e => return Err(e.into()),
            },
        }
    }

    debug!("Save to {:?}", args.output);

    std::fs::write(args.output, document.to_string())?;

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
            resize: None,
            size: None,
            measures: vec![],
            hide: vec![],
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
            resize: None,
            size: None,
            measures: vec![],
            hide: vec![],
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
            resize: None,
            size: None,
            measures: vec![],
            hide: vec![],
        };
        cmd(args)
    }
}
