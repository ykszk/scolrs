use crate::cli::{Direction, SvgArgs};
use anyhow::{Context, Result};
use labelme_rs::{image::GenericImageView, LabelMeData, LabelMeDataWImage};
use log::debug;
use scolrs::{draw_coronal, draw_sagittal, ColorPaletts, DrawParam, ScolDesc, Spine};

pub fn cmd(args: SvgArgs) -> Result<()> {
    let draw_param = if let Some(filename) = args.config {
        let s = std::fs::read_to_string(&filename)
            .with_context(|| format!("reading file {:?}", filename))?;
        toml::from_str(&s)?
    } else {
        DrawParam::default()
    };

    debug!("Loading {:?}", args.input);
    let s = std::fs::read_to_string(&args.input)
        .with_context(|| format!("reading file {:?}", &args.input))?;
    let data: LabelMeData = s.try_into()?;
    let orig_wd = std::env::current_dir()?;
    if let Some(parent) = args.input.parent() {
        std::env::set_current_dir(parent)?;
    }
    let mut data = LabelMeDataWImage::try_from(data)?;
    if args.input.parent().is_some() {
        std::env::set_current_dir(orig_wd)?;
    }
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

    let scol = Spine::try_from(&data.data)?;

    let label_colors = if let Some(filename) = args.label_colors {
        ColorPaletts::new(
            labelme_rs::load_label_colors(&filename)
                .with_context(|| format!("reading file {:?}", filename))?,
        )
    } else {
        ColorPaletts::new(labelme_rs::LabelColorsHex::default())
    };
    let line_colors = if let Some(filename) = args.line_colors {
        let reader = std::fs::File::open(&filename)
            .with_context(|| format!("reading file {:?}", filename))?;
        ColorPaletts::new(scolrs::load_line_colors(reader)?)
    } else {
        ColorPaletts::new(scolrs::LineColors::default())
    };

    let document = match args.direction {
        Direction::Frontal => {
            let curve_apex_set = if let Some(filename) = args.curve_set {
                let reader = std::fs::File::open(&filename)
                    .with_context(|| format!("reading file {:?}", filename))?;
                let cs: ScolDesc = labelme_rs::serde_json::from_reader(reader)?;
                Some((cs.curves, cs.apices))
            } else {
                None
            };
            draw_coronal(
                data,
                scol,
                draw_param,
                svg_size,
                label_colors,
                line_colors,
                curve_apex_set,
            )
        }
        Direction::Lateral => {
            draw_sagittal(data, scol, draw_param, svg_size, label_colors, line_colors)
        }
    };

    std::fs::write(args.output, document.to_string())?;
    Ok(())
}
