use crate::cli::{Plane, SvgArgs};
use anyhow::{Context, Result};
use labelme_rs::{image::GenericImageView, LabelMeDataWImage};
use log::debug;
use scolrs::{draw_coronal, draw_sagittal, ColorPalette, DrawParam, SagittalMeasure, ScolDesc};

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

    let label_colors = if let Some(filename) = args.label_colors {
        ColorPalette::new(
            labelme_rs::load_label_colors(&filename)
                .with_context(|| format!("Load label color {:?}", filename))?,
        )
    } else {
        ColorPalette::new(labelme_rs::LabelColorsHex::default())
    };
    let line_colors = if let Some(filename) = args.line_colors {
        let reader = std::fs::File::open(&filename)
            .with_context(|| format!("Load line color {:?}", filename))?;
        ColorPalette::new(scolrs::load_line_colors(reader)?)
    } else {
        ColorPalette::new(scolrs::LineColors::default())
    };

    let document = match args.direction {
        Plane::Coronal => {
            let coronal_points = scolrs::CoronalPoints::try_from(&data.data)?;
            let curve_apex_set = if let Some(filename) = args.curve_set {
                let reader = std::fs::File::open(&filename)
                    .with_context(|| format!("Load curve set {:?}", filename))?;
                let cs: ScolDesc = labelme_rs::serde_json::from_reader(reader)?;
                Some((cs.curves, cs.apices))
            } else {
                None
            };
            draw_coronal(
                data,
                coronal_points,
                draw_param,
                svg_size,
                label_colors,
                line_colors,
                curve_apex_set,
            )
        }
        Plane::Sagittal => {
            let sagittal_points = scolrs::SagittalPoints::try_from(&data.data)?;
            let measures: Vec<SagittalMeasure> = if args.measures.is_empty() {
                SagittalMeasure::all()
            } else {
                args.measures
            };
            draw_sagittal(
                data,
                sagittal_points,
                measures,
                args.hide,
                draw_param,
                svg_size,
                label_colors,
                line_colors,
            )
        }
    };

    debug!("Save to {:?}", args.output);

    std::fs::write(args.output, document.to_string())?;
    Ok(())
}
