use crate::cli::{Plane, SvgArgs};
use anyhow::{Context, Result};
use labelme_rs::{image::GenericImageView, LabelMeData, LabelMeDataWImage};
use log::debug;
use scolrs::{
    draw_coronal, draw_sagittal, parse_measures, ColorPalette, ColorPalettes, CoronalMeasure,
    CoronalPoints, CoronalPointsIR, DrawParam, SagittalMeasure, SagittalPoints, SagittalPointsIR,
    ScolDesc,
};

// enum NativeFormat {
//     Coronal(Box<CoronalPoints>),
//     Sagittal(Box<SagittalPoints>),
// }

// impl NativeFormat {
//     fn image_data(&self) -> Option<&Vec<u8>> {
//         match self {
//             NativeFormat::Coronal(cp) => Some(&cp.image_data),
//             NativeFormat::Sagittal(sp) => Some(&sp.image_data),
//         }
//     }
// }

pub fn cmd(args: SvgArgs) -> Result<()> {
    let draw_param = if let Some(filename) = args.config {
        let s = std::fs::read_to_string(&filename)
            .with_context(|| format!("Load config file {:?}", filename))?;
        toml::from_str(&s)?
    } else {
        DrawParam::default()
    };

    debug!("Loading {:?}", args.input);

    let (mut data, image_data) = if args.labelme {
        let data: LabelMeDataWImage = args
            .input
            .as_path()
            .try_into()
            .with_context(|| format!("Load LabelMeData from {:?}", &args.input))?;
        (data, None)
    } else {
        let json_str = std::fs::read_to_string(&args.input)
            .with_context(|| format!("Load native format data from {:?}", &args.input))?;
        match args.direction {
            Plane::Coronal => {
                let ir: CoronalPointsIR = serde_json::from_str(&json_str)?;
                let data = LabelMeData::from(ir.clone());
                let data_w_image = LabelMeDataWImage::try_from(data)?;
                let cp: CoronalPoints = ir.try_into()?;
                (data_w_image, Some(cp.image_data))
            }
            Plane::Sagittal => {
                let ir: SagittalPointsIR = serde_json::from_str(&json_str)?;
                let data = LabelMeData::from(ir.clone());
                let data_w_image = LabelMeDataWImage::try_from(data)?;
                let sp: SagittalPoints = ir.try_into()?;

                (data_w_image, Some(sp.image_data))
            }
        }
    };

    // let mut data: LabelMeDataWImage = args
    //     .input
    //     .as_path()
    //     .try_into()
    //     .with_context(|| format!("Load LabelMeData from {:?}", &args.input))?;

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

    let palettes = ColorPalettes {
        label_colors,
        line_colors,
    };

    let document = match args.direction {
        Plane::Coronal => {
            let mut coronal_points = scolrs::CoronalPoints::try_from(&data.data)?;
            if let Some(image_data) = image_data {
                coronal_points.image_data = image_data;
                coronal_points.scale();
            }
            let curve_apex_set = if let Some(filename) = args.curve_set {
                let reader = std::fs::File::open(&filename)
                    .with_context(|| format!("Load curve set {:?}", filename))?;
                let cs: ScolDesc = labelme_rs::serde_json::from_reader(reader)?;
                Some((cs.curves, cs.apices))
            } else {
                None
            };
            let draws: Vec<CoronalMeasure> = if args.measures.is_empty() {
                CoronalMeasure::all_draws()
            } else {
                let v = parse_measures(&args.measures);
                match v {
                    Ok(v) => v,
                    Err(e) => return Err(anyhow::anyhow!(e)),
                }
            };
            let hide = parse_measures(&args.hide);
            let hide = match hide {
                Ok(v) => v,
                Err(e) => return Err(anyhow::anyhow!(e)),
            };
            draw_coronal(
                data,
                coronal_points,
                (draws, hide),
                draw_param,
                svg_size,
                palettes,
                curve_apex_set,
            )
        }
        Plane::Sagittal => {
            let mut sagittal_points = scolrs::SagittalPoints::try_from(&data.data)?;
            if let Some(image_data) = image_data {
                sagittal_points.image_data = image_data;
                sagittal_points.scale();
            }
            let draws: Vec<SagittalMeasure> = if args.measures.is_empty() {
                SagittalMeasure::all_draws()
            } else {
                let v = parse_measures(&args.measures);
                match v {
                    Ok(v) => v,
                    Err(e) => return Err(anyhow::anyhow!(e)),
                }
            };
            let hide = parse_measures(&args.hide);
            let hide = match hide {
                Ok(v) => v,
                Err(e) => return Err(anyhow::anyhow!(e)),
            };
            draw_sagittal(
                data,
                sagittal_points,
                draws,
                hide,
                draw_param,
                svg_size,
                palettes,
            )
        }
    };

    debug!("Save to {:?}", args.output);

    std::fs::write(args.output, document?.to_string())?;
    Ok(())
}
