use crate::cli::{Plane, SvgArgs};
use anyhow::{Context, Result};
use labelme_rs::{image::GenericImageView, LabelMeData, LabelMeDataWImage};
use log::debug;
use scolrs::{
    draw_coronal, draw_sagittal, parse_measures, ColorPalette, ColorPalettes, CoronalMeasure,
    CoronalPoints, CoronalPointsIR, DrawParam, SagittalMeasure, SagittalPoints, SagittalPointsIR,
    ScolDesc,
};

pub fn cmd(args: SvgArgs) -> Result<()> {
    let mut draw_param = if let Some(filename) = args.config {
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
                let data_w_image = LabelMeDataWImage::try_from_data_and_path(data, &args.input)?;
                let cp: CoronalPoints = ir.try_into()?;
                (data_w_image, Some(cp.image_metadata))
            }
            Plane::Sagittal => {
                let ir: SagittalPointsIR = serde_json::from_str(&json_str)?;
                let data = LabelMeData::from(ir.clone());
                let data_w_image = LabelMeDataWImage::try_from_data_and_path(data, &args.input)?;
                let sp: SagittalPoints = ir.try_into()?;

                (data_w_image, Some(sp.image_metadata))
            }
        }
    };

    if let Some(image_data) = image_data.as_ref() {
        // mean spacing
        let draw_scale = (image_data.spacing_xy.0 + image_data.spacing_xy.1) / 2.0;
        draw_param
            .scale(draw_scale)
            .map_err(|e| anyhow::anyhow!(e))?;
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
                coronal_points.image_metadata = image_data;
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
                sagittal_points.image_metadata = image_data;
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

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use std::path::PathBuf;

    fn output_path(name: &str) -> Result<PathBuf> {
        if let Ok(dir) = std::env::var("TEST_OUTPUT_DIR") {
            let path = PathBuf::from(dir).join("scol").join(name);
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
        let resize = Some("1024x1024".to_string());
        SvgArgs {
            input: Default::default(),
            output: Default::default(),
            config,
            label_colors,
            line_colors,
            resize,
            ..Default::default()
        }
    }

    fn _test_case(case: &str, native: bool) -> Result<()> {
        let mut svg_args = gen_svg_args();

        let data_dir = PathBuf::from("../../tests/data/");
        if native {
            svg_args.input = data_dir.join(case).join("frontal_native.json");
            svg_args.output = output_path(&format!("{}_frontal_native.svg", case))?;
        } else {
            svg_args.input = data_dir.join(case).join("frontal.json");
            svg_args.output = output_path(&format!("{}_frontal.svg", case))?;
        }
        svg_args.labelme = !native;
        svg_args.direction = Plane::Coronal;
        cmd(svg_args.clone())?;

        if native {
            svg_args.input = data_dir.join(case).join("lateral_native.json");
            svg_args.output = output_path(&format!("{}_lateral_native.svg", case))?;
        } else {
            svg_args.input = data_dir.join(case).join("lateral.json");
            svg_args.output = output_path(&format!("{}_lateral.svg", case))?;
        }
        svg_args.labelme = !native;
        svg_args.direction = Plane::Sagittal;
        cmd(svg_args)?;

        Ok(())
    }

    /// Entry point for debugging
    #[test]
    fn svg_cmd_scol_case1() -> Result<()> {
        _test_case("case1", false)?;

        // test bending
        let mut svg_args = gen_svg_args();
        let data_dir = PathBuf::from("../../tests/data/");
        svg_args.input = data_dir.join("case1/left_lateral_bend.json");
        svg_args.output = output_path("case1_left_lateral_bend.svg")?;
        svg_args.curve_set = Some(data_dir.join("case1/curve_set.json"));
        svg_args.labelme = true;
        cmd(svg_args.clone())?;

        svg_args.input = data_dir.join("case1/right_lateral_bend.json");
        svg_args.output = output_path("case1_right_lateral_bend.svg")?;
        cmd(svg_args)?;
        Ok(())
    }

    #[test]
    fn svg_cmd_scol_case2() -> Result<()> {
        _test_case("case2", false)?;
        _test_case("case2", true)
    }

    #[test]
    fn svg_cmd_scol_case3() -> Result<()> {
        _test_case("case3", false)
    }

    #[test]
    fn svg_cmd_scol_case4() -> Result<()> {
        _test_case("case4", false)
    }
}
