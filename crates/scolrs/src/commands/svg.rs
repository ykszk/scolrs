use std::{
    io::{BufRead, BufReader},
    path::Path,
};

use crate::cli::{Plane, SvgArgs, SvgArgsCommon, SvgNdjsonArgs, SvgSubCommands};
use anyhow::{Context, Result};
use labelme_rs::{
    image::GenericImageView, LabelMeData, LabelMeDataLine, LabelMeDataWImage, ResizeParam,
};
use log::{debug, info};
use scolrs::{
    draw_coronal, draw_sagittal, ApexSet, ColorPalette, ColorPalettes, CoronalMeasure,
    CoronalPoints, CoronalPointsIR, CoronalPointsIRLine, CurveSet, DrawParam, ImageMetadata,
    SagittalMeasure, SagittalPoints, SagittalPointsIR, SagittalPointsIRLine, ScolDesc,
    ScolDescLine,
};

fn process_one(
    mut svg_common: ReadSvgArgCommon,
    mut data: LabelMeDataWImage,
    image_data: Option<ImageMetadata>,
    curve_apex_set: Option<(CurveSet, ApexSet)>,
    output: &std::path::Path,
) -> Result<()> {
    if let Some(image_data) = image_data.as_ref() {
        // mean spacing
        let draw_scale = (image_data.spacing_xy.0 + image_data.spacing_xy.1) / 2.0;
        svg_common
            .draw_param
            .scale(draw_scale)
            .map_err(|e| anyhow::anyhow!(e))?;
    }

    if let Some(resize_param) = svg_common.resize_param {
        data.resize(&resize_param);
    }
    let svg_size = if let Some(svg_size_param) = svg_common.svg_size_param {
        data.data
            .scale(svg_size_param.scale(data.image.width(), data.image.height()));
        svg_size_param.size(data.image.width(), data.image.height())
    } else {
        data.image.dimensions()
    };
    let svg_size = (svg_size.0 as usize, svg_size.1 as usize);

    let document = match svg_common.subcommand {
        SvgSubCommands::Coronal(subcommand) => {
            let mut coronal_points = scolrs::CoronalPoints::try_from(&data.data)?;
            if let Some(image_data) = image_data {
                coronal_points.image_metadata = image_data;
                coronal_points.scale()?;
            }

            let draws = subcommand
                .measures
                .unwrap_or_else(CoronalMeasure::all_draws);
            let hide = subcommand.hide;
            draw_coronal(
                data,
                coronal_points,
                (draws, hide),
                svg_common.draw_param,
                svg_size,
                svg_common.palettes,
                curve_apex_set,
            )
        }
        SvgSubCommands::Sagittal(subcommand) => {
            let mut sagittal_points = scolrs::SagittalPoints::try_from(&data.data)?;
            if let Some(image_data) = image_data {
                sagittal_points.image_metadata = image_data;
                sagittal_points.scale();
            }

            let draws = subcommand
                .measures
                .unwrap_or_else(SagittalMeasure::all_draws);
            let hide = subcommand.hide;
            draw_sagittal(
                data,
                sagittal_points,
                draws,
                hide,
                svg_common.draw_param,
                svg_size,
                svg_common.palettes,
            )
        }
    };

    debug!("Save to {:?}", output);

    std::fs::write(output, document?.to_string())?;
    Ok(())
}

pub fn cmd(args: SvgArgs) -> Result<()> {
    let svg_common = load_svg_common(args.svg_args)?;
    let curve_apex_set = if let Some(filename) = args.curve_set {
        let reader = std::fs::File::open(&filename)
            .with_context(|| format!("Load curve set {:?}", filename))?;
        let cs: ScolDesc = labelme_rs::serde_json::from_reader(reader)?;
        Some((cs.curves, cs.apices))
    } else {
        None
    };
    let (data, image_data) = if svg_common.labelme {
        let data: LabelMeDataWImage = args
            .input
            .as_path()
            .try_into()
            .with_context(|| format!("Load LabelMeData from {:?}", &args.input))?;
        (data, None)
    } else {
        let json_str = std::fs::read_to_string(&args.input)
            .with_context(|| format!("Load native format data from {:?}", &args.input))?;
        load_native_format(json_str, &args.input, svg_common.direction)?
    };

    debug!("Loading {:?}", args.input);
    process_one(svg_common, data, image_data, curve_apex_set, &args.output)
}

fn load_native_format(
    json_str: String,
    json_path: &Path,
    direction: Plane,
) -> Result<(LabelMeDataWImage, Option<ImageMetadata>)> {
    Ok(match direction {
        Plane::Coronal => {
            let ir: CoronalPointsIR = serde_json::from_str(&json_str)?;
            let data = LabelMeData::from(ir.clone());
            let data_w_image = LabelMeDataWImage::try_from_data_and_path(data, json_path)?;
            let cp: CoronalPoints = ir.try_into()?;
            (data_w_image, Some(cp.image_metadata))
        }
        Plane::Sagittal => {
            let ir: SagittalPointsIR = serde_json::from_str(&json_str)?;
            let data = LabelMeData::from(ir.clone());
            let data_w_image = LabelMeDataWImage::try_from_data_and_path(data, json_path)?;
            let sp: SagittalPoints = ir.try_into()?;

            (data_w_image, Some(sp.image_metadata))
        }
    })
}

#[derive(Clone)]
struct ReadSvgArgCommon {
    draw_param: scolrs::DrawParam,
    resize_param: Option<labelme_rs::ResizeParam>,
    svg_size_param: Option<labelme_rs::ResizeParam>,
    palettes: ColorPalettes,
    direction: Plane,
    subcommand: SvgSubCommands,
    labelme: bool,
}

fn load_svg_common(args: SvgArgsCommon) -> Result<ReadSvgArgCommon> {
    let draw_param = if let Some(filename) = args.config.as_ref() {
        let s = std::fs::read_to_string(filename)
            .with_context(|| format!("Load config file {:?}", filename))?;
        toml::from_str(&s)?
    } else {
        DrawParam::default()
    };
    let resize_param = args
        .resize
        .map(|s| ResizeParam::try_from(s.as_str()))
        .transpose()?;
    let svg_size_param = args
        .size
        .map(|s| ResizeParam::try_from(s.as_str()))
        .transpose()?;
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

    let direction = match args.subcommand {
        SvgSubCommands::Coronal(_) => Plane::Coronal,
        SvgSubCommands::Sagittal(_) => Plane::Sagittal,
    };

    Ok(ReadSvgArgCommon {
        draw_param,
        resize_param,
        svg_size_param,
        palettes,
        direction,
        subcommand: args.subcommand,
        labelme: args.labelme,
    })
}

pub fn cmd_ndjson(args: SvgNdjsonArgs) -> Result<()> {
    let mut curve_map = if let Some(filename) = args.curve_set.as_ref() {
        info!("Load curve set {:?}", filename);
        let reader = BufReader::new(std::fs::File::open(filename)?);
        let mut curve_map: std::collections::HashMap<String, (CurveSet, ApexSet)> =
            Default::default();
        for line in reader.lines() {
            let line = line?;
            let cs: ScolDescLine = labelme_rs::serde_json::from_str(&line)?;
            curve_map.insert(cs.filename.clone(), (cs.content.curves, cs.content.apices));
        }
        Some(curve_map)
    } else {
        None
    };
    let svg_common = load_svg_common(args.svg_args)?;

    let reader = if args.input.as_os_str() == "-" {
        Box::new(BufReader::new(std::io::stdin())) as Box<dyn BufRead>
    } else {
        Box::new(BufReader::new(std::fs::File::open(&args.input)?))
    };
    for line in reader.lines() {
        let line = line?;
        let (data, image_data, filename) = if svg_common.labelme {
            let data_line: LabelMeDataLine = line.as_str().try_into()?;
            let data = LabelMeDataWImage::try_from_data_and_path(data_line.content, &args.input)?;
            (data, None, data_line.filename)
        } else {
            let json_str = line.as_str();
            match svg_common.direction {
                Plane::Coronal => {
                    let ir: CoronalPointsIRLine = serde_json::from_str(json_str)?;
                    let data = LabelMeData::from(ir.content.clone());
                    let data_w_image =
                        LabelMeDataWImage::try_from_data_and_path(data, &args.input)?;
                    let cp: CoronalPoints = ir.content.try_into()?;
                    (data_w_image, Some(cp.image_metadata), ir.filename)
                }
                Plane::Sagittal => {
                    let ir: SagittalPointsIRLine = serde_json::from_str(json_str)?;
                    let data = LabelMeData::from(ir.content.clone());
                    let data_w_image =
                        LabelMeDataWImage::try_from_data_and_path(data, &args.input)?;
                    let sp: SagittalPoints = ir.content.try_into()?;
                    (data_w_image, Some(sp.image_metadata), ir.filename)
                }
            }
        };
        debug!("Processing {:?}", filename);

        let output = args.output.join(
            Path::new(filename.as_str())
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string()
                + ".svg",
        );
        let curve_set = curve_map.as_mut().and_then(|m| m.remove(&filename));
        if args.curve_set.as_ref().is_some() && curve_set.is_none() {
            return Err(anyhow::anyhow!("Curve set not found for {}", filename));
        }
        process_one(svg_common.clone(), data, image_data, curve_set, &output)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::cli::SvgArgsCommon;

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
        let svg_args = SvgArgsCommon {
            config,
            label_colors,
            line_colors,
            resize,
            ..Default::default()
        };
        SvgArgs {
            svg_args,
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
        svg_args.svg_args.labelme = !native;
        svg_args.svg_args.subcommand = SvgSubCommands::Coronal(Default::default());
        cmd(svg_args.clone())?;

        if native {
            svg_args.input = data_dir.join(case).join("lateral_native.json");
            svg_args.output = output_path(&format!("{}_lateral_native.svg", case))?;
        } else {
            svg_args.input = data_dir.join(case).join("lateral.json");
            svg_args.output = output_path(&format!("{}_lateral.svg", case))?;
        }
        svg_args.svg_args.labelme = !native;
        svg_args.svg_args.subcommand = SvgSubCommands::Sagittal(Default::default());
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
        svg_args.svg_args.labelme = true;
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
