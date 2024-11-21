use std::{
    io::{BufRead, BufReader},
    path::Path,
};

use crate::cli::{Plane, SvgArgs, SvgArgsCommon, SvgNdjsonArgs, SvgSubCommands};
use anyhow::{Context, Result};
use labelme_rs::{
    image::GenericImageView, LabelMeData, LabelMeDataLine, LabelMeDataWImage, ResizeParam,
};
use log::debug;
use rayon::prelude::*;
use scolrs::{
    draw_coronal, draw_sagittal, ColorPalette, ColorPalettes, CoronalMeasure,
    CoronalPointsAndCurve, CoronalPointsAndCurveIR, CoronalPointsAndCurveIRLine, DrawParam,
    ImageMetadata, MeasureAndDraw, SagittalMeasure, SagittalPoints, SagittalPointsIR,
    SagittalPointsIRLine,
};

enum PointData {
    Coronal(Box<CoronalPointsAndCurve>),
    Sagittal(Box<SagittalPoints>),
}

/// Maintain redundant point data in sync with scaling and resizing
struct PointDataWithImage {
    data: PointData,
    data_image: LabelMeDataWImage,
}

impl PointDataWithImage {
    fn resize(&mut self, param: &ResizeParam) {
        self.data_image.resize(param);
        match &mut self.data {
            PointData::Coronal(cp) => {
                cp.coronal_points = (&self.data_image.data).try_into().unwrap()
            }
            PointData::Sagittal(sp) => {
                *sp = SagittalPoints::try_from(&self.data_image.data)
                    .unwrap()
                    .into()
            }
        }
    }

    fn new_coronal(data: CoronalPointsAndCurve, data_image: LabelMeDataWImage) -> Self {
        Self {
            data: PointData::Coronal(data.into()),
            data_image,
        }
    }

    fn new_sagittal(data: SagittalPoints, data_image: LabelMeDataWImage) -> Self {
        Self {
            data: PointData::Sagittal(data.into()),
            data_image,
        }
    }
}

fn process_one(
    svg_common: ReadSvgArgCommon,
    mut point_with_image: PointDataWithImage,
    output: &std::path::Path,
) -> Result<()> {
    if let Some(resize_param) = svg_common.resize_param {
        point_with_image.resize(&resize_param);
    }
    let svg_size = if let Some(_svg_size_param) = svg_common.svg_size_param {
        // data.data
        //     .scale(svg_size_param.scale(data.image.width(), data.image.height()));
        // svg_size_param.size(data.image.width(), data.image.height())
        unimplemented!("svg_size_param")
    } else {
        point_with_image.data_image.image.dimensions()
    };
    let svg_size = (svg_size.0 as usize, svg_size.1 as usize);

    let document = match svg_common.subcommand {
        SvgSubCommands::Coronal(subcommand) => {
            let draws = subcommand
                .measures
                .unwrap_or_else(CoronalMeasure::all_draws);
            let mut hide = subcommand.hide;
            for group in subcommand.hide_group {
                hide.append(&mut (&group).into());
            }
            let coronal_set = match point_with_image.data {
                PointData::Coronal(cp) => cp,
                _ => unreachable!(),
            };
            draw_coronal(
                point_with_image.data_image.image,
                *coronal_set,
                (&draws, &hide),
                svg_common.draw_param,
                svg_size,
                svg_common.palettes,
            )
        }
        SvgSubCommands::Sagittal(subcommand) => {
            let draws = subcommand
                .measures
                .unwrap_or_else(SagittalMeasure::all_draws);
            let hide = subcommand.hide;
            let sagittal_points = match point_with_image.data {
                PointData::Sagittal(sp) => sp,
                _ => unreachable!(),
            };
            draw_sagittal(
                point_with_image.data_image.image,
                *sagittal_points,
                &draws,
                &hide,
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
    let point_data_with_image: PointDataWithImage = if svg_common.labelme {
        let data: LabelMeDataWImage = args
            .input
            .as_path()
            .try_into()
            .with_context(|| format!("Load LabelMeData from {:?}", &args.input))?;
        let image_data = if svg_common.pull_spacing {
            Some(ImageMetadata::try_from(Path::new(&data.data.imagePath))?)
        } else {
            None
        };

        match svg_common.direction {
            Plane::Coronal => {
                let mut cp: CoronalPointsAndCurve = (&data.data).try_into()?;
                if let Some(image_data) = image_data {
                    cp.coronal_points.image_metadata = image_data;
                }
                PointDataWithImage::new_coronal(cp, data)
            }
            Plane::Sagittal => {
                let mut sp: SagittalPoints = (&data.data).try_into()?;
                if let Some(image_data) = image_data {
                    sp.image_metadata = image_data;
                }
                PointDataWithImage::new_sagittal(sp, data)
            }
        }
    } else {
        let json_str = std::fs::read_to_string(&args.input)
            .with_context(|| format!("Load native format data from {:?}", &args.input))?;
        load_native_format(json_str, &args.input, svg_common.direction)?
    };

    debug!("Loading {:?}", args.input);
    process_one(svg_common, point_data_with_image, &args.output)
}

fn load_native_format(
    json_str: String,
    json_path: &Path,
    direction: Plane,
) -> Result<PointDataWithImage> {
    Ok(match direction {
        Plane::Coronal => {
            let ir: CoronalPointsAndCurveIR = serde_json::from_str(&json_str)?;
            let cp = CoronalPointsAndCurve::try_from(&ir)?;
            let data = LabelMeData::from(ir.coronal_points);
            let data_w_image = LabelMeDataWImage::try_from_data_and_path(data, json_path)?;
            PointDataWithImage::new_coronal(cp, data_w_image)
        }
        Plane::Sagittal => {
            let ir: SagittalPointsIR = serde_json::from_str(&json_str)?;
            let data = LabelMeData::from(ir.clone());
            let data_w_image = LabelMeDataWImage::try_from_data_and_path(data, json_path)?;
            let sp: SagittalPoints = ir.try_into()?;
            PointDataWithImage::new_sagittal(sp, data_w_image)
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
    pull_spacing: bool,
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
        pull_spacing: args.pull_spacing,
    })
}

pub fn cmd_ndjson(args: SvgNdjsonArgs) -> Result<()> {
    if let Some(jobs) = args.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()?;
    }

    let svg_common = load_svg_common(args.svg_args)?;

    let reader = if args.input.as_os_str() == "-" {
        Box::new(BufReader::new(std::io::stdin())) as Box<dyn BufRead>
    } else {
        Box::new(BufReader::new(std::fs::File::open(&args.input)?))
    };
    let lines = reader.lines().collect::<Result<Vec<_>, _>>()?;
    lines.into_par_iter().try_for_each(|line| -> Result<()> {
        let (point_with_image_data, filename): (PointDataWithImage, String) = if svg_common.labelme
        {
            let data_line: LabelMeDataLine = line.as_str().try_into()?;
            let data = LabelMeDataWImage::try_from_data_and_path(data_line.content, &args.input)?;
            let image_data = if svg_common.pull_spacing {
                Some(ImageMetadata::try_from(Path::new(&data.data.imagePath))?)
            } else {
                None
            };
            match svg_common.direction {
                Plane::Coronal => {
                    let mut cp = CoronalPointsAndCurve::try_from(&data.data)?;
                    if let Some(image_data) = image_data {
                        cp.coronal_points.image_metadata = image_data;
                    }
                    (
                        PointDataWithImage::new_coronal(cp, data),
                        data_line.filename,
                    )
                }
                Plane::Sagittal => {
                    let mut sp = SagittalPoints::try_from(&data.data)?;
                    if let Some(image_data) = image_data {
                        sp.image_metadata = image_data;
                    }
                    (
                        PointDataWithImage::new_sagittal(sp, data),
                        data_line.filename,
                    )
                }
            }
        } else {
            let json_str = line.as_str();
            match svg_common.direction {
                Plane::Coronal => {
                    let ir_line: CoronalPointsAndCurveIRLine = serde_json::from_str(json_str)?;
                    let cp = CoronalPointsAndCurve::try_from(&ir_line.content)?;
                    let data = LabelMeData::from(ir_line.content.coronal_points);
                    let data_w_image =
                        LabelMeDataWImage::try_from_data_and_path(data, &args.input)?;
                    (
                        PointDataWithImage::new_coronal(cp, data_w_image),
                        ir_line.filename,
                    )
                }
                Plane::Sagittal => {
                    let ir: SagittalPointsIRLine = serde_json::from_str(json_str)?;
                    let data = LabelMeData::from(ir.content.clone());
                    let data_w_image =
                        LabelMeDataWImage::try_from_data_and_path(data, &args.input)?;
                    let sp: SagittalPoints = ir.content.try_into()?;
                    (
                        PointDataWithImage::new_sagittal(sp, data_w_image),
                        ir.filename,
                    )
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
        process_one(svg_common.clone(), point_with_image_data, &output)?;
        Ok(())
    })?;
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
