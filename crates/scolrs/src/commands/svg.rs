use std::{
    io::{BufRead, BufReader},
    path::Path,
};

use crate::cli::{SvgArgs, SvgArgsCommon, SvgNdjsonArgs, SvgSubCommands};
use anyhow::{Context, Result};
use labelme_rs::{
    image::{DynamicImage, GenericImageView},
    LabelMeData, LabelMeDataWImage, ResizeParam,
};
use log::debug;
use rayon::prelude::*;
use scolrs::{
    draw::{draw_coronal, draw_sagittal, ColorPalette, ColorPalettes, DrawError},
    ContentFilename, CoronalMeasure, CoronalPointsAndCurve, CoronalPointsAndCurveLine, DrawParam,
    HasImageMetadata, ImageMetadata, MeasureAndDraw, PointDataWithImage, SagittalMeasure,
    SagittalPoints, SagittalPointsLine, Scalable, UpdatePoints,
};
use serde::Deserialize;
use svg::node::element;

trait AssociatedMeasureAndDraw {
    type Measure;
    fn draw(
        image: DynamicImage,
        sagittal_points: Self,
        draws: &[Self::Measure],
        hide: &[Self::Measure],
        draw_param: DrawParam,
        svg_size: (usize, usize),
        palettes: ColorPalettes,
    ) -> Result<element::SVG, DrawError>;
}

impl AssociatedMeasureAndDraw for CoronalPointsAndCurve {
    type Measure = CoronalMeasure;

    fn draw(
        image: DynamicImage,
        sagittal_points: Self,
        draws: &[Self::Measure],
        hide: &[Self::Measure],
        draw_param: DrawParam,
        svg_size: (usize, usize),
        palettes: ColorPalettes,
    ) -> Result<element::SVG, DrawError> {
        draw_coronal(
            image,
            sagittal_points,
            (draws, hide),
            draw_param,
            svg_size,
            palettes,
        )
    }
}

impl AssociatedMeasureAndDraw for SagittalPoints {
    type Measure = SagittalMeasure;

    fn draw(
        image: DynamicImage,
        sagittal_points: Self,
        draws: &[Self::Measure],
        hide: &[Self::Measure],
        draw_param: DrawParam,
        svg_size: (usize, usize),
        palettes: ColorPalettes,
    ) -> Result<element::SVG, DrawError> {
        draw_sagittal(
            image,
            sagittal_points,
            draws,
            hide,
            draw_param,
            svg_size,
            palettes,
        )
    }
}

fn process_one<T>(
    svg_common: ReadSvgArgCommon,
    mut point_with_image: PointDataWithImage<T>,
    draws: &[T::Measure],
    hide: &[T::Measure],
    output: &std::path::Path,
) -> Result<()>
where
    T: UpdatePoints + AssociatedMeasureAndDraw + HasImageMetadata + Clone + Scalable,
    <T as AssociatedMeasureAndDraw>::Measure: MeasureAndDraw + Clone + Copy + PartialEq,
{
    if let Some(resize_param) = svg_common.resize_param {
        point_with_image.resize(&resize_param);
    }
    let svg_size = if let Some(_svg_size_param) = svg_common.svg_size_param {
        // TODO: implement
        // data.data
        //     .scale(svg_size_param.scale(data.image.width(), data.image.height()));
        // svg_size_param.size(data.image.width(), data.image.height())
        unimplemented!("svg_size_param")
    } else {
        point_with_image.data_image.image.dimensions()
    };
    let svg_size = (svg_size.0 as usize, svg_size.1 as usize);

    let document = T::draw(
        point_with_image.data_image.image,
        point_with_image.data,
        draws,
        hide,
        svg_common.draw_param,
        svg_size,
        svg_common.palettes,
    )?;

    debug!("Save to {:?}", output);

    std::fs::write(output, document.to_string())?;
    Ok(())
}

fn load_from_native_json_file<T>(path: &Path) -> Result<PointDataWithImage<T>>
where
    T: UpdatePoints + HasImageMetadata + Clone,
    for<'de> T: Deserialize<'de>,

    // <T as TryFromJson>::Error: std::marker::Sync + std::marker::Send + std::error::Error + 'static,
    LabelMeData: From<T>,
{
    let json = std::fs::read_to_string(path)?;
    // let cp = T::try_from_native_json(&json).with_context(|| format!("Load {:?}", path))?;
    let cp: T = serde_json::from_str(&json)?;
    let data = LabelMeData::from(cp.clone());
    let data_w_image = LabelMeDataWImage::try_from_data_and_path(data, path)?;
    Ok(PointDataWithImage::new(cp, data_w_image))
}

fn load_from_labelme_json_file<T>(path: &Path, pull_spacing: bool) -> Result<PointDataWithImage<T>>
where
    T: UpdatePoints + HasImageMetadata + Clone + 'static,
    for<'de> T: Deserialize<'de>,

    for<'a> T: TryFrom<&'a LabelMeData>,
    for<'a> <T as TryFrom<&'a LabelMeData>>::Error:
        std::marker::Sync + std::marker::Send + std::error::Error + 'static,
{
    let data_image: LabelMeDataWImage = path
        .try_into()
        .with_context(|| format!("Load LabelMeData from {:?}", path))?;

    let mut cp = T::try_from(&data_image.data)?;
    if pull_spacing {
        let metadata = ImageMetadata::try_from(Path::new(&data_image.data.imagePath))?;
        *cp.image_metadata_mut() = metadata;
    }
    Ok(PointDataWithImage::new(cp, data_image))
}

fn load_native_or_lableme_json_file<T>(
    path: &Path,
    labelme: bool,
    pull_spacing: bool,
) -> Result<PointDataWithImage<T>>
where
    T: UpdatePoints + HasImageMetadata + Clone + 'static,
    for<'de> T: Deserialize<'de>,

    LabelMeData: From<T>,
    for<'a> T: TryFrom<&'a LabelMeData>,
    for<'a> <T as TryFrom<&'a LabelMeData>>::Error:
        std::marker::Sync + std::marker::Send + std::error::Error + 'static,
{
    if labelme {
        load_from_labelme_json_file(path, pull_spacing)
    } else {
        load_from_native_json_file(path)
    }
}

pub fn cmd(args: SvgArgs) -> Result<()> {
    let (svg_common, subcommand) = load_svg_common(args.svg_args)?;
    match subcommand {
        SvgSubCommands::Coronal(svg_sub_coronal_args) => {
            let data: PointDataWithImage<CoronalPointsAndCurve> = load_native_or_lableme_json_file(
                &args.input,
                svg_common.labelme,
                svg_common.pull_spacing,
            )?;
            let draws = svg_sub_coronal_args
                .measures
                .unwrap_or_else(CoronalMeasure::all_draws);
            let mut hide = svg_sub_coronal_args.hide;
            for group in svg_sub_coronal_args.hide_group {
                hide.append(&mut (&group).into());
            }
            process_one(svg_common, data, &draws, &hide, &args.output)?;
        }
        SvgSubCommands::Sagittal(svg_sub_sagittall_args) => {
            let data: PointDataWithImage<SagittalPoints> = load_native_or_lableme_json_file(
                &args.input,
                svg_common.labelme,
                svg_common.pull_spacing,
            )?;
            let draws = svg_sub_sagittall_args
                .measures
                .unwrap_or_else(SagittalMeasure::all_draws);
            let hide = svg_sub_sagittall_args.hide;
            process_one(svg_common, data, &draws, &hide, &args.output)?;
        }
    };
    Ok(())
}

#[derive(Clone)]
struct ReadSvgArgCommon {
    draw_param: scolrs::DrawParam,
    resize_param: Option<labelme_rs::ResizeParam>,
    svg_size_param: Option<labelme_rs::ResizeParam>,
    palettes: ColorPalettes,
    labelme: bool,
    pull_spacing: bool,
}

fn load_svg_common(args: SvgArgsCommon) -> Result<(ReadSvgArgCommon, SvgSubCommands)> {
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
        ColorPalette::new(scolrs::draw::load_line_colors(reader)?)
    } else {
        ColorPalette::new(scolrs::draw::LineColors::default())
    };

    let palettes = ColorPalettes {
        label_colors,
        line_colors,
    };

    Ok((
        ReadSvgArgCommon {
            draw_param,
            resize_param,
            svg_size_param,
            palettes,
            labelme: args.labelme,
            pull_spacing: args.pull_spacing,
        },
        args.subcommand,
    ))
}

fn load_from_native_json_line<S>(
    json: &str,
    path: &Path,
) -> Result<(
    PointDataWithImage<<S as ContentFilename>::ContentType>,
    String,
)>
where
    S: ContentFilename,
    for<'de> S: Deserialize<'de>,
    <S as ContentFilename>::ContentType: UpdatePoints + HasImageMetadata + Clone,
    LabelMeData: From<<S as ContentFilename>::ContentType>,
{
    let content_filename: S = serde_json::from_str(json)?;
    let (content, filename) = content_filename.content_filename();
    let data = LabelMeData::from(content.clone());
    let data_w_image = LabelMeDataWImage::try_from_data_and_path(data, path)?;
    Ok((PointDataWithImage::new(content, data_w_image), filename))
}

pub fn cmd_ndjson(args: SvgNdjsonArgs) -> Result<()> {
    if let Some(jobs) = args.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()?;
    }

    let (svg_common, subcommand) = load_svg_common(args.svg_args)?;

    let reader = if args.input.as_os_str() == "-" {
        Box::new(BufReader::new(std::io::stdin())) as Box<dyn BufRead>
    } else {
        Box::new(BufReader::new(std::fs::File::open(&args.input)?))
    };
    let lines = reader.lines().collect::<Result<Vec<_>, _>>()?;
    lines.into_par_iter().try_for_each(|line| -> Result<()> {
        match subcommand.clone() {
            SvgSubCommands::Coronal(svg_sub_coronal_args) => {
                let (data, filename) =
                    load_from_native_json_line::<CoronalPointsAndCurveLine>(&line, &args.input)?;
                let draws = svg_sub_coronal_args
                    .measures
                    .unwrap_or_else(CoronalMeasure::all_draws);
                let mut hide = svg_sub_coronal_args.hide;
                for group in svg_sub_coronal_args.hide_group {
                    hide.append(&mut (&group).into());
                }

                let output = args.output.join(
                    Path::new(filename.as_str())
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .to_string()
                        + ".svg",
                );
                process_one(svg_common.clone(), data, &draws, &hide, &output)?;
            }
            SvgSubCommands::Sagittal(svg_sub_sagittall_args) => {
                let (data, filename) =
                    load_from_native_json_line::<SagittalPointsLine>(&line, &args.input)?;
                let draws = svg_sub_sagittall_args
                    .measures
                    .unwrap_or_else(SagittalMeasure::all_draws);
                let hide = svg_sub_sagittall_args.hide;

                let output = args.output.join(
                    Path::new(filename.as_str())
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .to_string()
                        + ".svg",
                );
                process_one(svg_common.clone(), data, &draws, &hide, &output)?;
            }
        };
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
