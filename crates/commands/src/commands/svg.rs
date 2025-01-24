use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::{
    io::{BufRead, BufReader, Stdout},
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
    draw::{draw_coronal, draw_implant, draw_sagittal, ColorPalette, ColorPalettes, DrawError},
    head_neck::{LateralPoints, LateralPointsLine, NeckLateralDraw},
    implant::{LabelMeOptionalDetectron2, LabelMeOptionalDetectron2Line, ScrewSpine},
    ContentFilename, CoronalDraw, CoronalPointsAndCurve, CoronalPointsAndCurveLine, DrawParam,
    HasImageMetadata, ImageMetadata, ImplantDraw, MeasureAndDraw, PointDataWithImage, SagittalDraw,
    SagittalPoints, SagittalPointsLine, Scalable,
};
use serde::Deserialize;
use svg::node::element;
use tar;

trait FsWrite: Send + Sync {
    fn write(&mut self, path: &Path, content: String) -> Result<()>;
}

struct FsWriter;

impl FsWrite for FsWriter {
    fn write(&mut self, path: &Path, content: String) -> Result<()> {
        std::fs::write(path, content)?;
        Ok(())
    }
}

struct StdoutWriter;

impl FsWrite for StdoutWriter {
    fn write(&mut self, _path: &Path, content: String) -> Result<()> {
        println!("{}", content);
        Ok(())
    }
}

struct TarWriter {
    builder: Box<tar::Builder<Stdout>>,
}

impl FsWrite for TarWriter {
    fn write(&mut self, path: &Path, content: String) -> Result<()> {
        let mut header = tar::Header::new_gnu();
        header.set_size(content.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        self.builder
            .append_data(&mut header, path, content.as_bytes())?;
        Ok(())
    }
}

impl FsWrite for Box<dyn FsWrite> {
    fn write(&mut self, path: &Path, content: String) -> Result<()> {
        self.as_mut().write(path, content)
    }
}

impl FsWrite for Arc<Mutex<Box<dyn FsWrite>>> {
    fn write(&mut self, path: &Path, content: String) -> Result<()> {
        self.lock().unwrap().write(path, content)
    }
}

trait AssociatedMeasureAndDraw {
    type Draw;
    fn draw(
        image: DynamicImage,
        sagittal_points: Self,
        draws_hide: (&[Self::Draw], &[Self::Draw]),
        draw_param: DrawParam,
        resize_param: Option<ResizeParam>,
        svg_size: (usize, usize),
        palettes: ColorPalettes,
    ) -> Result<element::SVG, DrawError>;
}

impl AssociatedMeasureAndDraw for CoronalPointsAndCurve {
    type Draw = CoronalDraw;

    fn draw(
        image: DynamicImage,
        sagittal_points: Self,
        draws_hide: (&[Self::Draw], &[Self::Draw]),
        draw_param: DrawParam,
        resize_param: Option<ResizeParam>,
        svg_size: (usize, usize),
        palettes: ColorPalettes,
    ) -> Result<element::SVG, DrawError> {
        draw_coronal(
            image,
            sagittal_points,
            draws_hide,
            draw_param,
            resize_param,
            svg_size,
            palettes,
        )
    }
}

impl AssociatedMeasureAndDraw for ScrewSpine {
    type Draw = ImplantDraw;

    fn draw(
        image: DynamicImage,
        screw_spine: Self,
        draws_hide: (&[Self::Draw], &[Self::Draw]),
        draw_param: DrawParam,
        resize_param: Option<ResizeParam>,
        svg_size: (usize, usize),
        palettes: ColorPalettes,
    ) -> Result<element::SVG, DrawError> {
        draw_implant(
            image,
            screw_spine,
            draws_hide,
            draw_param,
            resize_param,
            svg_size,
            palettes,
        )
    }
}

impl AssociatedMeasureAndDraw for SagittalPoints {
    type Draw = SagittalDraw;

    fn draw(
        image: DynamicImage,
        sagittal_points: Self,
        draws_hide: (&[Self::Draw], &[Self::Draw]),
        draw_param: DrawParam,
        resize_param: Option<ResizeParam>,
        svg_size: (usize, usize),
        palettes: ColorPalettes,
    ) -> Result<element::SVG, DrawError> {
        draw_sagittal(
            image,
            sagittal_points,
            draws_hide,
            draw_param,
            resize_param,
            svg_size,
            palettes,
        )
    }
}

impl AssociatedMeasureAndDraw for LateralPoints {
    type Draw = NeckLateralDraw;

    fn draw(
        image: DynamicImage,
        sagittal_points: Self,
        draws_hide: (&[Self::Draw], &[Self::Draw]),
        draw_param: DrawParam,
        resize_param: Option<ResizeParam>,
        svg_size: (usize, usize),
        palettes: ColorPalettes,
    ) -> Result<element::SVG, DrawError> {
        scolrs::draw::draw_on_image(
            image,
            sagittal_points,
            draws_hide,
            draw_param,
            resize_param,
            svg_size,
            palettes,
        )
    }
}

fn process_one<T, S: FsWrite>(
    svg_common: ReadSvgArgCommon,
    point_with_image: PointDataWithImage<T>,
    draws: &[T::Draw],
    hide: &[T::Draw],
    output: &std::path::Path,
    mut writer: S,
) -> Result<()>
where
    T: AssociatedMeasureAndDraw + HasImageMetadata + Clone + Scalable,
    <T as AssociatedMeasureAndDraw>::Draw: Clone + Copy + PartialEq,
{
    // if let Some(resize_param) = svg_common.resize_param {
    //     point_with_image.resize(&resize_param);
    // }
    let svg_size = if let Some(svg_size_param) = svg_common.svg_size_param {
        match svg_size_param {
            ResizeParam::Percentage(_) => panic!("Percentage is not supported for svg size"),
            ResizeParam::Size(w, h) => {
                if w > h {
                    let image_aspect_ratio = point_with_image.data_image.image.width() as f64
                        / point_with_image.data_image.image.height() as f64;
                    let adjusted_height = (w as f64 / image_aspect_ratio) as u32;
                    (w, adjusted_height)
                } else {
                    let image_aspect_ratio = point_with_image.data_image.image.height() as f64
                        / point_with_image.data_image.image.width() as f64;
                    let adjusted_width = (h as f64 / image_aspect_ratio) as u32;
                    (adjusted_width, h)
                }
            }
        }
    } else {
        point_with_image.data_image.image.dimensions()
    };
    let svg_size = (svg_size.0 as usize, svg_size.1 as usize);

    let document = T::draw(
        point_with_image.data_image.image,
        point_with_image.data,
        (draws, hide),
        svg_common.draw_param,
        svg_common.resize_param,
        svg_size,
        svg_common.palettes,
    )?;

    debug!("Save to {:?}", output);

    writer.write(output, document.to_string())?;
    Ok(())
}

fn load_from_native_json_file<T>(path: &Path) -> Result<PointDataWithImage<T>>
where
    T: HasImageMetadata + Clone,
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
    T: HasImageMetadata + Clone + 'static,

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
    T: HasImageMetadata + Clone + 'static,
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
    let writer = if args.output.as_os_str() == "-" {
        Box::new(StdoutWriter) as Box<dyn FsWrite>
    } else {
        Box::new(FsWriter) as Box<dyn FsWrite>
    };
    match subcommand {
        SvgSubCommands::Coronal(svg_sub_coronal_args) => {
            let data: PointDataWithImage<CoronalPointsAndCurve> = load_native_or_lableme_json_file(
                &args.input,
                svg_common.labelme,
                svg_common.pull_spacing,
            )?;
            let draws = svg_sub_coronal_args
                .measures
                .unwrap_or_else(CoronalDraw::all);
            let mut hide = svg_sub_coronal_args.hide;
            for group in svg_sub_coronal_args.hide_group {
                hide.append(&mut (&group).into());
            }
            process_one(svg_common, data, &draws, &hide, &args.output, writer)?;
        }
        SvgSubCommands::Sagittal(svg_sub_sagittall_args) => {
            let data: PointDataWithImage<SagittalPoints> = load_native_or_lableme_json_file(
                &args.input,
                svg_common.labelme,
                svg_common.pull_spacing,
            )?;
            let draws = svg_sub_sagittall_args
                .measures
                .unwrap_or_else(SagittalDraw::all);
            let hide = svg_sub_sagittall_args.hide;
            process_one(svg_common, data, &draws, &hide, &args.output, writer)?;
        }
        SvgSubCommands::Neck(svg_sub_neck_args) => {
            let data: PointDataWithImage<LateralPoints> = load_native_or_lableme_json_file(
                &args.input,
                svg_common.labelme,
                svg_common.pull_spacing,
            )?;
            let draws = svg_sub_neck_args
                .measures
                .unwrap_or_else(NeckLateralDraw::all);
            let hide = svg_sub_neck_args.hide;
            process_one(svg_common, data, &draws, &hide, &args.output, writer)?;
        }
        SvgSubCommands::CoronalImplant(svg_sub_implant_args) => {
            let data: LabelMeOptionalDetectron2 =
                serde_json::from_str(&std::fs::read_to_string(&args.input)?)?;
            let screw_spine = data.screw_spine()?;
            let data_w_image =
                LabelMeDataWImage::try_from_data_and_path(data.labelme, &args.input)?;
            let data = PointDataWithImage::new(screw_spine, data_w_image);

            let draws = svg_sub_implant_args
                .measures
                .unwrap_or_else(ImplantDraw::all);
            let hide = svg_sub_implant_args.hide;
            process_one(svg_common, data, &draws, &hide, &args.output, writer)?;
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
    <S as ContentFilename>::ContentType: HasImageMetadata + Clone,
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
    let (writer, output_dir) = if args.output.as_os_str() == "-" {
        let output_dir = PathBuf::default();
        let builder = tar::Builder::new(std::io::stdout());
        (
            Box::new(TarWriter {
                builder: Box::new(builder),
            }) as Box<dyn FsWrite>,
            output_dir,
        )
    } else {
        (Box::new(FsWriter) as Box<dyn FsWrite>, args.output)
    };
    let writer = Arc::new(Mutex::new(writer));
    let lines = reader.lines().collect::<Result<Vec<_>, _>>()?;
    lines.into_par_iter().try_for_each(|line| -> Result<()> {
        let writer = writer.clone();
        match subcommand.clone() {
            SvgSubCommands::Coronal(svg_sub_coronal_args) => {
                let (data, filename) =
                    load_from_native_json_line::<CoronalPointsAndCurveLine>(&line, &args.input)?;
                let draws = svg_sub_coronal_args
                    .measures
                    .unwrap_or_else(CoronalDraw::all);
                let mut hide = svg_sub_coronal_args.hide;
                for group in svg_sub_coronal_args.hide_group {
                    hide.append(&mut (&group).into());
                }

                let output = output_dir.join(
                    Path::new(filename.as_str())
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .to_string()
                        + ".svg",
                );
                process_one(svg_common.clone(), data, &draws, &hide, &output, writer)?;
            }
            SvgSubCommands::Sagittal(svg_sub_sagittall_args) => {
                let (data, filename) =
                    load_from_native_json_line::<SagittalPointsLine>(&line, &args.input)?;
                let draws = svg_sub_sagittall_args
                    .measures
                    .unwrap_or_else(SagittalDraw::all);
                let hide = svg_sub_sagittall_args.hide;

                let output = output_dir.join(
                    Path::new(filename.as_str())
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .to_string()
                        + ".svg",
                );
                process_one(svg_common.clone(), data, &draws, &hide, &output, writer)?;
            }
            SvgSubCommands::Neck(svg_sub_neck_args) => {
                let (data, filename) =
                    load_from_native_json_line::<LateralPointsLine>(&line, &args.input)?;
                let draws = svg_sub_neck_args
                    .measures
                    .unwrap_or_else(NeckLateralDraw::all);
                let hide = svg_sub_neck_args.hide;

                let output = output_dir.join(
                    Path::new(filename.as_str())
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .to_string()
                        + ".svg",
                );
                process_one(svg_common.clone(), data, &draws, &hide, &output, writer)?;
            }
            SvgSubCommands::CoronalImplant(svg_sub_implant_args) => {
                let data_line: LabelMeOptionalDetectron2Line = serde_json::from_str(line.as_str())?;
                let screw_spine = data_line.content.screw_spine()?;
                let data_w_image = LabelMeDataWImage::try_from_data_and_path(
                    data_line.content.labelme,
                    &args.input,
                )?;
                let data = PointDataWithImage::new(screw_spine, data_w_image);

                let draws = svg_sub_implant_args
                    .measures
                    .unwrap_or_else(ImplantDraw::all);
                let hide = svg_sub_implant_args.hide;

                let output = output_dir.join(
                    Path::new(data_line.filename.as_str())
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .to_string()
                        + ".svg",
                );
                process_one(svg_common.clone(), data, &draws, &hide, &output, writer)?;
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
        let resize = Some("800x800".to_string());
        let size = Some("1024x1024".to_string());
        let svg_args = SvgArgsCommon {
            config,
            label_colors,
            line_colors,
            resize,
            size,
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
            svg_args.output = output_path(&format!("scoliosis/{}_frontal_native.svg", case))?;
        } else {
            svg_args.input = data_dir.join(case).join("frontal.json");
            svg_args.output = output_path(&format!("scoliosis/{}_frontal.svg", case))?;
        }
        svg_args.svg_args.labelme = !native;
        svg_args.svg_args.subcommand = SvgSubCommands::Coronal(Default::default());
        cmd(svg_args.clone())?;

        if native {
            svg_args.input = data_dir.join(case).join("lateral_native.json");
            svg_args.output = output_path(&format!("scoliosis/{}_lateral_native.svg", case))?;
        } else {
            svg_args.input = data_dir.join(case).join("lateral.json");
            svg_args.output = output_path(&format!("scoliosis/{}_lateral.svg", case))?;
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
        svg_args.output = output_path("scoliosis/case1_left_lateral_bend.svg")?;
        svg_args.svg_args.labelme = true;
        cmd(svg_args.clone())?;

        svg_args.input = data_dir.join("case1/right_lateral_bend.json");
        svg_args.output = output_path("scoliosis/case1_right_lateral_bend.svg")?;
        cmd(svg_args)?;
        Ok(())
    }

    #[test]
    fn svg_cmd_scol_case2() -> Result<()> {
        _test_case("case2", false)?;
        _test_case("case2_cropped", false)?;
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

    #[test]
    fn svg_cmd_scol_case5() -> Result<()> {
        // test bending
        let mut svg_args = gen_svg_args();
        svg_args.svg_args.subcommand = SvgSubCommands::CoronalImplant(Default::default());
        let data_dir = PathBuf::from("../../tests/data/");
        svg_args.input = data_dir.join("case5/frontal_postop.json");
        svg_args.output = output_path("scoliosis/case5_frontal_postop.svg")?;
        svg_args.svg_args.labelme = true;
        cmd(svg_args)?;

        Ok(())
    }

    // neck
    #[test]
    fn test_svg_cmd_neck_case1() -> Result<()> {
        let mut svg_args = gen_svg_args();
        svg_args.svg_args.subcommand = SvgSubCommands::Neck(Default::default());

        let data_dir = PathBuf::from("../../tests/data/");
        svg_args.input = data_dir.join("neck_case1/lateral_lateral_points.json");
        svg_args.output = output_path("neck_case1/neck_case1_lateral.svg")?;
        cmd(svg_args.clone())?;

        if std::env::var("TEST_OUTPUT_DIR").is_err() {
            return Ok(());
        }

        // test html command
        let html_args = crate::cli::HtmlArgs {
            input: svg_args.output,
            output: None,
            selector: vec!["g.Component".to_string()],
            title: None,
        };
        crate::commands::html::cmd(html_args)?;

        Ok(())
    }

    #[test]
    fn test_svg_cmd_neck_case2() -> Result<()> {
        let mut svg_args = gen_svg_args();
        svg_args.svg_args.resize = Some("768x768".to_string());
        svg_args.svg_args.subcommand = SvgSubCommands::Neck(Default::default());

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
        let mut html_args = crate::cli::HtmlArgs {
            input: output_path("neck_case2/extension_lateral.svg")?,
            output: None,
            selector: vec!["g.Component".to_string()],
            title: None,
        };
        crate::commands::html::cmd(html_args.clone())?;

        html_args.input = output_path("neck_case2/flexion_lateral.svg")?;
        crate::commands::html::cmd(html_args)?;

        // test catalog command with directory input
        let catalog_args = crate::cli::CatalogArgs {
            input: vec![output_path("neck_case2")?],
            output: output_path("neck_case2_catalog.html")?,
            title: Some("Neck Case 2".to_string()),
            selector: vec!["g.Component".to_string()],
            jobs: None,
        };
        crate::commands::catalog::cmd(catalog_args)?;

        // test catalog command with multiple directories
        let catalog_args = crate::cli::CatalogArgs {
            input: vec![
                output_path("neck_case1/neck_case1_lateral.svg")?,
                output_path("neck_case2/flexion_lateral.svg")?,
            ],
            output: output_path("neck_cases_catalog.html")?,
            title: Some("Neck Cases".to_string()),
            selector: vec!["g.Component".to_string()],
            jobs: None,
        };
        crate::commands::catalog::cmd(catalog_args)?;

        // test catalog command with html input
        let catalog_args = crate::cli::CatalogArgs {
            input: vec![
                output_path("neck_case2/flexion_lateral.html")?,
                output_path("neck_case2/extension_lateral.html")?,
            ],
            output: output_path("neck_catalog_from_html.html")?,
            title: Some("Neck Case 2 from HTML".to_string()),
            selector: vec!["g.Component".to_string()],
            jobs: None,
        };
        crate::commands::catalog::cmd(catalog_args)?;

        Ok(())
    }
}
