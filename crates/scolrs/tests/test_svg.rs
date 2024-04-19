use anyhow::{Context, Result};
use labelme_rs::{LabelMeData, LabelMeDataWImage, ResizeParam};
use scolrs::{
    draw_coronal, draw_sagittal, ApexSet, ColorPalette, CoronalPoints, CurveSet, DrawParam,
    SagittalMeasure, SagittalPoints, Spine,
};
use std::path::{Path, PathBuf};
use svg::Document;

fn test_data_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

fn tmp_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
}

const SVG_SIZE: (usize, usize) = (1024usize, 1024usize);

fn _test_svg(
    json_filename: &Path,
    curve_apex_set: Option<(CurveSet, ApexSet)>,
    coronal: bool,
) -> Result<Document> {
    let data_dir = test_data_directory();
    let draw_param = DrawParam::default();

    let label_colors = data_dir.join("colors.yaml");
    let line_colors = data_dir.join("line_colors.csv");

    let mut data: LabelMeDataWImage = json_filename
        .try_into()
        .with_context(|| format!("Load LabelMeDataWImage from {:?}", &json_filename))?;

    let svg_size_param = ResizeParam::Size(SVG_SIZE.0 as u32, SVG_SIZE.1 as u32);
    data.data
        .scale(svg_size_param.scale(data.image.width(), data.image.height()));
    let size_param = svg_size_param.size(data.image.width(), data.image.height());
    let svg_size = (size_param.0 as usize, size_param.1 as usize);

    let label_colors = ColorPalette::new(
        labelme_rs::load_label_colors(&label_colors)
            .with_context(|| format!("Load label color file {:?}", label_colors))?,
    );
    let line_colors = ColorPalette::new(scolrs::load_line_colors(
        std::fs::File::open(&line_colors)
            .with_context(|| format!("Load line color file {:?}", line_colors))?,
    )?);
    let document = if coronal {
        let coronal_points = CoronalPoints::try_from(&data.data)?;
        draw_coronal(
            data,
            coronal_points,
            draw_param,
            svg_size,
            label_colors,
            line_colors,
            curve_apex_set,
        )
    } else {
        let sagittal_points = SagittalPoints::try_from(&data.data)?;
        let measures = SagittalMeasure::all();
        let hide = Vec::new();
        draw_sagittal(
            data,
            sagittal_points,
            measures,
            hide,
            draw_param,
            svg_size,
            label_colors,
            line_colors,
        )
    };

    Ok(document)
}

fn test_coronal_sagittal(case: &str) -> Result<(Document, Document)> {
    let data_dir = test_data_directory();
    let frontal = data_dir.join(format!("{}/frontal.json", case));
    let d1 = _test_svg(&frontal, None, true)?;
    let lateral = data_dir.join(format!("{}/lateral.json", case));
    let d2 = _test_svg(&lateral, None, false)?;
    Ok((d1, d2))
}

fn test_sagittal_coronal_write(case: &str) -> Result<()> {
    let tmp_dir = tmp_directory();
    let (d1, d2) = test_coronal_sagittal(case)?;
    let out1 = tmp_dir.join(format!("{}_frontal.svg", case));
    std::fs::write(out1, d1.to_string())?;
    let out2 = tmp_dir.join(format!("{}_lateral.svg", case));
    std::fs::write(out2, d2.to_string())?;
    Ok(())
}

fn test_bend() -> Result<(Document, Document)> {
    let data_dir = test_data_directory();
    let json_filename = data_dir.join("case1/frontal.json");
    let s = std::fs::read_to_string(json_filename)?;
    let data = LabelMeData::try_from(s)?;
    let spine = Spine::try_from(&data)?;
    let (curves, apex_set, _major_curve) = spine.identify_curves();
    let curve_apex_set = Some((curves, apex_set));
    let coronal = true;
    let d_left = _test_svg(
        &data_dir.join("case1/left_lateral_bend.json"),
        curve_apex_set.clone(),
        coronal,
    )?;
    let d_right = _test_svg(
        &data_dir.join("case1/right_lateral_bend.json"),
        curve_apex_set,
        coronal,
    )?;
    Ok((d_left, d_right))
}

#[test]
fn test_svg_case1() -> Result<()> {
    test_coronal_sagittal("case1")?;
    test_bend()?;
    Ok(())
}
#[ignore]
#[test]
fn test_svg_case1_write() -> Result<()> {
    test_sagittal_coronal_write("case1")?;
    let (d_left, d_right) = test_bend()?;
    let tmp_dir = tmp_directory();
    let out1 = tmp_dir.join("case1_left_lateral_bend.svg");
    std::fs::write(out1, d_left.to_string())?;
    let out2 = tmp_dir.join("case1_right_lateral_bend.svg");
    std::fs::write(out2, d_right.to_string())?;
    Ok(())
}

#[test]
fn test_svg_case2() -> Result<()> {
    test_coronal_sagittal("case2")?;
    Ok(())
}
#[ignore]
#[test]
fn test_svg_case2_write() -> Result<()> {
    test_sagittal_coronal_write("case2")?;
    Ok(())
}

#[test]
fn test_svg_case3() -> Result<()> {
    test_coronal_sagittal("case3")?;
    Ok(())
}
#[ignore]
#[test]
fn test_svg_case3_write() -> Result<()> {
    test_sagittal_coronal_write("case3")?;
    Ok(())
}

#[test]
fn test_svg_case4() -> Result<()> {
    test_coronal_sagittal("case4")?;
    Ok(())
}
#[ignore]
#[test]
fn test_svg_case4_write() -> Result<()> {
    test_sagittal_coronal_write("case4")?;
    Ok(())
}
