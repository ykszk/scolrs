use crate::neck::cli::SvgArgs;
use anyhow::{Context, Result};
use labelme_rs::{image::GenericImageView, LabelMeDataWImage};
use log::debug;
use scolrs::head_neck::{
    C1Sac, CervicalPoints, LaminalPoints, NeckSagittalComponent, OptionalPoints,
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

    let common_components: Vec<Box<dyn CommonComponent>> =
        vec![Box::new(CervicalPoints(&cervical_points))];

    for component in common_components {
        let g = component.draw(&painter, &mut label_colors, &mut line_colors)?;
        document = document.add(g);
    }

    let neck_sagittal_components: Vec<Box<dyn NeckSagittalComponent>> = vec![
        Box::new(LaminalPoints(&cervical_points.lamina)),
        Box::new(OptionalPoints(&cervical_points)),
        Box::new(C1Sac(&cervical_points)),
    ];

    for component in neck_sagittal_components {
        let g = component.draw(&painter, &mut label_colors, &mut line_colors)?;
        document = document.add(g);
    }

    debug!("Save to {:?}", args.output);

    std::fs::write(args.output, document.to_string())?;

    Ok(())
}
