use std::fs::File;
use std::io::{self, BufRead, BufReader};

use crate::neck::cli::ConvArgs;
use anyhow::{bail, Result};
use labelme_rs::{LabelMeData, LabelMeDataLine};
use log::warn;
use scolrs::head_neck::{LateralPointsIR, LateralPointsIRLine, TryConvertContentFilename};
use scolrs::{
    CoronalPointsIR, CoronalPointsIRLine, HasImageMetadata, SagittalPointsIR, SagittalPointsIRLine,
};

use super::cli::ConvFormat;

fn process_ndjson(
    from: ConvFormat,
    to: ConvFormat,
    reader: Box<dyn BufRead>,
    mut writer: Box<dyn io::Write>,
    pull_spacing: bool,
) -> Result<()> {
    for line in reader.lines() {
        let line = line?;
        match (from, to) {
            (ConvFormat::Labelme, ConvFormat::LateralPoints) => {
                let from_data = LabelMeDataLine::try_from(line.as_str())?;
                let to_data: LateralPointsIRLine =
                    TryConvertContentFilename::try_convert_from(from_data)?;
                serde_json::to_writer(&mut writer, &to_data)?;
            }
            (ConvFormat::LateralPoints, ConvFormat::Labelme) => {
                let from_data = LateralPointsIRLine::try_from(line.as_str())?;
                let to_data: LabelMeDataLine =
                    TryConvertContentFilename::<LateralPointsIRLine>::try_convert_from(from_data)?;
                serde_json::to_writer(&mut writer, &to_data)?;
            }
            (ConvFormat::Labelme, ConvFormat::ScoliosisCoronal) => {
                let from_data = LabelMeDataLine::try_from(line.as_str())?;
                let mut to_data: CoronalPointsIRLine =
                    TryConvertContentFilename::try_convert_from(from_data)?;
                if pull_spacing {
                    to_data.content.pull_image_metadata()?;
                }
                serde_json::to_writer(&mut writer, &to_data)?;
            }
            (ConvFormat::ScoliosisCoronal, ConvFormat::Labelme) => {
                let from_data = CoronalPointsIRLine::try_from(line.as_str())?;
                let to_data: LabelMeDataLine =
                    TryConvertContentFilename::<CoronalPointsIRLine>::try_convert_from(from_data)?;
                serde_json::to_writer(&mut writer, &to_data)?;
            }
            (ConvFormat::Labelme, ConvFormat::ScoliosisSagittal) => {
                let from_data = LabelMeDataLine::try_from(line.as_str())?;
                let mut to_data: SagittalPointsIRLine =
                    TryConvertContentFilename::try_convert_from(from_data)?;
                if pull_spacing {
                    to_data.content.pull_image_metadata()?;
                }
                serde_json::to_writer(&mut writer, &to_data)?;
            }
            (ConvFormat::ScoliosisSagittal, ConvFormat::Labelme) => {
                let from_data = SagittalPointsIRLine::try_from(line.as_str())?;
                let to_data: LabelMeDataLine =
                    TryConvertContentFilename::<SagittalPointsIRLine>::try_convert_from(from_data)?;
                serde_json::to_writer(&mut writer, &to_data)?;
            }
            (from, to) => {
                if from == to {
                    bail!("No conversion needed")
                } else {
                    bail!("Invalid conversion from {:?} to {:?}", from, to)
                }
            }
        }
        writeln!(&mut writer)?;
    }
    Ok(())
}

fn get_pixel_spacing(path: &str) -> Result<Option<(f64, f64)>> {
    use dicom_dictionary_std::tags;
    let obj = dicom_object::open_file(path)?;
    let spacing = obj.get(tags::PIXEL_SPACING);
    if let Some(spacing) = spacing {
        let spacing = spacing.to_multi_float64()?;
        return Ok(Some((spacing[0], spacing[1])));
    } else {
        let spacing = obj.get(tags::IMAGER_PIXEL_SPACING);
        if let Some(spacing) = spacing {
            let spacing = spacing.to_multi_float64()?;
            warn!("Using Imager Pixel Spacing (0018,1164) instead of Pixel Spacing (0028,0030)");
            return Ok(Some((spacing[0], spacing[1])));
        }
    }
    Ok(None)
}

trait PullImageMetadata {
    fn pull_image_metadata(&mut self) -> Result<()>;
}

impl<T> PullImageMetadata for T
where
    T: HasImageMetadata,
{
    fn pull_image_metadata(&mut self) -> Result<()> {
        let metadata = self.image_metadata_mut();
        if metadata.path.ends_with(".dcm")
            || metadata.path.ends_with(".DCM")
            || metadata.path.ends_with(".dicom")
            || metadata.path.ends_with(".DICOM")
        {
            if let Some(spacing) = get_pixel_spacing(&metadata.path)? {
                metadata.spacing_xy = spacing;
                metadata.unit = "mm".to_string();
            } else {
                warn!("No pixel spacing found in dicom: {:?}", metadata.path);
            }
        } else {
            warn!("No dicom: {:?}", metadata.path);
        }
        Ok(())
    }
}

fn process_json(
    from: ConvFormat,
    to: ConvFormat,
    reader: Box<dyn BufRead>,
    mut writer: Box<dyn io::Write>,
    pull_spacing: bool,
) -> Result<()> {
    match (from, to) {
        (ConvFormat::Labelme, ConvFormat::LateralPoints) => {
            let from_data: LabelMeData = serde_json::from_reader(reader)?;
            let mut to_data: LateralPointsIR = from_data.try_into()?;
            if pull_spacing {
                to_data.pull_image_metadata()?;
            }
            serde_json::to_writer(&mut writer, &to_data)?
        }
        (ConvFormat::LateralPoints, ConvFormat::Labelme) => {
            let from_data: LateralPointsIR = serde_json::from_reader(reader)?;
            let to_data: LabelMeData = from_data.try_into()?;
            serde_json::to_writer(&mut writer, &to_data)?
        }
        (ConvFormat::Labelme, ConvFormat::ScoliosisCoronal) => {
            let from_data: LabelMeData = serde_json::from_reader(reader)?;
            let mut to_data: CoronalPointsIR = from_data.try_into()?;
            if pull_spacing {
                to_data.pull_image_metadata()?;
            }
            serde_json::to_writer(&mut writer, &to_data)?
        }
        (ConvFormat::ScoliosisCoronal, ConvFormat::Labelme) => {
            let from_data: CoronalPointsIR = serde_json::from_reader(reader)?;
            let to_data: LabelMeData = from_data.into();
            serde_json::to_writer(&mut writer, &to_data)?
        }
        (ConvFormat::Labelme, ConvFormat::ScoliosisSagittal) => {
            let from_data: LabelMeData = serde_json::from_reader(reader)?;
            let mut to_data: SagittalPointsIR = from_data.try_into()?;
            if pull_spacing {
                to_data.pull_image_metadata()?;
            }
            serde_json::to_writer(&mut writer, &to_data)?
        }
        (ConvFormat::ScoliosisSagittal, ConvFormat::Labelme) => {
            let from_data: SagittalPointsIR = serde_json::from_reader(reader)?;
            let to_data: LabelMeData = from_data.into();
            serde_json::to_writer(&mut writer, &to_data)?
        }
        (from, to) => {
            if from == to {
                bail!("No conversion needed")
            } else {
                bail!("Invalid conversion from {:?} to {:?}", from, to)
            }
        }
    };
    Ok(())
}

pub fn cmd(args: ConvArgs) -> Result<()> {
    let reader: Box<dyn BufRead> = if let Some(input) = args.input {
        Box::new(BufReader::new(File::open(input)?))
    } else {
        Box::new(BufReader::new(io::stdin()))
    };
    let writer: Box<dyn io::Write> = if let Some(output) = args.output {
        Box::new(File::create(output)?)
    } else {
        Box::new(io::stdout())
    };

    if args.ndjson {
        process_ndjson(args.from, args.to, reader, writer, args.pull_spacing)
    } else {
        process_json(args.from, args.to, reader, writer, args.pull_spacing)
    }
}

// TODO: Add tests for pull_spacing
