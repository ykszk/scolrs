use std::fs::File;
use std::io::{self, BufRead, BufReader};

use crate::neck::cli::ConvArgs;
use anyhow::{bail, Result};
use labelme_rs::{LabelMeData, LabelMeDataLine};
use scolrs::head_neck::{LateralPointsIR, LateralPointsIRLine, TryConvertContentFilename};
use scolrs::{
    CoronalPointsIR, CoronalPointsIRLine, DicomError, PullImageMetadata, SagittalPointsIR,
    SagittalPointsIRLine,
};

use super::cli::ConvFormat;

fn conv_and_write_line<From, To>(
    writer: &mut Box<dyn io::Write>,
    from_str: String,
    pull_spacing: bool,
) -> Result<()>
where
    To: TryConvertContentFilename<From, Error: std::error::Error + Send + Sync>,
    <To as scolrs::head_neck::TryConvertContentFilename<From>>::Error: 'static,
    To: serde::Serialize + PullIfImplemented,
    From: scolrs::ContentFilename,
    From: for<'a> TryFrom<&'a str, Error = serde_json::Error>,
{
    let from = From::try_from(from_str.as_str())?;
    let mut to: To = TryConvertContentFilename::try_convert_from(from)?;
    if pull_spacing {
        to.pull_if_implemented()?;
    }
    serde_json::to_writer(writer, &to)?;
    Ok(())
}

fn conv_and_write<From, To>(
    writer: &mut Box<dyn io::Write>,
    from_str: String,
    pull_spacing: bool,
) -> Result<()>
where
    To: TryFrom<From, Error: std::error::Error + Send + Sync>,
    <To as TryFrom<From>>::Error: 'static,
    To: serde::Serialize + PullIfImplemented,
    From: for<'a> TryFrom<&'a str, Error = serde_json::Error>,
{
    let from = From::try_from(from_str.as_str())?;
    let mut to: To = To::try_from(from)?;
    if pull_spacing {
        to.pull_if_implemented()?;
    }
    serde_json::to_writer(writer, &to)?;
    Ok(())
}

/// Trait to help convert_and_write
trait PullIfImplemented {
    /// Pull image metadata if implemented
    ///
    /// Just return Ok(()) if not implemented
    fn pull_if_implemented(&mut self) -> Result<(), DicomError>;
}

impl PullIfImplemented for LateralPointsIRLine {
    fn pull_if_implemented(&mut self) -> Result<(), DicomError> {
        self.content.pull_image_metadata()
    }
}
impl PullIfImplemented for CoronalPointsIRLine {
    fn pull_if_implemented(&mut self) -> Result<(), DicomError> {
        self.content.pull_image_metadata()
    }
}
impl PullIfImplemented for SagittalPointsIRLine {
    fn pull_if_implemented(&mut self) -> Result<(), DicomError> {
        self.content.pull_image_metadata()
    }
}
impl PullIfImplemented for LabelMeDataLine {
    fn pull_if_implemented(&mut self) -> Result<(), DicomError> {
        Ok(())
    }
}

impl PullIfImplemented for LateralPointsIR {
    fn pull_if_implemented(&mut self) -> Result<(), DicomError> {
        self.pull_image_metadata()
    }
}
impl PullIfImplemented for CoronalPointsIR {
    fn pull_if_implemented(&mut self) -> Result<(), DicomError> {
        self.pull_image_metadata()
    }
}
impl PullIfImplemented for SagittalPointsIR {
    fn pull_if_implemented(&mut self) -> Result<(), DicomError> {
        self.pull_image_metadata()
    }
}
impl PullIfImplemented for LabelMeData {
    fn pull_if_implemented(&mut self) -> Result<(), DicomError> {
        Ok(())
    }
}

fn process_ndjson(
    from: ConvFormat,
    to: ConvFormat,
    reader: Box<dyn BufRead>,
    mut writer: Box<dyn io::Write>,
    pull_spacing: bool,
) -> Result<()> {
    // shorthands to keep match hands one-line
    let pull = pull_spacing;
    type LMDLine = LabelMeDataLine;
    type LatPtsLine = LateralPointsIRLine;
    type CorPointsLine = CoronalPointsIRLine;
    type SagPointsLine = SagittalPointsIRLine;
    for line in reader.lines() {
        let line = line?;
        match (from, to) {
            (ConvFormat::Labelme, ConvFormat::LateralPoints) => {
                conv_and_write_line::<LMDLine, LatPtsLine>(&mut writer, line, pull)?;
            }
            (ConvFormat::LateralPoints, ConvFormat::Labelme) => {
                conv_and_write_line::<LatPtsLine, LMDLine>(&mut writer, line, pull)?;
            }
            (ConvFormat::Labelme, ConvFormat::ScoliosisCoronal) => {
                conv_and_write_line::<LMDLine, CorPointsLine>(&mut writer, line, pull)?;
            }
            (ConvFormat::ScoliosisCoronal, ConvFormat::Labelme) => {
                conv_and_write_line::<CorPointsLine, LMDLine>(&mut writer, line, pull)?;
            }
            (ConvFormat::Labelme, ConvFormat::ScoliosisSagittal) => {
                conv_and_write_line::<LMDLine, SagPointsLine>(&mut writer, line, pull)?;
            }
            (ConvFormat::ScoliosisSagittal, ConvFormat::Labelme) => {
                conv_and_write_line::<SagPointsLine, LMDLine>(&mut writer, line, pull)?;
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

fn process_json(
    from: ConvFormat,
    to: ConvFormat,
    reader: Box<dyn BufRead>,
    mut writer: Box<dyn io::Write>,
    pull_spacing: bool,
) -> Result<()> {
    let json_str = std::io::read_to_string(reader)?;

    match (from, to) {
        (ConvFormat::Labelme, ConvFormat::LateralPoints) => {
            conv_and_write::<LabelMeData, LateralPointsIR>(&mut writer, json_str, pull_spacing)?;
        }
        (ConvFormat::LateralPoints, ConvFormat::Labelme) => {
            conv_and_write::<LateralPointsIR, LabelMeData>(&mut writer, json_str, pull_spacing)?;
        }
        (ConvFormat::Labelme, ConvFormat::ScoliosisCoronal) => {
            conv_and_write::<LabelMeData, CoronalPointsIR>(&mut writer, json_str, pull_spacing)?;
        }
        (ConvFormat::ScoliosisCoronal, ConvFormat::Labelme) => {
            conv_and_write::<CoronalPointsIR, LabelMeData>(&mut writer, json_str, pull_spacing)?;
        }
        (ConvFormat::Labelme, ConvFormat::ScoliosisSagittal) => {
            conv_and_write::<LabelMeData, SagittalPointsIR>(&mut writer, json_str, pull_spacing)?;
        }
        (ConvFormat::ScoliosisSagittal, ConvFormat::Labelme) => {
            conv_and_write::<SagittalPointsIR, LabelMeData>(&mut writer, json_str, pull_spacing)?;
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
