use std::fs::File;
use std::io::{self, BufRead, BufReader};

use crate::neck::cli::ConvArgs;
use anyhow::{bail, Result};
use labelme_rs::{LabelMeData, LabelMeDataLine};
use scolrs::head_neck::{LateralPointsIR, LateralPointsIRLine, TryConvertContentFilename};

use super::cli::ConvFormat;

fn process_ndjson(
    from: ConvFormat,
    to: ConvFormat,
    reader: Box<dyn BufRead>,
    mut writer: Box<dyn io::Write>,
) -> Result<()> {
    for line in reader.lines() {
        let line = line?;
        match (from, to) {
            (super::cli::ConvFormat::Labelme, super::cli::ConvFormat::LateralPoints) => {
                let from_data = LabelMeDataLine::try_from(line.as_str())?;
                let to_data: LateralPointsIRLine =
                    TryConvertContentFilename::try_convert_from(from_data)?;
                serde_json::to_writer(&mut writer, &to_data)?;
            }
            (super::cli::ConvFormat::LateralPoints, super::cli::ConvFormat::Labelme) => {
                let from_data = LateralPointsIRLine::try_from(line.as_str())?;
                let to_data: LabelMeDataLine =
                    TryConvertContentFilename::<LateralPointsIRLine>::try_convert_from(from_data)?;
                serde_json::to_writer(&mut writer, &to_data)?;
            }
            (from, to) => {
                anyhow::ensure!(from == to, "Bug in conversion logic");
                bail!("No conversion needed")
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
) -> Result<()> {
    match (from, to) {
        (ConvFormat::Labelme, ConvFormat::LateralPoints) => {
            let from_data: LabelMeData = serde_json::from_reader(reader)?;
            let to_data: LateralPointsIR = from_data.try_into()?;
            serde_json::to_writer(&mut writer, &to_data)?
        }
        (ConvFormat::LateralPoints, ConvFormat::Labelme) => {
            let from_data: LateralPointsIR = serde_json::from_reader(reader)?;
            let to_data: LabelMeData = from_data.try_into()?;
            serde_json::to_writer(&mut writer, &to_data)?
        }
        (from, to) => {
            anyhow::ensure!(from == to, "Bug in conversion logic");
            bail!("No conversion needed")
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
        process_ndjson(args.from, args.to, reader, writer)
    } else {
        process_json(args.from, args.to, reader, writer)
    }
}
