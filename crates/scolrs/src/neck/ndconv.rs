use std::fs::File;
use std::io::{self, BufRead, BufReader};

use crate::neck::cli::NdconvArgs;
use anyhow::{bail, Result};
use labelme_rs::LabelMeDataLine;
use scolrs::head_neck::{LateralPointsIRLine, TryConvertContentFilename};

pub fn cmd(args: NdconvArgs) -> Result<()> {
    let reader: Box<dyn BufRead> = if let Some(input) = args.input {
        Box::new(BufReader::new(File::open(input)?))
    } else {
        Box::new(BufReader::new(io::stdin()))
    };
    let mut writer: Box<dyn io::Write> = if let Some(output) = args.output {
        Box::new(File::create(output)?)
    } else {
        Box::new(io::stdout())
    };

    for line in reader.lines() {
        let line = line?;
        match (args.from, args.to) {
            (super::cli::NdConvFormat::Labelme, super::cli::NdConvFormat::LateralPoints) => {
                let from_data = LabelMeDataLine::try_from(line.as_str())?;
                let to_data: LateralPointsIRLine =
                    TryConvertContentFilename::try_convert_from(from_data)?;
                serde_json::to_writer(&mut writer, &to_data)?;
            }
            (super::cli::NdConvFormat::LateralPoints, super::cli::NdConvFormat::Labelme) => {
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
