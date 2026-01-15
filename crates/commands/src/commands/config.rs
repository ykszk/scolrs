use crate::cli::{self, ConfigArgs, ConfigName};
use anyhow::Result;

enum Format {
    Toml,
    Json,
}

impl Format {
    fn from_cli(f: cli::ConfigFileFormat) -> Self {
        match f {
            cli::ConfigFileFormat::Toml => Format::Toml,
            cli::ConfigFileFormat::Json => Format::Json,
        }
    }
}

fn print_in_format<T: serde::ser::Serialize>(config: &T, format: Format) -> Result<()> {
    let string = match format {
        Format::Json => serde_json::to_string_pretty(config)?,
        Format::Toml => toml::to_string_pretty(config)?,
    };
    println!("{}", string);
    Ok(())
}

fn print_config<T: Default + serde::de::DeserializeOwned + serde::ser::Serialize>(
    format: Format,
) -> Result<()> {
    let config_builder = config::Config::builder()
        .add_source(config::Config::try_from(&T::default())?)
        .build()?;
    let config: T = config_builder.try_deserialize()?;
    print_in_format(&config, format)?;
    Ok(())
}

pub fn cmd(args: ConfigArgs) -> Result<()> {
    let format = Format::from_cli(args.format);
    match args.name {
        ConfigName::Asm => print_config::<scolrs::asm::AsmConfig>(format)?,
        ConfigName::Icp => print_config::<scolrs::asm::icp::IcpConfig>(format)?,
        ConfigName::Draw => print_config::<scolrs::draw::DrawParam>(format)?,
        ConfigName::Linecolor => {
            let colors = scolrs::draw::ColorPalettes::default_line_colors();
            print_in_format(&colors, format)?
        }
        ConfigName::Labelcolor => {
            let colors = scolrs::draw::ColorPalettes::default_label_colors();
            print_in_format(&colors, format)?
        }
    }
    Ok(())
}
