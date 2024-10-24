use std::{
    fs::File,
    io::{BufWriter, Write},
};

use crate::neck::cli::CatalogArgs;
use anyhow::{Context, Result};

const STYLE: &str = r#"
<style>
body {
	display: flex;
	flex-direction: row;
	flex-wrap: wrap;
}
    </style>
"#;

pub fn cmd(args: CatalogArgs) -> Result<()> {
    let glob_pattern = args.input.join("*.svg");
    let mut writer = BufWriter::new(File::create(&args.output)?);
    writer.write_all("<html>\n".as_bytes())?;
    if let Some(title) = args.title {
        writer.write_all(format!("<head><title>{}</title></head>\n", title).as_bytes())?;
    }
    writer.write_all(STYLE.as_bytes())?;
    writer.write_all("<body>\n".as_bytes())?;
    let mut paths: Vec<_> =
        glob::glob(glob_pattern.to_str().unwrap())?.collect::<Result<_, _>>()?;
    paths.sort();
    for path in paths {
        let filename = path.file_stem().unwrap().to_string_lossy();

        let svg = std::fs::read_to_string(&path).with_context(|| format!("Reading {:?}", path))?;
        writer.write_all(format!("<div id=\"{}\">\n", filename).as_bytes())?;
        writer.write_all(format!("<h2>{}</h2>\n", filename).as_bytes())?;
        writer.write_all(svg.as_bytes())?;
        writer.write_all("</div>\n".as_bytes())?;
    }
    writer.write_all("</body></html>\n".as_bytes())?;
    Ok(())
}
