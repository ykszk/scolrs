use std::fs::File;
use std::io::{BufWriter, Write};

use crate::neck::cli::HtmlArgs;
use anyhow::{Context, Result};
use scolrs::draw::wrap_in_html;

pub fn cmd(args: HtmlArgs) -> Result<()> {
    let svg = std::fs::read_to_string(args.input.as_path())
        .with_context(|| format!("Failed to read file: {:?}", args.input))?;
    let title = args.title.map_or_else(
        || {
            args.input
                .file_stem()
                .map_or(String::from("neckrs html"), |stem| {
                    stem.to_string_lossy().into_owned()
                })
        },
        |s| s,
    );
    let html = wrap_in_html(svg, args.selector, title)?;
    let mut writer = BufWriter::new(File::create(
        args.output
            .unwrap_or_else(|| args.input.with_extension("html")),
    )?);
    writer.write_all(html.as_bytes())?;

    Ok(())
}
