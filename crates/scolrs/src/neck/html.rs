use std::fs::File;
use std::io::{BufWriter, Write};

use crate::neck::cli::HtmlArgs;
use anyhow::{Context, Result};
use scraper::{Html, Selector};

pub fn cmd(args: HtmlArgs) -> Result<()> {
    let svg = std::fs::read_to_string(args.input.as_path())
        .with_context(|| format!("Failed to read file: {:?}", args.input))?;
    let document = Html::parse_document(&svg);

    let mut templates = tera::Tera::default();
    templates.autoescape_on(vec![]);
    templates.add_raw_templates(vec![
        ("html.jinja", include_str!("../templates/html.jinja")),
        (
            "checkbox.jinja",
            include_str!("../templates/checkbox.jinja"),
        ),
    ])?;
    let javascript = include_str!("../templates/capture.js");
    let mut elements: Vec<_> = Vec::new();
    for selector in args.selector {
        let selector = Selector::parse(&selector).unwrap_or_else(|_| {
            panic!("Failed to parse selector: {}", &selector);
        });
        elements.extend(document.select(&selector));
    }

    let mut checkboxes = vec![];
    for element in elements {
        let mut context = tera::Context::new();
        let id = element.value().attr("id").context("`id` not defined")?;
        context.insert("id", &id);
        context.insert("label", &element.value().attr("data-label").unwrap_or(id));
        context.insert(
            "description",
            &element.value().attr("data-description").unwrap_or(""),
        );
        let visibility = element.value().attr("visibility").unwrap_or("visible");
        let checked = if visibility == "visible" {
            "checked"
        } else {
            ""
        };
        context.insert("checked", checked);
        checkboxes.push(templates.render("checkbox.jinja", &context)?);
    }

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

    let mut context = tera::Context::new();
    context.insert("svg", &svg);
    context.insert("checkboxes", &checkboxes.join("\n"));
    context.insert("title", &title);
    context.insert("javascript", &javascript);
    context.insert(
        "save_module",
        &include_str!("../templates/save_module.html"),
    );

    let html = templates.render("html.jinja", &context)?;
    let mut writer = BufWriter::new(File::create(
        args.output
            .unwrap_or_else(|| args.input.with_extension("html")),
    )?);
    writer.write_all(html.as_bytes())?;

    Ok(())
}
