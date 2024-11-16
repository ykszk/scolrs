use std::{
    fs::File,
    io::{BufWriter, Write},
};

use crate::neck::cli::CatalogArgs;
use anyhow::{bail, Context, Result};
use indexmap::IndexMap;
use rayon::prelude::*;
use scraper::{Html, Selector};

/// Shorthand for writing a string to a file
trait WriteString {
    fn ws(&mut self, s: &str) -> std::io::Result<()>;
}

impl WriteString for BufWriter<File> {
    fn ws(&mut self, s: &str) -> std::io::Result<()> {
        self.write_all(s.as_bytes())
    }
}

pub fn cmd(args: CatalogArgs) -> Result<()> {
    if let Some(jobs) = args.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()
            .unwrap();
    }

    let glob_pattern = args.input.join("*.svg");
    let mut writer = BufWriter::new(File::create(&args.output)?);
    writer.ws("<html>\n")?;
    writer.ws("<head>")?;
    if let Some(title) = args.title {
        writer.ws(&format!("<title>{}</title>\n", title))?;
    }
    let style = include_str!("../templates/catalog.css");
    writer.ws("<style>\n")?;
    writer.ws(style)?;
    writer.ws("</style>\n")?;
    writer.ws("</head>\n")?;
    writer.ws("<body>\n")?;

    writer.ws("<script>\n")?;
    let javascript = include_str!("../templates/catalog.js");
    writer.ws(javascript)?;
    let javascript = include_str!("../templates/capture.js");
    writer.ws(javascript)?;
    writer.ws("</script>\n")?;

    let mut templates = tera::Tera::default();
    templates.autoescape_on(vec![]);
    templates.add_raw_templates(vec![
        (
            "image_container.jinja",
            include_str!("../templates/catalog_image_container.jinja"),
        ),
        (
            "checkbox.jinja",
            include_str!("../templates/checkbox.jinja"),
        ),
        (
            "catalog_popup.jinja",
            include_str!("../templates/catalog_popup.jinja"),
        ),
    ])?;

    let checkboxes: IndexMap<String, String> = IndexMap::new();
    let checkboxes = std::sync::Arc::from(std::sync::Mutex::new(checkboxes));
    let selectors: Result<Vec<_>, _> = args
        .selector
        .iter()
        .map(|s| Selector::parse(s.as_str()))
        .collect();
    let selectors = match selectors {
        Ok(selectors) => selectors,
        Err(e) => {
            bail!("Error parsing selector: {}", e);
        }
    };

    let mut paths: Vec<_> =
        glob::glob(glob_pattern.to_str().unwrap())?.collect::<Result<_, _>>()?;
    paths.sort();
    let divs: Result<Vec<_>> = paths
        .into_par_iter()
        .map(|path| -> Result<String> {
            let filename = path.file_stem().unwrap().to_string_lossy();
            let svg =
                std::fs::read_to_string(&path).with_context(|| format!("Reading {:?}", path))?;
            let document = Html::parse_document(&svg);
            let mut elements: Vec<_> = Vec::new();
            for selector in selectors.iter() {
                elements.extend(document.select(selector));
            }
            for element in elements {
                let id = element.value().attr("id").context("`id` not defined")?;
                if checkboxes.lock().unwrap().contains_key(id) {
                    continue;
                }
                let label = element.value().attr("data-label").unwrap_or(id);
                let description = element.value().attr("data-description").unwrap_or("");
                let checked = "checked";
                let mut context = tera::Context::new();
                context.insert("id", &id);
                context.insert("label", &label);
                context.insert("description", &description);
                context.insert("checked", checked);

                checkboxes.lock().unwrap().insert(
                    id.to_string(),
                    templates.render("checkbox.jinja", &context)?,
                );
            }

            let mut context = tera::Context::new();
            context.insert("img", &svg);
            context.insert("id", &filename);
            Ok(templates.render("image_container.jinja", &context)?)
        })
        .collect();
    let divs = divs?;

    for div in divs {
        writer.ws(&div)?;
    }

    let checkboxes = std::sync::Arc::into_inner(checkboxes)
        .unwrap()
        .into_inner()
        .unwrap();

    let mut context = tera::Context::new();
    context.insert(
        "checkboxes",
        &checkboxes.into_iter().map(|(_, v)| v).collect::<String>(),
    );
    context.insert(
        "save_module",
        &include_str!("../templates/save_module.html"),
    );
    let div_popup = templates.render("catalog_popup.jinja", &context)?;
    writer.ws(&div_popup)?;
    writer.ws("</body></html>\n")?;
    Ok(())
}
