use std::{
    fs::File,
    io::{BufWriter, Read, Write},
};

use crate::cli::CatalogArgs;
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

/// A trait for getting the path and content of a file
///
/// Abstracts over reading from a file or stdin in tar format
trait PathAndContent: Send + Sync {
    fn get(self: Box<Self>) -> (std::path::PathBuf, String);
}

impl PathAndContent for std::path::PathBuf {
    fn get(self: Box<Self>) -> (std::path::PathBuf, String) {
        let content = std::fs::read_to_string(self.as_ref()).unwrap();
        (*self, content)
    }
}

impl PathAndContent for (std::path::PathBuf, String) {
    fn get(self: Box<Self>) -> (std::path::PathBuf, String) {
        *self
    }
}

pub fn cmd(args: CatalogArgs) -> Result<()> {
    if let Some(jobs) = args.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()
            .unwrap();
    }

    let mut writer = BufWriter::new(
        File::create(&args.output).with_context(|| format!("Writing to {:?}", args.output))?,
    );
    writer.ws("<html>\n")?;
    writer.ws("<head><meta charset=\"utf-8\">")?;
    if let Some(title) = args.title {
        writer.ws(&format!("<title>{}</title>\n", title))?;
    }
    let style = include_str!("../../../scolrs/src/templates/catalog.css");
    writer.ws("<style>\n")?;
    writer.ws(style)?;
    writer.ws("</style>\n")?;
    writer.ws("</head>\n")?;
    writer.ws("<body>\n")?;

    writer.ws("<script>\n")?;
    let javascript = include_str!("../../../scolrs/src/templates/catalog.js");
    writer.ws(javascript)?;
    let javascript = include_str!("../../../scolrs/src/templates/capture.js");
    writer.ws(javascript)?;
    writer.ws("</script>\n")?;

    let mut templates = tera::Tera::default();
    templates.autoescape_on(vec![]);
    templates.add_raw_templates(vec![
        (
            "image_container.jinja",
            include_str!("../../../scolrs/src/templates/catalog_image_container.jinja"),
        ),
        (
            "checkbox.jinja",
            include_str!("../../../scolrs/src/templates/checkbox.jinja"),
        ),
        (
            "catalog_popup.jinja",
            include_str!("../../../scolrs/src/templates/catalog_popup.jinja"),
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

    let mut paths: Vec<Box<dyn PathAndContent>> = Vec::new();
    for input in args.input {
        if input.is_dir() {
            for ext in ["*.svg", "*.html"].iter() {
                let glob_pattern = input.join(ext);
                let mut svgs: Vec<_> =
                    glob::glob(glob_pattern.to_str().unwrap())?.collect::<Result<_, _>>()?;
                svgs.sort();
                paths.append(
                    svgs.into_iter()
                        .map(|p| Box::new(p) as _)
                        .collect::<Vec<_>>()
                        .as_mut(),
                );
            }
        } else if input.is_file() {
            paths.push(Box::new(input));
        } else if input.as_os_str() == "-" {
            let mut archive = tar::Archive::new(std::io::stdin());
            let mut path_and_content: Vec<Box<_>> = Vec::new();
            for entry in archive.entries()? {
                let mut entry = entry?;
                let mut content = String::new();
                entry.read_to_string(&mut content)?;
                path_and_content.push(Box::new((entry.path()?.to_path_buf(), content)));
            }
            path_and_content.sort_by(|a, b| a.0.cmp(&b.0));
            paths.extend(path_and_content.into_iter().map(|p| Box::new(*p) as _));
        } else {
            bail!("Invalid input file: {:?}", input)
        }
    }

    if paths.is_empty() {
        bail!("No files found");
    }

    let divs: Result<Vec<_>> = paths
        .into_par_iter()
        .map(|path| -> Result<String> {
            let (path, content) = path.get();
            let svg = if path.extension().unwrap_or_default() == "html" {
                let document = Html::parse_document(&content);

                document
                    .select(&Selector::parse("svg").unwrap())
                    .next()
                    .context("No `svg` element found")?
                    .html()
            } else {
                content
            };
            let filename = path.file_stem().unwrap().to_string_lossy();
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
        &include_str!("../../../scolrs/src/templates/save_module.html"),
    );
    let div_popup = templates.render("catalog_popup.jinja", &context)?;
    writer.ws(&div_popup)?;
    writer.ws("</body></html>\n")?;
    Ok(())
}
