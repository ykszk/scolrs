use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;

pub trait Ndjson {
    fn is_ndjson(&self) -> bool;
}

impl Ndjson for PathBuf {
    /// Check if the file extension is `ndjson` or `jsonl`.
    fn is_ndjson(&self) -> bool {
        self.extension()
            .is_some_and(|e| e == "ndjson" || e == "jsonl")
    }
}

pub trait CreateReaderWriter {
    /// Create a reader for the given path or stdin if the path is `-`.
    fn create_reader(&self) -> Result<Box<dyn BufRead>, anyhow::Error>;
    /// Create a writer for the given path or stdout if the path is `-`.
    fn create_writer(&self) -> Result<Box<dyn Write>, anyhow::Error>;
}

fn _create_reader(path: &Path) -> Result<Box<dyn BufRead>, anyhow::Error> {
    let reader: Box<dyn BufRead> = if path.as_os_str() == "-" {
        Box::new(BufReader::new(std::io::stdin()))
    } else {
        Box::new(BufReader::new(
            File::open(path).with_context(|| format!("Opening file {:?}", path))?,
        ))
    };
    Ok(reader)
}

fn _create_writer(path: &Path) -> Result<Box<dyn Write>, anyhow::Error> {
    let writer: Box<dyn Write> = if path.as_os_str() == "-" {
        Box::new(BufWriter::new(std::io::stdout()))
    } else {
        Box::new(BufWriter::new(
            File::create(path).with_context(|| format!("Creating file {:?}", path))?,
        ))
    };
    Ok(writer)
}

impl CreateReaderWriter for Path {
    fn create_reader(&self) -> Result<Box<dyn BufRead>, anyhow::Error> {
        _create_reader(self)
    }

    fn create_writer(&self) -> Result<Box<dyn Write>, anyhow::Error> {
        _create_writer(self)
    }
}

impl CreateReaderWriter for Option<PathBuf> {
    fn create_reader(&self) -> Result<Box<dyn BufRead>, anyhow::Error> {
        let reader: Box<dyn BufRead> = if let Some(path) = self {
            _create_reader(path)?
        } else {
            Box::new(BufReader::new(std::io::stdin()))
        };
        Ok(reader)
    }

    fn create_writer(&self) -> Result<Box<dyn Write>, anyhow::Error> {
        let writer: Box<dyn Write> = if let Some(path) = self {
            _create_writer(path)?
        } else {
            Box::new(BufWriter::new(std::io::stdout()))
        };
        Ok(writer)
    }
}

pub fn read_heatmaps(heatmap_path: &Path) -> Result<ndarray::Array3<f32>, anyhow::Error> {
    let metaimage = metaimage::MetaImage::read(heatmap_path)
        .with_context(|| format!("Reading heatmap from {:?}", heatmap_path))?;
    let heatmaps_dyn = metaimage.data.into_f32_array().with_context(|| {
        format!(
            "Converting heatmap data to f32 array for image {:?}",
            heatmap_path
        )
    })?;
    let heatmaps = heatmaps_dyn
        .into_dimensionality::<ndarray::Ix3>()
        .with_context(|| {
            format!(
                "Converting heatmap data to 3D array for image {:?}",
                heatmap_path
            )
        })?;
    Ok(heatmaps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_ndjson() {
        let path = PathBuf::from("file.ndjson");
        assert!(path.is_ndjson());
        let path = PathBuf::from("file.jsonl");
        assert!(path.is_ndjson());
        let path = PathBuf::from("file.json");
        assert!(!path.is_ndjson());
        let path = PathBuf::from("file");
        assert!(!path.is_ndjson());
    }
}
