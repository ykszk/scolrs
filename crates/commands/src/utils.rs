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
