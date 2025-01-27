use std::path::PathBuf;

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
