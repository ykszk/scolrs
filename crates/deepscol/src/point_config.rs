use std::ops::Range;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointSetConfig {
    pub labels: Vec<String>,
    /// Maximum number of points for each label
    pub max_counts: Vec<usize>,
    pub ch_range_rgb: (Range<usize>, Range<usize>, Range<usize>),
}

impl PointSetConfig {
    pub fn spine() -> Self {
        let labels = [
            "TL",
            "TR",
            "BL",
            "BR",
            "Shoulder",
            "Clavicle",
            "Pelvis",
            "Iliac",
            "FemoralHead",
            "C7-TL",
            "C7-TR",
            "S-TL",
            "S-TR",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let max_counts = vec![
            20, // TL
            20, // TR
            19, // BL
            19, // BR
            2,  // Shoulder
            2,  // Clavicle
            2,  // Pelvis
            2,  // Iliac
            2,  // FemoralHead
            1,  // C7-TL
            1,  // C7-TR
            1,  // S-TL
            1,  // S-TR
        ];
        let ch_range_rgb = (0..2, 2..4, 4..9); // exclude last 4 points for heatmap
        PointSetConfig {
            labels,
            max_counts,
            ch_range_rgb,
        }
    }
    pub fn neck_lateral() -> Self {
        let labels: Vec<_> = [
            "TL",
            "TR",
            "BL",
            "BR",
            "Brow",
            "Sella",
            "ExternalAuditoryCanal",
            "Orbit",
            "Occipital",
            "Chin",
            "Lamina",
            "Manubrium",
            "PosteriorHardPalate",
            "AnteriorC1Arch",
            "AnteriorDens",
            "PosteriorDens",
            "Lamina1",
            "C3_BL",
            "C3_BR",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let max_counts = vec![
            7, // TL
            7, // TR
            8, // BL
            8, // BR
            1, // Brow
            1, // Sella
            1, // ExternalAuditoryCanal
            1, // Orbit
            1, // Occipital
            1, // Chin
            8, // Lamina
            1, // Manubrium
            1, // PosteriorHardPalate
            1, // AnteriorC1Arch
            1, // AnteriorDens
            1, // PosteriorDens
            1, // Lamina1
            1, // C3_BL
            1, // C3_BR
        ];
        let ch_range_rgb = (0..2, 2..4, 4..(labels.len() - 2)); // exclude last 2 points for heatmap
        PointSetConfig {
            labels,
            max_counts,
            ch_range_rgb,
        }
    }
}
