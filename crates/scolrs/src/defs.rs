use std::fmt::Display;

use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};

pub const CORNER_LABELS: [&str; 4] = ["TL", "TR", "BL", "BR"];
pub const VERTEBRAL_LABELS: [&str; 18] = [
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2", "L3",
    "L4", "L5", "L6",
];

/// Indices for vertebrae
#[repr(u8)]
#[derive(Deserialize_repr, Serialize_repr, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VertebralIndex {
    T1 = 0,
    T2 = 1,
    T3 = 2,
    T4 = 3,
    T5 = 4,
    T6 = 5,
    T7 = 6,
    T8 = 7,
    T9 = 8,
    T10 = 9,
    T11 = 10,
    T12 = 11,
    L1 = 12,
    L2 = 13,
    L3 = 14,
    L4 = 15,
    L5 = 16,
    L6 = 17,
}

impl Display for VertebralIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<u8> for VertebralIndex {
    fn from(value: u8) -> Self {
        unsafe { ::std::mem::transmute(value) }
    }
}

/// Indices for vertebrae and discs between vertebrae
#[repr(u8)]
#[derive(Deserialize_repr, Serialize_repr, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VertebraDiscIndex {
    T1 = 0,
    DiscT1T2 = 1,
    T2 = 2,
    DiscT2T3 = 3,
    T3 = 4,
    DiscT3T4 = 5,
    T4 = 6,
    DiscT4T5 = 7,
    T5 = 8,
    DiscT5T6 = 9,
    T6 = 10,
    DiscT6T7 = 11,
    T7 = 12,
    DiscT7T8 = 13,
    T8 = 14,
    DiscT8T9 = 15,
    T9 = 16,
    DiscT9T10 = 17,
    T10 = 18,
    DiscT10T11 = 19,
    T11 = 20,
    DiscT11T12 = 21,
    T12 = 22,
    DiscT12L1 = 23,
    L1 = 24,
    DiscL1L2 = 25,
    L2 = 26,
    DiscL2L3 = 27,
    L3 = 28,
    DiscL3L4 = 29,
    L4 = 30,
    DiscL4L5 = 31,
    L5 = 32,
    DiscL5L6 = 33,
    L6 = 34,
}

impl From<VertebralIndex> for VertebraDiscIndex {
    fn from(index: VertebralIndex) -> Self {
        let v_i = index as u8;
        unsafe { ::std::mem::transmute(2 * v_i) }
    }
}

impl From<u8> for VertebraDiscIndex {
    fn from(value: u8) -> Self {
        unsafe { ::std::mem::transmute(value) }
    }
}

fn default_radius() -> f64 {
    1.5
}
fn default_line_width() -> f64 {
    1.0
}
fn default_font_size() -> String {
    "24px".into()
}
fn default_unit_font_size() -> String {
    "12px".into()
}
fn default_text_stroke() -> String {
    "black".into()
}
fn default_text_stroke_width() -> f64 {
    1.0
}
fn default_text_fill() -> String {
    "white".into()
}

fn default_len_unit() -> String {
    "px".into()
}

/// Drawing parameters
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DrawParam {
    /// Point radius
    #[serde(default = "default_radius")]
    pub radius: f64,
    /// Line width
    #[serde(default = "default_line_width")]
    pub line_width: f64,

    /// `stroke` for texts
    #[serde(default = "default_font_size")]
    pub font_size: String,
    /// `stroke` for small texts (e.g. unit)
    #[serde(default = "default_unit_font_size")]
    pub unit_font_size: String,
    /// `stroke` for texts
    #[serde(default = "default_text_stroke")]
    pub text_stroke: String,
    /// `stroke-width` for texts
    #[serde(default = "default_text_stroke_width")]
    pub text_stroke_width: f64,
    /// `fill` for texts
    #[serde(default = "default_text_fill")]
    pub text_fill: String,
    /// unit
    #[serde(default = "default_len_unit")]
    pub len_unit: String,
}

impl Default for DrawParam {
    fn default() -> Self {
        Self {
            radius: default_radius(),
            line_width: default_line_width(),
            font_size: default_font_size(),
            unit_font_size: default_unit_font_size(),
            text_stroke: default_text_stroke(),
            text_stroke_width: default_text_stroke_width(),
            text_fill: default_text_fill(),
            len_unit: default_len_unit(),
        }
    }
}

impl DrawParam {
    pub fn text_style(&self) -> String {
        format!(
            "text {{font-size: {}; font-family:sans-serif; stroke: {}; stroke-width: {}; fill: {}; text-anchor: middle; dominant-baseline: central}}",
            self.font_size,
            self.text_stroke,
            self.text_stroke_width,
            self.text_fill
        ) + &format!(
            "\ntspan.unit {{font-size: {}; dominant-baseline: hanging}}",
            self.unit_font_size,
        )
    }
    pub fn line_style(&self) -> String {
        format!(
            "line, polyline, polygon, path {{stroke-width: {}; fill: none}}",
            self.line_width
        )
    }
    pub fn point_style(&self) -> String {
        format!("circle {{stroke-width: {}}}", self.line_width)
    }
    pub fn style(&self) -> String {
        format!(
            "{}\n{}\n{}",
            self.text_style(),
            self.line_style(),
            self.point_style()
        )
    }

    fn scale_font_size(font_size: &str, scale: f64) -> std::result::Result<String, String> {
        let re = regex::Regex::new(r"^([\d.]+)(\D+)").unwrap();
        if let Some(caps) = re.captures(font_size.trim()) {
            let value = caps
                .get(1)
                .unwrap()
                .as_str()
                .parse::<f64>()
                .map_err(|e| e.to_string())?;
            let unit = caps.get(2).map(|v| v.as_str()).unwrap_or_default();
            Ok(format!("{}{}", value * scale, unit))
        } else {
            Err(format!("Invalid font size format: {}", font_size))
        }
    }

    pub fn scale(&mut self, scale: f64) -> std::result::Result<(), String> {
        if scale == 0.0 {
            return Ok(());
        }

        if scale < 0.0 {
            return Err("Negative scale".into());
        }

        self.radius *= scale;
        self.line_width *= scale;
        self.text_stroke_width *= scale;
        self.font_size = Self::scale_font_size(&self.font_size, scale)?;
        self.unit_font_size = Self::scale_font_size(&self.unit_font_size, scale)?;
        Ok(())
    }
}
