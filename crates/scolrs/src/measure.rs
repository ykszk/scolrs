use indexmap::IndexMap;

use crate::{
    draw::{
        ConfidenceComponent, ConfidenceDisplay, MapKey, MeasureComponent, MeasureError,
        ReductionMethod,
    },
    HasImageMetadata, PointConfidence, Scalable, ScaledType,
};
use serde::{Deserialize, Serialize};
use std::result;

#[derive(Serialize, Deserialize, Default)]
pub struct MeasureResult<K, T>
where
    K: MapKey,
{
    pub measurements: IndexMap<K, result::Result<T, MeasureError>>,
    pub confidences: Option<IndexMap<K, result::Result<T, MeasureError>>>,
    pub unit_of_length: String,
}

// Trait to convert MeasureResult<String, Vec<T>> to MeasureResult<String, T> by flattening
pub trait FlattenResult {
    type FlatType;
    fn into_flat(self) -> Self::FlatType;
}

impl FlattenResult for MeasureResult<String, f64> {
    type FlatType = MeasureResult<String, f64>;
    /// No-op for already flat results
    fn into_flat(self) -> Self::FlatType {
        self
    }
}

impl FlattenResult for MeasureResult<String, Vec<f64>> {
    type FlatType = MeasureResult<String, f64>;
    fn into_flat(self) -> Self::FlatType {
        let mut measurements: IndexMap<String, result::Result<f64, MeasureError>> = IndexMap::new();
        for (k, v) in self.measurements {
            if let Ok(v) = v {
                if v.len() == 1 {
                    measurements.insert(k, Ok(v[0]));
                } else {
                    for (i, value) in v.into_iter().enumerate() {
                        let key = format!("{}_{}", k, i + 1);
                        measurements.insert(key, Ok(value));
                    }
                }
            }
        }
        let mut confidences: Option<IndexMap<String, result::Result<f64, MeasureError>>> = None;
        if let Some(confs) = self.confidences {
            let mut conf_map: IndexMap<String, result::Result<f64, MeasureError>> = IndexMap::new();
            for (k, v) in confs {
                if let Ok(v) = v {
                    if v.len() == 1 {
                        conf_map.insert(k, Ok(v[0]));
                    } else {
                        for (i, value) in v.into_iter().enumerate() {
                            let key = format!("{}_{}", k, i + 1);
                            conf_map.insert(key, Ok(value));
                        }
                    }
                }
            }
            confidences = Some(conf_map);
        }
        MeasureResult {
            measurements,
            confidences,
            unit_of_length: self.unit_of_length,
        }
    }
}

pub struct TransposedEntry<T> {
    pub measurement: result::Result<T, MeasureError>,
    pub confidence: Option<result::Result<T, MeasureError>>,
}

pub struct TransposedResult<K, T>
where
    K: MapKey,
{
    pub entries: IndexMap<K, TransposedEntry<T>>,
    pub unit_of_length: String,
}

impl<K, T> MeasureResult<K, T>
where
    K: MapKey,
    T: std::fmt::Display + Clone,
{
    /// Transpose the result from (measure[key], confidence[key]) to (key: {measurement, confidence})
    pub fn into_transposed(self) -> TransposedResult<K, T> {
        let mut transposed: IndexMap<K, TransposedEntry<T>> = Default::default();
        for (key, measurement) in self.measurements {
            let confidence = self
                .confidences
                .as_ref()
                .and_then(|confs: &IndexMap<K, Result<T, MeasureError>>| confs.get(&key).cloned());
            transposed.insert(
                key,
                TransposedEntry {
                    measurement,
                    confidence,
                },
            );
        }
        TransposedResult {
            entries: transposed,
            unit_of_length: self.unit_of_length,
        }
    }
}

impl<V, T> MeasureResult<V, T>
where
    V: MapKey + ToString,
{
    /// Change key type to String
    pub fn into_string_map(self) -> MeasureResult<String, T> {
        let measurements: IndexMap<String, result::Result<T, MeasureError>> = self
            .measurements
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let confidences: Option<IndexMap<String, result::Result<T, MeasureError>>> = self
            .confidences
            .map(|confs| confs.into_iter().map(|(k, v)| (k.to_string(), v)).collect());
        MeasureResult {
            measurements,
            confidences,
            unit_of_length: self.unit_of_length,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct MeasureLine<K, T>
where
    K: MapKey,
{
    pub filename: String,
    pub content: MeasureResult<K, T>,
}

/// Calculate measures and their confidences for given data and measures.
pub fn measure_x<T, U, V: ConfidenceDisplay>(
    data: ScaledType<T>,
    measures: Vec<U>,
    reduction_method: ReductionMethod,
) -> MeasureResult<U, V>
where
    T: HasImageMetadata + Scalable + PointConfidence,
    U: MapKey + std::fmt::Debug,
    for<'a, 'b> (&'a U, &'b ScaledType<T>): Into<Box<dyn MeasureComponent<ValueType = V> + 'b>>
        + Into<Box<dyn ConfidenceComponent<ValueType = V> + 'b>>,
{
    let confidences = if data.0.get_confidence().is_some() {
        let confs = confidence_x(&data, &measures, reduction_method);
        Some(confs)
    } else {
        None
    };
    let mut measurements: IndexMap<U, _> = Default::default();
    for measure in measures.into_iter() {
        let spinal_measure: Box<dyn MeasureComponent<ValueType = V>> = (&measure, &data).into();
        measurements.insert(measure, spinal_measure.measure());
    }
    let result = MeasureResult {
        measurements,
        confidences,
        unit_of_length: data.0.image_metadata().unit.clone(),
    };
    result
}

type ConfResult<U, V> = IndexMap<U, result::Result<V, MeasureError>>;

fn confidence_x<T, U, V: ConfidenceDisplay>(
    data: &ScaledType<T>,
    measures: &[U],
    reduction_method: ReductionMethod,
) -> ConfResult<U, V>
where
    T: Scalable,
    U: MapKey + std::fmt::Debug,
    for<'a, 'b> (&'a U, &'b ScaledType<T>): Into<Box<dyn ConfidenceComponent<ValueType = V> + 'b>>,
{
    let mut measurements: IndexMap<U, result::Result<V, MeasureError>> = Default::default();
    for measure in measures {
        log::debug!("Calculating confidence for measure {:?}", measure);
        let spinal_measure: Box<dyn ConfidenceComponent<ValueType = V>> = (measure, data).into();
        let conf = spinal_measure.confidence(reduction_method);
        if let Some(conf) = conf {
            measurements.insert(measure.clone(), conf);
        }
    }
    measurements
}

pub type NeckMeasurements = MeasureResult<String, Vec<f64>>;
