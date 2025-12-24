use indexmap::IndexMap;

use crate::{
    draw::{ConfidenceComponent, MeasureComponent, MeasureError, ReductionMethod},
    head_neck::NeckMeasureComponent,
    HasImageMetadata, PointConfidence, Scalable, ScaledType,
};
use serde::{Deserialize, Serialize};
use std::result;

type Measurement = result::Result<f64, MeasureError>;
type MeasurementVec = result::Result<Vec<f64>, MeasureError>;

#[derive(Serialize, Deserialize, Default)]
pub struct MeasureResult<V, T>
where
    V: std::hash::Hash + Eq + std::cmp::Ord,
{
    pub measurements: IndexMap<V, result::Result<T, MeasureError>>,
    pub confidences: Option<IndexMap<V, result::Result<f64, MeasureError>>>,
    pub unit_of_length: String,
}

pub struct TransposedEntry<T> {
    pub measurement: result::Result<T, MeasureError>,
    pub confidence: Option<result::Result<f64, MeasureError>>,
}

pub struct TransposedResult<V, T>
where
    V: std::hash::Hash + Eq + std::cmp::Ord,
{
    pub entries: IndexMap<V, TransposedEntry<T>>,
    pub unit_of_length: String,
}

impl<V, T> MeasureResult<V, T>
where
    V: std::hash::Hash + Eq + std::cmp::Ord,
{
    pub fn into_transposed(self) -> TransposedResult<V, T> {
        let mut transposed: IndexMap<V, TransposedEntry<T>> = Default::default();
        for (key, measurement) in self.measurements {
            let confidence = self.confidences.as_ref().and_then(
                |confs: &IndexMap<V, Result<f64, MeasureError>>| confs.get(&key).cloned(),
            );
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
    V: std::hash::Hash + Eq + std::cmp::Ord + ToString,
{
    /// Change key type to String
    pub fn into_string_map(self) -> MeasureResult<String, T> {
        let measurements: IndexMap<String, result::Result<T, MeasureError>> = self
            .measurements
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let confidences: Option<IndexMap<String, result::Result<f64, MeasureError>>> = self
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
pub struct MeasureLine<V, T>
where
    V: std::hash::Hash + Eq + std::cmp::Ord,
{
    pub filename: String,
    pub content: MeasureResult<V, T>,
}

/// Calculate measures and their confidences for given data and measures.
pub fn measure_x<T, U>(
    data: ScaledType<T>,
    measures: Vec<U>,
    reduction_method: ReductionMethod,
) -> MeasureResult<U, f64>
where
    T: HasImageMetadata + Scalable + PointConfidence,
    U: std::hash::Hash + Eq + std::cmp::Ord + Clone + std::fmt::Debug,
    for<'a, 'b> (&'a U, &'b ScaledType<T>):
        Into<Box<dyn MeasureComponent + 'b>> + Into<Box<dyn ConfidenceComponent + 'b>>,
{
    let confidences = if data.0.get_confidence().is_some() {
        let confs = confidence_x(&data, &measures, reduction_method);
        Some(confs)
    } else {
        None
    };
    let mut measurements: IndexMap<U, Measurement> = Default::default();
    for measure in measures.into_iter() {
        let spinal_measure: Box<dyn MeasureComponent> = (&measure, &data).into();
        measurements.insert(measure, spinal_measure.measure());
    }
    let result = MeasureResult {
        measurements,
        confidences,
        unit_of_length: data.0.image_metadata().unit.clone(),
    };
    result
}

type ConfResult<U> = IndexMap<U, result::Result<f64, MeasureError>>;

fn confidence_x<T, U>(
    data: &ScaledType<T>,
    measures: &[U],
    reduction_method: ReductionMethod,
) -> ConfResult<U>
where
    T: Scalable,
    U: std::hash::Hash + Eq + std::cmp::Ord + Clone + std::fmt::Debug,
    for<'a, 'b> (&'a U, &'b ScaledType<T>): Into<Box<dyn ConfidenceComponent + 'b>>,
{
    let mut measurements: IndexMap<U, result::Result<f64, MeasureError>> = Default::default();
    for measure in measures {
        log::debug!("Calculating confidence for measure {:?}", measure);
        let spinal_measure: Box<dyn ConfidenceComponent> = (measure, data).into();
        let conf = spinal_measure.confidence(reduction_method);
        if let Some(conf) = conf {
            measurements.insert(measure.clone(), conf);
        }
    }
    measurements
}

pub fn measure_all(
    measures: Vec<Box<dyn NeckMeasureComponent + '_>>,
) -> IndexMap<String, MeasurementVec> {
    let measures: Vec<_> = measures
        .into_iter()
        .map(|m| {
            let result = m.measure();
            (m.id().to_string(), result)
        })
        .collect();
    IndexMap::from_iter(measures)
}

pub type NeckMeasurements = MeasureResult<String, Vec<f64>>;
