use ndarray::Axis;
use scolrs::lenke::{
    BendReasonAngles, CurveType, IsStructural, LumbarModifier, MajorCurve, MinorReason,
    RegionalCurveType, SagittalModifier, StructuralReason, Study, KYOPHOSIS_CURVE_MT,
    KYOPHOSIS_CURVE_PT, KYOPHOSIS_CURVE_TLL,
};
use scolrs::Scalable;
use scolrs::{
    draw::MeasureComponent, CoronalMeasure, CoronalPoints, CoronalPointsAndCurve, Curve, Spine,
    VertebralIndex,
};

use anyhow::{Context, Result};
use labelme_rs::LabelMeData;
use pretty_assertions::assert_eq;
use std::path::{Path, PathBuf};
use std::sync::Once;

static INIT: Once = Once::new();
fn setup() {
    INIT.call_once(|| {
        env_logger::init();
    });
}

fn load_spine(filename: &Path) -> Result<Spine> {
    let s = std::fs::read_to_string(filename).with_context(|| format!("Opening {:?}", filename))?;
    let data: LabelMeData = s.try_into()?;
    Ok(Spine::try_from(&data)?)
}

fn load_lateral_points(filename: &Path) -> Result<CoronalPoints> {
    let s = std::fs::read_to_string(filename).with_context(|| format!("Opening {:?}", filename))?;
    let data: LabelMeData = s.try_into()?;
    Ok(CoronalPoints::try_from(&data)?)
}

fn test_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests")
}

fn data_directory() -> PathBuf {
    test_directory().join("data")
}

fn test_curve(
    json_filename: &str,
    major_curve: MajorCurve,
    sup: VertebralIndex,
    inf: VertebralIndex,
) -> Result<()> {
    let json_filename = data_directory().join(json_filename);
    let scol = load_lateral_points(&json_filename)?;

    // test verticalize spine
    let mut spine = scol.spine.clone();
    spine.verticalize();

    // check if top and bottom centroids are vertically aligned
    let centroids = spine.c_c7tl.clone();
    let top = centroids.index_axis(Axis(0), 0);
    let bottom = centroids.index_axis(Axis(0), centroids.len_of(Axis(0)) - 1);

    assert_ne!((top[0] - bottom[0]).abs(), 0.001);

    let curve_desc = scol.identify_curves();
    let (curve_set, apex_set, major_curve_result) =
        (curve_desc.curves, curve_desc.apices, curve_desc.major_curve);
    assert_eq!(major_curve_result.unwrap(), major_curve);
    let (mt_curve, _angle) = curve_set.mt.unwrap();
    assert_eq!(mt_curve.sup, sup as usize);
    assert_eq!(mt_curve.inf, inf as usize);

    assert!(apex_set.pt.is_some());
    assert!(apex_set.mt.is_some());
    assert!(apex_set.tll.is_some());

    Ok(())
}

#[test]
fn test_curves_case1() -> Result<()> {
    test_curve(
        "case1/frontal.json",
        MajorCurve::MT,
        VertebralIndex::T5,
        VertebralIndex::T12,
    )
}

/// Ignore temporarily during development
#[ignore]
#[test]
fn test_curves_case2() -> Result<()> {
    let json_filename = data_directory().join("case2/frontal.json");
    let scol = load_lateral_points(&json_filename)?;

    let curve_desc = scol.identify_curves();
    let (curve_set, apex_set, major_curve) =
        (curve_desc.curves, curve_desc.apices, curve_desc.major_curve);
    // no strict testing of curve positions because case 2 is hard to determine curve with some certainty.
    assert!(curve_set.pt.is_none());
    assert!(curve_set.mt.is_some());
    assert!(curve_set.tll.is_some());

    // largest curve is tll though.
    assert_eq!(major_curve.unwrap(), MajorCurve::TLL);
    assert!(apex_set.pt.is_none());
    assert!(apex_set.mt.is_some());
    assert!(apex_set.tll.is_some());

    Ok(())
}

#[test]
fn test_curves_case3() -> Result<()> {
    test_curve(
        "case3/frontal.json",
        MajorCurve::MT,
        VertebralIndex::T7,
        VertebralIndex::T12,
    )
}

fn load_study(
    coronal_filename: &str,
    sagittal_filename: Option<&str>,
    left_filename: Option<&str>,
    right_filename: Option<&str>,
) -> Result<Study> {
    let coronal_filename = data_directory().join(coronal_filename);
    let coronal = load_lateral_points(&coronal_filename)?;

    let sagittal = sagittal_filename.map(|filename| {
        let filename = data_directory().join(filename);
        load_spine(&filename).unwrap()
    });

    let left = left_filename.map(|filename| {
        let filename = data_directory().join(filename);
        load_spine(&filename).unwrap()
    });

    let right = right_filename.map(|filename| {
        let filename = data_directory().join(filename);
        load_spine(&filename).unwrap()
    });

    Ok(Study::new(coronal, left, right, sagittal))
}

/// Fascilitator for tests
trait Wild<T> {
    /// Short hand for `as_ref().unwrap().angle(curve).unwrap()`
    fn angle_wild(&self, curve: T) -> f64;
}

impl Wild<&Option<(Curve, f64)>> for Option<Spine> {
    fn angle_wild(&self, curve: &Option<(Curve, f64)>) -> f64 {
        self.as_ref().unwrap().angle(&curve.ref_unwrap().0).unwrap()
    }
}

impl Wild<&Curve> for Option<Spine> {
    fn angle_wild(&self, curve: &Curve) -> f64 {
        self.ref_unwrap().angle(curve).unwrap()
    }
}

/// Fascilitator for tests
trait RefUnwrap<T> {
    fn ref_unwrap(&self) -> &T;
}

impl<T> RefUnwrap<T> for Option<T> {
    fn ref_unwrap(&self) -> &T {
        self.as_ref().unwrap()
    }
}

/// Ignore temporarily during development
#[ignore]
#[test]
fn test_lenke_case1() -> Result<()> {
    setup();
    let study = load_study(
        "case1/frontal.json",
        Some("case1/lateral.json"),
        Some("case1/left_lateral_bend.json"),
        Some("case1/right_lateral_bend.json"),
    )?;
    let curve_desc = study.coronal.identify_curves();
    let (curve_set, apex_set, major_curve) =
        (curve_desc.curves, curve_desc.apices, curve_desc.major_curve);

    let chart = study.chart(&curve_set, major_curve.unwrap());

    assert_eq!(
        chart.mt.ref_unwrap(),
        &RegionalCurveType::Structural(StructuralReason::Major())
    );
    let mut reason = MinorReason::with_coronal((IsStructural::T, curve_set.pt.ref_unwrap().1));
    reason.bend.left = Some((
        IsStructural::F,
        BendReasonAngles::new(
            curve_set.pt.ref_unwrap().1,
            study.left_bend.angle_wild(&curve_set.pt).abs(),
        ),
    ));
    reason.bend.right = Some((
        IsStructural::T,
        BendReasonAngles::new(
            curve_set.pt.ref_unwrap().1,
            study.right_bend.angle_wild(&curve_set.pt).abs(),
        ),
    ));

    reason.sagittal = Some((
        IsStructural::F,
        (
            KYOPHOSIS_CURVE_PT.clone(),
            study.sagittal.angle_wild(&KYOPHOSIS_CURVE_PT),
        ),
    ));
    assert_eq!(
        chart.pt.ref_unwrap(),
        &RegionalCurveType::NonStructural(reason)
    );

    let mut reason = MinorReason::with_coronal((IsStructural::T, curve_set.tll.ref_unwrap().1));

    reason.bend.left = Some((
        IsStructural::T,
        BendReasonAngles::new(
            curve_set.tll.ref_unwrap().1,
            study.left_bend.angle_wild(&curve_set.tll).abs(),
        ),
    ));
    reason.bend.right = Some((
        IsStructural::T,
        BendReasonAngles::new(
            curve_set.tll.ref_unwrap().1,
            study.right_bend.angle_wild(&curve_set.tll).abs(),
        ),
    ));

    reason.sagittal = Some((
        IsStructural::F,
        (
            KYOPHOSIS_CURVE_TLL.clone(),
            study.sagittal.angle_wild(&KYOPHOSIS_CURVE_TLL),
        ),
    ));

    assert_eq!(
        chart.tll.ref_unwrap(),
        &RegionalCurveType::Structural(StructuralReason::Minor(reason))
    );

    let curve_type = chart.classify().unwrap();
    assert_eq!(curve_type, CurveType::Type3);
    println!("{:?}: {:?}", curve_type, chart);

    assert_eq!(
        study.coronal.lumbar_modifier(apex_set.tll.unwrap()),
        LumbarModifier::AorB
    );
    assert_eq!(
        SagittalModifier::from(study.sagittal.angle_wild(&scolrs::lenke::T5T12_CURVE)),
        SagittalModifier::Normokyphosis
    );
    Ok(())
}

/// Ignore temporarily during development
#[ignore]
#[test]
fn test_lenke_case2() -> Result<()> {
    setup();
    let study = load_study("case2/frontal.json", Some("case2/lateral.json"), None, None)?;
    let curve_desc = study.coronal.identify_curves();
    let (curve_set, apex_set, major_curve) =
        (curve_desc.curves, curve_desc.apices, curve_desc.major_curve);

    let chart = study.chart(&curve_set, major_curve.unwrap());

    let mut reason = MinorReason::with_coronal((IsStructural::F, curve_set.mt.ref_unwrap().1));

    reason.sagittal = Some((
        IsStructural::F,
        (
            KYOPHOSIS_CURVE_MT,
            study.sagittal.angle_wild(&KYOPHOSIS_CURVE_MT),
        ),
    ));

    assert_eq!(
        chart.mt.ref_unwrap(),
        &RegionalCurveType::NonStructural(reason)
    );

    assert_eq!(
        chart.tll.ref_unwrap(),
        &RegionalCurveType::Structural(StructuralReason::Major())
    );

    let curve_type = chart.classify().unwrap();
    assert_eq!(curve_type, CurveType::Type5);
    println!("{:?}: {:?}", curve_type, chart);

    assert_eq!(
        study.coronal.lumbar_modifier(apex_set.tll.unwrap()),
        LumbarModifier::C
    );
    assert_eq!(
        SagittalModifier::from(study.sagittal.angle_wild(&scolrs::lenke::T5T12_CURVE)),
        SagittalModifier::Hypokyphosis
    );
    Ok(())
}

#[test]
fn test_lenke_case3() -> Result<()> {
    setup();
    let study = load_study("case3/frontal.json", Some("case3/lateral.json"), None, None)?;
    let curve_desc = study.coronal.identify_curves();
    let (curve_set, apex_set, major_curve) =
        (curve_desc.curves, curve_desc.apices, curve_desc.major_curve);

    let chart = study.chart(&curve_set, major_curve.unwrap());

    assert_eq!(
        chart.mt.ref_unwrap(),
        &RegionalCurveType::Structural(StructuralReason::Major())
    );
    let mut reason = MinorReason::with_coronal((IsStructural::T, curve_set.pt.ref_unwrap().1));

    reason.sagittal = Some((
        IsStructural::T,
        (
            KYOPHOSIS_CURVE_PT.clone(),
            study.sagittal.angle_wild(&KYOPHOSIS_CURVE_PT),
        ),
    ));
    assert_eq!(
        chart.pt.ref_unwrap(),
        &RegionalCurveType::Structural(StructuralReason::Minor(reason))
    );

    let mut reason = MinorReason::with_coronal((IsStructural::F, curve_set.tll.ref_unwrap().1));

    reason.sagittal = Some((
        IsStructural::F,
        (
            KYOPHOSIS_CURVE_TLL.clone(),
            study.sagittal.angle_wild(&KYOPHOSIS_CURVE_TLL),
        ),
    ));

    assert_eq!(
        chart.tll.ref_unwrap(),
        &RegionalCurveType::NonStructural(reason)
    );

    let curve_type = chart.classify().unwrap();
    assert_eq!(curve_type, CurveType::Type2);
    println!("{:?}: {:?}", curve_type, chart);

    assert_eq!(
        study.coronal.lumbar_modifier(apex_set.tll.unwrap()),
        LumbarModifier::AorB
    );
    assert_eq!(
        SagittalModifier::from(study.sagittal.angle_wild(&scolrs::lenke::T5T12_CURVE)),
        SagittalModifier::Normokyphosis
    );
    Ok(())
}

#[test]
fn test_lenke_case4() -> Result<()> {
    setup();
    let study = load_study("case4/frontal.json", Some("case4/lateral.json"), None, None)?;

    let data: LabelMeData = data_directory()
        .join("case4/frontal.json")
        .as_path()
        .try_into()?;
    let coronal_points = scolrs::CoronalPoints::try_from(&data)?;

    assert_eq!(coronal_points.clavicle.0.len_of(Axis(0)), 2);
    assert_eq!(coronal_points.shoulder.0.len_of(Axis(0)), 2);
    assert_eq!(coronal_points.femoral_head.0.len_of(Axis(0)), 2);

    let data: LabelMeData = data_directory()
        .join("case4/lateral.json")
        .as_path()
        .try_into()?;
    let sagittal_points = scolrs::SagittalPoints::try_from(&data)?;
    assert_eq!(sagittal_points.femoral_head.0.len_of(Axis(0)), 2);

    let curve_desc = study.coronal.identify_curves();
    let (curve_set, apex_set, major_curve) =
        (curve_desc.curves, curve_desc.apices, curve_desc.major_curve);

    let chart = study.chart(&curve_set, major_curve.unwrap());

    assert_eq!(
        chart.mt.ref_unwrap(),
        &RegionalCurveType::Structural(StructuralReason::Major())
    );

    let mut reason = MinorReason::with_coronal((IsStructural::F, curve_set.pt.ref_unwrap().1));

    reason.sagittal = Some((
        IsStructural::F,
        (
            KYOPHOSIS_CURVE_PT.clone(),
            study.sagittal.angle_wild(&KYOPHOSIS_CURVE_PT),
        ),
    ));
    assert_eq!(
        chart.pt.ref_unwrap(),
        &RegionalCurveType::NonStructural(reason)
    );

    let mut reason = MinorReason::with_coronal((IsStructural::F, curve_set.tll.ref_unwrap().1));

    reason.sagittal = Some((
        IsStructural::F,
        (
            KYOPHOSIS_CURVE_TLL.clone(),
            study.sagittal.angle_wild(&KYOPHOSIS_CURVE_TLL),
        ),
    ));

    assert_eq!(
        chart.tll.ref_unwrap(),
        &RegionalCurveType::NonStructural(reason)
    );

    let curve_type = chart.classify().unwrap();
    assert_eq!(curve_type, CurveType::Type1);
    println!("{:?}: {:?}", curve_type, chart);

    assert_eq!(
        study.coronal.lumbar_modifier(apex_set.tll.unwrap()),
        LumbarModifier::AorB
    );
    assert_eq!(
        SagittalModifier::from(study.sagittal.angle_wild(&scolrs::lenke::T5T12_CURVE)),
        SagittalModifier::Hypokyphosis
    );
    Ok(())
}

#[derive(Debug, PartialEq)]
enum Sign {
    Pos,
    Neg,
    Zero,
}

impl Sign {
    fn new(value: f64) -> Self {
        if value > 0.0 {
            Sign::Pos
        } else if value < 0.0 {
            Sign::Neg
        } else {
            Sign::Zero
        }
    }
}

#[test]
fn test_measure_case4() -> Result<()> {
    let json_filename = data_directory().join("case4/frontal.json");
    let json_str = std::fs::read_to_string(&json_filename)?;
    let lm = LabelMeData::try_from(json_str.as_str())?;
    let coronal = CoronalPointsAndCurve::try_from(&lm)?;
    let coronal = coronal.into_scaled()?;

    use CoronalMeasure::*;
    use Sign::*;

    // let measures = CoronalMeasure::all();
    let measure_and_pos = [
        (Avt, Neg),
        (T1TiltAngle, Neg),
        (CoronalBalance, Neg),
        (ClavicleAngle, Neg),
        (ShoulderHeight, Neg),
        (PelvicObliquity, Pos),
        (SacralObliquity, Neg),
        (LegLengthDiscrepancy, Pos),
        (LSTV, Pos),
    ];
    // for measure in measures {
    for (measure, sign) in measure_and_pos.into_iter() {
        let m = Box::<dyn MeasureComponent<ValueType = f64>>::from((&measure, &coronal));
        let value = m.measure()?;
        assert_eq!(
            Sign::new(value),
            sign,
            "Measure {:?} sign mismatch with value {}",
            measure,
            value
        );
    }
    Ok(())
}
