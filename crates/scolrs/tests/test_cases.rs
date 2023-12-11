use scolrs::{
    BendReasonAngles, CurveType, IsStructural, LumbarModifier, MajorCurve, MinorReason,
    RegionalCurveType, Spine, StructuralReason, Study, VertebraDiscIndex, VertebralIndex,
    KYOPHOSIS_CURVE_MT, KYOPHOSIS_CURVE_PT, KYOPHOSIS_CURVE_TLL,
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
    let data: LabelMeData = s.as_str().try_into()?;
    Ok(Spine::try_from(&data)?)
}

fn test_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests")
}

fn data_directory() -> PathBuf {
    test_directory().join("data")
}

#[test]
fn test_curves_case1() -> Result<()> {
    let json_filename = data_directory().join("case1/frontal.json");
    let scol = load_spine(&json_filename)?;

    let (curve_set, apex_set, major_curve) = scol.identify_curves();
    assert_eq!(major_curve.unwrap(), MajorCurve::MT);
    let (mt_curve, _angle) = curve_set.mt.unwrap();
    assert_eq!(mt_curve.sup, VertebralIndex::T5 as usize);
    assert_eq!(mt_curve.inf, VertebralIndex::T12 as usize);
    let (pt_curve, _angle) = curve_set.pt.unwrap();
    assert_eq!(pt_curve.inf, mt_curve.sup);
    assert_eq!(pt_curve.sup, VertebralIndex::T1 as usize);
    let (tll_curve, _angle) = curve_set.tll.unwrap();
    assert_eq!(tll_curve.sup, mt_curve.inf);
    assert_eq!(tll_curve.inf, VertebralIndex::L5 as usize);

    assert!(apex_set.pt.is_some());
    assert!(apex_set.tll.is_some());
    assert_eq!(apex_set.mt.unwrap(), VertebraDiscIndex::DiscT7T8);

    Ok(())
}

#[test]
fn test_curves_case2() -> Result<()> {
    let json_filename = data_directory().join("case2/frontal.json");
    let scol = load_spine(&json_filename)?;

    let (curve_set, apex_set, major_curve) = scol.identify_curves();
    // no strict testing of curve positions because case 2 is hard to determine curve with some certainty.
    assert!(curve_set.pt.is_some());
    assert!(curve_set.mt.is_some());
    assert!(curve_set.tll.is_some());

    // largest curve is tll though.
    assert_eq!(major_curve.unwrap(), MajorCurve::TLL);
    assert!(apex_set.pt.is_some());
    assert!(apex_set.mt.is_some());
    assert!(apex_set.tll.is_some());

    Ok(())
}

#[test]
fn test_curves_case3() -> Result<()> {
    let json_filename = data_directory().join("case3/frontal.json");
    let scol = load_spine(&json_filename)?;

    let (curve_set, apex_set, major_curve) = scol.identify_curves();
    assert_eq!(major_curve.unwrap(), MajorCurve::MT); // largest curve is PT but major curve is MT
    let (mt_curve, _angle) = curve_set.mt.unwrap();
    assert_eq!(mt_curve.sup, VertebralIndex::T7 as usize);
    assert_eq!(mt_curve.inf, VertebralIndex::T12 as usize);
    let (pt_curve, _angle) = curve_set.pt.unwrap();
    assert_eq!(pt_curve.inf, mt_curve.sup);
    assert_eq!(pt_curve.sup, VertebralIndex::T2 as usize);
    let (tll_curve, _angle) = curve_set.tll.unwrap();
    assert_eq!(tll_curve.sup, mt_curve.inf);
    assert_eq!(tll_curve.inf, VertebralIndex::L4 as usize);

    assert!(apex_set.pt.is_some());
    assert!(apex_set.mt.is_some());
    assert!(apex_set.tll.is_some());

    Ok(())
}

#[test]
fn test_lenke_case1() -> Result<()> {
    setup();
    let json_filename = data_directory().join("case1/frontal.json");
    let frontal_scol = load_spine(&json_filename)?;

    let (curve_set, apex_set, major_curve) = frontal_scol.identify_curves();

    let json_filename = data_directory().join("case1/left_lateral_bend.json");
    let left_scol = load_spine(&json_filename)?;
    let json_filename = data_directory().join("case1/right_lateral_bend.json");
    let right_scol = load_spine(&json_filename)?;
    let json_filename = data_directory().join("case1/lateral.json");
    let lateral_scol = load_spine(&json_filename)?;

    assert_eq!(frontal_scol.c_c7tl.len(), left_scol.c_c7tl.len());
    assert_eq!(frontal_scol.c_c7tl.len(), right_scol.c_c7tl.len());
    assert_eq!(frontal_scol.c_c7tl.len(), lateral_scol.c_c7tl.len());

    let study = Study::full(frontal_scol, left_scol, right_scol, lateral_scol);

    let chart = study.chart(&curve_set, major_curve.unwrap());

    assert_eq!(
        chart.mt.as_ref().unwrap(),
        &RegionalCurveType::Structural(StructuralReason::Major())
    );
    let mut reason = MinorReason::with_coronal((IsStructural::T, curve_set.pt.as_ref().unwrap().1));
    reason.bend.left = Some((
        IsStructural::F,
        BendReasonAngles::new(
            curve_set.pt.as_ref().unwrap().1,
            study
                .left_bend
                .as_ref()
                .unwrap()
                .angle(&curve_set.pt.as_ref().unwrap().0)
                .unwrap()
                .abs(),
        ),
    ));
    reason.bend.right = Some((
        IsStructural::T,
        BendReasonAngles::new(
            curve_set.pt.as_ref().unwrap().1,
            study
                .right_bend
                .as_ref()
                .unwrap()
                .angle(&curve_set.pt.as_ref().unwrap().0)
                .unwrap()
                .abs(),
        ),
    ));

    reason.sagittal = Some((
        IsStructural::F,
        (
            KYOPHOSIS_CURVE_PT.clone(),
            study
                .sagittal
                .as_ref()
                .unwrap()
                .angle(&KYOPHOSIS_CURVE_PT)
                .unwrap(),
        ),
    ));
    assert_eq!(
        chart.pt.as_ref().unwrap(),
        &RegionalCurveType::NonStructural(reason)
    );

    let mut reason =
        MinorReason::with_coronal((IsStructural::T, curve_set.tll.as_ref().unwrap().1));

    reason.bend.left = Some((
        IsStructural::T,
        BendReasonAngles::new(
            curve_set.tll.as_ref().unwrap().1,
            study
                .left_bend
                .unwrap()
                .angle(&curve_set.tll.as_ref().unwrap().0)
                .unwrap()
                .abs(),
        ),
    ));
    reason.bend.right = Some((
        IsStructural::T,
        BendReasonAngles::new(
            curve_set.tll.as_ref().unwrap().1,
            study
                .right_bend
                .unwrap()
                .angle(&curve_set.tll.as_ref().unwrap().0)
                .unwrap()
                .abs(),
        ),
    ));

    reason.sagittal = Some((
        IsStructural::F,
        (
            KYOPHOSIS_CURVE_TLL.clone(),
            study
                .sagittal
                .as_ref()
                .unwrap()
                .angle(&KYOPHOSIS_CURVE_TLL)
                .unwrap(),
        ),
    ));

    assert_eq!(
        chart.tll.as_ref().unwrap(),
        &RegionalCurveType::Structural(StructuralReason::Minor(reason))
    );

    let curve_type = chart.classify().unwrap();
    assert_eq!(curve_type, CurveType::Type3);
    println!("{:?}: {:?}", curve_type, chart);

    assert_eq!(
        study.coronal.lumbar_modifier(apex_set.tll.unwrap()),
        LumbarModifier::AorB
    );
    Ok(())
}

#[test]
fn test_lenke_case2() -> Result<()> {
    setup();
    let json_filename = data_directory().join("case2/frontal.json");
    let frontal_scol = load_spine(&json_filename)?;

    let (curve_set, apex_set, major_curve) = frontal_scol.identify_curves();

    let json_filename = data_directory().join("case2/lateral.json");
    let lateral_scol = load_spine(&json_filename)?;

    assert_eq!(frontal_scol.c_c7tl.len(), lateral_scol.c_c7tl.len());

    let study = Study {
        coronal: frontal_scol,
        left_bend: None,
        right_bend: None,
        sagittal: Some(lateral_scol),
    };

    let chart = study.chart(&curve_set, major_curve.unwrap());

    let mut reason = MinorReason::with_coronal((IsStructural::F, curve_set.mt.as_ref().unwrap().1));

    reason.sagittal = Some((
        IsStructural::F,
        (
            KYOPHOSIS_CURVE_MT,
            study
                .sagittal
                .as_ref()
                .unwrap()
                .angle(&KYOPHOSIS_CURVE_MT)
                .unwrap(),
        ),
    ));

    assert_eq!(
        chart.mt.as_ref().unwrap(),
        &RegionalCurveType::NonStructural(reason)
    );
    let mut reason = MinorReason::with_coronal((IsStructural::F, curve_set.pt.as_ref().unwrap().1));

    reason.sagittal = Some((
        IsStructural::F,
        (
            KYOPHOSIS_CURVE_PT,
            study
                .sagittal
                .as_ref()
                .unwrap()
                .angle(&KYOPHOSIS_CURVE_PT)
                .unwrap(),
        ),
    ));

    assert_eq!(
        chart.pt.as_ref().unwrap(),
        &RegionalCurveType::NonStructural(reason)
    );

    assert_eq!(
        chart.tll.as_ref().unwrap(),
        &RegionalCurveType::Structural(StructuralReason::Major())
    );

    let curve_type = chart.classify().unwrap();
    assert_eq!(curve_type, CurveType::Type5);
    println!("{:?}: {:?}", curve_type, chart);

    assert_eq!(
        study.coronal.lumbar_modifier(apex_set.tll.unwrap()),
        LumbarModifier::C
    );
    Ok(())
}

#[test]
fn test_lenke_case3() -> Result<()> {
    setup();
    let json_filename = data_directory().join("case3/frontal.json");
    let frontal_scol = load_spine(&json_filename)?;

    let (curve_set, apex_set, major_curve) = frontal_scol.identify_curves();

    let json_filename = data_directory().join("case3/lateral.json");
    let lateral_scol = load_spine(&json_filename)?;

    assert_eq!(frontal_scol.c_c7tl.len(), lateral_scol.c_c7tl.len());

    let study = Study {
        coronal: frontal_scol,
        left_bend: None,
        right_bend: None,
        sagittal: Some(lateral_scol),
    };

    let chart = study.chart(&curve_set, major_curve.unwrap());

    assert_eq!(
        chart.mt.as_ref().unwrap(),
        &RegionalCurveType::Structural(StructuralReason::Major())
    );
    let mut reason = MinorReason::with_coronal((IsStructural::T, curve_set.pt.as_ref().unwrap().1));

    reason.sagittal = Some((
        IsStructural::T,
        (
            KYOPHOSIS_CURVE_PT.clone(),
            study
                .sagittal
                .as_ref()
                .unwrap()
                .angle(&KYOPHOSIS_CURVE_PT)
                .unwrap(),
        ),
    ));
    assert_eq!(
        chart.pt.as_ref().unwrap(),
        &RegionalCurveType::Structural(StructuralReason::Minor(reason))
    );

    let mut reason =
        MinorReason::with_coronal((IsStructural::F, curve_set.tll.as_ref().unwrap().1));

    reason.sagittal = Some((
        IsStructural::F,
        (
            KYOPHOSIS_CURVE_TLL.clone(),
            study
                .sagittal
                .as_ref()
                .unwrap()
                .angle(&KYOPHOSIS_CURVE_TLL)
                .unwrap(),
        ),
    ));

    assert_eq!(
        chart.tll.as_ref().unwrap(),
        &RegionalCurveType::NonStructural(reason)
    );

    let curve_type = chart.classify().unwrap();
    assert_eq!(curve_type, CurveType::Type2);
    println!("{:?}: {:?}", curve_type, chart);

    assert_eq!(
        study.coronal.lumbar_modifier(apex_set.tll.unwrap()),
        LumbarModifier::AorB
    );
    Ok(())
}

#[test]
fn test_lenke_case4() -> Result<()> {
    setup();
    let json_filename = data_directory().join("case4/frontal.json");
    let frontal_scol = load_spine(&json_filename)?;

    let (curve_set, apex_set, major_curve) = frontal_scol.identify_curves();

    let json_filename = data_directory().join("case4/lateral.json");
    let lateral_scol = load_spine(&json_filename)?;

    assert_eq!(frontal_scol.c_c7tl.len(), lateral_scol.c_c7tl.len());

    let study = Study {
        coronal: frontal_scol,
        left_bend: None,
        right_bend: None,
        sagittal: Some(lateral_scol),
    };

    let chart = study.chart(&curve_set, major_curve.unwrap());

    assert_eq!(
        chart.mt.as_ref().unwrap(),
        &RegionalCurveType::Structural(StructuralReason::Major())
    );

    let mut reason = MinorReason::with_coronal((IsStructural::F, curve_set.pt.as_ref().unwrap().1));

    reason.sagittal = Some((
        IsStructural::F,
        (
            KYOPHOSIS_CURVE_PT.clone(),
            study
                .sagittal
                .as_ref()
                .unwrap()
                .angle(&KYOPHOSIS_CURVE_PT)
                .unwrap(),
        ),
    ));
    assert_eq!(
        chart.pt.as_ref().unwrap(),
        &RegionalCurveType::NonStructural(reason)
    );

    let mut reason =
        MinorReason::with_coronal((IsStructural::F, curve_set.tll.as_ref().unwrap().1));

    reason.sagittal = Some((
        IsStructural::F,
        (
            KYOPHOSIS_CURVE_TLL.clone(),
            study
                .sagittal
                .as_ref()
                .unwrap()
                .angle(&KYOPHOSIS_CURVE_TLL)
                .unwrap(),
        ),
    ));

    assert_eq!(
        chart.tll.as_ref().unwrap(),
        &RegionalCurveType::NonStructural(reason)
    );

    let curve_type = chart.classify().unwrap();
    assert_eq!(curve_type, CurveType::Type1);
    println!("{:?}: {:?}", curve_type, chart);

    assert_eq!(
        study.coronal.lumbar_modifier(apex_set.tll.unwrap()),
        LumbarModifier::AorB
    );
    Ok(())
}
