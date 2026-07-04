use ndarray::Axis;
use scolrs::draw::{AsMeasure, ColorPalettes};
use scolrs::head_neck::{LateralPoints, NeckLateralDraw, NeckLateralMeasure};
use scolrs::lenke::{
    BendReasonAngles, CurveType, IsStructural, LumbarModifier, MajorCurve, MinorReason,
    RegionalCurveType, SagittalModifier, StructuralReason, Study, KYOPHOSIS_CURVE_MT,
    KYOPHOSIS_CURVE_PT, KYOPHOSIS_CURVE_TLL,
};
use scolrs::{
    draw::MeasureComponent, CoronalMeasure, CoronalPoints, CoronalPointsAndCurve, Curve,
    MeasureAndDraw, Spine, VertebralIndex,
};
use scolrs::{CoronalDraw, SagittalDraw, SagittalPoints, Scalable};

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
        self.as_ref().unwrap().angle(&curve.ref_unwrap().0)
    }
}

impl Wild<&Curve> for Option<Spine> {
    fn angle_wild(&self, curve: &Curve) -> f64 {
        self.ref_unwrap().angle(curve)
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
fn test_coronal_measure_cases() -> Result<()> {
    fn inner(json_path: PathBuf, measure_and_pos: &[(CoronalMeasure, Sign)]) -> Result<()> {
        println!("Testing measures in {:?}", json_path);
        let json_str = std::fs::read_to_string(&json_path)?;
        let lm = LabelMeData::try_from(json_str.as_str())?;
        let coronal = CoronalPointsAndCurve::try_from(&lm)?;
        let coronal = coronal.into_scaled()?;

        for (measure, sign) in measure_and_pos.iter() {
            let m = Box::<dyn MeasureComponent<ValueType = f64>>::from((measure, &coronal));
            let value = m.measure()?;
            assert_eq!(
                Sign::new(value),
                *sign,
                "Measure {:?} sign mismatch with value {}",
                measure,
                value
            );
        }
        Ok(())
    }

    use CoronalMeasure::*;
    use Sign::*;

    // case2
    let json_filename = data_directory().join("case2/frontal.json");
    let measure_and_pos = [(ShoulderHeight, Pos), (LegLengthDiscrepancy, Neg)];
    inner(json_filename, &measure_and_pos)?;

    // case3
    let json_filename = data_directory().join("case3/frontal.json");
    let measure_and_pos = [(ShoulderHeight, Neg), (LegLengthDiscrepancy, Neg)];
    inner(json_filename, &measure_and_pos)?;

    // case4
    let json_filename = data_directory().join("case4/frontal.json");
    let measure_and_pos = [
        (CobbPT, Pos),
        (CobbMT, Neg),
        (CobbTLL, Pos),
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
    inner(json_filename, &measure_and_pos)?;
    Ok(())
}

#[test]
fn test_neck_measure_cases() -> Result<()> {
    fn inner(
        json_path: PathBuf,
        measure_and_signes: &[(NeckLateralMeasure, &[Sign])],
    ) -> Result<()> {
        println!("Testing measures in {:?}", json_path);
        let json_str = std::fs::read_to_string(&json_path)?;
        let lm = LabelMeData::try_from(json_str.as_str())?;
        let lateral = LateralPoints::try_from(&lm)?;
        let lateral = lateral.into_scaled()?;

        for (measure, signes) in measure_and_signes.iter() {
            let m = Box::<dyn MeasureComponent<ValueType = Vec<f64>>>::from((measure, &lateral));
            let values = m.measure()?;
            for (i, (value, sign)) in values.iter().zip(signes.iter()).enumerate() {
                assert_eq!(
                    Sign::new(*value),
                    *sign,
                    "Measure {:?} sign mismatch with value {} at index {}",
                    measure,
                    value,
                    i
                );
            }
        }
        Ok(())
    }

    use NeckLateralMeasure::*;
    use Sign::*;

    // neck_case1
    let json_filename = data_directory().join("neck_case1/lateral.json");
    let measure_and_pos = [(OccipitocervicalInclination, &[Pos][..])];
    inner(json_filename, &measure_and_pos)?;

    // neck_case2
    // let json_filename = data_directory().join("neck_case2/lateral.json");
    // inner(json_filename, &measure_and_pos)?;
    Ok(())
}

#[test]
fn test_measure_coronal_draw_equality() -> Result<()> {
    fn inner(json_path: PathBuf) -> Result<()> {
        println!("Testing equalities in {:?}", json_path);
        let json_str = std::fs::read_to_string(&json_path)?;
        let lm = LabelMeData::try_from(json_str.as_str())?;
        let coronal = CoronalPointsAndCurve::try_from(&lm)?;
        let coronal = coronal.into_scaled()?;

        let draws = CoronalDraw::all();
        let measure_and_draws: Vec<_> = draws
            .into_iter()
            .filter_map(|draw| {
                let measure = draw.as_measure();
                measure.map(|m| (m, draw))
            })
            .collect();
        let palettes = ColorPalettes::default();
        let mut label_colors = palettes.label_colors;
        let mut line_colors = palettes.line_colors;
        let draw_param = scolrs::draw::DrawParam::default();
        let svg_size = (500, 500);
        let mut painter = scolrs::draw::Painter::new(draw_param, svg_size);
        for (measure, draw) in measure_and_draws.into_iter() {
            println!("  Testing measure-draw {:?}", measure);
            let m = Box::<dyn MeasureComponent<ValueType = f64>>::from((&measure, &coronal));
            let measure_result = m.measure();
            if let Err(e) = &measure_result {
                match e {
                    scolrs::draw::MeasureError::UnableToMeasure(_) => {
                        panic!("Unable to measure: {:?}", measure)
                    }
                    _ => continue,
                }
            }
            let measured_value = measure_result.unwrap();
            let d = Box::<dyn scolrs::draw::DrawComponent>::from((&draw, &coronal));
            let group = d.draw(&mut painter, &mut label_colors, &mut line_colors)?;

            let data_attr = group
                .get_attributes()
                .get("data-value")
                .context("data-value attribute not found in drawn group")?;
            let drawn_value: f64 = data_attr
                .parse()
                .context("failed to parse data-value attribute")?;
            assert!(
                (measured_value - drawn_value).abs() < 1e-6,
                "Measure {:?} value mismatch in {}: measured {}, drawn {}",
                measure,
                json_path.display(),
                measured_value,
                drawn_value
            );
        }
        Ok(())
    }
    for case in ["case1", "case2", "case3", "case4"] {
        let json_path = data_directory().join(format!("{}/{}.json", case, "frontal"));
        inner(json_path)?;
    }
    Ok(())
}

#[test]
fn test_sagittal_measure_draw_equality() -> Result<()> {
    fn inner(json_path: PathBuf) -> Result<()> {
        println!("Testing equalities in {:?}", json_path);
        let json_str = std::fs::read_to_string(&json_path)?;
        let lm = LabelMeData::try_from(json_str.as_str())?;
        let lateral = SagittalPoints::try_from(&lm)?;
        let lateral = lateral.into_scaled()?;

        let draws = SagittalDraw::all();
        let measure_and_draws: Vec<_> = draws
            .into_iter()
            .filter_map(|draw| {
                let measure = draw.as_measure();
                measure.map(|m| (m, draw))
            })
            .collect();
        println!("Found {} measure-draw pairs", measure_and_draws.len());
        let palettes = ColorPalettes::default();
        let mut label_colors = palettes.label_colors;
        let mut line_colors = palettes.line_colors;
        let draw_param = scolrs::draw::DrawParam::default();
        let svg_size = (500, 500);
        let mut painter = scolrs::draw::Painter::new(draw_param, svg_size);
        for (measure, draw) in measure_and_draws.into_iter() {
            println!("  Testing measure-draw {:?}", measure);
            let m = Box::<dyn MeasureComponent<ValueType = f64>>::from((&measure, &lateral));
            let measure_result = m.measure();
            if let Err(e) = &measure_result {
                match e {
                    scolrs::draw::MeasureError::UnableToMeasure(_) => {
                        panic!("Unable to measure: {:?}", measure)
                    }
                    _ => continue,
                }
            }
            let measured_value = measure_result.unwrap();
            let d = Box::<dyn scolrs::draw::DrawComponent>::from((&draw, &lateral));
            let group = d.draw(&mut painter, &mut label_colors, &mut line_colors)?;

            let data_attr = group
                .get_attributes()
                .get("data-value")
                .context("data-value attribute not found in drawn group")?;
            let drawn_value: f64 = data_attr
                .parse()
                .context("failed to parse data-value attribute")?;
            assert!(
                (measured_value - drawn_value).abs() < 1e-6,
                "Measure {:?} value mismatch in {}: measured {}, drawn {}",
                measure,
                json_path.display(),
                measured_value,
                drawn_value
            );
        }
        Ok(())
    }
    for case in ["case1", "case2", "case3", "case4"] {
        let json_path = data_directory().join(format!("{}/{}.json", case, "lateral"));
        inner(json_path)?;
    }
    Ok(())
}

#[test]
fn test_neck_measure_draw_equality() -> Result<()> {
    fn inner(json_path: PathBuf) -> Result<()> {
        println!("Testing equalities in {:?}", json_path);
        let json_str = std::fs::read_to_string(&json_path)?;
        let lm = LabelMeData::try_from(json_str.as_str())?;
        let lateral = LateralPoints::try_from(&lm)?;
        let lateral = lateral.into_scaled()?;

        let draws = NeckLateralDraw::all();
        let measure_and_draws: Vec<_> = draws
            .into_iter()
            .filter_map(|draw| {
                let measure = draw.as_measure();
                measure.map(|m| (m, draw))
            })
            .collect();
        println!("Found {} measure-draw pairs", measure_and_draws.len());
        let palettes = ColorPalettes::default();
        let mut label_colors = palettes.label_colors;
        let mut line_colors = palettes.line_colors;
        let draw_param = scolrs::draw::DrawParam::default();
        let svg_size = (500, 500);
        let mut painter = scolrs::draw::Painter::new(draw_param, svg_size);
        for (measure, draw) in measure_and_draws.into_iter() {
            println!("  Testing measure-draw {:?}", measure);
            let m = Box::<dyn MeasureComponent<ValueType = Vec<f64>>>::from((&measure, &lateral));
            let measure_result = m.measure();
            if let Err(e) = &measure_result {
                match e {
                    scolrs::draw::MeasureError::UnableToMeasure(_) => {
                        panic!("Unable to measure: {:?}", measure)
                    }
                    _ => continue,
                }
            }
            let measured_value = measure_result.unwrap();
            let d = Box::<dyn scolrs::draw::DrawComponent>::from((&draw, &lateral));
            let group = d.draw(&mut painter, &mut label_colors, &mut line_colors)?;

            let data_attr = group
                .get_attributes()
                .get("data-value")
                .context("data-value attribute not found in drawn group")?;
            let drawn_values: Vec<f64> = data_attr
                .split(' ')
                .map(|s| {
                    s.parse::<f64>()
                        .context("failed to parse data-value attribute")
                })
                .collect::<Result<_, _>>()?;

            // let draw_values =
            //     if let Some(data_value_attr) = group.get_attributes().get("data-value") {
            //         let drawn_value: f64 = data_value_attr
            //             .parse()
            //             .context("failed to parse data-value attribute")?;
            //         vec![drawn_value]
            //     } else if let Some(data_values_attr) = group.get_attributes().get("data-values") {
            //         let drawn_values: Vec<f64> = data_values_attr
            //             .split(' ')
            //             .map(|s| {
            //                 s.parse::<f64>()
            //                     .context("failed to parse data-values attribute")
            //             })
            //             .collect::<Result<_, _>>()?;
            //         drawn_values
            //     } else {
            //         panic!("Neither data-value nor data-values attribute found in drawn group");
            //     };
            assert!(
                measured_value
                    .iter()
                    .zip(drawn_values.iter())
                    .all(|(m, d)| (m - d).abs() < 1e-6),
                "Measure {:?} value mismatch in {}: measured {:?}, drawn {:?}",
                measure,
                json_path.display(),
                measured_value,
                drawn_values
            );
        }
        Ok(())
    }
    for case in ["neck_case1", "neck_case2"] {
        for dir in ["lateral", "extension_lateral", "flexion_lateral"] {
            let json_path = data_directory().join(format!("{}/{}.json", case, dir));
            if !json_path.exists() {
                continue;
            }
            inner(json_path)?;
        }
    }
    Ok(())
}
