use crate::{ContentFilename, ScaledType};
use crate::{HasImageMetadata, ImplantDraw, Scalable};
use detectron2::DetectedBox;
use indexmap::IndexMap;
use labelme_rs::LabelMeData;
use ndarray::{Array, Array1, Axis, Slice};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{ImageMetadata, ScolError, Spine};

#[derive(Debug, Clone)]
pub struct Rectangle {
    /// top left corner
    pub tl: (f64, f64),
    /// bottom right corner
    pub br: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct Implant {
    pub screw: Vec<Rectangle>,
    pub hook: Vec<Rectangle>,
    pub transverse: Vec<Rectangle>,
    pub rod: Vec<Rectangle>,
}

#[derive(Debug, Clone)]
pub struct Screw {
    /// bounding box of the screw
    pub bb: Rectangle,
    /// vertebrae that the screw is attached to
    pub vertebra: usize,
    /// screws is on the left side of the vertebra
    pub left: Option<bool>,
}

impl Screw {
    pub fn with_pos(bb: Rectangle, vertebra: usize, left: bool) -> Self {
        let left = Some(left);
        Self { bb, vertebra, left }
    }

    pub fn new(bb: Rectangle, vertebra: usize) -> Self {
        let left = None;
        Self { bb, vertebra, left }
    }
}

impl From<&LabelMeData> for Implant {
    fn from(data: &LabelMeData) -> Self {
        let mut screw = Vec::new();
        let mut hook = Vec::new();
        let mut transverse = Vec::new();
        let mut rod = Vec::new();

        for shape in &data.shapes {
            if shape.shape_type == "rectangle" {
                let tl = (
                    shape.points[0].0.min(shape.points[1].0),
                    shape.points[0].1.min(shape.points[1].1),
                );
                let br = (
                    shape.points[0].0.max(shape.points[1].0),
                    shape.points[0].1.max(shape.points[1].1),
                );
                let rectangle = Rectangle { tl, br };

                match shape.label.to_lowercase().as_str() {
                    "screw" => screw.push(rectangle),
                    "hook" => hook.push(rectangle),
                    "transverse" => transverse.push(rectangle),
                    "rod" => rod.push(rectangle),
                    _ => (),
                }
            }
        }
        Self {
            screw,
            hook,
            transverse,
            rod,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImplantSpine {
    pub spine: Spine,
    pub implant: Implant,
}

trait VecStats {
    fn mean(&self) -> f64;
    fn std(&self) -> f64;
}

impl VecStats for Vec<f64> {
    fn mean(&self) -> f64 {
        self.iter().sum::<f64>() / self.len() as f64
    }

    fn std(&self) -> f64 {
        let mean = self.mean();
        (self.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.len() as f64).sqrt()
    }
}

impl TryFrom<&LabelMeData> for ImplantSpine {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let spine = Spine::try_from(data)?;
        let implant = Implant::from(data);
        Ok(Self { spine, implant })
    }
}

/// The allowed distance factor is used to determine the maximum distance between a rectangle and a vertebra
/// to be considered a valid pairing. This factor is multiplied by the mean plate length of the spine.
const ALLOWED_DIST_FACTOR: f64 = 1.0;

trait ScrewVertebraUtils {
    fn sort_by_y(&mut self);
    fn count_rect_per_vertebra(&self, n_vertebrae: usize) -> Vec<u8>;
}

impl ScrewVertebraUtils for &mut [ScrewVertebra] {
    fn sort_by_y(&mut self) {
        self.sort_by(|a, b| {
            let (_x1, y1) = a.c_rect;
            let (_x2, y2) = b.c_rect;
            y1.partial_cmp(&y2).unwrap()
        });
    }
    fn count_rect_per_vertebra(&self, n_vertebrae: usize) -> Vec<u8> {
        let mut count = vec![0; n_vertebrae];
        self.iter().for_each(|ScrewVertebra { i_vert, .. }| {
            count[*i_vert] += 1;
        });
        count
    }
}

fn _refine_pairing_two_sided(
    rectangles: &[Rectangle],
    mut pairs: Vec<ScrewVertebra>,
    vertebra_centroids: ndarray::ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 2]>>,
    is_left: bool,
) -> Vec<Screw> {
    let n_vertebrae = vertebra_centroids.len_of(Axis(0));
    let rect_count_per_vertebra = pairs.as_mut_slice().count_rect_per_vertebra(n_vertebrae);
    let too_many_rects = rect_count_per_vertebra.iter().any(|&count| count > 1);

    let optimal_pairs: Vec<Screw> = if too_many_rects {
        log::warn!("Too many rectangles per vertebra");
        // find optimal assignment
        let cost_func = |i_rect: usize, i_vert: usize| {
            let (x, y): (f64, f64) = pairs[i_rect].c_rect;
            let vc = vertebra_centroids.index_axis(Axis(0), i_vert);
            ((x - vc[0]).powi(2) + (y - vc[1]).powi(2)).sqrt()
        };

        let optimal_assignments = ordered_assignment(n_vertebrae, pairs.len(), cost_func);

        pairs
            .iter()
            .zip(optimal_assignments.assignments.iter())
            .map(|(sv, &i_vert)| Screw::with_pos(rectangles[sv.i_rect].clone(), i_vert, is_left))
            .collect()
    } else {
        pairs
            .iter()
            .map(|x| Screw::with_pos(rectangles[x.i_rect].clone(), x.i_vert, is_left))
            .collect::<Vec<_>>()
    };
    optimal_pairs
}

fn refine_two_sided_pairings(
    rectangles: &[Rectangle],
    left_pairs: Vec<ScrewVertebra>,
    right_pairs: Vec<ScrewVertebra>,
    vertebra_centroids: ndarray::ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 2]>>,
) -> Vec<Screw> {
    let left_refined = _refine_pairing_two_sided(rectangles, left_pairs, vertebra_centroids, true);
    let right_refined =
        _refine_pairing_two_sided(rectangles, right_pairs, vertebra_centroids, false);
    let mut both_pairs = left_refined;
    both_pairs.extend(right_refined);
    both_pairs
}

fn refine_one_sided_pairings(
    rectangles: &[Rectangle],
    pairs: Vec<ScrewVertebra>,
    vertebra_centroids: ndarray::ArrayBase<ndarray::ViewRepr<&f64>, ndarray::Dim<[usize; 2]>>,
    is_left: bool,
) -> Vec<Screw> {
    // choose the closest two screws to the vertebrae
    let mut chosen_pairs: Vec<Screw> = Vec::new();
    for i_vert in 0..vertebra_centroids.len_of(Axis(0)) {
        let mut closest = pairs
            .iter()
            .filter(|sv| sv.i_vert == i_vert)
            .collect::<Vec<_>>();
        closest.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        closest.truncate(2);
        chosen_pairs.extend(
            closest
                .iter()
                .map(|sv| Screw::with_pos(rectangles[sv.i_rect].clone(), i_vert, is_left)),
        );
    }
    chosen_pairs
}

impl ImplantSpine {
    fn pair_impl(&self, screw_rects: &[Rectangle]) -> Vec<Screw> {
        let (left_pairs, right_pairs) = pair_to_closest(screw_rects, &self.spine);

        let vertebra_centroids = self.spine.c_c7tl.slice_axis(Axis(0), Slice::from(1..));

        let one_sided = left_pairs.is_empty() || right_pairs.is_empty();
        let refined_pairs = if one_sided {
            let (pairs, is_left) = if left_pairs.is_empty() {
                (right_pairs, false)
            } else {
                (left_pairs, true)
            };
            refine_one_sided_pairings(screw_rects, pairs, vertebra_centroids, is_left)
        } else {
            refine_two_sided_pairings(screw_rects, left_pairs, right_pairs, vertebra_centroids)
        };

        if refined_pairs.len() != screw_rects.len() {
            log::warn!(
                "Number of screws does not match number of rectangles. {} != {}",
                refined_pairs.len(),
                screw_rects.len()
            );
        }
        refined_pairs
    }

    pub fn pair_screw(&self) -> Vec<Screw> {
        self.pair_impl(&self.implant.screw)
    }

    pub fn screw_spine(&self, image_metadata: ImageMetadata) -> ScrewSpine {
        let screws = self.pair_screw();
        ScrewSpine {
            spine: self.spine.clone(),
            screws,
            image_metadata,
        }
    }
}

#[derive(Debug, Clone, HasImageMetadata)]
pub struct ScrewSpine {
    pub spine: Spine,
    pub screws: Vec<Screw>,
    pub image_metadata: ImageMetadata,
}

impl TryFrom<&LabelMeData> for ScrewSpine {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let implant_spine = ImplantSpine::try_from(data)?;
        let screws = implant_spine.pair_screw();
        let spine = implant_spine.spine;
        let image_metadata = ImageMetadata::from(data.clone());
        Ok(Self {
            spine,
            screws,
            image_metadata,
        })
    }
}

impl Scalable for ScrewSpine {
    type Error = std::convert::Infallible;

    fn _impl_scale(&mut self) -> Result<(), Self::Error> {
        let scale_xy = ndarray::array![
            self.image_metadata.spacing_xy.0,
            self.image_metadata.spacing_xy.1
        ];
        self.spine.scale(scale_xy.view());
        self.screws.iter_mut().for_each(|screw| {
            screw.bb.tl.0 *= scale_xy[0];
            screw.bb.tl.1 *= scale_xy[1];
            screw.bb.br.0 *= scale_xy[0];
            screw.bb.br.1 *= scale_xy[1];
        });
        Ok(())
    }
}

impl ScrewSpine {
    /// The number of screws per vertebra
    pub fn count_screws(&self) -> Vec<usize> {
        let mut count = vec![0; self.spine.v_c7tl.0.len_of(Axis(0)) - 1];
        for screw in &self.screws {
            count[screw.vertebra] += 1;
        }
        count
    }

    /// Convert to LabelMeData with Screws and optionally Vertebrae
    ///
    /// `group_id` is the vertebrae index starting from 1
    pub fn to_labelme(
        &self,
        flags: IndexMap<String, bool>,
        include_vertebrae: bool,
    ) -> LabelMeData {
        let mut shapes = Vec::new();
        for screw in &self.screws {
            let points = vec![screw.bb.tl, screw.bb.br];
            let shape = labelme_rs::Shape {
                label: "Screw".to_string(),
                points,
                shape_type: "rectangle".to_string(),
                flags: Default::default(),
                group_id: Some(screw.vertebra + 1),
            };
            shapes.push(shape);
        }
        if include_vertebrae {
            let vertebrae: std::collections::HashSet<_> =
                self.screws.iter().map(|screw| screw.vertebra).collect();
            for i_vert in vertebrae.into_iter() {
                let points = self.spine.v_c7tl.0.index_axis(Axis(0), i_vert + 1);
                // from (tl, tr, bl, br) to (tl, tr, br, bl)
                let points = vec![
                    (points[[0, 0]], points[[0, 1]]),
                    (points[[1, 0]], points[[1, 1]]),
                    (points[[3, 0]], points[[3, 1]]),
                    (points[[2, 0]], points[[2, 1]]),
                ];

                let shape = labelme_rs::Shape {
                    label: "Vertebra".to_string(),
                    points,
                    shape_type: "polygon".to_string(),
                    flags: Default::default(),
                    group_id: Some(i_vert + 1),
                };
                shapes.push(shape);
            }
        }
        shapes.sort_by_key(|shape| shape.group_id.unwrap());
        LabelMeData {
            version: "4.5.6".to_string(),
            flags,
            shapes,
            imageWidth: self.image_metadata.width,
            imageHeight: self.image_metadata.height,
            imagePath: self.image_metadata.path.clone(),
            imageData: None,
        }
    }
}

use crate::draw::{
    DrawComponent, DrawError, Named, Painter, VertebralLabels, CLASS_ANNOTATION, CLASS_POLYGON,
};
use svg::node::element;

const IMPLANT_COMPONENT_CLASS: &str = "CoronalComponent";
pub trait ImplantComponent: DrawComponent {
    fn default_group(&self) -> element::Group {
        self.default_group_w_classes(&["Component", IMPLANT_COMPONENT_CLASS])
    }
}

/// Screws and vertebrae
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POLYGON])]
struct Screws<'a>(&'a ScrewSpine);
impl ImplantComponent for Screws<'_> {}
impl DrawComponent for Screws<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut crate::draw::ColorPalette,
        _line_colors: &mut crate::draw::ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let spine = &self.0.spine;

        let style = r#".screw {fill:transparent}
        .screw:hover ~ .screw-vertebra {animation: pulse 0.6s ease-in infinite;}
        .screw:hover {fill:white;fill-opacity:0.5;}
        .screw-vertebra {transform-origin: 50% 50%;transform-box: fill-box;}
        @keyframes pulse{
          25%  {transform: scale(0.8);}
          75%  {transform: scale(1.2);}
        }"#;

        let mut g = self
            .default_group()
            .set("fill", "none")
            .add(element::Style::new(style));

        let mut screws_per_vertebra: Vec<Vec<_>> = vec![Vec::new(); spine.v_c7tl.0.len_of(Axis(0))];

        for screw in &self.0.screws {
            screws_per_vertebra[screw.vertebra].push(screw);
        }

        for (i, screws) in screws_per_vertebra.iter().enumerate() {
            let line_color = crate::draw::TAB10_NEW_TAB10[i % crate::draw::TAB10_NEW_TAB10.len()];
            if screws.is_empty() {
                continue;
            }
            let mut g_screw_vertebra = element::Group::new().set("stroke", line_color);
            for screw in screws {
                let rect = Array::from_shape_vec(
                    (2, 2),
                    vec![screw.bb.tl.0, screw.bb.tl.1, screw.bb.br.0, screw.bb.br.1],
                )
                .unwrap();
                let title = if i < 12 {
                    format!("T{}", i + 1)
                } else {
                    format!("L{}", i - 11)
                };
                let bbox = painter
                    .rectangle(rect)
                    .set("class", "screw")
                    .add(painter.title(title.as_str()));

                g_screw_vertebra = g_screw_vertebra.add(bbox);
                if let Some(left) = screw.left {
                    let point_pos: Array1<f64> = if left {
                        // draw point on the left side of the sc
                        ndarray::array![screw.bb.tl.0, (screw.bb.tl.1 + screw.bb.br.1) / 2.0]
                    } else {
                        // draw point on the right side of the sc
                        ndarray::array![screw.bb.br.0, (screw.bb.tl.1 + screw.bb.br.1) / 2.0]
                    };
                    let point = painter.point(point_pos).set("fill", line_color);
                    g_screw_vertebra = g_screw_vertebra.add(point);
                }
            }
            let mut corners = spine.v_c7tl.0.index_axis(Axis(0), i + 1).to_owned(); // vertebra + 1 to skip C7

            // Change point-order from (tl, tr, bl, br) to (tl, tr, br, bl)
            corners.swap((2, 0), (3, 0)); // bl.x <-> br.x
            corners.swap((2, 1), (3, 1)); // bl.y <-> br.y

            let vertebra = painter
                .polygon(corners)
                .set("stroke-dasharray", "5,5")
                .set("class", "screw-vertebra");
            g_screw_vertebra = g_screw_vertebra.add(vertebra);

            g = g.add(g_screw_vertebra);
        }
        Ok(g)
    }
}

impl<'a, 'b> From<(&'b ImplantDraw, &'a ScaledType<ScrewSpine>)> for Box<dyn DrawComponent + 'a> {
    fn from((draw, screw_spine): (&'b ImplantDraw, &'a ScaledType<ScrewSpine>)) -> Self {
        let screw_spine = &screw_spine.0;
        match draw {
            ImplantDraw::Screws => Box::new(Screws(screw_spine)),
            ImplantDraw::VertebralLabels => Box::new(VertebralLabels(&screw_spine.spine)),
        }
    }
}

pub mod detectron2 {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct Output {
        pub instances: Instances,
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct Instances {
        /// bounding box coordinates in (x1, y1, x2, y2) format
        pub pred_boxes: Vec<(f64, f64, f64, f64)>,
        pub scores: Vec<f64>,
        pub pred_classes: Vec<usize>,
        // pub pred_masks: Vec<Mask>,
        /// keypoint coordinates in (x, y, score) format
        pub pred_keypoints: Vec<Vec<(f64, f64, f64)>>,
        pub image_size: (usize, usize),
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct DetectedBox {
        /// bounding box coordinates in (x1, y1, x2, y2) format
        pub coords: (f64, f64, f64, f64),
        pub score: f64,
    }

    impl Instances {
        pub fn boxes(&self, class: usize) -> Vec<DetectedBox> {
            self.pred_boxes
                .iter()
                .zip(self.scores.iter())
                .zip(self.pred_classes.iter())
                .filter_map(|((coords, &score), &c)| {
                    if c == class {
                        Some(DetectedBox {
                            coords: *coords,
                            score,
                        })
                    } else {
                        None
                    }
                })
                .collect()
        }
    }
}

#[derive(Debug, Clone)]
struct ScrewVertebra {
    i_rect: usize,
    c_rect: (f64, f64),
    i_vert: usize,
    dist: f64,
    /// displacement of the rectangle (screw) center from the vertebra center along the x-axis
    dx: f64,
}

/// Define a trait to use [pair_to_closest] with both [Rectangle] and [DetectedBox]
trait CalculateCentroid {
    fn centroid(&self) -> (f64, f64);
}

impl CalculateCentroid for Rectangle {
    fn centroid(&self) -> (f64, f64) {
        ((self.tl.0 + self.br.0) / 2.0, (self.tl.1 + self.br.1) / 2.0)
    }
}

impl CalculateCentroid for DetectedBox {
    fn centroid(&self) -> (f64, f64) {
        (
            (self.coords.0 + self.coords.2) / 2.0,
            (self.coords.1 + self.coords.3) / 2.0,
        )
    }
}

// Helper function to determine if the screws are all on one side
fn detect_one_sided_configuration(pairs: &[ScrewVertebra]) -> bool {
    // Only need to check if we have enough pairs
    if pairs.len() <= 2 {
        return false;
    }

    // Sample a subset of pairs to check their relative displacements
    for reference_pair in pairs.iter() {
        let reference_y = reference_pair.c_rect.1;

        // Calculate displacement vectors to other pairs
        let displacements: Vec<_> = pairs
            .iter()
            .map(|sv| {
                let dx = sv.dx;
                let dy = sv.c_rect.1 - reference_y;
                (dx, dy, (dx.powi(2) + dy.powi(2)).sqrt())
            })
            .collect();

        // Sort by distance and take the closest 3
        let mut closest = displacements;
        closest.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        closest.truncate(3);

        // If any displacement has |dx| > |dy|, it suggests a one-sided configuration
        if closest.iter().any(|(dx, dy, _)| dx.abs() > dy.abs()) {
            return false;
        }
    }

    true
}

/// Pair rectangles to the closest vertebrae
///
/// Returns two lists of ScrewVertebra: left and right
/// The left and right lists are sorted by the y-coordinate of the rectangle centroid
fn pair_to_closest<T: CalculateCentroid>(
    rectangles: &[T],
    spine: &Spine,
) -> (Vec<ScrewVertebra>, Vec<ScrewVertebra>) {
    let max_allowed_distance = ALLOWED_DIST_FACTOR * crate::draw::mean_plate_length(spine);
    log::debug!("max_allowed_distance: {}", max_allowed_distance);
    let vertebrae = spine.v_c7tl.0.slice_axis(Axis(0), Slice::from(1..));
    let vertebra_centroids = spine.c_c7tl.slice_axis(Axis(0), Slice::from(1..));
    let rectangle_centroids = rectangles
        .iter()
        .map(|rectangle| rectangle.centroid())
        .collect::<Vec<_>>();

    // list of potential vertebrae for each rectangle
    let potential_vertebrae_per_rect: Vec<_> = rectangle_centroids
        .iter()
        .enumerate()
        .map(|(i_rect, (x, y))| {
            let potential_vertebrae: Vec<_> = vertebrae
                .axis_iter(Axis(0))
                .enumerate()
                .filter_map(|(i_vert, vertebra)| {
                    // distances from the rectangle centroid to the corner points of the vertebra
                    let distances = vertebra.map_axis(Axis(1), |xy| {
                        ((x - xy[0]).powi(2) + (y - xy[1]).powi(2)).sqrt()
                    });
                    let min_distance = distances.iter().cloned().fold(f64::INFINITY, f64::min);
                    if min_distance < max_allowed_distance {
                        let dx = x - vertebra_centroids.index_axis(Axis(0), i_vert)[0];
                        Some(ScrewVertebra {
                            i_rect,
                            c_rect: (*x, *y),
                            i_vert,
                            dist: min_distance,
                            dx,
                        })
                    } else {
                        None
                    }
                })
                .collect();
            potential_vertebrae
        })
        .collect();
    log::debug!(
        "number of potential vertebrae per rect: {:?}",
        potential_vertebrae_per_rect
            .iter()
            .map(|v| v.len())
            .collect::<Vec<_>>()
    );

    // simple pairing with closest vertebrae
    let mut pairs: Vec<_> = potential_vertebrae_per_rect
        .iter()
        .filter_map(|potential_vertebrae| {
            let closest_vertebrae = potential_vertebrae
                .iter()
                .min_by(|sv1, sv2| sv1.dist.partial_cmp(&sv2.dist).unwrap())
                .map(|x| x.to_owned());
            closest_vertebrae
        })
        .collect();

    if pairs.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // sort pairs by dx
    pairs.sort_by(|a, b| a.dx.partial_cmp(&b.dx).unwrap());

    // check if it's one-sided
    let is_one_sided = detect_one_sided_configuration(&pairs);
    if is_one_sided {
        let mean_dx = pairs.iter().map(|sv| sv.dx).collect::<Vec<_>>().mean();
        if mean_dx > 0.0 {
            log::debug!("all screws are on the right side");
            return (Vec::new(), pairs);
        } else {
            log::debug!("all screws are on the left side");
            return (pairs, Vec::new());
        }
    }

    // split pairs into two groups: left and right groups
    // split minimize the sum of standard deviations of dx in each group
    let mut sum_stds = Vec::new();
    for i in 1..pairs.len() - 1 {
        let (left, right) = pairs.split_at(i);
        let left_dx = left.iter().map(|sv| sv.dx).collect::<Vec<_>>();
        let right_dx = right.iter().map(|sv| sv.dx).collect::<Vec<_>>();
        let left_std = left_dx.std();
        let right_std = right_dx.std();
        sum_stds.push(left_std + right_std);
    }
    let min_sum_std = sum_stds.iter().cloned().fold(f64::INFINITY, f64::min);
    let i_split = sum_stds.iter().position(|&x| x == min_sum_std).unwrap();
    let (mut left_pairs, mut right_pairs) = pairs.split_at_mut(i_split + 1);
    left_pairs.sort_by_y();
    right_pairs.sort_by_y();
    log::debug!(
        "number of left and right pairs: {} {}",
        left_pairs.len(),
        right_pairs.len()
    );

    (left_pairs.to_vec(), right_pairs.to_vec())
}

pub fn pair_screw(screws: &[detectron2::DetectedBox], spine: &Spine) -> Vec<Screw> {
    let (left_pairs, right_pairs) = pair_to_closest(screws, spine);

    let vertebra_centroids = spine.c_c7tl.slice_axis(Axis(0), Slice::from(1..));
    let rectangles = screws
        .iter()
        .map(|screw| Rectangle {
            tl: (screw.coords.0, screw.coords.1),
            br: (screw.coords.2, screw.coords.3),
        })
        .collect::<Vec<_>>();

    let refined_pairs =
        refine_two_sided_pairings(&rectangles, left_pairs, right_pairs, vertebra_centroids);

    if refined_pairs.len() != rectangles.len() {
        log::warn!(
            "Number of screws does not match number of rectangles. {} != {}",
            refined_pairs.len(),
            rectangles.len()
        );
    }
    refined_pairs
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LabelMeDetectron2 {
    #[serde(flatten)]
    pub labelme: LabelMeData,
    #[serde(flatten)]
    pub detectron2: detectron2::Output,
}

impl LabelMeDetectron2 {
    pub fn pair_screw(&self) -> Result<ScrewSpine, ScolError> {
        let screws = pair_screw(
            &self.detectron2.instances.boxes(1),
            &Spine::try_from(&self.labelme).unwrap(),
        );
        let spine = Spine::try_from(&self.labelme)?;
        let image_metadata = ImageMetadata::from(self.labelme.clone());
        Ok(ScrewSpine {
            spine,
            screws,
            image_metadata,
        })
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, ContentFilename)]
pub struct LabelMeDetectron2Line {
    pub content: LabelMeDetectron2,
    pub filename: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LabelMeOptionalDetectron2 {
    #[serde(flatten)]
    pub labelme: LabelMeData,
    #[serde(flatten)]
    pub detectron2: Option<detectron2::Output>,
}

impl LabelMeOptionalDetectron2 {
    pub fn screw_spine(&self) -> Result<ScrewSpine, ScolError> {
        if let Some(detectron2) = &self.detectron2 {
            let screws = pair_screw(
                &detectron2.instances.boxes(1),
                &Spine::try_from(&self.labelme)?,
            );
            let spine = Spine::try_from(&self.labelme)?;
            let image_metadata = ImageMetadata::from(self.labelme.clone());
            Ok(ScrewSpine {
                spine,
                screws,
                image_metadata,
            })
        } else {
            // check if `Screw` with group_id exists
            let group_id_exists = self
                .labelme
                .shapes
                .iter()
                .any(|shape| shape.label == "Screw" && shape.group_id.is_some());

            let image_metadata = ImageMetadata::from(self.labelme.clone());
            if group_id_exists {
                log::debug!("Use group_id to pair screws");
                let mut screws = Vec::new();
                for shape in &self.labelme.shapes {
                    if shape.label == "Screw" {
                        match shape.group_id {
                            Some(group_id) => {
                                // group_id is 1-indexed
                                let vertebra = group_id - 1;
                                let rect = Rectangle {
                                    tl: shape.points[0],
                                    br: shape.points[1],
                                };
                                screws.push(Screw::new(rect, vertebra));
                            }
                            None => panic!("Screw without group_id found"),
                        }
                    }
                }
                let spine = Spine::try_from(&self.labelme)?;
                Ok(ScrewSpine {
                    spine,
                    screws,
                    image_metadata,
                })
            } else {
                log::debug!("Use pairing algorithm to pair screws because no group_id found");
                let implant = ImplantSpine::try_from(&self.labelme)?;
                let screws = implant.pair_screw();
                Ok(ScrewSpine {
                    spine: implant.spine,
                    screws,
                    image_metadata,
                })
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LabelMeOptionalDetectron2Line {
    pub content: LabelMeOptionalDetectron2,
    pub filename: String,
}

/// Represents a dynamic programming state key
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct State {
    object_idx: usize,
    slot_idx: usize,
}

/// Represents the result of the assignment problem
#[derive(Debug)]
pub struct AssignmentResult {
    pub min_cost: f64,
    pub assignments: Vec<usize>,
}
pub fn ordered_assignment<F>(n_slots: usize, m_objects: usize, cost_function: F) -> AssignmentResult
where
    F: Fn(usize, usize) -> f64,
{
    // Memoization caches
    let mut dp: HashMap<State, f64> = HashMap::new();
    let mut prev: HashMap<State, Option<usize>> = HashMap::new();

    // Recursive solver function
    fn solve<F>(
        state: State,
        m_objects: usize,
        n_slots: usize,
        cost_function: &F,
        dp: &mut HashMap<State, f64>,
        prev: &mut HashMap<State, Option<usize>>,
    ) -> f64
    where
        F: Fn(usize, usize) -> f64,
    {
        // Base cases
        if state.object_idx == m_objects {
            return 0.0;
        }
        if state.slot_idx == n_slots {
            return f64::INFINITY; // Represents invalid state
        }

        // Check if we've already solved this state
        if let Some(&cost) = dp.get(&state) {
            return cost;
        }

        // Try skipping current slot
        let skip_cost = solve(
            State {
                object_idx: state.object_idx,
                slot_idx: state.slot_idx + 1,
            },
            m_objects,
            n_slots,
            cost_function,
            dp,
            prev,
        );

        // Try using current slot
        let use_cost = cost_function(state.object_idx, state.slot_idx)
            + solve(
                State {
                    object_idx: state.object_idx + 1,
                    slot_idx: state.slot_idx + 1,
                },
                m_objects,
                n_slots,
                cost_function,
                dp,
                prev,
            );

        // Store the better choice
        let best_cost = skip_cost.min(use_cost);
        dp.insert(state.clone(), best_cost);

        // Store which choice was made
        let slot_idx = state.slot_idx;
        prev.insert(
            state,
            if use_cost < skip_cost {
                Some(slot_idx)
            } else {
                None
            },
        );

        best_cost
    }

    // Initial state
    let initial_state = State {
        object_idx: 0,
        slot_idx: 0,
    };

    // Solve the problem
    let min_cost = solve(
        initial_state.clone(),
        m_objects,
        n_slots,
        &cost_function,
        &mut dp,
        &mut prev,
    );

    log::debug!("min_cost: {}", min_cost);
    if min_cost == f64::INFINITY {
        panic!("No valid assignment found");
    }

    // Reconstruct the solution
    let mut assignments = Vec::new();
    let mut curr_state = initial_state;

    while curr_state.object_idx < m_objects {
        if let Some(&Some(slot)) = prev.get(&curr_state) {
            assignments.push(slot);
            curr_state.object_idx += 1;
        }
        curr_state.slot_idx += 1;
    }

    AssignmentResult {
        min_cost,
        assignments,
    }
}

// Example usage and tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_case() {
        let n_slots = 3;
        let m_objects = 2;

        // Cost matrix for object i to slot j assignments
        let costs = [
            vec![1.5, 2.4, 3.1], // costs for object 0
            vec![2.2, 1.6, 2.4],
        ];

        let cost_function = |i: usize, j: usize| costs[i][j];

        let result = ordered_assignment(n_slots, m_objects, cost_function);
        assert!((result.min_cost - 3.1).abs() < 1e-10); // 1.5 + 1.6
        assert_eq!(result.assignments, vec![0, 1]);
    }

    #[test]
    fn test_complex_case() {
        let n_slots = 6;
        let m_objects = 3;

        // Cost matrix for object i to slot j assignments
        let costs = [
            vec![2.3, 1.1, 0.4, 0.5, 2.2, 1.1], // costs for object 0
            vec![3.2, 2.2, 1.3, 0.4, 0.8, 1.3], // costs for object 1
            vec![3.3, 1.3, 2.2, 1.1, 0.4, 0.7],
        ];

        let cost_function = |i: usize, j: usize| costs[i][j];

        let result = ordered_assignment(n_slots, m_objects, cost_function);
        assert_eq!(result.assignments, vec![2, 3, 4]);
        assert!((result.min_cost - 1.2).abs() < 1e-10); // optimal combination
    }
}
