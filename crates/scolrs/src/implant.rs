use crate::{ContentFilename, ScaledType};
use crate::{HasImageMetadata, ImplantDraw, Scalable};
use detectron2::DetectedBox;
use indexmap::IndexMap;
use labelme_rs::LabelMeData;
use ndarray::{Array, Array1, ArrayView2, Axis, Slice};
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

pub const LABEL_SCREW: &str = "Screw";
pub const LABEL_SCREW_LEFT: &str = "ScrewLeft";
pub const LABEL_SCREW_RIGHT: &str = "ScrewRight";

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

    pub fn new(bb: Rectangle, vertebra: usize, left: Option<bool>) -> Self {
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

pub trait VecStats {
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

impl From<ScrewSpine> for ImplantSpine {
    fn from(screw_spine: ScrewSpine) -> Self {
        ImplantSpine {
            spine: screw_spine.spine,
            implant: Implant {
                screw: screw_spine.screws.iter().map(|s| s.bb.clone()).collect(),
                hook: Vec::new(),
                transverse: Vec::new(),
                rod: Vec::new(),
            },
        }
    }
}

/// The allowed distance factor is used to determine the maximum distance between a rectangle and a vertebra
/// to be considered a valid pairing. This factor is multiplied by the mean plate length of the spine.
const ALLOWED_DIST_FACTOR: f64 = 1.0;

pub trait ScrewVertebraUtils {
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

impl ScrewVertebraUtils for Vec<ScrewVertebra> {
    fn sort_by_y(&mut self) {
        self.sort_by(|a, b| {
            let (_x1, y1) = a.c_rect;
            let (_x2, y2) = b.c_rect;
            y1.partial_cmp(&y2).unwrap()
        });
    }
    fn count_rect_per_vertebra(&self, _n_vertebrae: usize) -> Vec<u8> {
        unimplemented!()
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
            .map(|(sv, &i_vert)| Screw::with_pos(rectangles[sv.i_screw].clone(), i_vert, is_left))
            .collect()
    } else {
        pairs
            .iter()
            .map(|x| Screw::with_pos(rectangles[x.i_screw].clone(), x.i_vert, is_left))
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
                .map(|sv| Screw::with_pos(rectangles[sv.i_screw].clone(), i_vert, is_left)),
        );
    }
    chosen_pairs
}

impl ImplantSpine {
    pub fn pair_screw(&self) -> Vec<Screw> {
        pair_screw_rects(&self.implant.screw, &self.spine)
    }

    pub fn screw_spine(self, image_metadata: ImageMetadata) -> ScrewSpine {
        let screws = self.pair_screw();
        ScrewSpine {
            spine: self.spine,
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
        label_lr: bool,
    ) -> LabelMeData {
        let mut shapes = Vec::new();
        for screw in &self.screws {
            let points = vec![screw.bb.tl, screw.bb.br];
            let label = if label_lr {
                match screw.left {
                    Some(true) => LABEL_SCREW_LEFT.to_string(),
                    Some(false) => LABEL_SCREW_RIGHT.to_string(),
                    None => LABEL_SCREW.to_string(),
                }
            } else {
                LABEL_SCREW.to_string()
            };
            let shape = labelme_rs::Shape {
                label,
                points,
                shape_type: "rectangle".to_string(),
                flags: Default::default(),
                group_id: Some(screw.vertebra + 1),
            };
            shapes.push(shape);
        }
        if include_vertebrae {
            let labels4 = vec!["TL", "TR", "BL", "BR"];
            let labels2 = vec!["TL", "TR"];
            for (i_vert, vert) in self.spine.v_c7tl.0.axis_iter(Axis(0)).enumerate() {
                let labels = if i_vert < self.spine.v_c7tl.0.len_of(Axis(0)) - 1 {
                    // all four corner points for vertebrae except the last one
                    &labels4
                } else {
                    // Last vertebra (Sacrum) has only TL and TR points
                    &labels2
                };
                for (i, label) in labels.iter().enumerate() {
                    let point = (vert[[i, 0]], vert[[i, 1]]);
                    let shape = labelme_rs::Shape {
                        label: label.to_string(),
                        points: vec![point],
                        shape_type: "point".to_string(),
                        flags: Default::default(),
                        group_id: Some(i_vert),
                    };
                    shapes.push(shape);
                }
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
pub struct ScrewVertebra {
    /// index of the rectangle in the screws list
    pub i_screw: usize,
    /// centroid of the rectangle (screw)
    pub c_rect: (f64, f64),
    /// index of the vertebra in the spine
    pub i_vert: usize,
    /// distance from the rectangle centroid to the vertebra centroid
    pub dist: f64,
    /// displacement of the rectangle (screw) center from the vertebra center along the x-axis
    pub dx: f64,
}

impl ScrewVertebra {
    pub fn create(
        i_screw: usize,
        c_rect: (f64, f64),
        i_vert: usize,
        dx: f64,
        vertebra: ArrayView2<f64>,
    ) -> Self {
        let min_dist = vertebra
            .map_axis(Axis(1), |xy| {
                ((c_rect.0 - xy[0]).powi(2) + (c_rect.1 - xy[1]).powi(2)).sqrt()
            })
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        Self {
            i_screw,
            c_rect,
            i_vert,
            dist: min_dist,
            dx,
        }
    }
}

/// Define a trait to use [pair_to_closest] with both [Rectangle] and [DetectedBox]
pub trait CalculateCentroid {
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

/// Helper function to determine if the screws are all on one side
/// If there are two or more screws that are vertically aligned, pairs are not one-sided
pub fn detect_one_sided_configuration(pairs: &[ScrewVertebra]) -> bool {
    // Only need to check if we have enough pairs
    if pairs.len() < 2 {
        return true;
    }

    // Sample a subset of pairs to check their relative displacements
    for reference_pair in pairs.iter() {
        let reference_y = reference_pair.c_rect.1;

        let mut closest = pairs
            .iter()
            .filter(|sv| sv.i_screw != reference_pair.i_screw)
            .collect::<Vec<_>>();
        closest.sort_by(|a, b| {
            let dx_a = reference_pair.c_rect.0 - a.c_rect.0;
            let dy_a = reference_pair.c_rect.1 - a.c_rect.1;
            let dist_a = (dx_a * dx_a + dy_a * dy_a).sqrt();

            let dx_b = reference_pair.c_rect.0 - b.c_rect.0;
            let dy_b = reference_pair.c_rect.1 - b.c_rect.1;
            let dist_b = (dx_b * dx_b + dy_b * dy_b).sqrt();

            dist_a.partial_cmp(&dist_b).unwrap()
        });
        closest.truncate(3);

        let displacements = closest.iter().map(|sv| {
            let dx = sv.c_rect.0 - reference_pair.c_rect.0;
            let dy = sv.c_rect.1 - reference_y;
            (dx, dy)
        });

        // Check if a displacement vector is horizontal (less than 30 degrees)
        for (dx, dy) in displacements {
            // Handle edge cases properly
            if dx.abs() < 1e-6 {
                return false; // Vertical vector, definitely not horizontal
            }

            // Calculate angle from horizontal
            let angle_rad = (dy / dx).abs().atan();
            let angle_deg = angle_rad.to_degrees();

            // Consider horizontal if angle is less than threshold
            const MAX_ANGLE_DEG: f64 = 30.0;
            if angle_deg < MAX_ANGLE_DEG {
                return false; // Found a horizontal vector, not one-sided
            }
        }
    }

    true
}

/// Pair rectangles to the closest vertebrae
///
/// Returns two lists of ScrewVertebra: left and right
/// The left and right lists are sorted by the y-coordinate of the rectangle centroid
fn pair_to_closest(
    screws: &[Rectangle],
    spine: &Spine,
) -> (Vec<ScrewVertebra>, Vec<ScrewVertebra>) {
    let max_allowed_distance = ALLOWED_DIST_FACTOR * crate::draw::mean_plate_length(spine);
    log::debug!("max_allowed_distance: {}", max_allowed_distance);
    let vertebrae = spine.t1_to_sac_vertebrae();
    let vertebra_centroids = spine.t1_to_sac_centroids();

    let rectangle_centroids = screws
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
                    let dx = x - vertebra_centroids.index_axis(Axis(0), i_vert)[0];
                    let screw_vertebra =
                        ScrewVertebra::create(i_rect, (*x, *y), i_vert, dx, vertebra);
                    if screw_vertebra.dist < max_allowed_distance {
                        Some(screw_vertebra)
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
    let pairs: Vec<_> = potential_vertebrae_per_rect
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

    split_pairs(screws, pairs)
}

fn split_pairs(
    screws: &[Rectangle],
    pairs: Vec<ScrewVertebra>,
) -> (Vec<ScrewVertebra>, Vec<ScrewVertebra>) {
    // Early return for small inputs
    if pairs.len() <= 1 {
        if pairs[0].dx > 0.0 {
            return (Vec::new(), pairs);
        } else {
            return (pairs, Vec::new());
        }
    }

    split_pairs_brute_force(screws, pairs)
}

/// Split pairs into left and right groups
/// The split is done by minimizing the sum of differences of dx in each group
/// For efficiency, some obvious screws are pre-split into hard_left and hard_right
fn split_pairs_brute_force(
    screws: &[Rectangle],
    mut pairs: Vec<ScrewVertebra>,
) -> (Vec<ScrewVertebra>, Vec<ScrewVertebra>) {
    // sort by y-coordinate
    pairs.sort_by(|a, b| a.c_rect.1.partial_cmp(&b.c_rect.1).unwrap());

    let (hard_left, hard_right) = pre_split_easy_screws(screws, &pairs);

    // Find indices that are not in either hard_left or hard_right
    let unsorted_indices: Vec<_> = (0..pairs.len())
        .filter(|i| !hard_left.contains(i) && !hard_right.contains(i))
        .collect();

    log::debug!(
        "hard_left: {:?}, hard_right: {:?}, unsorted_indices: {:?}",
        hard_left,
        hard_right,
        unsorted_indices
    );

    split_unsorted_screws(pairs, hard_left, unsorted_indices)
}

/// Split unsorted screws into left and right groups
///
/// `pairs` must be sorted by y-coordinate.
pub fn split_unsorted_screws(
    pairs: Vec<ScrewVertebra>,
    hard_left: Vec<usize>,
    unsorted_indices: Vec<usize>,
) -> (Vec<ScrewVertebra>, Vec<ScrewVertebra>) {
    let mut min_sum_diff = f64::INFINITY;
    let mut best_split = 0u64;

    let bitmask_hard_left = hard_left.iter().fold(0u64, |acc, &idx| acc | (1 << idx));

    // Try all possible splits for unsorted pairs
    for unsorted_bitmask in 0u64..(1 << unsorted_indices.len()) {
        // Create a bitmask for the current split
        // Start with the hard left bitmask
        let mut bitmask = bitmask_hard_left;

        // Apply unsorted pairs
        for (i, &idx) in unsorted_indices.iter().enumerate() {
            if unsorted_bitmask & (1 << i) != 0 {
                bitmask |= 1 << idx;
            }
        }

        // Calculate sum of differences for both groups
        let mut sum_diff = 0.0;
        let mut prev_left_x: Option<f64> = None;
        let mut prev_right_x: Option<f64> = None;

        for (i, pair) in pairs.iter().enumerate() {
            let is_left = (bitmask & (1 << i)) != 0;
            let x = pair.c_rect.0;

            if is_left {
                if let Some(prev_x) = prev_left_x {
                    sum_diff += (x - prev_x).abs();
                }
                prev_left_x = Some(x);
            } else {
                if let Some(prev_x) = prev_right_x {
                    sum_diff += (x - prev_x).abs();
                }
                prev_right_x = Some(x);
            }
        }

        // Update if this split is better
        if sum_diff < min_sum_diff {
            min_sum_diff = sum_diff;
            best_split = bitmask;
        }
    }

    // Divide pairs according to the best split found
    let n_left_pairs = (0..pairs.len())
        .filter(|i| (best_split & (1 << i)) != 0)
        .count();
    let mut left_pairs = Vec::with_capacity(n_left_pairs);
    let mut right_pairs = Vec::with_capacity(pairs.len() - n_left_pairs);

    for (i, pair) in pairs.into_iter().enumerate() {
        if (best_split & (1 << i)) != 0 {
            left_pairs.push(pair);
        } else {
            right_pairs.push(pair);
        }
    }

    // Sort results by y-coordinate
    left_pairs.sort_by_y();
    right_pairs.sort_by_y();

    // Make sure left is actually on the left side
    if !left_pairs.is_empty() && !right_pairs.is_empty() {
        let left_mean_x =
            left_pairs.iter().map(|sv| sv.c_rect.0).sum::<f64>() / left_pairs.len() as f64;
        let right_mean_x =
            right_pairs.iter().map(|sv| sv.c_rect.0).sum::<f64>() / right_pairs.len() as f64;

        if left_mean_x > right_mean_x {
            std::mem::swap(&mut left_pairs, &mut right_pairs);
        }
    }

    (left_pairs, right_pairs)
}

fn pre_split_easy_screws(
    screws: &[Rectangle],
    pairs: &[ScrewVertebra],
) -> (Vec<usize>, Vec<usize>) {
    // pre split obvious screws
    // Find vertically aligned screws and split them into hard_left and hard_right
    let mut hard_left = Vec::new();
    let mut hard_right = Vec::new();
    for (i_pair, ref_pair) in pairs.iter().enumerate() {
        // find vertically aligned screws
        let ref_screw = &screws[ref_pair.i_screw];
        let ref_width = (ref_screw.tl.0 - ref_screw.br.0).abs();
        let ref_height = (ref_screw.tl.1 - ref_screw.br.1).abs();
        for pair in pairs.iter() {
            if pair.i_screw == ref_pair.i_screw {
                continue;
            }
            let screw = &screws[pair.i_screw];
            let width = (screw.tl.0 - screw.br.0).abs();
            let mean_width = (width + ref_width) / 2.0;
            let dx = (pair.c_rect.0 - ref_pair.c_rect.0).abs();
            if dx < 0.5 * mean_width {
                // Too close horizontally
                continue;
            }
            let height = (screw.tl.1 - screw.br.1).abs();
            let mean_height = (height + ref_height) / 2.0;
            let dy = (pair.c_rect.1 - ref_pair.c_rect.1).abs();
            if dy < 0.5 * mean_height {
                // Close enough vertically
                let dx = pair.c_rect.0 - ref_pair.c_rect.0;
                if dx > 0.0 {
                    hard_left.push(i_pair);
                } else {
                    hard_right.push(i_pair);
                }
                break;
            }
        }
    }
    (hard_left, hard_right)
}

pub fn pair_screw_rects(screw_rects: &[Rectangle], spine: &Spine) -> Vec<Screw> {
    let (left_pairs, right_pairs) = pair_to_closest(screw_rects, spine);

    log::debug!(
        "left pairs: {:?}, right pairs: {:?}",
        left_pairs.len(),
        right_pairs.len()
    );

    let vertebra_centroids = spine.c_c7tl.slice_axis(Axis(0), Slice::from(1..));

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

pub fn pair_screw(screws: &[detectron2::DetectedBox], spine: &Spine) -> Vec<Screw> {
    let screw_rects = screws
        .iter()
        .map(|screw| Rectangle {
            tl: (screw.coords.0, screw.coords.1),
            br: (screw.coords.2, screw.coords.3),
        })
        .collect::<Vec<_>>();
    pair_screw_rects(&screw_rects, spine)
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
            log::debug!("Using Detectron2 output to pair screws");
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
            let image_metadata = ImageMetadata::from(self.labelme.clone());

            // check if `Screw` with group_id exists
            let screw_group_ids = self
                .labelme
                .shapes
                .iter()
                .filter(|shape| {
                    shape.label == LABEL_SCREW
                        || shape.label == LABEL_SCREW_LEFT
                        || shape.label == LABEL_SCREW_RIGHT
                })
                .map(|shape| shape.group_id)
                .collect::<Vec<_>>();
            let group_id_exists = screw_group_ids.iter().any(|&id| id.is_some());
            if group_id_exists {
                let group_id_exists_all = screw_group_ids.iter().all(|&id| id.is_some());
                if !group_id_exists_all {
                    panic!("Screw without group_id found")
                }
                log::debug!("Use group_id to pair screws");
                let mut screws = Vec::new();
                for shape in &self.labelme.shapes {
                    if shape.label.starts_with(LABEL_SCREW) {
                        let group_id = shape.group_id.unwrap(); // guaranteed to be Some
                        let vertebra = group_id - 1; // group_id is 1-indexed
                        let rect = Rectangle {
                            tl: shape.points[0],
                            br: shape.points[1],
                        };
                        let left = match shape.label.as_str() {
                            LABEL_SCREW => None,
                            LABEL_SCREW_LEFT => Some(true),
                            LABEL_SCREW_RIGHT => Some(false),
                            _ => {
                                return Err(ScolError::Value(format!(
                                    "Unknown screw label: {}",
                                    shape.label
                                )));
                            }
                        };
                        screws.push(Screw::new(rect, vertebra, left));
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
