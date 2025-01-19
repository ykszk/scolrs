use crate::ContentFilename;
use crate::{HasImageMetadata, ImplantDraw, Scalable};
use labelme_rs::LabelMeData;
use ndarray::{Array, Axis, Slice};
use serde::{Deserialize, Serialize};

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
}

impl Screw {
    pub fn new(bb: Rectangle, vertebra: usize) -> Self {
        Self { bb, vertebra }
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

impl TryFrom<&LabelMeData> for ImplantSpine {
    type Error = ScolError;

    fn try_from(data: &LabelMeData) -> Result<Self, Self::Error> {
        let spine = Spine::try_from(data)?;
        let implant = Implant::from(data);
        Ok(Self { spine, implant })
    }
}

impl ImplantSpine {
    fn pair_impl(&self, rectangles: &[Rectangle]) -> Vec<Screw> {
        let max_allowed_distance = 0.5 * crate::draw::mean_plate_length(&self.spine);
        log::debug!("max_allowed_distance: {}", max_allowed_distance);
        let vertebrae = self.spine.v_c7tl.0.slice_axis(Axis(0), Slice::from(1..));
        let rectangle_centroids = rectangles
            .iter()
            .map(|rectangle| {
                (
                    (rectangle.tl.0 + rectangle.br.0) / 2.0,
                    (rectangle.tl.1 + rectangle.br.1) / 2.0,
                )
            })
            .collect::<Vec<_>>();
        // list of potential vertebrae for each rectangle
        let potential_vertebrae_per_rect: Vec<_> = rectangle_centroids
            .iter()
            .enumerate()
            .map(|(i_rect, (x, y))| {
                let potential_vertebrae: Vec<_> = vertebrae
                    .axis_iter(Axis(0))
                    .enumerate()
                    .filter_map(|(i, vertebra)| {
                        let distances = vertebra.map_axis(Axis(1), |xy| {
                            ((x - xy[0]).powi(2) + (y - xy[1]).powi(2)).sqrt()
                        });
                        let min_distance = distances.iter().cloned().fold(f64::INFINITY, f64::min);
                        if min_distance < max_allowed_distance {
                            Some((rectangles[i_rect].clone(), i, min_distance))
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
        // try simple pairs with closest vertebrae first
        let pairs: Vec<_> = potential_vertebrae_per_rect
            .iter()
            .filter_map(|potential_vertebrae| {
                let closest_vertebrae = potential_vertebrae
                    .iter()
                    .min_by(|(_, _, dist1), (_, _, dist2)| dist1.partial_cmp(dist2).unwrap())
                    .map(|x| x.to_owned());
                closest_vertebrae
            })
            .collect();
        let mut rect_count_per_vertebra: Vec<u8> = vec![0; vertebrae.len_of(Axis(0))];
        pairs.iter().for_each(|(_, i, _)| {
            rect_count_per_vertebra[*i] += 1;
        });
        log::debug!("rect_count_per_vertebra: {:?}", rect_count_per_vertebra);
        let too_many_rect_per_vertebra = rect_count_per_vertebra.iter().any(|&count| count > 2);
        if too_many_rect_per_vertebra {
            log::error!("Too many rectangles per vertebra");
        }
        pairs
            .into_iter()
            .map(|(rect, i, _)| Screw::new(rect, i))
            .collect()
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

    fn scale(&mut self) -> Result<(), Self::Error> {
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
        let mut g = self.default_group().set("fill", "none");

        let mut screws_per_vertebra: Vec<Vec<_>> = vec![Vec::new(); spine.v_c7tl.0.len_of(Axis(0))];

        for screw in &self.0.screws {
            screws_per_vertebra[screw.vertebra].push(screw);
        }

        for (i, screws) in screws_per_vertebra.iter().enumerate() {
            let line_color = crate::draw::TAB10_NEW_TAB10[i % 20];
            if screws.is_empty() {
                continue;
            }
            let mut g_screw_vertebra = element::Group::new().set("stroke", line_color);
            let mut corners = spine.v_c7tl.0.index_axis(Axis(0), i + 1).to_owned(); // vertebra + 1 to skip C7

            // Change point-order from (tl, tr, bl, br) to (tl, tr, br, bl)
            corners.swap((2, 0), (3, 0)); // bl.x <-> br.x
            corners.swap((2, 1), (3, 1)); // bl.y <-> br.y

            let vertebra = painter.polygon(corners).set("stroke-dasharray", "5,5");
            g_screw_vertebra = g_screw_vertebra.add(vertebra);

            for screw in screws {
                let rect = Array::from_shape_vec(
                    (2, 2),
                    vec![screw.bb.tl.0, screw.bb.tl.1, screw.bb.br.0, screw.bb.br.1],
                )
                .unwrap();
                let bbox = painter.rectangle(rect);
                g_screw_vertebra = g_screw_vertebra.add(bbox);
            }
            g = g.add(g_screw_vertebra);
        }
        Ok(g)
    }
}

/// Screws and vertebrae with hover effect
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POLYGON])]
pub struct HoverScrews<'a>(&'a ScrewSpine);
impl ImplantComponent for HoverScrews<'_> {}
impl DrawComponent for HoverScrews<'_> {
    fn draw(
        &self,
        painter: &mut Painter,
        _label_colors: &mut crate::draw::ColorPalette,
        line_colors: &mut crate::draw::ColorPalette,
    ) -> Result<element::Group, DrawError> {
        let spine = &self.0.spine;
        let mut style = ".hidden {opacity:0;transition:opacity 0.3s ease;}\n".to_string();
        let screw_color = line_colors.get_or_new("Screw");
        style.push_str(format!(".screw {{stroke:{screw_color};fill:transparent}}\n").as_str());
        let vertebra_color = line_colors.get_or_new("ScrewVertebra");
        style.push_str(format!(".screw-vertebra {{stroke:{vertebra_color};fill:none}}\n").as_str());
        for i in 0..self.0.screws.len() {
            let hover = format!(".screw{i}:hover ~ .screw{i}-vertebra {{opacity:1}}\n");
            style.push_str(&hover);
        }
        let mut g = self.default_group().add(element::Style::new(style));

        for (i, screw) in self.0.screws.iter().enumerate() {
            let rect = Array::from_shape_vec(
                (2, 2),
                vec![screw.bb.tl.0, screw.bb.tl.1, screw.bb.br.0, screw.bb.br.1],
            )
            .unwrap();
            let bbox = painter
                .rectangle(rect)
                .set("class", format!("screw screw{}", i));
            g = g.add(bbox);
            let mut corners = spine
                .v_c7tl
                .0
                .index_axis(Axis(0), screw.vertebra + 1)
                .to_owned(); // vertebra + 1 to skip C7

            // Change point-order from (tl, tr, bl, br) to (tl, tr, br, bl)
            corners.swap((2, 0), (3, 0)); // bl.x <-> br.x
            corners.swap((2, 1), (3, 1)); // bl.y <-> br.y

            let vertebra = painter.polygon(corners).set(
                "class",
                format!("hidden screw-vertebra screw{}-vertebra", i),
            );
            g = g.add(vertebra);
        }
        Ok(g)
    }
}

impl<'a, 'b> From<(&'b ImplantDraw, &'a ScrewSpine)> for Box<dyn DrawComponent + 'a> {
    fn from((draw, screw_spine): (&'b ImplantDraw, &'a ScrewSpine)) -> Self {
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

    // impl From<&Instances> for Vec<Box> {
    //     fn from(instance: &Instances) -> Self {
    //         instance
    //             .pred_boxes
    //             .iter()
    //             .zip(instance.scores.iter())
    //             .zip(instance.pred_classes.iter())
    //             .map(|((coords, &score), &class)| Box {
    //                 coords: *coords,
    //                 score,
    //                 class,
    //             })
    //             .collect()
    //     }
    // }
}

pub fn pair_screw(screws: &[detectron2::DetectedBox], spine: &Spine) -> Vec<Screw> {
    let max_allowed_distance = 1.0 * crate::draw::mean_plate_length(spine);
    log::debug!("max_allowed_distance: {}", max_allowed_distance);
    let vertebrae = spine.v_c7tl.0.slice_axis(Axis(0), Slice::from(1..));
    let screw_centroids = screws
        .iter()
        .map(|screw| {
            (
                (screw.coords.0 + screw.coords.2) / 2.0,
                (screw.coords.1 + screw.coords.3) / 2.0,
            )
        })
        .collect::<Vec<_>>();
    // pair each screw with the closest vertebra
    // one vertebra can be paired with at most two screws
    // optimal pairing minimizes the sum of distances between paired screws and vertebrae
    let potential_vertebrae_per_screw: Vec<_> = screw_centroids
        .iter()
        .enumerate()
        .map(|(i_screw, (x, y))| {
            let potential_vertebrae: Vec<_> = vertebrae
                .axis_iter(Axis(0))
                .enumerate()
                .filter_map(|(i, vertebra)| {
                    let distances = vertebra.map_axis(Axis(1), |xy| {
                        ((x - xy[0]).powi(2) + (y - xy[1]).powi(2)).sqrt()
                    });
                    let min_distance = distances.iter().cloned().fold(f64::INFINITY, f64::min);
                    if min_distance < max_allowed_distance {
                        Some((screws[i_screw].clone(), i, min_distance))
                    } else {
                        None
                    }
                })
                .collect();
            potential_vertebrae
        })
        .collect();
    log::debug!(
        "number of potential vertebrae per screw: {:?}",
        potential_vertebrae_per_screw
            .iter()
            .map(|v| v.len())
            .collect::<Vec<_>>()
    );
    // try simple pairs with closest vertebrae first
    let pairs: Vec<_> = potential_vertebrae_per_screw
        .iter()
        .filter_map(|potential_vertebrae| {
            let closest_vertebrae = potential_vertebrae
                .iter()
                .min_by(|(_, _, dist1), (_, _, dist2)| dist1.partial_cmp(dist2).unwrap())
                .map(|x| x.to_owned());
            closest_vertebrae
        })
        .collect();
    let mut screw_count_per_vertebra: Vec<u8> = vec![0; vertebrae.len_of(Axis(0))];
    pairs.iter().for_each(|(_, i, _)| {
        screw_count_per_vertebra[*i] += 1;
    });
    log::debug!("screw_count_per_vertebra: {:?}", screw_count_per_vertebra);
    let too_many_screw_per_vertebra = screw_count_per_vertebra.iter().any(|&count| count > 2);
    if !too_many_screw_per_vertebra {
        return pairs
            .into_iter()
            .map(|(screw, i, _)| {
                Screw::new(
                    Rectangle {
                        tl: (screw.coords.0, screw.coords.1),
                        br: (screw.coords.2, screw.coords.3),
                    },
                    i,
                )
            })
            .collect();
    }
    log::debug!("Too many screws per vertebra");
    let mut screws_per_vertebra: Vec<Vec<_>> = vec![Vec::new(); vertebrae.len_of(Axis(0))];
    for pair in pairs {
        screws_per_vertebra[pair.1].push(pair.0);
    }
    let refined_screws_per_vertebra: Vec<_> = screws_per_vertebra
        .into_iter()
        .map(|mut screws| {
            if screws.len() < 3 {
                return screws;
            };
            screws.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
            screws.truncate(2);
            screws
        })
        .collect();
    refined_screws_per_vertebra
        .into_iter()
        .enumerate()
        .flat_map(|(i, screws)| {
            screws
                .into_iter()
                .map(|screw| {
                    Screw::new(
                        Rectangle {
                            tl: (screw.coords.0, screw.coords.1),
                            br: (screw.coords.2, screw.coords.3),
                        },
                        i,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
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
            let implant = ImplantSpine::try_from(&self.labelme)?;
            let screws = implant.pair_screw();
            Ok(ScrewSpine {
                spine: implant.spine,
                screws,
                image_metadata: ImageMetadata::from(self.labelme.clone()),
            })
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LabelMeOptionalDetectron2Line {
    pub content: LabelMeOptionalDetectron2,
    pub filename: String,
}
