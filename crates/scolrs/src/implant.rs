use crate::{HasImageMetadata, ImplantDraw, Scalable};
use labelme_rs::LabelMeData;
use ndarray::{Array, Axis, Slice};

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
        // pair each rectangle with the closest vertebra
        // one vertebra can be paired with at most two rectangles
        // optimal pairing minimizes the sum of distances between paired rectangles and vertebrae
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
        if !too_many_rect_per_vertebra {
            return pairs
                .into_iter()
                .map(|(rect, i, _)| Screw::new(rect, i))
                .collect();
        }
        log::error!("Too many rectangles per vertebra");
        Vec::new()
    }

    pub fn pair_screw(&self) -> Vec<Screw> {
        self.pair_impl(&self.implant.screw)
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

/// Screws and
#[derive(Named)]
#[draw_type([CLASS_ANNOTATION, CLASS_POLYGON])]
struct Screws<'a>(&'a ScrewSpine);
impl ImplantComponent for Screws<'_> {}
impl DrawComponent for Screws<'_> {
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
