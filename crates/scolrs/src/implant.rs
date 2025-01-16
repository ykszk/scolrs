use labelme_rs::LabelMeData;
use ndarray::{Axis, Slice};

use crate::{ScolError, Spine};

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
    fn impl_list_potential_pairs(
        mut potential_vertebrae_per_rect: Vec<Vec<(Rectangle, usize, f64)>>,
        rect_count_per_vertebra: Vec<u8>,
    ) -> Vec<Vec<(Rectangle, usize, f64)>> {
        if potential_vertebrae_per_rect.is_empty() {
            return Vec::new();
        }
        let last_rect = potential_vertebrae_per_rect.pop().unwrap();
        let mut leaves = Vec::new();
        for (rect, i_vert, dist) in last_rect {
            if rect_count_per_vertebra[i_vert] < 2 {
                let mut new_rect_count_per_vertebra = rect_count_per_vertebra.clone();
                new_rect_count_per_vertebra[i_vert] += 1;
                let new_leaves = Self::impl_list_potential_pairs(
                    potential_vertebrae_per_rect.clone(),
                    new_rect_count_per_vertebra,
                )
                .into_iter()
                .map(|mut leaf| {
                    leaf.push((rect.clone(), i_vert, dist));
                    leaf
                })
                .collect::<Vec<_>>();
                leaves.extend(new_leaves);
            }
        }
        leaves
    }

    fn list_potential_pairs(
        potential_vertebrae_per_rect: Vec<Vec<(Rectangle, usize, f64)>>,
        n_vertebrae: usize,
    ) -> Vec<Vec<(Rectangle, usize, f64)>> {
        // potential_vertebrae_per_rect.reverse();
        let rect_count_per_vertebra: Vec<u8> = vec![0; n_vertebrae];
        Self::impl_list_potential_pairs(potential_vertebrae_per_rect, rect_count_per_vertebra)
    }

    fn pair_impl(&self, rectangles: &[Rectangle]) -> Vec<(Rectangle, usize)> {
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
        let too_many_rect_per_vertebra = rect_count_per_vertebra.iter().any(|&count| count > 2);
        if !too_many_rect_per_vertebra {
            return pairs.into_iter().map(|(rect, i, _)| (rect, i)).collect();
        }
        panic!("Too many rectangles per vertebra");

        let potential_pairs =
            Self::list_potential_pairs(potential_vertebrae_per_rect, vertebrae.len_of(Axis(0)));
        if potential_pairs.is_empty() {
            log::warn!("No potential pairs found");
            return Vec::new();
        }
        println!("{:?}", potential_pairs);
        let costs = potential_pairs
            .iter()
            .map(|leaf| leaf.iter().map(|(_, _, dist)| dist).sum::<f64>())
            .collect::<Vec<_>>();
        let min_cost = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let best_leaf = potential_pairs
            .iter()
            .enumerate()
            .find_map(|(i, leaf)| {
                if costs[i] == min_cost {
                    Some(leaf)
                } else {
                    None
                }
            })
            .unwrap();
        best_leaf
            .iter()
            .map(|(_, i, _)| (rectangles[*i].clone(), *i))
            .collect()
    }

    pub fn pair_screw(&self) -> Vec<(Rectangle, usize)> {
        self.pair_impl(&self.implant.screw)
    }
}
