# Shape Normalization for Cervical Spine Motion Analysis

## Purpose

Cervical spine kinematic analysis compares vertebral positions across poses (flexion,
neutral, extension). Raw annotations conflate two distinct sources of variation:

- **Motion** — the true biological signal of interest
- **Shape and annotation variation** — differences in vertebral morphology across
  patients, and inconsistency in how corners are placed by the annotator

Shape normalization separates these two sources so that downstream analysis reflects
pure motion.

---

## What is normalized away

| Source | Description |
|---|---|
| Annotator placement error | The same corner clicked slightly differently across the three images |
| Vertebral shape variation | No two vertebrae are identical; corner geometry differs across patients |
| X-ray magnification | Source-to-film distance varies between exposures, introducing a scale difference |

None of these are motion. All three are removed by the normalization.

---

## Procedure (per vertebra)

**Neutral is the reference.** Flexion and extension are registered to it.

1. **Register** — find the rigid transform (rotation + translation) that best aligns
   the flexion corners to the neutral corners, and repeat for extension. One global
   scale factor is shared across all vertebrae to account for magnification.

2. **Compute the mean shape** — after registration, average the corner positions in
   neutral space across all three aligned poses. This mean shape is a cleaner
   estimate of the true vertebral geometry than any single annotation.

3. **Back-project** — invert each registration transform to carry the mean shape back
   into the original flexion and extension image frames. The result is a set of
   corners in each pose that represents the mean geometry *as if it had been annotated
   consistently*.

---

## What remains after normalization

After back-projection, the corners in each pose differ only because of motion —
annotation noise and shape variability have been averaged out. The transforms between
poses (rotation angle and translation at each vertebral level) now cleanly estimate
**intervertebral kinematics**.

---

## Why neutral is the reference

Neutral is the natural biomechanical baseline and is typically the least deformed
image geometrically, making registrations from flexion and extension numerically
more stable. It is also the pose against which clinical measurements (Cobb angle,
listhesis) are conventionally reported.

---

## Scope of normalization

Normalization is applied **per vertebra** (C2 through T1). Each vertebra is treated
independently, so the method makes no assumption about the relative positions of
adjacent vertebrae — those relative positions are precisely what the kinematic
analysis measures.