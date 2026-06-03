use std::{
    fs::File,
    io::{self, BufWriter, Read, Write},
};

use anyhow::{bail, Context, Result};
use scolrs::{
    head_neck::LateralPoints,
    reg::polygon_pairs::{self, PolygonPair},
};

use crate::cli::RegisterArgs;

#[derive(Debug, Clone, Copy)]
struct PairSpec {
    top_row: usize,
    include_top: bool,
}

fn read_input(path: &std::path::Path, field_name: &str) -> Result<String> {
    if path.as_os_str() == "-" {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .with_context(|| format!("read {field_name} from stdin"))?;
        Ok(buf)
    } else {
        std::fs::read_to_string(path).with_context(|| format!("read {field_name} from {path:?}"))
    }
}

fn build_polygon_pairs(
    moving: &LateralPoints,
    fixed: &LateralPoints,
) -> Result<(Vec<PolygonPair>, Vec<PairSpec>)> {
    let moving_corners = &moving.corners.0;
    let fixed_corners = &fixed.corners.0;

    if moving_corners.dim().1 != 4 || fixed_corners.dim().1 != 4 {
        bail!("Expected corners with shape (N, 4, 2)");
    }
    if moving_corners.dim() != fixed_corners.dim() {
        bail!(
            "Moving/fixed corners shape mismatch: moving={:?}, fixed={:?}",
            moving_corners.dim(),
            fixed_corners.dim()
        );
    }
    if moving_corners.dim().0 < 2 {
        bail!("Need at least 2 corner rows (including dummy TL/TR row)");
    }

    let n_rows = moving_corners.dim().0;
    let mut pairs = Vec::with_capacity(n_rows - 1);
    let mut specs = Vec::with_capacity(n_rows - 1);

    // Pair strategy:
    // - first non-dummy row: (bl, br)
    // - remaining rows: (tl, tr, bl, br)
    for top_row in 0..n_rows {
        let include_top = top_row > 0;
        let moving_poly = if include_top {
            ndarray::arr2(&[
                [
                    moving_corners[[top_row, 0, 0]],
                    moving_corners[[top_row, 0, 1]],
                ],
                [
                    moving_corners[[top_row, 1, 0]],
                    moving_corners[[top_row, 1, 1]],
                ],
                [
                    moving_corners[[top_row, 2, 0]],
                    moving_corners[[top_row, 2, 1]],
                ],
                [
                    moving_corners[[top_row, 3, 0]],
                    moving_corners[[top_row, 3, 1]],
                ],
            ])
        } else {
            ndarray::arr2(&[
                [
                    moving_corners[[top_row, 2, 0]],
                    moving_corners[[top_row, 2, 1]],
                ],
                [
                    moving_corners[[top_row, 3, 0]],
                    moving_corners[[top_row, 3, 1]],
                ],
            ])
        };
        let fixed_poly = if include_top {
            ndarray::arr2(&[
                [
                    fixed_corners[[top_row, 0, 0]],
                    fixed_corners[[top_row, 0, 1]],
                ],
                [
                    fixed_corners[[top_row, 1, 0]],
                    fixed_corners[[top_row, 1, 1]],
                ],
                [
                    fixed_corners[[top_row, 2, 0]],
                    fixed_corners[[top_row, 2, 1]],
                ],
                [
                    fixed_corners[[top_row, 3, 0]],
                    fixed_corners[[top_row, 3, 1]],
                ],
            ])
        } else {
            ndarray::arr2(&[
                [
                    fixed_corners[[top_row, 2, 0]],
                    fixed_corners[[top_row, 2, 1]],
                ],
                [
                    fixed_corners[[top_row, 3, 0]],
                    fixed_corners[[top_row, 3, 1]],
                ],
            ])
        };

        pairs.push(PolygonPair::new(moving_poly, fixed_poly)?);
        specs.push(PairSpec {
            top_row,
            include_top,
        });
    }

    Ok((pairs, specs))
}

pub fn cmd(args: RegisterArgs) -> Result<()> {
    if args.moving.as_os_str() == "-" && args.fixed.as_os_str() == "-" {
        bail!("Both moving and fixed cannot be '-' because stdin can only be read once");
    }

    let moving_str = read_input(&args.moving, "moving")?;
    let fixed_str = read_input(&args.fixed, "fixed")?;

    let moving: LateralPoints =
        serde_json::from_str(&moving_str).context("parse moving as LateralPoints")?;
    let mut fixed: LateralPoints =
        serde_json::from_str(&fixed_str).context("parse fixed as LateralPoints")?;

    let (pairs, specs) = build_polygon_pairs(&moving, &fixed)?;
    let result = polygon_pairs::register(&pairs, 1.0, 50, 1e-8)
        .context("register moving to fixed corners")?;

    log::debug!("Registration result: {:?}", result);

    // Apply each per-level transform back to the moving corners.
    for (pair_index, spec) in specs.iter().enumerate() {
        let top_row = spec.top_row;
        let moving_poly = if spec.include_top {
            ndarray::arr2(&[
                [
                    moving.corners.0[[top_row, 0, 0]],
                    moving.corners.0[[top_row, 0, 1]],
                ],
                [
                    moving.corners.0[[top_row, 1, 0]],
                    moving.corners.0[[top_row, 1, 1]],
                ],
                [
                    moving.corners.0[[top_row, 2, 0]],
                    moving.corners.0[[top_row, 2, 1]],
                ],
                [
                    moving.corners.0[[top_row, 3, 0]],
                    moving.corners.0[[top_row, 3, 1]],
                ],
            ])
        } else {
            ndarray::arr2(&[
                [
                    moving.corners.0[[top_row, 2, 0]],
                    moving.corners.0[[top_row, 2, 1]],
                ],
                [
                    moving.corners.0[[top_row, 3, 0]],
                    moving.corners.0[[top_row, 3, 1]],
                ],
            ])
        };
        let reg_poly = polygon_pairs::transform(&moving_poly, &result, pair_index)?;

        if spec.include_top {
            fixed.corners.0[[top_row, 0, 0]] = reg_poly[[0, 0]];
            fixed.corners.0[[top_row, 0, 1]] = reg_poly[[0, 1]];
            fixed.corners.0[[top_row, 1, 0]] = reg_poly[[1, 0]];
            fixed.corners.0[[top_row, 1, 1]] = reg_poly[[1, 1]];
            fixed.corners.0[[top_row, 2, 0]] = reg_poly[[2, 0]];
            fixed.corners.0[[top_row, 2, 1]] = reg_poly[[2, 1]];
            fixed.corners.0[[top_row, 3, 0]] = reg_poly[[3, 0]];
            fixed.corners.0[[top_row, 3, 1]] = reg_poly[[3, 1]];
        } else {
            fixed.corners.0[[top_row, 2, 0]] = reg_poly[[0, 0]];
            fixed.corners.0[[top_row, 2, 1]] = reg_poly[[0, 1]];
            fixed.corners.0[[top_row, 3, 0]] = reg_poly[[1, 0]];
            fixed.corners.0[[top_row, 3, 1]] = reg_poly[[1, 1]];
        }
    }

    let writer: Box<dyn Write> = if args.output.as_os_str() == "-" {
        Box::new(io::stdout())
    } else {
        Box::new(
            File::create(&args.output)
                .with_context(|| format!("create output {:?}", args.output))?,
        )
    };
    let mut writer = BufWriter::new(writer);
    serde_json::to_writer_pretty(&mut writer, &fixed)?;
    writeln!(&mut writer)?;

    Ok(())
}
