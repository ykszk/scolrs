use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
};

use anyhow::{bail, Context, Result};
use ndarray::{s, Array2, Array3, Axis};
use scolrs::{
    head_neck::{LateralPoints, LateralPointsLine},
    reg::polygon_pairs::{self, PolygonPair},
};

use crate::cli::{VertebraArgs, VertebraNormalizeArgs, VertebraRegisterArgs, VertebraSubCommands};

#[derive(Debug, Clone, Copy)]
struct PairSpec {
    row: usize,
    include_top: bool,
}

/// Extract the polygon a [`PairSpec`] selects from a corner array.
fn extract_poly(corners: &Array3<f64>, spec: &PairSpec) -> Array2<f64> {
    if spec.include_top {
        corners.index_axis(Axis(0), spec.row).to_owned()
    } else {
        corners.slice(s![spec.row, 2.., ..]).to_owned()
    }
}

/// Write a polygon back into the location a [`PairSpec`] selects.
fn assign_poly(corners: &mut Array3<f64>, spec: &PairSpec, poly: &Array2<f64>) {
    if spec.include_top {
        corners.index_axis_mut(Axis(0), spec.row).assign(poly);
    } else {
        corners.slice_mut(s![spec.row, 2.., ..]).assign(poly);
    }
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

    if moving_corners.dim().1 != 4
        || fixed_corners.dim().1 != 4
        || moving_corners.dim().2 != 2
        || fixed_corners.dim().2 != 2
    {
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
    let mut pairs = Vec::with_capacity(n_rows);
    let mut specs = Vec::with_capacity(n_rows);

    // Pair strategy:
    // - first non-dummy row: (bl, br)
    // - remaining rows: (tl, tr, bl, br)
    for row in 0..n_rows {
        let include_top = row > 0;
        let moving_poly = extract_poly(moving_corners, &PairSpec { row, include_top });
        let fixed_poly = extract_poly(fixed_corners, &PairSpec { row, include_top });

        pairs.push(PolygonPair::new(moving_poly, fixed_poly)?);
        specs.push(PairSpec { row, include_top });
    }

    Ok((pairs, specs))
}

fn cmd_register(args: VertebraRegisterArgs) -> Result<()> {
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
        let moving_poly = extract_poly(&moving.corners.0, spec);
        let reg_poly = polygon_pairs::transform(&moving_poly, &result, pair_index)?;
        assign_poly(&mut fixed.corners.0, spec, &reg_poly);
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

/// Normalize one set of poses. The first line is the reference; all others are
/// registered to it, averaged into a mean shape, and back-projected into each frame.
fn normalize_group(lines: &[LateralPointsLine]) -> Result<Vec<LateralPointsLine>> {
    if lines.len() < 2 {
        let filename = lines.first().map(|l| l.filename.as_str()).unwrap_or("");
        log::warn!(
            "pose set with {} line(s) (first: {filename:?}) has nothing to register; \
             passing it through unchanged",
            lines.len()
        );
        return Ok(lines.to_vec());
    }

    let reference = &lines[0].content;

    // Register each non-reference pose onto the reference. All poses share the same
    // corner layout, so the resulting `PairSpec`s are identical across registrations.
    let mut regs = Vec::with_capacity(lines.len() - 1);
    let mut shared_specs: Option<Vec<PairSpec>> = None;
    for line in &lines[1..] {
        let (pairs, specs) = build_polygon_pairs(&line.content, reference)?;
        let result = polygon_pairs::register(&pairs, 1.0, 50, 1e-8)
            .context("register pose to reference corners")?;
        shared_specs = Some(specs);
        regs.push(result);
    }
    let specs = shared_specs.expect("at least one non-reference pose");

    // Mean shape, computed per polygon in the reference frame: average the reference
    // corners with every other pose mapped into the reference frame.
    let n = lines.len() as f64;
    let mut mean_corners = reference.corners.0.clone();
    for (pair_index, spec) in specs.iter().enumerate() {
        let mut acc = extract_poly(&reference.corners.0, spec);
        for (line, result) in lines[1..].iter().zip(regs.iter()) {
            let moving_poly = extract_poly(&line.content.corners.0, spec);
            let mapped = polygon_pairs::transform(&moving_poly, result, pair_index)?;
            acc = acc + mapped;
        }
        acc.mapv_inplace(|v| v / n);
        assign_poly(&mut mean_corners, spec, &acc);
    }

    let mut out = Vec::with_capacity(lines.len());

    // The reference pose keeps the mean shape as-is (it lives in the reference frame).
    let mut reference_line = lines[0].clone();
    reference_line.content.corners.0 = mean_corners.clone();
    out.push(reference_line);

    // Each other pose receives the mean shape back-projected into its own frame.
    for (line, result) in lines[1..].iter().zip(regs.iter()) {
        let mut out_line = line.clone();
        for (pair_index, spec) in specs.iter().enumerate() {
            let mean_poly = extract_poly(&mean_corners, spec);
            let back = polygon_pairs::transform_inverse(&mean_poly, result, pair_index)?;
            assign_poly(&mut out_line.content.corners.0, spec, &back);
        }
        out.push(out_line);
    }

    Ok(out)
}

fn cmd_normalize(args: VertebraNormalizeArgs) -> Result<()> {
    let reader: Box<dyn BufRead> = if args.input.as_os_str() == "-" {
        Box::new(BufReader::new(io::stdin()))
    } else {
        Box::new(BufReader::new(
            File::open(&args.input).with_context(|| format!("open {:?}", args.input))?,
        ))
    };

    // Each line is one pose. Blank lines delimit independent pose sets (sub-ndjson).
    let mut groups: Vec<Vec<LateralPointsLine>> = Vec::new();
    let mut current: Vec<LateralPointsLine> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            if !current.is_empty() {
                groups.push(std::mem::take(&mut current));
            }
            continue;
        }
        let parsed = serde_json::from_str(&line).context("parse line as LateralPointsLine")?;
        current.push(parsed);
    }
    if !current.is_empty() {
        groups.push(current);
    }
    if groups.is_empty() {
        bail!("no input poses");
    }
    log::info!(
        "Read {} pose sets ({} lines total)",
        groups.len(),
        groups.iter().map(|g| g.len()).sum::<usize>()
    );

    let mut writer: Box<dyn Write> = if let Some(output) = args.output.as_ref() {
        Box::new(BufWriter::new(
            File::create(output).with_context(|| format!("create output {output:?}"))?,
        ))
    } else {
        Box::new(io::stdout())
    };

    // Normalize each set independently, preserving the blank-line delimiters on output.
    for (group_index, group) in groups.iter().enumerate() {
        if group_index > 0 {
            writeln!(writer)?;
        }
        let normalized =
            normalize_group(group).with_context(|| format!("normalize pose set {group_index}"))?;
        for line in &normalized {
            writeln!(writer, "{}", serde_json::to_string(line)?)?;
        }
    }

    Ok(())
}

pub fn cmd(args: VertebraArgs) -> Result<()> {
    match args.command {
        VertebraSubCommands::Register(args) => cmd_register(args),
        VertebraSubCommands::Normalize(args) => cmd_normalize(args),
    }
}
