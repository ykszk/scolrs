use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct PolygonPair {
    pub p: Array2<f64>,
    pub q: Array2<f64>,
}

impl PolygonPair {
    pub fn new(p: Array2<f64>, q: Array2<f64>) -> Result<Self, PolygonPairError> {
        if p.ndim() != 2 || q.ndim() != 2 {
            return Err(PolygonPairError::InvalidShape(
                "P and Q must be rank-2 arrays".to_string(),
            ));
        }
        if p.ncols() != 2 || q.ncols() != 2 {
            return Err(PolygonPairError::InvalidShape(
                "P and Q must both have shape (N_k, 2)".to_string(),
            ));
        }
        if p.raw_dim() != q.raw_dim() {
            return Err(PolygonPairError::InvalidShape(
                "P and Q must have the same shape".to_string(),
            ));
        }
        Ok(Self { p, q })
    }
}

#[derive(Debug, Clone)]
pub struct PairResult {
    pub theta: f64,
    pub r: [[f64; 2]; 2],
    pub t: [f64; 2],
}

#[derive(Debug, Clone)]
pub struct RegistrationResult {
    pub s: f64,
    pub pairs: Vec<PairResult>,
    pub n_iter: usize,
    pub objective_history: Vec<f64>,
}

#[derive(Debug, thiserror::Error)]
pub enum PolygonPairError {
    #[error("Shape error: {0}")]
    InvalidShape(String),
    #[error("Value error: {0}")]
    Value(String),
    #[error("Runtime error: {0}")]
    Runtime(String),
}

#[derive(Debug, Clone, Copy)]
struct PairCache {
    p_bar: [f64; 2],
    q_bar: [f64; 2],
    h: [[f64; 2]; 2],
    sigma: f64,
}

#[inline]
fn mat_vec_mul(m: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] {
    [
        m[0][0] * v[0] + m[0][1] * v[1],
        m[1][0] * v[0] + m[1][1] * v[1],
    ]
}

#[inline]
fn row_times_r_t(x: f64, y: f64, r: [[f64; 2]; 2]) -> [f64; 2] {
    [x * r[0][0] + y * r[0][1], x * r[1][0] + y * r[1][1]]
}

fn objective(pairs: &[PolygonPair], s: f64, rs: &[[[f64; 2]; 2]], ts: &[[f64; 2]]) -> f64 {
    let mut total = 0.0;
    for (idx, pair) in pairs.iter().enumerate() {
        let r = rs[idx];
        let t = ts[idx];
        for i in 0..pair.p.nrows() {
            let x = pair.p[[i, 0]];
            let y = pair.p[[i, 1]];
            let mapped = row_times_r_t(x, y, r);
            let dx = s * mapped[0] + t[0] - pair.q[[i, 0]];
            let dy = s * mapped[1] + t[1] - pair.q[[i, 1]];
            total += dx * dx + dy * dy;
        }
    }
    total
}

fn precompute(pairs: &[PolygonPair]) -> Vec<PairCache> {
    let mut cache = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let n = pair.p.nrows() as f64;

        let p_bar = [
            pair.p.column(0).iter().copied().sum::<f64>() / n,
            pair.p.column(1).iter().copied().sum::<f64>() / n,
        ];
        let q_bar = [
            pair.q.column(0).iter().copied().sum::<f64>() / n,
            pair.q.column(1).iter().copied().sum::<f64>() / n,
        ];

        let mut h = [[0.0; 2]; 2];
        let mut sigma = 0.0;

        for i in 0..pair.p.nrows() {
            let px = pair.p[[i, 0]] - p_bar[0];
            let py = pair.p[[i, 1]] - p_bar[1];
            let qx = pair.q[[i, 0]] - q_bar[0];
            let qy = pair.q[[i, 1]] - q_bar[1];

            h[0][0] += px * qx;
            h[0][1] += px * qy;
            h[1][0] += py * qx;
            h[1][1] += py * qy;

            sigma += px * px + py * py;
        }

        cache.push(PairCache {
            p_bar,
            q_bar,
            h,
            sigma,
        });
    }

    cache
}

fn step_a(cache: &[PairCache], s: f64) -> (Vec<[[f64; 2]; 2]>, Vec<[f64; 2]>) {
    let mut rs = Vec::with_capacity(cache.len());
    let mut ts = Vec::with_capacity(cache.len());

    for c in cache {
        // In 2D, maximizing tr(HR) over rotations yields theta from these two scalars.
        let a = c.h[0][0] + c.h[1][1];
        let b = c.h[0][1] - c.h[1][0];
        let theta = b.atan2(a);
        let ct = theta.cos();
        let st = theta.sin();
        let r = [[ct, -st], [st, ct]];
        let rp = mat_vec_mul(r, c.p_bar);
        let t = [c.q_bar[0] - s * rp[0], c.q_bar[1] - s * rp[1]];

        rs.push(r);
        ts.push(t);
    }

    (rs, ts)
}

fn step_b(cache: &[PairCache], rs: &[[[f64; 2]; 2]]) -> Result<f64, PolygonPairError> {
    let numerator = cache
        .iter()
        .zip(rs.iter())
        .map(|(c, r)| {
            c.h[0][0] * r[0][0] + c.h[0][1] * r[1][0] + c.h[1][0] * r[0][1] + c.h[1][1] * r[1][1]
        })
        .sum::<f64>();

    let denominator = cache.iter().map(|c| c.sigma).sum::<f64>();
    if denominator == 0.0 {
        return Err(PolygonPairError::Value(
            "All source polygons are degenerate (zero variance).".to_string(),
        ));
    }

    let s = numerator / denominator;
    if s <= 0.0 {
        return Err(PolygonPairError::Runtime(format!(
            "Scale update produced s = {s:.4} <= 0. Check polygon orientation consistency."
        )));
    }

    Ok(s)
}

pub fn register(
    pairs: &[PolygonPair],
    s_init: f64,
    max_iter: usize,
    tol: f64,
    verbose: bool,
) -> Result<RegistrationResult, PolygonPairError> {
    if pairs.is_empty() {
        return Err(PolygonPairError::Value(
            "Need at least one polygon pair.".to_string(),
        ));
    }

    let n_tot = pairs.iter().map(|p| p.p.nrows()).sum::<usize>();
    if n_tot < 12 {
        return Err(PolygonPairError::Value(format!(
            "Need N_tot >= 12 point correspondences for an overdetermined system (got {n_tot})."
        )));
    }

    let cache = precompute(pairs);

    let mut s = s_init;
    let (mut rs, mut ts) = step_a(&cache, s);
    let mut e_prev = objective(pairs, s, &rs, &ts);
    let mut history = vec![e_prev];

    if verbose {
        println!("  iter  0  |  E = {e_prev:.6e}  |  s = {s:.6}");
    }

    let mut n_iter = 0usize;
    for iteration in 1..=max_iter {
        let updated = step_a(&cache, s);
        rs = updated.0;
        ts = updated.1;

        s = step_b(&cache, &rs)?;

        let e = objective(pairs, s, &rs, &ts);
        history.push(e);

        if verbose {
            println!("  iter {iteration:2}  |  E = {e:.6e}  |  s = {s:.6}");
        }

        let rel_change = (e - e_prev).abs() / (e_prev.abs() + 1e-300);
        n_iter = iteration;
        if rel_change < tol {
            if verbose {
                println!(
                    "  Converged after {iteration} iterations (rel delta E = {rel_change:.2e})"
                );
            }
            break;
        }

        e_prev = e;
    }

    if max_iter > 0 && n_iter == max_iter && verbose {
        println!("  Reached max_iter={max_iter} without convergence.");
    }

    let pair_results = rs
        .iter()
        .zip(ts.iter())
        .map(|(r, t)| PairResult {
            theta: r[1][0].atan2(r[0][0]),
            r: *r,
            t: *t,
        })
        .collect::<Vec<_>>();

    Ok(RegistrationResult {
        s,
        pairs: pair_results,
        n_iter,
        objective_history: history,
    })
}

pub fn transform(
    p: &Array2<f64>,
    result: &RegistrationResult,
    pair_index: usize,
) -> Result<Array2<f64>, PolygonPairError> {
    if p.ndim() != 2 || p.ncols() != 2 {
        return Err(PolygonPairError::InvalidShape(
            "P must have shape (N, 2).".to_string(),
        ));
    }

    let pr = result.pairs.get(pair_index).ok_or_else(|| {
        PolygonPairError::Value(format!("pair_index out of bounds: {pair_index}"))
    })?;

    let mut out = Array2::<f64>::zeros((p.nrows(), 2));
    for i in 0..p.nrows() {
        let x = p[[i, 0]];
        let y = p[[i, 1]];
        let mapped = row_times_r_t(x, y, pr.r);
        out[[i, 0]] = result.s * mapped[0] + pr.t[0];
        out[[i, 1]] = result.s * mapped[1] + pr.t[1];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn rot(theta: f64) -> [[f64; 2]; 2] {
        let c = theta.cos();
        let s = theta.sin();
        [[c, -s], [s, c]]
    }

    fn apply_known(p: &Array2<f64>, s: f64, r: [[f64; 2]; 2], t: [f64; 2]) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((p.nrows(), 2));
        for i in 0..p.nrows() {
            let x = p[[i, 0]];
            let y = p[[i, 1]];
            let mapped = [x * r[0][0] + y * r[0][1], x * r[1][0] + y * r[1][1]];
            out[[i, 0]] = s * mapped[0] + t[0];
            out[[i, 1]] = s * mapped[1] + t[1];
        }
        out
    }

    fn angle_diff(a: f64, b: f64) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut d = (a - b).abs();
        while d > two_pi {
            d -= two_pi;
        }
        d.min(two_pi - d)
    }

    #[test]
    fn register_recovers_noiseless_pairs() {
        let s_true = 1.35;

        let p1 = array![
            [-1.0, 0.0],
            [0.0, 0.5],
            [1.0, 0.0],
            [0.5, -0.8],
            [-0.6, -0.7],
            [0.2, 1.1],
        ];
        let p2 = array![
            [-2.0, -1.0],
            [-1.0, 1.2],
            [1.5, 1.0],
            [2.0, -0.3],
            [0.0, -1.4],
            [-1.6, 0.2],
        ];

        let th1 = 0.31;
        let th2 = -0.87;
        let t1 = [2.0, -1.5];
        let t2 = [-0.75, 3.2];

        let q1 = apply_known(&p1, s_true, rot(th1), t1);
        let q2 = apply_known(&p2, s_true, rot(th2), t2);

        let pairs = vec![
            PolygonPair::new(p1, q1).expect("valid pair 1"),
            PolygonPair::new(p2, q2).expect("valid pair 2"),
        ];

        let res = register(&pairs, 1.0, 100, 1e-12, false).expect("registration succeeds");

        assert!((res.s - s_true).abs() < 1e-10);
        assert_eq!(res.pairs.len(), 2);
        assert!(angle_diff(res.pairs[0].theta, th1) < 1e-10);
        assert!(angle_diff(res.pairs[1].theta, th2) < 1e-10);
        assert!((res.pairs[0].t[0] - t1[0]).abs() < 1e-10);
        assert!((res.pairs[0].t[1] - t1[1]).abs() < 1e-10);
        assert!((res.pairs[1].t[0] - t2[0]).abs() < 1e-10);
        assert!((res.pairs[1].t[1] - t2[1]).abs() < 1e-10);
    }

    #[test]
    fn transform_matches_known_targets() {
        let p = array![
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.5],
            [2.0, -0.25],
            [-0.5, -1.5],
            [1.5, 1.25],
            [0.25, -0.75],
            [-1.2, 0.4],
            [0.9, -1.1],
            [1.1, 0.8],
            [-0.7, 1.3],
            [0.4, 0.2],
        ];
        let s_true = 1.2;
        let th = 0.42;
        let t = [3.0, -2.0];
        let q = apply_known(&p, s_true, rot(th), t);

        let pairs = vec![PolygonPair::new(p.clone(), q.clone()).expect("valid pair")];
        let res = register(&pairs, 0.9, 100, 1e-12, false).expect("registration succeeds");
        let mapped = transform(&p, &res, 0).expect("transform succeeds");

        for i in 0..mapped.nrows() {
            assert!((mapped[[i, 0]] - q[[i, 0]]).abs() < 1e-10);
            assert!((mapped[[i, 1]] - q[[i, 1]]).abs() < 1e-10);
        }
    }

    #[test]
    fn register_errors_when_total_points_too_small() {
        let p1 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let q1 = p1.clone();
        let p2 = array![[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]];
        let q2 = p2.clone();

        let pairs = vec![
            PolygonPair::new(p1, q1).expect("valid pair 1"),
            PolygonPair::new(p2, q2).expect("valid pair 2"),
        ];

        let err = register(&pairs, 1.0, 10, 1e-8, false).expect_err("must fail for N_tot < 12");
        match err {
            PolygonPairError::Value(msg) => assert!(msg.contains("N_tot >= 12")),
            other => panic!("unexpected error type: {other:?}"),
        }
    }

    #[test]
    fn register_errors_when_source_is_degenerate() {
        let p = array![
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ];
        let q = array![
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.4],
            [0.3, 0.6],
            [0.4, 0.8],
            [0.5, 1.0],
            [0.6, 1.2],
            [0.7, 1.4],
            [0.8, 1.6],
            [0.9, 1.8],
            [1.0, 2.0],
            [1.1, 2.2],
        ];

        let pairs = vec![PolygonPair::new(p, q).expect("valid pair")];
        let err =
            register(&pairs, 1.0, 10, 1e-8, false).expect_err("must fail for zero source variance");
        match err {
            PolygonPairError::Value(msg) => assert!(msg.contains("degenerate")),
            other => panic!("unexpected error type: {other:?}"),
        }
    }
}
