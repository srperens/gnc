//! BD-rate (Bjontegaard Delta rate) computation.
//!
//! Computes the average bitrate difference between two codecs at matched quality,
//! using piecewise cubic interpolation of RD curves.

/// A single rate-distortion point.
#[derive(Debug, Clone, Copy)]
pub struct RdPoint {
    pub bpp: f64,
    pub psnr: f64,
}

/// Compute BD-rate between two RD curves.
///
/// Returns the percentage bitrate difference of `test` vs `reference`.
/// Negative = test is better (fewer bits at same quality).
/// Positive = test is worse.
///
/// Both curves must have at least 4 points, sorted by PSNR ascending.
pub fn bd_rate(reference: &[RdPoint], test: &[RdPoint]) -> Option<f64> {
    if reference.len() < 4 || test.len() < 4 {
        return None;
    }

    let mut ref_sorted: Vec<RdPoint> = reference.to_vec();
    let mut test_sorted: Vec<RdPoint> = test.to_vec();
    ref_sorted.sort_by(|a, b| a.psnr.partial_cmp(&b.psnr).unwrap());
    test_sorted.sort_by(|a, b| a.psnr.partial_cmp(&b.psnr).unwrap());

    // Use log(bpp) for integration (standard BD-rate practice)
    let ref_log: Vec<(f64, f64)> = ref_sorted
        .iter()
        .filter(|p| p.bpp > 0.0)
        .map(|p| (p.psnr, p.bpp.ln()))
        .collect();
    let test_log: Vec<(f64, f64)> = test_sorted
        .iter()
        .filter(|p| p.bpp > 0.0)
        .map(|p| (p.psnr, p.bpp.ln()))
        .collect();

    if ref_log.len() < 4 || test_log.len() < 4 {
        return None;
    }

    // Overlapping PSNR range
    let psnr_min = ref_log[0].0.max(test_log[0].0);
    let psnr_max = ref_log[ref_log.len() - 1]
        .0
        .min(test_log[test_log.len() - 1].0);

    if psnr_min >= psnr_max {
        return None;
    }

    // Integrate log(bpp) over overlapping PSNR range using piecewise cubic
    let ref_integral = integrate_piecewise_cubic(&ref_log, psnr_min, psnr_max);
    let test_integral = integrate_piecewise_cubic(&test_log, psnr_min, psnr_max);

    let avg_diff = (test_integral - ref_integral) / (psnr_max - psnr_min);

    // Convert from log-domain difference to percentage
    Some((avg_diff.exp() - 1.0) * 100.0)
}

/// Compute BD-PSNR between two RD curves.
///
/// Returns the average PSNR difference of `test` vs `reference` at matched bitrate.
/// Positive = test is better (higher PSNR at same bitrate).
pub fn bd_psnr(reference: &[RdPoint], test: &[RdPoint]) -> Option<f64> {
    if reference.len() < 4 || test.len() < 4 {
        return None;
    }

    let mut ref_sorted: Vec<RdPoint> = reference.to_vec();
    let mut test_sorted: Vec<RdPoint> = test.to_vec();
    ref_sorted.sort_by(|a, b| a.bpp.partial_cmp(&b.bpp).unwrap());
    test_sorted.sort_by(|a, b| a.bpp.partial_cmp(&b.bpp).unwrap());

    // Use log(bpp) as x-axis
    let ref_pts: Vec<(f64, f64)> = ref_sorted
        .iter()
        .filter(|p| p.bpp > 0.0)
        .map(|p| (p.bpp.ln(), p.psnr))
        .collect();
    let test_pts: Vec<(f64, f64)> = test_sorted
        .iter()
        .filter(|p| p.bpp > 0.0)
        .map(|p| (p.bpp.ln(), p.psnr))
        .collect();

    if ref_pts.len() < 4 || test_pts.len() < 4 {
        return None;
    }

    let bpp_min = ref_pts[0].0.max(test_pts[0].0);
    let bpp_max = ref_pts[ref_pts.len() - 1]
        .0
        .min(test_pts[test_pts.len() - 1].0);

    if bpp_min >= bpp_max {
        return None;
    }

    let ref_integral = integrate_piecewise_cubic(&ref_pts, bpp_min, bpp_max);
    let test_integral = integrate_piecewise_cubic(&test_pts, bpp_min, bpp_max);

    Some((test_integral - ref_integral) / (bpp_max - bpp_min))
}

/// Piecewise cubic interpolation and integration over [a, b].
///
/// Points are (x, y) sorted by x. Uses natural cubic spline.
fn integrate_piecewise_cubic(points: &[(f64, f64)], a: f64, b: f64) -> f64 {
    let n = points.len();
    if n < 4 {
        // Fallback to trapezoidal if too few points
        return integrate_trapezoidal(points, a, b);
    }

    // Compute cubic spline coefficients
    let coeffs = cubic_spline_coefficients(points);

    // Integrate each segment that overlaps [a, b]
    let mut integral = 0.0;
    for i in 0..n - 1 {
        let x0 = points[i].0;
        let x1 = points[i + 1].0;

        // Clip to [a, b]
        let seg_start = x0.max(a);
        let seg_end = x1.min(b);
        if seg_start >= seg_end {
            continue;
        }

        let (ai, bi, ci, di) = coeffs[i];
        // Polynomial: y(t) = ai + bi*(x-x0) + ci*(x-x0)^2 + di*(x-x0)^3
        // Integrate from seg_start to seg_end
        let t0 = seg_start - x0;
        let t1 = seg_end - x0;

        let antideriv = |t: f64| -> f64 {
            ai * t + bi * t * t / 2.0 + ci * t * t * t / 3.0 + di * t * t * t * t / 4.0
        };

        integral += antideriv(t1) - antideriv(t0);
    }

    integral
}

/// Compute natural cubic spline coefficients.
/// Returns (a, b, c, d) for each segment where y = a + b(x-xi) + c(x-xi)^2 + d(x-xi)^3.
fn cubic_spline_coefficients(points: &[(f64, f64)]) -> Vec<(f64, f64, f64, f64)> {
    let n = points.len();
    if n < 2 {
        return vec![];
    }

    let mut h = vec![0.0; n - 1];
    let mut alpha = vec![0.0; n];

    for i in 0..n - 1 {
        h[i] = points[i + 1].0 - points[i].0;
    }

    for i in 1..n - 1 {
        if h[i - 1].abs() < 1e-15 || h[i].abs() < 1e-15 {
            continue;
        }
        alpha[i] = 3.0 / h[i] * (points[i + 1].1 - points[i].1)
            - 3.0 / h[i - 1] * (points[i].1 - points[i - 1].1);
    }

    // Solve tridiagonal system for c coefficients (natural spline: c[0] = c[n-1] = 0)
    let mut c = vec![0.0; n];
    let mut l = vec![1.0; n];
    let mut mu = vec![0.0; n];
    let mut z = vec![0.0; n];

    for i in 1..n - 1 {
        l[i] = 2.0 * (points[i + 1].0 - points[i - 1].0) - h[i - 1] * mu[i - 1];
        if l[i].abs() < 1e-15 {
            continue;
        }
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    for j in (0..n - 1).rev() {
        c[j] = z[j] - mu[j] * c[j + 1];
    }

    let mut coeffs = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let ai = points[i].1;
        let bi = if h[i].abs() < 1e-15 {
            0.0
        } else {
            (points[i + 1].1 - points[i].1) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
        };
        let ci_coeff = c[i];
        let di = if h[i].abs() < 1e-15 {
            0.0
        } else {
            (c[i + 1] - c[i]) / (3.0 * h[i])
        };
        coeffs.push((ai, bi, ci_coeff, di));
    }

    coeffs
}

/// Simple trapezoidal integration fallback.
fn integrate_trapezoidal(points: &[(f64, f64)], a: f64, b: f64) -> f64 {
    let mut integral = 0.0;
    for i in 0..points.len() - 1 {
        let x0 = points[i].0.max(a);
        let x1 = points[i + 1].0.min(b);
        if x0 >= x1 {
            continue;
        }
        // Linear interpolation for y at clipped boundaries
        let t = points[i + 1].0 - points[i].0;
        if t.abs() < 1e-15 {
            continue;
        }
        let y0 = points[i].1 + (points[i + 1].1 - points[i].1) * (x0 - points[i].0) / t;
        let y1 = points[i].1 + (points[i + 1].1 - points[i].1) * (x1 - points[i].0) / t;
        integral += (y0 + y1) / 2.0 * (x1 - x0);
    }
    integral
}

/// Parse an RD curve CSV file (as produced by `gnc rd-curve`).
/// Expects columns: q, qstep, psnr, ssim, bpp, encode_ms, decode_ms
pub fn parse_rd_csv(path: &str) -> Result<Vec<RdPoint>, Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();

    let psnr_idx = headers.iter().position(|h| h == "psnr");
    let bpp_idx = headers.iter().position(|h| h == "bpp");

    let mut points = Vec::new();

    if let (Some(pi), Some(bi)) = (psnr_idx, bpp_idx) {
        for result in reader.records() {
            let record = result?;
            let psnr: f64 = record[pi].parse()?;
            let bpp: f64 = record[bi].parse()?;
            points.push(RdPoint { bpp, psnr });
        }
    }

    Ok(points)
}

/// Parse RD CSV with header-based column lookup (more robust version).
pub fn load_rd_curve(path: &str) -> Result<Vec<RdPoint>, Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();

    let psnr_idx = headers
        .iter()
        .position(|h| h == "psnr")
        .ok_or("CSV missing 'psnr' column")?;
    let bpp_idx = headers
        .iter()
        .position(|h| h == "bpp")
        .ok_or("CSV missing 'bpp' column")?;

    let mut points = Vec::new();
    for result in reader.records() {
        let record = result?;
        let psnr: f64 = record[psnr_idx].parse()?;
        let bpp: f64 = record[bpp_idx].parse()?;
        points.push(RdPoint { bpp, psnr });
    }

    points.sort_by(|a, b| a.psnr.partial_cmp(&b.psnr).unwrap());
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bd_rate_identical_curves() {
        let curve: Vec<RdPoint> = vec![
            RdPoint {
                psnr: 30.0,
                bpp: 0.5,
            },
            RdPoint {
                psnr: 35.0,
                bpp: 1.0,
            },
            RdPoint {
                psnr: 40.0,
                bpp: 2.0,
            },
            RdPoint {
                psnr: 45.0,
                bpp: 4.0,
            },
        ];

        let rate = bd_rate(&curve, &curve).unwrap();
        assert!(
            rate.abs() < 0.1,
            "BD-rate of identical curves should be ~0%, got {rate:.2}%"
        );
    }

    #[test]
    fn test_bd_rate_better_codec() {
        // Reference codec
        let reference = vec![
            RdPoint {
                psnr: 30.0,
                bpp: 1.0,
            },
            RdPoint {
                psnr: 35.0,
                bpp: 2.0,
            },
            RdPoint {
                psnr: 40.0,
                bpp: 4.0,
            },
            RdPoint {
                psnr: 45.0,
                bpp: 8.0,
            },
        ];
        // Test codec uses half the bitrate at every quality
        let test = vec![
            RdPoint {
                psnr: 30.0,
                bpp: 0.5,
            },
            RdPoint {
                psnr: 35.0,
                bpp: 1.0,
            },
            RdPoint {
                psnr: 40.0,
                bpp: 2.0,
            },
            RdPoint {
                psnr: 45.0,
                bpp: 4.0,
            },
        ];

        let rate = bd_rate(&reference, &test).unwrap();
        // Should be approximately -50% (test uses half the bits)
        assert!(
            rate < -40.0 && rate > -60.0,
            "BD-rate should be ~-50%, got {rate:.2}%"
        );
    }

    #[test]
    fn test_bd_psnr_better_codec() {
        let reference = vec![
            RdPoint {
                psnr: 30.0,
                bpp: 0.5,
            },
            RdPoint {
                psnr: 35.0,
                bpp: 1.0,
            },
            RdPoint {
                psnr: 40.0,
                bpp: 2.0,
            },
            RdPoint {
                psnr: 45.0,
                bpp: 4.0,
            },
        ];
        // Test codec has 3 dB better PSNR at every bitrate
        let test = vec![
            RdPoint {
                psnr: 33.0,
                bpp: 0.5,
            },
            RdPoint {
                psnr: 38.0,
                bpp: 1.0,
            },
            RdPoint {
                psnr: 43.0,
                bpp: 2.0,
            },
            RdPoint {
                psnr: 48.0,
                bpp: 4.0,
            },
        ];

        let psnr_diff = bd_psnr(&reference, &test).unwrap();
        assert!(
            psnr_diff > 2.5 && psnr_diff < 3.5,
            "BD-PSNR should be ~3.0 dB, got {psnr_diff:.2} dB"
        );
    }
}
