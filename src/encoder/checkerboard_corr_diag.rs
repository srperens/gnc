//! Checkerboard spatial-correlation diagnostic for wavelet subbands.
//!
//! Gated behind `GNC_CHECKER_CORR=1` env var. Zero overhead when disabled.
//!
//! **Gate question**: within a single wavelet subband, are adjacent coefficients
//! spatially correlated in magnitude? If so, a checkerboard k-selection scheme
//! (even-indexed positions use a different k from odd-indexed positions, each
//! trained on the statistical context of their neighbors) could reduce coding cost.
//!
//! **Measurement**:
//! For each subband in each tile, split coefficients into horizontally-adjacent
//! pairs: (x, y) paired with (x+1, y) within the same subband row.
//!
//!   corr = Pearson(|a_i|, |b_i|)
//!        = (E[|a||b|] - E[|a|]*E[|b|]) / (std(|a|) * std(|b|))
//!
//! Aggregated per-subband-group over all tiles and both halves (left=even columns,
//! right=odd columns) simultaneously.
//!
//! **Gate criterion**:
//!   |corr| > 0.30 for ≥3 subbands → PROCEED
//!   |corr| < 0.20 for most subbands → CLOSE

use super::rans::compute_subband_group;

/// Per-subband accumulator for horizontal-pair Pearson correlation.
#[derive(Default, Clone)]
struct SubbandCorrStats {
    label: String,
    /// Sum of |a_i| (left element of each pair)
    sum_a: f64,
    /// Sum of |b_i| (right element of each pair)
    sum_b: f64,
    /// Sum of |a_i|²
    sum_a2: f64,
    /// Sum of |b_i|²
    sum_b2: f64,
    /// Sum of |a_i| * |b_i|
    sum_ab: f64,
    /// Number of pairs
    n: u64,
}

impl SubbandCorrStats {
    /// Pearson correlation coefficient r(|a|, |b|).
    /// Returns None if fewer than 2 pairs or zero variance.
    fn pearson(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let n = self.n as f64;
        let mean_a = self.sum_a / n;
        let mean_b = self.sum_b / n;
        let cov = self.sum_ab / n - mean_a * mean_b;
        let var_a = (self.sum_a2 / n - mean_a * mean_a).max(0.0);
        let var_b = (self.sum_b2 / n - mean_b * mean_b).max(0.0);
        let denom = var_a.sqrt() * var_b.sqrt();
        if denom < 1e-10 {
            return None; // near-constant subband (all zeros or uniform)
        }
        Some(cov / denom)
    }
}

/// Subband group label for display.
fn group_label(group: usize, num_levels: u32) -> String {
    match group {
        0 => "LL".to_string(),
        1 => "finest(merged)".to_string(),
        g => {
            let lfd = (g - 2) / 2 + 1;
            let level = (num_levels as usize).saturating_sub(1 + lfd);
            let is_hh = (g - 2) % 2 == 1;
            if is_hh {
                format!("HH(L{})", level)
            } else {
                format!("LH+HL(L{})", level)
            }
        }
    }
}

/// Accumulate horizontal-neighbor pairs from a single tile.
///
/// For each row, iterates all columns in steps of 2, forming pairs (lx, lx+1).
/// Both elements must fall in the same subband group; if the subband boundary
/// falls on an odd column the trailing element is skipped.
fn analyze_tile(
    coeffs: &[f32],
    tile_size: u32,
    num_levels: u32,
    stats: &mut [SubbandCorrStats],
) {
    let ts = tile_size as usize;

    for ly in 0..tile_size {
        let mut lx = 0u32;
        while lx + 1 < tile_size {
            let g_a = compute_subband_group(lx, ly, tile_size, num_levels);
            let g_b = compute_subband_group(lx + 1, ly, tile_size, num_levels);

            if g_a == g_b && g_a < stats.len() {
                // Both elements in the same subband — valid horizontal pair.
                let a = coeffs[ly as usize * ts + lx as usize].abs() as f64;
                let b = coeffs[ly as usize * ts + (lx + 1) as usize].abs() as f64;
                let st = &mut stats[g_a];
                st.sum_a += a;
                st.sum_b += b;
                st.sum_a2 += a * a;
                st.sum_b2 += b * b;
                st.sum_ab += a * b;
                st.n += 1;
                lx += 2; // non-overlapping pair: advance past both
            } else {
                // Subband boundary between lx and lx+1 — skip this column.
                lx += 1;
            }
        }
    }
}

/// Analyze a full quantized plane.
fn analyze_plane(
    plane: &[f32],
    padded_w: u32,
    padded_h: u32,
    tile_size: u32,
    num_levels: u32,
) -> Vec<SubbandCorrStats> {
    let num_groups = (num_levels as usize * 2).max(2);
    let mut stats: Vec<SubbandCorrStats> = (0..num_groups)
        .map(|g| SubbandCorrStats {
            label: group_label(g, num_levels),
            ..Default::default()
        })
        .collect();

    let tiles_x = (padded_w / tile_size) as usize;
    let tiles_y = (padded_h / tile_size) as usize;
    let ts = tile_size as usize;

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let mut tile_coeffs = vec![0.0f32; ts * ts];
            for row in 0..ts {
                let src_start = (ty * ts + row) * padded_w as usize + tx * ts;
                let dst_start = row * ts;
                tile_coeffs[dst_start..dst_start + ts]
                    .copy_from_slice(&plane[src_start..src_start + ts]);
            }
            analyze_tile(&tile_coeffs, tile_size, num_levels, &mut stats);
        }
    }

    stats
}

/// Print the correlation table to stderr.
fn print_table(plane_label: &str, stats: &[SubbandCorrStats]) {
    eprintln!();
    eprintln!("=== Checkerboard Correlation Diagnostic ({plane_label}) ===");
    eprintln!(
        "{:<18} | {:>10} | {:>8} | {:>8}",
        "Subband", "n_pairs", "corr(|a|,|b|)", "verdict"
    );
    eprintln!("{}", "-".repeat(60));

    let mut n_pass = 0usize;
    let mut n_subbands = 0usize;

    for st in stats {
        if st.n == 0 {
            continue;
        }
        n_subbands += 1;
        match st.pearson() {
            None => {
                eprintln!("{:<18} | {:>10} |           N/A | skip", st.label, st.n);
            }
            Some(r) => {
                let verdict = if r.abs() > 0.30 {
                    n_pass += 1;
                    "PASS"
                } else if r.abs() > 0.20 {
                    "WEAK"
                } else {
                    "FAIL"
                };
                eprintln!(
                    "{:<18} | {:>10} | {:>13.4} | {}",
                    st.label, st.n, r, verdict
                );
            }
        }
    }

    eprintln!("{}", "-".repeat(60));
    eprintln!(
        "  Gate: |corr|>0.30 for ≥3 subbands → PROCEED. Got {}/{} subbands passing.",
        n_pass, n_subbands
    );
    if n_pass >= 3 {
        eprintln!("  *** GATE PASSED *** checkerboard k-context is worth implementing.");
    } else {
        eprintln!("  *** GATE FAILED *** spatial correlation too weak; EMA already captures variance.");
    }
}

/// Run the checkerboard-correlation diagnostic on quantized coefficient planes.
///
/// Called from `pipeline.rs` after quantization, before entropy coding, on the
/// first I-frame only. Gated behind `GNC_CHECKER_CORR=1`.
#[allow(clippy::too_many_arguments)]
pub fn run_multi_plane(
    ctx: &crate::GpuContext,
    y_buf: &wgpu::Buffer,
    co_buf: &wgpu::Buffer,
    cg_buf: &wgpu::Buffer,
    padded_w: u32,
    padded_h: u32,
    chroma_w: u32,
    chroma_h: u32,
    tile_size: u32,
    num_levels: u32,
) {
    eprintln!("[checker_corr] diagnostic active (GNC_CHECKER_CORR)");

    let luma_pixels = (padded_w * padded_h) as usize;
    let chroma_px = (chroma_w * chroma_h) as usize;

    let y_plane = crate::gpu_util::read_buffer_f32(ctx, y_buf, luma_pixels);
    let co_plane = crate::gpu_util::read_buffer_f32(ctx, co_buf, chroma_px);
    let cg_plane = crate::gpu_util::read_buffer_f32(ctx, cg_buf, chroma_px);

    let y_stats = analyze_plane(&y_plane, padded_w, padded_h, tile_size, num_levels);
    print_table("Y (I-frame)", &y_stats);

    let co_stats = analyze_plane(&co_plane, chroma_w, chroma_h, tile_size, num_levels);
    print_table("Co (I-frame)", &co_stats);

    let cg_stats = analyze_plane(&cg_plane, chroma_w, chroma_h, tile_size, num_levels);
    print_table("Cg (I-frame)", &cg_stats);
}
