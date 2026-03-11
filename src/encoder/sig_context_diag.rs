//! Significance-map context diagnostic for backlog item #53.
//!
//! Gated behind `GNC_SIG_CONTEXT=1` env var. Zero overhead when disabled.
//!
//! Measures the conditional entropy of the significance map given two context
//! signals:
//!   1. **Above-neighbor context**: is the coefficient directly above (same
//!      column, row-1, clamped to subband boundary) significant?
//!   2. **Parent context**: is the corresponding coefficient in the
//!      one-level-coarser subband (at row//2, col//2 within that subband)
//!      significant?
//!
//! Prints a table to stderr with per-subband-group and aggregate results.
//! Does NOT modify the bitstream or affect encoding.

use super::rans::compute_subband_group;

/// Per-subband-group context statistics.
#[derive(Default, Clone)]
struct SubbandContextStats {
    /// Human-readable label for the subband group
    label: String,

    // --- Above-neighbor context ---
    n_total_above_sig: u64,
    n_sig_given_above_sig: u64,
    n_total_above_zero: u64,
    n_sig_given_above_zero: u64,

    // --- Parent context ---
    n_total_parent_sig: u64,
    n_sig_given_parent_sig: u64,
    n_total_parent_zero: u64,
    n_sig_given_parent_zero: u64,
}

/// Compute binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p).
/// Returns 0 for degenerate p=0 or p=1.
fn h2(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

/// Flat significance entropy for a subband: H(f_sig).
fn h_flat(n_sig: u64, n_total: u64) -> f64 {
    if n_total == 0 {
        return 0.0;
    }
    h2(n_sig as f64 / n_total as f64)
}

/// Conditional entropy H(sig | context) given the two context buckets.
/// Weighted average: w0 * H(p0) + w1 * H(p1) where w = fraction of coeffs
/// in that context bucket.
fn h_conditional(
    n_sig_c0: u64, n_total_c0: u64,
    n_sig_c1: u64, n_total_c1: u64,
) -> f64 {
    let n_total = n_total_c0 + n_total_c1;
    if n_total == 0 {
        return 0.0;
    }
    let p0 = if n_total_c0 > 0 { n_sig_c0 as f64 / n_total_c0 as f64 } else { 0.0 };
    let p1 = if n_total_c1 > 0 { n_sig_c1 as f64 / n_total_c1 as f64 } else { 0.0 };
    let w0 = n_total_c0 as f64 / n_total as f64;
    let w1 = n_total_c1 as f64 / n_total as f64;
    w0 * h2(p0) + w1 * h2(p1)
}

/// Find the subband region boundaries for a coefficient at tile position (lx, ly).
///
/// Returns `(subband_x0, subband_y0, subband_w, subband_h)` — the top-left
/// corner and dimensions of the sub-subband (LH, HL, or HH) that this
/// coefficient belongs to. For LL, returns (0, 0, ll_size, ll_size).
fn subband_region(lx: u32, ly: u32, tile_size: u32, num_levels: u32) -> (u32, u32, u32, u32) {
    let mut region = tile_size;
    for _level in 0..num_levels {
        let half = region / 2;
        if lx >= half || ly >= half {
            // This coefficient is in the detail subbands of this wavelet level.
            let (x0, y0) = if lx >= half && ly < half {
                // HL: cols [half..region), rows [0..half)
                (half, 0)
            } else if lx < half {
                // LH: cols [0..half), rows [half..region)
                (0, half)
            } else {
                // HH: cols [half..region), rows [half..region)
                (half, half)
            };
            return (x0, y0, half, half);
        }
        region = half;
    }
    // LL
    (0, 0, region, region)
}

/// Compute the tile-local position of the **above neighbor** for coefficient (lx, ly).
///
/// Returns `None` if (lx, ly) is on the top row of its subband (no above neighbor).
fn above_neighbor_pos(lx: u32, ly: u32, tile_size: u32, num_levels: u32) -> Option<(u32, u32)> {
    let (_, y0, _, _) = subband_region(lx, ly, tile_size, num_levels);
    if ly == y0 {
        None // first row of subband — no above neighbor
    } else {
        Some((lx, ly - 1))
    }
}

/// Compute the tile-local position of the **parent** coefficient for (lx, ly).
///
/// The parent is in the one-level-coarser subband, at the position corresponding
/// to (row//2, col//2) within the same sub-subband type (LH→LH, HL→HL, HH→HH).
///
/// Returns `None` for LL (no parent) and for the deepest detail level (group 1,
/// whose "parent" would be LL — we skip that because LL has a very different
/// distribution and mixing it would distort the measurement).
fn parent_pos(lx: u32, ly: u32, tile_size: u32, num_levels: u32) -> Option<(u32, u32)> {
    let group = compute_subband_group(lx, ly, tile_size, num_levels);

    // LL (group 0) has no parent; deepest detail (group 1) maps to LL which is
    // statistically very different — omit to keep the measurement clean.
    if group <= 1 {
        return None;
    }

    // Decode the level from the group index.
    // Directional grouping: group 0 = LL, group 1 = finest merged,
    // groups 2..num_levels*2 = pairs (LH+HL, HH) from finest+1 to coarsest.
    // group = 2 + (finest_lfd - 1) * 2 + is_hh  where lfd = num_levels-1-level.
    // For lfd: deepest lfd=0 → group 1 (handled above).
    //          next:  lfd=1 → group 2 or 3.
    // Reverse: lfd = (group - 2) / 2 + 1.  level = num_levels - 1 - lfd.
    let lfd = (group - 2) / 2 + 1;
    let level = (num_levels as usize).saturating_sub(1 + lfd);

    // Current subband region (at this level's `region` = tile_size >> level)
    let region = tile_size >> level;
    let half = region / 2;

    // Determine sub-subband type and local position within it
    let (sb_x0, sb_y0) = if lx >= half && ly < half {
        (half, 0) // HL
    } else if lx < half && ly >= half {
        (0, half) // LH
    } else {
        (half, half) // HH
    };

    // Local position within this sub-subband
    let local_x = lx - sb_x0;
    let local_y = ly - sb_y0;

    // Parent subband: one level coarser (level + 1), same sub-subband type.
    // At level+1, region = tile_size >> (level+1) = half, half_parent = half/2.
    let parent_region = half; // = tile_size >> (level + 1)
    let parent_half = parent_region / 2;

    // Parent sub-subband origin within the tile (at level+1's coarser region)
    // The coarser level occupies [0..parent_region) × [0..parent_region) within the tile.
    let (parent_sb_x0, parent_sb_y0) = if sb_x0 > 0 && sb_y0 == 0 {
        (parent_half, 0) // HL at coarser level
    } else if sb_x0 == 0 && sb_y0 > 0 {
        (0, parent_half) // LH at coarser level
    } else {
        (parent_half, parent_half) // HH at coarser level
    };

    // Parent local position: (local_x // 2, local_y // 2)
    let parent_lx = parent_sb_x0 + local_x / 2;
    let parent_ly = parent_sb_y0 + local_y / 2;

    Some((parent_lx, parent_ly))
}

/// Subband group label for display.
fn group_label(group: usize, num_levels: u32) -> String {
    match group {
        0 => "LL".to_string(),
        1 => "finest(merged)".to_string(),
        g => {
            // g = 2 + (lfd-1)*2 + is_hh, lfd = (g-2)/2 + 1, level = num_levels-1-lfd
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

/// Run the significance-context diagnostic on a single tile's quantized coefficients.
///
/// `coeffs`: `tile_size × tile_size` f32 values in raster order (row-major).
/// `tile_size` and `num_levels` describe the wavelet decomposition.
/// `stats`: per-group statistics accumulators (indexed by group 0..num_groups).
fn analyze_tile(
    coeffs: &[f32],
    tile_size: u32,
    num_levels: u32,
    stats: &mut [SubbandContextStats],
) {
    let ts = tile_size as usize;
    let n = ts * ts;
    debug_assert_eq!(coeffs.len(), n, "tile coeff count mismatch");

    for idx in 0..n {
        let lx = (idx % ts) as u32;
        let ly = (idx / ts) as u32;
        let group = compute_subband_group(lx, ly, tile_size, num_levels);
        let is_sig = coeffs[idx] != 0.0;

        // --- Above-neighbor context ---
        let above_sig = match above_neighbor_pos(lx, ly, tile_size, num_levels) {
            None => {
                // No above neighbor: count as "above=zero" context
                false
            }
            Some((ax, ay)) => coeffs[ay as usize * ts + ax as usize] != 0.0,
        };

        debug_assert!(group < stats.len(), "group {} out of range for {} stats entries", group, stats.len());
        let st = &mut stats[group];
        if above_sig {
            st.n_total_above_sig += 1;
            if is_sig { st.n_sig_given_above_sig += 1; }
        } else {
            st.n_total_above_zero += 1;
            if is_sig { st.n_sig_given_above_zero += 1; }
        }

        // --- Parent context (skip LL and deepest merged) ---
        if let Some((px, py)) = parent_pos(lx, ly, tile_size, num_levels) {
            let parent_sig = coeffs[py as usize * ts + px as usize] != 0.0;
            if parent_sig {
                st.n_total_parent_sig += 1;
                if is_sig { st.n_sig_given_parent_sig += 1; }
            } else {
                st.n_total_parent_zero += 1;
                if is_sig { st.n_sig_given_parent_zero += 1; }
            }
        }
    }
}

/// Run the significance-context diagnostic on a full quantized plane (raster f32 buffer).
///
/// `plane`: full-plane buffer in raster order (padded_w × padded_h).
/// `padded_w` / `padded_h`: padded frame dimensions.
/// `tile_size` / `num_levels`: wavelet decomposition parameters.
///
/// Returns per-group stats accumulated over all tiles.
fn analyze_plane(
    plane: &[f32],
    padded_w: u32,
    padded_h: u32,
    tile_size: u32,
    num_levels: u32,
) -> Vec<SubbandContextStats> {
    let num_groups = num_levels as usize * 2; // directional grouping: LL + num_levels*2-1 detail groups, rounded up
    let num_groups = num_groups.max(2); // at least LL + one detail group
    let mut stats: Vec<SubbandContextStats> = (0..num_groups)
        .map(|g| SubbandContextStats {
            label: group_label(g, num_levels),
            ..Default::default()
        })
        .collect();

    let tiles_x = (padded_w / tile_size) as usize;
    let tiles_y = (padded_h / tile_size) as usize;
    let ts = tile_size as usize;

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            // Extract tile coefficients into a contiguous buffer
            let mut tile_coeffs = vec![0.0f32; ts * ts];
            for row in 0..ts {
                let src_row = ty * ts + row;
                let src_col0 = tx * ts;
                let src_start = src_row * padded_w as usize + src_col0;
                let dst_start = row * ts;
                tile_coeffs[dst_start..dst_start + ts]
                    .copy_from_slice(&plane[src_start..src_start + ts]);
            }
            analyze_tile(&tile_coeffs, tile_size, num_levels, &mut stats);
        }
    }

    stats
}

/// Print the diagnostic table to stderr.
fn print_table(plane_label: &str, stats: &[SubbandContextStats], padded_w: u32, padded_h: u32, tile_size: u32) {
    let total_pixels = (padded_w * padded_h) as f64;

    // Aggregate across all groups
    let mut agg_n_sig = 0u64;
    let mut agg_n_total = 0u64;
    let mut agg_above_sig_c0 = 0u64; let mut agg_above_total_c0 = 0u64;
    let mut agg_above_sig_c1 = 0u64; let mut agg_above_total_c1 = 0u64;
    let mut agg_parent_sig_c0 = 0u64; let mut agg_parent_total_c0 = 0u64;
    let mut agg_parent_sig_c1 = 0u64; let mut agg_parent_total_c1 = 0u64;

    for st in stats {
        let n_total = st.n_total_above_sig + st.n_total_above_zero;
        let n_sig = st.n_sig_given_above_sig + st.n_sig_given_above_zero;
        agg_n_sig += n_sig;
        agg_n_total += n_total;
        agg_above_sig_c0 += st.n_sig_given_above_zero; agg_above_total_c0 += st.n_total_above_zero;
        agg_above_sig_c1 += st.n_sig_given_above_sig;  agg_above_total_c1 += st.n_total_above_sig;
        agg_parent_sig_c0 += st.n_sig_given_parent_zero; agg_parent_total_c0 += st.n_total_parent_zero;
        agg_parent_sig_c1 += st.n_sig_given_parent_sig;  agg_parent_total_c1 += st.n_total_parent_sig;
    }

    eprintln!();
    eprintln!("=== Significance Context Diagnostic ({plane_label}) ===");
    eprintln!(
        "{:<18} | {:>8} | {:>6} | {:>7} | {:>7} | {:>7} | {:>10} | {:>11}",
        "Subband", "n_coeff", "f_sig", "H_flat", "H_above", "H_parent", "gain_above", "gain_parent"
    );
    eprintln!("{}", "-".repeat(90));

    for st in stats {
        let n_total = st.n_total_above_sig + st.n_total_above_zero;
        if n_total == 0 {
            continue;
        }
        let n_sig = st.n_sig_given_above_sig + st.n_sig_given_above_zero;
        let f_sig = n_sig as f64 / n_total as f64;
        let h_flat = h_flat(n_sig, n_total);

        let h_above = h_conditional(
            st.n_sig_given_above_zero, st.n_total_above_zero,
            st.n_sig_given_above_sig, st.n_total_above_sig,
        );
        let gain_above = h_flat - h_above;

        let has_parent = st.n_total_parent_sig + st.n_total_parent_zero > 0;
        let (h_parent_str, gain_parent_str) = if has_parent {
            let h_p = h_conditional(
                st.n_sig_given_parent_zero, st.n_total_parent_zero,
                st.n_sig_given_parent_sig, st.n_total_parent_sig,
            );
            let gain_p = h_flat - h_p;
            (format!("{:.4}", h_p), format!("{:+.4}", gain_p))
        } else {
            ("N/A    ".to_string(), "N/A       ".to_string())
        };

        eprintln!(
            "{:<18} | {:>8} | {:>6.3} | {:>7.4} | {:>7.4} | {} | {:>+10.4} | {}",
            st.label,
            n_total,
            f_sig,
            h_flat,
            h_above,
            h_parent_str,
            gain_above,
            gain_parent_str,
        );
    }

    eprintln!("{}", "-".repeat(90));

    // Aggregate row
    if agg_n_total > 0 {
        let agg_f_sig = agg_n_sig as f64 / agg_n_total as f64;
        let agg_h_flat = h_flat(agg_n_sig, agg_n_total);
        let agg_h_above = h_conditional(agg_above_sig_c0, agg_above_total_c0, agg_above_sig_c1, agg_above_total_c1);
        let agg_gain_above = agg_h_flat - agg_h_above;
        let agg_has_parent = agg_parent_total_c0 + agg_parent_total_c1 > 0;
        let (agg_h_parent_str, agg_gain_parent_str) = if agg_has_parent {
            let h_p = h_conditional(agg_parent_sig_c0, agg_parent_total_c0, agg_parent_sig_c1, agg_parent_total_c1);
            let gain_p = agg_h_flat - h_p;
            (format!("{:.4}", h_p), format!("{:+.4}", gain_p))
        } else {
            ("N/A    ".to_string(), "N/A       ".to_string())
        };
        eprintln!(
            "{:<18} | {:>8} | {:>6.3} | {:>7.4} | {:>7.4} | {} | {:>+10.4} | {}",
            "AGGREGATE",
            agg_n_total,
            agg_f_sig,
            agg_h_flat,
            agg_h_above,
            agg_h_parent_str,
            agg_gain_above,
            agg_gain_parent_str,
        );

        // Estimated bpp improvement from above-neighbor context.
        // Significance bits per pixel = f_sig (1 bit each) + overhead already in Rice.
        // The gain in bits/significance-check * checks/pixel gives bpp improvement.
        // checks/pixel ≈ 1 (one sig bit per coefficient, one coeff per pixel for Y plane).
        let sig_bits_per_pixel = agg_n_total as f64 / total_pixels;
        let bpp_gain_above = agg_gain_above * sig_bits_per_pixel;
        let bpp_gain_parent = if agg_has_parent {
            let h_p_val = h_conditional(agg_parent_sig_c0, agg_parent_total_c0, agg_parent_sig_c1, agg_parent_total_c1);
            (agg_h_flat - h_p_val) * sig_bits_per_pixel
        } else {
            0.0
        };

        eprintln!();
        eprintln!(
            "  H_flat={:.4}  H_above={:.4}  H_parent={}  gain_above={:.4} bits/sig-check",
            agg_h_flat, agg_h_above,
            if agg_has_parent { format!("{:.4}", h_conditional(agg_parent_sig_c0, agg_parent_total_c0, agg_parent_sig_c1, agg_parent_total_c1)) } else { "N/A".to_string() },
            agg_gain_above,
        );
        eprintln!(
            "  Estimated bpp improvement (above-neighbor): {:.4} bpp",
            bpp_gain_above
        );
        if agg_has_parent {
            eprintln!(
                "  Estimated bpp improvement (parent-context): {:.4} bpp",
                bpp_gain_parent
            );
        }
        eprintln!(
            "  Tile size: {}×{}, levels: N/A (see call site), total_coeffs: {}, frame_px: {:.0}",
            tile_size, tile_size, agg_n_total, total_pixels
        );
    }
}

/// Run the significance-context diagnostic on quantized coefficient planes.
///
/// Called from `pipeline.rs` after quantization, before entropy coding, on the
/// first I-frame only. Handles both 4:4:4 (all planes same size) and non-444
/// (Co/Cg use chroma dimensions).
///
/// `ctx`: GPU context for buffer readback.
/// `y_buf`: quantized Y plane (GPU buffer, f32, padded_w × padded_h).
/// `co_buf`: quantized Co plane.
/// `cg_buf`: quantized Cg plane.
/// `padded_w`, `padded_h`: padded luma frame dimensions.
/// `chroma_w`, `chroma_h`: padded chroma frame dimensions.
/// `tile_size`, `num_levels`: wavelet decomposition parameters.
#[allow(clippy::too_many_arguments)] // 3 GPU buffers + 6 dimension/config scalars; no natural grouping
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
    eprintln!("[sig_context] diagnostic active (GNC_SIG_CONTEXT)");

    let luma_pixels = (padded_w * padded_h) as usize;
    let chroma_px = (chroma_w * chroma_h) as usize;

    // Read quantized planes back from GPU
    let y_plane = crate::gpu_util::read_buffer_f32(ctx, y_buf, luma_pixels);
    let co_plane = crate::gpu_util::read_buffer_f32(ctx, co_buf, chroma_px);
    let cg_plane = crate::gpu_util::read_buffer_f32(ctx, cg_buf, chroma_px);

    // Analyze each plane — Y at luma dims, Co/Cg at chroma dims
    let y_stats = analyze_plane(&y_plane, padded_w, padded_h, tile_size, num_levels);
    print_table("Y (I-frame)", &y_stats, padded_w, padded_h, tile_size);

    let co_stats = analyze_plane(&co_plane, chroma_w, chroma_h, tile_size, num_levels);
    print_table("Co (I-frame)", &co_stats, chroma_w, chroma_h, tile_size);

    let cg_stats = analyze_plane(&cg_plane, chroma_w, chroma_h, tile_size, num_levels);
    print_table("Cg (I-frame)", &cg_stats, chroma_w, chroma_h, tile_size);
}
