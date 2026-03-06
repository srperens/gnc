//! Per-frame encode diagnostics for debugging and analysis.
//!
//! Gated behind `GNC_DIAGNOSTICS=1` env var or `--diagnostics` CLI flag.
//! Zero overhead when disabled — all collection is behind runtime checks.

use crate::{CompressedFrame, EntropyData, FrameType, GpuContext, MotionField, ResidualStats};
use super::cfl;

/// Bit budget breakdown: where the bytes go in a compressed frame.
#[derive(Debug, Default)]
pub struct BitBudget {
    /// Motion vector data (delta-coded varint MVs + skip bitmap)
    pub mv_bytes: usize,
    /// Tile overhead: headers, Rice k-params, skip bitmaps, stream lengths
    pub tile_header_bytes: usize,
    /// Actual coefficient bitstream data (Rice/rANS/Huffman streams)
    pub coefficient_bytes: usize,
    /// CfL alpha side information
    pub cfl_bytes: usize,
    /// Adaptive quantization weight map
    pub weight_map_bytes: usize,
    /// Intra prediction modes
    pub intra_bytes: usize,
    /// Total frame size in bytes
    pub total_bytes: usize,
}

/// Rice-specific entropy coding efficiency statistics.
#[derive(Debug, Default)]
pub struct RiceEfficiency {
    /// Total bits across all tile streams
    pub total_stream_bits: u64,
    /// Total non-zero coefficients (estimated from tile data)
    pub total_nonzero_coeffs: u64,
    /// Total coefficients across all tiles
    pub total_coeffs: u64,
    /// Average Rice k parameter for magnitudes
    pub avg_k_mag: f64,
    /// Average Rice k parameter for zero-run lengths
    pub avg_k_zrl: f64,
    /// Number of tiles where all 256 streams are empty (skip_bitmap == 0xFF)
    pub tiles_all_skipped: usize,
    /// Total tiles
    pub total_tiles: usize,
}

/// Check if diagnostics are enabled (cached after first call).
pub fn enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("GNC_DIAGNOSTICS").is_ok())
}

/// Enable diagnostics programmatically (e.g. from CLI flag).
pub fn enable() {
    std::env::set_var("GNC_DIAGNOSTICS", "1");
}

/// Read back a GPU staging buffer containing f32 residual values and compute statistics.
/// The residual is in YCoCg-R Y-plane space (roughly [0,255] range for the original signal).
pub fn compute_residual_stats(
    ctx: &GpuContext,
    staging_buf: &wgpu::Buffer,
    buf_size: u64,
    pixel_count: usize,
) -> ResidualStats {
    let buf_slice = staging_buf.slice(..buf_size);
    let (tx, rx) = std::sync::mpsc::channel();
    buf_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = buf_slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&data);

    let n = pixel_count.min(floats.len());
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut near_zero = 0usize;

    for &v in &floats[..n] {
        let abs_v = v.abs() as f64;
        sum_abs += abs_v;
        sum_sq += abs_v * abs_v;
        if abs_v < 1.0 {
            near_zero += 1;
        }
    }

    let mean_abs = sum_abs / n as f64;
    let variance = sum_sq / n as f64 - (sum_abs / n as f64).powi(2);
    let stddev = if variance > 0.0 { variance.sqrt() } else { 0.0 };
    let near_zero_pct = near_zero as f64 / n as f64 * 100.0;

    drop(data);
    staging_buf.unmap();

    ResidualStats {
        mean_abs,
        stddev,
        near_zero_pct,
        pixel_count: n,
    }
}

/// MV statistics for a single direction.
#[derive(Debug, Default)]
pub struct MvStats {
    pub count: usize,
    pub zero_count: usize,
    pub mean_abs: f64,
    pub median_abs: f64,
    pub max_abs: u32,
    /// Histogram: [0, 1-2, 3-4, 5-8, 9-16, 17+]
    pub histogram: [usize; 6],
}

/// Per-frame diagnostics collected after encoding.
#[derive(Debug)]
pub struct FrameDiagnostics {
    pub frame_idx: usize,
    pub frame_type: FrameType,
    pub encoded_bytes: usize,
    pub bpp: f64,
    pub qstep: f32,

    // I-frame size for ratio computation
    pub last_iframe_bytes: Option<usize>,

    // Motion estimation (P/B frames)
    pub total_macroblocks: usize,
    pub split_count: usize,       // blocks with block_size=8
    pub skip_count: usize,        // blocks with MV=(0,0)
    pub mv_raw: Option<MvStats>,  // raw MVs
    pub mv_delta: Option<MvStats>, // delta MVs (after prediction) — future enhancement

    // B-frame specific
    pub fwd_mv_stats: Option<MvStats>,
    pub bwd_mv_stats: Option<MvStats>,
    pub mode_fwd_count: usize,
    pub mode_bwd_count: usize,
    pub mode_bidir_count: usize,

    // Rice entropy stats
    pub tiles_with_skip: usize,
    pub total_tiles: usize,
    pub subbands_skipped: usize,
    pub subbands_total: usize,

    // Residual statistics (spatial domain, after MC before wavelet)
    pub residual: Option<ResidualStats>,
    pub residual_co: Option<ResidualStats>,
    pub residual_cg: Option<ResidualStats>,

    // Bit budget breakdown
    pub bit_budget: Option<BitBudget>,

    // Rice entropy efficiency
    pub rice_efficiency: Option<RiceEfficiency>,

    // Warnings
    pub warnings: Vec<String>,
}

fn compute_mv_stats(vectors: &[[i16; 2]]) -> MvStats {
    if vectors.is_empty() {
        return MvStats::default();
    }

    let count = vectors.len();
    let mut zero_count = 0usize;
    let mut sum_abs = 0u64;
    let mut max_abs = 0u32;
    let mut abs_values: Vec<u32> = Vec::with_capacity(count);
    let mut histogram = [0usize; 6];

    for &[dx, dy] in vectors {
        let abs_mv = (dx.unsigned_abs() as u32).max(dy.unsigned_abs() as u32);
        abs_values.push(abs_mv);
        sum_abs += abs_mv as u64;
        if abs_mv > max_abs {
            max_abs = abs_mv;
        }
        if dx == 0 && dy == 0 {
            zero_count += 1;
        }
        match abs_mv {
            0 => histogram[0] += 1,
            1..=2 => histogram[1] += 1,
            3..=4 => histogram[2] += 1,
            5..=8 => histogram[3] += 1,
            9..=16 => histogram[4] += 1,
            _ => histogram[5] += 1,
        }
    }

    abs_values.sort_unstable();
    let median_abs = abs_values[count / 2] as f64;
    let mean_abs = sum_abs as f64 / count as f64;

    MvStats {
        count,
        zero_count,
        mean_abs,
        median_abs,
        max_abs,
        histogram,
    }
}

/// Collect diagnostics from an encoded frame.
pub fn collect(
    frame_idx: usize,
    compressed: &CompressedFrame,
    qstep: f32,
    last_iframe_bytes: Option<usize>,
) -> FrameDiagnostics {
    let mut diag = FrameDiagnostics {
        frame_idx,
        frame_type: compressed.frame_type,
        encoded_bytes: compressed.byte_size(),
        bpp: compressed.bpp(),
        qstep,
        last_iframe_bytes,
        total_macroblocks: 0,
        split_count: 0,
        skip_count: 0,
        mv_raw: None,
        mv_delta: None,
        fwd_mv_stats: None,
        bwd_mv_stats: None,
        mode_fwd_count: 0,
        mode_bwd_count: 0,
        mode_bidir_count: 0,
        tiles_with_skip: 0,
        total_tiles: 0,
        subbands_skipped: 0,
        subbands_total: 0,
        residual: compressed.residual_stats,
        residual_co: compressed.residual_stats_co,
        residual_cg: compressed.residual_stats_cg,
        bit_budget: None,
        rice_efficiency: None,
        warnings: Vec::new(),
    };

    // MV stats
    if let Some(ref mf) = compressed.motion_field {
        collect_mv_stats(&mut diag, mf);
    }

    // Entropy stats
    collect_entropy_stats(&mut diag, &compressed.entropy);

    // Bit budget breakdown
    diag.bit_budget = Some(collect_bit_budget(compressed));

    // Rice efficiency
    diag.rice_efficiency = collect_rice_efficiency(&compressed.entropy);

    // Sanity warnings
    collect_warnings(&mut diag);

    diag
}

fn collect_mv_stats(diag: &mut FrameDiagnostics, mf: &MotionField) {
    let block_size = mf.block_size;
    let total = mf.vectors.len();
    diag.total_macroblocks = total;

    // For 8x8 split MVs: a "non-split" 16x16 block appears as 4 identical 8x8 MVs.
    // Count unique groups of 4 where all MVs match = not split.
    // Count where they differ = split to 8x8.
    if block_size == 8 {
        // We don't have the original 16x16 grid to compare against,
        // but skip_count counts zero MVs regardless.
        // split_count would require the 16x16 MVs. For now, just report MV stats.
        diag.skip_count = mf.vectors.iter().filter(|v| v[0] == 0 && v[1] == 0).count();
    } else {
        diag.skip_count = mf.vectors.iter().filter(|v| v[0] == 0 && v[1] == 0).count();
    }

    diag.mv_raw = Some(compute_mv_stats(&mf.vectors));

    // B-frame: separate fwd/bwd stats
    if let Some(ref bwd) = mf.backward_vectors {
        diag.fwd_mv_stats = Some(compute_mv_stats(&mf.vectors));
        diag.bwd_mv_stats = Some(compute_mv_stats(bwd));
    }

    if let Some(ref modes) = mf.block_modes {
        for &m in modes {
            match m {
                0 => diag.mode_fwd_count += 1,
                1 => diag.mode_bwd_count += 1,
                _ => diag.mode_bidir_count += 1,
            }
        }
    }
}

fn collect_entropy_stats(diag: &mut FrameDiagnostics, entropy: &EntropyData) {
    if let EntropyData::Rice(ref tiles) = entropy {
        diag.total_tiles = tiles.len();
        for tile in tiles {
            let skip_bits = tile.skip_bitmap;
            if skip_bits != 0 {
                diag.tiles_with_skip += 1;
            }
            diag.subbands_skipped += skip_bits.count_ones() as usize;
            diag.subbands_total += tile.num_groups as usize;
        }
    }
}

/// Estimate the serialized size of delta-coded zigzag varint MVs with skip bitmap.
/// This matches the GP12 encoding in format.rs (but avoids full serialization).
fn estimate_mv_delta_size(vectors: &[[i16; 2]], block_size: u32, width: u32, tile_size: u32) -> usize {
    let n = vectors.len();
    if n == 0 {
        return 0;
    }
    // Skip bitmap: ceil(N/8) bytes
    let bitmap_bytes = n.div_ceil(8);
    // Count non-skip blocks and estimate varint sizes
    let padded_w = width.div_ceil(tile_size) * tile_size;
    let blocks_x = (padded_w / block_size) as usize;
    let blocks_y = if blocks_x > 0 { n / blocks_x } else { 0 };
    let mut varint_bytes = 0usize;
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let idx = by * blocks_x + bx;
            if idx >= n {
                break;
            }
            if vectors[idx][0] == 0 && vectors[idx][1] == 0 {
                continue; // skip — no bytes
            }
            // Estimate: each varint is 1-3 bytes. For small deltas (most cases), 1 byte.
            // Median predictor makes deltas small. Assume ~2 bytes per non-skip MV.
            varint_bytes += 2;
        }
    }
    bitmap_bytes + varint_bytes
}

/// Compute bit budget breakdown from the compressed frame.
fn collect_bit_budget(frame: &CompressedFrame) -> BitBudget {
    let mv_bytes = frame.motion_field.as_ref().map_or(0, |mf| {
        let w = frame.info.width;
        let ts = frame.info.tile_size;
        let fwd = estimate_mv_delta_size(&mf.vectors, mf.block_size, w, ts);
        let bwd = mf.backward_vectors.as_ref().map_or(0, |v| {
            estimate_mv_delta_size(v, 16, w, ts)
        });
        let modes = mf.block_modes.as_ref().map_or(0, |m| m.len());
        // Also include block_size(u16) + num_blocks(u32) header = 6 bytes
        6 + fwd + bwd + modes
    });
    let cfl_bytes = frame
        .cfl_alphas
        .as_ref()
        .map_or(0, |a| a.alphas.len() * std::mem::size_of::<i16>());
    let weight_map_bytes = frame
        .weight_map
        .as_ref()
        .map_or(0, |wm| wm.len() * std::mem::size_of::<f32>());
    let intra_bytes = frame.intra_modes.as_ref().map_or(0, |m| m.len());

    // Break down tile data into header overhead vs coefficient data
    let (tile_header_bytes, coefficient_bytes) = match &frame.entropy {
        EntropyData::Rice(tiles) => {
            let mut header = 0usize;
            let mut coeff = 0usize;
            for tile in tiles {
                // Total tile size minus stream data = header overhead
                let tile_total = tile.byte_size();
                let data_bytes = tile.stream_data.len();
                header += tile_total - data_bytes;
                coeff += data_bytes;
            }
            (header, coeff)
        }
        _ => {
            // For non-Rice: report all tile data as coefficient data
            (0, frame.entropy.byte_size())
        }
    };

    let total_bytes = mv_bytes + tile_header_bytes + coefficient_bytes + cfl_bytes + weight_map_bytes + intra_bytes;

    BitBudget {
        mv_bytes,
        tile_header_bytes,
        coefficient_bytes,
        cfl_bytes,
        weight_map_bytes,
        intra_bytes,
        total_bytes,
    }
}

/// Compute Rice-specific entropy efficiency stats.
fn collect_rice_efficiency(entropy: &EntropyData) -> Option<RiceEfficiency> {
    let tiles = match entropy {
        EntropyData::Rice(tiles) => tiles,
        _ => return None,
    };
    if tiles.is_empty() {
        return None;
    }

    let mut total_stream_bits = 0u64;
    let mut total_coeffs = 0u64;
    let mut k_mag_sum = 0f64;
    let mut k_zrl_sum = 0f64;
    let mut k_count = 0usize;
    let mut tiles_all_skipped = 0usize;

    for tile in tiles {
        // Stream data bits
        let tile_stream_bytes: u64 = tile.stream_lengths.iter().map(|&l| l as u64).sum();
        total_stream_bits += tile_stream_bytes * 8;
        total_coeffs += tile.num_coefficients as u64;

        // K parameter averages (only for non-skipped subbands)
        for g in 0..tile.num_groups as usize {
            let skipped = (tile.skip_bitmap >> g) & 1 != 0;
            if !skipped {
                k_mag_sum += tile.k_values[g] as f64;
                k_zrl_sum += tile.k_zrl_values[g] as f64;
                k_count += 1;
            }
        }

        // All-skipped tiles: skip_bitmap has all num_groups bits set
        let ng = tile.num_groups.min(8);
        let all_mask = if ng >= 8 { 0xFFu8 } else { (1u8 << ng) - 1 };
        if all_mask != 0 && tile.skip_bitmap & all_mask == all_mask {
            tiles_all_skipped += 1;
        }
    }

    // Estimate non-zero coefficients: total_coeffs minus those in skipped subbands.
    // For a rough estimate: each tile has num_coefficients total. Skipped subbands
    // contribute proportionally. The stream data encodes only non-zero info.
    // A more precise estimate: stream bits / avg_bits_per_coeff, but we compute it
    // from the total bits and total coefficients for an upper bound.
    let total_tiles = tiles.len();
    let avg_k_mag = if k_count > 0 { k_mag_sum / k_count as f64 } else { 0.0 };
    let avg_k_zrl = if k_count > 0 { k_zrl_sum / k_count as f64 } else { 0.0 };

    Some(RiceEfficiency {
        total_stream_bits,
        total_nonzero_coeffs: 0, // Not precisely known without coefficient readback
        total_coeffs,
        avg_k_mag,
        avg_k_zrl,
        tiles_all_skipped,
        total_tiles,
    })
}

fn collect_warnings(diag: &mut FrameDiagnostics) {
    match diag.frame_type {
        FrameType::Predicted | FrameType::Bidirectional => {
            let ft = if diag.frame_type == FrameType::Predicted { "P" } else { "B" };

            // P/B frame should be much smaller than I-frame
            if let Some(iframe_bytes) = diag.last_iframe_bytes {
                if iframe_bytes > 0 {
                    let ratio = diag.encoded_bytes as f64 / iframe_bytes as f64;
                    if ratio > 0.7 {
                        diag.warnings.push(format!(
                            "WARN: {}-frame ratio vs I-frame = {:.2} (>0.7) — temporal prediction may not be effective",
                            ft, ratio
                        ));
                    }
                }
            }

            // MV skip rate
            if let Some(ref mv) = diag.mv_raw {
                if mv.count > 0 {
                    let skip_pct = diag.skip_count as f64 / mv.count as f64 * 100.0;
                    if skip_pct == 0.0 {
                        diag.warnings.push(format!(
                            "WARN: {}-frame MV zero rate = 0% — skip/zero-MV detection may not be working",
                            ft
                        ));
                    }
                }
            }

            // Rice skip bitmap
            if diag.total_tiles > 0 && diag.tiles_with_skip == 0 {
                diag.warnings.push(format!(
                    "WARN: {}-frame subband skip bitmap never set — skip bitmap not working?",
                    ft
                ));
            }

            // Residual energy
            if let Some(ref r) = diag.residual {
                if r.mean_abs > 15.0 {
                    diag.warnings.push(format!(
                        "WARN: {}-frame Y residual mean_abs={:.1} (>15) — MC not reducing signal effectively",
                        ft, r.mean_abs
                    ));
                }
                if r.near_zero_pct < 50.0 {
                    diag.warnings.push(format!(
                        "WARN: {}-frame Y near_zero={:.0}% (<50%) — most pixels have significant residual",
                        ft, r.near_zero_pct
                    ));
                }
            }

            // Bit budget: P-frame coefficient data vs I-frame
            if let (Some(ref bb), Some(iframe_bytes)) =
                (&diag.bit_budget, diag.last_iframe_bytes)
            {
                if iframe_bytes > 0 && bb.coefficient_bytes > 0 {
                    let coeff_ratio = bb.coefficient_bytes as f64 / iframe_bytes as f64;
                    if coeff_ratio > 0.9 {
                        diag.warnings.push(format!(
                            "WARN: {}-frame coefficient data {:.0}% of I-frame total — residuals not significantly smaller",
                            ft, coeff_ratio * 100.0
                        ));
                    }
                }
            }

            // Rice efficiency
            if let Some(ref re) = diag.rice_efficiency {
                if re.avg_k_mag > 5.0 {
                    diag.warnings.push(format!(
                        "WARN: {}-frame Rice avg_k_mag={:.1} (>5) — magnitudes large, quantization may be too gentle",
                        ft, re.avg_k_mag
                    ));
                }
            }
        }
        FrameType::Intra => {}
    }
}

/// Print diagnostics to stderr.
pub fn print(diag: &FrameDiagnostics) {
    let ft = match diag.frame_type {
        FrameType::Intra => "I",
        FrameType::Predicted => "P",
        FrameType::Bidirectional => "B",
    };

    eprintln!();
    eprint!(
        "Frame {} [{}] size={} bpp={:.2} q={:.1}",
        diag.frame_idx, ft, diag.encoded_bytes, diag.bpp, diag.qstep
    );

    if let Some(iframe_bytes) = diag.last_iframe_bytes {
        if iframe_bytes > 0 && diag.frame_type != FrameType::Intra {
            eprint!(
                " ratio_vs_iframe={:.2}",
                diag.encoded_bytes as f64 / iframe_bytes as f64
            );
        }
    }
    eprintln!();

    // MV stats for P-frames
    if diag.frame_type == FrameType::Predicted {
        if let Some(ref mv) = diag.mv_raw {
            let total = mv.count;
            if total > 0 {
                let zero_pct = diag.skip_count as f64 / total as f64 * 100.0;
                eprintln!(
                    "  MV zero:  {}/{} ({:.0}%)",
                    diag.skip_count, total, zero_pct
                );
                eprintln!(
                    "  MV raw:   mean_abs={:.1} median_abs={:.0} max_abs={}",
                    mv.mean_abs, mv.median_abs, mv.max_abs
                );
                eprintln!(
                    "  MV hist:  [0]={} [1-2]={} [3-4]={} [5-8]={} [9-16]={} [17+]={}",
                    mv.histogram[0],
                    mv.histogram[1],
                    mv.histogram[2],
                    mv.histogram[3],
                    mv.histogram[4],
                    mv.histogram[5],
                );
            }
        }
    }

    // MV stats for B-frames
    if diag.frame_type == FrameType::Bidirectional {
        if let Some(ref fwd) = diag.fwd_mv_stats {
            eprintln!(
                "  Fwd MV:   mean_abs={:.1} median={:.0} max={}",
                fwd.mean_abs, fwd.median_abs, fwd.max_abs
            );
        }
        if let Some(ref bwd) = diag.bwd_mv_stats {
            eprintln!(
                "  Bwd MV:   mean_abs={:.1} median={:.0} max={}",
                bwd.mean_abs, bwd.median_abs, bwd.max_abs
            );
        }
        let mode_total = diag.mode_fwd_count + diag.mode_bwd_count + diag.mode_bidir_count;
        if mode_total > 0 {
            eprintln!(
                "  Modes:    fwd={} bwd={} bidir={} (of {})",
                diag.mode_fwd_count, diag.mode_bwd_count, diag.mode_bidir_count, mode_total
            );
        }
    }

    // Rice entropy stats
    if diag.total_tiles > 0 {
        eprintln!(
            "  Rice: tiles_with_skipped_subbands={}/{} ({:.0}%)",
            diag.tiles_with_skip,
            diag.total_tiles,
            diag.tiles_with_skip as f64 / diag.total_tiles as f64 * 100.0
        );
        if diag.subbands_total > 0 {
            eprintln!(
                "  Rice: total_subbands_skipped={}/{} ({:.0}%)",
                diag.subbands_skipped,
                diag.subbands_total,
                diag.subbands_skipped as f64 / diag.subbands_total as f64 * 100.0
            );
        }
    }

    // Residual statistics (per-channel)
    if let Some(ref r) = diag.residual {
        eprintln!(
            "  Residual Y:  mean_abs={:.2} stddev={:.2} near_zero={:.0}%",
            r.mean_abs, r.stddev, r.near_zero_pct
        );
    }
    if let Some(ref r) = diag.residual_co {
        eprintln!(
            "  Residual Co: mean_abs={:.2} stddev={:.2} near_zero={:.0}%",
            r.mean_abs, r.stddev, r.near_zero_pct
        );
    }
    if let Some(ref r) = diag.residual_cg {
        eprintln!(
            "  Residual Cg: mean_abs={:.2} stddev={:.2} near_zero={:.0}%",
            r.mean_abs, r.stddev, r.near_zero_pct
        );
    }

    // Bit budget breakdown
    if let Some(ref bb) = diag.bit_budget {
        if bb.total_bytes > 0 {
            eprintln!("  Bit budget:");
            let pct = |b: usize| b as f64 / bb.total_bytes as f64 * 100.0;
            if bb.mv_bytes > 0 {
                eprintln!(
                    "    MV data:         {:>7.1} KB  ({:.1}%)",
                    bb.mv_bytes as f64 / 1024.0,
                    pct(bb.mv_bytes)
                );
            }
            if bb.tile_header_bytes > 0 {
                eprintln!(
                    "    Tile headers:    {:>7.1} KB  ({:.1}%)",
                    bb.tile_header_bytes as f64 / 1024.0,
                    pct(bb.tile_header_bytes)
                );
            }
            eprintln!(
                "    Coefficient data:{:>7.1} KB  ({:.1}%)",
                bb.coefficient_bytes as f64 / 1024.0,
                pct(bb.coefficient_bytes)
            );
            if bb.cfl_bytes > 0 {
                eprintln!(
                    "    CfL alphas:      {:>7.1} KB  ({:.1}%)",
                    bb.cfl_bytes as f64 / 1024.0,
                    pct(bb.cfl_bytes)
                );
            }
            if bb.weight_map_bytes > 0 {
                eprintln!(
                    "    AQ weight map:   {:>7.1} KB  ({:.1}%)",
                    bb.weight_map_bytes as f64 / 1024.0,
                    pct(bb.weight_map_bytes)
                );
            }
            if bb.intra_bytes > 0 {
                eprintln!(
                    "    Intra modes:     {:>7.1} KB  ({:.1}%)",
                    bb.intra_bytes as f64 / 1024.0,
                    pct(bb.intra_bytes)
                );
            }
            eprintln!(
                "    Total:           {:>7.1} KB",
                bb.total_bytes as f64 / 1024.0,
            );
        }
    }

    // Rice efficiency
    if let Some(ref re) = diag.rice_efficiency {
        let bits_per_coeff = if re.total_coeffs > 0 {
            re.total_stream_bits as f64 / re.total_coeffs as f64
        } else {
            0.0
        };
        eprintln!(
            "  Rice: bits/coeff={:.2} avg_k_mag={:.1} avg_k_zrl={:.1} all_skip_tiles={}/{}",
            bits_per_coeff,
            re.avg_k_mag,
            re.avg_k_zrl,
            re.tiles_all_skipped,
            re.total_tiles,
        );
    }

    // Warnings
    for w in &diag.warnings {
        eprintln!("  {}", w);
    }
}

// ---------------------------------------------------------------------------
// Temporal wavelet potential diagnostic
// ---------------------------------------------------------------------------

/// Per-subband temporal redundancy statistics.
pub struct TemporalSubbandStats {
    /// Subband name: "LL", "LH", "HL", "HH"
    pub name: &'static str,
    /// % of coefficients where current == previous (exact match)
    pub identical_pct: f64,
    /// % of coefficients where |current - previous| <= dead_zone threshold
    pub within_dz_pct: f64,
    /// Mean of |current - previous| across all coefficients
    pub mean_abs_diff: f64,
}

/// Temporal wavelet diagnostic for one frame (all 3 color planes).
pub struct TemporalWaveletDiag {
    pub y: Vec<TemporalSubbandStats>,
    pub co: Vec<TemporalSubbandStats>,
    pub cg: Vec<TemporalSubbandStats>,
}

/// Read a GPU staging buffer and return its contents as Vec<f32>.
///
/// The buffer must have been copied to with MAP_READ usage. After reading,
/// the buffer is unmapped.
pub fn read_plane_f32(
    ctx: &GpuContext,
    staging_buf: &wgpu::Buffer,
    buf_size: u64,
    pixel_count: usize,
) -> Vec<f32> {
    let buf_slice = staging_buf.slice(..buf_size);
    let (tx, rx) = std::sync::mpsc::channel();
    buf_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).ok();
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = buf_slice.get_mapped_range();
    let floats: &[f32] = bytemuck::cast_slice(&data);
    let result = floats[..pixel_count.min(floats.len())].to_vec();
    drop(data);
    staging_buf.unmap();
    result
}

/// Compare quantized wavelet coefficients of the original signal between
/// consecutive frames. This measures temporal redundancy in the spatial
/// wavelet domain — the quantity a temporal Haar wavelet would exploit.
///
/// Both `current` and `previous` must be quantized wavelet coefficients
/// of the **original** (uncompensated) frame, not MC residuals.
pub fn compute_temporal_wavelet(
    current: &[f32],
    previous: &[f32],
    padded_w: u32,
    padded_h: u32,
    tile_size: u32,
    num_levels: u32,
    qstep: f32,
) -> Vec<TemporalSubbandStats> {
    // Aggregate into 4 groups: LL, LH, HL, HH
    // Each accumulates across all levels and tiles.
    struct Accum {
        count: u64,
        identical: u64,
        within_dz: u64,
        sum_abs_diff: f64,
    }
    let mut groups = [
        Accum { count: 0, identical: 0, within_dz: 0, sum_abs_diff: 0.0 },
        Accum { count: 0, identical: 0, within_dz: 0, sum_abs_diff: 0.0 },
        Accum { count: 0, identical: 0, within_dz: 0, sum_abs_diff: 0.0 },
        Accum { count: 0, identical: 0, within_dz: 0, sum_abs_diff: 0.0 },
    ];

    let dz_threshold = (qstep / 2.0) as f64;
    let tiles_x = padded_w / tile_size;
    let tiles_y = padded_h / tile_size;
    let n = current.len().min(previous.len());

    for i in 0..n {
        let px = (i as u32) % padded_w;
        let py = (i as u32) / padded_w;
        let tx = px / tile_size;
        let ty = py / tile_size;
        if tx >= tiles_x || ty >= tiles_y {
            continue;
        }
        let lx = px % tile_size;
        let ly = py % tile_size;

        let sb_idx = cfl::compute_subband_index(lx, ly, tile_size, num_levels);

        // Map subband index to group: 0=LL, LH=1+level*3+0, HL=1+level*3+1, HH=1+level*3+2
        let group = if sb_idx == 0 {
            0 // LL
        } else {
            match (sb_idx - 1) % 3 {
                0 => 1, // LH
                1 => 2, // HL
                _ => 3, // HH
            }
        };

        let diff = (current[i] - previous[i]).abs() as f64;
        let g = &mut groups[group];
        g.count += 1;
        if diff == 0.0 {
            g.identical += 1;
        }
        if diff <= dz_threshold {
            g.within_dz += 1;
        }
        g.sum_abs_diff += diff;
    }

    let names = ["LL", "LH", "HL", "HH"];
    groups
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let c = g.count.max(1) as f64;
            TemporalSubbandStats {
                name: names[i],
                identical_pct: g.identical as f64 / c * 100.0,
                within_dz_pct: g.within_dz as f64 / c * 100.0,
                mean_abs_diff: g.sum_abs_diff / c,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Temporal wavelet GOP diagnostics
// ---------------------------------------------------------------------------

/// Per-frame quality pair (PSNR dB, SSIM).
pub type FrameQuality = (f64, f64);

/// Coefficient statistics computed from entropy-decoded quantized coefficients.
#[derive(Debug, Default)]
pub struct CoeffStats {
    pub mean_abs: f64,
    pub stddev: f64,
    /// Percentage of coefficients that are exactly zero.
    pub zero_pct: f64,
    pub total: usize,
}

fn compute_coeff_stats_from_frame(frame: &CompressedFrame) -> CoeffStats {
    if let EntropyData::Rice(ref tiles) = frame.entropy {
        let mut sum_abs: i64 = 0;
        let mut sum_sq: f64 = 0.0;
        let mut zeros: i64 = 0;
        let mut total: i64 = 0;
        for t in tiles {
            let coeffs = super::rice::rice_decode_tile(t);
            for v in coeffs {
                let abs_v = v.unsigned_abs() as i64;
                sum_abs += abs_v;
                sum_sq += (abs_v as f64) * (abs_v as f64);
                if v == 0 {
                    zeros += 1;
                }
                total += 1;
            }
        }
        if total == 0 {
            return CoeffStats::default();
        }
        let mean_abs = sum_abs as f64 / total as f64;
        let variance = sum_sq / total as f64 - mean_abs * mean_abs;
        let stddev = if variance > 0.0 { variance.sqrt() } else { 0.0 };
        CoeffStats {
            mean_abs,
            stddev,
            zero_pct: zeros as f64 / total as f64 * 100.0,
            total: total as usize,
        }
    } else {
        CoeffStats::default()
    }
}

/// Print temporal wavelet GOP diagnostics to stderr.
///
/// Called per GOP after encoding + decoding. Matches the existing per-frame
/// diagnostics style but adapted for the temporal wavelet decomposition.
#[allow(clippy::too_many_arguments)]
pub fn print_temporal_gop_diagnostics(
    gop_idx: usize,
    gop_size: usize,
    mode: crate::TemporalTransform,
    quality: u32,
    highpass_mul: f32,
    group: &crate::TemporalGroup,
    per_frame_quality: &[FrameQuality],
    all_i_bpp: Option<f64>,
    width: u32,
    height: u32,
) {
    let pixels = width as f64 * height as f64;
    let mode_str = match mode {
        crate::TemporalTransform::Haar => "Haar",
        crate::TemporalTransform::LeGall53 => "LeGall-5/3",
        crate::TemporalTransform::None => "none",
    };
    let num_levels = group.high_frames.len();

    // Compute total GOP size
    let low_bytes = group.low_frame.byte_size();
    let mut high_bytes_per_level: Vec<usize> = Vec::new();
    let mut high_count_per_level: Vec<usize> = Vec::new();
    let mut total_high_bytes: usize = 0;
    for lvl in &group.high_frames {
        let lvl_bytes: usize = lvl.iter().map(|f| f.byte_size()).sum();
        high_bytes_per_level.push(lvl_bytes);
        high_count_per_level.push(lvl.len());
        total_high_bytes += lvl_bytes;
    }
    let total_bytes = low_bytes + total_high_bytes;
    let total_kb = total_bytes as f64 / 1024.0;
    let total_bpp = total_bytes as f64 * 8.0 / pixels / gop_size as f64;

    // Per-frame quality stats
    let (avg_psnr, avg_ssim, temporal_consistency) = if !per_frame_quality.is_empty() {
        let n = per_frame_quality.len() as f64;
        let avg_p = per_frame_quality.iter().map(|q| q.0).sum::<f64>() / n;
        let avg_s = per_frame_quality.iter().map(|q| q.1).sum::<f64>() / n;
        let consistency = if per_frame_quality.len() >= 2 {
            let max_p = per_frame_quality.iter().map(|q| q.0).fold(f64::NEG_INFINITY, f64::max);
            let min_p = per_frame_quality.iter().map(|q| q.0).fold(f64::INFINITY, f64::min);
            max_p - min_p
        } else {
            0.0
        };
        (avg_p, avg_s, consistency)
    } else {
        (0.0, 0.0, 0.0)
    };

    // --- Header ---
    eprintln!();
    eprintln!(
        "GOP {} [{} frames] temporal_wavelet={} levels={} q={} mul={:.1}",
        gop_idx, gop_size, mode_str, num_levels, quality, highpass_mul
    );
    eprintln!(
        "  Total: {:.1} KB  bpp={:.2}  psnr_avg={:.2} dB  temporal_consistency={:.2} dB",
        total_kb, total_bpp, avg_psnr, temporal_consistency
    );

    if let Some(baseline_bpp) = all_i_bpp {
        if baseline_bpp > 0.0 {
            let ratio = total_bpp / baseline_bpp;
            let saving_pct = (1.0 - ratio) * 100.0;
            eprintln!(
                "  Ratio vs all-I: {:.2} ({:+.0}%)",
                ratio, -saving_pct
            );
        }
    }

    // --- Temporal decomposition ---
    let is_53 = mode == crate::TemporalTransform::LeGall53;
    eprintln!();
    eprintln!("  Temporal decomposition:");
    // For 5/3: high_frames[0][0] is s1 (second lowpass), not highpass
    let effective_low_bytes = if is_53 && !group.high_frames.is_empty() && !group.high_frames[0].is_empty() {
        low_bytes + group.high_frames[0][0].byte_size()
    } else {
        low_bytes
    };
    let effective_low_count = if is_53 { 2 } else { 1 };
    let _effective_high_bytes = total_bytes - effective_low_bytes;

    let low_pct = if total_bytes > 0 { effective_low_bytes as f64 / total_bytes as f64 * 100.0 } else { 0.0 };
    let low_bpp = effective_low_bytes as f64 * 8.0 / pixels / effective_low_count as f64;
    eprintln!(
        "    Lowpass (L):   {} frame{} {:>7.1} KB  ({:.1}%)  avg_bpp={:.2}",
        effective_low_count,
        if effective_low_count != 1 { "s" } else { " " },
        effective_low_bytes as f64 / 1024.0, low_pct, low_bpp
    );

    // Highpass levels (skip s1 from level 0 for 5/3)
    for (lvl, (bytes, count)) in high_bytes_per_level.iter().zip(high_count_per_level.iter()).enumerate() {
        let (adj_bytes, adj_count) = if is_53 && lvl == 0 && *count > 1 {
            // Subtract s1 (first frame in level 0) from highpass stats
            let s1_bytes = group.high_frames[0][0].byte_size();
            (*bytes - s1_bytes, *count - 1)
        } else {
            (*bytes, *count)
        };
        if adj_count == 0 { continue; }
        let pct = if total_bytes > 0 { adj_bytes as f64 / total_bytes as f64 * 100.0 } else { 0.0 };
        let avg_bpp = if adj_count > 0 { adj_bytes as f64 * 8.0 / pixels / adj_count as f64 } else { 0.0 };
        eprintln!(
            "    Highpass L{}:   {} frame{} {:>7.1} KB  ({:.1}%)  avg_bpp={:.2}",
            lvl,
            adj_count,
            if adj_count != 1 { "s" } else { " " },
            adj_bytes as f64 / 1024.0,
            pct,
            avg_bpp,
        );
    }

    // --- Per-frame details ---
    eprintln!();

    // Lowpass frame
    let skip_coeff = !per_frame_quality.is_empty();
    print_temporal_frame_detail("L", 0, &group.low_frame, width, height, skip_coeff);

    // Highpass frames by level
    // For 5/3 mode: high_frames[0] = [s1, d0, d1] — first frame is actually second lowpass
    let is_53 = mode == crate::TemporalTransform::LeGall53;
    for (lvl, frames) in group.high_frames.iter().enumerate() {
        for (fi, frame) in frames.iter().enumerate() {
            let label = if is_53 && lvl == 0 && fi == 0 {
                "L1".to_string() // s1 is a lowpass, not highpass
            } else {
                format!("H{}", lvl)
            };
            let idx = if is_53 && lvl == 0 && fi > 0 { fi - 1 } else { fi };
            print_temporal_frame_detail(&label, idx, frame, width, height, skip_coeff);
        }
    }

    // --- Reconstructed quality ---
    if !per_frame_quality.is_empty() {
        eprintln!();
        eprintln!("  Reconstructed quality:");
        for (i, (psnr, ssim)) in per_frame_quality.iter().enumerate() {
            eprintln!(
                "    frame {}: psnr={:.2} dB  ssim={:.4}",
                i, psnr, ssim
            );
        }
        eprintln!(
            "    avg:     psnr={:.2} dB  ssim={:.4}",
            avg_psnr, avg_ssim
        );
    }

    // --- Warnings ---
    let mut warnings: Vec<String> = Vec::new();

    // Lowpass dominance
    if total_bytes > 0 && low_pct > 60.0 {
        warnings.push(format!(
            "WARN: lowpass = {:.0}% of GOP (>60%) — temporal decorrelation may be weak",
            low_pct
        ));
    }

    // Highpass not sparse enough — skip CPU Rice decode when PSNR is already available
    if per_frame_quality.is_empty() {
        let coeff_stats = compute_coeff_stats_from_frame(&group.low_frame);
        if coeff_stats.total > 0 && coeff_stats.zero_pct < 30.0 {
            warnings.push(format!(
                "WARN: lowpass zero_pct={:.0}% (<30%) — signal has high energy after temporal transform",
                coeff_stats.zero_pct
            ));
        }

        // Check first highpass level sparsity (skip s1 for 5/3 — it's lowpass)
        if let Some(h0_frames) = group.high_frames.first() {
            let skip = if is_53 { 1 } else { 0 };
            for (fi, hf) in h0_frames.iter().enumerate().skip(skip) {
                let hs = compute_coeff_stats_from_frame(hf);
                if hs.total > 0 && hs.zero_pct < 50.0 {
                    warnings.push(format!(
                        "WARN: highpass L0[{}] zero_pct={:.0}% (<50%) — temporal difference has high energy",
                        fi, hs.zero_pct
                    ));
                }
            }
        }
    }

    // Temporal consistency
    if temporal_consistency > 3.0 {
        warnings.push(format!(
            "WARN: temporal_consistency={:.2} dB (>3.0 dB) — quality varies significantly across frames",
            temporal_consistency
        ));
    }

    if !warnings.is_empty() {
        eprintln!();
        for w in &warnings {
            eprintln!("  {}", w);
        }
    }
}

/// Print detail for a single temporal wavelet frame (lowpass or highpass).
/// `skip_coeff_stats` suppresses the CPU Rice decode used to compute mean_abs/stddev/zero%
/// when the caller already has PSNR from a GPU decode (avoids redundant full-frame decodes).
fn print_temporal_frame_detail(
    label: &str,
    idx: usize,
    frame: &CompressedFrame,
    width: u32,
    height: u32,
    skip_coeff_stats: bool,
) {
    let pixels = width as f64 * height as f64;
    let bytes = frame.byte_size();
    let bpp = bytes as f64 * 8.0 / pixels;
    let q = frame.config.quantization_step;

    eprintln!(
        "  Frame {} [{}] size={} bpp={:.2} q={:.1}",
        idx, label, bytes, bpp, q
    );

    // Rice stats
    if let EntropyData::Rice(ref tiles) = frame.entropy {
        let total_tiles = tiles.len();
        let mut tiles_with_skip = 0usize;
        let mut subbands_skipped = 0usize;
        let mut subbands_total = 0usize;
        for tile in tiles {
            if tile.skip_bitmap != 0 {
                tiles_with_skip += 1;
            }
            subbands_skipped += tile.skip_bitmap.count_ones() as usize;
            subbands_total += tile.num_groups as usize;
        }
        if total_tiles > 0 {
            eprintln!(
                "    Rice: tiles_with_skipped_subbands={}/{} ({:.0}%)  subbands_skipped={}/{}",
                tiles_with_skip,
                total_tiles,
                tiles_with_skip as f64 / total_tiles as f64 * 100.0,
                subbands_skipped,
                subbands_total
            );
        }
    }

    // Coefficient stats — skipped when per_frame_quality is available (GPU decode already ran)
    if !skip_coeff_stats {
        let cs = compute_coeff_stats_from_frame(frame);
        if cs.total > 0 {
            eprintln!(
                "    Coefficients: mean_abs={:.2} stddev={:.2} zero={:.0}%",
                cs.mean_abs, cs.stddev, cs.zero_pct
            );
        }
    }

    // Bit budget
    let bb = collect_bit_budget(frame);
    if bb.total_bytes > 0 {
        let pct = |b: usize| b as f64 / bb.total_bytes as f64 * 100.0;
        eprintln!("    Bit budget:");
        if bb.tile_header_bytes > 0 {
            eprintln!(
                "      Tile headers:    {:>7.1} KB  ({:.1}%)",
                bb.tile_header_bytes as f64 / 1024.0,
                pct(bb.tile_header_bytes)
            );
        }
        eprintln!(
            "      Coefficient data:{:>7.1} KB  ({:.1}%)",
            bb.coefficient_bytes as f64 / 1024.0,
            pct(bb.coefficient_bytes)
        );
        if bb.cfl_bytes > 0 {
            eprintln!(
                "      CfL alphas:      {:>7.1} KB  ({:.1}%)",
                bb.cfl_bytes as f64 / 1024.0,
                pct(bb.cfl_bytes)
            );
        }
        if bb.weight_map_bytes > 0 {
            eprintln!(
                "      AQ weight map:   {:>7.1} KB  ({:.1}%)",
                bb.weight_map_bytes as f64 / 1024.0,
                pct(bb.weight_map_bytes)
            );
        }
    }

    // Rice efficiency
    if let Some(re) = collect_rice_efficiency(&frame.entropy) {
        let bits_per_coeff = if re.total_coeffs > 0 {
            re.total_stream_bits as f64 / re.total_coeffs as f64
        } else {
            0.0
        };
        eprintln!(
            "    Rice: bits/coeff={:.2} avg_k_mag={:.1} avg_k_zrl={:.1} all_skip_tiles={}/{}",
            bits_per_coeff,
            re.avg_k_mag,
            re.avg_k_zrl,
            re.tiles_all_skipped,
            re.total_tiles,
        );
    }
}

/// Print temporal wavelet potential diagnostic.
pub fn print_temporal_wavelet(
    frame_idx: usize,
    frame_type: FrameType,
    y_stats: &[TemporalSubbandStats],
    co_stats: &[TemporalSubbandStats],
    cg_stats: &[TemporalSubbandStats],
) {
    let ft = match frame_type {
        FrameType::Intra => "I",
        FrameType::Predicted => "P",
        FrameType::Bidirectional => "B",
    };
    eprintln!("  Frame {} [{}] temporal_wavelet_potential:", frame_idx, ft);
    for (label, stats) in [("Y", y_stats), ("Co", co_stats), ("Cg", cg_stats)] {
        for s in stats {
            eprintln!(
                "    {:<2} {}: identical={:.1}%  within_dz={:.1}%  mean_abs_diff={:.2}",
                label, s.name, s.identical_pct, s.within_dz_pct, s.mean_abs_diff,
            );
        }
    }
}
