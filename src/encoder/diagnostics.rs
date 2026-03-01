//! Per-frame encode diagnostics for debugging and analysis.
//!
//! Gated behind `GNC_DIAGNOSTICS=1` env var or `--diagnostics` CLI flag.
//! Zero overhead when disabled — all collection is behind runtime checks.

use crate::{CompressedFrame, EntropyData, FrameType, MotionField};

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
        warnings: Vec::new(),
    };

    // MV stats
    if let Some(ref mf) = compressed.motion_field {
        collect_mv_stats(&mut diag, mf);
    }

    // Entropy stats
    collect_entropy_stats(&mut diag, &compressed.entropy);

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

    // Warnings
    for w in &diag.warnings {
        eprintln!("  {}", w);
    }
}
