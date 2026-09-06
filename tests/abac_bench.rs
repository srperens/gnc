//! Throughput bench for the abac coder variants. **Ignored by default — run it on an idle machine.**
//!
//! ```text
//! cargo test --release --test abac_bench -- --ignored --nocapture
//! ```
//!
//! Up to five Claude sessions share this M1, and wall-clock figures are worthless while any of
//! them is working: the same input has timed 25.2, 31.1 and 37.5 ms across runs, a 48% spread.
//! COORDINATION.md has the rule. This bench is built so one idle-machine run settles the whole
//! grid rather than requiring a measurement session per question.
//!
//! Two design choices make it survive some load, and they are the reason to use it rather than
//! timing things ad hoc:
//!
//! - **Paired within a run.** Every variant is timed back-to-back on the same input in the same
//!   process, so they see the same load. Comparing two numbers from different runs is what
//!   produced the earlier null results.
//! - **Median of repeats, and the spread is printed.** If `spread` is large the run was not idle
//!   and the absolute numbers should be discarded — the ratios may still hold.

use gnc::encoder::abac::Coder;
use gnc::encoder::abac_gpu::{verify_against_cpu, GpuAbacDecoder};
use gnc::GpuContext;
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();
fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// Coefficients shaped like a quantised wavelet subband: mostly zero, non-zeros clustered, heavy
/// tail. Uniform noise would leave the significance contexts idle and flatter every variant
/// equally, which would make the bench useless for choosing between them.
fn synth_plane(w: usize, h: usize, seed: u64) -> Vec<i32> {
    let mut out = vec![0i32; w * h];
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for y in 0..h {
        for x in 0..w {
            let busy = ((y / 16) + (x / 16)) % 3 == 0;
            let r = next();
            if busy && r % 3 != 0 {
                let mag = match r % 100 {
                    0..=59 => 1,
                    60..=84 => 2,
                    85..=95 => 3 + (r % 7) as i32,
                    _ => 50 + (r % 4000) as i32,
                };
                out[y * w + x] = if r % 2 == 0 { mag } else { -mag };
            }
        }
    }
    out
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "throughput bench: needs an idle machine, see COORDINATION.md"]
fn abac_decode_throughput_grid() {
    const REPEATS: usize = 7;
    // A padded 1080p luma plane. A 4:4:4 frame is three of these.
    let (w, h) = (2048usize, 1280usize);
    let plane = synth_plane(w, h, 11);
    let mcoeff = (w * h) as f64 / 1e6;

    let ctx = gpu();
    let decoder = GpuAbacDecoder::new(ctx);

    println!(
        "\nabac decode throughput — {w}x{h} ({mcoeff:.2} Mcoeff), median of {REPEATS} runs\n\
         A 4:4:4 frame is 3 planes, so frame ms = 3 x the ms column.\n"
    );
    println!(
        "  {:<10} {:>4} {:>8} {:>10} {:>11} {:>9} {:>8}",
        "coder", "cb", "bytes", "ms", "Mcoeff/s", "frame fps", "spread"
    );

    let mut baseline: Option<f64> = None;
    for cb in [32usize, 64] {
        for coder in [Coder::Interval, Coder::Range] {
            let mut times = Vec::with_capacity(REPEATS);
            let mut bytes = 0usize;
            for _ in 0..REPEATS {
                let (b, s) = verify_against_cpu(ctx, &decoder, &plane, w, h, cb, coder);
                bytes = b;
                times.push(s);
            }
            let lo = times.iter().cloned().fold(f64::MAX, f64::min);
            let hi = times.iter().cloned().fold(0.0, f64::max);
            let med = median(times);
            let frame_fps = 1.0 / (med * 3.0);
            if baseline.is_none() {
                baseline = Some(med);
            }
            println!(
                "  {:<10} {:>4} {:>8} {:>10.2} {:>11.1} {:>9.1} {:>7.0}%",
                format!("{coder:?}"),
                cb,
                bytes,
                med * 1e3,
                mcoeff / med,
                frame_fps,
                (hi / lo - 1.0) * 100.0
            );
        }
    }
    println!(
        "\n  If the spread column is more than ~10% the machine was not idle: discard the absolute\n\
         numbers. Variants within one row group were timed back-to-back, so their ratio is more\n\
         robust to load than any single figure.\n"
    );
}

/// CPU decode for the same grid. Worth having beside the GPU numbers: it is the fallback if the
/// GPU path never gets fast enough, and it is also the sanity check that a GPU figure is
/// plausible — the GPU should not be slower than one CPU core at this block count.
#[test]
#[ignore = "throughput bench: needs an idle machine, see COORDINATION.md"]
fn abac_cpu_decode_throughput() {
    const REPEATS: usize = 5;
    let (w, h) = (1024usize, 1024);
    let plane = synth_plane(w, h, 13);
    let mcoeff = (w * h) as f64 / 1e6;

    println!("\nabac CPU decode (single thread) — {w}x{h} ({mcoeff:.2} Mcoeff)\n");
    println!("  {:<10} {:>4} {:>10} {:>11}", "coder", "cb", "ms", "Mcoeff/s");
    for cb in [32usize, 64] {
        for coder in [Coder::Interval, Coder::Range] {
            // Encode once, then time the decode of every block.
            let mut blocks = Vec::new();
            let mut by = 0;
            while by < h {
                let bh = cb.min(h - by);
                let mut bx = 0;
                while bx < w {
                    let bw = cb.min(w - bx);
                    let mut blk = Vec::with_capacity(bw * bh);
                    for y in 0..bh {
                        let row = (by + y) * w + bx;
                        blk.extend_from_slice(&plane[row..row + bw]);
                    }
                    blocks.push((coder.encode_block(&blk, bw), blk.len(), bw));
                    bx += cb;
                }
                by += cb;
            }
            let mut times = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let t0 = std::time::Instant::now();
                for (bytes, count, bw) in &blocks {
                    let _ = coder.decode_block(bytes, *count, *bw);
                }
                times.push(t0.elapsed().as_secs_f64());
            }
            let med = median(times);
            println!(
                "  {:<10} {:>4} {:>10.2} {:>11.1}",
                format!("{coder:?}"),
                cb,
                med * 1e3,
                mcoeff / med
            );
        }
    }
    println!();
}
