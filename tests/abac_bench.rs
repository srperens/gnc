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
//! - **Best of many repeats, with the median printed beside it.** For a deterministic kernel on
//!   fixed input every error source only makes a reading *slower*, so the minimum is the
//!   least-contaminated estimate. `best` is the headline and `med/best` is the diagnostic: when
//!   it is far above 1.0 the run was not settled.
//!
//! # Run it single-threaded
//!
//! ```text
//! cargo test --release --test abac_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Two things corrupted the first idle-machine run of this bench, and neither was load:
//!
//! - **The GPU ramps its clocks.** Three consecutive processes on identical input read 66.5,
//!   45.3 and 34.9 ms — a 1.9x spread, monotonically decreasing. A freshly-idle M1 starts in a
//!   low power state and needs on the order of a second of sustained work to boost. Seven
//!   repeats did not outlast the ramp, so the median was measuring the ramp and the spread
//!   column blamed a machine that was in fact idle.
//! - **Cargo runs test functions concurrently by default**, so the CPU bench below and the GPU
//!   grid above were contending for the same device and interleaving their output. Hence
//!   `--test-threads=1`.

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
    // 24, not 7: seven dispatches do not outlast the GPU's clock ramp (see the module comment).
    const REPEATS: usize = 24;
    // A padded 1080p luma plane. A 4:4:4 frame is three of these.
    let (w, h) = (2048usize, 1280usize);
    let plane = synth_plane(w, h, 11);
    let mcoeff = (w * h) as f64 / 1e6;

    let ctx = gpu();
    let decoder = GpuAbacDecoder::new(ctx);

    println!(
        "\nabac decode throughput — {w}x{h} ({mcoeff:.2} Mcoeff), best of {REPEATS} dispatches\n\
         A 4:4:4 frame is 3 planes, so frame ms = 3 x the ms column.\n"
    );
    println!(
        "  {:<10} {:>4} {:>8} {:>10} {:>11} {:>9} {:>9}",
        "coder", "cb", "bytes", "best ms", "Mcoeff/s", "frame fps", "med/best"
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
            let best = times.iter().cloned().fold(f64::MAX, f64::min);
            let med = median(times);
            let frame_fps = 1.0 / (best * 3.0);
            if baseline.is_none() {
                baseline = Some(best);
            }
            println!(
                "  {:<10} {:>4} {:>8} {:>10.2} {:>11.1} {:>9.1} {:>8.2}x",
                format!("{coder:?}"),
                cb,
                bytes,
                best * 1e3,
                mcoeff / best,
                frame_fps,
                med / best
            );
        }
    }
    println!(
        "\n  med/best near 1.0 means the run was settled and the absolute numbers can be quoted.\n\
         Well above it means the GPU was still ramping or another session was working — the\n\
         ratios between rows may still hold, since variants were timed back-to-back.\n\
         Real coefficients decode faster than these synthetic ones (GNC_ABAC_COMPARE=1 on a real\n\
         encode reports the same measurement on a real frame); use this grid for ratios and that\n\
         one for absolute figures.\n"
    );
}

/// CPU decode for the same grid. Worth having beside the GPU numbers: it is the fallback if the
/// GPU path never gets fast enough, and it is also the sanity check that a GPU figure is
/// plausible — the GPU should not be slower than one CPU core at this block count.
#[test]
#[ignore = "throughput bench: needs an idle machine, see COORDINATION.md"]
fn abac_cpu_decode_throughput() {
    const REPEATS: usize = 9;
    let (w, h) = (1024usize, 1024);
    let plane = synth_plane(w, h, 13);
    let mcoeff = (w * h) as f64 / 1e6;

    println!("\nabac CPU decode (single thread) — {w}x{h} ({mcoeff:.2} Mcoeff)\n");
    println!(
        "  {:<10} {:>4} {:>10} {:>11}",
        "coder", "cb", "ms", "Mcoeff/s"
    );
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

/// Encode throughput, which is what ENT-5 exists to change.
///
/// Decision 0017 keeps abac opt-in for three reasons and the second is this one: **129 ms of CPU
/// encode per 1080p frame against Rice's 23 ms.** So the figure that matters is not Mcoeff/s in
/// the abstract, it is the GPU encode step's cost per frame against that 129 ms — and the CPU arm
/// is timed in the same process on the same input, because two numbers from two runs is how the
/// earlier null results happened.
///
/// The GPU rows time the whole `encode_plane_to_tiles` call: dispatches, the lengths readback, the
/// stream readback and the host-side cut into tiles. That is the honest unit — it is what the
/// encoder pays — and it is why the CPU row times `abac_encode_tile` over the same plane rather
/// than a bare `encode_block` loop.
///
/// Both output-sizing modes are here because which is faster is the open question ENT-5 was asked
/// to measure rather than pick: `CountThenEmit` runs the coder twice into an exactly-sized buffer,
/// `BoundedSlots` runs it once into a provably-bounded one and then compacts. The bytes are
/// identical either way (`tests/abac_gpu_encode.rs`), so this is purely a throughput choice.
#[test]
#[ignore = "throughput bench: needs an idle machine, see COORDINATION.md"]
fn abac_encode_throughput_grid() {
    use gnc::encoder::abac_gpu_encode::{GpuAbacEncoder, Sizing};
    use gnc::encoder::abac_tile::abac_encode_tile;
    use wgpu::util::DeviceExt;

    // 24, not 7: seven dispatches do not outlast the GPU's clock ramp.
    const REPEATS: usize = 24;
    // A padded 1080p luma plane at the shipped tile size: 8x5 tiles of 256 px, 1000 code-blocks.
    let (ts, tx, ty, levels, cb) = (256u32, 8usize, 5usize, 5u32, 64u32);
    let (w, h) = (tx * ts as usize, ty * ts as usize);
    let plane = synth_plane(w, h, 17);
    let mcoeff = (w * h) as f64 / 1e6;

    let ctx = gpu();
    let mut enc = GpuAbacEncoder::new(ctx);
    let floats: Vec<f32> = plane.iter().map(|&v| v as f32).collect();
    let input = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bench_input"),
            contents: bytemuck::cast_slice(&floats),
            usage: wgpu::BufferUsages::STORAGE,
        });

    println!(
        "\nabac encode throughput — one {w}x{h} plane ({mcoeff:.2} Mcoeff), best of {REPEATS}\n\
         A 4:4:4 frame is 3 planes, so the frame column is 3 x the plane column.\n\
         Reference: decision 0017 records 129 ms/frame of CPU encode against Rice's 23 ms.\n"
    );
    println!(
        "  {:<26} {:>8} {:>10} {:>10} {:>11} {:>9}",
        "path", "bytes", "plane ms", "frame ms", "Mcoeff/s", "med/best"
    );

    let row = |label: String, times: Vec<f64>, bytes: usize| {
        let best = times.iter().cloned().fold(f64::MAX, f64::min);
        let med = median(times);
        println!(
            "  {:<26} {:>8} {:>10.2} {:>10.2} {:>11.1} {:>8.2}x",
            label,
            bytes,
            best * 1e3,
            best * 3e3,
            mcoeff / best,
            med / best
        );
    };

    for coder in [Coder::Range, Coder::Interval] {
        for sizing in [Sizing::CountThenEmit, Sizing::BoundedSlots] {
            // One untimed call so the cached buffers are at their final size and the shader is
            // warm; without it the first repeat measures buffer creation.
            let _ =
                enc.encode_plane_to_tiles(ctx, &input, w, tx, ty, ts, levels, cb, coder, sizing);
            let mut times = Vec::with_capacity(REPEATS);
            let mut bytes = 0usize;
            for _ in 0..REPEATS {
                let t0 = std::time::Instant::now();
                let tiles = enc
                    .encode_plane_to_tiles(ctx, &input, w, tx, ty, ts, levels, cb, coder, sizing);
                times.push(t0.elapsed().as_secs_f64());
                bytes = tiles.iter().map(|t| t.block_data.len()).sum();
            }
            row(format!("GPU {coder:?}/{sizing:?}"), times, bytes);
        }
    }

    // The CPU arm: the same plane, the same tiles, the coder abac ships with. Fewer repeats
    // because it is two orders of magnitude slower and the ramp argument does not apply to it.
    for coder in [Coder::Range, Coder::Interval] {
        let mut times = Vec::with_capacity(3);
        let mut bytes = 0usize;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            let mut b = 0usize;
            for tiy in 0..ty {
                for tix in 0..tx {
                    let mut coeffs = Vec::with_capacity(ts as usize * ts as usize);
                    for y in 0..ts as usize {
                        let r = (tiy * ts as usize + y) * w + tix * ts as usize;
                        coeffs.extend_from_slice(&plane[r..r + ts as usize]);
                    }
                    b += abac_encode_tile(&coeffs, ts, levels, cb, coder)
                        .block_data
                        .len();
                }
            }
            times.push(t0.elapsed().as_secs_f64());
            bytes = b;
        }
        row(format!("CPU {coder:?} (1 thread)"), times, bytes);
    }

    println!(
        "\n  med/best near 1.0 means the run was settled and the absolute numbers can be quoted.\n\
         The bytes column must be identical across every row: the GPU encoder is bit-exact\n\
         against the CPU one, so any difference here is a bug and not a trade-off.\n"
    );
}
