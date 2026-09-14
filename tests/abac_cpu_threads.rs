//! ENT-15 — what the abac coder costs on N CPU threads, against the whole GPU.
//!
//! The GPU encode runs one thread per code-block and the blocks are independent, which is the
//! textbook shape for a GPU. It still loses to a **single** CPU thread (`abac_bench`: 32.64 ms per
//! plane on the GPU against 27.47 ms on one core). This measures the obvious follow-up question,
//! which nothing in the tree had asked: what does the same coder do on the cores this machine
//! already has?
//!
//! Ignored by default — it is a measurement, not a gate, and it wants an idle machine:
//!
//! ```text
//! cargo test --release --test abac_cpu_threads -- --ignored --nocapture --test-threads=1
//! ```

use gnc::encoder::abac::Coder;
use gnc::encoder::abac_tile::abac_encode_tile;
use std::time::Instant;

/// Deterministic wavelet-ish coefficients: mostly zero, a heavy tail, structured by subband.
fn synth_tiles(tiles: usize, ts: usize) -> Vec<Vec<i32>> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..tiles)
        .map(|_| {
            (0..ts * ts)
                .map(|i| {
                    let r = next();
                    // Coarse region dense, fine regions sparse — like a real subband layout.
                    let sparse = i > (ts * ts) / 16;
                    let zero_odds = if sparse { 12 } else { 2 };
                    if r % zero_odds != 0 {
                        0
                    } else {
                        let m = (r >> 8) % 64;
                        if r & 1 == 0 {
                            m as i32
                        } else {
                            -(m as i32)
                        }
                    }
                })
                .collect()
        })
        .collect()
}

#[test]
#[ignore = "measurement, not a gate — needs an idle machine"]
fn abac_cpu_thread_scaling() {
    // One padded 1080p luma plane at the shipped tile size: 8x5 tiles of 256 px.
    let (ts, tiles) = (256usize, 40usize);
    let planes = synth_tiles(tiles, ts);
    let mcoeff = (tiles * ts * ts) as f64 / 1e6;
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());

    println!(
        "\nabac CPU encode — one {mcoeff:.2} Mcoeff plane ({tiles} tiles of {ts}), \
         best of 5, {cores} logical cores"
    );
    println!(
        "Against the same plane on the GPU (abac_bench, BoundedSlots): 32.64 ms, 80.3 Mcoeff/s.\n"
    );
    println!(
        "  {:<10} {:>10} {:>12} {:>10}",
        "threads", "ms", "Mcoeff/s", "vs 1 thread"
    );

    let mut base = 0.0f64;
    for &n in &[1usize, 2, 4, 8, cores.max(1)] {
        if n > cores {
            continue;
        }
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = Instant::now();
            let chunk = planes.len().div_ceil(n);
            std::thread::scope(|s| {
                for part in planes.chunks(chunk) {
                    s.spawn(move || {
                        for tile in part {
                            let out = abac_encode_tile(tile, ts as u32, 5, 32, Coder::Range);
                            std::hint::black_box(&out);
                        }
                    });
                }
            });
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
        }
        if base == 0.0 {
            base = best;
        }
        println!(
            "  {:<10} {:>10.2} {:>12.1} {:>9.2}x",
            n,
            best,
            mcoeff / (best / 1000.0),
            base / best
        );
    }
    println!(
        "\nThe coder is embarrassingly parallel over code-blocks on either device. The question\n\
         this answers is which device is the right one, and it is not a matter of taste."
    );
}
