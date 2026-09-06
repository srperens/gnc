//! The GPU abac decoder must reproduce the CPU coder bit-for-bit.
//!
//! This matters more than most equivalence tests. An adaptive arithmetic decoder that diverges
//! from its encoder by one bit does not fail — it carries on and produces plausible-looking
//! garbage, because every subsequent symbol is decoded from a corrupted interval and a corrupted
//! context. An off-by-one in the context bucket (WGSL has `firstLeadingBit` where Rust has
//! `leading_zeros`, which differ by one) reconstructs a whole wrong image without any error.
//!
//! Synthesises its own coefficients so it runs without test material.

use gnc::encoder::abac::Coder;
use gnc::encoder::abac_gpu::{verify_against_cpu, GpuAbacDecoder};
use gnc::GpuContext;
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();
fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// Coefficients shaped like a quantised wavelet subband: mostly zero, non-zeros clustered, a
/// heavy tail. Uniform noise would leave the significance contexts untested, which is the point.
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
            // Busy regions, so the neighbourhood context has structure to find.
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

#[test]
fn gpu_decode_matches_cpu_coder() {
    let ctx = gpu();
    let decoder = GpuAbacDecoder::new(ctx);

    for &(w, h, cb, seed) in &[
        (64usize, 64usize, 64usize, 1u64),
        (128, 128, 64, 2),
        (256, 256, 64, 3),
        (256, 256, 32, 5),
        // Non-multiple geometry: subband edges produce partial blocks, and the shader takes a
        // block's stream end from the next block's offset, so ragged blocks exercise that.
        (200, 100, 64, 6),
        (67, 33, 32, 7),
        // A realistic plane, so the timing is not dominated by fixed dispatch overhead:
        // 2048x1280 is a padded 1080p luma plane, 640 blocks at 64px.
        (2048, 1280, 64, 8),
        (2048, 1280, 32, 9),
    ] {
        let plane = synth_plane(w, h, seed);
        for coder in [Coder::Interval, Coder::Range] {
            let (bytes, gpu_s) = verify_against_cpu(ctx, &decoder, &plane, w, h, cb, coder);
            let mc = (w * h) as f64 / 1e6;
            eprintln!(
                "  {w}x{h} cb={cb} {coder:?}: {bytes} B, GPU decode {:.3} ms ({:.1} Mcoeff/s)",
                gpu_s * 1e3,
                mc / gpu_s
            );
        }
    }
}

/// All-zero and all-saturated planes are the cases where an interval coder is most likely to
/// mis-handle its renormalisation, and they cost nothing to check.
#[test]
fn gpu_decode_handles_degenerate_planes() {
    let ctx = gpu();
    let decoder = GpuAbacDecoder::new(ctx);
    for coder in [Coder::Interval, Coder::Range] {
        let zero = vec![0i32; 128 * 128];
        verify_against_cpu(ctx, &decoder, &zero, 128, 128, 64, coder);
        let ones = vec![1i32; 128 * 128];
        verify_against_cpu(ctx, &decoder, &ones, 128, 128, 64, coder);
        let big = vec![-9999i32; 64 * 64];
        verify_against_cpu(ctx, &decoder, &big, 64, 64, 64, coder);
    }
}
