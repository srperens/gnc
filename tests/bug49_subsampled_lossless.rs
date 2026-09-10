//! BUG-49 — `q=100` on subsampled input must not drift across a tile.
//!
//! `chroma_downsample.wgsl` box-averages 2 samples (4:2:2) or 4 (4:2:0), so its output is a
//! multiple of 0.5 or 0.25 — *fractional*. A lossless configuration codes that with a step-1
//! quantiser, which rounds; and `med_predict.wgsl` predicts from `src` while the decoder
//! predicts from its own reconstruction, so the rounding error is not a half-LSB but the seed
//! of a DPCM drift that grows along the scan inside each tile.
//!
//! Measured on bbb_1080p at q=100 4:2:2 before the fix, mean |Co error| by distance from the
//! tile origin: **0.40 at the origin, 1.39 at 4-15, 2.55 at 16-63, 6.38 at 192-255**, peak 34.
//! dE00 was 2.7x to 20x worse than q=99 at the same chroma format, and 4:2:2 — which keeps
//! twice the chroma of 4:2:0 — came out *worse* than 4:2:0 on all three stills.
//!
//! The fix rounds the averaged plane to integers when the configuration is lossless, so the
//! plane the encoder commits to is one a lossless transform can carry exactly. What this test
//! asserts is the shape of the failure rather than a recorded number: **error must not grow
//! with distance from the tile origin.** A flat profile is the signature of a half-LSB round;
//! a rising one is the signature of an accumulating prediction mismatch, which is the defect.
//!
//! Mutation-tested: removing the `round` from `chroma_downsample.wgsl` reproduces the real
//! BUG-49 failure here — far-corner error over near-origin error goes to **2.25x (4:2:2) and
//! 1.98x (4:2:0)**, against 0.98x and 0.99x with the fix in. The threshold is 1.5, which both
//! arms clear by a wide margin in both directions.

use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::{ChromaFormat, EntropyCoder, GpuContext};
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();

fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// A picture whose chroma is *fractional under the box filter but almost lossless to
/// subsample*, which is what isolates the drift from the subsampling error it hides behind.
///
/// `R = b + d`, `G = b + e`, `B = b` with `d`,`e` a one-bit dither gives `Co = d`, `Cg = e`,
/// `Y = b` in YCoCg-R. The box averages then land on 0, 0.5 and 1 — *varying* fractional parts,
/// which is the trigger: a plane whose samples all share one fractional part has integral MED
/// residuals and codes exactly even with the bug present. Meanwhile the subsampled picture is
/// never more than one level from the original, so a correct encode leaves a fraction of a
/// level and the drift has nothing to hide behind.
///
/// Two earlier drafts could not detect the bug at all, and both failures are worth keeping:
/// per-pixel colour *noise* buried a ~4-level drift under ~5.5 levels of subsampling error,
/// and a `(x&1)` dither made every box average exactly 0.5, so every residual was integral.
///
/// `b` carries a gradient and mild noise so the luma plane is not trivially predictable;
/// shifting R, G and B together moves `Y` alone and leaves `Co`/`Cg` untouched.
fn chroma_detail(w: u32, h: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (w * h * 3) as usize];
    let mut s = 0x1234_5678u32;
    for y in 0..h as usize {
        for x in 0..w as usize {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = (((s >> 24) & 0x0f) as f32) - 7.5;
            let b = (((x * 100) / w as usize + (y * 60) / h as usize) as f32 + 60.0 + n).round();
            let i = (y * w as usize + x) * 3;
            out[i] = b + ((s >> 9) & 7) as f32;
            out[i + 1] = b + ((s >> 17) & 7) as f32;
            out[i + 2] = b;
        }
    }
    out
}

/// Mean absolute RGB error grouped by Chebyshev distance from the origin of the chroma plane's
/// own 256x256 tile — the coordinate the MED chain restarts on.
fn error_by_tile_distance(
    orig: &[f32],
    dec: &[f32],
    w: u32,
    h: u32,
    tile: u32,
    shift_x: u32,
    shift_y: u32,
) -> Vec<(u32, f64)> {
    let bands = [(0u32, 31u32), (32, 95), (96, 191), (192, 255)];
    bands
        .iter()
        .map(|&(lo, hi)| {
            let (mut sum, mut n) = (0.0f64, 0usize);
            for y in 0..h {
                for x in 0..w {
                    let d = ((x >> shift_x) % tile).max((y >> shift_y) % tile);
                    if d < lo || d > hi {
                        continue;
                    }
                    let i = ((y * w + x) * 3) as usize;
                    for c in 0..3 {
                        sum += (orig[i + c] - dec[i + c]).abs() as f64;
                        n += 1;
                    }
                }
            }
            (lo, if n == 0 { 0.0 } else { sum / n as f64 })
        })
        .collect()
}

fn drift_ratio(fmt: ChromaFormat, shift_x: u32, shift_y: u32) -> (f64, Vec<(u32, f64)>) {
    // 512x512 so the 4:2:0 chroma plane is a full 256x256 tile: a smaller frame cannot show a
    // drift that only becomes visible far from the tile origin.
    let (w, h) = (512u32, 512u32);
    let img = chroma_detail(w, h);

    let mut config = gnc::quality_preset(100);
    config.chroma_format = fmt;
    config.entropy_coder = EntropyCoder::Rice;
    assert!(
        config.is_lossless(),
        "q=100 must be a lossless configuration or this test is measuring something else"
    );

    let mut enc = EncoderPipeline::new(gpu());
    let frame = enc.encode(gpu(), &img, w, h, &config);
    let dec = DecoderPipeline::new(gpu()).decode(gpu(), &frame);

    let bands = error_by_tile_distance(&img, &dec, w, h, config.tile_size, shift_x, shift_y);
    let near = bands[0].1;
    let far = bands[bands.len() - 1].1;
    // A picture this codec reproduces perfectly would divide 0 by 0; the synthetic above always
    // loses something to the subsample, so `near` is comfortably above zero.
    assert!(near > 0.0, "no error at all near the tile origin: {bands:?}");
    (far / near, bands)
}

#[test]
fn q100_422_chroma_error_does_not_grow_across_a_tile() {
    let (ratio, bands) = drift_ratio(ChromaFormat::Yuv422, 1, 0);
    assert!(
        ratio < 1.5,
        "4:2:2 q=100 error grows {ratio:.2}x from the tile origin to its far corner — that is \
         MED drift from a fractional chroma plane (BUG-49), not a half-LSB round. Bands \
         (distance, mean |RGB err|): {bands:?}"
    );
}

#[test]
fn q100_420_chroma_error_does_not_grow_across_a_tile() {
    let (ratio, bands) = drift_ratio(ChromaFormat::Yuv420, 1, 1);
    assert!(
        ratio < 1.5,
        "4:2:0 q=100 error grows {ratio:.2}x from the tile origin to its far corner — that is \
         MED drift from a fractional chroma plane (BUG-49), not a half-LSB round. Bands \
         (distance, mean |RGB err|): {bands:?}"
    );
}

/// The 4:4:4 control: nothing in this fix may touch the path that was already exact.
#[test]
fn q100_444_stays_bit_exact() {
    let (w, h) = (512u32, 512u32);
    let img = chroma_detail(w, h);
    let mut config = gnc::quality_preset(100);
    config.chroma_format = ChromaFormat::Yuv444;
    config.entropy_coder = EntropyCoder::Rice;

    let mut enc = EncoderPipeline::new(gpu());
    let frame = enc.encode(gpu(), &img, w, h, &config);
    let dec = DecoderPipeline::new(gpu()).decode(gpu(), &frame);

    let max = img
        .iter()
        .zip(&dec)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(max, 0.0, "q=100 4:4:4 is no longer bit-exact");
}
