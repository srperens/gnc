//! BUG-45 — a lossless configuration must be bit-exact on a *fractional* source too.
//!
//! `is_lossless()` is a statement about the settings. Bit-exactness also needs integer samples,
//! and until 2026-09-10 the encoder only *warned* when it did not have them. That warning fired
//! on every frame of every Y4M encode and was coded through anyway: the Y4M reader's BT.601
//! matrix is `1.164*(Y-16)` and friends, so **98.9% of the samples in a 1080p frame arrive
//! fractional** — even at `C444`, because the matrix and not the chroma upsample is what makes
//! them. Measured on `bbb.y4m` before the fix: `benchmark-sequence -q 100` returned **PSNR
//! 32.89 dB at 8.42 bpp**, drifting 32.48–33.16 frame to frame, which is visible as a tile-shaped
//! flicker in a player. After: **PSNR inf on four sequences**.
//!
//! The mechanism is `0080`'s, one producer over. The step-1 quantiser rounds the *residual*, and
//! `med_predict.wgsl` is open-loop by design — the encoder predicts from `src` while the decoder
//! predicts from its own reconstruction — so the rounding accumulates along each tile's diagonal
//! scan instead of staying a half-LSB.
//!
//! What is asserted here is the invariant, not a number: **a lossless encode of a fractional
//! source reproduces `round(source)` exactly.** Rounding is what the encoder now commits to, so
//! that is the picture it must return.

use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::{EntropyCoder, GpuContext};
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();

fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// A picture whose samples land off an integer the way a BT.601 conversion leaves them —
/// a gradient with structure, plus a per-channel fraction that varies from pixel to pixel.
///
/// **Varying fractions are load-bearing.** A source whose samples all share one fractional part
/// has integral MED residuals and codes exactly even with the bug present; that is the trap
/// `tests/bug49_subsampled_lossless.rs` documents from the other direction.
fn fractional(w: u32, h: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (w * h * 3) as usize];
    let mut s = 0x9e37_79b9u32;
    for y in 0..h as usize {
        for x in 0..w as usize {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let i = (y * w as usize + x) * 3;
            let base = ((x * 90) / w as usize + (y * 70) / h as usize) as f32 + 50.0;
            for c in 0..3 {
                let frac = ((s >> (7 * c + 3)) & 0x7) as f32 / 8.0;
                out[i + c] = base + ((s >> (5 * c + 1)) & 0xf) as f32 + frac;
            }
        }
    }
    out
}

fn roundtrip(img: &[f32], w: u32, h: u32) -> Vec<f32> {
    let mut config = gnc::quality_preset(100);
    config.entropy_coder = EntropyCoder::Rice;
    assert!(config.is_lossless(), "q=100 must be a lossless configuration");
    let mut enc = EncoderPipeline::new(gpu());
    let frame = enc.encode(gpu(), img, w, h, &config);
    DecoderPipeline::new(gpu()).decode(gpu(), &frame)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn q100_on_a_fractional_source_reproduces_the_rounded_source_exactly() {
    let (w, h) = (512u32, 512u32);
    let img = fractional(w, h);
    assert!(
        img.iter().filter(|v| v.fract() != 0.0).count() > img.len() / 2,
        "the test picture must actually be fractional, or it proves nothing"
    );

    let decoded = roundtrip(&img, w, h);
    let expected: Vec<f32> = img.iter().map(|v| v.round()).collect();

    let d = max_abs_diff(&expected, &decoded);
    assert_eq!(
        d, 0.0,
        "q=100 on a fractional source is not bit-exact against round(source): max abs error {d}. \
         Before BUG-45 was fixed this read tens of levels, drifting across each tile (0081)."
    );
}

/// The control: an integral source was always exact and must stay that way.
#[test]
fn q100_on_an_integral_source_is_untouched() {
    let (w, h) = (512u32, 512u32);
    let img: Vec<f32> = fractional(w, h).iter().map(|v| v.round()).collect();
    let decoded = roundtrip(&img, w, h);
    assert_eq!(
        max_abs_diff(&img, &decoded),
        0.0,
        "q=100 on an integral source is no longer bit-exact"
    );
}
