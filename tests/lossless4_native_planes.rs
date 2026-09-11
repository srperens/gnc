//! LOSSLESS-4 — a Y'CbCr source round-trips as itself, and costs fewer bits than its RGB.
//!
//! GNC's historical path converts RGB to YCoCg-R, which is right for an RGB source and wrong for
//! a Y4M one. The BT.601 matrix that produced that RGB is not integer-invertible, so the file's
//! own samples cannot come back however good the codec is; and coding the matrix's output costs
//! **39.2% more bits** than coding the planes as they arrived (four sequences at q=100,
//! 2026-09-11, and independently matching FFV1's own +64.2% penalty for coding RGB).
//!
//! GP21 records which the planes are. What is asserted here is the pair of claims that justified
//! the generation bump: the samples survive, and the file is smaller.

use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::{ChromaFormat, ColorSpace, EntropyCoder, GpuContext};
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();

fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// Y'CbCr planes with real structure: a luma gradient with texture, and chroma that varies
/// independently of it, so a colour transform cannot be a no-op on this content.
fn planes(w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0f32; w * h];
    let mut cb = vec![0.0f32; w * h];
    let mut cr = vec![0.0f32; w * h];
    let mut s = 0x5eed_1234u32;
    for j in 0..h {
        for i in 0..w {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = ((s >> 25) & 0x7) as f32;
            y[j * w + i] = (16.0 + (i * 200 / w) as f32 + n).round();
            cb[j * w + i] = (128.0 + ((j * 90 / h) as f32 - 45.0)).round();
            cr[j * w + i] = (128.0 + ((i * 70 / w) as f32 - 35.0)).round();
        }
    }
    (y, cb, cr)
}

fn config(q: u32) -> gnc::CodecConfig {
    let mut c = gnc::quality_preset(q);
    c.entropy_coder = EntropyCoder::Rice;
    c.chroma_format = ChromaFormat::Yuv444;
    c
}

#[test]
fn q100_native_planes_come_back_exactly() {
    let (w, h) = (512usize, 512usize);
    let (y, cb, cr) = planes(w, h);

    let mut enc = EncoderPipeline::new(gpu());
    let frame = enc.encode_planar(gpu(), &y, &cb, &cr, w as u32, h as u32, &config(100));
    assert_eq!(
        frame.config.color_space,
        ColorSpace::YCbCrNative,
        "encode_planar must record the colour space, or the decoder will invert a transform that \
         never happened and return a plausible wrong picture"
    );

    // The decoder emits interleaved planes; for a native source those are Y'CbCr, not RGB.
    let out = DecoderPipeline::new(gpu()).decode(gpu(), &frame);
    assert_eq!(out.len(), w * h * 3);

    let mut worst = 0.0f32;
    for p in 0..w * h {
        for (c, src) in [&y, &cb, &cr].iter().enumerate() {
            worst = worst.max((out[p * 3 + c] - src[p]).abs());
        }
    }
    assert_eq!(
        worst, 0.0,
        "q=100 on native planes is not bit-exact: worst sample error {worst}. This is the whole \
         correctness claim of LOSSLESS-4 — that a Y'CbCr source gets its own samples back."
    );
}

/// The rate half. Coding the planes must beat coding the RGB they convert to — that is the 39.2%
/// the item was re-priced on, and a regression here would be silent otherwise.
#[test]
fn native_planes_cost_fewer_bits_than_the_rgb_they_convert_to() {
    let (w, h) = (512usize, 512usize);
    let (y, cb, cr) = planes(w, h);

    // The same picture as RGB, through BT.601 the way the Y4M reader does it.
    let mut rgb = vec![0.0f32; w * h * 3];
    for p in 0..w * h {
        let yy = 1.164_f32 * (y[p] - 16.0);
        let pb = cb[p] - 128.0;
        let pr = cr[p] - 128.0;
        rgb[p * 3] = (yy + 1.596 * pr).clamp(0.0, 255.0);
        rgb[p * 3 + 1] = (yy - 0.392 * pb - 0.813 * pr).clamp(0.0, 255.0);
        rgb[p * 3 + 2] = (yy + 2.017 * pb).clamp(0.0, 255.0);
    }

    let mut enc = EncoderPipeline::new(gpu());
    let native = gnc::format::serialize_compressed(&enc.encode_planar(
        gpu(),
        &y,
        &cb,
        &cr,
        w as u32,
        h as u32,
        &config(100),
    ))
    .len();
    let via_rgb =
        gnc::format::serialize_compressed(&enc.encode(gpu(), &rgb, w as u32, h as u32, &config(100)))
            .len();

    assert!(
        native < via_rgb,
        "coding native planes ({native} B) did not beat coding the RGB they convert to \
         ({via_rgb} B) — the 39.2% that justified GP21 has regressed or reversed"
    );
}
