//! Diagnostics must not change the bitstream.
//!
//! They did: the temporal-wavelet diagnostic ran a second full wavelet transform through the
//! encoder's shared GPU buffers, clobbering the motion-compensation reference. Every P-frame
//! after the second then encoded far worse — a 32% larger file on blue_sky at q=50, with
//! residuals as large as the raw frame difference, and false "temporal prediction may not be
//! effective" warnings to match. Measurements taken with `GNC_DIAGNOSTICS=1` were measuring
//! the observer.
//!
//! This test synthesises its own frames so it runs without test material.

use gnc::encoder::pipeline::EncoderPipeline;
use gnc::format::serialize_sequence;
use gnc::GpuContext;
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();
fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// A translating pattern with enough detail that motion compensation matters: get the
/// reference wrong and the residual jumps to the size of the raw frame difference.
fn synth_frame(idx: usize, w: u32, h: u32) -> Vec<f32> {
    let shift = (idx * 5) as f32;
    let mut out = vec![0.0f32; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 + shift;
            let fy = y as f32;
            let v = (0.5
                + 0.25 * ((fx * 0.11).sin() * (fy * 0.07).cos())
                + 0.15 * ((fx * 0.31 + fy * 0.23).sin()))
            .clamp(0.0, 1.0);
            // Pixels are 0..255 floats, as `load_image_rgb_f32` produces.
            let i = ((y * w + x) * 3) as usize;
            out[i] = v * 255.0;
            out[i + 1] = (v * 0.8 + 0.1).clamp(0.0, 1.0) * 255.0;
            out[i + 2] = (1.0 - v * 0.6).clamp(0.0, 1.0) * 255.0;
        }
    }
    out
}

fn encode_all(diagnostics: bool, w: u32, h: u32, n: usize) -> Vec<u8> {
    gnc::encoder::diagnostics::set_enabled(diagnostics);
    let ctx = gpu();
    let mut encoder = EncoderPipeline::new(ctx);
    let mut config = gnc::quality_preset(50);
    // The default preset is all-intra (keyframe_interval = 1). The corruption lives in the
    // P-frame path, so the sequence has to actually contain P-frames.
    config.keyframe_interval = 9;
    let frames = encoder.encode_sequence_streaming(
        ctx,
        n,
        |i| synth_frame(i, w, h),
        w,
        h,
        &config,
        30.0,
    );
    // Per-frame sizes make a failure readable: the corruption's signature is one collapsed
    // frame followed by several inflated ones, not a uniform shift.
    eprintln!(
        "diagnostics={diagnostics}: frame sizes {:?}",
        frames.iter().map(|f| f.byte_size()).collect::<Vec<_>>()
    );
    serialize_sequence(&frames, (30, 1))
}

#[test]
fn diagnostics_do_not_change_the_bitstream() {
    // The corruption only showed from the third frame on, so a short sequence would pass
    // regardless of whether the bug is present.
    let (w, h, n) = (512u32, 512u32, 6usize);
    let quiet = encode_all(false, w, h, n);
    let loud = encode_all(true, w, h, n);
    gnc::encoder::diagnostics::set_enabled(false);
    assert_eq!(
        quiet.len(),
        loud.len(),
        "enabling diagnostics changed the encoded size ({} -> {} bytes, {:+.1}%) — a diagnostic \
         is writing through encoder state",
        quiet.len(),
        loud.len(),
        (loud.len() as f64 / quiet.len() as f64 - 1.0) * 100.0
    );
    assert!(
        quiet == loud,
        "enabling diagnostics changed the bitstream at equal length"
    );
}
