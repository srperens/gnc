//! LOSSLESS-2 — a lossless P-frame that costs more than an I-frame must not be coded.
//!
//! At `q=100` every frame decodes bit-exact (BUG-39, `docs/decisions/0064`), so the two
//! candidates for one frame are the *same picture* and the choice between them is bytes alone.
//! Measured on real content: a P-frame costs **+74.6% to +80.8%** of its own I-frame on camera
//! content and **−2.8% to −3.3%** on animation. The encoder now re-codes the losing case as an
//! I-frame.
//!
//! What is verified here, on synthetic content chosen so the two directions are unambiguous:
//!   1. A smooth picture with fresh grain per frame — the P residual is the difference of two
//!      independent grain fields, so it costs more than intra-predicting the smooth picture —
//!      must come out **all-Intra**, and still bit-exact.
//!   2. A static sequence — the P residual is zero — must **keep its P-frames**, which is the
//!      half that proves the rule is a comparison and not a blanket "no P-frames at q=100".
//!   3. The gate: at q=99 nothing is bit-exact, the comparison is not free, and the frame types
//!      must be unchanged whatever the sizes are.

use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::{EntropyCoder, GpuContext};
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();

fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// A smooth picture plus a *fresh* small noise field per frame — the camera-content mechanism
/// in miniature. MED predicts the smooth part almost exactly, so an I-frame costs about one
/// noise field; the temporal difference of two independent fields has twice the variance, so a
/// P-frame costs about half a bit per sample more. That is why real camera content pays +75% to
/// +80% for a lossless P-frame.
///
/// **Not full-amplitude noise.** A first draft used uniform 0..255 per frame and the two
/// candidates came out within **±0.03%** of each other (245 309 B against 245 382 B): both paths
/// hit the same incompressible floor, so pure noise is a *tie* and cannot test a direction.
fn grainy(w: u32, h: u32, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut out = ramp(w, h);
    for v in out.iter_mut() {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let n = (((s >> 24) & 0x0f) as f32) - 7.5;
        *v = (*v + n).clamp(0.0, 255.0).round();
    }
    out
}

/// Something with structure, so the still path is not coding pure noise in the static test.
fn ramp(w: u32, h: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (w * h * 3) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = (y * w as usize + x) * 3;
            out[i] = ((x * 255) / w as usize) as f32;
            out[i + 1] = ((y * 255) / h as usize) as f32;
            out[i + 2] = (((x + y) * 255) / (w + h) as usize) as f32;
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn encode(frames: &[Vec<f32>], w: u32, h: u32, q: u32) -> Vec<gnc::CompressedFrame> {
    let refs: Vec<&[f32]> = frames.iter().map(|f| f.as_slice()).collect();
    let mut config = gnc::quality_preset(q);
    config.entropy_coder = EntropyCoder::Rice;
    config.keyframe_interval = 9;
    // Independent noise is a scene cut at every frame, and the detector would force keyframes
    // for a reason that has nothing to do with this item — the first draft of this test passed
    // on that instead of on the rule it claims to check. Off, so the P path is actually taken.
    config.scene_cut_threshold = 0.0;
    let mut enc = EncoderPipeline::new(gpu());
    enc.encode_sequence(gpu(), &refs, w, h, &config)
}

#[test]
fn a_grainy_sequence_is_coded_all_intra_at_q100() {
    let (w, h) = (256u32, 256u32);
    let frames: Vec<Vec<f32>> = (0..5).map(|i| grainy(w, h, 7 + i)).collect();
    let cf = encode(&frames, w, h, 100);

    let types: Vec<gnc::FrameType> = cf.iter().map(|f| f.frame_type).collect();
    assert!(
        types.iter().all(|t| *t == gnc::FrameType::Intra),
        "a P-frame whose residual is the difference of two independent grain fields costs more \
         than an I-frame of the same picture, so every frame should have been re-coded as \
         Intra; got {types:?}"
    );

    // The guarantee that makes the choice free must survive the re-code.
    let dec = DecoderPipeline::new(gpu());
    let decoded = dec.decode_sequence(gpu(), &cf);
    for (i, (d, o)) in decoded.iter().zip(&frames).enumerate() {
        assert_eq!(
            max_abs_diff(o, d),
            0.0,
            "frame {i} is not bit-exact after the LOSSLESS-2 re-code"
        );
    }
}

#[test]
fn a_static_sequence_keeps_its_p_frames_at_q100() {
    let (w, h) = (256u32, 256u32);
    let base = ramp(w, h);
    let frames: Vec<Vec<f32>> = (0..5).map(|_| base.clone()).collect();
    let cf = encode(&frames, w, h, 100);

    assert_eq!(
        cf[0].frame_type,
        gnc::FrameType::Intra,
        "frame 0 is the keyframe"
    );
    let inter: Vec<gnc::FrameType> = cf[1..].iter().map(|f| f.frame_type).collect();
    assert!(
        inter.iter().all(|t| *t == gnc::FrameType::Predicted),
        "a P-frame over a static picture codes a zero residual and must win, or LOSSLESS-2 has \
         become a blanket refusal of lossless P-frames rather than a comparison; got {inter:?}"
    );

    let dec = DecoderPipeline::new(gpu());
    let decoded = dec.decode_sequence(gpu(), &cf);
    for (i, (d, o)) in decoded.iter().zip(&frames).enumerate() {
        assert_eq!(max_abs_diff(o, d), 0.0, "frame {i} is not bit-exact");
    }
}

#[test]
fn the_comparison_does_not_run_below_q100() {
    let (w, h) = (256u32, 256u32);
    let frames: Vec<Vec<f32>> = (0..5).map(|i| grainy(w, h, 7 + i)).collect();
    let cf = encode(&frames, w, h, 99);

    let inter: Vec<gnc::FrameType> = cf[1..].iter().map(|f| f.frame_type).collect();
    assert!(
        inter.iter().all(|t| *t == gnc::FrameType::Predicted),
        "at q=99 the two candidates are not the same picture, so the byte comparison is not \
         free and must not run; got {inter:?}"
    );
}
