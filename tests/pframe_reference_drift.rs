//! BUG-25 — the encoder's P-frame reference must be reconstructed with the residual quantiser
//! step, not the intra one.
//!
//! `encode_pframe` quantises P residuals at `res_qstep = quantization_step * p_qp_scale`
//! (TUNE-6's taper) and tells the decoder the same value. Until 2026-09-07 its six *local
//! decode* dequantise dispatches read `config.quantization_step` instead, so the reference the
//! encoder predicted from differed from the decoder's by `quantization_step / res_qstep`. Every
//! P-frame that predicts from another P then inherits a picture no decoder holds, and the error
//! compounds down the GOP.
//!
//! **Why this test looks at the shape of a GOP rather than at one number, and why the threshold
//! is 1.5 dB rather than zero.** A plain quantiser cannot accumulate error in a closed loop —
//! each frame's error is the quantisation error of its own residual, bounded independently of
//! GOP position. A *dead zone* can: a coefficient it zeroes leaves an error the loop has no way
//! to correct, and the next residual carries that error back into the same dead zone and has it
//! zeroed again. Inter residuals get twice the intra dead zone (`GNC_INTER_DZ_MUL`), so some
//! slow monotone drift along a GOP is legitimate here and this test must not forbid it.
//!
//! What separates the two is magnitude, measured on this exact input at `07c01b1` and after the
//! fix: **the defect drifts 4.04 dB over eight P-frames (36.51 -> 32.47) where the dead zone
//! alone drifts 0.86 dB (36.51 -> 35.65)**. 1.5 dB sits between them with room on both sides.
//! On real content the defect was far larger — crowd_run q=90 ki=9 at scale 1.25 read
//! 47.95 / 41.10 / 38.27 / … / 34.14 before reverting to 49.24 at the next I-frame.
//!
//! **Why it uses q=50 and no environment variable.** The taper is keyed on the quantiser step,
//! and `p_qp_scale` is 1.0 for every step at or below 2.8 — i.e. for all q >= 85, where the
//! wrong value and the right one coincide and the defect is invisible. q=50 is qstep 8.0, well
//! into the 1.25 region, so this exercises the **shipped default** rather than an override.
//! `GNC_P_QP_SCALE` would reach it too, but the environment is process-global and cargo runs
//! `#[test]` bodies as threads: a `set_var` here would change what a concurrently running test
//! encodes, which is precisely the race that masked a real decoder bug in `abac_bitstream`
//! (COORDINATION, BUG-21). Nothing in this file touches the environment.

use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::bench::quality;
use gnc::{EntropyCoder, GpuContext};
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();
fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// Textured content — a flat field would quantise to nothing and hide any drift.
fn synth(w: u32, h: u32) -> Vec<f32> {
    let mut d = Vec::with_capacity((w * h * 3) as usize);
    let mut rng: u32 = 0x9e37_79b9;
    for y in 0..h {
        for x in 0..w {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let n = (rng % 24) as f32 - 12.0;
            let fine = if ((x / 2) + (y / 3)) % 2 == 0 { 18.0 } else { 0.0 };
            let coarse = ((x / 40) * 37 % 200) as f32;
            let ramp = y as f32 / h as f32 * 120.0;
            d.push((coarse + ramp + fine + n).clamp(0.0, 255.0).round());
            d.push((ramp * 1.4 + fine + n).clamp(0.0, 255.0).round());
            d.push((255.0 - coarse + n).clamp(0.0, 255.0).round());
        }
    }
    d
}

/// A pure horizontal pan, so motion estimation finds an exact match and the only thing that can
/// degrade a later P-frame is the reference it predicted from.
fn panned(base: &[f32], w: u32, h: u32, dx: usize) -> Vec<f32> {
    let mut f = vec![0.0f32; base.len()];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let sx = (x + dx) % w as usize;
            for c in 0..3 {
                f[(y * w as usize + x) * 3 + c] = base[(y * w as usize + sx) * 3 + c];
            }
        }
    }
    f
}

#[test]
fn p_frame_quality_does_not_decay_along_a_gop() {
    let ctx = gpu();
    let (w, h) = (256u32, 256u32);
    let base = synth(w, h);
    let frames: Vec<Vec<f32>> = (0..9).map(|i| panned(&base, w, h, i * 2)).collect();
    let refs: Vec<&[f32]> = frames.iter().map(|f| f.as_slice()).collect();

    // q=50 -> qstep 8.0 -> p_qp_scale 1.25. ki=9 over 9 frames is one I followed by eight Ps,
    // so the last P is seven references deep — the longest chain the defect could compound over.
    let mut config = gnc::quality_preset(50);
    config.entropy_coder = EntropyCoder::Rice;
    config.keyframe_interval = 9;

    let mut enc = EncoderPipeline::new(ctx);
    let cf = enc.encode_sequence(ctx, &refs, w, h, &config);
    let dec = DecoderPipeline::new(ctx);
    let decoded = dec.decode_sequence(ctx, &cf);

    let psnr: Vec<f64> = decoded
        .iter()
        .zip(&frames)
        .map(|(d, o)| quality::psnr(o, d, 255.0))
        .collect();

    assert_eq!(
        cf[0].frame_type,
        gnc::FrameType::Intra,
        "frame 0 must be the I-frame this GOP predicts from"
    );
    assert!(
        cf[1..].iter().all(|f| f.frame_type == gnc::FrameType::Predicted),
        "frames 1..8 must all be P-frames, or this tests nothing: {:?}",
        cf.iter().map(|f| f.frame_type).collect::<Vec<_>>()
    );

    // The first P predicts from the I-frame, where both sides agree even with the defect present,
    // so it is the reference point rather than part of the evidence. Every later P is compared
    // against it: a mismatch shows up as monotone decay away from this value.
    let first_p = psnr[1];
    let last_p = psnr[8];
    let worst_p = psnr[1..].iter().copied().fold(f64::INFINITY, f64::min);

    // 1.5 dB: the defect drifts 4.04 dB on this input, the dead zone alone 0.86 dB (see the
    // module comment). Asserting the drift rather than an absolute dB figure keeps this a gate on
    // the mismatch and not on the codec's rate/quality point, which is free to move.
    assert!(
        first_p - worst_p < 1.5,
        "P-frame quality decays along the GOP: first P {:.2} dB, worst P {:.2} dB, last P \
         {:.2} dB (all frames: {:?}). This is more drift than the inter dead zone accounts \
         for, so it is an encoder/decoder reference mismatch — check that every dequantise in \
         encode_pframe's local decode uses res_qstep and not config.quantization_step (BUG-25).",
        first_p,
        worst_p,
        last_p,
        psnr.iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>()
    );
}
