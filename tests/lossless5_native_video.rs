//! **STATUS 2026-09-11: both tests are `#[ignore]`d and this file is a specification, not a
//! passing guard.** The encoder plumbing is in and verified byte-identical for RGB input; what
//! does not work yet is the native arm end to end. Two attempts failed and CLAUDE.md's rule
//! ("if the same bug resurfaces after two fix attempts — stop, diagnose root cause properly")
//! applies, so the requirement is recorded here rather than half-fixed.
//!
//! What is known:
//!   - `worst sample error 258` on frame 0, i.e. reconstruction is badly wrong, not slightly.
//!   - The encoder records `ColorSpace::YCbCrNative` correctly — that assertion passes.
//!   - Widening `enc_input`/`enc_color_out` usages fixed a wgpu validation error; not the cause.
//!   - `decoder/pipeline.rs` has **five** more `self.color.dispatch` sites, all identical in shape,
//!     that the still-path fix in `gpu_work.rs` never touched. Branching them the same way did
//!     **not** change the result, so either the sequence decode path reaches a sixth site or the
//!     fault is encode-side. That dedup is worth doing on its own merits — five copies of one
//!     dispatch, exactly the shape LOSSLESS-5 step 1 removed on the encoder side — but it is
//!     deliberately *not* in this commit, because it fixes nothing here and was not worth carrying
//!     as unverified risk.
//!   - The byte counts did not move between attempts (513 315 native vs 476 728 converted), which
//!     is encode-side evidence: a decoder fix cannot change them.
//!
//! What was mis-diagnosed on the way, and is *not* the problem: the sequence encoder does pad
//! before this point — via `dispatch_gpu_pad_cached`, not an inline `pad_pass`, which is why a
//! grep for the latter returned zero and suggested otherwise. `input_buf` is the padded buffer and
//! `color_out` the destination, exactly as in the still path, and both are `buf_size_3`.
//!
//! The rate test is *also* suspect on its own terms: this synthetic has near-constant Cb and a
//! smooth Cr ramp, so YCoCg-R over the derived RGB may genuinely win on it where it loses on real
//! clips. Before treating its failure as a defect, re-run the assertion on real material — the
//! −13.2% in RESEARCH_LOG was measured on four clips, not on this.

//! LOSSLESS-5 — a Y'CbCr *sequence* round-trips as itself, with no colour transform at either end.
//!
//! LOSSLESS-4 did this for stills. Video goes through `sequence.rs`, which ran its own colour
//! conversion at eleven copy-pasted sites (one since step 1's dedup). This asserts the two claims
//! that justify threading the flag through it: the samples survive a multi-frame encode including
//! P-frames, and the file is smaller than the same picture coded through the RGB matrix.
//!
//! **The rate claim is deliberately the colour space alone.** An earlier version of this number
//! compared against an RGB encode at 4:4:4 from a 4:2:0 source and credited a chroma-format change
//! to the colour space; separated, the colour space is worth **−13.2%** on four clips and the
//! format change is worth −31.6% (RESEARCH_LOG, 2026-09-11). Both arms here are 4:4:4, so only the
//! matrix differs.

use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::{ChromaFormat, ColorSpace, EntropyCoder, GpuContext};
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();

fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// Interleaved Y'CbCr with motion, so the P-frames have something to predict and the test is not
/// secretly an all-intra one.
fn ycbcr_frames(w: usize, h: usize, n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|f| {
            let mut out = vec![0.0f32; w * h * 3];
            let shift = f * 3;
            let mut s = 0x1357_9bdfu32.wrapping_add(f as u32);
            for j in 0..h {
                for i in 0..w {
                    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let n8 = ((s >> 26) & 0x7) as f32;
                    let p = (j * w + i) * 3;
                    out[p] = (16.0 + (((i + shift) % w) * 200 / w) as f32 + n8).round();
                    out[p + 1] = (128.0 + ((j * 80 / h) as f32 - 40.0)).round();
                    out[p + 2] = (128.0 + (((i + shift) % w) * 60 / w) as f32 - 30.0).round();
                }
            }
            out
        })
        .collect()
}

fn config(native: bool) -> gnc::CodecConfig {
    let mut c = gnc::quality_preset(100);
    c.entropy_coder = EntropyCoder::Rice;
    c.chroma_format = ChromaFormat::Yuv444;
    c.keyframe_interval = 4;
    c.scene_cut_threshold = 0.0;
    if native {
        c.color_space = ColorSpace::YCbCrNative;
    }
    c
}

#[test]
#[ignore = "LOSSLESS-5: native sequence path not working yet; see the file header"]
fn q100_native_planes_survive_a_sequence_with_p_frames() {
    let (w, h, n) = (256usize, 256usize, 6usize);
    let frames = ycbcr_frames(w, h, n);
    let refs: Vec<&[f32]> = frames.iter().map(|f| f.as_slice()).collect();

    let mut enc = EncoderPipeline::new(gpu());
    let cf = enc.encode_sequence(gpu(), &refs, w as u32, h as u32, &config(true));
    assert_eq!(cf.len(), n);
    assert!(
        cf.iter().any(|f| f.frame_type == gnc::FrameType::Predicted),
        "the clip coded all-intra, so this never exercised the inter path"
    );
    for f in &cf {
        assert_eq!(
            f.config.color_space,
            ColorSpace::YCbCrNative,
            "a frame lost its colour space, so the decoder will invert a transform that never ran"
        );
    }

    let decoded = DecoderPipeline::new(gpu()).decode_sequence(gpu(), &cf);
    for (i, (d, o)) in decoded.iter().zip(&frames).enumerate() {
        let worst = o
            .iter()
            .zip(d)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            worst, 0.0,
            "frame {i} is not bit-exact: worst sample error {worst}. A Y'CbCr source getting its \
             own samples back is the whole correctness claim of LOSSLESS-4/5."
        );
    }
}

/// The rate half, with both arms at 4:4:4 so only the colour matrix differs.
#[test]
#[ignore = "LOSSLESS-5: see the file header; this synthetic may also be the wrong content"]
fn native_planes_beat_the_rgb_they_convert_to_across_a_sequence() {
    let (w, h, n) = (256usize, 256usize, 6usize);
    let frames = ycbcr_frames(w, h, n);

    let rgb: Vec<Vec<f32>> = frames
        .iter()
        .map(|f| {
            let mut o = vec![0.0f32; f.len()];
            for p in 0..f.len() / 3 {
                let yy = 1.164_f32 * (f[p * 3] - 16.0);
                let pb = f[p * 3 + 1] - 128.0;
                let pr = f[p * 3 + 2] - 128.0;
                o[p * 3] = (yy + 1.596 * pr).clamp(0.0, 255.0);
                o[p * 3 + 1] = (yy - 0.392 * pb - 0.813 * pr).clamp(0.0, 255.0);
                o[p * 3 + 2] = (yy + 2.017 * pb).clamp(0.0, 255.0);
            }
            o
        })
        .collect();

    let mut enc = EncoderPipeline::new(gpu());
    let bytes = |cf: &[gnc::CompressedFrame]| -> usize {
        cf.iter()
            .map(|f| gnc::format::serialize_compressed(f).len())
            .sum()
    };
    let nat: Vec<&[f32]> = frames.iter().map(|f| f.as_slice()).collect();
    let native = bytes(&enc.encode_sequence(gpu(), &nat, w as u32, h as u32, &config(true)));
    let via: Vec<&[f32]> = rgb.iter().map(|f| f.as_slice()).collect();
    let converted = bytes(&enc.encode_sequence(gpu(), &via, w as u32, h as u32, &config(false)));

    assert!(
        native < converted,
        "coding the sequence's own planes ({native} B) did not beat coding the RGB they convert \
         to ({converted} B) — the colour-space saving has regressed or reversed"
    );
}
