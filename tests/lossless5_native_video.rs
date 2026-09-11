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
//!
//! A third test covers 4:2:0, where the interleaved buffer is the only door into the sequence
//! encoder and the chroma has to survive both the resampling and the motion vectors.

use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::{ChromaFormat, ColorSpace, EntropyCoder, GpuContext};
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();

fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// Interleaved Y'CbCr that **translates**, so the P-frames have something to predict and the
/// assertion below is not secretly an all-intra test.
///
/// The texture is a function of the *shifted* column, noise included, so frame `f` is frame 0
/// scrolled by `3f` pixels and nothing else. Two earlier versions of this generator did not
/// survive the `Predicted` assertion, and both failures were about `q=100` rather than about the
/// colour space:
///
///   1. The noise was reseeded per frame, so motion compensation had nothing to find at all.
///   2. With the noise made positional but the picture still a smooth ramp, MED's intra
///      prediction coded it so cheaply that LOSSLESS-2 re-coded every P-frame as an I-frame
///      anyway — a correct decision about rate that left the inter path untested.
///
/// So the content is blocks *and* texture: expensive enough to intra-code that an exactly
/// matched P-frame wins (~114 kB I against ~30 kB P here), which is the regime this test needs
/// and the one `GNC_LOSSLESS_INTRA_RECODE` exists to force. Nothing here sets it — the
/// environment is process-global and cargo runs tests as threads (see
/// `tests/pframe_reference_drift.rs`), so the content has to earn its P-frames.
fn ycbcr_frames(w: usize, h: usize, n: usize) -> Vec<Vec<f32>> {
    // Position-seeded, so the value at column `c` is the same in every frame that shows it.
    let tex = |c: usize, j: usize| -> f32 {
        let mut s = 0x1357_9bdfu32
            .wrapping_add(c as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add((j as u32).wrapping_mul(40_503));
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((s >> 26) & 0x7) as f32
    };
    (0..n)
        .map(|f| {
            let mut out = vec![0.0f32; w * h * 3];
            let shift = f * 3;
            for j in 0..h {
                for i in 0..w {
                    let c = (i + shift) % w;
                    let blk = ((c / 32) + (j / 32) * 4) % 7;
                    let p = (j * w + i) * 3;
                    out[p] = (16.0 + (blk * 28) as f32 + tex(c, j)).round();
                    out[p + 1] = (128.0 + (blk as f32) * 7.0 - 20.0 + tex(c + 7, j)).round();
                    out[p + 2] = (128.0 + (blk as f32) * 5.0 - 15.0 + tex(c, j + 3)).round();
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

/// 4:2:0, where the interleaved buffer is the encoder's only door and the chroma has to survive
/// both the resampling and the motion vectors.
///
/// **Two claims, and the content is built for the second one.**
///
/// 1. The CLI reaches the sequence encoder with interleaved triples, so a 4:2:0 Y4M's half-size
///    chroma is nearest-neighbour replicated to luma resolution on the way in and the encoder's
///    own box filter takes it back down. That is exact — the average of four copies of one
///    integer is that integer — but it is exact because of what a *shader* does, so it is
///    asserted rather than assumed.
///
/// 2. The chroma motion vector is the luma one arithmetic-shifted right by a bit, so a *full-pel*
///    luma vector is a **half-pel chroma** vector and `bilinear_ref` puts back exactly the
///    fraction BUG-39's rounding exists to remove. The encoder therefore rounds luma vectors to
///    two pixels at 4:2:0, and this asserts the invariant directly: every coded component is a
///    multiple of 8 quarter-pels.
///
/// **A static background with one moving band** is what makes claim 2 testable. A synthetic that
/// pans as a whole cannot: at `q=100` LOSSLESS-2 compares each P-frame against an I-frame of the
/// same picture, an odd-pixel pan mis-predicts every block once the vectors are rounded, and the
/// P-frame is sent back to I before any chroma vector is used — measured at every odd step from
/// 1 to 5. With most of the frame static the P-frame wins on the still part and the moving band
/// still carries the odd vectors. Run against the pre-fix encoder this content reads a worst
/// sample error of **0.5** with 236-256 of 256 vectors not multiples of 8, so it discriminates.
#[test]
fn chroma_4_2_0_survives_the_interleaved_round_trip() {
    let (w, h, n) = (256usize, 256usize, 4usize);
    // One band moves 3 px per frame — odd, so the rounding has something to do.
    let frames = moving_band_ycbcr(w, h, n, 3);
    let refs: Vec<&[f32]> = frames.iter().map(|f| f.as_slice()).collect();

    let mut cfg = config(true);
    cfg.chroma_format = ChromaFormat::Yuv420;
    cfg.normalize_for_chroma();

    let mut enc = EncoderPipeline::new(gpu());
    let cf = enc.encode_sequence(gpu(), &refs, w as u32, h as u32, &cfg);
    assert!(
        cf.iter().any(|f| f.frame_type == gnc::FrameType::Predicted),
        "the clip coded all-intra, so no chroma motion vector was ever derived"
    );

    for (i, f) in cf.iter().enumerate() {
        let Some(mf) = f.motion_field.as_ref() else {
            continue;
        };
        let stray = mf
            .vectors
            .iter()
            .filter(|v| v[0] % 8 != 0 || v[1] % 8 != 0)
            .count();
        assert_eq!(
            stray, 0,
            "frame {i} carries {stray} of {} motion vectors that are not a multiple of 8 \
             quarter-pels, so their halves are half-pel in chroma and the prediction is \
             fractional (BUG-39 cause 4, chroma side)",
            mf.vectors.len()
        );
    }

    let decoded = DecoderPipeline::new(gpu()).decode_sequence(gpu(), &cf);
    for (i, (d, o)) in decoded.iter().zip(&frames).enumerate() {
        let worst_of = |c: usize| {
            o.iter()
                .skip(c)
                .step_by(3)
                .zip(d.iter().skip(c).step_by(3))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };
        let (y, cb, cr) = (worst_of(0), worst_of(1), worst_of(2));
        assert_eq!(
            (y, cb, cr),
            (0.0, 0.0, 0.0),
            "frame {i} is not bit-exact at 4:2:0: worst Y {y}, Cb {cb}, Cr {cr}"
        );
    }
}

/// A still picture with one horizontally moving band, chroma replicated 2x2 the way a 4:2:0
/// source looks once it has been interleaved.
fn moving_band_ycbcr(w: usize, h: usize, n: usize, step: usize) -> Vec<Vec<f32>> {
    let tex = |c: usize, j: usize| -> f32 {
        let mut s = 0x1357_9bdfu32
            .wrapping_add(c as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add((j as u32).wrapping_mul(40_503));
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((s >> 26) & 0x7) as f32
    };
    (0..n)
        .map(|f| {
            let mut out = vec![0.0f32; w * h * 3];
            for j in 0..h {
                for i in 0..w {
                    let c = if (h / 4..h / 2).contains(&j) {
                        i + f * step
                    } else {
                        i
                    };
                    let blk = ((c / 32) + (j / 32) * 4) % 7;
                    let p = (j * w + i) * 3;
                    out[p] = (16.0 + (blk * 28) as f32 + tex(c, j)).round();
                    out[p + 1] = (128.0 + (blk as f32) * 7.0 - 20.0 + tex(c + 7, j)).round();
                    out[p + 2] = (128.0 + (blk as f32) * 5.0 - 15.0 + tex(c, j + 3)).round();
                }
            }
            // Replicate each 2x2 chroma block from its top-left sample.
            let src = out.clone();
            for j in 0..h {
                for i in 0..w {
                    let s0 = ((j & !1) * w + (i & !1)) * 3;
                    let d = (j * w + i) * 3;
                    out[d + 1] = src[s0 + 1];
                    out[d + 2] = src[s0 + 2];
                }
            }
            out
        })
        .collect()
}
