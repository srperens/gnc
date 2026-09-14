//! BUG-57 — a `q=100` P-frame must keep the residual of a tile that is mostly padding.
//!
//! The tile skip pass declares a tile static when its **mean** per-pixel zero-MV SAD falls under
//! `qstep/2`, and then deletes that tile's quantised coefficients: the decoder reconstructs it
//! from the motion prediction with no residual at all. That is a rate/quality trade, and at the
//! bit-exact rung there is no quality to trade.
//!
//! **The mean is what makes a corner tile lie.** A tile on the right or bottom edge is padded out
//! to `tile_size`, and the padding is identical between frames — SAD exactly zero — so it drags
//! the mean down in proportion to how much of the tile is padding. At 1080p with `tile_size=256`
//! the bottom-right tile is 128x56 of picture inside 256x256, i.e. **89% padding**, and on
//! `bbb.y4m` at q=100 that was enough to declare it static while its real 11% still differed by
//! +/-1: frame 2 decoded at 75.31 dB with 3 606 luma samples wrong, all inside x 1792..1919,
//! y 1024..1079. `--tile-size 128` made it vanish, because the same corner is then 56 of 128 rows.
//!
//! This test reproduces the geometry at a size that runs in a fraction of a second: 320x300 pads
//! to 512x512, so the corner tile is **4.3% picture**. Against the pre-fix encoder it fails with
//! *2 709 samples wrong over x 256..318, y 256..298* — the corner rectangle exactly, and nothing
//! else.
//!
//! **Two things about the content are load-bearing, and both were got wrong first.**
//!
//! 1. The change must stop before the last row and column. Padding replicates the picture's edge,
//!    so an edge that moves makes the padding move with it, the tile's mean stops being a lie, and
//!    the tile is never declared static.
//! 2. The change must be one **no motion vector can predict**. The first version alternated a
//!    checkerboard, `(i + j + f) % 2` — which is the previous frame shifted by one pixel, so the
//!    search found it, `mean_mc` collapsed, and the shader's second condition
//!    (`mean_sad <= mean_mc * (1 + margin)`) correctly refused the skip. A uniform per-frame
//!    offset over the interior is invisible to the search and reaches the branch under test.
//!
//! `GNC_SKIP_DIAG=1` prints the skip-tile count per P-frame; on this content it reads 2 of 4.

use gnc::decoder::pipeline::DecoderPipeline;
use gnc::encoder::pipeline::EncoderPipeline;
use gnc::{ChromaFormat, ColorSpace, EntropyCoder, GpuContext};
use std::sync::OnceLock;

static GPU: OnceLock<GpuContext> = OnceLock::new();

fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// A moving band across the top, so the frame earns its P-frame, and a bottom-right corner that
/// changes by exactly +/-1 per frame — small enough for the mean of a mostly-padding tile to call
/// it static, and not small enough to be zero.
fn frames(w: usize, h: usize, n: usize) -> Vec<Vec<f32>> {
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
                    // Top third pans by two pixels a frame; the rest stands still.
                    let c = if j < h / 3 { i + f * 2 } else { i };
                    let blk = ((c / 32) + (j / 32) * 4) % 7;
                    let mut y = 16.0 + (blk * 28) as f32 + tex(c, j);
                    // The corner the tile grid pads: a single level of change per frame, and
                    // **not on the last row or column**. Padding replicates the picture's edge,
                    // so an edge that moves makes the padding move with it and the tile's mean
                    // stops being a lie — which is what the first version of this test got wrong
                    // and why it passed against the unfixed encoder.
                    if (256..w - 1).contains(&i) && (256..h - 1).contains(&j) {
                        y += (f % 2) as f32;
                    }
                    let p = (j * w + i) * 3;
                    out[p] = y.round();
                    out[p + 1] = (128.0 + (blk as f32) * 7.0 - 20.0 + tex(c + 7, j)).round();
                    out[p + 2] = (128.0 + (blk as f32) * 5.0 - 15.0 + tex(c, j + 3)).round();
                }
            }
            out
        })
        .collect()
}

#[test]
fn q100_keeps_the_residual_of_a_mostly_padded_corner_tile() {
    // 320x300 pads to 512x512 at tile_size 256: the corner tile is 64x44 of picture in 256x256.
    let (w, h, n) = (320usize, 300usize, 4usize);
    let frames = frames(w, h, n);
    let refs: Vec<&[f32]> = frames.iter().map(|f| f.as_slice()).collect();

    let mut cfg = gnc::quality_preset(100);
    cfg.entropy_coder = EntropyCoder::Rice;
    cfg.color_space = ColorSpace::YCbCrNative;
    cfg.chroma_format = ChromaFormat::Yuv444;
    cfg.keyframe_interval = 8;
    cfg.scene_cut_threshold = 0.0;
    cfg.set_tile_size(256);
    cfg.normalize_for_chroma();

    let mut enc = EncoderPipeline::new(gpu());
    let cf = enc.encode_sequence(gpu(), &refs, w as u32, h as u32, &cfg);
    assert!(
        cf.iter().any(|f| f.frame_type == gnc::FrameType::Predicted),
        "the clip coded all-intra, so no tile was ever offered to the skip pass"
    );

    let decoded = DecoderPipeline::new(gpu()).decode_sequence(gpu(), &cf);
    for (i, (d, o)) in decoded.iter().zip(&frames).enumerate() {
        // Report where, not just how much: a corner-only failure is the signature of this bug and
        // a whole-frame one is something else.
        let (mut n_bad, mut worst) = (0usize, 0.0f32);
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        for (k, (a, b)) in o.iter().zip(d).enumerate() {
            let e = (a - b).abs();
            if e > 0.0 {
                let (x, y) = ((k / 3) % w, (k / 3) / w);
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                n_bad += 1;
                worst = worst.max(e);
            }
        }
        assert_eq!(
            n_bad, 0,
            "frame {i} is not bit-exact: {n_bad} samples wrong, worst {worst}, over \
             x {x0}..{x1} y {y0}..{y1}. A tile whose residual the encoder deleted reads as a \
             rectangle on a tile boundary (BUG-57)."
        );
    }
}
