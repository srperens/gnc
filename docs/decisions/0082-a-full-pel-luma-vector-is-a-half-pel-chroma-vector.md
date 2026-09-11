# 0082 — A full-pel luma vector is a half-pel chroma vector

**Date:** 2026-09-11
**Item:** LOSSLESS-5
**Status:** accepted

## Context

BUG-39 cause 4 rounds motion vectors to full-pel whenever the configuration is lossless: a
step-1.0 quantiser cannot survive a fractional prediction, and `bilinear_ref` only returns a
reference sample unchanged when the vector lands on a whole pixel.

At subsampled chroma the vector does not stay whole. `motion_mv_scale.wgsl` derives the chroma
displacement by arithmetic right-shift — `dx_c = dx >> 1` at 4:2:0 — so a multiple of 4 quarter-
pels becomes a multiple of 2. That is **half-pel in chroma units**, the interpolator averages two
reference samples, and the fractional prediction the rounding exists to remove is back in Cb and
Cr. 4:2:2 reaches the same place by another route: it does chroma MC in the luma domain on an
NN-upsampled plane and box-filters the result back down, and an odd vector makes that box straddle
two different source columns.

Measured at q=100, 8 frames, `bbb.y4m`: luma exact, every P-frame 66.5–70.4 dB at 4:2:0, 7 of 8
frames inexact at 4:2:2.

**Why it was never seen.** Through the RGB path a Y4M's own samples cannot come back at all — the
BT.601 matrix is not integer-invertible and BUG-45 rounds the fractional input — so "P-frames are
not bit-exact at 4:2:0" was indistinguishable from that already-known loss. LOSSLESS-5's native
arm removed the excuse.

## Decision

Raise the rounding quantum to **8 quarter-pels (two luma pixels) on each axis the chroma planes
subsample**, and leave it at 4 elsewhere: (8, 8) at 4:2:0, (8, 4) at 4:2:2, (4, 4) at 4:4:4. The
constraint is applied to the **luma** vector, before it is coded.

## What was not chosen

**Rounding the encoder's scaled chroma vectors (`mv_chroma_buf`) instead.** This is the obvious
fix and it is wrong. The decoder derives chroma vectors by halving whatever the bitstream carries
and does no rounding of its own, so the two sides disagree about the prediction and the error
compounds through the GOP — measured, 55.6 dB on the first P-frame falling to **46.4 dB** by the
eighth. Making it right would mean changing the decoder, which changes how existing lossless 4:2:0
P-frames decode, which is a bitstream generation bump. A vector that is already a multiple of 8
halves to a multiple of 4 on both sides for free.

**Leaving it.** The alternative to fixing it is that `q=100` is not bit-exact at subsampled
chroma, which makes the rung's whole guarantee false wherever a P-frame survives.

## What it costs

Prediction accuracy, on the subsampled axes only. With `GNC_LOSSLESS_INTRA_RECODE=0` so P-frames
are kept whatever they cost — PNG source, q=100, 4 frames, 4:2:0 — the totals move
**+0.56% / +2.87% / +2.79% / +2.11%** on bbb / blue_sky / crowd_run / old_town_cross.

**Shipped, that cost is not paid.** LOSSLESS-2 already re-codes a lossless P-frame that costs more
than an I-frame, and at q=100 it throws every P-frame out on all three camera clips. On bbb, the
one clip that keeps them, the change is **−1.16%** at 4:2:0 and **−2.82%** at 4:2:2 — the MV field
costs fewer bits and the residual does not get worse. So the measured shipped effect across four
clips is −1.16%, 0.00%, 0.00%, 0.00%.

The +2.9% is recorded because it is what a future change to LOSSLESS-2's rule would start paying,
and because a synthetic that pans by an odd number of pixels pays it in full: the same content
costs **+72%** with the vectors rounded, which is the regime this trade is worst in.

It also repairs the RGB path, where the defect was invisible rather than absent: PNG source,
4:2:0, q=100, P-frames forced, per-frame PSNR goes 38.60 → **38.63 dB**, exactly its own I-frame's
figure.

## Consequences

- Lossless 4:2:0 and 4:2:2 *sequence* encodes change bytes. 144 regression runs against `c633cc4`:
  132 byte-identical, and the 12 that move are exactly this arm.
- No decoder change and no bitstream generation. `mv_round_fullpel.wgsl` gains a per-axis quantum;
  `dispatch_mv_round_fullpel` takes `(i32, i32)` and debug-asserts that only a subsampled axis is
  coarse.
- B-frames still do not round at all — they are off by default (`b_pyramid`), and a lossless
  B-frame was never bit-exact. Not addressed here.
