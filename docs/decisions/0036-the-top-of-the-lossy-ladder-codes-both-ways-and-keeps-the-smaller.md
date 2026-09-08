# 0036 — RATE-2: above q=95 the encoder codes both ways and keeps the smaller file

**Date:** 2026-09-08
**Status:** accepted
**Item:** BACKLOG RATE-2 (P1)
**Changes a default.** `quality_preset(q)` for q = 95..=99 now sets `lossless_fallback`, and a
still-image encode at those rungs may return a **bit-exact** file. Files are unchanged below q=95
and at q=100, and no sequence is affected.

## The defect

On every photographic image measured, the top of the wavelet ladder spends **more bytes than the
MED lossless path while delivering worse pixels.** Reproduced on this commit, four stills, bytes:

| image | q=100 (bit-exact) | q=99 (wavelet) | penalty | dominated from |
|---|---|---|---|---|
| bbb_1080p | 3 235 737 | 3 536 493 @ 59.59 dB | **+9.29%** | q=98 |
| blue_sky_1080p | 2 153 118 | 3 026 470 @ 60.14 dB | **+40.56%** | q=95 |
| kristensara_720p | 927 600 | 1 260 606 @ 59.59 dB | **+35.90%** | q=96 |
| touchdown_1080p | 2 610 478 | 3 384 366 @ 59.56 dB | **+29.65%** | q=96 |

Mean at q=99: **+28.85% of the bitrate for a worse picture than bit-exact.** Every figure matches
RATE-2's original filing at `fa32a26`, so the defect is unchanged by everything that landed since.

**It was created by an improvement.** LOSSLESS-1 made q=100 14.9% cheaper by coding MED residuals
instead of wavelet coefficients, which pushed the lossless price *below* the top of the lossy
ladder. The two paths are never compared, because one is called "lossy" and the other "lossless",
and no test covered the crossover.

## The decision

**At q = 95..=99, encode both ways and keep the smaller file.**

`EncoderPipeline::encode` codes the configured wavelet path, then codes
`lossless_sibling(config)` — `quality_preset(100)`'s MED path carrying the caller's `--abac`,
`--cpu-encode` and `--tile-size` — and returns whichever serialises smaller.

**Why this needs no rate/quality trade-off rule, unlike every other choice in this codec.** When
the lossless candidate wins it wins on *both* axes at once: fewer bytes **and** bit-exact pixels.
There is no BD-rate to compute and no metric to choose (CLAUDE.md's table does not apply, because
nothing is being traded). That is also why "keep the smaller" is the whole rule — the case where
the lossy file is smaller is the case where it is the right answer.

**Why it cannot be a preset constant.** The boundary is content-dependent — q=95 on blue_sky, q=96
on kristensara and touchdown, q=98 on bbb — because it sits exactly where MED does well.
`smoothramp512`, `flat512` and `noise512` are **not** dominated at all, since MED is poor on ramps
and on noise. A constant would be wrong on most images in one direction or the other; measuring
per image is the only honest answer, and it is what the codec is for.

**Why the decoder needs nothing.** `transform_type` is a frame-header byte written at
`format.rs:301` and read back at `:969`, independent of the quality byte. A q=97 file carrying
`transform_type = 2` decodes on every existing build — verified by round-tripping through the
shipped `gnc decode` and comparing raw RGB md5 against the source. **No format change, no GP
version.**

## What it buys, measured

Rate against the file the same command produced before, four stills:

| | q=95 | q=96 | q=97 | q=98 | q=99 |
|---|---|---|---|---|---|
| bbb_1080p | 0.00% | 0.00% | 0.00% | −1.39% | −8.50% |
| blue_sky_1080p | −5.34% | −10.24% | −17.26% | −23.48% | −28.86% |
| kristensara_720p | 0.00% | −5.30% | −13.51% | −20.41% | −26.42% |
| touchdown_1080p | 0.00% | −3.12% | −10.79% | −17.31% | −22.87% |
| **mean** | **−1.33%** | **−4.67%** | **−10.39%** | **−15.65%** | **−21.66%** |

And on **12 of those 20 points the output became bit-exact**, from 52.5–60.1 dB. Verified outside
the harness: `gnc encode` → `gnc decode` → raw RGB md5 identical to the source on every point that
switched, and *correctly not* identical on bbb q=97, where the lossy file is genuinely smaller and
is kept.

## Three refusals, and one of them was found by a test going red

- **Already lossless** — nothing to compare against.
- **Not the wavelet.** `--dct` is an explicit request for a third transform, and swapping it for
  MED would make the flag mean something else. This was caught by
  `test_block_dct_quality_preset` failing, which is exactly what that test is for: it asserts a
  DCT config at q=99 stays a DCT config.
- **Inside a sequence** — the important one, below.

## Sequences refuse it, and the reason is a real defect

Measured on bbb, 4 frames, ki=2, q=99, with the fallback reaching the I-frames: the I-frames come
out bit-exact as intended and **the P-frames referencing them decode at 9.80 dB against 60.69 dB**,
while the sequence *grows* from 13.08 MB to 15.28 MB. A MED I-frame carries `wavelet_levels = 0`
and `transform_type = 2`, and the P-frame path's reference cannot reconstruct from it.

So the flag is cleared in two funnels — `build_ip_config` and the three temporal-wavelet config
sites in `main.rs` — and again inside `encode_sequence`, and sequence output is **byte-identical**
to the previous build (13 078 463 B on that run, P-frames back at 60.69 dB).

**A bit-exact reference ought to be the best reference there is, so this is worth fixing rather
than avoiding. Filed as RATE-3.** The clearing in three places rather than one is deliberate: the
warm-up encodes in `main.rs` call `encode` directly with the sequence config, so a single gate
inside `encode_sequence` would still have paid for a second encode and printed a canary for a path
that will not take it.

## What was not chosen, and what it would have cost

- **Stop advertising q=95-99**, which RATE-2 called the honest alternative. It is honest and it is
  worse: the rungs are not broken, they are *mispriced against a sibling*, and the sub-unit
  quantiser steps up there are earning their bits — qstep 0.75 buys 3.9 dB over qstep 1.0 for 13%
  more bits on bbb, a normal RD slope. Removing them would delete a working operating point to
  avoid making a comparison.
- **A boundary constant in the preset** — see "why it cannot be a preset constant". The measured
  boundaries span q=95 to q=98 on four images and three synthetic images have no boundary at all.
- **Predicting the winner instead of coding both.** Cheaper, and it is exactly the kind of
  estimate this repository keeps retracting. The second encode is the MED path, which is the
  cheaper of the two — 8.35 ms against the wavelet's ~21–24 ms in the same run — so the cost is a
  fraction rather than a doubling. **Those milliseconds are load-contaminated and must not be
  quoted as throughput** (COORDINATION); the honest statement of the cost is a *count*: **two
  encodes instead of one, at q = 95..=99 only.**
- **Fixing the inter path now.** That is RATE-3, and it is a different item with a different
  gate — it needs the P-frame reference to reconstruct from a MED frame, which is encoder *and*
  decoder work.
- **Widening the window below q=95.** Every image measured has the lossy file smaller there, so
  the second encode would be pure cost. The window is where the defect is, not where it might be.

## Caveats a later reader needs

- **RD ladders now flatten at the top, and that is correct but consequential.** `rd-curve` on
  kristensara at q = 90,95,97,99,100 returns 8.052083 bpp with infinite PSNR for the last three
  points. `bd_rate` already filters non-finite PSNR (`bench/bdrate.rs:34`) and returns `None`
  below four usable points, so nothing computes silently wrong — **but a ladder that reaches q≥95
  now has fewer usable points than the same ladder had before this change**, and QUAL-1 measured
  that changing a ladder's extent moves the figure by 1.0 PSNR points on its own. **Do not compare
  a BD-rate across this commit.** State the q values with any BD-rate that goes above q=90.
- **`-q 97` can now return a file with no distortion at all.** That is a strict improvement but it
  does change what the number means: q is a ceiling on distortion, not a target. Anyone reading q
  as "approximately this PSNR" will be surprised at the top of the range.
- **The canary prints on every frame that takes the path**, whichever way it goes, so "the
  fallback ran and kept the lossy file" is distinguishable from "the fallback did not run".
  `GNC_LOSSLESS_FALLBACK=0` restores the previous behaviour for measurement.
- **Only 4:4:4 photographic stills were measured.** Subsampled chroma and synthetic content are
  untested here; the synthetic images are known *not* to be dominated, so the fallback should
  simply never fire on them, which is a prediction rather than a measurement.
