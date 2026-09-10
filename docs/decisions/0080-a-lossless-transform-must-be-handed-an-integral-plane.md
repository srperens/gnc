# 0080 — A lossless transform must be handed an integral plane

Date: 2026-09-10
Item: BACKLOG BUG-49 (P1)
Status: accepted

## Context

`q=100` on subsampled input was **2.7x to 20x worse in colour than `q=99`** at the same chroma
format, on 6 of 6 points, with p95 dE00 above 4.8 everywhere — a lossless mode losing to the lossy
rung below it on the axis lossless is supposed to own. And 4:2:2, which keeps twice the chroma of
4:2:0, came out *worse* than 4:2:0 at q=100 while being better at every other quality. 4:4:4 at
q=100 was exact.

BUG-49 named two suspects: a `MedPredict` branch in the resample, or a plane-extent class of bug.
**Both are wrong.** The resample is one code path for every quality — box filter down, nearest
neighbour up, no branch on transform type — and the extents agree, because `chroma_padded_height`
*is* `chroma_tiles_y * tile_size` by construction.

## What it actually was

**The box filter manufactures non-integer samples, and a lossless transform cannot carry them.**
It averages 2 samples at 4:2:2 and 4 at 4:2:0, so the plane it writes is a multiple of 0.5 or
0.25. At q=100 the step-1 quantiser then rounds the residual. That is exactly the mechanism
BUG-45 documents for a caller's fractional samples — *arising inside the pipeline*, where no
guard was looking, and reachable from a PNG.

**On the MED path a half-LSB does not stay a half-LSB.** `med_predict.wgsl` is open-loop by
design, and says so: *"at lossless the encoder's reconstruction is its input, so every thread can
read the neighbours it needs straight from `src`."* That is true only while coding is exact. Once
the residual is rounded, the encoder predicts from `src` and the decoder predicts from its own
reconstruction, and the difference **accumulates along the DPCM chain** until the tile resets it.

Measured on bbb_1080p at q=100 4:2:2 — mean |Co error| by Chebyshev distance from the origin of
the chroma plane's own 256x256 tile:

| distance | 0 | 1–3 | 4–15 | 16–63 | 64–127 | 128–191 | 192–255 |
|---|---|---|---|---|---|---|---|
| **q=100 (broken)** | 0.40 | 0.56 | 1.39 | 2.55 | 4.35 | 5.79 | **6.38** |
| q=99, same format (control) | 0.60 | 0.34 | 0.35 | 0.36 | 0.36 | 0.35 | **0.35** |
| **q=100 (fixed)** | 0.40 | 0.24 | 0.25 | 0.26 | 0.26 | 0.25 | **0.25** |

The q=99 control is flat, because a wavelet's error does not compound along a scan. The broken
q=100 row is a **16x ramp**, and that shape *is* the diagnosis: nothing about a resample, a filter
or an extent grows with distance from a tile origin. After the fix the row is flat at 0.25, which
is not error left over but the rounding itself — the expected |E| of rounding a quarter-integer,
against a reference that is the *unrounded* box average.

The lossless 5/3 wavelet arm (`GNC_MED=0`) confirms the split: it was **0.22 dE00 above the
subsampling floor and perfectly flat** — the rounding, without the amplifier.

## The decision

**Round the averaged plane to integers when the configuration is lossless**, in
`chroma_downsample.wgsl`, gated by a new `round_output` flag on `ChromaResampleParams` and reached
through `ChromaResampler::dispatch_with_rounding`. The encoder then commits to a plane a lossless
transform can carry exactly, which is what every 8-bit 4:2:0 pipeline stores anyway.

**What was not chosen, and what it would have cost:**

- **Make MED closed-loop.** Correct, and it throws away the property the path was built on: the
  forward pass is embarrassingly parallel *because* it is open-loop. The decoder's wavefront
  already costs 4.9x on the pass that carries it; paying that on the encoder too, to avoid a
  half-LSB, is the wrong trade.
- **Round inside `med_predict.wgsl` instead.** One shader, no new uniform field — and it fixes
  only MED. The lossless 5/3 arm has the same fractional input and was measurably wrong too
  (0.9640 dE00 against a 0.7482 floor on bbb 4:2:2). Rounding at the downsample fixes both, and
  states the invariant where it belongs: *the plane handed to a lossless transform is integral.*
- **A reversible (lifting) chroma downsample.** Would remove the rounding rather than absorb it,
  and is the only option that could beat this one on quality. It changes the subsampled signal,
  so it is a bitstream-visible design change for a gain measured at **0.0000 dE00** — the rounded
  and unrounded floors are identical to four decimal places on all three stills. Not worth it.
- **Refuse `q=100` at non-444.** Rejected by the same argument as `0078` rejects refusing frames:
  the caller can legitimately want it, and it now works.

**No generation bump.** The decoder is untouched, and every previously written file decodes to the
same pixels. What changed is which residuals the encoder produces.

## Result

Decoded output now matches an independent CPU model of *YCoCg-R → box average → round → nearest
upsample → inverse* **exactly — max absolute RGB error 0, on 6 of 6 points**. The codec is
bit-exact for the signal it codes; the whole remaining dE00 is the subsample itself, which is
lossy by definition and always was.

| still | fmt | dE00 before | **after** | floor | Y-PSNR (YCoCg-R) before → after | bytes |
|---|---|---|---|---|---|---|
| blue_sky | 4:2:2 | 2.3066 | **0.0792** | 0.0792 | 65.24 → **70.35** dB | −3.43% |
| blue_sky | 4:2:0 | 1.9488 | **0.0996** | 0.0996 | 64.18 → **66.98** dB | −0.12% |
| kristensara | 4:2:2 | 2.1422 | **0.0976** | 0.0976 | 63.54 → **71.69** dB | −3.02% |
| kristensara | 4:2:0 | 1.8875 | **0.1116** | 0.1116 | 66.89 → **70.98** dB | −0.50% |
| bbb | 4:2:2 | 2.7226 | **0.7482** | 0.7482 | 57.40 → **68.16** dB | −3.24% |
| bbb | 4:2:0 | 2.5567 | **0.9648** | 0.9648 | 59.91 → **64.58** dB | −0.51% |

Every after-figure equals the independently predicted floor to four decimals. 4:2:2 beats 4:2:0 on
all three stills — **the inversion is gone**. Files got *smaller* as well: integral residuals code
better than fractional ones.

**Lossy output does not move at all.** Every q=95 and q=99 row is byte-identical before and after,
as is q=100 4:4:4 — the flag is off unless `is_lossless()`.

## What this does not settle

`q=100` at 4:2:2/4:2:0 is bit-exact **for its own subsampled plane**, not against the source: the
subsample is lossy and `is_lossless()` still reports true. That is BUG-45's territory, one layer
up, and this record does not close it.

`0078`'s refusal of the RATE-2 lossless fallback at non-444 was justified by this defect, so its
premise is now void — but taking it off is a separate decision with its own measurement, because
on 2 of 6 points the bit-exact candidate wins rate and ~10 dB of luma while giving back 0.5–1.7%
of dE00, and because the fallback has never once run at non-444. **RATE-5.**
