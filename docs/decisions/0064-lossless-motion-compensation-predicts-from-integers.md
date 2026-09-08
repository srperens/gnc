# 0064 — BUG-39 cause 4: lossless motion compensation predicts from integers

**Date:** 2026-09-08
**Status:** accepted
**Item:** BACKLOG BUG-39 (P1) — **fixed.** A `q=100` sequence now decodes **bit-exact on every
frame**, I and P alike, verified outside the harness on 48 frames.
**Bitstream-visible:** at `q=100` every motion vector is a multiple of 4 (full-pel). No format
change, no decoder change — the decoder reads the vectors it is given and takes the exact-sample
path through the same interpolator.

## The defect

`motion_compensate.wgsl` interpolates the reference bilinearly at quarter-pel positions. A sub-pel
vector therefore makes the **prediction fractional**, `cur − pred` fractional with it, and a
lossless configuration quantises the residual at step 1.0 — which rounds it. Rounding is precisely
what a lossless configuration must not do, so the round trip lost up to half a sample per pixel,
amplified into RGB by the inverse colour transform: **51.5 dB instead of bit-exact** after `0054`
fixed cause 3.

Nothing was wrong with the arithmetic. Encoder and decoder agreed exactly; they agreed on a
picture that was not the source.

## The measurement that isolated it, before any code was written

An 8-pixel **integer-shift** sequence (ffmpeg crop+pad of a real frame), coded at `q=100`:

```
Residual Y: mean_abs=0.01 stddev=0.05 near_zero=100%    ← blocks that landed on full-pel
Frame 1 [P] PSNR=75.59 dB                               ← and the ones that did not
```

The residual is **exactly zero wherever the vector is full-pel**, and the entire 75.6 dB (rather
than `inf`) comes from the blocks the split search refined to sub-pel. That is the whole diagnosis
in one run, and it cost nothing: the content was built to have a known integer motion, so the
search's own refinement is the only variable left.

## What was chosen

**Round the motion vectors to full-pel when the configuration is lossless**, in place, at the
point where the vector field is final — after `tile_skip_motion` and MV smoothing, before both
motion compensation and the MV entropy coding. `src/shaders/mv_round_fullpel.wgsl`, dispatched
from `encode_pframe` behind `config.is_lossless()`.

One place, two consumers, no way for them to disagree. And the decoder needs no counterpart at
all: it uses the vectors the bitstream carries, so `bilinear_ref`'s `fx == 0 && fy == 0`
early-out returns a reference sample unchanged there too.

`GNC_LOSSLESS_FULLPEL=0` restores sub-pel vectors — the arm that prices the trade.

## Numbers

**Bit-exactness — the item's success criterion, verified outside the harness.** Real container
(`encode-sequence` → `.gnv` → `decode-sequence`), raw RGB md5 of every decoded frame against its
source PNG, 3 sequences × ki=2 and ki=9 × 8 frames:

| | ki=2 | ki=9 |
|---|---|---|
| crowd_run | **8/8 bit-exact** | **8/8** |
| old_town_cross | **8/8** | **8/8** |
| bbb | **8/8** | **8/8** |

**48 of 48 frames**, md5-identical. The in-harness reading agrees: every inter frame prints
`PSNR inf` where it read 51.54–58.23 dB before.

**What full-pel costs**, same encode with `GNC_LOSSLESS_FULLPEL=0` against the default (bytes,
8 frames, `q=100`):

| | sub-pel | full-pel | cost |
|---|---|---|---|
| crowd_run ki=2 | 35 132 592 | 35 712 641 | +1.65% |
| crowd_run ki=9 | 41 814 962 | 43 003 751 | +2.84% |
| old_town_cross ki=2 | 34 658 827 | 35 209 443 | +1.59% |
| old_town_cross ki=9 | 41 545 518 | 42 778 003 | +2.97% |
| bbb ki=2 | 25 371 086 | 25 484 805 | +0.45% |
| bbb ki=9 | 24 808 854 | 25 183 274 | +1.51% |

**Mean +1.83%, worst +2.97%** — and the sub-pel arm it is compared against is not bit-exact, so
this is the price of the guarantee, not a regression against an equal-quality alternative.

**No effect anywhere else.** `q=99` is identical **to the byte** on all six points; the gate is
`config.is_lossless()`, which is false for every rung from q=1 to q=99.

## Why not round the prediction instead

H.264 lossless keeps sub-pel motion and rounds the *interpolated prediction* to an integer, which
is strictly better prediction than full-pel for the same guarantee. It was not chosen here, and
the reason is plumbing rather than principle:

- The rounding flag has to reach three consumers — the encoder's forward MC, the encoder's local
  decode and the decoder — through `MotionCompensateParams`, which is built in **three** places
  including a **cache** (`buffer_cache.rs`) keyed on geometry and not on configuration. A cached
  params buffer built for a lossy encode and reused for a lossless one in the same process is a
  silent wrong answer, so the cache key would have to change too.
- Full-pel rounding needs one shader, one dispatch, no struct change, no cache key, no decoder
  change — and the measurement above says it costs **1.83% of bytes**.

That is a good trade today. It stops being one if the number grows: **if the P path is ever worth
more than a couple of percent at `q=100`, revisit this** — the prediction-rounding route is the
better codec and the numbers to beat are in the table above.

## What else was rejected

- **Skipping the sub-pel stage in `block_match_split.wgsl`.** Same effect, but it needs a flag in
  the search shader and leaves the *smoothing* path able to reintroduce sub-pel values. Rounding
  after the field is final cannot be bypassed by a later stage.
- **Rounding on the CPU between readback and MC.** The vectors live on the GPU and are read back
  for entropy coding; doing it there would put an ordering constraint between two stages that do
  not otherwise have one.
- **Doing it for B-frames as well.** Bidirectional MC averages two predictions, so `(p₀ + p₁)/2`
  is half-integer even at full-pel — full-pel is *necessary but not sufficient* there. B-frames
  are off by default (`b_pyramid_enabled()`); `GNC_B_PYRAMID=1` at `q=100` is still not bit-exact
  and now for exactly one stated reason.

## Caveats

- **4:4:4 only.** 4:2:0 box-filters both planes in chroma-domain MC, which is fractional by
  construction, and 4:2:0 is not a lossless format anyway.
- **B-frames are not bit-exact**, above.
- **The rate question this exposes is `LOSSLESS-2`**, not this record: at `q=100` the inter path
  costs **+38.1% (crowd_run) and +39.5% (old_town_cross)** against coding every frame intra, and
  **−1.6% on bbb**. Both arms are now bit-exact, so that is an exact rate comparison at identical
  pixels rather than a BD-rate estimate — the cleanest form that comparison can take.
