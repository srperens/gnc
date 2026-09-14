# 0084 — A bit-exact encode may not delete a tile's residual

**Date:** 2026-09-11
**Item:** BUG-57
**Status:** accepted

## Context

`tile_skip_motion.wgsl` declares a P-frame tile *static* when its mean per-pixel zero-MV SAD is
below `qstep/2` **and** no worse than the motion-compensated error. The encoder then does two
things: it zeroes that tile's 8x8 motion vectors, and — after quantisation — it zeroes the tile's
**quantised coefficients**, so the decoder reconstructs it from the prediction with no residual.

That is a rate/quality trade, and it was running at `q=100`, where there is no quality to trade.

**The mean is what makes a border tile lie.** A tile on the right or bottom edge is padded out to
`tile_size`, and the padding replicates the picture's edge, so it is identical between frames and
contributes a SAD of exactly zero. At 1080p with `tile_size=256` the bottom-right tile is 128x56
of picture inside 256x256 — **89% padding** — so the mean is diluted by roughly 9x before the
threshold sees it.

Measured on `bbb.y4m`, q=100, 4:2:0: frame 2 decoded at **75.31 dB**, 3 606 luma samples wrong,
over exactly `x 1792..1919, y 1024..1079` — the visible part of that one tile — with a worst error
of 10. `GNC_SKIP_DIAG=1` reads `skip_tiles=1/40` on that frame and `0/40` on its neighbours.
`--tile-size 128` made it vanish, because the same corner is then 56 of 128 rows rather than 56 of
256 and clears the threshold.

**Why it survived this long.** At `q=100` the encoder's reference is the *source*
(`reference_from_source`), so the error never enters the next frame's prediction and never appears
as drift — frame 3 was bit-exact again. And no Y4M encode was bit-exact end to end before
LOSSLESS-5, so one frame in eight being slightly wrong had nothing to stand out against.

## Decision

Do not zero coefficients when `config.is_lossless()`. The zero-MV forcing above it stays.

`GNC_TILE_SKIP_THRESH`, the coefficient-domain skip that is off by default, is refused at lossless
for the same reason, with a canary so a measurement run says so rather than quietly losing
exactness.

## What was not chosen

**Turning the whole pass off at lossless** (`GNC_TILE_SKIP_MOTION_MUL=0`). It is the tidier
statement — the forcing's stated purpose is to make the residual small *so that the quantiser
zeroes it*, and at step 1.0 the quantiser zeroes nothing, so the mechanism is gone even though the
code runs. But it is **rate-neutral**, measured on six points (bbb at 4:2:0 / 4:2:2 / 4:4:4, ki=8
and ki=2, 8 frames): **−0.009%, −0.005%, +0.011%, +0.006%, −0.004%, +0.000%**. Sign varies, every
figure is under 0.011%, and CLAUDE.md's rule for that is to move on. Removing code for no measured
gain is a change, not a simplification, so the diff stays at the one line that is actually wrong.

**Raising the threshold, or making it exclude padding from the mean.** Both are tuning of a lossy
tool, and neither makes `q=100` exact — only smaller or larger. A threshold that is *sometimes*
wrong at the bit-exact rung is still wrong there.

## What it costs

Only the clip that keeps P-frames at `q=100` pays anything; the three camera clips code all-intra
(`0070`) and are byte-identical. `bbb.y4m`, 8 frames:

| | before | after | delta |
|---|---|---|---|
| 4:2:0 ki=8 | 11 863 639 | 11 871 842 | **+0.069%** |
| 4:2:0 ki=2 | 12 280 971 | 12 291 502 | +0.086% |
| 4:2:2 ki=8 | 13 440 827 | 13 448 505 | +0.057% |
| 4:2:2 ki=2 | 13 965 386 | 13 975 917 | +0.075% |
| 4:4:4 ki=8 | 17 773 215 | 17 789 345 | +0.091% |
| 4:4:4 ki=2 | 17 957 819 | 17 977 284 | +0.108% |

Under a tenth of a percent, for the difference between a rung that is bit-exact and one that is
bit-exact except when it is not.

## Consequences

- **The RGB / PNG path does not move at all**: 144 `encode-sequence` runs md5'd against the
  pre-fix binary — four clips x {q=50, 75, 100} x {4:4:4, 4:2:0, 4:2:2} x {Rice, abac} x
  {ki=2, ki=9} — byte-identical, `q=100` included. On PNG content the corner tile never falls
  under the threshold, which is also why BASELINE's PNG-sourced lossless rows were honest.
- `tests/lossless_tile_skip.rs` is the guard, and it discriminates: against the unfixed encoder it
  fails with *2 709 samples wrong over x 256..318, y 256..298*.
- **`GNC_SKIP_DIAG`'s readback moved out of the gated block.** It was inside, so with the zeroing
  off it copied nothing and the readback reported a stale `skip_tiles=0` — the instrument for this
  bug nearly argued the diagnosis away after the fix was already in. It now reports what the pass
  decided either way, and says when the coefficients were kept.
