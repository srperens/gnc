# 0054 — BUG-39 cause 3: a zero-level transform is an identity, not a no-op

**Date:** 2026-09-08
**Status:** accepted
**Item:** BACKLOG BUG-39 (P1) — **third cause found and fixed.** `q=100` video goes from 26.30 dB
to 51.54 dB inter PSNR and stops drifting down the GOP. It is still not bit-exact.
**Bitstream-visible:** no format change. The bits a `q=100` (or `--dct`) sequence emits for a
P/B frame change completely, because they were the wrong bits.

## What 0042 said cause 3 was, and why that was wrong

0042 left cause 3 as a design question: "the residual is quantised at the P-frame taper (up to
1.25× the intra step) with a dead zone", so making a lossless P-frame meant suppressing both and
paying for it in rate.

**Neither is active at `q=100`.** The taper is keyed on the quantiser step and is 1.0 for any
step at or below 2.8; `q=100` is step 1.0. The dead zone comes from `dead_zone_referenced`, which
is 0.0 at `q=100` (`normalized_for_lossless`, BUG-30). The encoder prints both, and has since
INTER-1:

```
p_qp_scale=1.0000 (taper, default 1.0000), intra_qstep=1.0000 res_qstep=1.0000
  inter_dz_mul=1.00 dz_intra=0.000 dz_referenced=0.000 dz_res=0.000
```

So the two levers cause 3 named were already where a lossless configuration wants them, and the
26 dB had a different cause. **The canary that refutes it was in the default diagnostics output
the whole time** — the same shape as 0042's own lesson, one level up: 0042 diffed the two things
that had to be equal, and then wrote a mechanism paragraph for the remaining gap without reading
the numbers the encoder already prints.

## The actual cause

`WaveletTransform::forward` runs `for level in 0..levels`, so at `levels == 0` **it dispatches
nothing and never writes `output_buf`.** `WaveletTransform::inverse` copies `input_buf` into
`output_buf` before its own loop, so *its* zero-level case is the identity.

The two halves therefore disagreed about what "no transform" means. `encode_pframe` calls
`transform.forward(mc_out → plane_b → plane_c)` and then quantises **`plane_c`**, so at
`levels == 0` it quantised and transmitted whatever the previous frame had left there, and the
decoder — whose inverse *is* an identity — faithfully added that leftover to its motion
prediction.

Reached by every P and B frame in a `q=100` sequence, because LOSSLESS-1 sets
`wavelet_levels = 0` for the MED path, and a residual is always wavelet-coded whatever the
sequence's own transform is (0042 cause 2). `--dct` sequences hit it for the same reason.

**Fixed** by giving `forward` the same identity copy `inverse` already had. The early return in
`inverse` is the same change from the other side: it also removes a `levels - 1` underflow that
panics in a debug build.

## Numbers

3 sequences, 8 frames, ki=2 and ki=9, 4:4:4, shipped defaults. Inter-frame PSNR range across the
GOP (the I-frames are bit-exact, `inf`, before and after):

| | before | after | drift down the GOP |
|---|---|---|---|
| crowd_run ki=2 | 26.30 – 26.76 | **51.54 – 53.62** | |
| crowd_run ki=9 | 21.77 – 26.51 | **50.41 – 51.54** | 4.74 dB → **1.13 dB** |
| old_town_cross ki=2 | 29.69 – 29.91 | **51.83 – 52.52** | |
| old_town_cross ki=9 | 27.04 – 29.91 | **50.58 – 51.83** | 2.87 dB → **1.25 dB** |
| bbb ki=2 | 32.92 – 33.15 | **58.10 – 58.23** | |
| bbb ki=9 | 24.76 – 32.99 | **56.22 – 58.12** | 8.23 dB → **1.90 dB** |

PSNR leads here and VMAF is not quoted: `q=100` is above the q>85 line in CLAUDE.md where VMAF is
saturated, and these frames are 50+ dB.

**No effect at lossy quality, which is the gate that matters:** at q=99 all six points are
identical to the byte and to two decimals of PSNR (crowd_run ki=9: 39 570 472 B and 60.61–60.64
in both arms). The fix is unreachable wherever `wavelet_levels >= 1`, i.e. q=1..99.

## The rate half, and it is not the answer anyone expected

The residual that is now coded is real, so it costs real bits. crowd_run, 4 frames, ki=2:

| | bytes |
|---|---|
| I+P before | 11 369 827 |
| I+P after | **17 584 089** |
| all-intra (unchanged) | 12 932 312 |

**At `q=100` the inter path is now 36% larger than coding every frame intra, and still not
bit-exact.** Before the fix it looked 12% *smaller* than all-intra — a saving bought by
transmitting a stale buffer instead of the residual. A motion-compensated residual at
quarter-pel precision is noise-like and, with no quantiser to throw any of it away, more
expensive than the MED-predicted frame it replaces.

So "should `q=100` video use P-frames at all" is now a live question with a number attached, and
it is **not** the question BUG-39 was filed to answer. Recorded here rather than acted on.

## What is left of BUG-39, and it is one named mechanism

51.54 dB is not `inf`. The remaining error is **sub-pel prediction rounding**: MC interpolates
the reference bilinearly at quarter-pel positions, so the prediction is fractional, the residual
`cur − pred` is fractional, and quantising it at step 1.0 rounds it. Error ≤ 0.5 per sample in
YCoCg-R, amplified into RGB by the inverse colour transform — 51.5 dB is the right size for that,
and bbb reads 58 dB because more of its blocks are full-pel or zero.

The fix is the one H.264 lossless uses: **round the prediction to an integer** in a lossless
configuration, on both sides, so the residual is an integer and step 1.0 is exact. It is a
`round()` in `motion_compensate.wgsl` behind a params flag gated on `config.is_lossless()`, which
the decoder can derive from the frame header it already carries — no new bitstream field.
Untried; that is the next measurement, and it needs a rate number too, since rounding the
prediction changes the residual.

## What was not chosen

- **Suppressing the taper and the dead zone**, which is what 0042 said cause 3 required. Both are
  already neutral at `q=100`; the change would have been inert and would have "explained" the
  remaining 26 dB.
- **Making `forward` reject `levels == 0`.** An assertion would have found this bug, but the
  callers are right to pass 0 — "no transform" is a real configuration, and the decoder already
  implements it as the identity. The asymmetry was the defect, not the call.
- **Fixing it at the call site** (`if levels == 0 { copy }` in `encode_pframe`). Three call sites
  in the P path, three more in the B path, and the same trap for the next caller. It belongs in
  the transform.
- **Chasing 4:2:0 bit-exactness.** Chroma-domain MC box-filters both planes, which is fractional
  by construction. Out of scope, and 4:2:0 is not a lossless format anyway.

## Caveats

- **`--dct` sequences are corrected and unmeasured**, exactly as 0042's cause 2 left them. Same
  mechanism, same reason to flag rather than claim.
- **B-frames are not measured.** They are off by default (`b_pyramid_enabled()`), and
  `encode_bframe` reaches the same `forward` with the same `levels`, so the fix applies to them
  unmeasured.
- **The success criterion is still not met.** BUG-39 asked for bit-exact on every frame at
  `q=100`. This is 51.5 dB. The item stays open on the mechanism above.
