# 0028 — The dead zone is worth 3 points of the intra gap, and it must never be global

**Date:** 2026-09-08
**Status:** accepted
**Item:** BACKLOG INTRA-1 (P0), step 2c
**Builds on:** `0026` (the gap decomposes), `0027` (cross-tile allocation rejected)

## What was left

After `0024`, `0026` and `0027`, INTRA-1 had eliminated its obvious candidates and ~9 of the 27.1
RGB points against JPEG 2000 9/7 were unexplained. Two cheap untested things remained: the
**deadzone and quantiser rounding rule** against J2K's, and the wavelet's **tile-boundary
handling**.

## 1. The boundary handling is already correct — and `0027` said otherwise

`0027` and its RESEARCH_LOG entry describe `transform_97.wgsl` as one that "replicates the edge
sample where J2K uses symmetric extension". **That is wrong, and it is corrected here.**

The shader's edge rules read, in the *split* arrays: `low[half] := low[half-1]` on the predict step,
`high[-1] := high[0]` on the update step. Those look like replication — but the arrays are the
polyphase split of the interleaved signal, and whole-sample symmetric extension of `x` says
`x[-1] = x[1]` and `x[N] = x[N-2]`. In split terms `x[N] = low[half]` and `x[N-2] = low[half-1]`,
and `x[-1] = high[-1]` and `x[1] = high[0]`. They are the same operation.

Checked numerically rather than left as algebra: GNC's lifting against a textbook 9/7 run on an
explicitly symmetric-extended signal, 1000 random signals over five lengths — **worst relative
difference 0.000e+00, lowpass and highpass**. Identical.

It was also already visible in `0026`'s measurement and nobody read it that way: boundary synthesis
basis norms came out within 1–6% of interior ones, which a wrong extension would not produce.

**Candidate closed. Worth 0 points.**

## 2. The dead zone is real, and it is worth ~3 points — on stills

GNC quantises as `floor(|v|/step + 0.5)` after a `|v| < dead_zone * step` test. `floor(x + 0.5)` is
already zero below `0.5*step`, so **any `dead_zone <= 0.5` is a no-op**. Production interpolates
0.5 at q=85 to 0.05 at q=92 to 0.0 at q>=96 — so **GNC has no dead zone at all in its own operating
range**, and its zero bin is `1.0*step` wide. JPEG 2000's irreversible quantiser truncates
(`floor(|v|/step)`), so its zero bin is `2.0*step` — twice as wide.

Swept on four stills, six quality points, `--abac`:

| | rate at 48 dB | 50 dB | 52 dB | gap to J2K (RGB) |
|---|---|---|---|---|
| production | — | — | — | **+27.2%** |
| dz 0.6 | **−3.1%** | **−2.2%** | **−2.5%** | **+24.1%** |
| dz 0.75 | −1.0% | −0.6% | −2.0% | +24.9% |
| dz 0.9 | +9.4% | +2.5% | +0.5% | +29.6% |
| dz 1.0 | +13.8% | +4.5% | +2.9% | +33.1% |

**dz ≈ 0.6 removes about 3 points of the gap.** And J2K's own width is *worse* for GNC, by a lot —
because GNC reconstructs at the round-to-nearest bin centre, so widening the zero bin without moving
the reconstruction point buys rate and pays for it immediately in distortion.

At matched rate, averaged over the four images, dz 0.6 is better on **every** axis at once:

| bpp | RGB PSNR | dE00 |
|---|---|---|
| 4.0 | +0.24 dB | −2.9% |
| 5.0 | +0.38 dB | −4.5% |
| 6.0 | +0.28 dB | −3.2% |
| 8.0 | +0.29 dB | −3.9% |

VMAF cross-check at the same q: **−8 to −9% of rate for a VMAF move of −0.01, worst −0.02**, against
a 0.5-point block threshold. VMAF is saturated at 96.9–97.2 here, so it is a "no perceptual
objection" reading, not the lead — PSNR and dE00 lead at this operating point and both improve.

## 3. But it must never be a global default, and that is why sequences were measured

The dead zone applies to **P-frame residuals too**, and there it is a regression. Three sequences,
16 frames, ki=9, 4:4:4, evaluated at **matched rate** against the production ladder:

| sequence | mean PSNR, q=85/90/95 | worst-frame PSNR |
|---|---|---|
| bbb_extended | −0.62 / −0.21 / +0.30 | −1.10 / −0.85 / −1.53 |
| old_town_cross | −0.83 / −0.41 / −0.55 | −1.60 / −1.62 / −1.87 |
| crowd_run | −0.79 / −0.27 / −0.47 | −1.51 / −1.44 / −1.93 |

**Worst-frame is negative on 9 of 9 points, by up to 1.93 dB.** For a contribution codec that is the
metric that matters most (QUAL-1), and it is the one that moves most.

The mechanism is not mysterious: a motion-compensated residual is already sparse and small, so a
dead zone zeroes a much larger fraction of coefficients that were carrying real signal, and the
error then **propagates down the prediction chain** rather than being confined to one picture.

## The decision

**Do not change the `dead_zone` preset.** Filed as **INTRA-2 (P1)**: apply the dead zone to
**I-frames only**, then re-gate on both stills and sequences. `sequence.rs` already varies quantiser
parameters per frame type (`GNC_P_QP_SCALE`), so there is a place to put it; it is a feature, not a
preset tweak, and it needs its own gate.

**Had this shipped on the stills evidence alone it would have been a clean four-image win that cost
up to 1.93 dB of worst-frame quality on every sequence measured.** The stills gate was green on
three metrics — PSNR, dE00 and VMAF — and all three were measuring the wrong thing for the P path.

## Found on the way: BUG-30, and it is BUG-15's hole one knob over

`GNC_DEAD_ZONE=0.6` at q=100 produced a file **3.4% smaller and not bit-exact**, with no warning.
`is_lossless()` gates on `dead_zone == 0.0`, so a config carrying a dead zone reported *not
lossless*; `normalized_for_lossless` then skipped it and the integer-exact colour and lifting paths
were switched off — the guarantee gave way instead of the knob.

Fixed by splitting **lossless intent** (transform and step) from **bit-exactness** (intent plus the
knobs that can spoil it), and normalising on intent: the dead zone is now forced to 0.0 with a
warning, exactly as `chroma_weight` already was. q=100 is byte-identical at 927 600 B and bit-exact
with `GNC_DEAD_ZONE` set to 0.6 or 1.0. Test asserts bit-exactness, not a PSNR threshold — a
threshold is what let 55 dB pass for lossless in BUG-15.

## What was not chosen

- **Shipping dz 0.6 for stills only via the quality preset.** The preset is shared by both paths;
  gating on frame type is the correct place and belongs in INTRA-2.
- **Matching J2K's quantiser exactly** (truncation plus a reconstruction offset at `(|q|+r)*step`).
  It is the principled version of this lever and would need a decoder change, so it is scoped into
  INTRA-2 rather than done here — and the sweep already shows the naive wide zone is the wrong
  half of it.
