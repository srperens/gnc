# 0026 — INTRA-1 step 2: the JPEG 2000 gap decomposes, and a third of it is not a coding deficiency

**Date:** 2026-09-07
**Status:** accepted
**Item:** BACKLOG INTRA-1 (P0), step 2, first instalment
**Builds on:** `0024` (step 1 — the gap is upstream of the entropy coder)

## What step 2 was given

Step 1 established that GNC spends within 7.5% of the entropy of its own coefficients, so at most
~7.5 of the +27.1% RGB gap to JPEG 2000 9/7 is entropy coding and ~19.6 points is upstream. Step 2
lists four upstream candidates, cheapest first: tiling, quantiser shape and per-subband step,
lifting normalisation, code-block geometry.

Three of them are now measured. **The harness reproduces ENT-4's headline exactly — `J2K 9/7` reads
+27.1% RGB and +48.3% Y — so the baseline is validated, not assumed.**

## The decomposition

| cause | RGB points | how it was established |
|---|---|---|
| **Chroma allocation against an RGB metric** | **8.5** | YCoCg-R synthesis norms + measured sweep |
| **Entropy coder headroom** | **≤7.5** | decision `0024`; half of it is ENT-6 small-block cold start |
| **Tiling, 256px against a whole picture** | **≤12.4 for J2K, 0.6 realisable in GNC** | J2K given GNC's tiling; GNC given a bigger tile |
| **Lifting normalisation** | **~0** | closed, see below |
| remainder | **~9.8** | tiling is the leading suspect and GNC does not realise it |

Applying the first two in sequence: +27.2% → +18.7% (chroma) → **+9.8%** (entropy ceiling).

## 1. Lifting normalisation — closed, it is correct

`transform_97.wgsl` uses the Daubechies factorisation with `low *= K, high *= 1/K`,
K = 1.149604398. Mirroring that lifting exactly in Python, building the analysis matrix on a
256-sample signal at 5 levels and inverting it, the **synthesis basis L2 norms are 0.984–1.066**
across all six bands, and boundary basis functions sit within 1–6% of interior ones.

So the transform is near-orthonormal and **uniform quantiser steps are within ~5% of MSE-optimal**,
which at typical RD slopes is worth a fraction of a percent. The production default is already
uniform (`SubbandWeights::uniform`; the perceptual ladder is opt-in behind `GNC_PHYSICAL_WEIGHTS`).
This candidate cannot hold points and is closed.

## 2. The colour transform is where the normalisation error actually is

The same check applied to the **colour** transform is not clean.

| | Y | Co / Cb | Cg / Cr | spread |
|---|---|---|---|---|
| GNC, YCoCg-R synthesis norms | 1.7321 | **0.7071** | **0.8660** | **2.45** |
| JPEG 2000, ICT synthesis norms | 1.7321 | 1.8051 | 1.5734 | 1.15 |

RGB-MSE-optimal steps scale as 1/norm, so GNC's chroma steps should be **2.45x (Co) and 2.00x (Cg)**
the luma step. Production uses **1.2** for both (CHROMA-1, chosen on a colour-aware criterion), so
chroma is quantised about twice as finely as RGB MSE wants. J2K's ICT is nearly norm-balanced, so
*its* uniform steps are automatically near-optimal for the metric — it gets for free what GNC has to
apply explicitly.

**Measured, four images, six quality points, BD-rate against production:**

| chroma weight | RGB PSNR | Y-PSNR | gap to J2K, RGB | gap to J2K, Y |
|---|---|---|---|---|
| 1.2 (production) | — | — | **+27.2%** | **+48.2%** |
| 1.6 | −4.9% | −10.5% | +21.0% | +32.0% |
| 2.0 | −7.0% | −19.3% | **+18.7%** | +21.5% |
| 2.45 | −7.4% | −27.4% | +18.4% | **+13.5%** |

Two things make this more than a sweep. The RGB gain **saturates at 2.0–2.45**, exactly where the
synthesis norms say the optimum is — the theory named the operating point before the measurement
found it. And the Y-PSNR gap **collapses from +48.2% to +13.5%**, which explains the one thing
about ENT-4's numbers nobody had accounted for: why the luma gap was so much larger than the RGB
gap. It was larger because GNC spends bits on chroma that a luma metric cannot see.

### What this does and does not license

**It does not license changing the default.** CHROMA-1 set 1.2 as the largest value costing nothing
on MEAS-8's criterion — 95% of pixels below the JND — and the colour cost of moving is real and
measured here: mean dE00 at q=90 goes **0.445 → 0.585** at cw 2.0 (+31%), and 0.077 → 0.172 at q=99.
CLAUDE.md's rule that chroma questions need a chroma-aware metric applies in both directions; this
is a luma-metric argument for a chroma change and cannot settle it alone.

**What it does license is a correction to how the gap is quoted.** 8.5 of the 27.1 RGB points, and
34.7 of the 48.2 Y points, are GNC deliberately buying colour accuracy that the comparison metric is
blind to. Quoting +27.1% as "the intra coding gap" overstates GNC's coding deficiency by about a
third; quoting +48.3% overstates it by more than two thirds. The honest form names the allocation.

## 3. Tiling — the candidate is real for J2K and does not transfer to GNC

Giving OpenJPEG GNC's tiling (`-t 256,256`, same transform, same five levels) moves the gap from
**+27.1% to +14.7%** on full frames, and from +21.8% to +12.0% on padding-free 1024x512 crops. So
256px tiling costs *JPEG 2000* 8.8–12.4 points.

**It does not follow that GNC would gain that by untiling, and the direct test says it would not.**
Doubling GNC's own tile to 512 is worth **−0.6% RGB / −0.9% Y**, against **3.8%** for J2K over the
same step — a 6x asymmetry. GNC does not convert tile size into rate the way J2K does. Whatever J2K
gains from one tile, GNC is not currently able to collect, and the leading hypothesis is that a large
part of it is **global rate allocation across the picture** rather than wavelet reach: untiled J2K
runs PCRD over the whole image, GNC has no cross-tile allocation at all above q=80 (AQ is 30–80).
EBCOT part 1 closed PCRD at *code-block* granularity **inside** a tile (0.00 dB); cross-tile
allocation is a different lever and has never been measured.

**A confounder found on the way, which reverses the sign of the naive experiment.** A full-frame
tile-size comparison charges the larger tile for padding: 1920x1080 pads to 2048x1280 at tile 256
and to **2048x1536** at tile 512, 20% more coefficients, which reads as **+6.1% rate for tile 512**
— the opposite of the truth. Any future tile-size question must be measured on content that is a
multiple of both sizes.

## What was not chosen, and what it would have cost

- **Making GNC's wavelet span a whole frame.** `transform_97.wgsl` keeps a `array<f32, 512>` in
  workgroup memory, so 512 is its hard ceiling and a whole-frame transform is a shader rewrite, not
  a config change. With GNC realising 0.6% at the one doubling that *is* testable, there is no rate
  case for that rewrite today. It also conflicts with GOALS rule 3 — tile independence is what buys
  the parallel decode, the error resilience and the seeking.
- **Shipping chroma weight 2.0.** See above; it is a colour regression measured in dE00 and the
  decision belongs to a colour-aware criterion, not to this item.
- **Candidate 4, code-block geometry inside abac.** Step 1 already bounded the whole coder at 7.5%
  and located half of that in small blocks (ENT-6). Re-sweeping cb here would have re-measured a
  bounded quantity.

## Caveat

**`--tile-size 1024` silently destroys the image** — 7.19 dB, no error from either encode or decode,
because the shader's workgroup array is sized for 512. Found while pricing this candidate, filed as
**BUG-26 (P1)**. Any earlier measurement taken at a tile size above 512 is void.
