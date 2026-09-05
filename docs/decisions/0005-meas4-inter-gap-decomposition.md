# 0005 — The inter gap is prediction quality, not the coding model

**Date:** 2026-09-05. **Status:** accepted. **Closes:** MEAS-4.

## Question

GNC's inter frames save far less than H.264's. Is that because of GNC's *coding model* —
tile-wide wavelet on MC residuals, context-free entropy, no block skip — or because of the
*prediction* the model is asked to code? The answer decides whether a rebuilt inter pipeline
(per-block transforms, block skip, context-adaptive entropy) is worth designing.

The 2026-03 sweep answered neither: it tuned parameters of the existing design rather than
testing the design. This measurement builds nothing — it bounds what a different model could
achieve on GNC's own residuals, offline.

## Method

The encoder dumps its spatial-domain motion-compensated residual (post-MC, pre-transform) for
every P/B frame under `GNC_DUMP_RESIDUAL=<dir> GNC_DIAGNOSTICS=1`, in 4:4:4 so all three planes
share the luma geometry. `scripts/meas4_oracle.py` then simulates **both** models on the same
residuals:

- **wavelet model** — GNC's own: 3-level dyadic DWT, uniform quantizer with dead zone,
  per-subband entropy;
- **rival model** — oracle 16x16 block skip + 8x8 DCT + per-coefficient-position entropy.

Both are given an *ideal* entropy coder (Shannon entropy of the quantized coefficients) and the
rival is additionally given an oracle skip decision and charged no MV cost. So this compares the
two models at their respective ceilings; no implementation can beat either figure.

Three methodological points decided the outcome and are worth recording, because getting any of
them wrong produces a confidently wrong answer:

1. **Compare at matched distortion, not matched qstep.** The first run compared bits at
   qstep=4 for both, which is meaningless — the two transforms land at different MSE. The script
   now sweeps a quantizer ladder for both models and interpolates the rival's rate at the
   wavelet model's operating distortion.
2. **Normalise the wavelet.** A lifting DWT is not orthonormal, so an unnormalised version
   spends quantization error unevenly across subbands and loses to an orthonormal DCT for
   reasons of scaling alone. Each subband is divided by the measured L2 norm of its synthesis
   basis, restoring the Parseval-like property the DCT already has. Before this correction the
   rival looked 41% better; after it, 4%.
3. **Exclude the padding and aggregate per frame.** The dump is the padded plane (2048x1280 for
   1080p) whose padding is identically zero — leaving it in drives every "fraction of blocks
   below threshold" figure toward 1. And GNC reports bpp as all three planes over the luma pixel
   count, so averaging per plane understates every figure by exactly 3x.

The simulation is validated by cross-check: GNC's measured coefficient bitrate lands within
-14% (BBB) and -2% (touchdown) of the simulated wavelet model at its own operating point, so the
simulation is a faithful — slightly pessimistic — proxy for the real encoder.

## Results

1080p, 17 frames, ki=9, 4:4:4, 15 inter frames per sequence, `GNC_REF_DEBLOCK=0`. Two quality
points: q=75 (qstep 4.0, broadcast quality) and q=25 (qstep 16.0, low bitrate), because the
oracle skip fraction is strongly quality-dependent and a single point would not generalise.

**4b — model vs model at equal distortion (the decision experiment):**

| quality | sequence | wavelet model | rival model | rival advantage | oracle-skippable |
|---|---|---|---|---|---|
| q=75 | BBB | 1.6238 bpp | 1.5610 bpp | **+3.9%** | 2.1% |
| q=75 | touchdown | 1.7219 bpp | 1.3321 bpp | **+22.6%** | 0.0% |
| q=25 | BBB | 0.3159 bpp | 0.3257 bpp | **−3.1%** | 20.8% |
| q=25 | touchdown | 0.2143 bpp | 0.2522 bpp | **−17.7%** | 49.7% |

The MEAS-4b decision rule required ≥40% for "the model is the cap". Nothing comes close. At high
quality the rival is 4–23% ahead; at low bitrate — where block skip finally has something to
skip, 21–50% of blocks — the rival is 3–18% **behind**. The wavelet model is competitive or
better across the range GNC operates in.

The skip figure is the more telling one at q=75: **2.1% (BBB) and 0.0% (touchdown)** of blocks
are skippable. Block skip is one of H.264's largest inter tools, and on GNC's residuals at
broadcast quality there is essentially nothing to skip — the prediction leaves error nearly
everywhere.

**4c — entropy context ceiling:** a 1-neighbour context model recovers at most **2.7%** (BBB) /
**2.2%** (touchdown) of coefficient bits at q=75, and 3.4% / 3.1% at q=25.

**4a — residual subband energy:** 97–99% in detail subbands. The originally proposed gate
(">40% detail means transform mismatch") cannot discriminate: an MC residual is high-pass by
construction, so it passes trivially regardless. Recorded as a gate that should not be used.

**4d — where H.264's own inter gain comes from** (x264, --qp 26, same 17 frames):

| | temporal saving vs all-I | multi-ref + B | CABAC | sub-block partitions |
|---|---|---|---|---|
| BBB | 89.2% | **+29.2%** | +8.4% | +1.3% |
| touchdown | 86.5% | **+31.5%** | +9.3% | +1.0% |

(GNC for comparison: 48.9% / 29.8% saving vs all-I.)

## Decision

**The inter gap is a prediction problem, not a coding-model problem.** Do not design a hybrid
per-block inter pipeline: at equal distortion it buys 4–23% at high quality and *loses* 3–18% at
low bitrate, against a large rewrite that would conflict with tile independence and GPU-parallel
entropy. Context-adaptive
entropy coding is worth ≤3% and is not the answer either.

Two independent lines of evidence point the same way. GNC's residuals have almost nothing an
oracle could skip, meaning the prediction leaves error nearly everywhere. And x264's own
ablation says its single biggest inter lever — by a factor of three over CABAC and thirty over
partitioning — is **multi-reference and B-frame prediction**, which is prediction, not coding.

## What this opens up

This is a more encouraging result than the "structural gap" framing it replaces. GNC currently
uses **single-reference P-frames**. The measurement says the lever that matters most for H.264
is exactly the one GNC does not have, and multi-reference prediction is an ordinary,
well-understood, GPU-parallel technique — not a pipeline rewrite.

Backlog item **#25 (multi-reference P-frames)** was deferred in 2026-03 for want of evidence.
This is that evidence; it is promoted to the top of the inter work. Its original gate (add a
periodic-motion sequence and show >15% non-adjacent references in an MV histogram) still stands.
