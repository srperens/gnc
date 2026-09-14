# 0087 — Clamping motion compensation to the visible picture cancels the saving it was meant to collect

**Date:** 2026-09-15
**Item:** PAD-2
**Status:** accepted — the candidate is **rejected on measurement**, and it takes PAD-2's proposed
shape with it
**Machine:** Apple M1 Pro, 16 GB, Metal — `gnc gpu-info`. One of two Macs (COORDINATION).
**Binary:** `codec-fingerprint v1 027e58bc` with the flag off — **unchanged, nothing ships.**
`ba5fdb18` with `GNC_MC_CLAMP_VISIBLE=1`, which is the canary that the path runs at all.

## Context

PAD-1 made the coded padding a cheap flat fade and measured it worth −4.63% of intra rate. It is
refused on any frame something predicts from, because the same fade costs up to **−3.86 dB** on the
worst frame of a P-chain. PAD-2 is the item for collecting that saving on inter anyway, and it
proposed a shape:

> Let the encoder write the cheap fill, and have **both** sides overwrite the padding with edge
> replication of the *decoded* picture before it is used as a reference.

with a cheaper first step to test the premise:

> If it is edge blocks with outward motion vectors specifically, then clamping MC's reads to the
> *visible* bounds instead of the padded ones is a much smaller change than a bitstream version.

Both rest on one premise: **the fill hurts because motion compensation predicts from it.** Clamping
a reference read to the visible extent is exactly equivalent to the reference having replicated
padding, so the cheap step tests the expensive design as well as itself.

## What was built

`visible_w`/`visible_h` in `MotionCompensateParams` and in `motion_compensate.wgsl`, threaded from
the picture size the encoder and decoder both already hold — the luma planes, the 4:4:4 and 4:2:2
chroma planes carried at luma dimensions, and the 4:2:0 chroma planes at their own rounded-up
extent. `motion::mc_extent` is the single place that decides, so the two sides cannot disagree by
each deciding for themselves. Gated on `GNC_MC_CLAMP_VISIBLE`; off, every clamp is bit-identical
to before and `gnc fingerprint` says so.

The bidir shaders declare only the first eight fields of that struct and read the same buffer,
which is legal and means **bidir MC does not get the clamp**. B-frames are off by default
(`0033`), so nothing shipped is affected.

## Result — it fails on both halves at once

`scripts/meas_pad1_inter.py`, three sequences × q ∈ {85, 92}, 17 frames, ki=9, 4:4:4. The figure
that decides is worst-frame PSNR, not the mean (`0039`). Forced `GNC_PAD_FILL=decay` against
`replicate`, each arm measured under its own clamp setting:

| | mean rate | worst dWORST | points regressing > 0.3 dB |
|---|---|---|---|
| MC clamped to the **padded plane** (shipped) | **−8.06%** | −3.860 dB | 2 of 6 |
| MC clamped to the **visible picture** (candidate) | **−0.61%** | −3.630 dB | 2 of 6 |

**1. It does not fix the quality regression.** −3.860 → −3.630 dB on bbb_extended at q=92, and
2 of 6 points still regress. **So the premise is false: the loss is not edge blocks predicting
from the padding.** If it were, removing the padding from prediction entirely would have removed
it.

**2. It destroys the rate win, and the mechanism is the point.** −8.06% → **−0.61%**. The fill's
inter saving exists *because the reference contains the same fill*: the padding's P-frame residual
is then ≈ 0 and costs nothing. Clamp the reads and MC predicts the padding region from the
replicated edge instead, so the fade becomes a residual that has to be coded on every P-frame —
and the encoder pays back exactly what the cheap fill saved.

**The saving and the hazard are the same fact.** The fill is cheap on inter *only* while it is also
what prediction sees.

## Decision

**Reject the candidate, and with it PAD-2's proposed shape.** "Write the cheap fill, replicate it
away in the reference" is not a smaller version of the design — it *is* the design, tested at zero
bitstream cost, and it collects 0.61% instead of 8%. Building the bitstream version would have
bought the same nothing for a format generation.

**Keep the plumbing, off.** It is inert by default (fingerprint unchanged), it is the instrument
that produced this result, and it is what the next hypothesis needs. The numbers are in the doc
comment beside the fields so nobody re-derives them.

## What was rejected, and what it would have cost

- **The bitstream version of PAD-2** — a decoding-process change plus a format generation, in a
  tree where generation numbers are already contested (BUG-51). Refuted before it was started, by
  the flag that made the same experiment free.
- **Restricting motion vectors so no block's footprint leaves the picture.** Encoder-only, no
  bitstream change, and it would also stop MC reading the fill. Not run, because result 1 says
  reading the fill is not what costs the dB — it would inherit the same false premise and, unlike
  the clamp, it would also cost legitimate outward motion.

## What this leaves, and it is a better question than the one PAD-2 asked

**bbb_extended regresses −2.400 dB at q=85 and −3.630 at q=92 with the padding removed from
prediction entirely.** Whatever the fill does on that clip, it does it to the *visible* pixels.
The remaining hypothesis, which this record does not test:

**A border tile's wavelet coefficients reconstruct the padding and the visible part together.**
Changing the fill changes those coefficients, so it changes how quantisation error lands inside the
picture near the edge — and a reference whose border is reconstructed slightly differently
propagates that down the GOP until the next keyframe. That is consistent with everything measured:
it is invisible on stills (PAD-1 measured −4.63% at *identical visible quality*, a single frame
with no chain), it is one-clip (bbb_extended's border content), and PAD-2 already records that
**INTER-2 made it worse** — *"a lower inter dead zone stopped quantising away part of the
prediction error the fill causes"* — which is a statement about coded residual inside the picture,
not about what MC reads outside it.

**How to test it cheaply, before anyone writes a shader:** measure PSNR of the reference frame's
visible pixels *within one tile-width of the right and bottom edges* against its interior, decay
against replicate. `quality::psnr_tile_boundary` already exists for the neighbouring question and
`GNC_TILE_BOUNDARY=1` already wires it into `benchmark`. If the loss is concentrated at the border,
the hypothesis holds and PAD-2 becomes a question about border tiles — which is **TILE-2**, already
filed, already priced, and a bigger prize than this one.
