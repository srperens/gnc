# 0014 — `chroma_weight` stays at 1.2 above q=85, and it is an intra lever only

**Date:** 2026-09-06
**Status:** Accepted

## The decision

`chroma_weight` no longer drops to 1.0 above q=85; it stays at **1.2** all the way up. Two things
were rejected along the way, and both rejections matter more than the change:

- **1.5 was not chosen**, even though it measures better on luma by a wide margin.
- **The hypothesis that GNC's luma gap is partly an allocation choice was eliminated.** It is not.

## Why the knob was worth sweeping

0013 found GNC ahead of x264 on colour at matched rate while 7.4–8.8 dB behind on luma. Two codecs
allocating differently between luma and chroma is a *configuration* difference, not a coding
efficiency one — so part of the +90.5% might have been reclaimable by moving the split, which is a
one-line change rather than a new algorithm. `chroma_weight` had never been swept; the values were
recorded in the source as "fixed guesses".

The trap was written down before starting: this exact knob was swept once on VMAF, looked like a
free 15% rate saving, and reversed sign once measured with a chroma-aware metric. VMAF is luma-only
*and* saturated above q=85 — doubly blind here.

## The frontier

Four images × q=75/85/92/96 × seven weights. BD-rate over each arm's own rate/quality curve, never
at fixed q. Luma in YCoCg-R, colour as CIEDE2000 computed on −dE00 so the sign convention matches:

| weight | luma BD-rate | colour BD-rate | exchange rate |
|---|---|---|---|
| 0.8 | +9.8% | −0.4% | — (worse on luma) |
| 1.0 | +1.3% | 0.0% | — |
| **1.2** | **−5.2%** | **+1.2%** | **4.3:1** |
| 1.5 | −12.9% | +4.0% | 3.2:1 |
| 2.0 | −21.8% | +9.7% | 2.2:1 |
| 3.0 | −32.0% | +22.2% | 1.5:1 |

Monotone on all four images, no crossings. Re-baselined on constant 1.0 rather than the shipped
mixed policy the figures are −6.5% / −14.1% / −23.0% / −32.9% against +1.3% / +4.0% / +9.5% /
+21.7%, so the conclusion does not depend on the baseline choice.

**The frontier is steep and the old 1.0 above q=85 sat off it.** In dB at matched rate, weight 1.5
against 1.0 gives +0.95 (bbb), +1.34 (blue_sky), +1.48 (touchdown), +1.35 (kristensara) — mean
**+1.28 dB** against a pre-registered success criterion of ≥0.5 dB.

## Why 1.2 and not 1.5

The luma side of the hypothesis passed comfortably, so the choice came down to what the colour side
costs — and the criterion as written could not be used.

It said "mean dE00 below x264's (0.611–0.949)". **That comparison is invalid**: 0013's x264 figures
are 4:2:0 *video*, this sweep is 4:4:4 *stills*. Different chroma format, different content.
Substituted MEAS-8's internal criterion — 95% of pixels below the JND — counting how many of the
four images pass:

| weight | q=75 | q=85 | q=92 | q=96 |
|---|---|---|---|---|
| shipped | 1.81 (0/4) | 1.08 (1/4) | 0.91 (2/4) | 0.73 (4/4) |
| **1.2** | 1.81 (0/4) | 1.23 (1/4) | 0.98 (2/4) | 0.79 (4/4) |
| 1.5 | 2.00 (0/4) | 1.39 (1/4) | 1.13 (**1/4**) | 0.87 (**2/4**) |
| 2.0 | 2.30 (0/4) | 1.64 (**0/4**) | 1.36 (1/4) | 0.99 (2/4) |

**1.2 is the largest weight that costs nothing** — pass counts identical to the shipped policy at
every q. 1.5 buys its extra luma by trading away exactly the criterion a contribution codec is
judged on, and colour fidelity is one of the few places GNC measurably leads (POSITIONING §4). A
1.28 dB luma gain is not worth surrendering a differentiator when 1.2 is available for free.

## The important half: this does not touch the video gap

The stills result does **not** transfer to the shipped configuration. The first guess for why was
the chroma format, since 4:2:0 halves the chroma material. That guess was wrong, and the control
says so:

| configuration | weight 2.0 vs shipped, bytes |
|---|---|
| 4:4:4 stills | ≈ −25% |
| 4:4:4 video, all-intra (ki=1) | **−20.8%** |
| 4:4:4 video, P-chain (ki=9) | **−2.9%** |
| 4:2:0 video, P-chain (ki=9) | −1.5% |

It is **inter**, not the format. After motion compensation there is almost no chroma residual left
to coarsen, so the effect lives in I-frames and is diluted sevenfold by the P-chain. At q=92 4:2:0,
luma moves **0.01 dB** across the knob's whole useful range.

**So the +90.5% is genuine luma coding deficit, not an allocation artefact, and intra is the only
route — by elimination rather than assumption.** That is the result this work was for; the shipped
default change is a by-product.

The lever stays real where intra dominates, and that is not a corner case: all-intra is a normal
contribution mode — lowest latency, best generation survival — in exactly the 4:4:4 / 4:2:2 formats
POSITIONING calls a gate.

## Two measurement errors, both caught in our own harness

**Luma from decoded RGB is contaminated by the very thing under test.** Perturb only Co/Cg and the
reconstructed RGB moves, so BT.709 Y moves with it. On kristensara at q=92, weight 1.0 → 3.0:

| luma measured as | 1.0 | 3.0 | apparent loss |
|---|---|---|---|
| BT.709 Y from RGB | 51.94 | 51.38 | **−0.56 dB** |
| YCoCg-R Y (what GNC codes) | 50.975 | 50.825 | **−0.15 dB** |

A **3.7x overstatement**, pointing the wrong way — it makes coarsening chroma look expensive in
luma when it is nearly free. `scripts/ypsnr_de00.py` now reports YCoCg-R luma, BT.709 luma and
dE00 together, so the contamination is visible rather than assumed away.

**And VMAF saw none of it.** The shipped change cuts 6% of the bits at q=90 and VMAF reads **97.08
before and 97.08 after** — not a small change, *no* change. Judged on VMAF alone this is a free 6%
with no cost whatsoever, which is precisely the illusion behind the earlier "free 15%". This is now
recorded next to the row in BASELINE.md and in CLAUDE.md's metric guidance.

## Correctness

Verified **before** the change, not after: q=100 output is byte-identical at every weight, because
the quantiser is bypassed at lossless — so lossless cannot be affected. q=75 output is unchanged
(it already used 1.2) and q=92 matches a forced `GNC_CHROMA_WEIGHT=1.2` exactly, confirming the new
default is the measured configuration and nothing else moved. 198 tests pass; both clippy targets
clean.

BASELINE's q=90 row moves 50.41 → 50.06 dB and 8.58 → 8.07 bpp. That is a move along the RD curve,
recorded per BASELINE's regression rule: most of the 0.35 dB is chroma leaking into an RGB metric,
the codec's own luma moves 0.055 dB, and dE00 goes 0.325 → 0.353 mean with p95 0.721 → 0.772, both
far under the JND.
