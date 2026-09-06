# 0011 — P-frames are quantised 1.25× coarser than intra

**Date:** 2026-09-06
**Status:** Accepted (reverses a 2026-09-05 rejection)

## Why this was on the table at all

BUG-7 forced MEAS-4 to be re-run on uncorrupted residual dumps, and its conclusion held: the
coding model is not the inter gap (a DCT-plus-oracle-skip rival is 5.7% *worse* than the wavelet
at matched distortion), the motion search is not the gap (GNC beats a ±32 full-search oracle with
quarter-pel by 6% on SATD), and context modelling has a 10.4% ceiling.

Three plausible causes ruled out in one afternoon leaves only the unglamorous category: things a
mature encoder does that GNC does not. The cheapest item on that list was rate allocation, and
GNC had none — `GNC_P_QP_SCALE` defaulted to 1.0, so a predicted frame was quantised exactly as
finely as an intra frame.

That is the one asymmetry every mature codec exploits, for a reason that is not subtle: the
I-frame is referenced by every P-frame that follows it, so bits spent there are reused across the
GOP, and bits spent on a P-frame are not. x264 runs P about four QP steps coarser, roughly 1.6×.

## The measurement

VMAF BD-rate against 1.0, four sequences, ki=9:

| sequence | 1.15 | 1.25 | 1.5 |
|---|---|---|---|
| blue_sky | — | **−0.59%** | +2.93% |
| aerial | −0.63% | **−0.51%** | −2.36% |
| bbb17 | — | **−5.27%** | −8.11% |
| old_town | −6.00% | **−6.77%** | −6.39% |
| mean | −3.32% | **−3.29%** | −3.48% |

1.25 improves all four. 1.5 has the same mean but regresses blue_sky by 2.9%, so it buys nothing
except a way to lose on one kind of content. 1.25 it is.

The effect scales with GOP length, which is the shape the argument predicts — a longer GOP has
more P-frames to spread the saving across. On old_town at ki=17, 1.25 reaches 84.07 mean VMAF at
0.65 bpp where 1.0 needs about 0.82 bpp for the same quality, roughly **−20%**.

## Reversing the earlier rejection

The 2026-09-05 sweep tried this lever and rejected it: *"worse than lowering q uniformly; VMAF min
falls 94→71 as reference error propagates."*

That objection is the right one to raise. Coarser P-frames degrade the reference that each
subsequent P-frame predicts from, error compounds down the GOP, and a VMAF *mean* hides a
collapsing tail completely. It is why every number here is quoted with its minimum.

It does not reproduce. Mean-versus-min spread on old_town at ki=17, q=35 is 3.32 VMAF points at
scale 1.0 and 2.75 at 1.25 — the spread *narrows*, and it still narrows at 1.6. And the test that
actually settles it compares at matched **rate** rather than matched q, because two encodes at the
same q are not at the same operating point:

| | bpp | VMAF mean | VMAF min |
|---|---|---|---|
| scale 1.0, q=28 | 0.66 | 82.00 | 78.14 |
| **scale 1.25, q=35** | **0.65** | **84.07** | **81.32** |

At the same bitrate the coarser-P encode is +2.07 VMAF mean and **+3.18 VMAF min**. The worst
frame is better, not worse.

A plausible account of the original 94→71 is that it was measured with `GNC_DIAGNOSTICS=1`, which
BUG-7 shows destroys P-frame prediction from the third frame of a sequence onward — precisely the
frames where a propagation argument would appear confirmed. That is a hypothesis about a
measurement no longer in the tree, not a claim; what is established is that the collapse is not a
property of this lever.

## What is genuinely given up

Per-frame PSNR now declines across a GOP: blue_sky at q=50 runs 41.3 → 35.8 dB where it previously
ran 41.3 → 38.3. That decline is real and should not be waved away.

It is also what moving to a lower-rate operating point looks like, and at matched rate the quality
floor is higher, per the table above. PSNR and VMAF disagree in sign here. VMAF is the primary
metric, for the documented reason that it catches perceptual regressions PSNR misses — but a
disagreement in *sign* is worth re-examining if a future change makes GOP-tail quality a complaint
rather than a number.

## Note on scope

This is one constant. It is not progress against the 5–7× video gap in any structural sense, and
should not be read as such. It is the first thing that paid after three larger hypotheses were
ruled out, which is mostly an argument for reading the bit budget before designing experiments.
