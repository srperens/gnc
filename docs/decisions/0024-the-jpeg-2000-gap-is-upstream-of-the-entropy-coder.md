# 0024 — INTRA-1 step 1: the remaining JPEG 2000 gap is upstream of the entropy coder

**Date:** 2026-09-07
**Status:** accepted
**Item:** BACKLOG INTRA-1 (P0), step 1
**Supersedes nothing. Redirects INTRA-1 to its step-2 branch.**

## The question this had to answer

ENT-4 measured GNC with `--abac` at **+27.1% of rate against JPEG 2000 in irreversible 9/7 mode**
at matched RGB PSNR, on the same transform at the same depth. abac closed exactly half the original
54.2% gap. Nothing in the repository accounted for the other half.

INTRA-1 step 1 was written as a fork with one number in it: **does GNC spend close to the entropy of
its own coefficients?**

- Close ⇒ the coder is done and the problem is upstream — quantiser shape, per-subband step,
  lifting normalisation, tiling.
- Materially more ⇒ there is coder headroom left after abac, and the offline context estimate
  understated it for a third time.

## What was measured

`src/encoder/coef_entropy_diag.rs`, gated behind `GNC_COEF_ENTROPY=1`. It takes the **shipped**
abac tiles — not a re-encode, not a Python DWT — decodes them back to the coefficients the
bitstream actually carries, and prices those coefficients per plane and per subband:

| column | model |
|---|---|
| `shipped` | the bytes the bitstream really spends, per code-block, plus each block's length field |
| `H0` | zeroth-order entropy of the signed symbols, pooled per plane and subband |
| `Hctx` | conditional entropy of abac's **own** binarisation under abac's **own** 18 contexts, pooled |
| `Hnb` | magnitude as one symbol under a 50-context causal neighbourhood, KT model cost charged |
| `Hnb0` | the same with no model cost charged at all — an absolute floor |
| `Hbig` | a 4× wider template (200 contexts, reaching two coefficients out), KT cost charged |

Four images (`bbb_1080p`, `blue_sky_1080p`, `kristensara_720p`, `touchdown_1080p` — the ENT-4 set),
six quality points (q=60, 75, 85, 90, 95, 99), 4:4:4, tile 256, 5 levels, cb 64, `--abac`.

**Canaries.** The per-band rows account for **99.95%** of the serialized abac payload (the rest is
the seven-byte per-tile header), and the coefficients are abac's own decode of its own stream, so
the comparison cannot be against a different set of coefficients than the one that was paid for.

## The number

**At q ≥ 85 — the contribution range — the shipped rate is 7.5% above the most generous of the
three bounds** (5.3% to 10.3% across 16 points; 8.8% over all 24 points including the lossy end).

The gap to JPEG 2000 is 27.1 points. Taking every one of those 7.5 points out leaves **+17.6%**.

> **Entropy coding can account for at most about 7.5 of the 27.1 points — 28% of the gap.
> Roughly 72%, about 19.6 points, is upstream of the coder.**

Two facts make that a floor on the conclusion rather than an artefact of the model chosen:

- **Widening the context found nothing.** `Hbig` reaches two coefficients further in both
  directions, 200 contexts against 50, and lands *above* `Hnb` on the large bands once its model
  cost is charged and below it only in the tiny ones. The local-context model has saturated.
- **Charging no model cost at all buys 1.6%.** `Hnb0` is not reachable by any real coder — it
  signals its tables for free — and it still leaves 19 of the 27 points unexplained.

## Where the 7.5% that *is* the coder's actually sits

| | share of rate | shipped vs bound | share of the headroom |
|---|---|---|---|
| levels 1–2 (full 64×64 code-blocks) | 82% | **+4.1%** (q=90) | 46% |
| LL + levels 3–5 (blocks smaller than 64px) | 18% | **+25.9%** (q=90) | 54% |

Half the remaining coder headroom is not a modelling gain at all: it is cold-start on blocks too
small to adapt. At tile 256 with 5 levels, the LL and the level-3/4/5 bands are 32, 16 and 8 px
square, so each becomes one short code-block — 64 coefficients to learn 18 context probabilities
on. On the bands where abac gets a full block it is within 4.1% of a bound that charges nothing for
being generous, which is as close to done as this measurement can show.

## The decision

**INTRA-1 goes down its step-2 branch: the gap is upstream.** In the order the item lists them —
whole-frame transform against 256px tiles, quantiser shape and per-subband step derivation, lifting
normalisation, code-block geometry.

**A richer entropy context is not the answer and should not be scoped as one.** This is now the
third measurement pointing the same way: the offline EBCOT estimate put the full neighbourhood at
−16.4% against abac's vertical-only −11.7% (≈5.3% apart), and this measures 4–7% on real shipped
coefficients. Two independent methods agreeing to within two points is the strongest cross-check
either number has.

## What was not chosen, and what it would have cost

- **Testing a parent / cross-subband context.** SPIHT and EZW condition on the coefficient in the
  coarser band; this measurement covers only causal *spatial* neighbourhoods. It is the one model
  class that could still find something, and it was left out because a 4× wider spatial template
  found nothing — a parent term is not plausibly worth 20 points when two more rings of neighbours
  are worth zero. If step 2 comes back empty, this is the thing to try before concluding.
- **Fixing the small-block cold start now.** It is worth about 4% of the file and is a real item
  (signalled initial probabilities, or letting deep subbands share one code-block). It is filed
  rather than done, because doing it here would have mixed a coder change into the measurement
  that says the coder is not the problem.
- **Re-running the four-arm `meas9_contribution.py` ladder to re-derive +27.1% first.** ENT-4
  measured it the same day against a pinned commit and this diagnostic does not depend on the
  ladder's absolute value — it splits whatever the gap is. Re-deriving it would have cost ~25
  minutes and moved nothing.

## Caveats a later reader needs

- **+27.1% is a BD-rate over the ladder; the headroom here is a rate ratio at fixed q.** Because
  abac is lossless recoding of the same coefficients, a rate saving of *x*% moves the BD-rate by
  approximately *x* points, but the two are not the same quantity and the arithmetic above treats
  them as if they were.
- **The bounds pool statistics across a whole plane's worth of a subband.** No real coder has that;
  it is deliberate, so that "even with priors it could never have, the coder is within 7.5%" is the
  strong form of the claim.
- **This says nothing about chroma allocation.** ENT-4's Y-PSNR gap (+48.3%) is larger than the RGB
  one and an entropy coder cannot move bits between planes; that is MEAS-9's finding and is
  untouched here.
