# 0063 — abac bypasses three quarters of its own bits at q=99, and the Exp-Golomb prefix is where the deficit is

**Date:** 2026-09-08
**Item:** ENT-9 step 1 (and 1b, the pricing)
**Status:** accepted — step 1 answered, step 2 scoped and ordered; no default moves, no bitstream change

## Decision

**ENT-9 step 1 is answered and the item is not small.** abac context-codes exactly three binary
decisions per coefficient — significant, `>1`, `>2` — and sends the Exp-Golomb order-0 remainder
and the sign as bypass bits at p=1/2. Measured on the shipped tiles:

**At q=99, 74.6% / 74.8% / 46.7% of abac's own bits are bypassed** — sent with no model at all.
The context model touches a quarter of the file on two of three sequences. (Figures re-taken after
RATE-3; see the section below. Pre-RATE-3 they were 75.1% / 75.5% / 43.6%.)

**Step 2 goes to candidate A first, and candidate B is below the gate.** Both were priced in the
same read-only pass, before either was built:

- **A — context the Exp-Golomb unary prefix** (6 buckets × 4 positions = 24 extra contexts):
  **−2.44% to −9.07% of the coder's bits at q=99**, clearing ENT-9's ≥2% gate on three of three.
- **B — context the sign** on the left and up neighbours' signs (3×3 = 9 contexts): **−0.57% to
  −1.29% at q=99**, below that gate on three of three.

## Re-taken after RATE-3 (2026-09-08, same session) — the conclusion strengthens

RATE-3 landed between the measurement and the merge, lifting the lossless-fallback gate for
sequences at q=95..=99. That changes the I-frame a P frame predicts from, so it changes these
coefficients. Re-run on a pinned post-RATE-3 binary (`ba9e1c6e…` at `56c7b6c`), first P frame,
inter:

| sequence | q | bypassed | A: prefix ctx | B: sign ctx |
|---|---|---|---|---|
| crowd_run | 90 | 50.1% *(identical)* | −2.79% *(identical)* | −2.00% *(identical)* |
| crowd_run | 95 | 59.8% | −4.43% | −1.62% |
| crowd_run | 99 | **74.6%** | **−8.20%** | −1.17% |
| bbb_extended | 90 | 29.9% *(identical)* | −0.55% *(identical)* | −0.81% *(identical)* |
| bbb_extended | 95 | 34.3% | −0.88% | −0.72% |
| bbb_extended | 99 | **46.7%** | **−2.44%** | −0.57% |
| old_town_cross | 90 | 47.3% *(identical)* | −2.85% *(identical)* | −2.33% *(identical)* |
| old_town_cross | 95 | 58.3% | −4.55% | −1.82% |
| old_town_cross | 99 | **74.8%** | **−9.07%** | −1.29% |

**The q=90 control is identical to the digit on all three sequences and all three columns**, which
is what makes the q≥95 rows a change rather than a re-measurement.

**One conclusion strengthens and it is the load-bearing one.** Candidate A now clears ENT-9's
≥2% gate on **three of three** sequences (−2.44% / −8.20% / −9.07%) where the pre-RATE-3 figures
cleared it on two and reached 1.84% on the third. bbb_extended moved most — bypass 43.6% → 46.7%
— which is consistent rather than surprising: RATE-3's bit-exact reference carries detail a lossy
one had quantised away, so the residual against it holds larger magnitudes, and larger magnitudes
are exactly what falls out of the three coded decisions into the bypassed suffix.

**Candidate B is below the gate on three of three either way** (−0.57% to −1.29%), so its
disposition does not move.

The pre-RATE-3 tables below stay as they were taken and are correct for `f3f7254`; where the two
disagree, these are the current figures.

## Why this was worth measuring before building anything

ENT-3 (`0045`) found abac's saving against Rice decaying monotonically with quality — on P-frame
bytes −20.6% at q=90 to −14.5% at q=99 on bbb_extended, and −12.2% → −4.3% / −11.9% → −3.7% on
crowd_run and old_town_cross. That is a mechanism-shaped result with no mechanism attached, and
the two available readings pointed in opposite directions: either abac is running out of
statistical structure to exploit (nothing to do), or it is coding a shrinking *share* of the file
(a lever). The split above distinguishes them, and it cost a `printf`: every term was already
being accumulated by `coef_entropy_diag` for `Hctx`, and the bypass is countable exactly, since a
p=1/2 bit costs one bit.

## Evidence

`GNC_COEF_ENTROPY_INTER=1` (first P frame) and `GNC_COEF_ENTROPY=1` (first I frame), crowd_run /
bbb_extended / old_town_cross, ki=9, 4:4:4, `--abac`. Share of `Hctx` that abac bypasses:

| sequence | q=75 | q=90 | q=95 | q=99 |
|---|---|---|---|---|
| crowd_run, inter | 34.7% | 50.1% | 60.0% | **75.1%** |
| crowd_run, intra | 43.1% | 52.7% | 60.6% | **73.7%** |
| bbb_extended, inter | — | 29.9% | — | **43.6%** |
| bbb_extended, intra | — | 49.6% | — | **61.6%** |
| old_town_cross, inter | — | 47.3% | — | **75.5%** |
| old_town_cross, intra | — | 46.3% | — | **72.4%** |

Broken out at crowd_run q=99 inter: significant 7.1%, `>1` 9.6%, `>2` 8.2% context-coded;
**Exp-Golomb 58.5%**, sign 16.6% bypassed. The suffix, not the sign, is the mass.

**The candidates, priced at `f3f7254` — pre-RATE-3, superseded by the table above but kept because the argument below is built on them (inter, negative = smaller):**

| sequence | q | bypassed | A: prefix ctx | B: sign ctx | A+B |
|---|---|---|---|---|---|
| crowd_run | 90 | 50.1% | −2.79% | −2.00% | −4.79% |
| crowd_run | 99 | 75.1% | **−8.31%** | −1.18% | −9.49% |
| bbb_extended | 90 | 29.9% | −0.55% | −0.81% | −1.36% |
| bbb_extended | 99 | 43.6% | **−1.84%** | −0.50% | −2.34% |
| old_town_cross | 90 | 47.3% | −2.85% | −2.33% | −5.18% |
| old_town_cross | 99 | 75.5% | **−9.35%** | −1.31% | −10.66% |

## The mechanism is confirmed twice, not once

**The bypass share predicts which sequence keeps its advantage.** At q=99, inter: bbb_extended
bypasses 43.6% and keeps a −14.5% saving against Rice; crowd_run bypasses 75.1% and keeps −4.3%;
old_town_cross bypasses 75.5% and keeps −3.7%. Monotone across all three, and the ordering is the
same one `0045` measured independently.

**And the fix is largest exactly where the deficit is.** Candidate A is worth −8.31% and −9.35% on
the two sequences whose saving collapsed, and −1.84% on the one that did not (post-RATE-3: −8.20%,
−9.07% and −2.44%, same ordering). A lever that is biggest where the problem is biggest is the
shape of a real mechanism rather than a coincidence.

**The bounds agree with each other, which they did not have to — once put on one denominator.**
`0045` reported shipped sitting **+12.5% over `Hnb`** on crowd_run inter at q=99, which is `Hnb`
sitting **11.1% below shipped** (12.5/112.5); candidate A's **−8.31% of `Hctx`** is **−8.28% of
shipped**, since `Hctx` is 0.997 of shipped there. So A recovers 8.28 of the 11.1 points available
to a whole-magnitude model with 24 contexts instead of 50 — *less*, as it must be, and three
quarters of it. The two figures are not directly comparable as printed, and quoting "8.31 against
12.5" would have been comparing a share of the coder's bits with a ratio of two bounds.

And `hnb_bits()` adds `sign_bits` unmodelled, so candidate B's −1.18% sits **outside** that bound
rather than inside it: the ceiling on crowd_run q=99 is ≈12.3 points of shipped, not 11.1, and no
bound in this repository had priced the sign before today.

## Why candidate A and not the others

Two things collapsed on inspection, and both save work:

- **"A context for the first suffix bit" and "more `>k` decisions" are the same lever.** The
  Exp-Golomb prefix is a unary code, so its bit `i` *is* the decision "is the magnitude past
  threshold `i`" — exactly what `>1` and `>2` are, continued past where abac stops asking. ENT-9
  filed them as two candidates; they are one, and candidate A is its general form (all prefix
  positions, not just the first).
- **The mantissa is left bypassed on purpose.** It is the low bits of a magnitude and there is no
  causal information about it, so modelling it would add contexts for noise. The measurement
  charges it as bypass in every arm.

### Step 2 milestone 1 — candidate A survives real per-block adaptation

`0063` named this the first thing step 2 must check, and named the reason: step 1b's bound pools
statistics per plane and subband, so it charges no adaptation and lets every block share one set
of counts, while a real implementation cold-starts **24 new contexts per 64×64 code-block** on the
same 4096 symbols the existing 18 learn from. That is the effect that collapsed abac's own
256-stream variant from −6.6% to −0.7%, so the pooled figure was not evidence about it either way.

`adapt_bits_prefix_ctx` runs abac's **real probability engine** over the real shipped code-blocks —
same `Prob`, same `ADAPT_SHIFT`, same cold start, charging −log2 p per decision — with the
Exp-Golomb prefix context-coded instead of bypassed. Both arms cold, so only the binarisation
differs:

| sequence | q | step 1b, pooled | **milestone 1, adaptation charged** |
|---|---|---|---|
| crowd_run | 90 | −2.79% | **−2.83%** |
| crowd_run | 99 | −8.20% | **−8.37%** |
| bbb_extended | 90 | −0.55% | **−0.39%** |
| bbb_extended | 99 | −2.44% | **−2.49%** |
| old_town_cross | 90 | −2.85% | **−2.63%** |
| old_town_cross | 99 | −9.07% | **−8.70%** |

**The win survives intact — within ±0.4 points everywhere, and larger with adaptation charged on
three of the six points.** The 256-stream precedent does not transfer, and the reason it does not
is worth keeping: that case gave each coder ~256 symbols to learn 18 contexts on, whereas here the
24 new contexts sit inside a 4096-coefficient block and are exercised only by coefficients with
|v| > 2 — still hundreds to thousands of decisions each at the qualities that matter. Where the
adaptive arm *beats* the pooled bound, it is doing something a pooled estimate cannot: tracking
statistics that vary within the block.

**The denominator was checked rather than assumed, and it barely moves.** Every candidate-A figure
above is a share of *the coder's own bits*, while ENT-9's gate is a share of **total rate** — not
the same denominator, and the difference runs against the item. Adding the per-block length
fields, which ride along unchanged in both arms, moves it by **≤0.01 points** (crowd_run −8.37% →
−8.37%, bbb_extended −2.49% → −2.49%, old_town_cross −8.70% → −8.69%): the fields are a few KB
against 2.9–5.0 MB of abac tile bytes per frame. What is still uncounted is frame headers and
motion vectors, which abac does not code — at q=99 the tiles dominate the frame, so the total-rate
figure will be close but strictly smaller, and only a real encode settles it. **The gate is still
not cleared; the bound is.**

**What this clears, precisely.** It clears the *bound* on three of three sequences at q=99
(−2.49% / −8.37% / −8.70%). It does **not** yet clear ENT-9's gate, which is ≥2% of **total rate**
at bit-identical pixels — that is a real encode, and total rate carries the per-block length
fields and container overhead these figures exclude. The remaining risk is now implementation
cost, not whether the signal is there.

## What was not chosen

**Building candidate A into the bitstream.** It changes the format, so the CPU coder,
`abac_encode.wgsl` and `abac_decode.wgsl` move together and get re-verified byte-exact three ways
— ENT-5-scale work and its own claim. Milestone 1 above is deliberately the cheap half: it uses
the real engine on real blocks and answers the only question that could have killed the item, for
no bitstream risk at all.

**Candidate B, for now.** −0.50% to −1.31% at q=99 is below ENT-9's own ≥2% gate, and it is a
gate this repository has already used to close ENT-6 at 1.3%. It costs 9 contexts and a signed
neighbour array in both shaders, and it should not be spent on its own. Worth re-pricing *after*
A lands, since A changes the denominator — and worth knowing it is real rather than zero, which
is more than was known this morning.

**Charging the candidates a model cost.** Both are reported as ideal-adaptive bounds with no
signalling and no adaptation loss, the same convention as `Hnb`/`Hbig` — generous to the
alternative on purpose. ENT-6 measured abac's real adaptation loss at under 0.7% of rate, so the
realisable figure is the same order, not a different one. A candidate that fails a generous bound
fails; A does not fail it.

## What this does not settle

**Every figure is the *first* P frame.** The diagnostic fires once through a `OnceLock`, so it
prices a P frame predicting from an I frame. Later P frames predict from P frames and their
residual statistics differ — reference drift is the subject of BUG-27 and RATE-3 — so the bypass
share deeper in a GOP is unmeasured, in an unobvious direction. Step 2's gate is on whole-file
rate and does not inherit this.

**Whether the realisable win survives adaptation.** The bounds charge no adaptation loss. ENT-6
measured abac's real loss at under 0.7% of rate, so the same order is expected, not the same
number — and 24 new contexts learn on the same 4096-symbol code-block that already carries 18, so
the per-context sample count falls. That is exactly the effect that collapsed the 256-stream
variant from −6.6% to −0.7% (`abac.rs`, "Why code-blocks and not GNC's 256 streams"), at a much
more extreme ratio. It is the first thing step 2 should check on real blocks rather than pooled.

## Cost

Read-only additions to `coef_entropy_diag`, inside the existing `GNC_COEF_ENTROPY` /
`GNC_COEF_ENTROPY_INTER` gates: three counters, two context arrays (24 and 9), a signed
neighbour array, and a report block. No shipped code path changes and the bitstream cannot move —
verified, not assumed: same input with the gate set and unset both hash `756c0cbd…`.
