# 0063 — abac bypasses three quarters of its own bits at q=99, and the Exp-Golomb prefix is where the deficit is

**Date:** 2026-09-08
**Item:** ENT-9 step 1 (and 1b, the pricing)
**Status:** accepted — step 1 answered, step 2 scoped and ordered; no default moves, no bitstream change

## Decision

**ENT-9 step 1 is answered and the item is not small.** abac context-codes exactly three binary
decisions per coefficient — significant, `>1`, `>2` — and sends the Exp-Golomb order-0 remainder
and the sign as bypass bits at p=1/2. Measured on the shipped tiles:

**At q=99, 75.1% / 75.5% / 43.6% of abac's own bits are bypassed** — sent with no model at all.
The context model touches a quarter of the file on two of three sequences.

**Step 2 goes to candidate A first, and candidate B is below the gate.** Both were priced in the
same read-only pass, before either was built:

- **A — context the Exp-Golomb unary prefix** (6 buckets × 4 positions = 24 extra contexts):
  **−1.84% to −9.35% of the coder's bits at q=99.**
- **B — context the sign** on the left and up neighbours' signs (3×3 = 9 contexts): **−0.50% to
  −1.31% at q=99**, below ENT-9's ≥2% gate on all three sequences.

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

**The candidates, priced (inter, negative = smaller):**

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
the two sequences whose saving collapsed, and −1.84% on the one that did not. A lever that is
biggest where the problem is biggest is the shape of a real mechanism rather than a coincidence.

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

## What was not chosen

**Building candidate A in this session.** It changes the bitstream, so it needs the CPU coder,
`abac_encode.wgsl` and `abac_decode.wgsl` moved together and re-verified byte-exact three ways,
which is ENT-5-scale work and its own claim. The bound is what decides whether that is worth
starting, and it is; step 2 is filed with this ordering.

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
