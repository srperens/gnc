# 0068 — The I-frame ledger is one frame deep, the penalty is paid once, and the fix is priced but not built

**Date:** 2026-09-08
**Status:** accepted as a measurement and a design decision. **No code shipped; the encoder is
unchanged and every RATE-3 figure stands.**
**Item:** RATE-4 (P2 → P3)
**Builds on:** `0044` (RATE-3 shipped the fallback on sequence I-frames and measured this as its
cost), `0036` (the still-image version), RATE-2

> **The design below is retired unbuilt, and the cause it was priced against was the wrong one —
> `0072`, the same day.** RATE-4's regressions were `lossless_sibling` not carrying
> `pad_fill_decay`, so the bit-exact candidate was coded with a still's padding while acting as a
> reference (BUG-47). With that one line fixed, the exact per-GOP ledger measured here changes the
> choice on **0 of 38 GOPs** and its mean equals the per-frame ledger's to the byte. There is
> nothing left for a lookahead to win.
>
> What survives is not the recommendation but the three measurements, and they are worth keeping
> for the next ledger question in this codec: the penalty is paid by the **first** P-frame and does
> not propagate; a one-frame lookahead reproduced the exact per-GOP decision on 33 of 33 GOPs; and
> **a margin constant would have passed all twelve points**, which is still the strongest argument
> against fitting one. The recommendation to settle the source-copy half first is what led to
> BUG-47, so it was right for a reason it did not know.

## The question

RATE-3 lets a sequence I-frame keep whichever of two candidates is smaller — a lossy wavelet frame
or a bit-exact MED one — **judged on that frame's own bytes**. Inside a sequence that is the wrong
ledger: a bit-exact I-frame carries detail a lossy one quantised away, so the P-frames predicted
from it are larger. Two of RATE-3's twelve points regress for exactly that reason (bbb q=99, +0.58%
at ki=2 and +0.40% at ki=9).

RATE-4 said the honest fix "compares *sequence* bytes, which needs the GOP encoded both ways or a
model of the residual cost", and banned a margin constant. This measures which of those it is.

## What was measured

`scripts/meas_rate4.py`, RATE-3's parameters unchanged — three sequences (bbb 8 frames, crowd_run
10, old_town_cross 10), q ∈ {95, 99}, ki ∈ {2, 9}, `benchmark-sequence`, I+P+B arm only. Two arms
from one binary differing only in `GNC_LOSSLESS_FALLBACK`, so per-GOP totals are exact rather than
modelled. Binary hash-recorded before the sweep; the whole sweep was run twice and every figure is
byte-identical.

**GOP independence was checked, not assumed** — it is the premise the whole unit rests on. No
B-frames at either interval, `rate_ctrl` is `None` without `--bitrate`, `pending_me` is reset at
every keyframe and `gpu_ref_planes` is overwritten rather than accumulated (verified by the RATE-3
session). The harness adds two assertions of its own: both arms must produce the same frame-type
partition before being differenced, and any GOP whose I-frame bytes agree across the arms while its
totals differ is a refusal, not a warning. **0 of 38 GOPs disagreed.**

| | mean of 12 points | worst point | points worse than control |
|---|---|---|---|
| today (per-frame ledger) | **−4.28%** | **+0.58%** | 2 of 12 |
| exact per-GOP ledger | **−4.37%** | +0.00% | 0 of 12, by construction |

## Three findings, in the order that changes the design

### 1. The prize is 0.09 points of mean, and the item is about the regression, not the rate

An exact per-GOP ledger moves the mean from −4.28% to −4.37%. It cannot be worse than the control
at any point, because the control is one of its two arms — which is what RATE-4's ban on a margin
constant was reaching for and could not express. But it buys **0.09 points**, and 5 of 38 GOPs
change their choice. So this is a correctness-of-decision item, not a compression item, and it
should be priced as one.

### 2. The penalty is paid by the *first* P-frame and does not propagate

Per-frame bytes inside each ki=9 GOP whose arms chose differently, bit-exact arm minus lossy arm
(frame 0 is the I-frame):

| sequence | q | I | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 |
|---|---|---|---|---|---|---|---|---|---|---|
| bbb | 99 | −279 336 | **+321 525** | +20 104 | +10 539 | +9 967 | +7 003 | +4 823 | +2 877 | |
| crowd_run | 95 | −645 809 | **+212 583** | +6 768 | +725 | −3 698 | −802 | −375 | −672 | +2 154 |
| crowd_run | 99 | −1 549 505 | **+284 074** | +16 754 | +1 705 | +5 803 | −1 903 | +1 486 | −3 027 | −6 351 |
| old_town_cross | 95 | −575 709 | **+187 647** | +7 328 | +2 931 | +3 363 | +1 517 | +1 496 | −79 | +5 051 |
| old_town_cross | 99 | −1 489 769 | **+249 662** | +10 385 | +6 955 | +3 943 | −5 186 | +2 043 | +2 687 | +790 |

**P2 is 2–6% of P1, and P2..P8 together are 1–17% of it.** The mechanism is plain once seen: P2's
reference is P1's *reconstruction*, which is lossy in both arms, so the extra detail a bit-exact
I-frame carries is re-coded once and gone. It never becomes a property of the GOP.

That inverts the item's own framing. **The ledger does not need the GOP encoded both ways.** A
one-frame lookahead — compare `I + P1` under each candidate's reference — reproduces the exact
per-GOP decision on **33 of 33 GOPs where the arms chose differently**, at one extra P-frame encode
per GOP instead of the losing arm's whole GOP. At ki=2 the two are the same thing; at ki=9 it is an
eightfold difference in cost.

It is an approximation, and the table says by how much: the one-frame ledger is *optimistic about
bit-exact* by the sum of P2.., which is 4 100 B (crowd_run q=95) to 55 313 B (bbb q=99) on GOPs of
roughly 20 MB. So it can be wrong only for a GOP whose true delta sits inside that band. It carries
evidence, not a construction guarantee — that distinction is the whole reason to write it down.

### 3. A margin constant would have passed this twelve-point test, and must still be refused

The item bans a fitted constant. The numbers show the temptation is real: the P1 penalty is roughly
**independent of the I-frame saving** (187 647–321 525 B across six sequence/q pairs, against
savings spanning 279 336–1 549 505 B), so "keep the bit-exact frame iff its saving exceeds T"
reproduces all twelve points for any T in **(279 336, 575 709) B**.

**That window is defined by two of the three sequences**, the penalty already varies 1.7× inside
this small set, every sequence here is 1080p so the constant is untested against resolution, and
the one point that decides the lower bound is the one regression the item exists to fix. A constant
fitted here would be a number that passes its own test set and nothing else. **The ban is correct
and is now measured rather than argued** — which is more useful than the ban was.

## The decision

**The design is the one-frame lookahead; it is priced and deliberately not built.**

The recommendation is to leave it unbuilt until one of two things changes, and RATE-4 is demoted
**P2 → P3** to say so:

- **The mean is 0.09 points** and the two regressions are on one sequence at one quality point.
- **The implementation lands in the fragile place.** Comparing `I + P1` under both candidates means
  holding two live I-frame references: build reference A, encode P1, build reference B, encode P1,
  then restore the winner's reference for P2. The cheap version snapshots and restores
  `gpu_ref_planes` (~24 MB at 1080p 4:4:4) rather than re-encoding, so the marginal cost is one
  extra I and one extra P per GOP. That side channel — `encode_once` leaving quantised planes for
  `local_decode_iframe_gpu` — has already produced four separate defects: `0040`, `0042`, RATE-3's
  own gate, and the ordering constraint recorded in `encode`'s comment. Two live references is
  precisely the shape all four had.
- **The two halves of RATE-4 interact, and the other one should go first.** If the source-copy
  route works (a bit-exact frame's reference *is* its colour-converted source), RATE-3's third
  encode disappears and the ledger's marginal cost roughly halves. That route is refuted at q=100
  and **unexplained at q=95..99, where the encoder's and decoder's references match to 0.0000** —
  which is exactly the case the ledger cares about. Settling that changes this item's price, so it
  is the cheaper thing to do first.

## What was rejected

- **Encode the whole sequence both ways.** Exact, never regresses, and doubles sequence encode
  while breaking the streaming shape of `encode_sequence`. Finding 2 makes it unnecessary: almost
  none of the information it buys is outside the first P-frame.
- **Decide once per sequence from the first GOP, then commit.** On this data it collects the entire
  win, because the choice never varies within a sequence — every one of the 5 flips is in bbb q=99,
  and there either all GOPs flip or none. But that uniformity is measured on 8- and 10-frame clips
  with no scene change; the first GOP of a 300-frame sequence is not evidence about its last.
  Cheaper than the lookahead and with a guarantee that holds only for homogeneous content.
- **Extrapolate the GOP from the measured P1 penalty** (`saving + n_P × penalty`). Refuted by the
  data that suggested it: on crowd_run q=95 ki=9 it predicts the lossy arm by a wide margin
  (−645 809 + 8×212 583) where the exact GOP prefers **bit-exact** (−429 126). Finding 2 says why —
  multiplying a one-frame cost by the GOP length is the one thing the decay curve forbids.
- **A margin constant.** See finding 3. It would have passed.

## What this does not claim

- **No encode-time figure.** Seven other sessions were on this machine; every cost above is a count
  of encodes, not a measurement of them (COORDINATION rule 1). "One extra P-frame per GOP" is an
  arithmetic statement about how many times the coder runs.
- **No quality claim.** No pixels moved: the encoder is untouched, and both arms are existing
  code paths.
- **Nothing about q ≤ 90.** The fallback is gated to q = 95..=99 and so is all of this.
