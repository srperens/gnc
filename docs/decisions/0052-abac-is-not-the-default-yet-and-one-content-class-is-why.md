# 0052 — abac is not the default yet, and one content class is why

**Date:** 2026-09-14
**Item:** ENT-10 (step 2 — the decision) — blocked on **ENT-14**
**Status:** accepted
**Machine:** Apple M1 Pro, 16 cores, 16 GB, Metal — `gnc gpu-info`. One of two Macs (COORDINATION).
**Binary:** `codec-fingerprint v1 fb2fe82a` throughout (post-`0051`). **Unchanged by this record:
the default did not move.**

*(Number out of sequence: `scripts/claim dr` fills the first free id, and 0051–0053 were vacated by
BUG-19's renumber.)*

## Context

ENT-10 asks whether abac should replace Rice as the default entropy coder above the rANS cutoff.
It has been open since 2026-09-08 and has always been blocked on cost. `0051` removed most of the
cost — abac's GPU encode was occupancy-starved, not slow, and cb=64 → 32 bought 2.3x encode and 2x
decode. The phase-1 rule (GOALS §1, owner 2026-09-14) settles the rest: **a rate win is worth an
encode cost while the rate gap to H.264 is open.**

So the rate case was measured properly, and it is strong.

## What was measured, and it says yes

All at identical pixels — entropy coding is lossless over the same quantised coefficients, and the
measured PSNR delta is **0.00 dB in every channel at every point**.

**Stills** (`gnc benchmark`, 3 photographic images, abac vs Rice):

| q | 25 | 50 | 75 | 85 | 90 | 95 | 99 | 100 |
|---|---|---|---|---|---|---|---|---|
| bbb | −10.1% | −12.1% | −12.4% | −13.5% | −13.2% | −12.7% | −14.9% | −14.9% |
| crowd_run | −11.8% | −13.2% | −12.4% | −12.5% | −11.9% | −13.0% | −13.0% | −13.0% |
| stockholm | −14.0% | −15.0% | −14.1% | −14.1% | −13.3% | −12.2% | −11.8% | −11.8% |

**Sequences** (`benchmark-sequence -n 9 -k 9 --throughput`, 4 clips, inter):

| | q=90 | q=95 | q=99 |
|---|---|---|---|
| crowd_run | −12.3% | −12.8% | −12.8% |
| rush_hour | **−16.8%** | −14.1% | −13.4% |
| stockholm | −13.1% | −11.9% | −11.8% |
| bbb | **−17.4%** | −15.9% | −14.1% |

**Three things this settles.** (1) **The saving does not decay on inter** — `0045` recorded it
falling to −3.7% at q=99; post-ENT-9 it holds at −11.8% to −17.4%, so that entry is superseded by
the coder having changed under it, not by an error. (2) **It does not decay at q=100 either**: both
coders stay bit-exact (PSNR `inf`) and abac is 11.8% to 14.9% smaller. (3) **The rANS cutoff is
still the right boundary and did not need moving** — abac *loses* to rANS below it and wins above
it, on all three images, at the same q: q=15 reads +5.8% / +0.3% / −0.1%, q=20 reads +1.6% / −2.8%
/ −3.4%, q=25 reads −6.0% / −7.2% / −7.9%.

The flip was made on that evidence.

## And then a unit test refused it

`regression_gradient_q50` and `_q75` failed within seconds: **bpp 0.3379 against a 0.2397 ceiling,
+41%.** On a 512x512 synthetic gradient:

| q | Rice | abac cb=64 | abac cb=32 (today's default) |
|---|---|---|---|
| 25 | 3 227 B | +100.6% | **+234.5%** |
| 50 | 3 636 B | +88.4% | +207.2% |
| 75 | 5 232 B | +39.5% | +122.1% |
| 90 | 7 849 B | +9.1% | **+64.7%** |

**The cause is measured.** `GNC_DIAGNOSTICS=1` at q=25 prints `abac_blocks=280 (empty=0)` per
plane — **840 code-blocks, not one recognised as empty.** The three planes code 9 641 bytes, i.e.
**~11.5 bytes per block**, against Rice's 3 227 for the whole picture. Every code-block pays a
terminated arithmetic interval plus a 4-byte length word whether or not it contains a single
significant coefficient. **On sparse content the file is nothing but floor.**

## Decision

**abac does not become the default. The blocker is filed as ENT-14 (P0) and the flip lands when it
closes.** `quality_preset` keeps Rice above the rANS cutoff, `--abac` stays the opt-in it has been
since `0017`, and `0051`'s cb=32 stays because it wins on every real picture and buys the 2.3x.

**This is not caution about an edge case.** Flat frames, fades, title cards, letterbox bars and —
the one that matters — **static P-frames** are ordinary contribution content. GOALS says in as many
words that *"a static studio shot should cost almost nothing"*. A coder with an 11.5-byte floor per
code-block cannot deliver that, and **every sequence row above is busy content**: nobody has yet
run abac on a static shot, and the gradient predicts what that will show.

**The honest reading of the good numbers is that they are real and incomplete.** −10% to −17% at
identical pixels across eight quality points, three images, four sequences and both sides of the
intra/inter line is not a fluke and will not evaporate. It was measured at the end of the range
where GNC is interesting and not at the end where it is sparse — which is the failure mode GOALS
names directly: *"what is not acceptable is a strategy that only works at one end and is quietly
measured only there."* The test caught it because a synthetic gradient is in the regression suite
for exactly this reason.

## What was rejected

**Shipping it and widening the gradient baselines.** The baselines are not wrong; the coder is.
Moving a ceiling to admit a 41% regression would have turned the one instrument that caught this
into an instrument that cannot catch it again.

**Shipping it per operating point — abac above q=85, Rice below.** Tempting, because the penalty
falls with q (+234.5% at q=25, +64.7% at q=90). Rejected: the gradient is still **+64.7% at q=90**,
which is inside the contribution range GNC is *for*, so the threshold would have to sit above the
range it was supposed to serve. The variable that predicts the penalty is **content sparsity**, not
quality, and a q threshold is a proxy for the wrong thing — `0051`'s expired decode note is the
same mistake one layer down.

**Choosing per tile by coding both and keeping the smaller.** The machinery exists (RATE-2 and
LOSSLESS-3 do exactly this at frame level) and it would work. Rejected as a *fix*: it doubles
encode to paper over a defect whose real repair is one bit per block. Reasonable as a fallback
later if ENT-14 turns out harder than it looks; it is not the first move.

**Reverting `0051`'s cb=32.** cb=64 is less bad on the gradient (+9.1% against +64.7% at q=90) but
still loses, so it does not solve this — and it costs 2.3x encode and 2x decode on everything else.
The floor is the defect at both sizes. **But the two interact: `cb` must not be tuned again until
ENT-14 is fixed**, because halving the block count halves the floor and would flatter any future
cb sweep for the wrong reason.

## Consequences

- **ENT-10 stays open at P0**, now with its rate side fully measured and one named blocker instead
  of an open question. When ENT-14 closes, the flip is a one-line change plus this record amended.
- **ENT-14 (P0) filed** with the diagnosis, the standard fix (signal an empty block in a bit, as
  JPEG 2000 signals zero bit-planes), and five success criteria written before the work.
- **ENT-13 (P1) filed** at the owner's request for abac's remaining *structural* cost — it codes
  every frame twice, and the second pass exists only to learn the first one's answer.
- **`0045` is superseded on the inter figure**: −3.7% at q=99 has become −11.8% to −17.4%, through
  ENT-9 and `0051` rather than through an error in it.
- **BASELINE's `--abac` rows predate `0051`** and are at cb=64. Re-take is MEAS-11, already open,
  and should wait for ENT-14 so it is taken once.
- `test_entropy_coder_follows_quality` now carries the reason abac is absent, so the next reader
  does not re-derive it.
