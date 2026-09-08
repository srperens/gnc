# 0072 — The sibling's padding was the regression, and a bit-exact frame's reference is its source

**Date:** 2026-09-08
**Status:** accepted. Two changes ship; one previously-recommended design is retired unbuilt.
**Item:** RATE-4 (P3) — **closed, criterion met**
**Corrects figures in:** `0044` (RATE-3's mean and its two regressions), `0068` (the ledger's prize)
**Files:** BUG-47 (fixed here), BUG-45 (warned here, open)

## What changed

1. **`lossless_sibling` now inherits `pad_fill_decay`** (BUG-47). It did not, so a bit-exact
   I-frame kept at q = 95..=99 was coded with **decay-filled** padding while the sequence encoder
   had deliberately cleared that flag for every frame something predicts from (`0039`, and the
   INTRA-2 reasoning beside it). One line.
2. **A bit-exact I-frame's reference is built from its colour-converted source** rather than by
   dequantising and inverting what was coded, so **RATE-3's third encode is gone** on those
   frames.
3. **`0068`'s one-frame lookahead ledger is retired unbuilt.** After (1) there is nothing left for
   it to win.

## 1. The regression was the padding, not the ledger

RATE-3 shipped the fallback on sequence I-frames and measured two regressions: bbb q=99, +0.58% at
ki=2 and +0.40% at ki=9 (`0044`). RATE-4 was filed on the theory that the *ledger* was wrong — the
candidate is chosen on one frame's bytes and paid for by the next one's — and `0068` measured that
theory carefully: an exact per-GOP ledger was worth **0.09 points of mean** and removed both
regressions.

It was the wrong cause. RATE-3's own harness, `scripts/meas_rate3.py`, unchanged parameters:

| | mean of 12 | best | worst point | worst P move | points worse than control |
|---|---|---|---|---|---|
| before (`0044`) | −4.28% | −13.16% | **+0.58%** | −0.01 dB | **2 of 12** |
| after the one-line fix | **−6.09%** | **−16.19%** | **+0.00%** | −0.01 dB | **0 of 12** |

**RATE-4's success criterion is met in full** — "no point in RATE-3's table larger than the control
arm, mean no worse than −4.28%, worst P within 0.1 dB" — and met by a field assignment rather than
by the machinery `0068` priced.

**Why padding mattered this much.** The padded region is not a rounding detail at 1080p: a
1920×1080 frame is coded at tile-aligned dimensions, and motion compensation reads that region
whenever a block near the frame edge is predicted. `0039` made the fill a lever and the sequence
encoder clears `pad_fill_decay` for referenced frames because a decayed pad is a *worse reference*
— it flattens edge detail the next frame then has to re-code. `lossless_sibling` starts from
`quality_preset(100)`, which sets the flag back to `true`, and it was not in the carry-over list.
So the one frame in the sequence most likely to be a reference was the one padded as if it were a
still.

**And `0068`'s ledger is now worth nothing measurable**: with the fill fixed, the per-GOP oracle
changes the choice on **0 of 38 GOPs** and its mean equals today's to the byte. The per-frame
ledger is already optimal on this data. That is a better outcome than building the lookahead, and
it is the second time in this item that measuring the cause beat implementing the fix for a
mis-attributed one.

## 2. A bit-exact frame's reference is its colour-converted source

`0040` point 4 tried this and reverted; RATE-4's first attempt tried it and read **254.0039**
against the decoder's reference at q=100, which is where it stopped. Both tested a patch. This
tests the **premise**, on the CPU, with no patch involved
(`a_bit_exact_frames_reference_is_its_colour_converted_source`): compute YCoCg-R forward from the
source and compare it against the reference the decoder actually holds.

**It is the same picture, exactly** — 0 of 65 536 pixels differing on all three planes — at q=100
MED *and* at q=99 with the bit-exact sibling kept. Those two had to agree, and saying why is what
made the measurement worth taking: `lossless_sibling` is `quality_preset(100)` with only
how-to-code fields carried over, so they are the same transform, the same reversible colour path
and the same branch of `local_decode_iframe_gpu`. **A measurement that separates them is measuring
the instrument.**

So `local_decode_iframe_gpu` colour-converts the source into the reference and returns, and
`encode_as_reference` stops paying the third encode for those frames. **The win is a count, not a
time** (`0058`'s rule): three intra encodes per qualifying I-frame become two, on 52 of 52
qualifying I-frames across the sweep. Encode time is not measured — seven other sessions were on
this machine (COORDINATION rule 1).

**It reads `input_buf`, and that is the whole point.** `plane_a` / `co_plane` / `cg_plane` belong to
whichever candidate ran *last*, so in the fallback case they hold the lossy candidate's **plain**
YCoCg — fractional values where the reference is made of reversible integers. Both previous
attempts read candidate-dependent buffers. The source RGB is identical for both candidates, so
`input_buf` is the one buffer whose contents do not depend on the order of the encodes.

**Except in its padding, which is how BUG-47 was found.** The first version of this change was
byte-identical at q=100 and moved bytes on 10 of 24 sequence points at q = 95..=99 — the fallback
cases, and only those. The two candidates were leaving *differently padded* sources in
`input_buf`. Forcing `GNC_PAD_FILL=replicate` made all 24 identical, which named the cause; the
one-line inheritance fixed it at the source. **The route did not find the padding defect by
accident: it is the first thing that ever compared the two candidates' preprocessing.**

### The gate

`scripts/rate4_ref_source_gate.py`. Arms differ only in `GNC_REF_FROM_SOURCE`; **24 of 24 points
byte-identical**, per frame and in total, across three sequences × {q=95, 99, 100} × {ki=2, 9} plus
4:2:0 at q=99 and q=100. The route fired **52 times** and the script fails if it fires where it
must not: 4:2:0 (0 times — the reference holds nearest-neighbour-upsampled chroma there, which the
source planes are not equal to), the forced-off arm (0 times), or if q=100 4:4:4 ever fires 0 times
(vacuous). Byte identity is the strongest statement this change is allowed to make, and it is why
the rate figures above belong to the padding fix alone.

## 3. BUG-45: `is_lossless()` is a claim about the settings, not about the input

The 254.0039 row is now fully explained, and it is not about q=100. It was measured on
`make_gradient_frame`, whose samples are `x / 256 * 255` — **fractional**. At q=100 the quantiser
step is 1.0 and MED's residual is a difference of *integers*; give it fractional f32 and the step
rounds, so the reconstruction leaves the source and the file is **lossy while every
`is_lossless()` in the codec reports true.** Measured both ways in
`lossless_at_q100_is_a_claim_about_integer_input`: integer input 0.0000, fractional input 254.0039.

Same family as BUG-15 (`chroma_weight` 1.2 at q=100) and BUG-30 (`GNC_DEAD_ZONE` at q=100) — a
setting outside the guarantee, silently taken. It stayed invisible because PNG and Y4M input is
integral, so only an API caller passing `&[f32]` can reach it.

**Warned, not refused.** Samples cannot be normalised without changing the picture, so the options
were to say so or to reject the frame, and rejecting a frame a caller may legitimately want coded
lossily is worse than telling them what they got. `reference_from_source` uses the same predicate
to decline building a reference the reconstruction does not equal. BUG-45 stays open for whoever
decides that refusing is better.

## What was rejected

- **Building `0068`'s one-frame lookahead.** Priced at one extra P-frame encode per GOP and 33-of-33
  agreement with the exact ledger — and now worth **0 of 38 flips**. Retired unbuilt.
- **Gating the source-built reference on `!lossless_fallback`** — i.e. shipping it for q=100
  sequences only, where it was already byte-identical, and leaving the fallback case to the third
  encode. That was the safe answer while the fallback rows moved, and it would have shipped the
  smaller half of the win *and* left BUG-47 in the tree, since nothing else compares the two
  candidates' preprocessing.
- **A tolerance in the byte-identity gate.** The two paths are meant to produce the same picture, so
  the assertion is equality; a tolerance would have swallowed BUG-47 exactly.

## What this does not claim

- **No encode-time figure.** Seven sessions on the machine. Three encodes becoming two is a count.
- **4:2:2 and 4:2:0 rate is unmeasured.** The padding fix applies wherever the sibling is used, so
  subsampled sequences should move in the same direction, but only 4:4:4 was measured. The *route*
  is refused there and that refusal is asserted.
- **Stills are unaffected, and that is checked rather than argued**: for a still both configs carry
  `pad_fill_decay = true`, so the inheritance changes nothing — bbb 1080p at q=95 and q=99 is
  byte-identical with the fill forced either way (2 336 979 B and 3 257 157 B).
