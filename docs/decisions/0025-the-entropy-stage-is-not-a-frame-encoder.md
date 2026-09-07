# 0025 — The entropy stage is not a frame encoder

**Date:** 2026-09-07
**Status:** accepted
**Items:** ARCH-3 (P1), BUG-18 (P1)
**Supersedes nothing. Lifts the retraction recorded in `457bbda` / COORDINATION "Two retractions".**

## The decision

`gpu_entropy_encode` now selects **where the entropy stage runs, and nothing else**. The two
independent whole-frame P and B encoders it used to choose between are one encoder: the batched
single-command-encoder pipeline. When the entropy stage runs on the CPU, the same pipeline runs,
the same coefficients come out of it, and they are read back and coded after the submit.

The second implementation — ~360 lines for P, ~200 for B, each with its own motion estimation and
its own local decode — is deleted rather than fixed.

## Why this and not the alternatives

Three options were on the table in ARCH-3, and they are not substitutes.

**Fix the non-batched implementation.** This was the obvious repair and it is the wrong one. The
two implementations did not differ in a bug; they differed in *design*. The batched one runs
4-stage pyramid ME (±96 px effective), tile-skip motion, optional MV smoothing and a look-ahead;
the other ran a single-level ±32 px search and none of the rest. Reconciling them means porting
the whole of one into the other, after which they are the same code — so the honest end state was
always "one implementation", and repairing first only pays for the trip twice. It also leaves the
class intact: the *next* coder that lands decode-first still gets routed somewhere by a flag about
entropy.

**Give abac a GPU encoder.** Worth doing, and now claimed as ENT-5, but it does not fix this. It
removes abac from the list of coders that trip the defect and leaves the defect. Bitplane is still
on that list, Huffman was on it in a different way (below), and the point of ARCH-3 is that the
list is not the problem.

**Separate the concerns.** Chosen. It is the smallest change of the three, it removes the class
rather than an instance, and it makes BUG-18's open cause 2 — "why do two implementations
disagree" — stop being a question, because there is one implementation.

## What it cost

Deletion, not addition: about 1200 lines removed against 400 added, counting whitespace-insensitively
(`git diff -w`); the raw diff is much larger because deleting the `if` dedented the surviving
branch. Two capabilities
went with the removed implementation, and both were already dead:

- **The temporal MV predictor.** `encode_pframe` took a `predictor_mvs` buffer and returned its
  own 16x16 MVs so the caller could feed them to the next frame. Only the deleted implementation
  read it; the batched one has used the pyramid predictor since it was written. So the shipped
  encoder never did temporal MV prediction, and the plumbing that said it might was removed
  along with the only code that could have used it. Rebuilding it later is a real change with a
  real measurement, which is the honest state.
- **`GNC_NO_LOOKAHEAD_ME`'s partner path.** Unaffected — that lever still selects whether the
  look-ahead's vectors are reused.

Nothing else: the default configuration is byte-identical, verified below.

## The second instance, found by measuring

Removing the frame-encoder conflation was not enough to make "the entropy choice does not touch
the pixels" true. `dispatch_zero_skip_tiles_by_map` — which zeroes quantised coefficients for
tiles the motion search flagged as static — was gated on `entropy_mode == Rice`, while
`dispatch_tile_skip_motion`, which zeroes those tiles' *motion vectors*, ran for every coder. So
abac and bitplane paid skip mode's prediction cost and collected none of its rate saving, and Rice
and abac coded **different coefficients for the same frame**.

That gate is now removed too. Which coefficients get coded is a rate/quality decision every coder
shares; only the size of the reward is coder-specific, since Rice has a compact
`TILE_FLAG_ALL_SKIP` and the others simply code the zeros.

This one is worth recording separately because of *how* it was found: it is invisible on a
synthetic full-frame pan, where no tile is ever static, and it showed up immediately on 1080p
content. A unit test written against convenient content would have certified the fix as complete.
The test now runs both a pan and a half-frozen frame.

## What it fixes that was not asked for

**`--huffman` video was broken on `main`.** Huffman has no GPU encoder wired into the inter path,
but it was not on the list of coders forced onto the second implementation, so it took the batched
pipeline — which pushed nothing into `huffman_tiles`. Every P-frame it wrote carried an empty tile
vector, and decoding one panics in `frame_data.rs`. Nothing caught it because no test encoded
Huffman video. It works now, and `every_coder_codes_a_p_frame` is the test that would have.

## The evidence

Measured against a pinned baseline binary built from `07c01b1` in a detached worktree, per
COORDINATION rule 1 and the "a sibling worktree's build is not a baseline" note.

**The default path does not move.** `.gnv` byte-identical on **54 of 54** configurations: bbb (8
frames) and bbb_extended (18), ki=2 and 9, q=50/75/90, 4:4:4 / 4:2:2 / 4:2:0, B-pyramid on and
off. So this change invalidates no measurement in the repository.

**BUG-18 is closed.** `tests/bug18_locate.rs`, the two arms differing only in
`gpu_entropy_encode`:

| q | ki | frame 1 (first P) | frame 2 | frame 3 | after |
|---|---|---|---|---|---|
| 50 | 9 | 28.8 | 55.8 | 62.9 | **0.000** |
| 90 | 9 | 4.24 | 4.65 | 4.59 | **0.000** |
| 50 | 2 | 28.8 | 0.000 | 26.8 | **0.000** |
| 90 | 2 | 4.24 | 0.000 | 5.27 | **0.000** |

Zero differing samples at every point, not a tolerance.

**abac's inter figure is measurable again, and it is not −14.4%.** Rice against abac, 18 frames,
ki=9, 4:4:4, decoded PNGs hashed to prove the pixels are bit-identical rather than inferring it
from matching PSNR:

| sequence | q=50 | q=75 | q=90 |
|---|---|---|---|
| bbb_extended | −16.3% | −22.9% | −19.7% |
| crowd_run | −20.7% | −18.4% | −12.1% |
| old_town_cross | −22.7% | −21.8% | −12.0% |

**−12.0% to −22.9% at bit-identical pixels.** There is no rate/quality trade to argue about here:
the two files decode to the same bytes. The retracted figure was measured with abac on the broken
frame encoder and Rice on the working one; this replaces it. It says nothing about throughput —
abac still has no GPU encoder (ENT-5) — and nothing about whether abac should be the default,
which `0017` decides on other grounds.

## What this does not fix

The encoder's local decode dequantises P-frame residuals with `config.quantization_step` while the
forward pass quantises them with `res_qstep = quantization_step × p_qp_scale`, so above q≈70 the
encoder's reference drifts from the decoder's by that factor. That was true on both implementations
and is true on the one that remains, so it is untouched here and unaffected by this change. It
belongs to **BUG-8**.
