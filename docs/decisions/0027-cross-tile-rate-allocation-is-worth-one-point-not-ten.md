# 0027 — Cross-tile rate allocation is worth 1 point, not 10 — hypothesis rejected

**Date:** 2026-09-07
**Status:** accepted
**Item:** BACKLOG INTRA-1 (P0), step 2, second instalment
**Builds on:** `0026` (the gap decomposes), `0024` (the gap is upstream of the entropy coder)

## The hypothesis, and why it was worth testing

Decision `0026` found an asymmetry it could not explain. Giving JPEG 2000 GNC's 256px tiling costs
it **8.8–12.4 points**, but doubling GNC's own tile to 512 is worth only **0.6%** — a 6x difference
over the same step. Something J2K gets from a whole-picture configuration, GNC cannot collect.

The leading explanation was **global rate allocation**. Untiled OpenJPEG runs PCRD across the whole
image, so it can spend more on a hard region and less on an easy one. GNC gives every tile the same
quantiser step above q=80 — adaptive quantisation is enabled only for `q` in `30..=80`. EBCOT part 1
closed PCRD at *code-block* granularity **inside** a tile (0.00 dB); across tiles it had never been
measured.

## What was measured

`scripts/meas_cross_tile_rd.py` — an **oracle**, deliberately not an implementation. Encode at every
q on a 12-rung ladder with `--abac`; take per-tile rate (`GNC_TILE_RATE=1`, summed over the three
planes) and per-tile squared error in RGB against the original. Then for a Lagrangian lambda give
every tile independently the q minimising `D_t + lambda * R_t`, and compare at **matched total
distortion**.

The oracle is generous on purpose: it sees the future, it is charged nothing for signalling a
per-tile q (about a byte per tile, 0.02% here), and it may pick any rung for any tile. If a free,
clairvoyant allocator cannot find the points, a built one will not either.

**Per-tile distortion is scored on visible pixels only.** The tile grid sits on the *padded* plane,
so the right column and bottom row of tiles extend past the picture; scoring the padding would
credit those tiles with error they do not carry and make them look free to coarsen — the one way
this measurement could have favoured the oracle by accident.

## The result

**−0.95% at q ≥ 85**, mean over four images (bbb −1.60%, blue_sky −1.19%, kristensara −0.14%,
touchdown −0.88%).

Against a remaining unexplained gap of ~9.8 points, that is about **one point of ten**. The
hypothesis is rejected: cross-tile rate allocation is not where the JPEG 2000 gap is.

**And the mechanism is visible, not merely inferred.** On kristensara the oracle picks a *single* q
for every one of the 15 tiles at q = 92, 94, 96 and 98 — the spread column reads `92-92`, `94-94`,
`96-96`, `98-98`. Uniform q is not merely close to optimal there, it *is* the optimum. That is the
same argument EBCOT part 1 made for code-blocks — uniform scalar quantisation of a near-orthonormal
transform under MSE puts everything at the same RD slope — and `0026` measured that GNC's transform
really is near-orthonormal (synthesis norms 0.984–1.066). The two findings are the same fact at two
scales, and there is nothing left for an allocator to move.

## Two instrument faults found on the way, both of which produced a wrong answer first

Recorded because each one produced a plausible number that would have been published.

1. **The oracle "lost" by +0.24%.** A clairvoyant allocator cannot do worse than fixed q — the
   all-tiles-same-q allocation is inside its own search space. The cause was comparing two fitted
   curves and integrating between them. Replaced with an exact matched-distortion comparison, and a
   **dominance assertion** now fires rather than reporting an impossible saving.
2. **The "purely spatial" column was identical to the raw column, because the convex envelope step
   was described in a comment and never implemented.** Without it the interpolation walks the
   Pareto staircase itself, so a ladder rung sitting *above* its own chord is treated as reachable
   and its inefficiency is credited to the spatial lever. **bbb's q=90 rung is 5.3% above the chord
   between q=88 and q=92**, and q=92 is 2.0% above — that is a mispriced rung (RATE-2's territory),
   not a cross-tile gain. Fixing it moved the headline from −1.21% to **−0.95%**.

The second one is the more instructive: the comment was right and the code was wrong, and the two
columns agreeing exactly to the hundredth of a percent across 48 rows was the only visible symptom.

## What INTRA-1 has now eliminated

| candidate | worth | how |
|---|---|---|
| entropy coding | ≤7.5 points | `0024` |
| chroma allocation vs an RGB metric | 8.5 points, and it is not a deficiency | `0026` |
| lifting normalisation | ~0 | `0026` |
| tiling, as wavelet reach | 0.6% realisable in GNC | `0026` |
| **cross-tile rate allocation** | **0.95%** | **this record** |

**~9 points remain and the obvious candidates are now spent.** What has *not* been tested: the
deadzone and quantiser rounding rule against J2K's, and the wavelet's tile-boundary handling
(`transform_97.wgsl` replicates the edge sample where J2K uses symmetric extension). Both are
cheap. Neither is obviously worth 9 points, which is itself worth saying out loud.

## What was not chosen

- **Building a per-tile rate controller anyway.** At a 0.95% ceiling for a *clairvoyant* allocator,
  a real one — which must search or model — would land well under that, against a bitstream change
  and an encoder search. CLAUDE.md's "know when to stop" applies.
- **Extending AQ above q=80.** It was the cheap way to capture this lever, and there is no longer a
  lever to capture. AQ's own range is a perceptual-masking decision and is untouched by this.
- **Re-deriving the 9.8-point remainder before testing.** The oracle bounds the lever independently
  of what the remainder turns out to be, so it stays valid if the remainder moves.
