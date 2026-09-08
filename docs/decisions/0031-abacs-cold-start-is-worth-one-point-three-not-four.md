# 0031 — ENT-6: abac's cold start is worth 1.3% of rate, not 4%, and the 4% was a bound artefact

**Date:** 2026-09-08
**Status:** accepted
**Item:** BACKLOG ENT-6 (P2) — closed by measurement.
**Corrects a figure in** `docs/decisions/0024` **and in RESEARCH_LOG's INTRA-1 step 1 entry.**
Follows from `0030`, which redirected ENT-7's part 2 here.

## The question, and why the existing number could not answer it

ENT-6 was filed on this: at tile 256 with 5 levels the LL and level-3/4/5 bands are 8, 16 and 32
px square, so each becomes **one short code-block** — 64 to 1024 coefficients on which to adapt 18
context probabilities, against 4096 in a full 64×64 block — and abac opens every context at
p = 1/2. `0024` measured those bands at **+25.9% over the entropy bound at q=90** while carrying
18% of the rate, and concluded the cold start was worth **about 4% of the file**.

**That 4% cannot be collected and the measurement could not have shown it.** Every column in
`coef_entropy_diag` is a bound whose probabilities are pooled over a whole plane's worth of a
subband. The pooling *is* the thing being priced, so "shipped vs bound" on a short block mixes two
different costs — the coder's cold start, which is fixable, and the difference between a per-block
adaptive model and a plane-wide oracle, which is not. A bound cannot separate them.

## What was measured instead

`src/encoder/abac_init_diag.rs`: a **simulation of the coder**, not a bound. It walks each
code-block exactly as `abac::encode_block` does, drives the **shipped** `Prob::update` — imported,
not reimplemented, so it cannot drift — and charges `−log2 p` per context-coded decision plus one
bit per bypassed decision. Only the initialisation changes between arms, so everything the
simulation does not model is present on both sides and cancels.

Four stills, q = 85/90/95/99, 4:4:4, tile 256, 5 levels, cb 64, `--abac` — the parameters of
`0024`, which is what makes this comparable to it.

**Canary.** The cold arm must land on what the bitstream really spent, and it does: 1.16% under,
which decomposes into 0.29% of per-block length fields and **41.4 bits per code-block** of coder
overhead. That per-block figure is checked against the real encoders in a unit test over 240
engine/geometry/spread/density combinations: the simulation is a strict lower bound on both
engines in all 240, and the worst overhead is **81.3 bits** — the range coder at 64×64. 41.4 sits
inside a band that was measured rather than chosen.

## The numbers, as a percentage of what abac's bitstream actually spent

**Candidate 1 — signalled initial probabilities, 18 bytes per (plane, subband), once per frame:**

| | q=85 | q=90 | q=95 | q=99 |
|---|---|---|---|---|
| bbb_1080p | −1.33% | −1.07% | −0.84% | −0.59% |
| blue_sky_1080p | −1.47% | −1.26% | −0.91% | −0.58% |
| kristensara_720p | −1.79% | −1.52% | −1.13% | −0.68% |
| touchdown_1080p | −1.34% | −1.15% | −0.82% | −0.58% |
| **mean** | **−1.48%** | **−1.25%** | **−0.93%** | **−0.61%** |

**Candidate 2 — stop cutting code-blocks on subband boundaries below `cb`:** −0.32% to −0.57%,
mean **−0.40%**. **Both together: no better than candidate 1 alone** (−1.10% against −1.11% at bbb
q=90, relative to the simulated band-aligned total). They are substitutes — both attack the same
cold start — not complements.

## The decision

**ENT-6 is closed. Neither candidate ships.** Its own criterion was **≥2% of total rate at q=90 on
all four stills**, and the best variant reaches **1.07–1.52%** — above the item's 1% close-floor
on 4 of 4 images and below its 2% ship bar on 4 of 4. Three things turn "in the ambiguous band"
into "no":

1. **The effect shrinks with quality, and GNC's home range is the top.** −1.48% at q=85 falls to
   −0.61% at q=99. `0024`'s bound ratio moves the *other* way (+23.1/25.9/28.9/34.2% at
   q=85/90/95/99), which is exactly the artefact: at higher q the blocks carry more symbols, so
   adaptation converges earlier relative to the total even as the pooled bound pulls further
   ahead. GNC is a contribution codec (GOALS §1) and the item is worth least where it matters most.
2. **Candidate 1 makes the entropy encode two-pass.** The table is that image's own per-band
   statistics, so the encoder must walk every coefficient once to gather them and again to code.
   That is a doubling of the entropy stage's encode work for 1.25%, on a coder whose encode time
   per frame `0017` has still never measured on an idle machine.
3. **It needs a bitstream change either way** — a frame-level table and a GP version for
   candidate 1, a changed partition for candidate 2 — and both shaders must load the table or the
   new geometry. CLAUDE.md's "would we ship this? A complex change for 0.3 dB is probably not
   worth the maintenance cost."

## What was not chosen, and what it would have cost

- **Landing candidate 2 anyway, on simplification grounds.** It is genuinely attractive as
  *cleanup*: −0.40% of rate, block count per 1080p 4:4:4 frame **3000 → 1920** (so 36% fewer
  length fields), and `code_blocks_banded` collapses into a plain `cb` grid. At tile 256 with cb
  64 the *only* block that changes is each tile's top-left one — level-1 bands are 128px and
  level-2 bands are exactly 64px, so both are cut identically either way. Against it: fewer blocks
  is **less parallelism**, and abac is one thread per code-block on both sides. 1920 threads still
  saturates 20 GPU cores, but the direction is wrong and **ENT-8 would change this calculus
  entirely** — if a block becomes 32 threads instead of 1, block size stops being the parallelism
  knob and candidate 2 becomes free of that objection. **Revisit candidate 2 if and only if ENT-8
  lands**, or if someone is already changing the partition for another reason.
- **A shipped, offline-trained table instead of a signalled one.** The 864 B is 0.05% of the
  frame, so signalling costs essentially nothing, and `0030` measured what a cross-trained table
  loses: 5.7 points of the model gap, concentrated in chroma, where leave-one-out context misses
  run to 2.44% against ≤0.31% on luma. Signalling is strictly better here and cheaper to reason
  about.
- **Per-tile signalling, which is what the item proposed.** 288 B per tile × 120 tiles is 34.5 kB,
  **1.9% of a 1080p frame at q=90**, against a 2% target. Measured net: **+0.4% to +1.2% — larger
  files.** The item's own design was a loss and nothing in it said so; the factor of 40 between
  per-tile and per-frame signalling is the single most important number in this record.
- **A faster adaptation rate for the first symbols of a block** (a two-speed `ADAPT_SHIFT`). Not
  measured. It is the one remaining variant that needs no header, no partition change and no
  second pass — only a different update rule — and it is therefore the cheapest thing left if
  anyone wants to reopen this. It cannot beat the 1.48% ceiling measured here, because that
  ceiling is what a *perfect* initialisation buys.

## Caveats a later reader needs

- **These are simulated coder bits, not encoded bytes.** The arm-to-arm difference is sound
  because the unmodelled per-block costs are identical on both sides, but no bitstream was
  produced and no file was written. A shipped implementation would land within the per-block flush
  band, not exactly on these figures.
- **The warm table is that image's own statistics, pooled per (plane, subband).** That is
  achievable by a two-pass encoder rather than an oracle — but it is the *ceiling* for this
  design: a finer table adapts better and costs more header, which is the trade the per-tile
  column prices and rejects.
- **Intra only, stills only.** Whether a warm start pays more on inter residuals is untested;
  `0025`'s figures are the inter baseline and this says nothing about them.
- **`0024` and its RESEARCH_LOG entry still say "worth about 4% of the file".** That sentence is
  corrected by this record rather than edited out of them, because the 4% is a correct reading of
  what `0024` measured — a bound ratio — and the error was in treating it as a collectable rate.
