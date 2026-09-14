# 0051 — abac's code-block default is 32, not 64

**Date:** 2026-09-14
**Item:** ENT-10 (step 1 — making abac affordable before deciding whether it is the default)
**Status:** accepted
**Machine:** Apple M1 Pro, 16 cores, 16 GB, Metal — `gnc gpu-info`. One of two Macs (COORDINATION).
**Binary:** `codec-fingerprint v1 6a9fa6bd` before, `fb2fe82a` after — this changes what the
encoder writes, and the fingerprint says so.

*(Number out of sequence on purpose: `scripts/claim dr` allocates the first free id and 0051–0053
were vacated by BUG-19's renumber. Nothing in the tree cites them.)*

## Context

ENT-10 asks whether abac should be the default entropy coder. It is worth **−14.1% to −17.5%** of
the bits, and the standing argument against it has always been one number: **cost.** `0017` shipped
it opt-in for that reason and left "the 1.69x decode debt on the performance list".

The owner, reading the re-measured cost, did not accept the number as a property of the codec:

> *"Vad jag funderar på om vi kan få abac inte så tung först? Vi vill ju helt klart ha
> förbättringen i lägre bitrate, men abac är för mycket latency nu.. något känns knas med den."*

That instinct is correct, and the diagnosis took one diagnostic line. **abac's GPU encode runs one
thread per code-block, and at cb=64 a 1080p plane has 1000 code-blocks** (`GNC_DIAGNOSTICS=1`:
`abac_blocks=1000`). One thousand threads, each serially coding 4 096 coefficients bit by bit, on
a GPU with 16 cores. Rice, for comparison, has 256 independent streams *per tile* — 10 240 per
plane. **abac was not expensive; it was empty.**

## What was measured

Three stills × q ∈ {90, 99} × {Rice, abac cb=64, abac cb=32} × {CountThenEmit, BoundedSlots},
`gnc benchmark -n 5`, idle machine. Bytes are identical between the two sizing modes in 12 of 12
cells, as their documentation promises, so rate is a function of `cb` alone.

| | rate vs Rice | encode vs Rice | frame decode vs Rice |
|---|---|---|---|
| **cb=64** (was the default) | −14.1% to −17.5% | **5.0x to 12.0x** | **2.6x to 4.8x** |
| **cb=32** (is now) | −11.8% to −14.9% | **2.2x to 5.1x** | **1.3x to 2.1x** |

**6 of 6 points agree in both directions.** Halving the code-block edge quadruples the code-block
count, and thread count *is* code-block count on both sides since ENT-5 put the decoder on the GPU
too. It costs ~2.5 to 3.5 points of rate and buys **~2.3x off encode and ~2x off decode**.

Two controls that matter:

- **The pixels do not move.** `gnc benchmark` reads 49.89 dB at both cb=64 and cb=32 on bbb q=90 —
  identical to two decimals in all three channels. Entropy coding is lossless over the same
  quantised coefficients; `cb` moves only the bytes.
- **32 is a knee, not a slide.** cb=16 buys ~8% more encode speed than cb=32 for **+17%** rate
  against cb=64, where cb=32 costs +3.8%. cb=8 is worse on both. Going further is not on offer.

**And abac's encode cost is nearly content-independent, which the ratio column hides.** At
cb=32/BoundedSlots, 1080p encode is 73.4 ms on bbb and 73.3 ms on stockholm — the same — while
Rice moves 39.5 → 15.7 ms with the content. So abac reads "11.96x" on stockholm and "1.86x" on
bbb *for the same absolute work*. **For a latency question the absolute number is the one that
means something**, and it is ~73–80 ms per 1080p frame at cb=32 against ~127–199 ms at cb=64.

## The measurement this replaces, and why it was not wrong when it was written

`DEFAULT_CB`'s own comment justified 64 like this: *"−13.8% rate at 33.0 ms of entropy decode,
where cb=32 is −10.9% at 31.4 ms — cb=64 dominates."*

**The rate half still reproduces** — a 2.9-point gap then, 2.5 to 3.5 points now. **The decode half
does not, because the decoder it was measured on no longer exists.** ENT-5 moved abac decode onto
the GPU, one thread per code-block, and that is exactly the axis `cb` controls. A 33.0 → 31.4 ms
reading is a 5% move; on today's path the whole frame decode moves **2x**. A five-percent
difference cannot be the shadow of a two-fold one, which is how you can tell the two numbers are
about different code.

**This is the `gnc fingerprint` failure mode in its natural habitat** (COORDINATION, "a number
carries its codec"): a correct measurement, recorded as a constant, outliving the thing it
measured — and sitting in a `const`'s doc comment where nothing re-runs it.

## Decision

**`DEFAULT_CB` = 32.** `GNC_ABAC_CB` still overrides it, and the encoder still refuses anything
above 64 or not a power of two.

**This is not a format change.** `cb` is written per tile, so a decoder reads whatever the stream
says and every file written at cb=64 keeps decoding. `codec-fingerprint` moves `6a9fa6bd` →
`fb2fe82a` because the *encoder's output* moves, which is the tool doing its job.

## What was rejected

**Making `BoundedSlots` the default sizing.** It is worth a further **1.1x to 1.6x** on encode at
**byte-identical output** (12 of 12 cells), which is not nothing — but next to cb's 2.3x it is the
small half, and it is the half that carries risk. `CountThenEmit` codes everything twice and is
exact by construction; `BoundedSlots` codes once against a ceiling computed in
`abac_encode.wgsl`, and if that ceiling is ever wrong the encoder **panics** (`check_flags`,
BUG-22's failure mode). Taking a 2.3x that cannot fail and declining a 1.2x that can is the right
order to do these in. The mode stays reachable as `GNC_ABAC_GPU_SIZING=slots`, it is now measured
rather than merely offered, and re-pricing it is ENT-10's business once the bound has a test that
tries to break it.

**Fixing the remaining cost by chasing occupancy further.** cb=16 says that lever is spent. What is
left is structural: two full coder passes, a `read_buffer_u32` between them, and a `poll(Wait)` per
pass. That is **ENT-8** (stripes within a code-block) and **PERF-5** (device-wide waits), and
neither is needed to make ENT-10's decision.

**Deciding ENT-10 itself here.** This record only moves the cost side. Whether abac becomes the
default still needs sequences as well as stills, inter as well as intra — `0045` records the
saving decaying on inter — and its own record.

## Consequences

- **ENT-10's trade is now −11.8% to −14.9% of the bits at ~2.3x encode and ~1.7x decode**, against
  −14.1% to −17.5% at 5.0–12.0x and 2.6–4.8x. That is a different decision, on the same evidence
  base, and it is the one the phase-1 rule (GOALS §1: a rate win is worth an encode cost) was
  written to resolve.
- `0017`'s "1.69x decode debt" figure is superseded twice over — by ENT-10's 2026-09-08 re-take
  (3.19x) and by this change (~1.7x at the new default). Any future quote needs a `cb` beside it.
- `all_zero_tile_is_cheap` asserted a flat 512-byte ceiling on a 256x256 tile and now scales with
  the code-block count. An empty block still costs a terminated interval and a length word, so the
  floor rises with the block count: 25 blocks → 100, 512 → 763 bytes. **A correct consequence of
  the change read as a regression**, which is what a flat bound on a derived quantity does.
- BASELINE's `--abac` rows are taken at cb=64 and are now at a non-default setting. Re-take is
  **MEAS-11**, already open.
