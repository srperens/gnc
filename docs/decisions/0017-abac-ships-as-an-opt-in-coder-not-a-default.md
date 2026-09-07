# 0017 — abac ships as an opt-in coder, and Rice stays the default

**Date:** 2026-09-07
**Status:** accepted
**Item:** ABAC-SHIP (BACKLOG, "EBCOT — evaluating in halves", Part 6)

## Context

abac — adaptive binary arithmetic coding over independent code-blocks — codes 16.6% to 18.8%
fewer bits than Rice at *identical* pixels, and 13.4% fewer at lossless. It is the largest
single-mechanism compression gain measured in this repository. It also costs about **1.69× frame
decode**, measured on an idle machine, because Rice decodes 256 branch-free streams per tile while
abac runs one serial adaptive coder per code-block.

GOALS §5 ("form first, then speed", set by the owner 2026-09-06) had already resolved the *keep or
drop* question in favour of keeping it: during the form phase, a change that closes a real
compression gap is worth taking even when it costs decode time, because a fast architecture that
is behind on compression is the harder problem to fix afterwards. What that decision did **not**
settle is whether abac should become the default coder now that it can write files.

## Decision

Ship abac as a **selectable** coder — `--abac`, `EntropyCoder::Abac`, bitstream entropy type 5 —
and leave Rice as the default at every quality preset.

## Why not make it the default

1. **The throughput debt is real and is not yet paid.** 1.69× frame decode is a measured number,
   not a fear, and GNC is positioned on concurrent streams per GPU and on latency
   (docs/POSITIONING.md), not on bitrate alone. Making abac the default would move the headline
   throughput figure by a factor the project has not yet decided to spend, and it would move it
   silently, in a commit whose subject is about compression.

2. **Encode is CPU-side and single-threaded.** 129 ms per 1080p frame against Rice's 23 ms. The
   coder is serial per symbol by construction; that is affordable for an opt-in mode and not for a
   default. It is also fixable — the work is parallel across ~3000 code-blocks — but it is not
   fixed today, and defaults should not carry work that has not been done.

   **Updated 2026-09-07 (ENT-5): the mechanism is gone, the number is not.** abac now encodes on
   the GPU, one thread per code-block, bit-exact against `abac.rs` — 98 of 98 whole-file
   comparisons byte-identical. So "encode is CPU-side and single-threaded" is no longer true.
   **What has not been measured is how long the GPU encode takes**: the machine had four sessions
   on it and COORDINATION forbids a wall-clock figure under load. Until that run happens this
   reason has lost its cause and kept its 129 ms, which is not the same as being discharged.
   `tests/abac_bench.rs::abac_encode_throughput_grid` is the instrument.
   See [0024](0024-the-gpu-abac-encoder-counts-before-it-writes.md).

3. **Inter frames are unmeasured.** This reason was withdrawn on 2026-09-07 when inter was
   measured at −14.4%, and then **reinstated the same day when that measurement was retracted**:
   abac's video path is the CPU-entropy P-frame path, and BUG-18 shows that path encodes every
   P-frame wrong — the first P after an I already diverges from the GPU path, and costs 2.1-2.8×
   the bytes. So the number was measured with abac on a defective arm.

   The reason therefore stands, and stands more strongly than when it was written: abac's video
   path is not merely unmeasured, it is currently **known-defective for reasons that have nothing
   to do with abac**. A default has to be right for video, and the video path has to be fixed
   before it can even be measured.

   **Update 2026-09-07, later the same day — this reason is now discharged.** ARCH-3 fixed the
   routing and BUG-18 with it, and inter is measured: **−12.0% to −22.9% against Rice at
   bit-identical pixels**, nine of nine points across bbb_extended / crowd_run / old_town_cross at
   q=50/75/90 (`docs/decisions/0025`, RESEARCH_LOG). abac pays on inter in the same band as intra.
   **The decision is unchanged** — reasons 1 and 4 are untouched, and reason 2 is the binding one.
   (Reason 2 was annotated the same day by ENT-5, which landed the GPU encoder: the 129 ms is no
   longer *CPU* encode time, it is simply the only encode time anyone has measured. Binding either
   way, and for the same reason — nobody has been able to time the new path on an idle machine.) Reason 3 should now be read as *answered yes*, not as
   *unmeasured*: it no longer argues against a default, it simply stops arguing for one.

4. **A default change is the expensive kind of change to reverse.** Every measurement in the repo
   that quotes a file size would need re-checking against a new default. Rice output is currently
   byte-identical before and after this change at q=25/50/75/90/100, which means this commit
   invalidates nothing. That property is worth more right now than a better default.

## What was rejected, and what it would have cost

**Making abac the default at q ≥ 85 only**, where the contribution operating point is and where the
gain is largest (−17.3% mean at q=90). Rejected: it would have made the decode-time cliff a
function of the quality preset, so a user moving q=84 → q=86 would see frame decode jump 1.69×
with no other change. The codebase already carries one lesson of this shape — TUNE-5 keyed a
quantiser scale to a preset index and had to be re-keyed to the physically relevant quantity
(TUNE-6). A throughput cliff hidden behind a quality knob is the same mistake.

**Shipping the Interval engine as well as Range, with Interval as default.** Rejected on
measurement: Interval costs 96.3 ms of entropy decode against Range's 33.0 ms — 2.9× — for 0.7
points of rate. Both engines remain reachable (`GNC_ABAC_CODER`) because the earlier −19% to −25%
figures were taken on Interval and reproducing them has to stay possible, but Range at cb=64
dominates every other cell measured and is what the encoder writes.

**Deriving the engine from the environment at decode time** instead of spending a byte per tile on
it. Rejected: the two engines share a binarisation but not a bitstream, and an adaptive arithmetic
decoder given the wrong one does not fail — it decodes every later symbol from a corrupted
interval *and* a corrupted context and reconstructs a plausible wrong image. One byte per tile
removes a failure mode that produces no error.

**Larger code-blocks.** cb=128 codes better still (−20.0% at q=55 against cb=64's −19.2%), and at
128 a subband is one block. Not taken: `abac_decode.wgsl` keeps two rows of neighbour magnitudes
per thread in workgroup memory and is sized for 64. The encoder refuses a larger `cb` at encode
time rather than writing a file the GPU cannot decode.

## Consequences

- The frame magic is now **GP18** for every frame this encoder writes, including Rice frames. GP18
  adds nothing but entropy type 5, so a GP18 Rice frame is byte-identical to the GP17 one apart
  from four bytes — but any pinned fixture or external decoder expecting GP17 needs updating.
- BASELINE.md does not move.
- The 1.69× decode debt stays on the performance list with a measured cost attached, rather than
  becoming a re-measurement project later.
