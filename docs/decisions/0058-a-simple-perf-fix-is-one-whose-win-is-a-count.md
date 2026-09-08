# 0058 — A simple perf fix is one whose win is a count

**Date:** 2026-09-08
**Status:** accepted
**Items:** PERF-1 (P2)
**Supersedes nothing. Retires the 31.7 fps figure from every place that still asserted it.**

> **Renumbered 0058 from 0027 by BUG-19 on 2026-09-08.** `0027` was taken by INTRA-1's
> [cross-tile rate allocation is worth one point, not ten](0027-cross-tile-rate-allocation-is-worth-one-point-not-ten.md),
> which reserved the number through `scripts/claim` and was committed twenty-six minutes earlier,
> so two files carried `0027` on `main` for a day. Nothing in the record below changed. Commit
> messages, and any citation older than this date, call it `0027`.

## The decision

`docs/SIMPLE_PERF_FIXES.md` proposed twelve host-side changes and ranked them by how easy they
looked. PERF-1 landed five of them — items 1, 3, 5, 6 and 7 — and each one is recorded as a
**count**, not a time:

| what | before | after |
|---|---|---|
| `load_frame` calls per display index | 2.62 | 1.00 |
| `poll(Wait)` per I-frame (q=75, 4:4:4) | 3 | 1 |
| `MAP_READ` buffers per `encode()` with CfL off | 2 | 0 |
| `queue.submit` per `encode()` | 2 | 1 |
| decode pack allocation per frame | 1.22 MB, zero-filled | reused |

Eight sessions share one GPU, so a wall-clock delta on this machine is not evidence and the repo
has retracted results that rested on one — and since BUG-29 the machine label itself is not
trustworthy either, which is a second reason not to quote a time here. A count is reproducible under load, and each of these
counts is of *the thing being removed* — which also makes it the canary. Three of them print
under `GNC_PROFILE` permanently.

**What this rules out is quoting a time.** Nothing in PERF-1 claims an fps improvement, because
nothing in PERF-1 measured one. The items are justified by what they delete, and the bitstream is
proven unchanged: 24 artefacts hashed identical before and after every commit.

## Why item 4 was verified and not landed

Item 4 — "`create_buffer_init` + `create_bind_group` on every dispatch" — is the largest item in
the scan by site count and it was the most confident: "easy". All fourteen sites are real and at
the exact lines given. It is still not simple, and the reason is a rule this repository already
wrote down in `rice_gpu.rs`:

> On Metal/wgpu, `queue.write_buffer` is staged: only the last write before `queue.submit` takes
> effect.

The proposed fix is to replace a per-dispatch `create_buffer_init` with one cached uniform buffer
plus `write_buffer`, "the way Rice already does". Rice can do that because its parameters are
constant across the batch. `quantize.rs:224` is dispatched 6+ times per P-frame with *different*
parameters inside one submit; giving it one cached UBO would hand every one of those dispatches
the last write's parameters. Same for the per-plane sites in `transform.rs` and `motion.rs`.

The mechanism that makes it safe already exists and is not `write_buffer`: the wavelet uses
**dynamic offsets into a persistent buffer**, one slot per plane per direction. Porting the other
sites to that is a real change with a real design in it, not a mechanical substitution — so it is
PERF-2, filed with this reasoning, rather than a fifth commit here.

The bind-group half is separable and safe wherever the bound buffers are stable (crop, pack,
colour convert). It is worth microseconds against a 25 ms frame. It is in PERF-2 too, at the
bottom.

## What was rejected along the way

**Merging the I-frame submits without checking (item 6).** The scan calls this "easy, modest …
cheap, obviously correct". It is correct here, but only after checking that no `write_buffer`
between the two phases feeds a preprocess dispatch and that no readback sits between them. Had
either been true, the merge would have silently changed which parameters the pad/colour/
deinterleave passes read — a corruption with no error and no failing test, since both submits
already produce correct output in isolation. "Obviously correct" is the phrase that should have
triggered the check, not replaced it.

**Timing anything.** The tempting version of this item is to run `benchmark-sequence` before and
after and quote the fps. Four sessions were on the GPU throughout. The number would have been
real, reproducible only by accident, and worth nothing — the same trap as the 48% instrument
spread that killed the three abac decode shader opts.

## The figure this retires

The scan's step-1 claim that 31.7 fps "is not reproducible" and that "GOALS still cites it" was
true, and understated. `GOALS.md:103` and `BASELINE.md:103` retracted the figure; `GOALS.md:118`,
`GOALS.md:216` and `README.md:75` went on asserting it, the last of those as the public headline
for video-sequence throughput. A number can be retracted in one file and still be the first thing
a reader sees in another. **Every occurrence in the repository is now a retraction**, and each
replacement names which of BASELINE's three quantities it is quoting — A (GPU encode phase, 12.2
fps), B (encoder loop, 5.6) or C (end to end, 5.0).

## Cost of being wrong

Low and bounded. Every change here is bitstream-identical by construction and verified as such on
24 artefacts covering 4:4:4, 4:2:0, Rice and abac, stills and sequences, encode and decode. If one
of them is wrong it is wrong about *speed*, and the counts say plainly what was removed, so the
claim can be re-checked without re-running anything on a busy GPU.
