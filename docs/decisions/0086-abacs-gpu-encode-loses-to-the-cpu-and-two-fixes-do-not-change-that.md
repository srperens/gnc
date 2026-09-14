# 0086 — abac's GPU encode loses to the CPU, and the two obvious fixes do not change that

**Date:** 2026-09-14
**Item:** ENT-15
**Status:** accepted — two optimisations **rejected on measurement**, one architectural question **opened**
**Machine:** Apple M1 Pro, 16 GPU cores, 10 CPU cores, 16 GB, Metal — `gnc gpu-info`. One of two
Macs (COORDINATION).
**Binary:** `codec-fingerprint v1 027e58bc`, unchanged either side — **nothing here ships.**

## Context

`0053`'s sizing grid produced a figure nobody had gone looking for: on one 2.62 Mcoeff plane, the
**single-threaded CPU** coder runs at **27.47 ms** against the **whole GPU's 32.64 ms**. The owner's
reaction was the right one — *"det är inte klokt"* — because this is the textbook GPU workload:
2800 independent code-blocks, one thread each, no shared state between them.

Two structural explanations were available, both fixable, and both are now refuted or nearly so.

## Hypothesis 1 — SIMD divergence between blocks. **Refuted, and the fix is a regression.**

A Metal SIMD group runs at its slowest lane, so a group of 32 blocks costs the *maximum* of its
32. The dispatch is sorted by **area**; content makes two equal-area blocks differ by an order of
magnitude in coded bytes, and since ENT-14 an empty block returns immediately — so an empty block
sharing a group with a busy one saves bytes and, apparently, no time.

`slot_sizes` is an exact per-block cost that is already in hand (pass 1's byte count, or the
bound). Sorting by it instead:

| | lanes charged / lanes needed |
|---|---|
| by area (shipped) | 2.38x–2.94x **by bytes**, **1.00x by area** |
| by bytes (candidate) | 1.03x by bytes |

**The metric went 2.94x → 1.03x and the encoder got 3% to 8% slower**, on 6 of 6 points (three
stills × q ∈ {90,99}).

**The proxy was wrong.** A block's *time* is dominated by the per-coefficient traversal — its
**area** — not by the bytes it emits. Sorting by bytes mixes a 32×32 block with an 8×8 one whenever
they happen to code to similar lengths, and the group then costs the larger area. **The shipped
area sort was already optimal: 1.00x on the proxy that predicts time.** There is no block-level
packing left to win.

`simd_waste` now prints both columns and says which is which, because the byte column looks like a
3x opportunity and is not one.

## Hypothesis 2 — occupancy starved by workgroup memory. **Real, and worth 3%.**

The shader holds `probs` (1344 words) and `rows` (1024 words) = **9472 B per 32-thread workgroup**,
296 B per thread. On a 32 KB-per-core budget that is 3 resident workgroups, 96 threads per core.

`rows` is sized for `MAX_BLOCK_W = 64` while the shipped default is now **cb=32** (`0051`), so it
is over-provisioned by exactly 2x. Halving it: 9472 → **7424 B**, 3 workgroups → 4.

**Measured gain: 0.3% to 5.9%, mean ~3%** (same six points). A 33% occupancy increase bought 3%, so
occupancy is a real term and a small one.

**Not taken**, because the saving is 3% and the cost is `cb=64` support: `MAX_BLOCK_W` is a
compile-time constant sizing a workgroup array, and cb=64 is still reachable through
`GNC_ABAC_CB`. A WGSL `override` could size the array per pipeline and keep both; that is worth
doing *with* a change that makes it matter, not for 3% on its own.

## What the numbers actually say

| path | ms/plane | passes | lanes | Mcoeff/s | coeff/s **per lane** |
|---|---|---|---|---|---|
| GPU Range / CountThenEmit | 51.91 | 2 | 2800 | 100.9 | 36 051 |
| GPU Range / BoundedSlots | 32.64 | 1 | 2800 | 80.3 | 28 668 |
| **CPU Range, one thread** | **27.47** | 1 | 1 | **95.4** | **95 376 775** |

**One CPU lane does the work of about 3 300 GPU lanes on this algorithm.** That is the whole
finding. An adaptive binary arithmetic coder is a bit-serial state machine with a data-dependent
branch per symbol and a threadgroup read-modify-write per binary decision — the exact shape a
CPU's branch predictor and out-of-order engine exist for, and the exact shape a SIMD lane is worst
at. **More lanes do not fix it**, which is independently visible: cb=16 quadruples the block count
and buys ~8%.

Hypotheses 1 and 2 mattered because they would have meant the GPU implementation was leaving
something on the table. It is not. **It is running a workload the hardware is bad at, competently.**

## The question this opens, which is not this record's to answer

The coder is embarrassingly parallel over code-blocks on **either** device, and this machine has
ten CPU cores. Measured (`tests/abac_cpu_threads.rs`, same plane, 1-thread figure matching the
bench to 0.2%):

| CPU threads | ms/plane | Mcoeff/s | vs 1 thread |
|---|---|---|---|
| 1 | 27.52 | 95.2 | 1.00x |
| 2 | 14.10 | 185.9 | 1.95x |
| 4 | 8.00 | 327.5 | 3.44x |
| **8** | **4.37** | **599.3** | **6.29x** |
| 10 | 5.32 | 492.4 | 5.17x |

**Eight CPU threads are 7.5x faster than the entire GPU at this stage**, scaling near-linearly.

That is a real tension with GOALS' "everything runs as wgpu compute shaders", and it is a
**direction question for the owner, not an engineering one** — which is why nothing here changes.
What an honest version would have to price, and none of it is measured yet:

- **The readback.** Coefficients live on the GPU. A CPU entropy stage needs ~10.5 MB per plane
  back over the bus, ~31 MB per 4:4:4 frame, plus a sync point. On unified memory that may be
  nearly free and on a discrete card it is not — and **the discrete card is where the density
  claim lives** (MEAS-15). A measurement on one Mac cannot settle this.
- **What it costs the browser.** The WASM decoder is verified (GOALS §1) and a CPU encode path in
  a browser is a different animal from a native thread pool.
- **Whether it is the coder or the codec.** Rice's entropy stage is ~10 240 streams per plane
  against abac's 2800, and per-lane it is no faster. If the CPU wins this comfortably on abac it
  may win on Rice too, and nobody has measured that either.

Filed as **ENT-16**.

## Decision

**Nothing ships from ENT-15.** The two candidate optimisations are rejected with numbers, the
diagnostic that would have misled the next person now carries both columns, and the architectural
question is filed rather than answered.

**What ENT-13 should take from this**, since it is the item that inherits the cost problem:
`simd_waste` reads 1.00x by area, cb=16 buys 8%, and threadgroup memory buys 3%. **Every
lane-count lever on the GPU is spent.** What is left is doing less work *per coefficient* — in a
non-empty block most coefficients are still zero, and each one costs a coded "not significant" bit
plus a threadgroup read-modify-write. JPEG 2000 answers that with a run-length mode in the
significance pass, and it would be a rate lever *and* a time lever. That, and ENT-16, are the two
directions the evidence supports. Neither is "tune the dispatch".
