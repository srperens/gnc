# 0035 — the fused histogram is dead work off the rANS path, and the fix is a second entry point

**Date:** 2026-09-08
**Item:** BUG-35 (partial — the default path; the rANS half stays open)
**Status:** accepted

## The problem

`quantize_histogram_fused.wgsl:main` declares **23800 B** of workgroup memory against a device
created with `max_compute_workgroup_storage_size: 16384` — **7416 B over**. 20480 B of that is one
array, `shared_hist: array<atomic<u32>, 5120>`.

Unlike BUG-31's abac shaders, this one is **not opt-in**. `EncoderPipeline::new` constructs
`FusedQuantizeHistogram` unconditionally, and `use_fused_qh` is
`config.use_fused_quantize_histogram && use_gpu_encode && !use_cfl` — true for Rice, which is the
default, whenever CfL is off. CfL is off above q=85, which is GNC's stated home range (GOALS §1).

## What the measurement showed, and it changed the fix

The histogram has exactly one consumer: the rANS batch encoder's
`encode_3planes_skip_histogram`. And the entropy branch checks Rice **first**, so the Rice path
never reaches it. On every path but rANS the shader was computing a 20 KB histogram in workgroup
atomics, writing it to device memory, and nobody ever read it.

Counted rather than argued — `GNC_PROFILE` now reports each dispatch's entry point:

| configuration | fused dispatches | of which with histogram |
|---|---|---|
| q=90, Rice, 4:4:4 (**the default**) | 3 | **0** |
| q=100, Rice | 3 | **0** |
| q=90, Rice, 4:2:0 | 3 | **0** |
| q=15, rANS, 4:4:4 | 3 | **3** |
| q=15, rANS, 4:2:0 | 3 | **0** |
| q=50 / q=70 (CfL on) | 0 | 0 |
| q=90, abac | 0 | 0 |

## What was chosen

A second entry point, `main_quantize_only`, which runs the same quantiser and never references
`shared_hist`: **3264 B**, measured. The quantise and histogram phases became two functions called
by two entry points, so nothing is duplicated (GOALS rule 9).

**A runtime flag would not have worked, and that is the whole reason for a second entry point.**
Declared workgroup memory is charged per entry point at pipeline creation. An
`if (params.write_histogram != 0u)` guard leaves the array referenced, so it is still allocated and
the pipeline is still refused. Only an entry point that cannot name the array gets under budget.

**The histogram pipeline is now created lazily, and that is what actually fixes the browser.**
It is *pipeline creation* a conformant WebGPU implementation refuses, not dispatch — so building
`main` eagerly in `FusedQuantizeHistogram::new` would fail every browser encode even when nothing
ever asks for a histogram. `main_quantize_only` is built eagerly; `main` is built on first use, in
a `OnceCell`. On the default path it is never built.

The `fused_qh_needs_hist` flag mirrors the *consumer's branch condition* rather than testing the
entropy coder, because branch order is what decides it: Rice is checked first and never reaches the
rANS arm, while Huffman without the 4:4:4 batch layout falls through to it and does consume the
tables. "coder == rANS" would have quietly stopped feeding that case.

## What was rejected

**Shrink `shared_hist` to fit.** To get the shader under budget the arena would have to drop from
5120 to ≤3266 entries. Rejected as the *first* move because it makes an existing hazard worse:
`total_hist_entries` is the sum of up to 12 per-group alphabets, each clamped at 4096, so it can
reach 49152 — and **nothing checks it against 5120.** `atomicStore`/`atomicAdd` past the end are
clamped by naga's bounds policy, so an overflow silently corrupts frequencies rather than failing.
A smaller arena overflows sooner. Any shrink needs the guard first, which is why it stays in
BUG-35 rather than being bundled here.

That the guard is missing here specifically is worth recording, because the neighbouring rANS
encode shader *does* have one, on the host, with a clear message: "tile 13 needs 6658 cumfreq
entries but the encode shader's workgroup table holds 4097". The fused histogram arena has no
equivalent.

**Gate `use_fused_qh` on the coder instead.** One line, and it stops the *dispatch* — but
`FusedQuantizeHistogram::new` would still create the over-budget pipeline, so a browser would still
refuse it. This is the same distinction BUG-31 turned on.

**Raise the limit request.** Ruled out in `0032` for reasons that apply unchanged.

## Evidence

- **10 of 10 encodes byte-identical** before and after, rebuilt from stashed sources for the
  "before" arm: Rice q=50/90/100 4:4:4, Rice q=90 4:2:0, q=15 (rANS default), `--rans` q=50 and
  q=70, `--abac` q=90, and sequences at ki=1 and ki=9 (which cover the I-frame and P-frame dispatch
  sites). Re-checked on the final build.
- **The Rice arm being identical is also the proof the histogram was dead.** The histogram phases
  only read the quantised buffer and write `hist_output`; if anything on that path had depended on
  them, the file would have moved.
- **A permanent canary, not just a print:**
  `fused_qh_does_not_build_the_histogram_pipeline_on_the_default_path` asserts zero histogram
  dispatches on a Rice encode *and* that the quantise path ran at all, so it cannot pass by
  asserting nothing. Byte-identity alone could not catch a flag stuck at `true`.
- 236 tests pass, 0 failures, re-run after rebase. `cargo clippy --release` and
  `--target wasm32-unknown-unknown --lib` both clean. `--tests` warnings unchanged from `main` at
  90 (BUG-20's pile did not grow).

## Not measured

**Throughput.** Removing 20 KB of workgroup atomics and a full re-read of the quantised buffer from
the default path should help, and it takes the shader from 23800 B to 3264 B, which is a large
occupancy change. No number: the machine was shared. Owed on an idle machine, and BUG-32 rules out
`benchmark-sequence`'s wall clock as the instrument.

**What the shader now costs on the rANS path is unchanged**, by construction — same entry point,
same bytes.
