# 0046 — `benchmark-sequence --throughput` skips the metrics the default must keep

**Date:** 2026-09-08
**Item:** BUG-32
**Status:** accepted

## Context

`benchmark-sequence` scored PSNR and SSIM on the CPU for every frame of *two* encodes (I+P and
all-I) and retained the decoded sequence in RAM. Measured on an RTX 4000 Ada, 8 frames at
q=90 ki=1: **2726 ms wall against 376 ms of encode — 86% not encode.** Cost per frame grew
superlinearly with length (341 ms at 8 frames, 6.9 s at 120) because the retained f32 RGB
sequence is about 3 GB per arm at 120 frames of 1080p.

`gpu_tier_bench.py --density` computed `frames / wall` from that command, so a concurrency
sweep measured how well N SSIM computations share the CPU. `--density-still` worked around
it by looping `benchmark` on one frame. The fix proper was left as this item.

## Decision

**Add `--throughput`. Leave the default alone.**

The default path still decodes, still prints PSNR/SSIM, still runs the all-I arm. That is the
honest command for a quality number. The timing path is opt-in and does three things:

1. Does not call `decode_sequence` (no retained decoded copy).
2. Does not run `psnr` / `ssim_approx`.
3. Does not run the all-I comparison encode.

It still prints per-frame bytes, bpp, frame type, and the encode-phase fps (BASELINE quantity
A). `--vmaf` conflicts: VMAF needs a decode.

`--density` now passes `--throughput`. `--density-still` stays; it has no sequence load.

## What was not chosen

- **Making the default skip metrics.** The command's name is `benchmark-sequence` and every
  existing quality table that came from it depends on the printed PSNR. Silent speed would
  have been a silent metric hole.
- **Gating the all-I arm on `--ab`.** That would change the default. `--ab` already means
  something else (temporal vs I+P). The second arm is skipped only with `--throughput`.
- **Deleting `--density` in favour of `--density-still`.** Still frames cannot exercise the
  inter path, which is the configuration MEAS-5 exists to run.

## Canary

`[bug32] throughput=1 metrics=0 i_only=0 decode_retained=0` on stderr. The default path must
not print it, and must still print `SSIM` and `All I-frames`.
