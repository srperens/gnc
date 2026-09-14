# 0085 — One device for N streams is measured, and rejected

**Date:** 2026-09-14
**Item:** PERF-4 (and a correction to MEAS-5's scaling column)
**Status:** accepted
**Machine:** Apple M1 Pro, 16 GB, Metal — `gnc gpu-info`. **One of two Macs**; the Platform Notes
describe the other (M5 Pro, 20 GPU cores, 64 GB). See "What this does not settle".
**Binary:** `codec-fingerprint v1 6a9fa6bd`

## Context

`scripts/gpu_tier_bench.py --density-still` answers "how many GNC streams does one GPU carry" by
launching N copies of the binary. PERF-4 was filed on an external reviewer's diagnosis of that
harness's own numbers: the sweep flattens at ~1.7-1.85x from N=4 while the GPU draws 2-49 W and
each process costs ~1.8 GB and ~15 s of pipeline compilation, so what flattens is plausibly
**process duplication**, not the GPU. The prescription was one device, one set of compiled
pipelines, N submissions in flight, inside a single process.

That prescription is now measured, because `gnc density` runs exactly it. The result does not
support it.

## What was measured

`gnc density` runs N encode streams as threads in one process and times **only** the steady-state
window, behind a barrier, after every stream has built its pipelines and encoded a warm-up frame.
Three arms, same binary, same frame, q=90, 16 iterations per stream:

| | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| **one shared device** | 21.90 fps | 28.39 | 28.67 | 28.39 |
| | 1.00x | 1.30x | **1.31x** | 1.30x |
| **one device per stream** | 21.69 fps | 27.13 | 34.16 | 33.49 |
| | 1.00x | 1.25x | **1.57x** | 1.54x |
| `--serial` (same total frames, one stream) | 21.74 fps | | | |

Three findings, in the order they matter.

**1. Removing the processes does not raise the ceiling.** The in-process arms flatten at
**1.3x-1.6x**, at or *below* what the process sweep reports (1.69x on a Mac, 1.85x on Windows).
Every per-process cost the diagnosis named — N devices, N pipeline builds, N copies of the frame,
N process images — is gone in the shared arm, and the ceiling did not move. **The reviewer's
diagnosis is not what caps this sweep.**

**2. Sharing the device is worse than not sharing it**, 1.31x against 1.57x. The mechanism is in
the tree and is not subtle: the readback path waits with `device.poll(Maintain::Wait)`, which is
**device-wide** — it returns when everything submitted on that device has completed, not when the
caller's own submission has. Put N streams on one device and each stream's readback waits for all
N. There are ~40 such call sites (`gpu_util::poll_wait` and its callers), so the fix is
`Maintain::WaitForSubmissionIndex` threaded from each `queue.submit`, and it is filed as PERF-5
rather than done here — because even a perfect fix only recovers the 1.57x that a device per
stream already gets for free.

**3. The ceiling is a pixel rate, and that identifies it.** A fixed per-frame or per-dispatch
overhead flattens in **frames** per second and reads the same at every resolution. A GPU
throughput limit flattens in **pixels** per second. Measured at three resolutions, one device per
stream, best row at each:

| resolution | N=1 | ceiling | Mpixel/s at the ceiling |
|---|---|---|---|
| 1280x720 | 46.00 fps | 76.47 fps (N=4) | **70.5** |
| 1920x1080 | 21.69 fps | 34.16 fps (N=4) | **70.8** |
| 3840x2160 | 7.58 fps | 10.72 fps (N=2) | **88.9** |

The fps ceiling moves by 7x across that range. The pixel rate does not — 70, 71, 89, rising
slightly with frame size as the dispatches get large enough to fill the machine. **GNC is
throughput-bound on this GPU at roughly 70-90 Mpixel/s**, and one 1080p stream already uses 45 of
it, i.e. ~64%. Two streams reach the ceiling. That is the honest density statement for this
machine, and it is a statement about the GPU, not about processes.

## Decision

**Do not build a shared-device, N-stream architecture.** It is more code, it couples the streams
through a device-wide wait, and the number it is supposed to buy is already available from N
independent devices — which is what N processes also get, minus the process image.

**Keep the process-per-stream deployment model.** It costs 230 MB RSS per process against ~80 MB
per additional in-process stream (164 MB for the first), which is a real difference and the only
argument the shared build had left. At a ceiling of two 1080p streams per GPU, ~150 MB of
duplication is not a constraint. Process isolation also keeps crash containment and tenancy
separation, which PERF-4 correctly flagged as possibly a *requirement* — it is now free rather
than a trade.

**Keep `gnc density` as the measurement.** `--density-still` is not wrong, but it answers a
different question and its scaling column is not a concurrency measurement: its N=1 row reads
**7.27 fps** against a true steady-state 23.4 fps on the same machine and frame, because its wall
clock contains device creation, pipeline build, PNG decode, the CPU quality metrics and process
teardown. Dividing by an N=1 row that is 3x too slow is what produced a scaling figure larger than
the one an idealised in-process build can reach. `--density-inproc` now runs both in-process arms
and prints a Mpixel/s column beside the fps.

## What was rejected, and what it would have cost

- **Submission-scoped waits, done now.** ~40 call sites across `sequence.rs`, `pipeline.rs`,
  `gpu_util.rs`, each needing the `SubmissionIndex` from its own submit. Rejected because its
  entire prize is the gap between 1.31x and 1.57x on an architecture this record is rejecting.
  Filed as PERF-5 because the *single-stream* version of the same question — 45 Mpixel/s against a
  70 Mpixel/s ceiling, i.e. a third of the GPU idle while one stream runs — is worth more than the
  density question and has the same root.
- **Sharing the compiled pipelines between streams.** The premise was 15 s of pipeline compilation
  per instance. On this machine, with a warm OS shader cache, `EncoderPipeline::new` takes
  **24.8 ms** and device creation **12.9 ms**; the first ever run after a build took 1 987 ms.
  So the 15 s is a cold-cache, first-run cost, not a per-instance one, and sharing pipelines saves
  25 ms per stream. It would also require splitting `EncoderPipeline` into immutable program state
  and mutable per-stream buffers, which is the `ARCH-4` refactor.
- **Quoting the 1.8 GB/process ceiling on this machine.** It does not reproduce: `gnc benchmark`
  peaks at **230 MB** RSS at 1080p. That figure is Windows/Vulkan and inter-frame; it should be
  re-taken rather than carried, and it is not what limits the Mac.

## What this does not settle

**The hardware. This is an integrated laptop GPU, and GNC's target is a server card.** Everything
above was measured on an Apple M1 Pro: 16 cores, no dedicated VRAM, LPDDR5 shared with 18 CPU
cores, a ~30 W part. A contribution encoder is deployed on an L40S, an A10G, an RTX 6000 Ada — a
300 W part with 300-900 GB/s of dedicated bandwidth and 5-10x the shader cores. **A throughput
ceiling measured on the first tells you almost nothing about the second**, and "70-90 Mpixel/s" is
a floor with a machine name attached, not a property of the codec.

**And the evidence already in this file points the other way.** BASELINE's CANARY-1 row has an
**RTX 4000 Ada** — itself only a 130 W workstation card, not a datacenter one — encoding 1080p at
**71.7 fps single-stream** against this machine's ~21.7. That is ~3x on one stream from a card two
tiers below the target, which is what one would expect if the ceiling is shader-bound and scales
with the GPU. It is *not* the same measurement (CANARY-1 times the encode loop; `gnc density`
times a steady-state window of full `encode()` calls) and it is cross-backend, so it is a hint and
not a result.

**Nobody has run `gnc density` on a discrete GPU.** That is now the measurement, it is one command,
and **MEAS-15** is where it lives — re-pointed by the owner the same day from "the other Mac" to
"a professional GPU in a server hall", because the two Macs are both integrated parts and the
claim is not about either of them.

**GOALS §1 now separates the two claims this record sits between.** *Reach* — GNC runs anywhere
there is a GPU, phone to Raspberry Pi to Chromebook — and *scale* — one card carries many streams
where fixed-function silicon caps out. **Read every number here as evidence for reach**, where it
is encouraging: a ~30 W laptop chip carries a 1080p stream with a third of the GPU still idle.
Read as evidence about a server card it is worth nothing, and it must not reach POSITIONING in
that role.


**Whether a bigger GPU buys proportionally more pixels per second.** That is MEAS-5 Claim B, it is
the thing GOALS calls the single most important measurement, and this record moves it rather than
answering it: the unit is now **Mpixel/s**, not instances, and the question is whether that number
tracks GPU size. It needs a second and third GPU, which this machine cannot provide.

**Which Mac the 2026-09-08 rows came from — answered 2026-09-14, and it hands MEAS-5 a cheap
experiment.** There are **two** Macs: the **M5 Pro / 20 GPU cores / 64 GB** of CLAUDE.md's
Platform Notes, and the **M1 Pro / 16 cores / 16 GB** every number above was taken on. Both notes
were correct and neither said which machine, which is the BUG-29 shape one level up.

What that is worth: **the second machine is a 25%-larger GPU of a later generation, already in the
owner's hands.** Claim B restated in this record's unit — *does a bigger GPU carry proportionally
more Mpixel/s* — is therefore measurable today with `gnc density`, on one input, with no NVIDIA
hardware. It is a weaker test than a discrete card (two points, same vendor, confounded generation
and core count) and it is a far cheaper one. Filed as **MEAS-15**.

**The 8K arm.** `bbb_8k.png` cannot be encoded at all: `enc_raw_input` wants 398 MB against a
256 MiB `max_buffer_size` (the wgpu default GNC deliberately requests, GOALS rule 4). Filed as
BUG-58. A fourth resolution point would have strengthened finding 3; three is what there is.
