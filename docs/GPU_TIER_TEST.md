# The GPU tier test — what it is, and what it can settle

*Written 2026-09-06 for a Windows 11 laptop with an Intel integrated GPU and an NVIDIA
RTX 2000 Ada. Nothing in it is specific to that machine.*

*Updated 2026-09-08, before a third round on that laptop. It has now been run twice —
2026-09-07 on an RTX 4000 Ada under Linux, 2026-09-08 on the laptop — and both rounds
changed what the next one should do. The changes are in "Where this stands" below; read
that first, because two of the commands this file used to give are now the wrong ones.*

Two numbers at the top of GOALS §4's target table were unmeasured when this was written, and
they are the two the entire positioning rests on: **concurrent streams per GPU** and **latency
per frame**. Everything else in this repository — the compression figures, the entropy coder
work, the lossless path — is secondary to them, because if GNC does not actually scale with
the card then the reason to prefer it over a fixed-function encoder mostly evaporates.

Latency has since been measured (MEAS-6: ~80 ms round trip, **0 frames** of reordering delay,
`docs/decisions/0033`). This harness measures the other one, and a cheaper thing that must be
true before it can mean anything.

## Where this stands

| question | state after two rounds |
|---|---|
| **CANARY-1** — does encode time move between GPUs at all? | **PASS, twice.** 34.5x against a CPU rasteriser (2026-09-07); **2.05x between two real GPUs** (2026-09-08). Settled; a re-run is now a regression check, not an experiment |
| **Test 3** — does GNC run on Windows? | **PARTIAL.** Vulkan works, and its output is **byte-identical across two different-vendor GPUs**. DX12 enumerates but does not compile the shader set. GL exposes no compute on this class of card |
| **MEAS-5** — how far does concurrency scale? | **Measured and not quotable.** The one clean point is N=2 at 2.01x; N=4 collapsed and N=8 exhausted RAM. Both bottlenecks are per-process startup and per-process memory, neither is GPU compute |
| **MEAS-5** — against fixed-function? | Intel QSV reached **4.54x at N=8** against GNC's 2.01x at N=2. **NVENC is still owed**, now for a driver reason |

**Three things changed underneath the previous round, and each one changes what this round
should run.**

**1. BUG-25 is fixed, and the previous round did not have the fix.** That round was built from
`f17bf1b`; the fix is `51a9ac6`, and `f17bf1b` **predates it** by 66 deleted lines of
`block_match_split.wgsl`. So every `0xC0000005` / `0xC0000409` P-frame crash in that round was
one invalid SPIR-V module, not a driver bug — GNC was shipping SPIR-V that naga 24 emitted
invalid, upstream `gfx-rs/wgpu#7048`. Check the build before believing anything about inter:

```bash
git merge-base --is-ancestor 51a9ac6 HEAD && echo "has the BUG-25 fix"
```

Two consequences. The P-frame arm, which the previous round could not run **at all**, is now
the first thing to try — and **it has not been re-run on this hardware.** It works on NVIDIA
RTX 4000 Ada and on lavapipe with byte-identical frame sizes; Intel Arc Pro and Windows NVIDIA
are an *expectation*, not a measurement, and confirming them is one `encode-sequence` and is
owed. See RESEARCH_LOG 2026-09-08 and `docs/decisions/0029`.

**2. `--density` measures SSIM, not the GPU (BUG-32).** `benchmark-sequence` spends **86% of
its wall clock** on CPU-side PSNR and SSIM, for two encode arms, and retains the whole decoded
sequence — so `frames / wall` swept concurrently measures how well N metric computations share
the CPU. Use **`--density-still`**, which sweeps `benchmark` (24% non-GPU overhead, and the
fixed part amortises across `--iterations`) and samples GPU power. **Know its limit:**
`--density-still` is one still frame, so it is **intra by construction** and cannot answer the
shipped-configuration question. Inter density needs BUG-32's proper fix — a flag on
`benchmark-sequence` that skips the metrics, the second arm and the sequence retention — and
until that exists, no clean inter density number is available on any hardware.

**3. `--all` could not run, and now can.** It passed one `-i` to every mode, and `--tier`
needs a still frame while `--density`/`--hwenc` need a Y4M clip. `-i` is now the still and
`--clip` is the sequence; a still handed to the clip modes is refused rather than silently
producing 0 completions, which printed as a driver session cap — the exact finding this harness
exists to detect. `--all` now runs `--tier`, `--density-still`, `--hwenc`; it no longer runs
the BUG-32 sweep.

## Run it

```bash
cargo build --release
git merge-base --is-ancestor 51a9ac6 HEAD && echo "has the BUG-25 fix"   # see above

python scripts/gpu_tier_bench.py --list           # what GPUs does this machine have?
python scripts/gpu_tier_bench.py --tier          -i test_material/frames/bbb_1080p.png
python scripts/gpu_tier_bench.py --density-still -i test_material/frames/bbb_1080p.png --adapter nvidia
python scripts/gpu_tier_bench.py --hwenc  --clip <clip>.y4m --encoder h264_nvenc
python scripts/gpu_tier_bench.py --all -i <frame>.png --clip <clip>.y4m
```

`--json report.json` keeps the raw numbers. `--density` still exists, prints its own BUG-32
warning, and is the only mode that reaches the inter path — read its rows as completion counts,
not as throughput.

The binary picks its GPU from the environment, so one build serves every device:

| variable | effect |
|---|---|
| `GNC_GPU_ADAPTER=<substring>` | first adapter whose name contains it, case-insensitive |
| `GNC_GPU_BACKEND=vulkan\|dx12\|metal\|gl` | restrict to one backend |
| `GNC_GPU_POWER=high\|low` | preference when no adapter name is given |
| `GNC_GPU_INFO=1` | print the adapter actually in use to stderr |

A `GNC_GPU_ADAPTER` that matches nothing is a hard error listing what is present.
Silently falling back to the default adapter would file a measurement of one GPU under
the name of another, which is precisely the failure the canary rule exists to catch.
Every run that sets one of these variables prints a `[gpu] …` line naming the device.

**Two things that cost time on the previous Windows round.** AppLocker on a managed machine can
block executables run from the winget per-user install location — Python and ffmpeg both had to
be copied to a policy-allowed path to run at all. And if `CARGO_TARGET_DIR` is set, the
harness's default `target/release/gnc.exe` is not where the binary lands: pass `--binary`
explicitly.

## Test 1 — CANARY-1: does encode time move between GPUs at all?

`--tier` runs the single-frame encode/decode loop on each adapter in turn and reports
the best of several processes.

**Why it comes first.** In 2011 BeHardware measured the shipping GPU H.264 encoders of
the day and found them performing *identically* on a 100 EUR card and a 330 EUR card —
because they were never compute-bound at all. The GPU was doing far less than the
marketing implied. A laptop with an integrated and a discrete GPU is the cheapest
possible version of that experiment, and until it passes, no throughput number from
this project means anything.

**Pass:** encode time drops substantially on the discrete GPU. **Fail:** the two come out
within ~15% of each other — the script says so in plain words when it happens — which would
mean the measurement is dominated by something other than GPU compute.

**Answered, on both arms.**

| round | slow arm | fast arm | encode spread |
|---|---|---|---|
| 2026-09-07, Linux | llvmpipe (CPU rasteriser) 480.74 ms | RTX 4000 Ada 13.95 ms | **34.46x** |
| 2026-09-08, Windows laptop | Intel Arc Pro 36.45 ms | RTX 2000 Ada 17.75 ms | **2.05x** (decode 2.35x) |

Settle ratios (the harness's median/best) were 1.00–1.09 throughout, so neither table is a
clock-ramp artefact.

**This file used to predict "roughly an order of magnitude" between the integrated and the
discrete GPU, and that was wrong about the size while being right about the mechanism.**
Between two *real* GPUs the spread is ~2x: the Arc Pro is a capable part and the RTX 2000 Ada
is a small mobile one. The 34.5x is the honest figure for GPU-against-CPU and says GNC is
compute-bound; **2x is the honest figure for real-GPU-to-real-GPU**, and it is the one to quote
for tier scaling. Both pass the same <1.15x failure test, comfortably.

Nothing here needs an NVENC comparison or quality matching. **What is left for this round is a
regression check**, and it is worth doing precisely because it is cheap: the shader set changed
under `51a9ac6`, and `--tier` on the same two adapters is the check that the tier ratio did not
move with it. CANARY-1's result belongs in the regression suite permanently for the same reason.

## Test 2 — MEAS-5: how far does concurrency scale, and against what?

`--density-still` launches N concurrent GNC processes and reports aggregate throughput plus
sampled GPU power; `--hwenc` does the same sweep through the machine's fixed-function encoder.

The structural argument is that a chip's hardware encoder blocks stay roughly constant however
large the GPU is, while shader throughput scales with the card — so a bigger GPU should buy
more GNC instances and does not buy more NVENC blocks. The published multi-tenancy literature
agrees that concurrency converts *idle* GPU into *useful* GPU rather than creating GPU.

### What the previous round measured, and why it is not an answer

RTX 2000 Ada, 60-frame 1080p 4:2:0 clip, q=90, ki=1 (all-intra, because P-frames crashed on that
pre-fix build), Rice:

| N | completed | wall s | aggregate fps | scaling vs N=1 |
|---|---|---|---|---|
| 1 | 1/1 | 16.6 | 3.61 | 1.00x |
| 2 | 2/2 | 16.5 | 7.27 | **2.01x** |
| 4 | 4/4 | 127.8 | 1.88 | 0.52x |
| 8 | 0/8 | ~205 (aborted) | — | RAM exhausted |

**Two implementation artefacts own that whole table, and both must be stated or the number
misleads.**

- **Per-process startup dominates.** The N=1 run is 16.6 s wall of which the GPU encode is
  **1.6 s**. The other ~15 s is Vulkan pipeline compilation plus clip load — a fixed cost paid
  once per instance. The aggregate-fps column is mostly measuring startup parallelism.
- **Per-process memory exhausts RAM.** Each process holds ~2 GB at 60 frames because it buffers
  the whole clip in several representations; a 120-frame 4:4:4 clip reached **4.3 GB** in one
  process. At N=8 that is ~16 GB, which is why the run was aborted. **Memory scales with frame
  count**, so the density sweep and the "use 120 frames or more" advice are in direct conflict
  on a 32 GB machine.

Short clips are startup-dominated, long clips are memory-dominated, and `--density-still` avoids
both by not being a clip at all — at the price of being intra-only.

### What to run this round

1. **`--density-still` on both adapters.** No CPU metrics, no clip buffering, and it samples
   power. Keep `--iterations` large (the fixed ~0.7 s startup amortises) and read the **power**
   column, not utilisation: `nvidia-smi` reported `utilization.gpu 100%` while the card drew
   43–46 W of a 130 W limit. Utilisation is an activity flag; power is the occupancy signal.
2. **The inter arm, for correctness before throughput.** `encode-sequence` with ki=9 on both
   Vulkan adapters is the measurement BUG-25's fix owes and nobody has taken on this hardware.
   A P-frame at a plausible bpp against the I-frame's is the canary that motion compensation
   actually ran.
3. **NVENC, if the driver allows it.** The previous round was stopped by ffmpeg 9.0.1 requiring
   **NVIDIA driver 610+ / nvenc API 13.1** against the machine's **13.0** — 0/N completed.
   Either update the driver first or the NVENC column is owed again. Intel QSV was substituted
   last time (`h264_qsv`, preset veryslow, global_quality 18) and reached 4.54x at N=8 with all
   8 completing and ~1 s startup.
4. **Do not re-run `--density`** for a throughput figure. If it is run at all, it is for the
   completion counts on the inter path.

### What this machine can and cannot settle

**It can** establish the scaling *slope* over three tiers — the dev Mac, Intel integrated and
the RTX 2000 Ada — which is more informative than any single point, and it tests whether GNC
keeps completing as N rises where a fixed-function encoder might not.

**It cannot** settle the density claim in its strong form. An RTX 2000 Ada is a small mobile
part; it is the least favourable version of GNC's own argument, which is about large GPUs.
Winning here is strong evidence. Losing here is not disproof, and the honest reading is the
slope, not the endpoint. **The previous round did lose here** — QSV out-scaled GNC — and the
gap was entirely GNC's per-process startup and memory rather than GPU compute, which makes it a
to-do list rather than a refutation.

**Three things to know before reading the fixed-function rows, or they will be misread:**

- **The session cap will not appear on this card.** The documented 12-concurrent-session limit
  is a GeForce driver restriction; professional RTX Ada parts do not carry it. On this laptop
  the comparison is pure throughput, and the absence of a cap is expected rather than a
  refutation.
- **The strongest fact in the positioning cannot be tested here at all.** A100, H100 and B200
  ship with *no* NVENC whatsoever — an idle AI fleet has zero encode capacity. That stays a
  sourced claim about hardware neither of us has.
- **The rows are not quality-matched.** The script says so in its own output. Bitrate and
  distortion parity is `scripts/meas1_vs_h264.py`'s job; read these tables as session-count and
  scaling only. A throughput comparison between two encoders sitting at different quality points
  is not a comparison. The GOP length is matched — `--keyframe-interval` goes through as
  ffmpeg's `-g`, since an all-intra GNC arm against a 250-frame-GOP NVENC arm was a comparison
  of GOP lengths wearing a throughput label.

**The dev-machine density table is a candidate for the same defect, not a baseline.** The
~2x-aggregate-at-N=8 figure POSITIONING quotes was taken 2026-09-05 by a method BACKLOG records
as unrecorded, on a machine this repository labelled M1 and which is an M5 Pro (BUG-29), and it
has exactly the shape CPU-bound work on N cores produces. It predates this harness, so BUG-32
cannot be pinned on it — and it needs re-taking with `--density-still` on an idle Mac before it
is quoted again.

## Test 3 — the free one: does GNC run on Windows at all?

GOALS rule 4 claims Metal, Vulkan, DX12 and WebGPU. **Partly answered, and the answer is
"Vulkan".** From the previous round, single intra frame, q=90 bbb:

| adapter | backend | result |
|---|---|---|
| Intel Arc Pro | Vulkan | works — 2091447 bytes, PSNR 50.06, 36.94 ms |
| NVIDIA RTX 2000 Ada | Vulkan | works — 2091447 bytes, PSNR 50.06, 20.11 ms |
| Intel Arc Pro | **DX12** | **crash (exit 101) at shader compile** |
| NVIDIA RTX 2000 Ada | **DX12** | **crash (exit 101) at shader compile** |
| Microsoft Basic Render Driver (WARP) | DX12 | panics even on a single intra frame |

- **Vulkan output is byte-identical across two different-vendor GPUs** (2091447 bytes on both).
  That is a stronger cross-backend result than the Metal-vs-Vulkan comparison, which differed by
  one byte — though at a different operating point (this is q=90, near-lossless; the 1-byte diff
  was q=75), so it is not proof the lossy path is bit-exact, only that it was here.
- **The DX12 backend does not run GNC at all.** Every DX12 encode dies at pipeline creation,
  before a pixel is read, with an FXC HLSL compile error: `X3695: race condition writing to
  shared` in **`block_match_bidir.wgsl`** (line 260). naga's generated HLSL trips FXC's
  groupshared race check; the same WGSL compiles fine under Vulkan/naga-SPIR-V. This is distinct
  from BUG-25 — different backend, different compiler, different shader.

**BUG-40 step 1 landed 2026-09-08:** `match_bidir_pipeline` (and the two bidir MC pipelines) are
lazy, same rule as `split_pipeline`. An intra DX12 encode should now fail on a shader it
actually uses, or complete. **That has not been re-run on Windows.** If this round predates
the merge, the DX12 rows are the old crash and nothing more. If it includes the fix, a still
encode on `GNC_GPU_BACKEND=dx12` is the first real DX12 measurement this project has. B-frame
dispatch on DX12 may still die on FXC `X3695` in `block_match_bidir.wgsl` — that is step 2,
and it is not this round's default path.

`GNC_GPU_BACKEND=vulkan` against `GNC_GPU_BACKEND=dx12` on the same card remains the comparison
worth having — two backends of the same shaders on identical hardware, where any divergence in
output is a portability bug worth more than the throughput numbers. It cannot be run until DX12
compiles.

**Still unexercised on Vulkan:** 4:2:2 and 10-bit. Both ship and both are only ever measured on
Metal.

## Measuring honestly on a laptop

COORDINATION.md's timing rules apply, and a laptop makes three of them sharper:

- **An idle machine is necessary and not sufficient.** A GPU ramps its clocks, and three
  consecutive processes on identical input have read 66.5, 45.3 and 34.9 ms on a genuinely
  idle dev machine — a 1.9x spread, monotonically decreasing. The script takes the **best** of
  several processes, because every error source only makes a reading slower, and prints
  `median / best` beside it: near 1.0 means quotable, well above means keep only the ratios.
- **Thermal throttling is the laptop-specific hazard**, and it pushes the other way from
  the clock ramp. Run on mains power, on a hard surface, with the power profile at maximum.
  Density is a question about *sustained* throughput, so if the sustained figure is well
  below the best-of-N figure, that gap is itself the result — report both.
- **Watch free RAM before raising N.** A density sweep that gets OOM-killed at N=8 costs the
  whole level and can take the machine with it. Check the per-process footprint at N=1 first and
  multiply; the previous round hit ~2 GB per process at 60 frames and 4.3 GB at 120.

Do not tune anything against numbers from this machine while other work is running on it.
Compression figures (bpp, PSNR, VMAF, dE00) are deterministic and safe at any load;
throughput figures are not.

## What to record

Whatever comes out, it belongs in RESEARCH_LOG.md **with the machine and the commit stated.**
The commit is not bookkeeping: the previous round's four "independent" driver crashes collapsed
into one cause the moment someone ran `git merge-base --is-ancestor` on the commit that entry
had recorded, and *"reproduced on an independent driver" is only independent if the builds are.*

CANARY-1's result belongs in the regression suite as a permanent check. The failure mode it
guards against is silent: a pipeline that is not running where we think it is looks exactly like
a pipeline that is, right up until someone compares two GPUs.
