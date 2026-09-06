# CLAUDE.md — AI Instructions for GNC

All project goals, design rules, and priorities are in **[GOALS.md](GOALS.md)** — the single source of truth. This file contains only AI-specific instructions and technical reference.

**Up to five Claude sessions work on this repository concurrently. Start by moving into your own
git worktree — [COORDINATION.md](COORDINATION.md) rule 0 — then read the rest of that file and
claim your area in it.** It lists what is in flight, what each recent change invalidated, and the
measurement rules that have already cost retracted results. The shared checkout is for reading,
for the coordination and log files, and for merging; it is not where you work.

## Build & Run

```bash
cargo build --release
cargo run --release -- benchmark -i test_material/frames/bbb_1080p.png -q 75 --vmaf
cargo run --release -- rd-curve -i test_material/frames/bbb_1080p.png --compare-codecs --vmaf
cargo test --release
```

## Architecture

Modular pipeline with swappable stages:
1. Color space conversion (YCoCg-R, integer-exact lossless path available)
2. Transform (CDF 9/7 wavelet for lossy q=1-99, LeGall 5/3 for lossless q=100; 5 levels at q≥25, 4 below)
3. Quantization (adaptive with perceptual subband weights, CfL chroma prediction at q=50-85, fused quantize+histogram shader)
4. Entropy coding — three backends:
   - **Rice+ZRL** (fastest): 256 fully independent streams per tile, significance map + Golomb-Rice + zero-run-length
   - **rANS** (default): 32 interleaved streams per tile, per-subband frequency tables
   - **Bitplane**: block-based, fully parallel decode
5. Video: I/P/B frames, half-pel motion estimation, hierarchical block matching, CBR/VBR rate control
6. Container: GNV1 sequence format with frame index, keyframe seeking, error resilience (per-tile CRC-32)

Shader source is in `src/shaders/*.wgsl` (32 shaders). Rust host code is in `src/encoder/` and `src/decoder/`.

## Platform Notes

- Dev machine: Apple M1 — 8 GPU cores, ~2.6 TFLOPS FP32, no FP64, 32KB threadgroup memory, max 1024 threads/workgroup
- WASM target must work — avoid features not available in WebGPU (e.g. some storage texture formats, push constants)
- WGSL shaders are the single source — transpiled per backend by naga

## Code Style

- Rust, edition 2021. Keep shader code (WGSL) simple and readable — comment non-obvious GPU-specific tricks.
- **Zero clippy warnings** — `cargo clippy --release` and `cargo clippy --release --target wasm32-unknown-unknown` must both be clean. Fix warnings before committing. Prefer fixing the code over suppressing; `#[allow(clippy::…)]` is OK on individual items with justification but **blanket allows** (module-level `#![allow(…)]`, `dead_code` on entire impls, etc.) are **not acceptable**.
- **No `unsafe`** unless absolutely unavoidable. Prefer safe abstractions.
- Each pipeline stage is a separate module; new experiments go in `src/experiments/`.
- Don't commit test material to git (it's in `.gitignore`).

## Research Protocol — Skeptical Scientific Method

The team's core principle: **correctness over speed, measurement over assumption, skepticism over optimism.**

### Quality metrics
**VMAF is the primary quality metric.** PSNR is a secondary cross-check only.

**VMAF scores the luma plane only.** It cannot see chroma being degraded, so it cannot validate
any decision about a chroma parameter — chroma weighting, CfL range, chroma format trade-offs. A
2026-09-05 sweep of `chroma_weight` looked like a free 15% rate saving on VMAF and shrank to
+0.3 dB, direction-reversing, once measured with a metric that includes chroma. **Chroma
questions need a chroma-aware metric; VMAF answers luma questions.**
- VMAF catches perceptual regressions that PSNR misses (proven: TILE_ENERGY_ZERO_THRESH ghosting bug, chroma subsampling penalty)
- Always run `--vmaf` on `benchmark` and `rd-curve`. Success criteria must include VMAF.
- VMAF binary: `/opt/homebrew/Cellar/libvmaf/3.0.0/bin/vmaf` (also in PATH as `vmaf`)
- Tolerances: VMAF regression >0.5 points = BLOCK. PSNR regression >0.3 dB = flag but investigate.

### Before any experiment
1. **State the hypothesis clearly** — what do we expect to change and why?
2. **Question whether it's the right experiment** — does this address the actual bottleneck? Is there a simpler approach we're overlooking? Are we solving the right problem?
3. **Define success criteria with numbers** — "better" is not a criterion. "≥1.5 dB PSNR and ≥0.5 VMAF at same bpp" is.

### During implementation
4. **Verify the change is actually active** — add diagnostic output confirming the new code path runs. A feature that silently doesn't execute is worse than no feature.
5. **Test on ≥3 diverse sequences** — one sequence can mislead. Use high-motion (crowd_run), low-motion (rush_hour), and mixed (stockholm) at minimum.

### After measurement
6. **Challenge the numbers** — Do they make physical sense? A 0.01 dB improvement is noise. A 5 dB improvement on one sequence but 0 on others suggests a bug, not a breakthrough.
7. **Check for measurement artifacts** — Is the test actually exercising the new code? Are we comparing apples to apples (same q, same content, same frame count)?
8. **Reproduce before celebrating** — Run twice. If results vary by >0.1 dB or >5% bpp, investigate variance before claiming improvement.
9. **Ask: would we ship this?** — A complex change for 0.3 dB is probably not worth the maintenance cost. Simplicity has value.

### Logging
- Log ALL experiments in `RESEARCH_LOG.md` — including failures and abandoned approaches. Failed experiments are data.
- Always compare against baseline AND previous best.
- Include raw numbers, not just deltas.

## How to work

### Operating Philosophy

- **Correctness is non-negotiable.** A fast codec with subtle bugs is worthless. Verify every change end-to-end.
- **Measure everything, trust nothing.** Numbers that look too good probably are. Numbers that look unchanged might mean the code isn't running.
- **Challenge your own work.** After implementing something, actively try to prove it's wrong before calling it done.
- **Know when to stop.** If an approach yields <1% improvement after honest measurement, move on. Don't polish a dead end.
- **Iterate toward the best codec possible** using known techniques. Read literature, compare against state of the art, identify the biggest gaps, and close them systematically.

### The loop

One agent, in one context, carries an item from hypothesis to committed measurement. See
[LOOP.md](LOOP.md) for the step list. In short: pick from [BACKLOG.md](BACKLOG.md) by
value-to-effort, question whether it is still the right item, measure the *before*, change, measure
the *after* on ≥3 sequences and ≥2 quality points, run the three clean-build checks, write the
numbers into [RESEARCH_LOG.md](RESEARCH_LOG.md) including the failures, commit with the numbers in
the message.

### On subagents — the role-based team was tried and retired (2026-09-06)

This file used to prescribe an eleven-role pipeline — Visionary → Research Scientist → Researcher →
Team Lead → Builder → Critic → Tester → Validator → Documentation Agent, with a Regression Guard
and a Performance Profiler on the side. It was run. It consumed a large amount of context and did
not produce better work than a single agent following the rules below. It is gone, along with the
`opencode` scaffolding (`AGENTS.md`, `run-team.sh`, `.opencode/`) and the three abandoned worktrees
that were its only visible output. **Do not reinstate it.** Three reasons it did not fit this
project:

1. **The binding constraint here is measurement, and measurement is mechanical.** `benchmark
   --vmaf` is deterministic. A persona does not make it more truthful; running it on the right
   sequence, at the right operating point, against a pinned commit does. Every retracted result in
   this repo — the `chroma_weight` sign reversal, the 8-bit-truncated 10-bit run, the unnormalised
   DWT, the inflated `byte_size()`, the −3.7% BD-rate that was −2.3% — came from a harness error,
   not from a missing reviewer.
2. **Role boundaries force handoffs, and handoffs lose the context that catches those errors.**
   "Researcher does NOT write production code", "Profiler does NOT suggest fixes" means the person
   who understands the claim is never the one holding the code. None of the errors above would have
   been caught by a Critic reading a diff. They were caught by someone who knew the harness *and*
   the claim at the same time.
3. **The hardware is one M1 with 8 GPU cores.** Parallel agents contend for the single resource
   every measurement needs — the same fps run reads 20% slower with another session busy — and a
   shared working tree makes their numbers mutually invalid (COORDINATION.md rule 1).

**What subagents are still worth spawning:** read-only fan-out where the answer is a conclusion and
the cost is reading. Searching a 400 KB `RESEARCH_LOG.md` and a 90 KB `BACKLOG.md` for whether an
idea has already been measured and rejected. Literature search on a specific mechanism. Both return
a paragraph, not a diff, and neither needs the GPU. Spawn those freely; do the codec work yourself.

**What survives from the protocol is the rules, not the roles.** They are checklists, and they work
without an org chart:

### Hard Rules

- Temporal lifting operates on spatial wavelet subbands — never raw pixels
- Separate GPU buffer per plane (Y/Cb/Cr) — no aliased write_buffer calls
- Single command encoder per GOP for spatial wavelet dispatches — no inter-frame races
- All tests must pass after every change
- Zero clippy warnings after every change
- If the same bug resurfaces after two fix attempts — stop, diagnose root cause properly, do not loop
- **No silent features** — every new code path must have a way to verify it actually executes

### Quality Rules (added 2026-03-11 — lessons from sloppy execution)

- **Neutral bpp = SEND BACK, not CLOSE.** If a feature predicts ≥3% bpp gain and measures 0.0%, that is a bug, not a null result. Demand an explanation of yourself before accepting it. "It compiles" is not evidence it runs correctly.
- **Builder must show a canary metric.** Every implementation must include a logged count or value proving the new code path executes on real data (e.g., `skip_tiles=47`, `context_switches=1203`). A feature without a canary is not done.
- **Domain declaration is mandatory before implementation.** Researcher must explicitly state: "this change operates on [spatial residual | quantized coefficients | wavelet coefficients | bitstream | reference buffer]" and justify why that domain is correct. Researcher role includes challenging this — a skip implemented in spatial domain in a wavelet codec is wrong by construction.
- **Measurements must match baseline parameters exactly.** Same ki, same frame count, same sequence, same chroma format. Any deviation invalidates the comparison. State all parameters explicitly.
- **"Neutral" after claimed improvement requires investigation, not acceptance.** Ask: Did the code actually run? Was the threshold ever exceeded? Is there a log line proving the new path was taken? If none of these can be answered — the feature is broken, not neutral.

### Checkpoints

At every natural checkpoint (feature complete, priority item done):
1. Run `cargo test --release` — fix failures before proceeding
2. Log progress in `RESEARCH_LOG.md` with full numbers
3. Commit with descriptive message
4. Update `BACKLOG.md` status and `BASELINE.md` if improved
5. Continue automatically with next item

## Key Documents

- **[GOALS.md](GOALS.md)** — Rules, priorities, current state, non-goals
- **[docs/POSITIONING.md](docs/POSITIONING.md)** — What GNC is for, what the contribution market
  requires, and where GNC measurably stands against it. The reasoning behind GOALS §1
- **[BACKLOG.md](BACKLOG.md)** — Agent team backlog with status tracking
- **[BASELINE.md](BASELINE.md)** — Benchmark regression baseline
- **[docs/BITSTREAM_SPEC.md](docs/BITSTREAM_SPEC.md)** — Bitstream format specification
- **[RESEARCH_LOG.md](RESEARCH_LOG.md)** — Experiment log
- **[README.md](README.md)** — Public project description
- **[docs/archive/](docs/archive/)** — Historical documents (MILESTONES.md, INSTRUCTION.md, etc.)
