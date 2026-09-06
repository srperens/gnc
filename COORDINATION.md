# Coordination between concurrent Claude sessions

**Up to five Claude sessions work on this repository at the same time.** Sharing one checkout
between them does not work: it has already cost a retracted BD-rate figure, a measurement premise
invalidated mid-experiment, and one session nearly deleting another's in-flight code. This file is
how we avoid that. **Read it before starting, update it when you claim or release something.**

## Rule 0 — start by moving into your own worktree (2026-09-06)

**Do this before reading anything else, before your first build, before your first measurement.**
The shared checkout `/Users/per/src/gnc` is for reading, for the coordination and log files, and
for merging. It is not where you work.

```bash
git worktree add <your-scratchpad>/wt <base-sha>     # pin the base explicitly
ln -sfn /Users/per/src/gnc/test_material/frames <your-scratchpad>/wt/test_material/frames
```

- **Pin the worktree to a commit**, not to `main`, so nobody else's merge moves the ground under a
  running sweep (rule 1 below is a consequence of this one).
- **Use a separate `target/`** — the worktree gets its own by default. It costs one full build
  (~2 min) and buys you a build that no other session invalidates mid-run.
- **Symlink the test material** rather than copying it; it is gitignored and large.
- Do codec work there. Commit there. Only touch the shared checkout to update
  `COORDINATION.md` / `RESEARCH_LOG.md` / `BACKLOG.md` / `BASELINE.md` and to merge your branch.
- When you are done, `git worktree remove` it. Three worktrees from 2026-03 sat abandoned for six
  months with uncommitted diffs in them.

Editing `src/` in the shared checkout while four other sessions do the same is how the failures in
the next section happened. The sharpest example, 17:30 on 2026-09-06: the abac session's 16-bit
entropy coder was reverted to HEAD's 32-bit version while its GPU decode shader — a direct port of
the 16-bit one — was left in place, along with a `mod.rs` declaration and a `gpu_util.rs` helper
that vanished from under the module needing them. Nothing warned anyone. The symptom was a
bit-exactness test failing at coefficient 8, which reads exactly like a shader bug and is not one.
**A half-reverted entropy coder is the worst case of this failure mode**, because an arithmetic
decoder that disagrees with its encoder does not error — it produces plausible garbage. A shared working tree makes every session's numbers mutually invalid —
not just risky, invalid, because no number can be attributed to a known state of the code.

## The three rules that have actually bitten us

**1. A number is only valid against a commit.** The working tree can change under you without any
signal. On 2026-09-06 a BD-rate figure moved from −3.7% to −2.3% purely because the coefficient
path changed between the sweep and the commit, and it had to be corrected in BACKLOG, RESEARCH_LOG
and a decision record after publication. Before quoting a benchmark number: either commit first
and measure the committed state, or measure in a worktree pinned to a hash
(`git worktree add <dir> <sha>`). If a number looks 0.1 dB off from one taken earlier in the same
session, suspect the tree before suspecting the metric.

**2. Measure the range the project cares about, not the range that is convenient.** GNC is a
*contribution* codec (GOALS §1). TUNE-5 was measured at q=15-50, shipped, and then found to cost
10.3 dB at q=99 — the operating point that actually matters. A sweep that stops at q=50 has not
tested this codec.

**3. Above about q=80, VMAF is saturated and PSNR must lead.** On old_town at q≥85 VMAF reads
identical for two settings whose worst-frame PSNR differs by 4.8 dB. CLAUDE.md's "VMAF primary"
rule is right in the lossy range and wrong near lossless. State which metric led and why.

## Claimed right now

Update this section when you start and when you finish. "Claimed" means *do not edit these files*
without saying so here first.

| files / area | session | state |
|---|---|---|
| `src/encoder/abac*.rs`, `src/shaders/abac_decode.wgsl`, `tests/abac_gpu.rs` | EBCOT/abac track | **active** — GPU decode shader landed and bit-exact; measuring the 16-bit interval's rate cost |
| `src/encoder/sequence.rs` P-frame quantiser block | EBCOT/abac track | **released** — TUNE-6 landed, see below |
| `src/encoder/intra.rs`, `src/shaders/intra_predict.wgsl` | lossless/intra track | **active** — fixing BUG-13 (encoder/decoder predictor mismatch) |
| `src/encoder/rice.rs`, `rice_gpu.rs`, `src/shaders/rice_*.wgsl` | tile-geometry track | **active** — BUG-11 (Rice stream mapping is tile-width-blind) + BUG-12 |
| repo hygiene + `CLAUDE.md` / `AGENTS.md` / `run-team.sh` / `.opencode/` | tile-geometry track | **active** — retiring the dead opencode agent team, de-duplicating the rule docs |
| everything else | unclaimed | — |

## Landed today, and what each one invalidates

- **BUG-13 filed** — `GNC_INTRA_PRED=1` (new knob) produces corrupt output at every quality: max
  error 197-255 from q=50 to q=100, and at q=100 it loses 62 dB against a bit-exact baseline.
  **The historical "-11.76 dB / +29%" measurement that set `intra_prediction: false` was measuring
  this bug**, not the idea. Error accumulates toward the bottom-right of every 32x32 block, which
  is an encoder/decoder predictor mismatch. Nothing else is invalidated -- the feature has always
  been off by default -- but the *conclusion* recorded against it is.
- **q=100 verified bit-exact lossless** on all three entropy coders. GOALS' "no true lossless with
  Rice" was stale and is corrected. GNC beats JPEG 2000 lossless by 10.8% and PNG by 7.8%; loses to
  FFV1 by 27% and x264 `-qp 0` by 43%, both of which predict against the neighbour.
- **Note on numbering:** BUG-11 was assigned twice on 2026-09-06 (Rice tile width, and intra
  prediction). The intra one has been renumbered **BUG-13**. Check this file before taking a number.

Newest first. If you have measurements taken before one of these, they are suspect.

- **Tile-size sweep (BUG-11 / ENT-1 / BUG-12)** — measurement only, no code change. Result: 256 px
  is a local optimum and both 128 and 512 are worse *through Rice*, but the sign reverses on the
  rANS path (512 gains 14–20%). Rice hardcodes tile width 256 in its `i % 256` stream mapping.
  **Invalidates the conclusion of every past tile-size experiment, #47 included** — they were all
  scored through the coder that penalises the larger-tile arm. Also: `--tile-size` never reached
  its own level ceiling (BUG-12), so those runs were capped at 5 levels too.
- **TUNE-6** — the P-frame quantiser scale now follows the quantiser step (1.25× at step ≥ 4.6,
  tapering to 1.0 at step ≤ 2.8). Fixes BUG-10. **Invalidates any inter measurement taken between
  TUNE-5 and now at q > 80.**
- **abac (EBCOT part 3)** — new standalone module plus a `GNC_ABAC_COMPARE=1` diagnostic. No
  bitstream change, nothing wired into the pipeline. Invalidates nothing.
- **BUG-7** — `GNC_DIAGNOSTICS=1` was inflating encoded files by 32% by clobbering the
  motion-compensation reference. **Invalidates every sequence measurement ever taken with that
  flag**, including MEAS-4's residual dumps. `tests/diagnostics_neutral.rs` now guards it.
- **MEAS-2** — AQ off below q=30 (its gradient was inverted), reference deblocking off by default
  (measured neutral-to-negative). **Invalidates AQ and deblock measurements from before today.**
- **FMT-2 / GP17** — Rice-coded stream-length tables. Changes all file sizes; invalidates absolute
  byte figures, not ratios measured within one build.
- **BUG-6** — 5 wavelet levels at q ≥ 25. Changes the LL subband AQ measures variance on, and the
  per-group table count rANS pays for. **Any AQ or entropy-coder tuning from before this is
  stale** — that is how the rANS/Rice crossover came to sit exactly at q=25.

## Conventions worth keeping

- **Every new code path needs a canary** — a logged count or value proving it ran on real data.
  `GNC_DIAGNOSTICS=1` prints `groups=N deep_skipped=M`, `rice_tiles=N/N`, and so on. A feature
  without one is not done (CLAUDE.md).
- **A result that beats its own theoretical ceiling is a bug.** That is how the abac comparison's
  bad baseline was caught (−35% against a ceiling of −13.7%).
- **A toggle measured on the wrong metric reads as dead weight.** CfL looks like a loss on VMAF
  and is a clear win on CIEDE2000, because VMAF is luma-only. Anything touching chroma needs
  `scripts/chroma_metric.py`.
- **Point measurements at fixed q cannot judge a rate/quality trade**, and they always flatter the
  option that spends more bits. Use BD-rate, or compare at matched rate. Three separate wrong
  conclusions today came from this one error.
