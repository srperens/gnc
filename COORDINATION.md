# Coordination between concurrent Claude sessions

Two Claude sessions work in this checkout at the same time. That has cost real work today —
a published BD-rate figure had to be retracted, a measurement premise was invalidated
mid-experiment, and one session nearly deleted the other's in-flight code. This file is how we
avoid that. **Read it before starting, update it when you claim or release something.**

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
| `src/encoder/abac.rs`, `src/encoder/abac_compare.rs` | EBCOT/abac track | **active** — CPU reference landed, GPU decode shader next |
| `src/encoder/sequence.rs` P-frame quantiser block | EBCOT/abac track | **released** — TUNE-6 landed, see below |
| everything else | unclaimed | — |

## Landed today, and what each one invalidates

Newest first. If you have measurements taken before one of these, they are suspect.

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
