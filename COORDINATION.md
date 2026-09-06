# Coordination between concurrent Claude sessions

**Five Claude sessions work on this repository. Do not work in `/Users/per/src/gnc` directly.
Your first action in a session is to move into your own git worktree.**

Sharing one working directory does not work, and we have the scars: a published BD-rate figure had
to be retracted because the tree changed mid-experiment, a measurement premise was invalidated
under a running sweep, one session nearly deleted another's in-flight code, and on 2026-09-06 two
sessions' edits to `abac.rs` and `gpu_util.rs` overwrote each other and were lost outright.

## Rule 0 — start here, every session

```bash
# Pick a short name for the area you are working on, not for the session.
git -C /Users/per/src/gnc worktree add -b <area> /Users/per/src/gnc-<area> main
cd /Users/per/src/gnc-<area>
ln -s /Users/per/src/gnc/test_material test_material   # gitignored, ~GB, do not copy it
```

Then work only in that directory, and add your row to the table below. Each worktree has its own
`target/`, so builds no longer block on each other's cargo lock — that alone is worth the disk.

When your work is ready:

```bash
cargo test --release && cargo clippy --release          # the usual gates, in your worktree
git -C /Users/per/src/gnc-<area> push -u origin <area>  # or merge to main if you own it
```

To merge into main, rebase onto it first so the history stays linear and conflicts surface in your
worktree rather than in someone else's:

```bash
git fetch origin && git rebase origin/main
cargo test --release                                   # rebase can break things silently
```

Clean up when the area is done: `git -C /Users/per/src/gnc worktree remove /Users/per/src/gnc-<area>`.

**`/Users/per/src/gnc` itself stays on `main` and is for merging and reading, not for editing.**

## Worktrees currently out

Add your row when you create one, remove it when you are done.

| worktree | branch | area |
|---|---|---|
| `/Users/per/src/gnc-abac` | `abac` | adaptive binary code-block entropy coder (EBCOT follow-up), CPU + GPU |

## The four rules that have actually bitten us

**1. A number is only valid against a commit.** The tree can change under you with no signal. A
BD-rate figure moved from −3.7% to −2.3% purely because the coefficient path changed between the
sweep and the commit, and it had to be corrected in three documents after publication. Working in
your own worktree fixes most of this; still, measure a committed state, or a worktree pinned to a
hash (`git worktree add <dir> <sha>`).

**2. Measure the range the project cares about, not the range that is convenient.** GNC is a
*contribution* codec (GOALS §1). TUNE-5 was measured at q=15-50, shipped, and then found to cost
10.3 dB at q=99 — the operating point that actually matters. A sweep that stops at q=50 has not
tested this codec.

**3. Above about q=80, VMAF is saturated and PSNR must lead.** On old_town at q≥85 VMAF reads
identical for two settings whose worst-frame PSNR differs by 4.8 dB. CLAUDE.md's "VMAF primary"
rule is right in the lossy range and wrong near lossless. State which metric led, and why.

**4. A point measurement at fixed q cannot judge a rate/quality trade**, and it always flatters
the option that spends more bits. Use BD-rate, or compare at matched rate. At least four separate
wrong conclusions have come from this one error.

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
