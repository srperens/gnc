# Coordination between concurrent Claude sessions

**Five Claude sessions work on this repository. Do not work in the shared checkout directly.
Your first action in a session is to move into your own git worktree.**

Sharing one working directory does not work, and we have the scars: a published BD-rate figure had
to be retracted because the tree changed mid-experiment, a measurement premise was invalidated
under a running sweep, one session nearly deleted another's in-flight code, and on 2026-09-06 two
sessions' edits to `abac.rs` and `gpu_util.rs` overwrote each other and were lost outright.

## Rule 0 — start here, every session

```bash
# Derive the paths, never type them — see CLAUDE.md, "Never commit secrets or local
# infrastructure detail". Pick a short name for the *area*, not for the session.
REPO=$(git rev-parse --show-toplevel)
AREA=<area>
git -C "$REPO" worktree add -b "$AREA" "$REPO-$AREA" main
cd "$REPO-$AREA"

# Link the test material in. NEVER run this in the shared checkout, and never with -f:
# `frames` there is itself a symlink to the real ~31 GB directory, and `ln -sfn` pointed it
# at itself on 2026-09-06, breaking every session's measurements at once.
[ "$PWD" != "$REPO" ] || { echo "refusing: you are in the shared checkout"; return 1 2>/dev/null || exit 1; }
ln -s "$(readlink "$REPO/test_material/frames" || echo "$REPO/test_material/frames")" \
      test_material/frames
```

`test_material/` itself is tracked (it holds the fetch script), so symlink `frames` inside it
rather than replacing the directory. Then work only in that directory, and add your row to the
table below. Each worktree has its own
`target/`, so builds no longer block on each other's cargo lock — that alone is worth the disk.

When your work is ready:

```bash
cargo test --release && cargo clippy --release   # the usual gates, in your worktree
git push -u origin "$AREA"                       # or merge to main if you own it
```

To merge into main, rebase onto it first so the history stays linear and conflicts surface in your
worktree rather than in someone else's:

```bash
git fetch origin && git rebase origin/main
cargo test --release                                   # rebase can break things silently
```

Clean up when the area is done: `git -C "$REPO" worktree remove "$REPO-$AREA"`.

**The shared checkout stays on `main` and is for merging and reading, not for editing.**

## Worktrees currently out

Add your row when you create one, remove it when you are done. Write the worktree **relative to
the shared checkout** — no absolute paths, no session ids.

| worktree | branch | area |
|---|---|---|
| `../gnc-abac`, `.claude/worktrees/abac` (`abac-gate`) | `abac` | **released — question answered, see BACKLOG Part 6.** The idle-machine bench is run. Range at cb=64 costs **1.69× frame decode for −16.7% rate** at q=90; Interval costs 3.99×. Rice's own entropy stage is 47% of frame decode, which caps any entropy work at 1.9×. What remains is a positioning call, not an engineering one. |
| `../gnc-abac` | `abac` | same worktree, now on **BUG-8** — the encoder's local decode diverges from the real decoder down a GOP. |

## Timing: an idle machine is necessary and NOT sufficient (added 2026-09-06, after an idle run still lied)

**The GPU ramps its clocks, and a repeat count chosen for a CPU benchmark will not outlast the
ramp.** On a genuinely idle machine (load 3.8, GPU free), three consecutive *processes* on
identical input read **66.5, 45.3 and 34.9 ms** — a 1.9× spread, monotonically decreasing. A
freshly-idle M1 starts in a low power state and needs on the order of a second of sustained work
to boost. `abac_bench` had a `spread` column that blamed a busy machine for exactly this, so the
instrument was reporting the ramp *and* misattributing it.

Two rules that follow, and they cost nothing:

- **Take the best of ~24 repeats, not the median of ~7.** For a deterministic kernel on fixed
  input every error source — clock ramp, another session, scheduler noise — only makes a reading
  *slower*. The minimum is therefore the least-contaminated estimate, and `median / best` is a
  free settled-or-not diagnostic: near 1.0 means quotable, well above means keep only the ratios.
- **Run benches with `--test-threads=1`.** Cargo runs `#[test]` functions concurrently by default,
  so two GPU benches in one file contend for the device and interleave their output. This was
  happening and nothing warned about it.

**Isolating one GPU stage without timestamp queries:** dispatch it k times behind an env var and
difference. Rice's entropy decode is idempotent, so `GNC_RICE_DISPATCH_REPEAT=k` gives frame times
of 29.50 / 85.16 / 141.55 ms at k = 1 / 5 / 9 — slope 14.0 ms, two independent estimates agreeing
to 0.6%. That is how the abac question stopped being a bracket and became a number, and the same
trick works for any idempotent stage.

## Timing: the machine is shared, so throughput numbers are not measurable during a session

Up to five sessions compile and run GPU work on this one M1 at the same time. That makes every
wall-clock figure unreliable while anyone else is working — on 2026-09-06 the same abac decode
input timed **25.2, 31.1 and 37.5 ms across three runs**, a 48% spread on identical work, and
three targeted shader optimisations against three different suspected bottlenecks all returned
exactly nothing. Three plausible fixes measuring null is far likelier to be a broken instrument
than three wrong hypotheses.

So:

- **Compression figures (bpp, PSNR, VMAF, dE00) are safe.** They are deterministic and unaffected
  by load. Measure and quote them freely.
- **Throughput figures are not.** Do not tune against them, and do not record one without saying
  the machine was loaded. An optimisation validated against noise looks justified and is not.
- **Build the alternatives behind switches and measure them later, together.** When something has
  several plausible implementations, implement them all, make them selectable, and add a bench
  that runs the whole set in one command. Then one idle-machine run settles it. `abac` does this:
  `GNC_ABAC_CODER` selects the entropy coder variant and
  `cargo test --release --test abac_bench -- --ignored --nocapture` times every combination.

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
identical for two settings whose worst-frame PSNR differs by 4.8 dB. State which metric led, and
why. CLAUDE.md now carries this as a table rather than a blanket "VMAF primary" rule, and QUAL-1
put a magnitude on it: widening a BD-rate ladder moved the VMAF figure by **47.5 points on
average, 110 at worst**, while the PSNR figure moved **1.0**.

**4. A point measurement at fixed q cannot judge a rate/quality trade**, and it always flatters
the option that spends more bits. Use BD-rate, or compare at matched rate. At least four separate
wrong conclusions have come from this one error.

## Landed today, and what each one invalidates

- **CHROMA-1 — `chroma_weight` now stays at 1.2 above q=85 instead of dropping to 1.0.**
  Changes default output at **q ≥ 85 only**: BASELINE's q=90 row moves 50.41 → 50.06 dB and
  8.58 → 8.07 bpp, which is −5.2% luma BD-rate for +1.2% on colour. q=25/50/75 are byte-identical
  and q=100 stays bit-exact (the quantiser is bypassed there). **Any q ≥ 85 file size measured
  before this is stale**; ratios within one build are fine.
  Two things worth carrying: **VMAF read 97.08 before and after, on 6% fewer bits** — luma-only and
  saturated, so a VMAF-only verdict here is worthless, which is the same illusion that made the
  2026-09-05 sweep look like a free 15%. And **luma measured as BT.709 Y from decoded RGB is
  contaminated by chroma error** — it overstated the loss 3.7x. Use YCoCg-R Y (`ypsnr_de00.py`).
  Also settled by elimination: the +90.5% video gap is **not** an allocation artefact. The knob is
  intra-only (−20.8% all-intra against −2.9% on a ki=9 P-chain), so intra really is the only route.

- **QUAL-1 — the headline gap figure is corrected.** At the contribution operating point GNC needs
  **+90.5% BD-rate on PSNR** (about 1.9x), not the **+456.7% to +672.1%** recorded from MEAS-1.
  Nothing in the coder changed; the old figure was measured at distribution bitrates.
  **Invalidates every use of the 5-7x figure**, including in POSITIONING and any argument that
  single-digit improvements are pointless — against 1.9x they are not.
  Two further things to carry: **never quote a VMAF BD-rate above about q=85** (widening the ladder
  moved it 47.5 points on average, 110 on old_town, while PSNR moved 1.0); and **at matched rate
  GNC beats x264 on dE00** while losing 7.4-8.8 dB of luma, so quote luma and colour together or
  the number misleads in whichever direction suits.
  Also: **MEAS-1's stated sources are not reproducible** — `bbb.y4m` has 8 frames, not 17, and no
  `touchdown` sequence exists in the tree.

- **The test material was unreachable for about ten minutes (18:04, restored).** `ln -sfn` run
  inside the shared checkout pointed `test_material/frames` at itself, so every worktree's link
  resolved to a loop and no session could read a test image. Restored to the real directory and
  rule 0's command is now guarded against being run in the shared checkout. **Lost in the process:
  the `frames/hdr/` 10-bit material**, which was generated rather than fetched — regenerate it with
  `scripts/png16.py` if a 10-bit measurement is needed. Nothing else was lost, and no committed
  result depended on it.

- **BUG-11 + BUG-12** — Rice's stream mapping is now tile-width-aware (column-major, cut into 256
  contiguous segments) and the wavelet-level ceiling now follows the tile size actually in use.
  **Tile 256 output is byte-identical at all 12 measured points**, so nothing in BASELINE.md moves
  and no measurement taken at the default tile size is affected. What *is* invalidated: **every
  tile-size result in this repo, #47 and this morning's sweep included** — all of them scored the
  larger-tile arm through a coder that penalised it by 13–19%. Corrected figure: 512 over 256 is
  **−0.91% BD-rate**, so the geometry is worth ~1% and the default stays at 256. Also closed:
  6 wavelet levels at tile 512 is worth −0.1%, so deeper decomposition is not a lever either.
  **Use `CodecConfig::set_tile_size()`, never assign `tile_size` directly** — the level ceiling
  depends on it.

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

- **For the abac track, a result you did not ask for.** The `GNC_ABAC_COMPARE=1` harness was gated
  on `TransformType::Wavelet`, so it could not see the new lossless path. Gate widened (one
  `matches!`); `subbands()` at `num_levels = 0` already yields the right single region. **abac on
  MED residuals measures −14.3% mean on real coefficients** (bbb −14.9, blue_sky −15.6,
  kristensara −15.5, touchdown −11.2). With LOSSLESS-1 that is −27% against this morning's
  lossless and **+7.7% behind FFV1, from +48.4%**. Rate only. Your GPU decode throughput gate is
  now the only thing between that and a shipped result. Not implemented here — it is your code.

- **LOSSLESS-1** — q=100 now codes MED prediction residuals instead of wavelet coefficients
  (`TransformType::MedPredict`, bitstream `transform_type = 2`, `GNC_MED=0` to revert). Files are
  **14.9% smaller, still bit-exact**. **Invalidates every lossless figure in this repo**, including
  the "beats JPEG 2000 by 10.8%, loses to FFV1 by 27%" line — the FFV1 gap is now +25.8% measured
  as +48.4% before the change, so that comparison was against a different baseline than it reads.
  Also note `is_lossless()` now decides per transform type; anything keying on it should be
  re-read.

- **ENT-1** — subband-rANS frequency tables are now Exp-Golomb packed (bit 31 of the tile's
  `num_groups` word). Files shrink 11.7–26.6% on that coder at bit-identical quality; the preset
  path (q ≤ 20) shrinks 4.5%. **Invalidates every absolute byte or bpp figure for the rANS coder**,
  and it invalidates the *reason* for the Rice/rANS crossover at q=20 — re-sweep before quoting it.
  Landed inside commit `0e8987c`, which swept up another session's uncommitted work; the change was
  verified and measured only afterwards.

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
