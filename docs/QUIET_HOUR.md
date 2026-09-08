# The quiet hour — a run list for the idle-machine queue

*Written 2026-09-08 at the project owner's request. Nothing in it is a new measurement; it is the
order to collect five already-parked ones in.*

Five backlog items are parked on one precondition, in these words or near them: *"needs an idle
GPU; eight sessions share one."* They are `ENT-10`, `MEAS-6`, `PERF-3`, `MEAS-5` and `BUG-38`.
None of them is blocked on an idea. **Together they are blocked on about an hour of a machine that
nobody else is using**, and one of them decides a default worth 23 points of the headline BD-rate.

This is the list to work through when that hour exists. It is ordered so that stopping early still
banks the most valuable result, and so the one step that conflicts with everybody runs last.

**Who runs it:** one session, holding a claim, in its own worktree — not the shared checkout.
Take each item with `scripts/claim take` as you reach it and **drop it when that run is done**. Do
not hold all five for the hour: a parked item is invisible to seven other sessions.

```bash
REPO=$(git rev-parse --show-toplevel)
LOG=$(mktemp -t quiet-hour)        # transcribe into RESEARCH_LOG.md at the end
echo "log: $LOG"
```

---

## 0. The gate — five checks, none of them skippable

The hour is worthless if the machine is not actually idle. On 2026-09-08 a 30-minute window was
lost to a subagent that ran x264 encodes, and it was caught only because the session responsible
volunteered the fact afterwards. Nothing in the harness detects it for you.

```bash
# 1. Nothing held but the parked rows. A `blocked-*` holder is fine; a session holder is not.
scripts/claim list

# 2. Every worktree clean. Not a formality — on 2026-09-08 three worktrees held uncommitted
#    files under claims marked OWNER UNIDENTIFIABLE.
git -C "$REPO" worktree list --porcelain | awk '/^worktree /{print $2}' \
  | while read -r w; do printf '%s: ' "$w"; git -C "$w" status --short --untracked-files=no | wc -l; done
# `--untracked-files=no` is load-bearing: without it this counts results directories like
# `meas10_out/` and blocks BUG-38 forever. That happened on 2026-09-08.

# 3. No stray load, CPU or GPU. COORDINATION rule 1 is about concurrent load full stop,
#    and the x264 window above was CPU-only.
pgrep -l 'cargo|rustc|x264|x265|aomenc|ffmpeg|vmaf|python' || echo "clear"

# 4. Record the machine. BUG-29 exists because nobody ever did this: every throughput
#    figure in this repository labelled "M1" is of unknown provenance.
cargo run --release -- gpu-info | tee -a "$LOG"

# 5. One binary for everything below, and record what it produces rather than what it is.
git rev-parse HEAD | tee -a "$LOG"
cargo run --release -- fingerprint | tee -a "$LOG"   # COORD-6 / docs/decisions/0075
```

**If check 3 is not clear, stop and find the process.** Do not average over load: a run taken
during a `cargo test` measured 20% slower than the same run after it (BASELINE, "How to read the
fps figures in this file").

---

## Run 1 — `ENT-10`, and it is first for a reason

It is the only item here that decides a **default**, and the default is worth
**+89.2% → +66.0%** BD-rate against H.264 at contribution quality — same pixels, PSNR-Y identical
to two decimals, only the bytes move (MEAS-10). Nothing has to be built to collect it. Both of
`docs/decisions/0017`'s two standing reasons are wall-clock numbers and both are missing.

```bash
scripts/claim take ENT-10 "the quiet hour: 0017's two open reasons are both wall-clock"

# 1a. abac GPU encode ms/frame — ENT-5 criterion 3. The instrument already exists.
cargo test --release --test abac_bench -- --ignored --nocapture --test-threads=1 2>&1 | tee -a "$LOG"

# 1b. The decode-debt re-take, abac against Rice. THREE runs per coder, minimum.
```

**Why three runs and not one.** The same abac decode has read **25.2 / 31.1 / 37.5 ms** across
three runs on this shared machine (ENT-8). That is a **1.48× spread against a 1.69× effect** — the
noise is nearly the size of the thing being measured, which is the entire reason this item is
parked rather than open. Report **n=3, median plus min and max**. A single number here is not a
measurement.

**Success criterion:** a re-taken decode ratio whose spread is small enough that 1.69× is
confirmed or refuted rather than restated. If the spread is still >5%, the machine was not idle —
go back to the gate.

**Then the decision, which is a decision record and not a commit.** Two things to argue it from
honestly:

- **Argue from intra, not from video.** `0045` measured abac against Rice on inter through the
  contribution range and the saving **decays monotonically with quality** — P-frame bytes go
  −20.6 / −12.2 / −11.9% at q=90 to **−14.2 / −4.3 / −3.7% at q=99**. GNC is a contribution codec
  and q=95–99 is its own range, so the −16.6% to −18.8% headline belongs to a range GOALS §1 says
  GNC is not primarily for.
- **The answer is probably not a global flip.** GNC already selects strategy per operating point
  in three places — MED replaces the wavelet at q=100, the entropy coder follows quality, the
  wavelet depth follows the tile size — and GOALS §1 blesses exactly this: *"Several internal
  strategies, selected by quality and bitrate, is a legitimate design."* Measure where the line
  falls and put the default there.

---

## Run 2 — `MEAS-6`'s coding half

The structural half is exact and stays: **0 frames** of reordering delay at the default
(`docs/decisions/0033`). What is owed is the ~80 ms coding half, which is a non-idle measurement
taken on a machine labelled M1 that is an M5 Pro.

```bash
scripts/claim take MEAS-6 "the quiet hour: the ~80ms coding half"
```

Use `benchmark-sequence` with **`--throughput`**. BUG-32 added that flag on 2026-09-08 precisely
because the default path spends **86% of its wall clock on CPU-side PSNR and SSIM** across two
encode arms, so any figure taken without it partly measures how fast SSIM runs.

**Report which quantity it is.** BASELINE distinguishes **A** (GPU encode phase), **B** (the
encoder loop's own figure) and **C** (end to end, wall clock, PNG input) — A is 2.4× C — and a
fourth thing has been called "encode fps" in that file as well. State the letter every time.

Glass-to-glass stays out of scope. Nobody has built the instrument and it needs capture hardware,
not a quiet hour.

---

## Run 3 — `PERF-3` item 8's A/B

**Why it earns a slot in a scarce hour:** items 9, 10 and 11 are each priced on bytes of bus
traffic saved, and item 8 showed that premise is untested. One A/B answers for all three — and may
delete them, which is worth as much as confirming them.

```bash
scripts/claim take PERF-3 "the quiet hour: item 8's A/B answers items 9-11"
```

`GNC_RICE_DISPATCH_REPEAT` isolates the slice; run it both ways on the one binary from the gate.

**State the question so that "no" is a result:** is the decode side bandwidth-bound at all? If it
is not, three backlog items are mispriced and should be closed rather than carried.

---

## Run 4 — `MEAS-5`'s `--density-still` half

Claim B — that a bigger GPU buys more GNC instances while it does not buy more NVENC blocks — needs
a discrete NVIDIA card with driver 610+ / nvenc API 13.1 and **stays blocked**. The still-frame
density re-take is the half an idle Mac unblocks.

```bash
scripts/claim take MEAS-5 "the quiet hour: the --density-still half only"
python scripts/gpu_tier_bench.py --density-still -i test_material/frames/bbb_1080p.png 2>&1 | tee -a "$LOG"
```

Keep `--iterations` large so the fixed ~0.7 s startup amortises, and **read the power column, not
utilisation**: `nvidia-smi` reported `utilization.gpu 100%` while the card drew 43–46 W of a 130 W
limit. Utilisation is an activity flag; power is the occupancy signal.

**Know the mode's limit and do not overstate the result.** `--density-still` is one still frame, so
it is **intra by construction** and cannot answer the shipped-configuration question. Do **not**
re-park MEAS-5 as answered — Claim B is the item, and this is one input to it.

---

## Run 5 — `BUG-38`, and only last

573 `cargo fmt` diffs across 61 files, 44 of which changed on `main` within 24 hours. `0066`
decided to keep the rule and land one atomic commit, and parked the item on exactly this
precondition: *"unpark when claim list shows nothing held and every worktree is clean."*

**Last, for two reasons.** It invalidates nothing measured above — formatting is behaviour-neutral
— and it conflicts with every session that restarts afterwards, which is one conflict per session
for zero behaviour change if it lands while sessions are live.

```bash
scripts/claim take BUG-38 "the quiet hour: the one precondition 0066 named"
cargo fmt
cargo test --release
cargo clippy --release --all-targets
cargo clippy --release --target wasm32-unknown-unknown --lib
```

All four clean before the commit, then one atomic commit per `0066`.

---

## What to log, per run

Into `RESEARCH_LOG.md`, and into `BASELINE.md` if a figure there moves:

| field | why it is required |
|---|---|
| the command, verbatim | four different quantities have been called "encode fps"; the command is what disambiguates |
| commit hash **and** `gnc fingerprint` | COORD-6 / `0075`: two numbers carrying the same fingerprint are comparable and two carrying different ones are not |
| the `gpu-info` device line | BUG-29: every "M1" in this repository is of unknown provenance |
| n, median, min, max | a single timing number on this machine is not a measurement — see Run 1 |
| which BASELINE quantity (A / B / C) | required by BASELINE's own fps section |
| whether the gate's check 3 was clear | so a later reader can trust the row or discard it |

## Stop rules

- **Two runs of the same thing differing by >5% means the machine is not idle.** Stop, find the
  process, restart the run. Do not average.
- **Do not start Run 5** until Runs 1–4 are logged and committed. It is the only step that touches
  files other sessions will conflict with.
- **Drop each claim as you finish it.** A held claim removes work from seven other sessions.
- If the hour runs out mid-list, the remaining items are still parked and still correct. **Runs 1
  and 2 are the ones worth protecting.**

## What this list deliberately does not contain

- **`ENT-8` step 2.** It is parked on the same precondition but it is ENT-5-scale work, not a
  measurement — it does not fit an hour and it is not a reason to spend one.
- **The laptop round.** `MEAS-5`'s remaining half, the first real DX12 measurement (unblocked by
  BUG-40 on 2026-09-08 and not re-run) and the NVENC column all need the Windows laptop and a
  driver update, not this machine. They live in **[GPU_TIER_TEST.md](GPU_TIER_TEST.md)**.
