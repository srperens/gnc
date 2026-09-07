# 0021 — Builds share one compilation cache, and cargo no longer assumes it owns the machine

**Date:** 2026-09-07
**Status:** Accepted. Machine-local configuration; no codec change, no bitstream change.

## The decision

Two settings in `~/.cargo/config.toml` — **not** in the repository's `.cargo/config.toml`:

```toml
[build]
rustc-wrapper = "sccache"
jobs = 6
```

## What was measured first

The machine at 20:14, with five sessions in their gate runs:

| | |
|---|---|
| load average (1/5/15 min) | **47.0 / 38.9 / 37.4** on 18 cores |
| processes | 9 `cargo`, 20 `rustc`, 16 `claude` |
| resident memory | 61 GB of 64, 1.9 GB unused, 731 MB swap in use |
| `target/` on disk | **~10.6 GB** across ten worktrees, 576 MB – 1.4 GB each |
| `Cargo.lock` | **265 packages**, 11 of them direct |
| `.cargo/config.toml` | **absent**, in the repo and in `$HOME` alike |

Those last two lines are the whole diagnosis, and they are two separate problems that had been
read as one:

1. **The work was redundant, not merely simultaneous.** Ten worktrees each compiled the same 264
   dependency crates — the wgpu/naga tree — into their own `target/`. Giving each worktree its own
   `target/` escaped the target lock, which was worth it, but it also multiplied the work by ten.
2. **No cargo on this machine knew it was sharing.** With no config anywhere, every invocation took
   the default `-j` = 18. Five concurrent gate runs asked for **90 parallel rustc jobs on 18
   cores**, which is the load average above.

## The risk that had to be cleared before enabling a cache

`src/` has **61 `include_str!` sites** pulling the WGSL files into the Rust crates, and CLAUDE.md
makes the shaders the single source for shader code. A cache whose hash did not cover included
files would serve a stale object after a shader edit: the session would measure a codec it had not
written, and the mismeasurement would be invisible. That is the failure class behind the retracted
results in [0012](0012-tile-geometry-scored-through-a-broken-coder.md) and
[0015](0015-the-entropy-coder-comparison-is-withdrawn.md), so it was tested rather than assumed.

Tested on a deliberately minimal two-file crate — one `.rs` with an `include_str!` of one
`.wgsl` — so the answer cost seconds instead of a 265-crate build:

| step | required | measured |
|---|---|---|
| build shader version A | miss | `misses 1` |
| `cargo clean`, shader unchanged | **hit** — proves the cache is consulted at all | `hits 1` |
| `cargo clean`, shader edited to version B | **miss** — proves the hash covers the file | `misses 2` |
| artefact contents | B present, A absent | `B: 2   A: 0` |

sccache 0.17.0 with rustc 1.97.1 discovers included files through `--emit=dep-info`. Safe here.

## What it bought

Priming from one worktree: **151 cacheable compilations**, `Average compiler 2.725 s`, so roughly
**411 s of rustc work** stored. A second, empty worktree at the same commit then built with:

| | |
|---|---|
| cache hit rate | **80.67 %** — 242 hits, 58 misses |
| cost of a hit | **0.077 s** against 2.725 s to compile — ~35x |
| non-cacheable | 118 calls, all `crate-type` — proc-macros and binaries, which still build per worktree |

**No wall-clock claim is made.** The second build took 231 s, but the priming build's time is
polluted by lock-blocking and by four other sessions, so there is no clean baseline to compare it
against. The hit rate is the load-independent figure and it is the one quoted. Timing this properly
needs the idle machine COORDINATION.md already demands, and it was not available.

## Validated against the gates, not just against "it compiles"

A compilation cache fails by handing back a stale object, and a build that exits 0 says nothing
about that. So the documented gate was run on the rebased tree with sccache in the loop, at
default test parallelism:

| gate | result |
|---|---|
| `cargo test --release` | **212 passed, 0 failed** across twelve test binaries |
| `cargo clippy --release` | **0** (excluding the `block v0.1.6` future-incompat notice from a dependency) |
| `cargo clippy --release --target wasm32-unknown-unknown --lib` | **0** |
| sccache during that run | 74.23 % hit rate, and `0` read errors, `0` write errors, `0` cache errors, `0` timeouts |

The first attempt at this gate went red on `abac_handles_subsampled_chroma`, which is exactly what
a stale cache entry would look like. It was not one: the failure is BUG-17, documented in
COORDINATION.md against the very commit this branch started from, root-caused to a process-global
`std::env::set_var` plus GPU device contention, and fixed on a later `main`. Three serial re-runs
of the same test on the old base went 7/7 green each time, and after rebasing onto the fix it
passes at default parallelism. **Worth stating plainly because the reasoning is reusable: when a
cache is newly enabled, every subsequent test failure will look like the cache. The way out is a
serial re-run and the commit history, not a guess.**

## The cache is worth more the busier the machine is

Unplanned, and the most useful number in the exercise. The same two figures, measured at two
different machine loads:

| | load ~36 | load ~70 |
|---|---|---|
| `Average compiler` | 2.725 s | **13.481 s** |
| `Average cache read hit` | 0.077 s | **0.086 s** |
| a hit is cheaper by | ~35x | **~157x** |

Compiling got 4.9x more expensive under contention. Reading from the cache did not measurably
change. **So the cache and the job cap are not substitutes for each other:** the cap lowers
contention, and the cache makes whatever contention remains far cheaper. This also means the
benefit is largest at exactly the moment five sessions hit their gates together, which is the case
the whole change exists for.

## What was not chosen

- **A shared `CARGO_TARGET_DIR`.** Removes the duplication too, and needs no new tool. But it
  reinstates the *target* lock that per-worktree `target/` directories were introduced to escape,
  so it converts duplicated work into serialised work. sccache shares artefacts without sharing a
  lock.
- **A build semaphore** — a `scripts/build` wrapper holding one of N slots, `flock` via `python3`
  since macOS has no `flock(1)`. It bounds bursts properly, where a static `jobs` cap only bounds
  them on average. It was not chosen *yet* because it works only if every session calls the
  wrapper, and that is a convention; [0016](0016-claiming-an-item-is-a-ref-cas-not-a-table-row.md)
  is this repository's record of what conventions are worth against eight concurrent sessions. Held
  in reserve if `jobs = 6` proves insufficient — the trigger is a load average above ~30 with the
  cap in place.
- **A shared GNU make jobserver.** Cargo is a jobserver client, so one token pool across all
  sessions would give a true global budget with no idle waste — strictly better than a static cap.
  It needs a long-lived process holding the fifo, and every build hangs if it dies. Not worth that
  failure mode before knowing whether the static cap suffices.
- **Either setting in the repository's `.cargo/config.toml`.** `jobs = 6` describes an 18-core
  machine and `rustc-wrapper` presumes sccache is installed; both would be wrong on any other
  machine and on CI. CLAUDE.md's rule against committing local infrastructure detail points the
  same way.

## Cost and limits

- The cache is capped at 25 GiB in `~/Library/Application Support/Mozilla.sccache/config`
  (602 GiB free on this disk). Verified picked up: `Max cache size 25 GiB`.
- The first build after this lands still misses everywhere; the cache pays from the second worktree
  on. Existing worktrees keep their warm `target/` and see nothing until a clean build or a rebase
  that invalidates deps.
- **A lone session now gets 6 of 18 cores** and is slower than before. Override with `-j18` or
  `CARGO_BUILD_JOBS=18` when genuinely alone — which, at eight instances, is rare.
- Proc-macros and binaries are never cached (`crate-type`), so a floor of per-worktree work remains.
- **The package-cache lock is untouched.** sccache does not go near it.

## The package-cache lock: `--offline` was tested and does not help

COORDINATION.md records builds blocking on `~/.cargo/.package-cache`. The obvious candidate was
`--offline`: `Cargo.lock` is committed and every crate is already in the registry, so resolution
should need nothing exclusive. **It made no difference.** Holding the lock from another process and
running a no-op build both ways:

| | lock held by another process | lock free |
|---|---|---|
| `cargo build --release` | blocked past 15 s, `Blocking waiting` printed | finished in 2 s |
| `cargo build --release --offline` | **blocked past 15 s, `Blocking waiting` printed** | finished in 1 s |

Cargo acquires the lock whether or not anything needs fetching, so the flag is not a way around it.
The advice in COORDINATION.md stands unchanged: a build queued behind other sessions is normal, not
a hang. Anyone attacking this next should not re-test `--offline`.
