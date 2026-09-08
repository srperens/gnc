# 0062 — Test code is code, so the native clippy gate is `--all-targets`

**Date:** 2026-09-08
**Item:** BUG-20
**Status:** accepted — gate widened, all 91 warnings fixed, none suppressed

## Context

CLAUDE.md's Code Style rule says **zero clippy warnings** and named the gate as
`cargo clippy --release` plus `cargo clippy --release --target wasm32-unknown-unknown --lib`.
Both were clean. But `cargo clippy --release` compiles the lib and the bins and **never reads a
test**, so nothing in the loop ever linted test code. `--all-targets` reported **91 warnings** —
90 in the lib's own `#[cfg(test)]` modules and 1 in `tests/requested_limits.rs`.

BUG-20 recorded 88 on 2026-09-07 and a later entry recorded 90; it was 91 when this item was
picked up. The count drifts upward on its own, which is the whole problem: the rule said zero and
the check could not see them.

The 91, by lint:

| n | lint | substance |
|---|---|---|
| 38 | `field_reassign_with_default` | style — `let mut c = T::default(); c.f = v;` |
| 27 | `needless_range_loop` | style — index loop over a slice of known length |
| 17 | `unnecessary_cast` | style — `as i32` on an expression already `i32` |
| 4 | `unused_variables` | **one dead binding**, three unused bindings |
| 2 | `assertions_on_constants` | **two runtime asserts over compile-time constants** |
| 1 | `needless_borrow` | style |
| 1 | `manual_div_ceil` | style |
| 1 | `manual_range_contains` | style |

Two of the eight were worth having found, and neither is a "noisy in test code" lint:

- **`assertions_on_constants`** pointed at `rans_gpu_encode.rs`'s BUG-35 guard test, which
  asserted `MAX_GROUPS * MAX_GROUP_ALPHABET > SHARED_HIST_ENTRIES` and `3266 <
  SHARED_HIST_ENTRIES` at **run time**. Both are relations between `const usize` values: the
  assertions cannot fail while the test runs, and they cannot fail at all — they can only stop
  being true when someone edits a constant, which is a build-time event that a runtime test
  reports late and only if the test is run. They are now `const _: () = assert!(…)`, so shrinking
  the arena fails the build.
- **`unused_variables`** found `storage_dst` in `rice_gpu.rs` — a `BufferUsages` value computed
  and never used, while the buffers around it spell their usage inline. Deleted. The other three
  are two GPU read-backs in a debug test whose values are never asserted on and one unused `y` in
  a horizontal-gradient generator; those are prefixed with `_`, which keeps the read-back doing
  its work and says in the source that nothing checks it.

## Decision

**Widen the native gate to `cargo clippy --release --all-targets` and fix all 91.** CLAUDE.md and
LOOP.md now name that form, and CLAUDE.md says why the two gates are asymmetric — native is
`--all-targets`, wasm stays `--lib` (BUG-24: the CLI is not a wasm artifact).

**Nothing was suppressed.** No `#[allow]` was added, at item level or any other level. Every
warning is a fixed line.

## The one way `needless_range_loop` can make a test weaker, and what was done about it

`for i in 0..n { a[i] = … }` panics when `a` is shorter than `n`. `a.iter_mut().enumerate()` runs
`a.len()` times and cannot. Where `n` and `a.len()` are the same expression that is a pure win;
where `n` comes from a *separate* computation, the rewrite converts an out-of-bounds panic into a
**loop that silently does less work** — which is the vacuous-pass failure mode this project keeps
running into (see the scene-cut guard in `test_bframe_yuv420_*`, which asserts the frame types
before measuring precisely so a degenerate sequence cannot pass quietly).

Four of the 27 sites were that shape, so the bound is now written down instead of being an
accident of indexing:

- `pipeline_tests.rs`: `tiles` holds every plane's tiles and the loop wants one plane's worth →
  `assert!(tiles.len() >= tiles_per_plane)`, then `.take(tiles_per_plane)`.
- `rice_gpu.rs`, three sites: the loop bound was `tile_sz * tile_sz` / `coeffs_per_tile` while the
  slice came from `rice_decode_tile` → `assert_eq!(decoded.len(), …)` at the top of the body, and
  the iteration is over the whole slice.

The fifth candidate needed nothing: `plane_tiles` is `&tiles[p * tpp..(p + 1) * tpp]`, so its
length *is* the bound, and the `.take()` there was redundant and was removed. The five
`compressed`/`psnr` sites are all preceded by an existing `assert_eq!(len, 9)`.

## What was not chosen

- **Writing down that the zero-warning rule covers shipped code and not tests.** This was the
  live alternative and BUG-20 put it first. Rejected on what the 91 contained: the gate that does
  not read the tests is the only gate this project has, since there is no CI — and the two
  substantive findings above sat in test code specifically. A rule that exempts tests would have
  kept both. It would also have to explain why `tests/requested_limits.rs`, which exists to stop
  a third `Limits` override landing silently (BUG-34, `0047`), is not code.
- **`#[allow(clippy::needless_range_loop)]` on the noisy sites.** 27 of the 91 are that one lint,
  and it is the most arguable of the eight in test code — `for (i, cf) in
  compressed.iter().enumerate().take(8).skip(1)` is not plainly better than `for i in 1..8`.
  Rejected because it is 27 individual justifications for the same non-reason, which is a
  module-level allow written out longhand, and CLAUDE.md refuses those.
- **A `[lints.clippy]` table in `Cargo.toml` turning the style lints down.** Cargo's lints table
  is per-package, not per-target, so it cannot say "quieter in tests" — it would relax the lint
  for shipped code too, which is the opposite of the point.
- **`cargo clippy --workspace --all-targets --all-features`.** No workspace, and `--all-features`
  would pull feature combinations the loop never builds. `--all-targets` is the smallest form that
  closes the gap.

## Cost

`--all-targets` compiles the test, bench and example targets under `clippy-driver`, which
`cargo test --release` does not share, so the gate is slower than `cargo clippy --release` was.
**Not timed:** eight sessions were live on this machine (load average 44), and CLAUDE.md's rule is
that a timing figure taken under load is not a figure. The compile is of code the loop already
builds, and it is the same one-command shape as before.

## Scope of the change

Every edit is inside `#[cfg(test)]` code or an integration test target — checked mechanically,
and the eleven `src/` files touched split two ways. For **nine** of them the first changed line is
below that file's own `#[cfg(test)]` marker (`codec_compare` 326>318, `abac` 726>707, `bitplane`
737>701, `cfl` 907>792, `huffman` 946>871, `rans` 1934>1906, `rans_gpu_encode` 2082>2031, `rice`
1011>976, `rice_gpu` 1599>1581). The other **two** have no marker of their own because they *are*
test files: `src/{encoder,decoder}/pipeline_tests.rs` are reached only through
`#[cfg(test)] #[path = "pipeline_tests.rs"] mod tests;` in the two `pipeline.rs`. Plus
`tests/requested_limits.rs`, which is an integration test target. So the shipped build is
unchanged by construction rather than by measurement, and no encode figure in the repository
moves.

## What this leaves open

`cargo fmt --check` is red the same way, and worse: **566 diffs in 61 files, 504 of them in 44
files under `src/`**. GOALS §9 requires `cargo fmt` clean and neither CLAUDE.md nor LOOP.md names
it as a gate. Filed as **BUG-38** rather than fixed here — reformatting 44 shipped modules
conflicts with every session in flight and carries no behaviour, so it wants a quiet tree.
