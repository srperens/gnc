# 0047 — The storage-buffer override is 9, not 10, and it is a named exception

**Date:** 2026-09-08
**Item:** BUG-34
**Status:** accepted — override kept, reduced, and asserted

## Context

`GpuContext` asked for `max_storage_buffers_per_shader_stage: 10` on top of
`wgpu::Limits::default()`, whose value is **8** (wgpu-types 24.0.0, and the WebGPU spec).
CLAUDE.md's portability table printed the 10 and said GNC asks for wgpu's defaults so the
same shaders run under WebGPU (GOALS rule 4). Both halves were false for that row: 10 is
an override, and a conformant implementation held to 8 would fail `request_device`.

The adapter here offers 31, so nothing failed. `gnc gpu-info` had been printing the proof.

Counted from the shaders (naga, per compute entry point, only `var<storage>` the entry
point actually uses):

| storage buffers | shaders |
|---|---|
| **9** | `block_match_bidir.wgsl:main` only |
| 7 | `motion_compensate_bidir.wgsl:main`, `motion_compensate_bidir_chroma.wgsl:main` |
| ≤6 | the rest (file-level `grep var<storage>` overcounts shaders with several entry points) |

So **10 was one of unused slack.** 9 is the least request that still creates the B-frame
matcher. 8 is the spec default and needs that one shader to shed a buffer.

## Decision

**Keep one named override, at 9. Do not merge `block_match_bidir`'s buffers in this item.**

`gnc::required_limits()` is the single `Limits` value passed to `request_device` (and to
the two probe examples). It is `Limits::default()` with
`max_storage_buffers_per_shader_stage = 9`. `tests/requested_limits.rs` compares the two
structs with `PartialEq`, so a new override cannot land as a one-line field update, and
counts storage buffers per entry point against that 9, so a tenth binding fails the test
rather than the browser.

`gnc gpu-info` prints the override against `Limits::default()` rather than only against
the adapter.

## What was not chosen

- **`Limits::default()` unmodified (8).** That is the success criterion's other arm, and it
  is the right end state. Reaching it means merging two of `block_match_bidir.wgsl`'s nine
  storage bindings — `fwd_motion_vectors` + `bwd_motion_vectors` (both `read_write array<i32>`)
  or `predictor_fwd_mvs` + `predictor_bwd_mvs` (both `read array<i32>`). Not done here:
  B-frames have been off by default since BUG-5, the same file is BUG-25's crash site, and
  **BUG-40 holds it** (eager pipeline creation / DX12 FXC X3695). A merge in a low-traffic
  shader with three open reasons for caution is the item's judgement call, and the call is
  no. The test will fail in the helpful direction if that shader ever drops to 8: it asserts
  the heaviest entry point is still `block_match_bidir` at 9.
- **Leaving the request at 10.** One unused slot is how the override went unnoticed; it is
  not a safety margin worth keeping.
- **Raising the request to the adapter's 31.** That is the opposite of rule 4.

## Canary

`cargo test --release --test requested_limits -- --nocapture` prints
`[bug34] … max block_match_bidir.wgsl:… at 9 storage buffers; request 9`.
`gnc gpu-info` prints `storage buffers / stage overrides Limits::default() (8)`.
The default path does not print `[bug34]`.
