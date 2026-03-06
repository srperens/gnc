# 0003 — Explicit Layout Verification for WGSL Uniform Buffer Parameters

- **Date:** 2026-03-06
- **Status:** Accepted

## Context

A subtle parameter mismatch in `dispatch_tile_energy_reduce()` caused the GPU tile
energy reduction (TER) feature to be nearly non-functional for several sessions:

The Rust side built a `[u32; 8]` params array by hand. An extra zero was inserted at
index 3 (byte offset 12), intended as padding. However, the corresponding WGSL struct
had no padding at that position — `low_thresh` was declared immediately after
`tile_size` at offset 12. The shader therefore read:

```
Rust layout:        [padded_w, padded_h, tile_size, 0_PAD, low_thresh, high_thresh, max_mul, 0]
WGSL interpretation:[padded_w, padded_h, tile_size, low_thresh,         high_thresh, max_mul, _pad, _pad2]
```

Effect: `low_thresh = 0.0`, `high_thresh = 0.5` (was `low_thresh`'s correct value).
Every tile with any energy at all exceeded `high_thresh`, so `mul` clamped to 1.0
everywhere. The adaptive quantization was completely suppressed.

This class of bug is hard to detect because:
- The encoder compiled and ran without error
- Output bpp was plausible (slightly higher than expected, not obviously broken)
- The mismatch only manifests as wrong parameter values inside the shader — no crash,
  no assertion, no NaN that propagates visibly
- The GPU TER appeared to "work" (correct code path taken) but had no effect

## Decision

**All WGSL uniform structs with more than two fields must have a corresponding byte
offset comment block in the Rust code that constructs the params array.**

Format:

```rust
// TileEnergyReduceParams layout (matches WGSL struct — no implicit padding):
//   offset  0: padded_w    (u32)
//   offset  4: padded_h    (u32)
//   offset  8: tile_size   (u32)
//   offset 12: low_thresh  (f32)
//   offset 16: high_thresh (f32)
//   offset 20: max_mul     (f32)
//   offset 24: _pad        (u32)
//   offset 28: _pad2       (u32)  — 16-byte struct alignment
let params_data: [u32; 8] = [
    padded_w,
    padded_h,
    tile_size,
    low_thresh.to_bits(),   // offset 12
    high_thresh.to_bits(),  // offset 16
    max_mul.to_bits(),      // offset 20
    0,                      // _pad
    0,                      // _pad2
];
```

Additionally:

- **WGSL structs used as uniform bindings must include explicit `@align` and/or
  `@size` annotations** when their natural alignment could differ from what a
  Rust `repr(C)` struct or hand-built array would produce.
- **For non-trivial structs**, prefer a `#[repr(C)]` Rust struct with `bytemuck::Pod`
  over a raw `[u32; N]` array. The struct self-documents the layout and makes type
  mismatches a compile error.
- **Whenever a new shader parameter struct is added or modified**, the WGSL struct and
  the Rust params construction must be reviewed side by side and the byte offsets
  verified.

## Consequences

- The comment block adds ~10 lines of documentation per dispatch function. Worth it:
  layout bugs cost far more in debugging time than the comments cost to maintain.
- Moving to `#[repr(C)]` structs with `bytemuck::Pod` is the stronger long-term fix
  and should be adopted when existing dispatch functions are refactored. It makes the
  Rust compiler enforce the layout contract at compile time rather than relying on
  manual comment maintenance.
- This decision applies to all shader param construction in `src/encoder/pipeline.rs`
  and any future dispatch helpers. Existing call sites should be audited in a single
  pass and retrofitted with offset comments.

## Background: WGSL Struct Alignment Rules

WGSL follows WebGPU's uniform buffer layout rules (roughly: members aligned to their
own size, structs 16-byte aligned). Key points:

- `f32`, `i32`, `u32`: 4-byte aligned, 4 bytes
- `vec2<f32>`: 8-byte aligned, 8 bytes
- `vec3<f32>`: 16-byte aligned, 12 bytes (struct alignment becomes 16)
- `vec4<f32>`: 16-byte aligned, 16 bytes
- Structs: aligned to the largest member alignment, rounded up to 16 bytes

Rust's `[u32; N]` is always 4-byte aligned with no implicit padding. A mismatch
occurs when the WGSL struct has padding that the Rust array does not (or vice versa).
The safe pattern is: construct the array with explicit padding entries that mirror
any WGSL-side padding, and document both sides with offset comments.
