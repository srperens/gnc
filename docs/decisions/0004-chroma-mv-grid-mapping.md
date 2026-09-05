# 0004 — Chroma MC indexes the MV field through an explicit grid mapping

**Date:** 2026-09-05. **Status:** accepted. **Closes:** BUG-1.

## Context

4:2:0 pyramid B-frames at layers 2–3 reconstructed far below their bitrate: on BBB at q=75,
B-frames measured 32.7–36.2 dB while the same frames in 4:4:4 measured 39.2–41.4 dB. B₄ and
P-frames were unaffected in both formats. The bug had survived earlier fix attempts (archived
item #14) because each attempt reasoned about *which count to dispatch* rather than about what
the shader was actually indexing.

## What was actually wrong

`motion_compensate_bidir_chroma.wgsl` derives a block index from the chroma pixel position:

```
block_idx = (y / 4) * (chroma_w / 4) + (x / 4)
```

and uses it directly as an index into the MV and block-mode fields. That is only valid when the
MV field is on the same grid as the chroma 4×4 block grid. Two different grids reach this shader:

| Frame type | MV field grid | Entries at 1080p | Matches chroma grid? |
|---|---|---|---|
| P (and B₄, coded as P) | 8×8 luma split MVs | 40960 | yes — 1:1 |
| True B | 16×16 luma ME MVs | 10240 | **no** — half resolution on each axis |

So for every true B-frame the shader read the MV field with the wrong row stride. This had two
distinct consequences, and previous analysis had only ever identified the second:

1. **Spatial scrambling.** Even for in-range indices, chroma block *(bx, by)* read the MV
   belonging to an unrelated luma block, because the read assumed 256 columns where the field
   had 128. Prediction was wrong everywhere, not just past some boundary.
2. **Tail divergence.** Indices beyond the field's length resolved differently on each side:
   the encoder read out of bounds from a short, freshly allocated buffer, while the decoder read
   *in bounds* from its persistent `mv_buf`, which `ensure_var_buf` grows but never clears, and
   so still held the previous P-frame's split MVs. Encoder and decoder therefore predicted from
   different motion, and the coded residual was added to the wrong prediction.

`block_modes` was indexed by the same expression and was wrong in the same two ways.

A third defect surfaced while fixing this: luma and chroma planes are padded to a tile multiple
*independently* (1080p → luma 2048×1280, chroma 1024×768), so the chroma block grid is not a
fixed multiple of the MV grid — 192 chroma rows against 80 MV rows, a ratio of 2.4. Any mapping
derived from grid *dimensions* is therefore both non-integer and wrong; and the surplus rows
index past the field on both the B and the P path.

## Decision

State the mapping explicitly instead of assuming the grids coincide.

- **`ChromaMvGrid`** (`src/encoder/motion.rs`) derives the row stride and the per-axis shift from
  *block geometry* — the MV block size in luma pixels against the luma pixels one chroma block
  spans — never from the ratio of padded grid dimensions. Encoder and decoder both construct it
  from this one constructor, so they cannot drift apart.
- The shader takes `mv_blocks_x`, `mv_blocks_y` and the two shifts, maps chroma block → MV block,
  and **clamps** to the field extent. Padding blocks beyond the real image now resolve
  identically on both sides instead of relying on out-of-bounds behaviour.
- `mv_scale` is dispatched with the frame's own MV count on both sides. There is no tail to fill,
  so there is nothing left to disagree about.

The P path maps to shift (0,0) and an identical stride, so it is unchanged by construction —
confirmed by 4:4:4 output being bit-identical before and after.

## Why not the alternative

The obvious smaller fix — zero-fill the tails on both sides so encoder and decoder agree — was
rejected. It would have made the two sides *consistent* while leaving the prediction *wrong*:
the bottom of every chroma plane predicted from a zero MV and the top from scrambled MVs, with
the residual coder paying to correct both. Agreement is necessary but is not the goal; correct
prediction is.

## Measured effect

1080p, q=75, 17 frames, ki=9, 4:2:0, `GNC_REF_DEBLOCK=0`. Per-frame PSNR of true B-frames:

| | BBB before | BBB after | touchdown before | touchdown after |
|---|---|---|---|---|
| worst B-frame | 32.75 dB | 37.33 dB | 31.36 dB | 35.02 dB |
| B-frame range | 32.8–36.2 | 37.3–39.9 | 31.4–34.8 | 35.0–38.0 |
| B₄ (P path) | 40.81 | 40.81 | 39.40 | 39.40 |

Sequence VMAF and rate:

| | VMAF mean | VMAF min | bpp |
|---|---|---|---|
| BBB before | 95.52 | 91.10 | 1.79 |
| BBB after | **96.13** | **93.68** | **1.76** |
| touchdown before | 97.17 | 92.86 | 2.06 |
| touchdown after | **97.59** | **94.96** | **2.05** |

Quality rose while rate fell on both sequences. That combination is the evidence that this is a
prediction fix and not a masking change: a better prediction produces both a smaller residual to
code and a better reconstruction. A change that merely re-aligned encoder and decoder would have
improved quality at *higher* rate.

## Canary

`GNC_DIAGNOSTICS=1` prints per B-frame:

```
[bframe_chroma_mv] enc grid: mv_blocks_x=128 shift=(1,1) me_blocks=10240 chroma_blocks=49152
```

This is what exposed the independent-padding defect above — `chroma_blocks` is not
`me_blocks << 2`, which is visible at a glance and was not visible from reading the code.
