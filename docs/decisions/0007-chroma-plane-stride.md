# 0007 — Chroma MC must use the chroma plane's own stride, and the MV field's

**Date:** 2026-09-05. **Status:** accepted. **Closes:** BUG-3.

## Symptom

4:2:0 inter frames reconstructed 15 dB below their bitrate at some resolutions and were fine at
others, with no obvious pattern. I-frames were unaffected everywhere; 4:4:4 was unaffected
everywhere.

## Finding the rule

A sweep over frame sizes and tile sizes produced a clean split, and the first hypothesis drawn
from it — "breaks when the chroma plane is smaller than one tile" — was **falsified** by
384x384, where the chroma plane (192) is smaller than the tile (256) and the output is healthy.

Reading the data by tile counts instead gave the rule, and a non-square test settled the axis:

| geometry | luma tile grid | result |
|---|---|---|
| 768x512 | 3 x 2 | broken |
| 512x768 | 2 x 3 | **healthy** |
| 1920x1088 | 8 x 5 | healthy |

Only the *horizontal* tile count matters — the signature of a wrong row stride, not of a wrong
region size. The condition is `padded_w != 2 * chroma_padded_w`, which happens exactly when
`tiles_x` is odd. That includes **1280x720**, where 5 luma tile columns pad the chroma plane to
768 while `padded_w / 2` is 640.

## Root cause

Two separate off-by-stride errors, one on each side, in the P-frame 4:2:0 chroma path:

- **Encoder** (`buffer_cache.rs`): `mc_fwd_params_chroma420` was built from `padded_w / 2,
  padded_h / 2` under a comment asserting "chroma dims = padded/2". The chroma plane is padded to
  a tile multiple *independently* of luma, so that is false whenever `tiles_x` is odd. The
  shader's pixel row stride was wrong while the dispatch covered the correct pixel count.
- **Decoder** (`gpu_work.rs`): passed the correct chroma dimensions, but then derived
  `blocks_x = chroma_padded_w / 4` and used it to index the motion-vector field — which lives on
  the *luma* 8x8 split grid with stride `padded_w / 8`. At 720p that is 192 against 160.

So each side had one of the two strides right and the other wrong, in different ways. This is the
same defect class as BUG-1: code assuming the chroma grid and the MV grid coincide.

## Decision

Say which grid is being indexed, everywhere, instead of deriving it from whichever dimension is
in scope. `motion_compensate.wgsl` now takes `mv_blocks_x` / `mv_blocks_y` alongside its own
`blocks_x`, and clamps blocks past the field's end so encoder and decoder resolve padding
identically. `compensate()` takes an explicit `mv_grid: Option<(u32, u32)>` — `None` for luma and
4:4:4, where the grids genuinely coincide; the luma split grid for subsampled chroma.

The encoder's cached chroma MC params are built from the real `chroma_padded_w/h`, which required
threading those through `ensure_cached` and adding them to the cache key — the chroma padding
changes with the chroma format, so without that a format switch would keep params with the wrong
stride.

## Measured effect

Synthetic translating texture, 4:2:0, q default, 9 frames, anchor P-frame PSNR:

| geometry | before | after |
|---|---|---|
| 1280x720 | 23.63 dB | **37.92 dB** |
| 768x768 | 23.86 dB | **37.93 dB** |
| 256x256 | 20.57 dB | **37.93 dB** |
| 512x512 | 37.91 dB | 37.91 dB (unchanged) |
| 1920x1088 | 37.92 dB | 37.92 dB (unchanged) |

Real content, 1080p q=75 ki=9 4:2:0, 17 frames, `GNC_REF_DEBLOCK=0`:

| | VMAF mean | VMAF min | total bytes |
|---|---|---|---|
| BBB before | 96.19 | 94.74 | 7 760 719 |
| BBB after | 96.19 | 94.74 | **7 457 987** (−3.9%) |
| touchdown before | 97.59 | 94.96 | 9 046 580 |
| touchdown after | 97.65 | 95.01 | **8 691 484** (−3.9%) |

1080p has an even `tiles_x`, so its horizontal stride was already correct — but its *height* was
also wrong (`padded_h / 2` = 640 against a chroma plane of 768 rows), and the shader's
`total_pixels` guard left rows 640–767 unwritten. Those rows are padding, so quality was
unaffected, but the stale contents were still wavelet-transformed and entropy-coded. That is the
3.9% — the encoder was paying to code garbage in the bottom of every chroma plane at the
project's primary resolution.

## Note on the gate

The originally recorded BUG-3 gate ("confirm it tracks `chroma_plane < tile_size`") was wrong,
and the sweep that falsified it took four measurements. Worth keeping as a reminder that a rule
inferred from a handful of black-box samples is a hypothesis, not a diagnosis — the axis test
(768x512 against 512x768) is what actually identified this as a stride bug and pointed at the
code.
