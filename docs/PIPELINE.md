# GNC Encode Pipeline

*Updated 2026-09-06 — reflects current codebase on `main`*

## Overview

All stages run as wgpu compute shaders (WGSL). Tiles are 256x256 and fully independent — no cross-tile dependencies at any stage. Default entropy backend is Rice+ZRL.

```
RGB input
  → Pad (edge-replicate to tile boundary)
  → Color convert (RGB → YCoCg-R)
  → Deinterleave (split into Y, Co, Cg planes)
  → Wavelet transform (CDF 9/7 lossy, LeGall 5/3 lossless)
  → Adaptive quantization (variance-based weight map)
  → Quantize (per-subband weights, dead-zone, adaptive QP)
  → CfL chroma prediction (q=50–85, least-squares alpha)
  → Entropy coding (Rice+ZRL default, rANS/Huffman available)
```

## Stages

### 1. Padding — `pad.wgsl`

Pads input to tile-aligned dimensions using edge replication. Workgroup size 256, one thread per pixel.

### 2. Color Space — `color_convert.wgsl`

**RGB → YCoCg-R** via lifting steps. Lossless mode (q=100) uses `floor()` for integer-exact reversibility. Lossy mode uses f32 intermediates. One thread per pixel.

### 3. Deinterleave — `deinterleave.wgsl`

Scatters interleaved YCoCg-R into three separate plane buffers: Y (luminance), Co (chroma orange), Cg (chroma green).

### 4. Wavelet Transform — `transform_97.wgsl` / `transform_53.wgsl`

Separable 2D wavelet (row pass → column pass), repeated for 4–5 decomposition levels (5 at q≥25, 4 below — BUG-6).

The tile size sets the ceiling, not the image size: each level halves the tile, and the floor of 8 samples means 256 px carries 5 levels, 512 carries 6, 128 carries 4. Use `CodecConfig::set_tile_size()` rather than assigning `tile_size` directly — it re-derives that ceiling in both directions (BUG-12).

| Mode | Filter | Use case |
|------|--------|----------|
| Lossy (q=1–99) | CDF 9/7 | Better energy compaction |
| Lossless (q=100) | LeGall 5/3 | Integer-exact roundtrip |

Produces LL (lowpass, top-left) + detail subbands LH, HL, HH per level. 8 dispatches per level × 4–5 levels (5 at q≥25).

**Alternative:** Block DCT-8×8 (`dct8_fused.wgsl`) — fused forward+quantize+inverse mega-kernel. Matches CDF 9/7 RD in fewer dispatches. Currently experimental.

### 5. Adaptive Quantization — `variance_map.wgsl` + `weight_map_normalize.wgsl`

Active when `aq_strength > 0` (default at q ≤ 80).

1. **Variance analysis:** Per-block (8×8 in LL domain) variance of Y's LL subband
2. **Weight normalization:** Log-domain conversion → geometric mean normalization to 1.0 → 3×3 spatial smoothing

Output: multiplicative per-block weight map that modulates the quantization step.

### 6. Quantization — `quantize.wgsl` / `quantize_histogram_fused.wgsl`

```
q[i] = round(coeff[i] / (step × subband_weight × aq_weight)) × sign(coeff[i])
```

- **Dead zone:** coefficients in `[-dead_zone, +dead_zone]` map to zero
- **Subband weights:** uniform by default; `GNC_PHYSICAL_WEIGHTS` selects the perceptual curve
- **Chroma:** `chroma_weight` 1.0–1.5, tightening with quality (1.0 at q≥85)
- **Quality step:** log-interpolated from preset anchors (q=1,10,25,50,75,85,92,99,100). The
  ladder above q=92 was capped at qstep 2.0 by a stale rANS constraint until 2026-09-06; q=92/96/99
  now give 51.7 / 55.2 / 59.8 dB where they previously gave the same picture three times
- **P-frames** are quantised more coarsely than I-frames, scaled 1.25x at qstep ≥ 4.6 and tapering
  to 1.0 at qstep ≤ 2.8 (TUNE-6)

**Fused path** (`quantize_histogram_fused.wgsl`): combines quantization + entropy histogram in one kernel. Used when CfL is off. Saves one full GPU pass.

### 7. Chroma-from-Luma (CfL) — `cfl_alpha.wgsl` + `cfl_forward.wgsl`

Active at q=50–85. Predicts chroma from reconstructed luma to reduce chroma entropy.

1. **Alpha computation** (per-tile, per-subband): `α = Σ(Y·C) / Σ(Y²)` — least-squares fit, quantized to 14-bit
2. **Forward prediction:** `residual = chroma − α × Y_reconstructed`

Encoder transmits residuals (lower entropy) + alpha values. Decoder reverses with inverse prediction.

### 8. Entropy Coding

Three backends exist. **Rice+ZRL is the default above q=20**; rANS is selected at q≤20, where it
codes better. The crossover was re-swept on 2026-09-06 after rANS's frequency tables were packed
and it stayed where it was.

#### Rice+ZRL (default) — `rice_encode.wgsl`

- **256 fully independent streams per tile** — maximum GPU parallelism
- Per coefficient: zero-bit → (if nonzero) sign + Golomb-Rice magnitude code
- Zero-run-length extension for efficient zero runs
- Rice parameter `k` chosen per subband group; zero-run `k` has two magnitude contexts
- Shared memory usage < 1 KB → excellent occupancy on M1
- **Stream layout:** streams walk the tile **column-major**, cut into 256 contiguous segments, so
  the previous symbol in a stream is the coefficient directly above it — the vertical adjacency
  the adaptive `k` and the zero runs are tuned against. Until 2026-09-06 the mapping was `i % 256`,
  which gives that property only at a 256 px tile and interleaved distant columns at any other
  width (BUG-11). At 256 px the two are identical coefficient for coefficient

#### rANS (parked) — `rans_encode.wgsl`

- 32 interleaved streams per tile
- Requires histogram (`rans_histogram.wgsl`) → normalization to 12-bit (`rans_normalize.wgsl`) → encode
- Per-subband frequency tables, optional context-adaptive mode
- Frequency tables are coded as alternating zero-run and value, both Exp-Golomb order 0 (ENT-1,
  2026-09-06). They used to be flat `u16` and cost 23–26% of every file; packing them made the
  file 11.7–26.6% smaller at bit-identical quality
- Better compression at low rate, but a sequential dependency the Rice path does not have

#### Huffman (parked) — `huffman_encode.wgsl`

- 64-symbol alphabet + exp-Golomb escape codes
- Per-tile codebook construction

## Current Performance (1080p, bbb reference, M1 GPU)

| Quality | PSNR | BPP | Encode FPS | Decode FPS |
|---------|------|-----|------------|------------|
| q=25 | 33.2 dB | 1.71 | 39 | 72 |
| q=50 | 37.7 dB | 2.37 | 40 | 60 |
| q=75 | 42.8 dB | 4.01 | 40 | 59 |
| q=90 | 50.5 dB | 8.90 | 40 | 63 |

Sequence encode: **31.7 fps** (1080p, q=75, Rice, keyframe interval 8, I+P+B frames)

> These throughput figures are indicative to about ±25%. Up to five sessions share this machine and
> the same workload has timed 25.2, 31.1 and 37.5 ms across three runs. Three different quantities
> have also been called "encode fps" and they differ by 2.4x — see BASELINE.md. The compression
> columns are deterministic and carry no such caveat.

## Shader Inventory (encode path)

| Shader | Stage | Notes |
|--------|-------|-------|
| `pad.wgsl` | Padding | Edge-replicate |
| `color_convert.wgsl` | Color | RGB ↔ YCoCg-R |
| `deinterleave.wgsl` | Plane split | Y/Co/Cg |
| `transform_97.wgsl` | Wavelet | CDF 9/7 (lossy) |
| `transform_53.wgsl` | Wavelet | LeGall 5/3 (lossless) |
| `variance_map.wgsl` | AQ | Block variance |
| `weight_map_normalize.wgsl` | AQ | Weight map |
| `quantize.wgsl` | Quantize | Standard path |
| `quantize_histogram_fused.wgsl` | Quantize | Fused with histogram |
| `cfl_alpha.wgsl` | CfL | Alpha computation |
| `cfl_forward.wgsl` | CfL | Chroma prediction |
| `rice_encode.wgsl` | Entropy | Rice+ZRL (default) |
| `rans_histogram.wgsl` | Entropy | rANS histogram |
| `rans_normalize.wgsl` | Entropy | rANS freq tables |
| `rans_encode.wgsl` | Entropy | rANS encode |
| `huffman_encode.wgsl` | Entropy | Huffman encode |
| `dct8_fused.wgsl` | Transform | Block DCT-8 (experimental) |
