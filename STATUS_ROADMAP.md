# GNC — Project Status & Roadmap

## What GNC Is

GNC (GPU-Native Codec) is a video/image codec designed from scratch around GPU
parallelism, rather than adapting CPU-era algorithms. The core premise: as
resolutions scale from 1080p to 4K to 8K, a tile-based GPU-native architecture
scales nearly linearly — more tiles means more parallel workgroups, not more
sequential work.

**Key design principles:**
- Every tile is fully independent — no cross-tile dependencies
- Runs as wgpu compute shaders (WGSL), targeting Metal, Vulkan, DX12, WebGPU
- Patent-free, open source
- Single codec covering the full quality spectrum (lossless to ~1:300)

---

## Current State (2026-02-23)

### What works
- End-to-end encode/decode pipeline on GPU
- LeGall 5/3 integer wavelet transform (lossless-capable, 1-4 levels)
- Per-subband weighted quantization (uniform, perceptual, custom presets)
- Dead-zone quantization (configurable threshold)
- Per-tile rANS entropy coding (32 interleaved streams per tile, ZRL)
- Chroma-from-Luma (CfL) prediction (per-tile per-subband alpha)
- Adaptive quantization infrastructure (weight maps, SSIM-guided — in progress)
- GPU rANS decoder (CPU encoder, GPU decoder)
- Pipelined decode path (overlap GPU/CPU for throughput)
- CLI: encode (`--cfl`), decode, benchmark, sweep (7 experiment sets)
- GPC6 bitstream format (backward-compatible evolution from GPC4/5)

### Compression performance (blue_sky 1920x1080)

**Baseline (uniform quantization, no dead zone):**

| QStep | PSNR    | BPP  |
|-------|---------|------|
| 4     | 43.47 dB | 5.44 |
| 8     | 38.19 dB | 3.32 |
| 16    | 32.68 dB | 1.97 |

**With CfL enabled (1.7-3.9% bitrate reduction, quality-neutral):**

| QStep | PSNR    | BPP  | Savings |
|-------|---------|------|---------|
| 4     | 43.46 dB | 5.35 | -1.7%   |
| 8     | 38.31 dB | 3.24 | -2.5%   |
| 16    | 32.87 dB | 1.89 | -3.9%   |

**Best combined settings (perceptual weights + dead zone 0.75):**

| QStep | PSNR    | BPP  | vs baseline |
|-------|---------|------|-------------|
| 8     | 34.18 dB | 1.66 | -50%        |
| 16    | 29.21 dB | 0.87 | -56%        |

From initial i16 packing (48 bpp) to current best (1.66 bpp at q=8): **29x
total compression**. The pipeline improvements stack: rANS (7.4x), wavelet
(2.2x), dead zone + subband weights (1.8x combined).

### Throughput (blue_sky 1920x1080, Apple M1)
- Encode: 188 ms (5.3 fps) — bottleneck is CPU-side rANS
- Decode: 36 ms sequential (27.6 fps), 34 ms pipelined (29.3 fps)

### What was tried and rejected
- **Zero-run-length coding**: rANS already handles zeros efficiently. ZRL
  added 0-1.4% savings — not worth the complexity. Code remains but is a
  negative result.

### Hardware compatibility
- Requires `max_storage_buffers_per_shader_stage >= 4`
- Modern GPUs (NVIDIA, AMD, Apple Silicon, recent Intel) work fine
- Old/integrated GPUs (e.g. Intel HD 4000, GT 635M) hit hardware limits
- Fix applied: use `adapter.limits()` instead of `Limits::default()`

---

## The Core Thesis

Traditional codecs (H.264, HEVC, AV1) are shaped by CPU constraints —
sequential prediction chains, complex motion estimation, serial entropy coding.
They do not scale well with resolution because their fundamental dependencies
are sequential.

GNC's tile-independent design means:
- 4K = 4x more tiles = 4x more parallel workgroups — no architectural changes
- 8K follows the same logic
- The GPU does the scaling automatically

This is a genuine architectural advantage, not just a performance trick.

### Target use cases (where GNC's design fits better than alternatives)
- **Real-time video** — live streaming, video conferencing, game capture
- **Professional intermediate codec** — GPU-accelerated, low-latency proxy format
- **VR/XR** — where latency and GPU availability matter more than compression ratio
- **High-resolution pipelines** — 4K/8K where sequential codecs bottleneck

GNC is not trying to beat AV1 on compression efficiency. It occupies a
different point in the design space: simpler, parallel, lower latency.

---

## Design Vision: Full Q Spectrum

The goal is a single codec with one Q parameter covering the full range:

| Q range | Mode | Wavelet |
|---------|------|---------|
| Lossless | Reversible | LeGall 5/3 (integer, current) |
| High quality | Near-lossless | LeGall 5/3 with fine quantization |
| Medium quality | Lossy | CDF 9/7 (better energy compaction) |
| Aggressive | High compression | Haar or reduced wavelet levels |

Each tile runs its wavelet independently — the GPU doesn't care which variant
is selected. This could even be adaptive per-tile based on local content.

---

## What To Do Next

### 1. Per-subband entropy coding (high ROI, moderate effort)
The rANS coder uses one frequency table per tile across all subbands. LL
coefficients have very different distributions from HH detail. Splitting
statistics by subband within each tile would improve compression 10-20%
with minimal complexity — the subband index computation already exists.
Still fully parallel, no cross-tile dependencies.

### 2. GPU rANS encode (throughput bottleneck)
CPU-side rANS encoding is the 3.6x throughput bottleneck (188ms encode vs
36ms decode). Moving rANS encode to GPU compute shaders could push encode
to 30-50 fps at 1080p. The GPU rANS decoder already exists as reference.

### 3. CDF 9/7 for lossy mode
LeGall 5/3 is optimal for lossless but suboptimal for lossy. CDF 9/7 (the
JPEG 2000 lossy wavelet) gives significantly better energy compaction and would
close much of the gap to JPEG 2000. Requires keeping both 5/3 (lossless) and
9/7 (lossy) paths — straightforward with tile-independent architecture.

### 4. Rate control (target bitrate)
Currently you pick qstep and get whatever bpp falls out. Real-time use cases
need target-bpp encoding. A per-tile qstep adjustment pass (encode, measure,
adjust) is GPU-friendly and essential for any streaming application.

### 5. Finish adaptive quantization
Infrastructure is in place (weight maps, aq_strength, SSIM-guided pipeline).
Needs tuning and validation. Expected gain: redistribute bits from smooth
regions to textured regions for better perceptual quality at same bitrate.

### 6. Temporal extension (video)
The biggest compression opportunity. Two GPU-native approaches worth exploring:
- **GPU block matching**: find best matching tile in previous frame via compute
  shader, encode residuals only. Massively parallel, patent-free.
- **3D wavelet in time**: apply LeGall 5/3 across N frames in the time
  dimension. Simpler but less effective without motion compensation.

### 7. Tile index table for random access
Tiles are independent but the bitstream is sequential. Adding a tile offset
table in the header enables O(1) random access — critical for region-of-interest
decode in VR/AR and partial frame updates.

### 8. Benchmark against JPEG and JPEG 2000
Use standard test images (Kodak dataset, Xiph frames). Measure PSNR, SSIM,
encode time, decode time. Establish a real baseline to track progress against.

### 9. WebAssembly / WebGPU build
wgpu supports WASM targets. A browser-runnable demo would be the easiest way
to demonstrate GNC across platforms including mobile, without native app
packaging.

---

## What GNC Is Not Trying To Be

- A replacement for AV1 or HEVC on compression efficiency
- A codec for archival or offline encoding where time doesn't matter
- A one-person replication of decades of codec research

GNC's value is in its architecture, not its current compression numbers. The
numbers will improve. The architecture is the thesis.
