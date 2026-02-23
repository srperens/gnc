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
- CDF 9/7 lossy wavelet (implemented, not yet default)
- Per-subband weighted quantization (uniform, perceptual, custom presets)
- Dead-zone quantization (configurable threshold)
- Per-tile rANS entropy coding (32 interleaved streams per tile)
- **Per-subband entropy coding** (4 frequency tables per tile, 8-20% savings)
- Chroma-from-Luma (CfL) prediction (per-tile per-subband alpha)
- GPU rANS decoder (CPU encoder, GPU decoder)
- Pipelined decode path (overlap GPU/CPU for throughput)
- CLI: encode (`--cfl`, `--per-subband`, `--wavelet 97`), decode, benchmark, sweep (8 experiment sets)
- GPC9 bitstream format (backward-compatible evolution from GPC4-8)

### Compression performance (bbb_1080p 1920x1080)

**Baseline (LeGall 5/3, uniform quantization, no dead zone):**

| QStep | PSNR    | BPP  |
|-------|---------|------|
| 4     | 43.41 dB | 6.55 |
| 8     | 37.88 dB | 4.14 |
| 16    | 32.55 dB | 2.39 |

**With per-subband entropy (quality-neutral, 8-20% savings):**

| QStep | PSNR    | BPP  | Savings |
|-------|---------|------|---------|
| 4     | 43.41 dB | 6.03 | -7.9%   |
| 8     | 37.88 dB | 3.62 | -12.6%  |
| 16    | 32.55 dB | 1.92 | -19.5%  |

**CDF 9/7 wavelet (same qstep, +4-5 dB quality):**

| QStep | PSNR    | BPP  | PSNR gain vs 5/3 |
|-------|---------|------|-------------------|
| 4     | 46.77 dB | 7.41 | +3.36 dB |
| 8     | 41.92 dB | 4.96 | +4.04 dB |
| 16    | 37.41 dB | 3.17 | +4.86 dB |

**Best config: CDF 9/7 + per-subband entropy + CfL (quality-neutral only):**

| QStep | PSNR    | BPP  | vs 5/3 baseline |
|-------|---------|------|-----------------|
| 8     | 42.11 dB | 3.76 | +4.2 dB, -9% bpp |
| 16    | 37.62 dB | 2.23 | +5.1 dB, -7% bpp |
| 32    | 33.51 dB | 1.25 | +6.2 dB, -2% bpp |

**Best config: CDF 9/7 + per-subband + CfL + perceptual + dz=0.75:**

| QStep | PSNR    | BPP  | vs 5/3 baseline |
|-------|---------|------|-----------------|
| 8     | 37.65 dB | 2.17 | -0.2 dB, **-48% bpp** |
| 16    | 33.25 dB | 1.18 | +0.7 dB, **-51% bpp** |
| 32    | 29.47 dB | 0.62 | +2.2 dB, **-52% bpp** |

### Throughput (blue_sky 1920x1080, Apple M1)
- Encode: 188 ms (5.3 fps) — bottleneck is CPU-side rANS
- Decode: 36 ms sequential (27.6 fps), 34 ms pipelined (29.3 fps)

### What was tried and rejected
- **Zero-run-length coding**: rANS already handles zeros efficiently. ZRL
  added 0-1.4% savings — not worth the complexity. Negative result.

### Hardware compatibility
- Requires `max_storage_buffers_per_shader_stage >= 4`
- Modern GPUs (NVIDIA, AMD, Apple Silicon, recent Intel) work fine
- Old/integrated GPUs (e.g. Intel HD 4000, GT 635M) hit hardware limits
- Fix applied: use `adapter.limits()` instead of `Limits::default()`

---

## Honest Assessment: Is GNC Any Good?

### Compression efficiency — competitive with JPEG, 1.5x behind JPEG 2000

**Measured comparison at matched PSNR on bbb_1080p (1920x1080):**

| PSNR target | JPEG (libjpeg-turbo) | GNC best config | JPEG 2000 (OpenJPEG) | GNC vs JPEG |
|-------------|---------------------|-----------------|---------------------|-------------|
| ~35 dB | 2.21 bpp | ~1.6 bpp | ~1.0 bpp | **-27%** |
| ~38 dB | 4.49 bpp | ~2.5 bpp | ~1.6 bpp | **-44%** |
| ~33 dB | 1.51 bpp | ~1.2 bpp | ~0.7 bpp | **-20%** |

GNC "best config" = CDF 9/7 + per-subband entropy + CfL + perceptual weights + dead zone 0.75.

**Content-dependent results (measured, all 1080p):**
- **Animation (bbb_1080p)**: GNC beats JPEG by 20-44%
- **Nature (blue_sky)**: GNC ties to beats JPEG by 0-23%
- **Sports (touchdown)**: JPEG wins by 6-32%

On smooth/animated content, wavelets dominate DCT. On dense textures, JPEG's
8×8 DCT remains competitive. The JPEG 2000 gap is consistently ~1.5-1.7x
across all content types — this is structural (context modeling, trellis quant).

### What closed the gap (feature stacking)

From "3-5x worse than JPEG" to "competitive or better":

| Feature | Improvement | Quality impact |
|---------|-------------|----------------|
| CDF 9/7 wavelet | +4-5 dB PSNR at same Q | Quality-neutral (better) |
| Per-subband entropy | 12-31% bitrate reduction | Quality-neutral |
| CfL prediction | 9-10% bitrate reduction (with 9/7) | Quality-neutral |
| Perceptual weights | 25-30% bitrate reduction | ~2-3 dB PSNR trade |
| Dead zone 0.75 | 15-20% bitrate reduction | ~1 dB PSNR trade |

Combined: ~67% bitrate reduction vs LeGall 5/3 baseline at matched quality.

### Remaining gap to JPEG 2000

1. **No context modeling** — EBCOT uses spatial context to predict coefficients.
   GNC's rANS treats each coefficient independently. Expected improvement if
   added: 15-25%.
2. **No trellis quantization** — JPEG 2000 optimizes quantized values for
   rate-distortion. GNC uses simple rounding.
3. **30 years of optimization** — OpenJPEG is a mature implementation.

### Where GNC already has genuine advantages

1. **Decode speed**: 29 fps at 1080p on M1 GPU, with headroom for optimization.
   JPEG 2000 decode is typically 2-5 fps on CPU. Even hardware-accelerated H.264
   decode has higher latency due to serial dependencies.

2. **Resolution scaling**: 4K = 4x tiles = 4x parallel workgroups, no
   architectural changes. Sequential codecs need algorithmic redesign for 8K.

3. **Simplicity**: ~4K lines of Rust + ~300 lines of WGSL. The entire codec is
   comprehensible by one person. AV1's reference encoder is 400K+ lines.

4. **Patent-free**: No MPEG-LA, no VVC patent pool, no licensing uncertainty.

5. **Tile independence**: Enables partial decode, region-of-interest, error
   resilience — all for free from the architecture.

### The verdict

GNC with full feature stack is **competitive with JPEG** on animation and
nature content, and within 1.5-1.7x of JPEG 2000. The decoder runs at
real-time speed (29 fps at 1080p on M1). The gap to JPEG 2000 is real but
structural — closing it requires context-modeled entropy coding, which is
the next major architectural decision.

For video use cases, temporal prediction is the elephant in the room —
without it, GNC is an image codec. With it, the 3-10x compression
multiplier would make the JPEG 2000 gap irrelevant for most applications.

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

## What To Do Next (Prioritized)

### Completed
- [x] Per-subband entropy coding — 8-31% quality-neutral bitrate savings
- [x] CfL (Chroma-from-Luma) prediction — 2-14% savings
- [x] Dead-zone quantization — 18-33% savings (quality trade)
- [x] Perceptual subband weights — 25-32% savings (quality trade)
- [x] CDF 9/7 wavelet — implemented, +4-5 dB PSNR at same Q
- [x] GPU rANS decoder
- [x] Pipelined decode
- [x] Feature stacking verified — CDF 9/7 + per-subband + CfL compose correctly
- [x] Competitive benchmark — JPEG and JPEG 2000 measured on same images

### 1. Make CDF 9/7 + per-subband entropy the default
The data is in: CDF 9/7 + per-subband entropy are always beneficial for lossy
and compose correctly with all other features. Should be default for lossy
mode (keep LeGall 5/3 for lossless). Also enable per-subband entropy by
default — it's quality-neutral with 12-31% savings.

### 2. GPU rANS encode (throughput bottleneck)
CPU-side rANS encoding is the 3.6x throughput bottleneck (188ms encode vs
36ms decode). Moving rANS encode to GPU compute shaders could push encode
to 30-50 fps at 1080p. The GPU rANS decoder already exists as reference.

### 4. Rate control (target bitrate)
Currently you pick qstep and get whatever bpp falls out. Real-time use cases
need target-bpp encoding. A per-tile qstep adjustment pass (encode, measure,
adjust) is GPU-friendly and essential for any streaming application.

### 5. Temporal extension (video)
The biggest compression opportunity. Two GPU-native approaches worth exploring:
- **GPU block matching**: find best matching tile in previous frame via compute
  shader, encode residuals only. Massively parallel, patent-free.
- **3D wavelet in time**: apply LeGall 5/3 across N frames in the time
  dimension. Simpler but less effective without motion compensation.

This is where GNC could go from "interesting research" to "genuinely useful" —
temporal prediction alone typically provides 3-10x compression on video.

### 6. Context-modeled entropy coding (moderate effort, 15-25% savings)
The rANS coder treats each coefficient independently. Using spatial context
from already-decoded neighbors in the same subband to modulate frequencies
could save 15-25%. This is what closes the gap to JPEG 2000's EBCOT.
Challenge: context modeling introduces decode-order dependencies that may
conflict with massive parallelism. Need a design that preserves tile
independence while adding intra-tile context.

### 7. Tile index table for random access
Tiles are independent but the bitstream is sequential. Adding a tile offset
table in the header enables O(1) random access — critical for region-of-interest
decode in VR/AR and partial frame updates.

### 8. Finish adaptive quantization
Infrastructure is in place (weight maps, aq_strength, SSIM-guided pipeline).
Needs tuning and validation. Expected gain: redistribute bits from smooth
regions to textured regions for better perceptual quality at same bitrate.

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
