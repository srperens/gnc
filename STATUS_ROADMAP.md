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

## Current State

### What works
- End-to-end encode pipeline runs on GPU
- LeGall 5/3 integer wavelet transform (lossless-capable)
- Uniform scalar quantization with configurable QStep
- Per-tile rANS entropy coding (32 interleaved streams per tile)
- CLI: encode, decode, benchmark, sweep commands

### Compression performance (baseline)
| QStep | PSNR   | BPP   |
|-------|--------|-------|
| 4     | 46.2 dB | 2.70 |
| 8     | 40.4 dB | 2.32 |
| 16    | 34.5 dB | 1.93 |

At QStep 8, GNC is roughly comparable to JPEG (1992). JPEG 2000 and H.264
intra-only are meaningfully better (~1.5–1.8 bpp at similar PSNR). This gap
is expected for an early-stage research codec without tuned quantization or
contextual entropy.

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

### 1. Fix the decode hardware limit (immediate)
The decoder's rANS shader uses 4 storage buffers but some hardware only allows
2 per shader stage. Split into two bind groups or restructure to use fewer
buffers simultaneously.

### 2. Switch to CDF 9/7 for lossy mode
LeGall 5/3 is optimal for lossless but suboptimal for lossy. CDF 9/7 (the
JPEG 2000 lossy wavelet) gives significantly better energy compaction and would
close much of the gap to JPEG 2000. This is the single highest-leverage
compression improvement.

### 3. Adaptive quantization per tile
Compute local variance in a GPU pass before quantization. Scale QStep per tile
based on content complexity. The receiver can replicate the same calculation —
no extra signaling needed. Large perceptual quality improvement for free.

### 4. Contextual entropy coding
Currently entropy coding uses flat probability estimates. Modeling symbol
probabilities based on wavelet subband position (low-freq vs high-freq) would
meaningfully improve compression. Still fully parallel — context is computed
per-tile.

### 5. Temporal extension (video)
The biggest compression opportunity. Two GPU-native approaches worth exploring:
- **3D wavelet in time**: apply LeGall 5/3 across N frames in the time
  dimension. No explicit motion estimation. Fully parallel per tile group.
- **GPU block matching**: find best matching tile in previous frame via compute
  shader, encode residuals only. Massively parallel, patent-free.

### 6. Benchmark honestly against JPEG and JPEG 2000
Use standard test images (Kodak dataset, Xiph frames). Measure PSNR, SSIM,
encode time, decode time. Establish a real baseline to track progress against.

### 7. WebAssembly / WebGPU build
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
