# GNC — Goals, Rules & Priorities

## 1. What GNC Is

GNC is a patent-free **video codec** designed from scratch for GPU parallelism. Everything runs as wgpu compute shaders (WGSL) — cross-platform on Metal, Vulkan, DX12, and WebGPU/WASM. The core idea: tile-independent processing with thousands of parallel threads instead of sequential CPU-era algorithms.

## 2. Design Rules

1. **Patent-free** — No patented techniques, period. If it's patented, we don't use it.
2. **GPU-first** — Everything runs in compute shaders. No CPU fallback paths. CPU reference implementations only for validation/testing.
3. **Massive parallelism via tile independence** — No cross-tile dependencies at any stage. Each tile encodes/decodes in isolation. This is what enables thousands of parallel GPU threads.
4. **Cross-platform** — Must work on Metal, Vulkan, DX12, and WebGPU (WASM). No backend-specific features. WGSL shaders are the single source.
5. **No f64 in shaders** — Target hardware (M1, mobile GPUs) has no hardware double precision.
6. **Open source only** — All dependencies must be open source.
7. **English only** — All code, comments, docs, and commit messages in English.
8. **Measure everything** — Every change benchmarked: PSNR, SSIM, bpp, encode/decode FPS. Compare against baseline and previous best. Optionally compare against relevant codecs (H.264, H.265, AV1, MJPEG, JPEG XS, ProRes) for context.
9. **No code duplication** — Extract shared logic. Code must pass `cargo fmt` and `cargo clippy` with zero warnings.
10. **No legacy** — Nobody runs GNC in production. We can break the bitstream format, change the container, rename fields, restructure anything. No backward compatibility constraints.
11. **Video codec first** — GNC is a video codec, not an image codec. Sequence encode/decode performance is the primary metric. Single-frame performance only matters as a component of video throughput.

## 3. Current State

**Current results (1080p, bbb reference frame, Rice+ZRL entropy):**

| Quality | PSNR | BPP | Encode | Decode |
|---------|------|-----|--------|--------|
| q=25 | 33.2 dB | 1.73 | 40 fps | 70 fps |
| q=50 | 37.5 dB | 2.42 | 42 fps | 61 fps |
| q=75 | 42.1 dB | 4.09 | 40 fps | 61 fps |
| q=90 | 49.2 dB | 8.96 | 41 fps | 66 fps |
| q=100 | lossless | 12.8 | — | — |

**What works:**
- Full I/P/B frame video pipeline with motion estimation, rate control, GNV1 container
- 3 entropy backends (Rice+ZRL, rANS, Bitplane), all GPU compute shaders
- Quality spectrum from lossless to extreme compression (q=1-100)
- 112 tests, golden-baseline regression, 5 conformance bitstreams
- WASM/WebGPU decoder builds (263 KB)

**Known gaps:**
- Compression ~1.4x behind JPEG 2000 (structural — context modeling gap)
- Lossless: 12.8 bpp vs JPEG 2000's ~3.5 bpp
- 8-bit only (10-bit not implemented)
- 4:4:4 only (4:2:2 not implemented)
- Sequence encode 6.5 fps (target: 30+ fps)
- Single-frame encode 40 fps (target: 60 fps)

## 4. Priorities

**P1: Video encode speed**
- Sequence encode from 6.5 fps to 30+ fps (this is the real video codec metric)
- Motion estimation readback is the bottleneck — needs GPU-side pipelining
- Fused GPU kernels (wavelet+quantize, quantize+histogram)
- Target: real-time 1080p30 sequence encode

**P2: Compression efficiency**
- Context-adaptive entropy (15-25% bpp reduction)
- Improve lossless (12.8 bpp to <5 bpp)
- Fix remaining CfL/AQ edge cases

**P3: Broadcast features**
- 10-bit content support
- 4:2:2 chroma subsampling

**P4: Transform exploration**
- DCT, hybrid wavelet/DCT, Haar

**P5: Research/experimental**
- PVQ, learned lifting, Huffman entropy, non-separable fused wavelet

## 5. Non-Goals

- **Beating AV1/H.265 on compression ratio** — We occupy a different design point: parallel, low-latency, patent-free. We compete on speed and simplicity, not maximum compression.
- **CPU decode path** — GPU-only by design. No software fallback.
- **Backward compatibility** — No legacy bitstreams to support (rule 10).
- **Neural/ML compression** — Extreme complexity for marginal gains. Not worth it for GPU-native design.
- **Maximum single-thread performance** — We scale with parallelism, not clock speed.
