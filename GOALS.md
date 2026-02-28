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

**Entropy coder: Rice+ZRL** (256 independent streams/tile, fully GPU-parallel, patent-free).
rANS and Huffman exist in the codebase but are parked — Rice is the default for all paths.

| Quality | PSNR | BPP | Encode | Decode |
|---------|------|-----|--------|--------|
| q=25 | 33.2 dB | 1.71 | 39 fps | 72 fps |
| q=50 | 37.7 dB | 2.37 | 40 fps | 60 fps |
| q=75 | 42.8 dB | 4.01 | 40 fps | 59 fps |
| q=90 | 50.5 dB | 8.90 | 40 fps | 63 fps |

*(Single-frame, 1080p bbb reference, M1 GPU, 2026-02-27)*

**Sequence encode: 31.7 fps** (1080p, q=75, Rice, ki=8, 10 frames I+P+B)

**What works:**
- Full I/P/B frame video pipeline with motion estimation, rate control, GNV1 container
- Rice+ZRL entropy (GPU encode + decode), rANS and Huffman available but parked
- Fused quantize+histogram shader
- 128+ tests, golden-baseline regression, 5 conformance bitstreams
- 33 WGSL compute shaders
- WASM/WebGPU decoder builds (263 KB)

**Key GPU architecture insight:** On M1, shared memory occupancy dominates performance. 16KB shared memory = 2 workgroups/core (full occupancy). Rice uses < 1KB shared → excellent occupancy.

**Known gaps:**
- Sequence encode 31.7 fps → target 60 fps
- Single-frame encode 40 fps → target 60 fps
- 8-bit only (10-bit not implemented)
- 4:4:4 only (4:2:2 not implemented)
- No true lossless with Rice (near-lossless 56 dB at q=100)

## 4. Priorities

**P1: Video encode speed → 60 fps**
- ✓ 30 fps achieved (31.7 fps, 1080p, q=75, Rice, I+P+B)
- Next: multi-frame GPU overlap, fused wavelet+quantize kernel, reduce ME cost
- Target: real-time 1080p60 sequence encode

**P2: Broadcast features**
- 10-bit content support
- 4:2:2 chroma subsampling

**P3: Compression efficiency** (parked)
- rANS and Huffman are implemented but parked — revisit when speed targets are met
- Fix Rice lossless, improve low-quality compression

**P4: Transform exploration**
- DCT, hybrid wavelet/DCT, Haar

**P5: Research/experimental**
- PVQ, learned lifting, non-separable fused wavelet

## 5. Non-Goals

- **Beating AV1/H.265 on compression ratio** — We occupy a different design point: parallel, low-latency, patent-free. We compete on speed and simplicity, not maximum compression.
- **CPU decode path** — GPU-only by design. No software fallback.
- **Backward compatibility** — No legacy bitstreams to support (rule 10).
- **Neural/ML compression** — Extreme complexity for marginal gains. Not worth it for GPU-native design.
- **Maximum single-thread performance** — We scale with parallelism, not clock speed.
