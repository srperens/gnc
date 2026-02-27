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

Two entropy coders with different tradeoffs. Rice+ZRL is faster; rANS compresses better at low quality.

**Rice+ZRL — fast path (256 independent streams/tile):**

| Quality | PSNR | BPP | Encode | Decode |
|---------|------|-----|--------|--------|
| q=25 | 33.2 dB | 1.71 | 39 fps | 72 fps |
| q=50 | 37.7 dB | 2.37 | 40 fps | 60 fps |
| q=75 | 42.8 dB | 4.01 | 40 fps | 59 fps |
| q=90 | 50.5 dB | 8.90 | 40 fps | 63 fps |

**rANS — compression path (32 interleaved streams/tile):**

| Quality | PSNR | BPP | Encode | Decode |
|---------|------|-----|--------|--------|
| q=25 | 33.2 dB | 1.29 | 31 fps | 42 fps |
| q=50 | 37.7 dB | 2.30 | 30 fps | 36 fps |
| q=75 | 42.8 dB | 4.22 | 29 fps | 34 fps |
| q=90 | 51.0 dB | 9.65 | 31 fps | 33 fps |
| q=100 | lossless | 13.06 | 31 fps | 32 fps |

*(All: 1080p bbb reference frame, M1 GPU, 2026-02-27)*

**What works:**
- Full I/P/B frame video pipeline with motion estimation, rate control, GNV1 container
- 3 entropy backends (Rice+ZRL, rANS, Bitplane), all GPU compute shaders
- Fused quantize+histogram shader, fused rANS normalize+encode shader
- True lossless encode/decode (rANS, q=100, bit-exact roundtrip)
- 114+ tests, golden-baseline regression, 5 conformance bitstreams
- 30 WGSL compute shaders (6,348 lines)
- WASM/WebGPU decoder builds (263 KB)

**Patent risk note:** Microsoft holds patent US11234023B2 on rANS modifications. Our rANS implementation has potential exposure. Rice and Huffman are patent-free — strategic reason to migrate away from rANS as the primary entropy coder.

**Key GPU architecture insight:** On M1, shared memory occupancy dominates performance. 16KB shared memory = 2 workgroups/core (full occupancy). Exceeding 16KB halves occupancy → ~20% regression. This is why Rice (< 1KB shared) and Huffman (8KB shared) are fast, while rANS (16KB+) is slower. ALU cost (e.g. integer multiply vs division) is negligible by comparison.

**Known gaps:**
- Sequence encode 18.1 fps Rice (target: 30+ fps) — ME ±32 (P) / ±16 (B), temporal MV prediction, GPU Rice in P/B pipeline, parallelized bidir half-pel
- Single-frame encode 40 fps Rice, 30 fps rANS (target: 60 fps)
- Rice at q=25 is +33% bpp vs rANS (per-subband k_zrl implemented, gap is structural)
- Rice q=100 is near-lossless (56 dB) not bit-exact — only rANS supports true lossless
- Lossless: 13.06 bpp vs JPEG 2000's ~3.5 bpp (structural context-modeling gap)
- 8-bit only (10-bit not implemented)
- 4:4:4 only (4:2:2 not implemented)

## 4. Priorities

**P1: Video encode speed**
- Sequence encode from 6.5 fps to 30+ fps (this is the real video codec metric)
- Multi-frame GPU overlap — pipeline frame N+1 while entropy-coding frame N
- Motion estimation readback elimination — keep MVs on GPU, serialize from GPU buffers
- Fused wavelet+quantize kernel (quantize+histogram already fused)
- Target: real-time 1080p30 sequence encode

**P2: Compression efficiency**
- **Canonical Huffman entropy** — next planned implementation. Same 256-stream architecture as Rice but with distribution-adaptive codewords instead of fixed Golomb-Rice. Expected: 1-5% overhead vs rANS (vs Rice's 3-7%). See `docs/HUFFMAN_PLAN.md`.
- ~~Adaptive Rice k parameter per subband~~ ✓ Done (1-2% bpp gain, gap is structural not parametric)
- Fix Rice lossless — currently near-lossless (56 dB), should be bit-exact like rANS
- Improve lossless (13 bpp to <5 bpp — context modeling is the structural gap)
- Improve low-quality compression (q<25 range)

**P3: Broadcast features**
- 10-bit content support
- 4:2:2 chroma subsampling

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
