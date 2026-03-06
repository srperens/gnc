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

## 4. Goals

GNC should become a **good, robust codec** — not optimized along a single axis. We iterate across multiple dimensions simultaneously, looking for combinations of techniques that work well together. No single property is a hard blocker for the others.

**Target properties (all of these, no strict order):**

| Property | Current | Target |
|----------|---------|--------|
| Encode speed | 31.7 fps (seq, 1080p q=75) | 60 fps |
| Compression | competitive vs MJPEG/JPEG XS | competitive with H.264 at matched bitrate |
| Quality range | q=1–100 functional | smooth, predictable quality curve |
| Robustness | basic test coverage | no artifacts, stable across q and content |
| Bitstream | GNV1/GNV2 defined | well-specified, documented |

VC-2 (Dirac) demonstrates that a patent-free wavelet codec can reach H.264-class compression. That is our compression reference point — not just intra codecs.

**How we iterate:**

- Pick the next backlog item based on what provides the most overall value right now — not what happens to be listed as P1
- Rotate between compression, speed, and robustness — progress in one area does not unlock another
- Always measure on ≥3 sequences and multiple q levels — a codec that is only good over a narrow quality band is not a good codec
- Technology choices are driven by: patent freedom, GPU parallelism, measurable improvement

## 5. Design Philosophy

**Correctness over speed.** A codec with subtle bugs is worthless. Verify every change end-to-end. A fast encoder that produces subtly wrong output is not a working encoder.

**Measure before assuming.** Numbers that look too good probably are. Numbers that look unchanged might mean the code isn't running. Run twice. Test on diverse content. Compare against baseline.

**Simplicity has value.** A complex change for 0.3 dB gain is probably not worth the maintenance cost. When two approaches produce similar results, prefer the simpler one. Clever code that nobody understands will break.

**Low-latency by design.** Tile independence is not just about parallelism — it also enables low-latency decode and random seek without full GOP decode. Preserve this property in every pipeline stage.

**Broad content coverage.** A codec that is only good on one type of content is not a good codec. Always validate on high-motion (crowd_run), low-motion (rush_hour), and mixed (stockholm) sequences. Synthetic tests are for correctness, not quality measurement.

**Challenge your own work.** After implementing something, actively try to prove it is wrong before calling it done. Reproduce results before celebrating. If the same bug resurfaces twice — stop and diagnose the root cause properly.

## 6. Non-Goals

- **Beating AV1/H.265 on compression ratio** — We occupy a different design point: parallel, low-latency, patent-free. We compete on speed and simplicity, not maximum compression.
- **CPU decode path** — GPU-only by design. No software fallback.
- **Backward compatibility** — No legacy bitstreams to support (rule 10).
- **Neural/ML compression** — Extreme complexity for marginal gains. Not worth it for GPU-native design.
- **Maximum single-thread performance** — We scale with parallelism, not clock speed.
