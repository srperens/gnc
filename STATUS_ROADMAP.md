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

## Current State (2026-02-27)

### What works

**Image compression:**
- End-to-end GPU encode/decode pipeline
- CDF 9/7 lossy wavelet (q=1-99), LeGall 5/3 lossless wavelet (q=100), 3-4 levels
- Adaptive quantization with perceptual subband weights, dead zone, SSIM-guided spatial weighting
- Chroma-from-Luma (CfL) prediction at q=50-85 (14-bit i16 precision)
- Three entropy coders: Rice+ZRL (256 streams, fastest), rANS (32 streams, default), Bitplane
- Per-subband entropy coding (separate frequency tables per wavelet level)
- Smooth monotonic quality curve q=1..100 (validated)
- Bit-exact lossless round-trip at q=100

**Video:**
- I/P/B frame types with motion-compensated prediction
- Half-pel bilinear motion estimation, hierarchical coarse-to-fine block matching (±64px)
- B-frame scheduling (I B B P B B P pattern), bidirectional prediction
- CBR and VBR rate control with R-Q model and VBV buffer
- GNV1 sequence container with frame index table and keyframe seeking

**Infrastructure:**
- GP11 bitstream format (backward-compatible with GP10/GPC9/GPC8)
- Per-tile CRC-32 checksums, corrupt tile detection and recovery
- 5 conformance test bitstreams
- Golden-baseline regression tests at q=25/50/75/90
- BD-rate computation, multi-codec comparison (GNC vs JPEG vs JPEG 2000)
- WebGPU/WASM build (263 KB), browser decode demo
- Criterion benchmarks for encode, decode, roundtrip

### Compression performance (bbb_1080p 1920x1080)

**Rice+ZRL entropy coder (fastest):**

| Quality | PSNR | BPP | Encode | Decode |
|---------|------|-----|--------|--------|
| q=25 | 33.2 dB | 1.73 | 40 fps | 70 fps |
| q=50 | 37.5 dB | 2.42 | 42 fps | 61 fps |
| q=75 | 42.1 dB | 4.09 | 40 fps | 61 fps |
| q=90 | 49.2 dB | 8.96 | 41 fps | 66 fps |
| q=100 | inf | 12.80 | — | — |

**rANS entropy coder (default, best compression at low bitrate):**

| Quality | PSNR | BPP | Encode | Decode |
|---------|------|-----|--------|--------|
| q=25 | 33.2 dB | 1.29 | 29 fps | 34 fps |
| q=50 | 37.7 dB | 2.30 | 30 fps | 36 fps |
| q=75 | 42.8 dB | 4.22 | 29 fps | 34 fps |
| q=90 | 51.0 dB | 9.65 | 30 fps | 34 fps |

**Key finding:** Rice+ZRL beats rANS in bpp at q>=75 while being 1.5-2x faster.
At q=25, Rice is +34% overhead but dramatically faster. Rice eliminates the sequential
state chain that limits rANS — all 256 streams encode independently.

### Multi-codec comparison

**BD-rate vs JPEG (libjpeg-turbo):** ~-1% on animation (GNC wins), ~+50% on sports.
**BD-rate vs JPEG 2000 (OpenJPEG):** ~+50% (gap closed from ~+60% baseline).

### 4K performance

4K is 34-71% *faster* per pixel than 1080p (better GPU occupancy):
- 4K encode: 101ms (82 MP/s) vs 1080p: 34ms (61 MP/s)
- 4K decode: 70ms (118 MP/s) vs 1080p: 30ms (69 MP/s)

### Video sequence performance

- Mixed I+P+B: 6.5 fps at 1080p (up from 0.8 fps)
- All-I frames: 9.8 fps
- P-frame savings: 33% on animation, 0% on camera pan (motion estimation limitation)
- B-frames: ~20% smaller than P-frames

---

## Honest Assessment

### Where GNC excels

1. **Decode speed**: 61-70 fps at 1080p (Rice) on M1 GPU. JPEG 2000 is typically 2-5 fps on CPU.
2. **Resolution scaling**: 4K = 4x tiles = 4x parallel workgroups. 4K per-pixel throughput
   is actually *faster* than 1080p.
3. **Simplicity**: ~15K lines of Rust + ~3K lines of WGSL. The entire codec is comprehensible
   by one person. AV1's reference encoder is 400K+ lines.
4. **Patent-free**: No MPEG-LA, no VVC patent pool, no licensing uncertainty.
5. **Tile independence**: Partial decode, region-of-interest, error resilience — all for free.
6. **Cross-platform**: Single shader source runs on Metal, Vulkan, DX12, and WebGPU.

### Remaining gaps

1. **Compression vs JPEG 2000**: ~1.4x bpp gap. Structural — EBCOT's context modeling and
   trellis quantization are inherently more efficient than per-coefficient Rice/rANS.
2. **Lossless**: 12.8 bpp vs JPEG 2000's ~3.5 bpp. Needs bitplane/CABAC entropy to close.
3. **Camera pan content**: Motion estimation with 16x16 blocks and ±64px search loses to
   whole-frame motion in traditional codecs. Variable block sizes would help.
4. **Sequence speed**: 6.5 fps I+P+B, target is 10+ fps. ME readback is the bottleneck.
5. **60fps encode**: 40 fps with Rice, 29 fps with rANS. GPU ALU throughput is the limit.

---

## Roadmap (What's Next)

### Near-term: Make Rice the default entropy coder

Rice+ZRL matches or beats rANS at q>=50 while being 1.5-2x faster. The remaining +34%
gap at q=25 could be closed with:
- Adaptive k_zrl per subband (instead of per-tile)
- Context-adaptive Rice parameter switching
- Better zero-run modeling for high-frequency subbands

### Medium-term: Close the lossless gap

Lossless 12.8 bpp is 3.7x worse than JPEG 2000. This requires fundamentally different
entropy coding (bitplane refinement or context-adaptive binary coding) rather than
incremental Rice/rANS improvements.

### Medium-term: Improve motion estimation

Variable block sizes (8x8 alongside 16x16) with RD-based split decisions would improve
P-frame compression on high-motion content. GPU-native approach: evaluate all candidates
in parallel, pick lowest cost.

### Long-term: Real-time 1080p60

Current bottleneck is GPU ALU throughput in the entropy coding pass. Possible paths:
- Fused wavelet+quantize shader (eliminate intermediate buffer read/write)
- Wider Rice streams (512 per tile) for higher parallelism
- Multi-frame pipelining (overlap frame N+1 transform with frame N entropy)

---

## Target Use Cases

GNC's architecture fits best where latency, GPU availability, and simplicity matter
more than maximum compression:

- **Real-time video** — live streaming, video conferencing, game capture
- **Professional intermediate codec** — GPU-accelerated, low-latency proxy format
- **VR/XR** — where latency and GPU availability matter more than compression ratio
- **High-resolution pipelines** — 4K/8K where sequential codecs bottleneck
- **Web delivery** — WebGPU/WASM decode in browser with no plugins

GNC is not trying to beat AV1 on compression efficiency. It occupies a different
point in the design space: simpler, parallel, lower latency, patent-free.
