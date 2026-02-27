# GPU-Native Broadcast Codec — Original Project Brief

> **Historical document** (2026-02-22). This was the initial project brief. Current AI instructions are in CLAUDE.md. Current project status is in README.md and STATUS_ROADMAP.md. All Phase 1-3 items below have been completed.

## Vision

Design and iterate on a patent-free, GPU-native video codec that exploits massive parallelism to achieve broadcast-quality compression. This is both a practical project (alternative to JPEG XS for ST 2110) and a research exploration (what new compression techniques does GPU-native design unlock?).

The key hypothesis: traditional codecs are shaped by CPU constraints. By designing from scratch for thousands of GPU threads, we can achieve competitive compression with simpler per-thread logic and zero sequential dependencies.

## Language & Platform

- **Rust** with `wgpu` for compute shaders (cross-platform: Vulkan, Metal, DX12)
- WGSL for shader code
- No patent-encumbered techniques — everything must be royalty-free
- Target: real-time encoding of 1080p50/60 and 4K broadcast content

## Project Structure

```
gpu-codec/
├── Cargo.toml
├── README.md
├── RESEARCH_LOG.md          # Track every experiment: what was tried, results, insights
├── src/
│   ├── lib.rs
│   ├── encoder/
│   │   ├── mod.rs
│   │   ├── pipeline.rs      # Composable encoding pipeline
│   │   ├── color.rs         # Color space conversion (YCoCg-R, others)
│   │   ├── transform.rs     # Transform stage (wavelets, DCT, learned, hybrid)
│   │   ├── quantize.rs      # Quantization strategies
│   │   └── entropy.rs       # Entropy coding (rANS, tANS, etc.)
│   ├── decoder/
│   │   ├── mod.rs
│   │   └── pipeline.rs
│   ├── shaders/
│   │   ├── color_convert.wgsl
│   │   ├── transform.wgsl
│   │   ├── quantize.wgsl
│   │   └── entropy.wgsl
│   ├── bench/
│   │   ├── mod.rs
│   │   ├── quality.rs       # PSNR, SSIM, VMAF measurement
│   │   ├── throughput.rs    # FPS, latency measurement
│   │   └── compare.rs       # Compare variants side by side
│   └── experiments/
│       └── mod.rs           # Switchable experiment configurations
├── test_material/            # Broadcast test content (not in git)
│   └── README.md            # Instructions for obtaining test material
└── results/                  # Benchmark results, graphs
    └── .gitkeep
```

## Core Design Principles

1. **Tile-independent processing** — Each tile must be encodable/decodable independently. No cross-tile dependencies. This gives us parallelism AND broadcast-friendly random access.

2. **Modular pipeline** — The codec is a composable pipeline of stages. Each stage can be swapped independently for experimentation:
   - Color space conversion
   - Transform (spatial → frequency/sparse)
   - Quantization
   - Entropy coding

3. **GPU-first** — Everything runs in compute shaders by default. CPU fallback only for validation. Don't adapt CPU algorithms to GPU — design for GPU from the start.

4. **Measure everything** — Every experiment must produce comparable metrics: PSNR, SSIM, bitrate (bpp), encode time, decode time, GPU memory usage.

## Research Directions to Explore

### Phase 1: Baseline
- Implement the simplest possible GPU codec: YCoCg-R → LeGall 5/3 wavelet → uniform quantization → simple entropy coding
- Get the full pipeline working end-to-end with benchmarks
- This is the baseline everything else is compared against

### Phase 2: Component Exploration
Try alternatives for each pipeline stage independently:

**Transforms:**
- LeGall 5/3 wavelet (baseline)
- Haar wavelet (simplest, most parallel)
- DCT (8x8, 16x16) — how does it compare when GPU-native?
- Hybrid: wavelet for low frequencies, DCT for high
- Small learned transforms (train offline, deploy as shader constants)
- Sparse pursuit: let GPU threads compete to find sparse representations

**Entropy Coding:**
- No entropy coding (just quantized coefficients) — what's the baseline?
- Per-tile rANS — known to be GPU-friendly
- tANS with precomputed tables
- Simple Huffman per-tile
- Golomb-Rice for coefficient magnitudes
- Hybrid: different coding for different subbands

**Quantization:**
- Uniform scalar
- Dead-zone quantization
- Perceptually weighted (frequency-dependent)
- Content-adaptive (GPU analyzes tile complexity, adjusts QP)
- Rate-distortion optimized (GPU evaluates multiple QP per tile in parallel)

**Color Space:**
- YCoCg-R (baseline, lossless for integer)
- YCbCr variants
- ICtCp (perceptually uniform, used in HDR)
- Direct RGB with decorrelation transform

### Phase 3: Novel Ideas
These are speculative — the point is to try them:

- **Massively parallel motion search**: For inter-frame, use thousands of GPU threads to search reference frames. Can brute-force search beat sophisticated CPU heuristics?
- **Per-pixel adaptive transform**: Instead of fixed block/tile transforms, let GPU choose transform per region
- **Compression as parallel optimization**: Frame each tile as an optimization problem, let GPU threads run gradient descent or similar
- **Neural micro-transforms**: Very small (3x3, 5x5) learned kernels that replace fixed transforms
- **Multi-resolution parallel coding**: Encode multiple resolutions simultaneously, use GPU to find optimal bit allocation across scales

## Iteration Protocol

For each experiment:

1. **Describe** the hypothesis in RESEARCH_LOG.md before implementing
2. **Implement** the variant as a swappable pipeline component
3. **Benchmark** against baseline and all previous best results:
   - Test material: at least 3 different broadcast content types (talking head, sports/motion, graphics/text)
   - Metrics: PSNR, SSIM at target bitrates (50 Mbps, 100 Mbps, 200 Mbps for 1080p50)
   - Throughput: frames per second encode and decode
   - Latency: single-frame encode/decode time
4. **Analyze** — why did it work or not? What does this tell us about the design space?
5. **Log** everything in RESEARCH_LOG.md with dates
6. **Decide** next experiment based on accumulated insights

## Benchmark Requirements

- Use standard test sequences or derive from openly available broadcast content
- Always compare: raw throughput, quality at matched bitrate, bitrate at matched quality
- Generate comparison images (crop interesting regions, show difference maps)
- Track results in a structured format (CSV or JSON) for later analysis

## Constraints

- No patented techniques (check before implementing — H.264/5/6 patent pools, MPEG-LA, etc.)
- Must handle 10-bit content (broadcast standard)
- Must support 4:2:2 chroma subsampling (broadcast standard)
- Target: visually lossless at ≤ 200 Mbps for 1080p50
- Stretch goal: visually lossless at ≤ 500 Mbps for 2160p50

## Getting Started

```bash
cargo init gpu-codec
cd gpu-codec
# Add dependencies: wgpu, image, clap, serde, csv
# Implement Phase 1 baseline first
# Run: cargo run --release -- encode --input test.raw --output test.gpuc --benchmark
```

Start with Phase 1. Get a working end-to-end pipeline with benchmarks before exploring alternatives. The baseline doesn't need to be good — it needs to be correct and measurable.
