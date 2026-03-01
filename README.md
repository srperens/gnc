# GNC — GPU-Native Codec

Research project exploring what video compression looks like when designed from scratch for GPU parallelism, rather than adapting CPU-era algorithms.

**Rust + wgpu compute shaders (WGSL). Cross-platform: Metal, Vulkan, DX12, WebGPU/WASM. Patent-free.**

## Why

Traditional codecs (H.264, HEVC, AV1) are shaped by decades of CPU constraints — sequential processing, complex prediction modes, intricate entropy coding with state chains. GPUs offer thousands of parallel threads, but these codecs can't exploit them.

GNC asks: if you start from zero with a GPU-first mindset, what do you end up with?

The answer so far: tile-independent processing, fully parallel entropy coding (256 independent streams per tile), and wavelet transforms that map naturally to GPU workgroups. The result competes with JPEG on compression while encoding/decoding at 40–70 fps at 1080p — including a full I/P/B video pipeline.

## Current Results (1080p, bbb reference, M1 GPU)

### Single-frame (Rice+ZRL entropy)

| Quality | PSNR | BPP | Encode | Decode |
|---------|------|-----|--------|--------|
| q=25 (high compression) | 33.2 dB | 1.71 | 39 fps | 72 fps |
| q=50 (balanced) | 37.7 dB | 2.37 | 40 fps | 60 fps |
| q=75 (good quality) | 42.8 dB | 4.01 | 40 fps | 59 fps |
| q=90 (high quality) | 50.5 dB | 8.90 | 40 fps | 63 fps |

### Video sequence

**31.7 fps** (1080p, q=75, keyframe interval 8, I+P+B frames)

## Architecture

Everything runs as wgpu compute shaders. The pipeline:

```
RGB → YCoCg-R → Wavelet → Quantize → Entropy Code → Bitstream
         ↕          ↕          ↕            ↕
     (lossless   (CDF 9/7   (adaptive,   (Rice+ZRL:
      integer)   or 5/3)     CfL, AQ)    256 streams)
```

Each tile (256x256) is fully independent — no cross-tile dependencies. This gives parallelism, random access, and error resilience for free. See [`docs/PIPELINE.md`](docs/PIPELINE.md) for a detailed stage-by-stage breakdown.

### Pipeline stages

1. **Color space** — YCoCg-R via lifting (integer-exact, lossless-capable)
2. **Wavelet transform** — CDF 9/7 for lossy (q=1–99), LeGall 5/3 for lossless (q=100), 3–4 decomposition levels
3. **Adaptive quantization** — Per-block variance analysis on LL subband, geometric mean normalization, 3×3 spatial smoothing
4. **Quantization** — Uniform scalar with perceptual subband weights, dead zone, adaptive QP from AQ weight map. Fused quantize+histogram kernel when CfL is off.
5. **Chroma-from-Luma (CfL)** — Per-tile per-subband least-squares alpha (14-bit), active at q=50–85. Encodes chroma residuals instead of raw coefficients.
6. **Entropy coding** — Rice+ZRL (default): significance map + Golomb-Rice + zero-run-length, 256 independent streams per tile. rANS (32 streams) and Huffman (64-symbol) available but parked.

### Video features

- **I/P/B frames** — motion-compensated prediction with half-pel bilinear interpolation
- **Motion estimation** — hierarchical coarse-to-fine block matching (16x16, ±64px search)
- **Rate control** — CBR and VBR modes with R-Q model
- **Container** — GNV1 format with frame index table, keyframe seeking
- **Error resilience** — per-tile CRC-32 checksums, corrupt tile detection and recovery

## Build & Run

```bash
cargo build --release
```

### Encode / decode a single image

```bash
gnc encode -i input.png -o output.gpuc -q 75
gnc decode -i output.gpuc -o output.png
```

### Benchmark

```bash
gnc benchmark -i input.png -q 75              # Rice+ZRL (default)
gnc benchmark -i input.png -q 75 --rans       # rANS entropy (better compression)
```

### Rate-distortion curve

```bash
gnc rd-curve -i input.png                     # sweep q=10..100, output CSV
gnc rd-curve -i input.png --compare-codecs    # also compare vs JPEG, JPEG 2000
```

### Encode / decode video sequence

```bash
gnc encode-sequence -i "frames/%04d.png" -o video.gnv -q 75 --keyframe-interval 8
gnc decode-sequence -i video.gnv -o "output/%04d.png"
gnc decode-sequence -i video.gnv -o "output/%04d.png" --seek 5.0  # seek to 5s
```

### Rate-controlled encoding

```bash
gnc encode-sequence -i "frames/%04d.png" -o video.gnv --bitrate 10M --rate-mode vbr
```

### Run tests

```bash
cargo test --release    # 141 tests: unit, regression, conformance
```

## Test Material

```bash
cd test_material && bash fetch_test_frames.sh
```

Downloads representative broadcast frames from [Xiph.org](https://media.xiph.org/) (requires ffmpeg and curl).

## Entropy Coders

GNC has three entropy coding backends, all running as GPU compute shaders:

| Coder | Streams/tile | Compression | Speed | Patent risk |
|-------|-------------|-------------|-------|-------------|
| **Rice+ZRL** (default) | 256 | 4.01 bpp @ q=75 | **1.5–2× faster** | None |
| rANS (`--rans`) | 32 | 4.22 bpp @ q=75 | Baseline | Possible (MS patent) |
| Huffman | Per-tile | 64-symbol + escape | Moderate | None |

Rice is the default because it eliminates the sequential state chain that limits rANS. Each of the 256 streams encodes independently — no shared state, no synchronization, minimal shared memory (< 1 KB vs rANS's 16 KB frequency tables). rANS and Huffman are available but parked — they'll be revisited once speed targets are met.

## Quality Spectrum

Smooth, monotonic quality scaling from lossless to extreme compression:

```
q=100  Lossless     — bit-exact round-trip (LeGall 5/3 integer wavelet)
q=90   High quality — 49 dB PSNR, near-transparent
q=75   Production   — 42 dB PSNR, good general-purpose quality
q=50   Balanced     — 37 dB PSNR, CfL + adaptive quantization
q=25   Compressed   — 33 dB PSNR, broadcast-suitable
q=5    Extreme      — 27 dB PSNR, 0.47 bpp (preview/thumbnail)
```

## WebGPU / WASM

The full decoder compiles to WebAssembly (263 KB) and runs in browsers via WebGPU:

```bash
wasm-pack build --target web --release
```

Browser demo in `examples/web/index.html`.

## Project Structure

```
src/
├── lib.rs              Core types, quality_preset(), codec config
├── main.rs             CLI (encode, decode, benchmark, rd-curve, ...)
├── format.rs           Bitstream serialization (GP11 frame, GNV1 sequence)
├── encoder/
│   ├── pipeline.rs     Encoder orchestration
│   ├── sequence.rs     Video sequence, B-frames, rate control
│   ├── rice.rs         CPU Rice encoder/decoder (reference)
│   ├── rice_gpu.rs     GPU Rice encoder/decoder
│   ├── rans.rs         CPU rANS encoder/decoder
│   ├── rans_gpu_encode.rs  GPU rANS encoder
│   ├── huffman_gpu.rs  GPU Huffman encoder
│   ├── motion.rs       Motion estimation and compensation
│   ├── cfl.rs          Chroma-from-Luma prediction
│   ├── adaptive.rs     Adaptive quantization
│   ├── fused_block.rs  Block DCT-8×8 mega-kernel
│   └── ...
├── decoder/
│   ├── pipeline.rs     Decoder orchestration
│   ├── frame_data.rs   Frame data upload
│   └── gpu_work.rs     GPU dispatch
├── shaders/            33 WGSL compute shaders
│   ├── rice_encode.wgsl, rice_decode.wgsl
│   ├── rans_encode.wgsl, rans_decode.wgsl
│   ├── transform_97.wgsl, transform_53.wgsl
│   ├── block_match.wgsl, motion_compensate.wgsl
│   └── ...
├── bench/              BD-rate, codec comparison, quality metrics
└── experiments/        Experimental features

tests/
├── quality_regression.rs   Golden-baseline regression (q=25/50/75/90)
├── conformance.rs          5 conformance bitstreams + corruption tests
└── golden_baselines.toml   Reference PSNR/SSIM/bpp values
```

## Documentation

- [`docs/PIPELINE.md`](docs/PIPELINE.md) — Detailed encode pipeline description
- [`docs/BITSTREAM_SPEC.md`](docs/BITSTREAM_SPEC.md) — Complete bitstream format specification (GP11 frame, GNV1 sequence)
- [`RESEARCH_LOG.md`](RESEARCH_LOG.md) — Experiment log with hypotheses, results, analysis

## License

All code is patent-free. No H.264/5/6 patent pool or MPEG-LA encumbered techniques. All dependencies are open source.
