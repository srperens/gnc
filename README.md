# GNC — GPU-Native Codec

Research project exploring what video compression looks like when designed from scratch for GPU parallelism, rather than adapting CPU-era algorithms.

**Rust + wgpu compute shaders (WGSL). Cross-platform: Metal, Vulkan, DX12, WebGPU/WASM. Patent-free.**

## Why

Traditional codecs (H.264, HEVC, AV1) are shaped by decades of CPU constraints — sequential processing, complex prediction modes, intricate entropy coding with state chains. GPUs offer thousands of parallel threads, but these codecs can't exploit them.

GNC asks: if you start from zero with a GPU-first mindset, what do you end up with?

The answer so far: tile-independent processing, fully parallel entropy coding (256 independent streams per tile), and wavelet transforms that map naturally to GPU workgroups. It runs a full I/P/B video pipeline in real time at 1080p on an eight-core integrated GPU.

GNC targets **contribution** — the high-quality, low-latency link between a camera or a production facility and whatever comes next — not distribution to viewers. That choice sets the operating point everything below is measured at, and it is why the comparisons are against H.264 near lossless rather than at streaming bitrates.

## Status

**Working end to end:** I/P/B video pipeline with motion estimation, 8- and 10-bit, 4:4:4 / 4:2:2 / 4:2:0, three interchangeable entropy coders, and bit-exact lossless at `q=100`. Runs on Metal, Vulkan, DX12 and WebGPU.

**Where it stands against H.264** (measured 2026-09-06, `scripts/meas1_vs_h264.py`, 1080p, ki=9, x264 at defaults):

- **Contribution quality: +90.5% BD-rate on PSNR** — about 1.9x the bitrate of x264 for the same luma quality, across three sequences.
- **Colour: GNC spends more of its budget on chroma.** At rate matched to 1%, GNC scores better CIEDE2000 on all three sequences (0.611 vs 0.684 mean on bbb) with fewer pixels past the just-noticeable threshold — while sitting 7.4–8.8 dB behind on luma. That is an allocation difference, and it is not yet known whether it is more than one: the test that would settle it is giving x264 the same allocation (`--chroma-qp-offset`) and re-measuring CIEDE2000 at the same total rate. Until that runs, this row says where the bits went, not that the transform preserves colour better.
- **Lossless: the best wavelet result in the field.** 1.99:1 at `q=100`, beating JPEG 2000 lossless by 10.8% and PNG by 7.8%; behind FFV1 by 27% and x264 `-qp 0` by 43%, both of which predict against the neighbouring pixel rather than across scales.
- **Latency: ~80 ms round trip** intra or P-only, ~240 ms with the B-pyramid, 1080p on an M1
  (MEAS-6). That is the low-latency-HEVC band, **not** the JPEG XS band — JPEG XS codes 1–32
  lines and EBU measures it under one frame. The 256-line tile floor is not reachable today: the
  pipeline processes whole frames, so the practical floor is one full frame whatever the tile
  size. See [`docs/POSITIONING.md`](docs/POSITIONING.md) for where that leaves GNC against the
  incumbents in this segment, and note that the +90.5% figure above is measured against x264,
  which is a sanity anchor rather than a competitor here — JPEG XS, J2K, VC-2 and ProRes are.
- At *distribution* bitrates the gap is much larger. GNC is not built for that operating point.

**Off by default, and why:** the B-frame pyramid (costs 7–31% in rate on camera content and 160 ms in latency), temporal wavelet mode (loses 2–5 dB on high motion), and motion-compensated temporal filtering (measured 1.04–1.14x *worse* than a P-frame chain on every sequence tested).

See [`RESEARCH_LOG.md`](RESEARCH_LOG.md) for every measurement, including the ones that failed — roughly two dozen ideas have been tested and rejected, and they are written up as carefully as the wins.

## Current Results (1080p, bbb reference, M1 GPU)

### Single-frame (Rice+ZRL entropy)

| q | PSNR | BPP | VMAF | levels |
|---|------|-----|------|--------|
| 25 | 35.51 dB | 1.60 | 90.25 | 5 |
| 50 | 40.30 dB | 2.73 | 95.02 | 5 |
| 75 | 44.84 dB | 4.53 | 96.58 | 5 |
| 90 | 50.06 dB | 8.07 | 97.08 | 5 |

*Single-frame, 1080p bbb reference, Rice, 4:4:4. **[BASELINE.md](BASELINE.md) is the single source
for these** — this table was three separate copies from 2026-02-27 and had drifted more than 2 dB.
Throughput columns are deliberately absent: see BASELINE's fps section for why no single "encode
fps" exists.*

### Video sequence

**31.7 fps** (1080p, q=75, keyframe interval 8, I+P+B frames)

> **On the throughput figures above.** Three different quantities have been called "encode fps" in
> this project and they differ by 2.4x — the GPU encode phase, the encoder loop, and end-to-end
> wall clock. The figures here are the encoder loop. They were also measured on a machine that is
> not reliably idle: the same workload has timed 25.2, 31.1 and 37.5 ms across three runs, a 48%
> spread on identical work. **Treat every fps number in this README as indicative to about ±25%.**
> The compression figures (bpp, PSNR, CIEDE2000) are deterministic and carry no such caveat.

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
2. **Wavelet transform** — CDF 9/7 for lossy (q=1–99), LeGall 5/3 for lossless (q=100), 5 decomposition levels at q≥25, 4 below
3. **Adaptive quantization** — Per-block variance analysis on LL subband, geometric mean normalization, 3×3 spatial smoothing
4. **Quantization** — Uniform scalar with perceptual subband weights, dead zone, adaptive QP from AQ weight map. Fused quantize+histogram kernel when CfL is off.
5. **Chroma-from-Luma (CfL)** — Per-tile per-subband least-squares alpha (14-bit), active at q=50–85. Encodes chroma residuals instead of raw coefficients.
6. **Entropy coding** — Rice+ZRL (default): significance map + Golomb-Rice + zero-run-length, 256 independent streams per tile. rANS (32 streams), Huffman (64-symbol), and Bitplane also available but parked.

### Video features

- **I/P/B frames** — motion-compensated prediction with half-pel bilinear interpolation
- **Motion estimation** — hierarchical coarse-to-fine block matching (16x16, ±32px search)
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
gnc benchmark -i input.png -q 75 --rans       # rANS entropy (see Entropy Coders)
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

### Run tests

```bash
cargo test --release    # 148 tests: unit, regression, conformance
```

## Test Material

```bash
cd test_material && bash fetch_test_frames.sh
```

Downloads representative broadcast frames from [Xiph.org](https://media.xiph.org/) (requires ffmpeg and curl).

## Entropy Coders

GNC has four entropy coding backends, all running as GPU compute shaders:

| Coder | Streams/tile | Coding | Speed | Patent risk |
|-------|-------------|--------|-------|-------------|
| **Rice+ZRL** (default) | 256 | Golomb-Rice + zero-run | **1.5–2× faster** | None |
| rANS (`--rans`) | 32 | Range asymmetric numeral systems | Baseline | Possible (MS patent) |
| Huffman (parked) | 256 | 64-symbol + escape | Moderate | None |
| Bitplane (parked) | Per-block | Sign + magnitude bitplanes | Moderate | None |

*No compression column: **the coders have never been measured against each other on one commit.**
The figures that stood here (Rice 4.01 bpp, rANS 4.22 bpp @ q=75) were taken at an operating point
that no longer exists — q=75 has since moved from 42.17 dB / 3.83 bpp to 44.84 dB / 4.53 bpp with
uniform weights and 5 levels — and GP17 (Rice-coded stream-length tables) shrank Rice's headers at
bit-identical output without touching rANS. Quoting 4.53 against 4.22 would compare three changes
at once. Rice is expected to still win, and by more than before, since header overhead scales with
stream count and Rice runs 256 streams to rANS's 32 — but that is a prediction, not a measurement.
[BASELINE.md](BASELINE.md) carries Rice only.*

Rice is the default because it eliminates the sequential state chain that limits rANS. Each of the 256 streams encodes independently — no shared state, no synchronization, minimal shared memory (< 1 KB vs rANS's 16 KB frequency tables). rANS, Huffman, and Bitplane are available but parked — they'll be revisited once speed targets are met.

## Quality Spectrum

Smooth, monotonic quality scaling from lossless to extreme compression:

```
q=100  Lossless     — bit-exact round-trip (LeGall 5/3 integer wavelet)
q=90   High quality — near-transparent
q=75   Production   — good general-purpose quality
q=50   Balanced     — CfL + adaptive quantization
q=25   Compressed   — broadcast-suitable
q=5    Extreme      — preview/thumbnail
```

*Deliberately without dB figures. This block used to carry its own set (q=75 → 42 dB, q=50 → 37,
q=25 → 33) which was a third copy of the 2026-02-27 numbers and had drifted 2–3 dB from the table
above. [BASELINE.md](BASELINE.md) is the single source; the Current Results table quotes it, and
nothing else in this file should.*

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
├── shaders/            WGSL compute shaders
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
