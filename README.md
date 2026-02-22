# GNC — GPU-Native Codec

Research project exploring what video compression looks like when designed from scratch for GPU parallelism, rather than adapting CPU-era algorithms.

## Why

Traditional codecs (H.264, HEVC, AV1) are shaped by decades of CPU constraints — sequential processing, complex prediction modes, intricate entropy coding. GPUs offer thousands of parallel threads but these codecs can't fully exploit them.

This project asks: if you start from zero with a GPU-first mindset, what do you end up with? Can brute-force parallelism and simpler per-thread logic compete with sophisticated sequential algorithms?

## How

Everything runs as wgpu compute shaders (WGSL), targeting all backends — Metal, Vulkan, DX12, and WebGPU (WASM).

The codec is a modular pipeline of swappable stages:

1. **Color space conversion** — RGB to YCoCg-R (lossless, integer-only)
2. **Wavelet transform** — LeGall 5/3 lifting scheme, multi-level, tile-independent
3. **Quantization** — Uniform scalar with optional dead zone
4. **Entropy coding** — Per-tile rANS

Each tile is fully independent — no cross-tile dependencies. This gives parallelism and random access for free.

## Status

Early research. The baseline pipeline works end-to-end with benchmarks.

| QStep | PSNR | BPP (3-level wavelet + rANS) |
|---|---|---|
| 4 | 46.2 dB | 2.70 |
| 8 | 40.4 dB | 2.32 |
| 16 | 34.5 dB | 1.93 |

## Build & Run

```bash
cargo build --release
cargo run --release -- encode -i input.png -o output.gpuc
cargo run --release -- decode -i output.gpuc -o output.png
cargo run --release -- benchmark -i input.png
cargo run --release -- sweep -i input.png
```

## Test Material

```bash
cd test_material && bash fetch_test_frames.sh
```

Downloads representative broadcast frames from [Xiph.org](https://media.xiph.org/) (requires ffmpeg and curl).

## License

All code is patent-free. No H.264/5/6 patent pool or MPEG-LA encumbered techniques. All dependencies are open source.
