# CLAUDE.md — GPU-Native Video Codec Research

## Project Overview

Research project exploring GPU-native video compression. The core question: what does a codec look like when designed from scratch for massive GPU parallelism instead of adapted from CPU thinking? Rust + wgpu compute shaders (WGSL). Cross-platform via wgpu (Metal, Vulkan, DX12, WebGPU/WASM).

## Build & Run

```bash
cd gpu-codec
cargo build --release
cargo run --release -- encode --input test.raw --output test.gpuc --benchmark
cargo test
```

## Key Constraints

- **No patented techniques** — no H.264/5/6 patent pool or MPEG-LA encumbered methods. Check before implementing.
- **GPU-first** — everything runs in compute shaders. Don't port CPU algorithms; design for thousands of parallel threads from the start.
- **Tile-independent** — no cross-tile dependencies. Each tile encodes/decodes in isolation.
- **Cross-platform** — must work on Metal, Vulkan, DX12, and WebGPU (WASM). Don't use backend-specific features.

## Architecture

Modular pipeline with swappable stages:
1. Color space conversion (YCoCg-R baseline)
2. Transform (LeGall 5/3 wavelet baseline)
3. Quantization (uniform scalar baseline)
4. Entropy coding (rANS baseline)

Shader source is in `gpu-codec/src/shaders/*.wgsl`. Rust host code is in `gpu-codec/src/encoder/` and `gpu-codec/src/decoder/`.

## Platform Notes

- Cross-platform via wgpu: Metal (macOS), Vulkan (Linux/Windows/Android), DX12 (Windows), WebGPU (WASM/browser)
- Dev machine: Apple M1 — 8 GPU cores, ~2.6 TFLOPS FP32, **no FP64**, 32KB threadgroup memory, max 1024 threads/workgroup
- WASM target must work — avoid features not available in WebGPU (e.g. some storage texture formats, push constants)
- WGSL shaders are the single source — transpiled per backend by naga

## Code Style

- Rust, edition 2021. Code must pass `cargo fmt` and `cargo clippy` with no warnings.
- **No code duplication** — extract shared logic rather than copy-pasting. If two stages do similar things, factor it out.
- All code, comments, docs, and commit messages in **English**
- Keep shader code (WGSL) simple and readable — comment non-obvious GPU-specific tricks
- Each pipeline stage is a separate module; new experiments go in `src/experiments/`
- Benchmark every change: PSNR, SSIM, bitrate (bpp), encode/decode FPS, latency
- **All dependencies must be open source** — no proprietary or closed-source libraries

## Research Protocol

- Log experiments in `RESEARCH_LOG.md` with hypothesis, implementation, results, analysis
- Always compare against baseline and previous best
- Test on varied content types when possible

## AI Team Lead Protocol

Claude operates as **team lead** for this project:

1. **Parallel execution** — Launch multiple agents for independent tasks to maximize throughput.
2. **Plan-driven** — Follow `MILESTONES.md` milestone by milestone.
3. **At every natural checkpoint** (milestone, sub-deliverable, or significant feature complete):
   - Run `cargo test --release` regression tests — fix any failures before proceeding.
   - Log progress in `RESEARCH_LOG.md` (hypothesis, implementation, results, analysis).
   - Update `MILESTONES.md` with completion status.
   - Commit code with descriptive message.
4. **Continue automatically** — After committing, pick up the next item in the plan without waiting.

## Don'ts

- Don't add CPU fallback paths unless explicitly asked — this is GPU-only by design
- Don't introduce cross-tile dependencies — parallelism is the whole point
- Don't use f64 in shaders — M1 has no hardware double precision
- Don't commit test material to git (it's in `.gitignore`)
