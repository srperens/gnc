# CLAUDE.md — AI Instructions for GNC

All project goals, design rules, and priorities are in **[GOALS.md](GOALS.md)** — the single source of truth. This file contains only AI-specific instructions and technical reference.

## Build & Run

```bash
cargo build --release
cargo run --release -- benchmark -i test_material/frames/bbb_1080p.png -q 75
cargo run --release -- rd-curve -i test_material/frames/bbb_1080p.png --compare-codecs
cargo test --release
```

## Architecture

Modular pipeline with swappable stages:
1. Color space conversion (YCoCg-R, integer-exact lossless path available)
2. Transform (CDF 9/7 wavelet for lossy q=1-99, LeGall 5/3 for lossless q=100; 3-4 levels adaptive)
3. Quantization (adaptive with perceptual subband weights, CfL chroma prediction at q=50-85, fused quantize+histogram shader)
4. Entropy coding — three backends:
   - **Rice+ZRL** (fastest): 256 fully independent streams per tile, significance map + Golomb-Rice + zero-run-length
   - **rANS** (default): 32 interleaved streams per tile, per-subband frequency tables
   - **Bitplane**: block-based, fully parallel decode
5. Video: I/P/B frames, half-pel motion estimation, hierarchical block matching, CBR/VBR rate control
6. Container: GNV1 sequence format with frame index, keyframe seeking, error resilience (per-tile CRC-32)

Shader source is in `src/shaders/*.wgsl` (32 shaders). Rust host code is in `src/encoder/` and `src/decoder/`.

## Platform Notes

- Dev machine: Apple M1 — 8 GPU cores, ~2.6 TFLOPS FP32, no FP64, 32KB threadgroup memory, max 1024 threads/workgroup
- WASM target must work — avoid features not available in WebGPU (e.g. some storage texture formats, push constants)
- WGSL shaders are the single source — transpiled per backend by naga

## Code Style

- Rust, edition 2021. Keep shader code (WGSL) simple and readable — comment non-obvious GPU-specific tricks.
- **Zero clippy warnings** — `cargo clippy --release` and `cargo clippy --release --target wasm32-unknown-unknown` must both be clean. Fix warnings before committing. Prefer fixing the code over suppressing; `#[allow(clippy::…)]` is OK on individual items with justification but **blanket allows** (module-level `#![allow(…)]`, `dead_code` on entire impls, etc.) are **not acceptable**.
- **No `unsafe`** unless absolutely unavoidable. Prefer safe abstractions.
- Each pipeline stage is a separate module; new experiments go in `src/experiments/`.
- Don't commit test material to git (it's in `.gitignore`).

## Research Protocol

- Log experiments in `RESEARCH_LOG.md` with hypothesis, implementation, results, analysis.
- Always compare against baseline and previous best.
- Test on varied content types when possible.

## AI Team Lead Protocol

Claude operates as **team lead** for this project:

1. **Parallel execution** — Launch multiple agents for independent tasks to maximize throughput.
2. **Plan-driven** — Follow priorities in `GOALS.md`.
3. **At every natural checkpoint** (significant feature, priority item complete):
   - Run `cargo test --release` — fix any failures before proceeding.
   - Log progress in `RESEARCH_LOG.md`.
   - Commit code with descriptive message.
4. **Continue automatically** — After committing, pick up the next item without waiting.

## Key Documents

- **[GOALS.md](GOALS.md)** — Rules, priorities, current state, non-goals
- **[docs/BITSTREAM_SPEC.md](docs/BITSTREAM_SPEC.md)** — Bitstream format specification
- **[RESEARCH_LOG.md](RESEARCH_LOG.md)** — Experiment log
- **[README.md](README.md)** — Public project description
- **[docs/archive/](docs/archive/)** — Historical documents (MILESTONES.md, INSTRUCTION.md, etc.)
