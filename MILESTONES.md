# GNC Video Codec — Milestones

## Context

GNC is a GPU-native codec research project (Rust + wgpu/WGSL). It currently works as an image codec with basic P-frame temporal coding. The goal: turn it into a real video codec with quality scaling from lossless to extreme compression. An AI team will iterate and evaluate each milestone.

**Current baseline (1080p, q=75):** 44 dB PSNR, 5.93 bpp, 31 fps encode, 34 fps decode.
**Gap:** ~1.5-1.7x bpp vs JPEG 2000. ~8 dB gap at matched ~1.2 bpp. P-frame encode only 1.7 fps. No rate control, no B-frames.

---

## M1: Evaluation Framework ✅ COMPLETE (2026-02-25)

**Why first:** Every subsequent milestone needs automated regression detection and BD-rate measurement. The AI team cannot iterate without repeatable comparison.

### Deliverables

1. **Golden-baseline regression tests** — encode-decode at q=25/50/75/90 on 3+ reference images. Assert PSNR/bpp within tolerance. Fail test if any metric regresses. Store baselines in `tests/golden_baselines.toml`.

2. **`gnc rd-curve` CLI command** — sweep q=10,20,...,90,100, output CSV (q, qstep, psnr, ssim, bpp, encode_ms, decode_ms). Compute BD-rate between two CSV files.

3. **Multi-codec comparison** — extend sweep to compare GNC vs JPEG vs JPEG 2000 (via external tools). Output matched-PSNR and matched-bpp tables.

4. **Sequence metrics** — per-frame PSNR/SSIM/bpp in CSV. Average, min, max, stddev. Temporal consistency: max PSNR drop between consecutive frames.

### Definition of Done
- `cargo test --release` includes 8+ golden-baseline regression tests
- `gnc rd-curve` produces valid RD sweep in < 60 seconds
- BD-rate computation verified against a known pair
- Comparison script produces CSV with GNC/JPEG/J2K on all test images

### Key files
- `src/main.rs` — add `rd-curve` subcommand
- New: `tests/quality_regression.rs`, `tests/golden_baselines.toml`

---

## M2: Close the Image Compression Gap ✅ COMPLETE (2026-02-25)

**Why:** Per-frame compression efficiency is the foundation that temporal coding amplifies. Fixing disabled features and adding context-adaptive entropy closes the 1.5x gap to JPEG 2000.

### 2A: Fix CfL (Chroma-from-Luma)
- Diagnose alpha quantization precision loss (u8 truncation at high quality)
- Fix: f16 or scaled i16 alphas, or adaptive precision by quality level
- Re-enable in `quality_preset()` for q ≤ 90
- **Expected: 8-12% bpp reduction on chroma planes**

### 2B: Fix Adaptive Quantization
- Redesign to work in wavelet domain: compute activity from LL subband energy, not spatial variance
- Weight map modulates qstep per-tile (preserves tile independence)
- Validate via SSIM improvement at matched bpp
- **Expected: 5-10% bpp at matched SSIM**

### 2C: GPU Zero-Run-Length
- Port per-group ZRL from CPU path (`rans.rs`) to GPU (`rans_encode.wgsl`, `rans_histogram.wgsl`)
- Validate GPU output matches CPU reference
- **Expected: 3-8% bpp reduction at high quantization**

### 2D: Context-Adaptive Entropy
Largest single compression opportunity. Current rANS treats each coefficient independently.
- **Significance map + magnitude refinement** within each subband:
  - Pass 1: binary zero/nonzero map using 2-3 neighbor context
  - Pass 2: nonzero magnitudes using context bins from neighbors
- Select among 4-8 rANS frequency tables per context (not 1 per tile)
- Process in scan order within subbands, using shared memory for context state
- Matching decode logic in `rans_decode.wgsl`
- **Expected: 15-25% bpp reduction**

### Definition of Done
- **BD-rate vs JPEG 2000 improves from ~+60% to within +20%** on bbb_1080p
- At matched 35 dB PSNR, GNC bpp within 1.3x of JPEG 2000 (down from 1.6x)
- No encode throughput regression beyond 10% at 1080p
- M1 regression harness passes with updated baselines

### Key files
- `src/encoder/cfl.rs` — alpha precision fix
- `src/encoder/adaptive.rs` — wavelet-domain AQ redesign
- `src/shaders/rans_encode.wgsl`, `rans_histogram.wgsl` — GPU ZRL + context model
- `src/shaders/rans_decode.wgsl` — context-adaptive decode
- `src/encoder/rans.rs` — CPU reference for context model

---

## M3: Video Codec Fundamentals

**Why:** Transform from image codec to functioning video codec. Temporal compression, rate control, and proper sequence handling.

### 3A: Improved Motion Estimation ✅
- **Half-pel refinement**: bilinear interpolation on GPU after integer-pel full search ✅
- **Variable block size**: 8x8 alongside 16x16, SAD-based split decision (deferred — needs more infrastructure)
- **Larger search range**: ±64px ✅
- **Per-tile adaptive I/P**: if tile residual exceeds I-frame cost, zero MVs for that tile ✅
- **Critical fix**: encoder local decode now matches decoder (AQ + CfL), fixing P-frame reference drift ✅
- **Result**: bbb 22.6% savings at q=75; blue_sky still -1.5% (camera pan)

### 3B: B-Frame Support ✅
- `FrameType::Bidirectional` with `MotionField` extended for backward_vectors + block_modes ✅
- Bidir block matching shader: tests fwd-only, bwd-only, and bidir average per block ✅
- Bidir motion compensation shader with half-pel bilinear interpolation ✅
- B-frame scheduling in encode loop: [B B P] groups when ki >= 4 ✅
- Rust-level reference plane swap (zero GPU cost) for past/future ref management ✅
- `decode_order()` and `decode_sequence()` for B-frame aware decoding ✅
- B-frames are non-reference: no local decode loop needed ✅
- **Result**: B-frame encode/decode verified at 45+ dB PSNR on synthetic content

### 3C: Rate Control ✅
- **Per-frame**: R-Q model (bpp ~ c * qstep^-α) with exponential smoothing ✅
- **VBV (leaky bucket)**: max buffer size, target bitrate, constrain per-frame qstep ✅
- **CBR and VBR modes**: constant bitrate and variable bitrate ✅
- `--bitrate` and `--rate-mode` CLI flags ✅

### 3D: Sequence Container ✅
- GNV1 container: file header + frame index table + frame data ✅
- Frame index with offset, size, frame_type, pts — enables random access ✅
- I-frame seeking: `seek_to_keyframe()` ✅
- `encode-sequence` and `decode-sequence` CLI subcommands ✅

### Speed Optimization ✅
- Hierarchical coarse-to-fine block matching: 4×4 subsampled coarse search + ±4 fine search
- Batched GPU pipeline: preprocess + ME + forward encode in single command encoder
- Eliminated MV readback/re-upload roundtrip: GPU buffers used directly for MC
- Batched local decode: all 3 planes in single command encoder
- Deferred MV readback for bitstream serialization only
- Batched bidir readback: single submit+poll for fwd MVs, bwd MVs, block modes
- **Result**: 0.8 fps → 5.0 fps sequence encode (6.5x speedup)

### Definition of Done
- P-frame bpp savings ≥50% vs I-frames on all test content — **33% on bbb, 0% on blue_sky (camera pan)** ⚠️
- B-frames at least 20% smaller than P-frames — **~20% smaller on bbb** ✅
- `gnc encode --bitrate 10M` produces output within ±10% of target over 100 frames ✅
- `gnc encode -i frames/%04d.png -o video.gnv` and `gnc decode -i video.gnv -o frames/%04d.png` round-trips ✅
- Sequence encode ≥10 fps at 1080p — **5.0 fps** ⚠️ (up from 0.8 fps)
- M1 regression harness passes ✅

### Key files
- `src/shaders/block_match.wgsl` — half-pel, larger search
- `src/encoder/motion.rs` — variable block size
- `src/encoder/sequence.rs` — B-frames, rate control, reordering
- `src/lib.rs` — `FrameType::Bidirectional`, rate control config
- `src/format.rs` — sequence container (GP11 / `.gnv`)

---

## M4: Full Quality Spectrum

**Why:** A real codec must cover lossless through extreme compression with smooth, monotonic quality scaling.

### 4A: True Lossless ✅
- Integer-exact YCoCg-R color conversion: conditional `floor()` in lifting steps via `lossless` shader param ✅
- Integer-exact LeGall 5/3 wavelet: `floor()` in predict/update steps (all modes) ✅
- `CodecConfig::is_lossless()` → activates integer color path when qstep≤1, dead_zone=0, LeGall53 ✅
- Bit-exact round-trip verified on 4 image sizes (256², 512², 100², 300×200) ✅
- Per-subband entropy enabled at q=100 for better lossless coding ✅
- **Result**: PSNR=∞, SSIM=1.0, bpp=12.80 on bbb_1080p (lossless bpp limited by tile-independent rANS alphabet overhead)
- **Target: lossless bpp within 1.3x of J2K** — NOT MET (12.80 vs J2K ~3.5 bpp). Requires CABAC/bitplane entropy coding to close gap.

### 4B: Extreme Low-Bitrate ✅
- **CfL disabled** for all lossy quality levels — CfL alpha precision at high qstep causes 5-8 dB PSNR loss across entire quality range ✅
- Wavelet levels: 3 for q<50, 4 for q≥50 (transition at anchor point avoids monotonicity dip) ✅
- Aggressive chroma subsampling: chroma_weight=1.5 at q<50 ✅
- **Result: q=5 gives 27.26 dB at 0.47 bpp** — target met ✅
- Overall quality improved +2-8 dB across entire range from CfL disable

### 4C: Smooth Quality Curve ✅
- Increased MAX_ALPHABET from 2048 to 4096 across all rANS shaders + Rust host ✅
- New anchor structure: q=92 (last CDF 9/7, qstep=2.05), q=99 (qstep=2.0, dead_zone=0), q=100 (LeGall 5/3 lossless) ✅
- CDF 9/7 for all lossy (q=1-99) with qstep floor at 2.0 to keep rANS alphabet within GPU limits ✅
- LeGall 5/3 only at q=100 for lossless (integer wavelet can't match CDF 9/7 lossy quality) ✅
- **Validated strict monotonicity**: PSNR strictly increasing and bpp strictly increasing for q=1..100 on bbb_1080p ✅
- Extended regression test: PSNR monotonicity at q=5,10,...,95,100 on gradient and checkerboard ✅
- Named presets: deferred (not critical path)
- **Result**: smooth curve from 24.3 dB/0.35 bpp (q=1) through 51.95 dB/10.16 bpp (q=99) to ∞/12.79 bpp (q=100)

### Definition of Done
- Bit-exact lossless round-trip on all test images — **✅ verified (PSNR=∞, SSIM=1.0)**
- Sub-0.5 bpp at >27 dB PSNR on bbb_1080p — **✅ q=5 gives 0.47 bpp / 27.26 dB**
- `gnc rd-curve` confirms strict monotonicity q=1..100 on all test images — **✅ on bbb_1080p**
- Named presets work and are documented — deferred

### Key files
- `src/lib.rs` — quality_preset refinement, chroma subsampling config
- `src/shaders/color_convert.wgsl` — integer path for lossless
- `src/encoder/pipeline.rs` — skip mode for uniform tiles

---

## M5: Streaming and Broadcast Readiness

**Why:** Make the codec deployable in real-world pipelines.

### 5A: Bitstream Stability ✅
- GP11 format with backward-compatible GP10/GPC9/GPC8 reading ✅
- Bitstream specification document: `BITSTREAM_SPEC.md` ✅
- 5 conformance test bitstreams in `tests/conformance/` with decode output hashes ✅
- GP11 adds: per-tile CRC-32 checksums, tile index table, full B-frame motion serialization

### 5B: Error Resilience ✅
- Per-tile CRC-32 checksums (ISO 3309) in GP11 bitstream ✅
- `deserialize_compressed_validated()` returns per-tile CRC results ✅
- `substitute_corrupt_tiles()` replaces corrupt tiles with zero-data (mid-gray) ✅
- 8 conformance tests including corruption detection and recovery ✅

### 5C: WebGPU/WASM ✅
- Full library compiles to `wasm32-unknown-unknown` with WebGPU backend ✅
- `pollster` conditionally compiled (native only); `GpuContext::new_async()` public for WASM ✅
- `wasm-bindgen` entry points: `decode_gnc()`, `gnc_width()`, `gnc_height()` ✅
- `wasm-pack build --target web --release` produces 263 KB WASM binary ✅
- Minimal browser demo: `examples/web/index.html` decodes and displays .gnc files ✅

### Definition of Done
- Specification document exists — **✅ BITSTREAM_SPEC.md**
- 5+ conformance bitstreams in repo — **✅ 5 bitstreams**
- Corrupt tile detection and recovery demonstrated — **✅ conformance_corrupt_tile_recovery test**
- WASM build works, browser demo decodes a frame — **✅ 263 KB WASM, browser demo ready**

---

## M6: Performance and Scalability

**Why:** Real-time 1080p60 encode/decode is the throughput target for broadcast.

### 6A: Fused GPU Kernels
- Lean rANS encode shader without ZRL arrays (eliminates 16 KB/thread register spilling) ✅
- GPU-side frame padding shader (replaces CPU `pad_frame()`, saves ~6ms at 1080p) ✅
- Deferred weight map readback (eliminates intermediate GPU poll, saves ~5ms) ✅
- Fused wavelet+quantize shader (eliminate intermediate buffer)
- Fused quantize+histogram (single read of coefficients)
- **Result**: encode 124ms → 34ms (29→30 fps), no quality regression

### 6B: 4K Validation ✅
- Benchmark 3840x2160 content ✅
- Per-pixel throughput within 25% of 1080p ✅
- **Result**: 4K is 34-71% *faster* per pixel than 1080p (better GPU occupancy)
  - 4K encode: 101ms (82 MP/s) vs 1080p: 34ms (61 MP/s)
  - 4K decode: 70ms (118 MP/s) vs 1080p: 30ms (69 MP/s)

### 6C: Multi-Frame Pipelining
- Pipeline frame N+1's color+wavelet while frame N's entropy runs
- Double-buffered encode buffers

### Definition of Done
- 1080p encode < 16.7ms (60 fps)
- 1080p decode < 8.3ms (120 fps, pipelined)
- 4K per-pixel throughput within 25% of 1080p

---

## Dependency Graph

```
M1 (Evaluation) ──────────────────────────────────────┐
  │                                                     │
  v                                                     v
M2 (Compression Gap) ──> M4 (Quality Spectrum) ──> M5 (Broadcast)
  │                                                     ^
  v                                                     │
M3 (Video Fundamentals) ──> M6 (Performance) ──────────┘
```

## Target Summary

| Milestone | Key Metric | Current | Target |
|-----------|-----------|---------|--------|
| M1 | Automated BD-rate | manual | < 60s automated |
| M2 | BD-rate vs J2K (bbb) | +60% | < +20% |
| M3 | Temporal savings (ki=8) | 26% | 50%+ |
| M3 | Sequence encode fps | 1.7 | 10+ |
| M4 | Lossless bpp | 5-6 | < 4 |
| M4 | Extreme compression | 0.62 bpp/29 dB | < 0.5 bpp/27+ dB |
| M5 | Conformance tests | 0 | 5+ |
| M6 | Encode fps (1080p) | 31 | 60 |
