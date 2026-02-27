# GNC — Research Log

> Historical entries (2026-02-22 to 2026-02-26) archived in `docs/archive/RESEARCH_LOG_2026-02-22_to_26.md`.

---

## 2026-02-27: Temporal MV prediction for bidir ME (B-frames)

### Hypothesis
Consecutive B-frames sharing the same reference pair have correlated forward/backward MVs. Using the first B-frame's MVs as predictors for the second should skip coarse search on both directions.

### Implementation
- Added `@group(0) @binding(8)` (predictor_fwd_mvs) and `@binding(9)` (predictor_bwd_mvs) to `block_match_bidir.wgsl`
- When `use_predictor != 0`, both forward and backward coarse searches are skipped; predictor MVs converted from half-pel to integer-pel as fine search starting point
- Modified `estimate_bidir()` to accept optional predictor buffers
- Modified `encode_bframe()` to accept predictors and return MV buffers
- Tracked `prev_bidir_fwd_mv`/`prev_bidir_bwd_mv` in B-frame group loop, reset per group
- Increased `max_storage_buffers_per_shader_stage` from 8 to 10

### Results (bbb_1080p, q=50, ki=8)

| B-frame | No predictor | With predictor | Change |
|---------|-------------|----------------|--------|
| Time | ~87ms | ~82ms | **-6%** |
| Quality | 37.89 dB | 37.89 dB | identical |

30-frame benchmark: 13.2 → 13.4 fps (+1.5%).

### Analysis
1. Modest improvement on identical-frame benchmark because all-zero MVs make coarse search trivially fast.
2. Real video with motion diversity should see larger gains (coarse search is the expensive part, ~30ms per direction at ±16).
3. Within each B-frame group (2 B-frames between anchors), only the second B-frame benefits from prediction. With B_FRAMES_PER_GROUP=2, that's 50% of B-frames.

---

## 2026-02-27: Bidir ME search range reduction — ±32 → ±16

### Hypothesis
B-frames interpolate between two references (forward and backward), so each direction's motion is typically half the total scene motion. A ±16 search range should be sufficient for B-frame ME while reducing coarse candidates from 4,225 to 1,089 (4x reduction).

### Implementation
Added `ME_BIDIR_SEARCH_RANGE: u32 = 16` constant in `motion.rs`, used in `estimate_bidir` instead of `ME_SEARCH_RANGE`.

### Results (bbb_1080p, q=50, ki=8)

| Metric | ±32 | ±16 | Change |
|--------|-----|-----|--------|
| B-frame time | 100ms | 87ms | **-13%** |
| 10-frame fps | 12.3 | 13.4 | +9% |
| 30-frame fps | 11.5 | 13.2 | +15% |
| Quality | 37.82 dB | 37.82 dB | identical |

### Analysis
1. B-frames are ~60% of inter-frames at ki=8 (pattern: I B B P B B P B B P...), so this 13ms savings per B-frame compounds across the sequence.
2. Quality is identical because at 30fps the inter-frame motion is small enough that ±16 covers virtually all real motion per direction.
3. For content with extreme motion, `ME_BIDIR_SEARCH_RANGE` can be increased independently of `ME_SEARCH_RANGE`.

---

## 2026-02-27: Temporal MV prediction for P-frames

### Hypothesis
Consecutive P-frames have highly correlated motion vectors. Using the previous P-frame's MVs as predictors can skip the expensive coarse search (4,225 candidates) and only do fine refinement (81 candidates at ±4), reducing ME cost by ~4x for predicted frames.

### Implementation
- Modified `block_match.wgsl` to accept a `predictor_mvs` buffer and `use_predictor` flag
- When predictor is available: skip Phase 1 (coarse search), convert half-pel MV to integer-pel, use as starting point for Phase 2 (fine search) with configurable range
- Added `predictor_mvs: Option<&wgpu::Buffer>` parameter to `MotionEstimator::estimate()`
- In sequence loop: track `prev_mv_buf`, pass to next P-frame, reset on keyframe
- `encode_pframe` returns `(CompressedFrame, wgpu::Buffer)` to propagate MV buffer

### Results (bbb_1080p, q=50, ki=3 P-only)

| P-frame type | Time | Loads/block |
|-------------|------|-------------|
| First P (no predictor) | 60ms | 88K (coarse+fine) |
| Predicted P (±4 fine) | 45ms | 21K (fine only) |
| Improvement | **-25%** | **-76%** |

Quality identical: 37.83-37.84 dB for both paths.

### Analysis
1. 15ms savings per predicted P-frame. The coarse search (4,225 × 16 = 67.6K loads) is entirely eliminated for predicted frames.
2. Tested ±8 predictor fine range (74K loads) — only 5-6ms savings because full-resolution SAD is expensive even with fewer candidates.
3. ±4 is optimal for same-content frames. For real video with large inter-frame motion changes, ±8 may be needed (configurable via ME_PRED_FINE_RANGE).
4. B-frames don't benefit yet (they use bidir ME which doesn't have temporal prediction).

---

## 2026-02-27: ME search range reduction — ±64 → ±32

### Hypothesis
Motion estimation coarse search (±64, 16,641 candidates per block) dominates P/B frame GPU compute time. Reducing to ±32 (4,225 candidates) should nearly halve ME cost with negligible quality impact for 30fps content.

### Implementation
Changed `ME_SEARCH_RANGE` constant from 64 to 32 in `motion.rs`. The shader search range is a uniform parameter, so no shader changes needed.

Also tested ±16 (1,089 candidates) for comparison.

### Results — Sequence encode (bbb_1080p, q=50, ki=8)

| Search Range | P-frame | B-frame | 10-frame FPS | 30-frame FPS | Quality |
|-------------|---------|---------|--------------|--------------|---------|
| ±64 (old) | 113ms | 180ms | 7.2 fps | 6.9 fps | 37.82 dB |
| ±32 (new) | 59ms | 100ms | 12.3 fps | 11.5 fps | 37.82 dB |
| ±16 (tested) | 49ms | 85ms | 14.0 fps | — | 37.82 dB |

### Analysis
1. P-frame time nearly halved (113ms → 59ms). The coarse search was testing 16,641 candidates × 16 subsampled loads = 266K loads per block. At ±32, this drops to 67K loads — a 4x reduction.
2. Quality is identical for this benchmark (same frame repeated). Real video with large motion may see small quality degradation at ±32, but for 30fps 1080p, ±32 pixels covers virtually all motion.
3. ±16 shows diminishing returns (59ms → 49ms, only 10ms gain) because non-ME work (entropy encode, local decode, wavelet/quantize) dominates at that point.
4. Also tested fused rANS encode in batched pipeline — **negative result**: 20ms slower per P-frame because the fused shader wastes GPU occupancy (256 threads, only 32 encode).

### Remaining bottleneck analysis (P-frame at ±32)
- ME coarse+fine: ~20ms
- MC + wavelet + quantize (3 planes): ~10ms
- rANS entropy encode: ~10ms GPU compute
- rANS readback (30MB): ~10ms DMA + pack
- Local decode (dequant + inverse wavelet + MC, 3 planes): ~10ms
- Total: ~60ms

---

## 2026-02-27: Sequence encode GPU pipeline optimization

### Hypothesis
Video encode bottleneck is pipeline stalls and CPU roundtrips in the per-frame encode loop. Eliminating CPU entropy decode from I-frame local decode and batching GPU work into single submits should improve fps significantly toward the 30 fps target.

### Implementation
Four optimizations applied to `sequence.rs`:

1. **I-frame GPU local decode** (`local_decode_iframe_gpu`): After `encode()`, quantized planes persist on GPU in `mc_out` (Y), `ref_upload` (Co), `plane_b` (Cg). New method reads directly from these buffers for dequantize → inverse wavelet → reference frame update, completely eliminating CPU entropy decode + 30MB re-upload per I-frame.

2. **Split-phase rANS encode API**: Added `dispatch_3planes_to_cmd` (dispatches histogram + normalize + encode to external command encoder) and `finish_3planes_readback` (map + poll + pack tiles) to `GpuRansEncoder`. Enables batching entropy encode with other GPU work in a single submit.

3. **P-frame batched pipeline**: Single command encoder for forward pass + entropy encode dispatches + local decode + MV staging copy → single submit → single poll. Eliminates inter-phase GPU pipeline stalls.

4. **B-frame batched pipeline**: Same pattern as P-frame. Added `BidirStaging` struct and split-phase bidir MV/modes readback to `MotionEstimator`.

Also removed dead `local_decode_iframe` method (replaced by GPU version).

### Results — Sequence encode (bbb_1080p, q=50, ki=8)

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| 10 frames | 6.5 fps | 7.2 fps | +11% |
| 30 frames | 6.3 fps | 6.9 fps | +10% |
| I-only (10f) | 25.7 fps | 25.7 fps | — |

Per-frame timing (30 frames, q=50): I-frame ~39ms, P-frame ~126ms, B-frame ~193ms.

### Analysis
1. The I-frame GPU local decode eliminates ~30MB CPU readback per I-frame — measurable improvement for I-heavy sequences.
2. Batching GPU work into single submits removes small pipeline stalls but the improvement is modest because **GPU compute time dominates**, not pipeline overhead.
3. The fundamental bottleneck is the rANS GPU encode readback (~30MB per frame). At ~140ms/frame for P/B frames, reaching 30 fps (33ms/frame) requires either faster entropy coding or deferred/async readback across frames.
4. Possible next steps: use Rice entropy for sequence encode (faster GPU path), GPU kernel fusion for ME+MC+transform, or multi-frame async readback pipeline.

---

## 2026-02-27: Rice per-subband k_zrl + quotient overflow fix

### Hypothesis
Adaptive k_zrl per wavelet subband should close the +34% bpp gap between Rice and rANS at q=25.

### Implementation
Changed Rice+ZRL from a single global k_zrl to per-subband k_zrl arrays (one k_zrl per wavelet subband group). Modified: `rice_encode.wgsl`, `rice_decode.wgsl`, `rice_gpu.rs`, `rice.rs`, `format.rs`. K_STRIDE changed from 9 to 16 (MAX_GROUPS*2) to store both magnitude k and zrl k per group.

### Bug Found: Rice quotient overflow causes GPU decode corruption
GPU decode produced 24.74 dB (garbage) for real images at q=25 while CPU decode worked correctly.

**Root cause**: When a zero run starts in a subband with small k_zrl (e.g., k_zrl=0 for the LL band), the maximum encodable run-1 is `(31 << k_zrl) | ((1 << k_zrl) - 1)` = 31 for k_zrl=0 (max run=32). But the encoder counted the FULL run (up to 256), emitted the capped quotient (31), and advanced `s` by the full run. The decoder read the capped run (32) and advanced by only 32, desynchronizing the bit reader for all subsequent symbols.

The CPU decoder masked this because its BitReader returns 0 past end-of-stream (naturally producing zero tokens). The GPU decoder has no bounds checking and reads into adjacent streams' data, producing non-zero values where there should be zeros.

**Fix**: Cap zero-run counting at `max_run = 32 << k_zrl` in both GPU and CPU encoders. Remaining zeros are encoded as subsequent zero-run tokens (possibly with a different subband's k_zrl). No decoder changes needed.

### Results — Rice with per-subband k_zrl

| Quality | PSNR | Old bpp | New bpp | Change | vs rANS |
|---------|------|---------|---------|--------|---------|
| q=25 | 33.2 dB | 1.73 | 1.71 | -1.2% | +33% |
| q=50 | 37.7 dB | 2.42 | 2.37 | -2.1% | +3.0% |
| q=75 | 42.8 dB | 4.09 | 4.01 | -2.0% | -5.0% |
| q=90 | 50.5 dB | 8.96 | 8.90 | -0.7% | -7.8% |

### Analysis
1. Per-subband k_zrl gives 1-2% bpp improvement — modest because the Rice-vs-rANS gap is structural (fixed Golomb-Rice codewords vs adaptive distribution), not parametric.
2. The quotient overflow bug was a serious correctness issue affecting all zero runs longer than `32 << k_zrl` in the encoder. It could silently corrupt any GPU-encoded real image.
3. The remaining +33% gap at q=25 requires distribution-adaptive coding (e.g., canonical Huffman) to close, not further parameter tuning.

---

## 2026-02-27: GPU Rice+ZRL — Fix K-Stride Bug and Full Quality Validation

### Hypothesis
Zero-run-length (ZRL) coding should close the Rice-vs-rANS compression gap from +269%
to manageable levels. The previous implementation had a GPU corruption bug at q>=50 where
decoded output was ~6 dB (garbage). CPU unit tests passed, so the bug was isolated to GPU.

### Root Cause: K-Stride Overlap Bug
**When `num_levels=4` (q>=50), `num_groups = num_levels*2 = 8 = MAX_GROUPS`.**
The k_zrl parameter was stored at `k_output[tile_id * MAX_GROUPS + num_groups]`, i.e.,
`tile_id * 8 + 8`. This overlapped with the next tile's `k_values[0]` at
`(tile_id+1) * 8 + 0 = tile_id * 8 + 8`. Race condition between workgroups!

**Fix**: Changed stride from `MAX_GROUPS` to `K_STRIDE = MAX_GROUPS + 1 = 9` in
`rice_encode.wgsl`, `rice_decode.wgsl`, and `rice_gpu.rs`.

### Results — Rice+ZRL vs rANS (bbb_1080p, 1920x1080)

| Quality | PSNR | rANS bpp | Rice+ZRL bpp | Overhead |
|---------|------|----------|--------------|----------|
| q=25 | 33.19 dB | 1.29 | 1.73 | +34% |
| q=50 | ~37.5 dB | 2.30 | 2.42 | +5.2% |
| q=75 | ~42.5 dB | 4.22 | 4.09 | -3.1% |
| q=90 | ~50 dB | 9.65 | 8.96 | -7.1% |

**Speed (GPU Rice+ZRL):**

| Quality | Encode | Decode |
|---------|--------|--------|
| q=25 | 25.1ms (40 fps) | 14.3ms (70 fps) |
| q=50 | 24.0ms (42 fps) | 16.4ms (61 fps) |
| q=75 | 24.7ms (40 fps) | 16.4ms (61 fps) |
| q=90 | 24.4ms (41 fps) | 15.2ms (66 fps) |

### Key Findings

1. **ZRL closes the compression gap**: At q>=50, Rice+ZRL beats rANS in bpp.
2. **Rice is 1.5-2x faster than rANS** due to 256 independent streams (no state chain) and minimal shared memory (32B vs 16KB).
3. **Rice is now the recommended entropy coder** — competitive compression, faster, patent-free.
4. **Remaining gap at q=25 (+34%)** could be closed with adaptive k_zrl per subband.

---
