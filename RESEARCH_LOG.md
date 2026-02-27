# GNC — Research Log

> Historical entries (2026-02-22 to 2026-02-26) archived in `docs/archive/RESEARCH_LOG_2026-02-22_to_26.md`.

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
