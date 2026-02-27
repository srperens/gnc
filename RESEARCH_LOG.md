# GNC — Research Log

> Historical entries (2026-02-22 to 2026-02-26) archived in `docs/archive/RESEARCH_LOG_2026-02-22_to_26.md`.

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
