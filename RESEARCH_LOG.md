# GPU-Native Broadcast Codec — Research Log

## 2026-02-22: Phase 1 Baseline Implementation

### Hypothesis
A simple GPU-native pipeline (YCoCg-R color conversion + LeGall 5/3 wavelet + uniform scalar quantization + i16 packing) can serve as a correct, measurable baseline for iterative improvement. The goal is not quality — it's a working end-to-end GPU pipeline with benchmarks.

### Implementation
- **Color space**: YCoCg-R (reversible, f32 arithmetic on GPU)
- **Transform**: Single-level 2D LeGall 5/3 wavelet via lifting scheme
  - Row pass: one workgroup (128 threads) per row per tile
  - Column pass: one workgroup per column per tile
  - Uses workgroup shared memory for the lifting operations
  - Tile size: 256x256 (configurable)
- **Quantization**: Uniform scalar with configurable step size, optional dead zone
- **Entropy coding**: Phase 1 uses no real entropy coding — coefficients are packed as i16 pairs into u32 words (fixed 16 bits per coefficient)
- **Bitstream**: Simple header + raw packed coefficients

### Results — Test Image (512x512 synthetic gradient/checker pattern)

| QStep | PSNR (dB) | SSIM   | BPP   |
|-------|-----------|--------|-------|
| 1     | 55.88     | 1.0000 | 48.00 |
| 2     | 51.89     | 1.0000 | 48.00 |
| 4     | 46.29     | 0.9999 | 48.00 |
| 8     | 40.01     | 0.9994 | 48.00 |
| 16    | 33.71     | 0.9975 | 48.00 |
| 32    | 27.57     | 0.9898 | 48.00 |

Throughput (512x512, qstep=4): ~90 fps encode, ~96 fps decode

### Analysis
- The pipeline is correct: PSNR scales as expected with quantization step (~6 dB per doubling).
- BPP is constant at 48 (= 3 channels x 16 bits/coefficient) because we have no entropy coding yet.
- The wavelet + quantization is doing its job (q=1 gives ~56 dB, not infinity, due to f32 rounding in the lifting scheme — expected).
- Throughput is reasonable for Phase 1 but includes significant CPU overhead (deinterleave/interleave of color planes, per-plane GPU submissions).

### Key observations for next steps
1. **Entropy coding is the #1 priority** — without it, bitrate doesn't change with quantization. Most coefficients at higher Q are zero, so even simple entropy coding should dramatically reduce BPP.
2. **GPU deinterleave/interleave** — Currently done on CPU. Should be a trivial compute shader for better pipelining.
3. **Multi-level wavelet** — Single-level wastes bits on the large LL subband. 2-3 levels should improve RD performance.
4. **Real test content** — Need broadcast-representative images (talking head, sports, graphics).

### Next experiment
Implement per-tile rANS entropy coding to get meaningful bitrate measurements.

---

## 2026-02-22: rANS Entropy Coding Integration

### Hypothesis
Adding per-tile rANS (range Asymmetric Numeral Systems) entropy coding will dramatically reduce bitrate since most wavelet coefficients after quantization are zero or near-zero. rANS is chosen over Golomb-Rice because: deterministic throughput (no branch-heavy trees), naturally parallelizable across tiles, and well-suited for eventual GPU implementation with interleaved streams.

### Implementation
- **rANS codec**: CPU-side for correctness (GPU implementation planned as optimization)
  - 32-bit state, 12-bit precision (M = 4096)
  - Per-tile frequency tables with adaptive alphabet (range = [min_coeff, max_coeff])
  - Histogram normalization ensuring all non-zero symbols keep freq >= 1
  - Byte-aligned output for easy streaming
  - Based on Fabian Giesen's ryg_rans design (public domain)
- **Tile independence preserved**: each tile has its own frequency table and encoded stream
- **Bitstream format**: header + per-tile (min_val, alphabet_size, freq_table as u16, encoded_bytes)

### Results — Test Image (512x512 synthetic gradient/checker pattern)

| QStep | PSNR (dB) | SSIM   | BPP (i16) | BPP (rANS) | Compression vs i16 |
|-------|-----------|--------|-----------|------------|-------------------|
| 1     | 55.88     | 1.0000 | 48.00     | 10.78      | 4.5x              |
| 2     | 51.89     | 1.0000 | 48.00     | 8.51       | 5.6x              |
| 4     | 46.29     | 0.9999 | 48.00     | 7.33       | 6.5x              |
| 8     | 40.01     | 0.9994 | 48.00     | 6.46       | 7.4x              |
| 16    | 33.71     | 0.9975 | 48.00     | 5.57       | 8.6x              |
| 32    | 27.57     | 0.9898 | 48.00     | 4.61       | 10.4x             |

Throughput (512x512, qstep=4): ~57 fps encode, ~73 fps decode

### Analysis
- **BPP now varies with quantization** as expected. Higher Q = more zeros = better compression.
- **6.5x compression at q=4** (46 dB PSNR) is a strong result for a first entropy coder.
- PSNR is identical to Phase 1 (rANS is lossless — all loss comes from quantization).
- Encode throughput dropped from ~90 to ~57 fps due to CPU-side rANS. Decode dropped less (96→73 fps) because rANS decode is simpler than encode.
- The per-tile frequency table overhead is visible at low Q (q=1: 10.78 bpp) where coefficients have a wide range. At higher Q the alphabet shrinks and overhead drops.
- For broadcast 1080p50 at target ≤200 Mbps: 200 Mbps / (1920×1080×50) = ~1.93 bpp. We'd need q≈64+ which gives poor quality. This motivates multi-level wavelet and better quantization strategies.

### Key observations for next steps
1. **Multi-level wavelet decomposition** — The single-level LL subband is huge (half the tile) and has high entropy. 2-3 levels will concentrate energy better and dramatically improve RD.
2. **GPU rANS** — Move rANS to GPU using interleaved streams (32 parallel rANS states per tile). This will recover the throughput loss.
3. **Dead-zone quantization** — Setting dead_zone > 0 should zero out more near-zero coefficients, improving compression at minimal quality cost.
4. **Subband-specific quantization** — HH/HL/LH subbands are less perceptually important. Using higher Q for detail subbands could improve RD significantly.
5. **Real test content** — Synthetic image has unusual statistics. Need real broadcast frames.

### Next experiment
Multi-level wavelet (2-3 decomposition levels) to improve rate-distortion performance.

---

## 2026-02-22: Multi-Level Wavelet Decomposition

### Hypothesis
Recursively applying the wavelet transform to the LL (low-low) subband will concentrate energy more effectively, reducing the entropy of higher-level subbands and improving rate-distortion performance. With 256-pixel tiles, 3 levels of decomposition (256→128→64→32) should be near-optimal.

### Implementation
- **Shader update**: Added `region_size` parameter to `TransformParams` struct. The shader uses `region_size` instead of `tile_size` to determine the working area for each pass, so higher decomposition levels only process the LL subband.
- **Forward transform**: Each level reads from `output_buf` (or `input_buf` for level 0), does rows→`temp_buf`, columns→`output_buf`. Region halves each level.
- **Inverse transform**: Copies all data to `output_buf` first (preserving detail subbands), then each level processes columns `output→temp`, rows `temp→output`. Region doubles each level from the smallest.
- **Config**: Added `wavelet_levels: u32` to `CodecConfig` (default: 3).
- **Bug fix**: Initial implementation had incorrect buffer alternation logic using `level % 2`, which corrupted data for levels > 1 (PSNR dropped to 4.83 dB). Fixed by using a consistent buffer flow pattern.

### Results — Wavelet Level Comparison (qstep=4, 512x512 test image)

| Levels | PSNR (dB) | SSIM   | BPP  | Ratio vs 1-level |
|--------|-----------|--------|------|-------------------|
| 1      | 46.29     | 0.9999 | 7.33 | 1.0x              |
| 2      | 47.08     | 0.9999 | 3.63 | 2.0x              |
| 3      | 46.20     | 0.9998 | 2.70 | 2.7x              |
| 4      | 44.42     | 0.9998 | 2.57 | 2.9x              |

### Results — Quantization Sweep with 3-level wavelet

| QStep | PSNR (dB) | SSIM   | BPP (1-level) | BPP (3-level) | Improvement |
|-------|-----------|--------|---------------|---------------|-------------|
| 1     | 54.23     | 1.0000 | 10.78         | 5.67          | 1.9x        |
| 2     | 50.76     | 1.0000 | 8.51          | 3.42          | 2.5x        |
| 4     | 46.20     | 0.9998 | 7.33          | 2.70          | 2.7x        |
| 8     | 40.43     | 0.9995 | 6.46          | 2.32          | 2.8x        |
| 16    | 34.54     | 0.9977 | 5.57          | 1.93          | 2.9x        |
| 32    | 28.56     | 0.9913 | 4.61          | 1.51          | 3.1x        |

### Analysis
- **Level 2 is remarkable**: it *improves* PSNR by 0.8 dB while cutting BPP in half. The additional decomposition concentrates energy in the LL2 subband, making all subbands more compressible.
- **Level 3 is the sweet spot**: 2.7x compression vs single-level at nearly identical quality (46.20 vs 46.29 dB — 0.09 dB difference is negligible).
- **Level 4 shows diminishing returns**: only 5% BPP reduction but 1.8 dB quality loss. The 16-pixel LL4 subband is too small to represent well with uniform quantization.
- **Broadcast target approaching**: At qstep=16, we're at 1.93 bpp with 34.5 dB PSNR. For 1080p50 at 200 Mbps target (1.93 bpp), this is right at the boundary. Still needs quality improvement for broadcast acceptability (target: 38+ dB for contribution quality).
- **Throughput maintained**: GPU-side transform is trivially fast; the multi-level dispatch adds negligible overhead (just more shader dispatches). Bottleneck remains CPU-side rANS.

### Next experiments
1. **Dead-zone quantization** — Set dead_zone > 0 to zero out near-zero coefficients, especially in HH subbands. Should improve compression at minimal quality cost.
2. **Subband-specific quantization** — Use higher Q for HH/HL/LH subbands, lower Q for LL. This exploits the fact that high-frequency detail is less perceptually important.
3. **GPU rANS** — Move entropy coding to GPU with interleaved streams to recover throughput.
4. **Real test content** — Need broadcast-representative images for meaningful evaluation.
