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

---

## 2026-02-22: Dead-Zone Quantization

### Hypothesis
Widening the zero bin around zero (dead zone) should increase the number of zero-valued coefficients after quantization, improving entropy coding efficiency at a modest quality cost. In wavelet codecs, many detail coefficients are near-zero; aggressively zeroing them trades minimal perceptual quality for significant bitrate savings.

### Implementation
Dead-zone support was already present in the quantize shader. The parameter `dead_zone` is a fraction of `step_size`: values with `|val| < dead_zone * step_size` are mapped to zero. Values above the threshold use standard rounding.

### Results — blue_sky (1080p, nature/landscape)

| QStep | Dead Zone | PSNR (dB) | BPP  | BPP vs dz=0 |
|-------|-----------|-----------|------|-------------|
| 4     | 0.00      | 43.47     | 4.92 | baseline    |
| 4     | 0.75      | 42.70     | 4.04 | -18%        |
| 4     | 1.00      | 41.45     | 3.47 | -29%        |
| 8     | 0.00      | 38.19     | 2.91 | baseline    |
| 8     | 0.75      | 37.36     | 2.34 | -20%        |
| 8     | 1.00      | 36.02     | 1.95 | -33%        |
| 16    | 0.00      | 32.68     | 1.66 | baseline    |
| 16    | 0.75      | 31.83     | 1.24 | -25%        |
| 16    | 1.00      | 30.77     | 0.99 | -40%        |

### Results — touchdown_pass (1080p, sports/motion)

| QStep | Dead Zone | PSNR (dB) | BPP  | BPP vs dz=0 |
|-------|-----------|-----------|------|-------------|
| 4     | 0.00      | 42.98     | 5.32 | baseline    |
| 4     | 0.75      | 41.78     | 4.06 | -24%        |
| 4     | 1.00      | 40.03     | 3.22 | -39%        |
| 8     | 0.00      | 37.52     | 2.70 | baseline    |
| 8     | 0.75      | 36.39     | 1.91 | -29%        |
| 8     | 1.00      | 35.00     | 1.50 | -44%        |
| 16    | 0.00      | 32.26     | 1.31 | baseline    |
| 16    | 0.75      | 31.56     | 0.95 | -28%        |
| 16    | 1.00      | 30.49     | 0.77 | -41%        |

### Analysis
- **Dead zones below 0.5 have no effect** because standard `floor(abs/step + 0.5)` rounding already maps values below `0.5 * step` to zero. The dead zone parameter only adds value when it exceeds the implicit rounding threshold.
- **dz=0.75 is a strong operating point**: ~20-29% BPP reduction for only 0.7-1.2 dB PSNR loss. This is a favorable trade — losing less than 1 dB per 20%+ bitrate reduction.
- **dz=1.0 is aggressive**: 29-44% BPP reduction but 1.8-2.5 dB PSNR loss. Useful for very low bitrate targets.
- **Broadcast target**: At qstep=8 + dz=0.75, we get 37.4 dB at 2.34 bpp (blue_sky) and 36.4 dB at 1.91 bpp (touchdown). The 1080p50 target of ~1.93 bpp is achievable at usable quality.
- **Content dependency**: Sports content (touchdown) benefits more from dead-zone (more near-zero detail coefficients from complex textures) but also loses more quality per dB.

### Key observations
1. The dead-zone parameter space below 0.5 is wasted — should start sweep at 0.5.
2. For a production codec, dead_zone should default to 0.5 (no change) or 0.75 (modest bitrate savings with minimal quality impact).
3. **Subband-specific quantization is the real opportunity**: applying different dead zones or Q steps to LL vs LH/HL/HH subbands would allow preserving low-frequency quality while aggressively compressing high-frequency detail.

### Next experiments
1. **Subband-specific quantization** — Different Q steps or dead zones per wavelet subband.
2. **GPU rANS** — Move entropy coding to GPU to recover throughput.
3. **Perceptual quality tuning** — Use SSIM-optimized quantization weights.

---

## 2026-02-22: Subband-Specific Quantization

### Hypothesis
All wavelet coefficients currently receive the same quantization step regardless of subband. HH subbands contain perceptually unimportant high-frequency noise that can be quantized much harder. Applying per-subband weight multipliers to the quantization step (effective_step = base_step × weight) should reduce bitrate at similar or better perceptual quality. Chroma planes (Co/Cg) can also be quantized more aggressively since the HVS is less sensitive to chroma detail.

### Implementation
- **Shader rewrite** (`quantize.wgsl`): Now 2D-aware. Each thread computes its (x, y) position in the plane, derives tile-local coordinates (lx, ly), and walks from the outermost decomposition level inward to classify the coefficient into LL, LH, HL, or HH at the appropriate level. The subband weight is looked up from 16 packed f32s (4 × vec4) in the uniform buffer.
- **SubbandWeights struct** (`lib.rs`): Contains `ll` weight, per-level `[LH, HL, HH]` detail weights, and a `chroma_weight` multiplier. Two constructors:
  - `uniform(levels)` — all weights 1.0, identical to previous behavior (regression baseline)
  - `perceptual(levels)` — default perceptual preset (see table below)
- **Encoder/decoder pipelines**: Pre-pack luma weights and chroma weights (all weights × chroma_weight). Planes 1,2 (Co, Cg) use chroma-weighted variant. Encoder and decoder use identical weights for correct round-trip.
- **Bitstream format**: Bumped to GPC4. Header now includes full SubbandWeights (ll, num_detail_levels, per-level [LH, HL, HH], chroma_weight) so the decoder can reconstruct correctly without out-of-band information.
- **Experiments**: Added `subband_weight_experiments()` with 5 presets × 3 QSteps = 15 experiments.

### Subband weight layout in GPU uniform buffer
16 packed f32s (4 × vec4): `[LL, L0_LH, L0_HL, L0_HH, L1_LH, L1_HL, L1_HH, ...]`
Supports up to 5 decomposition levels (1 + 5×3 = 16 slots).

### Default perceptual weights

| Subband       | Weight | Rationale                                  |
|---------------|--------|--------------------------------------------|
| LL            | 1.0    | Preserve DC / low-frequency energy         |
| Level 1: LH, HL | 1.0 | Outermost detail — most visible edges      |
| Level 1: HH  | 1.5    | High-frequency diagonal — less perceptible |
| Level 2: LH, HL | 1.5 | Mid-frequency detail                       |
| Level 2: HH  | 2.0    | Less perceptible                           |
| Level 3: LH, HL | 2.0 | Inner detail subbands                      |
| Level 3: HH  | 3.0    | Least perceptible — quantize hardest       |
| Chroma mult.  | 1.5   | HVS less sensitive to chroma detail        |

### Experiment presets

| Preset           | Description                                           |
|------------------|-------------------------------------------------------|
| uniform          | All weights 1.0 (regression baseline)                 |
| perceptual       | Default table above, chroma 1.5×                      |
| aggressive_hh    | HH at 2.0/3.0/4.0, LH/HL at 1.0, chroma 1.0×        |
| chroma_save      | Uniform luma, chroma 2.0×                             |
| full_perceptual  | Perceptual subband weights + chroma 2.0×              |

### Verification status
- [x] Builds with no new warnings (`cargo clippy`)
- [x] All 10 existing tests pass (`cargo test`)
- [x] Regression check: `uniform()` weights produce identical PSNR to previous code (BPP ~0.1 higher due to other code changes since dead-zone experiments — GPU rANS decoder, buffer caching — not from subband quantization)
- [x] Perceptual weights experiment sweep
- [x] Results analysis and comparison

### Results — blue_sky (1920x1080, nature/landscape)

| Preset          | QStep | PSNR (dB) | BPP  | BPP vs uniform | PSNR cost |
|-----------------|-------|-----------|------|----------------|-----------|
| uniform         | 4     | 43.47     | 5.03 | baseline       | —         |
| perceptual      | 4     | 40.54     | 3.66 | **-27%**       | -2.93     |
| aggressive_hh   | 4     | 42.19     | 4.52 | -10%           | -1.28     |
| chroma_save     | 4     | 41.18     | 3.83 | -24%           | -2.29     |
| full_perceptual | 4     | 39.42     | 3.27 | **-35%**       | -4.05     |
| uniform         | 8     | 38.19     | 3.02 | baseline       | —         |
| perceptual      | 8     | 35.34     | 2.13 | **-29%**       | -2.85     |
| aggressive_hh   | 8     | 37.19     | 2.75 | -9%            | -1.00     |
| chroma_save     | 8     | 35.86     | 2.17 | -28%           | -2.33     |
| full_perceptual | 8     | 34.34     | 1.85 | **-39%**       | -3.85     |
| uniform         | 16    | 32.68     | 1.76 | baseline       | —         |
| perceptual      | 16    | 30.27     | 1.19 | **-32%**       | -2.41     |
| aggressive_hh   | 16    | 31.91     | 1.64 | -7%            | -0.77     |
| chroma_save     | 16    | 30.79     | 1.22 | -31%           | -1.89     |
| full_perceptual | 16    | 29.48     | 1.02 | **-42%**       | -3.20     |

### Results — touchdown_pass (1920x1080, sports/motion)

| Preset          | QStep | PSNR (dB) | BPP  | BPP vs uniform | PSNR cost |
|-----------------|-------|-----------|------|----------------|-----------|
| uniform         | 4     | 42.98     | 5.42 | baseline       | —         |
| perceptual      | 4     | 39.67     | 3.69 | **-32%**       | -3.31     |
| aggressive_hh   | 4     | 41.13     | 4.81 | -11%           | -1.85     |
| chroma_save     | 4     | 41.18     | 3.87 | -29%           | -1.80     |
| full_perceptual | 4     | 39.15     | 3.30 | **-39%**       | -3.83     |
| uniform         | 8     | 37.52     | 2.81 | baseline       | —         |
| perceptual      | 8     | 34.74     | 2.00 | **-29%**       | -2.78     |
| aggressive_hh   | 8     | 36.18     | 2.50 | -11%           | -1.34     |
| chroma_save     | 8     | 35.48     | 2.27 | -19%           | -2.04     |
| full_perceptual | 8     | 33.85     | 1.90 | **-32%**       | -3.67     |
| uniform         | 16    | 32.26     | 1.42 | baseline       | —         |
| perceptual      | 16    | 30.89     | 1.06 | **-25%**       | -1.37     |
| aggressive_hh   | 16    | 31.66     | 1.27 | -11%           | -0.60     |
| chroma_save     | 16    | 29.87     | 1.26 | -11%           | -2.39     |
| full_perceptual | 16    | 29.03     | 1.02 | **-28%**       | -3.23     |

### Analysis

**Perceptual preset is the clear winner for general use.** It consistently delivers 25-32% BPP reduction across all QSteps and both test images at a cost of 1.4-3.3 dB PSNR. The PSNR cost is dominated by chroma loss (which PSNR overweights relative to human perception), making the actual visual quality closer than the numbers suggest.

**Key findings:**

1. **Perceptual weights save ~30% bitrate.** Averaging across both images and all QSteps: -29% BPP at -2.6 dB PSNR. This is the single biggest compression improvement since multi-level wavelet.

2. **Chroma weighting alone is nearly as effective as subband weighting alone.** `chroma_save` (uniform luma, 2× chroma) gets -19% to -31% BPP. This confirms that chroma planes are significantly overallocated with uniform quantization.

3. **Aggressive HH is too narrow.** `aggressive_hh` (only HH harder, no chroma) saves just 7-11% BPP — not worth the complexity. The HH subbands alone don't contain enough coefficients to make a big dent.

4. **Full perceptual is for low-bitrate targets.** The most aggressive preset (`full_perceptual`) achieves 28-42% BPP reduction but at 3-4 dB PSNR cost. Useful when chasing sub-1.5 bpp targets.

5. **Broadcast target update:** With perceptual weights at q=8: blue_sky 2.13 bpp / 35.3 dB, touchdown 2.00 bpp / 34.7 dB. The 1080p50 target of ~1.93 bpp is nearly achievable at q=8 + perceptual, with usable (though not contribution-grade) quality.

6. **Rate-distortion efficiency:** The perceptual preset provides roughly 0.5 dB/10% — i.e., every 10% BPP savings costs about 0.5 dB. This is a favorable trade for content where SSIM matters more than PSNR (sports, nature).

### Next steps
1. Consider making `SubbandWeights::perceptual(levels)` the default in `CodecConfig`
2. Tune weights further: the current weights were chosen by intuition — a grid search over weight values could find a better operating point
3. Combine with dead-zone quantization (dz=0.75 + perceptual weights) for maximum compression
4. GPU rANS optimization remains the throughput bottleneck

---

## 2026-02-22: Combined Dead-Zone + Subband Weights

### Hypothesis
Dead-zone quantization and subband-specific weights attack different sources of redundancy: dead-zone zeros out near-zero coefficients across all subbands, while subband weights apply coarser quantization to perceptually unimportant subbands. They should stack for near-additive BPP savings.

### Implementation
No code changes to the pipeline — just a new experiment set (`combined_dz_subband_experiments`) sweeping 3 QSteps × 3 dead-zones (0, 0.5, 0.75) × 3 weight presets (uniform, perceptual, full_perceptual) = 27 configurations.

### Key finding: dz=0.5 has no effect
Dead-zone 0.5 produces identical results to dz=0.0 in every case. This confirms the earlier observation: standard `floor(abs/step + 0.5)` rounding already maps values below `0.5 * step` to zero, so the dead-zone parameter only adds value above 0.5. Future sweeps should skip dz < 0.5.

### Results — blue_sky (1920x1080)

| Weights | Dead Zone | QStep | PSNR (dB) | BPP  | vs uniform/dz=0 |
|---------|-----------|-------|-----------|------|------------------|
| uniform         | 0.00 | 4  | 43.47 | 5.03 | baseline |
| perceptual      | 0.00 | 4  | 40.54 | 3.66 | -27% |
| full_perceptual | 0.00 | 4  | 39.42 | 3.27 | -35% |
| uniform         | 0.75 | 4  | 42.70 | 4.15 | -17% |
| perceptual      | 0.75 | 4  | 39.49 | 2.95 | -41% |
| full_perceptual | 0.75 | 4  | 38.19 | 2.59 | -49% |
| uniform         | 0.00 | 8  | 38.19 | 3.02 | baseline |
| perceptual      | 0.00 | 8  | 35.34 | 2.13 | -29% |
| full_perceptual | 0.00 | 8  | 34.34 | 1.85 | -39% |
| uniform         | 0.75 | 8  | 37.36 | 2.45 | -19% |
| **perceptual**  | **0.75** | **8** | **34.18** | **1.66** | **-45%** |
| full_perceptual | 0.75 | 8  | 33.11 | 1.42 | -53% |
| uniform         | 0.00 | 16 | 32.68 | 1.76 | baseline |
| perceptual      | 0.00 | 16 | 30.27 | 1.19 | -32% |
| full_perceptual | 0.00 | 16 | 29.48 | 1.02 | -42% |
| uniform         | 0.75 | 16 | 31.83 | 1.35 | -23% |
| perceptual      | 0.75 | 16 | 29.21 | 0.87 | -51% |
| full_perceptual | 0.75 | 16 | 28.18 | 0.74 | -58% |

### Results — touchdown_pass (1920x1080)

| Weights | Dead Zone | QStep | PSNR (dB) | BPP  | vs uniform/dz=0 |
|---------|-----------|-------|-----------|------|------------------|
| uniform         | 0.00 | 4  | 42.98 | 5.42 | baseline |
| perceptual      | 0.00 | 4  | 39.67 | 3.69 | -32% |
| full_perceptual | 0.00 | 4  | 39.15 | 3.30 | -39% |
| uniform         | 0.75 | 4  | 41.78 | 4.16 | -23% |
| perceptual      | 0.75 | 4  | 38.45 | 2.86 | -47% |
| full_perceptual | 0.75 | 4  | 38.04 | 2.67 | -51% |
| uniform         | 0.00 | 8  | 37.52 | 2.81 | baseline |
| perceptual      | 0.00 | 8  | 34.74 | 2.00 | -29% |
| full_perceptual | 0.00 | 8  | 33.85 | 1.90 | -32% |
| uniform         | 0.75 | 8  | 36.39 | 2.02 | -28% |
| **perceptual**  | **0.75** | **8** | **33.77** | **1.47** | **-48%** |
| full_perceptual | 0.75 | 8  | 33.00 | 1.42 | -49% |
| uniform         | 0.00 | 16 | 32.26 | 1.42 | baseline |
| perceptual      | 0.00 | 16 | 30.89 | 1.06 | -25% |
| full_perceptual | 0.00 | 16 | 29.03 | 1.02 | -28% |
| uniform         | 0.75 | 16 | 31.56 | 1.06 | -25% |
| perceptual      | 0.75 | 16 | 29.84 | 0.78 | -45% |
| full_perceptual | 0.75 | 16 | 26.40 | 0.63 | -56% |

### Analysis

**The gains stack.** At q=8, perceptual alone saves ~29%, dz=0.75 alone saves ~19-28%, combined saves **45-48%**. Not perfectly additive but close — the two techniques attack largely orthogonal redundancy.

**Recommended operating points for 1080p50:**

| Target | Config | PSNR | BPP | Bitrate |
|--------|--------|------|-----|---------|
| High quality | q=4, perceptual, dz=0.75 | ~39 dB | ~2.9 bpp | ~300 Mbps |
| Contribution | q=8, perceptual, dz=0.75 | ~34 dB | ~1.5-1.7 bpp | ~170 Mbps |
| Distribution | q=16, perceptual, dz=0.75 | ~29-30 dB | ~0.8-0.9 bpp | ~90 Mbps |

**The full_perceptual preset offers diminishing returns** at high Q: at q=16+dz=0.75, touchdown drops to 26.4 dB which is visually poor. The aggressive chroma weight (2.0×) pushes blue channel PSNR down to 24.3 dB. The standard perceptual preset (chroma 1.5×) is the safer default.

**Cumulative compression progress** (blue_sky, q=8 operating point):
1. Baseline (i16 packing): 48.00 bpp
2. + rANS entropy coding: 6.46 bpp (7.4×)
3. + 3-level wavelet: 2.91 bpp (2.2×)
4. + Dead-zone 0.75: 2.34 bpp (1.2×)
5. + Perceptual subband weights: **1.66 bpp** (1.4×)
6. Total compression: **29× vs baseline**, from 48 bpp to 1.66 bpp

### Next steps
1. GPU rANS throughput optimization — CPU-side rANS is the bottleneck
2. Context-modeled entropy coding — could save another 10-20%
3. Consider making q=8 + perceptual + dz=0.75 the default codec profile

---

## 2026-02-22: Zero-Run-Length Coding Before rANS (Negative Result)

### Hypothesis
After quantization + dead-zone + perceptual subband weights, ~70-80% of wavelet coefficients are zero. The current rANS encoder treats each zero as an individual symbol costing ~0.3-0.5 bits. Consecutive zeros are common (especially in HH subbands) but rANS has no way to exploit run-length structure. Replacing N consecutive zeros with a single "zero-run-of-N" symbol before rANS encoding should reduce BPP by ~15-25% at identical PSNR, analogous to MJPEG's run-length coding before Huffman.

### Implementation
- **Symbol mapping**: Non-zero coefficients keep their value; consecutive zeros become `zrun_base + (N - 1)` where `zrun_base = max(|coeff|) + 1`. Symbol 0 never appears in the transformed stream.
- **MAX_ZERO_RUN = 256**: longer runs split into multiple symbols.
- **Per-stream ZRL**: applied per stride-32 stream after deinterleaving, before rANS encoding. Shared histogram built from all ZRL-transformed symbols across all 32 streams.
- **Adaptive per-tile**: encoder tries both plain and ZRL, picks whichever produces a smaller tile. Zero-density heuristic (< 60% zeros → skip ZRL) avoids double-encoding overhead on low-zero tiles.
- **GPU decoder**: shader reads `zrun_base` from tile_info; decode loop changed from fixed-count `for` to output-driven `while` with run expansion. When `zrun_base == 0`, the shader falls back to the original one-symbol-per-output path.
- **Backward compat**: `zrun_base = 0` in tile header means no ZRL (old tiles decode identically).
- **Bitstream**: `zrun_base` added to per-tile serialization (InterleavedRansTile header). File format stays GPC5 (ZRL is transparent per-tile).

### Results — blue_sky (1920x1080)

| Weights | Dead Zone | QStep | PSNR (dB) | BPP (prev) | BPP (ZRL) | Change |
|---------|-----------|-------|-----------|-----------|-----------|--------|
| uniform         | 0.00 | 4  | 43.47 | 5.03 | 5.02 | -0.2% |
| uniform         | 0.00 | 8  | 38.19 | 3.02 | 3.02 | 0%    |
| uniform         | 0.00 | 16 | 32.68 | 1.76 | 1.76 | 0%    |
| uniform         | 0.75 | 4  | 42.70 | 4.15 | 4.14 | -0.2% |
| uniform         | 0.75 | 8  | 37.36 | 2.45 | 2.44 | -0.4% |
| uniform         | 0.75 | 16 | 31.83 | 1.35 | 1.35 | 0%    |
| perceptual      | 0.75 | 4  | 39.49 | 2.95 | 2.95 | 0%    |
| perceptual      | 0.75 | 8  | 34.18 | 1.66 | 1.65 | -0.6% |
| perceptual      | 0.75 | 16 | 29.21 | 0.87 | 0.87 | 0%    |
| full_perceptual | 0.75 | 8  | 33.11 | 1.42 | 1.41 | -0.7% |
| full_perceptual | 0.75 | 16 | 28.18 | 0.74 | 0.73 | **-1.4%** |

### Results — touchdown_pass (1920x1080)

| Weights | Dead Zone | QStep | PSNR (dB) | BPP (prev) | BPP (ZRL) | Change |
|---------|-----------|-------|-----------|-----------|-----------|--------|
| uniform         | 0.00 | 4  | 42.98 | 5.42 | 5.41 | -0.2% |
| uniform         | 0.00 | 8  | 37.52 | 2.81 | 2.81 | 0%    |
| uniform         | 0.75 | 8  | 36.39 | 2.02 | 2.02 | 0%    |
| perceptual      | 0.75 | 8  | 33.77 | 1.47 | 1.47 | 0%    |
| perceptual      | 0.75 | 16 | 29.84 | 0.78 | 0.78 | 0%    |
| full_perceptual | 0.75 | 16 | 26.40 | 0.63 | 0.63 | 0%    |

### Throughput (blue_sky, 1920x1080, default config, 10 iterations)

| Metric | Value |
|--------|-------|
| Encode | 187.70 ms (5.3 fps) |
| Decode (sequential) | 36.24 ms (27.6 fps) |
| Decode (pipelined) | 34.15 ms (29.3 fps) |

Encode speed regressed ~1.7x due to the adaptive double-encoding (try both, pick smaller). The zero-density heuristic (skip ZRL when < 60% zeros) mitigates this for low-Q tiles. Decode speed is unaffected — the while-loop decoder runs identically whether processing ZRL or non-ZRL tiles.

### Analysis

**The hypothesis was wrong.** ZRL before rANS provides 0-1.4% BPP reduction — far short of the expected 15-25%. The adaptive approach correctly prevents regressions (PSNR is identical, BPP never increases), but the technique is essentially a no-op for this codec.

**Why ZRL fails before rANS (but succeeds before Huffman):**

1. **rANS already encodes zeros cheaply.** In a distribution where zeros are 70-80% of symbols, the zero symbol has frequency ~3000/4096. rANS encodes it for ~0.35 bits — nearly at the entropy limit. ZRL can't beat this for isolated zeros or short runs.

2. **Alphabet expansion overhead.** ZRL adds run symbols above `max_abs`, expanding the per-tile frequency table by up to `MAX_ZERO_RUN × 2 = 512 bytes`. For 256×256 tiles at moderate bitrates, this overhead often exceeds the stream savings.

3. **Short runs dominate.** Wavelet coefficient zeros are distributed across interleaved subbands. After stride-32 deinterleaving, zero runs within each stream tend to be short (1-5 zeros) because non-zero LL/LH/HL coefficients break up the runs. A run of 3 zeros costs one run symbol (~2-4 bits) vs three zero symbols (~1.0 bits). The savings per run is marginal.

4. **MJPEG's RLC works differently.** MJPEG uses run-length coding before Huffman, where each symbol costs minimum 1 bit. Encoding 256 zeros costs minimum 256 bits with Huffman; one run symbol costs ~8-16 bits. That's 16-32x savings. With rANS, 256 zeros cost ~90 bits total (0.35 bits each); one run symbol costs ~8-10 bits. That's only ~10x savings, and the alphabet overhead eats most of it.

**Where ZRL provides marginal benefit (up to 1.4%):**
- High quantization (q=16) + aggressive dead-zone (0.75) + full_perceptual weights
- These settings maximize zero density (>95%) and create longer runs in HH subbands
- Even here, the benefit barely exceeds the frequency table overhead

### Conclusion

ZRL before rANS is a conceptual mismatch. rANS is near-optimal for symbol-by-symbol entropy coding — its strength IS efficiently encoding highly probable symbols. Run-length coding is valuable before fixed-length or minimum-1-bit codes (Huffman), but adds negligible value before an arithmetic-family coder that already exploits skewed distributions.

The adaptive encoder infrastructure (zrun_base field, GPU while-loop decoder, per-tile ZRL flag) is retained with zero overhead on non-ZRL tiles. If a future entropy coder with higher per-symbol overhead is explored, ZRL could be re-evaluated.

### Next steps
1. **Context-modeled entropy coding** — the real opportunity for BPP reduction. Neighboring coefficients in the same subband are correlated; using context from already-decoded neighbors to modulate rANS frequencies could save 10-20%.
2. **Trellis quantization** — optimize quantized values to minimize rate at a given distortion target, rather than simple rounding.
3. **GPU rANS encode** — move rANS encoding to GPU with 32 parallel streams to recover encode throughput.

---

## 2026-02-23: Per-Subband Entropy Coding

### Hypothesis
The rANS encoder builds one frequency table per tile covering all wavelet subbands. This mixes very different statistical distributions — LL coefficients (large positive values, narrow range) with outer HH coefficients (mostly zeros, wide tails). A single table wastes entropy capacity by averaging across these distributions. Splitting into 4 subband-group frequency tables should improve compression 10-20% at identical quality, with no cross-tile dependencies.

### Implementation
- **4 subband groups**: Group 0 = LL (innermost DC), Group 1 = level 0 detail (LH+HL+HH, outermost), Group 2 = level 1 detail, Group 3 = level 2 detail (innermost detail)
- **Group mapping**: `compute_subband_group(lx, ly, tile_size, num_levels)` walks from the outermost decomposition level inward, classifying each coefficient by its 2D position
- **Encoder**: `rans_encode_tile_interleaved_subband()` builds 4 separate histograms, normalizes each to sum=4096, encodes each stream with per-symbol table selection based on coefficient position
- **ZRL disabled**: per-subband tables already model zero-heavy distributions tightly; ZRL's alphabet expansion is counterproductive with per-group tables
- **New types**: `SubbandGroupFreqs` (min_val, alphabet_size, freqs, cumfreqs per group), `SubbandRansTile` (4 groups + 32 interleaved streams)
- **Serialization**: per-group `[min_val, alphabet_size, freqs as u16]`, then standard 32-stream layout
- **GPU decoder**: WGSL shader dispatches dual-path — `params.per_subband` flag selects single-table or multi-table decode. Multi-table path loads all group cumfreqs into shared memory, computes group from 2D position per output symbol
- **Format**: GPC9 (backward-compatible with GPC8), `--per-subband` CLI flag, entropy type 2 in bitstream
- **Config**: `per_subband_entropy: bool` in `CodecConfig` (default false)

### Files modified
- `src/encoder/rans.rs` — encoder, decoder, serialization, types, 7 new tests
- `src/encoder/rans_gpu.rs` — `prepare_decode_buffers_subband()`, updated params struct
- `src/shaders/rans_decode.wgsl` — complete rewrite with dual-path decode
- `src/lib.rs` — `CodecConfig`, `EntropyData::SubbandRans` variant
- `src/encoder/pipeline.rs` — `EntropyMode` enum, routing
- `src/decoder/pipeline.rs` — subband buffer preparation
- `src/main.rs` — GPC9 format, CLI flag, serialization
- `src/experiments/mod.rs` — `entropy_experiments()`

### Results — bbb_1080p (1920x1080, animated film)

| Mode | QStep | PSNR (dB) | BPP (single) | BPP (subband) | Savings |
|------|-------|-----------|-------------|--------------|---------|
| Single table | 4 | 43.41 | 6.55 | — | — |
| Per-subband  | 4 | 43.41 | — | 6.03 | **-7.9%** |
| Single table | 8 | 37.88 | 4.14 | — | — |
| Per-subband  | 8 | 37.88 | — | 3.62 | **-12.6%** |
| Single table | 16 | 32.55 | 2.39 | — | — |
| Per-subband  | 16 | 32.55 | — | 1.92 | **-19.5%** |

### Results — test_512 (512x512, synthetic)

| Mode | QStep | PSNR (dB) | BPP (single) | BPP (subband) | Savings |
|------|-------|-----------|-------------|--------------|---------|
| Single table | 4 | 46.20 | 2.34 | — | — |
| Per-subband  | 4 | 46.20 | — | 2.27 | -3.0% |
| Single table | 8 | 40.43 | 2.00 | — | — |
| Per-subband  | 8 | 40.43 | — | 1.95 | -2.5% |
| Single table | 16 | 34.54 | 1.65 | — | — |
| Per-subband  | 16 | 34.54 | — | 1.65 | -0.0% |

### Analysis

**Per-subband entropy coding delivers 8-20% bitrate savings on natural 1080p content at zero quality cost.** The effect scales with quantization: higher Q means more zeros in detail subbands, making the distribution mismatch between LL and HH groups more pronounced. At q16, nearly 20% of the bitrate was wasted by forcing a single table to span both distributions.

**Content dependency**: The synthetic 512x512 image shows minimal benefit (0-3%) because it has unusual statistics — relatively uniform coefficient distributions across subbands. Natural photographic/cinematic content (bbb_1080p) benefits far more because LL coefficients are genuinely different from detail coefficients.

**Why it scales with Q**: At low Q (q=4), most coefficients are non-zero across all subbands, so distribution overlap between groups is high. At high Q (q=16), detail subbands are 90%+ zeros while LL retains significant non-zero coefficients — the distribution mismatch is extreme, and per-subband tables capture this perfectly.

**Implementation is clean**: No cross-tile dependencies, same 32-stream interleaved structure, same GPU decode pattern. The only overhead is 4 frequency tables (~800 bytes) per tile instead of 1 (~500 bytes), which is negligible relative to the stream data savings.

### Cumulative compression progress (bbb_1080p, q=8 operating point)

| Step | BPP | Multiplier |
|------|-----|------------|
| Baseline (i16 packing) | 48.00 | — |
| + rANS entropy coding | ~6.5 | 7.4x |
| + 3-level wavelet | 4.14 | 1.6x |
| + Per-subband entropy | 3.62 | 1.14x |
| **Total** | **3.62** | **13.3x** |

Note: Dead-zone, perceptual weights, and CfL provide additional multiplicative savings but change quality, so they're not included in this lossless-path comparison.

### Next steps
1. **CDF 9/7 as default lossy wavelet** — 4-5 dB PSNR improvement at same qstep is the single largest remaining opportunity for RD efficiency
2. **Stack all improvements** — per-subband entropy + CfL + CDF 9/7 have never been tested together; combined savings could be substantial
3. **GPU rANS encode** — throughput bottleneck remains CPU-side
4. **Honest competitive benchmark** — need JPEG/JPEG2000 numbers on same test images

---

## 2026-02-23: Feature Stacking + Competitive Benchmark (JPEG, JPEG 2000)

### Hypothesis
CDF 9/7, per-subband entropy, and CfL were developed and tested independently. Stacking all three quality-neutral features (CDF 9/7 wavelet + per-subband entropy coding + CfL prediction) should compound their individual gains. Adding perceptual weights and dead-zone on top should push GNC into JPEG-competitive territory.

### Implementation
- New `best_config_experiments()` sweeping 8 feature combinations × 4 qsteps = 32 configs
- Python benchmark script (`scripts/benchmark_codecs.py`) measuring libjpeg-turbo (14 quality levels) and OpenJPEG JPEG 2000 (13 compression ratios) on the same test images
- All metrics computed identically: PSNR via scikit-image, BPP from file size

### Results — Feature stacking on bbb_1080p

| Config | q4 PSNR/BPP | q8 PSNR/BPP | q16 PSNR/BPP | q32 PSNR/BPP |
|--------|-------------|-------------|--------------|--------------|
| 53_base (old default) | 43.41/6.55 | 37.88/4.14 | 32.55/2.39 | 27.31/1.28 |
| 97_base (CDF 9/7 only) | 46.77/7.41 | 41.92/4.96 | 37.41/3.17 | 33.31/1.96 |
| 97_sb (+ per-subband) | 46.77/6.53 | 41.92/4.15 | 37.41/2.46 | 33.31/1.36 |
| 97_sb_cfl (+ CfL) | 46.91/5.97 | 42.11/3.76 | 37.62/2.23 | 33.51/1.25 |
| 97_sb_cfl_perc (+ perceptual) | 43.64/4.48 | 38.94/2.70 | 34.54/1.52 | 30.61/0.82 |
| 97_all_dz75 (+ dead zone) | 42.42/3.78 | 37.65/2.17 | 33.25/1.18 | 29.47/0.62 |

**Per-subband entropy savings with CDF 9/7 are larger than with LeGall 5/3:**
- q8: 4.96 → 4.15 bpp = **-16.3%** (was -12.6% with 5/3)
- q16: 3.17 → 2.46 bpp = **-22.4%** (was -19.5% with 5/3)
- q32: 1.96 → 1.36 bpp = **-30.6%**

CDF 9/7's better energy compaction creates sharper distribution differences between subbands, giving per-subband entropy tables more to work with.

**CfL savings with CDF 9/7 are also larger:**
- q8: 4.96 → 4.50 bpp = **-9.3%** (was ~3% with 5/3 at matched qstep)
- q16: 3.17 → 2.86 bpp = **-9.8%**

CDF 9/7's floating-point coefficients have stronger inter-plane correlation that CfL can exploit.

### Competitive comparison at matched PSNR — bbb_1080p

| PSNR target | JPEG | GNC (best) | JP2 | GNC vs JPEG | GNC vs JP2 |
|-------------|------|------------|-----|-------------|------------|
| ~35 dB | 2.21 bpp (q90) | ~1.6 bpp | ~1.0 bpp | **-27%** | +60% |
| ~38 dB | 4.49 bpp (q98) | ~2.5 bpp | ~1.6 bpp | **-44%** | +56% |
| ~33 dB | 1.51 bpp (q80) | ~1.2 bpp | ~0.7 bpp | **-20%** | +71% |

### Competitive comparison — blue_sky_1080p

| PSNR target | JPEG | GNC (best) | JP2 | GNC vs JPEG | GNC vs JP2 |
|-------------|------|------------|-----|-------------|------------|
| ~36 dB | 1.17 bpp (q85) | ~1.2 bpp | ~0.75 bpp | tied | +60% |
| ~39 dB | 2.29 bpp (q95) | ~1.8 bpp | ~1.2 bpp | **-21%** | +50% |

### Competitive comparison — touchdown_1080p

| PSNR target | JPEG | GNC (best) | JP2 | GNC vs JPEG | GNC vs JP2 |
|-------------|------|------------|-----|-------------|------------|
| ~35 dB | 0.68 bpp (q50) | ~0.9 bpp | ~0.55 bpp | +32% | +64% |
| ~37 dB | 1.32 bpp (q80) | ~1.5 bpp | ~1.2 bpp | +14% | +25% |
| ~39 dB | 1.60 bpp (q85) | ~1.7 bpp | ~1.3 bpp | +6% | +31% |

### Encode speed comparison (bbb_1080p, 1920x1080)

| Codec | Encode time | Notes |
|-------|-------------|-------|
| JPEG (libjpeg-turbo) | 15-33 ms | CPU, single-threaded |
| JPEG 2000 (OpenJPEG) | 417-456 ms | CPU, single-threaded |
| GNC | ~188 ms | CPU rANS + GPU transform/quantize |

### Analysis

**GNC with full feature stack is now competitive with JPEG on animation and nature content** (bbb_1080p, blue_sky). On sports/high-texture content (touchdown), JPEG retains a modest edge. This is a dramatic improvement from the "3-5x worse than JPEG" assessment made before feature stacking.

**Content-dependent performance:**
- Animation (bbb_1080p): GNC beats JPEG by 20-44%. Smooth gradients and large flat areas favor wavelets over 8×8 DCT. JPEG's blocking artifacts waste bits at high quality.
- Nature (blue_sky): GNC roughly ties JPEG at moderate quality, beats by 21% at high quality.
- Sports (touchdown): JPEG wins by 6-32%. Dense textures with lots of high-frequency detail favor DCT's fixed-size blocks over wavelets' global frequency decomposition.

**The JPEG 2000 gap is ~1.5-1.7x consistently.** This is structural — JPEG 2000 has context-modeled entropy coding (EBCOT), trellis quantization, and 30+ years of optimization. The gap won't close without adding context modeling.

**CDF 9/7 is the biggest single improvement.** On bbb_1080p at q16: LeGall 5/3 gets 32.55 dB; CDF 9/7 gets 37.41 dB — **4.86 dB better at the same bitrate.** This alone moves GNC from "3x worse than JPEG" to "competitive with JPEG."

**Feature stacking is multiplicative, not additive.** Individual savings:
- CDF 9/7 vs 5/3: +4-5 dB PSNR at same Q (effective ~30% bitrate reduction at matched quality)
- Per-subband entropy: 12-31% bitrate reduction (quality-neutral)
- CfL: 9-10% bitrate reduction with CDF 9/7 (quality-neutral)
- Perceptual weights: 25-30% bitrate reduction (quality trade)
- Dead zone 0.75: 15-20% bitrate reduction (quality trade)
Combined: all features together reduce bitrate by ~67% vs 5/3 baseline (bbb_1080p q8: 4.14 → 2.17 bpp at ~matched quality, or 4.14 → 1.18 at lower quality).

### Key insight: CDF 9/7 should be the default

The data is unambiguous. CDF 9/7 provides massive RD improvement with no downside for lossy encoding. The codec should default to CDF 9/7 for lossy, keeping LeGall 5/3 only for lossless mode.

### Next steps
1. **Make CDF 9/7 default** for lossy encoding (qstep > 1)
2. **Enable per-subband entropy by default** — quality-neutral, always beneficial
3. **GPU rANS encode** — 188ms encode is the remaining bottleneck
4. **Context-modeled entropy** — the only path to close the 1.5x JP2 gap
5. **Temporal prediction** — for video, this is where the real compression lives

---

## 2026-02-23: GPU rANS Entropy Encoding

### Hypothesis
CPU-side rANS encoding is the throughput bottleneck: 165-188ms encode vs 26-37ms decode at 1080p on M1. The encode pipeline reads back ~30MB of quantized coefficients from GPU to CPU, builds histograms, normalizes frequency tables, and serially encodes 32 interleaved rANS streams per tile. Moving histogram building and rANS encoding to GPU compute shaders should eliminate the large readback and parallelize the serial encoding, targeting 2-6x encode speedup.

### Implementation
Two new compute shaders + host code, integrated into the existing encode pipeline:

**`rans_histogram.wgsl`** — GPU histogram building (256 threads/workgroup, 1 workgroup/tile):
- Phase 1: parallel min/max reduction across 256 threads using shared memory tree reduction (pattern from `variance_map.wgsl`)
- Phase 2: atomic histogram building with `atomicAdd` on `var<workgroup>` shared memory
- Per-subband mode: 4 separate histograms (one per wavelet subband group), min/max tracked per group
- Output: fixed stride per tile (HIST_TILE_STRIDE = 2060 u32s) with `[min_val, alphabet_size, histogram[]]`

**`rans_encode.wgsl`** — GPU rANS encoding (32 threads/workgroup, 1 workgroup/tile):
- Each thread encodes one of 32 independent interleaved rANS streams
- Cumfreq tables loaded cooperatively into shared memory (same pattern as `rans_decode.wgsl`)
- Encodes in reverse coefficient order with byte-level renormalization
- Byte writing via `write_byte()` into packed u32 array (each stream writes to its own non-overlapping region — no atomics needed)
- Per-subband mode: selects cumfreq region based on 2D coefficient position via `compute_subband_group()`
- Output: stream bytes (MAX_STREAM_BYTES=4096 per stream) + metadata (write_ptr, final_state per stream)

**`rans_gpu_encode.rs`** — Host-side `GpuRansEncoder`:
- Two-pass pipeline: GPU histogram → CPU normalize → GPU encode → readback & pack
- `dispatch_histogram()`: creates params buffer, dispatches 1 workgroup per tile
- `normalize_histograms()`: reads back ~1MB histogram buffer, calls existing `normalize_histogram()` per tile/group, builds cumfreq tables
- `dispatch_encode()`: uploads cumfreq + tile_info, zero-initializes stream buffer (`mapped_at_creation`), dispatches 1 workgroup per tile
- `pack_tiles()`: reads back encoded streams, extracts per-stream bytes from write_ptr..MAX_STREAM_BYTES, packs into `InterleavedRansTile`/`SubbandRansTile` structs
- `encode_plane_to_tiles()`: high-level API used by encoder pipeline

**Pipeline integration** (`pipeline.rs`):
- `GpuRansEncoder` added as field of `EncoderPipeline`
- `use_gpu_encode` flag gated on `config.gpu_entropy_encode && entropy_coder != Bitplane`
- GPU encode path at all 3 entropy encode points: Y+CfL, chroma+CfL, non-CfL
- Quantized data stays on GPU (`plane_buf_b`) — no 30MB readback in GPU path
- CPU fallback via `--cpu-encode` flag for testing/debugging

### Data flow comparison

```
CPU encode (before):
  [GPU quantize] → readback 30MB → [CPU histogram] → [CPU rANS encode x120 tiles x32 streams]

GPU encode (after):
  [GPU quantize] → [GPU histogram] → readback ~1MB → [CPU normalize] → upload ~1MB →
  [GPU rANS encode] → readback ~5-15MB encoded streams → [CPU pack tiles]
```

### Results — Encode throughput at 1080p on M1

| Image | CPU Encode | GPU Encode | Speedup |
|---|---|---|---|
| blue_sky_1080p | 166.6 ms (6.0 fps) | 78.0 ms (12.8 fps) | **2.1x** |
| bbb_1080p | 168.0 ms (6.0 fps) | 79.0 ms (12.7 fps) | **2.1x** |
| touchdown_1080p | 164.0 ms (6.1 fps) | 77.7 ms (12.9 fps) | **2.1x** |

### Results — Quality verification

| Image | CPU PSNR | GPU PSNR | CPU BPP | GPU BPP |
|---|---|---|---|---|
| blue_sky_1080p | 43.47 dB | 43.47 dB | 5.02 | 5.03 |

PSNR is identical. BPP differs by <0.2% due to histogram normalization rounding (GPU builds histograms from f32 with `round()` vs CPU casting from f32 to i32 — minor floating-point path differences).

### Analysis

**2.1x overall encode speedup, consistent across all content types.** The entropy encoding step itself improved ~3.3x (from ~130ms to ~40ms), but the remaining ~40ms of non-entropy work (color convert, wavelet transform, quantize, buffer management) is unchanged and now dominates.

**Where the time goes (estimated breakdown of 78ms GPU encode):**
- Color conversion + deinterleave: ~8ms (CPU readback + deinterleave still on CPU)
- Wavelet transform (3 planes × GPU dispatch): ~5ms
- Quantize (3 planes × GPU dispatch): ~3ms
- GPU histogram dispatch + readback: ~5ms
- CPU normalize histograms: ~2ms
- GPU encode dispatch + readback: ~15ms
- Pack tiles + overhead: ~10ms
- Variance analysis (adaptive quant): ~5ms
- CfL prediction: ~10ms
- Buffer allocation/management: ~15ms

**Why not the predicted 6x speedup:** The plan estimated 188ms→30ms by assuming entropy encoding was the entire bottleneck. In reality, the 188ms included ~35-50ms of non-entropy pipeline work that doesn't benefit from GPU rANS. The entropy-specific improvement (3.3x) is close to expectations.

**Decode speed also improved slightly** (37ms→26ms) — not from GPU rANS encode, but from the decoder `plane_results` buffer now having correct `COPY_SRC` flags for temporal prediction, reducing unnecessary buffer recreation.

### Cumulative throughput progress (blue_sky_1080p)

| Step | Encode | Decode | Notes |
|------|--------|--------|-------|
| Phase 1 baseline | ~90 fps (512x512) | ~96 fps | i16 packing, no entropy |
| + rANS entropy | ~57 fps (512x512) | ~73 fps | CPU rANS bottleneck |
| + All features (1080p) | 5.3 fps | 27.6 fps | CPU rANS at scale |
| + GPU rANS encode | **12.9 fps** | **37.9 fps** | **2.1x encode, 1.4x decode** |

### Next steps
1. **GPU color deinterleave** — the CPU readback + deinterleave of 3 planes is ~8ms of unnecessary work
2. **Buffer reuse across frames** — allocating fresh GPU buffers per encode adds ~15ms overhead; caching buffers (like the decoder already does) would eliminate this
3. **Fused quantize+histogram** — a single shader pass that quantizes AND builds the histogram simultaneously would eliminate one GPU dispatch + the intermediate buffer
4. **Context-modeled entropy coding** — the remaining path to close the 1.5x JPEG 2000 gap
5. **Temporal prediction benchmarks** — P-frame encoding with motion estimation is implemented but not yet benchmarked

## 2026-02-23: Temporal Prediction (P-frames) Benchmarks

### Hypothesis
P-frame encoding with 16x16 block matching motion estimation should reduce bitrate for sequences with temporal redundancy (animation, talking heads) while maintaining quality. Camera-pan content with global motion may not benefit due to limited search range (±16 pixels) and block-level granularity.

### Implementation (already present)
- **Motion estimation**: GPU block matching on Y plane, 16x16 blocks, ±16 search range
- **Motion compensation**: GPU shader computes residual (encode) or reconstruction (decode)
- **Encoder local decode loop**: I-frame: entropy decode → dequant → inv wavelet → reference planes. P-frame: same + inverse MC to reconstruct reference.
- **Decoder**: maintains `reference_planes` buffers, applies MC for P-frames automatically
- **Encode sequence**: `encode_sequence()` schedules I/P frames based on `keyframe_interval`

### Results — BBB (1080p animation, 25fps, 10 consecutive frames)

| KI | I+P Size | I-only Size | Saving | I+P fps | I-only fps |
|----|----------|-------------|--------|---------|------------|
| 8 (IPPPPPPP) | 12.6 MB (4.85 bpp) | 17.0 MB (6.57 bpp) | **26.1%** | 1.7 | 3.9 |
| 4 (IPPP) | 13.0 MB (5.03 bpp) | 17.0 MB (6.57 bpp) | **23.4%** | 1.8 | 3.9 |
| 2 (IP) | 14.0 MB (5.41 bpp) | 17.0 MB (6.57 bpp) | **17.6%** | 2.2 | 3.8 |

P-frame quality: 43.07–43.26 dB vs I-frame 43.41 dB (only 0.15–0.34 dB degradation).

Individual P-frame sizes: ~4.24–4.55 bpp (35% smaller than I-frames at 6.57 bpp).

### Results — blue_sky (1080p nature, camera pan, 10 consecutive frames)

| KI | I+P Size | I-only Size | Saving | I+P fps | I-only fps |
|----|----------|-------------|--------|---------|------------|
| 8 | 14.2 MB (5.49 bpp) | 13.2 MB (5.11 bpp) | **-7.4%** | 1.7 | 4.2 |
| 4 | 13.8 MB (5.33 bpp) | 13.2 MB (5.11 bpp) | **-4.2%** | 1.9 | 4.1 |

P-frames are **larger** than I-frames for camera pan content. Quality drifts from 43.47→42.80 dB over 7 P-frames (ki=8).

### Analysis

**P-frames work well for animation (+26% savings) but fail for camera pan (-7%).** This matches expectations:

1. **BBB (animation)**: mostly static backgrounds with localized character motion. 16x16 block matching captures this efficiently. P-frames at ~4.3 bpp vs I-frames at 6.6 bpp — the temporal residual has much lower energy than the original signal.

2. **blue_sky (camera pan)**: global translational motion across the entire frame. 16x16 blocks with ±16 pixel search range can find local matches, but the ME overhead (motion vectors + residual coding overhead) exceeds the savings. The residual after MC still contains the quantization noise from the local decode loop, which is harder to compress than the original smooth gradient content.

**Speed impact**: P-frame encode is ~2x slower than I-frame encode due to:
- Motion estimation (block matching across search range)
- Motion compensation (3 planes × GPU dispatch)
- Local decode loop (entropy decode → dequant → inv wavelet → inv MC, all 3 planes)
- This brings I+P sequence throughput to 1.7 fps vs 3.9 fps for I-only

**Quality drift**: P-frames accumulate small quality losses through the encode→decode→reference→encode chain. BBB shows only 0.3 dB drift over 7 P-frames. blue_sky shows 0.6 dB drift, amplified by the larger residual.

### Test coverage (5 new tests, all passing)
- `test_encode_sequence_all_iframes` — ki=1 produces only I-frames
- `test_encode_sequence_ip_pattern` — ki=4 produces correct I/P/P/P pattern with MVs
- `test_pframe_roundtrip_quality` — encode I+P → decode → PSNR > 25 dB
- `test_pframe_identical_frames_correct_decode` — identical frames: good decode quality, near-zero MVs
- `test_sequence_decode_all_frames` — 5-frame I/P/P/I/P sequence roundtrips correctly

Total test count: 51 (all passing in release mode).

---

## 2026-02-24: Full GPU Encoder Pipeline Optimization

### Hypothesis
Moving CfL forward prediction to GPU, adding a GPU deinterleave shader, and refactoring the encoder pipeline to minimize CPU↔GPU round-trips should significantly improve encode throughput. The previous bottleneck was CPU-side work between GPU dispatches (~40 ms of the ~78 ms total encode time).

### Implementation
- **CfL forward on GPU**: New `cfl_alpha.wgsl` and `cfl_forward.wgsl` shaders replace CPU-side CfL forward prediction
- **GPU deinterleave**: New `deinterleave.wgsl` shader splits interleaved YCoCg into separate planes on GPU
- **Weight map normalization on GPU**: New `weight_map_normalize.wgsl` shader for adaptive quantization
- **Pipeline restructure**: Major refactor of `encoder/pipeline.rs` and `decoder/pipeline.rs` to reduce CPU stalls between GPU dispatches

### Results — 1080p Encode Throughput (q=8, CDF 9/7, CfL, per-subband rANS GPU)

| Image | Encode (before) | Encode (after) | Speedup | Decode |
|-------|-----------------|----------------|---------|--------|
| bbb_1080p | 78.0 ms (12.8 fps) | **33.0 ms (30.3 fps)** | **2.4x** | 25.4 ms (39.3 fps) |
| blue_sky_1080p | 78.0 ms (12.8 fps) | **34.7 ms (28.8 fps)** | **2.3x** | 25.0 ms (40.0 fps) |
| touchdown_1080p | 77.7 ms (12.9 fps) | **40.0 ms (25.0 fps)** | **1.9x** | 25.8 ms (38.8 fps) |

Quality unchanged (verified PSNR/BPP match previous results):

| Image | PSNR (dB) | SSIM | BPP |
|-------|-----------|------|-----|
| bbb_1080p | 43.41 | 0.9997 | 6.57 |
| blue_sky_1080p | 43.47 | 0.9997 | 5.03 |
| touchdown_1080p | 42.98 | 0.9983 | 5.43 |

### Analysis
- **2.0–2.4x encode speedup** across all content types, with no quality/compression change
- Encoder now runs at **25–30 fps for 1080p** (~60 MP/s), approaching real-time 1080p30
- The gains come from eliminating CPU round-trips: CfL forward, deinterleave, and weight map normalization all moved to GPU shaders
- Decode throughput unchanged at ~39–40 fps (was already GPU-native)
- touchdown is slightly slower encode (40 ms vs 33 ms) — likely due to higher coefficient complexity requiring more rANS work

### Cumulative throughput progress (1080p encode)

| Version | Encode FPS | Bottleneck |
|---------|-----------|------------|
| CPU rANS baseline | 6.0 fps | CPU entropy encoding |
| + GPU rANS encode | 12.8 fps | CPU pipeline overhead (CfL, deinterleave) |
| + Full GPU pipeline | **28–30 fps** | GPU compute (approaching hardware limit) |

Total improvement from original CPU-bottlenecked pipeline: **~5x faster encode**.

### Next steps for temporal prediction
1. **Global motion estimation** — detect and compensate global translation/rotation before block matching (would fix blue_sky)
2. **Sub-pixel ME** — half-pel or quarter-pel refinement for better residual reduction
3. **Larger search range** — ±32 or ±64 pixels for faster motion (blue_sky pans ~10 px/frame)
4. **Adaptive I/P decision** — skip P-frame when residual energy exceeds I-frame estimate
5. **P-frame encode speed** — the local decode loop is expensive; could be GPU-accelerated with GPU rANS decode in the encoder
6. **B-frames** — bi-directional prediction for further compression gains

---

## 2026-02-24: Encoder Buffer Caching

### Hypothesis
The encoder allocates ~15 GPU buffers per encode() call (~60MB+ for 1080p). Caching these buffers across frames should eliminate allocation overhead and improve throughput, especially for multi-frame sequences. The decoder already implements buffer caching successfully.

### Implementation
- **CachedEncodeBuffers struct**: 15 GPU buffers persisted in `EncoderPipeline` across encode() calls
  - 3-channel: `input_buf`, `color_out`
  - Single-plane work: `plane_a/b/c`, `co_plane`, `cg_plane`, `recon_y`
  - AQ: `variance_buf`, `wm_scratch`, `weight_map_buf`
  - CfL: `raw_alpha`, `dq_alpha` (variable size, 2× growth)
  - P-frame: `mc_out`, `ref_upload`, `recon_out`
- **Resolution-aware**: buffers are reallocated only when padded dimensions change
- **P-frame local decode optimization**: decoded residual stays on GPU (`plane_a`) for inverse MC instead of readback+reupload — eliminates ~8MB transfer per plane per P-frame
- Removed unused `create_plane_buffer` and `read_buffer_f32` helpers; replaced with `read_buffer_f32_cached` and `ensure_var_buf`

### Results — 1080p Encode/Decode Throughput (bbb_1080p, q=8, CfL, GPU rANS)

| Platform | Encode | Decode | Decode (pipelined) |
|----------|--------|--------|--------------------|
| Windows DX12 (native GPU) | **34.4 ms (29.1 fps)** | **15.0 ms (66.9 fps)** | **11.9 ms (84.1 fps)** |
| WSL2 (virtualized GPU) | 133.7 ms (7.5 fps) | 109.3 ms (9.2 fps) | 110.0 ms (9.1 fps) |

Quality unchanged: PSNR 43.41 dB, SSIM 0.9997, BPP 6.57

### Analysis
- Encode throughput on native DX12 matches the ~30 fps measured before buffer caching — first-frame allocation cost is amortized over 20 iterations, so the benchmark already captured steady-state performance
- The real win is for **multi-frame encoding** (encode_sequence) where buffer allocation was repeated per frame — now zero allocations after frame 1
- WSL2 is ~4× slower than native DX12 due to GPU virtualization overhead
- **Decode improved significantly**: 66.9 fps (was ~39 fps before decoder buffer caching) — the decoder buffer caching from the previous session is now paying off
- Pipelined decode reaches **84.1 fps** — well above real-time 1080p60

### Cumulative throughput progress (1080p, native DX12)

| Version | Encode FPS | Decode FPS |
|---------|-----------|------------|
| CPU rANS baseline | 6.0 | ~40 |
| + GPU rANS encode | 12.8 | ~40 |
| + Full GPU pipeline | 28–30 | ~40 |
| + Buffer caching | **29** | **67 (84 pipelined)** |

### Next steps
1. **Benchmark encode_sequence** — measure per-frame encode time for frames 2+ to quantify buffer caching benefit for temporal encoding
2. **Profile remaining bottlenecks** — GPU rANS encode is likely the dominant cost now (~15ms); histogram+normalize+encode could be further optimized
3. **Reduce pad_frame CPU cost** — currently allocates+copies each frame; could upload directly and pad on GPU

---

## 2026-02-25: GPU Per-Subband rANS Alphabet Overflow Fix + Quality Presets + GPU Batching

### Context
While enabling 4 wavelet levels (from 3) for higher quality presets, PSNR crashed from ~44 dB to ~13 dB. This turned out to be a latent bug in the GPU per-subband rANS encoder that was always present but only manifested when subband alphabet sizes exceeded `MAX_GROUP_ALPHABET`.

### Bug Investigation
Systematic isolation identified the root cause through binary search over config flags:

| Config change | PSNR (dB) | Status |
|---------------|-----------|--------|
| Baseline (3 levels, no AQ, no per_subband) | 43.04 | OK |
| + AQ enabled | 27.96 | BAD — AQ broken |
| + per_subband GPU encode | 13.48 | BAD — GPU encode broken |
| + per_subband CPU encode | 44.08 | OK — CPU path works |

**AQ bug (Adaptive Quantization)**: AQ applies spatial-domain variance weights to wavelet-domain coefficient positions. In wavelet domain, position (x,y) doesn't correspond to the same spatial location — AQ weights are meaningless. Disabled for now.

**GPU per-subband rANS bug**: The histogram shader clamped group alphabet sizes to `MAX_GROUP_ALPHABET` (was 512), but the encode shader did NOT clamp symbols. When any subband group's value range exceeded 512, the encoder read out-of-bounds `shared_cumfreq` data, producing a corrupt bitstream.

Quality sweep confirmed the bug threshold:
- q=50 (qstep=0.73): max LL alphabet ~480 → OK
- q=55 (qstep=0.55): max LL alphabet ~640 → OVERFLOW → PSNR crash

CDF 9/7 wavelet has DC gain ~1.15^8 ≈ 3.06× after 4 levels, amplifying the LL subband range.

### Fixes Applied

1. **Symbol clamping** in `rans_encode.wgsl` — match histogram shader's clamp behavior
2. **MAX_GROUP_ALPHABET**: 512 → 2048 across all shaders (`rans_histogram.wgsl`, `rans_normalize.wgsl`, `rans_encode.wgsl`) and Rust host code (`rans_gpu_encode.rs`)
3. **HIST_TILE_STRIDE**: updated to 16401 = 1 + 8×(2+2048) in all files
4. **shared_cumfreq**: reduced from 7700 → 4096 entries (30.8KB → 16.4KB shared memory), improving GPU occupancy. Practical cumfreq sums (LL~2048 + 4 detail groups ~500 each) well under 4096.
5. **Quality presets**: 4 wavelet levels for q≥60, quality-dependent chroma weights, AQ disabled

### GPU Submission Batching

Batched the 3-plane (Y/Co/Cg) wavelet transform + quantize into a single GPU command encoder submission:

- **Before**: 3 separate command encoders, 3 `queue.submit()` calls (one per plane)
- **After**: 1 command encoder with all 3 planes' dispatches sequentially, 1 `queue.submit()`
- CfL dependency (chroma needs reconstructed Y) satisfied by sequential dispatch order within the encoder
- CfL alpha readback deferred: Co/Cg alphas copied to MAP_READ staging buffers within the encoder, read back after the single submit with one poll

I-frame GPU submissions reduced from 5 to 3:
1. Preprocess (color convert + deinterleave)
2. 3-plane wavelet + quantize (NEW: batched)
3. 3-plane rANS encode (already batched)

### Results — Quality Sweep (bbb_1080p, GPU per-subband rANS)

| Quality | PSNR (dB) | BPP  | Notes |
|---------|-----------|------|-------|
| q=40    | 35.57     | 1.85 | 3 levels, LeGall 5/3 |
| q=50    | 37.98     | 2.75 | 3 levels |
| q=60    | 40.39     | 3.77 | 4 levels, CDF 9/7, CfL, per-subband |
| q=70    | 43.01     | 5.21 | 4 levels |
| q=75    | 44.08     | 5.93 | 4 levels (default) |
| q=80    | 45.80     | 6.90 | 4 levels |
| q=90    | 51.88     | 10.33 | 4 levels |

All quality levels produce correct monotonically increasing PSNR.

### Results — Throughput (1080p, q=75, macOS Metal)

| Image | Encode | Decode | Decode (pipelined) |
|-------|--------|--------|--------------------|
| bbb_1080p | 31.8 ms (31.5 fps) | 29.1 ms (34.3 fps) | 28.9 ms (34.6 fps) |
| blue_sky_1080p | 31.7 ms (31.6 fps) | 27.0 ms (37.0 fps) | 27.4 ms (36.5 fps) |

Encode improved from 33.5ms → 31.8ms (5% speedup) due to GPU submission batching. Criterion roundtrip benchmark shows statistically significant -2.5% improvement.

### Files Modified
- `src/shaders/rans_encode.wgsl` — symbol clamping, shared_cumfreq 7700→4096
- `src/shaders/rans_histogram.wgsl` — MAX_GROUP_ALPHABET 512→2048, HIST_TILE_STRIDE→16401
- `src/shaders/rans_normalize.wgsl` — MAX_GROUP_ALPHABET→2048, HIST_TILE_STRIDE→16401
- `src/shaders/rans_decode.wgsl` — shared_cumfreq 7700→4096
- `src/encoder/rans_gpu_encode.rs` — Rust constants updated to match shaders
- `src/encoder/rans.rs` — per-group ZRL encoding for detail subbands (CPU path)
- `src/encoder/rans_gpu.rs` — ZRL integration for GPU decode path
- `src/encoder/pipeline.rs` — batched 3-plane command encoder, deferred CfL alpha readback
- `src/lib.rs` — quality preset: 4 levels, chroma weights, AQ disabled

### Analysis
- The alphabet overflow bug was latent since per-subband GPU encoding was introduced — it only manifested when `qstep` was small enough for the LL subband range to exceed 512
- Increasing MAX_GROUP_ALPHABET to 2048 handles all practical cases (CDF97 DC gain at 4 levels with typical 8-bit content)
- The shared_cumfreq reduction from 7700→4096 entries was key for recovering GPU performance — 30.8KB was nearly filling the 32KB threadgroup memory limit, crushing occupancy
- GPU submission batching provides incremental improvement; the GPU compute itself dominates latency

### Remaining known issues
- **AQ fundamentally broken**: spatial-domain weights applied to wavelet-domain positions. Needs redesign to compute weights per wavelet subband (e.g., from LL subband energy) rather than spatial variance
- **BPP still high**: 5.93 bpp at 44 dB for bbb_1080p. Target was ~1 bpp. Main opportunities: better ZRL on GPU path (currently CPU-only), tighter frequency tables, context modeling
- **local_decode_iframe**: still 3 separate submits (hard sequential dependency on plane_a buffer reuse)
