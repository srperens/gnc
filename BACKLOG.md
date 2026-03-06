# GNC Backlog

Status: `todo` | `active` | `done` | `blocked`

## Baseline (v0.1-spatial, commit 617d8e6)

See [BASELINE.md](BASELINE.md) for current benchmark numbers.

## Items

### 1. Fix temporal Haar per-tile adaptive mul
- **Status:** done (2026-03-06)
- **Root cause:** Two bugs — (1) `map_energy_to_mul` threshold miscalibrated (all real energies clamped to floor), (2) floor was 0.8 making highpass finer than lowpass (backwards)
- **Fix:** Recalibrated curve [1.0, max_mul] with log-linear interpolation, thresholds 0.5-10.0
- **Result:** rush_hour -37% bpp, crowd_run -4% bpp, stockholm +7% bpp (needs per-tile mode)

### 2. Per-tile temporal mode selection
- **Status:** todo
- **Problem:** Entire frame uses same temporal mode; static tiles waste bits with full spatial encode
- **Success criteria:** Tiles with low motion energy use Haar, high motion use All-I; measurable bpp improvement on mixed-motion content
- **Depends on:** #1

### 3. Encode performance -> 60 fps
- **Status:** active (partial — 2026-03-06)
- **Problem:** 15-16 fps temporal Haar end-to-end; PNG decode dominates benchmark
- **Progress:** Batch GPU syncs: 15 → 3 blocking polls/GOP. Pure encode: ~306ms → ~252ms/GOP (~31.7fps). CfL disabled for temporal highpass (unreliable on residuals).
- **Remaining bottleneck:** Unknown — need per-stage profiling of `encode_temporal_wavelet_gop_haar` before next optimization
- **Next step:** Add per-stage Instant timers (spatial wavelet, temporal Haar, Rice, readback) then tackle #4
- **Success criteria:** >= 40 fps pure encode on crowd_run 1080p q=75 (stretch: 60 fps)

### 4. Tile size experiment
- **Status:** todo (next after per-stage profiling)
- **Problem:** M1 GPU underutilized at 40 tiles/plane with 256x256 tiles; 128x128 gives 4× more workgroups
- **Hypothesis:** 128×128 tiles reduce GPU idle time in light temporal kernels, saving 15-30% per-GOP encode time. Counter-hypothesis: 4× more dispatch calls add CPU overhead, making it neutral.
- **Success criteria:** ≥15% reduction in pure encode time/GOP, no bpp/PSNR change. Failure: <5% change.

### 5. 4:2:0 + 10-bit
- **Status:** todo
- **Problem:** 4:4:4 wastes chroma bits; 8-bit limits HDR content
- **Success criteria:** Working encode/decode, expected 15-25% bpp improvement
- **Note:** Breaking bitstream change, do both together

### 6. CfL in temporal mode
- **Status:** done (Phase 4)
- **Note:** Completed in temporal wavelet Phase 4

### 7. LL subband prediction
- **Status:** todo
- **Problem:** Delta prediction between adjacent LL tiles could reduce redundancy
- **Success criteria:** Skip if < 2% bpp gain on benchmark suite
- **Note:** Low priority, may not be worth the complexity

### 8. Rate control
- **Status:** todo
- **Problem:** No constant bitrate mode
- **Success criteria:** CBR mode with < 5% bitrate overshoot on 10s windows
