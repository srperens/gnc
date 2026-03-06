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
- **Status:** active (2026-03-06) — partial implementation, quality validation pending
- **Problem:** Entire frame uses same temporal mode; static tiles waste bits with full spatial encode
- **Success criteria:** Tiles with low motion energy use Haar, high motion use All-I; measurable bpp improvement on mixed-motion content
- **Depends on:** #1
- **Done:** GPU tile_energies readback (raw mean_abs per tile, binding 4 in TER shader). Per-tile zeroing in Pass B weight map: tiles with energy > 12.0 → TILE_ZERO_MUL=1000 → all coefficients quantized to zero.
- **Result:** crowd_run -38% bpp, bbb/rush_hour 0% change (no hot tiles). Quality not validated — zeroing = temporal blur (LL average only), not true All-I.
- **Next:** Visual quality validation. True All-I per tile requires bitstream format change (per-tile mode flag). Consider TILE_ENERGY_ZERO_THRESH tuning (12.0 is aggressive; 15-20 might be better). See ADR for struct layout verification lesson (0003).

### 3. Encode performance -> 60 fps
- **Status:** active (2026-03-06)
- **Per-stage breakdown** (crowd_run 1080p q=75 GOP=8, steady-state, after GPU TER):
  - high_enc (7 high frames, batch Rice): 88-130ms (variable, ~45%) — mostly GPU compute
  - spatial_wl (8 frames CDF 9/7): ~64ms (28%)
  - upload (write_buffer 8 frames): ~22ms (10%)
  - low_enc: ~18ms (8%), temporal_haar: ~15ms (7%), aq_readback: 4.2ms (2%)
  - **TOTAL: ~215-232ms/GOP = 35-37 fps pure encode**
- **Done: GPU-side tile energy reduction** — aq_readback 34ms → 4.2ms; struct layout bug fixed
- **Done: async upload pipelining** — write next GOP's frames during high_enc (~100ms GPU); upload=0ms steady state
  - Result: 195-208ms/GOP → 39.2fps average (SPLIT total), some GOPs at 40+fps
- **Status:** ~done — averaging 39.2fps vs 40fps target. Remaining variance driven by high_enc content complexity.
- **Success criteria:** >= 40 fps pure encode on crowd_run 1080p q=75

### 4. Tile size experiment
- **Status:** done (2026-03-06) — hypothesis invalidated by architecture constraint
- **Result:** 128×128 tiles FAIL at runtime. GPU buffer binding limit: Rice encoder allocates `num_tiles × 256 streams × 4096 B` upfront. At 1080p: 135 tiles × 256 × 4096 = 134MB > WebGPU 128MB max_storage_buffer_binding_size. Not viable without Rice buffer refactor.
- **Learning:** Minimum viable tile size at 1080p with current Rice architecture is ~160×160 (84 tiles, ~88MB). Power-of-2 constraint and wavelet level requirements may prevent non-256 sizes anyway.
- **Next:** See #4b if smaller tiles are still desired after re-evaluating the Rice buffer strategy.

### 4a. Fix benchmark input: PNG → Y4M/YUV (prerequisite for #4)
- **Status:** done (2026-03-06)
- **Result:** `benchmark-sequence` now accepts `.y4m` input. Y4M I/O = 16% overhead (vs PNG 45%). End-to-end fps now 27.2 fps (vs 16.4), GNC-only fps 32.6 fps. Temporal Haar warmup bug also fixed.
- **Note:** Y4M files must be created from PNG sequences via ffmpeg. Xiph originals available at media.xiph.org/video/derf/

### 5. 4:2:0 + 10-bit
- **Status:** todo
- **Problem:** 4:4:4 wastes chroma bits; 8-bit limits HDR content
- **Success criteria:** Working encode/decode, expected 15-25% bpp improvement
- **Note:** Breaking bitstream change, do both together

### 6. CfL in temporal mode
- **Status:** done (Phase 4)
- **Note:** Completed in temporal wavelet Phase 4

### 4b. Rice MAX_STREAM_BYTES reduction (enables future tile experiments)
- **Status:** todo (P3 — do after speed target met)
- **Problem:** 4096 bytes/stream is worst-case ceiling; at q=75 actual max is ~320 bytes/stream. Current value blocks 128×128 tiles at 1080p (134MB > 128MB WebGPU limit).
- **Approach:** Measure actual stream byte distribution across benchmark suite, pick safe ceiling (e.g. 1024 bytes) with overflow guard. Validates/invalidates without touching encoding algorithm.
- **Success criteria:** No encode/decode regression; 128×128 tiles no longer fail at runtime.

### 7. LL subband prediction
- **Status:** todo (P3 — probably skip)
- **Problem:** Delta prediction between adjacent LL tiles could reduce redundancy
- **Success criteria:** Skip if < 2% bpp gain on benchmark suite
- **Note:** Low priority, may not be worth the complexity

### 8. Rate control
- **Status:** todo (P2 — needed before broadcast use)
- **Problem:** No constant bitrate mode
- **Success criteria:** CBR mode with < 5% bitrate overshoot on 10s windows
