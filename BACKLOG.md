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

### 5. 4:2:2 and 4:2:0 chroma subsampling
- **Status:** done (2026-03-06)
- **Result:** Full end-to-end encode/decode for 4:2:2 and 4:2:0. BPP reduction: 4:2:2 11-30%, 4:2:0 21-43% across bbb/blue_sky/touchdown. PSNR loss 0.2-5.6 dB (all-channel metric; perceptual loss is smaller). Four bugs found and fixed: wavelet uniform buffer slot aliasing, WGSL struct field order mismatch, missing chroma edge-replication padding, double entropy encoding for non-444. Nearest-neighbor upsample used; bilinear deferred to #9. 10-bit deferred to #10.

### 6. CfL in temporal mode
- **Status:** done (Phase 4)
- **Note:** Completed in temporal wavelet Phase 4

### 4b. Rice MAX_STREAM_BYTES reduction (enables future tile experiments)
- **Status:** done (2026-03-08)
- **Result:** MAX_STREAM_BYTES now tile-size-dependent: 1024 for ≤128px tiles (64 symbols/stream,
  7× headroom), 4096 for 256px tiles (unchanged). Overflow guard fixed: was silent byte-drop,
  now GPU atomicStore + CPU panic with message on overflow. 128×128 tiles at 1080p now work
  (135 × 256 × 1024 = 33MB < 128MB WebGPU limit). New test: test_rice_128x128_tiles.

### 7. LL subband prediction
- **Status:** todo (P3 — probably skip)
- **Problem:** Delta prediction between adjacent LL tiles could reduce redundancy
- **Success criteria:** Skip if < 2% bpp gain on benchmark suite
- **Note:** Low priority, may not be worth the complexity

### 8. Rate control
- **Status:** done (2026-03-08)
- **Implementation:** Virtual buffer model (R-Q model + VBV, online least-squares fit)
  wired into temporal wavelet benchmark-sequence path. I+P+B path was already wired.
- **Result:** Temporal wavelet CBR at 10–20 Mbps converges to <0.1% target deviation
  by GOP 8. 10s steady-state window <1% deviation. Success criterion (<5%) met.
- **Notes:** 2 Mbps at 1080p hits codec minimum (qstep ceiling); expected for high-compression
  targets. First 2s startup transient excluded from criterion. Use:
  `benchmark-sequence --temporal-wavelet haar --bitrate 10M --rate-mode cbr`

### 11. VMAF integration — standard quality metric across all benchmarks
- **Status:** done (2026-03-06)
- **Problem:** VMAF is only available on `benchmark-sequence --vmaf`. Single-frame `benchmark` and `rd-curve` commands lack VMAF. Chroma subsampling (4:2:2/4:2:0) and future quality improvements cannot be properly validated without perceptual metrics. PSNR missed the TILE_ENERGY_ZERO_THRESH ghosting bug entirely; VMAF caught it.
- **Approach:**
  1. Add `--vmaf` flag to `benchmark` (single-frame): encode → decode to temp PNG → Y4M → vmaf CLI → report score
  2. Add `--vmaf` flag to `rd-curve`: include VMAF column alongside PSNR and bpp
  3. Run VMAF baseline on current chroma variants (444 vs 422 vs 420 at q=50/75) and log in RESEARCH_LOG.md
  4. Document VMAF call convention in CLAUDE.md
- **Success criteria:** `cargo run -- benchmark -i ... -q 75 --vmaf` prints VMAF score. `rd-curve --vmaf` outputs VMAF column. Chroma 422/420 VMAF baseline logged.
- **Effort:** ~half day (libvmaf already at `/opt/homebrew/Cellar/libvmaf/3.0.0/bin/vmaf`; pattern exists in `benchmark-sequence`)

### 9. Bilinear chroma upsampling for 4:2:2 / 4:2:0
- **Status:** invalidated (2026-03-08) — experiment run, hypothesis disproven
- **Problem:** Current upsample shaders use nearest-neighbor.
- **Outcome:** Bilinear caused VMAF −0.93 pts and PSNR −0.60 dB regression at q=75. Reverted.
  Bilinear does NOT fix tile-edge artifacts (artifacts are inter-tile, not intra-tile).
  NN upsampling is correct for current architecture.
- **Kept:** dispatch cleanup (no dummy sentinel values), multi-tile tests for 422/420.
- **See:** RESEARCH_LOG.md 2026-03-08 entry for full analysis.

### 12. CPU SIMD path (long-term, low priority)
- **Status:** todo (P5 — far future, contingent on codec maturity)
- **Motivation:** Broadcast contribution niche — same hole as VC-2/Dirac and JPEG XS: low latency, high quality, patent-free, low complexity. For broader adoption, a CPU-only path removes the GPU dependency and enables use on hardware without a capable GPU (servers, edge devices, FPGA/ASIC targets). Also enables WebAssembly decode on browsers without WebGPU (e.g. Firefox today).
- **Approach:** Portable SIMD via `std::portable_simd` or `wide` crate — single code path that compiles to NEON (M1/ARM), AVX2 (x86), and WASM SIMD128. GPU path remains primary; SIMD path is a secondary fallback tier.
- **WASM note:** WASM SIMD128 is well-supported (Chrome 91+, Firefox 89+, Safari 16.4+) and trivial to ship — just add `-C target-feature=+simd128` to the wasm-pack build.
- **Prerequisite:** Codec must first reach competitive compression/latency/quality. No point optimizing a SIMD path for an algorithm that may still change fundamentally.
- **Success criteria:** CPU SIMD decode of a 1080p frame within 2× real-time at target quality. No GPU required.
- **Note:** Primary goal of this project is to explore whether AI-driven iteration can produce something competitive in this space. SIMD path is downstream of that question.

### 10. 10-bit support
- **Status:** todo (P3)
- **Problem:** Codec is 8-bit only; no HDR or high-fidelity camera content can be encoded without precision loss.
- **Success criteria:** Bit-exact encode/decode roundtrip for 10-bit input in [0, 1023]. No regression on existing 8-bit tests.
- **Note:** Infrastructure partially in place — `bit_depth` field already exists in `FrameInfo` and the bitstream header. Main work is widening internal buffers and shader arithmetic from u8/u16 to u16/u32 where needed.
- **Note:** Not a bitstream-breaking change if the `bit_depth` header field is already versioned correctly.
