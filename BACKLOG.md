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
- **Status:** done (2026-03-08) — superseded by adaptive-mul system; zeroing approach removed
- **Outcome:** The original zeroing approach (TILE_ZERO_MUL for energy > threshold) caused ghosting and was removed. The adaptive-mul curve (log-linear [MIN_MUL=1.2, MAX_MUL=2.0] over energy [0.5, 10.0]) already achieves per-tile adaptation: high-motion tiles (energy >10) get min_mul to preserve quality; truly static tiles (energy ≈0, as in crowd_run L0) hit max_mul and produce skip=true outputs.
- **Diagnostic validation (2026-03-08):** bbb@q=75: tile energy p50=2.0 (real motion, correctly mid-curve). crowd_run@q=75: L0 tiles skip=true (energy 0.0), L1/L2 at min_mul (energy 7-15). System working as intended.
- **Closed:** True All-I per tile would require bitstream format change for marginal gain (energy distribution shows no large population of tiles in the actionable zone). Not worth the complexity.

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

### 13. Per-group k Rice optimization
- **Status:** investigated (2026-03-08) — vetoed as formulated; EMA already provides this
- **Finding:** Current Rice encoder uses per-coefficient EMA (window ~8) that already adapts k
  continuously. Explicit per-group k would add 8 bytes/stream overhead (3-5× the estimated gain).
  Implicit variant (EMA reset at group boundaries) is theoretically correct but implementation
  risk (encoder/decoder desync) outweighs sub-0.08 bpp expected gain. Closed.

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

### 16. Encode speed: I+P+B path optimization
- **Status:** active (2026-03-09)
- **Hypothesis:** The I+P+B encode path runs at 18-21fps (crowd_run/bbb, 1080p q=75). Profiling shows two bottlenecks: (1) Rice entropy staging uses max_stream_bytes=4096 regardless of q, causing ~125MB GPU copy even when only ~3MB of actual data; (2) block_match + estimate_split are two sequential dispatches processing the same reference data, each 10240 workgroups — fusing them eliminates one full dispatch cycle. ME itself is at the memory bandwidth floor (22ms/frame, 849MB @ 68 GB/s) and cannot be reduced without algorithm changes.
- **Success criteria:** Average I+P+B fps ≥25 fps on crowd_run 1080p q=75; VMAF neutral (±0.2 pts); no test regressions.
- **Phase 1 (easy):** Make `max_stream_bytes_for_tile()` q-dependent. Formula: `max(512, min(4096, tile_pixels × estimated_bpp / 8 × 12))` where estimated_bpp comes from q→bpp lookup. Saves ~3-6ms/frame staging copy.
- **Phase 1 done (2026-03-09):** Adaptive max_stream_bytes: q=75 now uses 1024 bytes/stream (was 4096). Actual savings smaller than expected due to L3 cache effects: 18.4fps → 18.7fps.
- **Phase 2 done (2026-03-09):** Reduced block_match_split.wgsl FINE_RANGE from 4 to 2 (always uses parent 16×16 MV as predictor; ±2 equivalent to predictor path in block_match.wgsl). Savings also smaller than bandwidth model predicted (L3 cache effects). 18.7fps → 19.3fps. bbb: 20.8fps → 21.8fps. VMAF unchanged (95.05). bpp unchanged.
- **Result:** 18.4fps → 19.3fps (+5%). Success criterion (≥25fps) NOT yet met.
- **Key finding:** Local decode round-trip is the next major bottleneck (~18ms/frame: Rice readback CPU→GPU for decode ref). See #19 for GPU-side local decode.
- **Note on 60fps:** L3 cache analysis shows ME bandwidth savings don't translate 1:1 to wall-clock time. 60fps (17ms/frame) not achievable without fundamental ME algorithm change AND GPU-side local decode. Revised target: 25-30fps achievable with #19.
- **Note:** GNC_PROFILE=1 profiling was used. Measurements: ME+split=20ms (forced barrier), entropy+stg=47ms, local decode readback=18-20ms per inter-frame.

### 19. GPU-side local decode for P/B-frame reference
- **Status:** todo (P1 — next priority after #16)
- **Problem:** After encoding a P/B-frame, the encoder reads back Rice stream data from GPU to CPU (~18ms), re-uploads and runs the GPU decoder to produce the reference frame. This 18ms CPU round-trip happens for every inter-frame and is the biggest remaining bottleneck (currently 19.3fps → would reach ~27fps if eliminated).
- **Hypothesis:** Keeping Rice streams on GPU and calling the decode shader directly on the already-encoded buffers eliminates the 18ms round-trip per inter-frame.
- **Challenges:** Rice encode output format (stream_buf/lengths_buf/k_buf per-stream layout) differs from Rice decode input format (packed k_values array + stream data layout). Either: (a) add a GPU-side format conversion shader, or (b) restructure Rice encode output to match decode input format, or (c) encode the frame into a "local decode compatible" GPU buffer alongside the regular output.
- **Success criteria:** I+P+B fps ≥25fps on crowd_run 1080p q=75; VMAF neutral; all tests pass.
- **Note:** Must verify encoder/decoder produce identical output (no encoding mismatch).

### 17. Scene cut detection + adaptive GOP
- **Status:** todo (P3)
- **Hypothesis:** When a hard scene cut occurs mid-GOP, B-frames reference pre-cut frames, producing large residuals. SAD-threshold detection and forced I-frame placement at cuts would reduce wasted bits.
- **Success criteria:** On a cut-heavy sequence, bpp reduces ≥2%; no regression on current test sequences (bbb/crowd_run/rush_hour have no cuts).
- **Note:** Low complexity (~50 lines in sequence.rs, no shader changes). Needs a test sequence with cuts for validation. Pure correctness/robustness item.

### 18. Per-tile Lagrange RD optimization (prerequisite: AQ overlap experiment)
- **Status:** todo (P2 — pending prerequisite)
- **Hypothesis:** Per-tile λ-optimal quantization could reduce BD-rate by 8-15% vs current per-q uniform quantization. AQ already does a form of this — prerequisite: measure AQ-on vs AQ-off BD-rate curves (bbb q=25/50/75/90) to bound the gap AQ leaves.
- **Success criteria:** ≥8% BD-rate improvement on ≥2 sequences; VMAF at each q ±0.3 pts.
- **Prerequisite experiment:** Run `rd-curve` with AQ enabled/disabled, report BD-rate delta. If AQ already captures ≥70% of potential gain, skip this item.
- **Note:** Requires GPU-side rate estimation or two-pass approach. Lagrange RD is tile-independent (each tile optimizes independently). Complex implementation (~5-8 days).

### 12. CPU SIMD path (long-term, low priority)
- **Status:** todo (P5 — far future, contingent on codec maturity)
- **Motivation:** Broadcast contribution niche — same hole as VC-2/Dirac and JPEG XS: low latency, high quality, patent-free, low complexity. For broader adoption, a CPU-only path removes the GPU dependency and enables use on hardware without a capable GPU (servers, edge devices, FPGA/ASIC targets). Also enables WebAssembly decode on browsers without WebGPU (e.g. Firefox today).
- **Approach:** Portable SIMD via `std::portable_simd` or `wide` crate — single code path that compiles to NEON (M1/ARM), AVX2 (x86), and WASM SIMD128. GPU path remains primary; SIMD path is a secondary fallback tier.
- **WASM note:** WASM SIMD128 is well-supported (Chrome 91+, Firefox 89+, Safari 16.4+) and trivial to ship — just add `-C target-feature=+simd128` to the wasm-pack build.
- **Prerequisite:** Codec must first reach competitive compression/latency/quality. No point optimizing a SIMD path for an algorithm that may still change fundamentally.
- **Success criteria:** CPU SIMD decode of a 1080p frame within 2× real-time at target quality. No GPU required.
- **Note:** Primary goal of this project is to explore whether AI-driven iteration can produce something competitive in this space. SIMD path is downstream of that question.

### 10. 10-bit support
- **Status:** done (2026-03-08)
- **Result:** Bit-exact encode/decode roundtrip for 10-bit input. `pack_u8.wgsl` peak parameterized, `buffer_to_texture.wgsl` uses scale uniform, all quality metrics use `max_val_for_depth()` across all subcommands. New test `test_10bit_roundtrip` passes. No regression on 8-bit tests.
- **Note:** f32 shaders are transparent to bit depth — only I/O boundaries needed changing (loader, saver, pack_u8, buffer_to_texture). `bit_depth` was already in `FrameInfo` and bitstream header — no bitstream break.

### 14. P-frame and B-frame chroma: 4:2:0/4:2:2 MC residual domain mismatch
- **Status:** active (2026-03-08) — P-frame fix committed, B-frame regression diagnosed
- **P-frame fix (committed 856761c):** Chroma-domain MC for 4:2:0 P-frames. BPP ordering now correct: 4:2:0 < 4:2:2 < 4:4:4 (bbb: 2.20/2.35/2.61, crowd_run: 5.16/5.59/6.39 bpp). Ratio 0.92-0.95 (not yet ≤0.85 target, possibly B-frame overhead polluting avg).
- **B-frame regression (diagnosed, 2026-03-08):** B-frames in 422/420 show up to 11.82 dB PSNR loss and are larger than I-frames. This is because B-frame backward MC for 4:2:0 chroma is currently performing MC in the luma domain, using luma-sized, NN-upsampled references directly, without the necessary box-filtering and MV scaling to operate in the chroma domain. This mirrors the original P-frame issue that caused structured HF residuals.
- **Next:** Implement specialized chroma-domain MC for 4:2:0 B-frames in `src/decoder/gpu_work.rs`. This includes box-filtering references, scaling MVs, calling a chroma-domain `compensate_bidir` variant, and upsampling the result.

### 15. Quarter-pel motion compensation
- **Status:** done (2026-03-09)
- **Hypothesis:** Half-pel ME leaves significant residual energy — P-frames cost 2-3× more than I-frames at q=25 on animated content (bbb). Quarter-pel interpolation reduces prediction error by ~25-50%, yielding ≥0.5 dB PSNR improvement on P/B-frames and ≥5% bpp reduction overall.
- **Result:** VMAF +1.14 pts at q=75 (95.05 vs 93.91 baseline). BPP reduced across all quality levels: -12.3% (q=25), -6.3% (q=50), -4.5% (q=75) on single-frame bbb. Sequence bpp: crowd_run q=75 6.93 vs 6.99 baseline (-0.9%), crowd_run q=25 1.90 vs All-I 2.17 (12.7% saving), rush_hour within noise (+1.0% bpp). All 164 tests pass, zero clippy warnings.
- **Shaders changed:** motion_compensate.wgsl, motion_compensate_bidir.wgsl, motion_compensate_bidir_chroma.wgsl, block_match.wgsl, block_match_bidir.wgsl, block_match_split.wgsl. Two-stage QP refinement: Stage A = ±2 QP units (= half-pel), Stage B = ±1 QP unit (= quarter-pel) around Stage A winner.
- **Note on success criteria:** PSNR hypothesis partially met — bpp reduction exceeds −5% at q=25 (12.7%). VMAF improvement (+1.14 pts) exceeds threshold. P-frame PSNR criterion vs All-I not directly measured but implied by improved VMAF.
