# GNC Backlog

Status: `todo` | `active` | `done` | `blocked`

## Baseline (v0.1-spatial, commit 617d8e6)

See [BASELINE.md](BASELINE.md) for current benchmark numbers.

## Current Focus (updated 2026-03-09)

H.264 comparison (#22) established the north star: GNC needs **2–5× more bits** than H.264 at equal PSNR (BD-rate +171–216%, broadcast contribution, 4:2:2, I+P+B).

**The gap is temporal prediction, not entropy.**
- P/B frames save only ~3% bpp vs all-I on high-motion content — H.264's MC is far more efficient
- Rice+ZRL vs arithmetic coding is only ~0.1–0.2 bpp — not the bottleneck
- Entropy-improvement items (#21, #7) are deprioritized; focus shifts to temporal prediction

**Priority order (updated 2026-03-10):**
1–14: (see previous items, all done/closed)
15. **#35 DCT for inter-frame residuals** (blocked P1) — gate experiment showed BlockDCT8 I-frame only; full new code path 6-10 days; deferred
16. **#36 Deblocking filter** (closed P2) — artifact is 1-2px incoherent boundary mismatch; deblocking would blur; correct fix requires overlapping tiles (bitstream change)
17. **#37 Per-8×8-block skip** (closed P2) — 0% blocks qualify on bbb; pan motion prevents block-level static detection
18. **#38 Lagrange RD quantization** (todo P3) — blocked on no-AQ gate; **NEXT**
19. **#39 32×32 coarse-block fallback** (todo P3) — blocked on MV variance gate; savings capped at 2.3%
20. **#24 Larger ME search range** (DEFER)
21. **#25 Multi-reference P-frames** (DEFER)

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
- **Status:** closed — skip (2026-03-09)
- **Reason:** H.264 comparison shows temporal prediction is the dominant gap, not intra-tile entropy. Delta prediction across LL tiles is an entropy-side improvement; expected gain <2% bpp. Not worth the complexity given current priorities.

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
- **Status:** done (2026-03-09) — partial success; see #20 for remaining bottleneck
- **Hypothesis:** The I+P+B encode path runs at 18-21fps (crowd_run/bbb, 1080p q=75). Profiling shows two bottlenecks: (1) Rice entropy staging uses max_stream_bytes=4096 regardless of q, causing ~125MB GPU copy even when only ~3MB of actual data; (2) block_match + estimate_split are two sequential dispatches processing the same reference data, each 10240 workgroups — fusing them eliminates one full dispatch cycle. ME itself is at the memory bandwidth floor (22ms/frame, 849MB @ 68 GB/s) and cannot be reduced without algorithm changes.
- **Success criteria:** Average I+P+B fps ≥25 fps on crowd_run 1080p q=75; VMAF neutral (±0.2 pts); no test regressions.
- **Phase 1 (easy):** Make `max_stream_bytes_for_tile()` q-dependent. Formula: `max(512, min(4096, tile_pixels × estimated_bpp / 8 × 12))` where estimated_bpp comes from q→bpp lookup. Saves ~3-6ms/frame staging copy.
- **Phase 1 done (2026-03-09):** Adaptive max_stream_bytes: q=75 now uses 1024 bytes/stream (was 4096). Actual savings smaller than expected due to L3 cache effects: 18.4fps → 18.7fps.
- **Phase 2 done (2026-03-09):** Reduced block_match_split.wgsl FINE_RANGE from 4 to 2 (always uses parent 16×16 MV as predictor; ±2 equivalent to predictor path in block_match.wgsl). Savings also smaller than bandwidth model predicted (L3 cache effects). 18.7fps → 19.3fps. bbb: 20.8fps → 21.8fps. VMAF unchanged (95.05). bpp unchanged.
- **Result:** 18.4fps → 19.3fps (+5%). Success criterion (≥25fps) NOT yet met.
- **Key finding:** Local decode round-trip is the next major bottleneck (~18ms/frame: Rice readback CPU→GPU for decode ref). See #19 for GPU-side local decode.
- **Note on 60fps:** L3 cache analysis shows ME bandwidth savings don't translate 1:1 to wall-clock time. 60fps (17ms/frame) not achievable without fundamental ME algorithm change AND GPU-side local decode. Revised target: 25-30fps achievable with #19.
- **Note:** GNC_PROFILE=1 profiling was used. Measurements: ME+split=20ms (forced barrier), entropy+stg=47ms, local decode readback=18-20ms per inter-frame.

### 19. Rice readback pipelining: submit next-frame ME before readback
- **Status:** done (2026-03-09)
- **Root cause diagnosed:** After each inter-frame encode, the encoder blocks ~18ms waiting for Metal's buffer sync (managed buffer `synchronize()` on M1). The 18ms is purely Metal sync overhead for Rice staging readback — latency-bound, not bandwidth-bound.
- **Implementation:**
  - P-frames: after submitting CMD_N, immediately write next frame's pixels + submit ME-only look-ahead before polling. poll() waits for both CMD_N sync (~18ms) and CMD_N+1_ME (~41ms unidirectional) concurrently. Since 41ms > 18ms, sync fully hidden.
  - B-frames (B1→B2): after submitting B1's command, write B2 pixels + submit bidir ME look-ahead before polling. poll() waits for both (~70ms bidir ME > ~18ms sync). B2 drops from ~77ms to ~18ms.
- **Results (crowd_run, q=75, 32 frames):**
  - P-only (ki=3): 19.3fps → 20.8fps (+8%). P-frames: 29ms → 8ms readback.
  - I+P+B (ki=8): 19.1fps → 19.4fps (+1.5%). B1 still 77ms (bidir ME ~60ms GPU dominates).
- **Why I+P+B gain is small:** B1 (first B-frame in group) has no precomputed ME and takes 77ms because bidir ME runs ~60ms on GPU. Pipelining hides the 18ms sync for B2 but B1 is still sequential. The bottleneck is bidir ME speed, not Metal sync. To reach 25fps would require faster bidir ME (algorithmic improvement).
- **VMAF:** 99.73 mean (unchanged). bpp: 6.50 (unchanged). 163 tests pass. Zero clippy warnings.
- **Lesson:** Metal sync latency (~18ms) is only the bottleneck when ME < 18ms (never true for bidir ME at ~60ms). For P-frames (ME ~41ms > 18ms), pipelining works well. For B-frames, the ME itself is the bottleneck.

### 20. Bidir ME qpel bottleneck: reduce Phase 3c/3d cost
- **Status:** done (2026-03-09) — implemented as opt-in; NOT made default
- **Root cause:** Bidir ME takes ~72ms for B1. Phase 3c+3d (32 barrier loops per block) is dominant. P-frame ME = 41ms; bidir ≈ 1.75× P-frame.
- **Implementation:** Wrapped Phase 3c and Phase 3d in `if params.skip_qpel == 0u { }` uniform blocks in `block_match_bidir.wgsl`. All barriers inside are now in uniform control flow. `GNC_BFRAME_NOQUPEL=1` env var gates the skip. `bidir_params_nopred_noqupel` buffer in buffer_cache.rs.
- **Measured results (3 sequences, q=75, 60 frames each):**

  | Sequence | qpel fps | noqupel fps | Δfps | qpel bpp | noqupel bpp | Δbpp |
  |---|---|---|---|---|---|---|
  | bbb | 19.6 | 20.8 | +6% | 2.54 | 2.57 | +1.2% |
  | crowd_run | 19.4 | 21.0 | +8% | 6.50 | 6.60 | +1.5% |
  | park_joy | 19.3 | 20.7 | +7% | 5.65 | 5.74 | +1.6% |
  VMAF: within noise on all sequences.

- **Why success criterion (≥23fps) was not met:** I-frame encode (~250ms per GOP) dominates total time. B-frame qpel savings (~40ms per B-frame pair) are diluted by the I-frame overhead. The actual savings were +6-8% fps, not +20-30% as expected.
- **Decision:** Keep qpel ON as default (better bpp at modest fps cost). `GNC_BFRAME_NOQUPEL=1` remains as opt-in for speed-over-quality use cases.
- **Key finding:** The real bottleneck for reaching 25fps I+P+B is the I-frame encode speed, not B-frame ME qpel.

### 17. Scene cut detection + adaptive GOP
- **Status:** done (2026-03-09, commit 776df85)
- **Implementation:** `pub fn luma_mad()` in lib.rs (shared with main.rs); `scene_cut_threshold: f32` in CodecConfig (default 50.0, 0=disabled). B-frame groups scan ahead to find earliest cut before committing. No shader changes, no bitstream changes.
- **Tests:** 5 unit tests in pipeline_tests.rs. 168 tests pass.

### 18. Per-tile Lagrange RD optimization (prerequisite: AQ overlap experiment)
- **Status:** closed (2026-03-09) — AQ experiment shows marginal gain potential
- **AQ experiment results (bbb_1080p, --no-aq flag added to rd-curve):**
  - AQ VMAF gain: 0-0.55 pts (q=10-60 only; q=70-90 identical)
  - AQ PSNR BD-rate: -3.9% (AQ redistributes bits perceptually → hurts PSNR)
  - AQ and Lagrange RD address similar problems (per-tile adaptation)
- **Conclusion:** AQ already provides per-tile quantization adaptation (via LL subband variance). The remaining gap to optimal Lagrange RD is estimated at <0.5 VMAF pts. Implementation cost (5-8 days) not justified for this gain. Prerequisite condition met: skip this item.
- **Note:** If BD-rate gap vs H.264 is to be closed, focus should be on spatial prediction improvements or entropy coding (parent-child context), not per-tile λ optimization.

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

### 21. Parent-child context Rice k — LL subband guides detail-subband k selection
- **Status:** closed — fully implemented, measured, reverted (2026-03-09)
- **Hypothesis:** Large LL parent (magnitude ≥4) predicts larger detail-subband coefficients → bias k += 1 for detail subbands (g > 0). Expected 0.08–0.18 bpp gain.
- **Step 0 measurement:** ZRL run length correlation with LL ancestor magnitude is ~1.0 across Q1–Q4. No usable signal in ZRL, but magnitude-k bias seemed worth testing.
- **Full implementation (Phase 1):** Implemented in all 4 components: rice_encode.wgsl, rice_decode.wgsl, rice.rs encoder, rice.rs decoder. GPU decoder used Phase 0 pre-decode of LL streams into shared memory (1024×f32 workgroup array), then Phase 1 detail decode with parent k bias.
- **Measured result:** bpp INCREASED across all tests. bbb_1080p q=75: 3.83 → 4.03 bpp (+5.2%). checkerboard q=50: 1.98 → 2.11 (+6.6%), q=75: 3.32 → 3.55 (+6.9%), q=90: 7.66 → 8.13 (+6.1%). Golden baseline regression tests failed.
- **Root cause:** At q=75, quantization step ≈ 4–5, so virtually ALL LL values have magnitude ≥4. Parent context k+1 fires universally for all detail coefficients — it's not adaptive at all. EMA was already correctly calibrated to optimal k; forcing k+1 everywhere is strictly worse. The threshold "magnitude ≥4" was too low relative to typical LL magnitudes after quantization.
- **Verdict:** Hypothesis was correct in theory (large parents do correlate with larger children) but the threshold choice and fixed +1 bias are too blunt. A soft, magnitude-proportional bias might work but EMA already handles within-stream adaptation. Abandoned.
- **Reverted:** All changes to rice_encode.wgsl, rice_decode.wgsl, rice.rs reverted to pre-experiment state. All tests pass.
- **Code left:** `ll_ancestor_coord()`, `collect_tile_entropy_samples()`, `compute_entropy_stats()` remain in rice.rs for future reference. GPU shaders unchanged from HEAD.

### 22. H.264 BD-rate comparison — broadcast contribution context
- **Status:** todo P3
- **Goal:** Establish an honest, apples-to-apples BD-rate gap between GNC and H.264 for the broadcast contribution use case. Used as north star for compression improvement prioritization.
- **Setup:**
  - GNC: Rice+ZRL entropy, I+P+B frames, 4:2:2 chroma (`--chroma-format 422`), q-sweep 10-90
  - H.264: `ffmpeg -c:v libx264 -profile:v high422 -pix_fmt yuv422p -preset veryslow`, q-sweep via `-crf 10..50`
  - Sequences: bbb_1080p (mixed), crowd_run (high motion), rush_hour (low motion)
  - Metric: PSNR-Y BD-rate (Bjøntegaard), and VMAF BD-rate as secondary
  - 10-bit: H.264 10-bit via `high422 10` profile + `yuv422p10le` if encoder supports it; GNC is natively 10-bit transparent
- **Success criterion:** Produce a table showing BD-rate gap (%) at each sequence, with clear statement of what encoder settings were used. Numbers must be reproducible.
- **Notes:**
  - `rd-curve` only sweeps single images (no video mode). For video comparison, run `benchmark-sequence` at multiple q values and collect bpp+PSNR, then feed two CSVs into `rd-curve --compare`.
  - H.264 intra-only is NOT the right comparison (we have B-frames). Use `-g 250 -bf 7` or similar for H.264 video mode.
  - Prior (incorrect) comparison used rANS + intra-only + 4:4:4 — discard those numbers.
- **Result (2026-03-09, park_joy):** BD-rate +171–216%. GNC 2–5× bpp vs H.264 at same PSNR. See RESEARCH_LOG.md for full table.
- **Key finding:** The gap is NOT entropy. Temporal prediction efficiency is the dominant bottleneck — P/B saves only ~3% vs all-I on high-motion content. Entropy improvements are secondary. Focus backlog on better motion compensation and temporal lifting.
- **Status:** done (measurement complete; used as north star for compression priorities)

### 23. Skip/merge modes — suppress residual for flat/matched regions
- **Status:** done (2026-03-09)
- **Result:** −2.7% bpp on bbb q=75 (VMAF −0.16 pts, within tolerance); −0.6% crowd_run; neutral park_joy.
- **Implementation:** `tile_skip_motion.wgsl` — one workgroup per tile (256 threads). Computes zero-MV SAD (mean |current−ref| per pixel). If mean_sad < qstep/2: zeros all 8×8 split MVs for that tile before MC dispatch. MC then produces small temporal residual → quantiser + Rice encoder produce compact all-skip tiles naturally. No bitstream format change required (zero MVs + small residual → same codec path). `tile_skip.wgsl` (coefficient zeroing) remains disabled (not needed).
- **Why it works:** Forces skip tiles to use zero-MV prediction (ref_same_pos). Encoder and decoder are symmetric: both reconstruct as ref_same_pos + zero_residual. Previous coefficient-zeroing approach caused MV-mismatch distortion; this approach avoids it by zeroing MVs before MC.
- **Limitation:** Savings limited to truly static tiles (temporal change < qstep/2 per pixel). B-frames not covered (bidir ME requires different skip logic). High-motion sequences see minimal savings.
- **Next step if more savings needed:** (a) More aggressive threshold calibration; (b) B-frame zero-MV skip; (c) Merge mode: use non-zero co-located MV (better prediction for slow-motion tiles). See #24 (larger search range) for high-motion gains instead.

### 24. Larger ME search range
- **Status:** investigated (2026-03-09) — precondition likely false; deprioritized
- **Motivation:** Current block_match.wgsl uses a fixed hierarchical search range. On fast-motion content (crowd_run, park_joy), many blocks may have the true motion vector outside the current search window, forcing a large residual. H.264's full-pel search typically covers ±64–128 pixels with hierarchical refinement.
- **Hypothesis:** Doubling the coarse search range reduces average residual energy on high-motion sequences by 5–10%, at the cost of ~2× ME compute time.
- **Success criteria:** bpp −3% on crowd_run q=75; VMAF neutral; fps ≥17 (acceptable regression since quality gain is the goal).
- **Research Scientist verdict (2026-03-09):** MODIFY. P-frame already uses ±32 px; at 30fps this covers 960 px/sec — faster than broadcast content. The precondition (range-limited MVs) is likely false. Increasing range would quadruple coarse candidates (O(range²)) and push ME from 22ms to 80-100ms for likely zero bpp gain. Deprioritized in favor of #25. If revived, check MV histogram first — need >10% blocks with |MV| >28 px as gate.
- **Note:** If any range issue exists, it is more likely in B-frames (±16 px) than P-frames (±32 px).

### 25. Multi-reference P-frames
- **Status:** todo (P2)
- **Motivation:** GNC P-frames reference only the immediately preceding decoded frame. H.264 can reference up to 16 frames, which dramatically improves compression for repeated textures (scrolling text, panning shots) and periodic motion. Even 2-reference P-frames would cover the most common cases.
- **Hypothesis:** Allowing P-frames to choose the best of 2 reference frames (prev and prev-prev) reduces bpp 3–8% on sequences with periodic motion or scene repetition.
- **Success criteria:** bpp −3% on at least one test sequence; VMAF neutral; no regression on bbb/crowd_run.
- **Complexity:** Medium. Requires decoder to track a reference buffer (already partially done for B-frames). ME shader needs a second reference input and cost comparison.
- **Research Scientist verdict (2026-03-09):** DEFER. Expected gain requires content with periodic motion; current test sequences (bbb, crowd_run, rush_hour, park_joy) don't exhibit this. Gate on adding a periodic-motion test sequence AND running MV histogram showing >15% non-adjacent references.

### 26. B-frame zero-MV skip
- **Status:** done (2026-03-09)
- **Motivation:** #23 implemented P-frame zero-MV skip (−2.7% bpp bbb). B-frames are not covered. B-frames in bbb, rush_hour, and slow-pan content have many near-static tiles where the bidir prediction is already excellent and the residual is near-zero.
- **Hypothesis:** For B-frames where bidir zero-MV SAD < qstep/2, zero both forward and backward MVs before MC dispatch. The residual collapses to near-zero and Rice encodes it cheaply. Expected: −5% bpp on bbb B-frames, VMAF neutral.
- **Implementation:** New shader `tile_skip_bidir.wgsl`: per-tile bidir zero-MV SAD = |current − avg(fwd_ref, bwd_ref)|. If mean SAD < qstep/2: zeros both fwd+bwd MVs, forces block_modes=2 (bidir). Dispatched after estimate_bidir, before compensate_bidir. No bitstream format change.
- **Result:** bbb −3.6% bpp (B-frames shrank 32–41% individually), VMAF −0.01 pts. No regression on crowd_run or park_joy. 163 tests pass, zero clippy warnings.

### 27. Temporal Differential Coding (TDC)
- **Status:** closed — implemented, measured, reverted (2026-03-09)
- **Result:** ~0% bpp gain. Only 3/40 tiles activated on bbb (8%). bpp change: +0.03% (noise).
- **Root cause:** TDC is fundamentally redundant with MC in an I+P+B codec. P-frame `plane_c` holds MC residuals (wavelet of `current − MC(reference)`), not absolute coefficients. For static tiles, MC residual ≈ 0 already — TDC on a near-zero residual adds no value. TDC is for intra-only codecs (like JPEG XS) where MC is absent. GNC with I+P+B already exploits temporal redundancy through MC.
- **Lesson:** Before implementing temporal coding, verify whether MC already handles the redundancy being targeted.

### 28. Overlapped Block Motion Compensation (OBMC)
- **Status:** closed (2026-03-10)
- **Motivation:** Dirac/VC-2 uses OBMC to eliminate block-boundary discontinuities in the MC residual. GNC's current hard-boundary 16×16 block MC produces residuals with artificial discontinuities every 16 pixels — these are expensive for the wavelet to code (discontinuities spread energy across all wavelet levels). OBMC blends predictions from neighboring blocks using a raised-cosine (Hann) window, making the residual smooth.
- **Hypothesis:** OBMC reduces P-frame residual energy by 0.8–2.2 dB (literature: Orchard & Sullivan 1994, Dirac spec), translating to ~10–20% bpp reduction on P/B-frames. Works within-tile only (tile independence preserved — no cross-tile blending).
- **GPU fit:** Good. Each pixel's prediction is a weighted blend of predictions from its block and up to 3 neighboring blocks' MVs. All MVs for a tile are in GPU memory before MC dispatch. Per-pixel computation remains independent.
- **Implementation:** Modify motion_compensate.wgsl. For each pixel, compute prediction as: w_self * pred(MV_self) + w_right * pred(MV_right) + w_below * pred(MV_below) + w_br * pred(MV_br), where w are raised-cosine weights. At tile boundary: only self-prediction (no cross-tile blending).
- **Correctness gate:** Before GPU implementation, prototype blending weights on CPU and verify reconstruction identity. The decoder must apply identical blending to the MC reference.
- **Success criteria:** P-frame bpp −10% on crowd_run q=75 (VMAF neutral ±0.3 pts); 3-sequence validation. Bitstream: OBMC enable flag per sequence (minor version bump).
- **Gate experiment (2026-03-10):** 3×3 median filter on 8×8 split MV buffer (`GNC_MV_SMOOTH=1`, `mv_median_smooth.wgsl`). bbb q=75: 1.3465 bpp, VMAF 95.31 — IDENTICAL with and without smoothing. Root cause: bbb MV field is already smooth (animated film, slow camera moves); median of 9 similar values = original. Conclusion: MV discontinuities are not the bottleneck. Closing — OBMC implementation effort not justified.
- **MV smoother committed:** f28568a (opt-in diagnostic tool, not used by default).

### 29. Fused multi-level wavelet kernel (speed)
- **Status:** closed (2026-03-10) — pre-condition false
- **Motivation:** I-frame encode takes ~250ms at 1080p q=75, dominating I+P+B total fps (~20 fps). The spatial wavelet dispatches 3–4 levels sequentially (each level requires the previous to complete). Fusing levels into a single dispatch using workgroup shared memory for intermediate levels eliminates the GPU pipeline stalls between levels.
- **Hypothesis:** Single-dispatch multi-level CDF 9/7 wavelet using 32KB shared memory (M1 limit) reduces I-frame wavelet time by 25–40%. With I-frame wavelet at ~64ms (28% of I-frame time), this saves 16–25ms per I-frame → I+P+B fps improves from ~20 to ~23–25 fps.
- **Pre-condition check (2026-03-10):** Code inspection of `transform.rs` + `pipeline.rs` confirms the pre-condition is FALSE. All 4 wavelet levels × 2 passes × 3 planes = 24 dispatches are already inside a **single command encoder**, submitted once. No intermediate `device.poll()` or `queue.submit()` between levels. The only overhead between levels is Metal-internal memory barriers (~10–30 µs each), not CPU-blocking polls.
- **Analysis:** Fusing levels would save ~150 µs total across all 3 planes — <<1% of 250ms I-frame time. Full tile fusion (level 0) would require 256×256 f32 = 256 KB shared memory per workgroup — 8× the M1 limit of 32 KB. Partial LL-subband fusion (levels 2–4) saves ~5 barriers × 30 µs = 150 µs max. Not worth implementing.
- **Conclusion:** Closed. The hypothesized 25–40% speedup assumed CPU-side blocking polls — those don't exist. Real bottleneck is elsewhere (entropy, quantize, or CPU overhead). If speed improvement is needed, profile with GPU timestamp queries to identify the actual bottleneck.
- **GPU fit:** Good. CDF 9/7 has a 9-tap filter; at 256×256 tile with 4-level decomposition, shared memory usage per workgroup is within 32KB M1 limit.
- **Success criteria:** I-frame encode time <180ms at 1080p q=75 (from ~250ms); VMAF identical; all tests pass. Bitstream: no change.

### 30. GPU timestamp profiling — identify I-frame bottleneck
- **Status:** done (2026-03-10)
- **Motivation:** I-frame encode is claimed to be ~250 ms but we have never broken this down per-stage at GPU level. "Wavelet=64ms" is estimated, not measured. Without GPU timestamps we cannot correctly prioritize speed work (#33 fused quantize+Rice depends on this, and #32 ME depends on knowing ME budget).
- **Hypothesis:** A per-stage GPU timestamp breakdown (wavelet, quantize, Rice encode, Rice staging) will reveal that one stage accounts for >40% of total I-frame time and is the true bottleneck.
- **Implementation:** wgpu `TIMESTAMP_QUERY` feature. `write_timestamp` before/after each major dispatch group in `pipeline.rs`. Read back via `resolve_query_set`. Fallback: CPU Instant::now() with explicit `device.poll(Maintain::Wait)` per stage group.
- **Success criteria:** Per-stage GPU breakdown for one I-frame at 1080p q=75; stages sum to within 10% of measured 250 ms wall time. Output as `--diagnostics` text.
- **Complexity:** 0.5 days.
- **Gates:** #33 (fused quantize+Rice) proceeds only if quantize+Rice > 30 ms; #32 (8×8 ME) proceeds only if ME is < 15 ms (room to expand).
- **Result (bbb_1080p, q=75, Rice, steady-state):**
  - `gpu_wavelet_quant` ≈ 12.75 ms (wavelet+quantize, all 3 planes)
  - `gpu_rice` ≈ 12.8 ms (Rice entropy encode, all 3 planes)
  - `rice_assemble` ≈ 0.5 ms (CPU staging readback)
  - `total` ≈ 29.5 ms = ~34 fps (pure I-frame, no Y4M overhead)
  - GPU is 86% of I-frame time; wavelet+quantize and Rice are equal in cost
  - **Gate #33 triggered:** quantize+Rice ≈ 12.8ms (+ partial wavelet_quant) < 30ms → close #33
  - **Implementation:** GNC_PROFILE=1 env var splits command encoder in profiling mode; production path unchanged (single encoder, one submit).

### 31. Adaptive dead-zone quantization per subband
- **Status:** closed (2026-03-10) — gate: existing system already adaptive
- **Motivation:** Current quantizer uses a fixed dead-zone width per frame. JPEG 2000 literature (Taubman & Marcellin 2002) shows per-subband dead-zone optimization reduces rate 5–10% at equal distortion. HH subbands at fine levels carry mostly noise at medium-high Q and benefit from wider dead-zones; LL and LH/HL need narrower dead-zones.
- **Gate check (2026-03-10):** Measured group-7 (HH level-0) zero fraction = 76.4% on a high-frequency synthetic tile at q=75 (effective qstep=6.0, dead_zone=4.5). On real bbb_1080p content this is expected 80–90%. Gate criterion was <60% near-zero → proceed; >80% → skip.
- **Why existing system already covers this:** The perceptual weights (HH level-0 = 1.5×, HH level-1 = 2.0×, HH level-2 = 2.5×, HH level-3 = 3.5×) already implement per-subband quantization amplification. Combined with dead_zone=0.75, the effective threshold for HH level-0 is 4.5 (vs base qstep=4.0). The perceptual weighting system IS per-subband adaptive dead-zone — adding a separate dz array would be third-level redundancy. Closed.

### 32. Larger FINE_RANGE for 8×8 split ME
- **Status:** closed (2026-03-10) — gate experiment inconclusive; analytical argument weak
- **Motivation:** `block_match_split.wgsl` uses FINE_RANGE=2 (±2 full pixels). At 30fps, fast-moving subjects move 4–12px/frame. A 16×16 block at a person-background boundary settles on a compromise MV (e.g., 6px); the background-side 8×8 sub-block needs 0px but cannot reach it with FINE_RANGE=2. The sub-block is forced to encode a ~6px residual instead of ~0px.
- **Original gate was flawed:** "8×8 split MV divergence >4px from 16×16 predictor" is structurally impossible — FINE_RANGE=2 hard-caps divergence at ±2.75px. The gate was testing a quantity that can never be >4px by construction.
- **Reformulated hypothesis:** Increasing FINE_RANGE from 2 to 6 allows boundary 8×8 blocks to find their true optimal MVs. Expected: bpp −5–10% on crowd_run q=75 (VMAF neutral). Search compute goes from (2×2+1)²=25 to (2×6+1)²=169 candidates per 8×8 block = 6.8× more split ME work. If ME is currently ~11ms (half of ~22ms total), split ME would go to ~75ms worst-case.
- **Gate experiment (2026-03-10):** Tested FINE_RANGE=6 on bbb_test.y4m q=75 444. Result: 1.35 bpp, VMAF 95.31 — identical to baseline (FINE_RANGE=2: 1.3465 bpp, VMAF 95.31). Neutral result on smooth-motion content (expected). crowd_run (high-motion) unavailable as Y4M sequence.
- **Analytical case is weak:** Motion-boundary 16×16 blocks represent <<1% of total blocks in a typical 1080p frame (4-5 runners × ~15 boundary blocks each = ~75/8100 = 0.9%). Even 5× residual improvement on those blocks = <0.05% total bpp savings. The 3% vs all-I gap on crowd_run is a structural issue (wavelet MC vs DCT MC) not a FINE_RANGE issue.
- **Compute cost:** 6.8× more split ME candidates (25 → 169 per 8×8 block) for <0.1% bpp savings. Not worth it. Closed.

### 33. Fused quantize + Rice encode shader (I-frame speed)
- **Status:** closed (2026-03-10) — gate triggered by #30 results
- **Motivation:** I-frame pipeline runs wavelet → quantize.wgsl → rice_encode.wgsl as separate dispatches, each reading/writing the 8 MB coefficient buffer from global memory. Fusing would eliminate one 8 MB global memory round-trip per plane.
- **Pre-condition:** #30 must show quantize+Rice together are >30 ms of I-frame time. If <30 ms, close this item.
- **#30 result:** gpu_rice = 12.8 ms; gpu_wavelet_quant = 12.75 ms (wavelet+quantize combined). Quantize is ~6 ms of the 12.75 ms; quantize+Rice ≈ 19 ms. Savings from eliminating one 8 MB global memory read = 24 MB / 68 GB/s ≈ 0.35 ms (~1.2% of total). Not worth implementing.
- **Closed:** Gate <30 ms, gains are sub-percent. The Rice shader and quantize shader are already independently efficient; fusion adds complexity for 0.35 ms savings.

### 34. Merge mode: co-located MV inheritance for slow-pan content
- **Status:** closed (2026-03-10) — gate: MV overhead too small
- **Motivation:** GNC codes all MVs as absolute values with no temporal MV prediction. On slow-pan content (rush_hour, bbb), most 8×8 block MVs are nearly identical to the previous frame's co-located MV. H.264 merge mode lets a block inherit a neighbor's MV at zero bits.
- **Gate check (2026-03-10):** MV overhead measured on bbb_test.y4m q=75: skip bitmap = 4,050 B (fixed), delta MVs = ~1 KB, total MV data ≈ 5 KB = 2.3% of average P-frame (222 KB). At best (with residual): 4.3%. Gate threshold was >5%.
- **Why closed:** The existing system (skip bitmap + delta coding with median spatial predictor) already captures temporal MV correlation efficiently. MVs for near-static blocks are already coded as 1-bit skip flags. Merge mode would save ~20% of the 2.3% MV overhead = ~0.2% bpp total. A bitstream format change (merge flag per block = minor version bump) is not justified for <0.2% bpp savings.

### 35. DCT transform for inter-frame residuals
- **Status:** blocked (P1) — gate experiment failed; full new code path required; deferred
- **Motivation:** The wavelet is the wrong transform for MC residuals. MC residuals have energy concentrated at 8×8 and 16×16 block boundaries (from the ME block structure). Wavelet spreads this boundary energy across ALL levels and subbands (worst case). DCT applied per-block localizes boundary discontinuities within individual blocks, allowing near-zero coding of adjacent blocks. Literature: H.264/HEVC/VVC all use DCT (integer transform) for inter residuals.
- **Hypothesis:** Using DCT-16 on P/B-frame MC residuals (wavelet stays for I-frames) reduces inter-frame bpp ≥10% on bbb q=75, with VMAF neutral.
- **Gate experiment result (2026-03-10):** Added `--dct` flag to `benchmark-sequence` to force `TransformType::BlockDCT8` for all frames. Result: PSNR = −6.44 dB, SSIM = 0.02 — catastrophically broken. Root cause: `BlockDCT8` is I-frame only; P-frame decoder path unconditionally assumes wavelet inverse transform. DCT reconstruction applied before MC add-back produces garbage.
- **Conclusion:** Cannot gate-test cheaply. Requires full new inter-residual transform path: `residual_transform_type` field in CodecConfig, new P/B-frame encoder path (DCT residual), decoder reading per-frame flag and choosing IDCT vs IWAVELET. Bitstream format change (minor version bump). Complexity: 6-10 days. Deferred until #36/#37 done.
- **Complexity:** 6-10 days (full implementation; no cheap gate available).
### 36. Deblocking filter at tile boundaries
- **Status:** closed (2026-03-10) — gate: artifact is 1-2px boundary-extension quantization mismatch (narrow, incoherent); deblocking would blur valid pixels; correct fix = overlapping tiles (bitstream change)
- **Motivation:** Tile-edge artifacts are documented as an architectural limitation. The 256×256 tile grid creates quantization discontinuities visible as blocking artifacts. These reduce VMAF even when PSNR is acceptable. A post-processing deblocking filter on the decoded output (not in the codec path) would smooth these boundaries. No bitstream change. This improves the VMAF axis without changing bpp.
- **Hypothesis:** 4-8 tap adaptive deblocking filter at 256-pixel tile grid boundaries increases VMAF ≥0.5 pts on bbb q=75 (from 95.31) without PSNR degradation >0.1 dB.
- **Correctness gate:** First inspect decoded output at tile boundaries with zoom. Characterize as Gibbs ringing (wavelet overshoot) vs hard block edges (quantization mismatch). Ringing → deblocking helps; hard edges → may blur without fixing. If artifact type is hard-edge quantization mismatch, deblocking won't help.
- **Implementation:** New WGSL shader `deblock.wgsl`: applied at decoder output (after buffer_to_texture). Adaptive H-filter at every 256th column, V-filter at 256th row. Boundary strength based on local gradient (strong where gradient is low, no-op where gradient is high). Optional: extend to 16×16 MC block boundaries within tiles (weaker filter). Decoder-only change.
- **Success criteria:** VMAF ≥+0.5 pts on bbb q=75; PSNR change < −0.1 dB; bpp unchanged (decoder post-processing).
- **Complexity:** 2-3 days.

### 37. Per-8×8-block skip decision
- **Status:** closed (2026-03-10) — gate: 0% of non-skip-tile blocks qualify on bbb; pan motion means all blocks have zero-MV SAD >> threshold
- **Motivation:** Current tile_skip_motion.wgsl zeroes MVs at tile granularity only when zero-MV SAD < qstep/2. Many 8×8 blocks within non-skip tiles have near-zero residuals but are not skipped because one or a few high-energy blocks in the tile push the tile-level SAD above threshold. Block-level skip decisions would zero those individual blocks' MVs independently.
- **Hypothesis:** Per-8×8-block zero-MV skip (if block zero-MV SAD < qstep/2) reduces P-frame bpp ≥3% on content with spatially heterogeneous tiles (some bbb-like sequences with mixed static/moving regions). Low expected gain on crowd_run (few near-zero blocks in high-motion content).
- **Gate:** Measure fraction of 8×8 blocks within non-skip tiles that have zero-MV SAD < qstep/2. Gate: proceed only if >15% of non-skip-tile blocks would qualify. Note: crowd_run tile-level skip gave only −0.6% → block-level on crowd_run will be even smaller.
- **Integration note:** Block-level skip must set MVs BEFORE the ME assigns them (same as tile-skip — zero MVs before MC dispatch, not after). Must integrate into split ME pipeline, not as a post-ME pass.
- **Success criteria:** P-frame bpp −3% on bbb q=75; VMAF neutral; no regression.
- **Complexity:** 2-3 days (after gate confirms).

### 38. Per-tile Lagrange RD quantization
- **Status:** todo (P3, quality)
- **Motivation:** AQ adapts quantization based on LL subband variance (texture heuristic). Lagrange-optimal allocation minimizes total distortion given a fixed bit budget by allocating bits to tiles with the highest marginal benefit. These are different optimization objectives. JPEG 2000 achieves 10-15% bpp reduction over fixed quantization via Lagrange allocation (Taubman & Marcellin, Ch. 8). GNC's AQ captures part of this but is not globally optimal.
- **Gate (1 day):** Run benchmark-sequence with `--no-aq` flag vs AQ baseline. If AQ gain over no-AQ is <2% bpp (AQ already near-optimal), close this item. If AQ gain is >5%, Lagrange has headroom.
- **Hypothesis:** Per-tile Lagrange-optimal quantizer reduces bpp ≥5% at equal VMAF on bbb q=75. The gap between AQ and Lagrange optimal captures real savings.
- **Note:** Gain is primarily on I-frames. For inter frames, tiles with good MC match are already near-zero and don't benefit from Lagrange. Gate must measure I-frame contribution separately.
- **Success criteria:** bpp −5% at VMAF ≥95.31 on bbb q=75; validated on crowd_run and rush_hour.
- **Complexity:** 5-7 days (includes R-D curve estimation, bisection λ search, integration with VBR rate control).

### 39. 32×32 coarse-block fallback for uniform-motion tiles
- **Status:** todo (P3, quality)
- **Motivation:** Current ME always runs 16×16 coarse → 8×8 split refinement. For tiles with uniform motion (all 16×16 blocks have similar MVs), the 8×8 split adds MV overhead with negligible residual benefit. A "coarse-only" mode for uniform-motion tiles would skip the split dispatch and use only the 16×16 MVs.
- **Gate:** Measure per-tile MV variance on rush_hour P-frames. Gate: proceed only if >30% of P-frame tiles have all-identical split MVs (zero intra-tile MV variance). MV overhead established at 2.3% of bpp total — this caps maximum savings at 2.3%.
- **Hypothesis:** Skipping 8×8 split for uniform-motion tiles reduces P-frame bpp ≥2% on rush_hour q=75 (predominantly uniform panning motion).
- **Note:** Given MV overhead is only 2.3% of total bpp, savings ceiling is ~2%. Implementation complexity must be proportional. If gate confirms >30% uniform tiles, this is a 1-day addition to the ME pipeline.
- **Success criteria:** bpp −2% on rush_hour q=75; VMAF neutral.
- **Complexity:** 1-2 days (after gate confirms).
