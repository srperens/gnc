# GNC Backlog

Status: `todo` | `active` | `done` | `blocked`

## Baseline (v0.1-spatial, commit 617d8e6)

See [BASELINE.md](BASELINE.md) for current benchmark numbers.

## Current Focus (updated 2026-03-10)

H.264 comparison (#22) established the north star: GNC needs **2–5× more bits** than H.264 at equal PSNR (BD-rate +171–216%, broadcast contribution, 4:2:2, I+P+B).

**The gap is temporal prediction quality, not entropy or transform mismatch.**
- P/B frames save only ~3% bpp vs all-I on high-motion content — H.264's MC is far more efficient
- Rice+ZRL vs arithmetic coding is only ~0.1–0.2 bpp — not the bottleneck
- #35 (DCT for inter residuals) deferred by RS: OBMC gate (0% bpp change from MV smoothing) falsifies the "block-boundary energy" premise; realistic gain 2-5%, not the 10% criterion
- New focus: #43 multi-reference P-frames (gate first), then #42 hierarchical B-frames

**Priority order (updated 2026-03-10):**
1–14: (see previous items, all done/closed)
15. **#35 DCT for inter-frame residuals** (CLOSED — gate: LL 88% of residual energy, detail <15%; premise falsified)
16. **#40 4×4 sub-block ME** (CLOSED — bpp +15–26%; MV overhead dominates; hypothesis falsified)
17. **#41 Adaptive intra tiles in P/B frames** (CLOSED — gain < 1% at q=75; LL-dominant residuals = camera motion)
18. **#44 DC subband offset correction** (CLOSED — DC shift < 1% bpp)
19. **#45 Adaptive GOP** (CLOSED — ki=2 bpp worse than ki=8)
20. **#43 Multi-reference P-frames** (CLOSED — pyramid ME ±96px already covers gap)
21. **#42 Hierarchical B-frame GOP** (DONE 2026-03-10) — crowd_run −3.4% bpp, park_joy −3.9%, VMAF neutral
22. **#24 Pyramid ME** (done 2026-03-10) — ±96px range, park_joy −3.4% bpp

**Next items (RS-approved, 2026-03-10):**
1. **#46 LL subband spatial prediction** — CLOSED (gate fail: crowd_run mean_ratio=1.536, park_joy 1.705 > 0.85)
2. **#48 Chroma qpel for 4:2:0 B-frames** — CLOSED (gate fail: Co/Y=0.57, Cg/Y=0.43, both <1.2; chroma MC working correctly)
3. **#50 Fast I-frame Rice skip** — CLOSED (gate fail: 0% qualifying tiles at q=75)
4. **#49 P-frame reference from pyramid pool** — DONE (B₄-as-P forward-only; park_joy −0.2% bpp, crowd_run +0.3%, VMAF neutral)
5. **#47 Overlapping tile windows** — gate PASSED (0.66 dB boundary gap), correct design documented. Structural code in place (overlap=0 is no-op). Full implementation needs: FrameInfo.overlap_pixels + separate coefficient buffer + decoder crop step + bitstream bump.

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
- **Status:** done (2026-03-10) — all chroma bugs fixed
- **P-frame fix (committed 856761c):** Chroma-domain MC for 4:2:0 P-frames. BPP ordering now correct: 4:2:0 < 4:2:2 < 4:4:4 (bbb: 2.20/2.35/2.61, crowd_run: 5.16/5.59/6.39 bpp).
- **B-frame fix (2026-03-10):** Three bugs fixed across two sessions:
  1. `local_decode_bframe_to_pyramid_slot` used wrong buffer (co_plane vs ref_upload for Co) and luma dims for chroma. Fix: full 4:2:0 chroma-domain bidir inverse decode path.
  2. Pyramid slot gating: `copy_pyramid_slot_to_*` calls were gated on Yuv444. Fix: removed gate.
  3. **Root cause of B₂/B₃/B₅ garbage (25 dB):** `dispatch_mv_scale` in encode_bframe was called with `me_total_blocks` (8160) instead of `split_total_blocks` (32640). Encoder left entries 8160..32640 stale from previous B-frame; decoder zeroed them via OOB reads → mismatch. Fix: use `split_total_blocks` for both fwd and bwd MV scaling in B-frame 4:2:0 path (commit b9df1ec).
- **Final result (bbb 4:2:0):** All B-frames 36.08–36.37 dB PSNR, consistent. bpp 0.39–0.58 per B-frame.

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
- **Status:** done (2026-03-10) — pyramid ME implemented and always-on
- **Motivation:** Current block_match.wgsl uses a fixed hierarchical search range. On fast-motion content (crowd_run, park_joy), many blocks may have the true motion vector outside the current search window, forcing a large residual. H.264's full-pel search typically covers ±64–128 pixels with hierarchical refinement.
- **Hypothesis:** Doubling the coarse search range reduces average residual energy on high-motion sequences by 5–10%, at the cost of ~2× ME compute time.
- **Success criteria:** bpp −3% on crowd_run q=75; VMAF neutral; fps ≥17 (acceptable regression since quality gain is the goal).
- **Research Scientist verdict (2026-03-09):** MODIFY. P-frame already uses ±32 px; at 30fps this covers 960 px/sec — faster than broadcast content. The precondition (range-limited MVs) is likely false. Increasing range would quadruple coarse candidates (O(range²)) and push ME from 22ms to 80-100ms for likely zero bpp gain. Deprioritized in favor of #25. If revived, check MV histogram first — need >10% blocks with |MV| >28 px as gate.
- **Note:** If any range issue exists, it is more likely in B-frames (±16 px) than P-frames (±32 px).
- **2026-03-10 UPDATE — REOPENED:** crowd_run MV histogram shows 12-40% of P-frame blocks have |MV| > 17px, max_abs = 155-169px. ME_SEARCH_RANGE=32 cannot find these. P-frames cost 90-100% of I-frame on crowd_run. RS prior verdict was wrong (assumed 30fps, actual crowd_run MV is far larger than estimated).
- **Revised implementation — pyramid ME:**
  1. New `downsample_4x.wgsl` shader: 4× average downscale of current and reference Y-plane
  2. Run block_match at 4× resolution with ±24px range → covers ±96px at full resolution
  3. Scale pyramid MVs ×4 and use as initial predictor for full-res block_match (FINE_RANGE=4)
  4. Net range: ±96px at full resolution. Net compute: ~2× current ME (vs 9× for naive range increase)
- **Updated success criteria:** P-frame bpp −20% on crowd_run q=75; VMAF neutral; enc fps ≥15.
- **Complexity:** 2-3 days (pyramid downscale shader + pipeline integration).

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
- **Status:** CLOSED (2026-03-10) — gate fails: LL carries 85–88% of residual energy; detail subbands <15% (gate required >40%); premise falsified
- **Motivation:** The wavelet is the wrong transform for MC residuals. MC residuals have energy concentrated at 8×8 and 16×16 block boundaries (from the ME block structure). Wavelet spreads this boundary energy across ALL levels and subbands (worst case). DCT applied per-block localizes boundary discontinuities within individual blocks, allowing near-zero coding of adjacent blocks. Literature: H.264/HEVC/VVC all use DCT (integer transform) for inter residuals.
- **Hypothesis:** Using DCT-16 on P/B-frame MC residuals (wavelet stays for I-frames) reduces inter-frame bpp ≥10% on bbb q=75, with VMAF neutral.
- **Gate experiment result (2026-03-10):** Added `--dct` flag to `benchmark-sequence` to force `TransformType::BlockDCT8` for all frames. Result: PSNR = −6.44 dB, SSIM = 0.02 — catastrophically broken. Root cause: `BlockDCT8` is I-frame only; P-frame decoder path unconditionally assumes wavelet inverse transform. DCT reconstruction applied before MC add-back produces garbage.
- **RS verdict (2026-03-10):** DEFER. The OBMC gate experiment (0% bpp change from MV smoothing) directly contradicts the premise that block-boundary energy dominates inter residuals. VC-2 achieves H.264-class compression with wavelet inter residuals — the gap is prediction quality, not transform mismatch. Realistic gain estimate: 2-5% on bbb (smooth MV field), ~0% on crowd_run (chaotic residual). The 10% criterion is likely unachievable without block-boundary structure that the OBMC evidence suggests is absent.
- **Mandatory gate before reconsideration:** Measure P-frame residual subband energy distribution on crowd_run (1 day). If detail subbands carry >40% of residual energy, the hypothesis gains renewed support. If LL dominates, close.
- **Complexity:** 6-10 days (full implementation; bitstream format change required).
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
- **Status:** closed (2026-03-10) — gate: AQ uses +0.5–1.4% MORE bits than no-AQ (not saving bits); Lagrange exploitable gap <1.5% bpp; 5-7 day impl not justified
- **Motivation:** AQ adapts quantization based on LL subband variance (texture heuristic). Lagrange-optimal allocation minimizes total distortion given a fixed bit budget by allocating bits to tiles with the highest marginal benefit. These are different optimization objectives. JPEG 2000 achieves 10-15% bpp reduction over fixed quantization via Lagrange allocation (Taubman & Marcellin, Ch. 8). GNC's AQ captures part of this but is not globally optimal.
- **Gate (1 day):** Run benchmark-sequence with `--no-aq` flag vs AQ baseline. If AQ gain over no-AQ is <2% bpp (AQ already near-optimal), close this item. If AQ gain is >5%, Lagrange has headroom.
- **Hypothesis:** Per-tile Lagrange-optimal quantizer reduces bpp ≥5% at equal VMAF on bbb q=75. The gap between AQ and Lagrange optimal captures real savings.
- **Note:** Gain is primarily on I-frames. For inter frames, tiles with good MC match are already near-zero and don't benefit from Lagrange. Gate must measure I-frame contribution separately.
- **Success criteria:** bpp −5% at VMAF ≥95.31 on bbb q=75; validated on crowd_run and rush_hour.
- **Complexity:** 5-7 days (includes R-D curve estimation, bisection λ search, integration with VBR rate control).

### 39. 32×32 coarse-block fallback for uniform-motion tiles
- **Status:** closed (2026-03-10) — analytical: max savings 2.3% (MV overhead cap); rush_hour unavailable; 30% of 2.3% = 0.7% savings ceiling; not worth 1-2 day impl
- **Motivation:** Current ME always runs 16×16 coarse → 8×8 split refinement. For tiles with uniform motion (all 16×16 blocks have similar MVs), the 8×8 split adds MV overhead with negligible residual benefit. A "coarse-only" mode for uniform-motion tiles would skip the split dispatch and use only the 16×16 MVs.
- **Gate:** Measure per-tile MV variance on rush_hour P-frames. Gate: proceed only if >30% of P-frame tiles have all-identical split MVs (zero intra-tile MV variance). MV overhead established at 2.3% of bpp total — this caps maximum savings at 2.3%.
- **Hypothesis:** Skipping 8×8 split for uniform-motion tiles reduces P-frame bpp ≥2% on rush_hour q=75 (predominantly uniform panning motion).
- **Note:** Given MV overhead is only 2.3% of total bpp, savings ceiling is ~2%. Implementation complexity must be proportional. If gate confirms >30% uniform tiles, this is a 1-day addition to the ME pipeline.
- **Success criteria:** bpp −2% on rush_hour q=75; VMAF neutral.
- **Complexity:** 1-2 days (after gate confirms).

### 40. 4×4 sub-block ME for high-energy tiles
- **Status:** CLOSED (2026-03-10) — hypothesis falsified; reverted
- **Gate result (2026-03-10):** SAD ratio 16×16 vs 8×8: 4.45–4.96× (all PROCEED). Gate was wrong premise — low SAD ratio ≠ low MV overhead.
- **Validation result (2026-03-10):** bpp +15–26% on all 3 sequences (bbb +25%, crowd_run +16%, park_joy +26%). VMAF neutral (bbb −0.13 pts) or improved (+0.59/+0.60 pts). Clearly BLOCK.
- **Root cause:** 4×4 MVs quadruple MV entries (32,400 → 129,600 per frame). At ~4 bytes/MV, that's ~390 KB extra MV data per 1080p frame — 50–60% of current compressed frame size. Residual savings from finer MVs are far smaller. Design flaw: "always output 4×4 MVs" without RD gate is unconditionally harmful.
- **Lesson:** Smaller blocks require per-block RD split decision (compare saved residual vs MV cost) to be net positive. H.264 does this per macroblock. Without this gate, 4×4 ME always increases total bits. A future #40b could add an RD gate, but estimated effort is high (need 8×8 SAD fed into 4×4 shader) and #41/#42 are better priority.

### 41. Adaptive intra tiles in P/B frames
- **Status:** CLOSED (2026-03-10) — gain < 1% at q=75; not worth bitstream format change
- **Gate result (2026-03-10):** Nominally passed: near_zero=11–17%, ratio vs I-frame = 0.98–1.00 on some P-frames on crowd_run. But LL residual mean_abs_diff=40.69 vs LH/HL=2.37/2.73 → LL-dominant. This is camera/crowd motion shifting DC-level across tiles, not tile misprediction. Switching those tiles to intra doesn't save bits because intra also costs ~I-frame rate for high-motion tiles.
- **Gate gate:** max possible gain: tiles that switch save only the MV overhead (2.3% total). With ~10% of tiles qualifying, savings ≤0.23%. Revised upper bound: < 1% total bpp savings.
- **Lesson:** Per-tile intra mode requires bitstream format change AND per-tile mode decision complexity. Only worthwhile if gain is reliably ≥3%. On crowd_run the MC fails globally (per-frame ratio = 0.98–1.0), not per-tile. Fix needs better global MC, not per-tile intra fallback.

### 42. Hierarchical B-frame GOP (pyramid reference structure)
- **Status:** done (2026-03-10) — commit 4bddc59
- **Motivation:** GNC uses flat B-frame referencing (all B-frames reference the same I and P anchors). A pyramid structure (I B4 B2 B1 B3 B2 B1 P where B-frames at each level reference closer frames) reduces temporal distance for inner B-frames without adding I-frames. HEVC and VP9 both use this for high efficiency.
- **Gate experiment result (2026-03-10 — INVALID PROXY):** ki=5 (B-frames at ≤2 frame distance) = 6.51 bpp vs ki=8 (≤4 frame distance) = 6.45 bpp on crowd_run. ki=5 is WORSE because it forces more I-frames. This is not a valid proxy for hierarchical B-frames within a fixed-length GOP.
- **Conclusion:** Cannot gate via ki=N. Must implement partial hierarchical structure to measure benefit. RS hypothesis card needed.
- **Hypothesis (draft, needs RS approval):** Within ki=8 GOP, encode layer-1 B (frame 4) first, then layer-2 B (frames 2,6) referencing I+layer-1 and layer-1+P respectively, then layer-3 (frames 1,3,5,7). Temporal distance for layer-3 = 1 frame; layer-2 = 2 frames; layer-1 = 4 frames. Expected bpp −8% on bbb, −4% on crowd_run; VMAF neutral.
- **Success criteria (draft):** bpp −5% on bbb q=75; VMAF neutral ±0.3 pts; no regression on crowd_run or park_joy.
- **Complexity:** 4-6 days. Encoding order ≠ display order (layers encoded outer-to-inner). Decoder must buffer decoded frames at each layer as reference. Bitstream: per-B-frame reference indices (currently implicit, both = I/P anchors). Minor version bump required.
- **RS Hypothesis (approved 2026-03-10):**
  - Coding order: I₀ P₈ B₄ B₂ B₆ B₁ B₃ B₅ B₇ (outer-to-inner)
  - Layer 1: B₄ refs I₀, P₈ (distance 4)
  - Layer 2: B₂ refs I₀+B₄; B₆ refs B₄+P₈ (distance 2)
  - Layer 3: B₁B₃B₅B₇ refs adjacent layer-2 frames (distance 1)
  - Expected: bbb −5–10%, crowd_run −1–4% (chaotic motion limits gain)
  - Fail criterion: bbb < 2% → flat structure still active; crowd_run = 0% expected (OK)
  - Diagnostic required: print actual ref indices used per B-frame (GNC_BFRAME_PYRAMID=1)
  - Risk: reference frame index mgmt silently using wrong frame
- **Implementation summary (2026-03-10):** B_FRAMES_PER_GROUP changed 2→7 (group size 8).
  GP14 bitstream: MotionField.fwd_ref_idx/bwd_ref_idx (Option<u8>) added. 5-slot reference
  pool per encoder and decoder. Encoder local-decodes B₄/B₂/B₆ into pyramid slots using
  mc_bidir_inv_params (mode=1 reconstruct) — critical: wrong params (mode=0) caused −0.11 dB
  on all layer-3 B-frames. Decoder always loads refs from saved pool slots (never relies on
  transient buffer state). All 163 tests pass; zero clippy warnings; WASM clean.
- **Validation (2026-03-10):** crowd_run −3.4% bpp (6.21→6.00), VMAF neutral (99.13). park_joy −3.9% bpp (4.94→4.75), VMAF neutral (99.14). bbb Y4M has only 8 frames — insufficient for ki=9 group (needs 10+); no valid comparison. ki fix: use_bframes gate updated to ki>=B_FRAMES_PER_GROUP+2=9; BenchmarkSequence default ki 8→9 (638b77a).
- **Final verdict:** SHIPPED — real improvement, VMAF safe. Note: bbb test material needs ≥10 frames for ki=9 validation.

### 43. Multi-reference P-frames (2 references per tile)
- **Status:** CLOSED (2026-03-10) — Researcher meta-gate analysis
- **Motivation:** crowd_run fails MC because occlusion makes ref[N−1] wrong for many blocks. A tile at (x,y) in frame N may not exist in N−1 (occluded) but exists in N−2. H.264 allows up to 16 reference frames per macroblock. GNC currently supports only 1 reference per P-frame. Adding ref[N−2] as a second candidate and picking the lower SAD could address occlusion.
- **Hypothesis:** Allowing each P-tile to choose from ref[N−1] or ref[N−2] (dual-reference) reduces bpp on crowd_run ≥5% at q=75, VMAF ≥ −0.3 pts. bbb gain ≈2–4%.
- **Success criteria:** crowd_run bpp ≤ 6.10 (≥5% reduction from 6.45); VMAF ≥ 98.6; bbb bpp ≤ 2.56.
- **Fail criterion:** crowd_run bpp change < 2% → dual reference not being used or occlusion not the bottleneck.
- **Gate (before full implementation):** Add diagnostic: for each P-tile ME, compute SAD against both ref[N−1] and ref[N−2]. Print fraction of tiles where ref[N−2] wins. If < 5% of tiles prefer ref[N−2], close.
- **Implementation:** Block match runs twice (against both refs), picks lower SAD. Store 1-bit ref_idx per tile. Decoder maintains 2-frame reference buffer (already partially true with B-frames). Minor bitstream format flag.
- **Complexity:** 3-4 days.

### 44. DC subband offset correction per P-frame
- **Status:** CLOSED (2026-03-10) — physics analysis rules out meaningful DC shift
- **Motivation:** LL mean_abs_diff=40.69 on crowd_run P-frames may indicate a systematic DC offset (global luminance drift, AGC, or scene luminance change). If so, subtracting the mean LL difference between current and reference frame before coding would reduce LL residual at near-zero bitstream cost (1 scalar per frame).
- **Hypothesis:** crowd_run LL residual has non-zero signed mean (> ±5) per P-frame. Correcting it reduces bpp ≥3%.
- **Gate (mandatory before implementation):** Print per-P-frame mean of signed LL residual coefficients. If mean is < 3 (symmetric around zero), close immediately — the energy is from random motion, not DC shift. This is a one-line diagnostic addition to `--diagnostics`.
- **Success criteria:** crowd_run bpp ≤ 6.26 (≥3% reduction); LL residual mean drops to < 2 after correction.
- **Fail criterion:** LL signed mean < 3 per frame → close without implementation.
- **Complexity:** < 1 day if gate passes. One scalar per P-frame added to bitstream header.

### 45. Adaptive GOP length (MC-futility detector)
- **Status:** CLOSED (2026-03-10) — gate failed; ki=2 crowd_run = 6.73 bpp > ki=8 6.45 bpp
- **Motivation:** crowd_run P/B frames cost 98–100% of I-frame cost. The GOP structure forces encoding B/P frames that provide no compression benefit. If an adaptive mode detected "MC futility" (post-ME SAD ≈ I-frame cost) and switched to short GOP or all-I for those segments, we'd eliminate the MV overhead + residual overhead for near-useless inter frames.
- **Hypothesis:** crowd_run with adaptive ki (short GOP when MC SADs are high) reduces bpp ≥7% on crowd_run while maintaining VMAF.
- **Gate (mandatory first):** Run crowd_run with ki=2 manually. If bpp at ki=2 > 6.45 (ki=8 current), I-frame overhead dominates and adaptive GOP can only make things worse → veto. If ki=2 < 6.45, proceed.
- **Success criteria:** crowd_run bpp ≤ 6.00 (≥7% reduction); VMAF ≥ 98.8.
- **Complexity:** 2-3 days. Reuse scene-cut detector logic; threshold on post-ME SAD ratio.

### 46. LL subband spatial prediction (cross-tile delta coding)
- **Status:** CLOSED (2026-03-10) — gate failed
- **Gate result (2026-03-10):** crowd_run P-frame mean_ratio=1.536, max_ratio=1.821 (gate threshold > 0.85). park_joy mean_ratio=1.705, max_ratio=1.982. LL residual tiles are spatially anti-correlated — the inter-tile LL variation *exceeds* the per-tile magnitude. Delta coding would increase bitrate, not reduce it.
- **Root cause analysis:** High-motion P-frame LL residuals reflect per-tile MC prediction error, which depends on local motion complexity. Adjacent tiles have independent motion → independent prediction errors → no spatial correlation in residual domain. The RS hypothesis "spatial signal continuity carries over to residual LL" is falsified: MC removes the spatial continuity, leaving spatially-decorrelated residuals.
- **Diagnostic code:** `GNC_LL_SPATIAL=1` env var in `encode_pframe` (diagnostic only, no bitstream change).

### 47. Overlapping tile windows (cross-tile wavelet lifting)
- **Status:** blocked — gate PASSED but correct implementation deferred (design complexity)
- **Motivation:** Tile-edge artifacts (#36 gate) are 1–2px boundary mismatch from symmetric reflection at tile borders. Correct fix: each tile encodes `(tile_size + 2*overlap)^2` coefficients using actual neighbor pixels; decoder decodes the full extended block and crops to central tile_size×tile_size.
- **Gate result (2026-03-10):** `GNC_TILE_BOUNDARY=1` diagnostic: boundary_psnr=41.56 dB, interior_psnr=42.21 dB, gap=**0.66 dB** at q=75 (threshold 0.5 dB). Also q=25: 0.81 dB, q=50: 0.82 dB. **PROCEED.**
- **Design (Approach A — full overlap, correct):**
  - Add `overlap_pixels: u8` to FrameInfo (and format.rs serialization)
  - Encoder: physical_tile_size = tile_size + 2*overlap. Read from `[tile_origin - overlap, tile_origin + tile_size + overlap)` (clamp at image edges). Write ALL `physical_tile_size^2` coefficients to a SEPARATE coefficient buffer (not the padded image buffer, which is too small).
  - Coefficient buffer per plane: `total_plane_tiles * physical_tile_size^2 * sizeof(f32)`
  - All downstream shaders (quantize, entropy) use `physical_tile_size` for per-tile coefficient count.
  - Decoder: read overlap_pixels from FrameInfo; allocate `(T+2o)^2` coefficient buffer; inverse wavelet on `(T+2o)^2`; crop central T×T via GPU crop shader or CPU copy.
  - Transform shaders: `workgroup_size(256)` and `shared_data[512]`, `shared_low/high[256]` (already done by Builder). Remove `skip/out_cnt` trimming — write ALL half coefficients.
  - **overlap=0 is a no-op** (identical to current behavior, fully tested).
- **What NOT to do:** Do NOT use a "trim to tile_size^2" approach. The decoder can't correctly invert encoder coefficients that were computed with different boundary conditions. This was attempted and caused 5.60 dB boundary gap (worse than before).
- **Structural changes already present:** `CodecConfig.overlap_pixels` (default 0), `transform_97.wgsl` has `overlap` in Params and enlarged shared memory. Encoder panics if `overlap_pixels > 0` until decoder + bitstream are complete.
- **Success criteria:** bbb q=75 boundary gap < 0.1 dB (was 0.66 dB); VMAF ≥ 95.55 (was 95.05); bpp overhead < 2%.
- **Complexity:** 4–6 days. Requires bitstream version bump.

### 48. Quarter-pel chroma MC for 4:2:0 B-frames
- **Status:** CLOSED (2026-03-10) — gate: chroma residual already below luma; MC working correctly
- **Gate result (bbb_test.y4m, q=75, 4:2:0):** Co/Y mean_abs ratio = 0.57, Cg/Y = 0.43. Both well below 1.2 threshold. After fixing the stale mv_chroma_buf bug (#14), 4:2:0 B-frame chroma MC is functioning correctly — chroma residual is smaller than luma as expected. No evidence of chroma MC being the bottleneck.
- **Note:** Also note that 4:2:0 B-frame chroma currently uses wrong MV spatial mapping (16×16 luma MV applied to 4×4 chroma blocks with stride mismatch), but it's consistent in encoder and decoder. Fixing mapping would improve quality but is a separate item.

### 49. P-frame reference selection from pyramid pool
- **Status:** todo
- **RS hypothesis (2026-03-10):** P-frame currently references only the most recent I or P anchor. With the 5-slot pyramid pool from #42, the decoded B₄ (frame 4) is available as a closer reference for the P-frame (frame 8). Using B₄ instead of I₀ halves temporal distance from 8 to 4 frames on slow-motion content, reducing residual energy.
- **Falsifiable claim:** On bbb (slow-motion), P-frame ME SAD against B₄ (closest decoded B) is < 80% of SAD against I₀. If SAD ratio > 0.9, temporal distance is not the bottleneck.
- **Gate result (2026-03-10):** crowd_run: 52.5% tiles prefer B₄_src, mean_SAD_ratio=1.016 (PASS). park_joy: 85.0% tiles prefer B₄_src, mean_SAD_ratio=0.776 (PASS). Gate threshold was >20%.
- **Architecture constraint discovered:** P₈ is encoded before B₄ in coding order (I₀→P₈→B₄→...). B₄ is not decoded when P₈ is encoded. Implementation requires coding order change to I₀→B₄→P₈→B₂→B₆→... (B₄ encoded as P-frame first, P₈ uses decoded B₄ as reference). Significant refactor.
- **Gate (1 day):** Add diagnostic: compute SAD of P-frame tiles against both I₀ and B₄. Print fraction of tiles where B₄ wins and mean SAD ratio. Gate: proceed if > 20% of tiles prefer B₄.
- **Success criteria:** bbb bpp −3% at q=75; VMAF neutral. crowd_run expected ~0% (chaotic motion).
- **Fail criterion:** < 20% of tiles prefer B₄ → close (I-frame is better reference, B₄ has accumulated error).
- **Complexity:** 3–4 days (revised up from 2–3 due to coding order refactor). Coding order: I₀→B₄→P₈→B₂→B₆→...; B₄ encodes as fwd-only P-frame; P₈ uses decoded B₄ from pyramid slot 0.

### 50. Fast I-frame Rice skip for near-zero tiles
- **Status:** CLOSED (2026-03-10) — gate failed: 0% qualifying tiles at q=75
- **Gate result:** `Rice: all_skip_tiles=0/120` at q=75 on bbb_1080p. Zero tiles have >80% zeros at this quality level — quantization is aggressive enough that all tiles have non-trivial coefficient distributions. The speed axis gain is zero.

