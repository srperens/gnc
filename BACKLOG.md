# GNC Backlog

Status: `todo` | `active` | `done` | `blocked`

Only **open** items live here. All completed, closed, and vetoed items (66 of them,
with gate experiments and measurements) are archived verbatim in
[docs/archive/BACKLOG_CLOSED.md](docs/archive/BACKLOG_CLOSED.md).
Current state and priorities are described in [GOALS.md](GOALS.md).

## Baseline (v0.1-spatial, commit 617d8e6)

See [BASELINE.md](BASELINE.md) for current benchmark numbers.

## Current Focus (updated 2026-09-04)

**Mode: Measurement and bugfixing. No new features.**

The measurement campaign (parts 8–13) established a new baseline with uniform subband
weights, and the 2026-03 experiment sweep (~40 gated experiments, see archive) exhausted
the cheap and medium-cost *incremental* inter-compression ideas. Temporal compression is still
a goal (see [GOALS.md](GOALS.md) §4) — the open question is what shape it should take, and
MEAS-4 is designed to answer that before anything gets built. Until then the priorities are:

1. Fix known bugs (BUG-1)
2. Finish the measurement campaign (MEAS-1/2/3) — honest VMAF-based video numbers
3. Toggle features to identify dead weight and incorrect implementations
4. Let measurements drive the next action

**Known facts (2026-03-11, uniform weights):**
- Spatial BD-rate vs H.264 all-I: **+13.9%** — reasonable for a wavelet codec; GNC wins above ~36 dB
- Spatial BD-rate vs JPEG 2000 (4:4:4): **+28.3%** — gap narrows to ~11% at high quality
- Temporal: GNC I+P+B saves ~17–27% vs all-I. H.264 saves ~60–70% → ~3–4× gap in the current
  I/P/B design. Whether that is inherent or a wrong-model problem is unmeasured — that is MEAS-4.
- All above is PSNR-based (RGB). VMAF-based video comparison vs H.264 still missing (MEAS-1).

## Active priority list

### BUG-1 — 4:2:0 pyramid B-frame chroma bug (**DONE 2026-09-05**)
True B-frames in 4:2:0 reconstructed 4–6 dB below their bitrate; B₄ and P-frames unaffected;
4:4:4 unaffected.

**Root cause** (measured, not the one the diagnosis led with): the chroma MC shader indexed the
MV and block-mode fields with the chroma 4×4 block grid's row stride, but a true B-frame's MV
field is on the 16×16 luma ME grid — half the resolution on each axis. Every chroma block read a
spatially unrelated MV. The encoder/decoder tail divergence identified in
[docs/BUG-1_DIAGNOSIS.md](docs/BUG-1_DIAGNOSIS.md) was real but secondary. A third defect found
by the canary: luma and chroma pad to a tile multiple independently, so the two grids are not
proportional (1080p → 192 chroma rows vs 80 MV rows) and the surplus rows indexed past the field
on the P path too.

**Fix:** `ChromaMvGrid` states the mapping explicitly, derived from block geometry and built from
one constructor on both sides; the shader clamps to the field extent; `mv_scale` is dispatched
with the frame's own MV count. See
[docs/decisions/0004-chroma-mv-grid-mapping.md](docs/decisions/0004-chroma-mv-grid-mapping.md).

**Result** (1080p q=75, ki=9, 4:2:0, `GNC_REF_DEBLOCK=0`): worst B-frame +4.6 dB (BBB) / +3.7 dB
(touchdown); VMAF mean +0.61 / +0.42, VMAF min +2.58 / +2.10; bpp −1.4% / −0.4%. Quality up and
rate down together. B₄, P and all 4:4:4 output bit-identical. Canary: `GNC_DIAGNOSTICS=1` prints
`[bframe_chroma_mv] enc grid: ...` per B-frame.

### BUG-2 — Pyramid reference-buffer defects (todo, P2)
Two format-independent defects identified in the BUG-1 diagnosis and **not** addressed by that
fix. Neither was implicated in the chroma collapse; both are wrong by inspection.

- **B₇'s encoder backward reference is stale.** `sequence.rs` `layer3_order`: B₅'s setup loads B₆
  into `gpu_bwd_ref_planes`, then B₇ hits a no-op arm whose comment claims the future P is still
  there. The encoder predicts B₇ from B₆ while the decoder loads P₈. Fix direction:
  `1 => self.copy_pyramid_slot_to_bwd_ref(ctx, 4, plane_size)`.
- **End-of-group reference restore is gated on 4:4:4.** In 4:2:0 pyramid the `else` branch calls
  `swap_ref_planes()`, leaving `gpu_ref_planes` = B₆ rather than decoded P₈, so the next group's
  P-frame is encoded against the wrong reference. Fix direction: use the slot-4 restore
  unconditionally.

Gate: per-frame PSNR in 4:4:4 pyramid should show B₇ measurably worse than B₁/B₃/B₅ today, and
multi-GOP 4:2:0 should show drift from P₁₆ onward. Measure before fixing — if neither shows, the
reasoning is wrong somewhere.

### BUG-3 — 4:2:0 collapses when the chroma plane is smaller than one tile (todo, P2)
Found while building the BUG-1 regression test (2026-09-05). At 512x512 and 1024x1024 a 4:2:0
pyramid GOP is healthy (I 38.1 dB, B 33–38, P 37.9). At **256x256 with tile_size=256** the whole
GOP degrades progressively — I0 38.1, then 31.6 / 28.0 / 25.9 / 25.6 / 22.5 / 23.0 / 23.9, P8
**20.6** — while the identical content in 4:4:4 stays flat at 42–44 dB across every frame.

P-frames are affected too, so this is **not** the BUG-1 chroma MV mapping (that path is
bit-identical for P). The distinguishing condition is that the chroma plane (128x128) is smaller
than one tile, so the suspicion is chroma tile/padding handling at sub-tile plane sizes.

Reproduce: `test_bframe_yuv420_chroma_mv_grid` with w=h=256 instead of 512. Gate before fixing:
confirm it tracks `chroma_plane < tile_size` rather than the absolute resolution, e.g. 256x256
with `tile_size=128` should be healthy if the theory holds.

### MEAS-1 — Correct video comparison GNC vs H.264 (todo)
Run H.264 (libx264, full inter, standard GOP) on crowd_run + park_joy, 10 frames, yuv420p.
Measure PSNR and bpp per CRF value. Compute BD-rate against GNC 4:2:0 I+P+B (after BUG-1 fixed).
Also GNC 4:4:4 vs H.264 yuv444p for fair comparison without chroma handicap.

### MEAS-2 — Feature toggling: what contributes and how much? (todo)
Systematic toggle measurement on crowd_run + park_joy, 10 frames, 4:4:4, q=75:
- AQ on/off (GNC_NO_AQ)
- CfL on/off
- Pyramid QP scale on/off (GNC_L3_QP_SCALE=1.0 vs 1.5)
- Pyramid B-frames vs flat B-frames (pyramid_enabled=false)
- B-frames vs P-only (ki=1)
- Rice vs rANS
Each toggle: report bpp + VMAF delta. Goal: identify dead weight and negative features.

### MEAS-3 — RD-curve on sequences (todo)
Run rd-curve (q=25–90) on crowd_run and park_joy with 4:4:4, measure VMAF at each point.
Current rd-curve lacks --chroma-format and --vmaf on sequences. A benchmark loop may suffice.

### MEAS-4 — Inter-model gap decomposition: oracle experiments (todo, after MEAS-1)
**Question:** Is GNC's 3–4× inter-efficiency gap vs H.264 caused by the *model* (tile-wide
wavelet on MC residuals + context-free entropy) rather than by GPU-parallelism per se? If yes,
a hybrid inter pipeline (per-block transform + block skip + context entropy — all GPU-parallel)
has real headroom. Measure the ceiling offline before building anything. No bitstream changes;
run in 4:4:4 (does not depend on BUG-1).

- **MEAS-4a — P-residual subband energy distribution** (the #35 gate that was never run, ~0.5 day):
  Diagnostic that logs per-subband energy of P/B-frame residual wavelet coefficients on
  crowd_run, park_joy, rush_hour at q=75. Falsifiable: if detail subbands carry >40% of residual
  energy, the wavelet is spreading block-structured energy (transform mismatch real). If LL
  dominates, transform is not the problem — close MEAS-4b's DCT half.
- **MEAS-4b — Oracle block-skip/DCT bound** (~1–2 days, offline): Dump spatial-domain MC
  residuals (pre-transform) per P/B-frame to disk. Offline script computes: (1) fraction of
  16×16 blocks with residual energy below the q=75 quantization threshold ("oracle-skippable");
  (2) fraction of total residual energy in the top 10/25/50% of blocks; (3) estimated bpp of an
  oracle coder: per-block 8×8 DCT + uniform quantizer + Shannon entropy of quantized coeffs
  + 1 bit/block skip signaling. Compare against measured GNC inter bpp at same quality.
  Falsifiable: oracle bound ≥40% below current GNC inter bpp on ≥2/3 sequences → the model
  (not prediction quality) is the cap, hybrid inter pipeline has real ceiling. Oracle bound
  <20% below → the gap is prediction quality, not the coding model; a rebuilt inter pipeline
  would not pay for itself, so close this line of attack and look elsewhere for temporal gain.
- **MEAS-4c — Entropy context ceiling** (~1 day, offline, extends #53): On the same dumped
  quantized coefficients, compute empirical conditional entropy with 1–2 neighbor/parent
  contexts vs measured Rice bpp. Bounds what 4–16 context-adaptive streams per tile (still
  thousands of parallel streams per frame) would recover vs the current 256 context-free streams.
- **MEAS-4d — H.264 feature ablation** (~0.5 day, no GNC code): Decompose H.264's ~60–70%
  temporal saving using x264 flags on the same sequences: baseline vs `--no-cabac` (context
  entropy contribution), `--partitions none` (block partitioning), `--ref 1 --bframes 0`
  (multi-ref/B contribution). Shows how much of H.264's inter gain comes from skip/partitioning
  vs entropy vs prediction — i.e., which parts GNC is actually missing.

**Decision rule:** MEAS-4b is the verdict. ≥40% oracle headroom → write a design doc for a
hybrid inter pipeline (wavelet I-frames + per-block DCT inter residuals + block skip + context
entropy). <20% → the current inter model is not the bottleneck; the gap is prediction quality,
and further work on the coding side is not where the temporal win lives.
**Requires the dev machine** (GPU + test material) for the residual dumps; the offline analysis
scripts are CPU-only.

## Noted — revisit only if conditions change

### ARCH-1 — Hybrid temporal: Haar for low motion, I+P+B for high motion (noted 2026-03-11)

**Observation:** GNV1 (I+P+B) and GNV2 (temporal Haar) are currently two separate architectural
tracks that are never combined. Measurements show B-frames beat Haar on motion-heavy content
(ducks q=75: 531M vs 596M), but they are essentially equal on low-motion content (rush_hour: both 78M).

**Hybrid idea:** Select temporal strategy per GOP based on motion energy:
- High motion → I+P+B (ME-based)
- Low motion → temporal Haar (pure temporal decorrelation)

**Why it probably does not pay off:** IPB naturally degrades to near-zero residuals on low motion
(MV≈0, skip-mode). Haar adds no measurable gain on top of that. Cost = dual decoder pipelines
+ per-GOP bitstream signaling.

**More promising variant:** MCTF (motion-compensated temporal filtering) — Haar *with* ME, as used
in Dirac/VC-2. Beats pure IPB in the literature but is a substantial project — a plausible
candidate for the "right shape" question in [GOALS.md](GOALS.md) §4. Parked until MEAS-4 bounds
how much headroom a rebuilt inter path actually has.

**Status:** Noted, nothing to implement now.

### 25. Multi-reference P-frames
- **Status:** deferred (P3)
- **Motivation:** GNC P-frames reference only the immediately preceding decoded frame. H.264 can reference up to 16 frames, which dramatically improves compression for repeated textures (scrolling text, panning shots) and periodic motion. Even 2-reference P-frames would cover the most common cases.
- **Hypothesis:** Allowing P-frames to choose the best of 2 reference frames (prev and prev-prev) reduces bpp 3–8% on sequences with periodic motion or scene repetition.
- **Success criteria:** bpp −3% on at least one test sequence; VMAF neutral; no regression on bbb/crowd_run.
- **Complexity:** Medium. Requires decoder to track a reference buffer (already partially done for B-frames). ME shader needs a second reference input and cost comparison.
- **Research Scientist verdict (2026-03-09):** DEFER. Expected gain requires content with periodic motion; current test sequences (bbb, crowd_run, rush_hour, park_joy) don't exhibit this. Gate on adding a periodic-motion test sequence AND running MV histogram showing >15% non-adjacent references.
- **Note:** the tile-level dual-reference variant (#43) was separately CLOSED (2026-03-10) — see archive. Revisit alongside MEAS-4's outcome rather than on its own.

### 61. Resolution-adaptive pipeline scaling (4K / 8K / 12K readiness)
- **Status:** todo (P2 — no action needed until 4K test material available, but design must account for this)
- **Motivation:** All pipeline parameters are currently calibrated for 1080p in pixels. At higher resolutions, the same physical scene motion and structure occupies proportionally more pixels. A fixed 256×256 tile at 8K covers ~6% of frame height vs ~24% at 1080p. A fixed ±96px ME search range at 4K corresponds to only ±48px of equivalent scene coverage. A fixed 4-level wavelet at any resolution produces a 16×16 LL subband regardless of how much scene content that represents.
- **Parameters that must scale with resolution:**
  - **Wavelet levels:** 4 @ 1080p → 5 @ 4K → 6 @ 8K → 7 @ 12K. Keeps LL subband representing ~same angular frequency.
  - **ME search range:** `ME_SEARCH_RANGE = base_range × (width / 1920)`. Keeps equivalent scene coverage constant.
  - **ME block sizes (once #60 is done):** nominal block size should scale similarly.
  - **AQ region size:** AQ energy map resolution should track resolution.
- **Parameters that do NOT scale:**
  - **Tile size (256×256):** Hardware-constrained. M1 threadgroup memory (32KB) sets the ceiling. More tiles at higher resolution = more GPU threads = parallelism scales automatically. This is a feature, not a limitation.
  - **Rice stream count (256 per tile):** Tied to tile size, stays fixed.
- **Falsifiable claim:** At 4K (3840×2160), increasing wavelet levels from 4→5 and ME search range proportionally reduces I-frame bpp ≥3% and P-frame bpp ≥5% vs the 1080p-calibrated baseline run at 4K, at VMAF neutral.
- **Gate:** Run GNC on a 4K test sequence with 4 vs 5 wavelet levels. No code change required — wavelet level is already a parameter. If bpp/VMAF difference < 1% → close.
- **Success criteria:** Pipeline parameters auto-select based on input resolution. All 1080p benchmarks unaffected (backward compatible).
- **Complexity:** Low for wavelet levels (already parameterized) and ME range (one formula). Medium for making auto-selection robust across resolutions.
- **Note:** Tiles remain 256×256 by hardware necessity. The design insight is that tile count scales with resolution (more tiles = more GPU threads), while semantic parameters (wavelet depth, ME range) must be resolution-relative, not pixel-absolute.
- **Design work required:** Before implementing resolution scaling, the team needs a general design document and explicit coding rules around pixel-absolute vs resolution-relative parameters. Every threshold, block size, search range, and energy value in the pipeline that is expressed in pixels is implicitly a 1080p assumption. This includes #59 (SAD threshold), #60 (block sizes, λ), ME search range, AQ energy map granularity, pyramid downsampling ratios, and any future parameters. The design document should establish: (1) which parameters are pixel-absolute by necessity (tile size — hardware), (2) which must be expressed relative to resolution (`width/1920` scale factor), (3) which should be per-pixel normalized (SAD, distortion, λ), and (4) a naming/commenting convention so future code makes the assumption explicit. This should be written before any 4K implementation work begins.

### 12. CPU SIMD path (long-term, low priority)
- **Status:** todo (P5 — far future, contingent on codec maturity)
- **Motivation:** Broadcast contribution niche — same hole as VC-2/Dirac and JPEG XS: low latency, high quality, patent-free, low complexity. For broader adoption, a CPU-only path removes the GPU dependency and enables use on hardware without a capable GPU (servers, edge devices, FPGA/ASIC targets). Also enables WebAssembly decode on browsers without WebGPU (e.g. Firefox today).
- **Approach:** Portable SIMD via `std::portable_simd` or `wide` crate — single code path that compiles to NEON (M1/ARM), AVX2 (x86), and WASM SIMD128. GPU path remains primary; SIMD path is a secondary fallback tier.
- **WASM note:** WASM SIMD128 is well-supported (Chrome 91+, Firefox 89+, Safari 16.4+) and trivial to ship — just add `-C target-feature=+simd128` to the wasm-pack build.
- **Prerequisite:** Codec must first reach competitive compression/latency/quality. No point optimizing a SIMD path for an algorithm that may still change fundamentally.
- **Success criteria:** CPU SIMD decode of a 1080p frame within 2× real-time at target quality. No GPU required.
- **Note:** Primary goal of this project is to explore whether AI-driven iteration can produce something competitive in this space. SIMD path is downstream of that question.

## Recently shipped (details in archive)

- **#64 Pyramid L3 QP scale** — DONE (2026-03-11). crowd_run −11.0% bpp, park_joy −10.4%.
- **#65 Subband weight fix** — DONE (2026-03-11). +2.28 dB PSNR, +1.51 VMAF at q=75. BD-rate ~18% bpp saving at equal VMAF.
- **#42 Hierarchical B-frame GOP** — DONE (2026-03-10). crowd_run −3.4%, park_joy −3.9%.
- **#60 Adaptive block-size ME** — DONE (2026-03-11). Neutral VMAF.
- **#49 B₄-as-P forward-only** — DONE (2026-03-10). Neutral.
