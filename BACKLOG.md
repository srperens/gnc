# GNC Backlog

Status: `todo` | `active` | `done` | `blocked`

Only **open** items live here. All completed, closed, and vetoed items (66 of them,
with gate experiments and measurements) are archived verbatim in
[docs/archive/BACKLOG_CLOSED.md](docs/archive/BACKLOG_CLOSED.md).
Current state and priorities are described in [GOALS.md](GOALS.md).

## Baseline (v0.1-spatial, commit 617d8e6)

See [BASELINE.md](BASELINE.md) for current benchmark numbers.

## Current Focus (updated 2026-09-05)

**Mode: Measurement and bugfixing. No new features.**

**Positioning fixed 2026-09-05 (see GOALS §1): GNC is a contribution codec.** It does not try to
beat H.264 on bitrate; it aims for H.264-class quality that runs on any GPU and scales with the
card, against fixed-function encoders with session limits. Two consequences for this list: the
headline metrics are now concurrent streams per GPU (MEAS-5) and latency (MEAS-6), neither of
which has ever been measured; and compression targets move to the contribution operating point,
so historical BD-rate measured at distribution bitrates describes a use case GNC is not built for.
Both intra and inter remain goals — going all-intra was considered and rejected.

The measurement campaign (parts 8–13) established a new baseline with uniform subband
weights, and the 2026-03 experiment sweep (~40 gated experiments, see archive) exhausted
the cheap and medium-cost *incremental* inter-compression ideas. Temporal compression is still
a goal (see [GOALS.md](GOALS.md) §4) — the open question is what shape it should take, and
MEAS-4 is designed to answer that before anything gets built. Until then the priorities are:

1. Fix known bugs (BUG-1 done; BUG-2, BUG-3 open)
2. Finish the measurement campaign (MEAS-1/2/3) — honest VMAF-based video numbers
3. Toggle features to identify dead weight and incorrect implementations
4. Let measurements drive the next action

MEAS-4 (done 2026-09-05) established that the inter gap is **not** in the coding model. The
follow-up hunt for where it *is* returned four negative results in a row (multi-reference,
sub-pel interpolation filter, motion search quality, and the earlier transform/entropy work),
and then the premise itself failed inspection: the "GNC saves 38.5% vs x264's 86.9%" figure
compares GNC's RGB PSNR against x264's YUV PSNR, which are not the same quantity.

**MEAS-1 is therefore a hard prerequisite for any further inter work.** Until there is a
trustworthy, VMAF-based, like-for-like comparison, there is no reliable number saying how large
the inter gap is, and targeting it is guesswork. See RESEARCH_LOG 2026-09-05.

**Known facts (2026-03-11, uniform weights):**
- Spatial BD-rate vs H.264 all-I: **+13.9%** — reasonable for a wavelet codec; GNC wins above ~36 dB
- Spatial BD-rate vs JPEG 2000 (4:4:4): **+28.3%** — gap narrows to ~11% at high quality
- Temporal: GNC I+P+B saves ~17–27% vs all-I (48.9%/29.8% on bbb/touchdown at q=75, 4:4:4).
  x264 saves 86–89% on the same content → the gap is real and large. **MEAS-4 located it in
  prediction quality, not in the coding model.**
- All above is PSNR-based (RGB). VMAF-based video comparison vs H.264 still missing (MEAS-1).

## Active priority list

### BUG-5 — B-frames stop paying at contribution quality (todo, P0)
Measured 2026-09-05 on 17 byte-identical 1080p frames (bbb, 4:4:4, Rice, fixed qstep, rate
control off) — content where the correct answer for every inter frame is "nothing changed".

| config | per inter frame | all_skip_tiles |
|---|---|---|
| P-only (`ki=8`, no B) | **3 246 B** | 120/120 every frame |
| B-pyramid (`ki=17`) | **54 059 B** avg | 8-95/120, varies |

The residual reaching the quantiser is statistically identical on both paths
(`mean_abs=0.83-0.84, near_zero=68%`) and the skip threshold is the same function
(`tile_skip_threshold`, `sequence.rs:3638` for P and `sequence.rs:6192` for B). So identical
input and an identical threshold produce 120/120 skip on one path and 8/120 on the other.

**The bits do not pay for themselves.** The B-pyramid buys +0.79 dB for 813 KB. Spending the same
bits on the I-frame and letting every inter frame all-skip reaches the same quality for **34.4%
fewer bytes** (1 348 021 B @ 45.04 dB vs 2 055 422 B @ 45.10 dB). On this content the temporal
path is worse than not coding inter frames at all.

**Working hypothesis (untested):** averaging two independently reconstructed references puts the
prediction a half quantiser step off, so the residual escapes the dead zone almost everywhere,
where a single reference's residual is exactly that reference's quantisation error and quantises
back to zero.

**Blocks/affects:**
- **TUNE-1** — its -24% for longer GOPs was read as "GNC uses too few B-frames", but B is the
  defective path here. The mechanism behind that number is not established and the headroom after
  a fix is probably larger. Do not change the default GOP rule until BUG-5 is understood.
- **MEAS-1 and ARCH-2** were both measured at `ki=9` with B-frames on. How much of the reported
  5-7x gap is design and how much is this defect is currently unknown.

**Confirmed on real content 2026-09-05.** Four 1080p sequences x 17 frames x 4 qsteps, 4:4:4,
Rice, fixed qstep, VMAF-based BD-rate of B-pyramid (`ki=17`) against P-only (`ki=8`). Negative =
B-pyramid cheaper:

| sequence | content | full range | high-quality end |
|---|---|---|---|
| bbb | animation | -37.2% | -35.3% |
| touchdown | camera, sport | -9.4% | **+8.2%** |
| old_town | camera, pan | +7.9% | **+7.4%** |
| speed_bag | camera, high motion | +15.2% | **+31.4%** |

The B path is **not globally broken** — it wins at distribution bitrates, and on animation it wins
everywhere. It loses at the high-quality end on all three camera sequences, by 7-31%, *while
P-only is handicapped by two extra I-frames* (`ki=8` emits 3 I-frames over 17 frames, `ki=17`
emits 1). The static-content result above is the extreme case of the same effect. A finer
quantiser makes the hypothesised half-step averaging offset relatively larger, which fits.

**Note on test material:** bbb is the one sequence where B wins at high quality, and it is this
repo's primary test sequence. Historical inter conclusions drawn from bbb need re-checking on
camera content.

**Cheap conditional fix, available before any root-cause work:** disable or shorten the B-pyramid
above a quality threshold. Worth 7-31% at contribution quality on camera content; a configuration
change, not a bitstream change. Root cause still worth finding — it may be worth more.

**Next step:** finer qstep sweep to locate the crossover per sequence, and a 4:2:0 cross-check.
Full measurement in RESEARCH_LOG 2026-09-05.

### MEAS-5 — Concurrent streams per GPU vs NVENC (todo, P0 — never measured)
The whole contribution-codec positioning (GOALS §1) rests on one unproven claim: a GPU's shader
throughput scales with the card while its fixed-function encoder blocks do not, so a big GPU
should run more GNC instances than it runs NVENC sessions. **Nobody has measured this.**

Measure: N concurrent 1080p encode+decode instances on one GPU, at contribution quality, until
throughput per instance drops below realtime. Report the N, and the same N for NVENC/VideoToolbox
on the same machine. Needs at least the M1 and one discrete NVIDIA card; ideally an Intel iGPU too.

Per-stream, compute will lose to fixed function — that is expected and not the claim. The claim is
about the aggregate. If the aggregate also loses, the positioning is wrong and we need to know
that before more compression work.

### MEAS-6 — Latency per frame (todo, P1 — never measured)
Contribution links need sub-frame latency, and tile independence is supposed to buy it. Measure
end-to-end frame latency (submit → decoded frame available), not throughput fps. Currently
unquantified.

### FMT-1 — 10-bit support (todo, P1)
8-bit is the main format gap for broadcast contribution. 4:2:2 and 4:2:0 already work, so bit
depth is the remaining piece. Cheap now — GOALS rule 10 says the bitstream can still break freely
and there are no users — and expensive once there is a spec, conformance streams and deployments.

Not a research item: buffer formats, upload/download paths, PNG/Y4M I/O, bitstream fields, and the
VMAF/PSNR comparison harness at 10-bit. No f64 needed; 10-bit fits f32/i32 comfortably.

**Partly present already (verified 2026-09-05).** `gnc encode` accepts `--bit-depth 10` and the
still-image encode/decode round-trip runs. **`gnc encode-sequence` has no bit-depth option at
all**, so the video path — the one the market actually requires — is 8-bit. Scope is therefore
"extend the existing still-image 10-bit path through the sequence pipeline and container", not
"implement 10-bit from scratch". Verify the still path is genuinely 10-bit end to end (the decoded
PNG's actual bit depth was not confirmed) before relying on it.

**External requirement, confirmed 2026-09-05.** EBU R 153 specifies 10-bit 4:2:2 Y'C'BC'R for live
UHD/HDR contribution and forbids SDR transfer functions; EBU TR 091's entire codec test matrix is
10-bit BT.2100 HLG with no 8-bit or 4:2:0 test point anywhere. A codec that cannot ingest 10-bit
4:2:2 cannot be entered into the industry's reference evaluation. This is a gate, not a feature.

Note the interaction with the operating point: 10-bit costs bits at low bitrate and almost nothing
at contribution quality, which is another reason to stop optimising against distribution bitrates.

### ARCH-2 — Inter residuals cannot skip locally (**CLOSED 2026-09-05** — measured, unreachable)
Measured 2026-09-05. At matched VMAF on bbb, GNC's inter frames cost **8-10x** H.264's (P: 304-380
KB vs 39 KB; B: 108 KB vs 14 KB) while intra costs only ~1.9x. On a fully static sequence — 17
identical frames — x264 codes P-frames at 181 bytes and B-frames at 76; GNC needs ~18 KB.

Ruled out by measurement, none of them the cause: multi-reference (~1-5%), sub-pel interpolation
filter (neutral), motion search quality (GNC beats an offline oracle), context entropy (<=3.4%),
pyramid QP scaling (-6% rate for -1.2 VMAF), tile size, dead zone (moves along the same RD curve,
not off it).

**Hypothesis:** GNC transforms the whole 256x256 tile with a wavelet, so the smallest region it
can decline to code is a tile. H.264 skips per 16x16 macroblock for ~1 bit. Well-predicted areas
therefore cost GNC ~0.5 bits per coefficient where they cost H.264 nothing.

**Note on MEAS-4:** it concluded a rebuilt inter model was not worth it, comparing models at
matched *residual distortion*. That comparison structurally cannot see the value of skip, because
skipping trades distortion for rate. The conclusion should not be relied on for this question.

**Block-wise inter coding was investigated and rejected (2026-09-05).**
`scripts/meas_block_skip_rd.py` compared GNC's tile wavelet against 16x16 blocks with an 8x8 DCT
and a per-block RD skip decision, on GNC's own residuals. At matched residual PSNR it is 30-39%
*worse* on bbb and 30-34% *better* on touchdown — content-dependent, and nowhere near the ~8x
needed. Smaller tiles are not an option either: each tile costs ~290 bytes of header, so 16x16
tiles would mean ~2.3 MB of headers per 1080p frame, and tile=64 measured 70% more bits at worse
quality than tile=256.

**Closed: all three routes to fine-grained skip measured and rejected.**

| route | result | why |
|---|---|---|
| shrink tiles | +70% bits at worse quality (64px) | ~290 B fixed header per tile |
| block-based transform | −39% to +34%, content-dependent | wavelet compaction offsets the skip gain |
| mask sub-blocks inside the wavelet | 5-30% worse at every sub-block size | synthesis support rings across region edges |

**Parallelism is not the constraint and more tiles do not help.** Each tile already carries 256
independent entropy streams, so 1080p/256px runs 10 240 independent streams per frame on an
8-core M1. Tile count is a rate knob, not a speed knob; the per-tile header is the price of the
stream independence that makes decode parallel. The design choice that makes GNC fast is the same
one that makes its inter coding weak.

**Skip granularity confirmed as the binding constraint (2026-09-05).** On a pure pan, backing
the inter quantiser off 3x cuts 31% of the bitrate and slightly *improves* VMAF — GNC was making
inter frames better than the I-frame they predict from. On real content the same change loses to
simply lowering q. The adaptive version GNC already has (`dispatch_tile_skip`, now wired for
P-frames) is worse than the q-curve on real content because 256x256 is too coarse: a tile
survives whole or dies whole. Combined with the block-transform result (±30%), GNC can neither
skip finely with its current transform nor gain enough from changing it. That is where the 8x
sits.

Tunables added for measurement, all defaulting to current behaviour: `GNC_INTER_DZ_MUL`,
`GNC_TILE_SKIP_THRESH`, `GNC_P_QP_SCALE`, `GNC_SPLIT_LAMBDA_SCALE`.

**Earlier notes, both now tested and negative:**
1. **No rate-distortion decisions anywhere.** GNC quantizes at the configured qstep and codes
   whatever comes out. x264's ablation puts its RD mode decision at +22%.
2. **Reference quality.** No in-loop deblocking; references carry wavelet ringing spread over the
   tile. The inter residual's mean |value| (2.63) sits near the ~2.0 noise floor the reference
   itself imposes — so much of each inter frame is re-coded reference noise. If that holds, the
   fix is better references, not better residual coding. **This is the more promising of the
   two.**

### BUG-4 — Inter frames degrade at odd tile-column counts (todo, P2)
At `--tile-size 128` on 1080p (15 tile columns, odd) several inter frames land 4-6 dB below their
neighbours while I-frames are unaffected; VMAF min falls from 94 to 80. Same condition as BUG-3
(`padded_w != 2 * chroma_padded_w`), which was fixed for the chroma MC stride — something else in
that family remains. Reproduce: `benchmark-sequence -n 17 -k 9 -q 70 --chroma-format 420
--tile-size 128`.


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

### BUG-2 — Pyramid reference-buffer defects (**DONE 2026-09-05**)
Two defects from the BUG-1 diagnosis, both measured before fixing. Writeup:
[docs/decisions/0006-pyramid-reference-restore.md](docs/decisions/0006-pyramid-reference-restore.md).

- **B₇'s backward reference was stale.** Its `bwd_idx = 1` arm was a no-op asserting the future P
  was still in the bwd buffer; B₁/B₃/B₅ had each overwritten it, leaving B₆. Now loads slot 4
  explicitly. 4:4:4 ki=9: B₇ 39.21 → **40.17 dB** at 22% fewer bits.
- **End-of-group reference restore was gated on 4:4:4.** In 4:2:0 the `else` branch left the
  forward reference holding B₆ instead of the decoded anchor P, so the next group's P was encoded
  against a reference the decoder does not have. Gate removed. 4:2:0 ki=17: P₁₆ 30.39 →
  **40.35 dB**, sequence VMAF 84.10 → **95.68** (min 69.74 → 94.72), bpp −2.8%.

**Why no test caught it:** every sequence test used ki ≤ 9, where the frame after a group is an
I-frame and the restored reference is never read. Regression test
`test_multi_group_yuv420_anchor_pframe` uses ki=17.

### BUG-3 — 4:2:0 chroma MC used the wrong row stride (**DONE 2026-09-05**)
Logged with a gate that turned out to be **wrong** ("breaks when chroma plane < tile size" —
falsified by 384x384, which is healthy). The real rule, from a sweep plus a non-square test:
breakage depends only on the *horizontal* tile count, i.e. `padded_w != 2 * chroma_padded_w`,
which holds whenever `tiles_x` is odd — **including 1280x720**, where inter frames measured
23.6 dB. Writeup:
[docs/decisions/0007-chroma-plane-stride.md](docs/decisions/0007-chroma-plane-stride.md).

Two off-by-stride errors, one per side: the encoder built the chroma MC params from
`padded_w / 2` (false when tiles_x is odd), the decoder derived the MV index from the chroma
block grid rather than the luma split grid the MVs actually live on. Fixed the BUG-1 way — state
both grids explicitly and clamp.

**Result:** 720p anchor P 23.63 → **37.92 dB**; 768x768 23.86 → 37.93; 256x256 20.57 → 37.93;
512x512 and 1920x1088 unchanged (controls). On real 1080p content, identical VMAF at
**−3.9% bitrate** on both sequences — the old height (640 vs 768 chroma rows) left the bottom of
every chroma plane unwritten, and the stale contents were still being coded.

**Follow-up worth doing:** audit for other places deriving one plane's geometry from another's
by a fixed factor. Three defects this session came from that single assumption.

### TUNE-1 — Default keyframe interval fragments the B-pyramid (todo, P1)
`ki=9` exactly matches the 8-frame pyramid group, so trailing frames form a group too short for a
pyramid and degrade to a P-chain. Measured at 1080p q=70 4:2:0:

| 17 frames | mix | rate | VMAF |
|---|---|---|---|
| ki=9 (default) | 2I+8P+7B | 5 102 044 | 95.50 |
| ki=17 | 1I+2P+14B | **−24%** | 95.02 |

| 33 frames | mix | rate | VMAF |
|---|---|---|---|
| ki=9 (default) | 5I+7P+21B | 8 244 027 | 95.53 |
| ki=33 | 2I+10P+21B | **−16%** | 95.07 |

Worth ~11% BD-rate on 33 frames, more on shorter ones. P-frames are references whose error
propagates, so they cannot be coded coarsely; B-frames are disposable. x264 spends 4 P and 11 B
where GNC spends 8 P and 7 B over the same 17 frames.

**Superseded in part by BUG-5 (2026-09-05).** The -24% above was measured at q=70 4:2:0, a
distribution operating point. Longer GOPs mean more B-frames, and at contribution quality on
camera content B-frames *cost* 7-31%. Do not change the default GOP rule on the strength of this
number until BUG-5 is resolved.

**Not a free win either way:** longer GOPs mean coarser seeking and weaker error resilience, both of which
matter for broadcast contribution. Needs a decision on the default, and probably a smarter rule
than a fixed interval — e.g. never emit a group too short for a full pyramid.

### MEAS-1 — Correct video comparison GNC vs H.264 (**DONE 2026-09-05**)
Harness: `scripts/meas1_vs_h264.py`. VMAF-scored, one normalised reference for both codecs,
BD-rate over the overlapping quality range. 1080p 4:2:0, x264 at defaults.

| | bbb | touchdown | old_town |
|---|---|---|---|
| full video (ki=9) | **+456.7%** | **+493.9%** | **+672.1%** |
| intra only (ki=1) | +54.6% | +46.3% | — |

**GNC needs roughly 5-7x the bitrate of H.264 for the same VMAF on video.** Intra accounts for
about +50%; inter multiplies the gap a further 8-10x. Supersedes the +13.9% spatial figure, which
was PSNR on stills rather than VMAF on video.

The gap is multiples, not percentages. Work targeting single-digit-percent improvements is not
addressing it.

### BUG-6 — Wavelet decomposition capped at 4 levels; 5 panics (todo, P2)
`GNC_WAVELET_LEVELS=5` panics at `rice_gpu.rs:942` (`index out of bounds: the len is 1000 but the
index is 1000`). `MAX_GROUPS = 8` with `num_groups = levels * 2` puts 4 levels exactly at the
ceiling, and the per-tile skip bitmap is a single `u8`, so 8 groups is the hard limit.

JPEG 2000 typically uses 5 levels, and moving 3 → 4 measured 5-17% better (see TUNE-2), so the
cap sits right where the gains might continue. Widening needs `MAX_GROUPS`, `K_STRIDE`, the GPU
buffer layout and a wider skip bitmap — the last is a bitstream change.

Gate before building: offline, levels 5 and 6 add only 0.2% and 0.1% over 4. But offline
understated the 3→4 step by 5x (1.2% predicted, 6% measured), because Rice adapts `k` per subband
and an ideal-entropy model does not see that. So measure in-codec with a temporary widening
before committing to the format change.

### TUNE-2 — Wavelet levels default (**DONE 2026-09-05**)
The quality preset used 3 levels below q=50. Measured 5-17% worse bitrate at equal or better
quality on bbb, touchdown and kristensara at q=25/40/49, at no speed cost. Default is now 4
everywhere.

### Closed by measurement 2026-09-05 — do not re-test
- **Coefficient-level RDOQ**: +0.1% at best (`scripts/meas_rdoq.py`). GNC's uniform quantiser
  plus dead zone is already on its RD curve; this is why every dead-zone and QP sweep moved along
  the curve rather than off it.
- **Per-tile RD bit allocation** (PCRD's idea without truncatable codes): 0% within noise at
  every rate (`scripts/meas_pcrd.py`). A uniform step already equalises the RD slope across tiles.
  JPEG 2000's gain comes from truncating *embedded* per-code-block streams, which Rice cannot do.
  **This retires the repo's standing hypothesis that PCRD accounts for ~89% of the intra gap.**
- **Block intra prediction**: already measured at −11.76 dB / +29% bitrate (hence
  `intra_prediction: false`), and worth only ~+6% to H.264 over JPEG 2000 regardless.

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

### MEAS-4 — Inter-model gap decomposition (**DONE 2026-09-05**)
**Answer: the inter gap is prediction quality, not the coding model.** Full method, numbers and
caveats in [docs/decisions/0005-meas4-inter-gap-decomposition.md](docs/decisions/0005-meas4-inter-gap-decomposition.md).

At matched distortion on GNC's own dumped residuals, an idealised DCT + oracle-block-skip model
beats GNC's wavelet model by only 3.9% (BBB) / 22.6% (touchdown) at q=75, and *loses* by 3.1% /
17.7% at q=25. The decision rule required ≥40% to justify a hybrid inter pipeline. Context-
adaptive entropy coding is worth ≤3.4%. Oracle-skippable blocks at q=75: 2.1% / 0.0% — GNC's
prediction leaves residual energy nearly everywhere, so H.264's skip tool would have nothing to
work with.

x264 ablation on the same content says H.264's own biggest inter lever is multi-reference and
B-frame prediction (+29–32%), three times CABAC (+8–9%) and thirty times sub-block partitioning
(+1%). Both lines of evidence point at prediction.

Tooling, reusable: `GNC_DUMP_RESIDUAL=<dir> GNC_DIAGNOSTICS=1` (4:4:4) dumps spatial MC
residuals; `scripts/meas4_oracle.py` runs 4a/4b/4c.

**Consequence:** #25 (multi-reference P-frames) is promoted out of deferred — see below. Do not
spend effort on per-block inter transforms, block skip, or context entropy for inter.

### 25. Multi-reference P-frames (**WITHDRAWN 2026-09-05** — measured, not worth it)
Promoted to P1 by MEAS-4 on an x264 ablation that turned out to conflate multi-reference with
B-frames. Separated: `--ref 1` alone costs **+0.2% to +5.5%** at matched quality, while
`--bframes 0` costs +22% to +41%. GNC already has B-frames.

The item's own gate (`scripts/meas_multiref_gate.py`) passes on only 2 of 4 sequences (10.1% /
22.0% / 7.8% / 25.8% of blocks preferring frame n−2), and the SAD reduction from best-of-two is
2.1–4.9% everywhere. The sequence chosen specifically as the best case — speed_bag, literally
periodic motion — scores lowest. Expected gain is below the item's own 3% success criterion, for
a bitstream format change.

Revisit only if MEAS-1 shows an inter gap that nothing cheaper explains.

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
in Dirac/VC-2. Beats pure IPB in the literature but is a substantial project. MEAS-4 found the
inter gap to be prediction quality rather than the coding model, and MCTF is a *prediction*
technique, so it stays live as a candidate — but behind #25, which is far cheaper and targets
the same lever.

**Status:** Noted, nothing to implement now.

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
