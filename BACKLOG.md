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

**Positioning fixed 2026-09-05 (see [docs/POSITIONING.md](docs/POSITIONING.md) and GOALS §1): GNC is a contribution codec.** It does not try to
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

MEAS-4's residual dumps were taken with `GNC_DIAGNOSTICS=1`, which BUG-7 (2026-09-06) shows
clobbered the motion-compensation reference from the third frame of every sequence onward. **It has
been re-run on clean dumps and its conclusion holds**: the coding model is not the gap (a
DCT-plus-oracle-skip rival is +5.7% worse), the motion search is not the gap (GNC beats a
full-search oracle by 6%), and context modelling has a 10.4% ceiling. What the corruption did
invalidate is claims about the residual's absolute size, which were about 6x too large.

The honest inter numbers are also better than this list carried: **inter saves 33-73%**, not
17-27%. x264 saves 86-89%, so the gap is real, but it is not the near-total failure the corrupted
diagnostics implied.

With the model, the search and the entropy ceiling all ruled out, what is left is what a mature
encoder does that GNC does not. The first item off that list paid: **TUNE-5** found that GNC spent
the same bits on a P-frame as on an I-frame, and separating them is −3.3% BD-rate at ki=9 and about
−20% at ki=17.

Separately, the old premise had already failed inspection: the "GNC saves 38.5% vs x264's 86.9%"
figure compares GNC's RGB PSNR against x264's YUV PSNR, which are not the same quantity.

**MEAS-1 is therefore a hard prerequisite for any further inter work.** Until there is a
trustworthy, VMAF-based, like-for-like comparison, there is no reliable number saying how large
the inter gap is, and targeting it is guesswork. See RESEARCH_LOG 2026-09-05.

**Known facts (2026-03-11, uniform weights):**
- Spatial BD-rate vs H.264 all-I: **+13.9%** — reasonable for a wavelet codec; GNC wins above ~36 dB
- Spatial BD-rate vs JPEG 2000 (4:4:4): **+28.3%** — gap narrows to ~11% at high quality
- Temporal: GNC I+P+B saves ~17–27% vs all-I (48.9%/29.8% on bbb/touchdown at q=75, 4:4:4).
  x264 saves 86–89% on the same content → the gap is real and large. **Where it lies is again an
  open question — see BUG-7 and the reopened MEAS-4.** For reference, on blue_sky at q=50 a
  P-frame really costs 0.55–0.61 of an I-frame, not the ~1.0 the diagnostics were reporting.
- All above is PSNR-based (RGB). VMAF-based video comparison vs H.264 still missing (MEAS-1).

## Active priority list

### BUG-5 — B-frames stop paying on camera content (**FIXED 2026-09-06** — pyramid off by default)
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

**Finer sweep 2026-09-06: there is no quality crossover. It is content, not quality.** Rate of
the B-pyramid relative to P-only at matched VMAF, per qstep (positive = pyramid costs more):

| qstep | bbb (animation) | old_town | speed_bag | touchdown |
|---|---|---|---|---|
| 4.0 | -34.3% | +5.7% | +26.7% | +7.3% |
| 5.0 | -37.1% | +7.1% | +7.3% | +6.6% |
| 6.0 | -39.1% | +7.1% | +4.0% | +3.9% |
| 7.0 | - | +6.6% | - | +0.8% |
| 8.0 | - | +16.3% | - | -6.5% |
| 9.0 | - | +19.7% | - | - |

The pyramid loses on camera content at nearly every rate point tested, and old_town gets *worse*
at high qstep. It wins 34-39% on animation everywhere. **A quality threshold would have been the
wrong fix**; the earlier reading of "wins at distribution bitrates" was an artefact of integrating
BD-rate over the whole range.

**Chroma caveat closed 2026-09-06.** Re-measured with CIEDE2000 (MEAS-7) at matched rate: the
pyramid's colour error differs by **−0.019 / −0.034 / +0.017 dE00** on the three camera sequences,
against a just-noticeable difference of about 1.0 — one to two orders of magnitude below JND, and
sign-inconsistent. No hidden chroma effect. On animation the pyramid is better on both metrics
(−0.36 dE00, +2.40 VMAF), consistent with it being a content bet. The default is validated on
both halves.

**Fix:** default the pyramid off on the quality-preset path, keep it available via
`GNC_B_PYRAMID=1`. Justified on two independent measurements — rate on camera content, and 160 ms
of reordering latency (MEAS-6) that applies regardless of content.

### BUG-10 — P-frame quality saturates (**CORRECTED AND CLOSED 2026-09-06** — it is TUNE-5)
**The original diagnosis in this entry was wrong.** It is not a structural ceiling and none of the
three suspects it listed (reference precision, interpolation filter, reconstruction clamping) is
involved. The cause is `TUNE-5` (1e238f9): `GNC_P_QP_SCALE` defaults to 1.25, quantising P-frames
25% coarser than intra, and each frame inherits its predecessor's error down the chain.

Verified, touchdown, 8 frames, q=99, ki=8:

| | rate | avg | min (worst P) | stddev |
|---|---|---|---|---|
| scale 1.25 (default) | 11.81 bpp | 49.46 | 45.01 | 5.60 |
| scale 1.0 | 13.00 bpp | **59.80** | **59.77** | **0.03** |

The ceiling vanishes; +10.1% bits for +10.3 dB. The explicit-qstep figures in the original entry
were also confounded — `--qstep` leaves `-q` at its default of 75, so `dead_zone` stayed 0.75.

**What survives is more important than the bug.** With the scale at 1.0, 1I+7P costs 13.00 bpp
against 13.25 bpp for all-intra — motion compensation earns **1.9%** at contribution quality.
TUNE-5 made the inter path look cheap by paying in quality rather than bits. TUNE-5 is not wrong;
it was measured at distribution bitrates. **The lever should follow the operating point** — that is
the open item, see below.

### TUNE-6 — P-frame quantiser scale now follows the operating point (**DONE 2026-09-06**)
`GNC_P_QP_SCALE` was a flat 1.25, right at distribution bitrates (TUNE-5) and wrong at
contribution quality, where it cost 10.3 dB to save 10% of bits (BUG-10, found by the concurrent
session). Now keyed on the **quantiser step**: 1.25× at step ≥ 4.6, tapering linearly to 1.0 at
step ≤ 2.8. Those breakpoints are q=70 and q=85 on the default ladder.

Keyed on the step rather than on `q` because the step is the physically relevant quantity — how
coarse a P-frame may be depends on how much quantisation error there is to hide behind, not on a
preset index — and because `q` is not available in `encode_pframe` at all under `--qstep` or rate
control, both of which set the step directly.

Measured at **matched rate**, PSNR average and worst frame:

| sequence | q | Δ avg | Δ worst |
|---|---|---|---|
| old_town | 65 | +0.94 dB | −0.06 dB |
| old_town | 75 | +0.93 dB | −0.66 dB |
| old_town | 85 | +0.40 dB | **−2.2 dB** |
| old_town | 99 | **−3.8 dB** | **−14.2 dB** |
| aerial | 70 | +0.56 dB | −0.47 dB |
| aerial | 80 | +0.81 dB | −0.52 dB |
| aerial | 90 | −0.21 dB | **−2.75 dB** |

Both sequences turn between q=80 and q=90, and it is the **worst frame** that pays — which matters
more than the average for a contribution codec. Tapered rather than stepped so the RD curve has no
cliff at the boundary. Verified after the change: old_town at q=99 is back to 59.74 dB average and
59.74 worst (was 54.01/43.60), and q=50 keeps the 1.64 bpp the flat 1.25 bought.

**Metric note, and it generalises:** VMAF is useless here. On old_town at q≥85 it reads
99.64/96.80 for both settings while the rate differs 14% and worst-frame PSNR differs 4.8 dB — it
has saturated. **Above about q=80, PSNR leads and VMAF is the cross-check**, the reverse of
CLAUDE.md's usual rule. Recorded in COORDINATION.md.


Measured 2026-09-06, touchdown, 17 frames, ki=8, Rice, 4:4:4. `min` is the worst P-frame, `max`
the I-frame:

| q | P-frame (min) | I-frame (max) |
|---|---|---|
| 85 | 40.52 | 48.95 |
| 92 | 43.02 | 51.40 |
| 96 | 43.68 | 55.19 |
| 99 | **44.01** | **59.88** |

The I-frame gains 10.9 dB across the range; the P-frame gains 3.5 and flattens. With an explicit
qstep the P-frame stops improving entirely below qstep 1.0 (36.82 / 36.47 / 36.59 at 1.0 / 0.5 /
0.25). **Inter frames have a quality ceiling independent of the quantiser.**

**Cost at the contribution operating point:** all-intra buys +9.0 dB (touchdown) and +9.5 dB
(old_town) for 3-5% more bits, and halves the BD-rate gap to x264 (+350% -> +177%, +219% -> +118%).

**This is probably the single largest defect the project has measured**, and it explains several
results previously treated as separate: all-intra beating ki=8 at matched quality, the B-pyramid
ceasing to pay as the quantiser gets finer (BUG-5), and "inter saves 17-27%" holding only at equal
qstep.

**Ruled out:** the inter dead zone (`GNC_INTER_DZ_MUL` at 2.0/1.0/0.0 is byte-identical at q=99 —
`dead_zone` is already 0 there); the B-pyramid (these are P-only); motion search quality (beats an
offline oracle).

**Cause found 2026-09-06 (commit 217cb25) — it is not a ceiling.** `GNC_P_QP_SCALE`, the
deliberate 1.25x coarser quantiser on predicted frames, compounding down the reference chain. Set
it to 1.0 and the ceiling disappears completely. Touchdown, 8 frames, ki=8, q=99:

| frame | default (1.25) | `GNC_P_QP_SCALE=1.0` |
|---|---|---|
| 0 [I] | 59.87 | 59.87 |
| 1 [P] | 57.91 | 59.82 |
| 2 [P] | 48.89 | 59.80 |
| 4 [P] | 46.03 | 59.79 |
| 7 [P] | 45.01 | **59.77** |

Same shape on old_town (I 59.74; P 57.80 → 44.30 by default, flat 59.74 with the lever off). The
first step is **1.94 dB, which is exactly 20·log10(1.25)** — the lever's own cost, paid once. What
follows is that cost being re-paid against an already-coarsened reference, frame after frame, until
it asymptotes. The candidates previously listed here (reference precision, bilinear interpolation,
clamping in reconstruction) are ruled out by the fact that all of them are still present in the
`GNC_P_QP_SCALE=1.0` run, which is flat.

**What this costs, and the second finding underneath it.** Turning the lever off raises the rate
about 9% (touchdown 11.4–12.1 → 12.8–13.3 bpp). At that point a P-frame costs **more than the
I-frame it predicts from** (13.0 against 12.59 bpp). So at contribution quality, motion
compensation on these sequences buys nothing at all — the residual is essentially noise. The lever
was hiding that by paying in quality instead of bits, which is why all-intra looked like a free
+9 dB.

**Fix, not yet made:** the lever is sound where it was measured (q≈35, −3.3% BD-rate) and wrong at
the top of the range. Make it descend with q, exactly as the qstep anchors were fixed for QUAL-1.
Above roughly q=90 it should be 1.0.

**One data point this does not explain.** The handset-qstep row above (36.82 / 36.47 / 36.59 at
qstep 1.0 / 0.5 / 0.25) sits far below the q=99 numbers. Suspect `dead_zone` was not 0 in that run,
so the inter dead-zone doubling was live where at q=99 it is not. Re-check before closing.

**Next step is diagnosis, not a fix.** Encode a P-frame with a forced zero motion field on
byte-identical frames at qstep 0.25 and check whether the reconstruction is bit-exact. That
separates precision from prediction in one run. Full measurement in RESEARCH_LOG 2026-09-06.

### BUG-11 — Rice's stream mapping hardcodes tile width 256 (todo, **P1**)
Rice maps coefficient *i* to stream `i % 256`. At tile width 256 each stream is exactly one tile
column, so the previous symbol in a stream is the pixel above — the property the vertical-context
result (−11.7%) calls free. At any other width the modulus interleaves spatially distant columns
into one stream and the adaptive *k* tracks a mixture.

Measured 2026-09-06 (worktree `d3744f5`, padding-neutral crops): across the q=20/q=25 coder switch
the tile-size effect **reverses sign** on all four images — rANS gains 14–20% going from 256 to 512
px tiles, Rice loses 15–22% — at PSNR matched to 0.04 dB. See RESEARCH_LOG, "Tile size: hypothesis
falsified".

Two consequences. **Every tile-size experiment in this repo, #47 included, was measured through a
coder that penalises the larger-tile arm** — those results do not bound what the geometry is worth.
And any future tile-size change is blocked behind this.

Fix: make the mapping tile-width-aware, so a stream is one column at any width (or a contiguous
column, if the stream count must stay at 256). Then re-run the tile-size sweep.

### ENT-1 — Per-tile frequency tables cost about 9% of the file at 1 bpp (todo, P1)
Fell out of the tile-size measurement. On the rANS path, bbb at q=20, quadrupling tile area (256 →
512 px, so 4x fewer tables per coefficient) saves **14.2%** with per-subband tables and only
**5.5%** with a single table per tile. About 9 points of that is table overhead rather than
transform continuity.

The tables are being amortised by geometry, which is the wrong lever — the geometry is worth ~5%
and is blocked behind BUG-11, while the tables can be attacked directly: share them across tiles
within a plane, or code them differentially against a neighbour or a default. JPEG 2000 and the
video codecs both do a version of this. Worth more than the experiment that found it.

Caveat: single-table mode is a worse coder overall, so the 9/5 split is indicative, not exact.
Measure properly before building.

### BUG-12 — `--tile-size` cannot reach its own wavelet-level ceiling (todo, P3)
`quality_preset` ends with `cfg.wavelet_levels = min(levels, cfg.max_wavelet_levels())`, computed
against the **default** tile size. `main.rs` then sets `config.tile_size = tile_size` from the CLI,
after the clamp has already run. So `--tile-size 512` is capped at 5 levels though a 512 px tile
allows 6, and `GNC_WAVELET_LEVELS=6` is silently ignored with it.

Harmless for the shipped presets (256 px everywhere) but it means every past `--tile-size`
experiment ran with a hidden cap. Fix: re-derive the ceiling after the CLI overrides land.

### BUG-11 — intra prediction produces corrupt output at every quality (todo, P1)
`GNC_INTRA_PRED=1` (added 2026-09-06 to gate the lossless hypothesis) does not reconstruct
correctly at any setting: max error 197-255 from q=50 to q=100, and at q=100 — where there is no
quantiser and the transform is reversible — it loses 62 dB against a bit-exact baseline. PSNR sits
at 33.9-38.3 dB and barely responds to q, which is a systematic reconstruction error rather than a
coding cost.

**The measurement that disabled this feature was measuring the bug.** 49.88 - 38.33 = 11.55 dB at
q=90 against the recorded "-11.76 dB" behind `intra_prediction: false`. It is recorded as "the idea
does not work"; what was shown is "the implementation does not work".

**Where it is:** error accumulates toward the bottom-right of every 32x32 block (row 0 col 0: 6,
row 31 col 31: 200), affecting 99.9% of blocks and 74.6% of pixels. That is the signature of an
encoder/decoder mismatch in a sequential predictor -- encoder predicting from *original*
neighbours, decoder from *reconstructed* ones -- so error compounds along both scan directions.
First block row and column are also dirty, so it is not only boundary initialisation.

**Blocks:** the lossless-prediction hypothesis (RESEARCH_LOG 2026-09-06). GNC loses to FFV1 by 27%
and x264 `-qp 0` by 43% at q=100, and both win by decorrelating against the neighbour rather than
the scale. Whether that transfers here is **untested, not refuted** — it cannot be measured until
this reconstructs.

### BUG-9 — rANS panics below qstep 1.0 instead of rejecting the configuration (todo, P2)
Panics with `range start index 4294963272 out of range for slice of length 5242880` at
`rans_gpu_encode.rs:1813` in `pack_tiles`. Measured 2026-09-06: rANS OK at 2.0 and 1.5, panics at
1.0; Rice is fine to at least 0.5.

**Repro line corrected.** `--qstep 1.0 --rans` does *not* reproduce it — `--rans` is a no-op flag
kept for backward compatibility, so that command encodes with Rice and succeeds. rANS is reached
through the preset only, so the panic needs `gnc encode -q 15 --qstep 1.0`.

**Stated cause was wrong.** This is not the symbol alphabet. `rans_encode.wgsl` writes each stream
*backwards* from the end of a fixed 4 KB buffer (`MAX_STREAM_BYTES = 4096`, `write_ptr` starting at
the top and decremented with no bound check). When a stream needs more than 4 KB the pointer
underflows: 4294963272 is 2^32 − 4024, i.e. the stream overran by 4024 bytes. Rice survives the
same qstep for two reasons — it sizes that buffer from qstep (`max_stream_bytes_for_tile`) and it
carries a per-tile overflow flag that the shader sets. rANS has neither.

**Recommended fix — the cheap half only.** Bounds-check `write_ptr` on the host (about three lines:
`write_ptr > MAX_STREAM_BYTES` means that stream overflowed) and return a named error, or port
Rice's overflow flag. Do **not** make rANS work at fine qstep: it is selected only at q<=20 where
4 KB is ample, and it measures worse than Rice above q=20, so the capability has no user. The
value here is turning a wrapped-pointer crash into a sentence.

### QUAL-1 — Re-run MEAS-1 at the contribution operating point (todo, P1)
The quality ladder above q=92 was dead until 2026-09-06 — q=92, 96 and 99 produced the same
picture, capped at qstep 2.0 by an rANS constraint that no longer applied. **Every
contribution-quality comparison in this repo predating that fix is invalid at the top of the
range**, because GNC was pinned while the competitor was not. The distribution-bitrate figures
(+306% to +617%) are unaffected.

Re-run MEAS-1 with `--q 92,96,99` against `--crf 4,8,12` now that GNC reaches 60.7 dB where it
previously stopped at 50.5. Note VMAF saturates around 99.8 and is useless at this end; use PSNR
and CIEDE2000.

### MEAS-5 — Concurrent streams per GPU vs NVENC (partly answered 2026-09-05, P0)

**The thesis was never one claim. It is two, and they are not equally strong.**

**Claim A — "no session cap, and it runs where NVENC does not" — HOLDS. Fully sourced.**
- NVENC's consumer limit is **12 concurrent sessions per *system***, explicitly *"the combined
  number ... on all non-qualified cards present in the system"*. **A second GeForce buys zero.**
- **A100, H100 and B200 ship with zero NVENC.** NVIDIA's Hopper whitepaper states it outright.
  The most valuable GPUs in the world cannot encode video at all — an idle AI fleet has no encode
  capacity whatsoever. This is the strongest and most under-used fact we have.
- **GeForce driver licence §2.8 prohibits datacenter deployment**, so NVENC at density legally
  requires professional or datacenter SKUs regardless of the session counter.
- Engine counts are flat or sublinear against compute (Ampere: 1 NVENC across 4.2x the SMs), and
  **per-engine throughput grew +14% from Turing to Blackwell while shader FP32 grew ~6x.**

**Claim B — "more aggregate throughput than the card's own NVENCs" — STILL UNPROVEN, and the
first local measurement is sobering.** M1, 1080p, N concurrent encode processes:

| instances | aggregate fps (two runs) |
|---|---|
| 1 | 7.02 / 6.22 |
| 2 | 11.13 / 9.51 |
| 4 | 11.51 / 12.23 |
| 8 | 14.15 / 13.38 |

**~2x aggregate at N=8, most of it already at N=2.** A single 1080p encode does not saturate the
M1, so there is real headroom — but it is far from linear, and the published multi-tenancy
literature agrees: concurrency converts *idle* GPU into *useful* GPU, it does not create GPU.
NVIDIA's own consolidation study measured time-slicing at 0.76 req/s where MIG gave 1.00.

**Still to do:** the same measurement on a discrete NVIDIA card, head to head against NVENC at
both P1 and P7 presets (P7 is nearer GNC's quality target and roughly 4x easier to win), and on an
H100 where the NVENC column is a zero.

**Blocked on a definition problem — fix this first.** At BASELINE's own stated parameters this
session measured **13.6 fps** for the GPU encode phase (`benchmark-sequence`) and **7.8 fps** end
to end (`encode-sequence`, incl. PNG decode and container write), against BASELINE's stated
**31.7 fps**. The CLI's own help concedes PNG input inflates the cost. Three numbers are in
circulation for "GNC encode fps" and GOALS quotes one without saying which. **Pin the definition
before any density claim rests on it.**

### MEAS-6 — Latency per frame (first pass done 2026-09-06, P1)

**The B-pyramid costs 8 frames of lookahead before any coding runs.** From the encoder's own
diagnostics, `ki=17` encodes in the order `0[I] 4[B] 8[P] 2[B] 6[B] 1[B] 3[B] ...` — frame 1
cannot be encoded until frame 8 has arrived. At 50 fps that is **160 ms of structural delay**.
`ki=8` (P-only) encodes in display order: **zero reordering delay**. This is not a tuning
parameter; it is what a hierarchical pyramid is.

Coding time, 1080p, M1, all-intra: GPU encode ~47 ms/frame, decode ~35 ms/frame (upper bound,
includes PNG write), **codec round trip ~80 ms**.

| | latency |
|---|---|
| JPEG XS | 1-32 lines; EBU measured < 1 frame |
| NDI High Bandwidth | < 16 ms |
| **GNC, intra or P-only** | **~80 ms** |
| **GNC, B-pyramid (default)** | **~240 ms** |
| low-latency HEVC | 120-3060 ms (EBU, real vendors) |

**GNC's default configuration sits in the low-latency-HEVC band, not the JPEG XS band.**

**Converges with BUG-5.** The B-pyramid already measured as *costing* 7-31% at contribution
quality on camera content. It now also costs 160 ms. Two independent measurements, one
conclusion: **the hierarchical B-pyramid is the wrong default for this operating point.** This is
now a supported configuration change rather than a hypothesis. It does not argue against inter
coding — P-frames have zero reordering delay and were the better performer at contribution quality.

**Still to do:** glass-to-glass instrumentation (capture-to-input, output-to-network,
output-to-display are all unmeasured). Note the ~256-line tile floor is not currently reachable:
the pipeline processes whole frames, so the practical floor is one full frame regardless of tile
size.

### CANARY-1 — Encode time must move across GPU tiers (todo, P1)
BeHardware's 2011 study found the shipping GPU H.264 encoders performed *identically* on 100 EUR
and 330 EUR cards, because they were never compute-bound at all — the GPU was doing far less than
the marketing implied. That is exactly the silent-feature failure CLAUDE.md's quality rules exist
to catch.

**If GNC's encode time does not move between the M1 and a discrete GPU, the pipeline is not
running where we think it is.** Add a cross-tier scaling check to the regression suite and keep it
permanently. Cheap, and it guards the single assumption the whole project rests on.

### FMT-1 — 10-bit support (**DONE 2026-09-06**)
**Measured 2026-09-06 (MEAS-8): 8-bit, not compression, is what limits GNC's colour fidelity.**
A single-LSB perturbation of an 8-bit image already gives p95 dE00 of 1.16–1.98 and puts
8.5–36.6% of pixels above the just-noticeable difference. GNC at q=99 measures *better* than that
(p95 1.07 against 1.95 on kristensara). So the codec is already operating below the floor its own
container format imposes, and no quality setting can cross it — which is why q=99 is barely
better than q=92.

For a codec positioned on contribution, where the output feeds grading, that makes bit depth the
first-order problem and the compression tuning second-order.

8-bit is the main format gap for broadcast contribution. 4:2:2 and 4:2:0 already work, so bit
depth is the remaining piece. Cheap now — GOALS rule 10 says the bitstream can still break freely
and there are no users — and expensive once there is a spec, conformance streams and deployments.

Not a research item: buffer formats, upload/download paths, PNG/Y4M I/O, bitstream fields, and the
VMAF/PSNR comparison harness at 10-bit. No f64 needed; 10-bit fits f32/i32 comfortably.

**Still-image path fixed 2026-09-06.** The decoder wrote 8-bit PNGs regardless of what the frame
was coded at — all three call sites used `save_image_rgb_f32` (hardcoded 8) rather than the
`_bits` variant beside it — so a 10-bit encode was truncated at the last step. Fixed; the decoded
PNG now reads IHDR bit depth 16 and a 10-bit q=100 round-trip is bit-exact. On a smooth 10-bit
gradient the 10-bit path measures dE00 0.0028 against the 8-bit path's 0.1669, for 45% more bits.
Guarded by `test_10bit_survives_the_frame_header`.

Needed `scripts/png16.py`, since Pillow can neither write nor read 16-bit RGB PNGs (it truncates
silently on open) — without it the measurement would have shown no benefit and looked like a
codec failure.

**Video path done 2026-09-06.** `encode-sequence` now takes `--bit-depth`, wired through frame
loading and `CodecConfig`. Verified: decoded frames are 16-bit PNGs, an I-frame at q=100 is
bit-exact, and P-frames differ by 8–11 units of 1023 — motion compensation, which is not lossless
at any q, rather than a bit-depth defect. Guarded by
`test_10bit_survives_the_sequence_container`.

The encode side and the bitstream already handled 10 bits; the gaps were a missing CLI flag and
three decoder call sites writing 8-bit output unconditionally.

**Confirmed on genuine content 2026-09-06.** Two Sintel frames from Xiph's `sintel-4k-png16` set
(really 16-bit, 781/840 distinct levels against 196/212 at 8 bits), cropped to 1920x1088:

| | q=55 | q=70 | q=85 |
|---|---|---|---|
| 8-bit | dE00 0.408 / 0.364 | 0.376 / 0.350 | 0.348 / 0.316 |
| 10-bit | 0.147 / 0.147 | 0.139 / 0.150 | **0.086 / 0.080** |

Tripling the bitrate improves 8-bit colour accuracy by 13-15% and 10-bit by 42-45%. **At matched
bitrate 10-bit is 2.1-2.4x more accurate.** So 10-bit is a better use of the same bits for colour
fidelity, not just a format checkbox.

**Harness plumbed 2026-09-06.** `meas1_vs_h264.py --depth 8|10` drives the whole chain. Two
codec defects surfaced doing it: the **Y4M reader parsed the colourspace tag but discarded the
bit-depth suffix**, reading any 10-bit file as 8-bit (half the samples, noise out); and
`benchmark-sequence` had no `--bit-depth` and six hardcoded 8-bit PNG load sites. Both fixed.

First 10-bit numbers on Netflix Chimera (intra-only): BD-rate +131% VMAF / +251% PSNR-Y, worse
than the +46% measured on 8-bit intra elsewhere. One sequence, and a hard one — a dark interior
where GNC's VMAF saturates above 97 by q=25. Needs more 10-bit material before it means anything.

**Old note:** the RD harness still measures at 8 bits. The chain is verified working
(`ffmpeg -strict -1` for 10-bit Y4M, x264 `--input-depth 10 --output-depth 10 --profile high444`,
vmaf scores 10-bit Y4M directly) and the source-material problem is solved — `sintel-4k-png16`
for stills, Netflix Chimera 10-bit Y4M on the same server for video. Plumbing, not new work.

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

### BUG-4 — Tile-skip used an absolute threshold (**DONE 2026-09-06**)
Not the BUG-3 family after all — the odd-tile-column guess was wrong. 4:4:4 was affected too, so
not chroma, and only P-frames, so not the shared path.

`tile_skip_motion` declared a tile static when its mean zero-MV SAD fell below `0.5 · qstep` and
zeroed all its motion vectors. That mean is taken over a whole tile, so its meaning depends on
tile area: at 256px a tile with a moving object still contains enough static background to stay
above the threshold; at 128px the same motion fills the tile and drops under it. Tiles with real
motion were being told they were static.

Now compares against the motion the search found — skip only when zero-MV error is also no worse
than the motion-compensated error (`GNC_TILE_SKIP_MC_MARGIN`, default 0). Measured at q=70, 4:2:0:
bbb **+0.36 VMAF** net at tile 256 and **+2.1** at tile 128; touchdown neutral at both. Positive
at the default size too — the old rule was slightly wrong everywhere and only visibly wrong when
tiles were small.


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

### TUNE-1 — Default keyframe interval (**CLOSED 2026-09-06** — keep the default)
Re-measured after BUG-5 turned the B-pyramid off. With P-only coding, GOP length is worth
**−1.7% to +2.2%** at matched VMAF across four sequences — nothing. The −24% below came entirely
from the pyramid, not from GOP length. The seeking and error-resilience arguments for a short GOP
now win uncontested. On camera content *shorter* is even mildly cheaper (−1 to −12%); only
animation prefers longer.

**Spun off as an open question:** pushing the same sweep to `ki=1` showed the repo's standing
"inter saves 17–27% vs all-I" is an **equal-qstep comparison** — at equal qstep inter saves 17–56%
but is also 1.2–3.9 VMAF worse. At matched quality all-intra is cheaper by 39% (old_town) and 12%
(touchdown) on VMAF, but PSNR disagrees in sign on touchdown (+5.4%). Needs more rate points and a
chroma cross-check before anything is concluded. See RESEARCH_LOG 2026-09-06.

### Superseded detail — original TUNE-1 (measured with the B-pyramid on)
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

### MEAS-8 — What quality does colour fidelity require? (**DONE 2026-09-06**)
Measured on four images with `scripts/chroma_metric.py`. Mean dE00 crosses the JND of 1.0 at
about q=70. For the stricter and more relevant criterion — 95% of pixels below JND — q≥85 on easy
content, q≥92 on faces and skies.

**And the limit is 8-bit, not the codec.** Perturbing every pixel by a single LSB gives p95 dE00
of 1.16–1.98 and puts 8.5–36.6% of pixels above JND. GNC at q=99 measures *better* than that
(p95 1.07 on kristensara against 1.95 for one LSB). Lab is strongly non-linear in dark and
saturated regions, so no quantiser setting can cross that floor in 8 bits — which is why q=99 is
barely better than q=92.

**Consequence: FMT-1 (10-bit) is the binding constraint on contribution-grade colour, not
compression.** Promoted accordingly. It also bounds what any future chroma work can be worth.

### MEAS-8 — original statement
Measured while settling `chroma_weight`: at the current default, mean CIEDE2000 sits at
**1.0–1.5 across q = 30–70** on bbb and kristensara — at or above the nominal just-noticeable
difference of 1. Only bbb at q=70 (0.77) is comfortably below.

GNC is positioned as a contribution codec, and contribution feeds grading and further processing,
where colour fidelity has to survive. So the operating point is not a free choice: the codec needs
a documented minimum q for colour error below JND, per content class. Present evidence suggests
roughly q≥70 for easy content and higher for faces, but that is two images.

Measure across the full test set with `scripts/chroma_metric.py`, then state the floor in GOALS.

### Chroma weight — settled, do not re-sweep on VMAF alone
Raising `chroma_weight` from 1.3 to 2.0 or 3.0 moves bits from chroma to luma: at matched rate,
VMAF rises 0.26–1.27 and dE00 worsens 0.013–0.141. A genuine trade, not the free 15% a VMAF-only
sweep suggested. **Left at 1.3** — contribution feeds downstream grading, so trading colour
fidelity for luma sharpness is the wrong direction for this market.

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

### BUG-7 — Diagnostics corrupted the encoder: 32% larger files with `GNC_DIAGNOSTICS=1` (**FIXED 2026-09-06**)
The temporal-wavelet diagnostic ran a second full wavelet transform through the encoder's *shared*
GPU buffers, clobbering the motion-compensation reference. Every P-frame after the second then
encoded against garbage. blue_sky, 8 frames, q=50: **2,808,848 bytes quiet vs 3,703,862 with
diagnostics (+31.9%)**. Residual Y mean-abs 2.5 real, 14.7 reported — i.e. reported as large as
the raw frame difference, meaning MC contributing nothing.

Gated behind `GNC_DIAG_TWAV=1`; a diagnostics-enabled run is now byte-identical to a quiet one.

**What it invalidates:**
- `ratio_vs_iframe` and every "temporal prediction may not be effective" warning. Real ratios on
  blue_sky q=50 are **0.55–0.61**, not the 1.02–1.06 that was being reported and believed.
- Residual statistics from the third frame of any sequence onward.
- **MEAS-4 (reopened below).** Its residual dumps used `GNC_DIAGNOSTICS=1`.
- The bit-budget shares in FMT-2's first write-up. Corrected: tile headers ~4% of an I-frame, ~6%
  of a P-frame. The GP17 *gain* is unaffected — measured on file sizes with diagnostics off.

MEAS-1's 5–7x figure is unaffected (`meas1_vs_h264.py` encodes without diagnostics).

**Regression test:** `tests/diagnostics_neutral.rs` encodes six synthetic frames twice, with and
without diagnostics, and asserts byte-identical output. Verified to fail when the diagnostic is
re-enabled (+72.1%). Synthesises its own frames, and sets `keyframe_interval = 9` — the default
preset is all-intra and cannot exercise a P-frame bug.

**Why it hid so long:** the symptom looked like a codec result ("P-frames cost as much as
I-frames") rather than a bug, so it was recorded as a finding. And it was perfectly reproducible,
which read as evidence it was real — reproducibility separates a bug from noise, not a codec
property from an instrumentation artefact.

### EBCOT — evaluating in halves (**part 1 closed 2026-09-06**, part 2 open)
Proposed by the project owner. Well aimed: it targets the one mechanism this repo's log said could
not be tested by proxy — *"JPEG 2000's gain comes from truncating embedded per-code-block streams,
which Rice cannot do"*. EBCOT has two separable halves and they are being measured separately
before anything is built.

**Part 1 — PCRD-opt rate allocation: 0.00 dB. Closed.** `scripts/meas_ebcot_pcrd.py`. The existing
0% result was at *tile* granularity and did not bound EBCOT, because a 256px tile averages every
subband and kind of content while a 64px code-block is homogeneous. Re-measured at code-block
granularity: **+0.01 dB (bbb 64px), +0.00 (bbb 32px), +0.00 (blue_sky), −0.00 (touchdown)** — zero
at every rate from 0.05 to 3.5 bpp.

The reason is structural, which makes it more convincing than the number: uniform scalar
quantisation of a near-orthonormal transform under MSE puts every coefficient at the same RD slope,
and that slope depends on the *step*, not on the coefficient or its neighbours. Re-allocating
between groups cannot find a gain that is absent at the coefficient level — and coefficient-level
RDOQ already measured +0.1%. Granularity was never the issue.

**Part 2 — context-modelled bit-plane coder: about −9%. BUILD IT.**
`scripts/meas_ebcot_context.py`. Conditional entropy of every coded bit under EBCOT's context model
(9 zero-coding contexts by band orientation, sign contexts, 3 refinement contexts), against a
faithful simulation of GNC's own coder on the **same coefficients** — Rice+ZRL with per-band k, cut
into 256 interleaved independent streams each charged a length field.

| image | qstep 4 (GNC's operating point) | qstep 8 | qstep 16 |
|---|---|---|---|
| touchdown | −0.1% | −3.5% | −9.3% |
| bbb | −6.2% | −5.8% | −5.9% |
| kristensara | **−15.0%** | −18.7% | −22.3% |
| blue_sky | **−15.6%** | −16.6% | −17.5% |
| mean | **−9.2%** | −11.2% | −13.8% |

Take qstep 4 as the headline: those luma rates (0.98-1.75 bpp) match GNC's real operating point.
**~9% mean, 0-16% by content**, and roughly a third of the +28.3% intra gap to JPEG 2000 — which is
unsurprising, since it *is* JPEG 2000's coder. Larger than everything shipped today put together
(−5% BD-rate).

**The parallelism objection is answered.** GNC's 256-way stream split costs under 1% at qstep 4 and
3-5% at qstep 16 against one stream per subband. Independence is nearly free here, and EBCOT
code-blocks are independent, so a GPU implementation keeps the architecture.

**Why this disagrees with the "context-adaptive entropy ≤3.4%" entry:** that came from
`GNC_SIG_CONTEXT`, which models *two* signals (above-neighbour, parent-subband). EBCOT's model is
nine orientation-separated zero-coding contexts plus sign and refinement contexts, applied per
bit-plane. The old figure correctly measured a much weaker model and was read as a verdict on
context modelling in general.

**Corrected 2026-09-06 after measuring the table cost — build an adaptive binary context coder,
not JPEG 2000's EBCOT verbatim.**

GNC's 256 streams map coefficient *i* to stream `i % 256`, so in a 256-wide tile **each stream is
one tile column**: the vertical neighbour is already decoded, free, in the current architecture. A
vertical-magnitude context measures **−11.7% mean** at qstep 4 — better than EBCOT's own
significance-only contexts (−9.2%) and needing no restructuring. The full neighbourhood is −16.4%
but needs EBCOT's per-code-block sequential model, which costs the 256-way decode.

**But the whole gain depends on table cost, and that is why bit-planes exist.** Charging table bits
per alphabet symbol per context, mean at qstep 4: 8 bits → −10.9%, 16 → −8.6%, 32 → −4.0%,
**64 → +5.1%, a loss.** GNC's rANS signals *static* tables per tile, and a 6-bucket context
multiplies its table count from 10 to 60 — while rANS already loses to Rice above q=25 *because of*
per-group table cost (+8% to +32%, measured today).

An adaptive binary arithmetic coder carries no tables at all: contexts adapt as the decoder
decodes. To use one you need binary decisions, and that is what bit-plane decomposition is for. So
the bit-planes are not (here) about truncatability — part 1 measured that at 0.00 dB — **they are
what makes the contexts affordable.** EBCOT pays ~7 points of coding efficiency to get contexts for
free.

Plan: adaptive binary coder, bit-planes, **GNC's vertical magnitude context rather than JPEG 2000's
neighbourhood**, no PCRD. Expected −9% to −12%. Ship as a fourth `EntropyCoder` variant, gated and
measured against Rice at every quality like the other three.

**RESOLVED 2026-09-06 — build EBCOT's code-block design. −13.7% mean.**

Two wrong turns first, both instructive. GNC's 256 streams map coefficient *i* to stream `i % 256`,
so each stream is a tile column and the vertical neighbour is free — that much is true, and it
suggested keeping the existing streams and just adding a context. It does not work: pooling all
256 streams' statistics assumes **shared** probability estimates, and a parallel decode cannot
share them. With each stream adapting on its own ~256 symbols, the gain collapses from −6.6% to
**−0.7%** (warm start) or **+2.4%** (cold). So GNC's 256-way-per-tile parallelism is what makes
context modelling unaffordable — not through table cost, which was the previous hypothesis, but
through *statistics*: 256 symbols is too little to learn 18 context probabilities on.

**Code-blocks exist to solve exactly that.** A 64×64 block gives one coder 4096 symbols, and its
raster scan makes the full neighbourhood available rather than only the vertical. The parallelism
objection dissolves: a 1080p luma plane holds ~450 independent 64×64 code-blocks — ample GPU work,
even though it is not 256 per tile. Parallelism at *frame* scale was never the constraint.

Measured with cold-start adaptation (KT learning cost, no signalled tables) and a per-block length
field charged, at qstep 4 (GNC's operating point):

| image | code-block 64 | code-block 32 |
|---|---|---|
| touchdown | **−7.6%** | −6.2% |
| bbb | **−11.0%** | −9.8% |
| kristensara | **−18.0%** | −16.7% |
| blue_sky | **−18.3%** | −16.4% |
| mean | **−13.7%** | −12.3% |

Positive on all four, worst case −7.6%. Largest single-mechanism gain measured in this repo, and
roughly half the +28.3% intra gap to JPEG 2000 — as one should expect from adopting JPEG 2000's
coder.

**Build:** independent code-blocks, adaptive binary contexts, full neighbourhood, coefficient-major
scan inside a block. **Drop:** PCRD (part 1: 0.00 dB) and plane-major scan — its embedded
truncatability buys nothing here and costs the richer full-magnitude context a coefficient-major
scan allows.

**The one real risk: decode throughput.** Rice decodes 256 branch-free streams per tile. This is
~450 serial adaptive-binary coders per plane at several binary decisions per coefficient:
per-symbol cost up a lot, parallelism per tile down 16×. For a codec judged on concurrent streams
per GPU and latency, a 13.7% rate win that halves throughput may not be a win.

**Part 3 — CPU reference built and measured in-codec: −19% to −25%.** `src/encoder/abac.rs`
(adaptive binary arithmetic coding over code-blocks, textbook WNC coder) and
`src/encoder/abac_compare.rs` (`GNC_ABAC_COMPARE=1`, codes every tile of a real encode twice).
Against the **shipped** Rice tiles on identical coefficients:

| image | q=40 | q=55 | q=70 |
|---|---|---|---|
| bbb | −21.9% | −19.2% | −16.7% |
| blue_sky | −22.9% | −21.0% | −18.2% |
| touchdown | −24.4% | −22.1% | −19.3% |
| kristensara | **−25.5%** | −23.4% | −22.1% |
| mean | **−23.7%** | **−21.4%** | **−19.1%** |

Three checks, because a result this large is likelier to be a bug than a breakthrough: (1) Rice is
dispatched over the same three buffers the comparison reads back — the same coefficients, not an
equivalent signal; (2) every block is decoded and asserted equal to its input, and the subband
cutting asserts it covered `tile_size²` coefficients, which is the one failure mode no roundtrip
test would catch; (3) the first baseline was the CPU reference Rice and gave −35%, caught because
it beat the offline conditional-entropy ceiling, which is impossible.

It still exceeds the offline −13.7% ceiling, and those numbers are not comparable: the offline run
used a Python DWT with different normalisation, no AQ and a different crop, and its Rice baseline
was idealised with no per-tile headers. Part of the in-codec win is header structure — 25
code-blocks at 2 bytes against Rice's 16-byte header plus per-group k plus 256 length fields.

**Throughput, single-threaded and unoptimised:** 77-108 Mcoeff/s decode, so ~100 ms for a padded
1080p 4:4:4 frame on one core. Not fatal: a frame holds ~3000 independent code-blocks, so the work
is parallel at frame scale, and a production binary decoder is several times faster than this
textbook one. **Plausible-to-proceed, not a green light on fps.**

**Next, in order:**
1. **GPU decode shader and honest fps against Rice on an idle machine.** This is the gate.
2. Bitstream integration — a GP18 generation with `EntropyCoder::Abac`, code-block length fields,
   block size in the tile header. Nothing is integrated yet; `abac` is standalone and
   `abac_compare` is a diagnostic.
3. Inter frames. All of the above is intra; residual statistics differ and contexts may need
   re-tuning.

**Code-block size settled: 128px, i.e. one block per subband.** Swept on bbb at q=55: 16px
**+1.1% — worse than Rice**, 32px −15.1%, 64px −19.2%, 128px −20.0%, 256px identical to 128 (no
subband exceeds 128). Verified across images at 128px: blue_sky −24.5/−22.1/−18.9%, touchdown
−26.1/−23.2/−19.9%, kristensara **−27.6**/−25.0/−23.3% at q=40/55/70. Parallelism remains ample:
~1900 independent code-blocks per 1080p frame.

That 16px loses to Rice is the **third independent confirmation of one mechanism** today: a
context-adaptive coder needs symbols to learn on. It killed the 256-stream variant (256 symbols
each), it orders the block-size sweep, and it is why EBCOT uses code-blocks at all.

The 8×8 deep-subband concern flagged earlier is real but immaterial: LL plus the three level-5
subbands are 256 of 65536 coefficients, 0.4% of a tile, and they are already one block each. Not
worth merging subbands into a shared coder. Dropped.

### BUG-8 — The encoder's local decode diverges from the real decoder down a GOP (**OPEN, not diagnosed**)
bbb17, 17 frames, ki=17, q=50. Per-frame PSNR of the encoder's own reconstruction vs the actual
decoded file: gap −0.04 dB at frame 0, +0.08 at frame 8, **+0.23 at frame 16** — monotonic,
accumulating, and in the decoder's favour.

The bitstream is valid and the decoded output is slightly *better* than the encoder believes, so
this is not a correctness failure of the output. It is a correctness failure of the encoder's
model: the local decode loop is what rate control and any RD decision reads, and by the end of a
GOP it is wrong by a quarter of a dB.

**Not the deblock filter** — the divergence persists with `GNC_REF_DEBLOCK` off. Cause unknown.
Any RD or rate-control work should treat encoder-internal PSNR as suspect until this is diagnosed.
First step: find whether the local decode's reference buffer differs from the decoder's after one
P-frame, and if so by what.

### MEAS-2 — Feature toggling: what contributes and how much? (in progress 2026-09-06)
First toggle measured: **`GNC_REF_DEBLOCK` — neutral to negative, default flipped off.** Its own
commit measured 0.016% bpp / VMAF neutral and explained why (tile-boundary pixels are 0.78% of ME
decisions); re-measured with VMAF it is exactly neutral on old_town and aerial and *worse* on
bbb17 (+0.66 VMAF min with it off at q=30). Code retained rather than deleted — 0.1 VMAF margin,
correct implementation, and a second agent is editing this tree. Delete it if nothing revives it.

**Second: CfL — keeps its place, but only visible with the right metric.** On VMAF alone it reads
as a loss on two of three images (costs 2-3% rate, loses VMAF). VMAF is luma-only and CfL is a
chroma tool, so it can see the cost and not the benefit. On CIEDE2000 it is better on both axes at
once on bbb (9% smaller *and* better colour) and reaches a colour accuracy the others need 3-14%
more rate to match. **Any MEAS-2 toggle touching chroma must be scored with dE00** — the naive
sweep would have deleted a working feature.

**Third: AQ — gradient was inverted, now off below q=30.** BD-rate of turning AQ off:
**−4.58% at q=15-30** (AQ was costing rate where the old rule set its strength highest), −0.01% at
q=30-55, **+1.82% at q=55-80** (AQ earns its keep). New rule: off below 30, strength 0.15 from 30
to 80; re-verified +1.63% mean at that strength, positive on all four images. Strength is noise in
the upper range (0.15/0.20/0.30 all within 0.07 VMAF).

The TUNE-4 disagreement is the same error class as MEAS-4's equal-qstep comparison: point VMAF at
fixed q read "+0.1 to +0.55 VMAF for under 1% rate", but at matched quality the trade is 4-7% of
the bitrate. Also partly self-inflicted — BUG-6 changed the wavelet level count at q≥25 the same
day, and AQ measures variance on the LL subband.

**Fourth: Rice vs rANS — q≤20 boundary survives, for a new reason.** rANS keeps a ~2% mean edge
below q=20 and loses 8-32% from q=25 up. The cliff is at *exactly* q=25, where BUG-6 switches 4
wavelet levels to 5: rANS carries a frequency table per subband group, so a 5th level costs it two
more tables per tile, while GP17 made Rice's length table cheaper. **Re-check this boundary if the
wavelet-level rule moves again.** Tightening 20→15 rejected — 1.1% mean difference and neither
boundary is clean (rANS loses 5.7-8.2% on kristensara throughout its own range).

**Fifth: motion tile-skip — on the RD curve, kept.** At matched rate on aerial: skip on gives
0.91 bpp / 88.87 VMAF mean / 85.05 min; skip off gives 0.92 bpp / 88.58 / 85.71. +0.29 mean and
−0.66 min, i.e. trading tail quality for average, essentially on the curve. At matched *q* it looks
dramatic (12-31% rate for 0.2-1.7 VMAF) but that is movement along the curve.

Remaining: pyramid QP scales and B-pyramid (both behind BUG-5's pyramid-off default, so lower
value). MEAS-2's main finding is that three of five toggles were mis-tuned or mis-measured.

### MEAS-4 — Inter-model gap decomposition (**RE-RUN AND CLOSED 2026-09-06** on clean data)
Reopened because its residual dumps were taken with `GNC_DIAGNOSTICS=1` (BUG-7), which clobbered
the MC reference from the third frame of every sequence onward. Re-dumped with the diagnostic
gated off — **verified bitstream-neutral first**: the dump run and a quiet run produce
byte-identical files.

**The conclusion survives.** 4b (blue_sky, q=50, at the wavelet's operating point): wavelet
2.1528 bpp vs DCT-plus-oracle-skip 2.0291 — **the rival model is +5.7% worse**. 4c: context
modelling could recover at most **10.4%** of coefficient bits. `meas_me_quality.py` on clean
dumps: the offline full-search oracle is **6.0% worse on SATD** than GNC's shipped search
(previously reported 20.8% — the margin shrank by two thirds, the direction held).

**Why it survived a corrupted input:** 4b and 4c are ratios between two models simulated on the
*same* residual, so a corrupted residual moves both arms together and largely cancels. What the
corruption did invalidate is every claim about the residual's **absolute** size — "the prediction
is leaving error nearly everywhere" rested on magnitudes about 6x too large.

**Honest inter numbers** (diagnostics off, all-I vs I+P, 8 frames): blue_sky saves 38.3% / 37.4% /
33.0% at q=30/50/75; bbb17 saves 72.7% / 65.4% / 49.9%. **Inter saves 33-73%**, not the 17-27%
this backlog carried. x264 saves 86-89% on comparable content — the gap is real, but not the
near-total failure the corrupted diagnostics implied.

**What clean data rules out, all at once:** the coding model, the motion search, and context
modelling. None is the multiple-x deficit. What is left is what a mature encoder does that GNC does
not — starting with rate allocation, see TUNE-5.

### TUNE-5 — P-frames were quantised as finely as I-frames (**DONE 2026-09-06**)
`GNC_P_QP_SCALE` defaulted to 1.0: identical quantiser step for intra and predicted frames. Every
mature codec separates them (x264 runs P about 4 QP steps coarser, ~1.6x) because the I-frame is
referenced by every P that follows, so bits spent there are reused and bits spent on a P frame are
not. **Default is now 1.25**; the env var still overrides.

VMAF BD-rate against 1.0, ki=9:

| sequence | 1.15 | 1.25 | 1.5 |
|---|---|---|---|
| blue_sky | — | **−0.59%** | +2.93% |
| aerial | −0.63% | **−0.51%** | −2.36% |
| bbb17 | — | **−5.27%** | −8.11% |
| old_town | −6.00% | **−6.77%** | −6.39% |
| mean | −3.32% | **−3.29%** | −3.48% |

1.25 is better on all four. 1.5 gains nothing more on average and regresses blue_sky, so 1.25 is
the pick. **The win grows with GOP length** — on old_town at ki=17, 1.25 reaches the same VMAF as
1.0 at 0.65 bpp against ~0.82, about **−20%**.

**This reverses the 2026-09-05 rejection** ("worse than lowering q uniformly; VMAF min falls 94→71
as reference error propagates"). That was the right thing to check — coarser P degrades the
reference each P predicts from, and a mean hides a collapsing tail. It does not reproduce.
Mean-vs-min spread on old_town at ki=17, q=35: 3.32 VMAF points at 1.0, 2.75 at 1.25 — the spread
*narrows*, and still narrows at 1.6. Decisive test at matched **rate** rather than matched q:

| | bpp | VMAF mean | VMAF min |
|---|---|---|---|
| scale 1.0, q=28 | 0.66 | 82.00 | 78.14 |
| **scale 1.25, q=35** | **0.65** | **84.07** | **81.32** |

At the same bitrate the coarser-P encode is **+2.07 VMAF mean and +3.18 VMAF min** — the worst
frame is better, not worse. A plausible explanation for 94→71 is that it was measured with
`GNC_DIAGNOSTICS=1`, which BUG-7 shows destroys P-frame prediction from the third frame onward:
exactly the frames where a propagation argument would look confirmed.

**Flagged:** per-frame PSNR now declines across a GOP (blue_sky q=50: 41.3 → 35.8 dB, was
41.3 → 38.3). That decline is real, and is what a lower-rate operating point looks like; at matched
rate the floor is higher. PSNR and VMAF disagree in sign here, and VMAF is primary.

### FMT-2 — Stream-length tables cost more than the coefficients they describe (**DONE 2026-09-06**, GP17)
Each tile carries a 256-entry table of entropy-stream lengths — the price of 256 independent
streams per tile. As byte-aligned varints that is ~256 bytes per tile, ~30 KB per 1080p frame,
regardless of how much the tile actually holds. Never measured before.

Share of frame size, on blue_sky q=50 with the GP17 coding in place: tile headers are 4.4% of an
I-frame and ~6% of a P-frame, of which the length table is 12.6 KB of a 301 KB P-frame. Pre-GP17
the same table was 22.5 KB — about 7% of the frame. *(A first version of this entry quoted 5% / 17-31%
from runs taken with `GNC_DIAGNOSTICS=1`, which BUG-7 shows were corrupted encodes. The measured
size reductions below are unaffected: they were taken on actual file sizes with diagnostics off.)*

Priced three encodings before implementing: varint (existing), Exp-Golomb order 0 with a zero
bitmap, and Golomb-Rice with a per-tile `k`. **Rice wins at every point (−20% to −61%)**;
Exp-Golomb *loses* to varints on high-quality I-frames (+4% at q=50, +24% at q=75) where lengths
cluster near the 4096-byte stream cap. Per-tile best-of-three with a mode flag never beat plain
Rice, so there is no mode signalling — just a 4-bit `k` and 256 Rice codes.

Quality is bit-identical (headers only), so the whole size reduction is gain:

| | q=25/30 | q=40/50 | q=55/75 |
|---|---|---|---|
| stills (4 images) | −2.8% to **−7.6%** | −1.8% to −4.5% | −0.5% to −2.7% |
| video (bbb17, blue_sky, 8f) | **−6.9% / −7.6%** | −3.7% | −1.4% to −1.5% |

Largest at low bitrate — the contribution operating point — because the table is a fixed cost that
does not shrink with the coefficients.

**Bitstream:** GP17. Tile flag 0x08 marks a Rice-coded length table. A GP16 decoder has no such
flag, hence the bump. Generation tracking was also collapsed from eight `is_gpXX` booleans ORed
into a dozen chains (`is_gp15` and `is_gp16` appeared twice in one assert) into a single
`gen: u32` with `gen >= N` tests. Verified: GP16 files decode in the GP17 binary; GP17 files are
refused by the GP16 binary with "invalid magic" rather than misparsed.

**Canary:** `GNC_DIAGNOSTICS=1` prints
`Stream-length tables: 15.0 KB (varint would be 30.0 KB, -50%)  rice_tiles=120/120`.

**Near miss worth remembering:** the first decoder capped the Rice quotient at 64 as a
corrupt-input guard, which silently truncated any length above `64 << k` — over 2048 bytes at a
typical k=5, which occurs on high-quality I-frames. Termination never needed the cap
(`get_bit` returns 0 past the end of the buffer); the cap is a sanity bound and is now 65536.

**Does not touch the 5-7x video gap.** That gap is not in the headers.

### BUG-6 — Wavelet decomposition capped at 4 levels; 5 panics (**DONE 2026-09-06**)
The cap was `MAX_GROUPS = 8` with `num_groups = levels * 2`, which put 4 levels exactly at the
ceiling, plus a one-byte per-tile skip bitmap. Raised to 12 groups (6 levels) across both entropy
backends: `rice.rs`, `rice_gpu.rs`, `rans_gpu.rs`, `rans_gpu_encode.rs` and the five WGSL shaders.
The tile-info and k strides are now derived from `MAX_GROUPS` rather than written out as literals,
which is what the old `33`/`25`/`36` constants were.

**Bitstream:** the Rice skip bitmap is one byte for ≤8 groups and two little-endian bytes above
that. `num_groups` is already in the tile header, so no generation flag was needed and existing
files keep parsing — see [docs/BITSTREAM_SPEC.md](docs/BITSTREAM_SPEC.md) §2.4. The per-odd-stream
checkerboard-k block follows the same rule (stride 8, or 12 for wide tiles).

**Measured** at q=70, 5 levels against 4, on the shipped binary after the fix:

| image | bpp 4L → 5L | Δ rate | Δ PSNR | Δ VMAF |
|---|---|---|---|---|
| blue_sky_1080p | 3.47 → 3.33 | **−4.0%** | +1.40 dB | 0.00 |
| kristensara_720p | 2.28 → 2.24 | −1.8% | +0.86 dB | −0.08 |
| bbb_1080p | 4.17 → 4.13 | −1.0% | +0.01 dB | 0.00 |

*(The kristensara VMAF was first logged as +3.32, from an L4 reading of 93.49. Re-measured, L4 at
q=70 scores 96.89 and L5 96.81 — the +3.32 was a bad baseline reading, not a real jump. It was
flagged as suspicious at the time and it should have been.)*

Smooth content gains most, which is what a deeper decomposition should do.
`CodecConfig::max_wavelet_levels()` states the tile-size ceiling (5 for a 256 px tile);
`GNC_WAVELET_LEVELS` still overrides.

**Range settled by BD-rate, 2026-09-06 — 5 levels at q ≥ 25, 4 below.** Per-point VMAF at equal q
looks slightly *worse* with 5 levels, because 5 levels also removes 1–16% of the bits; the gain
only appears at equal quality. BD-rate on VMAF over q=25–70, four images:

| image | BD-rate (VMAF) |
|---|---|
| blue_sky_1080p | **−4.82%** |
| touchdown_1080p | −2.35% |
| kristensara_720p | −1.35% |
| bbb_1080p | −0.83% |
| mean | **−2.34%** |

*(These numbers were re-measured on the committed tree. The first set logged here — mean −3.73% —
came from an intermediate working-tree state while two sessions were editing the same checkout,
and overstated the gain. The sign and the cutoff are unchanged; the magnitudes are smaller.)*

**Lower cutoff (kept).** Over q=15–35 the sign flips: +5.92% bbb, +3.42% touchdown, +2.84%
kristensara (blue_sky still −5.73%), mean +1.61%. Below q≈25 the deep subbands quantise to
all-zero anyway, so their k values and rANS frequency tables are pure overhead — at q=15–20 five
levels costs *more* bits **and** about 1 VMAF point. Hence 25.

**Upper cutoff (removed).** The old q ≤ 80 cap was measuring the aliasing bug, not the transform.
Swept q=85/90/95/99 on all four images: all 16 points save 0.3–0.6% of the bits at PSNR and VMAF
identical to two decimals, and q=100 stays bit-exact lossless while shrinking 0.2%. No loss found
anywhere above q=25, so the cap is gone.

**Video** (old_town, 16 frames, q=30): I-only −4.6% and I+P −3.2% bitrate for −0.20 VMAF; aerial
q=30 I+P −6.0%. Same shape as stills, well inside the −0.5 VMAF block threshold.

**Root cause was wider than the skip bitmap.** Four separate places capped the codec at 8 subband
groups, and each had to be found by a different failure: the Rice k arrays (panic), the Rice
phase-1 accumulators (silent aliasing), the rANS group arrays (validation error), and
`quantize_histogram_fused.wgsl` — the fourth histogram producer, which still wrote at the 8-group
stride so every tile but tile 0 read back `num_groups=0` and the encoder overran its stream buffer.
All four now derive from one `MAX_GROUPS`/`RICE_MAX_GROUPS` constant per backend, and the rANS
decode tile-info offsets derive from it too instead of being hardcoded 33/34/66.

**Canary:** `GNC_DIAGNOSTICS=1` prints `groups=N deep_skipped=M` per frame. At 5 levels it reads
`groups=10` with `deep_skipped>0` at low rate; at 4 levels `groups=8 deep_skipped=0` always.
`deep_skipped` counts skips in groups ≥8, which only the two-byte bitmap can carry.

### TUNE-4 — Adaptive quantisation gradient was inverted (**DONE 2026-09-05**)
`aq_strength` was 0.2 above q=70 and 0.15 below, never swept. Measured: 0.3 below q=30 buys +0.1
to +0.55 VMAF for under 1% more rate on all three images; 0.45 and 0.6 fall back, so 0.3 is the
peak. Neutral-to-negative from q=40 up, and irrelevant above q=55. AQ helped precisely where it
was set weakest. New rule: 0.3 below q=30, unchanged above. ~1-5% BD-rate at low quality, and it
stacks with TUNE-3.

### TUNE-3 — Entropy coder now follows quality (**DONE 2026-09-05**)
rANS at q ≤ 20, Rice above. Measured at identical PSNR: rANS is 5-19% smaller below q=20 and
neutral-to-worse above, crossover content-dependent (kristensara turns at q=20, bbb and touchdown
not until above q=40). Costs ~8% encode and ~15% decode throughput. 4:4:4 only — the rANS GPU
path batches all three planes on the luma tile layout.

Also fixed: subsampled chroma combined with rANS/Huffman/Bitplane panicked in the encoder. A
legal configuration should degrade, not abort; `CodecConfig::normalize_for_chroma()` falls back to
Rice.

**Follow-up:** the `--rice` CLI help says Rice is "~30% worse compression". Measured, Rice is
*better* above q≈25 and by 8-12% at q=70. Text needs correcting.

### TUNE-2 — Wavelet levels default (**DONE 2026-09-05**)
The quality preset used 3 levels below q=50. Measured 5-17% worse bitrate at equal or better
quality on bbb, touchdown and kristensara at q=25/40/49, at no speed cost. Default is now 4
everywhere.

### MEAS-7 — A chroma-aware quality metric (**DONE 2026-09-06**)
`scripts/chroma_metric.py`: CIEDE2000 on decoded RGB, validated against all 16 critical Sharma
reference pairs to 1e-3 (`--selftest`). Report it next to VMAF — VMAF for luma structure, mean and
p95 dE00 for colour accuracy. dE00 ≈ 1 is the nominal just-noticeable difference.

Unblocks the chroma parameters that could not be tuned before: `chroma_weight`, the CfL
enablement range, chroma-format trade-offs.

### Original statement of the problem
VMAF scores luma only, so every chroma decision in this repo validated on VMAF is unvalidated:
`chroma_weight`, the CfL enablement range (q=50–85), chroma-format trade-offs. A 2026-09-05
`chroma_weight` sweep looked like a free 15% rate saving on VMAF and collapsed to +0.3 dB, with
the direction reversing at the low end, once measured with RGB PSNR.

Needed before any chroma parameter can be tuned. Candidates: VMAF with chroma-aware features,
CIEDE2000 / ΔE on the decoded RGB, or a weighted YUV-PSNR with defensible weights. Whatever is
chosen has to be justified, not just picked.

`GNC_CHROMA_WEIGHT` is in place so the sweep can be repeated the moment a metric exists.

### Closed by measurement 2026-09-06 — do not re-test
- **Non-adaptive reference filtering** (a general in-loop filter; GNC has only tile-seam
  deblocking, now off by default): a mild 3x3 low-pass on the
  reference buys **0.9-1.8% on prediction SATD** — consistent in sign on blue_sky, bbb17 and
  old_town, reversing if over-filtered. An edge-selective deblocking proxy is −0.16% to +0.41%,
  i.e. nothing. Too small for a normative bitstream filter the decoder must reproduce bit-exactly.
  **Bounds a non-adaptive filter only** — an edge-adaptive filter with per-block strength is not
  bounded by this, and the trade improves if reference quality ever matters more (longer GOP
  default, hierarchical references). `scripts/meas_ref_filter.py`.
  Do *not* try to bound this by predicting from the previous frame's clean source: that comparison
  is confounded (a decoded reference is the source *low-pass filtered by the quantiser*, not the
  source plus noise) and read −31.6% on bbb17 against +12.3% on old_town.
- **Quantiser cascade down the GOP** (P-frame step growing with distance from the keyframe): at
  exactly matched rate on old_town ki=17 a flat 1.25 step beats a +0.03/frame cascade by +0.95
  VMAF mean and **+5.04 VMAF min**; the mean-vs-min spread goes 2.29 → 6.40 points. Each P in a
  cascade predicts from a reference coded more coarsely than its own predecessor, so error
  compounds geometrically instead of settling. TUNE-5's argument for separating I from P does not
  extend to separating P from P. Lever removed.
- **Hierarchically coded MV zero mask** (2x2 group flag instead of one bit per block): −0% to +3%,
  sign flips with content. Helps only when the field is nearly all zero, which is when the MV field
  is a negligible share of an already tiny frame.

### Closed by measurement 2026-09-05 — do not re-test
- **Intra dead zone**: 0.75 is at its optimum; 0.4 and 1.5 both lose to changing q instead.
  Consistent with the RDOQ result.
- **Wavelet filter choice**: CDF 9/7 lossy, LeGall 5/3 at q=100 — already JPEG 2000's practice.
- **Huffman entropy backend**: 10-20% worse than Rice at every quality.
- **Bitplane entropy backend**: **2.2-2.6x worse** than Rice and the slowest of the four
  (57.7 ms against 33.0 at q=70). Too far off to be a tuning matter — it looks unfinished. Not
  worth carrying as a candidate.
- **Per-code-block Rice parameter adaptation** (JPEG 2000's code-block granularity): ≤2.8% at high
  rate, negative at low rate (`scripts/meas_codeblock_k.py`). Per-subband `k` is already right.
- **Subband quantiser weighting from synthesis norms** (what JPEG 2000 does): the existing
  `GNC_PHYSICAL_WEIGHTS` gradient pushes in that direction and loses to uniform by 8-14% at
  matched quality on all three images. GNC's CDF 9/7 applies the K normalisation, so its
  coefficients are already effectively normalised and a uniform step is correct.
- **Coefficient-level RDOQ**: +0.1% at best (`scripts/meas_rdoq.py`). GNC's uniform quantiser
  plus dead zone is already on its RD curve; this is why every dead-zone and QP sweep moved along
  the curve rather than off it.
- **Per-tile RD bit allocation** (PCRD's idea without truncatable codes): 0% within noise at
  every rate (`scripts/meas_pcrd.py`). A uniform step already equalises the RD slope across tiles.
  JPEG 2000's gain comes from truncating *embedded* per-code-block streams, which Rice cannot do.
  **This retires the repo's standing hypothesis that PCRD accounts for ~89% of the intra gap.**
- **Block intra prediction**: already measured at −11.76 dB / +29% bitrate (hence
  `intra_prediction: false`), and worth only ~+6% to H.264 over JPEG 2000 regardless.

### MEAS-2 — original statement (superseded by the in-progress entry above)
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

### MEAS-4 — Inter-model gap decomposition (**SUPERSEDED — original run, on dumps corrupted by BUG-7. See the re-run above.**)
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

### MCTF — motion-compensated temporal filtering (**REJECTED 2026-09-06** — gated, do not re-test)
`src/temporal.rs` is 129 lines with no motion compensation, so warping along motion vectors and
*then* filtering temporally — MC-EZBC / 3D-SPIHT's combination — was genuinely untested. Two
offline gates, identical motion vectors in both arms:

| gate | touchdown | old_town | speed_bag | bbb |
|---|---|---|---|---|
| open loop (closed/open residual) | 0.99x | 0.99x | 0.98x | 1.34x |
| multi-frame transform (MCTF/P-chain) | 1.04x | 1.05x | 1.13x | 1.14x |

The open loop wins nothing on camera content — real motion dominates reference noise 4-5x — and
the temporal transform is *worse* than a P-chain everywhere, because the level-2 highpass
differences two lowpass frames two apart, which align worse than originals. Stable under a finer
motion estimator (8x8/±16: 0.99→1.01x). Full measurement in RESEARCH_LOG 2026-09-06.

Reaches the same verdict as ICME 2006 and MPEG's deletion of the SVC temporal update step, from an
independent direction.

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
