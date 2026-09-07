# GNC Backlog

Status: `todo` | `active` | `done` | `blocked`

**This file is the queue, not the lock.** `scripts/claim` decides who works on what
([COORDINATION.md](COORDINATION.md), "Start of session"); take your item with
`scripts/claim next "<why>"`, which reads the headings below and claims one atomically. An
`(in progress)` marker here is documentation written afterwards and goes stale — never treat it
as evidence that an item is free or taken. `scripts/claim items` is the live answer.

**So a heading is a queue entry only if it looks like `### NAME-<n> — title (todo, P<n>)`.** Give
a new idea an ID of that shape and a priority, or `next` cannot offer it. A finished item is
marked `**DONE**` / `**CLOSED**` / `**FIXED**` / `**REJECTED**` and loses its priority marker, so
it drops out of the queue by construction. To park an item that nobody can start — `CANARY-1` and
`MEAS-5` both need a second GPU — hold it with
`scripts/claim take --as blocked-<reason> <ITEM> "<why>"` rather than deleting the priority.

Only **open** items live here. All completed, closed, and vetoed items (66 of them,
with gate experiments and measurements) are archived verbatim in
[docs/archive/BACKLOG_CLOSED.md](docs/archive/BACKLOG_CLOSED.md).
Current state and priorities are described in [GOALS.md](GOALS.md).

## Baseline (v0.1-spatial, commit 617d8e6)

See [BASELINE.md](BASELINE.md) for current benchmark numbers.

## Current Focus (updated 2026-09-06)

**Positioning: GNC is a contribution codec** ([docs/POSITIONING.md](docs/POSITIONING.md), GOALS §1).
It does not try to beat H.264 on bitrate; it aims for H.264-class quality that runs on any GPU and
scales with the card, against fixed-function encoders with session limits.

### What 2026-09-06 established, and what it changes

**1. There is no inter gap at the contribution operating point — for anyone.** Measured x264's own
inter saving: **+63.9% at crf 23, +0.6% at crf 12, −33.3% at crf 2**. Its crossover is around
crf 12; GNC's is between q=75 and q=92. At contribution quality both codecs break even, and GNC is
if anything better behaved (−2% against x264's −33% on raw rate). Near lossless the MC residual is
noise-like (Girod), so it costs about what the picture costs and the motion vectors are overhead.
**The +118–177% GNC is behind at contribution quality is an *intra* gap.** Stop spending inter
effort there; the inter gap is real only at distribution bitrates.

**2. A serial dependency costs 4.9x and is still 201 fps.** Measured directly
(`tests/wavefront_cost.rs`): 1.02 ms independent against 4.98 ms for a true per-pixel wavefront on
1080p 4:4:4. That is 25% of the 20 ms budget at 50 fps. **The parallelism objection that has been
used to decline tools is quantitatively much weaker than assumed** — a dependency is a tax on one
pass, not a disqualification, and no such pass is near the bottleneck (entropy coding, 51–85% of
runtime).

**3. Lossless works and is the best wavelet in the field.** `q=100` is bit-exact on all three
coders (GOALS was wrong). 1.99:1 beats JPEG 2000 lossless by 10.8% and PNG by 7.8%; loses to FFV1
by 27% and x264 `-qp 0` by 43%, both of which predict against the neighbour rather than the scale.

**4. Five conclusions this week turned on the operating point or configuration measured**, not on
the idea: TUNE-5, "inter saves 17–27%", BUG-10, BUG-13, and the intra-prediction rejection. The
standing rule is now: *name the mechanism you are testing, then check that the implementation
actually implements that mechanism* — and measure the range the project cares about, not the
convenient one.

### 5. The headline gap figure was wrong by 3x, and the target is now reachable (QUAL-1)

At the contribution operating point GNC needs **+90.5% BD-rate on PSNR** against x264 — about
1.9x — not the 5.6x recorded from MEAS-1, which was measured at distribution bitrates with the
quality ladder above q=92 dead. Nothing in the coder changed between the two measurements. This
matters for prioritisation more than for pride: **against 5–7x, single-digit improvements were
provably pointless; against 1.9x they accumulate into the target.** Every "this is too small to
bother with" judgement in this repo predating 2026-09-06 was made against the wrong denominator.

Also settled: **never quote a VMAF BD-rate above about q=85** (widening the ladder moved it 47.5
points on average, 110 on old_town, while PSNR moved 1.0). ~~And **GNC leads x264 on colour at
matched rate** (dE00 0.611 vs 0.684) while trailing 7.4–8.8 dB on luma~~ — **that colour lead is
withdrawn (CHROMA-2, 2026-09-07, decision 0020): the rate-matched control has x264 ahead on 6 runs
of 6, on five of them without needing a chroma-QP offset and while also leading luma. GNC has no
measured advantage over x264 on any axis at this operating point.**

### Priority order

1. **Intra at contribution quality** — the whole remaining +90.5% lives here, per findings 1 and 5.
   Inter breaks even at this operating point for x264 too, so this is the only place the gap is.
   **First instalment paid 2026-09-07 (ABAC-SHIP): −17.3% of intra rate at q=90, opt-in.** Against
   the corrected +90.5% denominator that is roughly a fifth of the gap, from one mechanism. The
   next largest known intra lever is still unbuilt — see EBCOT Part 7's open items.
2. **LOSSLESS-1** — both gates green (10–26% for ~5 ms/frame). The one lever this week that passed
   rather than failed. Buildable now.
3. **MEAS-5 / CANARY-1** — blocked on a discrete GPU. The entire strategic thesis rests on MEAS-5
   and it has never been measured.
4. ~~**QUAL-1**~~ — done 2026-09-06: the contribution gap is **+90.5% BD-rate on PSNR**, about
   1.9x, not the 5.6x recorded at distribution bitrates. ~~Follow-up: `GNC_CHROMA_WEIGHT`~~ — done
   too (CHROMA-1): the frontier is steep but intra-only, so it does **not** explain the video gap.
   That gap is genuine luma coding deficit, which is why item 1 is intra.
5. Bugs: BUG-14. BUG-9 closed 2026-09-07 (the cause was the cumfreq table, not the slot), BUG-12 closed 2026-09-06.

**Do not re-test** (measured and closed this week): MCTF, GOP length, the B-pyramid at contribution
quality, RD decisions, multi-reference, sub-pel filters, motion search, block transforms, sub-block
masking, smaller tiles, prediction *before* the wavelet.

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

### BUG-11 — Rice's stream mapping hardcoded tile width 256 (**DONE 2026-09-06**)
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

**Fixed 2026-09-06.** Streams now walk the tile in **column-major** order, cut into 256 contiguous
segments: `j = stream_id * symbols_per_stream + s`, raster index
`(j % tile_size) * tile_size + j / tile_size`. One expression, uniform in the width, and at 256 px
a segment is exactly one column so it reduces to the old mapping — **tile 256 output is
byte-identical at all 12 measured points**, which is both the correctness proof and the reason no
shipped preset moves.

At tile 512, same q, PSNR delta exactly 0.00 dB (the change touches entropy coding only):

| image | q=75 | q=90 | q=99 |
|---|---|---|---|
| bbb | −14.4% | −5.5% | −0.6% |
| blue_sky | −15.3% | −5.8% | −1.3% |
| touchdown | −12.9% | −5.6% | −0.5% |
| kristensara | −18.6% | −7.8% | −0.8% |

Encode/decode agree bit-exactly at tile 128, 256 and 512 (q=100 lossless roundtrip, max error 0).

**The tile-size conclusion reverses, and the prize is ~1%.** 512 now beats 256 on all 12 fixed-q
points, but BD-rate (q=60–95, PSNR-driven) is **−0.91% mean** — bbb −0.23%, blue_sky −0.84%,
touchdown −1.99%, kristensara −0.61%. The fixed-q points flatter 512 because its q=75 arm also
sits 0.03–0.23 dB lower in PSNR. So the *sign* was the coder, as hypothesised, but the geometry
underneath is worth about 1%, not the ~5% inferred from the rANS arm. **Not changing the default
tile size for 0.91%** — a 512 px tile quadruples the threadgroup working set and adds latency.
Full numbers in RESEARCH_LOG, 2026-09-06.

### ENT-1 — Per-tile frequency tables (**DONE 2026-09-06**)
Subband-rANS stored each normalised frequency as a flat `u16`, and the tables were **23–26% of
every file across the whole quality range** — not just at low bitrate, because the alphabet grows
with quality in step with the payload (202 symbols at q=20, 1791 at q=70). At q=70 they were 68%
zeros with an order-0 entropy of 27 KB against 205 KB stored.

Now coded as alternating zero-run and value, both Exp-Golomb order 0, byte-aligned per group.
Signalled by bit 31 of the tile's `num_groups` word, so the tile versions itself, no frame
generation was spent, and older files parse unchanged. Host-side only — the GPU shaders see the
same expanded buffers.

**Tables 65–82% smaller; the file 11.7–26.6% smaller, mean −16.6%, at bit-identical quality.**
Four images, q=50 and q=70. On the preset path (rANS at q ≤ 20) it is −4.5%.

**Follow-up: DONE 2026-09-06 — the crossover stays where it is.** Re-swept on four images at
q=25/40/55/70, exactly rather than by BD-rate (both coders quantise identically and entropy coding
is lossless, so equal q is matched quality — verified end to end, bbb q=70 reads PSNR 44.25 under
both). rANS went from 15–25% behind Rice to level or ahead on three of four images: mean −1.9% at
q=70, but that is touchdown at −7.8% pulling three flat ones, kristensara regresses 1.2–6.3%
everywhere, rANS costs ~15% decode (TUNE-3), and it cannot encode above q=75 at all (BUG-9 — a named refusal since 2026-09-07, a crash before that).
A content bet that is slower and cannot run at the operating point is not a new default.

**Re-measured by ENT-2 (2026-09-07) on whole frames rather than padding-neutral crops: same
verdict, different magnitudes.** The mean at q=25/40/55/70 reads +0.4 / +0.1 / +0.1 / −0.6% rather
than −1.5 / −1.4 / −1.5 / −1.9%, and the overflow starts at **q=77, not q=80**. Two harnesses
agreeing on the conclusion while disagreeing on the numbers is the reason to quote the setup and
not just the delta.

**Original follow-up text:** rANS is pinned to q ≤ 20 because Rice
measured better above it. With the tables packed, rANS at q=70 is level with Rice on bbb, ahead on
blue_sky (3.04 vs 3.07) and touchdown (3.18 vs 3.45), and 1% behind on kristensara — where it used
to be 15–25% behind everywhere. The crossover was measuring table overhead, not the coder. Verify
PSNR equality end-to-end first: it is identical by construction, and "by construction" has been
wrong three times in this repo today.

**Caveat:** the ABAC track needs no tables at all and measures −19 to −25% against Rice. If it
clears its GPU throughput gate this work is superseded, and it was built knowing that.

**The other half of that 14.2%, measured directly (BUG-11, 2026-09-06).** The 256→512 rANS gain
was split into ~9 points of table overhead and ~5% of "transform continuity plus length
amortisation". With Rice's stream mapping fixed, the geometry alone measures **−0.91% BD-rate**
(bbb −0.23, blue_sky −0.84, touchdown −1.99, kristensara −0.61). So continuity is worth ~1%, not
~5%, and the tables really were the whole lever — which is what this entry then went and fixed.
Rice carries no frequency tables, so those figures are independent of the packing above.

### BUG-12 — `--tile-size` could not reach its own wavelet-level ceiling (**DONE 2026-09-06**)
`quality_preset` ends with `cfg.wavelet_levels = min(levels, cfg.max_wavelet_levels())`, computed
against the **default** tile size. `main.rs` then sets `config.tile_size = tile_size` from the CLI,
after the clamp has already run. So `--tile-size 512` is capped at 5 levels though a 512 px tile
allows 6, and `GNC_WAVELET_LEVELS=6` is silently ignored with it.

Harmless for the shipped presets (256 px everywhere) but it means every past `--tile-size`
experiment ran with a hidden cap.

**Fixed 2026-09-06.** `CodecConfig::set_tile_size()` re-derives the ceiling from a recorded
`requested_wavelet_levels`, in both directions, and all five call sites in `main.rs` use it instead
of assigning `tile_size`. `tests/tile_size_levels.rs` pins it, including that the shipped presets
do not move.

And the level it unlocked is worth nothing: 6 levels against 5 at tile 512 measures **−0.0% to
−0.5%** of the bits (−0.1% at q=90, where PSNR is identical to 0.00 dB). Consistent with the
earlier −0.3% note. The q=75 arm is not interpretable — a 6th level shrinks the LL band that
adaptive quantisation measures variance on, and AQ is active at q=30–80, so PSNR swings by image
(−1.13 dB bbb, +0.56 dB kristensara). **Closed: deeper decomposition is not a lever.**

### BUG-13 — intra prediction produced corrupt output (**FIXED 2026-09-06**)
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

### BUG-15 — the wavelet lossless arm was not lossless (**FIXED 2026-09-07**)
`GNC_MED=0` at q=100, and any `--qstep 1 --wavelet 53` config below it, returned **53–56 dB with
dE00 0.5–0.9** instead of bit-exact output. `is_lossless()` checks the quantiser step, dead zone
and wavelet type but not the **subband weights**, and `pack_weights_chroma()` scales the step by
`chroma_weight` — which CHROMA-1 raised to 1.2 for every q >= 60, q=100 included.
`GNC_PHYSICAL_WEIGHTS=1` is the same hole on luma: 6 255 bytes at 49.2 dB on gradient512.

**Invisible because LOSSLESS-1 landed first.** q=100 had already been routed to MED, so the only
config CHROMA-1 broke was one nothing exercised; `conformance_lossless_q100` stayed green because
the MED branch flattens the weights for its own reasons. Neither change was wrong on its own — the
invariant they share (*a lossless config must not scale any quantiser weight*) lived in neither.

Fixed with `CodecConfig::normalized_for_lossless()`, called from `quality_preset` and again at the
encoder entry (CLI overrides land after the preset). With it, `GNC_MED=0` at q=100 is
**byte-identical to the pre-LOSSLESS-1 build `e872904`** — 671 507 / 53 811 / 937 420, all `inf`.
MED output is unchanged, so **LOSSLESS-1's −14.9%, the +25.8% FFV1 gap and the −14.3% abac
follow-up all stand**. The lossy side is untouched and asserted so (q=90 keeps 1.2).

Regression test `conformance_lossless_wavelet_arm_is_bit_exact`, verified to fail without the fix.
It asserts **pixel equality, not a PSNR threshold** — a `psnr > 45.0` assertion, which most tests
in that file use, reads 55 dB as a pass.

Seen in passing, both at q=100 with MED active, both silent: **`--huffman` emits 6.73 dB garbage**
(BUG-14's session has this) and **`--rans` falls back to the wavelet path**, byte-identical to
`GNC_MED=0`. Only the default coder (Rice) delivers the MED path.

### INTRA-NEARLOSSLESS — MED instead of the wavelet does not survive into the lossy range (**CLOSED BY MEASUREMENT 2026-09-07**)
Gate run and **failed on its criteria set beforehand** (pass: >=10% luma BD-rate, consistent in
sign). Closed-loop JPEG-LS near-lossless model, calibrated per image against the real q=100 file
(1.078-1.123x), `scripts/nearlossless_gate.py`:

| image | BD-rate luma | BD-rate dE00 |
|---|---|---|
| bbb | **+14.12%** | +106.25% |
| blue_sky | −27.19% | +59.66% |
| kristensara | −26.42% | +31.60% |
| touchdown | −22.46% | +54.37% |
| mean | −15.49% | +62.97% |

The mean reads as a win and is not one: bbb reverses (as in LOSSLESS-1, where animation was also
the weakest image at −5.8%) and colour is worse everywhere by more than the luma gain buys back.

**Mechanism, and it is a property of DPCM rather than of this implementation:** quantisation error
feeds back through the predictor, so the residual grows with the step and rate falls far more
slowly than quality does. kristensara, delta 1→2→3→4: 538 678 → 330 987 → 253 428 → 234 743 B for
exact → 51.1 → 49.9 → 46.4 dB. **The usable ladder is delta=1 or delta=2 and nothing between** —
bit-exact or ~51 dB, with no way to ask for 55 dB. A contribution codec needs that range.

**Do not re-test** the quantised-MED-instead-of-wavelet path. Scoped deliberately narrowly, after
the abac near-miss: what is closed is *replacing the wavelet with a quantised closed-loop MED
predictor in the lossy range*. Prediction as a **residual-domain** tool, context modelling on
wavelet coefficients (abac, now shipped), and MED at q=100 (LOSSLESS-1, shipped) are all
untouched by this.

The one point win, matched on colour rather than luma: kristensara delta=2 at dE00 0.575 against
GNC q=90 at 0.570 — 330 987 B / 51.15 dB against 407 248 B / 50.36 dB, so 18.7% fewer bits and
+0.79 dB. One rung is not a coding path, but it is why the gate was worth running.

Two modelling artefacts, both caught by checking monotonicity, both mine: a **fractional quantiser
step does not divide the integer pixel lattice** (rate *rose* as the step coarsened — δ=3 cost 20%
more than δ=2), and the **calibration point must actually be lossless** (the chroma multiplier made
the δ=1 rung lossy, folding a quantisation loss into the coder-overhead ratio). Four runs void
before that was found. A coarser quantiser producing more bits is a broken instrument, not a
finding.

### Superseded — INTRA-NEARLOSSLESS as filed (the q=99 observation that opened it)
Priority 1 is intra at contribution quality, and the first measurement of the ladder's top found a
hole in it. Four crops, default coder, YCoCg-R luma:

| image | q=99 | q=100 (MED, bit-exact) | |
|---|---|---|---|
| bbb | 2 454 001 B @ 59.92 dB | 2 415 436 B | **−1.6%** |
| blue_sky | 2 107 664 B @ 60.28 dB | 1 598 293 B | **−24.2%** |
| kristensara | 717 257 B @ 59.83 dB | 538 678 B | **−24.9%** |
| touchdown | 2 362 696 B @ 59.82 dB | 1 969 244 B | **−16.7%** |

**q=99 costs 1.6–24.9% more bits than bit-exact lossless and is 60 dB rather than perfect.** The
anchor ladder puts qstep at 0.75 there (1.30 at q=96), so above about q=96 GNC quantises at a step
below one — spending bits on precision an 8-bit output cannot show (which is RATE-1's mechanism)
while a mode that spends nothing on it is cheaper. Nobody re-measured the top of the ladder after
LOSSLESS-1 moved q=100 by −14.9%, which is what opened the hole.

Open question this was filed for: does MED prediction *instead of* the wavelet keep paying at
q=88–99 with a quantised closed-loop residual (JPEG-LS near-lossless)? Gate not yet run.

### LOSSLESS-1 — prediction-based lossless path (**BUILT AND MEASURED 2026-09-06**)
GNC is 27% behind FFV1 and 43% behind x264 `-qp 0` at q=100, and both win the same way: a
per-pixel median predictor whose error is entropy-coded **directly**, with no transform. GNC's own
block intra prediction was fixed and measured (BUG-13) and costs **4-8%** at lossless — but it
predicts *and then* transforms, which hands the wavelet a harder signal. That refutes prediction
before a wavelet; it says nothing about prediction instead of one.

Five differences from what FFV1 does: block modes vs per-pixel median; one mode per 8x8 block vs
every pixel; before the transform vs replacing it; SAD-selected vs n/a; resets every 32 px vs whole
frame; luma only vs all planes.

**Gate run and passed** (`scripts/lossless_gate.py`). MED/LOCO-I in GNC's own reversible YCoCg-R,
zeroth-order entropy of the residual, calibrated against FFV1's real output:

| image | vs GNC q=100 | vs FFV1 real |
|---|---|---|
| touchdown | **-19.6%** | +5.4% |
| bbb | **-9.5%** | +17.1% |
| blue_sky | **-26.2%** | +15.2% |
| kristensara | **-20.0%** | +6.8% |

The model is 5-17% *larger* than FFV1's real file, so it is conservative — FFV1 adds context
modelling and adaptive Golomb that the model lacks. **10-26% available, plausibly more.**

**Second gate, not yet run: throughput.** MED cannot produce pixel (x,y) before (x-1,y) — a serial
per-pixel dependency, which is what this architecture exists to avoid. Escapes: tiles (already
have them) and a wavefront within each tile (511 steps of up to 256 lanes for a 256px tile, versus
65 536 serial). But a wavefront is a different parallelism profile from one-thread-per-coefficient,
and "massively GPU-parallel" is the core claim.

**Throughput gate run and passed** (`tests/wavefront_cost.rs`). Identical arithmetic, the only
difference being whether neighbours come from a separate buffer or from the output the workgroup is
still writing. 1080p 4:4:4 padded, M1:

| | per frame | fps |
|---|---|---|
| independent (1 thread/px) | 1.02 ms | 980 |
| wavefront (511 diagonals, storageBarrier) | **4.98 ms** | **201** |

**4.9x, and still 201 fps.** 25% of the 20 ms budget at 50 fps, ~15% of GNC's current decode. The
barrier type barely matters (4.79 vs 4.98 ms with the weaker barrier), so the cost is **occupancy,
not synchronisation** — the short diagonals idle most of the workgroup.

**Both gates are green. This is buildable.** Scope: a second coding path (MED predict + entropy
code, no wavelet) selected at q=100, with the wavefront only on the decode side — encode-side MED
is fully parallel at lossless because the encoder's reconstruction equals its input.

**Built and shipped.** `TransformType::MedPredict`, bitstream `transform_type = 2`, selected at
q=100; `GNC_MED=0` restores the wavelet. Four images, every one verified bit-exact through
encode → file → decode:

| image | wavelet | MED | delta |
|---|---|---|---|
| bbb | 3 436 337 | 3 235 737 | −5.8% |
| touchdown | 3 195 978 | 2 724 578 | −14.7% |
| kristensara | 1 190 210 | 984 178 | −17.3% |
| blue_sky | 2 911 571 | 2 278 064 | **−21.8%** |
| mean | | | **−14.9%** |

**The FFV1 gap halves: +48.4% → +25.8%**, and GNC now beats PNG by ~19% on three of four images
where it previously lost to PNG on all four.

**Delivered 79% of the gate, uniformly short**, which located the remainder precisely: the gate
modelled zeroth-order entropy while Rice spends a significance bit per coefficient — nearly free
on sparse wavelet coefficients, close to a wasted bit per pixel on a dense MED residual. **The
remaining +26% against FFV1 is therefore an entropy-coding gap, not a prediction one.**

**Follow-up: MEASURED 2026-09-06, −14.3% more.** MED residuals through abac, real coefficients via
`GNC_ABAC_COMPARE=1` (the harness was gated on the wavelet path and could not see this one; gate
widened): bbb −14.9%, blue_sky −15.6%, kristensara −15.5%, touchdown −11.2%. **Together with
LOSSLESS-1 that is −27% against this morning's wavelet lossless, and it puts GNC +7.7% behind
FFV1 — from +48.4%.** Rate only; abac's GPU decode throughput gate is now the only thing in the
way, and it is the gate the abac track is already blocked on. Handed to that track rather than
implemented twice.

**Closed by measurement — do not re-test: a dense Rice mode for MED residuals.** Zigzag every
coefficient, no significance bit, no zero runs. Modelled at −6.6/−7.5/−7.4% on three images and
**+2.4% on blue_sky**, whose smooth sky makes run-length coding pay. Content-dependent ~5% for
changes to both Rice shaders, against −14.3% from a coder that already exists. The model was
validated against the shipped encoder to within 3% before being used to reject the idea.

**Follow-up (open, P3): decode throughput on the real path.** The 201 fps wavefront figure is the
isolated gate; the shipped decoder has never been timed, and the machine was too loaded to do it
(COORDINATION rule 1).

### CHROMA-1 — Is GNC's luma/chroma rate split on the frontier? (**DONE 2026-09-06**)
**Withdrawn 2026-09-07 — see the CHROMA-2 entry above; x264 wins colour 6 of 6 at matched rate.**
Falls out of QUAL-1. At rate matched to 1%, GNC beats x264 on dE00 on all three sequences while
sitting **7.4–8.8 dB behind on luma**. The two codecs allocate rate differently between luma and
chroma, so part of the +90.5% luma gap may be an allocation *choice* rather than a coding
efficiency deficit — and allocation is a one-line config change, not a new algorithm.

**Hypothesis (falsifiable):** GNC's `chroma_weight` (1.5 below q=40, tapering to 1.0 at q≥85)
spends more on chroma than the joint luma/colour optimum. If so, coarsening chroma and spending
the recovered bits on luma should improve luma at matched total rate.

**Success criterion:** ≥0.5 dB Y-PSNR at matched rate while mean dE00 stays below x264's at that
rate — i.e. buy luma without surrendering the colour lead, which is a real differentiator
(docs/POSITIONING.md §4). Below 0.2 dB, reject and record that the default is already on the
frontier. **If the frontier is flat, that is a useful answer too** — it means the +90.5% is
genuine coding deficit and intra work is the only route.

**The trap, documented before starting:** this exact knob was swept once on VMAF, looked like a
free 15% rate saving, and reversed sign once measured with a chroma-aware metric (CLAUDE.md).
VMAF is luma-only *and* saturated above q=85, so it is doubly blind here. Judge on Y-PSNR and
CIEDE2000 together, at matched rate — a fixed-q point measurement always flatters whichever arm
spends more bits.

---

**Answered 2026-09-06. Yes there is a frontier, it is steep, and the default sat off it — but it
is an intra lever, so it does not touch the +90.5% video gap.**

BD-rate against the shipped policy, four images x q=75/85/92/96, luma in YCoCg-R, colour as dE00:

| weight | luma BD-rate | colour BD-rate | exchange rate |
|---|---|---|---|
| **1.2** | **−5.2%** | **+1.2%** | **4.3:1** |
| 1.5 | −12.9% | +4.0% | 3.2:1 |
| 2.0 | −21.8% | +9.7% | 2.2:1 |
| 3.0 | −32.0% | +22.2% | 1.5:1 |

Monotone on all four images. Weight 1.5 is **+1.28 dB at matched rate** against a 0.5 dB criterion,
so the luma side of the hypothesis holds easily — but it pays by losing MEAS-8's "95% of pixels
under the JND" on two of four images at q=96. **1.2 is the largest weight that costs nothing
there**, so that is what shipped: `chroma_weight` no longer drops to 1.0 above q=85.

**The criterion as written was not testable.** It said "mean dE00 below x264's (0.611–0.949)", but
those are 4:2:0 *video* figures and this sweep is 4:4:4 *stills* — not comparable. MEAS-8's
internal criterion was substituted.

**Why it does not help the video gap — the control that settles it.** First guess was the chroma
format; wrong:

| configuration | weight 2.0 vs shipped |
|---|---|
| 4:4:4 stills | ≈ −25% |
| 4:4:4 video, all-intra (ki=1) | **−20.8%** |
| 4:4:4 video, P-chain (ki=9) | **−2.9%** |
| 4:2:0 video, P-chain (ki=9) | −1.5% |

It is **inter**. After motion compensation there is almost no chroma residual left to coarsen, so
the effect lives in I-frames and is diluted sevenfold by the P-chain. At q=92 4:2:0, luma moves
**0.01 dB** across the knob's whole useful range. **So the +90.5% is genuine luma coding deficit,
not an allocation artefact, and intra is the only route** — established by elimination rather than
assumption. The lever stays real for all-intra 4:4:4/4:2:2 contribution modes, which are not a
corner case for this codec.

**Two measurement errors caught, both in my own harness.** BT.709 Y from decoded RGB is
contaminated by chroma error and overstated the luma loss **3.7x** (−0.56 vs −0.15 dB); luma is now
taken in YCoCg-R. And VMAF read **97.08 before and after** the shipped change, on 6% fewer bits —
the same illusion that made the 2026-09-05 sweep look like a free 15%.

### BUG-17 — `abac_bitstream` failed under test parallelism (**FIXED 2026-09-07**)

`cargo test --release`, the gate CLAUDE.md prescribes, was **non-deterministic** on this file:
`abac_handles_subsampled_chroma` failed 2 runs in 4 at default parallelism, passed 4 of 4 with
`--test-threads=1`, and passed alone. Another session saw a *different* test fail
(`abac_survives_a_p_frame_chain`, `max |diff| 104745`) on its own runs.

**The question this entry asked was the right one, and the answer is the uncomfortable option.**
It asked: is the test wrong to share a GPU device with its neighbours, or is there shared state in
the abac path that concurrency merely exposes — because if the second, `--test-threads=1` would
hide a real bug rather than fix one. It is the second, twice over:

1. **A test called `std::env::set_var("GNC_ABAC_CODER", …)`.** The environment is process-global
   and cargo runs tests on parallel threads, so that test changed which arithmetic engine a
   *concurrently running* test encoded with. That explains both the flake's signature (a different
   test failing each run) and its magnitude (garbage, not a rounding difference — a plane decoded
   with the wrong engine does not fail, it produces a plausible wrong image).
2. **And it was masking a real decoder bug.** `CachedBuffers` held **one** `abac_coder` for the
   whole frame, written per plane in a loop, so it ended up holding the *last* plane's engine and
   planes 0 and 1 decoded with whatever plane 2 used. All three planes normally share an engine,
   so the field was right by accident and no test could see it.

**Fixed at the source, not worked around.** The coder and code-block size moved from the
environment into `CodecConfig` (`abac_coder`, `abac_code_block`, still seeded from
`GNC_ABAC_CODER` / `GNC_ABAC_CB`), and `abac_coder` in the decoder is now `[Coder; 3]`. The env
var was never only a test hazard: any embedder running two encodes on different threads had it.

Three consecutive parallel runs of the file afterwards: **8 passed, 0 failed**, and the full suite
is green at default parallelism. **So the gate stays `cargo test --release`, unqualified** — no
`--test-threads=1` requirement and no serialising mutex, because a red gate on this file should
mean something again.

Worth keeping for the shape: the flake was the only visible symptom of a latent bug that no
sequential test run could ever have caught, and the instinct to make it go away by serialising the
suite would have preserved it.


### BUG-19 — decision-record numbers collide, and two pairs are live on `main` (todo, P3)

`docs/decisions/` currently holds **two 0018s and two 0019s**:

```
0018-gnc-is-broad-on-purpose.md
0018-the-entropy-coders-are-level-and-0015s-prediction-was-wrong.md
0019-the-inter-paths-saving-was-an-equal-setting-figure.md
0019-the-pick-is-the-lock.md
```

A third case was already cleaned up by hand — `0020-the-colour-lead-over-x264-is-withdrawn.md`
was renumbered from 0018.

**Same mechanism as the MEAS-9 claim collision, on a resource nobody thought to lock.** Two
sessions run `ls docs/decisions/`, both compute "next is 0018", and neither can see the other:
reading, deciding and writing are three steps, and the number is only taken once the file is
committed. Determinism is what makes it reliable rather than unlikely, exactly as in
`docs/decisions/0019-the-pick-is-the-lock.md` — which is itself one of the duplicates, filed
against a number a peer session took in the same window.

**The mechanism already exists; it was documented one commit too late for these four.**
`scripts/claim take dr-<NNNN> "<title>"` reserves the number before the file is written and is
the same compare-and-swap as every other claim (COORDINATION.md, "The claim commands"). Nothing
enforces it yet — `claim` does not know what a decision record is.

**To close:** renumber the two later duplicates, fix every inbound reference (BACKLOG,
COORDINATION, RESEARCH_LOG and any sibling record that cites them), and decide whether `claim`
should learn to hand out the next free `dr-` number the way `next` hands out backlog items. The
renumbering is the boring half and the references are where it goes wrong — 0020 was renumbered
by hand and is worth checking for stale citations while here.

Filed 2026-09-07 by the `coord` session.

### BUG-20 — the clippy gate does not cover the test targets, and 88 warnings sit there (todo, P4)

CLAUDE.md requires **zero clippy warnings** and names the gate as `cargo clippy --release` plus
the wasm target. Both are clean. But `cargo clippy --release --all-targets` reports
**`gnc` (lib test) generated 88 warnings**, 24 of them auto-fixable — the gate as written does
not look at test code, so the rule and the check disagree about what "the code" means.

Measured 2026-09-07, twice, on a tree whose diff was documentation only.

**Not urgent, and not obviously a bug in the tests either** — some lints are noisier in test code
than they are worth silencing. The decision to make is which of the two is wrong: widen the gate
to `--all-targets` and clear the 88, or write down that the zero-warning rule covers shipped code
and not tests. Doing neither means the next person to run `--all-targets` re-discovers this and
has to decide it under time pressure.

The one number worth keeping either way: `cargo clippy --release` and
`cargo clippy --release --target wasm32-unknown-unknown --lib` are genuinely clean. The single
remaining `warning:` line on the native target is the future-incompatibility notice for the
third-party crate `block v0.1.6`, not a lint on this code.

Filed 2026-09-07 by the `coord` session.

### BUG-18 — the CPU-entropy P-frame path encodes every P-frame wrong (claimed, P1)

Found 2026-09-07 while measuring abac on inter (ABAC-SHIP); **not an abac defect** — the isolating
tests use Rice on both sides. One concrete cause found and fixed, at least one more open.

**`gpu_entropy_encode` does not merely move entropy coding between CPU and GPU.** In `sequence.rs`
it selects between **two independent implementations of the whole P-frame encode** — a batched
single-command-encoder pipeline ("forward + entropy + local decode", ~1460 lines) and a per-plane
one (~360 lines). They do not agree.

**Measured** (`tests/bug18_locate.rs`, `--ignored`; 1I+3P, 256×256, 4:4:4, Rice on both sides,
only `gpu_entropy_encode` varying):

| q | ki | frame 0 (I) | frame 1 (first P) | frame 2 | frame 3 |
|---|---|---|---|---|---|
| 50 | 9 | **0.000** | 28.8 | 55.8 | 62.9 |
| 90 | 9 | **0.000** | 4.24 | 4.65 | 4.59 |
| 50 | 2 | **0.000** | 28.8 | **0.000** | 26.8 |
| 90 | 2 | **0.000** | 4.24 | **0.000** | 5.27 |

Bytes at q=50, ki=9: GPU `[35366, 4504, 3997, 3461]`, CPU `[39974, 9373, 9341, 9578]` — the CPU
path's P-frames cost **2.1–2.8×** and do not shrink down the GOP as the GPU path's do. (The
I-frame's 13% is the known, harmless CPU-vs-GPU Rice coder difference: same pixels, worse coding.)

**Read the ki=2 rows.** With I,P,I,P every P predicts from the I immediately before it, so a
reference that fails to advance cannot matter — and the first P still diverges by the same 28.8.
**So this is not drift or accumulation: every P-frame is wrong on its own.** ki=9 adds compounding
on top (28.8 → 55.8 → 62.9), but the defect is per-frame. That killed the obvious hypothesis
(the non-batched branch never writes a reconstructed P back to `gpu_ref_planes`, which is true and
looked damning) — it must be a smaller effect than it appears, or masked by whatever comes first.

**Cause 1, found and fixed.** The non-batched branch quantised P residuals with
`config.quantization_step` while `res_config` — which is what goes into the frame header, and what
the decoder dequantises with — carried `res_qstep = quantization_step × p_qp_scale` (TUNE-6). So
its P-frames decoded **25% too large** wherever the scale exceeds 1.0. The comment directly above
the definition already stated the invariant ("The decoder dequantises from the stored config, so
both must use this value"); three dispatches ignored it. Fixed, and the comment now says that it
binds *every* quantise call on *either* path. This took q=50 from 74.5 to 62.9 — real, and not the
main term.

**Cause 2, open.** After the fix the first P still diverges by 28.8 at q=50 / 4.2 at q=90 and costs
2.1× the bytes, which points at a worse *prediction* rather than a worse quantisation: motion
estimation, motion compensation, or the buffers feeding them differ between the branches. The
batched branch runs ME on the GPU into `split_mv_buf`; the non-batched computes it separately.
That is where to look next.

**What it invalidates.** Anything encoded with `gpu_entropy_encode = false` on video — which is
**every abac video encode**, since abac has no GPU encode path. ABAC-SHIP's inter figure
(−14.4% at q=90) is retracted for exactly this reason. Intra is unaffected: single-frame encodes go
through `pipeline.rs` and were verified pixel-identical between the coders. Bitplane video is also
on this path.

Possibly the same root cause as **BUG-16** (the two *intra* encode paths disagreeing at q ≤ 30),
and it is worth checking against **BUG-8** ("the encoder's local decode diverges from the real
decoder down a GOP"), which may be this seen from the other side.


### BUG-16 — Rice's GPU and CPU encode paths disagree on the coefficients (todo, P2)

Found 2026-09-07 while shipping abac (ABAC-SHIP); **not an abac defect** and not chased there.
Both arms are the Rice coder, so this is the quantise stage, and it is on the **default** path.

| bbb, 4:4:4 | Rice GPU encode | Rice CPU encode |
|---|---|---|
| q=25 | 35.51 dB, 415 544 B | **35.63 dB**, 610 264 B |
| q=90 | 50.06 dB, 2 091 447 B | 50.06 dB, 2 275 767 B |
| synthetic q=75, 4:2:2 | — | max abs pixel diff **1.69** vs the GPU path |

Read the two rows together. At q=90 the paths agree on the picture to the decimal and differ only
in size — that is the *expected* difference, since the CPU reference lacks per-stream k and the
checkerboard k-context and is simply a worse coder. At q=25 they disagree on the picture, which
means **different coefficients**, and the GPU path emits the smaller *and* worse file: it is
discarding something the CPU path keeps.

**The obvious suspect is already ruled out.** The fused quantize+histogram shader runs only on the
GPU encode path (`use_fused_qh = config.use_fused_quantize_histogram && use_gpu_encode &&
!use_cfl`), so "fused is active" looked like the answer — but CfL is already off at q=90
(`GNC_NO_CFL=1` there changes nothing, 8.07 bpp either way), so fused is active at q=90 as well and
the paths still agree. Being on the fused path is necessary at most, not sufficient.

The remaining variable is the quantiser itself: dead zone 0.75 / step 16.0 at q=25 against ~0.05 /
~2.2 at q=90. q=50 and q=75 have the wide dead zone but CfL **on**, which disables fused, so they
cannot separate the two — which is why this only ever shows up below about q=30 and at subsampled
chroma. A dead-zone or rounding difference between the fused shader and the separate quantise
shader fits every row; a coding difference fits none of them.

**Why it matters beyond 0.12 dB:** the GPU path is the default, so the *shipped* encoder is the one
losing the quality, and the loss is invisible to any test that compares an encode against itself.
It also silently invalidates any experiment that compares a CPU-encoded arm against a GPU-encoded
one at q ≤ 30 — which is exactly the comparison an entropy-coder experiment wants to make.

**Repro:** `gnc benchmark -i <1080p png> -q 25 -n 1` against the same with `--cpu-encode`.
Pinned by `rice_gpu_and_cpu_encode_paths_differ_at_subsampled_chroma` in `tests/abac_bitstream.rs`,
which asserts the gap exists and is small, so a fix will fail that test and its comment says so.

### BUG-14 — Huffman's stream mapping had BUG-11 (**DONE 2026-09-07**)
`huffman_encode.wgsl`, `huffman_decode.wgsl` and `huffman_histogram.wgsl` all carry the same
`thread_id + s * STREAMS_PER_TILE` mapping that BUG-11 fixed in Rice, so a Huffman stream is one
tile column only at 256 px. Left alone deliberately: Huffman is not a default coder, is capped at
4 levels, and nothing measures through it. The fix is the same `stream_coeff_index` expression if
it ever matters.

The rANS fused histogram shader (`quantize_histogram_fused.wgsl`) has the same shape over 32
streams, but rANS *gains* 14–20% at tile 512, so its ordering is not costing it the same way. Not
filed as a bug — noted so nobody mistakes it for one.

**Fixed 2026-09-07.** `stream_coeff_index` in `huffman.rs` and in all three shaders, identical to
`rice.rs`. Four stills, host encoder (the GPU one cannot reach tile 512 — BUG-22), PSNR identical
at every point:

| tile | q=75 | q=90 |
|---|---|---|
| 128 | **−2.94%** mean (−2.09 to −3.88) | −1.48% |
| 256 | **byte-identical, 8/8 points, both encoders** | byte-identical |
| 512 | **−20.13%** mean (−17.17 to −23.16) | −11.90% |

Larger than Rice's −12.9 to −18.6%: Huffman also codes its zero runs with a per-group adaptive
`k_zrl`, so a stream that interleaves distant columns costs it twice. **256 is still the right
tile for Huffman** — after the fix 512 is +1.2% to +17.2% larger *and* lower in PSNR — the mapping
was never why 256 won, only why the margin read 28.8%. Byte-identity at 256 is asserted in
`test_stream_mapping_matches_legacy_at_256` rather than argued. Full numbers in RESEARCH_LOG.

The gate that measured this found the coder broken three ways first: **BUG-21**, **BUG-22** and
**BUG-23** below, all pre-existing, none of them the mapping.

### BUG-21 — Huffman built no codebook on the lossless path (**FIXED 2026-09-07**)
LOSSLESS-1 codes MED prediction residuals with `num_levels = 0`, so `num_groups = num_levels * 2`
was zero. `rice.rs` has `.max(1)` there; `huffman.rs` and `huffman_gpu.rs` did not. The host
encoder panicked (`index out of bounds: the len is 0 but the index is 0`); the GPU encoder built
no codebook, emitted no codes, and wrote a file that decoded at **4.1–9.7 dB with max error 255**
on all four stills at every tile size — and which was *smaller* than the same image at q=90, the
"beats its own theoretical ceiling" canary.

Fixed with `.max(1)` in both. `--huffman -q 100` at tile 256 is now **bit-exact lossless**, max
error 0, host and GPU encoders byte-identical: 1 076 689 bytes on kristensara_720p against Rice's
984 178, so Huffman is **+9.4% behind Rice at q=100**. Not a reason to un-park the coder.

### BUG-22 — Huffman's GPU per-stream output slot has no bound (**guarded 2026-09-07, not fixed**, P3)
`emit_byte` in `huffman_encode.wgsl` writes `stream_output[p_stream_word_base + p_word_pos]` with
nothing checking `p_word_pos` against `MAX_STREAM_WORDS`. A stream needing more than its 512-byte
slot spills into its neighbour's, and the host packs the neighbour's bytes back out. That is the
whole of the tile-512 corruption: **7.8–10.9 dB at q=90 on all four stills**, and 19–24 dB at
q=75 on two of them. Exactly BUG-9's shape in rANS, and guarded the same way — the host now
asserts with the tile, the stream and the byte count (`overflowed its 512-byte output slot
(563 bytes)`) instead of returning a picture.

**The real fix** is to size the slot from `symbols_per_stream` rather than fix it at 512 bytes:
worst case is about 4 bytes per symbol (significance + sign + an 8-bit code + exp-Golomb escape),
so 4 KB per stream at tile 512, about 37 MB of scratch for 1080p 4:4:4 — affordable, but it means
making `MAX_STREAM_WORDS` a parameter across the shader and the host. Not done: this is a parked
coder and **BUG-14's tile-512 arm is the only thing that wants it**, which the host encoder can
measure instead.

### BUG-23 — `clamp_code_lengths` does not terminate (**bounded 2026-09-07, not fixed**, P3)
It places excess code length by moving one symbol from length *j* to two at *j*+1, and only
lengths below the 8-bit maximum may donate — a donor pool of order 100 donations for a 64-symbol
alphabet, against an `excess_bits` that is not bounded by it. A steeply skewed histogram needs
over a thousand: simulated, a geometric distribution over 32 symbols exhausts the pool after 247
donations with 52 bits still to place. Measured, an encode of bbb_1080p at `-q 100 -t 512` spun at
**79% CPU for 8 minutes** before it was killed.

Unreachable until now only because BUG-21 meant no codebook was ever built on that path. The loop
now asserts when the pool is exhausted; the same encode fails in **0.148 s** with `62 bits of
excess left with no length below 8 to donate`. Failing rather than shortening the codes anyway is
deliberate — with excess left the lengths violate Kraft, so `assign_canonical_codes` would emit
codewords that are not a prefix code and the tile would decode to noise, trading a hang for silent
corruption.

**The real fix** is a length-limited construction: package-merge, or the cheap standard trick of
halving all frequencies and rebuilding until the natural depth fits (guaranteed to terminate — at
frequency 1 everywhere the alphabet is uniform and the depth is 6). Both change Huffman's
codebook, and therefore its bitstream, wherever clamping currently occurs. Not done for a parked
coder.

### BUG-24 — `clippy --target wasm32-unknown-unknown` fails on `main` (todo, P3)
11 × `no associated function or constant named 'new' found for struct GpuContext`, all in the
**bin** target: `GpuContext::new` is `#[cfg(not(target_arch = "wasm32"))]` and `main.rs` calls it
unconditionally. Reproduced on a clean tree at `bc851c7`, so it predates BUG-14's branch; it most
likely arrived with the GPU-selection work (`fcac02f`).

`cargo clippy --release --target wasm32-unknown-unknown --lib` is **clean, no warnings** — the
library is what WASM ships, and it is fine. So the failure is a CLI binary being type-checked for
a target it is never built for. Two honest resolutions: exclude the bin from the wasm target
(`required-features`, or a `#![cfg(not(target_arch = "wasm32"))]` on `main.rs`), or make the CLI's
context creation cfg-aware. Until one of them lands, CLAUDE.md's "both clippy targets must be
clean" cannot be satisfied as written, and every session hits it.

### BUG-9 — rANS's cumfreq table does not fit its shader, and the slot overflow was the symptom (**DONE 2026-09-07**)
Both limits are now refused by name, no configuration writes out of bounds any more, and the
capability itself is deliberately not extended. The entry below is kept because most of its
reasoning was right and one central claim was not.

**The recorded cause was backwards.** The entry said "this is not the symbol alphabet". It is —
just not via any one group's alphabet limit. Every subband group's cumfreq table for a tile is
loaded into *one* workgroup array in `rans_encode.wgsl`, so what has to fit is the **sum** of
`alphabet_size + 1` over the tile's groups. Past the end the shader read and wrote outside the
array, which is undefined, and the streams it then produced from those frequencies are what
overran their 4 KB slots. Measured with `--rans` at the default step, worst tile, Y plane:

| image | q=70 | q=75 | q=76 | q=77 | q=80 |
|---|---|---|---|---|---|
| kristensara_720p | 3504 | 4020 | **4165 refused** | 4317 | 4806 |
| blue_sky_1080p | 3510 | 4025 | **4173 refused** | 4325 | 4810 |
| bbb_1080p | 3406 | 3909 | 4052 | **4197 refused** | 4670 |
| touchdown_1080p | 3360 | 3853 | 3993 | **4138 refused** | 4600 |

The capacity is 4097. **Not one stream overflowed its slot at any point that completed** — on the
per-subband path the tables always give out first. That reproduces ENT-2's independently measured
ceiling exactly (q=75 encodes on all four, q=77 fails on all four, q=76 splits by content) and
supplies the mechanism: the split is the Y-plane alphabet crossing 4097 at different qualities.
Chroma is never close, at 600–2000 entries.

**The slot bug was real too, and it corrupted a neighbour.** `write_ptr` was decremented with no
bound check, and `stream_base_byte + write_ptr` is u32, so a decrement past zero wrapped to
`stream_base_byte - 1` and downwards — inside the *previous* stream's slot, with `write_byte`
ORing bits into data that was already correct. The shader now stops at the boundary and reports
the stream instead.

**Why the table limit is the worse of the two.** A slot overflow announces itself: the host sees a
wrapped pointer. A table overrun need not — a tile can overrun its tables and still emit streams
that fit, and then nothing downstream notices and the file is quietly wrong. The old behaviour was
also not deterministic; two builds of near-identical source disagreed about whether the same input
overflowed, which is what out-of-bounds workgroup access buys.

**An off-by-one that was always there.** A table of n symbols is n+1 cumulative frequencies, so at
`MAX_ALPHABET` the single-table path overran the array by exactly one entry — reachable with
`--no-per-subband --qstep 1.0`, which asks for 4097. The array is now `MAX_ALPHABET + 1`, four
bytes of the M1's 32 KB per threadgroup, so that path can use its own maximum alphabet.

**Still not extending the capability, and ENT-2's numbers say so from the other side.** rANS is
6–7% smaller than Rice at q<=20 and level at q=25–70, so the range runtime buffer sizing would
unlock is one where the coder measures level at best. Turning undefined behaviour into a sentence
was the whole value.

**Canary:** `GNC_DIAGNOSTICS=1` prints `rans_streams=N overflowed=M` and
`cumfreq_entries_max=N/4097` per plane. At q=15, where rANS is actually selected, the worst tile
asks for 361 of 4097.

**Verified:** byte-identical to the pinned parent commit on 4 images x q=1..100, `cargo test
--release` green, both clippy targets clean. Regression tests in
`tests/rans_stream_overflow.rs`. See decision record 0022.

---

Original entry, kept for its reasoning:


**The threshold recorded here was wrong and understated it.** "OK at qstep 2.0 and 1.5, panics at
1.0" came from `-q 15 --qstep 1.0`, whose preset carries 4 wavelet levels, different subband
weights and a different dead zone. In the **default** configuration bbb survives q=78
(qstep 3.594) and fails at q=80 (qstep 3.347) — it breaks at three times the recorded step. The
limit is a function of the whole configuration, not of qstep, so a static qstep guard would be the
wrong fix and a fallback keyed on one would misfire.

**Consequence:** rANS cannot be measured above **q=75** — not q=80; the ceiling was measured
twice, by ENT-2 and above. That is below the contribution operating point. Reaching it means threading a runtime buffer size through three encode shaders and six
allocation sites, the way Rice already sizes its own from qstep (`max_stream_bytes_for_tile`).

**Not doing it, on purpose.** The crossover re-sweep (below) says nobody wants what it would buy:
rANS is 1.9% smaller than Rice at q=70 on the mean of four images, regresses up to 6% on one of
them, and costs ~15% decode throughput. Recorded so the next person does not rediscover the crash
and assume the fix is cheap.

**Repro line corrected — and then corrected again.** This said `--rans` was a no-op kept for
backward compatibility. It is not: `src/main.rs` sets `EntropyCoder::Rans` and the bitstream's
`entropy_type` confirms it on 40 of 40 points (ENT-2). Only the flag's help text was stale.
`gnc encode -q 15 --qstep 1.0` reproduces it, and so does `--rans -q 77`.

**Stated cause was wrong — and this correction was itself wrong; see the top of the entry.**
This is not *only* the output slot. `rans_encode.wgsl` writes each stream
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

### QUAL-1 — Re-run MEAS-1 at the contribution operating point (**DONE 2026-09-06**)
The quality ladder above q=92 was dead until 2026-09-06 — q=92, 96 and 99 produced the same
picture, capped at qstep 2.0 by an rANS constraint that no longer applied. **Every
contribution-quality comparison in this repo predating that fix is invalid at the top of the
range**, because GNC was pinned while the competitor was not. The distribution-bitrate figures
(+306% to +617%) are unaffected.

**Done 2026-09-06. The contribution-quality gap is +90.5% BD-rate on PSNR — about 1.9x, against
the 5.6x mean recorded at distribution bitrates.**

1920x1080, 17 frames, ki=9, chroma 420, 8-bit, x264 at defaults — MEAS-1's parameters exactly,
only the operating point changed. q=85,92,96,99 against crf=1,2,4,8 (the first pass at
q=92-99/crf=4-12 left only 1.8 dB of curve overlap; extended to 6.5 dB).

| sequence | BD-rate PSNR-Y | overlap | BD-rate VMAF |
|---|---|---|---|
| bbb_extended | **+129.0%** | 49.4–55.9 dB | +122.6% |
| old_town_cross | **+71.9%** | 50.0–56.2 dB | +191.4% |
| crowd_run | **+70.6%** | 49.9–56.3 dB | +113.6% |
| **mean** | **+90.5%** | | +142.5% |

Nothing was fixed in the coder — the gap was always this size *here*; +456.7% to +672.1% was
measured somewhere else. Corroborates the Current Focus +118–177% intra gap independently, and
lands slightly better than it.

**Never quote a VMAF BD-rate above about q=85.** Widening the ladder moved the VMAF figure by a
mean of **47.5 points** (old_town: +81.1% → +191.4%) and the PSNR figure by **1.0 point**. On
old_town VMAF reads 99.62–99.68 across a 6 dB PSNR spread — no signal left to integrate. This is
COORDINATION rule 3 with a magnitude attached.

**Colour reverses at matched rate.** ~~(Withdrawn 2026-09-07 by CHROMA-2, decision 0020 — the
control reverses it back: x264 ahead on 6 runs of 6.)~~ CIEDE2000 on decoded RGB, rate matched to 1%: GNC 0.611 vs
x264 0.684 (bbb), 0.911 vs 0.949 (old_town) — GNC better on mean and 95th percentile on all three
sequences, with fewer pixels past the JND, **while losing luma by 7.4–8.8 dB at the same points.**
That is the two codecs allocating rate differently between luma and chroma, not GNC being better.
So a single luma BD-rate overstates the gap for a colour-weighted use case and understates the
luma deficit. Quote both. (crowd_run's pair is 1.08x on rate, so its dE00 win is partly bought;
the two matched pairs give +4.1% and +10.7%.)

**Correction to MEAS-1's record.** It states 17 frames on bbb / touchdown / old_town. `bbb.y4m` has
**8 frames** and there is no `touchdown` sequence in the tree; with `--frames 17` the GNC arm dies
on a missing PNG. Either MEAS-1 ran against sources no longer present or against fewer frames than
recorded — so its absolute figures are not reproducible as written. Sources used here, stated
explicitly and MD5-confirmed distinct: bbb_extended (24 frames), old_town_cross (200), crowd_run
(32).

**Follow-up worth doing (new, P2):** sweep `GNC_CHROMA_WEIGHT` at *matched total rate* to see
whether moving bits from chroma to luma closes part of the +90.5%. Judge on luma PSNR **and** dE00
together — that sweep was run once on VMAF, looked like a free 15%, and reversed sign on dE00.

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

**Harness built 2026-09-06, not yet run.** `scripts/gpu_tier_bench.py --density` and `--hwenc`
sweep concurrent instances of GNC and of the machine's fixed-function encoder over the same clip.
Two things to know before reading its NVENC rows: the 12-session cap is a GeForce driver
restriction and will not appear on a professional Ada part, and the rows are **not
quality-matched** — `meas1_vs_h264.py` is the harness for that.

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

**Harness built 2026-09-06, not yet run.** `scripts/gpu_tier_bench.py --tier` measures the
single-frame encode/decode loop on every GPU a machine exposes, best of N processes, and says in
plain words when the spread is under 15% — the failure signal. Needs a machine with more than one
GPU; a laptop with an integrated and a discrete GPU is the cheapest version of the experiment.
`docs/GPU_TIER_TEST.md` states what it proves and what it does not.

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

> **SUPERSEDED 2026-09-06 by QUAL-1.** These figures were measured at *distribution* bitrates
> (crf 18–38) with the quality ladder above q=92 dead, and on VMAF, which is saturated at the
> contribution end. Re-run at the operating point GNC is built for, the gap is **+90.5% BD-rate on
> PSNR — about 1.9x.** Nothing in the coder changed. The sources stated here are also not
> reproducible: `bbb.y4m` has 8 frames, not 17, and there is no `touchdown` sequence in the tree.
> Keep this entry as the historical distribution-bitrate figure; do not quote it as current.

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
unsurprising, since it *is* JPEG 2000's coder.

> **Both halves of that sentence are corrected by ENT-4 (2026-09-07).** The **+28.3%** was measured
> against OpenJPEG's *reversible 5/3* default, the wrong mode for a lossy comparison; measured with
> `-I` the intra gap is **+54.2%** on RGB PSNR. And the **~9%** was an offline model: the shipped
> coder measures **−16.0%** in-codec at identical pixels, 1.7x more. So the share is **a half, not a
> third** — 54.2% → 27.1% with `--abac`. Both errors were real and they pointed in opposite
> directions, which is why the answer had to be measured. Larger than everything shipped today put together
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

**Part 4 — GPU decode written, bit-exact, and throughput is the blocker (2026-09-06).**
`src/shaders/abac_decode.wgsl` + `src/encoder/abac_gpu.rs` + `tests/abac_gpu.rs`. One thread per
code-block. Verified bit-exact against the CPU coder across seven geometries including ragged
subband-edge blocks and degenerate planes — which caught two real bugs that produced *plausible
wrong images* rather than errors (WGSL `firstLeadingBit` vs Rust `leading_zeros` differ by one; a
`vec3` tail makes a uniform struct 32 bytes not 16).

| code-block | blocks per 1080p luma plane | decode |
|---|---|---|
| 64 px | 640 | ~50 Mcoeff/s |
| 32 px | 2560 | ~85-105 Mcoeff/s |

A 4:4:4 frame is 7.86 Mcoeff → **75-155 ms, 6-13 fps**. Rice is far faster. **On throughput this
does not pass.**

Two optimisations that did *not* help: moving the context scratch from device memory into
workgroup memory, and sorting blocks by size so a SIMD group holds equal-sized work. That both
failed points at the real cause — **one serial coder per thread wastes most of a SIMD group
because the divergence is data-dependent**, every lane's renormalisation loop running a different
number of iterations per symbol. In hindsight this is exactly why Rice uses 256 branch-free
streams per tile.

**Timing is not currently measurable.** Three targeted optimisations — context scratch into
workgroup memory, blocks sorted by size for SIMD-group uniformity, and thread-interleaved
workgroup arrays to kill a 32-way bank conflict — all returned *nothing*, while the same input
timed 25.2 / 31.1 / 37.5 ms across runs. A 48% spread on identical work with five sessions
hammering the GPU means the figures above are an order of magnitude and no finer, and those three
optimisations are untested rather than disproven. Re-time on an idle machine before touching the
shader again; optimising against noise produces changes that look justified and are not.

**Part 5 — two coder variants behind a switch (2026-09-06).** Since throughput is not measurable
while the machine is shared, both plausible engines are built and selectable, with a bench that
settles the grid in one idle run: `GNC_ABAC_CODER=interval|range`, `GNC_ABAC_CB=<px>`,
`cargo test --release --test abac_bench -- --ignored --nocapture`.

*Interval* is bit-renormalising (0-16 iterations per decision, data-dependent). *Range* is
byte-renormalising, LZMA/VP8 family — at most 3 iterations, usually 0 or 1, so **~8x fewer
iterations in the decoder's hottest loop**, and it needs no narrowed interval.

Rate on real coefficients (bbb, q=55, vs shipped Rice):

| coder | cb=32 | cb=64 | cb=128 |
|---|---|---|---|
| Interval | −15.0% | −19.2% | −20.0% |
| Range | −10.2% | −17.0% | −18.4% |

Throughput, paired within one run (provisional, machine loaded, but the *ratio* is paired):
Interval 38.1 / Range **84.2** Mcoeff/s at cb=64; 77.2 / **157.5** at cb=32. About 2x at both
sizes.

**Range at cb=64 dominates Interval at cb=32 on both axes** — −17.0% vs −15.0% rate *and* faster.
So if throughput matters the answer is the range coder with bigger blocks, not the interval coder
with smaller ones.

Range's rate penalty widens as blocks shrink because its final flush is 5 bytes per block: at
cb=32 that is 42 KB per frame. **Fixable, and the obvious next step if Range wins on speed.**

**Diagnosis narrowed by MEAS 75ca12b.** A trivially parallel shader does 7.86 M elements in
1.02 ms, and a wavefront *dependency* costs only 4.9×. So a serial dependency is cheap and abac's
problem is not that it has one. The difference is shape: a wavefront keeps 256 threads progressing
in lockstep, abac gives one thread 4096 sequential symbols with 31 lanes idle. That points at
per-symbol instruction count on a single lane — three context-coded decisions per coefficient,
each with a variable-length renormalisation loop — rather than memory behaviour, which is what all
three failed optimisations targeted. **The structural fix to try: replace the renormalisation loop
with a fixed-cost table lookup, as VP8's bool decoder does.** Coder-core rewrite, not a shader
tweak; the 16-bit interval work is a prerequisite either way.

**Three things would change the answer**, in order of leverage: (1) re-time on an idle machine —
cheapest and unambiguous; (2) a table-driven binary coder with fixed per-symbol cost instead of a
renormalisation loop, which is a coder-core rewrite; (3) accept it as an encode-side or CPU-decode
option, which is a product decision for the owner, not an engineering one.

**The rate result stands regardless: −19% to −25% at identical quality**, verified against the
shipped Rice coder with per-block roundtrip and coverage assertions.

**Part 6 — measured on REAL coefficients against Rice in the same process, at q=90
(2026-09-06). The Range coder makes this a live trade, not a rejection.**

`GNC_ABAC_COMPARE=1` now decodes the frame's *real* code-blocks on the GPU, verifies all 7 864 320
bit-exact against the CPU coder, then times seven dispatches. `tests/abac_bench.rs` times the same
grid on *synthesised* planes, which is a slightly different workload — significance density drives
the binary-decision count, and the synthetic proxy runs ~6% fast at cb=64. More importantly, only
a real frame can be paired against the codec's own whole-frame `Decode:` figure **inside one
process**, which is what makes a ratio survive a shared machine.

Identical real blocks, bbb, q=90, cb=64:

| coder | entropy-stage decode | rate vs shipped Rice |
|---|---|---|
| Interval | 113.9 ms (69.1 Mcoeff/s) | −14.5% |
| Range | 33.8–41.4 ms (190–232 Mcoeff/s) | −13.8% |

**~3× the throughput for 0.7 points of rate** — a wider speed gap and a much narrower rate gap
than the synthetic bench found at q=55, because Range's 5-byte-per-block flush amortises as blocks
fill. Both effects favour Range *more* at the operating point that matters. Range at cb=64 also
dominates Range at cb=32 (−13.8% vs −10.9%, 33.8 ms vs 31.4 ms).

Range rate at q=90, cb=64: bbb −13.8%, blue_sky −16.6%, touchdown −16.3%, kristensara −20.1%,
**mean −16.7%**.

**Cost per frame, three repeats at load 17 (the quietest window available):** abac Range entropy
stage ~39.5 ms against Rice's *whole frame* ~35.6 ms. The shared inverse wavelet and colour
transform cancel, so `abac_frame − rice_frame = abac_entropy − rice_entropy`, bracketing the
answer between **1.11×** (if Rice's entropy stage were its whole frame) and **2.11×** (if it were
free). Rice's frame decode is nearly q-insensitive — 29.3 / 37.6 / 32.5 ms at q=20 / 55 / 90,
non-monotonic, i.e. load rather than coefficient count — so a branch-free 256-stream coder's
entropy stage is small and flat, and the estimate sits near the ceiling: **realistically
~1.7–1.9× frame decode for ~16.7% of rate.**

**RESOLVED on an idle machine, same day.** `GNC_RICE_DISPATCH_REPEAT=k` issues Rice's entropy
dispatch k times (it is idempotent, so the slope isolates the stage without timestamp queries):
29.50 ms at k=1, 85.16 at k=5, 141.55 at k=9 — **14.0 ms per dispatch**, two independent slopes
agreeing to 0.6%. **Rice's entropy stage is 47% of frame decode**, which is also the ceiling on
any entropy-coder work here: a free entropy coder would buy 1.9× and no more.

Settled figures (bbb, q=90, cb=64, min and median within 1%): Range entropy **33.0 ms**
(238.6 Mcoeff/s), Interval **96.3 ms** (81.7). Since `abac_frame = rice_frame − rice_entropy +
abac_entropy`, the bracket collapses:

| coder | implied frame decode | vs Rice | rate |
|---|---|---|---|
| **Range** | 27.5 − 14.0 + 33.0 = **46.5 ms** | **1.69×** | −16.7% mean |
| Interval | 27.5 − 14.0 + 96.3 = **109.8 ms** | 3.99× | −14.5% |

**The decision is ~16.7% of rate for ~1.65-1.7× frame decode.** Under GOALS §5 "form first, then
speed" (owner, 2026-09-06) that resolves in favour of **keeping it**: during the form phase a
change that closes a real compression gap is worth taking even when it costs decode time, because
a fast architecture that is behind on compression is the harder problem to fix afterwards. The
1.69× is therefore **logged throughput debt**, not a veto — it goes on the list the later
performance push works from, with a measured cost rather than a re-measurement project.

Recommended configuration if taken up: **Range at cb=64**, which dominates every other cell
measured. Still outstanding before it could ship: bitstream integration (a GP18 generation with
`EntropyCoder::Abac`, per-block length fields, block size in the tile header) and inter frames —
all of the above is intra, and residual statistics differ.

Also fixed the measurement method itself: three consecutive processes on identical input read
66.5 / 45.3 / 34.9 ms on an *idle* machine — the GPU's clock ramp, not load. `abac_bench` now uses
24 dispatches and reports best-of with `med/best` as the settled-or-not diagnostic, and documents
`--test-threads=1` (its two tests were contending for the GPU). An idle machine is necessary and
not sufficient.

**Scope warning on an earlier verdict.** A "closed by measurement — do not re-test" note was
written for abac on the strength of the Interval coder alone, and it was wrong as a statement
about the idea: Range is ~3× faster on the identical workload. "Closed by measurement" is a claim
about a mechanism, not about a measurement, and a "do not re-test" note is the most expensive kind
of wrong because it is written precisely so nobody checks it again. Scope such notes to what was
actually varied.

**Part 7 — SHIPPED 2026-09-07 (ABAC-SHIP). GP18, entropy type 5, −16.6% to −18.8% at identical
pixels.** `EntropyCoder::Abac` / `--abac`, `src/encoder/abac_tile.rs`, GPU decode into `scratch_a`.
Measured through encode → file → GPU decode on four images; entropy coding is lossless, so both
arms decode to the **same pixels** and the rate delta is exact rather than a rate/quality trade:

| q | bbb | blue_sky | kristensara | touchdown | mean |
|---|---|---|---|---|---|
| 50 | −17.8% | −18.5% | −19.4% | −19.5% | **−18.8%** |
| 75 | −14.7% | −15.8% | −19.3% | −16.8% | **−16.6%** |
| 90 | −14.2% | −17.2% | −20.9% | −16.9% | **−17.3%** |
| 100 (lossless, bit-exact) | −14.2% | −14.5% | −15.0% | −10.0% | **−13.4%** |

The q=90 diagnostic predicted −16.7%; the real bitstream measures −17.3%, every image within 0.8
points, the mean slightly *better* for the reason predicted in advance (25 two-byte block headers
per tile against Rice's tile header plus 256 length fields). **Lossless takes the FFV1 gap from
+23.9% to +7.3%**, against an FFV1 level-3 gbrp encode run the same day rather than a figure
carried forward.

Rice stays the default and **BASELINE.md does not move** — BASELINE reproduces exactly on this
commit (q=75 44.84 dB / 4.53 bpp, q=90 50.06 dB / 8.07 bpp), and Rice files are identical before
and after apart from the four magic bytes, which is the whole of GP18. Reasoning in `docs/decisions/0017`. The ~1.69× decode debt is
unchanged and was not re-measured.

Two things found on the way. **The decode shader wrote `array<i32>` into `scratch_a`, which every
other entropy decoder writes as f32** — the file was already the right size and PSNR came back
`NaN`, because −1 as i32 is a quiet NaN as f32. Rate right, picture absent: a bpp-only benchmark
would have recorded the win. And **Rice's GPU and CPU encode paths do not produce the same
pixels** (35.51 vs 35.63 dB on bbb at q=25; max |diff| 1.69 at 4:2:2) — both arms are Rice, so it
is the quantise stage, not entropy coding. Pre-existing, not chased, now pinned by a test so it is
not re-found as an abac bug. It is why **q=25 is not quoted as a rate figure** above.

**Still open:**
1. ~~GPU decode shader and honest fps against Rice on an idle machine.~~ Done — Part 6.
2. ~~Bitstream integration.~~ Done — Part 7.
3. **Inter frames — measured, then RETRACTED the same day (BUG-18).** ~~−14.4% mean at q=90~~ — abac's video path is the CPU-entropy P-frame path, and that path encodes every P-frame wrong (diverges from the GPU path on the *first* P after an I, and costs 2.1-3x the bytes). The comparison put abac on a broken arm. **Re-measure after BUG-18 is fixed.** The original text follows for the record:
   Three sequences (crowd_run, old_town_cross, bbb_extended), 24 frames, ki=9, 4:4:4, I+P:
   −12.17% / −12.00% / −19.11% on the I+P bitstream, against −11.48% / −12.65% / −14.25% for the
   all-intra control from the same runs. The standing note here was that abac's contexts were
   tuned on intra coefficients so inter might pay less; **it pays slightly more** (−14.4% vs
   −12.8%). PSNR is exactly equal on two of three; bbb_extended differs 0.03 dB, so its −19.11%
   carries a small caveat. `--abac` had to be added to `benchmark-sequence` / `encode-sequence`
   first — the sequence path had no entropy-coder flag at all.
   **These are quality-matched to ≤0.03 dB, not pixel-exact like the intra rows** — on the inter
   path the two *encode paths* never agree exactly (BUG-22, filed from this measurement), and abac
   is CPU-encoded where Rice is GPU-encoded. **q=75 is not quoted at all**: the gap is worth
   0.54 dB there.
4. **CPU encode is 129 ms/frame against Rice's 23 ms (todo, P3).** Serial per symbol by
   construction, but parallel across ~3000 code-blocks and currently single-threaded. This is what
   stands between abac and being a candidate default, more than the decode debt does.

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

### BUG-8 — The encoder measured a reconstruction that never leaves it (**FIXED 2026-09-06** — a metric bug, not drift)
Filed as a suspected encoder/decoder divergence. It is not: the reference buffers never differ.
`bench::quality::psnr` compared raw `f32`, while the decoder's output is what `pack_u8.wgsl`
writes — `u32(clamp(f + 0.5, 0.0, peak))`.

Two effects pulling opposite ways. **Clamping helps** (a reconstruction overshooting 255 in a
bright sky is pulled back, and the overshoot grows down a GOP — which is what looked like drift).
**Rounding hurts** (up to half a level the float reconstruction never carries), and dominates at
high quality. The tell was that the gap **changes sign**: blue_sky last frame, decoder minus
encoder, +0.222 dB at q=25, +0.355 at q=50, +0.231 at q=75, **−0.530 at q=90**. Drift accumulates
one way; this does not. It was also content-dependent — flat on old_town_cross, +0.36 on
blue_sky — which a real divergence would not be.

Every metric now quantises both inputs to the output grid first, same expression and same order as
the pack shader. Gap collapses to ±0.005 dB. **This matters most where GNC is aimed:** at
contribution quality the old metric overstated by half a dB, and it is what rate control and any
RD decision read.

Two consequences, both correct: the checkerboard q90 golden baseline moved 50.82 → 50.19 dB (the
cost of rounding, previously uncharged — the codec did not change), and the PSNR monotonicity test
now stops at 55 dB, above which the 8-bit grid dominates and two encodes within ±1 of the source
can order either way. Verified that a plain gradient reconstructs *exactly* from q=92 up, so there
is no defect behind that.

### RATE-1 — Above ~q=90 an 8-bit encode buys precision it cannot emit (**ANSWERED NO 2026-09-07 — do not build the rule**)
**Measured across content, and the premise does not survive it.** The item said the gradient is
the best case and real content would show less. It shows *nothing*: **no real image reaches
bit-exactness anywhere below q=100**, and on bbb and blue_sky nothing gets within even 1 LSB. Every
rung above q=90 is still buying 8-bit-visible improvement — max error falls 6 -> 4 -> 2 -> 1 and the
share of differing pixels falls from ~54% at q=86 to ~7% at q=99. Recoverable share of the top
rate: **89.4% on the gradient, 91.8% on a two-axis ramp, 5.6% flat, 0.0% on all four photographic
stills.** A bit-depth-aware rate rule would recover nothing on the content this codec is for.

Original premise, kept for the record: on the test gradient q=90 costs 0.275 bpp and q=95 costs
1.142 bpp for bit-identical 8-bit output. That is real, and it is a synthetic axis-aligned ramp.

**The synthetic number is the trap, and it is worth naming.** A bit-depth rate rule validated on
flat512 or the gradient would have shown a large win and delivered nothing on any real image. That
is the same shape as two errors already in this log — VMAF validating a chroma decision, and
`opj_compress`'s reversible 5/3 default scoring JPEG 2000 in a lossy comparison: the instrument
agreed with the hypothesis because the content was chosen in a way that let it. Content selection
is part of the instrument.

Two things worth keeping from the sweep. The ladder is **not monotonic in rate** — flat512 costs
0.0450 bpp at q=86 and 0.0370 at q=90 — so "the first q that qualifies" is the wrong statistic
anywhere this is re-measured, and anything interpolating GNC by rate should flag a rung whose rate
falls while q rises (MEAS-9's harness now does). And for a 10-bit target the extra precision is real, so the ladder
itself was never the problem. Harness: `scripts/meas_rate1_precision.py`, measured at `fa32a26`.
Numbers in RESEARCH_LOG.

### RATE-2 — Above q≈95-98 the lossy ladder costs more than bit-exact lossless (todo, P1)

**On every real image measured, the top of the wavelet ladder spends more bytes than lossless while
delivering worse output.** Found by the RATE-1 sweep; it is a different defect and a bigger one.

| image | q=100 (MED, bit-exact) | q=99 (wavelet) | q=99 penalty | dominated from |
|---|---|---|---|---|
| bbb_1080p | 12.4836 bpp | 13.6439 bpp @ 59.59 dB | **+9.3%** | q=98 |
| blue_sky_1080p | 8.3068 | 11.6762 @ 60.14 dB | **+40.6%** | q=95 |
| kristensara_720p | 8.0521 | 10.9428 @ 59.59 dB | **+35.9%** | q=96 |
| touchdown_1080p | 10.0713 | 13.0570 @ 59.56 dB | **+29.6%** | q=96 |

Mean penalty at q=99: **+28.9% of the bitrate for a worse picture than bit-exact.** Measured at
`fa32a26` and verified outside the harness — blue_sky q=100 is 2153118 bytes whose decoded pixels
hash identically to the original (`ffmpeg -f rawvideo` md5 `46839b0e...`), against q=99's 3026470
bytes at 60.14 dB. Encoding is deterministic (two runs, identical md5).

**Cause, and it was created by an improvement.** LOSSLESS-1 made q=100 14.9% cheaper by coding MED
residuals instead of wavelet coefficients. COORDINATION notes it invalidated every lossless figure;
nothing noticed it moved the lossless price *below* the top of the lossy ladder. The two paths are
never compared, because one is "lossy" and the other "lossless", and no test covers the crossover.
Not universal — smoothramp512 (1.8024 bpp), flat512 (0.0865) and noise512 (29.9438) are **not**
dominated, because MED is poor on ramps and on noise. The dominance appears wherever MED does well,
i.e. on every photographic image.

**Cheapest fix:** at q>=95, encode both ways host-side and keep the smaller. That is the PCRD logic
already used within a tile, lifted to the transform choice, and it costs two encodes at the top of
the ladder. It has a bitstream-visible consequence to decide deliberately — a q=97 file would then
carry `transform_type = 2` — so it wants a decision record, not just a patch. The honest
alternative is to stop advertising q=95-99.

**Do not measure a contribution operating point at q=95-99 without knowing this.** Any BD-rate
whose ladder includes those rungs has scored GNC through its dominated range.

**Confirmed independently, different inputs, same day** (INTRA-NEARLOSSLESS session, found while
measuring the ladder's top for a different item — the two sessions collided on this defect and
both filed it as RATE-2; this entry is the one that stands). Padding-neutral crops (1536x1024,
1024x512 for kristensara) rather than full frames, so the penalties differ in size while agreeing
in sign and cause: q=99 against bit-exact q=100 is **+1.6% (bbb), +20.0% (touchdown), +31.9%
(blue_sky), +33.2% (kristensara)**.

That run adds the boundary in **qstep** terms, which is what a fix has to be written against.
Sweeping `--qstep` at q=99, the coarsest-first rung that still costs *less* than bit-exact:

| image | first non-dominated qstep | its quality |
|---|---|---|
| blue_sky | 1.6 | 52.51 dB |
| kristensara | 1.6 | 52.30 dB |
| touchdown | 1.3 | 53.58 dB |
| bbb | 0.9 | 57.21 dB |

The anchor ladder reaches qstep 1.30 at q=96 and 0.75 at q=99, so on three of four images the whole
band above q≈96 is dominated. It also rules out one framing: **sub-unit qstep is not wasted
precision in PSNR terms** — qstep 0.75 buys 3.9 dB over qstep 1.0 for 13% more bits on bbb
(59.92 vs 56.01 dB), a normal RD slope. The lossy rungs are priced correctly against each other and
mispriced only against lossless, which is what makes option (1) — compare and keep the smaller —
the right shape of fix rather than a ladder clamp.

### MEAS-2 — Feature toggling: what contributes and how much? (todo, P3)

**Back in the queue 2026-09-07.** This heading read `(in progress 2026-09-06)` for a day
with nobody holding a claim on it — the stale-marker failure that COORD-1 removed the
instruction for. Four toggles are measured below and the sweep is not finished, so it is
startable. **P3 is carried over from the stale marker, not measured — re-rank it if that
is wrong.**

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

**Follow-up (done by ENT-2, 2026-09-07):** the `--rice` CLI help said Rice is "~30% worse
compression"; the text is corrected in five subcommands. Note this entry's own figure — "Rice is
better above q≈25 and by 8-12% at q=70" — did not survive re-measurement either: after ENT-1 the
two coders are level above q=25 (mean +0.1%), and Rice is 6-7% *larger* at q<=20.

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

### MEAS-3 — RD-curve on sequences (**DONE 2026-09-07**)

No encoder change was needed: this item's premise ("rd-curve lacks --chroma-format and --vmaf on
sequences") is stale, `benchmark-sequence` has both. `scripts/meas3_sequence_rd.py` loops it.
`park_joy` is not in the test material; `old_town_cross` stood in, matching the QUAL-1 set.

Three sequences, 18 frames (two exact GOPs at ki=9), 4:4:4, q=25–95, two arms: ki=9 (shipped
inter) against ki=1 (all-intra). **BD-rate of inter against all-intra, positive meaning inter
needs more bits:**

| sequence | mean PSNR | worst-frame PSNR | VMAF (q≤85) |
|---|---|---|---|
| crowd_run | **+15.9%** | **+32.4%** | discarded — overlap 99.55–99.84, saturated |
| old_town_cross | **+22.2%** | **+35.4%** | +35.6% |
| bbb_extended | **−24.2%** | **−10.5%** | −9.9% |
| **mean** | **+4.6%** | **+19.1%** | |

**The inter path's rate saving does not survive being measured at matched quality.** GOALS §4's
"saves 17–27% vs all-I" is an equal-setting figure whose quality evidence was VMAF 99.09 vs 99.10;
at the same q the inter arm codes P and B frames coarser on purpose (TUNE-6), so on crowd_run at
q=70 it spends 4.7 bpp against intra's 8.0 **while sitting 7.6 dB lower**. Above q≈85 it stops
paying at all, and at q=95 it costs *more* than all-intra on two of three sequences. Full tables
and caveats in RESEARCH_LOG; the reversal is decision record 0019.

### INTER-1 — The inter path is a loss at contribution quality; decide what it is for (todo, P1)

MEAS-3 measured the shipped I/P/B configuration against all-intra as BD-rate and found **+4.6% on
mean PSNR, +19.1% on worst-frame PSNR**, winning only on low-motion animation (bbb_extended
−24.2%). Above q≈85 the saving is gone; at q=95 inter costs more. GNC is a contribution codec
(GOALS §1), so that is the operating point that matters.

**This is a positioning question with an engineering half, and neither is settled by MEAS-3's three
sequences at 18 frames.** What would settle it:

1. **Sweep ki** (1, 2, 4, 9) at q=85–99 on the same sequences. If the win only appears at short
   GOPs, the default is wrong rather than the feature.
2. **Price the worst frame deliberately.** TUNE-6's 1.25× P-frame quantiser scale is what makes the
   inter arm's quality uneven; at contribution quality the right scale may be 1.0. That is one
   constant and a re-measure.
3. **Then decide whether inter stays a default at all** for a contribution codec, or becomes an
   opt-in for the low-motion case where it demonstrably pays — the shape ABAC-SHIP used.

Do **not** start by deleting the inter path: it is −24.2% on animation, and MEAS-4 located the gap
in prediction quality, which is a fixable thing rather than a wrong architecture.

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

### CHROMA-2 — Is the colour result more than an allocation difference? (**DONE 2026-09-07** — no, and worse)

**Answer: it is an allocation artefact, and the control found something stronger than that.**
`scripts/meas_chroma2.py`, three sequences x {4:2:0, 4:4:4}, q=85, 24 frames, ki=9, x264's crf
bisected to within 1% of GNC's bytes for each `--chroma-qp-offset` in {0,-2,-4,-6,-8}:

| sequence | chroma | GNC dE00 | best x264 dE00 | offset | GNC luma (YCoCg-R) | x264 luma |
|---|---|---|---|---|---|---|
| bbb_extended | 420 | 1.183 | **1.107** | -8 | 46.73 | 46.59 |
| bbb_extended | 444 | 0.537 | **0.464** | -6 | 46.81 | 48.91 |
| old_town_cross | 420 | 0.916 | **0.550** | 0 | 46.22 | 53.32 |
| old_town_cross | 444 | 0.923 | **0.385** | 0 | 46.22 | 53.59 |
| crowd_run | 420 | 0.859 | **0.518** | -8 | 46.20 | 50.27 |
| crowd_run | 444 | 0.839 | **0.348** | 0 | 46.22 | 53.54 |

**x264 wins colour 6 of 6, and on five it needs no offset — it leads colour at offset 0 while also
leading luma by 4.1-7.4 dB.** There is no trade to price. The README, GOALS, BASELINE, POSITIONING
and CLAUDE.md rows are withdrawn; decision 0020 records it.

**The harness control that mattered more.** RGB -> yuv -> RGB with no codec at all costs dE00
**1.057** on bbb at 4:2:0 — ~90% of everything the codecs then scored — and on three of the six
runs x264's dE00 equalled that floor to three decimals, meaning its coded colour error was nil.
The floor is paid by the x264 arm only (GNC takes reference PNGs directly; YCoCg-R is reversible),
so the comparison is conservative: the handicapped arm still wins. Any two-codec colour comparison
here must state its floor.

**QUAL-1's colour table does not reproduce** — same nominal config now gives 32% more bytes *and*
worse dE00 — and it was taken an hour before CHROMA-1 changed q>=85 output, which COORDINATION
already flagged as invalidating q>=85 figures. Its luma BD-rate is untouched.

### Superseded statement of CHROMA-2 (todo, P1)

The README's one claimed win over x264 is CIEDE2000 on all three sequences at rate matched to 1%,
while trailing 7.4–8.8 dB on luma. Both halves are real, and together they say GNC spends a larger
share of the budget on chroma. They do **not** say the transform preserves colour better, and the
README now says so explicitly rather than implying the stronger claim.

**The control that settles it:** give x264 the same allocation — `--chroma-qp-offset -6`, or
whatever offset matches GNC's split — and re-measure CIEDE2000 at the same *total* rate on all
three sequences.

- If x264 takes the dE00 win back, the colour row is an allocation artefact. Say so and drop it.
- If it cannot — if the luma cost is disproportionate — then 4:4:4 wavelet plus CfL is doing
  something a block DCT at 8x8 does not, and that is a contribution argument worth making, since
  chroma keying and grading are exactly what contribution material has to survive.

Metric rules apply: dE00 via `scripts/chroma_metric.py`, luma in YCoCg-R via `scripts/ypsnr_de00.py`,
not luma computed from decoded RGB. Success criterion: state the offset, the matched rate, and the
dE00 delta per sequence. Half an afternoon.

### MEAS-9 — GNC against the codecs it actually competes with (**DONE 2026-09-07**)

Filed as "JPEG XS in `--compare-codecs`". Delivered as `scripts/meas9_contribution.py`: seven arms
— GNC, JPEG XS 4:4:4 and 4:2:2, ProRes 4444 and 422, VC-2, JPEG 2000 in both transform modes — on
four images, through **one metric path** (each arm decodes to an 8-bit RGB PNG; every figure comes
from that PNG, so no arm reports its own quality). Full numbers in RESEARCH_LOG 2026-09-07.

**Mean BD-rate, GNC vs each arm. Positive = GNC needs more bits at matched quality.**

| arm | Y-PSNR (YCoCg-R) | RGB PSNR |
|---|---|---|
| **J2K 9/7** (irreversible) | **+79.7%** | **+54.2%** |
| J2K 5/3 reversible (opj default) | +67.9% | +20.2% |
| JPEG XS 4:4:4 | +29.4% | **−10.2%** |
| ProRes 4444 | +29.3% | +20.2% |
| JPEG XS 4:2:2, ProRes 422, VC-2 | not computable — their ranges do not reach GNC's | |

**The load-bearing result: JPEG 2000 uses the same transform as GNC — 9/7 wavelet, five levels —
and needs 54% fewer bits on RGB and 80% fewer on luma, winning on dE00 at matched rate too.** When
the transform is the same, the gap is not the transform; it is the entropy coder and coefficient
modelling. J2K's is EBCOT. abac (shipped the same day, GP18, `--abac`, −17.3% at q=90) is about a
third of the RGB gap, so it is the right lever and not the whole answer. **Re-run this comparison
with `--abac` on** — one flag on the GNC arm — before deciding what else EBCOT-ish is worth
building.

Two results that change how figures must be quoted:

- **GNC protects chroma more than any of the five.** Y-PSNR minus RGB PSNR at ~4.5 bpp on bbb:
  GNC +1.39 dB, VC-2 +1.99, ProRes 4444 +2.33, J2K 9/7 +2.56, JPEG XS 4:4:4 **+5.50**. That single
  allocation difference flips the ranking against JPEG XS between the two metrics, so **neither
  number alone ranks GNC against a 4:4:4 incumbent.** Measured now against five independent codecs
  in the same direction, so it is a property of GNC, not of one comparison.
- **Content spread exceeds the codec difference.** ProRes 4444 ranges −4.3% to +52.4% RGB across
  four images; GNC is ahead on bbb (animation) and clearly behind on touchdown (crowd and grass
  texture, −4.8 dB RGB at matched rate). A single-image result on this axis is an anecdote.

**Three instrument errors, all of which flattered GNC**, and all now guarded in the harness:
BD-rate integrated over 0.1 dB of overlap read as −58.6% (now ≥3 dB required, fit windowed);
`opj_compress` defaults to **reversible 5/3**, the wrong mode for a lossy comparison, worth 2-3 dB
to J2K — with the default J2K looks 20% *behind*, with `-I` it is 54% ahead (**any J2K figure in
this repo taken without `-I` understates it**); and a saturated arm reads as a landslide. Validated
with `--selftest` (identical curves +0.000%, a uniformly 20%-cheaper reference exactly +25.000%,
narrow overlap refused) and byte-identical across two runs and two Python interpreters.

**Still open from this item, and each is a successor rather than a gap in the result:**

1. **Re-measure with `--abac`.** Cheapest and highest-value: it directly tests how much of the J2K
   gap the shipped coder closes.
2. **Inter.** Every arm here is all-intra on stills. The entropy gap on inter residuals is
   unmeasured and is where the H.264 gap is largest.
3. **Latency, which cannot come from this build.** The arm64 JPEG XS build has every SIMD kernel
   disabled, so rate and quality are exact and **no speed figure from it means anything.** The
   original request to sit the JPEG XS figure next to MEAS-6's latency row needs a machine with the
   SIMD paths, or an explicit note that no JPEG XS speed figure exists here.
4. **q=100 / lossless arms.** Excluded because of BUG-15 (the wavelet lossless path was not
   bit-exact on main until 2026-09-07). Every GNC arm here is q=60-99 on the default MED path;
   `--huffman` and `--rans` are unused, both having open defects at high q.

**How to build the JPEG XS arms** (kept, because the correction cost two sessions a wrong
conclusion). ffmpeg here knows the codec id and has no implementation (`-codecs` shows
`..VILS jpegxs`), and Homebrew has no `libjxs` or `svt-jpeg-xs` formula — but **only SVT-JPEG-XS's
build system is x86-only.** Its C sources carry `#else /* ARCH_X86_64 */` scalar fallbacks for
every dispatch, so the portable path was written and merely never selected.
`scripts/build_jpegxs_arm64.sh` + `scripts/svt-jpegxs-arm64.patch` (commit `bc851c7`) clone
upstream at a pinned commit, gate the nasm discovery / `-DARCH_X86_64` / nine ASM object libraries
on a detected `SVT_ARCH_X86`, and build to `${TMPDIR}/svt-jpegxs/Bin/Release/`. Nothing is
prebuilt — run the script. It verifies rather than trusting the link: 1920x1080 yuv422p at
`--bpp 3` round-trips to **PSNR y 44.484 dB**, reproduced from a clean clone by two sessions
independently.

Note JPEG XS is patented (GOALS, docs/POSITIONING.md) — this is a comparison, not a target to
adopt.

### ENT-4 — Re-run MEAS-9 with `--abac` (**DONE 2026-09-07** — abac closes half the JPEG 2000 gap)

**Result: −16.01% of rate at identical pixels (24/24 rungs bit-identical), and it closes exactly
half the RGB gap to JPEG 2000 — 54.2% → 27.1%.** Full numbers in RESEARCH_LOG 2026-09-07.

| arm | RGB, Rice | RGB, abac | Y, Rice | Y, abac |
|---|---|---|---|---|
| J2K 9/7 | +54.2% | **+27.1%** | +79.7% | **+48.3%** |
| ProRes 4444 | +20.2% | **+1.3%** | +29.3% | **+9.1%** |
| JPEG XS 4:4:4 | −10.2% | **−25.8%** | +29.4% | **+7.7%** |

With abac GNC matches ProRes 4444 and beats JPEG XS 4:4:4 by 25.8% on RGB PSNR; it stays behind
both on Y-PSNR, because an entropy coder does not move bits between planes and the luma/chroma
allocation difference MEAS-9 measured is untouched. The saving decays with rate (−18.0% at q=60 to
−10.7% at q=99), so **any single abac figure must name its q**. The q=90 mean is −17.32% against
ABAC-SHIP's independently measured −17.3%, which is the best cross-check either number has.

**Two recorded figures are corrected by this.** The EBCOT scoping said "~9% mean … roughly a third
of the +28.3% intra gap to JPEG 2000". The +28.3% was measured against **OpenJPEG's default
reversible 5/3** — provable, not suspected: the 2026-09-05 J2K ladder reads 3.00 bpp at 41.89 dB and
4.80 at 45.55 on bbb, which today's `J2K 5/3rev` arm reproduces exactly, while `-I` gives 43.85 and
48.58 dB at the same rates. And the offline model understated the coder by 1.7x (−9.2% modelled
against −16.0% in-codec), the same direction as the 3-to-4 wavelet levels case. Gap bigger, coder
better, "a third" → **a half**.

**Successor, and it is now the sharpest open question in intra: where is the remaining +27.1%?**
The offline work put EBCOT's full-neighbourhood context at −16.4% against abac's vertical-only
−11.7%, so a richer context model is worth single digits and costs the 256-way parallel decode —
nowhere near 27 points. The rest must be deadzone/quantisation detail, subband weighting, or
code-block geometry. **Nothing in the record accounts for it.**

**One flag on the GNC arm, ~25 minutes, and it replaces an extrapolation with a measurement.**
MEAS-9 found that JPEG 2000 in irreversible 9/7 mode — GNC's own transform at GNC's own depth —
needs **54.2% fewer bits on RGB PSNR and 79.7% on Y-PSNR**. ABAC-SHIP measured abac at −17.3% at
q=90, and RESEARCH_LOG currently says that is "about a third of the gap". **That third is an
extrapolation from a single quality point, not a BD-rate**, and it is the number the entropy-coding
priority rests on. Measure it:

```bash
"$(git rev-parse --show-toplevel)/.venv/bin/python" scripts/meas9_contribution.py \
    --images test_material/frames/{bbb_1080p,blue_sky_1080p,kristensara_720p,touchdown_1080p}.png \
    --arms gnc,jpegxs,prores444,j2k --csv meas9_abac.csv
```

The GNC arm needs `--abac` added to its `encode` invocation in `arm_gnc()` — a two-line change, or
an `--gnc-extra-args` option if it should be selectable. Everything else is already in place: the
harness validates itself (`--selftest`), the incumbent arms are deterministic and their published
rows can be reused, and the ceilings and canaries are unchanged.

**Success criteria, stated in advance.** abac is *lossless* re-coding of the same coefficients, so
pixels must be identical to the Rice run at every q — check that first, because if quality moves at
all the arm is measuring something other than the entropy coder. Then the BD-rate against J2K 9/7
should fall from +54.2% by roughly the measured rate saving. If it falls by much less, the −17.3%
does not hold across the ladder; if by much more, suspect the harness before celebrating.

**Why it matters beyond one number:** it says how much of the intra gap is entropy coding and how
much is left for coefficient modelling, which is the difference between "keep going down this road"
and "the remaining gap is somewhere else". Decision 0018 makes that the leading question, since
entropy coding is the one lever that pays on intra, inter, lossless and every chroma format at
once.

### ENT-3 — Does abac pay on inter residuals? (todo, P1, claimed and released unmeasured 2026-09-07)

**The inter half of the entropy question, which ABAC-SHIP explicitly left out of scope** ("intra
only; inter is out of scope for this row"). Nothing has been measured. Claimed at the end of the
2026-09-07 session and released without a single number, so this entry is the hypothesis, not a
result.

**The question:** abac against Rice on **P-frame residual coefficients**, same pixels, same GOP
structure, at q=75 and q=90 on ≥3 sequences. `GNC_ABAC_COMPARE=1` already reports rate on real
coefficients and was widened to see non-wavelet paths, so the instrument may need little work.

**Expect a smaller number than intra's −17%, and treat that as an answer rather than a
disappointment.** After motion compensation the residual is noise-like at high quality (Girod), so
there is less statistical structure for a context-adaptive coder to exploit than in an intra
subband. Two outcomes, both useful:

- **It pays comparably.** Then entropy coding is the whole entropy story, intra and inter, and the
  case for it is stronger than MEAS-9 alone makes it.
- **It pays much less.** Then the inter gap against H.264 lives in the *motion model* — subpel,
  block partitioning, OBMC — and not in the coder. That contradicts the standing assumption that
  entropy coding is the biggest inter gap, which is worth knowing before more effort goes there.

**Do not skip the domain declaration.** State whether the measurement operates on quantised
wavelet coefficients of the motion-compensated residual, or on something else, and check that the
implementation matches — five conclusions in this repo turned on that exact question.

Note the 2026-09-06 finding that inter breaks even at contribution quality is correct *at that
operating point* and must not be read as "inter does not matter": under decision 0018 inter has to
work across the whole range, and the low end is where inter earns its keep.

### ENT-2 — Rice vs rANS on one commit (**DONE 2026-09-07**)

Measured, four stills, one commit (`c0dd27f`; no `src/` change through `edf56bc`). Harness
`scripts/ent2_rice_vs_rans.py`, images pinned with SHA-256 in `frames_pinned/`. rANS against Rice,
negative meaning rANS is smaller:

| q | bbb | blue_sky | kristensara | touchdown | mean |
|---|---|---|---|---|---|
| 5 | −9.0% | −0.8% | +11.1% | −3.5% | −0.5% |
| 10 | −11.1% | −7.3% | +3.1% | −10.4% | **−6.4%** |
| 15 | −10.4% | −7.4% | +0.6% | −11.1% | **−7.1%** |
| 20 | −9.0% | −7.4% | +0.8% | −11.2% | **−6.7%** |
| 25 | −3.3% | +0.1% | +8.2% | −3.5% | +0.4% |
| 40 | −0.9% | +1.3% | +5.3% | −5.4% | +0.1% |
| 55 | +0.5% | +1.6% | +4.1% | −5.9% | +0.1% |
| 70 | +1.4% | +1.8% | −0.1% | −5.6% | −0.6% |
| ≥77 | crash | crash | crash | crash | — |

**The prediction in decision record 0015 is falsified.** It said Rice still wins and by more than
the withdrawn 4.01-vs-4.22 pair suggested. Above q=25 the coders are level on the mean; below q=20
rANS is 6–7% smaller. Decision record 0018 records the correction.

**No default moves.** Rice stays above q=20 on the parallelism argument, which no rate figure was
ever going to change; rANS stays at q≤20, where the mean supports it. What changed is the
documentation: the `--help` text in five subcommands (it called rANS the default and Rice "~30%
worse compression"), the README's Entropy Coders table (rate column restored, unsupported
"1.5–2× faster" removed), and `quality_preset`'s justification comment.

Three findings worth carrying:

- **`--rans` is not a no-op.** BUG-9's entry records it as one. `src/main.rs:1018` sets
  `EntropyCoder::Rans`, `encode` defaults to 4:4:4 so `normalize_for_chroma()` does not revert it,
  and the harness confirms it at bitstream level — it parses `entropy_type` out of the GP17 header,
  and 40 of 40 points carry the requested coder. Only the flag's help text was stale.
- **The q=20 cutoff is where `default_levels` goes 4 → 5**, and every image jumps in the same
  direction across it. rANS pays a frequency table per subband group; Rice adapts k per subband
  nearly free. The two constants must move together if either moves.
- **rANS's ceiling is q=75, not q=78.** q=75 encodes on all four images, q=77 fails on all four,
  q=76 splits by content. Handed to the BUG-9 owner; that entry's text is theirs to correct.

**Left open: the throughput half.** TUNE-3's ~8% encode / ~15% decode is quoted, not re-measured —
COORDINATION rule 1 forbids timing while other sessions build. Worth one idle-machine run, which
would also settle whether the removed "1.5–2×" was ever true.

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
