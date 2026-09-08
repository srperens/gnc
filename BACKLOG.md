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

At the contribution operating point GNC needs **+89.2% BD-rate on PSNR** against x264 — about
1.9x — not the 5.6x recorded from MEAS-1, which was measured at distribution bitrates with the
quality ladder above q=92 dead. (QUAL-1's +90.5% was the same ladder on 2026-09-06; MEAS-10
re-took it after INTER-2.) Nothing in the coder changed between the two measurements. This
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

0. ~~**INTRA-1 — where is the remaining 27%?**~~ — **ANSWERED 2026-09-08 by step 3, and closed as
   a queue entry.** 26.2 of the 27.1 points are named and **15.1 of them are not coding
   deficiencies** (8.5 chroma allocation, 6.6 tile padding), so the intra coding gap on these four
   images is closer to **+12%** than +27%. The remaining work is **PAD-1 (P1)** — the padding fill,
   −4.5% of shipped intra rate, gated on inter — and **INTRA-2 (P1)** — the dead zone on I-frames
   only, ~3 points. History below. Added 2026-09-07 after ENT-4. With `--abac` on, GNC
   needs 27.1% more bits than JPEG 2000 *using the same transform at the same depth*, and nothing in
   this repository accounts for it. Largest known compression gap. **Step 1 is done (2026-09-07):
   GNC spends within 7.5% of the entropy of its own coefficients at q >= 85, so entropy coding can
   account for at most ~7.5 of the 27.1 points and ~72% is upstream of the coder.** The item is now
   its step-2 branch — whole-frame transform against 256px tiles, quantiser shape and per-subband
   step, lifting normalisation, code-block geometry. Decision `docs/decisions/0024`.
   **Step 2, first instalment done (2026-09-07): the gap decomposes.** 8.5 points is chroma
   allocation against an RGB metric (YCoCg-R's synthesis norms, not a coding deficiency), ≤7.5 is
   the entropy coder, lifting normalisation is clean (~0), and tiling costs *JPEG 2000* 12.4 points
   but GNC realises only 0.6% of it. **~9.8 points remain**, and the open question is why a bigger
   tile buys GNC nothing when it buys J2K 3.8%. Decision `docs/decisions/0026`.
   **Step 2b done (2026-09-07): cross-tile rate allocation is rejected — a clairvoyant, free
   per-tile allocator saves 0.95%**, about one point of the remaining ten, and on one image the
   oracle picks the *same* q for every tile at q>=92. The obvious candidates are now spent;
   ~9 points remain. Decision `docs/decisions/0027`.
   **Step 2c done (2026-09-08): both remaining candidates settled.** Tile-boundary handling is
   already correct symmetric extension (0 points, and it corrects a wrong claim in 0027); the dead
   zone is worth **~3 points on stills** but is a **worst-frame regression on 9 of 9 sequence
   points**, so it is filed as **INTRA-2** rather than shipped. **~6 points remain, and every
   candidate the item listed has now been measured.** Decision `docs/decisions/0028`.
   **Step 3 done (2026-09-08): the remaining ~6 points are tile-alignment padding, and the
   accounting closes.** GNC pads every plane to whole tiles with edge replication and codes the
   padded plane, so a 1920x1080 frame is coded as **2048x1280 — 20.9% of the coded samples outside
   the picture**, while J2K in whole-picture mode codes none, and both arms are divided by the
   visible pixel count. Two methods agree: **+6.60%** projected from a content-controlled crop pair
   (quality held to 0.005 dB), **+6.70 points** as the drop in the cross-codec gap from native to
   padding-free content on one ladder (26.54 → 19.83 RGB, 51.17 → 40.66 Y) — means agreeing to
   0.10 points, per-image figures scattering ±2.2 with the content change. Like the chroma 8.5,
   this is **not** a coding deficiency —
   but unlike it, **two thirds is recoverable: a fill change is worth −4.5% of shipped intra rate**
   at unchanged visible quality, filed as **PAD-1**. Decision `docs/decisions/0034`.
1. **Intra at contribution quality** — the whole remaining +89.2% lives here, per findings 1 and 5.
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
5. Bugs: BUG-14. BUG-9 closed 2026-09-07 (the cause was the cumfreq table, not the slot), BUG-12
   closed 2026-09-06. **ARCH-3 and BUG-18 closed 2026-09-07** — one P/B frame encoder, so
   `gpu_entropy_encode` chooses only where entropy runs. Default output byte-identical (54/54);
   **abac's inter rate is measurable again at −12.0% to −22.9% against Rice at bit-identical
   pixels**, replacing the retracted −14.4%. `docs/decisions/0025`.

**Do not re-test** (measured and closed this week): MCTF, GOP length, the B-pyramid at contribution
quality, RD decisions, multi-reference, sub-pel filters, motion search, block transforms, sub-block
masking, smaller tiles, prediction *before* the wavelet.

## Active priority list

### DOC-2 — GOALS's target table was stale on the rows that set every session's priority (**FIXED 2026-09-08**)

**The internal source of truth had drifted while the public one stayed current**, which is the
dangerous direction: CLAUDE.md points every session at GOALS for priorities, and README — fixed by
DOC-1 — was already right about all of it.

**Four rows and one sentence, all corrected:**

| what GOALS said | what is true |
|---|---|
| "8-bit only (10-bit not implemented) — **the main format gap** for broadcast contribution" | **FMT-1 shipped 10-bit on 2026-09-06.** Verified here end to end on a genuine 10-bit 1080p source: **q=100 is bit-exact, max error 0 over 6 220 800 samples**, and q=90 reads 61.33 dB. FMT-1 had called bit depth *"the first-order problem"*, so this was the most misleading sentence in the file |
| Target table: "Latency per frame — **never measured**" | MEAS-6 measured it twice; **~80 ms round trip, 0 frames of reordering**, below the low-latency-HEVC band. `docs/decisions/0033` |
| "Bit depth: 8-bit → 10-bit, in the format from the start" | both ship; the target is met and the row now says so |
| "Compression (intra): +46–55% vs H.264 all-I (VMAF)" | presented uncaveated, while **BASELINE says that figure predates the high-q ladder fix and has not been re-run** — and it is a VMAF BD-rate, which BASELINE forbids quoting at the contribution end one line earlier. Now carries its caveats plus INTRA-1's better framing: the intra *coding* gap is nearer **+12%** than +27% |
| "The two metrics at the top of that table have never been measured … **They come before further compression work**" | the operative sentence, and it had stopped meaning anything: one metric is measured, the other is **parked on hardware, not on effort**. Restated so the ordering is actionable |

**Also retitled ENT-3**, whose heading asked a question its own body answers with nine measured
points. Not closed — q=95-99 and context retuning are genuinely open, and the frame mix behind the
table is unrecorded.

**Documentation only.** No code, no shader, no bitstream. The one measurement taken was the 10-bit
verification above, which confirms an existing claim rather than making a new one.

### DOC-1 — five stale prose claims in the public README (**DONE 2026-09-08**)

Line numbers below are where each was **found**, before the edit shifted them; the two live pointers (`README.md:192`, `abac_gpu_encode.rs:381`) are post-edit and were re-checked.

Found by auditing the top-level docs after BUG-29 and PERF-1 both swept them. The *numbers* were
current — PERF-1 had just retired 31.7 fps everywhere and BUG-29 had replaced "M1" with "M5 Pro"
in every labelled figure. **Every one of the five was prose that no measurement pointed at**,
which is the same category BUG-29 itself was: a sentence nobody re-reads because it carries no
number to check.

1. **`README.md:13` — "in real time at 1080p on an eight-core integrated GPU".** Both halves
   wrong. The core count is the hardware label BUG-29 retired; "real time" contradicts line 97 of
   the same file (5.0 fps end to end). BUG-29 fixed the two places that named the chip and missed
   this one *because* it names no figure.
2. **`README.md:5` and `:19` — "Cross-platform: Metal, Vulkan, DX12, WebGPU/WASM" / "Runs on
   Metal, Vulkan, DX12 and WebGPU", both stated as fact.** Of the four: Metal is measured; Vulkan
   runs intra only (BUG-25's `block_match_split` crash makes P/B unreachable); DX12 has never been
   run at all; WASM has never been verified in a browser and BUG-31 says a conformant WebGPU would
   fail every decode. Replaced with a **Portability, as measured** section giving the four rows and
   their evidence, plus the cross-backend bit-exactness result. This is the one that mattered: it
   was the public README asserting support the repository's own open bugs contradict.
3. **`README.md:19` — "three interchangeable entropy coders"** against line 192's "five entropy
   coding backends", in the same file. Now: five implemented, three selectable.
4. **`README.md:46` — "Where the remaining 27% lives is not yet known."** INTRA-1 decomposed it
   the same evening (`0026`–`0028`): entropy ≤7.5, chroma allocation 8.5 and not a deficiency,
   tiling 0.6%, cross-tile 0.95%, tile-boundary 0, dead zone ~3 intra-only. ~6 points remain.
5. **`docs/BITSTREAM_SPEC.md` §2.6** listed "the encoder, the CPU decoder and the GPU decoder" as
   the callers of `abac_tile::code_blocks`. ENT-5 added a fourth. Checked rather than assumed:
   `abac_gpu_encode.rs:381` calls `code_blocks` on the host and uploads the geometry, so the
   single-definition property the paragraph exists to assert is still true — the shader does not
   re-derive it. Sentence corrected and the reason recorded, since "a shader calls a Rust function"
   is exactly the claim a reader would doubt.

Documentation only; no code, no shader and no bitstream touched, so no measurement moves and the
GPU test suite was deliberately not run — it would have taken the GPU from seven sessions to prove
something a `git diff --stat` of two `.md` files already proves.

**No decision record.** Nothing was chosen; five claims were checked against evidence that already
existed and four of them lost. The one judgement call — softening the portability claim rather than
deleting it — is argued in the section itself.

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


### BUG-25 — `block_match_split.wgsl` killed two Vulkan drivers (**FIXED** 2026-09-08)

**FIXED, and it was fixed by `51a9ac6` — the defect-A commit — before anyone noticed.** Measured at
`766196a` on the bench box: `gnc encode-sequence` codes **1I + 2P on NVIDIA/Vulkan** (102244 /
30301 / 27563 bytes, 180.1 ms), `decode-sequence` round-trips it, and **lavapipe produces
byte-identical frame sizes**. `shader_probe` compiles **62 of 63** shaders through wgpu's real WGSL
path — the one failure is `blit.wgsl`, which has no `@compute` entry point, so the probe cannot
build a compute pipeline from it.

**All four recorded driver crashes were the invalid module.** NVIDIA/Linux and lavapipe were built
from `07c01b1`; the Windows NVIDIA and Intel Arc crashes, which read as independent confirmation
because they were *committed after* the fix, were built from **`f17bf1b` — which predates
`51a9ac6`** by 66 deleted lines of this shader. Four signatures, one cause. And lavapipe's
`Parent device is lost` is verbatim what upstream `#7198` reports for the same naga defect.

**Defect B is withdrawn, not fixed, because GNC never reached it.** `buffer: Restrict` does segfault
NVIDIA — verified again today — but wgpu asks for `buffer: Unchecked` on any adapter reporting
`robustBufferAccess2`, and both adapters here report it. The stale datum that held this up was "the
real WGSL path crashes": true when written, one commit out of date when the elimination argument
used it. `docs/decisions/0029`, RESEARCH_LOG 2026-09-08, and **BUG-33 closed**.

**Not re-measured, and not claimed: Intel Arc Pro and Windows NVIDIA.** Their crash was on the
invalid module too, so the expectation is that they are fine — an expectation, not a measurement.

**Owed:** an inter throughput figure on Vulkan, now measurable for the first time. Today's runs are
384x256 and 3 frames — correctness tests on a machine carrying other work, not a benchmark.

Older text follows, kept because the withdrawal is the point.


**Correction 2026-09-08 — the second one on this item. Defect B's *cause* is withdrawn; its
*measurements* are not.** Every driver result below was measured and stands. What is withdrawn is
the attribution. `wgpu-hal` 24.0.4's own rule is `buffer: robust_buffer_access2 ? Unchecked :
Restrict` (`adapter.rs:1899`), the cap is read from a **queried** `VK_EXT_robustness2`
(`adapter.rs:1595`, `:1372`), and **both** Vulkan implementations on the bench box report the
feature — the RTX per `vulkaninfo`, lavapipe since Mesa 22.2. So wgpu should be shipping
`buffer: Unchecked` here. Measured on the dev machine with no GPU: the faithful reconstruction of
that configuration (`caps_index_restrict`) is **byte-identical** to `index_restrict_only`, which was
measured **pipeline OK**, and it contains **zero `OpArrayLength`** where every crashing
configuration contains **48**. `docs/bug25/minimal_repro.spvasm` is nothing but an `OpArrayLength`
clamp, so on that reading it is **not a reduction of GNC's crash**. See **BUG-33** (rewritten) for
the four propositions and the single run that settles it, and RESEARCH_LOG 2026-09-08 for the
numbers. Also established there: **defect A is upstream `gfx-rs/wgpu#7048`, closed by PR #7239** —
our local `switch` rewrite duplicates a fix that already exists, which is worth knowing the next
time "upgrade wgpu" is priced.


**The premise this item was built on was wrong, and correcting it found two defects.** The item
said `spirv-val` passes all 62 shaders, so two unrelated drivers were dying on valid SPIR-V. That
validation was run with the `naga` **CLI at 30.0.1**; GNC ships **naga 24.0.0** via wgpu 24. The
module that was validated is not the module that reaches the driver. Rule that follows and outlives
this bug: **validate the artefact you ship, with the compiler you ship** — a tool on `PATH` is not
the one in `Cargo.lock`.

**Defect A — GNC shipped invalid SPIR-V. FIXED.** Under wgpu's options, naga 24 emits
`OpStore %1214` and `OpAccessChain … %1214` for a function-local temporary it never declares. The
temporary is how it dynamically indexes a **value-typed constant array**; the source is four
`let hpel_dx = array<i32, 8>(…)` / `qpel_*` tables indexed by a loop counter, and **only this
shader has them (4 occurrences; both siblings have 0)**. Swept the tree: **1 of 63 invalid before,
0 of 63 after.** Fixed by writing the 8-point diamond as a `switch`, which is how the earlier
half-pel search in the same file was already written; Metal output is **byte-identical**
(`d07cd62d6ab4da43`), because the offsets map one-to-one.

This also explains four older results at once: why deleting the quarter-pel section fixed it (the
arrays are there), why "8 candidates → 4" still crashed (still a dynamic index), why H1 compiled in
isolation (naga 24 gets it right in a small module), and why size/barrier/workgroup-variable counts
never discriminated.

**Defect B — the crash. NOT fixed, and it is not defect A.** The now-valid module still segfaults.
Emitting under each wgpu option separately isolates the trigger to
**`BoundsCheckPolicy::Restrict`**, which wgpu requests unconditionally for array indices:

| configuration | valid | driver |
|---|---|---|
| `bounds_unchecked` | yes | pipeline OK |
| naga default bounds | yes | pipeline OK |
| `bounds_restrict` | yes | **CRASH** |
| `wgpu_native` / `wgpu_polyfill` | yes | **CRASH** |

Debug names, `lang_version` and the workgroup zero-init mode make no difference, and removing the
four arrays entirely leaves the crash untouched. **Do not close this item on defect A.**

**Defect B is now characterised, down to 42 lines.** `spirv-reduce` took the valid crashing module
from 41 068 bytes to **524**, and the result is in `docs/bug25/` with instructions to reassemble
and reproduce:

```spirv
OpSelectionMerge %1638 None
OpBranchConditional %1897 %1553 %1554
%1553 = OpLabel
%1596 = OpArrayLength %uint %32 0     ; runtime-array length
%1597 = OpISub %uint %1596 %uint_1    ; length - 1
        OpBranch %1639
%1639 = OpLabel
        OpReturn                      ; returns instead of reaching the merge
%1638 = OpLabel                       ; the merge block, unreached
        OpReturn
```

`OpArrayLength` + `OpISub 1` **is** naga's `Restrict` storage-buffer check, `min(i, len - 1)`, and
it sits in a selection branch that returns rather than reaching its merge block. Splitting the
policy confirms which half:

| policy | driver |
|---|---|
| **`buffer: Restrict`**, index Unchecked | **CRASH** |
| `index: Restrict`, buffer Unchecked | pipeline OK |
| all Unchecked | pipeline OK |

**Correction to how that was first written.** This entry originally said wgpu requests
`buffer: Restrict` "whenever the adapter does not report `robustBufferAccess2`" — but this adapter
*does* report it (`vulkaninfo`: `robustBufferAccess2 = true`, extension present), so that reading of
`wgpu-hal` does not explain the crash. **The conclusion survives on different evidence: elimination.**
The real WGSL path crashes; a faithful reconstruction with `buffer: Unchecked` does **not** crash —
with wgpu's real writer flags and with its capability list — and one with `buffer: Restrict` does.
So the shipped module must carry `Restrict` on buffers. *Why* wgpu selects it here, given the
device reports the feature, is unexplained and is the next thing to find out, because it decides
whether the fix is upstream or a local patch.

**Also ruled out while reducing:** not driver stack exhaustion (reproduces at `ulimit -s` 8 MB,
64 MB and unlimited); and the full-size valid module segfaults **Mesa lavapipe too**, so the
"two independent compilers" argument is restored for defect B — it was simply measured on an
invalid module before. The 524-byte file is minimal *for NVIDIA only*, because the reduction's
oracle ran the default adapter; reduce again against lavapipe if a two-implementation reproducer
is wanted.

### Four candidate fixes tested. Three are dead; one is proven.

| candidate | result |
|---|---|
| **Upgrade wgpu/naga** | **DEAD.** naga **30** under wgpu's *identical* options crashes both drivers. The earlier naga-30 module that built a pipeline was emitted with the CLI's own bounds policy, so it compared two things at once and settled nothing. `examples/bug25_emit30.rs` holds the options fixed and moves only the compiler version. |
| **Remove the `let`-array construct** | **DEAD** for the crash. It fixed the *validity* defect and nothing else. |
| **Remove the early `return`** | **DEAD.** Deleting it outright still crashes, so the returning branch in the reduced module is an artefact of reduction, not the trigger. |
| **`buffer: Unchecked`** | **WORKS.** Compiles and builds a pipeline, under wgpu's real flags and capability list. This is the only proven fix. |

**So the fix is to stop wgpu asking for `Restrict` on buffers**, and the open question is why it
asks on an adapter that reports `robustBufferAccess2 = true`. If that is a wgpu defect the fix is
upstream and small; if it is deliberate, GNC needs a `[patch.crates-io]` pin. Either way the shader
does not need to change, which is worth knowing before anyone rewrites it.

It is also a legitimate driver bug report in its own right — a valid module should be rejected or
compiled, never segfault the compiler — and worth filing against both NVIDIA and Mesa, since the
full-size module takes down lavapipe too.

**Instruments, all committed:** `examples/spirv_probe.rs` (module shape, no GPU),
`examples/bug25_emit.rs` (emit under wgpu's options, one knob at a time),
`examples/spirv_pipeline_probe.rs` (one pipeline from raw SPIR-V, exit codes inverted for a
reducer), `scripts/bug25_interesting.sh` (the oracle, verified in both directions before use —
a reduction whose oracle accepts modules the driver never saw converges on garbage).

<details><summary>Superseded framing (kept: five of its hypotheses are still valid negatives)</summary>


**Status: the blast radius is fixed, the shader is not.** `split_pipeline` is now built on first
dispatch instead of in `MotionEstimator::new`, so intra encode, decode and CANARY-1 all work on
Vulkan. Dropped from P0 to P1 because nothing is blocked on it any more except MEAS-5. Full
numbers in RESEARCH_LOG 2026-09-07.

**Corrected 2026-09-07: inter on Vulkan does not die, it hangs.** This entry said P-frames "still
compile the shader on every P-frame and still dies", which was written from reading the call sites
rather than from running it. Measured on the box: `encode-sequence`, 4 frames, ki=2,
Vulkan/NVIDIA, sat for 3 minutes at **0.00 CPU time** with no output file and had to be killed,
while intra in the same run completed normally. **A zero-CPU hang is a different symptom from a
SIGSEGV and may be a different cause** — a lost device that wgpu absorbed and then a fence that
never signals, or something upstream of the shader entirely. Characterise it with one run before
assuming the P-frame path fails for the same reason pipeline creation did.

**Five hypotheses were tested and all five are wrong. Do not re-run these:**

| # | hypothesis | result |
|---|---|---|
| H1 | dynamic index into a `let`-declared array value | compiles in isolation |
| H3 | `workgroupBarrier()` in a function called inside a loop | compiles in isolation |
| H4 | nine workgroup variables live across many barriers | compiles in isolation |
| E2 | loop unrolling (all 6 refinement bounds made opaque) | **still segfaults** |
| E5 | iteration count (8 candidates → 4) | **still segfaults** |
| H6 | naga duplicates a barrier-carrying block, so invocations reach *different* `OpControlBarrier` instructions | **dead — measured, not argued** |
| H7 | the count of `var<workgroup>` (nine, against four in both siblings) | **dead — two shaders with 10 and 11 compile** |

**H6 and H7 were killed on the Mac, with no Vulkan and no remote box**, by
`examples/spirv_probe.rs` — it runs the same naga 24 that wgpu resolves, over all 64 shaders, and
reports the shape of the SPIR-V that would have been shipped. Numbers:

| shader | wgsl barriers | spv | in conditional blocks | blocks | merge depth | `var<workgroup>` | Vulkan |
|---|---|---|---|---|---|---|---|
| **block_match_split** | 29 | 30 | 18 | 268 | 6 | **9** | **dies** |
| block_match_bidir | 35 | 36 | **24** | 343 | 6 | 4 | OK |
| block_match | 18 | 19 | 12 | 188 | 6 | 4 | OK |
| quantize_histogram_fused | 18 | 19 | 11 | **389** | **7** | **11** | OK |
| rans_histogram | 15 | 16 | 8 | 300 | 7 | **10** | OK |

**H6:** SPIR-V barrier count is source + 1 in *every* shader, so naga duplicates nothing; and
barriers in conditionally-reached blocks do not discriminate — the two siblings that compile have
24 and 12 against the offender's 18. **H7:** nine workgroup variables is not even the maximum;
`quantize_histogram_fused` declares eleven and `rans_histogram` ten, and both compile on Vulkan.
The entry-point interface list is also identical in shape across working and failing shaders.

**So size, block count, merge depth, barrier count, barrier placement, workgroup-variable count
and interface shape are all eliminated** — across the whole shader set, not by pairwise
comparison. What survives is the quarter-pel section (E1 showed it is *required* for the crash)
and the specific instruction sequence naga emits for it. That is where the SPIR-V reduction has to
cut.

**Do not assume you need the hardware.** Most of what has been eliminated so far was eliminated
without it — `spirv_probe` reads the module that would have shipped, so hypotheses about naga's
output, the shape of the SPIR-V, limits and shader structure are all answerable on any machine.
What the box is needed for is the final step only: deciding whether a given *cut* still crashes.
Exhaust the offline questions first; box time is contended and the WGSL round that preceded this
spent five hypotheses' worth of it on questions that did not need it.

Also ruled out: size and barrier count. `block_match_split` is 806 lines / 17 loops / 31 barriers;
`block_match_bidir.wgsl` is 741 / 22 / 35 and compiles. Removing the quarter-pel section (E1) makes
it compile, so that section is *required* for the crash — but the WGSL-level bisect can only delete
whole statements, and four truncated variants were rejected by naga rather than crashing and were
counted as passes, so lines 666–697 are where the crash *appears*, not where it is proven to be.

**The next step is a SPIR-V-level reduction**, not more WGSL guessing: `spirv-dis` the module and cut
it down there. The one solid clue is that **two compilers sharing no code — NVIDIA's and Mesa
lavapipe's — both die on SPIR-V that `spirv-val` passes**, which points at the shape of naga's
output for this shader rather than at either driver.



Found 2026-09-07, the first time this project ever ran on non-Apple hardware. Full measurement in
RESEARCH_LOG 2026-09-07.

**GOALS rule 4, the README and the positioning all claim Metal, Vulkan, DX12 and WebGPU. On the
only non-Metal hardware GNC has ever been run on, it does not start.** Portability is the one axis
GNC is meant to win on outright (GOALS §1), so this is not a compatibility nit.

**Machine:** Ubuntu 24.04.3, kernel 6.8.0-136, NVIDIA RTX 4000 Ada (20 GB), driver 580.173.02,
Mesa lavapipe on LLVM 20.1.2 as a second Vulkan implementation. Built from `07c01b1`, cargo 1.97.1,
**clean release build, zero warnings, 1m35s.** wgpu 24.0.5. Input the pinned `bbb_1080p.png`
(`f83f355f…02bf`).

| backend / adapter | result |
|---|---|
| Vulkan, RTX 4000 Ada | **SIGSEGV** (139), no Rust panic |
| Vulkan, lavapipe | `Parent device is lost` |
| GL, RTX 4000 Ada | no compute support — a real backend limitation, not this defect |

**The offender is `src/shaders/block_match_split.wgsl`, and it was found by bisect.** A throwaway
probe that creates a device and **one** compute pipeline per process, run over all 62 WGSL files:
60 pass, `block_match_split` segfaults NVIDIA and loses the device on lavapipe. (`blit.wgsl` also
"failed" — it is vertex/fragment only and that was the probe's own artefact, recorded so a 1-of-62
result does not later get quoted as 2-of-62.)

Do not trust the label in the wgpu error. lavapipe reports `Parent device is lost` against
whichever pipeline is created *after* the loss, which is why three earlier readings blamed
`block_match_split_pipeline` for a device that a previous call had already killed — the label was
right by coincidence, not by evidence.

**The SPIR-V is valid, which is the interesting part.** naga converts all 62 shaders without
complaint and **`spirv-val` passes all 62**. Mesa's software Vulkan and NVIDIA's proprietary driver
share no compiler code, and the same single shader kills both while its sibling
`block_match.wgsl` compiles fine. Two independent compilers dying on the same valid input is weak
evidence of two driver bugs and strong evidence that something in this shader, or in naga's codegen
for it, is outside what implementations handle.

**Size and complexity are ruled out:**

| shader | lines | loops | barriers | `var<workgroup>` | Vulkan |
|---|---|---|---|---|---|
| **block_match_split** | 806 | 17 | 31 | **9** | **dies on both** |
| block_match_bidir | 741 | 22 | 35 | 4 | OK |
| block_match | 448 | 14 | 19 | 4 | OK |

`block_match_bidir` has more loops and more barriers and compiles. Workgroup *memory* is ~2.1 KB,
far under any limit. What is left is the count of workgroup variables, or a barrier reached under
non-uniform control flow — which WGSL forbids and naga does not fully diagnose. **Hypotheses, not
findings.** Bisect the shader.

</details>

### Two separable pieces of work — the second is **done**, the first is what remains

1. **The shader.** Cut `block_match_split.wgsl` down until it compiles, name the construct, fix it,
   and add the case to whatever guards it afterwards. If it turns out to be a driver bug on valid
   SPIR-V, the workaround still belongs in GNC — "the driver is wrong" does not make the codec run.
2. **Eager pipeline creation, which is our own doing and is the larger defect.**
   `block_match_split` is variable-block-size motion estimation: encoder-only, inter-only, never
   dispatched by a still-image encode. It is created **unconditionally in `MotionEstimator::new`**
   (`src/encoder/motion.rs:456`), so a shader a still never uses stops a still from encoding.
   **One broken shader becomes a dead codec.** Create it lazily, or behind the condition that
   dispatches it, and intra encode and decode start working on Vulkan while piece 1 is diagnosed.

   **What that does and does not unblock, checked rather than assumed.** `estimate_split` is called
   only from `sequence.rs`, never from `pipeline.rs`, so the still path never touches it — and
   `gpu_tier_bench.py` runs `--tier` through `gnc benchmark` on a single image but `--density`
   through `benchmark-sequence` on a clip. So **lazy creation unblocks CANARY-1 and leaves MEAS-5
   blocked**: both `estimate_split` call sites are unconditional inside the P-frame path, so every
   P-frame compiles the shader. (An all-intra density sweep at ki=1 would emit no P-frames and would
   run — but it measures intra concurrency, not the shipped configuration, and must be labelled that
   way if anyone quotes it.) CANARY-1 is the right first measurement regardless: until it passes, no
   throughput number from any GPU means anything. This is also the general fix: the next shader that trips a driver should
   cost its own feature, not the product.

**Land the probe first.** `examples/shader_probe.rs` — ~30 lines, device plus one compute pipeline
per process, no GPU work, no test material. It found this in one run and would have caught it the
day the shader landed. It was deliberately **not** committed by the session that filed this, which
was working in the shared checkout without a claim on any code; that is the first thing BUG-25's
owner should bring in.

**What this blocks.** CANARY-1 and MEAS-5, both of which finally have hardware. They are parked as
`linux-nvidia` rather than free, and the reason has changed from "no second GPU" to this bug.
### PERF-1 — verify `docs/SIMPLE_PERF_FIXES.md` and land the fixes that are free (**DONE 2026-09-08**)

**Result (2026-09-08).** 23 of the 24 cited `file:line` sites hold; the one miss is cosmetic
(item 10 cites `quantize.wgsl:170`, the dequant branch is at `:199`). The "already closed" table
checks out against RESEARCH_LOG and the archive. **Five items landed, each as a count, all
bitstream-identical on 24 artefacts:**

| item | before | after |
|---|---|---|
| 1 — `load_frame` calls per display index | 2.62 | **1.00** |
| 3 — `poll(Wait)` per I-frame (q=75 4:4:4) | 3 | **1** |
| 5 — CfL `MAP_READ` buffers per `encode()`, CfL off | 2 | **0** |
| 6 — `queue.submit` per `encode()` | 2 | **1** |
| 7 — decode pack allocation per frame | 1.22 MB zero-filled | **reused** |

Item 5 also found a live defect: the buffers labelled "never used" were mapped and read on the
`cfl_enabled && !use_cfl` path, and two garbage alphas were read back and then discarded. No
bitstream effect, which is why nothing caught it. The 31.7 fps citation is retired from GOALS,
README and POSITIONING. **Item 4 is verified and deliberately not landed — see PERF-2.** Items 2,
8, 9, 10 and 11 were out of scope by construction and are PERF-2/PERF-3.
RESEARCH_LOG 2026-09-08; decision `docs/decisions/0027`.

Filed 2026-09-07. `docs/SIMPLE_PERF_FIXES.md` landed on `main` in `e8a8a45` as a **scan**, and says
so itself: *"Not claimed as a BACKLOG item — this is a scan"*, and *"No throughput number in this
file is a new measurement"*. So it is twelve ranked assertions about host-side waste with no ID, no
priority and no verification — invisible to `scripts/claim next`, and read by every session as if it
were established. This item is what makes it either true or closed.

**Two halves, and the first is the point.** Verify the claims; then fix only the ones that are
mechanical and bitstream-identical. A candidate that survives verification but needs a shader, an
API change or a wall-clock number is *filed*, not done here — see "Not in scope".

**Why P2 and not P1.** Throughput is not the headline gap — the +90.5% BD-rate against x264 is, and
it is intra rate (INTRA-1, priority item 1). But verification costs no GPU at all, most of the fixes
are byte-identical by construction, and the file is currently a standing invitation for the next
session to act on unmeasured prose. Cheap, low-risk, and it stops a bad citation before it spreads.
Not P1 because until step 1 runs, nobody knows which of the twelve are real.

#### Step 1 — verify, and this needs no GPU and no idle machine

Every `file:line`, every "already closed" row against RESEARCH_LOG and `docs/archive/`, and the
three fps quantities against BASELINE's "How to read the fps figures". **Nine sites were spot-checked
against `e8a8a45` while filing this and all nine hold** — `sequence.rs` `|i| frames[i].to_vec()`,
`main.rs` `self.cache[i].clone()`, the scene-cut *"will call load_frame again"* comment, the
`prev_frame_luma` comment that claims "just the R channel" against four assignments that all store
the full interleaved RGB `frame_data`, the CfL `// Dummy buffers (never used)` `MAP_READ` pair,
`rice_decode.wgsl`'s one-byte `load_byte`, the whole-plane `copy_buffer_to_buffer` preamble in
`transform.rs`, and `color.rs`'s per-dispatch `create_buffer_init`. That is a good hit rate, not a
pass: the *quantities* are unverified, and the quantities are what decide whether a fix is worth the
diff.

**One contradiction is already confirmed and is in scope.** The doc says the 31.7 fps figure is not
reproducible and that GOALS still cites it. Both are true, and it is worse than one line:
`GOALS.md:103` and `BASELINE.md:103` say the figure is not reproducible, while `GOALS.md:118`,
`GOALS.md:216` and **`README.md:75`** still headline it. The public README carries a number this
repository has already retracted internally. Fix the three, say which of A/B/C replaces it, or say
"unmeasured on an idle machine" — do not quietly delete it.

#### Step 2 — fix, and only where the bitstream cannot move

Doc items **1** (`Arc<[f32]>`, stop the double `load_frame` after a non-cut, stop storing a second
full RGB copy as a "luma proxy"), **3** (one poll for Rice + MV + AQ + CfL instead of two to four),
**4** (cache uniforms and bind groups the way `GpuRiceEncoder::params_buf` already does), **5**
(drop the dummy CfL `MAP_READ` buffers), **6** (fold the I-frame preprocess submit into the main
encoder), **7** (reuse the decode `pack_decode_data` scratch). Land them as separate commits, not
one — item 4 alone touches a dozen modules and will conflict with anything else in flight.

#### Success criteria — counted, not timed

1. **Byte-identical output.** `cmp` the `.gnv`/`.gnc` files before and after on ≥3 sequences at
   q=75 and q=90. Every item in step 2 is bitstream-identical by construction; if a file moves by one
   byte, that item is a behaviour change and leaves this item.
2. **All three gates green** — `cargo test --release` and both clippy targets.
3. **Counts, from `GNC_PROFILE`, because the machine is shared.** `load_frame` calls = 1 per display
   index on the P-only path (the doc says 2–3 today); zero `create_buffer_init` on the steady-state
   I-frame path; `poll(Wait)` per I-frame down to 1; bytes cloned per frame reported before and after.
   **A candidate whose fix removes no counted copy, alloc or poll is closed with the count, not
   polished** — CLAUDE.md, "know when to stop".
4. **No fps claim without an idle machine**, and it must name which of A (12.2), B (5.6) or C (5.0)
   it is. Eight sessions share this GPU and the counts above are the honest instrument; a wall-clock
   delta taken under load is worth nothing (COORDINATION, and the 1.9x clock-ramp lesson).

**Canary.** The doc specifies its own and they are the right ones: log `load_frame` call count
against display index, and count `create_buffer_init` on the steady-state I-frame path in
`GNC_PROFILE`. Both are counts of the thing being removed, so they cannot read "improved" while the
old path still runs — CLAUDE.md, "no silent features".

#### Not in scope, and say so in the write-up rather than silently skipping

Doc item **2** (packed-u8 YUV upload — changes the encode input API), **8** (32-bit Rice bit window
— a shader change, mechanical but still a shader), **9**/**10**/**11** (plane copies, folding
dequant into the Rice store, fusing interleave/colour/crop/pack — all behind a switch plus an
idle-machine bench, the `GNC_ABAC_CODER` pattern). If step 1 says they hold, file them as PERF-2+
with the verified quantities; do not start them here. Likewise **do not re-propose** anything in the
doc's "Already closed" table — fused wavelet (#29), fused quantize+Rice (#33), the three abac decode
shader opts, `MEGA_KERNEL_PLAN.md`. The checkerboard two-pass Rice (item 12) is a product decision on
a null compression feature, not a cleanup; leave it.

**Coordination.** The worktree and branch already exist — `../gnc-perffix` on `perffix`, carrying
only the doc commit. Reuse it. Item 1 and item 3 touch `sequence.rs` on the P-frame path, which is
where **ARCH-3** and **BUG-18** live: if either is in flight, they land first and this rebases onto
them. Nothing here changes the bitstream or any measurement, so it invalidates no result in
RESEARCH_LOG — which is also why it must be able to prove that, by criterion 1.

### BUG-29 — every throughput figure was labelled with the wrong machine (**FIXED 2026-09-07**)

Found by accident: `examples/shader_probe.rs` prints the adapter it opened, and on the dev Mac it
printed **Apple M5 Pro**. CLAUDE.md's Platform Notes had said *"Apple M1 — 8 GPU cores, ~2.6 TFLOPS
FP32"*, and BASELINE, POSITIONING, GOALS, README and COORDINATION all repeated "M1".

`system_profiler` confirms: **Apple M5 Pro, 20 GPU cores, 18 CPU cores, 64 GB, Metal 4.**

**The figures are not fabricated — they were measured on *something*. What is gone is their
provenance.** When the machine changed is recorded nowhere, so no throughput number in this
repository can be reproduced from its label or compared against another one. Every affected file now
carries that warning instead of a chip name it cannot support. Nothing was re-measured and nothing
was deleted: deciding what is worth re-running is a separate call, and this item deliberately does
not make it.

**The larger finding, which the wrong label was hiding.** `gnc gpu-info` now prints device limits in
two columns, and the gap is the point:

| | adapter has | GNC requests |
|---|---|---|
| workgroup storage | 32768 B | **16384 B** |
| invocations / workgroup | 1024 | **256** |
| storage buffers / stage | 31 | **10** |
| max buffer size | 39813 MiB | 256 MiB |

So *"32KB threadgroup memory, max 1024 threads/workgroup"* was true of the **adapter** and was never
what the shaders could use — GNC asks for wgpu's defaults so the same shaders run under WebGPU
(GOALS rule 4). **GOALS' occupancy argument ("16KB shared memory = 2 workgroups/core") is therefore
about a self-imposed ceiling, not the chip's**, which is a materially different statement from the
one it appeared to make. Whether raising the request is worth losing WebGPU portability is
**unmeasured** and needs a decision record rather than a commit.

**Why it went unnoticed for months, and the fix for that rather than for the text:** nothing in the
tree printed the device or its limits, so no run could contradict the prose. `gnc gpu-info` now
does, and it is the first command `docs/GPU_TIER_TEST.md` tells you to run on a new machine.

**Also corrected in passing:** CLAUDE.md said `src/shaders/*.wgsl` holds 32 shaders; it holds 62.
The count is now simply not asserted. And CLAUDE.md's argument against parallel role-based agents
rested on *"the hardware is one M1 with 8 GPU cores"* — the contention argument survives, the
hardware claim in it does not, so it now says "one machine with one GPU".
### BUG-36 — concurrent `--vmaf` runs scored each other's frames, through a fixed temp filename (**FIXED 2026-09-08**)

Found 2026-09-08 while reading the VMAF path under MEAS-6. Every VMAF call site wrote its
reference and distorted Y4M to a **fixed** name under `std::env::temp_dir()` — `gnc_vmaf_ref.y4m`,
`gnc_ip_vmaf_ref.y4m`, `gnc_bench_vmaf_ref.y4m`, `gnc_rdcurve_vmaf_ref.y4m`, and the
`gnc_j2k_compare` directory. `TMPDIR` is per *user*, not per process, and COORDINATION.md's
working mode is **eight sessions on one machine**. Two concurrent `--vmaf` runs of the same
subcommand therefore opened the same two files and each scored whatever frames won the race.

**Measured, not inferred.** `benchmark-sequence --vmaf`, 9 frames, q=75, ki=9, two sequences whose
serial scores are bit-stable across repeated runs:

| run | old_town_cross | bbb_extended |
|---|---|---|
| serial, twice | 97.39 | 95.91 |
| concurrent, twice | 97.39 | **97.19** (+1.28) |
| concurrent, once | **96.37** (−1.02) | 95.91 |

**Exactly one of the pair is wrong in every concurrent run**, in whichever direction the race
decided. The error is 1.02–1.28 VMAF points against the **>0.5-point move CLAUDE.md calls a
BLOCK** — 2–2.5x the threshold, from nothing but another session existing.

**What makes it the bad kind of bug:** it is silent, and the wrong number is plausible. No error,
no warning, no implausible value — 97.19 reads as a perfectly ordinary score. VMAF is the lead
metric at q≤85 (CLAUDE.md), so this is the project's primary number being quietly replaced by
another session's clip. Nothing in a log distinguishes a contaminated run from a clean one.

**Fixed** by routing all nine sites through `gnc::session_temp_path()`, which stamps the process id
into the filename. Verified with the same canary: **6 of 6** concurrent runs now return the serial
values exactly, against 3 of 6 before. `tests/temp_path_collision.rs` scans `src/` and fails on any
`temp_dir()` outside the helper, so it cannot be reintroduced by writing a literal.

**What this does and does not invalidate.** It cannot be reconstructed after the fact — no run
records whether another session was in its VMAF window — so no specific past result is retracted
here. Two things bound the exposure: the four filenames are per-subcommand, so `benchmark` and
`benchmark-sequence` never collided with each other, and only the *overlap* of two VMAF windows
does damage. Longer windows are the higher risk: `rd-curve --vmaf` scores every quality point in
one process. **Rate figures are immune** — bytes are bytes; this reaches only what VMAF scored.

### BUG-37 — `benchmark-sequence` without `-q` silently codes the B-pyramid (**FIXED 2026-09-08**)

Found 2026-09-08 under MEAS-6. `benchmark-sequence`'s quality argument is `Option<u32>` with **no
default** (`src/main.rs:515`), where `benchmark` (:436), `encode-sequence` (:606) and
`benchmark-suite` (:765) all carry `default_value = "75"`. `build_ip_config` only calls
`quality_preset()` when quality is `Some`, and `quality_preset()` is the *only* place the B-pyramid
veto lives (`src/lib.rs:1021`). `CodecConfig::default()` still has `b_pyramid: true`.

So `benchmark-sequence -k 9` with no `-q` codes **`2I+2P+14B`** — the hierarchical pyramid that two
independent measurements rejected as a default on 2026-09-06 — while the same command with `-q 75`
codes `2I+16P+0B`. Verified both ways on the current build; the `B-pyramid suppressed` canary fires
only in the second. It also silently selects qstep 4.0 and LeGall 5/3 rather than a preset, so two
things move at once.

**Nothing recorded is contaminated:** every harness in `scripts/` passes `-q` — checked all five
that invoke `benchmark-sequence` (`meas1_vs_h264.py`, `gpu_tier_bench.py`, `meas_inter1_ki.py`,
`meas3_sequence_rd.py`, `meas_chroma2.py`). The exposure is interactive use, and the trap is that
the flag named *quality* is also the only thing selecting the *GOP structure*.

**It was five sites, not one.** `main.rs` had five `if let Some(q) { quality_preset(q) } else
{ CodecConfig { …, ..Default::default() } }` constructions — `build_ip_config` (:911) plus the
temporal-wavelet and warmup paths at :1643, :1698 and :2555, and the still-image `Encode` path at
:1042. Four of the five code sequences. So `default_value = "75"` would have closed one instance,
left the mechanism, and left three more already diverging.

**Fixed at the root instead.** `gnc::b_pyramid_enabled()` is now the single statement of the
shipped policy; `quality_preset()` and the new `gnc::manual_config(qstep)` both ask it, and all
five CLI sites go through one or the other. `CodecConfig::default()` is deliberately **unchanged**
at `b_pyramid: true` — `src/encoder/pipeline_tests.rs:87` and `:278` build a `Default` config
precisely to exercise the B-frame path, and flipping it would leave those tests green while
silently testing P-only, which is worse than the bug.

Verified on the command that was broken, `-k 9`, 18 frames:

| invocation | before | after |
|---|---|---|
| no `-q` | `2I+2P+14B`, silent | **`2I+16P+0B`, canary fires** |
| `-q 75` | `2I+16P+0B` | `2I+16P+0B`, **26911589 bytes both times** |
| `GNC_B_PYRAMID=1`, no `-q` | `2I+2P+14B` | `2I+2P+14B` (opt-in preserved) |

**Invalidates no measurement:** the `-q` path is byte-identical, and all five `scripts/` harnesses
pass `-q`. `tests/cli_shipped_config.rs` asserts the structural invariant — `main.rs` builds no
config from `Default::default()` — so a sixth site cannot quietly reintroduce it, plus that the
preset and manual paths agree and that the library default still permits B-frames.

No decision record: no default changed. The shipped default was already P-only since 2026-09-06;
this makes four CLI paths actually honour it.

### BUG-40 — the eager `block_match_bidir` pipeline is BUG-25's shape on DX12 (**FIXED 2026-09-08**, step 1)

**Step 1 landed 2026-09-08.** `match_bidir_pipeline`, `compensate_bidir_pipeline` and
`compensate_bidir_chroma_pipeline` are `OnceLock`s, compiled on first dispatch, same pattern as
`split_pipeline`. Decision `0049`. Metal byte-identical on a q=75 still (`0b5cc743…`) and a
9-frame ki=9 I+P sequence (`d75c72ee…`). Intra `GNC_PROFILE` is silent; a B-pyramid encode
prints `[bug40] match_bidir_pipeline=1`. DX12 intra is not re-run here — that is the laptop
round's measurement. **Step 2 (the FXC X3695 in `block_match_bidir.wgsl` on an actual B-frame
dispatch) is unfixed and needs Windows;** it is not a startable item of its own until someone
is on that machine.

Filed 2026-09-08 while updating `docs/GPU_TIER_TEST.md` for a third laptop round. Not a new
measurement — the crash was recorded on 2026-09-08 and never given an id.

**Every DX12 encode dies at pipeline creation**, before a pixel is read, with an FXC HLSL compile
error: `X3695: race condition writing to shared` in **`block_match_bidir.wgsl`** (line 260) —
on both Intel Arc Pro and NVIDIA RTX 2000 Ada. naga's generated HLSL trips FXC's groupshared
race check; the same WGSL compiles under Vulkan/naga-SPIR-V. Distinct from BUG-25: different
backend, different compiler, different shader.

**The reason it stops an *intra* encode is a pipeline-creation choice, not the shader.**
`split_pipeline` is lazy (`OnceCell`, `src/encoder/motion.rs:1183`) and stayed lazy after BUG-25
was fixed, on the rule rather than the bug — *a shader's cost, including the risk that it does
not compile, is paid by the feature that uses it and not by everything else.*
`match_bidir_pipeline` never got that treatment: `MotionEstimator::new` creates it eagerly
(`src/encoder/motion.rs:317`). B-frames have been off by default since BUG-5 and the pyramid has
been suppressed since 2026-09-06, so **the default path compiles a bidirectional
motion-estimation shader it will never dispatch, and one backend dies on it.** That is exactly
the failure BUG-25's laziness was introduced to stop.

**Two halves, and the cheap one comes first.**

1. **Make `match_bidir_pipeline` lazy**, mirroring `split_pipeline` — the pattern, the layout
   and the doc comment already exist a thousand lines below it. Also `compensate_bidir_pipeline`
   (`:364`), which is on the same feature. Costs nothing on any working backend and is testable
   here: Metal must stay byte-identical.
2. **Then the shader.** Whether FXC's complaint is a real groupshared race or an
   over-conservative check is unknown and is the part that needs a Windows machine. Note
   `block_match_bidir.wgsl` is also the file BUG-34 wants to shed a storage buffer from — three
   open reasons for caution in one low-traffic file.

**Success criterion:** an intra DX12 encode on Windows either completes or fails on something
that is not a shader it does not use. **Why P2:** it invalidates no measurement and blocks
nothing on Vulkan or Metal, but GOALS rule 4 claims DX12 and step 1 is close to free. Step 1
alone converts "DX12 does not run GNC" into a measurement.

### BUG-34 — GNC requests 10 storage buffers per stage against a default of 8 (**DONE 2026-09-08**)

Request is now **9**, via `gnc::required_limits()`. 10 was unused slack: naga counts
`block_match_bidir.wgsl:main` alone at 9 storage buffers, next heaviest at 7
(`motion_compensate_bidir` and `_chroma`). 9 is still an
override of `Limits::default()`'s 8 (and the WebGPU spec). Recorded in `docs/decisions/0047`.
`tests/requested_limits.rs` asserts the whole `Limits` struct against default plus that one
field, and that the heaviest entry point is still `block_match_bidir` at 9 — so a tenth
binding, or a new override, fails the test rather than a browser. **Not done: merging two
bindings to reach 8.** B-frames are off by default, the shader is BUG-25's crash site, and
BUG-40 holds the file. Canary: `[bug34] … max block_match_bidir.wgsl:… at 9 storage buffers;
request 9`. `gnc gpu-info` prints the override against default 8. No codec change; no
measurement moved.

Filed 2026-09-08 by ENT-7, found while checking a literature brief's claim about the WebGPU
default rather than by looking for it.

`src/lib.rs:1431` asks for `max_storage_buffers_per_shader_stage: 10` on top of
`wgpu::Limits::default()`, whose value for that field is **8** — verified in
`wgpu-types-24.0.0/src/lib.rs:1270`, and 8 is also the WebGPU specification's default. So
**CLAUDE.md's "GNC asks for wgpu's default limits, not the hardware's, so the same shaders run
under WebGPU (rule 4)" is not true for storage buffers**, and its portability table prints the 10
without flagging that it is an override. `gnc gpu-info` shows `storage buffers / stage: adapter
has 31, GNC requests 10`, which is exactly the line that should have made this obvious.

**Same class as BUG-31, different limit.** Nothing fails here — the adapter offers 31 — so this is
invisible on the dev machine and would surface as a failed `request_device` on a conformant
implementation held to the defaults.

**Three things to settle, and the third is the point.**

1. **Which shader needs the tenth buffer? None — this is already answered.** Verified 2026-09-08
   (found by the BUG-32/COORD-2 session, re-checked here):
   `for f in src/shaders/*.wgsl; do echo "$(grep -cE 'var<storage' $f)  $f"; done | sort -rn`
   puts **`block_match_bidir.wgsl` alone at 9** — bindings 1–9 storage, binding 0 the uniform —
   and the next four shaders at 7. So **10 → 9 in `src/lib.rs:1431` is free** and needs no shader
   change. That does not retire the override, since 9 is still above the default's 8, which is why
   it is not worth a commit on its own: getting to **8** means that one shader shedding one
   buffer, and it has two obvious merge candidates — `fwd_motion_vectors` + `bwd_motion_vectors`
   are both `read_write array<i32>`, and `predictor_fwd_mvs` + `predictor_bwd_mvs` are both
   `read array<i32>`. **Price the merge before starting it:** this is the B-frame bidirectional
   path, B-frames have been off by default since BUG-5, and this same file is the DX12 FXC X3695
   failure and was BUG-25's crash site. A low-traffic file with three open reasons for caution is
   either the ideal first target or the worst one, and that judgement is this item's to make.
2. **If the override has to stay, it is a decision, not a line of code.** CLAUDE.md already says
   raising a request "is unmeasured and would need a decision record, not a commit" — that applies
   to a request already raised.
3. **Nothing enforces this.** BUG-31's fix comes with a static test that sums `var<workgroup>`
   declarations per `.wgsl` file against `Limits::default()`. The same test should assert every
   field of the `Limits` GNC requests against `Limits::default()`, and fail on any override that
   is not annotated. That is the change that stops a third instance.

**Success criterion:** either the request is `Limits::default()` unmodified with all tests green,
or the override is recorded in a decision record and asserted by a test that names it. Plus
CLAUDE.md's portability prose corrected either way.

**Why P2.** Same reasoning as BUG-31 — no measurement is invalidated and nothing fails on this
machine — but the affected claim is a documented project rule, and step 1 may well be free.

### BUG-35 — five more compute entry points are over the workgroup budget; the default path is done, the rANS half is not (todo, P2)

**The default encode path is off the over-budget entry point, and the histogram it was computing
turned out to be dead work.** `quantize_histogram_fused.wgsl` gained
`main_quantize_only` — same quantiser, no `shared_hist`, **3264 B measured** against `main`'s
23800 B — and the histogram pipeline is now created **lazily**, because it is pipeline *creation*
a conformant WebGPU implementation refuses, not dispatch. On a Rice encode it is never created.

**The histogram had exactly one consumer** — the rANS batch encoder's
`encode_3planes_skip_histogram` — and the entropy branch tests Rice first, so the Rice path never
reached it. Everywhere else the shader was filling a 20 KB workgroup histogram with atomics,
writing it to device memory, and nobody read it. Counted, not argued (`GNC_PROFILE`):

| configuration | fused dispatches | of which with histogram |
|---|---|---|
| q=90 / q=100, Rice, 4:4:4 (**the default**) | 3 | **0** |
| q=90, Rice, 4:2:0 | 3 | **0** |
| q=15, rANS, 4:4:4 | 3 | **3** |
| q=15, rANS, 4:2:0 | 3 | **0** |
| q=50 / q=70 (CfL on, no fusion) | 0 | 0 |
| q=90, abac | 0 | 0 |

**Invalidates no measurement: 10 of 10 encodes byte-identical** before and after — Rice
q=50/90/100 4:4:4, Rice q=90 4:2:0, q=15, `--rans` q=50 and q=70, `--abac` q=90, and sequences at
ki=1 and ki=9. The Rice arm being identical *is* the proof the histogram was dead. Permanent canary:
`fused_qh_does_not_build_the_histogram_pipeline_on_the_default_path` asserts zero histogram
dispatches on Rice **and** that the quantise path ran, so it cannot pass by asserting nothing —
byte-identity alone would not have caught a flag stuck at `true`. Decision `0035`.

**Guard landed, shrink refused (2026-09-08).** `check_hist_arena_capacity` is the encode-side
twin of `check_cumfreq_capacity`: it sums per-group `alphabet_size` (not +1) and refuses a
tile over 5120. The shaders skip out-of-range atomics so naga's clamp is not the only bound.
Canary: `GNC_PROFILE=1` prints `[rans] hist_arena_max=N/5120 (tile T)` on every rANS encode;
Rice still prints `with_histogram=0` and never hits the check. Decision `0048`.

Measured on four stills, `--rans`, 4:4:4, this Mac, not idle:

| image | q=15 | q=25 | q=50 | q=70 | q=85 | q=90 |
|---|---|---|---|---|---|---|
| bbb_1080p | 313 | 940 | 1972 | 3428 | **5322 refuse** | **6648 refuse** |
| blue_sky_1080p | 320 | — | — | — | — | **6843 refuse** |
| touchdown_1080p | 335 | — | — | — | — | **6575 refuse** |
| kristensara_720p | — | — | — | — | — | **7004 refuse** |

q=15 is rANS's default range (preset picks it at q≤20) and sits at **6% of the arena**.
q=70 `--rans` (in 0035's identity gate) still fits at 3428. q≥85 `--rans` was already
silently corrupting the last bins; the check makes that a named refusal instead of a
wrong file. `--rans` at contribution quality was never a supported operating point
(BUG-9's cumfreq table refuses kristensara at q=76 for a smaller array).

**Shrinking 5120 → ≤3266 is rejected.** That is what would put `main` under 16384 B, and
it would refuse bbb at q=70 (3428 > 3266) — a configuration 0035 shipped as byte-identical.
Growing the arena to 7004 would take `main` to ~31 KB, which is the raise-the-limit option
`0032` already refused.

**What is still open.** The five over-budget rANS entry points, including `main` at 23800 B.
A browser still cannot create the histogram pipeline. Packing, a storage-buffer histogram,
or parking those shaders explicitly are the remaining answers; shrinking is not one of them.
`rans_decode.wgsl:main` and `rans_encode_lean.wgsl:main` sit at **exactly 16384 B**.
`rans_encode.wgsl:main` is **16388 B** (+4 B), which is the extra cumfreq slot BUG-9 added.

Original entry follows.

#### BUG-35, as originally filed

Found 2026-09-08 by `tests/workgroup_storage_limit.rs`, the check written for BUG-31. That item
was filed as "abac's two GPU shaders"; the sweep found **nine** entry points over budget across
five shaders. Four were abac's and are fixed. These five are not:

| shader:entry point | declares | over budget by | path |
|---|---|---|---|
| `rans_normalize_encode_fused.wgsl:main` | 33816 B | +17432 B | rANS (parked) |
| `quantize_histogram_fused.wgsl:main` | 23800 B | +7416 B | **default encode** |
| `rans_histogram.wgsl:main` | 23752 B | +7368 B | rANS (parked) |
| `rans_normalize.wgsl:main` | 18460 B | +2076 B | rANS (parked) |
| `rans_encode.wgsl:main` | 16388 B | **+4 B** | rANS (parked) |

Budget is 16384 B, which is what `GpuContext` requests and what the WebGPU spec guarantees as a
minimum. Nothing enforces it natively — see BUG-31 and `docs/decisions/0032` for why, and for the
reason a passing browser would not close this.

**`quantize_histogram_fused.wgsl` is the one that matters and it is not opt-in.**
`EncoderPipeline::new` constructs `FusedQuantizeHistogram::new(ctx)` unconditionally
(`src/encoder/pipeline.rs:727`), and it is the fused quantise+histogram stage CLAUDE.md lists as
part of the architecture. So the *default encoder* asks for 7416 B more than the device it created.
The encoder is not exposed to JS today, which is the only reason this is not already a browser
failure; GOALS §2 rule 4 does not distinguish encoder from decoder.

**`rans_encode.wgsl` at +4 B is worth its own line**, because it is the shape of a defect that a
tolerance would hide: it is 16388 B against 16384. One `vec4` of slack would fix it, and no
plausible occupancy argument is disturbed by it.

**Order of work.** `quantize_histogram_fused` first — it is the only one on a live path. The rANS
four are a parked backend (GOALS §5b: rANS must never be the default), so they are less urgent and
not less real; if the answer for them is "park the shaders too", that needs saying explicitly
rather than leaving the test's exception list as the record.

**Do not fix these by raising the limit request.** That is the one option BUG-31 ruled out with a
reason that applies unchanged here.

**Canary already in place:** the exception list in `tests/workgroup_storage_limit.rs` holds each
shader's *exact* current size, so a fix cannot land silently and the record cannot rot. Closing
this item means that array is empty and the machinery around it can go.

### BUG-31 — abac's two GPU shaders ask for more workgroup memory than the device is created with (**FIXED 2026-09-08**)

**Fixed by packing, not by narrowing the workgroup. 18688 B -> 6400 B on both shaders, and the
emitted bytes did not move: 98 of 98 whole-file comparisons identical, four decoded outputs hashed
equal before and after.** `rows` now stores one clamped byte per magnitude, four to a word, and the
clamp is exact rather than approximate — `bucket` saturates at `nb >= 1 << (NUM_BUCKETS - 2)`, so a
clamped contributor and the true one land in the same bucket and contributors below the clamp are
stored exactly. Rejected: `WG` 32 -> 28 (fits, trivially identical, idles 12.5% of the lanes and
still only one workgroup per core), `MAX_BLOCK_W` 64 -> 48 (invalidates every abac rate figure),
raising the request to 32768 (trades GOALS rule 4's portability axis), and moving `probs` back to
function scope (a measured regression the shader's own comment records). Decision `0032`.

**The item understated its own finding by more than half.** The check it needed —
`tests/workgroup_storage_limit.rs`, which computes declared workgroup storage per compute entry
point from naga and asserts it against `wgpu::Limits::default()` — found **nine** entry points over
budget, not four. The five that are not abac are **BUG-35**, and one of them is on the default
encode path. That test now guards the class: a new offender fails it, and a recorded one that
grows, shrinks or is fixed fails it too.

**Its stated first step was deliberately not made the gate.** "Run it in a browser; if it passes,
this is P3 documentation" does not follow — a browser that happens not to validate would not make
18688 B against a 16384 B device conformant, it would only hide the defect behind one
implementation's leniency. The deterministic test is the gate; a browser run is still worth having
as evidence and is still owed. Original entry follows.

Filed 2026-09-08 by ENT-5, found by reading **BUG-29**'s new limits table rather than by hitting
it. Nothing fails on this machine, and that is the whole point: the check that would fail is one
this stack does not perform and a browser does.

**The two numbers.** `abac_decode.wgsl` and `abac_encode.wgsl` each declare

```wgsl
var<workgroup> probs: array<u32, 576>;    //  2 304 B
var<workgroup> rows:  array<u32, 4096>;   // 16 384 B
```

**18 688 B**, used by every entry point that codes (`main`, `main_rc` in both files). `GpuContext`
requests `wgpu::Limits::default()` with only `max_storage_buffers_per_shader_stage` overridden, and
`Limits::default()` sets `max_compute_workgroup_storage_size: 16384` (wgpu-types 24.0.0). So the
shaders are **2 304 B over the limit the device was created with**, on both sides of the codec.

**Why it runs anyway, and why that is not reassuring.** `max_compute_workgroup_storage_size`
appears **zero** times in `wgpu-core-24.0.5`'s `validation.rs` and `device/resource.rs` — the
limit is negotiated with the adapter and reported back, and never checked against a shader at
pipeline creation. Native Metal therefore honours the 32 KB the hardware has and the requested
16 KB is a number nobody enforces. A conformant WebGPU implementation *does* enforce it: it is a
spec validation rule, and 16384 is the spec's guaranteed minimum, not an accident of wgpu's
defaults.

**The reachable consequence is bigger than abac.** `DecoderPipeline::new` constructs
`GpuAbacDecoder::new(ctx)` unconditionally (`src/decoder/pipeline.rs:296`), and the WASM entry
point `wasm::decode_gnc` builds a `DecoderPipeline` for every call. So if that validation bites,
it bites at pipeline construction — **every WASM decode fails, not only abac ones.** Rice files
included. `EncoderPipeline::new` now constructs `GpuAbacEncoder::new(ctx)` the same way, which is
ENT-5's doing; the encoder is not exposed to JS today, so the decoder is the path that matters.

**Unverified in a browser, and that is the first thing to do.** The chain above is four verified
static facts and one spec rule. Run the WASM decode in a browser before designing a fix: if it
passes, this is P3 documentation; if it fails, the failure will be at `DecoderPipeline::new` with a
validation error naming the workgroup size, and nothing else in the repository has ever exercised
that path.

### Three fixes, and they are not equivalent

1. **Ask for what we use** — `max_compute_workgroup_storage_size: 32768` in `required_limits`,
   beside the `max_storage_buffers_per_shader_stage: 10` already there. One line, no shader change,
   no bitstream change, and it makes the requirement honest rather than accidentally satisfied.
   The cost is real: device creation then *fails* on any adapter offering only the spec minimum,
   which is every conformant baseline device. GNC already requests above baseline for storage
   buffers (10 against WebGPU's 8), so this is a decision the project has made once before — but
   it makes it a second time and harder.
2. **32 threads → 24 per workgroup** — 14 016 B, fits the baseline, no shader logic change.
   **Read `abac_decode.wgsl`'s workgroup-memory comment before costing this**: both arrays are
   indexed `[i * WG + tid]` precisely so lane `tid` lands in Metal bank `tid`, and the comment
   records that getting that wrong is invisible — bit-exact and "dramatically slower". With
   WG = 24 against 32 banks the bank-per-lane property is gone, so this trades a limits violation
   for a bank-conflict pattern that no test can see. It needs the abac bench, on an idle machine.
3. **Pack the magnitudes four to a word** — `rows` becomes 4 096 B and the total 6 400 B, well
   inside the baseline, and the interleaving survives intact. Costs a shift-and-mask per
   neighbour read in the hottest loop of both shaders. Also on the bench.

A fourth path — replacing the coder outright with BPC-PaCo, which would delete both shaders — is
filed as **ENT-7**. It is speculative and must not gate this fix; ENT-7's own step 1 is this bug.

**Success criterion:** both shaders' declared workgroup storage ≤ 16 384 B with
`Limits::default()` *or* the requested limit raised deliberately with that trade recorded; and
`tests/abac_gpu.rs` plus `tests/abac_gpu_encode.rs` still green, since any of these fixes must be
bit-exact — abac's output is byte-identical between CPU and GPU on 98 of 98 files today and a fix
here must not move a single byte.

**Canary, and it cannot be a runtime check.** Native wgpu will not fail, so a test that builds the
pipelines proves nothing. The assertion has to be static: parse `var<workgroup>` declarations out
of the `.wgsl` files, sum them per file, and assert against
`wgpu::Limits::default().max_compute_workgroup_storage_size`. That runs in CI on any machine,
catches every shader rather than these two, and would have caught this on the day abac's decoder
landed.

**Why P2.** No measurement is invalidated, no shipped figure is wrong, and nothing fails on the
machine the project runs on — the same reasons BUG-28 is P2. Against that: CLAUDE.md makes "WASM
target must work" a hard requirement, the affected path is *every* decode on that target rather
than an opt-in coder's, and fix 1 is one line. It is P1 if a browser run confirms the failure, and
P3 if it shows the limit is not enforced there either.

### PERF-2 — the per-dispatch uniform buffers need dynamic offsets, not a cached UBO (todo, P3)

Filed 2026-09-08 by PERF-1, which verified the sites and then declined the fix as specified.

`docs/SIMPLE_PERF_FIXES.md` item 4 lists fourteen sites that build a uniform buffer and a bind
group on every dispatch, and proposes caching them "the way Rice already does" — one persistent
UBO written with `write_buffer`. **All fourteen sites are real and at the cited lines. The
proposed fix is wrong for most of them**, and the reason is already written down in
`rice_gpu.rs`: on Metal/wgpu `queue.write_buffer` is staged, so only the last write before
`queue.submit` takes effect. `quantize.rs:224` runs 6+ times per P-frame with *different*
parameters inside one submit; one cached UBO would give every one of those dispatches the last
write's parameters. Same shape at the per-plane sites in `transform.rs`, `motion.rs` and
`cfl.rs`.

**What actually works is in the tree already:** the wavelet writes all its slots up front and
binds with **dynamic offsets** into one persistent buffer. That is the port — a design change per
site, not a substitution — and it is what this item is.

Second, separable half: **bind-group caching where the bound buffers are stable** — crop, pack,
colour convert, and `dispatch_decode`'s per-plane bind group. `CachedBuffers` already caches
`buf_to_tex_bind_group` (`src/decoder/buffer_cache.rs:477`), so the pattern exists. Worth
microseconds against a 25 ms frame; do it only alongside the first half.

**Success criteria.** Bitstream identical on the 24-artefact set PERF-1 used. A count, not a time:
`create_buffer_init` + `create_bind_group` calls on the steady-state I-frame path, before and
after, printed under `GNC_PROFILE`. Below a 50% reduction in that count, close it — the scan's own
estimate for the whole of item 4 was 0.6 ms of command recording.

### PERF-3 — the decode-side bandwidth items, behind a switch and an idle machine (todo, P3)

Filed 2026-09-08 by PERF-1 as the remainder of `docs/SIMPLE_PERF_FIXES.md`. Claims verified in
step 1; none of the work started. These are the items that are *not* host-side bookkeeping, so
none of them can be closed on a count alone — each needs a real throughput number, which needs an
idle machine (COORDINATION).

- **Item 2 — packed-u8 YUV upload.** Y4M is already YUV; GNC converts it to RGB f32 on the CPU
  (scalar `row × col` loop, ~10 MB of f32 planes plus a 24.9 MB RGB allocation per frame) and then
  back to YCoCg-R on the GPU. Uploading packed u8 and converting in a shader is 4× less DMA and no
  CPU colour — but it changes the encode input API, which is why PERF-1 did not touch it. This is
  the only remaining host change that can close BASELINE's A→C gap on the streaming path.
- **Item 8 — 32-bit Rice bit window.** `rice_decode.wgsl` refills one *byte* at a time inside the
  unary loop of a stage that is 47% of I-frame decode. Encode already accumulates words. No
  bitstream change; mechanical but a shader.
- **Item 9 — the extra full-plane copies** (`transform.rs:320` inverse preamble,
  `gpu_work.rs:624/637/434/411/309`). Tens of MB per I-frame at 1080p.
- **Item 10 — fold dequant into the Rice store.** One dispatch and ~48 MB of traffic per frame.
  Not the closed encode-side fusion (#33); this is the decode dequant that runs *after* Rice has
  produced floats.
- **Item 11 — fuse interleave → inverse colour → crop → pack.** Four full-frame trips, ~100+ MB.

**Do them behind a switch, the `GNC_ABAC_CODER` pattern**, and measure the set together on an idle
machine rather than one at a time under load.

### BUG-32 — `benchmark-sequence` spends 86% of its wall clock on CPU quality metrics, so any throughput figure derived from it measures SSIM (**FIXED 2026-09-08**)

**Fixed with `--throughput`.** Default path unchanged (still prints PSNR/SSIM, still runs the
all-I arm). The flag skips CPU metrics, the second encode, and `decode_sequence` retention.
Canary: `[bug32] throughput=1 metrics=0 i_only=0 decode_retained=0`. `--vmaf` conflicts.
`gpu_tier_bench.py --density` now passes the flag. Decision `0046`.

Measured on this Mac, bbb_extended, n=8, q=90, Rice, not idle (so no fps quoted):

| | k=1 wall | k=1 encode printed | k=9 wall |
|---|---|---|---|
| default | 2.467 s | 142.5 ms + 142.4 ms I-only | 1.434 s |
| `--throughput` | **0.541 s** | 144.1 ms (no second arm) | **0.720 s** |

k=1 wall **4.56×**; bytes identical (15 825 673). The 86% RTX figure was the same shape.
Tests: `tests/bug32_throughput.rs`.

Original filing follows.

### BUG-32 — `benchmark-sequence` spends 86% of its wall clock on CPU quality metrics, so any throughput figure derived from it measures SSIM (original filing)

Found 2026-09-08 while running MEAS-5. Measured on an RTX 4000 Ada, Vulkan, `-q 90 -k 1 --rice`,
120-frame crowd_run clip:

| | wall | of which encode | not encode |
|---|---|---|---|
| `benchmark-sequence -n 8` | **2726 ms** | 208.7 ms (I+P) + 167.7 ms (I-only) | **86%** |
| `benchmark -n 8` | 934 ms | 175 ms GPU work | 759 ms, mostly fixed startup |

**Cause.** Per frame it runs `quality::psnr` and `quality::ssim_approx` on the CPU, for *both* its
I+P arm and its I-only arm, and `decoder.decode_sequence` decodes and retains the entire sequence.
Both arms run unconditionally on the default path: `run_baseline = !run_temporal || ab`, and
`--temporal-wavelet none` makes `run_temporal` false, so `--ab` is not needed to get the second
encode. That is four passes of work per measured frame.

**It degrades superlinearly with length**, because the retained sequence is ~3 GB per arm at 120
frames of 1080p f32: **341 ms/frame at 8 frames, 6.9 s/frame at 120**, where a single instance ran
**13m52s** for 120 frames against ~1.7 s of actual GPU encode. The same pathology, unexplained at
the time, is why a 24-frame `encode-sequence` run on the Mac took 54 minutes at 100% of one core
earlier the same night.

**What it invalidates.** `gpu_tier_bench.py --density` computes `aggregate_fps = frames / wall`, so
swept concurrently it measures **how well N SSIM computations share the CPU**, not GPU encode. The
GPU sat at **43–46 W of a 130 W limit** throughout, while `nvidia-smi` reported
`utilization.gpu 100%` — a activity flag, not saturation, and worth its own line in any future
throughput work.

**What it does not invalidate.** The encoder's *own printed* fps (208.7 ms for 8 frames here) times
the encode phase and is fine — that is BASELINE's quantity **A**. Compression figures from this
command are untouched: bytes, bpp and pixel identity are deterministic and do not care what the
wall clock did. Checked with the session that landed ARCH-3/BUG-18: none of its published numbers
are throughput, and decision 0025 says so explicitly.

**Candidate, not a finding: POSITIONING's M-series density table** (7.02 → 14.15 fps, "~2x at N=8,
most of it already at N=2") has exactly the shape CPU-bound work on N cores produces. But it was
taken 2026-09-05 by a method BACKLOG itself records as unrecorded, and this harness was built the
day after, so **it cannot be attributed to this code path.** Do not write it up as though it can.

**Worked around, not fixed.** `--density-still` (added with this item) sweeps `benchmark` instead:
no per-frame CPU metrics, ~705 ms fixed startup plus 6.8 ms/iteration of non-GPU work against
21.3 ms of GPU work — 24% overhead, and the fixed part amortises, so run large `--iterations`. It
also samples GPU power per level, because power is the occupancy signal utilisation is not.

**The fix proper** is a flag on `benchmark-sequence` that skips the metrics, the second arm and the
whole-sequence decode retention, so a throughput sweep can use the same clip as the hardware-encoder
arm. That was deliberately not done here: it touches a 4000-line handler whose blocks feed each
other's summaries, and MEAS-5 did not need it once the instrument was characterised. Whoever takes
it should keep the metrics on by default — the default should stay the honest one.

**A second defect in the same harness, fixed here.** `hwenc_density` never passed a GOP length, so
the fixed-function arm used its own default — 250 frames on NVENC — against whatever `-k` GNC was
given. An all-intra GNC arm against a 250-frame-GOP NVENC arm is a comparison of GOP lengths
wearing a throughput label. `--keyframe-interval` now goes through as ffmpeg's `-g`, and the row
label prints it so the two arms cannot silently drift apart again. Nothing was ever published from
that arm, so this invalidates no result — it would have invalidated the head-to-head MEAS-5 exists
to run.

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

### ARCH-3 — `gpu_entropy_encode` selected a whole P-frame pipeline, not just where entropy runs (**DONE 2026-09-07**)

**Fixed by separating the concerns, which was the option this entry argued for.** There is one
P-frame encoder and one B-frame encoder now — the batched pipeline — and `gpu_entropy_encode`
picks the entropy step inside it. The second implementation is deleted, not repaired: ~1200 lines
out against ~400 in (`git diff -w`). `docs/decisions/0025`, RESEARCH_LOG 2026-09-07.

**Result:** the two arms decode to **bit-identical pixels** at all four (q, ki) points in
`tests/bug18_locate.rs`, where they diverged by up to 62.9 before. The default configuration is
**byte-identical on 54 of 54** encodes against a baseline pinned at `07c01b1`, so nothing measured
is invalidated.

**A second instance of the same defect was found by measuring and is fixed too.** The unit-test
invariant held on synthetic content and failed on 5 of 9 points on 1080p:
`dispatch_zero_skip_tiles_by_map` was gated on `entropy_mode == Rice` while
`dispatch_tile_skip_motion` zeroed the same tiles' MVs for every coder, so abac paid skip mode's
cost and collected none of its saving, and the two coders coded different coefficients for one
frame. Ungated. **A full-frame pan has no static tiles, so a test written on convenient content
certifies this class of fix as complete when it is not** — `tests/arch3_entropy_stage.rs` now runs
a half-frozen frame as well.

**It also fixed `--huffman` video, which was broken on `main`** and is not mentioned anywhere
else: Huffman took the batched pipeline, which pushed nothing into `huffman_tiles`, so every
P-frame carried an empty tile vector and decoding one panicked. Verified on the pinned baseline,
so it was shipped. No test encoded Huffman video; `every_coder_codes_a_p_frame` does.

**And it removed a capability that had already stopped existing.** `encode_pframe` took a
`predictor_mvs` buffer and returned its own MVs for the caller to feed forward; only the deleted
implementation read them. GNC has never done temporal MV prediction on the shipped path, and the
signature no longer implies it might.

<details><summary>Original statement</summary>


Surfaced 2026-09-07 by the question "how can abac not have a GPU path in a GPU codec?" — it does,
and naming the confusion found the design defect underneath BUG-18.

**What the flag actually does.** It reads as "entropy-encode on the GPU". In `sequence.rs` it also
picks which of **two independent whole-frame P-frame implementations** runs: a batched
single-command-encoder pipeline ("forward + entropy + local decode", ~1460 lines) or a per-plane
one (~360 lines), each with its own local decode. So a coder that merely lacks a GPU *entropy
encoder* silently gets a different frame encoder — and BUG-18 shows that one encodes every P-frame
wrong.

**Which coders that hits, and why it is not about them.** Encode/decode shaders by coder:

| coder | GPU encode | GPU decode |
|---|---|---|
| Rice | `rice_encode.wgsl` | `rice_decode.wgsl` |
| rANS | `rans_encode.wgsl` (+ histogram, normalize) | `rans_decode.wgsl` |
| Huffman | `huffman_encode.wgsl` (+ histogram) | `huffman_decode.wgsl` |
| **abac** | **none** | `abac_decode.wgsl` |
| **Bitplane** | **none** | `bitplane_decode.wgsl` |

abac and bitplane are GPU-decoded — abac at one thread per code-block, ~3000 blocks per 1080p
frame, verified bit-exact against the CPU coder across seven geometries. Nothing about either coder
is broken. They are routed onto a defective *frame* encoder by a flag that should have had nothing
to do with frame encoding.

**The fix, and it is the smallest of the three available.** Separate the concerns: the entropy
encode choice should select the entropy step and nothing else, with both arms running the batched
pipeline. That removes the whole class of this bug rather than one instance of it, and it makes
BUG-18's cause 2 a question about one implementation instead of a difference between two.

The other two are worth doing for their own reasons and neither substitutes for this: **fix the
non-batched implementation** (needed anyway if the path survives), and **give abac a GPU encoder**
— no architectural obstacle, encode is as parallel as decode over the same code-blocks, the only
real complication being that a block's output size is not known in advance (two passes, or
worst-case allocation). That would also close the 129 ms/frame encode gap which is one of the
three reasons abac is not the default (`docs/decisions/0017`).

**Why P1 rather than a tidy-up.** It is the mechanism by which a missing shader becomes broken
video, it silently invalidated a published rate figure (ABAC-SHIP's inter −14.4%), and it will do
so again for the next coder that lands decode-first — which is the natural order for this project,
since decode is the side the product is judged on.

</details>

### BUG-18 — the CPU-entropy P-frame path encoded every P-frame wrong (**FIXED 2026-09-07**)

**Closed by ARCH-3, by construction rather than by finding cause 2.** With one frame encoder there
is no second implementation to disagree with, so "why do they diverge" stopped being a question.
`tests/bug18_locate.rs` now reads **0.000 with zero differing samples** at all four (q, ki) points
where it read 28.8 / 4.24 / 28.8 / 4.24 on the first P-frame, and the CPU arm's P-frames shrink
down the GOP instead of growing. The assertion lives in
`tests/arch3_entropy_stage.rs::entropy_stage_location_does_not_reach_the_pixels`; the test in
`abac_bitstream.rs` that asserted the bug was *present* is retired, with its lessons kept in place.

**abac's inter figure is measurable again and is not −14.4%.** At bit-identical pixels (decoded
PNGs hashed, not inferred from PSNR), abac against Rice over 18 frames at ki=9, 4:4:4:
**−12.0% to −22.9%** across bbb_extended / crowd_run / old_town_cross at q=50/75/90. See
RESEARCH_LOG and `docs/decisions/0025`.

**Not fixed and not caused here:** the encoder's local decode dequantises P residuals with
`config.quantization_step` while the forward pass uses `res_qstep = quantization_step ×
p_qp_scale`, so the encoder's reference drifts from the decoder's wherever the scale exceeds 1.0.
Both implementations did this; the surviving one still does. **Not BUG-8** (closed, a metric bug):
it is the defect `gnc-inter1` holds under the second, colliding `BUG-25`, and it takes id
**BUG-27**.

> **Now fixed — see `### BUG-27` below (INTER-1, 2026-09-07).** The three surviving dequantise
> sites take `res_qstep`. Your reading of it was right in every particular, including that both
> implementations had it, and it drifts in **both** directions rather than only above 1.0: forcing
> the scale to 0.90 spent 4% *more* bits for 5 dB less quality, which is what made it unmistakably
> a defect rather than a bad trade. Live on the default path at all q ≤ 80; byte-identical at
> q ≥ 85, where the taper is already 1.0 and the two values coincide.

<details><summary>Original statement</summary>


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

**The flag conflates two things, and that is why a missing shader turns into broken video.**
`gpu_entropy_encode` reads as "entropy-encode on the GPU". In `sequence.rs` it *also* selects which
whole-frame P pipeline runs. abac and bitplane have a GPU **decoder** (`abac_decode.wgsl`,
`bitplane_decode.wgsl`) but no GPU **encoder** shader, so selecting either coder silently swaps the
entire frame encoder for the other implementation — the defective one. Nothing about abac is
broken; a flag about entropy coding changes the encoder underneath it.

Two ways out, and they are independent: **separate the concerns** so the entropy-encode choice
stops selecting a P-frame pipeline (both coders should run the batched pipeline and differ only in
the entropy step), or **fix the non-batched implementation**. The first is the smaller change and
removes a whole class of this bug; the second is needed anyway if the path is to survive at all.
A third option is to give abac a GPU encoder — there is no architectural obstacle, encode is as
parallel as decode over the same ~3000 code-blocks, the only real complication being that a
block's output size is not known in advance (two passes, or worst-case allocation). That would
also close the 129 ms/frame encode gap that is one of the reasons abac is not the default.

**What it invalidates.** Anything encoded with `gpu_entropy_encode = false` on video — which is
**every abac video encode**, since abac has no GPU encode path. ABAC-SHIP's inter figure
(−14.4% at q=90) is retracted for exactly this reason. Intra is unaffected: single-frame encodes go
through `pipeline.rs` and were verified pixel-identical between the coders. Bitplane video is also
on this path.

**Consider ARCH-3 first.** If the entropy-encode choice stops selecting a P-frame pipeline, cause 2
stops being "why do two implementations disagree" and becomes "is this one implementation correct",
which is a smaller question — and the non-batched path may not need to survive at all.

Possibly the same root cause as **BUG-16** (the two *intra* encode paths disagreeing at q ≤ 30),
and it is worth checking against **BUG-8** ("the encoder's local decode diverges from the real
decoder down a GOP"), which may be this seen from the other side.


</details>

### BUG-33 — does wgpu ship `buffer: Restrict` here at all? (**CLOSED** 2026-09-08 — no, and the question rested on a stale datum)

**Answered: it does not, exactly as its source says.** Both adapters on the bench box report
`robustBufferAccess2 = true` (`vulkaninfo`, checked per device), so `wgpu-hal/src/vulkan/adapter.rs:1899`
resolves to `buffer: Unchecked` and the shipped module carries no clamp. There was never a
contradiction to explain: the item existed because "the real WGSL path crashes" was still on record
after `51a9ac6` had stopped it, and the crash it was reconciling against came from `bug25_emit`
modules that hard-code `Restrict`.

Confirmed behaviourally rather than by reading alone, which is what this item asked for:
`shader_probe block_match_split.wgsl --trusted` — bounds checks off, the one thing the flag changes
— is **OK**, and so is the same probe *without* the flag. Nothing to switch off.

So neither of the two fixes this item was scoping is needed: no `[patch.crates-io]` pin, and no
`create_shader_module_trusted` in the encoder (which would have wanted an `unsafe` call inside a
library that is `#![forbid(unsafe_code)]`). See `docs/decisions/0029`. Original text follows.


**Rewritten 2026-09-08 after the premise inverted.** This item was filed as "why does wgpu ask for
`buffer: Restrict` on an adapter that reports `robustBufferAccess2`?". Reading the path it asked to
have read answers it: **wgpu does not ask for it.** The remaining question is which of four
propositions is false, and that is a sharper and cheaper item than the original.

**The rule, read at all three sites** (`wgpu-hal` 24.0.4, the version in `Cargo.lock`):

* `vulkan/adapter.rs:1899`, in `device_from_raw` — the options every user shader is compiled with:
  `index: Restrict`, `buffer: robust_buffer_access2 ? Unchecked : Restrict`.
* `vulkan/device.rs:1831` and `:916` — all four policies `Unchecked` when
  `runtime_checks.bounds_checks` is false.

And `private_caps.robust_buffer_access2` (`adapter.rs:1595`) comes from **querying**
`VkPhysicalDeviceRobustness2FeaturesEXT`, which wgpu pushes into its `features2` chain whenever the
device *supports* the extension (`adapter.rs:1372`). Support decides it, not enablement. The RTX
4000 Ada reports the feature, and lavapipe has implemented `VK_EXT_robustness2` since Mesa 22.2 —
so **both** of the bench box's Vulkan implementations should be getting `buffer: Unchecked`.

**Measured on the dev machine, no GPU** (RESEARCH_LOG, 2026-09-08): the faithful reconstruction of
that configuration, `caps_index_restrict`, is **byte-identical** to `index_restrict_only`
(`sha256 537e7329…`), which was measured on the box as **pipeline OK** — so `capabilities:
Some([…])`, the previous "last untested candidate", is dead. And it contains **zero
`OpArrayLength`**, while every configuration that crashed contains **48**. `docs/bug25/minimal_repro.spvasm`
is nothing but an `OpArrayLength` clamp.

**So one of these four is false:**

1. the adapter reports `robustBufferAccess2` — recorded from `vulkaninfo`, never confirmed to be the
   physical device wgpu selected;
2. wgpu-hal chooses as read above;
3. `spirv_pipeline_probe` reproduces GNC's pipeline creation faithfully — it passes `layout: None`
   and lets wgpu derive the layout, while GNC binds an explicit one;
4. the crash is at compute-pipeline creation of this module at all.

**How to answer it — one run, and it ends the argument.** Dump the module wgpu hands
`vkCreateShaderModule` and `sha256` it against the 19 emitted configurations. Six lines in
`compile_stage` behind a `[patch.crates-io]` git pin, or GFXReconstruct's `gfxrecon-extract` if it
installs on the box.

* hash `537e7329…` → production is the module already measured **OK**, the crash is not in the
  module bytes, and the search moves to the pipeline layout. Note in that case that an NVIDIA report
  from August 2026 has `vkCreateComputePipeline` segfaulting with no validation output **when the
  descriptor set layout's first binding is not 0**, which is exactly the reduced reproducer's shape
  (its only binding is `Binding 4`).
* hash `3fe91fe0…` → `robustBufferAccess2` is not reaching wgpu, the original question is real, and
  the fix is roughly one line upstream.

**If a local switch is needed after all, it is not a `[patch.crates-io]` pin.** wgpu exposes the
knob: `Device::create_shader_module_trusted(desc, ShaderRuntimeChecks { bounds_checks: false,
force_loop_bounding: true })` reaches the `device.rs:1831` site and sets all four policies
`Unchecked`. Costs one `unsafe` call, and **must be gated to the Vulkan backend** or Metal codegen
moves with it and every Metal figure in this repository is invalidated.

**Do not start by rewriting the shader.** That has been tried twice and is not where the defect is.

### BUG-28 — ~~abac and Rice decode to different pixels on subsampled chroma~~ (**CLOSED 2026-09-08 — duplicate of BUG-16, and the blame was backwards**)

**Filed as BUG-26 and renumbered to BUG-28 the same day** — `intra1` filed a different BUG-26
(`--tile-size 1024` silently destroys the image, P1) minutes later, and one `scripts/claim take
BUG-26` then covered both headings, so neither could be claimed on its own. Mine moves because
theirs is held and being worked on; renumbering the held one would collide inside a live worktree.
BUG-27 stays reserved for the P-frame dequant defect, per COORDINATION — **now filed and fixed by
INTER-1**, see `### BUG-27`.

Found 2026-09-07 while closing ARCH-3, on the intra path, **and reproduced identically on `main`
at `1d67d29`** — so it is not caused by ARCH-3 and it is not about inter. Filing it because it
contradicts a standing claim, not because it was hit.

**The claim it contradicts.** abac's headline is "−16.6% to −18.8% **at identical pixels**", and
decision `0018` says abac "pays on intra, inter, lossless and every chroma format at once". Entropy
coding is lossless, so Rice and abac must decode a frame to the same bytes. At 4:4:4 they do, at
every quality tried. **At 4:2:2 and 4:2:0 they do not.**

**Measured**, single all-intra frame of bbb_extended, 1920x1080:

| chroma | q | pixels | max \|diff\| | differing samples | rice bytes | abac bytes |
|---|---|---|---|---|---|---|
| 4:2:2 | 50 | **DIFFER** | 12 | 266 151 (4.28%) | 612 776 | 514 311 |
| 4:2:2 | 75 | **DIFFER** | 5 | 68 187 (1.10%) | 1 005 253 | 871 762 |
| 4:2:2 | 90 | identical | 0 | 0 | 1 562 781 | 1 360 681 |
| 4:2:0 | 50 | **DIFFER** | 13 | 242 574 (3.90%) | 530 700 | 447 710 |
| 4:2:0 | 75 | **DIFFER** | 5 | 36 332 (0.58%) | 865 324 | 755 090 |
| 4:2:0 | 90 | identical | 0 | 0 | 1 327 234 | 1 161 662 |

Mean magnitude over the differing samples is 1.2-1.6, so this is a *small* difference over a
*large* area — the shape that PSNR to two decimals hides, which is how BUG-18's first two
explanations went wrong. It is not visible corruption; it is two coders coding different
coefficients.

**Where to look first.** The q boundary is the tell: it differs at q=50 and q=75 and agrees at
q=90. Two features switch off between those points — **adaptive quantisation** (on for
30 ≤ q ≤ 80) and **CfL** (on for 50 ≤ q ≤ 85) — and both write per-tile side data whose layout is
tile-count-dependent. On non-444 the chroma planes have a *different tile grid* from luma, and
`pipeline.rs` already carries an assert saying the rANS and Huffman batch dispatches cannot handle
that; Rice and abac each have their own per-plane path. One of the two is very likely indexing
that side data with the luma tile count. Read `use_cfl` and the AQ weight-map indexing in
`pipeline.rs` against the abac per-plane dispatch before anything else.

> **CLOSED 2026-09-08 as a duplicate of [BUG-16](#bug-16--rices-gpu-and-cpu-encode-paths-disagree-on-the-coefficients-todo-p2), whose root cause is now found and proved.**
> The entry below is kept because it is the better *symptom* description, but it is wrong on all
> three of its substantive claims and nobody should chase them:
>
> 1. **The blame is backwards. abac is correct; GPU Rice is wrong.** Adding a third coder settles
>    it: **CPU-Rice and abac agree with each other in 10 configurations out of 10**, across
>    q ∈ {25, 35, 40, 75, 90} × {4:4:4, 4:2:2}. Every divergence is GPU-Rice against both of them.
> 2. **It is not "on subsampled chroma".** At q ≤ 35 it happens at **4:4:4 too** — this entry only
>    missed it because 4:4:4 was tried at q ≥ 50, where CfL is on and hides it.
> 3. **It is not CfL or AQ side-data indexing**, which is what "where to look first" below sends
>    you at. CfL matters only because CfL-on *disables* the fused quantiser; it is otherwise
>    uninvolved.
>
> **Root cause: `src/shaders/quantize_histogram_fused.wgsl:441`**, the Phase-2 sparse-group
> dead-zone expansion — for non-LL groups that are ≥95% zero it re-quantises the surviving ±1
> coefficients to 0. `quantize.wgsl` has no such step, so the two quantisers are not equivalent and
> the fused one is *lossier by design*. Proved by disabling that one branch: **GPU-Rice then equals
> CPU-Rice at all 10 points, and every hash equals abac's.** Full evidence and the measured trade
> are in BUG-16.
>
> **A much sharper reproducer than the one below**, for whoever fixes BUG-16: encode a **grayscale**
> image (`ffmpeg -i in.png -vf format=gray,format=rgb24 gray.png`). Chroma is then exactly zero, so
> subsampling is lossless and any difference is pure luma — and the 4:2:2 failure window widens from
> the two points below (q=50, 75) to **every q ≤ 86**.

**Why P2 and not P1.** Nothing shipped is measured at non-444 with abac: every abac figure in the
repository (intra, lossless, and the inter figures added 2026-09-07) is 4:4:4. So no published
number is wrong. What is wrong is the *scope* claimed for them, and the next person to quote abac
on a 4:2:0 mezzanine would be quoting a coder that changes the picture.

**Do not close this by widening a tolerance.** The correct assertion is bit-exactness — a
`psnr > 45.0` check reads 55 dB as a pass, which is exactly how BUG-15 survived a day.

### BUG-16 — Rice's GPU and CPU encode paths disagree on the coefficients (**FIXED 2026-09-08**)

**Scope corrected 2026-09-08, and the correction is not mine.** This section first said "only
q <= 30 is affected", from a 4:4:4 sweep. The `intrasym` session, which found the same root cause
independently while working BUG-28, measured it on a **grayscale** source where chroma is exactly
zero and got the real answer: **4:4:4 differs at q <= 35, but 4:2:2 and 4:2:0 differ at q <= 86.**
Confirmed here on bbb_1080p with the flag on and off — 4:2:2 and 4:2:0 differ at q=50/75/85/86
(-0.70% down to -0.00%) and are byte-identical at q=90. So the claim that GNC's home range was
untouched holds only for 4:4:4 and only from q=90; my own 4:2:2 q=75 check was already evidence
against it and I did not connect the two.

**Cause: the fused quantiser had a sparse dead-zone expansion that no other quantise path had.**
Phase 1.5 of `quantize_histogram_fused.wgsl` re-quantised the remaining ±1 values to zero in
non-LL subband groups already ≥95% zero, up to 1.25× dead zone. `quantize.wgsl` and the CPU
quantiser have nothing of the kind, so the same `CodecConfig` produced different coefficients
depending on which quantiser ran — and the entry's own hypothesis ("a dead-zone or rounding
difference between the fused shader and the separate quantise shader") was right.

**All three rows of the table above are that one feature:**

| | recorded | now |
|---|---|---|
| q=25 GPU | 35.51 dB, 415 544 B | **35.63 dB, 425 944 B** |
| q=25 CPU | 35.63 dB, 610 264 B | unchanged |
| 4:2:2 q=75, GPU vs CPU pixels | max abs diff 1.69 | **max abs diff 0**, zero differing pixels |

The two arms now agree on the picture to three decimals. **The remaining 43% size difference is
expected, not a defect** — the CPU reference Rice coder lacks per-stream *k* and the checkerboard
*k*-context, which is what the entry already said about the q=90 row.

**Priced before deciding, with the same coder in both arms** (new `flags` bit 1; comparing against
`--cpu-encode` cannot price it, because that arm is independently worse). Three stills, q=15/25/30:
saves 2.17-4.12% of rate for 0.094-0.214 dB, i.e. **BD-rate −0.35% / +4.20% / −0.79%, mean
+1.02%** and direction-inconsistent. It buys nothing, so it is **off by default** and kept behind
`GNC_SPARSE_DZ=1` rather than deleted (three points is a thin ladder) or ported to the other two
quantisers (porting a wash). Decision `0038`.

**Scope is measured: only q ≤ 30.** Byte-identical either way at q=40/50/75/85/90/100 — above
q≈30 the dead zone is too narrow for a subband to reach 95% zeros. So GNC's home range is
untouched, and **BASELINE's q=25 row moved** to 35.63 dB / 1.64 bpp / VMAF 90.31 (from 35.51 /
1.60 / 90.25). VMAF moved **+0.06, an improvement**, far inside the 0.5-point tolerance. GOALS's
copy of that table is updated too.

Original entry follows.

#### BUG-16, as originally filed

#### BUG-16 — the root-cause proof, from the session that found it independently

Written by `intrasym` while working BUG-28, and kept in full: it proves the cause by a different
route, corrects this entry's scope, and its third-coder arbitration is a better argument for
turning the expansion off than the BD-rate above.


> **Found 2026-09-08 while working BUG-28, which is a duplicate of this and is now closed.**
>
> **The cause is `src/shaders/quantize_histogram_fused.wgsl:441`** — the Phase-2 *sparse-group
> dead-zone expansion*. For every non-LL subband group that is ≥95% zero after the first
> quantisation pass, it re-quantises the surviving `|q| == 1` coefficients to 0:
>
> ```wgsl
> if (zero_frac_x100 >= 95u) {
>     let t = f32(zero_frac_x100 - 95u) / 5.0;
>     shared_group_dz_mul[g] = 1.0 + 0.25 * clamp(t, 0.0, 1.0);
> ```
>
> **`quantize.wgsl` has no such step.** The two shaders are therefore not two implementations of
> one quantiser — the fused one is deliberately lossier, and which one runs is decided by
> `use_fused_qh = use_fused_quantize_histogram && use_gpu_encode && !use_cfl`. Nothing tells the
> user which quantiser coded their frame.
>
> **Proved, not inferred.** Changing that one threshold to `>= 101u` so the branch can never fire
> makes **GPU-Rice equal CPU-Rice at all 10 points measured** (q ∈ {25,35,40,75,90} × {4:4:4,
> 4:2:2}), and every resulting hash equals abac's. Restored afterwards; no source change is
> committed with this entry.
>
> **This entry's own reasoning was one step from it.** It said "being on the fused path is
> necessary at most, not sufficient" — correct. The sufficient condition is *sparsity*: a subband
> group crossing 95% zeros. That is why q=90 agrees with fused active, and why the failure follows
> coefficient statistics rather than any feature flag.
>
> **Third-coder arbitration says which side is wrong.** CPU-Rice and abac — independent coders,
> abac verified bit-exact against its own CPU reference — **agree in 10 of 10 configurations**.
> Every divergence is GPU-Rice against both. The shipped default encoder is the wrong one.
>
> **Corrected scope: this is not "below about q=30 and at subsampled chroma".** Measured on a
> grayscale 1080p frame, where chroma is exactly zero so subsampling is lossless and every
> difference is pure luma:
>
> | | GPU-Rice vs CPU-Rice |
> |---|---|
> | 4:4:4 | **differs at q ≤ 35**, agrees q ≥ 40 |
> | 4:2:2 / 4:2:0 | **differs at q ≤ 86**, agrees q ≥ 90 |
>
> A grayscale source is the reproducer to use: it widens the 4:2:2 window from two points to
> everything below q=90.
>
> **What the expansion is worth**, three 1080p stills, PSNR and size with the branch on (shipped)
> against off. The last column converts the PSNR loss into rate using each image's *own* local RD
> slope, measured between the two nearest points on its ladder, so the two halves are comparable:
>
> | image | point | PSNR on → off | size on → off | net |
> |---|---|---|---|---|
> | bbb | q=25 4:4:4 | 35.51 → 35.63 (+0.12 dB) | +2.50% | ~+1.1% win |
> | blue_sky | q=25 4:4:4 | 37.24 → 37.37 (+0.13 dB) | +2.40% | ~neutral |
> | touchdown | q=25 4:4:4 | 35.44 → 35.55 (+0.11 dB) | +3.69% | ~+0.6% win |
> | all three | q=40 4:4:4 | **0.00 dB** | **0.00%** | does not fire |
> | bbb / blue_sky / touchdown | q=50–75 4:2:2 | +0.03 to +0.06 dB | +0.14 to +1.85% | small win |
>
> **So the expansion is a marginal net win (0–1%), not free and not harmful — which is exactly why
> the fix is a decision and not a repair.** Three options, and the cheap one is not obviously
> right:
>
> 1. **Delete it from the fused shader.** Paths agree immediately, the shipped encoder stops being
>    worse than its own reference, and it costs the 0–1%. Needs a BD-rate ladder to confirm the
>    single-point arithmetic above, not four points.
> 2. **Implement it in `quantize.wgsl` too.** Keeps the win and makes the paths agree, but the
>    separate quantiser has no per-group zero counts — the fused shader only has them because it is
>    also building a histogram — so it means a second pass. It also changes **abac's** output, and
>    abac's published −16.6% to −18.8% would have to be re-measured.
> 3. **Gate it in config and apply it uniformly**, so "GNC's quantiser" has one definition and the
>    expansion is a named, documented tool rather than a side effect of which entropy coder was
>    picked.
>
> Whichever is chosen changes shipped rate/quality, so it wants a decision record and a real
> BD-rate ladder. **Not done here** — this session found and proved the cause and measured the
> trade; it did not pick the answer.
>
> Note `tests/abac_bitstream.rs::rice_gpu_and_cpu_encode_paths_differ_at_subsampled_chroma` pins
> the *current* disagreement, so any of the three fixes will fail it. That is intended, and its
> comment says so.


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

### BUG-22 — Huffman's GPU per-stream output slot has no bound (**FIXED 2026-09-08**)

**Fixed by sizing the slot from the work, and bounding the shader's writes as well.**
`STREAMS_PER_TILE` threads split a tile's coefficients evenly, and the worst case per symbol is
four bytes — significance bit, sign bit, an 8-bit code, an exp-Golomb escape — so four bytes per
symbol is an upper bound rather than an estimate. The host computes it, passes it as
`max_stream_words`, and **both** `stream_output` writes in `huffman_encode.wgsl` are bounded by it.
Correct sizing makes the configuration work; the shader bound makes a *wrong* size truncate one
stream instead of corrupting the next one. 4 KiB per stream at tile 512, ~37 MB of scratch for
1080p 4:4:4. The slot size is now part of the buffer cache's key.

**Measured recovery on the arm this bug defined** — q=90, tile 512, where it read 7.8-10.9 dB:
bbb 50.08 dB, blue_sky 49.95, kristensara 49.66, touchdown 49.57, max error 4 on all four.
BASELINE puts q=90 at 50.06 dB, so these are the operating point, not merely better — about
**+40 dB**. Decision `0037`.

**Why now, when the entry declined this fix.** It declined it because "BUG-14's tile-512 arm is the
only thing that wants it", which was true while the alternative was a guard. Once the guard is the
only thing between an ordinary command line and a crash, it stops being true.

Original entry follows.

#### BUG-22, as originally filed
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

### BUG-23 — `clamp_code_lengths` does not terminate (**FIXED 2026-09-08**)

**Fixed by building a real tree on scaled frequencies instead of patching a length histogram.**
While the natural tree is deeper than the 8-bit limit, halve every non-zero frequency and rebuild.
It terminates because 32 halvings take any `u32` to 1 and a uniform 64-symbol alphabet has depth 6
— an assert names that bound rather than trusting it — and **the result satisfies Kraft by
construction**, which is the property that matters: the old code could return with excess unplaced,
after which `assign_canonical_codes` would emit codewords that are not a prefix code. Rounding up
on the halving is load-bearing; a live symbol that scaled to zero would lose its code entirely.
`clamp_code_lengths` is gone. Decision `0037`.

**Rejected: package-merge.** Optimal, and a few hundred lines. Frequency scaling is not optimal —
it discards histogram resolution — but it is fifteen lines, terminates for a reason a reader can
check, and is what zlib and JPEG implementations do. For a parked coder, a construction nobody has
to audit is worth more than an optimal codebook. It is the upgrade if Huffman is ever unparked.

**These two bugs are one configuration failing twice.** `--huffman -q 100 -t 512` on bbb_1080p hit
this one first (hang, then refusal); fixing it produced a codebook and the encode hit BUG-22
immediately. That configuration now encodes in **0.63 s** and decodes **bit-exact lossless — max
error 0, zero wrong pixels**. The `should_panic` test became two tests: the skewed histogram is
length-limited and Kraft-complete, and a second one asserts its unconstrained tree really is deeper
than the limit so the first cannot pass for the wrong reason.

Original entry follows.

#### BUG-23, as originally filed
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

### COORD-2 — An id must come *from* the compare-and-swap, not be checked against it (**DONE 2026-09-08**)

**Built.** `scripts/claim bug "<why>"` and `scripts/claim dr "<why>"`. Each unions committed
`main` (BACKLOG `BUG-N`, `git ls-tree main docs/decisions/`) with live `refs/claims/*`, CAS
the first gap, retries on a lost race. `take dr-NNNN` remains for a number you already hold.
`claim selftest` now races 8 `bug` allocators and 8 `dr` allocators: **8 claimed, 8 distinct**
both times. Record `0050` (`0049` collided with BUG-40's merge after they dropped the
reservation). Decision `0050`.

Original filing follows.

### COORD-2 — An id must come *from* the compare-and-swap, not be checked against it (original filing)

`scripts/claim` made picking *work* atomic and it did not make picking an *id* atomic. The cost so
far, all on 2026-09-07: **`BUG-25` used twice** (Vulkan shader / P-frame dequant), **`BUG-26` used
twice** (abac-vs-Rice chroma / `--tile-size 1024`, forcing a renumber to `BUG-28`), the dequant
defect renumbered **twice** before landing as `BUG-27`, and decision records **0018, 0019 and 0024
each duplicated**. COORDINATION diagnoses it exactly and in two places: read-decide-write is three
steps, everyone reads the same `main:BACKLOG.md`, and the answer is deterministic — so being
deterministic is what makes the collision reliable rather than unlikely.

**`scripts/claim take BUG-NN` is not the fix and has been tried.** It excludes another *taker* of
that id; it cannot see a renumbering that exists only in someone's worktree, which is exactly how
`BUG-26` moved under a session that had reserved it correctly. Two sessions holding one id for two
different defects means the lock protected neither.

**And it is worse than "insufficient against a race" — 2026-09-08 shows it hands out a number that
has been committed for hours.** A live session reserved `dr-0029` while
`docs/decisions/0029-bug-25-was-one-defect-and-the-second-was-never-reachable.md` was already on
`main`, merged in `fcc3d33`. There was no race, no worktree-local renumbering and no second
reserver: `take dr-NNNN` is a CAS on `refs/claims/dr-NNNN` and **nothing in it ever looks at the
namespace it is reserving in**, so a number sitting in the shared checkout for hours reserves
cleanly. Caught by reading `scripts/claim list` against `ls docs/decisions/` during unrelated
cleanup, before the file was written.

**That matters because it is the case the interim habits do not cover.** COORDINATION's mitigation
is "reserve the number before you write the file", and this session did exactly that; reserving is
what produced the wrong number. Its worktree was based on `1437d72`, so `docs/decisions/` *there*
listed only up to `0028` — **reserving from a stale base is indistinguishable from reserving a free
number.**

Until `claim dr` exists, the habit that actually works is **`git ls-tree --name-only main
docs/decisions/`**, which reads committed `main` from any worktree with no fetch and no rebase,
since every worktree shares one `.git`. `ls docs/decisions/` is the wrong oracle in both
directions: it misses records committed after your base and shows records that are not on `main`
yet.

**But it is mitigation and not a fix, and it does not replace reserving** — a claim this item's own
text made in an earlier revision and which is withdrawn here. `ls-tree` covers a number already
committed; `claim list` plus taking the id covers a number held but not written; **neither covers
two sessions reading "0031 is free" inside the same minute**, which is the original `0018` race and
the case this item exists to close. Whether the habits would have caught the `0018`, `0024` and
`0027` pairs depends on whether the winner had committed before the loser looked, which is not
recorded — so those three are not evidence for either habit. The build below is the fix, and the
title is the design: the number comes *out* of the compare-and-swap.

**What to build:** `scripts/claim bug "<why>"` and `scripts/claim dr "<why>"`, each allocating the
next free number *as* the compare-and-swap that reserves it — the same mechanism `claim next`
already uses for work items, which `claim selftest` proves with 16 processes racing for one item
and 6 for one queue. The free-number scan has to read committed `main` (BACKLOG headings for bugs,
`docs/decisions/` for records) *and* the live `refs/claims/*`, then CAS the winner; a number that
loses the race retries rather than being handed out twice. Extend `claim selftest` with the
analogous assertion: **N processes calling `claim bug` at the same instant get N distinct
numbers.**

**Why P2 and not P1:** it costs time and churn rather than correctness, and no measurement has
been invalidated by it — still true on 2026-09-08. But it has now bitten on five distinct ids in
one day plus `0027` and `dr-0029` since, the fix is mechanical, and every session pays the tax.
**Left at P2 deliberately rather than bumped:** GOALS §4 says pick by value now and not by the
P-number, and the honest read is that this is cheap, frequent churn — the argument for doing it is
the tax and the `dr-0029` class the habits cannot cover, not a correctness risk that appeared. Three cheap habits until it exists, all from
COORDINATION: reserve the id before you write the heading; **push a filing quickly rather than
holding it in a worktree** — an id that exists only locally is invisible to the mechanism that
would protect it; and **read the namespace with `git ls-tree --name-only main
docs/decisions/` just before you write**, since reserving does not consult it and your working tree
is not it. Also worth pairing with the BUG-32 lesson: commit the heading or the
record stub *with* the reservation, so a reserved id that outlives its session is a filed item
rather than a claim on nothing.

### BUG-27 — the encoder's P-frame reference was dequantised with the intra qstep (**FIXED 2026-09-07**)

Found and fixed inside INTER-1. **Filed as BUG-25 in the worktree and renumbered to 27** per
COORDINATION's resolution of that double-used id — the Vulkan BUG-25 was pushed first, then
BUG-26 went to abac/Rice subsampled chroma. `arch3` found the same defect independently while
reading both P-frame implementations for ARCH-3 (`a312d6f`, decision 0025); neither of us had
fixed it. **It is not BUG-18 cause 2** — that was the forward *quantise* on the path ARCH-3 has
since deleted; this is the *dequantise* that rebuilds the encoder's reference, and it was in
**both** implementations, so deleting one did not remove it.

`encode_pframe` quantises P residuals at `res_qstep = quantization_step * p_qp_scale` (TUNE-6's
taper) and records that for the decoder; its local-decode dequantise dispatches read
`config.quantization_step` — luma, 4:2:0 chroma and 4:2:2 chroma. The encoder's reference
therefore differed from the decoder's by `quantization_step / res_qstep`, and every P predicting
from another P inherited a picture no decoder holds. Six sites when found, three after ARCH-3.

**Live on the default path at all q <= 80**, wherever the taper leaves 1.0. At q >= 85 the taper is
already 1.0, the two values coincide, and output is **byte-identical** — 27/27 verified across
three sequences x ki=2/4/9 x q=85/92/99 against a binary pinned at `07c01b1`. crowd_run 10 frames
ki=9 4:4:4, mean/worst-frame PSNR: q=70 goes 13161434 B 35.20/32.08 dB -> 13423148 B
**37.02/35.70 dB** (+2.0% bytes for +1.82/+3.62 dB); q=50 +1.6% for +0.90/+1.78; q=25 +0.4% for
+0.29/+0.41.

**Invalidated:** every inter figure at q <= 80 — MEAS-3 and decision 0019 (mean +4.6% -> -0.3%,
worst-frame +19.1% -> +8.0%), TUNE-5's -3.3%, and TUNE-6's own justification. BASELINE's q=75
sequence table too, though it was already stale for an unrelated reason (BUG-5).

Signature, for whoever meets this class again: a monotone PSNR ramp down a GOP that resets at each
I-frame, with the **first** P-frame correct in both directions — only a P predicting from a P can
inherit a wrong reference. Regression test `tests/pframe_reference_drift.rs` gates on drift
magnitude (4.04 dB with the defect, 0.86 dB from the inter dead zone alone, threshold 1.5) and
reaches the taper through q=50 rather than an env var, since a `set_var` in a `#[test]` is the race
that masked a real decoder bug in `abac_bitstream`. Decision record 0023.

### BUG-24 — `clippy --target wasm32-unknown-unknown` fails on `main` (**FIXED 2026-09-08**)

**Fixed by excluding the CLI from targets it cannot compile for, not by making it compile.** The
binary is now declared explicitly with `required-features = ["cli"]` and `cli` is in the default
feature set, so native builds are unchanged (`cargo build --release` still produces `gnc`) while
`--no-default-features` yields a genuinely bin-free wasm build. CLAUDE.md's gate now reads
`--lib`, matching what LOOP.md already said.

**Why not the other resolution.** Making the CLI's context creation cfg-aware would mean writing
wasm-specific code for a tool that cannot run there — `pollster` cannot block on wasm and the CLI
needs an adapter, a filesystem and ffmpeg. That is dead code carried for a gate, and the gate was
asking the wrong question: the library is what WASM ships, and it was clean all along.

**Verified:** `clippy --release --target wasm32-unknown-unknown --lib` clean;
`--no-default-features` clean; `clippy --release` (native) clean; native binary still built and
runs; full suite unchanged. Without a `[[bin]]` section the binary was auto-discovered and built
for every target, which is why no command-line convention could have fixed this on its own.

Original entry follows.

#### BUG-24, as originally filed
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

**Read that table with BUG-32 in hand.** It has exactly the shape CPU-bound work on N cores
produces, and the harness that would produce that shape spends 86% of its wall clock on CPU quality
metrics. But this table was taken 2026-09-05 by a method BACKLOG itself records as unrecorded, and
`gpu_tier_bench.py` was built the day after — so **the defect cannot be attributed to it.** That
makes it a candidate, not a retraction. It needs re-taking before it is quoted again, and
POSITIONING quotes it today.

**Harness built 2026-09-06; first run 2026-09-08, and that run characterised the instrument
rather than the GPU — see BUG-32.** `--density` computes `frames / wall` from `benchmark-sequence`,
which spends 86% of that wall on CPU-side quality metrics, so swept concurrently it measures how
well N SSIM computations share the CPU. **Use `--density-still` instead**: it sweeps `benchmark`,
which runs no per-frame metrics (24% non-GPU overhead, and the fixed part amortises across
`--iterations`), and it samples GPU power, because `utilization.gpu` reads 100% while the card draws
43 W of a 130 W budget. `--density` and `--hwenc`
sweep concurrent instances of GNC and of the machine's fixed-function encoder over the same clip.
Two things to know before reading its NVENC rows: the 12-session cap is a GeForce driver
restriction and will not appear on a professional Ada part, and the rows are **not
quality-matched** — `meas1_vs_h264.py` is the harness for that.

**Still to do:** the same measurement on a discrete NVIDIA card, head to head against NVENC at
both P1 and P7 presets (P7 is nearer GNC's quality target and roughly 4x easier to win), and on an
H100 where the NVENC column is a zero.

**~~Blocked on a definition problem — fix this first.~~ RESOLVED 2026-09-06, and this entry was
still carrying it as a live blocker on 2026-09-08.** The three numbers are now named and pinned in
[BASELINE.md](BASELINE.md): **A** — GPU encode phase (`benchmark-sequence`), 12.2 fps; **B** — the
encoder loop figure `encode-sequence` prints, 5.6 fps; **C** — end to end incl. PNG decode and
container write, 5.0 fps. A is 2.4x C. The 31.7 fps figure is retracted — it reproduces as none of
the three and its stated parameters are internally inconsistent (ki=8 is below the B-frame
threshold of 9) — and GOALS and the README were corrected under PERF-1 on 2026-09-07. CANARY-1's
single-frame encode/decode loop is a **fourth** quantity, nearest A and not comparable to B or C.
**Every quote must say which letter it is.** Nothing in MEAS-5 is blocked on this any more.

**What MEAS-5 still needs, and why it is not "free to pick up".** Claim A is closed. Claim B is a
pure throughput measurement, and it needs one of two things this project does not have on demand:

1. **A discrete NVIDIA card with a driver new enough for NVENC.** The 2026-09-08 Windows session
   got as far as the RTX 2000 Ada and was stopped by ffmpeg 9.0.1 requiring driver 610+ / nvenc API
   13.1 against the machine's 13.0. Intel QSV was substituted and **out-scaled GNC** (4.54x@N=8 vs
   GNC's 2.01x@N=2) — on the least-favourable hardware for GNC's argument, and entirely on GNC's
   per-process **startup and memory**, not on GPU compute.
2. **An idle Mac**, for the one part runnable here: re-taking the 2026-09-05 M-series density table
   with `--density-still`, which BUG-32 flags as a candidate and POSITIONING still quotes. Eight
   sessions share this machine; at load 39.5 an fps number is not a number (BASELINE, "Timing runs
   require an idle machine"). This is why the item was claimed and dropped again on 2026-09-08
   without a measurement — **nothing is retracted and nothing was run.**

The honest next step is neither of those: it is **amortising per-process startup and streaming the
clip instead of buffering it**, because those are what the two density runs actually measured. Until
they are fixed, a density number on any hardware measures pipeline compilation.

### MEAS-6 — Latency per frame (second pass 2026-09-08, P1)

**The default is no longer the B-pyramid, and three documents said it was for two days.** Found
2026-09-08: `quality_preset()` has vetoed the pyramid since **2026-09-06**
(`b_pyramid: … .unwrap_or(false)`, `src/lib.rs:1021`), on the two measurements below — but
POSITIONING, the README and this entry all still called it the "current default" and quoted
~240 ms as GNC's latency. **GNC's default latency is ~80 ms, and it is below the low-latency-HEVC
band (120 ms floor), not inside it.** See `docs/decisions/0033`.

Verified on the current build rather than read off the source, at the default `ki=9`:

| configuration | frame mix | canary | reordering delay |
|---|---|---|---|
| `benchmark-sequence -q 75` (default) | `2I+16P+0B` | `B-pyramid suppressed … zero reordering latency` | **0 frames** |
| `GNC_B_PYRAMID=1` | `2I+2P+14B` | silent | **8 frames** (160 ms at 50 fps) |

**The two halves of the latency figure have very different standing, and quoting them together
hides that.** The reordering delay is *structural* — 0 frames or 8, read from the encoder's own
frame-type output, unmovable by machine load. The coding time is a 2026-09-06 wall-clock
measurement on a **non-idle** machine, labelled M1 when it was the M5 Pro (BUG-29): GPU encode
~47 ms/frame, decode ~35 ms/frame (upper bound, includes PNG write), **round trip ~80 ms**.

| | latency |
|---|---|
| JPEG XS | 1-32 lines; EBU measured < 1 frame |
| NDI High Bandwidth | < 16 ms |
| **GNC, default (P-only, zero reordering)** | **~80 ms** |
| low-latency HEVC | 120-3060 ms (EBU, real vendors) |
| GNC, `GNC_B_PYRAMID=1` (opt-in) | ~240 ms |

**Still to do, and the first one is cheap.** Re-take the ~80 ms on an **idle** machine — BASELINE
already records that a run taken during a `cargo test` reads 20% slow, and this figure was taken
on a loaded box. It was not re-taken on 2026-09-08 either: the machine sat at load 21-39 with
other sessions running the test suite for the whole session, which is exactly why the structural
half was pinned instead. Then glass-to-glass, which needs instrumentation nobody has:
capture-to-input, output-to-network and output-to-display are all unmeasured.

**Converges with BUG-5.** The B-pyramid already measured as *costing* 7-31% at contribution
quality on camera content. It now also costs 160 ms. Two independent measurements, one
conclusion: **the hierarchical B-pyramid is the wrong default for this operating point.** This is
now a supported configuration change rather than a hypothesis. It does not argue against inter
coding — P-frames have zero reordering delay and were the better performer at contribution quality.

**Still to do:** glass-to-glass instrumentation (capture-to-input, output-to-network,
output-to-display are all unmeasured). Note the ~256-line tile floor is not currently reachable:
the pipeline processes whole frames, so the practical floor is one full frame regardless of tile
size.

### MEAS-10 — Re-take BASELINE against current HEAD (**DONE 2026-09-08**)

Pinned to `0a1b055`. Compression only; fps not quoted (load 2.6–3.9). Harness:
`scripts/meas10_rebaseline.sh` plus `scripts/meas1_vs_h264.py` on 17-frame y4m derived from the
PNG sequences. Canary: `GNC_PAD_FILL=replicate` reproduces the previous still rows exactly.

**Stills, bbb_1080p 4:4:4 Rice** — PAD-1 is why q=25/50/75 moved; q=90 already included it:

| q | before (BASELINE) | MEAS-10 | bpp |
|---|---|---|---|
| 25 | 35.63 dB / 1.64 / 90.31 | 35.63 / **1.57** / 90.31 | −4.3% |
| 50 | 40.30 dB / 2.73 / 95.02 | **40.25** / **2.60** / **95.07** | −4.8% |
| 75 | 44.84 dB / 4.53 / 96.58 | **44.64** / **4.31** / **96.55** | −4.9% |
| 90 | 49.89 dB / 7.21 / 97.06 | 49.89 / 7.21 / 97.06 | 0 |
| 100 | — | **PSNR inf** / 12.57 / 97.43 | bit-exact |

Four images at four q, plus q=100 on bbb. At q=75 RGB PSNR is −0.20 dB under PAD-1's decay
fill; PAD-1's own gate was q=80–94 (−0.001 dB). VMAF −0.03. Under both tolerances.

**Sequences, 10 frames ki=9 4:4:4, shipped default `2I+8P+0B`:**

| sequence | q=75 bpp / PSNR / VMAF | q=90 | vs I-only q=90 |
|---|---|---|---|
| crowd_run | 8.39 / 42.47 dB / 99.68 | 13.17 / 49.60 / 99.72 | **+5.3%** |
| old_town_cross | 8.39 / 42.36 dB / 99.38 | 13.04 / 49.59 / 99.70 | **+9.2%** |
| bbb_extended | 3.07 / 43.55 dB / 97.88 | 6.77 / 50.28 / 99.16 | −11.3% |

The withdrawn q=75 I+P+B rows (crowd_run 5.55 bpp / 39.04 dB) are not comparable. On camera
content at q=90, inter costs more than all-intra.

**QUAL-1 ladder re-run, 17 frames, 4:2:0, q=85/92/96/99 vs crf=1/2/4/8:**

| | bbb_extended | old_town_cross | crowd_run | **mean** |
|---|---|---|---|---|
| MEAS-10 | +128.5% | +70.2% | +68.8% | **+89.2%** |
| QUAL-1 | +129.0% | +71.9% | +70.6% | +90.5% |

−1.3 points, direction INTER-2 predicted (only q=85 of four rungs moved). VMAF BD-rate on
old_town is +2548% at 99.8–99.8 — not quoted. RATE-3 is still in flight; this is HEAD without
it. No codec change. No fps.

### CANARY-1 — Encode time must move across GPU tiers (**DONE 2026-09-07 — PASSES at 34x**)

**Answer: encode time moves 34.5x across device tiers.** The failure signal was the two devices
landing within ~15% of each other. Measured on an NVIDIA RTX 4000 Ada (Ubuntu 24.04.3, driver
580.173.02, wgpu 24.0.5), `scripts/gpu_tier_bench.py --tier`, bbb_1080p pinned at
`f83f355f…02bf`, from `main` plus BUG-25's lazy-pipeline change:

| device | backend | encode | decode | settle | spread |
|---|---|---|---|---|---|
| **NVIDIA RTX 4000 Ada** | Vulkan | **13.95 ms** (71.7 fps) | **7.29 ms** (137.2 fps) | 1.02 / 1.01 | **34.46x** |
| llvmpipe (LLVM 20.1.2) | Vulkan | 480.74 ms (2.1 fps) | 373.88 ms (2.7 fps) | 1.00 / 1.01 | |
| RTX 4000 Ada via GL | Gl | no compute support — dropped | | | |

**Reproduced on a second commit and a different machine load**, which is what makes it a
measurement rather than a reading: `07c01b1`+patch at load 3.67 gave 13.95 / 7.29 ms and 34.46x;
`d7353c9`+patch at load 4.5–6.4 gave **14.01 / 7.27 ms and 34.31x**. Encode agrees to 0.4%, decode
to 0.3%. `settle` is the harness's median/best ratio; at 1.00–1.02 these are not clock-ramp
artefacts, and the agreement across a 2x load difference is why.

**What it settles.** The 2011 BeHardware failure this canary exists for — shipping GPU H.264
encoders performing identically on a 100 EUR and a 330 EUR card because they were never
compute-bound — does not describe GNC. The work happens where we think it does.

**What it does not settle, and both halves matter.** The slow arm is **lavapipe, a CPU
rasterizer**, not a weaker GPU: this is a strong statement that GNC is compute-bound and a weak one
about scaling across GPU *tiers*, so the two-real-GPU version is still owed. And it is not MEAS-5 —
nothing here measures concurrency or compares against NVENC.

**Do not quote the M1 comparison as a result.** MEAS-6's ~47 ms encode / ~35 ms decode against
these numbers is ~3.8x on round trip, but that is cross-machine, cross-backend, at a possibly
different q, on a box at load 4.5. The controlled version is one command — run this same harness on
the M1 when the machine is idle — and it has not been run.



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
3. **Inter frames — retracted (BUG-18), and re-measured 2026-09-07 after ARCH-3 closed it.**
   ~~−14.4% mean at q=90~~ — abac's video path was the CPU-entropy P-frame path, and that path
   encoded every P-frame wrong. The comparison put abac on a broken arm.
   **The replacement figure is −12.0% to −22.9%, on nine of nine points at bit-identical pixels**
   (bbb_extended / crowd_run / old_town_cross, q=50/75/90, 18 frames, ki=9, 4:4:4; decoded PNGs
   hashed rather than PSNR compared). With one frame encoder the two coders code the same
   coefficients, so this is a pure rate delta with a quality delta of exactly zero — a stronger
   claim than the retracted one, which was quality-matched to ≤0.03 dB. RESEARCH_LOG 2026-09-07,
   `docs/decisions/0025`. The original retracted text follows for the record:
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

### BUG-39 — `q=100` video: three causes fixed, 12.45 → 51.54 dB, still not bit-exact (todo, P1)

**Third cause found and fixed, and it was not the one `0042` predicted.** `docs/decisions/0054`;
numbers in RESEARCH_LOG. 3 sequences, 8 frames, ki=2 and ki=9, 4:4:4, shipped defaults — inter
PSNR **26.30–26.76 → 51.54–53.62** (crowd_run ki=2), **21.77–26.51 → 50.41–51.54** (ki=9, drift
down the GOP 4.74 → 1.13 dB), and the same shape on old_town_cross (29.7 → 51.8–52.5) and bbb
(33.0 → 58.1). **q=99 is identical to the byte and to two decimals of PSNR on all six points**,
because the fix cannot be reached wherever `wavelet_levels >= 1`.

**Cause 3, fixed: a zero-level `forward` wrote nothing.** `WaveletTransform::forward` runs
`for level in 0..levels`, so at `levels == 0` it dispatches nothing and never writes
`output_buf`; `inverse` copies input to output before its own loop, so the decoder's zero-level
case *is* the identity. `encode_pframe` quantises `plane_c`, so the encoder transmitted whatever
the **previous frame** left there and the decoder added it to its prediction. Reached by every
P/B frame of a `q=100` sequence (MED sets `wavelet_levels = 0`) and of a `--dct` one.

**`0042`'s cause 3 was wrong about the mechanism and is corrected in place** (the original text
kept visible): the P-scale taper is already 1.0 at `q=100` and the dead zone already 0.000, both
printed by the encoder's own canary since INTER-1, so there was nothing to suppress.

**Cause 4, open, and it has a named fix rather than a hypothesis.** 51.54 dB is **sub-pel
prediction rounding**: bilinear quarter-pel interpolation makes the prediction fractional, so
`cur - pred` is fractional and step 1.0 rounds it (≤ 0.5 per sample in YCoCg-R, amplified into
RGB by the inverse colour transform — and bbb reads 58 dB because more of its blocks are full-pel
or zero). The fix is H.264 lossless's: round the prediction to an integer in a lossless
configuration, on both sides, so the residual is an integer and step 1.0 is exact. A `round()` in
`motion_compensate.wgsl` behind a params flag gated on `config.is_lossless()`, which the decoder
derives from the frame header it already carries — **no new bitstream field**. Needs a rate
number too, since rounding the prediction changes the residual. Untried.

**Success criterion unchanged:** every frame bit-exact at `q=100` on ≥3 sequences at ki=2 and 9,
verified outside the harness. **Not met.**

**Canary:** `zero_level_forward_is_an_identity_not_a_no_op` (verified to fail without the fix).

**Probe that looks decisive and is not**, recorded so nobody repeats it: a static sequence (the
same PNG four times) codes `q=100` P-frames bit-exact at 3 198 bytes **before** the fix too —
`all_skip_tiles=120/120`, so it went down the motion-skip path and never asked the transform for
anything. A zero-residual probe cannot test a residual path.

### LOSSLESS-2 — at `q=100` the inter path costs 36% more than all-intra (todo, P2)

**Measured, not argued.** crowd_run, 4 frames, ki=2, `q=100`, after BUG-39's cause-3 fix:

| | bytes |
|---|---|
| I+P | **17 584 089** |
| all-intra | 12 932 312 |

**+36%, and the P-frames are not bit-exact either** (51.5 dB against the I-frames' `inf`). Before
the fix the same comparison read −12%, but that saving was bought by transmitting a stale buffer
instead of the residual, so it was never real.

**Why it goes this way.** With no quantiser to discard anything, a quarter-pel MC residual is
noise-like and costs more to code than the MED-predicted frame it replaces. This extends
INTER-1 / `0023`'s line — the inter saving is already a wash at q=85–99 (−1.9% mean, −0.2%
worst-frame) — past the wash into a loss.

**The question:** should a lossless configuration code P-frames at all, or fall back to all-intra
(per frame, on an RD decision, or per sequence)? It bears on a GOALS §1 row, and the honest
answer may be that `q=100` video is all-intra by construction — which is what FFV1 does.

**Not startable before BUG-39's cause 4**, because rounding the prediction changes the residual
and therefore this number. Take it after, or take both.

### BUG-39 — `q=100` video: two causes fixed, 12.45 → 26.30 dB (superseded 2026-09-08)

**Two of three causes found, fixed and proven.** `docs/decisions/0042`; numbers in RESEARCH_LOG.
crowd_run, 10 frames, `q=100`: P-frames go from **9.06–21.37 dB to 21.63–26.51 dB** at ki=9 and
from 21.35–21.48 to **26.30–26.76** at ki=2, and the drift down the GOP is gone (the ki=9 span
collapses from 12.3 dB to 4.9 dB). Nothing moved at lossy quality — crowd_run q=99 ki=9 is
byte-identical at 49 328 550 B with P-frames 60.61–60.64.

**Cause 1, fixed:** `local_decode_iframe_gpu` called `transform.inverse` unconditionally, so a MED
I-frame's reference was built by inverting a transform it was not coded with. The decoder always
did this right and `med.inverse` always existed; only the encoder's copy lacked the branch.
Measured before: encoder reference against decoder reference differed by up to **33.0 on Y and
64.0 on Cg**, on 63 029 and 64 266 of 65 536 pixels, while the q=99 wavelet control was
bit-identical. Now 0.0000 on every plane, asserted by
`lossless_iframe_reference_matches_the_decoders`.

**Cause 2, fixed:** `encode_pframe` codes its residual with `transform.forward` **always**, but
cloned the sequence config, so a `q=100` P-frame advertised `transform_type = MedPredict` and the
decoder inverted a MED prediction over a wavelet residual. The label now says what the code does.
**This also corrects `--dct` sequences**, mislabelled the same way — not measured, flagged.

**Cause 3, open, and it is a design question rather than a patch.** P-frames at `q=100` are lossy
*by construction*: the residual is quantised at the P-frame taper (up to 1.25× the intra step) with
a dead zone, and `wavelet_levels` is 0. **Nothing in the P-frame path asks to be lossless when the
sequence is.** Fixing it means suppressing the taper and the dead zone for a lossless
configuration, and it needs a rate number as well as a quality one — a lossless P-frame is much
larger. Success criterion unchanged: every frame bit-exact at `q=100` on ≥3 sequences at ki=2 and
9, verified outside the harness.

**The instrument to use, and the reason the first two causes hid for so long:**
`read_reference_planes` exists on **both** pipelines and `test_pframe_reference_matches_decoder`
has been diffing them all along — with `CodecConfig::default()`, qstep 4.0, wavelet. The lossless
case was the untested axis, not a missing tool. Two hours of mechanism hypotheses (`0040`) against
ten minutes of diffing the two things that must be equal.

The original filing follows.

### BUG-39 — `q=100` video decodes at 12.45 dB: lossless sequences have never worked (original filing)

Filed 2026-09-08 by RATE-3, which found it while investigating something else and confirmed it is
**not** caused by RATE-2.

**Measured on `main`, no flags, shipped defaults:** crowd_run, 4 frames, ki=2, `-q 100`. The
I-frames are bit-exact (`PSNR inf`) and the **P-frames that reference them decode at 12.45 and
12.53 dB.** With `GNC_MED=0`, so the I-frames are lossless *wavelet* frames instead of MED, the
same case reads **44.18 and 46.15 dB** — still wrong, and differently wrong, so part of this is
MED-specific and part is not.

**Nothing in the repository recorded it**, and the reason it stayed invisible is worth keeping:
every lossless claim GNC makes is about **stills** — 1.99:1, "10.8% better than JPEG 2000
lossless", the FFV1 gap — and the one sentence that implied video (README's "bit-exact lossless at
`q=100`", in a paragraph about the I/P/B pipeline) is prose carrying no figure, which is exactly
the class DOC-1's two sweeps missed. That sentence is now corrected.

**What is already established** (RATE-3, `docs/decisions/0040`), so this does not start from zero:

- The reference's *quality* is not the limit. A perfect reference yields 52.17 dB P-frames, a
  deliberately broken 34.30 dB reference yields 34.17 dB, and the ordinary 59.5 dB lossy reference
  yields 60.62 dB. **A better reference producing a worse P-frame means the encoder and decoder
  disagree about the reference**, not that quantisation caps it.
- Two mechanisms are refuted with numbers: the colour transform's `floor` vs fractional lifting,
  and a geometry difference at `wavelet_levels = 0`. See `0040` before re-deriving either.
- Taking the reference from the colour-converted source instead of inverting the wrong transform
  moves this from 12.45 to 21.37 dB — broken either way, and reverted.

**The decisive measurement, and it has not been run:** read back the encoder's `gpu_ref_planes`
after a lossless I-frame and diff them against the decoder's own reference for the same frame.
Needs readback plumbing on both sides; gives an unambiguous answer.

**Success criterion:** `-q 100` on ≥3 sequences at ki=2 and 9 decodes **bit-exact on every frame**,
I and P alike, verified outside the harness (raw md5 against the source, the standard `0036` used).
Anything less than bit-exact at q=100 is a failure, not a partial win — there is no quantiser in
that configuration to blame.

**Canary:** the per-frame PSNR line already prints `inf` for a bit-exact frame; the gate is that
every frame prints it.

**Why P1.** It is a shipped codec producing 12 dB video at its highest quality setting. It also
gates RATE-3, and RATE-3 gates the inter half of RATE-2's 21.66%.

### RATE-3 — a bit-exact I-frame is not a drop-in reference (**investigated 2026-09-08, not fixed**, P1)

**Three attempts, two refuted hypotheses, one real fix kept, and a named next measurement.**
`docs/decisions/0040`. The gate `0036` shipped stays; the tree is byte-identical to it on stills
and back to 60.64/60.62 dB P-frames on sequences.

**Kept from this item:** `encode`'s two candidate encodes now run **sibling first, configured path
second**. `local_decode_iframe_gpu` builds an I-frame's reference from the quantised planes
`encode()` leaves on the GPU, so whichever candidate ran *last* decided the reference — with the
sibling second, bbb q=95 (where the lossy file is correctly kept) gave P-frames at **9.83 dB** and
**+40.55%** bytes. That was a latent bug in RATE-2 reachable the moment the gate is lifted.

**What is established and what is refuted is in `0039` and summarised in BUG-39 above. Read one of
them before touching this.** The short version: the reference's quality is not the limit, so the
encoder and decoder disagree about the reference, and the diff between them is the measurement that
settles it. Everything else tried was mechanism-guessing.

**This item is now downstream of BUG-39.** BUG-39 is the same cause seen at q=100 with no fallback
involved and no rate to win, so it is the cleaner place to find it. Take BUG-39 first; if it is
fixed, re-run `scripts/meas_rate3.py` (written for this item) and this becomes a rate question
again.

The original filing follows.

### RATE-3 — a bit-exact I-frame breaks the P-frames that reference it (original filing)

Filed 2026-09-08 by RATE-2, which found it by shipping its fix and testing the sequence path
before believing it.

**Measured.** bbb, 4 frames, ki=2, q=99, with RATE-2's lossless fallback reaching the I-frames:
the I-frames come out **bit-exact as intended** and the P-frames referencing them decode at
**9.80 dB against 60.69 dB** with the fallback off, while the sequence *grows* from 13 078 463 B to
15 276 618 B. A MED I-frame carries `wavelet_levels = 0` and `transform_type = 2`, and the P-frame
path's reference cannot reconstruct from it. RATE-2 therefore refuses the fallback inside every
sequence path, and the intra win stops at the sequence boundary.

**Why this is worth fixing rather than avoiding.** A bit-exact reference is the *best* reference
there is — no drift, no propagated error — so the inter half of RATE-2's win should be larger than
the intra half, not zero. And the defect is not really about RATE-2: it says the P-frame path
assumes its reference came from the wavelet, which is an assumption nothing else states and no
test covers.

**Where to look first.** The 9.80 dB says the reference is garbage rather than merely different, so
this is a wrong-buffer or wrong-geometry bug, not a quality loss. `wavelet_levels = 0` is the
likeliest trigger: the local decode in `encode_pframe`'s neighbourhood reconstructs from
coefficients, and with no subbands there are none to reconstruct from. Check whether the reference
is taken from the *decoded pixels* or from the encoder's coefficient buffer — if the latter, that
is the bug, and it is the same class as BUG-27 (the P-frame local decode used the intra qstep).

**Success criteria.** With the fallback allowed inside sequences: P-frame PSNR within 0.1 dB of
today's on bbb/crowd_run/old_town_cross at q=95 and 99, ki=2 and 9; total sequence bytes **not
larger** than today's on any of those six points; and the I-frames still bit-exact where the
fallback chooses them. Below that, keep the gate.

**Canary:** RATE-2's existing `GNC: RATE-2 lossless fallback` line already prints per frame; add
the frame index and type to it so an I-frame taking the path inside a sequence is visible in the
log rather than inferred from the byte count.

**Why P1.** It is the difference between RATE-2 being a stills fix and a codec fix, it unblocks
BASELINE's 1.9x-against-H.264 re-run (which is *not* unblocked by RATE-2 — see that entry), and
the 9.80 dB shape suggests a bug with a single cause rather than a tuning problem.

### RATE-2 — the top of the ladder codes both ways and keeps the smaller (**FIXED 2026-09-08**)

**Shipped.** `docs/decisions/0036`, numbers in RESEARCH_LOG "RATE-2 — the top of the lossy ladder
now codes both ways". At q = 95..=99 a still encode codes the wavelet path *and*
`lossless_sibling(config)` and returns whichever is smaller. Rate against the file the same
command produced before, mean of the four stills: **−1.33% / −4.67% / −10.39% / −15.65% / −21.66%
at q = 95/96/97/98/99**, and **12 of those 20 points became bit-exact** from 52.5–60.1 dB. All 20
choose correctly against the measured dominance boundaries (bbb q=98, blue_sky q=95, kristensara
and touchdown q=96). Verified outside the harness with `gnc encode` → `gnc decode` → raw RGB md5,
including the point where the lossy file is correctly kept.

**No format change and no GP version:** `transform_type` is a header byte independent of the
quality byte, so a q=97 file carrying `transform_type = 2` decodes on every existing build.

**It is intra-only, and RATE-3 says why.** Two things it does not do, both deliberate: `--dct` is
refused (an explicit third transform, caught by `test_block_dct_quality_preset` going red), and so
is any frame inside a sequence — a MED I-frame breaks the P-frame reference at **9.80 dB against
60.69 dB**. Sequence output is byte-identical either side of the commit.

**Two measurement consequences.** RD ladders now flatten at the top, correctly; and a BD-rate over
a ladder reaching q≥95 integrates over fewer points than before, so **do not compare a BD-rate
across this commit**. BASELINE's 1.9x-against-H.264 caveat is *updated, not lifted* — that ladder
is video, and the rung to re-run it against is RATE-3.

The original filing follows.

### RATE-2 — Above q≈95-98 the lossy ladder costs more than bit-exact lossless (original filing)

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

### INTER-1 — The inter path is a loss at contribution quality; decide what it is for (**DONE 2026-09-07** — the premise was 60% a bug; no default changes)

**Answer: the title is wrong, the default ki is the best of four, and TUNE-6's taper was right for
a reason nobody had measured.** `docs/decisions/0023`. At q=85-99 the shipped configuration is
**-1.9% mean / -0.2% worst-frame** BD-rate against all-intra, not a loss. Three findings:

1. **ki sweep (step 1): BD-rate improves monotonically with GOP length**, so ki=9 wins the
   3-sequence mean on both metrics (ki=2 -1.30%/+0.50%, ki=4 -1.73%/+0.03%, ki=9 -1.90%/-0.20%).
   The item predicted the win would be confined to short GOPs; it is at long ones. The sign is set
   by *content* — camera sequences lose at every ki, animation wins at every ki — so no default
   fixes both.
2. **BUG-27 found and fixed inside the item.** `encode_pframe`'s local-decode *dequantise*
   dispatches used the intra quantiser step while its quantise dispatches used `res_qstep`, so the
   encoder's reference disagreed with the decoder's whenever `p_qp_scale != 1.0` — i.e. on the
   default path at **all q <= 80**. Worth up to +1.82 dB mean / +3.62 dB worst-frame for +2.0%
   bytes. **Invalidated MEAS-3 and decision 0019** (mean +4.6% -> **-0.3%**, worst-frame +19.1% ->
   **+8.0%**), **TUNE-5** and **TUNE-6's own justification**. Byte-identical at q >= 85 (27/27
   verified), so nothing at the contribution operating point moved.
3. **P-scale priced properly (step 2): the taper stays, both endpoints now justified separately.**
   Below q=70, 1.25 is worth 3.7 points of mean BD-rate over a flat 1.0. Above q=85, 1.0 is kept
   even though 1.25 is 1.1 points *better* on worst-frame BD-rate, because at 1.0 the worst frame
   in a GOP equals all-intra's (59.50 vs 59.49 at q=99) and every coarser scale drops it (57.05 at
   1.25, 54.99 at 1.50). A ceiling guarantee, not a rate/quality trade — and invisible to a
   common-interval BD-rate, which is common *because* the coarse arms stop early.

**Step 3's decision: nothing changes.** ki stays 9, the taper stays as-is, and inter stays a
default rather than becoming opt-in — the case for demoting it was 0019's figure, and ~60% of its
worst-frame penalty was BUG-27. What remains is content-specific (old_town_cross +28.7%
worst-frame) and MEAS-4 already located it in prediction quality, not the coding model.
Harnesses: `scripts/meas_inter1_ki.py`, `scripts/meas_inter1_pscale.py`. Follow-up: **INTER-2**.

### INTER-2 — The inter dead zone is a large unpriced lever at the q=85 rung (**DONE 2026-09-08 — default 2.0 → 1.0, BD-rate −4.77%**)

Found inside INTER-1, not chased there. The q=85 rung behaves unlike every rung above it: the
inter arm goes *cheap and worse* (10.50 bpp against all-intra's 11.64, 2.9 dB down on the worst
frame) while from q=90 up it goes *dearer and slightly better*. The P-scale is 1.0 across all of
it, so the scale is not the cause. `inter_dz_mul` doubles the inter dead zone, and the intra dead
zone falls 0.5 -> 0.05 between q=85 and q=92, so the inter dead zone goes 1.0 -> ~0.1 across
exactly that boundary — large enough at q=85 to zero coefficients that matter, gone above q=90.

One **point** measurement, crowd_run q=85 ki=9 24 frames, which sizes the lever and cannot rank
the options (COORDINATION rule 4):

| | bytes | mean / worst-frame |
|---|---|---|
| `inter_dz_mul=2.0` (shipped) | 65293226 | 45.02 / 44.61 dB |
| `inter_dz_mul=1.0` | 74831067 | 47.89 / **47.48** dB |
| all-intra reference | 72379589 | 47.48 / 47.48 dB |

At 1.0 the worst frame lands exactly on all-intra's 47.48 — the same ceiling property the P-scale
has at 1.0, arrived at from a second knob, which is the interesting part. 12.7% of the rate and
2.87 dB of worst-frame is worth a BD-rate.

> **Priced 2026-09-08 and shipped: `inter_dz_mul` is now 1.0.** `docs/decisions/0043`. The point
> above reproduces byte-for-byte on today's `main`, and the ladder says the same thing everywhere.
>
> 4 rungs (q=70/75/80/85) x 3 sequences x 4 arms, 24 frames, ki=9, 4:4:4. BD-rate on PSNR against
> the old 2.0, integrated over each sequence's common interval across all arms:
>
> | sequence | mul=1.5 | **mul=1.0** | mul=0.0 |
> |---|---|---|---|
> | bbb_extended | −2.70% | −2.13% | **+12.48%** |
> | crowd_run | −3.01% | **−6.04%** | −2.40% |
> | old_town_cross | −2.76% | **−6.14%** | −3.34% |
> | mean | −2.82% | **−4.77%** | +2.25% |
>
> **Worst-frame improves at 12 of 12 points, by +2.44 to +5.23 dB**, and 1.0 beats both
> neighbours, so the optimum is bracketed. **0.0 is worse than shipped on animation (+12.48%)** —
> the inter dead zone earns its place, 2.0 just overshot.
>
> **VMAF could not decide this and said so.** crowd_run's four rungs span 99.86–99.88 — a
> 0.02-point interval across a 5.5 dB PSNR spread — and a BD-rate over it returns +35.41%. The
> q≤85 "VMAF leads" rule is a stills rule; this ladder runs at 4.9–12.0 bpp.
>
> **Scope measured, not argued: q ≤ 88 moves, q ≥ 89 is byte-identical**, and q=100 is byte-identical
> both ways. Also consolidated three inlined copies of the factor into
> `gnc::inter_dead_zone_mul()`, guarded by a test — it was one `unwrap_or` from BUG-37's shape.


**What to do:** BD-rate `inter_dz_mul` in {1.0, 1.5, 2.0, 3.0} against all-intra on the three
sequences at q=85-99 *and* q=25-70, mean and worst-frame, via `scripts/meas_inter1_pscale.py`
(it takes any `meas_inter1_ki.py` CSV and normalises the arms onto one interval — the dead zone
lowers the ceiling too, so the same trap applies). Report the top rung beside the BD-rate.
**Success criterion:** >=3% worst-frame BD-rate at q=85-99 without lowering the q=99 worst frame
below the all-intra figure. If the answer is "1.0 above q=85, 2.0 below", that is a second taper
and should be keyed on the quantiser step like the first one, not on `q`.

### INTER-1 — original statement, **SUPERSEDED** by the entry above and `docs/decisions/0023`

> Kept for the reasoning in it; carries no priority marker on purpose, so `scripts/claim next`
> cannot hand it out again.

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

### INTRA-1 — Where is the remaining 27%? (**ANSWERED 2026-09-08** — 26.2 of the 27.1 points named)

**Closed as a queue entry, not as a subject.** The item's own success criterion was "name where the
27% goes, with a number per cause that sums to roughly the measured gap", and step 3 met it: 8.5
chroma allocation + ≤7.5 entropy + 6.6 padding + ~3 dead zone + 0.6 tiling = **26.2 of 27.1**, with
~0.9 unexplained. It loses its priority marker so `next` stops offering a P0 whose remaining work
is 0.9 points of residue — **the work that is left lives in its two descendants, both filed with
numbers and gates:**

- **PAD-1 (P1)** — the fill change, −4.5% of shipped intra rate, gated on an inter measurement.
- **INTRA-2 (P1)** — the dead zone on I-frames only, ~3 points, blocked on the P path.

**15.1 of the 27.1 points are not coding deficiencies at all** (8.5 chroma allocation, 6.6
padding), so the honest form of the intra coding gap on these four images is **closer to +12% than
+27%**. Anyone re-opening this should start from that number, not the headline one.

**One pin worth carrying forward.** RATE-2 (`a7273ab`) made q=95..99 code both ways and keep the
smaller, so **any still figure here taken on a ladder reaching q>=95 is pinned to pre-`a7273ab`
code** — that includes step 1's and step 2's q=85..99 ladders. Nothing is retracted; they were
right for the code they ran on and will not reproduce byte for byte today. Step 3's figures were
re-measured on `c84fbd5` at q<=94 and moved by at most 0.25 points, with the headline projection
identical.

Decisions `0024`, `0026`, `0027`, `0028`, `0034`.

**The question.** With `--abac` on, GNC needs **+27.1% more bits than JPEG 2000 in 9/7 mode** at
matched RGB PSNR, and **+48.3%** at matched Y-PSNR (ENT-4, four images, one metric path). J2K uses
**the same transform at the same depth** — CDF 9/7, five levels. Half the original 54.2% gap was the
entropy coder and is now closed. **Nothing in this repository accounts for the other half.**

That makes this the biggest known compression gap in GNC, and it is not a tuning item: something
structural is costing a quarter of the bitrate and we cannot currently name it.

**ANSWERED 2026-09-08 (step 3). 26.3 of the 27.1 points are named**, and 15.2 of them are not
coding deficiencies at all: 8.5 is a deliberate chroma allocation the RGB metric cannot see
(`0026`) and **6.6 is GNC coding its own tile-alignment padding** (`0034`), which JPEG 2000 in
whole-picture mode does not do. The coding half is ≤7.5 entropy (`0024`), ~3 dead zone (`0028`,
filed as INTRA-2) and 0.6 tiling. **The one actionable piece is PAD-1: −4.6% of shipped intra rate
from a fill change, at unchanged visible quality.** The question below is kept as written because
the shape of the answer only makes sense against it.

**What is already ruled out, so nobody re-measures it:**

- **PCRD / rate allocation.** 0.00 dB at code-block granularity, at every rate from 0.05 to 3.5 bpp,
  with a structural reason (uniform scalar quantisation of a near-orthonormal transform under MSE
  puts every coefficient at the same RD slope). EBCOT part 1, closed.
- **Coefficient-level RDOQ.** +0.1%.
- **A richer entropy context.** The offline model puts EBCOT's full neighbourhood at −16.4% against
  abac's vertical-only −11.7%, so the remaining context headroom is single digits — and it costs the
  256-way parallel decode. It cannot be 27 points. Worth doing eventually; not the answer.
- **Tile geometry, as far as it has been measured.** 512 over 256 is −0.91%, and six levels at tile
  512 is −0.1% (BUG-11/BUG-12). Note both stop at 512 and neither tested a whole-frame transform.

### Step 1 — **DONE 2026-09-07. The gap is upstream: entropy coding can account for at most 28% of it.**

**Measured: GNC spends 7.5% more than the entropy of its own coefficients at q >= 85** (5.3%-10.3%
across 16 points; 8.8% over all 24, including the lossy end where the coder is looser). The gap to
J2K 9/7 is 27.1 points; taking every one of those 7.5 points out leaves **+17.6%**. So **entropy
coding accounts for at most ~7.5 of the 27.1 points (28%) and ~19.6 points (72%) is upstream.**

Tool: `src/encoder/coef_entropy_diag.rs`, `GNC_COEF_ENTROPY=1`. It decodes the **shipped** abac
tiles and prices the coefficients they carry against six models, per plane and per subband —
`H0`, abac's own context (`Hctx`), and three richer neighbourhood models. Four images, six quality
points. Per-band rows cover 99.95% of the abac payload, which is the coverage canary.

**Why the bound holds.** `Hbig` widens the context template 4x and finds **nothing**; `Hnb0`
charges no model cost at all and buys 1.6%. And it agrees with the existing offline estimate by a
completely different method (full-neighbourhood EBCOT −16.4% vs abac's −11.7% relative to Rice is
5.3% apart; this measures 4-7%).

**Where the coder's own 7.5% is**, and it is not the context model: on the bands that get a full
64x64 code-block (82% of the rate) abac is within **+4.1%** of the bound; on LL and levels 3-5,
whose blocks are 32/16/8px, it is **+25.9%** — 54% of the headroom on 18% of the rate. That is
cold-start on blocks too small to adapt, filed as **ENT-6**, worth about 4% of the file.

Full numbers in RESEARCH_LOG 2026-09-07 and `docs/decisions/0024`.

### Step 1 — the measurement that splits the gap in two (as originally specified)

**Compare what GNC spends against the entropy of GNC's own coefficients**, per subband, at a rate
matched to J2K. Both quantities are computable from a single encode:

- **If GNC spends close to its own coefficients' entropy**, the coder is done and the problem is
  *upstream*: the coefficients themselves are more expensive than J2K's. Then the suspects are
  quantisation (deadzone shape, per-subband step derivation from band gain), the lifting
  implementation's normalisation, and the tiling.
- **If GNC spends materially more than that entropy**, there is coder headroom left after abac, and
  the context model is worth more than the offline estimate suggested — which has now happened twice
  (the offline model understated abac by 1.7x).

One number, and it decides which of two entirely different investigations to run. `scripts/` already
has the machinery: `meas_ebcot_context.py` computes conditional entropy per subband against a
faithful simulation of the shipped coder, and `GNC_DIAGNOSTICS=1` reports per-group counts.

### Step 2 — **first instalment DONE 2026-09-07. Three of four candidates measured; ~9.8 points left.**

**The harness reproduces ENT-4's headline exactly** (`J2K 9/7` reads +27.1% RGB, +48.3% Y), so the
baseline is reproduced rather than assumed.

| cause | RGB points | status |
|---|---|---|
| chroma allocation against an RGB metric | **8.5** | measured; a deliberate perceptual trade, **not** a coding deficiency |
| entropy coder headroom | **≤7.5** | decision 0024; half is ENT-6's small-block cold start |
| **tile-alignment padding** | **6.6** | **step 3, decision 0034 — samples GNC codes and J2K does not; also not a coding deficiency, and 4.6 points of it are recoverable (PAD-1)** |
| dead zone / quantiser rounding rule | **~3** | step 2c, decision 0028 — intra-only, filed as INTRA-2 |
| tiling, 256px vs whole picture | **≤12.4 for J2K, 0.6 realisable in GNC** | measured in both codecs |
| lifting normalisation | **~0** | closed — synthesis norms 0.984–1.066, uniform steps within ~5% of MSE-optimal |
| cross-tile rate allocation | **0.95, rejected** | step 2b, decision 0027 |
| tile-boundary extension | **0** | already whole-point symmetric; the candidate was wrong |
| remainder | **~0.8** | **26.3 of 27.1 named** |

In sequence: +27.2% → +18.7% (chroma) → +9.8% (entropy ceiling) → **+3.1%** (padding) — and the
dead zone is roughly what is left. **The honest form of the intra coding gap on these four images
is closer to +12% than to +27%**, since 15.2 of the 27.1 points are allocation and padding rather
than coding.

**The colour transform is where the normalisation error is.** YCoCg-R synthesis norms are
(Y 1.7321, Co 0.7071, Cg 0.8660), spread **2.45**, so RGB-MSE-optimal chroma steps are 2.45x/2.00x
the luma step; production uses **1.2**. J2K's ICT has spread 1.15 and is therefore near-optimal for
the metric for free. Measured: cw 2.0 is −7.0% RGB / −19.3% Y BD-rate and takes the gap to +18.7%
RGB / +21.5% Y; the RGB gain **saturates at 2.0–2.45**, exactly where the norms put the optimum, and
the Y gap collapses from +48.2% to **+13.5%** — which is what explains why the luma gap always
exceeded the RGB gap. **Not a proposal to change the default:** mean dE00 at q=90 goes 0.445 → 0.585
(+31%), and CHROMA-1 chose 1.2 on a colour-aware criterion. What it changes is how the gap is
*quoted* — "+27.1%" overstates the coding deficiency by about a third, "+48.3%" by more than two
thirds.

**Tiling does not transfer.** J2K given GNC's 256px tiling goes +27.1% → +14.7%; GNC given a 512px
tile gains **−0.6% RGB / −0.9% Y** against J2K's 3.8% for the same step — a 6x asymmetry. The
leading hypothesis for the difference is **global rate allocation**, not wavelet reach: untiled J2K
runs PCRD over the whole picture, GNC has no cross-tile allocation at all above q=80 (AQ is 30–80).
EBCOT part 1 closed PCRD *inside* a tile at code-block granularity (0.00 dB); cross-tile allocation
is a different lever and has never been measured. **That is the next test, and it is cheap.**

**Measurement trap, recorded so nobody repeats it:** a full-frame tile-size comparison is confounded
by padding — 1920x1080 pads to 2048x1280 at tile 256 and **2048x1536** at tile 512, 20% more
coefficients, reading **+6.1% for tile 512** at identical PSNR when the real effect is −0.6%. Use
content that is a multiple of both tile sizes; this used centred 1024x512 crops.

**Step 2b — cross-tile rate allocation, REJECTED 2026-09-07.** The asymmetry above had one obvious
explanation: untiled J2K runs PCRD across the whole picture and GNC has no cross-tile allocation
above q=80. Measured with an oracle (`scripts/meas_cross_tile_rd.py`, `GNC_TILE_RATE=1`): a
clairvoyant allocator that sees the future, pays nothing to signal a per-tile q and may pick any
rung for any tile saves **0.95% at q>=85** (bbb −1.60%, blue_sky −1.19%, kristensara −0.14%,
touchdown −0.88%). Pre-declared floor was 3 points. **Rejected**, and the mechanism is visible: on
kristensara the oracle picks a *single* q for all 15 tiles at q = 92/94/96/98. Uniform q is not
close to optimal there, it is the optimum — EBCOT part 1's argument one scale up. Decision
`docs/decisions/0027`.

**Where the item stands after step 3: the accounting closes.** The two candidates this paragraph
used to list are both settled and neither was worth what was left. The dead zone is ~3 points and
is INTRA-2. **The tile-boundary handling was never a candidate: `transform_97.wgsl` already
implements whole-point symmetric extension** — the lifting steps substitute `low[half-1]` for
`low[half]` and `high[0]` for `high[-1]`, which is `x[N] = x[N-2]`, `x[-1] = x[1]` exactly, and the
inverse pass substitutes the same values. Settled by step 2c and re-derived by step 3; **0 points**.
The "boundary replication" comment at `transform_97.wgsl:78` is about the image-edge clamp in the
*overlap* load path, which does nothing on the default path (`overlap_pixels` is 0 and the plane is
already a whole number of tiles).

**The candidate pointed at the right phenomenon in the wrong place, and step 3 found it.** GNC does
replicate its picture edge, and it costs **6.6 of the 27.1 points** — but it happens in
`pad.wgsl`, on pixels, before the transform runs. See step 3 below.

Full numbers in RESEARCH_LOG 2026-09-07 and `docs/decisions/0026`, `0027`. Also found on the way:
**BUG-26** (fixed) — `--tile-size 1024` silently destroyed the image, and so did any tile size not
divisible by `2^levels`.

### Step 3 — **DONE 2026-09-08. The remaining ~6 points are padding, and 4.6 of them are recoverable.**

**GNC pads every plane up to a whole multiple of `tile_size` with edge replication
(`src/shaders/pad.wgsl`) and codes the padded plane.** A 1920x1080 frame is coded as **2048x1280 —
26.4% more coefficients, 20.9% of the coded samples outside the picture**, and the decoder crops
them. OpenJPEG in whole-picture mode codes 1920x1080 exactly. Both arms are divided by the
*visible* pixel count, so the GNC arm has carried a tax the J2K arm does not, in every cross-codec
figure since MEAS-9. RESEARCH_LOG had padding only as a **tile-size comparison** confound, which is
a different question.

**Canary first, because everything rests on one mechanism.** Rebuild the padded plane in Python,
encode it as a picture in its own right: `bbb_1080p` 1920x1080 -> 2048x1280 gives **1 793 794 B
against 1 793 794 B, identical** — and that is the figure this file already records for bbb at
q=90 with `--abac`. Same for kristensara (554 346 B both ways).

**Part 1's own control, measured rather than argued:** JPEG 2000 codes both crops exactly as given,
so its `A` -> `D` BD-rate is the content term — **−0.13% to +0.02% RGB** across the four images,
against GNC's +11.6% to +25.2% on the same pair. Under 1% of the effect.

**Two methods, means agreeing to 0.02 points:**

| | method | mean over the 4 ENT-4 images |
|---|---|---|
| projected from a content-controlled crop pair | GNC only, content fixed, quality held to 0.005 dB | **+6.60%** |
| drop in the cross-codec gap, native against padding-free | GNC `--abac` vs J2K 9/7, one ladder | **+6.70 points** (26.54 -> 19.83 RGB, 51.17 -> 40.66 Y) |

Per image, projected against measured: bbb 7.90/7.44, blue_sky 7.54/8.68, kristensara 2.65/2.25,
touchdown 8.30/9.53. **kristensara is the discriminating case** — it pads only 720 -> 768 and reads
a third of the 1080p figure in *both* methods, which a measurement of the crop's content change
would not do.

**Two thirds of it is a fill choice.** The padded samples are don't-care, so edge replication is a
choice; the canary means alternatives can be priced with no code change. BD-rate against the
shipped fill at unchanged visible quality (bbb q=90: 50.060 dB flat against 50.061 dB replicate):

| fill | mean RGB | mean Y |
|---|---|---|
| replicate, then fade to one scalar over 8 px | **−4.48%** | −4.52% |
| one scalar everywhere outside the picture | −4.34% | −4.42% |
| the same fade over 32 px | −4.12% | −4.14% |
| whole-point mirror of the picture into the padding | **+11.41%** | +11.28% |

So **replication was already the better of the two textbook extensions** — mirroring copies real
detail into the padding and costs 11.5 points more. What matters is being *flat in the direction of
extension*, not smooth at the seam.

**Not shipped here, and the reason is not in these numbers:** the decoder keeps the padded region
in the reference buffer and motion compensation reads it for edge blocks, where edge replication is
the standard choice. Filed as **PAD-1** with the inter gate named. The ceiling above a fill change
is 6.6 points, not 27, and reaching it means partial border tiles.

Harness `scripts/meas_intra1_padding.py`; full numbers in RESEARCH_LOG 2026-09-08 and
`docs/decisions/0034`.

### Step 2 — the candidates as originally specified, cheapest first

1. **Whole-frame transform vs 256px tiles.** GNC transforms 35 independent tiles on a 1080p frame;
   J2K transforms the whole picture. Each tile's LL is 8x8 at five levels, against a 60x34 LL for
   the frame — so GNC cannot exploit low-frequency correlation beyond 256 px, and codes 35
   independent DC structures. Measured up to 512 the effect was ~1%, but nobody has measured a
   single-tile frame. **Test without touching the architecture**: `CodecConfig::set_tile_size()` to
   the frame size, at the level ceiling that then applies, and BD-rate it. If it is worth a lot, that
   is a genuine conflict with GOALS rule 3 (no cross-tile dependencies) and belongs in a decision
   record, not in a commit — tile independence is what buys the parallel decode, the error
   resilience and the seeking.
2. **Quantiser shape and per-subband step.** J2K derives each band's step from the band gain of the
   irreversible 9/7 and uses a deadzone quantiser. GNC uses adaptive quantisation with perceptual
   subband weights, which are tuned for perceived quality and are being scored here on PSNR. **That
   mismatch alone could account for several points**, and it is measurable offline: re-quantise GNC's
   coefficients with J2K's band-gain-derived steps and re-measure rate at matched PSNR. Careful — a
   PSNR-optimal reweighting may be a perceptual regression, so any change needs dE00 and a VMAF
   cross-check at q<=85 before it ships.
3. **Lifting normalisation.** The 9/7 lifting steps must carry the correct scaling for the transform
   to be near-orthonormal; an error costs rate directly and silently. This repo has been here before
   ("an unnormalised lifting DWT loses to an orthonormal DCT on scaling alone" moved a result from
   41% better to 4% better). Cheap to verify: transform a delta and check subband gains against the
   published 9/7 values.
4. **Code-block geometry inside abac.** cb=64 was chosen as the best of the sizes measured for the
   rate/speed trade, not for rate alone. Re-sweep now that the coder is in the bitstream.

**Success criterion for the item as a whole:** name where the 27% goes, with a number per cause that
sums to roughly the measured gap. "We tried four things and got 3%" is an acceptable outcome only if
it comes with an account of the missing 24 — an unexplained gap is a standing invitation to guess,
and this repo has a long record of what guessing costs.

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

### BUG-26 — `--tile-size 1024` silently destroys the image (**FIXED 2026-09-07**)

Found 2026-09-07 by INTRA-1 step 2 while pricing the tiling candidate. **Nothing errors.**

```
gnc encode -i bbb_1080p.png -o t1024.gnc -q 90 --abac -t 1024   # exit 0
gnc decode -i t1024.gnc     -o t1024.png                        # exit 0
```

The result is **7.19 dB RGB PSNR at 901 865 bytes**, against 50.06 dB at 1 793 794 bytes for the
default tile 256. Half the bytes, and the picture is gone. No warning, no clamp, no assertion — the
CLI advertises `-t <TILE_SIZE>` with no documented range.

**Cause.** `src/shaders/transform_97.wgsl` declares `var<workgroup> shared_data: array<f32, 512>`
with `shared_low`/`shared_high` at 256, and says so in a comment sized for tile 256:
"MAX_OVERLAP = tile_size/2 - 1 = 127; physical_tile_size_max = 510. Workgroup size 256 covers
half_max=255 with one thread to spare." A 1024px tile needs 1024 elements and 512 threads. WGSL
out-of-bounds workgroup access is not a trap, so it reads and writes whatever is there and the
transform quietly returns nonsense that still round-trips through a valid bitstream.

**Tile 512 is fine** — verified 50.07 dB against tile 256's 50.06 dB on the same input, and −0.6%
RGB / −0.9% Y BD-rate on padding-free crops. So the ceiling is real but it is 512, not 256.

**FIXED 2026-09-07.** It was **two** silent defects, not one, and the boundary was measured rather
than guessed (kristensara_720p at q=90): 64/96/128/160/192/256/320/384/448/512 all read 49.6-49.7 dB;
**504 reads 41.0, 520 reads 24.8, 640 reads 11.2, 1024 reads 7.5, and 260 reads 20.1.**

- **Above 512** the shader reads past `array<f32, 512>` in workgroup memory.
- **Not divisible by `2^levels`**: `max_wavelet_levels()` derives its ceiling from `tile_size / 8`
  under *integer* division, so 260 (= 4 x 65) is handed five levels when only two halvings are clean.

`set_tile_size` now refuses both, naming the limit and the measured evidence — **refused, not
clamped**, because a clamp encodes something other than what was asked for, just as quietly.
`MIN_TILE_SIZE` / `MAX_TILE_SIZE` are public constants carrying the reason. Three tests in
`tests/tile_size_levels.rs`, one of which asserts every measured-good tile size is still accepted so
the guard cannot cost a working configuration. Tile 256 and 512 output byte-identical before and
after. One existing test row — `(1024, 7)` in `ceiling_follows_the_tile_size_in_use` — was asserting
the level arithmetic of a configuration that destroys the picture, and is removed.

**Original fix note**: reject a tile size the shader cannot serve, at config validation, with the
limit named in the error. Raising the ceiling is a separate and much larger job (the shader would need a
multi-pass or a larger workgroup) and is **not** what this item asks for — INTRA-1 step 2 measured
that GNC gains only 0.6% from 256 -> 512, so there is no rate case for chasing 1024 today.

**Canary:** the same encode must exit non-zero with a message naming 512, and `cargo test --release`
must carry a test that asserts it.

**Related, and worth knowing before measuring any tile-size question:** a full-frame tile-size
comparison is confounded by padding. 1920x1080 pads to 2048x1280 at tile 256 and to **2048x1536**
at tile 512 — 20% more coefficients — which reads as +6.1% rate for tile 512 that is entirely
padding and reverses the sign of the real effect. Measure tile size on content that is a multiple
of both sizes; `1024x512` centre crops are what INTRA-1 used.

### PAD-1 — GNC codes its own tile padding (**DONE 2026-09-08** — shipped for stills, refused on references)

**Shipped: −4.63% RGB / −4.60% Y of intra rate on stills, at unchanged visible quality, and
nothing changes for video.** `pad.wgsl` now replicates the picture edge and fades to flat over
8 px; the fade target is 8 strided samples of the edge line, which reproduces that line's exact
mean while needing no reduction, no per-frame uniform and no host pass. Measured on the *shipped
encoder* against `GNC_PAD_FILL=replicate`, four stills, q=80..94, `--abac`: bbb −5.86%, blue_sky
−4.99%, kristensara −1.75%, touchdown −5.90%. **The offline oracle predicted −4.63% / −4.60% —
agreement to two decimals on both metrics from independent implementations.**

**The inter gate failed and that is why this is not the default everywhere.** At ki=9 with the fill
forced on: crowd_run and old_town_cross move worst-frame PSNR by **+0.000 dB** for −6.9% to −10.1%
of rate, but **bbb_extended loses 2.47 dB of worst-frame PSNR at q=85 and 4.03 dB at q=92** — the
same shape as INTRA-2's dead zone, mean −2.28 dB against worst −4.03 dB. The control: **the same
clip at ki=1 loses nothing** (−5.65% of rate, PSNR identical), which pins the cause to the MC
reference rather than to the coding. Motion compensation is handed the *padded* dimensions and
clamps its reads to them (`src/decoder/gpu_work.rs:478`), so an edge block whose motion vector
points outward predicts from the fill.

So the policy is per-path and fail-safe: `quality_preset` opts in, `CodecConfig::default()`
refuses (the sequence path builds from it), all four `main.rs` funnels that already refuse RATE-2's
`lossless_fallback` refuse this too, both sites where the sequence encoder codes an I-frame through
`EncoderPipeline::encode` clear it, and every other padding dispatch there asserts replication.
**Sequence output is byte-identical to the pre-PAD-1 encoder.**

**Canaries, because a fill writes pixels nobody looks at and is silent by construction.**
`GNC_DIAGNOSTICS=1` prints which fill each path took and why;
`scripts/meas_intra1_padding.py --canary` checks the shader byte-for-byte against an independent
Python reimplementation and passes in *both* modes; `scripts/meas_pad1_inter.py` keeps three arms
and its forced-on arm is **expected to regress**, as the standing guard on this decision.

Two defects were found by verifying rather than assuming, and both would have shipped the 4 dB
loss: the pad uniform is shared and persistent so a sequence inherited `decay` from a previous
still (`benchmark-sequence` calls the still path several times per run), and sequence configs built
from `quality_preset` inherited the flag. Decision `docs/decisions/0040`.

**Still open, filed as PAD-2:** the inter half. Having the decoder re-replicate the picture edge
after reconstruction would let the encoder write a cheap fill while the reference stays
MC-friendly, collecting the remaining ~4.6% on video too — but it changes the decoding process and
needs a bitstream version.

### PAD-2 — collect the padding fill on inter, by re-replicating in the decoder (todo, **P2**)

Filed 2026-09-08 by PAD-1, which shipped the still half and measured exactly why the inter half
does not follow. **Worth roughly −7% to −10% of sequence rate** (that is what forcing the fill on
already measures), blocked on a bitstream-visible change.

**The problem in one line.** The fill is cheap to *code* and bad to *predict from*, and today both
sides use the same padded plane for both purposes.

**The shape that resolves it.** Let the encoder write the cheap fill, and have **both** sides
overwrite the padding with edge replication of the *decoded* picture before it is used as a
reference. Encoder and decoder then agree, the coded padding stays cheap, and motion compensation
sees a plausibly extended picture again. It changes the decoding process, so it needs a bitstream
version and old streams must keep the old behaviour.

**What is already measured, so nobody re-derives it:** the forced-on arm of
`scripts/meas_pad1_inter.py`, re-measured on `main` with INTER-2: crowd_run +0.000 dB at both q,
old_town_cross −0.900 / −0.490 at q=85 and +0.000 at q=92, bbb_extended −2.470 at q=85 and
**−4.030 at q=92**, mean rate −8.80%. **6 of 12 points regress, up from 3 before INTER-2** — a
lower inter dead zone stopped quantising away part of the prediction error the fill causes, so the
first figures understated it. Plus the ki=1 control that isolates the cause. **Success criterion:**
the rate win of the forced-on arm with worst-frame PSNR within 0.3 dB of replication on all three
sequences, both chroma formats.

**And there is prior art for exactly this split, which is worth trying before the decoder change.**
The Dirac specification (v2.2.3, §13.1.2 Note) recommends **edge extension for intra pictures and
*zero* extension for inter pictures**, and Schroedinger implements precisely that — it calls
`schro_frame_zero_extend` on the inter path and edge-extends on intra
(`schroencoder.c:2442-2453`). VC-2's copy of that Note dropped the inter clause only because VC-2
is intra-only. So a codec in this family already treats the two cases differently, which is what
PAD-1 concluded from measurement.

The mechanism is not the one PAD-1 tested, and that is why it is interesting: PAD-1 asked "which
fill predicts best", and zero extension instead makes the **padding's own residual** vanish, since
a zero-padded reference against a zero-padded current frame differences to exactly nothing. It says
nothing about visible edge blocks whose motion vectors reach outward, which is where PAD-1's 4.03 dB
went — so it may well not help, but it is cheap to measure with `GNC_PAD_FILL` extended by a
`zero` arm and it is the one candidate here with a shipping implementation behind it.

**Cheaper thing to check first, and it may make PAD-2 unnecessary for most content:** the loss is
concentrated on one clip. If it is edge blocks with outward motion vectors specifically, then
clamping MC's reads to the *visible* bounds instead of the padded ones is a much smaller change
than a bitstream version — and it is a question about `motion_compensate.wgsl`, not about padding.

**And the bigger prize behind both of these is partial border tiles, which is now priced from the
other side.** "Why not just use a tile size that divides the frame?" is the obvious question, and
it has a measured answer as of 2026-09-08 (`0039`): such a size exists — `gcd(W, H)` is 120 for
1080p, 80 for 720p, 240 for 2160p — but **no tile size divisible by 32 divides 1080, 720 or 2160**,
so none of them can carry five wavelet levels. Broadcast heights are not power-of-two friendly
(1080 = 8 x 135). Measured on bbb_1080p at q=90: tile 120 with zero padding and three levels costs
**+81% of rate** against tile 256 with 20.9% padding and five levels. So the padding is by far the
cheaper evil, and **having to choose at all is the defect** — partial border tiles the way JPEG
2000 has them decouple tile size from frame size and give both. That is a bigger item than PAD-2, and it is
filed as **TILE-1**; the number above is what justifies scoping it.

**The original filing** — the pre-shipping version of PAD-1, with the three candidate shapes and
the oracle figure of −4.5% — is superseded by the two decision records that came out of it
(`0034` for the measurement, `0039` for the decision) and is in git history. It is not reproduced
here because two of its claims are now wrong: the intra figure is −4.63% measured on the shipped
encoder rather than −4.5% from an oracle, and it said the still-path fill would need a bitstream
version, which it does not — `pad.wgsl` is compiled only in the encoder.

### TILE-1 — partial border tiles, so tile size and frame size stop being alternatives (todo, **P2**)

Filed 2026-09-08 by PAD-1, which shipped 4.6 of the 6.6 points the padding costs and then measured
why the obvious cheaper escape does not exist. **Worth the remaining ~2 points on stills and the
whole ~6.6 on any content, and it removes a choice the codec should not have to make.**

**The constraint, measured rather than reasoned about.** The tile grid pads every plane up to a
whole multiple of `tile_size`, so 1920x1080 is coded as 2048x1280 and **20.9% of the coded samples
are outside the picture** (`0034`). A tile size that divides the frame exactly does exist —
`gcd(W, H)`, so **120 at 1080p, 80 at 720p, 240 at 2160p** — but a tile must be divisible by
`2^levels`, and **no tile size divisible by 32 divides 1080, 720 or 2160 at all**. Broadcast
heights are not power-of-two friendly: 1080 = 8 x 135, 720 = 16 x 45, 2160 = 16 x 135.

So today the choice is a deep wavelet with padding or a shallow one without, and it is not close —
bbb_1080p at q=90 with `--abac`, tile 120 at three levels costs **+81% of rate** against tile 256
at five levels, at the same PSNR (3 056 603 B against 1 689 447 B). **The padding is by far the
cheaper evil, and having to choose at all is the defect.**

**What this item is.** Let the border tiles be *incomplete*, the way JPEG 2000 does: **one** tile
size as today, and the last row and column are simply clipped to what is left of the picture.
Tile size and frame size decouple, five levels and zero padding stop being alternatives, and the
padding tax goes to zero at every resolution rather than being reduced for stills.

**Prefer that framing over "use a second, smaller tile size at the border", because it needs
nothing signalled.** The frame header already carries `width` and `height` (`format.rs:80`) — it
has to, in order to crop the padding away — so the border extents are just `W mod tile_size` and
`H mod tile_size` and **both sides derive them with no bitstream change and no new config field**.
The two-tile-size phrasing describes the same grid while adding a value to choose, store and keep
consistent. It is the same mechanism with extra machinery.

**What it touches, which is why it is P2 and not P1.** Tile origins and the tile grid; every
shader that derives a position from `tile_size` (the wavelet's `region_size`, the quantiser's
subband walk, the entropy coders' per-tile stream mapping); the per-tile CRC and the seek
structures in the container. `transform_97.wgsl` also sizes its workgroup arrays from the tile, and
BUG-26 is the standing reminder that WGSL does not trap an out-of-bounds workgroup access.

**Success criterion.** Zero padded samples with five wavelet levels at **every** resolution in the
table above — including 720x486 and 1366x768, which are the ones that break a
remainder-factorisation approach and are therefore the real test; **>= 5% of intra rate** against today's default on the four stills (PAD-1 already collected
4.6 of the 6.6 on stills, so the incremental win there is small — the case is that it also applies
to video, where PAD-1 is refused); decoded output bit-exact against today's on any resolution that
is already a whole number of tiles, which is the regression guard that says the grid rework did not
change coding.

**The tax is resolution-dependent and 1080p is not the worst case.** Raised in review: there are
far too many resolutions for a 1080p-shaped fix to be worth anything. Correct, and the spread is
wider than the headline suggests — `W mod 256` / `H mod 256`, and how deep a *clipped* border tile
could go on today's shader:

| | remainder W | levels | remainder H | levels | padding today |
|---|---|---|---|---|---|
| 1920x1080 | 128 | 7 | 56 | **3** | 20.9% |
| 1280x720 | 0 | full | 208 | 4 | 6.2% |
| 3840x2160 | 0 | full | 112 | 4 | 6.2% |
| 2048x1080 (DCI 2K) | 0 | full | 56 | 3 | 15.6% |
| 720x576 (PAL) | 208 | 4 | 64 | 6 | **29.7%** |
| 720x486 (NTSC) | 208 | 4 | 230 | **1** | 11.0% |
| 1366x768 | 86 | **1** | 0 | full | 11.1% |
| 1920x818 (scope crop) | 128 | 7 | 50 | **1** | 25.1% |
| 7680x4320 | 0 | full | 224 | 5 | 0.7% |

**0.7% to 29.7%, worst on the small and odd formats** — PAL pads 720x576 to 768x768 and throws
away nearly a third. Also worth knowing: 720p, 2160p, DCI 4K, 1440p and 4320p have a zero
remainder in *width*, so for them only the height is a problem, which is a smaller job than the
general case.
**Two stages, and sourced research says the first is far cheaper than this item first assumed**
(2026-09-08; citations in RESEARCH_LOG).

*Stage 1 — pad to a multiple of `2^levels` instead of a multiple of `tile_size`.* **That removes
about 87% of the tax with no new mathematics at all.** The grid keeps 256 in the interior and the
border tiles are simply shorter, but every extent stays a multiple of 32, so `half = extent / 2`
stays valid at all five levels and there is no parity or odd-length handling to write. The one real
change is that the arithmetic must be driven by each tile's **extent** rather than by the global
`tile_size`.

| | pad to 256 (today) | pad to 32 |
|---|---|---|
| 1920x1080 | 20.9% | **0.7%** |
| 1280x720 | 6.2% | 2.2% |
| 3840x2160 | 6.2% | 0.7% |
| 2048x1080 | 15.6% | **0.7%** |
| 720x576 (PAL) | **29.7%** | 2.2% |
| 720x486 (NTSC) | 11.0% | 7.1% |
| 1366x768 | 11.1% | 0.7% |
| 1920x818 | 25.1% | 1.7% |
| **mean** | **14.1%** | **1.8%** |

At 1080p, 1080 rounds to 1088 and the last tile row is 64 tall — divisible by 32, so five levels
are free. **And this has direct precedent.** SMPTE ST 2042 (VC-2), the closest relative to this
codec, *defines* its subband dimensions by rounding the picture up to `2^depth`, and its decoder
strips the padding afterwards (`idwt_pad_removal`, clause 15.4.5); JPEG XR pads to its 16x16
macroblock and crops with a windowing margin. So padding is a recognised design — **what has no
precedent is padding to a 256-sample tile**, which costs up to 255 samples per axis where VC-2
costs at most 31 and JPEG XR at most 15.

*Stage 2 — full clipping, the JPEG 2000 way, for the remaining 1.8% mean (7.1% at NTSC).* Border
tiles take their true extent and each split becomes `low = ceil(n/2)`, `high = n - low`. **This is
much smaller for GNC than for a general implementation**, because J2K's parity apparatus collapses
here: tile origins are multiples of 256, so the subband start coordinate is even at every level and
J2K's `cas` is always 0. What is actually needed is that per-level `ceil` split, the degenerate
`n == 1` rule (a lone sample on an even coordinate passes through unchanged), and symmetric
extension written as **index clamping inside each lifting step** rather than as pre-extension —
every 9/7 lifting step reaches only +/-1 in the split domain, so clamping reproduces pre-extension
exactly. No level reduction is needed for short tiles: at 1080p the 56-row strip runs
56 -> 28 -> 14 -> 7 -> 4 -> 2 and nothing degenerates.

**This supersedes the plan this item was filed with, which had border tiles giving up levels.**
That was solving a self-inflicted problem — short tiles only lose depth because
`transform_97.wgsl:67` is `let half = ts / 2u`. J2K and JPEG XS both transform short border tiles
at *full* depth and neither reduces the level count. The old plan's arithmetic still stands as a
warning, though: a per-tile level count would have been a lottery on how the remainder factorises,
giving NTSC one level over 47% of its rows and an odd remainder none at all.

**The design principle, from three shipping wavelet codecs that all arrived at it independently:
the alignment belongs in the buffer allocation and the boundary filter, never in the coded sample
count.**

* **DjVu's IW44** rounds its buffer up to 32 (`bw = (w+0x1f) & ~0x1f`, `IW44Image.cpp:596`) but
  passes the **true** dimensions to the transform with the padded width used **only as row stride**
  (`forward(data16, iw, ih, bw, 1, 32)`, `IW44EncodeCodec.cpp:958`). The pad strip is never touched
  and stays zero. Same 5-level maximum as GNC. Its boundary handling is *degraded taps* — a 2-tap
  average near edges — rather than extension at all.
* **VC-5 / CineForm's successor** keeps no pad buffer whatsoever and handles the odd case inside
  the boundary filter: `input[column] + input[column]` under the comment "Duplicate the value in
  the last column" (`vc5_encoder/forward.c:654`), with dedicated top-row / bottom-row / middle-row
  variants. GoPro deliberately migrated to this from CineForm's `ROUNDUP(height, 8)` plus a
  signalled crop.
* **Indeo 4/5** clips tiles outright — `tile->width = FFMIN(band->width - x, t_width)`
  (FFmpeg `ivi.c:368`) — with band sizing by ceiling and `FFALIGN` used only as pitch. A shipped
  wavelet *video* codec doing exactly what Stage 2 proposes. Its tile size is a bitstream field
  with four legal values (64/128/256 or whole picture) and odd tile dimensions are refused.

**Implementation gotcha, and GNC has been bitten here before:** Dirac rounds **luma and chroma
independently** against the same `2^depth` (§13.5.5.1 says so explicitly), so the chroma pad is
*not* the luma pad halved. That is the same trap as BUG-3 — each plane pads to a tile multiple
independently — so whatever Stage 1 does must be derived per plane, not by shifting the luma
figure.

**Two mechanisms ruled out before anyone finds them:** MrSID's overlap-add (US5710835A) reconstructs
a seamless whole-image DWT but couples neighbouring tiles, which GOALS rule 3 forbids; and VC-5
Part 5's tiles-as-layers requires every tile to be identically sized, which forbids ragged edges by
construction. Also dead ends, checked: ADV601 has no sub-blocking at all (whole-field, fixed
geometry), REDCODE's patent contains no wavelet geometry, and ECW is line-streaming rather than
tiled.

**Worth checking one level down while in here:** abac's 64px code-blocks are anchored at the
subband origin, and T.800 truncates *its* border code-blocks the same way rather than padding them.

### INTRA-2 — apply the dead zone to I-frames only (**DONE 2026-09-08**)

**Shipped as: `dead_zone` floored at 0.6 over q=85..95, plus a new `dead_zone_referenced`
carrying the ladder's own value for the two places that must not see the raised one — an I-frame
inside a P-chain, and the inter residual path.** Decision `0041`.

**The criterion is met on all three clauses:**

| clause | result |
|---|---|
| ≥2% rate at matched RGB PSNR, four stills, q≥85 | **BD-rate −5.01%** (−3.31 / −4.67 / −6.96 / −5.09) |
| no worst-frame regression on any of the three sequences | **byte-identical** — regression is structurally impossible |
| dE00 no worse | **−5.2% mean** at matched rate, p95 better on 11 of 12, Y-PSNR **+0.39 dB** |

**What 0028's blocker was actually measuring.** `GNC_DEAD_ZONE=0.6` moved two things, because
`res_dead_zone = config.dead_zone * inter_dz_mul` with the multiplier at 2.0. It took the *inter*
dead zone from the ladder's 1.0 / 0.357 / 0.025 at q=85/90/95 to **1.2 at all three**, turning a
dormant no-op into a strongly active one at q=90 and q=95. With the inter dead zone held at the
ladder's value, worst-frame goes from −0.85…−1.93 dB to **0.00/+0.03 dB at q=85 and −0.12…−0.19 dB
at q=90/95.** So ~90% of the blocker was the inter dead zone, not the intra lever.

**And the residue decided the design.** Even with inter held, a referenced I-frame costs −0.19 dB
of worst-frame for −0.33% of rate at q=90 — arithmetic, not tuning: at ki=9 the I-frame is one
frame in sixteen so its saving dilutes 16:1, while worst-frame PSNR is *fully* exposed to it
because **the worst frame is the I-frame**. So the gate is on **being referenced**, not on being
intra — which is why the item's own title understates it, and it is the same conclusion PAD-1
reached one commit earlier from a different lever (`0039`), on the line directly above.

**Rejected:** gating on frame type alone (leaves the raised value on a referenced I-frame — the
−0.19 dB case); lowering the floor to 0.52, where the sequence cost is −0.01 dB but most of the
stills win is gone too; and the principled `(|q| + r) * step` truncation-plus-offset, which is
still the right long-term answer and still needs a decoder change.

**BASELINE's q=90 row moves** to 49.89 dB / 7.21 bpp / VMAF 97.06 (VMAF −0.01 against a 0.5
threshold). Its q=25/50/75 rows are outside the floor's range and unaffected.

**Two harness errors are recorded in `0041`** and are worth reading before the next sweep: a floor
applied *after* an env override silently disarms the override, and `np.interp` clamping compared
two different rates and flipped the sign of the dE00 result.

Original entry follows.

#### INTRA-2, as originally filed

Filed 2026-09-08 by INTRA-1 step 2c, which measured the lever and then measured why it cannot ship
as a preset. **~3 points of the intra JPEG 2000 gap, blocked on the P path.** Decision
`docs/decisions/0028`.

**What is proven.** GNC quantises as `floor(|v|/step + 0.5)` after a `|v| < dead_zone*step` test, so
any `dead_zone <= 0.5` is a no-op — and production interpolates 0.5 at q=85 to 0.0 at q>=96. **GNC
has no dead zone at all in its own operating range.** Setting it to ~0.6 on four stills, six quality
points, `--abac`:

- rate at fixed RGB PSNR: **−3.1% / −2.2% / −2.5%** at 48 / 50 / 52 dB
- gap to J2K 9/7 falls **+27.2% → +24.1%**
- at matched rate: **+0.24 to +0.38 dB RGB PSNR *and* −2.9% to −4.5% dE00** — better on every axis
- VMAF at the same q: **−8 to −9% rate for −0.01 VMAF** (worst −0.02, block threshold 0.5)

J2K's own width (effectively 1.0) is much *worse* for GNC — +13.8% rate at 48 dB — because GNC
reconstructs at the round-to-nearest bin centre, so a wider zero bin pays in distortion immediately.

**What blocks it.** The dead zone also hits P-frame residuals. Three sequences, 16 frames, ki=9,
4:4:4, at matched rate against the production ladder: mean PSNR −0.83 to +0.30 dB, and **worst-frame
PSNR negative on 9 of 9 points, by up to 1.93 dB.** Worst-frame is what a contribution codec is
judged on (QUAL-1). A motion-compensated residual is already sparse, so a dead zone zeroes far more
real signal, and the error propagates down the prediction chain instead of staying in one picture.

**The work.** Gate the dead zone on frame type. `sequence.rs` already varies quantiser parameters per
frame type (`GNC_P_QP_SCALE`), so there is a place for it. Then re-gate on **both** stills and
sequences — this item exists because the stills gate was green on three metrics and all three were
measuring the wrong thing for the P path.

**Worth scoping in at the same time:** J2K's actual quantiser is truncation plus a reconstruction
offset at `(|q| + r) * step`. That is the principled version of this lever, and the sweep above shows
the naive wide zero bin is the wrong half of it. It needs a decoder change.

**Success criterion:** ≥2% rate at matched RGB PSNR on four stills at q>=85, with **no worst-frame
regression on any of the three sequences** and dE00 no worse. Below that, close it.

**Canary:** a per-frame log line showing the dead zone actually differs between I and P frames, and
an assertion that a q=100 encode stays bit-exact (BUG-30).

### BUG-30 — a dead zone could silently defeat bit-exact lossless (**FIXED 2026-09-08**)

Found by INTRA-1 step 2c. **`GNC_DEAD_ZONE=0.6` at q=100 produced a file 3.4% smaller that was not
bit-exact, with no warning from encode or decode.** BUG-15's hole, one knob over.

**Cause.** `is_lossless()` gates on `dead_zone == 0.0`, so a configuration carrying a dead zone
reported *not lossless*; `normalized_for_lossless` then skipped it entirely and the integer-exact
colour conversion and lifting paths were switched off. The guarantee gave way instead of the knob —
backwards, and it is the same shape as the `chroma_weight` case BUG-15 fixed.

**Fix.** Split **lossless intent** (`is_lossless_intent()`: transform and step) from
**bit-exactness** (`is_lossless()`: intent plus the knobs that can spoil it), and normalise on
intent. The dead zone is forced to 0.0 with a printed warning, exactly as `chroma_weight` already
was. Verified: q=100 is byte-identical at 927 600 B and bit-exact with `GNC_DEAD_ZONE` unset, 0.6 or
1.0. Test `conformance_a_dead_zone_cannot_defeat_lossless` asserts **bit-exactness, not a PSNR
threshold** — a threshold is what let 55 dB pass for lossless in BUG-15.

**Invalidates:** any lossless figure taken with `GNC_DEAD_ZONE` set. No shipped default carried one,
so no published number moves.

### ENT-7 — replace abac with BPC-PaCo? No (**part 2 REJECTED 2026-09-08**; the WGSL half is BUG-31)

**Steps 2 and 3 are done and the answer is no.** Decision record
[0030](docs/decisions/0030-bpc-paco-is-not-the-sixth-entropy-coder.md), numbers in RESEARCH_LOG
"ENT-7 steps 2–3". Priced on GNC's own shipped coefficients with a seventh model in the
`GNC_COEF_ENTROPY=1` harness (`src/encoder/bpc_paco_diag.rs`), four stills, q=85 and 90, the
parameters of `0024`:

| mean over four stills | q=85 | q=90 |
|---|---|---|
| BPC-PaCo's model, oracle table trained on the image itself | −7.76% | −7.87% |
| leave-one-image-out table (what a stationary coder ships) | −1.96% | −2.20% |
| **+ its 32 fixed-length codeword streams per block** | **+2.81%** | **+1.85%** |
| dropping the cross-lane exchange (vs the first row) | +9.35% | +8.63% |

Fails the +2% criterion at q=85, scrapes it at q=90, and is +6.21%/+4.07% on bbb. The model's
7.9% is abac's cold start counted from the other side — `Y LL` −50.4%, sub-64px bands −12% to
−29%, full 64×64 blocks only −4% to −11% — so it belongs to **ENT-6**, not to a new backend. Two
mechanisms extracted instead: **ENT-6** takes the stationary priors, **ENT-8** takes the
two-column lockstep scan. Not taken: the fixed-length multi-codeword coder (4.0 pts here).

**Part 1 is untouched and lives in BUG-31**, which is where the WGSL fix was always specified.
Nothing in part 2's rejection changes it: WASM decode is still broken by 18 688 B of declared
workgroup storage against a 16 384 B device, and BUG-34 is now a second instance of the same
class. ENT-7 is out of rotation; take BUG-31.

**What the papers say, for anyone who reopens this.** BPC-PaCo's parallelism is *free* in rate —
its two-column lockstep scan sees an average of 4 already-coded neighbours, exactly what JPEG
2000's raster scan gets, and the authors' own ablation puts the entire penalty on the multiple
codeword streams. Its throughput needs `__shfl`, `__ballot` and `__popc`; the ballot **is** the
bitstream ordering rule, and WGSL guarantees no relationship between subgroup lane ids and
`local_invocation_index`, so a subgroup port would have a device-dependent bitstream. And the
authors retired the stationary model in 2023 for an adaptive sliding window that beats JPEG 2000
and HTJ2K at medium and high rates — **if the backend question ever returns, that paper is the
specification, not the 2016 one.** The reference CUDA has no licence file at all and cannot be
copied.

The original filing follows, unchanged, as the record of what was asked.

Filed 2026-09-08. Two halves; the second, if it pays, deletes the first.

**This item was filed twice on the same day by two sessions, and this is the merged version.**
The first filing (branch `claude/wgsl-bpc-paco-encoder-tvws1m`, pushed and unmerged, which is why
`scripts/claim` could not see it and why the second session found nothing) is everything above and
below under "Order of work", "Success criteria", "Canary" and "Decision record"; the second
filing's contribution is the four blocks that follow — what `0024` already bounds, the throughput
ceiling, the two things this must not be bought for, and steps 0 and 1, which are cheap and can
settle the item without a GPU. Same mechanism as the `0018` decision-record collision: both
sessions read the tree, neither could see the other. If the original branch is merged later,
expect a duplicate-content conflict in this section and keep *this* version.

**1. The broken WGSL.** BUG-31 is the concrete defect: `abac_encode.wgsl` and `abac_decode.wgsl`
each declare **18 688 B** of workgroup storage against the **16 384 B** the device is created with.
Native wgpu never checks it, a conformant WebGPU implementation must, and the reachable failure is
every WASM decode (`DecoderPipeline::new` builds the abac decoder unconditionally). The three
in-place fixes are priced in BUG-31 and are not restated here — this item exists because there is a
fourth option BUG-31 does not consider: stop patching abac's shaders and ask whether the coder
itself is the right one to carry forward.

**2. The candidate replacement: BPC-PaCo** (Bitplane Coding with Parallel Coefficient processing —
Aulí-Llinàs, Enfedaque, Moure; IEEE TIP 2017, plus their GPU implementation paper). It differs from
abac in exactly the two places that have cost us:

- **Stationary, offline-trained probabilities per context** instead of per-symbol adaptation.
  No adaptation means no per-symbol serial chain, so coefficients *within* a code-block code in
  parallel — abac is one thread per block walking a serial chain on both sides (legal under the
  Hard Rules, bounded per block, but it is why frame decode is 1.69x and why encode time per frame
  is still unmeasured on an idle machine, the hole 0017 has carried since ENT-5).
- Stationary probabilities also delete ENT-6's cold start outright: ENT-6 measured the short
  blocks at **+25.9% over the entropy bound at q=90** because abac opens every block at p = 1/2,
  and its fix 1 (signalled initial probabilities) is this mechanism in one-tile form. The offline
  `warm_start` result — +2.4% → −0.7% — is the same idea already measured once in this repo.

The literature figure is roughly 2–5% rate loss against an adaptive J2K-class coder in exchange for
much higher GPU throughput. Whether that holds on GNC's subbands and contexts must be measured, not
assumed — and abac's measured −16.6% to −18.8% against Rice is the bar, so a BPC-PaCo that gives
half of that back may still lose on rate what it wins on speed.

**What `0024` already bounds, and it should be read before anything is built.** Decision `0024`
priced the *shipped* abac tiles against the entropy of the coefficients they carry: GNC spends
within **7.5%** of it at q >= 85, `Hbig` widens the context template 4x and finds **nothing**, and
`Hnb0` charges no model cost at all and buys **1.6%**. So there is no double-digit rate win
available to *any* entropy coder here. Worse for part 2: a stationary model is strictly weaker than
an adaptive one wherever there are enough symbols to adapt on, and that is where **82% of the rate
lives — the full 64x64 blocks, where abac is already within +4.1% of the bound.** On those bands
BPC-PaCo should be expected to **lose**. That is exactly why the success criterion below is
"+2% of abac" rather than a gain, and it is the criterion most likely to bind.

**The throughput prize has a ceiling, and it must be quoted next to the 1.3x.** Rice's own entropy
stage is **47% of frame decode**, which caps *any* entropy work at **1.9x** (BACKLOG Part 6,
idle-machine bench) — a free entropy coder buys 1.9x and no more, against abac's measured
**1.69x frame decode for −16.7% rate** at q=90 / cb=64. So "decode >= 1.3x abac" is asking for a
large fraction of everything that is physically there, which makes it a good criterion and an
unlikely one. Quote both numbers or the 1.3x reads as modest.

**It is not the parked `Bitplane` backend.** That one is GNC's own sign-plus-magnitude bitplane
coder with no context modelling, and it measured **2.2-2.6x worse rate than Rice** (RESEARCH_LOG
2026-09-06). Its failure is not evidence about BPC-PaCo and BPC-PaCo's claims are not a defence of
it. What may be reusable is scaffolding, not conclusions: `src/shaders/bitplane_decode.wgsl`
already walks bitplanes on the GPU.

**Two things this must not be bought for.** *Truncatable embedded streams*, which BPC-PaCo keeps
along with the bitplane passes: worth **0.00 dB** here, at every rate from 0.05 to 3.5 bpp, with a
structural reason (EBCOT part 1 — uniform scalar quantisation of a near-orthonormal transform
under MSE puts every coefficient at the same RD slope). And *closing the JPEG 2000 gap*: INTRA-1
step 1 put **~72% of the remaining 27.1 points upstream of the coder entirely.**

**Order of work.**

1. **BUG-31's browser check and cheapest fix first.** Whatever coder wins must fit 16 384 B, and
   the WASM target is a hard requirement today; ENT-7's part 2 is speculative and must not gate it.
   BUG-31's static canary (sum `var<workgroup>` declarations per `.wgsl` file in a test) covers any
   new shader this item adds, so land that with the fix.
2. **Read the papers and extract two numbers — no GPU, no code.** The exact probability model
   (which neighbours, quantised into how many contexts, tables truly fixed or trained per image),
   the reported rate delta against an adaptive J2K-class coder on 9/7 *lossy* data, and the
   reported GPU throughput with the hardware it was measured on. Read-only fan-out, the kind of
   subagent CLAUDE.md says to spawn freely. **Gate:** if the reported rate loss exceeds the ~2-5%
   quoted above, part 2's rate half is dead on arrival and only throughput survives — which then
   has to argue against the 1.9x ceiling before a line is written.
3. **Price BPC-PaCo's model on GNC's own shipped coefficients, offline.** This is the decisive
   cheap measurement and the harness exists: `src/encoder/coef_entropy_diag.rs`
   (`GNC_COEF_ENTROPY=1`) decodes the shipped abac tiles and prices their coefficients against six
   models per plane and per subband, four images, six quality points, per-band rows covering
   99.95% of the payload. **Add a seventh model: BPC-PaCo's local-average stationary contexts**,
   and read the per-band rows against `Hctx` (abac's own). The LL and level-3/4/5 rows decide this
   item; levels 1-2 are where the loss is expected. Same images, same points, same parameters as
   `0024`, or it is not a comparison.
4. **CPU reference BPC-PaCo first, GPU port verified byte-exact against it** — the abac pattern
   (98 of 98 whole-file matches) is the standard, and `abac.rs` shows the shape.
5. Measure rate at identical pixels (it is lossless recoding of the same coefficients, same rule as
   ENT-6), then throughput on an idle machine, encode *and* decode.

**Success criteria, stated before implementation:** shaders' declared workgroup storage
≤ 16 384 B under `Limits::default()`; rate within **+2% of abac** at identical decoded pixels on
the four stills and ≥3 sequences at q = 85 and 90 (worse than that, keep abac and only BUG-31's
fix ships); decode ≥ **1.3x** abac's throughput at 1080p 4:4:4 measured idle, else the added coder
is not paying for its maintenance. Bit-exact CPU/GPU on the full artefact set, like abac.

**The likely outcome, said in advance so it is not a disappointment.** If step 3 shows the short
blocks improving, the cheap way to collect that is to give *abac* neighbourhood-derived initial
probabilities — ENT-6's item, one initialisation change, no new bitstream, no shader pair, no new
entropy type. **A sixth backend has to earn itself against that alternative**, because it costs a
GP version, an entropy type, a CPU reference coder, GPU encode *and* decode shaders, a
byte-exactness gate and a permanent maintenance surface across two command families. Threshold for
preferring the backend over folding the mechanism into abac: **>=3% of total rate beyond what
abac + ENT-6 reach**, or the 1.3x decode above. Below both, the right answer is ENT-6's fix and a
decision record explaining why the coder was not added.

**Canary:** a `coder=bpc-paco` line with per-tile code-block and symbol counts on encode and
decode, and the BUG-31 static workgroup-storage assertion in CI.

**Traps, each already paid for by someone here.**

- **No VMAF BD-rate above q≈85** — widening the ladder moved it 47.5 points on average, 110 at
  worst, while PSNR moved 1.0. This item lives at q >= 85: PSNR leads, and dE00
  (`scripts/ypsnr_de00.py`) answers anything touching chroma.
- **abac is lossless recoding of the same coefficients.** If decoded quality moves at all, the
  measurement is wrong, not the coder.
- **A new coder must be checked on both command families.** `--rans` was unreachable on video for
  a while and nobody noticed; stills and sequences select coders through different paths.
- **The WebGPU limits are the budget, not the adapter's** — 16 384 B workgroup storage, 256
  invocations per workgroup, 10 storage buffers per stage. A coefficient-parallel design that wants
  1024 threads or 32 KB of tables does not run here, which is part 1's whole point.

**Decision record required either way** — a shipped coder is a default-adjacent choice, and a
rejection is a recorded conclusion with numbers (the EBCOT entry is the template).

### ENT-8 — abac could code 16 stripes per code-block instead of one (**step 1 DONE 2026-09-08**, step 2 todo, P2)

**Step 1 passes and the recommended stripe width is 4, not BPC-PaCo's 2.** Numbers in RESEARCH_LOG
"ENT-8 step 1". Priced offline on the shipped abac tiles with `abac_init_diag`'s `Scan`, four
stills, q=85 and 90, as a percentage of total rate against the same coder in raster order:

| | k=2 (32 threads/block) | k=4 (16 threads) | k=8 (8 threads) |
|---|---|---|---|
| mean, q=85 | +0.74% | **+0.41%** | +0.23% |
| mean, q=90 | +0.65% | **+0.37%** | +0.20% |
| worst single point | +1.00% (blue_sky q=85) | +0.58% | +0.30% |

The gate was 1% of total rate. **k=2 hits exactly 1.00% on 1 of 8 points**, so the last doubling of
threads costs as much as the first four together; **k=4 buys 16x today's parallelism for two-fifths
of the budget** and is the recommendation. The cost falls as 0.55 per doubling against the 0.50
that "only the first column of a stripe pays" predicts — the residual is that the paying column is
where the left neighbour carries most, which is also why the **LH bands are the worst rows**
(`Y LH1` +1.65% against `Y HL1` +0.46% at k=2): LH is horizontally lowpass, so the left neighbour
is exactly the informative one there.

**Step 2 is untouched and it is the whole question.** What step 1 changes about it: the width is
settled, the rate cost is known, and BUG-31 left ~9 984 B of workgroup headroom so the exchange
buffer is not a constraint — but `tests/workgroup_storage_limit.rs` now asserts every entry point's
exact size, so an addition goes in its exception list as a number, never as a tolerance. What step
1 does **not** change: the exchange needs a `workgroupBarrier()` per phase, 64 rows x k phases per
block, and the authors' own shuffle-to-shared-memory substitution cost ~20% on a far less
exchange-dense kernel, so 20% is a floor on the loss. **Step 2 is an implementation item on the
scale of ENT-5** — a CPU coder variant, both shaders, and a byte-exactness gate — **whose entire
payoff is a throughput number that cannot be taken on this machine** (eight sessions, one GPU; the
same abac decode has read 25.2 / 31.1 / 37.5 ms across three runs). Do not start it without an
idle box.

**The decision record belongs to whoever ships step 2**, with the curve above as its input.
Nothing shipped here: no default moved and no conclusion reversed, so step 1 has none.

Original filing follows.

### ENT-8 — abac could code 32 stripes per code-block instead of one, at no context cost (original filing)

Filed 2026-09-08 by ENT-7 step 3, which found it while rejecting the coder it came from. **This is
a throughput item with a rate gate, not a rate item.**

**The mechanism, and it is the one genuinely surprising thing in the BPC-PaCo papers.** abac runs
**one GPU thread per code-block** on both sides. BPC-PaCo splits a block into vertical stripes two
columns wide, one thread per stripe, and steps them in lockstep — every thread codes the left
column of row *y*, then every thread codes the right column of row *y*. A left-column coefficient
then has **3** already-coded neighbours and a right-column one **5**, so the average is **4 —
exactly what a sequential raster scan gets** (TIP 2016 §III-A). The parallelism is bought by
*scheduling*, not by weakening the context, and the authors' own ablation confirms it costs
essentially nothing in rate.

**For a 64px code-block that is 32 threads where abac has 1.** ~3000 blocks per padded 1080p 4:4:4
frame today; this would be ~96 000 invocations.

**What has to be checked, in this order.**

1. **The rate cost on abac's template, offline, before any shader.** abac's context is
   `neighbour_sum` over left, up, up-left, up-right — 4 causal neighbours, all available in a
   raster scan. Under the lockstep scan an even-column coefficient loses the *left* neighbour and
   an odd-column one gains nothing it did not have, so the template degrades to 3 of 4 on half the
   coefficients. **That is a different arithmetic from BPC-PaCo's 8-neighbourhood and it does not
   inherit their result** — BPC-PaCo's average is unchanged because it reads 8 neighbours, abac
   reads 4 and they are all on the causal side. Price it with `src/encoder/bpc_paco_diag.rs`'s
   `visit_order`, which already produces the scan, against `Hctx` in the same harness. **Gate: if
   this costs more than 1% of total rate at q=85/90 on the four stills, stop here** — the
   throughput is not worth a rate regression, and 1% is roughly a quarter of everything abac has
   left (`0024`: +4.1% on the full blocks).
2. **Then the WGSL cost, which is the real question.** The exchange of the left neighbour across a
   stripe boundary needs one bit per coefficient per step between adjacent threads. WGSL has no
   portable subgroup shuffle, so it goes through workgroup storage plus a `workgroupBarrier()` per
   step — 64 rows × 2 phases per block. The authors measured the same substitution at **~20% on
   their DWT kernel**, which is far less exchange-dense, so treat 20% as a floor on the loss and
   the open question as whether 32× the invocations beats it.
3. **Barrier uniformity constrains the workgroup shape.** WGSL requires barriers in uniform
   control flow, and abac's per-block loop bounds are per-block, so one code-block per workgroup
   at `@workgroup_size(32)` is the shape that is legal without masking. That is 32 of 256
   invocations, which wastes 7/8 of the workgroup unless the tail is filled with something.

4. **The workgroup-storage budget is now known and it is generous — but it is also enforced.**
   BUG-31 (merged 2026-09-08, `9424405`) repacked `rows` in both abac shaders to one clamped byte
   per magnitude, four to a word: **18 688 B → 6 400 B per entry point**, so this item has about
   **9 984 B** of headroom under `Limits::default()`'s 16 384 B. The stripe exchange needs one bit
   per coefficient per step across stripe boundaries, which is nothing against that. **The
   enforcement is new too:** `tests/workgroup_storage_limit.rs` parses every `src/shaders/*.wgsl`
   with naga, sums `var<workgroup>` per compute entry point and asserts against
   `Limits::default()`, with each known offender's *exact* size in an exception list — so a size
   that moves in either direction fails. Adding a workgroup array here will trip it deliberately;
   update the exception list only with a number, never with a tolerance. (BUG-31's sweep also
   found five more offenders, filed as **BUG-35**, one of them in the *default* encoder path.)

**Success criteria.** Bit-exact CPU/GPU on the full artefact set, like ENT-5 (98 of 98 whole
files). Rate within **1%** of today's abac at q=85 and 90 on the four stills at bit-identical
decoded pixels — abac is lossless recoding, so if quality moves the measurement is wrong. Decode
**≥1.2×** today's abac at 1080p 4:4:4 on an idle machine, encode not worse. Below that, close it:
the 1.9× ceiling on all entropy work (BACKLOG Part 6) means there is not much to win and a
bit-exactness surface to maintain.

**Canary:** a per-frame count of stripes coded and barriers executed under `GNC_PROFILE`, and the
existing byte-exactness gate `scripts/ent5_gpu_encode_gate.sh` unchanged and green.

**Which instrument, because two of the obvious ones are wrong** (from BUG-32's session,
2026-09-08). `gpu_tier_bench.py --density` and any `frames / wall` from `benchmark-sequence` are
out: that command spends **86% of its wall clock on CPU quality metrics**, so its frames per second
is SSIM throughput. `--density-still` is the right instrument for the criterion above — it sweeps
`benchmark`, which loops on one still frame at 1080p 4:4:4 — **and for exactly that reason it
cannot say anything about inter**. If this item is ever extended to P/B residuals, the figure to
use is `benchmark-sequence`'s own **printed encode timing** (BASELINE quantity A, 208.7 ms for 8
frames in the BUG-32 run), which times the encode phase and is unaffected by the defect. Two
further cautions on `--density-still`: ~705 ms of fixed startup plus 6.8 ms per iteration of
non-GPU work, so 24% overhead at the default `--iterations` — run it large; and power sampling is
`nvidia-smi` only and swallows its own errors, so on Metal the power table is silently absent and
that is not evidence of an idle GPU. The rate half of this item is unaffected by all of it — bytes
are deterministic.

**A second thing rides on this item, from ENT-6 (2026-09-08, decision `0031`).** ENT-6's candidate
2 — stop cutting code-blocks on subband boundaries below `cb` — measures **−0.40% of rate** and is
also a simplification: 3000 → 1920 blocks per 1080p 4:4:4 frame, 36% fewer length fields,
`code_blocks_banded` collapsing into a plain grid, and at tile 256 with cb 64 only each tile's
top-left block changes at all. It was **not** taken, on one objection: fewer blocks is less
parallelism, and abac is one thread per code-block. **If this item lands, that objection is gone** —
a block would be 32 threads and block size would stop being the parallelism knob. So ENT-6's
candidate 2 becomes a free follow-on to ENT-8 and should be re-priced as part of it.

**Why P2.** It is the only remaining idea with a credible path to abac's decode cost, and step 1
is an afternoon with an existing harness. Against that: the throughput half cannot be measured on
a shared machine at all (COORDINATION), and the rate gate may kill it before the shader work
starts — which is why the gate is first.

### ENT-9 — abac context-codes three decisions and bypasses the rest; at q>=95 the rest is where the file is (todo, P2)

**Filed 2026-09-08 by the ENT-3 session, from its own numbers.** ENT-3 measured abac's saving
against Rice on P-frame bytes decaying monotonically with quality — bbb_extended −20.6% at q=90 to
−14.5% at q=99, crowd_run −12.2% to −4.3%, old_town_cross −11.9% to −3.7% (`0045`). The same
run's entropy bound says the shortfall is the **context template**, not adaptation: shipped sits
+12.5% over `Hnb` on inter at q=99 while adaptation loss is under 0.7%.

**Hypothesis.** abac context-codes exactly three binary decisions per coefficient — significant,
`>1`, `>2` — and sends the Exp-Golomb order-0 remainder of `(|v| - 3)` and the sign as **bypass**
bits at p=0.5 (`abac.rs`, `encode_block`). As the quantiser fines, magnitudes grow and the
population moves out of the three context-coded decisions and into the bypassed suffix. So the
decay is not abac running out of structure; it is abac coding a shrinking share of the file. Rice
codes exactly that population well, which is why the two converge.

**Step 1, and nothing should be built before it: measure the split.** What fraction of the shipped
bits at q=90 / 95 / 99 are (a) the three context-coded decisions, (b) the Exp-Golomb suffix, (c)
the sign? **This is not yet measured and the hypothesis above stands or falls on it.** If the
suffix is 15% of the file at q=99 the ceiling on this item is small; if it is 60% the item is the
largest thing left in the coder. `coef_entropy_diag` already walks abac's binarisation and
`abac_decode_tile` gives the coefficients back, so this is a counter in an existing read-only
diagnostic, not new coder work.

**Step 2, only if step 1 justifies it.** The cheap candidates, in ascending cost:

- **A context for the first suffix bit**, conditioned on the same magnitude bucket. One extra
  context set of 6; the bit is far from uniform when the neighbourhood is large.
- **A sign context** from the signs of the left and up neighbours. Wavelet subband signs are not
  independent along the direction of the band — HL is horizontally correlated, LH vertically —
  and this is the standard JPEG 2000 sign-context argument, which GNC has never priced.
- **More `>k` decisions** before the bypass starts (`>3`, `>4`), which is a straight
  context-count-for-rate trade and the one most likely to be a wash.

**Success criterion.** ≥2% of total rate at q=99 on ≥3 sequences at bit-identical pixels, which is
the same gate ENT-6 was closed against and the same bar its 1.3% failed. Below that, close it:
a context experiment worth 1% is not worth the decode dependency it adds.

**Why it is P2 and not P1.** abac is opt-in and `0017`'s case for it is *weaker* at the top of the
range after `0045`, so this improves a non-default coder in the range where it is least
convincing. It is filed because the mechanism is specific, the instrument exists, and step 1 is
an afternoon; it is not filed as urgent.

**Do not confuse this with ENT-6 or ENT-8.** ENT-6 priced the *initialisation* of the existing
contexts (1.3%, closed). ENT-8 prices *parallelising* the existing contexts at fixed rate. This
prices *which symbols get a context at all*, which neither touches.

### ENT-6 — abac's cold start is worth 1.3%, not 4% (**CLOSED by measurement 2026-09-08**)

**Closed. Neither candidate ships.** Decision record
[0031](docs/decisions/0031-abacs-cold-start-is-worth-one-point-three-not-four.md), numbers in
RESEARCH_LOG "ENT-6 — abac's cold start is worth 1.3% of rate, not 4%".

**Why the ~4% below is wrong, and it is the transferable part.** That figure is a *bound* ratio.
`coef_entropy_diag`'s columns pool their probabilities over a whole plane's worth of a subband, so
on a short block "shipped vs bound" mixes the cold start — which an initialisation fixes — with the
gap between a per-block adaptive model and a plane-wide oracle, which nothing fixes. **The pooling
is the thing being priced. A bound cannot price a change to initialisation.**

Measured instead by *simulating* the coder — `src/encoder/abac_init_diag.rs`, the shipped
`Prob::update` driven over the shipped coefficients, only the initialisation changing between arms
— on the four stills at q=85/90/95/99, as a percentage of what abac's bitstream really spent:

| | q=85 | q=90 | q=95 | q=99 |
|---|---|---|---|---|
| candidate 1, signalled table **once per frame** (864 B) | −1.48% | −1.25% | −0.93% | −0.61% |
| candidate 1, signalled **per tile** — the design below | +1.2% … +0.4% (a **loss**) | | | |
| candidate 2, drop the band-aligned cut below `cb` | −0.47% | −0.42% | −0.37% | −0.31% |
| both together | no better than candidate 1 alone — they are **substitutes** | | | |

Against the criterion below (≥2% at q=90, close under 1%) the best variant reaches **1.07–1.52%**.
Three things settle it: the effect **shrinks with quality** while the bound ratio grows, which is
the artefact itself; candidate 1 makes the entropy encode **two-pass**; and the per-tile design the
item proposed costs 34.5 kB of header against a 2% target, so it is larger files.

**What survives:** candidate 2 is a −0.40% rate win *and* a simplification (3000 → 1920 blocks per
1080p frame, 36% fewer length fields, `code_blocks_banded` collapses to a plain grid, and only each
tile's top-left block actually changes) whose one objection is that fewer blocks is less
parallelism. **ENT-8 removes that objection** — see its entry. And one untested variant is cheaper
than either: a two-speed `ADAPT_SHIFT`, faster for a block's first symbols, needs no header, no
partition change and no second pass. It cannot beat the −1.48% ceiling, so it is recorded in `0031`
rather than filed.

The original filing follows, unchanged, including the ~4% that this closes.

### ENT-6 — abac's deep subbands are one short code-block each (original filing, ~4% figure withdrawn)

Filed 2026-09-07 by INTRA-1 step 1, which found it while measuring something else. **Not the answer
to INTRA-1** — it is worth about 4% of the file and the J2K gap is 27 points — but it is the one
concrete, cheap entropy-coding win that measurement turned up, and it is half of all the headroom
abac has left.

**The mechanism.** `code_blocks()` never lets a block straddle a subband boundary, which is right —
the statistics either side differ. But at tile 256 with 5 levels the LL and the level-3/4/5 bands
are 32, 16 and 8 px square, so each becomes **one short code-block**: 64 coefficients to adapt 18
context probabilities on, against 4096 in a full 64x64 block. abac starts every block at p = 1/2.

**Measured**, against the most generous entropy bound on the same coefficients
(`GNC_COEF_ENTROPY=1`, four images, mean over the four):

| | share of rate | shipped vs bound |
|---|---|---|
| levels 1-2 (full 64x64 blocks) | 82% | +4.1% (q=90) |
| LL + levels 3-5 (blocks < 64px) | 18% | **+25.9%** (q=90) |

Worst individual rows on bbb at q=90: `Y LL` **+66.7%**, `Y HL5` **+47.3%**, `Y HL4` +30.9%. The
effect grows with quality: the small bands read +23.1% / +25.9% / +28.9% / +34.2% at q = 85/90/95/99.

**Two candidate fixes, and they are not exclusive.**

1. **Signalled initial probabilities per subband.** One byte per context per subband per tile, set
   from the encoder's own statistics. `AbacTile` already carries a header; 18 bytes per subband is
   ~0.3% of a tile at these rates. This is exactly what the offline work called `warm_start`, where
   it moved the 256-stream variant from +2.4% to −0.7% — the same mechanism one scale up.
2. **Let the deep subbands share one code-block.** LL + the three level-5 bands are 4x8x8 = 256
   coefficients; the level-4 set is 4x16x16 = 1024. Cutting one block per *level* instead of one
   per band gives the coder 4x the symbols. Costs the per-band homogeneity the current cut buys, so
   it must be measured, not assumed — the orientation difference is real.
3. **Trained stationary initial probabilities rather than signalled ones** — BPC-PaCo's
   mechanism. **ENT-7 step 3 has now measured it and it is the biggest of the three
   (2026-09-08, decision `0030`).** A stationary per-bitplane, per-subband model over the same
   shipped coefficients prices **7.8–7.9% below what abac spends**, and the saving sits exactly
   where this item says it should: `Y LL` **−50.4%**, the sub-64px bands −12% to −29%, the full
   64×64 blocks only −4% to −11%. Three things ENT-7 learned that this item should not have to
   rediscover:

   - **Use the table as a prior, not as the model.** Of that 7.9%, **5.7 points is what
     *stationarity* costs** — a table trained on three of the four stills and applied to the
     fourth gives back nearly everything. abac keeps adapting, so it pays that only on the first
     symbols of each block, which is precisely the defect being fixed.
   - **Chroma needs its own tables, or none.** The leave-one-out misses are ≤0.31% on Y and up to
     **2.44% on Co**, and every band where the trained table came out *worse* than abac is a
     chroma level-1 band. Four images is a thin corpus; the Y tables travelled and the chroma
     tables did not.
   - **Per bitplane matters.** The literature is explicit that a single pooled table degrades
     every corpus, and that LUTs must be built bitplane by bitplane
     (Aulí-Llinàs & Marcellin, *IEEE TM* 16(4), 2014, §IV).

   Also from that literature, and the direct confirmation of this item from outside: a stationary
   model *beats* adaptive JPEG 2000 as code-blocks shrink — lossless natural imagery, +0.04 bps at
   64×64, 0.00 at 32×32, **−0.10 at 16×16**, with half JPEG 2000's degradation over that range.
   Short blocks are where a prior wins, which is what this item is about.

**Success criterion:** ≥2% of total rate at q=90 on all four stills, at bit-identical decoded
pixels (abac is lossless recoding; if quality moves at all, something else changed). Below 1%,
close it — CLAUDE.md's "know when to stop".

**Canary:** `GNC_COEF_ENTROPY=1` prints the per-band `vs Hnb` column; the LL and level-3/4/5 rows
are the ones that must move, and levels 1-2 must not.

### ENT-5 — abac needs a GPU encoder, and it is what stands between abac and the default (**DONE 2026-09-07, one criterion outstanding**)

**Shipped.** `src/shaders/abac_encode.wgsl` + `src/encoder/abac_gpu_encode.rs`, one thread per
code-block, both arithmetic engines. **Bit-exact against the CPU encoder: 98 of 98 whole-file
comparisons byte-identical** (`scripts/ent5_gpu_encode_gate.sh` — four stills at q=60/75/90/99/100,
both engines, both output-sizing modes, 4:4:4/4:2:2/4:2:0, cb=16/32/64, plus an 8-frame bbb
sequence at ki=1 and ki=9). GPU encode -> file -> GPU decode is **max |diff| 0** against the Rice
decode of the same source at 1080p 4:4:4. Rate therefore cannot have moved, and did not.
`tests/abac_gpu_encode.rs` asserts the same per block over eight geometries plus degenerate planes.

**Criterion 3 — encode time per 1080p frame — is NOT measured.** Four sessions were on this M1 at
load 10.2 and COORDINATION forbids a wall-clock figure under load. The instrument exists and is
built to the same rules as the decode grid:
`cargo test --release --test abac_bench -- --ignored --nocapture --test-threads=1`. Until it runs,
**decision 0017's reason 2 has lost its mechanism and kept its 129 ms.** Whoever runs it should
also settle the sizing mode: `CountThenEmit` (2 coder passes, exactly-sized scratch) is the default
and `BoundedSlots` (1 coder pass, 22-29x scratch) is selectable with `GNC_ABAC_GPU_SIZING=slots`;
the bytes are identical either way, so the default can flip on one run. `docs/decisions/0024`.

**What it does not do:** abac is still opt-in — 0017's reason 1, the 1.69x decode, is untouched,
and reason 2's figure is unmeasured. (Reason 3 was discharged by ARCH-3, not here.) It does not
touch `use_gpu_encode` or
`sequence.rs` — abac is routed on `gpu_entropy_encode` directly, one condition inside
`encode_entropy`, deliberately orthogonal to ARCH-3 and to the fused quantiser that BUG-16 says
moves Rice's pixels.

The original item follows.


Filed 2026-09-07 after ENT-4. The mechanism has been named three times today — inside ARCH-3, inside
BUG-18 and inside decision 0017 — and never as an item with an ID, so `scripts/claim next` cannot
offer it. It is the third of the three reasons abac is opt-in, and the only one that is a missing
shader rather than a trade.

**Why it is P1 now rather than a tidy-up.** ENT-4 measured what abac is worth: **−16.0% of rate over
q=60-99 at identical pixels** (24/24 rungs bit-identical), closing **half** the RGB gap to JPEG 2000
and taking GNC level with ProRes 4444. That is the largest single compression lever in the codec, it
pays on intra, inter, lossless and every chroma format at once (decision 0018), and it is behind a
flag partly because of 129 ms/frame of CPU encode against Rice's 23 ms (`docs/decisions/0017`).

It is also the *cause* of the ARCH-3 / BUG-18 class of defect, not merely a neighbour of it: abac has
no GPU encode shader, so selecting abac flips `gpu_entropy_encode` and silently swaps the entire
P-frame encoder for the defective non-batched implementation. **ARCH-3 fixes the routing; this
removes what the routing is routing around.** Both are worth doing and neither substitutes for the
other.

**No architectural obstacle, and this is measured rather than asserted.** `abac_decode.wgsl` already
runs one thread per code-block — ~3000 blocks per 1080p frame — and is bit-exact against the CPU
coder across seven geometries. Encode is as parallel as decode over the same blocks, and abac's
per-symbol serial chain is confined to a block, which CLAUDE.md's bounded-dependency rule already
accepts and prices.

**The one real complication:** a block's coded size is not known before it is coded. Two ways, and
the item should measure rather than pick by taste — (a) worst-case allocation per block plus a
compaction pass, (b) two passes, count then emit. State which, and what the other would have cost.
BUG-22 is the standing example of what an unbounded per-stream output slot does, so whichever way it
goes, the bound must be explicit.

### Success criteria, stated before implementation

1. **Bit-exact against the CPU encoder.** The GPU encoder's bytes must be identical to the shipped
   CPU abac output, per block and per file, on the same seven geometries the decoder was verified
   across. Anything less is a different coder, not a faster one.
2. **Round trip.** GPU encode → file → GPU decode → max |diff| **0** against the Rice decode of the
   same source, at 1080p 4:4:4. That is the check ABAC-SHIP used and it caught a coder that produced
   correct rate and no picture.
3. **Encode time per 1080p frame, on an idle machine**, reported as one of BASELINE's three named fps
   quantities and saying which. Rice is 23 ms, abac on CPU is 129 ms. Report the number whatever it
   is: it is what decides whether reason 1 of decision 0017 is discharged, and four sessions share
   this GPU, so a figure taken under load is worth nothing (COORDINATION, and the 1.9x clock-ramp
   lesson).
4. **Rate unchanged.** −16.0% mean over q=60-99 must reproduce exactly. A bit-exact encoder cannot
   move it; if it moves, criterion 1 has failed and the rate figure is not the finding.

**Canary, and it is not a formality here.** A log line proving the GPU encoder ran — dispatched block
count and bytes emitted. A shader that silently falls back to the CPU coder passes criteria 1, 2 and
4 *by construction*, so those three cannot detect the one failure mode most likely to occur.
CLAUDE.md, "no silent features".

**Not in scope: making abac the default.** That is a separate decision needing decision 0017's other
two reasons re-priced — the 1.69x frame decode, and abac's inter behaviour (ENT-3, which BUG-18
blocks). This item discharges one of three, and its write-up should say plainly which two remain.

**Coordination.** Overlaps ARCH-3 in the entropy-encode dispatch. **ARCH-3 landed 2026-09-07 —
rebase onto it.** What changed under this item: there is one P/B frame encoder now, and
`inter_gpu_entropy_available()` in `entropy_helpers.rs` is the single place that says which coders
have a GPU entropy encoder. Adding abac's shader means adding `EntropyCoder::Abac` to that
predicate and a dispatch arm beside the Rice and rANS ones in `sequence.rs` — it no longer means
touching a frame encoder, which is the whole point of ARCH-3. abac video is also *correct* now, so
a GPU encoder can be checked against the CPU one for bit-exactness on inter, not only on intra.

### ENT-3 — abac on inter: answered end to end (**DONE 2026-09-08**)

**All three open questions are closed. Decision record `0045`, harness
`scripts/ent3_abac_inter.py`, log entry in RESEARCH_LOG.** Binary pinned and hash-recorded
(`364e6aaf…` at `f3f7254`), 18 frames, ki=9, 4:4:4, and every point decodes both arms and hashes
all 18 PNGs — **18 of 18 identical at all 18 points**, so no BD-rate is quoted and none is needed.

**1. The frame mix is `2I+16P+0B`.** Read out of three places, not reasoned about:
`benchmark-sequence` prints it on stdout and emits `GNC: B-pyramid suppressed (ki=9 would allow
it)` on stderr; `encode-sequence` prints `Encoded 18 frames (2I + 16P)` on all 36 encodes; and
`encode-sequence`'s `-q` defaults to 75, so that path cannot reach `CodecConfig::default()` at
all. `-q` was passed everywhere (BUG-37).

**2. The container ratio was never an inter figure — the halves are separable and now separated.**
At ki=9 a whole-file ratio mixes abac's known intra saving in. `encode-sequence` tags each frame
`[I]`/`[P]`, so **P-frames only**, abac against Rice:

| sequence | q=50 | q=75 | q=90 | q=95 | q=97 | q=99 |
|---|---|---|---|---|---|---|
| bbb_extended | −16.1% | −21.1% | −20.6% | −18.2% | −16.6% | −14.5% |
| crowd_run | −18.3% | −14.3% | −12.2% | −9.6% | −7.5% | −4.3% |
| old_town_cross | −21.6% | −14.7% | −11.9% | −9.5% | −7.2% | −3.7% |

I-frames from the same runs, for the contrast: −17.8/−14.7/−14.2/−12.3/−11.1/−9.5,
−16.0/−12.7/−11.4/−8.6/−6.4/−3.3, −20.2/−14.5/−12.7/−10.2/−8.0/−4.7.

**This entry's own prediction is falsified.** It said to expect a *smaller* number on inter than
intra's −17% because a motion-compensated residual is noise-like. Measured inside the same run,
**inter is the stronger half on two of three sequences** (bbb_extended −20.6% vs −14.2% at q=90),
and P −11.3% vs I −9.4% over all 18 points. So the entry's second branch — "then the inter gap
lives in the motion model, not the coder" — is **not** supported and must not be read out of this.

**The new finding is the decay.** Monotonic in q everywhere: two of three sequences lose two
thirds of the saving between q=90 and q=99. GNC's home range is q=95-99, so **the figure that
matters for positioning is the smallest one**, not the −12.0%-to−22.9% band `0025` published.

**3. Retuning the contexts for residual statistics: rejected, with a number.** New read-only
diagnostic `GNC_COEF_ENTROPY_INTER=1` prices the first P frame's shipped abac tiles the way
`GNC_COEF_ENTROPY=1` prices a still's. crowd_run, `TOTAL` rows — shipped above `Hnb` (50-context
richer neighbourhood): **q=95 intra +7.7% vs inter +6.7%; q=99 intra +12.8% vs inter +12.5%.**
Inter's headroom is *smaller*, so the 6 magnitude buckets leave the same amount on a residual as
on an intra subband and there is nothing inter-specific to collect. Adaptation loss is under 0.7%
throughout, so the gap is the **template**, not the cold start — an INTRA-1/ENT-6 question over
both populations, not an inter one.

**4. Correction that fell out of the reproduction — `0025`'s q=50 and q=75 columns are
superseded.** A detached worktree pinned at `a312d6f` (0025's own commit) reproduces its container
column **9 of 9 exactly**, so the harness is the same measurement and the mix behind it was
`2I+16P+0B`. Across the two commits, I-frame bytes are equal integers at all nine points and
P-frame bytes are equal integers **at q=90** but +51.9% to +217.1% at q=50 and +52.4% to +59.9% at
q=75 (Rice; abac similar). Inter-only, no-op at q=90, and far too large for BUG-27 (+0.4% to
+2.0%): **the dominant cause is INTER-2 (`0043`) halving `inter_dz_mul` 2.0 → 1.0.** The ladder's
dead zone is 0.75 at q=50 and q=75, so the inter dead zone went 1.5 → 0.75 and stopped zeroing a
large population of small residual coefficients; at q=90 it interpolates to ≈0.18, so the inter
value went ≈0.36 → ≈0.18 and **both are no-ops** — the quantiser is `floor(|v|/step + 0.5)` after
the dead-zone test, so anything at or below 0.5 changes nothing. Not bisected against BUG-27 and
INTRA-2's `dead_zone_referenced` split, which are in the same window; the magnitude says the dead
zone carries it. CLAUDE.md's abac range is corrected.

**Nothing shipped changes.** The only code is the env-gated diagnostic, and output is
byte-identical with the variable unset. abac stays opt-in and `0045` moves it no closer: a −3.7%
saving at q=99 is not an argument for a 1.69× frame decode, so `0017`'s inter reason is priced
*down* in the range GNC is for.

<details>
<summary>The entry as it stood before this was measured (kept for provenance)</summary>


**Retitled 2026-09-08 (DOC-2), because the old title asked a question this entry answers.** It read
"Does abac pay on inter residuals? (todo, P1, claimed and released unmeasured 2026-09-07)", which
was wrong twice over: ARCH-3 measured it as a side effect — **yes, −12.0% to −22.9% at bit-identical
pixels on nine points** — and "unmeasured" stopped being true on 2026-09-07. A session picking this
off the queue was being handed a solved question with the solution in its own body.

**What is actually open, in the order it should be done:**

1. **Which frame mix produced the nine-point table.** At ki=9 there are two (`2I+16P+0B` or
   `2I+2P+14B`) and `0025` does not record which. Settle it by reading `benchmark-sequence`'s own
   stdout, not by reasoning — and pass `-q`, since BUG-37 was exactly that trap.
2. **The contribution range proper, q=95-99**, which is GNC's home range and is where RATE-2 says
   the ladder misbehaves anyway.
3. **Whether the contexts are worth retuning for residual statistics** — they were tuned on intra
   coefficients.

Nothing below is retracted; the numbers stand for the range they were taken in.

**The inter half of the entropy question, which ABAC-SHIP explicitly left out of scope** ("intra
only; inter is out of scope for this row"). Nothing has been measured. Claimed at the end of the
2026-09-07 session and released without a single number, so this entry is the hypothesis, not a
result.

**Substantially answered as a side effect of ARCH-3 on 2026-09-07 — read this before claiming.**
Closing BUG-18 required proving that the entropy choice does not reach the pixels, which is exactly
this comparison's premise, so the numbers fell out of that gate. abac against Rice, 18 frames,
ki=9, 4:4:4, `.gnv` bytes, **bit-identical pixels on all nine points** (decoded PNGs hashed):

| sequence | q=50 | q=75 | q=90 |
|---|---|---|---|
| bbb_extended | −16.3% | −22.9% | −19.7% |
| crowd_run | −20.7% | −18.4% | −12.1% |
| old_town_cross | −22.7% | −21.8% | −12.0% |

**Superseded 2026-09-08 for q=50 and q=75 (`0045`).** q=90 reproduces exactly; the other two
columns were taken before INTER-2 halved the inter dead zone.

So the answer to "does abac pay on inter residuals" is **yes, −12.0% to −22.9%**, in the same band
as its intra −16.6% to −18.8% and wider at both ends. No BD-rate is needed: the quality delta is
exactly zero, not small. **What is left of this item** is the contribution range proper — q=95-99,
where RATE-2 says the ladder misbehaves anyway — and whether the contexts, tuned on intra
coefficients, are worth retuning for residual statistics. Neither is answered above.

**The table above does not say which frame mix produced it, and at ki=9 there are two (noted
2026-09-08).** `quality_preset()` sets `b_pyramid: false` unless `GNC_B_PYRAMID=1`
(`src/lib.rs:1021`, since 2026-09-06), while `CodecConfig::default()` sets it `true`
(`src/lib.rs:571`). So ki=9 codes either **2I+16P+0B** or **2I+2P+14B** depending on how the
config was built — very different residual statistics — and `0025` does not record which. Its
`0025:90` byte-identity sweep covered "B-pyramid on and off", but that is the 54-config identity
check, not the nine-point rate table. **Settle it by reading the run rather than reasoning:**
`benchmark-sequence` prints `2I+16P+0B` or `2I+2P+14B` on stdout, and emits
`GNC: B-pyramid suppressed (ki=9 would allow it)` on stderr only when it vetoes.

Two traps in that check, both found 2026-09-08 by the MEAS-6 session:

- **`benchmark-sequence`'s `-q` is `Option<u32>` with no default** (`src/main.rs:515`), unlike
  `benchmark`, `encode-sequence` and `benchmark-suite`. Without `-q` it falls through to
  `CodecConfig::default()` — so it codes the **pyramid**, at qstep 4.0 and LeGall 5/3 rather than a
  preset. **BUG-37.** Every harness in `scripts/` passes `-q`, so nothing recorded is contaminated;
  a hand-run reproduction of `0025` must pass it too.
- **`--vmaf` had a fixed temp filename**, so two concurrent sessions scored each other's frames —
  measured at 1.0–1.3 VMAF points in either direction, 2–2.5x the 0.5 BLOCK threshold, silent.
  **BUG-36**, fixed 2026-09-08. Rate figures here are byte ratios at bit-identical pixels and were
  never exposed; anything scored with VMAF before that fix was.

**An earlier version of this note said to wait for MEAS-6 because it was "reversing the GOP
default". That was wrong and is withdrawn** — the default moved on 2026-09-06 and MEAS-6 is
correcting the *documents* that still call the pyramid current, not changing a code path. Nothing
gates this item.

**The original question:** abac against Rice on **P-frame residual coefficients**, same pixels, same
GOP structure, at q=75 and q=90 on ≥3 sequences. `GNC_ABAC_COMPARE=1` already reports rate on real
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

</details>

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
