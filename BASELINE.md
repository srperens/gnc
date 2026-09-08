# GNC Benchmark Baseline

Last updated: 2026-09-08 (compression columns; fps columns are older — see below)
Baseline commit: `0a1b055` — MEAS-10 re-take after PAD-1, INTRA-2, INTER-2, BUG-16 and RATE-2.
Mode: Spatial-only, Rice entropy, uniform subband weights. P-only by default since BUG-5.

**Tile size 256 is unchanged by BUG-11 — byte-identical at all 12 measured points** — so every
compression figure in this file still stands. Non-256 tile figures from before 2026-09-06 do not:
they were scored through a stream mapping that penalised the larger-tile arm by 13–19%.

**Sequence numbers taken before 2026-09-06 with `GNC_DIAGNOSTICS=1` are not valid** — see BUG-7:
that flag clobbered the motion-compensation reference and inflated files by 32%. Re-measure
anything quoted from a diagnostics-enabled run.

## Partial session figure, 2026-09-06 — **stale, do not quote as the session total**

BD-rate on VMAF at matched quality, HEAD-at-the-time against the morning's commit `90ae8b1`, both
binaries built from pinned checkouts:

| stills | mean **−5.04%** | blue_sky −10.59%, touchdown −5.07%, kristensara −2.39%, bbb −2.09% |
|---|---|---|
| video | mean **−5.43%** | old_town −5.99%, aerial −4.87% |

It covers BUG-6, FMT-2/GP17 and TUNE-5 only. **Everything after it is missing**: TUNE-6 (which
partly reversed TUNE-5 above q=80), MEAS-2's AQ rule change and reference-deblock default flip,
and the work the other sessions landed the same day. The figure is not wrong for what it measured —
BD-rate on VMAF is unaffected by the BUG-8 metric fix, since VMAF was always computed from decoded
output — it is simply a mid-session snapshot that reads like a total.

A real session total needs one fresh sweep against `90ae8b1` from current HEAD. Not done: it is a
long measurement and the machine is shared.

## Single-Frame (bbb_1080p, Rice, 4:4:4)

Re-measured 2026-09-06 after BUG-6 made 5 wavelet levels the default at q ≥ 25 (no upper cutoff
— the earlier q ≤ 80 cap was measuring an aliasing bug, see RESEARCH_LOG).

| q   | PSNR     | BPP  | VMAF  | levels |
|-----|----------|------|-------|--------|
| 25  | 35.63 dB | 1.57 | 90.31 | 5 |
| 50  | 40.25 dB | 2.60 | 95.07 | 5 |
| 75  | 44.64 dB | 4.31 | 96.55 | 5 |
| 90  | 49.89 dB | 7.21 | 97.06 | 5 |

**MEAS-10 re-took every row on 2026-09-08 at `0a1b055`.** q=90 is unchanged from INTRA-2's
patch (already included PAD-1). q=25/50/75 moved because PAD-1's decay fill is now the still
default: bpp −4.3% / −4.8% / −4.9%. Canary: `GNC_PAD_FILL=replicate` reproduces the previous
rows **byte-for-byte in the printed columns** (q=50 40.30 dB / 2.73 / 95.02; q=75 44.84 dB /
4.53 / 96.58). At q=75 RGB PSNR drops **0.20 dB** under decay — PAD-1's own gate was q=80–94
where it measured −0.001 dB; VMAF moves −0.03, well under the 0.5 BLOCK. q=100 on this image
is bit-exact (`PSNR inf`, 12.57 bpp, VMAF 97.43 — VMAF is saturated and cannot see lossless).

**The q=90 row moved on 2026-09-08 (INTRA-2).** It read **50.06 dB / 8.07 bpp / VMAF 97.08**.
The dead zone is now 0.6 over q=85..95 where the ladder had 0.5 down to 0.0 — values the quantiser
treats as no dead zone at all — which is worth **BD-rate −5.01% on four stills** and, at matched
rate, **−5.2% dE00 and +0.39 dB of YCoCg-R Y-PSNR**. VMAF moved **−0.01**. `GNC_DEAD_ZONE` still
overrides it in both directions. Decision `0041`.

Two things this row does *not* say. **The bpp move is larger than INTRA-2's own −5.3%** at this
point, because PAD-1 (`0039`) landed the same day and this row had not been re-taken since; the
pre-INTRA-2 build measures 7.62 bpp at 50.06 dB here, not 8.07. And **sequences did not move at
all** — an I-frame that others predict from keeps the ladder's dead zone, verified byte-identical.

**The q=25 row moved on 2026-09-08 (BUG-16) and the others did not.** It read
**35.51 dB / 1.60 bpp / VMAF 90.25**; both sets reproduce on demand, because the change is a
default and not a rewrite — `GNC_SPARSE_DZ=1` gives the old row exactly.

What moved: the fused quantiser applied a sparse dead-zone expansion that no other quantise path
had, so the same configuration produced different coefficients depending on which quantiser ran.
It is now off by default. Priced with the same entropy coder in both arms over three stills at
q=15/25/30, it saved 2.17-4.12% of rate for 0.094-0.214 dB — **BD-rate +1.02% and
direction-inconsistent** (-0.35%, +4.20%, -0.79%), so it bought nothing. VMAF moved **+0.06**, an
improvement and far inside the 0.5-point tolerance.

**These rows are 4:4:4, where the change is confined to q <= 35** — q=40 and above are
byte-identical either way, verified at 40/50/75/85/90/100. That is why the other three rows are
untouched rather than re-measured.

**Do not generalise that to subsampled chroma.** At 4:2:2 and 4:2:0 the expansion fires up to
**q=86** (-0.70% at q=50, -0.05% at q=85, byte-identical from q=90), so any 4:2:2 or 4:2:0 figure
recorded before 2026-09-08 at q <= 86 was measured with it on. An earlier version of this note
said "only q <= 30", which was a 4:4:4 sweep mistaken for the whole answer.

**PSNR figures recorded before BUG-8 (2026-09-06) are not comparable to these.** The metric used
to compare the encoder's `f32` reconstruction; it now compares what the decoder actually emits,
`u32(clamp(f + 0.5, 0.0, peak))`. Clamping helps and rounding hurts, so the correction changes
sign with quality — on this exact table: q=25 +0.27 dB, q=50 −0.04, q=75 +0.37, **q=90 −0.61**.
bpp and VMAF are untouched, because VMAF was always computed from decoded output and was
therefore already honest. If a PSNR number here disagrees with an older one by a few tenths in
either direction, this is why; re-measure rather than reconciling.

The bpp column dropped again with GP17 (Rice-coded stream-length tables, 2026-09-06) at
**bit-identical output** — PSNR and VMAF are unchanged by construction, only the headers shrank.

**The q=90 row moved again with CHROMA-1 (2026-09-06): 50.41 → 50.06 dB, 8.58 → 8.07 bpp.**
`chroma_weight` now stays at 1.2 above q=85 instead of dropping to 1.0, which the sweep showed was
the wrong direction — **−5.2% luma BD-rate for +1.2% on colour**, a 4.3:1 trade. This is a move
along the RD curve, not a regression: most of that 0.35 dB is chroma leaking into an RGB metric,
and the codec's own YCoCg-R luma moves **0.055 dB** while dE00 goes 0.325 → 0.353 mean and
0.721 → 0.772 p95, both far under the JND. q=25/50/75 are untouched (they already used 1.2 or
higher) — q=75 re-measured at 44.84 dB / 4.53 bpp, identical to the row above, as a control.

**And note what VMAF did: nothing.** 97.08 before, 97.08 after, on 6% fewer bits. VMAF is luma-only
and saturated here, so judged on it alone this change reads as a free 6% rate saving with no cost
whatsoever. That is the same illusion that made a 2026-09-05 `chroma_weight` sweep look like a free
15% before a chroma-aware metric reversed its sign. **A VMAF-only verdict on anything touching
chroma is worthless**, and above q=85 a VMAF verdict on anything at all is close to it.

Against the 2026-09-05 rows, **q=25 moved to a different operating point** — 1.89 → 1.60 bpp
(−15%) for −0.77 VMAF. A VMAF drop that arrives with a 15% rate drop is a move along the RD curve,
not a regression, but it means **the q=25 row is not comparable to the old one point-for-point.**
Compare with BD-rate, or at matched rate.

Two corrections this row has already needed, both worth remembering. It was first logged as
34.94 dB / 1.60 bpp, from an uncommitted working tree shared with another session; that does not
reproduce — measure against a hash, not a checkout. It was then logged as 35.24 dB under the old
float metric; it is 35.51 dB now that the metric measures the decoder's real output.

Previous (perceptual weights, #64): q=75 → 42.17 dB / 3.83 bpp / VMAF 95.05

**The fps columns were dropped from this table**, not re-measured: the encode/decode figures in the
2026-09-05 version were taken by an unrecorded method, and this Mac was not idle on 2026-09-06.
See the section below before quoting any throughput number.

## How to read the fps figures in this file

> **Every "M1" in this file is an unreliable label (BUG-29, 2026-09-07).** The dev machine is an
> Apple M5 Pro (20 GPU cores); this file was written as if it were an M1 with 8, and when the
> machine changed is recorded nowhere. The figures were really measured on *something*, so they are
> not fabricated — but they cannot be compared with each other or reproduced from the label. Treat
> any throughput number below as valid only against the commit and harness it names, and re-measure
> before quoting one against another.


Three different quantities have been called "encode fps" here. State which one, every time:

| | what it times | measured 2026-09-06 (1080p, ki=8, Rice, **the Mac, labelled M1 — see above**, machine not idle) |
|---|---|---|
| **A — GPU encode phase** | `benchmark-sequence` with Y4M input | 12.2 fps median |
| **B — encoder loop** | the figure `encode-sequence` prints | 5.6 fps median |
| **C — end to end** | wall clock around `encode-sequence`, PNG input | 5.0 fps median |

**A is 2.4x C.** Use A to compare against another codec's encoder, C to claim throughput.
`benchmark-sequence --throughput` (BUG-32, 2026-09-08) is the CLI form of A without the CPU
PSNR/SSIM tax that used to be 86% of that command's wall clock. Default `benchmark-sequence`
still scores every frame; do not time it.

**Timing runs require an idle machine.** Two agents share this Mac; a run taken during a
`cargo test` measured 20% slower than the same run taken after it. Compression figures (bpp,
VMAF, dE00) are deterministic and unaffected; fps and latency are not.

**The 31.7 fps figure GOALS and the README quoted until 2026-09-07 is not reproducible** and matches none of the three. Its
stated parameters are also inconsistent — "ki=8 ... I+P+B", but ki=8 is below the B-frame
threshold of 9, and the encoder emits 2I+8P. Do not build a density claim on it.

## A second GPU, and the first figures not taken on the Mac (2026-09-07)

NVIDIA RTX 4000 Ada Generation, Ubuntu 24.04.3, driver 580.173.02, Vulkan, wgpu 24.0.5,
`scripts/gpu_tier_bench.py --tier` on the pinned bbb_1080p (`f83f355f…02bf`). From `main` plus
BUG-25's lazy-pipeline change — GNC did not start on Vulkan without it. *(History as of
2026-09-08: the shader was emitting invalid SPIR-V and `51a9ac6` fixed that, so GNC now starts on
Vulkan either way and codes inter frames there. The lazy pipeline stays on its own merits — a
shader's cost is paid by the feature that uses it. `docs/decisions/0029`.)*

| | encode | decode | round trip |
|---|---|---|---|
| **RTX 4000 Ada (Vulkan)** | **13.95 ms** (71.7 fps) | **7.29 ms** (137.2 fps) | **21.2 ms** |
| llvmpipe, CPU rasteriser (Vulkan) | 480.74 ms | 373.88 ms | 854.6 ms |
| Apple M5 Pro (Metal), MEAS-6 2026-09-06 — logged as "M1" | ~47 ms | ~35 ms | ~80 ms |

Reproduced at a second commit under 2x the load: 14.01 / 7.27 ms. **This is CANARY-1's quantity —
the single-frame encode/decode loop — and it is a fourth thing that has been called "encode fps" in
this file.** It is nearest quantity **A** (GPU encode phase) and is not comparable to B or C.

**Do not quote the Mac row against the RTX row as a speedup.** It is cross-machine, cross-backend,
possibly at a different q, and the Mac figure was taken by a different harness. The controlled
version is one command — this same harness on an idle Mac — and it has not been run.

**Cross-backend output, measured for the first time.** Same commit, same input: q=100 lossless is
**byte-identical** between Metal and Vulkan (`5c4539d8…`, 3 235 737 B), and q=75 lossy **differs by
one byte** (1 173 797 vs 1 173 796). Every file decodes to identical pixels on both backends. So
**the decoder is bit-exact across backends and the lossy encoder is not** — any regression test that
hashes lossy encoder output will fail across backends, and a conformance suite must require decoder
bit-exactness, not encoder reproducibility.

## Sequence Benchmarks (I+P, ki=9, 10 frames, 4:4:4)

The pre-2026-09-08 rows (crowd_run 5.55 bpp / 39.04 dB / VMAF 99.36, park_joy 4.43 bpp /
VMAF 99.37, both at q=75) are withdrawn. They were I+P+B through three defects at once:
the B-pyramid that `quality_preset` has vetoed since BUG-5, BUG-27's intra-qstep P-frame
reference (live at q ≤ 80), and INTER-2's double inter dead zone (live at q ≤ 88).
`park_joy` is not in the tree. `GNC_B_PYRAMID=1` reproduces the pyramid mix, not those numbers.

**Re-taken 2026-09-08 (MEAS-10) against the shipped default** — I+P, no B-pyramid, INTER-2's
inter dead zone, PAD-1 refused on references. 10 frames, ki=9, 4:4:4, Rice, `2I+8P+0B`.
Sources: crowd_run, old_town_cross, bbb_extended PNG sequences.

| sequence        | q  | bpp  | PSNR avg | PSNR min | VMAF  | I-only bpp | vs I-only |
|-----------------|----|------|----------|----------|-------|------------|-----------|
| crowd_run       | 75 | 8.39 | 42.47 dB | 42.27 dB | 99.68 | 8.76       | −4.2%     |
| crowd_run       | 90 | 13.17| 49.60 dB | 49.23 dB | 99.72 | 12.51      | **+5.3%** |
| old_town_cross  | 75 | 8.39 | 42.36 dB | 42.17 dB | 99.38 | 8.25       | **+1.7%** |
| old_town_cross  | 90 | 13.04| 49.59 dB | 49.22 dB | 99.70 | 11.94      | **+9.2%** |
| bbb_extended    | 75 | 3.07 | 43.55 dB | 42.37 dB | 97.88 | 4.53       | −32.3%    |
| bbb_extended    | 90 | 6.77 | 50.28 dB | 50.06 dB | 99.16 | 7.63       | −11.3%    |

The I-only column is the comparison that survives: on camera content at q=90, **inter costs more
than all-intra**. Animation still saves. That is INTER-1's finding on current HEAD, not a new one.
No fps is quoted — the machine was not idle.

## Lossless sequences (q=100, 8 frames, 4:4:4, Rice)

New section 2026-09-08 (LOSSLESS-2). There was no lossless *sequence* row here before, because
until BUG-39 (`0064`) `q=100` video was not bit-exact and there was nothing to regress against.
Container bytes, `encode-sequence`, and every frame md5-identical to its source PNG through
`decode-sequence`:

| sequence | ki=2 | ki=9 | mix at ki=9 |
|---|---|---|---|
| crowd_run | 25 856 146 | 25 856 146 | **8I+0P** |
| old_town_cross | 25 247 023 | 25 247 023 | **8I+0P** |
| blue_sky | 17 294 725 | 17 294 725 | **8I+0P** |
| bbb (animation) | 25 485 001 | 25 183 470 | 1I+7P |

**ki does not change the camera rows, and that is the feature, not a copy-paste.** A `q=100`
P-frame that costs more than an I-frame of the same picture is re-coded as an I-frame (`0070`), so
camera content converges to all-intra whatever the keyframe interval says. Before that change the
same rows read 35 712 641 / 43 003 751 (crowd_run) and 35 209 443 / 42 778 003 (old_town_cross) —
up to **+69% for identical pixels**. `GNC_LOSSLESS_INTRA_RECODE=0` reproduces the old numbers.

Animation keeps its P-frames and its rows are unchanged. **No fps is quoted: eight sessions shared
the GPU.**

## Reported bitrate correction (2026-09-05)

`CompressedFrame::byte_size()` counted motion vectors as 4 raw bytes per block while the
bitstream delta-codes them as varints, inflating reported inter-frame sizes by up to 9x and
sequence bpp by **27-58%**. Fixed; `byte_size()` now measures by serializing.

**Every sequence bpp figure in this file and in RESEARCH_LOG predating 2026-09-05 is inflated by
that much.** Single-frame (intra) figures are unaffected — I-frames carry no motion vectors.
Corrected reference points, bbb 1080p ki=9 4:2:0, 17 frames:

| q | reported before | corrected |
|---|---|---|
| 40 | 0.90 bpp | **0.57 bpp** |
| 70 | 1.54 bpp | **1.22 bpp** |

Per-frame at q=70: I 820 KB, P 304-380 KB, B 108 KB. For scale, x264 at matched VMAF spends
I 439 KB, P 39 KB, B 14 KB — intra is ~1.9x, inter is **8-10x**.

## Intra vs H.264 and JPEG 2000 (2026-09-05, like-for-like)

6 frames of bbb at 1080p, 4:4:4, all-intra, one PNG-derived reference, all three codecs scored by
the same `vmaf` binary. `scripts/meas1_vs_h264.py --chroma 444 --keyint 1`.

| | bpp @ VMAF 96 | bpp @ PSNR-Y 43 |
|---|---|---|
| GNC | 2.678 | 3.213 |
| H.264 intra (x264, i444) | 1.880 | 1.874 |
| JPEG 2000 (openjpeg) | 1.496 | 2.201 |

**GNC needs 1.42x H.264 intra and 1.79x JPEG 2000 at VMAF 96.**

This supersedes the +13.9% and +17.6% figures below, which are RGB PSNR on a single still image.
`rd-curve --compare-codecs` still reports +17.6% vs JPEG 2000, so the difference is methodology,
not drift — but VMAF on a shared reference is the measurement to quote.

**JPEG 2000 beats H.264 intra here**, so GNC is losing to another wavelet codec, not to a
fundamentally different design. Unlike the inter gap, intra has an existence proof that the rate
is reachable.

**Superseded and made worse by MEAS-9 (2026-09-07).** Two corrections. The 1.79x above was scored
on **VMAF**, which is luma-only and saturated at this operating point — do not quote it. And
`opj_compress` defaults to the **reversible 5/3** transform, so that arm was JPEG 2000 with the
wrong transform for a lossy comparison: `-I` gains it 2-3 dB at matched rate. Measured correctly,
on four images through one metric path, **GNC needs +54.2% BD-rate on RGB PSNR and +79.7% on
Y-PSNR against JPEG 2000 in 9/7 mode** — a larger gap than this section implies, and larger than
GNC's gap to ProRes 4444 (+20.2% / +29.3%) or JPEG XS 4:4:4 (**−10.2%** / +29.4%). The conclusion
that the rate is reachable stands and is now stronger: the transform is the same, so what is left
is the entropy coder. See RESEARCH_LOG 2026-09-07 and `scripts/meas9_contribution.py`.

**Read those figures with the padding tax in hand (INTRA-1 step 3, 2026-09-08, `0034`).** GNC pads
every plane to whole tiles and codes the padding, so a 1920x1080 frame is coded as 2048x1280 and
**20.9% of the coded samples are outside the picture**, while JPEG 2000 in whole-picture mode codes
none — and both arms are divided by the visible pixel count. That is worth **~6.6 of the points**
in any GNC-vs-J2K figure taken at native resolution: measured on one q ladder, with `--abac`, the
gap reads **+26.5% RGB / +51.2% Y on native frames and +19.8% / +40.7% on padding-free crops**. No
number here is retracted; every cross-codec figure taken at a resolution that is not a whole
number of tiles carries the tax. **A caution that came out of the same run: those native figures
are not ENT-4's +27.1% / +48.3% even though they are close on the mean** — ENT-4 used a q=60–99
ladder against q=80–98 here, and a BD-rate is only comparable to another over the same overlapping
quality range. Per image the ladder alone moves the figure up to 2.2 points.

**And every still figure in this section predates PAD-1 (2026-09-08, decision `0039`), which
returned −4.63% RGB / −4.60% Y of intra rate** by fading the tile padding flat instead of
replicating it. So the GNC-vs-J2K intra gap on stills is about 4.6 points smaller than the numbers
here, and any still byte count taken before that commit is reproducible only with
`GNC_PAD_FILL=replicate`. **Sequence figures are untouched and byte-identical** — the fill is
refused on anything a frame predicts from.

## Video vs H.264 — the headline number (MEAS-10, 2026-09-08)

`scripts/meas1_vs_h264.py`: one normalised reference through PNG for both codecs, the same `vmaf`
binary and arguments for every score, rate from the actual coded bitstream. 1920x1080, 17 frames,
ki=9, 4:2:0, 8-bit, x264 at its defaults. Pinned to `0a1b055`. Same ladder as QUAL-1:
q=85,92,96,99 against crf=1,2,4,8.

**BD-rate on PSNR-Y at contribution quality:**

| | bbb_extended | old_town_cross | crowd_run | **mean** |
|---|---|---|---|---|
| full video (ki=9), Rice | **+128.5%** | **+70.2%** | **+68.8%** | **+89.2%** |
| full video (ki=9), `--abac` | **+91.8%** | **+53.1%** | **+53.0%** | **+66.0%** |
| curve overlap | 49.9–56.0 dB | 49.8–55.9 dB | 49.8–56.0 dB | |
| QUAL-1 (2026-09-06), Rice | +129.0% | +71.9% | +70.6% | +90.5% |

**GNC Rice needs about 1.9x the bitrate of H.264 for the same luma PSNR at contribution quality.
`--abac` is 1.66x** — same pixels as Rice at every rung (PSNR-Y identical to two decimals),
only the bytes moved. That is the canary the path ran. Saving vs Rice decays with quality
(crowd_run −12.2% at q=85 to −3.7% at q=99), which is ENT-3's finding on this ladder.
Rice stays the default; quote **+89.2%** unless the command included `--abac`.

The move from QUAL-1's +90.5% to +89.2% is **1.3 points**, all in the direction INTER-2
predicted: only q=85 of this ladder sits in the inter-dead-zone change (q ≤ 88), so a
−4.77% GNC-vs-GNC BD-rate on that one rung dilutes to about a point against x264. VMAF
BD-rate on the same data is not a number (old_town **+2548%** Rice / **+2289%** abac at
VMAF 99.8–99.8). Do not quote it.

> **INTER-2's prediction is now a measurement.** The 2026-09-06 +90.5% stands as the QUAL-1
> record. Quote **+89.2%** for current HEAD. `docs/decisions/0041`.

**Caveat added 2026-09-07, and RATE-2's fix does NOT lift it (updated 2026-09-08).** Two of those
four rungs are inside GNC's dominated range: above q≈95–98 the lossy path costs *more bytes than
GNC's own bit-exact lossless* (+28.9% mean at q=99, +40.6% on blue_sky), and this ladder runs to
q=96 and q=99. So the +90.5% scores GNC partly through rungs it should never operate on, which
makes the figure **pessimistic against GNC by an unmeasured amount** — not wrong, but not a clean
1.9x either. The ladder is also not monotonic in rate (a rung's bpp can fall as q rises), so any
interpolation by rate should flag that.

**RATE-2 shipped on 2026-09-08 (`docs/decisions/0036`) and did not move this figure, because that
fix was intra-only. RATE-3 landed the same day (`docs/decisions/0044`) and does move it.** A still
at q=95–99 codes both ways and keeps the smaller (mean −21.66% at q=99), and since RATE-3 a
**sequence I-frame does too**: the gate that refused the fallback inside a sequence is lifted, worth
mean **−6.09%** of sequence bytes over three sequences at q ∈ {95, 99} and ki ∈ {2, 9}, up to
**−16.19%**, at a worst quality move of −0.01 dB.

**Those two figures were −4.28% and −13.16% until 2026-09-08 and moved for a fix, not a re-take**
(BUG-47, `docs/decisions/0072`): `lossless_sibling` did not carry `pad_fill_decay`, so the bit-exact
candidate was coded with a still's decay-filled padding while acting as a sequence reference. The
same twelve points now have **0 of 12 worse than the control**, where RATE-3 recorded two
regressions of +0.58% and +0.40%. The 4:2:0 ladder below is affected in the same direction and by
an unmeasured amount — the fix applies wherever the bit-exact sibling is used, and only 4:4:4 was
measured.

**So the ladder's top two rungs are stale.** It is q=85/92/96/99 and `quality_preset` sets the
fallback for q = 95..=99 only, so **q=96 and q=99 move; q=85/92 do not.** The direction favours GNC
and the size is not guessable from RATE-3's sweep: this ladder is 4:2:0 at ki=9, where RATE-3
measured −2.4% to −5.7%, not the −13% of its best point — and both ends of that range predate
BUG-47, so they are floors rather than estimates now. **+89.2% stands as recorded until
`meas1_vs_h264.py` is run again** — as with the INTER-2 note above, a predicted direction is not a
measurement. Two of the four rungs have now moved for two independent reasons (INTER-2 at q=85,
RATE-3 at q=96 and q=99), which makes re-running this ladder the highest-value measurement in the
file.

**Colour, at rate matched to 1%** — CIEDE2000 on decoded RGB, which VMAF cannot see:

| | bbb_extended | old_town_cross | crowd_run |
|---|---|---|---|
| GNC dE00 mean / p95 | 0.611 / 1.304 | 0.911 / 1.943 | 0.837 / 1.844 |
| x264 dE00 mean / p95 | 0.684 / 1.503 | 0.949 / 2.196 | 0.913 / 2.195 |

**These two rows are withdrawn (CHROMA-2, 2026-09-07, decision 0020).** They do not reproduce —
the same nominal configuration now gives 32% more bytes *and* worse dE00, and the table was taken
an hour before CHROMA-1 changed q>=85 output. The rate-matched control has **x264 ahead on colour
in 6 runs of 6**. Do not quote them; the luma BD-rate above is unaffected.

GNC is ahead on colour on all three while **7.4–8.8 dB behind on luma** at those same points — the
two codecs allocate rate differently between luma and chroma. Quote both numbers or the comparison
misleads in whichever direction suits. (crowd_run's pair is 1.08x on rate, so its colour win is
partly bought; the matched pairs give +4.1% and +10.7%.)

### Three rules for quoting these

1. **PSNR leads above q=85, not VMAF.** Widening the quality ladder from 1.8 dB to 6.5 dB of
   overlap moved the VMAF BD-rate by a mean of **47.5 points** (old_town +81.1% → +191.4%) and the
   PSNR BD-rate by **1.0 point**. VMAF reads 99.62–99.68 across a 6 dB PSNR spread: saturated, so
   the fit is on noise. Never quote a VMAF BD-rate at this end.
2. **Superseded: +456.7% / +493.9% / +672.1% (MEAS-1, 2026-09-05).** That was measured at
   *distribution* bitrates (crf 18–38) with the quality ladder above q=92 dead. Nothing in the
   coder changed between the two measurements. Treat it as a historical distribution-bitrate figure
   only. Its stated sources are also not reproducible: `bbb.y4m` has 8 frames, not 17, and no
   `touchdown` sequence exists in the tree.
3. **The +13.9% spatial figure is a third quantity** — PSNR on single stills against H.264 all-I.
   Not contradictory, just a different measurement. The intra-only figures (+54.6% / +46.3%,
   ki=1) predate the fix to the high-q ladder and have not been re-run.

## Regression Rules

Any change that regresses any sequence benchmark is rejected unless the regression is explained and accepted in writing in RESEARCH_LOG.md.
Tolerances: VMAF −0.5 pts (BLOCK), bpp +3% (BLOCK), PSNR −0.3 dB (flag).
