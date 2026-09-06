# GNC Benchmark Baseline

Last updated: 2026-09-06 (compression columns; fps columns are older — see below)
Baseline commit: `e872904` — after ENT-1 (packed frequency tables), BUG-11 (Rice stream mapping),
BUG-12, BUG-13 (intra prediction bit-exact) and QUAL-1.
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
| 25  | 35.51 dB | 1.60 | 90.25 | 5 |
| 50  | 40.30 dB | 2.73 | 95.02 | 5 |
| 75  | 44.84 dB | 4.53 | 96.58 | 5 |
| 90  | 50.06 dB | 8.07 | 97.08 | 5 |

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

Three different quantities have been called "encode fps" here. State which one, every time:

| | what it times | measured 2026-09-06 (1080p, ki=8, Rice, M1, **machine not idle**) |
|---|---|---|
| **A — GPU encode phase** | `benchmark-sequence` with Y4M input | 12.2 fps median |
| **B — encoder loop** | the figure `encode-sequence` prints | 5.6 fps median |
| **C — end to end** | wall clock around `encode-sequence`, PNG input | 5.0 fps median |

**A is 2.4x C.** Use A to compare against another codec's encoder, C to claim throughput.

**Timing runs require an idle machine.** Two agents share this Mac; a run taken during a
`cargo test` measured 20% slower than the same run taken after it. Compression figures (bpp,
VMAF, dE00) are deterministic and unaffected; fps and latency are not.

**The 31.7 fps figure quoted in GOALS is not reproducible** and matches none of the three. Its
stated parameters are also inconsistent — "ki=8 ... I+P+B", but ki=8 is below the B-frame
threshold of 9, and the encoder emits 2I+8P. Do not build a density claim on it.

## Sequence Benchmarks (I+P+B, q=75, ki=9, 10 frames, 4:4:4)

> **Stale as of 2026-09-06.** `quality_preset` now vetoes the hierarchical B-pyramid by default
> (BUG-5: it costs +3 to +19% at matched VMAF on camera content and 160 ms of reordering
> latency). Any `-q` run now produces I+P, not I+P+B. Set `GNC_B_PYRAMID=1` to reproduce the
> figures below. These numbers need re-measuring against the new default.

| sequence   | bpp  | PSNR avg | VMAF  | notes                                                        |
|------------|------|----------|-------|--------------------------------------------------------------|
| crowd_run  | 5.55 | 39.04 dB | 99.36 | uniform weights (pre: 5.34 bpp / VMAF 99.12, perceptual)    |
| park_joy   | 4.43 | —        | 99.37 | uniform weights (pre: 4.22 bpp / VMAF 99.12, perceptual)    |
| bbb        | —    | —        | —     | Y4M too short (8 frames) for ki=9                            |

Note: bpp increased at q=75 because quality also increased (+0.25 VMAF, +2.28 dB PSNR for single-frame).
BD-rate vs equal-VMAF comparison: uniform weights save ~18% bpp at matched VMAF.

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

## Video vs H.264 — the headline number (QUAL-1, 2026-09-06)

`scripts/meas1_vs_h264.py`: one normalised reference through PNG for both codecs, the same `vmaf`
binary and arguments for every score, rate from the actual coded bitstream. 1920x1080, 17 frames,
ki=9, 4:2:0, 8-bit, x264 at its defaults.

**BD-rate on PSNR-Y at contribution quality** — q=85,92,96,99 against crf=1,2,4,8:

| | bbb_extended | old_town_cross | crowd_run | **mean** |
|---|---|---|---|---|
| full video (ki=9) | **+129.0%** | **+71.9%** | **+70.6%** | **+90.5%** |
| curve overlap | 49.4–55.9 dB | 50.0–56.2 dB | 49.9–56.3 dB | |

**GNC needs about 1.9x the bitrate of H.264 for the same luma PSNR at contribution quality.**

**Colour, at rate matched to 1%** — CIEDE2000 on decoded RGB, which VMAF cannot see:

| | bbb_extended | old_town_cross | crowd_run |
|---|---|---|---|
| GNC dE00 mean / p95 | **0.611** / 1.304 | **0.911** / 1.943 | 0.837 / 1.844 |
| x264 dE00 mean / p95 | 0.684 / 1.503 | 0.949 / 2.196 | 0.913 / 2.195 |

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
