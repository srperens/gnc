# GNC Benchmark Baseline

Last updated: 2026-09-06 (compression columns; fps columns are older — see below)
Baseline commit: (BUG-6 — 5 wavelet levels at q ≥ 25)
Mode: Spatial-only, Rice entropy, uniform subband weights. P-only by default since BUG-5.

## Single-Frame (bbb_1080p, Rice, 4:4:4)

Re-measured 2026-09-06 after BUG-6 made 5 wavelet levels the default at q ≥ 25 (no upper cutoff
— the earlier q ≤ 80 cap was measuring an aliasing bug, see RESEARCH_LOG).

| q   | PSNR     | BPP  | VMAF  | levels |
|-----|----------|------|-------|--------|
| 25  | 34.94 dB | 1.60 | 90.21 | 5 |
| 50  | 40.34 dB | 2.76 | 95.02 | 5 |
| 75  | 44.47 dB | 4.56 | 96.57 | 5 |
| 90  | 51.02 dB | 8.60 | 97.08 | 5 |

Against the 2026-09-05 rows, q=50 and q=75 are unchanged within noise; **q=25 moved to a different
operating point** — 1.89 → 1.60 bpp (−15%) for −0.50 dB and −0.81 VMAF. Part of that is BUG-6
(isolated: −6.4% rate, +0.28 dB, −0.53 VMAF at q=25) and the rest predates it. A VMAF drop that
comes with a 15% rate drop is a move along the RD curve, not a regression — but it means **the
q=25 row is not comparable to the old one point-for-point.** Compare with BD-rate, or at matched
rate.

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

## Video vs H.264 (MEAS-1, 2026-09-05) — the headline number

Measured with `scripts/meas1_vs_h264.py`: VMAF-scored, one normalised reference for both codecs,
BD-rate over the overlapping quality range. 1080p, 4:2:0, 17 frames, x264 at its defaults.

| | bbb | touchdown | old_town |
|---|---|---|---|
| full video (ki=9) | **+456.7%** | **+493.9%** | **+672.1%** |
| intra only (ki=1, 8 frames) | +54.6% | +46.3% | — |

**GNC needs roughly 5-7x the bitrate of H.264 for the same VMAF on video.** Intra accounts for
about +50% of that; inter coding multiplies it a further 8-10x.

This supersedes the +13.9% spatial BD-rate quoted below and elsewhere in the repo. That figure is
PSNR-based, on single still images, against H.264 all-I — a different measurement, not a
contradictory one, but not the number that matters for a video codec. Quote the table above when
asked how GNC compares.

## Regression Rules

Any change that regresses any sequence benchmark without explicit Team Lead approval is rejected.
Tolerances: VMAF −0.5 pts (BLOCK), bpp +3% (BLOCK), PSNR −0.3 dB (flag).
