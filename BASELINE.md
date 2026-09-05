# GNC Benchmark Baseline

Last updated: 2026-09-05
Baseline commit: (subband weight fix — uniform weights default)
Mode: Spatial-only, I+P+B, Rice entropy, ki=9, 7B-per-group pyramid, uniform subband weights

## Single-Frame (bbb_1080p, Rice, 4:4:4)

| q   | PSNR     | BPP  | Encode  | Decode  | VMAF  |
|-----|----------|------|---------|---------|-------|
| 25  | 35.44 dB | 1.89 | 28.5 fps | 46.9 fps | 91.02 |
| 50  | 40.34 dB | 2.79 | 28.6 fps | 41.7 fps | 95.08 |
| 75  | 44.45 dB | 4.59 | 29.6 fps | 41.3 fps | 96.56 |
| 90  | 51.0 dB  | 9.65 | 39 fps   | 55 fps   | —     |

Previous (perceptual weights, #64): q=75 → 42.17 dB / 3.83 bpp / VMAF 95.05

## Sequence Benchmarks (I+P+B, q=75, ki=9, 10 frames, 4:4:4)

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
