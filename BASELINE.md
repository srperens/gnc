# GNC Benchmark Baseline

Last updated: 2026-03-11
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

## Regression Rules

Any change that regresses any sequence benchmark without explicit Team Lead approval is rejected.
Tolerances: VMAF −0.5 pts (BLOCK), bpp +3% (BLOCK), PSNR −0.3 dB (flag).
