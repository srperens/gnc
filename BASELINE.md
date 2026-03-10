# GNC Benchmark Baseline

Last updated: 2026-03-10
Baseline commit: 638b77a (hierarchical B-frame pyramid — #42 done)
Mode: Spatial-only, I+P+B, Rice entropy, ki=9, 7B-per-group pyramid

## Single-Frame (bbb_1080p, Rice, 4:4:4)

| q   | PSNR     | BPP  | Encode  | Decode  | VMAF  |
|-----|----------|------|---------|---------|-------|
| 25  | 32.89 dB | 1.50 | 35.1 fps | 62.8 fps | 85.10 |
| 50  | 37.53 dB | 2.22 | 35.9 fps | 52.1 fps | 89.68 |
| 75  | 42.17 dB | 3.83 | 35.3 fps | 52.0 fps | 95.05 |
| 90  | 51.0 dB  | 9.65 | 39 fps   | 55 fps   | —     |

## Sequence Benchmarks (I+P+B, q=75, ki=9, 10 frames, 4:4:4)

| sequence   | bpp  | PSNR avg | VMAF  | notes                                  |
|------------|------|----------|-------|----------------------------------------|
| crowd_run  | 6.00 | 38.57 dB | 99.13 | hierarchical pyramid #42 (pre: 6.15)   |
| park_joy   | 4.75 | ~39.3 dB | 99.14 | hierarchical pyramid #42 (pre: 4.77)   |
| bbb        | —    | —        | —     | Y4M too short (8 frames) for ki=9      |

## Regression Rules

Any change that regresses any sequence benchmark without explicit Team Lead approval is rejected.
Tolerances: VMAF −0.5 pts (BLOCK), bpp +3% (BLOCK), PSNR −0.3 dB (flag).
