# GNC Benchmark Baseline

Last updated: 2026-03-09
Baseline commit: 114a2f9 (quarter-pel ME, after B-frame 4:2:0 chroma fix)
Mode: Spatial-only, I+P+B, Rice entropy

## Single-Frame (bbb_1080p, Rice, 4:4:4)

| q   | PSNR     | BPP  | Encode  | Decode  | VMAF  |
|-----|----------|------|---------|---------|-------|
| 25  | 32.89 dB | 1.50 | 35.1 fps | 62.8 fps | 85.10 |
| 50  | 37.53 dB | 2.22 | 35.9 fps | 52.1 fps | 89.68 |
| 75  | 42.17 dB | 3.83 | 35.3 fps | 52.0 fps | 95.05 |
| 90  | 51.0 dB  | 9.65 | 39 fps   | 55 fps   | —     |

## Sequence Benchmarks (I+P+B, q=75, ki=8, 50 frames)

| sequence   | bpp  | PSNR avg | PSNR min | fps_enc | fps_dec |
|------------|------|----------|----------|---------|---------|
| crowd_run  | 6.93 | 38.80 dB | 38.34 dB | —       | —       |
| rush_hour  | 2.03 | 41.12 dB | 40.75 dB | —       | —       |

## Regression Rules

Any change that regresses any sequence benchmark without explicit Team Lead approval is rejected.
Tolerances: VMAF −0.5 pts (BLOCK), bpp +3% (BLOCK), PSNR −0.3 dB (flag).
