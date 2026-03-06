# GNC Benchmark Baseline

Last updated: 2026-03-06
Baseline commit: 617d8e6 (v0.1-spatial)
Mode: Spatial-only, I+P+B, Rice entropy, q=75

## Sequence Benchmarks (1080p, 120 frames)

| sequence   | bpp  | PSNR avg | PSNR min | fps_enc | fps_dec |
|------------|------|----------|----------|---------|---------|
| crowd_run  | 6.99 | 38.78 dB | -        | 20.8    | -       |
| park_joy   | 7.76 | 39.15 dB | -        | 20.7    | -       |
| rush_hour  | 2.01 | 41.15 dB | -        | 21.1    | -       |
| stockholm  | 4.15 | 38.78 dB | -        | 42.6    | -       |

## Single-Frame (bbb_1080p, Rice)

| q   | PSNR    | BPP  | Encode | Decode |
|-----|---------|------|--------|--------|
| 25  | 33.2 dB | 1.71 | 39 fps | 72 fps |
| 50  | 37.7 dB | 2.37 | 40 fps | 60 fps |
| 75  | 42.8 dB | 4.01 | 40 fps | 59 fps |
| 90  | 50.5 dB | 8.90 | 40 fps | 63 fps |

## Regression Rules

Any change that regresses any sequence benchmark without explicit Team Lead approval is rejected.
Fill in missing values (PSNR min, fps_dec) when Validator first runs the full suite.
