# GNC Temporal Wavelet — Roadmap

## Proven (CPU-staging benchmarks, crowd_run 50fps worst case)
- Temporal Haar in wavelet domain beats ME pipeline: 5.97 bpp vs 7.05 bpp at mul=1.5 (37.82 vs 38.80 dB — ~1 dB PSNR cost)
- At mul=2.0: 5.02 bpp (-29%) at 36.64 dB, mul=3.0: 3.96 bpp (-44%) at 34.93 dB
- Temporal consistency: 0.14 dB max drop vs 2.32 dB with I+P
- No motion estimation, fully tile-independent, fully parallelizable
- Estimated potential at full GPU implementation: 30-45% lower bitrate than ME at matched quality

## Phase 1 — GPU pipeline ✅ (2026-03-03)
1. ✅ **WGSL temporal Haar shader** — `shaders/temporal_haar.wgsl`, per-element Haar lifting, @workgroup_size(256). 1.2ms for 8-frame GOP.
2. ✅ **Multi-level GOP on GPU** — dyadic decomposition with snapshot buffers to prevent read-after-write aliasing between pairs within a level.
3. ✅ **Highpass qstep multiplier** — wired through GPU quantize path via `encode_from_gpu_wavelet_planes`.
4. ✅ **Decoder GPU path** — inverse temporal Haar on GPU, snapshot-based multilevel reconstruction.
5. ✅ **GPU Rice entropy** — temporal frames use GPU Rice encoder (1 submit per frame), 4× speedup over CPU entropy path.

**Results**: crowd_run 8 frames q=75 mul=2.0: 3.91 bpp / 35.82 dB, 40 fps encode+decode. -49% bitrate vs all-I, 0.62 dB temporal consistency.

## Phase 2 — Bitstream format
5. **GNV2 or GP12 extension** — new frame type for temporal wavelet GOP. Header fields: temporal_transform (none/haar/53), gop_levels, highpass_qstep_mul.
6. **Frame ordering in container** — lowpass first, then highpass L2→L1→L0 (decoder needs lowpass before inverse).
7. **Keyframe seeking** — lowpass frame = seekable entry point per GOP.

## Phase 3 — Temporal 5/3 lifting
8. **WGSL temporal 5/3 shader** — predict + update over 4 frames. Reuse spatial 5/3 lifting pattern temporally.
9. **Adaptive temporal transform selection** — mirrors spatial wavelet strategy (CDF 9/7 lossy, 5/3 lossless):
   - **No temporal transform**: ultra-low latency / pure I-frame mode (JPEG XS-like)
   - **Haar**: low latency / 25fps / high q
   - **5/3**: 50-60fps / moderate q / higher compression
   - Selectable per config or auto based on framerate + q target.

## Phase 4 — Optimization
10. **Re-enable CfL in temporal mode** — CfL on temporal wavelet coefficients (both lowpass and highpass).
11. **Adaptive highpass quantization per tile** — static tiles get higher mul, motion tiles get lower. Based on temporal variance.
12. **LL subband handling** — explore simple prediction for LL (small, 0.39% of coefficients but high energy).
13. **Benchmark suite** — automated A/B across all Xiph sequences (rush_hour, crowd_run, stockholm, old_town_cross, ducks_take_off, park_joy) with CSV output.

## Phase 5 — Validation
14. **Full 200-frame benchmarks** on all sequences, compare bitrate/PSNR/temporal consistency vs ME pipeline.
15. **Rate control** for temporal wavelet mode (CBR/VBR with temporal GOP structure).
16. **Broadcast demo** — encode real broadcast content, validate on production-representative material.

## Token budget estimate
- Phase 1-2: ~70% of effort (GPU + bitstream, needs Opus)
- Phase 3-4: ~20% (can partly use Sonnet for tests/benchmarks)
- Phase 5: ~10% (mostly running tests)