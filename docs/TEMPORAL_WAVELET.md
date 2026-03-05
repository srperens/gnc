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

## Phase 2 — Bitstream format ✅ (2026-03-03)
5. ✅ **GNV2 container format** — 34-byte header (magic, temporal_transform, gop_size, highpass_qstep_mul), 22-byte per-frame index entries with frame_role (lowpass/highpass/tail), temporal_level, gop_index, PTS.
6. ✅ **Frame ordering in container** — lowpass first, then highpass deepest-to-finest (L2→L1→L0). Each frame serialized as GP12 blob.
7. ✅ **Keyframe seeking** — `seek_to_temporal_keyframe()` finds GOP by PTS. Lowpass frames marked as seekable (frame_role=0).
8. ✅ **CLI integration** — `benchmark-sequence --temporal-wavelet haar -o file.gnv2` writes GNV2 container. Full roundtrip via `serialize_temporal_sequence` / `deserialize_temporal_sequence`.
9. ✅ **CLI decode** — `decode-sequence` auto-detects GNV2 magic, decodes temporal wavelet GOPs, writes output PNGs.
10. ✅ **WASM player** — `GnvPlayer` detects GNV2, decodes GOPs on demand via async `decode_temporal_group_rgba_wasm()` with per-frame GPU temporal Haar inverse + spatial inverse + pack u8 + async readback.

## Phase 2.5 — Full GPU decode + zero-copy player ✅ (2026-03-03)
11. ✅ **GPU entropy decode in temporal path** — replaced CPU `entropy_helpers::entropy_decode_plane()` with `prepare_frame_data()` + GPU Rice/rANS/Bitplane/Huffman dispatch. Zero CPU compute in decode pipeline. Shared `dispatch_entropy_decode()` helper for all 5 entropy backends.
12. ✅ **Zero-copy GNV2 playback** — `decode_and_present()` now handles GNV2 with GPU blit to canvas surface, same as GNV1. No more readback fallback.
13. ✅ **Two-phase GOP decode** — Phase 1 (once per GOP): entropy decode + dequantize + temporal Haar inverse → per-frame wavelet-domain GPU buffers. Phase 2 (per frame): inverse spatial wavelet + color convert + crop + blit. Eliminates GOP-boundary stutter.
14. ✅ **Per-GOP diagnostics** — `print_temporal_gop_diagnostics()` in `encoder/diagnostics.rs`: temporal decomposition breakdown, per-frame Rice stats, coefficient analysis, bit budget, reconstructed quality, content-aware warnings. Output to `.diag` files, displayed in web player.
15. ✅ **Demo parity** — `generate_demos_tw.sh` mirrors all GNV1 demos with matching content, quality, frame count, fps for direct comparison.

**Results**: 1080p 50fps: 3.9ms decode, 260fps presentation rate. GPU blit zero-copy. No requestAnimationFrame violations.

## Phase 2.6 — GOP decode optimization + player diagnostics ✅ (2026-03-03)
16. ✅ **GPU buffer caching** — `ensure_tw_bufs()` allocates frame/snapshot buffers once, reuses across GOPs. Eliminates 32 `create_buffer` calls (~256MB allocation) per GOP.
17. ✅ **Batched temporal Haar inverse** — all 3 planes in single `queue.submit()` (was 3 separate submits).
18. ✅ **Double-buffered GOP prefetch** — two `TwBufSet`s, next GOP decoded at midpoint of current GOP. Hides GOP-boundary latency during playback.
19. ✅ **Sub-ms timing** — `performance.now()` via web-sys replaces `Date.now()` (1ms resolution) for accurate per-frame seek/decode measurement.
20. ✅ **Timing log panel** — per-frame table (Frame, Seek ms, Decode ms, Total ms, FPS) with Clear/Copy CSV, summary stats (avg/min/max).
21. ✅ **Dropdown file selector** — grouped `<select>` (GNV1/GNV2 optgroups) replaces button grid.

## Phase 3 — Adaptive Haar + temporal 5/3 (experimental) ✅ (2026-03-05)
22. ✅ **WGSL temporal 5/3 shader** — `shaders/temporal_53.wgsl`, two-pass predict+update lifting. Available via `--temporal-wavelet 53` but not auto-selected (4-frame 5/3 produces 2 lowpass + 2 highpass, worse energy compaction than Haar's 1+3 split).
23. ✅ **Adaptive temporal transform selection** — `--temporal-wavelet auto` (default):
   - **fps < 1**: None (still image / slideshow)
   - **fps ≤ 25 or q ≥ 90**: Haar gop=2 (1 level, minimal latency)
   - **fps > 25**: Haar gop=4 (2 levels, 1 lowpass + 3 highpass, best compression)
   - Temporal wavelet is now the **default** encoding mode. MV (I+P+B) requires explicit `--temporal-wavelet none`.
24. ✅ **Diagnostics fix** — 5/3 mode now labels s1 as `[L1]` (second lowpass) instead of `[H0]`. Summary correctly separates lowpass and highpass byte counts.

**Results**: bbb 8 frames q=75 Haar gop=4: 1.77 bpp (vs 2.48 bpp for 5/3, vs 4.30 bpp all-I). Haar gop=4 = 59% bitrate reduction vs all-I.

## Phase 4 — Optimization
24. **Re-enable CfL in temporal mode** — CfL on temporal wavelet coefficients (both lowpass and highpass).
25. **Adaptive highpass quantization per tile** — static tiles get higher mul, motion tiles get lower. Based on temporal variance.
26. **LL subband handling** — explore simple prediction for LL (small, 0.39% of coefficients but high energy).
27. **Benchmark suite** — automated A/B across all Xiph sequences (rush_hour, crowd_run, stockholm, old_town_cross, ducks_take_off, park_joy) with CSV output.

## Phase 5 — Validation
28. **Full 200-frame benchmarks** on all sequences, compare bitrate/PSNR/temporal consistency vs ME pipeline.
29. **Rate control** for temporal wavelet mode (CBR/VBR with temporal GOP structure).
30. **Broadcast demo** — encode real broadcast content, validate on production-representative material.

## Token budget estimate
- Phase 1-2: ~70% of effort (GPU + bitstream, needs Opus) ✅
- Phase 2.5: ~5% (GPU decode + zero-copy player) ✅
- Phase 3-4: ~15% (can partly use Sonnet for tests/benchmarks)
- Phase 5: ~10% (mostly running tests)
