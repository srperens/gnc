# 0008 — Five wavelet levels, and the skip bitmap that had to grow a byte

**Date:** 2026-09-06. **Status:** accepted. **Closes:** BUG-6.

## Why the cap existed

Nothing had decided on four levels. `num_groups = wavelet_levels * 2` met a `MAX_GROUPS = 8` that
appeared in five places — `rice_gpu.rs`, `rans_gpu_encode.rs` and three shaders — and the Rice
tile header carried its per-group skip bitmap in a single `u8`. Four levels sat exactly on that
ceiling, so the limit read like a design choice while it was an accident of the first
implementation. `GNC_WAVELET_LEVELS=5` panicked.

That mattered because the 3 → 4 step had measured 5–17% better (TUNE-2), well above what an
ideal-entropy model predicted (1.2% against 6% measured) — Rice adapts `k` per subband, and an
offline entropy estimate cannot see that. An offline estimate was the only evidence that levels 5
and 6 were worthless, and it was the same estimate that had already been wrong by 5x.

## What was decided

**Widen to 12 groups (6 levels) in the code, and let the tile size decide the real limit.**
`CodecConfig::max_wavelet_levels()` derives it: the DWT runs per tile, each level halves the tile,
and the LL band has to stay at 8 samples or more for per-subband statistics to mean anything. A
256 px tile therefore allows 5, a 128 px tile 4.

**Widen the skip bitmap without a format generation.** Ten groups do not fit in a byte. The
alternative to a new magic was to key the width off a field the decoder already reads — and
`num_groups` is in the tile header, four bytes ahead of the bitmap. One byte at ≤8 groups, two
little-endian bytes above. Every file written before today parses byte-for-byte unchanged, and
GP16 did not have to become GP17. The per-odd-stream checkerboard-k block follows the same rule.

This is worth stating as a pattern: **a tile payload can version itself off its own header**, and
should, before a frame-level generation bump is considered.

## What the half-landed version taught

The widening was in the tree with the shader statistics arrays still eight wide, while the loops
around them ran to twelve. WGSL clamps an out-of-bounds workgroup index instead of trapping, so
groups 8–11 accumulated into group 7, the two deepest levels were coded with the wrong `k`, and
**every test still passed** — `k` travels in the bitstream, so the file decoded exactly. It was
simply larger than it should have been, and the measurements taken from it understated the gain.

The rule that follows: **when a capacity constant is raised, every array dimensioned by it has to
be found, including the ones in shader source that a Rust type check never sees.** A grep for the
old literal is the cheap version of that search. Correctness tests will not do it for you when the
data path is statistical rather than exact.

There was a fourth array, found after this was first written: `quantize_histogram_fused.wgsl`, a
separate producer of the same rANS histogram buffer, still on the 8-group stride. It broke in a way
worth recognising again — a writer and a reader disagreeing about a stride makes **element 0 look
correct and everything after it read as zero**, because only index 0 has a base offset of 0. Every
tile but the first came back with `num_groups = 0`.

## Result

5 levels at q ≥ 25, 4 below; `GNC_WAVELET_LEVELS` still overrides.

| image (q=70) | Δ rate | Δ PSNR | Δ VMAF |
|---|---|---|---|
| blue_sky_1080p | −4.0% | +1.40 dB | 0.00 |
| kristensara_720p | −1.7% | +0.86 dB | +3.32 |
| bbb_1080p | −1.0% | +0.01 dB | 0.00 |

Smooth content gains most, edge-heavy animation least, which is what a deeper decomposition should
do.

The range came from BD-rate on VMAF, not from a q sweep, and the distinction mattered: at equal q
the fifth level scores *slightly worse* VMAF, because it also removes 1–16% of the bits. Compared
at equal quality it is −1.8% to −6.9% BD-rate over q=25–70 (mean −3.7%) on four images.

**The lower cutoff is real.** Over q=15–35 the sign flips on three of four images (+2.2% to +4.3%);
below q≈25 the two deepest subbands quantise to all-zero on most tiles, so their k values and
per-group rANS frequency tables are overhead with nothing to pay for them. At q=15–20 five levels
costs more bits *and* about a VMAF point.

**The upper cutoff was not.** It was set against the buggy numbers, exactly as suspected below.
Swept q=85/90/95/99 on all four images with the accumulators correct: 16 of 16 points save 0.3–0.6%
of the bits at PSNR and VMAF identical to two decimals, and q=100 stays bit-exact lossless while
shrinking 0.2%. Removed.

## Left open

Nothing on the level count. The wider lesson stands: three of the four caps failed silently, and
one of them — the phase-1 accumulator aliasing — was invisible to every possible test, because the
bitstream carries `k` and the file still decoded exactly. It was only bigger.
