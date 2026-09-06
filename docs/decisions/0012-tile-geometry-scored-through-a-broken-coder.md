# 0012 — The tile-size default stays 256, and every past tile-size result was scored through a bug

**Date:** 2026-09-06
**Status:** Accepted (reverses the conclusion of BUG-11's own morning measurement, and of #47)

## The decision

Two things were decided, and only one of them is what the investigation set out to decide.

1. **Rice's stream mapping is now tile-width-aware** — streams walk the tile in column-major
   order, cut into 256 contiguous segments, rather than mapping coefficient *i* to stream
   `i % 256`.
2. **The default tile size stays at 256 px**, even though the fix reverses the sign of the
   tile-size comparison, because the corrected prize is about 1%.

## Why the mapping had to change

Rice maps a tile's coefficients across 256 independent streams for GPU parallelism. The property
the adaptive Rice parameter *k* and the zero-run coder are tuned against is that **the previous
symbol in a stream is the coefficient directly above** — vertical adjacency, which the
vertical-context result (−11.7%) calls free.

`i % 256` delivers that property at exactly one tile width. At 256 px, stream *s* is column *s*,
walked top to bottom. At 512 px it interleaves two columns 256 px apart into a single stream, and
*k* then tracks an EMA over a mixture of two unrelated regions. **The constant was a hidden
assumption about tile width, not a stream count.**

The replacement is one expression, uniform in the width:

```
j = stream_id * symbols_per_stream + s        // column-major position in the tile
raster_index = (j % tile_size) * tile_size + j / tile_size
```

At 256 px a segment is exactly one column, so this reduces to the old mapping **coefficient for
coefficient**. That was the falsifiable prediction stated before measuring: tile 256 output must
be byte-identical, and if it moved by a single byte the implementation was wrong rather than the
idea. It is byte-identical at all 12 measured points, which is simultaneously the correctness
proof and the reason no shipped preset moves.

At 512 px, same q, PSNR delta exactly 0.00 dB — the change touches entropy coding, never
reconstruction:

| image | q=75 | q=90 | q=99 |
|---|---|---|---|
| bbb | −14.4% | −5.5% | −0.6% |
| blue_sky | −15.3% | −5.8% | −1.3% |
| touchdown | −12.9% | −5.6% | −0.5% |
| kristensara | −18.6% | −7.8% | −0.8% |

The gain shrinks with q because near lossless almost nothing is zero, so the mixture *k* was
tracking barely costs anything.

## Why the default tile size does *not* change

This is the part worth recording, because the temptation ran the other way.

The morning's measurement concluded "256 px is a local optimum and both directions are worse,
twelve of twelve points favour 256". With the mapping fixed, 512 px beats 256 px on **all twelve**
fixed-q points, by 0.8% to 5.0%. A straight reversal.

Those points flatter it. At q=75 the 512 arm also sits 0.03–0.23 dB *lower* in PSNR, so part of the
saving is quality rather than efficiency — the standing error that a fixed-q point measurement
always favours whichever arm spends more bits. BD-rate over q=60–95, PSNR-driven, is the honest
figure:

| image | BD-rate, 512 vs 256 |
|---|---|
| bbb | −0.23% |
| blue_sky | −0.84% |
| touchdown | −1.99% |
| kristensara | −0.61% |
| **mean** | **−0.91%** |

**So the sign was the coder, exactly as hypothesised, and the geometry underneath is worth about
1% — not the ~5% that had been inferred from the rANS arm.** A 512 px tile quadruples the
threadgroup working set and adds latency, for 0.91%. GOALS' "would we ship this?" says no.

Deeper decomposition does not rescue it either: with BUG-12 fixed so `--tile-size 512` can reach
the 6 levels it allows, 6 levels against 5 measures **−0.1%** at q=90 where PSNR is identical.
That lever is closed.

## What this invalidates

**Every tile-size result in this repository, #47 and the morning's sweep included.** All of them
scored the larger-tile arm through a coder that penalised it by 13–19%, so none of them bound what
the geometry is worth. This is the second time in one day that a recorded conclusion turned out to
be a property of the measuring rather than the measured; the pattern is now explicit in
COORDINATION.md.

It also independently settles the other half of ENT-1's split. ENT-1 attributed the 256→512 rANS
gain to roughly 9 points of per-tile table overhead and about 5% of "transform continuity". With
the coder fixed, continuity measures 0.91%. **The tables were the whole lever**, which is what
ENT-1 then went and fixed.

## Correctness

Encode and decode agree bit-exactly at tile 128, 256 and 512 — q=100 lossless roundtrip on two
images, max error 0, zero mismatched pixels. That is the canary chosen deliberately: an
encoder/decoder disagreement about a symbol ordering breaks lossless first and loudest, where a
lossy comparison would show a plausible-looking small regression instead.

## Note on the other coders

`huffman_encode.wgsl`, `huffman_decode.wgsl` and `huffman_histogram.wgsl` carry the identical bug
and were deliberately left alone (filed as BUG-14): Huffman is not a default coder, is capped at 4
levels, and nothing measures through it. The rANS fused histogram shader has the same shape over 32
streams, but rANS *gains* at 512 px, so its ordering is not costing it the same way — noted so
nobody mistakes it for the same defect.
