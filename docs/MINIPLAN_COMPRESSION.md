# Miniplan: Compression Improvements

Three targeted changes to improve compression efficiency without sacrificing GPU parallelism.

## Current encode pipeline

```
RGB → color_convert (YCoCg-R) → deinterleave (Y/Co/Cg planes)
    → [CfL q50-85] → transform_97 → quantize → entropy (Rice)
```

## 1. Huffman per subband (replaces Rice) — DONE, not a win

**Result:** GPU Huffman encoder/decoder activated and benchmarked. GPU decode runs at
~59 fps (matching Rice's 61 fps), but compression is **8–11% worse** than Rice at all
quality levels. The overhead comes from Huffman table storage (~512 bytes/tile for
8 groups × 64 code lengths). Rice's single k-parameter per subband is more compact
and well-matched to geometric wavelet coefficient distributions.

| q   | Rice BPP | Huffman BPP | rANS BPP |
|-----|----------|-------------|----------|
| 25  | 1.71     | 1.84 (+8%)  | 1.29     |
| 50  | 2.37     | 2.63 (+11%) | 2.30     |
| 75  | 4.01     | 4.34 (+8%)  | 4.22     |
| 90  | 8.90     | 9.66 (+9%)  | 9.65     |

**Conclusion:** Keep Rice as default. Huffman backend works as an option but
doesn't improve compression. rANS remains best at low quality (–25% at q=25).

## 2. Intra prediction before transform_97

**Why:** Predicting pixels before wavelet reduces residual energy → fewer bits after quantization.

- New shader: per-block (8×8 or 16×16) prediction modes
  - DC (mean of neighbors), Horizontal, Vertical, Diagonal
- Encode residual (original − prediction) through wavelet instead of raw pixels
- Mode selection: fast SAD, encode mode index in bitstream (2 bits/block)
- Decoder reads mode, reconstructs prediction, adds decoded residual

## 3. CfL in spatial domain (move before transform_97)

**Why:** Chroma-from-luma correlation is stronger in spatial domain than after wavelet.

- Current CfL operates post-quantize — move to between deinterleave and transform_97
- Predict Co/Cg from downscaled Y (block DC), code residual through wavelet
- Requires pipeline reorder in both encoder and decoder

## Implementation order

1. **Huffman per subband** — most bpp gain, least invasive, code exists
2. **Intra prediction** — new shader + bitstream change
3. **CfL spatial** — most invasive pipeline reorder

## Success metrics

- Compare bpp at matched PSNR (BD-rate) against current Rice baseline
- Encode/decode fps must stay ≥30 fps (1080p)
- All existing tests must pass
