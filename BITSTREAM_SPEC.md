# GNC Bitstream Specification

**Version:** GP11 (frame codec), GNV1 (sequence container)
**Date:** 2026-02-26

This document specifies the GNC bitstream format at a level of detail sufficient for an independent implementation.

All multi-byte values are **little-endian**. All byte offsets are from the start of the enclosing structure unless otherwise noted.

---

## 1. Frame Codec (GP11)

A GP11 bitstream encodes a single image frame. The sequence container (GNV1, Section 2) wraps multiple GP11 frames for video.

### 1.1 Frame Header

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | char[4] | magic | `"GP11"` |
| 4 | 4 | u32 | width | Image width in pixels |
| 8 | 4 | u32 | height | Image height in pixels |
| 12 | 4 | u32 | bit_depth | Bits per channel (typically 8) |
| 16 | 4 | u32 | tile_size | Tile dimension in pixels (typically 256) |
| 20 | 4 | f32 | qstep | Quantization step size |
| 24 | 4 | f32 | dead_zone | Dead zone width for quantization |
| 28 | 4 | u32 | wavelet_levels | Number of wavelet decomposition levels |
| 32 | 1 | u8 | wavelet_type | 0 = LeGall 5/3, 1 = CDF 9/7 |
| 33 | 1 | u8 | per_subband | 0 = off, 1 = per-subband entropy coding |

**Derived values:**
- `tiles_x = ceil(width / tile_size)`
- `tiles_y = ceil(height / tile_size)`
- `padded_width = tiles_x * tile_size`
- `padded_height = tiles_y * tile_size`
- `total_tiles = tiles_x * tiles_y * 3` (Y, Co, Cg planes)

### 1.2 Subband Weights (variable length)

Immediately follows the fixed header at offset 34.

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | f32 | ll_weight | LL subband quantization weight |
| 4 | u32 | num_detail | Number of wavelet detail levels |
| num_detail * 12 | f32[3] | detail_weights | Per-level [LH, HL, HH] weights |
| 4 | f32 | chroma_weight | Chroma plane quantization weight multiplier |

### 1.3 CfL (Chroma-from-Luma) Side Info

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 1 | u8 | cfl_enabled | 0 = disabled, 1 = enabled |

If `cfl_enabled == 1`:

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | u32 | num_subbands | Number of subbands (1 + 3 * wavelet_levels) |
| 4 | u32 | num_cfl_tiles | tiles_x * tiles_y |
| 2 * num_cfl_tiles * num_subbands * 2 | i16[] | alphas | CfL prediction coefficients |

Alpha layout: `alphas[chroma_plane][tile][subband]` where chroma_plane is 0 (Co) or 1 (Cg). Values are quantized to [-16384, 16384] representing the range [-2.0, 2.0].

### 1.4 Adaptive Quantization

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | u32 | aq_enabled | 0 = disabled, 1 = enabled |
| 4 | f32 | aq_strength | AQ strength (0.0 - 1.0) |
| 4 | u32 | weight_map_len | Number of weight map entries (0 if disabled) |
| weight_map_len * 4 | f32[] | weight_map | Per-LL-block quantization weights |

### 1.5 Frame Type

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 1 | u8 | frame_type | 0 = Intra, 1 = Predicted, 2 = Bidirectional |

### 1.6 Motion Field (P-frames and B-frames only)

Present only when `frame_type != 0`:

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 2 | u16 | block_size | Motion estimation block size (typically 16) |
| 4 | u32 | num_blocks | Number of motion blocks |
| num_blocks * 4 | i16[2] | forward_vectors | Per-block [dx, dy] in half-pel units |

For B-frames (`frame_type == 2`), additionally:

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | u32 | bwd_count | Number of backward vectors (0 if none) |
| bwd_count * 4 | i16[2] | backward_vectors | Per-block [dx, dy] in half-pel units |
| 4 | u32 | modes_count | Number of block mode bytes (0 if none) |
| modes_count | u8[] | block_modes | Per-block: 0=forward, 1=backward, 2=bidirectional |

### 1.7 Entropy Section

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | u32 | entropy_type | 0 = InterleavedRans, 1 = Bitplane, 2 = SubbandRans |
| 4 | u32 | num_tiles | Total tile count (tiles_x * tiles_y * 3) |

### 1.8 Tile Index Table (GP11)

One entry per tile, immediately after `num_tiles`:

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | u32 | tile_size_bytes | Serialized size of this tile in bytes |
| 4 | u32 | tile_crc32 | CRC-32 (ISO 3309) of the tile's serialized bytes |

The tile index table has `num_tiles` entries (8 bytes each).

**CRC-32 specification:** Polynomial 0xEDB88320 (reflected), initial value 0xFFFFFFFF, final XOR 0xFFFFFFFF. Same as zlib, gzip, and PNG.

### 1.9 Tile Data

Tile data follows the tile index table. Tiles are concatenated in order:
- Plane 0 (Y): tile[0..tiles_per_plane)
- Plane 1 (Co): tile[tiles_per_plane..2*tiles_per_plane)
- Plane 2 (Cg): tile[2*tiles_per_plane..3*tiles_per_plane)

Within each plane, tiles are in raster order (row-major).

---

## 2. Tile Formats

### 2.1 InterleavedRans Tile (entropy_type = 0)

Uses 32 interleaved rANS streams sharing one frequency table.

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | i32 | min_val | Minimum coefficient value (symbol offset) |
| 4 | u32 | alphabet_size | Number of distinct symbols |
| 4 | u32 | num_coefficients | Total coefficients in this tile |
| 4 | i32 | zrun_base | ZRL base symbol (0 = no ZRL) |
| 32 * 4 | u32[32] | stream_lengths | Byte length of each of 32 streams |
| 32 * 4 | u32[32] | stream_states | Initial rANS state per stream |
| alphabet_size * 2 | u16[] | freqs | Normalized frequency table (sum = 4096) |
| variable | u8[] | stream_data | Concatenated stream bytes |

**Symbol mapping:** Symbol `s` maps to coefficient value `min_val + s`. When `zrun_base > 0`, symbols >= `zrun_base` encode zero-runs of length `s - zrun_base + 2`.

**rANS parameters:** RANS_M = 4096 (12-bit probability resolution). Renormalization range: state must stay in [RANS_M, RANS_M * 256).

### 2.2 SubbandRans Tile (entropy_type = 2)

Uses per-subband frequency tables with 32 interleaved rANS streams.

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | u32 | num_coefficients | Total coefficients |
| 4 | u32 | tile_size | Tile dimension |
| 4 | u32 | num_levels | Wavelet decomposition levels |
| 4 | u32 | num_groups | 1 + num_levels (LL + one per detail level) |

For each group (num_groups times):

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | i32 | min_val | Group minimum value |
| 4 | u32 | alphabet_size | Group alphabet size |
| 4 | i32 | zrun_base | ZRL base (0 = no ZRL; group 0 never uses ZRL) |
| alphabet_size * 2 | u16[] | freqs | Normalized frequencies (sum = 4096) |

After all groups:

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 32 * 4 | u32[32] | stream_lengths | Byte length per stream |
| 32 * 4 | u32[32] | stream_states | Initial rANS state per stream |
| variable | u8[] | stream_data | Concatenated stream bytes |

**Group assignment:** Group 0 = LL subband. Group `g` (g > 0) = all three subbands (LH, HL, HH) at wavelet level `g`. Coefficients within each stream are processed in raster scan order within each subband, advancing through groups sequentially.

### 2.3 Bitplane Tile (entropy_type = 1)

Block-based bitplane coding for GPU-parallel decode.

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | u32 | num_coefficients | tile_size^2 |
| 4 | u32 | tile_size | Tile dimension |
| 4 | u32 | num_blocks | Number of 32x32 blocks |
| num_blocks * 4 | u32[] | block_offsets | Byte offset of each block in block_data |
| 4 | u32 | block_data_len | Total bytes of block data |
| block_data_len | u8[] | block_data | Per-block bitplane data |

**Per 32x32 block format:**
```
max_bitplane: u8  (0 if all-zero block)
For p = (max_bitplane-1) down to 0:
  all_zero_flag: 1 bit
  If not all-zero:
    significance_map: 1024 bits (one per coefficient)
sign_bits: N bits (one per nonzero coefficient)
```

---

## 3. Sequence Container (GNV1)

Wraps multiple GP11 frames for video sequences.

### 3.1 File Header (28 bytes)

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | char[4] | magic | `"GNV1"` |
| 4 | 4 | u32 | version | 1 |
| 8 | 4 | u32 | width | Frame width |
| 12 | 4 | u32 | height | Frame height |
| 16 | 4 | u32 | frame_count | Number of frames |
| 20 | 4 | u32 | framerate_num | Framerate numerator (e.g. 30000) |
| 24 | 4 | u32 | framerate_den | Framerate denominator (e.g. 1001) |

### 3.2 Frame Index Table

Starts at offset 28. One entry per frame (21 bytes each):

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 8 | u64 | offset | Byte offset from file start to frame data |
| 4 | u32 | size | Frame data size in bytes |
| 1 | u8 | frame_type | 0 = Intra, 1 = Predicted, 2 = Bidirectional |
| 8 | u64 | pts | Presentation timestamp (frame number) |

### 3.3 Frame Data

Starts at offset `28 + frame_count * 21`. Concatenated GP11 frame bitstreams. Each frame is independently decodable given its reference frames.

**Random access:** To seek to time T, scan the frame index for the nearest preceding I-frame (frame_type == 0). Decode from that I-frame forward through any P/B-frames to reach the target.

---

## 4. Codec Pipeline

### 4.1 Color Space

**Forward (encode):** RGB -> YCoCg-R (reversible integer lifting for lossless mode)
- Y  = (R + 2G + B) >> 2
- Co = R - B
- t  = B + (Co >> 1)  [or floor for lossless]
- Cg = G - t

**Inverse (decode):** YCoCg-R -> RGB

### 4.2 Wavelet Transform

**LeGall 5/3** (lossless mode, q=100):
- Integer-exact lifting steps with `floor()` division
- Bit-exact round-trip guaranteed

**CDF 9/7** (lossy modes, q=1-99):
- Floating-point lifting steps
- 4 vanishing moments, better energy compaction

### 4.3 Quantization

Uniform scalar quantization with dead zone:
```
quantized = sign(x) * max(0, floor((|x| - dead_zone * qstep) / qstep + 0.5))
```

Adaptive quantization modulates `qstep` per-tile using the weight map:
```
effective_qstep = qstep * (1.0 + aq_strength * (weight - 1.0))
```

### 4.4 Tile Independence

Tiles are strictly independent: no cross-tile dependencies at any stage. Each tile can be encoded, decoded, and error-recovered independently. This is a fundamental design constraint enabling GPU parallelism.

---

## 5. Backward Compatibility

| Magic | Readable | Notes |
|-------|----------|-------|
| GP11 | Yes | Current version with CRC-32 and tile index |
| GP10 | Yes | Temporal coding, no CRC, no tile index |
| GPC9 | Yes | Per-subband entropy, no temporal |
| GPC8 | Yes | Baseline, no per-subband flag |
| GPC7 and older | No | Must re-encode |

When reading older formats, missing features are defaulted:
- No per-subband flag -> `per_subband_entropy = false`
- No frame_type -> `frame_type = Intra`
- No tile index -> no CRC validation available
- No B-frame motion -> `backward_vectors = None, block_modes = None`

---

## 6. Error Resilience

GP11 provides per-tile CRC-32 checksums. A decoder should:

1. Read the tile index table (sizes + CRCs) before tile data
2. Compute CRC-32 over each tile's serialized bytes
3. Compare against the stored CRC
4. For tiles that fail CRC:
   - **I-frames:** Substitute a zero-coefficient tile (decodes to mid-gray)
   - **P/B-frames:** Optionally substitute the corresponding tile from the reference frame
5. Continue decoding remaining valid tiles normally

The tile-independent architecture ensures a single corrupt tile does not affect other tiles.
