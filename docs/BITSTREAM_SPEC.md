# GNC Bitstream Specification

**Version:** GP17 (frame codec), GNV1 / GNV2 (containers)
**Date:** 2026-09-06

This document specifies the GNC bitstream format at a level of detail sufficient for an independent implementation.

All multi-byte values are **little-endian**. All byte offsets are from the start of the enclosing structure unless otherwise noted.

---

## 1. Frame Codec (GP11 … GP17)

A frame bitstream encodes a single image frame. The sequence container (GNV1, Section 3) wraps
multiple frames for video. The magic in the first four bytes names the generation; the encoder
writes **GP17** today, and Section 6 lists what each generation added. Field layouts below are
GP17 unless a row says otherwise.

The decoder tracks the generation as a single number parsed from the magic, and every field added
since GPC8 is gated on `gen >= N`; adding a generation is one entry in that table.

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
| 2 | u16 | block_size | Motion estimation block size (8 for variable block) |
| 4 | u32 | num_blocks | Number of motion blocks in the 2D grid |

**Delta-coded motion vectors (forward):**

Motion vectors are delta-coded using a median spatial predictor, preceded by an all-zero flag and
a skip bitmap:

1. **All-zero flag** (GP16): 1 byte, 1 if every vector in the field is (0,0). When set, nothing
   else follows — a static frame no longer pays 5 KB for an all-ones bitmap.
2. **Skip bitmap**: `ceil(num_blocks / 8)` bytes. Bit *i* = 1 means block *i* has MV = (0,0); no delta bytes follow.
3. **Delta MVs**: For each non-skip block (in raster order), two deltas (dx_delta, dy_delta).
   - **GP12–GP15**: zigzag-encoded unsigned varints, byte-aligned.
   - **GP16**: zigzag values written as **Exp-Golomb order 0** into one MSB-first bit stream,
     flushed to a byte boundary at the end of the field. Value *v* is `ceil(log2(v+2))-1` zero
     bits followed by *(v+1)* in binary, so a perfectly predicted vector costs one bit per
     component instead of two bytes.

The **median predictor** for block at grid position (bx, by):
- `left` = MV at (bx-1, by), or (0,0) if at left edge
- `above` = MV at (bx, by-1), or (0,0) if at top edge
- `above_right` = MV at (bx+1, by-1), or `above` if unavailable
- `predictor = (median(left.dx, above.dx, above_right.dx), median(left.dy, above.dy, above_right.dy))`
- `delta = actual_MV - predictor`

**Zigzag encoding**: `zigzag(x) = (x << 1) ^ (x >> 15)` maps signed i16 to unsigned u16 (0→0, -1→1, 1→2, ...).
**Varint encoding**: Standard unsigned varint (7 bits per byte, MSB continuation flag). Max 3 bytes for u16 range.

For B-frames (`frame_type == 2`), additionally:

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | u32 | bwd_count | Number of backward vectors (0 if none) |
| var | delta-coded | backward_vectors | Same delta+varint format as forward vectors |
| 4 | u32 | modes_count | Number of block mode bytes (0 if none) |
| modes_count | u8[] | block_modes | Per-block: 0=forward, 1=backward, 2=bidirectional |

### 1.7 Entropy Section

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | u32 | entropy_type | 0 = InterleavedRans, 1 = Bitplane, 2 = SubbandRans, 3 = Rice+ZRL, 4 = Huffman |
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
| 4 | u32 | num_groups | Low bits: `num_levels * 2` (LL + deepest merged + directional pairs). **Bit 31** = frequency tables are packed, see below |

For each group (num_groups times):

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | i32 | min_val | Group minimum value |
| 4 | u32 | alphabet_size | Group alphabet size |
| 4 | i32 | zrun_base | ZRL base (0 = no ZRL; group 0 never uses ZRL) |
| variable | u16[] or packed | freqs | Normalized frequencies (sum = 4096). Flat `u16` per symbol unless bit 31 of `num_groups` is set |

After all groups:

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 32 * 4 | u32[32] | stream_lengths | Byte length per stream |
| 32 * 4 | u32[32] | stream_states | Initial rANS state per stream |
| variable | u8[] | stream_data | Concatenated stream bytes |

**Packed frequency tables** (`num_groups & 0x8000_0000`). A normalised table is mostly empty —
68% zeros at q=70, mean nonzero entry in the single digits — so a flat `u16` per symbol made the
tables 23–26% of the file. Packed, each group's table is a bit stream of alternating
**zero-run length** and **value**, both **Exp-Golomb order 0**, MSB-first, flushed to a byte
boundary at the end of the group:

```
repeat until alphabet_size symbols have been placed:
    ue(run)        -- number of zero-frequency symbols to skip
    ue(freq - 1)   -- frequency of the symbol that follows the run
```

A trailing run with no value after it terminates the table. Runs and values alternate by
construction, so no prefix bit is spent. Exp-Golomb rather than Rice because the zero-coefficient
symbol routinely holds a frequency in the thousands; ue codes 4096 in 25 bits where unary would
need 4096. `num_groups` is read before the tables, so the tile versions itself and streams written
before this parse unchanged.

**Group assignment (directional splitting):** Group 0 = LL subband. Group 1 = deepest detail level (LH+HL+HH merged). For remaining levels (deep to shallow): even groups = LH+HL subbands, odd groups = HH subband. Total groups = num_levels * 2. This separates diagonal (HH) from horizontal/vertical (LH+HL) detail for tighter frequency distributions.

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

### 2.4 Rice+ZRL Tile (entropy_type = 3)

Significance map + Golomb-Rice magnitudes + zero-run-length, across **256 independent streams**
per tile. Coefficient *i* of the tile belongs to stream `i % 256` and is symbol `i / 256` of that
stream, so all 256 streams decode in parallel with no shared state.

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 4 | u32 | num_coefficients | tile_size² |
| 4 | u32 | tile_size | Tile dimension |
| 4 | u32 | num_levels | Wavelet decomposition levels |
| 4 | u32 | num_groups | `num_levels * 2` — subband grouping is the same as Section 2.2 |
| 1 | u8 | flags | Bit 0 (0x01) = varint stream lengths, bit 1 (0x02) = all-skip, bit 2 (0x04) = checkerboard-k block present, bit 3 (0x08) = Rice-coded stream lengths (GP17) |

**All-skip tiles** (`flags & 0x02`): the skip bitmap is the only remaining field; every
coefficient is zero and all 256 stream lengths are 0. A whole tile costs 18–19 bytes.

Otherwise, in order:

| Size | Type | Field | Description |
|------|------|-------|-------------|
| num_groups | u8[] | k_values | Rice parameter *k* for magnitudes, per subband group (0–15) |
| num_groups | u8[] | k_zrl_nz | Rice *k* for zero runs following a nonzero coefficient |
| num_groups | u8[] | k_zrl_z | Rice *k* for zero runs following a zero run or start-of-stream |
| 1 or 2 | u8 / u16 | skip_bitmap | Bit *g* = 1 means group *g* is entirely zero |
| 128 * ck_stride | u8[] | k_stream_odd | Present only when `flags & 0x04`; see below |
| variable | see below | stream_lengths | Byte length of each of the 256 streams |
| sum(lengths) | u8[] | stream_data | The 256 streams, concatenated in order |

**Stream-length table.** Three encodings, selected by the flags byte, most specific first:

| flags | encoding |
|---|---|
| bit 3 (0x08) | 4-bit `k`, then 256 Golomb-Rice codes, MSB-first, padded to a byte boundary. GP17 and later. |
| bit 0 (0x01) | 256 byte-aligned varints, 1–3 bytes each. |
| neither | Legacy: 256 × u16 little-endian. |

A Rice code with parameter `k` is the quotient `v >> k` as that many 1 bits and a terminating 0,
then the low `k` bits of `v`. Unary-then-terminator, rather than zeros-then-one, means a read past
the end of the buffer terminates at quotient 0 instead of spinning.

The encoder searches `k = 0..15` for the fewest bits and falls back to varints if they are smaller
— in practice Rice always wins, by 20–61% of the table. It matters because 256 independent streams
per tile means the table is a fixed ~256 bytes per tile whatever the tile holds: 5% of an I-frame
and 17–31% of a P-frame. Note that an optimal `k` deliberately leaves single long streams with a
long unary run rather than raising `k` for all 256 entries, so a decoder must not cap the quotient
anywhere near the typical value — `64 << k` silently truncates real high-quality frames.

**Skip bitmap width.** One byte while `num_groups <= 8`, two little-endian bytes above that. Five
wavelet levels produce 10 groups, which no longer fit in a byte. `num_groups` sits in the tile
header and is read first, so a decoder knows the width without a format flag and tiles written
before 5 levels existed keep parsing byte-for-byte (BUG-6, 2026-09-06).

**Coefficient syntax**, MSB-first within each stream:

```
significance bit
  0 -> zero run: Rice(k_zrl_nz or k_zrl_z) + 1 zero coefficients.
       k_zrl_nz applies when the previous coefficient had |v| >= 2, k_zrl_z otherwise.
       Coefficients in skipped groups are consumed by the run without costing bits.
  1 -> sign bit (1 = negative), then Rice(k) carrying |v| - 1.
```

*k* is not constant within a stream: it tracks an EMA of recent magnitudes, held in fixed point
×16 with a window of ≈8 coefficients. Per group *g* the stream starts at
`ema = max(1, 1 << k_values[g]) << 4`, uses `k = floor(log2(ema >> 4))` clamped to 15, and
updates `ema += (rice_val << 1) - (ema >> 3)` after every nonzero. Encoder and decoder derive *k*
from the same decoded history, so only the per-group seed is transmitted.

**Checkerboard context** (`flags & 0x04`). Even streams are coded first; each odd stream warm-starts
from its left neighbour's final EMA rather than from `k_values`. `k_stream_odd[odd_idx * ck_stride + g]`
holds the blended seed for stream `2*odd_idx + 1`, group *g*, where
`ck_stride = 8` for tiles with ≤8 groups and 12 above that. Tiles encoded on the GPU derive the
same values in-shader from the decoded even streams and leave the block out entirely.

### 2.5 Huffman Tile (entropy_type = 4)

Canonical Huffman, per-subband tables. Not yet specified here — read `src/encoder/huffman.rs`.

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

## 4. Temporal Wavelet Container (GNV2)

Wraps temporal wavelet-encoded video as a sequence of GP12 frames organized into GOPs.

### 4.1 File Header (34 bytes)

| Offset | Size | Type | Field | Description |
|--------|------|------|-------|-------------|
| 0 | 4 | char[4] | magic | `"GNV2"` |
| 4 | 4 | u32 | version | 1 |
| 8 | 4 | u32 | width | Frame width |
| 12 | 4 | u32 | height | Frame height |
| 16 | 4 | u32 | frame_count | Total original frames |
| 20 | 4 | u32 | framerate_num | Framerate numerator |
| 24 | 4 | u32 | framerate_den | Framerate denominator |
| 28 | 1 | u8 | temporal_transform | 0 = none, 1 = Haar, 2 = LeGall 5/3 |
| 29 | 1 | u8 | gop_size | GOP size (2, 4, or 8) |
| 30 | 4 | f32 | highpass_qstep_mul | Highpass quantization multiplier |

### 4.2 Frame Index Table

Starts at offset 34. One entry per serialized frame (22 bytes each):

| Size | Type | Field | Description |
|------|------|-------|-------------|
| 8 | u64 | offset | Byte offset from file start to frame data |
| 4 | u32 | size | Frame data size in bytes |
| 1 | u8 | frame_role | 0 = lowpass (seekable), 1 = highpass, 2 = tail I-frame |
| 1 | u8 | temporal_level | Wavelet level (0 = finest). 0 for lowpass/tail. |
| 2 | u16 | gop_index | GOP index (0-based) |
| 2 | u16 | frame_index_in_gop | Position within GOP |
| 4 | u32 | pts | Presentation timestamp (display-order frame number) |

### 4.3 Frame Ordering

Within each GOP, frames are ordered: **lowpass first**, then highpass from deepest temporal level to finest (L2→L1→L0). Within each level, frames in order. After all GOPs: tail I-frames.

Example for GOP size 8 (3 temporal levels):
- Frame 0: lowpass (1 frame, frame_role=0)
- Frame 1: level 2 highpass (1 frame, frame_role=1, temporal_level=2)
- Frames 2–3: level 1 highpass (2 frames, temporal_level=1)
- Frames 4–7: level 0 highpass (4 frames, temporal_level=0)

### 4.4 Random Access

To seek to time T: compute `gop_index = T / gop_size`, find the lowpass frame (frame_role=0) of that GOP. Decode the entire GOP to reconstruct individual frames via inverse temporal wavelet.

---

## 5. Codec Pipeline

### 5.1 Color Space

**Forward (encode):** RGB -> YCoCg-R (reversible integer lifting for lossless mode)
- Y  = (R + 2G + B) >> 2
- Co = R - B
- t  = B + (Co >> 1)  [or floor for lossless]
- Cg = G - t

**Inverse (decode):** YCoCg-R -> RGB

### 5.2 Wavelet Transform

**LeGall 5/3** (lossless mode, q=100):
- Integer-exact lifting steps with `floor()` division
- Bit-exact round-trip guaranteed

**CDF 9/7** (lossy modes, q=1-99):
- Floating-point lifting steps
- 4 vanishing moments, better energy compaction

**Decomposition depth.** The transform runs per tile, so each level halves the tile; a 256 px tile
allows at most 5 levels before the LL band gets too small for per-subband statistics to mean
anything (`CodecConfig::max_wavelet_levels`). The quality preset uses **5 levels at q ≥ 25 and 4
below** — below that the two deepest subbands quantise to all-zero and their k values and rANS
frequency tables cost more than they save. `wavelet_levels` is in the frame header, so a decoder
never has to infer it.

### 5.3 Quantization

Uniform scalar quantization with dead zone:
```
quantized = sign(x) * max(0, floor((|x| - dead_zone * qstep) / qstep + 0.5))
```

Adaptive quantization modulates `qstep` per-tile using the weight map:
```
effective_qstep = qstep * (1.0 + aq_strength * (weight - 1.0))
```

### 5.4 Tile Independence

Tiles are strictly independent: no cross-tile dependencies at any stage. Each tile can be encoded, decoded, and error-recovered independently. This is a fundamental design constraint enabling GPU parallelism.

---

## 6. Backward Compatibility

| Magic | Readable | Notes |
|-------|----------|-------|
| GP17 | Yes | Current version: Golomb-Rice stream-length tables (tile flag 0x08) |
| GP16 | Yes | Exp-Golomb bit-coded MV deltas + all-zero flag byte |
| GP15 | Yes | Rice `k_zrl` split into `k_zrl_nz` + `k_zrl_z` per subband |
| GP14 | Yes | Per-block `fwd_ref_idx` / `bwd_ref_idx` for hierarchical pyramid B-frames |
| GP13 | Yes | GP12 + chroma_format byte in the header |
| GP12 | Yes | Delta-coded varint MVs with skip bitmap |
| GP11 | Yes | CRC-32 and tile index, raw i16 MVs |
| GP10 | Yes | Temporal coding, no CRC, no tile index |
| GPC9 | Yes | Per-subband entropy, no temporal |
| GPC8 | Yes | Baseline, no per-subband flag |
| GPC7 and older | No | Must re-encode |

When reading older formats, missing features are defaulted:
- No per-subband flag -> `per_subband_entropy = false`
- No frame_type -> `frame_type = Intra`
- No tile index -> no CRC validation available
- No B-frame motion -> `backward_vectors = None, block_modes = None`
- No chroma_format byte -> 4:4:4
- No ref indices -> `fwd = 0, bwd = 1`

Tile payloads carry no generation of their own; they are versioned by their own header fields.
The Rice skip-bitmap width (Section 2.4) is the current example — `num_groups` decides it, so
old and new tiles coexist in the same frame format.

---

## 7. Error Resilience

GP11 provides per-tile CRC-32 checksums. A decoder should:

1. Read the tile index table (sizes + CRCs) before tile data
2. Compute CRC-32 over each tile's serialized bytes
3. Compare against the stored CRC
4. For tiles that fail CRC:
   - **I-frames:** Substitute a zero-coefficient tile (decodes to mid-gray)
   - **P/B-frames:** Optionally substitute the corresponding tile from the reference frame
5. Continue decoding remaining valid tiles normally

The tile-independent architecture ensures a single corrupt tile does not affect other tiles.
