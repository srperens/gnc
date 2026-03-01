# Canonical Huffman GPU Entropy Coder — Implementation Plan

> **Status: IMPLEMENTED (2026-03)**
>
> This plan has been fully implemented. The final design diverged from the
> original plan in several significant ways — see [Implementation Notes](#implementation-notes-vs-original-plan)
> at the bottom of this document.
>
> Relevant source files:
> - `src/encoder/huffman.rs` — CPU codebook builder, encode/decode, serialization
> - `src/encoder/huffman_gpu.rs` — GPU host code (histogram, encode, decode)
> - `src/shaders/huffman_histogram.wgsl` — GPU histogram shader
> - `src/shaders/huffman_encode.wgsl` — GPU encode shader
> - `src/shaders/huffman_decode.wgsl` — GPU decode shader

## Context

We have three entropy coders: rANS (good compression, slow — 32 sequential streams), Rice+ZRL (fast — 256 parallel streams, matches or beats rANS at q>=50 but +34% overhead at q=25), and Bitplane. Canonical Huffman could close the remaining low-bitrate gap: Rice-like parallelism (256 independent streams, no state chain) with near-rANS compression (variable-length codes adapt to actual distributions instead of assuming geometric like Rice).

**Note (2026-02-27):** Rice+ZRL has largely closed the compression gap — it beats rANS at q>=75 and is within 5% at q=50. The main remaining gap is at q=25 (+34%). Huffman's value proposition is now narrower: closing that low-bitrate gap and potentially improving lossless compression.

## Design: Significance Map + Canonical Huffman (per-subband)

Same stream architecture as Rice. Same significance map encoding:
- bit=0 → zero coefficient
- bit=1 → sign bit + Huffman(|val|-1) using per-subband codebook

Key difference from Rice: instead of a single `k` parameter per subband, we have a full Huffman codebook (codeword + code length per symbol). This requires a **2-dispatch encode** (histogram on GPU, codebook on CPU, then encode on GPU).

### Encode Pipeline (2 GPU dispatches + CPU codebook)

```
GPU dispatch 1: Histogram
  256 threads/tile, cooperative magnitude frequency counting
  → magnitude_freq[group][symbol] via shared atomics
  → readback to CPU

CPU: Build codebooks
  For each subband group: freq[] → Huffman tree → canonical codes
  → codeword[symbol], codelen[symbol]
  Upload codebook buffer to GPU

GPU dispatch 2: Encode
  Load codebook into shared memory (8KB)
  Each thread encodes its stream independently
  Same bit-packing as Rice (emit_bit, emit_bits, emit_byte)
```

### Decode Pipeline (1 GPU dispatch)

```
GPU dispatch: Decode
  Load 8-bit prefix decode table into shared memory (8KB)
  Each thread decodes its stream independently
  Peek 8 bits → table lookup → symbol + code length
  Fallback to bit-by-bit for codes > 8 bits (rare)
```

### Alphabet & Escape Code

rANS uses MAX_ALPHABET=4096. With significance maps we only encode magnitudes (|val|-1), so the alphabet is smaller. Cap Huffman alphabet at 255 symbols (magnitudes 0..254). Symbol 255 = escape code → followed by raw 12-bit magnitude. This keeps shared memory small while handling rare large values.

### Shared Memory Budget (8KB total — excellent occupancy)

**Encoder shared memory:**
- `shared_code[8][256]` — packed u32 = `(codelen << 16) | codeword`
- 8 groups × 256 symbols × 4B = 8KB (sym 255 = escape code entry)

**Decoder shared memory:**
- `shared_decode[8][256]` — packed u32 = `(symbol << 16) | codelen`
- 8 groups × 256 entries × 4B = 8KB
- 8-bit primary table: for codes ≤ 8 bits, entry repeated 2^(8-len) times
- Codes > 8 bits: flag entry, fall back to bit-by-bit scan
- If decoded symbol = 255 (escape): read next 12 raw bits as magnitude

## Files

### New Files (6)

1. **`src/encoder/huffman.rs`** — CPU Huffman codebook builder + tile serialization
   - `HuffmanTile` struct (codebooks, stream_lengths, stream_data)
   - `build_canonical_codebook(freq: &[u32]) -> (Vec<u32>, Vec<u8>)` — codewords + lengths
   - `build_decode_table(codelen: &[u8], max_bits: u8) -> Vec<u32>` — prefix lookup table
   - `serialize_tile_huffman()` / `deserialize_tile_huffman()`
   - Pattern: follow `rice.rs` structure exactly

2. **`src/encoder/huffman_gpu.rs`** — GPU host code (buffer setup, dispatch, readback)
   - `GpuHuffmanEncoder` struct with histogram + encode pipelines
   - `CachedHuffmanBuffers` — reusable GPU buffers
   - `encode_3planes_to_tiles()` — batched 3-plane encoding
   - Two-pass dispatch: histogram → CPU codebook → encode
   - Pattern: follow `rice_gpu.rs` structure

3. **`src/shaders/huffman_histogram.wgsl`** — GPU histogram shader
   - 256 threads/tile, cooperative frequency counting
   - Per-subband magnitude histograms via shared atomics
   - Output: `hist_output[tile * MAX_GROUPS * MAX_ALPHABET + group * MAX_ALPHABET + symbol]`

4. **`src/shaders/huffman_encode.wgsl`** — GPU encode shader
   - Load codebook from buffer into shared memory
   - Per-thread encoding: significance map + Huffman codes
   - Reuse Rice bit-packing pattern (emit_bit, emit_bits, emit_byte, flush)
   - Output: stream_output[], stream_lengths[]

5. **`src/shaders/huffman_decode.wgsl`** — GPU decode shader
   - Load decode table from buffer into shared memory
   - Per-thread decoding: read_bit for significance, prefix-table decode for symbols
   - 8-bit primary table with bit-by-bit fallback
   - Output: reconstructed f32 coefficients

6. **`tests/huffman_tests.rs`** or additions to existing test files
   - Round-trip correctness (encode → serialize → deserialize → decode)
   - Codebook construction unit tests
   - GPU vs CPU agreement

### Modified Files (6)

7. **`src/lib.rs`** — Add `EntropyCoder::Huffman` variant + config defaults
8. **`src/encoder/mod.rs`** — Add `pub mod huffman; pub mod huffman_gpu;`
9. **`src/encoder/entropy_helpers.rs`** — Add `EntropyMode::Huffman` + dispatch logic
10. **`src/encoder/pipeline.rs`** — Wire Huffman into GPU encode path
11. **`src/decoder/pipeline.rs`** — Wire Huffman into decode path
12. **`src/format.rs`** — Add `EntropyData::Huffman(Vec<HuffmanTile>)`, entropy_type = 4, serialization

## Implementation Steps

### Step 1: CPU Huffman codebook builder (`huffman.rs`)

Core algorithm for `build_canonical_codebook()`:
1. Build min-heap of (frequency, symbol) pairs
2. Merge two lowest-frequency nodes repeatedly → Huffman tree
3. Extract code lengths from tree (DFS)
4. Clamp max code length to 15 bits (redistribute if needed)
5. Canonical assignment: sort symbols by (length, symbol), assign codes sequentially
   - Start with code=0 at shortest length
   - Increment code, left-shift when moving to next length

`HuffmanTile` struct:
```rust
pub struct HuffmanTile {
    pub num_coefficients: u32,
    pub tile_size: u32,
    pub num_levels: u32,
    pub num_groups: u32,
    pub codebooks: Vec<SubbandCodebook>,  // one per group
    pub stream_lengths: Vec<u32>,          // 256 entries
    pub stream_data: Vec<u8>,              // concatenated streams
}

pub struct SubbandCodebook {
    pub min_symbol: u32,       // offset for symbol mapping
    pub alphabet_size: u32,
    pub code_lengths: Vec<u8>,  // per symbol
    pub codewords: Vec<u16>,    // per symbol (max 15 bits)
}
```

Serialization format:
```
u32: num_coefficients
u32: tile_size
u32: num_levels
u32: num_groups
For each group:
  u32: min_symbol
  u32: alphabet_size
  u8[alphabet_size]: code_lengths
u16[256]: stream_lengths
u8[sum(stream_lengths)]: stream_data
```

### Step 2: GPU histogram shader (`huffman_histogram.wgsl`)

Simple cooperative histogram:
```
shared_hist: array<atomic<u32>, 2048>  // [group*256+mag], 8 groups × 256 symbols

Phase 1: Each thread scans its stream's coefficients
  - Compute subband group
  - If non-zero: atomicAdd(&shared_hist[group * 256 + min(|val|-1, 255)], 1)

Phase 2: Threads cooperatively write histograms to output buffer
```

Shared memory: 2048 × 4 = 8KB. Same occupancy as codebook phase.

### Step 3: GPU encode shader (`huffman_encode.wgsl`)

```
const MAX_ALPHABET: u32 = 256;
const MAX_GROUPS: u32 = 8;

var<workgroup> shared_code: array<u32, 2048>;  // [group*256+sym] = (len<<16)|codeword

Phase 1: Load codebook cooperatively
  for (i = thread_id; i < num_groups * 256; i += 256)
    shared_code[i] = codebook_buf[tile_id * 2048 + i];
  workgroupBarrier();

Phase 2: Encode (identical loop structure to Rice)
  for each coefficient in stream:
    if zero: emit_bit(0)
    else:
      emit_bit(1)
      emit_bit(sign)
      let mag = |val| - 1
      let sym = min(mag, 255)  // clamp; 255 = escape
      let packed = shared_code[group * 256 + sym]
      let len = packed >> 16
      let code = packed & 0xFFFF
      emit_bits(code, len)
      if mag >= 255: emit_bits(mag, 12)  // raw escape payload
  flush_remaining()
```

### Step 4: GPU decode shader (`huffman_decode.wgsl`)

```
var<workgroup> shared_decode: array<u32, 2048>;  // [group*256+peek] = (sym<<16)|len

Decode loop:
  for each coefficient position:
    let sig = read_bit()
    if sig == 0: output = 0.0
    else:
      let sign = read_bit()
      let peek = peek_bits(8)  // read 8 bits without consuming
      let entry = shared_decode[group * 256 + peek]
      let sym = entry >> 16
      let len = entry & 0xFFFF
      consume_bits(len)  // advance by actual code length
      var mag = sym
      if sym == 255: mag = read_bits(12)  // escape: raw 12-bit
      output = ±(mag + 1) based on sign
```

`peek_bits(n)` reads n bits without consuming. `consume_bits(n)` advances position.
For codes > 8 bits: entry has flag (len=0), fall back to bit-by-bit scan.

### Step 5: GPU host code (`huffman_gpu.rs`)

Follow `rice_gpu.rs` pattern:
```rust
pub struct GpuHuffmanEncoder {
    histogram_pipeline: wgpu::ComputePipeline,
    histogram_bgl: wgpu::BindGroupLayout,
    encode_pipeline: wgpu::ComputePipeline,
    encode_bgl: wgpu::BindGroupLayout,
    cached: Option<CachedHuffmanBuffers>,
}

// Two-pass encode:
fn encode_plane(&self, ctx, quantized_buf, info) -> HuffmanTile {
    // Pass 1: GPU histogram
    dispatch histogram shader → readback histograms

    // CPU: build codebooks from histograms
    for each group: build_canonical_codebook(hist)

    // Upload codebooks + decode tables to GPU

    // Pass 2: GPU encode
    dispatch encode shader → readback streams + lengths

    // Pack into HuffmanTile
}
```

### Step 6: Integration (lib.rs, pipeline.rs, format.rs, entropy_helpers.rs)

- Add `EntropyCoder::Huffman` to enum in lib.rs
- Add `EntropyMode::Huffman` to entropy_helpers.rs
- Add `EntropyData::Huffman(Vec<HuffmanTile>)` to format.rs
- Wire encoder dispatch in pipeline.rs (same pattern as Rice)
- Wire decoder dispatch in decoder/pipeline.rs
- Entropy type = 4 in bitstream

### Step 7: Testing & benchmarking

- Unit tests: codebook construction, canonical code properties
- Round-trip tests: encode → decode at various quality levels
- Comparison benchmark: Huffman vs Rice vs rANS (PSNR, bpp, fps)
- Run on all test images (bbb, blue_sky, touchdown, kristensara)

## Key Risk: CPU Codebook Roundtrip

The histogram GPU→CPU→codebook→GPU roundtrip adds latency. Mitigation:
- Histogram readback is small (8 groups × 256 × 4B = 8KB per tile)
- Codebook construction is O(n log n) where n ≤ 256 — microseconds
- Codebook upload is small (8KB per tile)
- Net: probably < 1ms overhead for full frame
- If this becomes a bottleneck, the histogram could be built from the fused quantize+histogram shader (already exists for rANS)

## Verification

1. `cargo test --release` — all existing tests pass
2. New Huffman-specific tests pass
3. `cargo run --release -- benchmark -i test_material/frames/bbb_1080p.png -q 75 --entropy huffman`
4. Compare bpp/PSNR/fps against Rice and rANS
5. `cargo clippy` — zero warnings

---

## Implementation Notes vs Original Plan

The following changes were made during implementation. The plan above reflects
the **original** design; the actual code differs as noted below.

### Alphabet: 64 symbols (not 255)

The plan specified 255 symbols with escape at symbol 255. The implementation
uses **64 symbols** (`HUFFMAN_ALPHABET_SIZE = 64`) with escape at symbol 63.
64 covers >99% of wavelet magnitudes directly and cuts shared memory from 8KB
to 2KB for the encoder codebook (512 entries × 4B).

### Max code length: 8 bits (not 15)

Clamped to 8 bits (`HUFFMAN_MAX_CODE_LEN = 8`) so the 8-bit prefix decode
table resolves **every** code in a single lookup. The slow-path bit-by-bit
fallback exists in the CPU decoder but is never hit. The GPU decoder has no
slow path at all.

### Escape coding: exp-Golomb (not raw 12-bit)

Instead of a fixed 12-bit raw magnitude after the escape symbol, the
implementation uses **exp-Golomb** coding for the excess magnitude
(`magnitude - ESCAPE_SYM`). This adapts to the actual distribution of large
values rather than wasting bits on a fixed-width field.

### Zero-run-length coding added

The plan described a simple significance bit (0=zero, 1=non-zero). The
implementation uses **Rice-coded zero-run-length** (identical to the Rice
entropy coder): bit=0 signals a zero run, followed by Rice(run_length-1, k_zrl)
with per-subband k_zrl parameters. The GPU histogram shader also computes
ZRL statistics (sum + count per group) for optimal k_zrl selection.

### HuffmanTile struct simplified

The plan's `SubbandCodebook` struct (with `min_symbol`, `alphabet_size`,
`codewords`) was not implemented. Instead, `HuffmanTile` stores:
- `code_lengths: Vec<Vec<u8>>` — per-group code lengths only
- `k_zrl_values: Vec<u8>` — per-group Rice k for zero runs

Canonical codewords are reconstructed from code lengths at decode time (which
is the whole point of canonical Huffman — only lengths need to be transmitted).

### Shared memory usage (actual)

| Stage | Plan | Actual |
|---|---|---|
| Histogram | 8KB (2048 entries) | **2KB** (512 entries) + 64B ZRL stats |
| Encoder codebook | 8KB | **2KB** (512 entries) + 32B k_zrl |
| Decoder table | 8KB | **8KB** (2048 entries) + 32B k_zrl |

Smaller encoder/histogram footprint improves workgroup occupancy on M1.

### GPU stream buffer: 512 bytes (not 4096)

The GPU path uses `MAX_STREAM_BYTES = 512` per stream (vs 4096 in the CPU
fallback path), reducing GPU buffer allocation by 8x.

### Tests: inline (not separate file)

Tests are in `huffman.rs` as `#[cfg(test)] mod tests` (8 tests) rather than
a separate `tests/huffman_tests.rs` file.
