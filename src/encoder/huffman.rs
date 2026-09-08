//! Significance map + Canonical Huffman entropy coding.
//!
//! Same 256-stream architecture as Rice but with distribution-adaptive codewords
//! instead of fixed Golomb-Rice. Expected to close the low-bitrate gap (q<50)
//! where Rice has +34% overhead vs rANS.
//!
//! Per coefficient:
//!   - Zero run: bit=0 + Rice(run_length-1, k_zrl)  (same as Rice)
//!   - Non-zero: bit=1 + sign bit + Huffman(|val|-1)
//!
//! Huffman codebook is built per subband group from magnitude histogram.
//! Canonical codes are used — only code lengths need to be stored in the bitstream.

use super::rans::compute_subband_group;

/// Number of interleaved streams per tile (same as Rice).
pub const HUFFMAN_STREAMS_PER_TILE: usize = 256;

/// Max output bytes per stream.
pub const HUFFMAN_MAX_STREAM_BYTES: usize = 4096;

/// Huffman alphabet size. Symbols 0..(N-2) are direct magnitudes (|val|-1).
/// Symbol (N-1) is an escape code — followed by exp-Golomb coded excess magnitude.
/// 64 symbols covers >99% of wavelet magnitudes directly.
pub const HUFFMAN_ALPHABET_SIZE: usize = 64;

/// The escape symbol index (last symbol in alphabet).
pub const HUFFMAN_ESCAPE_SYM: usize = HUFFMAN_ALPHABET_SIZE - 1;

/// Maximum code length in bits. Capped at 8 so the GPU decoder's
/// 8-bit prefix table handles all codes in a single lookup (no slow path).
pub const HUFFMAN_MAX_CODE_LEN: u8 = 8;

/// Encoded tile using significance map + canonical Huffman coding with zero-run-length.
#[derive(Debug, Clone)]
pub struct HuffmanTile {
    pub num_coefficients: u32,
    pub tile_size: u32,
    pub num_levels: u32,
    pub num_groups: u32,
    /// Per-subband code lengths (code_lengths[g][sym] = number of bits for symbol sym in group g).
    /// Only code lengths are stored; canonical codes are reconstructed on decode.
    pub code_lengths: Vec<Vec<u8>>,
    /// Per-subband Rice parameter k for zero-run lengths (reuse Rice ZRL scheme).
    pub k_zrl_values: Vec<u8>,
    /// Bytes used by each of the 256 streams.
    pub stream_lengths: Vec<u32>,
    /// Concatenated stream data (all 256 streams).
    pub stream_data: Vec<u8>,
}

impl HuffmanTile {
    pub fn byte_size(&self) -> usize {
        // Header: 4 × u32 = 16 bytes
        // Per-group: 1 byte (alphabet_size) + code_lengths
        // k_zrl_values: num_groups bytes
        // stream_lengths: 256 × u16 = 512 bytes
        // stream_data
        let code_len_bytes: usize = self
            .code_lengths
            .iter()
            .map(|cl| 1 + cl.len()) // 1 byte for alphabet_size + code lengths
            .sum();
        16 + code_len_bytes
            + self.k_zrl_values.len()
            + self.stream_lengths.len() * 2
            + self.stream_data.len()
    }
}

/// Per-subband canonical Huffman codebook (used during encoding only).
#[derive(Debug, Clone)]
struct Codebook {
    /// Code length for each symbol (0 = symbol not present).
    code_lengths: Vec<u8>,
    /// Canonical codeword for each symbol.
    codewords: Vec<u32>,
}

/// Public wrapper: build codebook from frequency array, return (code_lengths, codewords).
/// Used by GPU host code to build codebooks from GPU histogram readback.
pub fn build_canonical_codebook_from_freq(freq: &[u32]) -> (Vec<u8>, Vec<u32>) {
    let cb = build_canonical_codebook(freq);
    (cb.code_lengths, cb.codewords)
}

/// Build a canonical Huffman codebook from frequency counts.
///
/// Steps:
/// 1. Build Huffman tree via min-heap
/// 2. Extract code lengths via DFS
/// 3. Clamp max length to HUFFMAN_MAX_CODE_LEN (redistribute)
/// 4. Assign canonical codes (sorted by length then symbol)
fn build_canonical_codebook(freq: &[u32]) -> Codebook {
    let n = freq.len();
    if n == 0 {
        return Codebook {
            code_lengths: Vec::new(),
            codewords: Vec::new(),
        };
    }

    // Count non-zero frequency symbols
    let active: Vec<usize> = (0..n).filter(|&i| freq[i] > 0).collect();

    if active.is_empty() {
        // All zeros — single dummy code
        return Codebook {
            code_lengths: vec![0; n],
            codewords: vec![0; n],
        };
    }

    if active.len() == 1 {
        // Single symbol — assign length 1
        let mut code_lengths = vec![0u8; n];
        code_lengths[active[0]] = 1;
        let mut codewords = vec![0u32; n];
        codewords[active[0]] = 0; // code = 0, 1 bit
        return Codebook {
            code_lengths,
            codewords,
        };
    }

    // Length-limited Huffman by frequency scaling (BUG-23).
    //
    // The natural tree can be far deeper than `HUFFMAN_MAX_CODE_LEN` — a geometric histogram over
    // 32 symbols reaches depth 32 — and the previous approach redistributed the excess by moving
    // one symbol from length j to two at j + 1. That donor pool is finite and the excess is not
    // bounded by it, so the loop could not always finish: it spun at 79% CPU for 8 minutes before
    // an assert was added, after which it refused the material instead of coding it.
    //
    // Build a *real* tree on scaled frequencies instead. While the tree is too deep, halve every
    // non-zero frequency and rebuild. Two properties carry it:
    //
    //   * **It terminates.** 32 halvings take any `u32` to 1, and a uniform alphabet of
    //     `HUFFMAN_ALPHABET_SIZE` = 64 symbols has depth 6 — inside the 8-bit limit. The assert
    //     below checks that bound rather than trusting it.
    //   * **The result satisfies Kraft by construction**, because it is an actual Huffman tree and
    //     not a patched length histogram. That is what makes `assign_canonical_codes` safe; the
    //     old path could leave excess in place and hand out codewords that were not a prefix code,
    //     which decodes to noise.
    //
    // Rounding up on the halving matters: a live symbol must never scale to zero and drop out of
    // the alphabet, because it would then have no code at all.
    let mut scaled: Vec<u32> = freq[..n].to_vec();
    let mut code_lengths = natural_code_lengths(&scaled, &active, n);
    let mut halvings = 0u32;
    while code_lengths.iter().any(|&l| l > HUFFMAN_MAX_CODE_LEN) {
        for f in scaled.iter_mut() {
            if *f > 1 {
                *f = (*f).div_ceil(2);
            }
        }
        halvings += 1;
        assert!(
            halvings <= 32,
            "length-limited Huffman did not converge after {halvings} halvings over {} active              symbols; every non-zero frequency should be 1 by now, and a uniform alphabet that              size fits {HUFFMAN_MAX_CODE_LEN}-bit codes",
            active.len(),
        );
        code_lengths = natural_code_lengths(&scaled, &active, n);
    }

    // Assign canonical codes
    let codewords = assign_canonical_codes(&code_lengths);

    Codebook {
        code_lengths,
        codewords,
    }
}

/// Code lengths from an unconstrained Huffman tree over `freq`, for the symbols in `active`.
///
/// Split out of `build_canonical_codebook` so the length-limiting loop can rebuild the tree on
/// scaled frequencies. Callers guarantee `active.len() >= 2`.
fn natural_code_lengths(freq: &[u32], active: &[usize], n: usize) -> Vec<u8> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    // Node: leaves are 0..n-1, internal nodes n.. in creation order.
    let mut node_freq: Vec<u64> = Vec::with_capacity(2 * n);
    let mut children: Vec<(usize, usize)> = Vec::with_capacity(n);
    for &f in freq.iter().take(n) {
        node_freq.push(f as u64);
    }

    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    for &sym in active {
        heap.push(Reverse((freq[sym] as u64, sym)));
    }

    while heap.len() > 1 {
        let Reverse((f1, n1)) = heap.pop().unwrap();
        let Reverse((f2, n2)) = heap.pop().unwrap();
        let new_id = node_freq.len();
        let combined = f1 + f2;
        node_freq.push(combined);
        children.push((n1, n2));
        heap.push(Reverse((combined, new_id)));
    }

    let mut code_lengths = vec![0u8; n];
    let root = heap.pop().unwrap().0 .1;

    fn assign_lengths(
        node: usize,
        depth: u8,
        n_leaves: usize,
        children: &[(usize, usize)],
        code_lengths: &mut [u8],
    ) {
        if node < n_leaves {
            code_lengths[node] = depth.max(1); // minimum 1 bit
        } else {
            let (left, right) = children[node - n_leaves];
            assign_lengths(left, depth + 1, n_leaves, children, code_lengths);
            assign_lengths(right, depth + 1, n_leaves, children, code_lengths);
        }
    }
    assign_lengths(root, 0, n, &children, &mut code_lengths);
    code_lengths
}

/// Assign canonical codes from code lengths.
/// Symbols sorted by (length, symbol), codes assigned sequentially.
fn assign_canonical_codes(lengths: &[u8]) -> Vec<u32> {
    let n = lengths.len();
    let mut codewords = vec![0u32; n];

    // Collect (symbol, length) pairs for active symbols
    let mut pairs: Vec<(usize, u8)> = (0..n)
        .filter(|&i| lengths[i] > 0)
        .map(|i| (i, lengths[i]))
        .collect();

    if pairs.is_empty() {
        return codewords;
    }

    // Sort by (length, symbol)
    pairs.sort_by_key(|&(sym, len)| (len, sym));

    // Assign canonical codes
    let mut code: u32 = 0;
    let mut prev_len = pairs[0].1;

    for (i, &(sym, len)) in pairs.iter().enumerate() {
        if i > 0 {
            code += 1;
            // Left-shift when moving to a longer code length
            code <<= (len - prev_len) as u32;
        }
        codewords[sym] = code;
        prev_len = len;
    }

    codewords
}

/// Reconstruct canonical codewords from code lengths (for decoder).
pub fn reconstruct_codes_from_lengths(lengths: &[u8]) -> Vec<u32> {
    assign_canonical_codes(lengths)
}

/// Build an 8-bit prefix decode table for fast Huffman decoding.
///
/// table[peek_8_bits] = (symbol << 16) | code_length
/// For codes <= 8 bits, the entry is replicated 2^(8-len) times.
/// For codes > 8 bits, code_length = 0 (flag for bit-by-bit fallback).
pub fn build_decode_table(lengths: &[u8], codewords: &[u32]) -> Vec<u32> {
    let mut table = vec![0u32; 256]; // 8-bit prefix table

    for (sym, (&len, &code)) in lengths.iter().zip(codewords.iter()).enumerate() {
        if len == 0 {
            continue;
        }
        if len > 8 {
            // Codes > 8 bits: mark entries with len=0 (fallback flag)
            // The decoder will do bit-by-bit scanning for these
            let prefix = (code >> (len - 8)) as usize;
            // Just mark the one entry — decoder checks len==0 to know it needs fallback
            if table[prefix] == 0 {
                table[prefix] = 0; // len=0 means "needs fallback"
            }
            continue;
        }
        // Codes <= 8 bits: replicate entry for all suffixes
        let padding = 8 - len;
        let base = (code << padding) as usize;
        let count = 1usize << padding;
        let entry = ((sym as u32) << 16) | (len as u32);
        for j in 0..count {
            table[base + j] = entry;
        }
    }

    table
}

// --- BitWriter / BitReader (same as Rice) ---

struct BitWriter {
    buf: Vec<u8>,
    current_byte: u8,
    bits_in_byte: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(HUFFMAN_MAX_STREAM_BYTES),
            current_byte: 0,
            bits_in_byte: 0,
        }
    }

    #[inline]
    fn write_bit(&mut self, bit: u8) {
        self.current_byte = (self.current_byte << 1) | (bit & 1);
        self.bits_in_byte += 1;
        if self.bits_in_byte == 8 {
            self.buf.push(self.current_byte);
            self.current_byte = 0;
            self.bits_in_byte = 0;
        }
    }

    #[inline]
    fn write_bits(&mut self, value: u32, count: u8) {
        for i in (0..count).rev() {
            self.write_bit(((value >> i) & 1) as u8);
        }
    }

    #[inline]
    fn write_rice(&mut self, value: u32, k: u8) {
        let quotient = value >> k;
        let remainder = value & ((1u32 << k) - 1);
        let q = quotient.min(31);
        for _ in 0..q {
            self.write_bit(1);
        }
        self.write_bit(0);
        if k > 0 {
            self.write_bits(remainder, k);
        }
    }

    #[inline]
    fn write_exp_golomb(&mut self, value: u32) {
        if value == 0 {
            self.write_bit(1);
            return;
        }
        let v = value + 1;
        let bits = 31 - v.leading_zeros() as u8;
        for _ in 0..bits {
            self.write_bit(0);
        }
        self.write_bits(v, bits + 1);
    }

    fn flush(mut self) -> Vec<u8> {
        if self.bits_in_byte > 0 {
            self.current_byte <<= 8 - self.bits_in_byte;
            self.buf.push(self.current_byte);
        }
        self.buf
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    #[inline]
    fn read_bit(&mut self) -> u8 {
        if self.byte_pos >= self.data.len() {
            return 0;
        }
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        bit
    }

    #[inline]
    fn read_bits(&mut self, count: u8) -> u32 {
        let mut value = 0u32;
        for _ in 0..count {
            value = (value << 1) | self.read_bit() as u32;
        }
        value
    }

    #[inline]
    fn read_rice(&mut self, k: u8) -> u32 {
        let mut quotient = 0u32;
        while self.read_bit() == 1 && quotient < 31 {
            quotient += 1;
        }
        let remainder = if k > 0 { self.read_bits(k) } else { 0 };
        (quotient << k) | remainder
    }

    /// Peek at the next `count` bits without consuming them.
    #[inline]
    fn peek_bits(&self, count: u8) -> u32 {
        let mut value = 0u32;
        let mut byte_pos = self.byte_pos;
        let mut bit_pos = self.bit_pos;
        for _ in 0..count {
            if byte_pos >= self.data.len() {
                value <<= 1;
                continue;
            }
            let bit = (self.data[byte_pos] >> (7 - bit_pos)) & 1;
            value = (value << 1) | bit as u32;
            bit_pos += 1;
            if bit_pos == 8 {
                bit_pos = 0;
                byte_pos += 1;
            }
        }
        value
    }

    /// Consume (skip) `count` bits.
    #[inline]
    fn consume_bits(&mut self, count: u8) {
        for _ in 0..count {
            let _ = self.read_bit();
        }
    }

    #[inline]
    fn read_exp_golomb(&mut self) -> u32 {
        let mut leading_zeros = 0u32;
        while self.read_bit() == 0 && leading_zeros < 20 {
            leading_zeros += 1;
        }
        if leading_zeros == 0 {
            return 0;
        }
        let rest = self.read_bits(leading_zeros as u8);
        (1u32 << leading_zeros) - 1 + rest
    }
}

/// Compute optimal Rice parameter k for zero-run lengths.
fn optimal_k_zrl(values: &[u32]) -> u8 {
    if values.is_empty() {
        return 0;
    }
    let sum: u64 = values.iter().map(|&v| v as u64).sum();
    let mean = sum / values.len() as u64;
    if mean == 0 {
        return 0;
    }
    (63 - mean.leading_zeros()).min(15) as u8
}

/// Tile-local raster index of symbol `s` in stream `stream_id`.
///
/// Streams walk the tile in **column-major** order, cut into `HUFFMAN_STREAMS_PER_TILE`
/// contiguous segments, so the previous symbol in a stream is the coefficient directly above it.
/// That vertical adjacency is what the ZRL runs and the adaptive `k_zrl` are tuned against.
///
/// BUG-14: this backend carried BUG-11's `stream_id + s * HUFFMAN_STREAMS_PER_TILE`, i.e.
/// `i % 256`. At the default 256 px tile a segment is exactly one column and the two mappings
/// agree coefficient for coefficient, so no shipped bitstream moves. At any other width `i % 256`
/// interleaved columns `tile_size / 256` apart into one stream, which broke both the zero runs and
/// the `k_zrl` they are coded with. Must stay in sync with the three `huffman_*.wgsl` shaders.
#[inline]
fn stream_coeff_index(
    stream_id: usize,
    s: usize,
    symbols_per_stream: usize,
    tile_size: usize,
) -> usize {
    // Column-major position within the tile: j = x * tile_size + y.
    let j = stream_id * symbols_per_stream + s;
    (j % tile_size) * tile_size + j / tile_size
}

/// Encode a tile of quantized coefficients using canonical Huffman with ZRL.
///
/// Token encoding (per stream):
///   bit=0 → zero run: Rice(run_length-1, k_zrl)
///   bit=1 → non-zero: sign bit + Huffman(min(|val|-1, ESCAPE_SYM))
///            if |val|-1 >= ESCAPE_SYM: followed by exp-Golomb(excess)
pub fn huffman_encode_tile(coefficients: &[i32], tile_size: u32, num_levels: u32) -> HuffmanTile {
    let num_coefficients = coefficients.len();
    // `.max(1)`, as in rice.rs: the MED lossless path (LOSSLESS-1) codes prediction
    // residuals with `num_levels = 0`, which used to make this zero groups and panic on the
    // first coefficient — `index out of bounds: the len is 0 but the index is 0`.
    let num_groups = (num_levels * 2).max(1) as usize;
    let symbols_per_stream = num_coefficients / HUFFMAN_STREAMS_PER_TILE;

    // Phase 1: Build magnitude histograms per subband group
    let max_direct = HUFFMAN_ESCAPE_SYM; // magnitudes 0..max_direct-1 are direct, rest use escape
    let mut group_freqs: Vec<Vec<u32>> = vec![vec![0u32; HUFFMAN_ALPHABET_SIZE]; num_groups];
    for (idx, &coeff) in coefficients.iter().enumerate() {
        if coeff != 0 {
            let y = (idx / tile_size as usize) as u32;
            let x = (idx % tile_size as usize) as u32;
            let g = compute_subband_group(x, y, tile_size, num_levels);
            let mag = (coeff.unsigned_abs() - 1) as usize;
            if mag < max_direct {
                group_freqs[g][mag] += 1;
            } else {
                group_freqs[g][HUFFMAN_ESCAPE_SYM] += 1;
            }
        }
    }

    // Phase 2: Build codebooks per group
    let codebooks: Vec<Codebook> = group_freqs
        .iter()
        .map(|freq| build_canonical_codebook(freq))
        .collect();

    // Phase 3: Compute k_zrl per subband (same as Rice)
    let mut group_run_lengths: Vec<Vec<u32>> = vec![Vec::new(); num_groups];
    for stream_id in 0..HUFFMAN_STREAMS_PER_TILE {
        let mut run = 0u32;
        let mut zrl_group = 0usize;
        for s in 0..symbols_per_stream {
            let coeff_idx =
                stream_coeff_index(stream_id, s, symbols_per_stream, tile_size as usize);
            if coefficients[coeff_idx] == 0 {
                if run == 0 {
                    let y = (coeff_idx / tile_size as usize) as u32;
                    let x = (coeff_idx % tile_size as usize) as u32;
                    zrl_group = compute_subband_group(x, y, tile_size, num_levels);
                }
                run += 1;
            } else if run > 0 {
                group_run_lengths[zrl_group].push(run - 1);
                run = 0;
            }
        }
        if run > 0 {
            group_run_lengths[zrl_group].push(run - 1);
        }
    }
    let k_zrl_values: Vec<u8> = group_run_lengths.iter().map(|v| optimal_k_zrl(v)).collect();

    // Phase 4: Encode 256 interleaved streams
    let mut all_stream_data = Vec::new();
    let mut stream_lengths = Vec::with_capacity(HUFFMAN_STREAMS_PER_TILE);

    for stream_id in 0..HUFFMAN_STREAMS_PER_TILE {
        let mut writer = BitWriter::new();
        let mut s = 0;

        while s < symbols_per_stream {
            let coeff_idx =
                stream_coeff_index(stream_id, s, symbols_per_stream, tile_size as usize);
            let coeff = coefficients[coeff_idx];

            if coeff == 0 {
                // Zero run with Rice ZRL (same as Rice coder)
                let y0 = (coeff_idx / tile_size as usize) as u32;
                let x0 = (coeff_idx % tile_size as usize) as u32;
                let g_zrl = compute_subband_group(x0, y0, tile_size, num_levels);

                let k = k_zrl_values[g_zrl];
                let max_run = 32u32 << k;
                let mut run = 1u32;
                while s + (run as usize) < symbols_per_stream && run < max_run {
                    let next_idx = stream_coeff_index(
                        stream_id,
                        s + run as usize,
                        symbols_per_stream,
                        tile_size as usize,
                    );
                    if coefficients[next_idx] != 0 {
                        break;
                    }
                    run += 1;
                }
                writer.write_bit(0); // token: zero run
                writer.write_rice(run - 1, k);
                s += run as usize;
            } else {
                // Non-zero: significance + sign + Huffman code
                writer.write_bit(1);
                writer.write_bit(if coeff < 0 { 1 } else { 0 });

                let magnitude = coeff.unsigned_abs() - 1;
                let y = (coeff_idx / tile_size as usize) as u32;
                let x = (coeff_idx % tile_size as usize) as u32;
                let g = compute_subband_group(x, y, tile_size, num_levels);

                let sym = if (magnitude as usize) < HUFFMAN_ESCAPE_SYM {
                    magnitude as usize
                } else {
                    HUFFMAN_ESCAPE_SYM
                };
                let cb = &codebooks[g];

                // Write Huffman code for symbol
                writer.write_bits(cb.codewords[sym], cb.code_lengths[sym]);

                // Escape: append exp-Golomb coded excess magnitude
                if sym == HUFFMAN_ESCAPE_SYM {
                    writer.write_exp_golomb(magnitude - HUFFMAN_ESCAPE_SYM as u32);
                }

                s += 1;
            }
        }

        let stream_bytes = writer.flush();
        stream_lengths.push(stream_bytes.len() as u32);
        all_stream_data.extend_from_slice(&stream_bytes);
    }

    // Extract code lengths for serialization
    let code_lengths: Vec<Vec<u8>> = codebooks
        .iter()
        .map(|cb| cb.code_lengths.clone())
        .collect();

    HuffmanTile {
        num_coefficients: num_coefficients as u32,
        tile_size,
        num_levels,
        num_groups: num_groups as u32,
        code_lengths,
        k_zrl_values,
        stream_lengths,
        stream_data: all_stream_data,
    }
}

/// Decode a Huffman-coded tile back to quantized coefficients.
pub fn huffman_decode_tile(tile: &HuffmanTile) -> Vec<i32> {
    let num_coefficients = tile.num_coefficients as usize;
    let symbols_per_stream = num_coefficients / HUFFMAN_STREAMS_PER_TILE;
    let mut coefficients = vec![0i32; num_coefficients];

    // Reconstruct codebooks from stored code lengths
    let codebooks: Vec<(Vec<u32>, Vec<u32>)> = tile
        .code_lengths
        .iter()
        .map(|lengths| {
            let codewords = reconstruct_codes_from_lengths(lengths);
            let decode_table = build_decode_table(lengths, &codewords);
            (codewords, decode_table)
        })
        .collect();

    let mut data_offset = 0usize;
    for stream_id in 0..HUFFMAN_STREAMS_PER_TILE {
        let stream_len = tile.stream_lengths[stream_id] as usize;
        let stream_data = &tile.stream_data[data_offset..data_offset + stream_len];
        let mut reader = BitReader::new(stream_data);

        let mut s = 0usize;
        while s < symbols_per_stream {
            let token = reader.read_bit();
            if token == 0 {
                // Zero run
                let first_idx = stream_coeff_index(
                    stream_id,
                    s,
                    symbols_per_stream,
                    tile.tile_size as usize,
                );
                let zy = (first_idx / tile.tile_size as usize) as u32;
                let zx = (first_idx % tile.tile_size as usize) as u32;
                let g_zrl = compute_subband_group(zx, zy, tile.tile_size, tile.num_levels);
                let run = reader.read_rice(tile.k_zrl_values[g_zrl]) + 1;
                // Zeros are already 0 in the output buffer
                s += run as usize;
            } else {
                // Non-zero coefficient
                let coeff_idx = stream_coeff_index(
                    stream_id,
                    s,
                    symbols_per_stream,
                    tile.tile_size as usize,
                );
                let sign = reader.read_bit();

                let y = (coeff_idx / tile.tile_size as usize) as u32;
                let x = (coeff_idx % tile.tile_size as usize) as u32;
                let g = compute_subband_group(x, y, tile.tile_size, tile.num_levels);

                let (_, decode_table) = &codebooks[g];
                let code_lengths_g = &tile.code_lengths[g];

                // Decode Huffman symbol using prefix table
                let peek = reader.peek_bits(8) as usize;
                let entry = decode_table[peek];
                let sym;
                let code_len = (entry & 0xFFFF) as u8;

                if code_len > 0 {
                    // Fast path: code is <= 8 bits
                    sym = entry >> 16;
                    reader.consume_bits(code_len);
                } else {
                    // Slow path: code > 8 bits, bit-by-bit scan
                    sym = decode_slow(&mut reader, code_lengths_g, &codebooks[g].0);
                }

                let magnitude = if sym >= HUFFMAN_ESCAPE_SYM as u32 {
                    // Escape code: read exp-Golomb coded excess magnitude
                    HUFFMAN_ESCAPE_SYM as u32 + reader.read_exp_golomb()
                } else {
                    sym
                };

                let val = (magnitude + 1) as i32;
                coefficients[coeff_idx] = if sign == 1 { -val } else { val };
                s += 1;
            }
        }

        data_offset += stream_len;
    }

    coefficients
}

/// Slow-path Huffman decode: bit-by-bit matching for codes > 8 bits.
fn decode_slow(reader: &mut BitReader, code_lengths: &[u8], codewords: &[u32]) -> u32 {
    // Build code bit-by-bit and match against codebook
    let mut code: u32 = 0;
    for bits_read in 1..=HUFFMAN_MAX_CODE_LEN {
        code = (code << 1) | reader.read_bit() as u32;
        // Check all symbols with this code length
        for (sym, (&len, &cw)) in code_lengths.iter().zip(codewords.iter()).enumerate() {
            if len == bits_read && cw == code {
                return sym as u32;
            }
        }
    }
    // Shouldn't reach here with valid data
    0
}

/// Serialize a HuffmanTile to bytes.
pub fn serialize_tile_huffman(tile: &HuffmanTile) -> Vec<u8> {
    let mut out = Vec::with_capacity(tile.byte_size());
    out.extend_from_slice(&tile.num_coefficients.to_le_bytes());
    out.extend_from_slice(&tile.tile_size.to_le_bytes());
    out.extend_from_slice(&tile.num_levels.to_le_bytes());
    out.extend_from_slice(&tile.num_groups.to_le_bytes());

    // Per-group code lengths
    for cl in &tile.code_lengths {
        // Trim trailing zeros to save space
        let effective_len = cl.iter().rposition(|&l| l > 0).map_or(0, |i| i + 1);
        out.push(effective_len as u8);
        out.extend_from_slice(&cl[..effective_len]);
    }

    // k_zrl values
    out.extend_from_slice(&tile.k_zrl_values);

    // Stream lengths as u16
    for &len in &tile.stream_lengths {
        out.extend_from_slice(&(len as u16).to_le_bytes());
    }

    // Stream data
    out.extend_from_slice(&tile.stream_data);
    out
}

/// Deserialize a HuffmanTile from bytes. Returns (tile, bytes_consumed).
pub fn deserialize_tile_huffman(data: &[u8]) -> (HuffmanTile, usize) {
    let mut pos = 0;

    let num_coefficients = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let tile_size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let num_levels = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let num_groups = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    // Per-group code lengths
    let mut code_lengths = Vec::with_capacity(num_groups as usize);
    for _ in 0..num_groups {
        let effective_len = data[pos] as usize;
        pos += 1;
        let mut cl = vec![0u8; HUFFMAN_ALPHABET_SIZE];
        cl[..effective_len].copy_from_slice(&data[pos..pos + effective_len]);
        pos += effective_len;
        code_lengths.push(cl);
    }

    // k_zrl values
    let k_zrl_values = data[pos..pos + num_groups as usize].to_vec();
    pos += num_groups as usize;

    // Stream lengths
    let mut stream_lengths = Vec::with_capacity(HUFFMAN_STREAMS_PER_TILE);
    for _ in 0..HUFFMAN_STREAMS_PER_TILE {
        let len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as u32;
        stream_lengths.push(len);
        pos += 2;
    }

    // Stream data
    let total_data: usize = stream_lengths.iter().map(|&l| l as usize).sum();
    let stream_data = data[pos..pos + total_data].to_vec();
    pos += total_data;

    (
        HuffmanTile {
            num_coefficients,
            tile_size,
            num_levels,
            num_groups,
            code_lengths,
            k_zrl_values,
            stream_lengths,
            stream_data,
        },
        pos,
    )
}

/// Pack multiple HuffmanTiles into bytes (used by format.rs).
pub fn pack_tiles_huffman(tiles: &[HuffmanTile]) -> (Vec<u8>, Vec<u32>) {
    let mut blob = Vec::new();
    let mut sizes = Vec::with_capacity(tiles.len());
    for tile in tiles {
        let serialized = serialize_tile_huffman(tile);
        sizes.push(serialized.len() as u32);
        blob.extend_from_slice(&serialized);
    }
    (blob, sizes)
}

/// Unpack HuffmanTiles from bytes with known sizes.
pub fn unpack_tiles_huffman(data: &[u8], tile_sizes: &[u32]) -> Vec<HuffmanTile> {
    let mut tiles = Vec::with_capacity(tile_sizes.len());
    let mut pos = 0;
    for &size in tile_sizes {
        let (tile, _) = deserialize_tile_huffman(&data[pos..pos + size as usize]);
        tiles.push(tile);
        pos += size as usize;
    }
    tiles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_codebook_basic() {
        // Simple frequency distribution: sym0=4, sym1=2, sym2=1, sym3=1
        let freq = vec![4, 2, 1, 1];
        let cb = build_canonical_codebook(&freq);
        // Most frequent symbol should get shortest code
        assert_eq!(cb.code_lengths[0], 1); // freq=4 → 1 bit
        assert!(cb.code_lengths[1] <= 3);
        assert!(cb.code_lengths[2] <= 3);
        assert!(cb.code_lengths[3] <= 3);
    }

    #[test]
    fn test_canonical_codebook_single_symbol() {
        let freq = vec![0, 0, 10, 0];
        let cb = build_canonical_codebook(&freq);
        assert_eq!(cb.code_lengths[2], 1);
        assert_eq!(cb.code_lengths[0], 0);
        assert_eq!(cb.code_lengths[1], 0);
        assert_eq!(cb.code_lengths[3], 0);
    }

    #[test]
    fn test_canonical_codebook_empty() {
        let freq = vec![0, 0, 0, 0];
        let cb = build_canonical_codebook(&freq);
        assert!(cb.code_lengths.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_decode_table_roundtrip() {
        let freq = vec![10, 5, 3, 1, 1, 0, 0, 0];
        let cb = build_canonical_codebook(&freq);
        let table = build_decode_table(&cb.code_lengths, &cb.codewords);

        // Verify each active symbol can be decoded
        for (sym, (&len, &code)) in cb.code_lengths.iter().zip(cb.codewords.iter()).enumerate() {
            if len == 0 || len > 8 {
                continue;
            }
            let padded = (code << (8 - len)) as usize;
            let entry = table[padded];
            assert_eq!((entry >> 16) as usize, sym, "Symbol mismatch for sym={}", sym);
            assert_eq!((entry & 0xFFFF) as u8, len, "Length mismatch for sym={}", sym);
        }
    }

    #[test]
    fn test_huffman_roundtrip_zeros() {
        let coefficients = vec![0i32; 256 * 64]; // all zeros
        let tile = huffman_encode_tile(&coefficients, 128, 3);
        let decoded = huffman_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_huffman_roundtrip_simple() {
        let mut coefficients = vec![0i32; 256 * 64];
        // Place some non-zero values
        for i in 0..100 {
            coefficients[i * 7] = ((i % 10) as i32 + 1) * if i % 2 == 0 { 1 } else { -1 };
        }
        let tile = huffman_encode_tile(&coefficients, 128, 3);
        let decoded = huffman_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_huffman_roundtrip_varied() {
        let mut coefficients = vec![0i32; 256 * 64];
        // Mix of magnitudes
        for (i, c) in coefficients.iter_mut().enumerate() {
            let v = (i % 256) as i32;
            *c = if v < 128 { 0 } else { (v - 128) * if i % 3 == 0 { -1 } else { 1 } };
        }
        let tile = huffman_encode_tile(&coefficients, 128, 3);
        let decoded = huffman_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_huffman_roundtrip_large_magnitudes() {
        let mut coefficients = vec![0i32; 256 * 64];
        // Test escape code path (magnitudes >= 255)
        coefficients[0] = 256;
        coefficients[256] = -512;
        coefficients[512] = 1000;
        coefficients[768] = -4000;
        let tile = huffman_encode_tile(&coefficients, 128, 3);
        let decoded = huffman_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_huffman_serialize_roundtrip() {
        let mut coefficients = vec![0i32; 256 * 64];
        for i in 0..200 {
            coefficients[i * 5] = ((i % 20) as i32 + 1) * if i % 2 == 0 { 1 } else { -1 };
        }
        let tile = huffman_encode_tile(&coefficients, 128, 3);
        let serialized = serialize_tile_huffman(&tile);
        let (deserialized, consumed) = deserialize_tile_huffman(&serialized);
        assert_eq!(consumed, serialized.len());
        let decoded = huffman_decode_tile(&deserialized);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_huffman_compression_vs_raw() {
        let mut coefficients = vec![0i32; 256 * 64];
        // Geometric-like distribution (many small values)
        for (i, c) in coefficients.iter_mut().enumerate() {
            let r = (i * 7 + 13) % 256;
            if r < 40 {
                *c = (r as i32 % 10 + 1) * if r % 2 == 0 { 1 } else { -1 };
            }
        }
        let tile = huffman_encode_tile(&coefficients, 128, 3);
        let raw_bytes = coefficients.len() * 4;
        let compressed_bytes = tile.byte_size();
        assert!(
            compressed_bytes < raw_bytes,
            "Huffman should compress: {} >= {} raw bytes",
            compressed_bytes,
            raw_bytes
        );
    }

    /// BUG-14: at 256 px the column-major mapping must agree with the legacy `i % 256` one,
    /// coefficient for coefficient. That is what makes the fix bitstream-neutral at the default
    /// tile size, so it is asserted rather than argued.
    #[test]
    fn test_stream_mapping_matches_legacy_at_256() {
        let tile_size = HUFFMAN_STREAMS_PER_TILE;
        let symbols_per_stream = tile_size * tile_size / HUFFMAN_STREAMS_PER_TILE;
        for stream_id in 0..HUFFMAN_STREAMS_PER_TILE {
            for s in 0..symbols_per_stream {
                assert_eq!(
                    stream_coeff_index(stream_id, s, symbols_per_stream, tile_size),
                    stream_id + s * HUFFMAN_STREAMS_PER_TILE,
                    "stream {stream_id} symbol {s}"
                );
            }
        }
    }

    /// Every coefficient of the tile must be visited exactly once, at any width. A mapping that
    /// dropped or doubled one would silently corrupt only part of the picture.
    #[test]
    fn test_stream_mapping_is_a_permutation() {
        for tile_size in [128usize, 256, 512] {
            let n = tile_size * tile_size;
            let symbols_per_stream = n / HUFFMAN_STREAMS_PER_TILE;
            let mut seen = vec![false; n];
            for stream_id in 0..HUFFMAN_STREAMS_PER_TILE {
                for s in 0..symbols_per_stream {
                    let idx = stream_coeff_index(stream_id, s, symbols_per_stream, tile_size);
                    assert!(idx < n, "tile {tile_size}: index {idx} out of range");
                    assert!(!seen[idx], "tile {tile_size}: index {idx} visited twice");
                    seen[idx] = true;
                }
            }
            assert!(seen.iter().all(|&b| b), "tile {tile_size}: coefficients left unvisited");
        }
    }

    /// A distribution too skewed for the natural tree must be *coded*, not refused, and not spun.
    ///
    /// History, because this test has meant three different things. The original
    /// `clamp_code_lengths` redistributed excess length by moving one symbol from length j to two
    /// at j + 1 — a finite donor pool against an unbounded excess — and on this histogram it had
    /// nothing left to donate and did it forever (79% CPU, 8 minutes, killed). BUG-23 then made it
    /// assert, and this test asserted the panic. Neither coded the material. The length-limited
    /// construction does, so the assertion is now about the codebook being usable.
    #[test]
    fn test_codebook_length_limits_a_steeply_skewed_distribution() {
        let mut freq = vec![1u32; 32];
        for (i, f) in freq.iter_mut().enumerate().skip(1) {
            *f = 1u32 << (i - 1);
        }
        let cb = build_canonical_codebook(&freq);

        // Every active symbol has a code, and no code exceeds the limit the decoder reads.
        for (sym, &f) in freq.iter().enumerate() {
            if f > 0 {
                assert!(
                    cb.code_lengths[sym] >= 1 && cb.code_lengths[sym] <= HUFFMAN_MAX_CODE_LEN,
                    "symbol {sym} has length {}, outside 1..={HUFFMAN_MAX_CODE_LEN}",
                    cb.code_lengths[sym]
                );
            }
        }

        // Kraft equality: a canonical prefix code over these lengths exists and is complete.
        // This is the property the old redistribution could violate while still returning, which
        // would have made `assign_canonical_codes` emit a non-prefix code and decode to noise.
        let kraft: f64 = cb
            .code_lengths
            .iter()
            .filter(|&&l| l > 0)
            .map(|&l| 2f64.powi(-(l as i32)))
            .sum();
        assert!(
            (kraft - 1.0).abs() < 1e-9,
            "Kraft sum is {kraft}, so these lengths are not a complete prefix code"
        );
    }

    /// The natural tree for this histogram is deeper than the limit, so the scaling loop must
    /// actually run. Without that, the test above would pass for the wrong reason.
    #[test]
    fn test_length_limiting_is_actually_exercised() {
        let mut freq = vec![1u32; 32];
        for (i, f) in freq.iter_mut().enumerate().skip(1) {
            *f = 1u32 << (i - 1);
        }
        let active: Vec<usize> = (0..freq.len()).filter(|&i| freq[i] > 0).collect();
        let natural = natural_code_lengths(&freq, &active, freq.len());
        let deepest = natural.iter().copied().max().unwrap_or(0);
        assert!(
            deepest > HUFFMAN_MAX_CODE_LEN,
            "the unconstrained tree is only {deepest} deep, so this histogram no longer tests              length limiting — pick a more skewed one"
        );
    }

    /// The host encoder and decoder must agree at every tile width, not just the default one.
    #[test]
    fn test_huffman_roundtrip_across_tile_sizes() {
        for tile_size in [128u32, 256, 512] {
            let n = (tile_size * tile_size) as usize;
            let mut coefficients = vec![0i32; n];
            for (i, c) in coefficients.iter_mut().enumerate() {
                let r = (i * 31 + 7) % 97;
                if r < 30 {
                    *c = (r as i32 % 12 + 1) * if r % 2 == 0 { 1 } else { -1 };
                }
            }
            let tile = huffman_encode_tile(&coefficients, tile_size, 3);
            let decoded = huffman_decode_tile(&tile);
            assert_eq!(coefficients, decoded, "roundtrip failed at tile {tile_size}");
        }
    }
}
