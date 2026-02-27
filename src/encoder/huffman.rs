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
/// Symbol (N-1) is an escape code — followed by raw 12-bit magnitude.
/// 32 symbols covers >95% of wavelet magnitudes directly while keeping
/// codebook headers small (32 bytes/group vs 256 bytes/group).
pub const HUFFMAN_ALPHABET_SIZE: usize = 32;

/// The escape symbol index (last symbol in alphabet).
pub const HUFFMAN_ESCAPE_SYM: usize = HUFFMAN_ALPHABET_SIZE - 1;

/// Maximum code length in bits. Prevents pathological codebooks.
pub const HUFFMAN_MAX_CODE_LEN: u8 = 15;

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

    // Build Huffman tree using a simple priority queue (Vec-based min-heap).
    // Node: (frequency, node_id). Leaf nodes = 0..n-1, internal = n..
    let mut node_freq: Vec<u64> = Vec::with_capacity(2 * n);
    let mut children: Vec<(usize, usize)> = Vec::with_capacity(n);

    // Initialize leaf nodes with their frequencies
    for i in 0..n {
        node_freq.push(freq[i] as u64);
    }

    // Min-heap of (freq, node_id) — use BinaryHeap with Reverse for min
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    for &sym in &active {
        heap.push(Reverse((freq[sym] as u64, sym)));
    }

    // Build tree by merging two lowest-frequency nodes
    while heap.len() > 1 {
        let Reverse((f1, n1)) = heap.pop().unwrap();
        let Reverse((f2, n2)) = heap.pop().unwrap();
        let new_id = node_freq.len();
        let combined = f1 + f2;
        node_freq.push(combined);
        children.push((n1, n2));
        heap.push(Reverse((combined, new_id)));
    }

    // Extract code lengths via DFS
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
            // Leaf node
            code_lengths[node] = depth.max(1); // minimum 1 bit
        } else {
            let (left, right) = children[node - n_leaves];
            assign_lengths(left, depth + 1, n_leaves, children, code_lengths);
            assign_lengths(right, depth + 1, n_leaves, children, code_lengths);
        }
    }

    assign_lengths(root, 0, n, &children, &mut code_lengths);

    // Clamp max code length to HUFFMAN_MAX_CODE_LEN
    clamp_code_lengths(&mut code_lengths, freq);

    // Assign canonical codes
    let codewords = assign_canonical_codes(&code_lengths);

    Codebook {
        code_lengths,
        codewords,
    }
}

/// Clamp code lengths to max_len by redistributing excess length.
/// Uses the algorithm from JPEG: iteratively shorten longest codes.
fn clamp_code_lengths(lengths: &mut [u8], _freq: &[u32]) {
    let max_len = HUFFMAN_MAX_CODE_LEN;

    // Count lengths
    let mut len_count = vec![0u32; (max_len as usize) + 2];
    let mut overflow = false;
    for &l in lengths.iter() {
        if l > 0 {
            if l > max_len {
                overflow = true;
                len_count[max_len as usize] += 1;
            } else {
                len_count[l as usize] += 1;
            }
        }
    }

    if !overflow {
        return;
    }

    // Count how many symbols exceeded max_len
    let mut excess_bits: i32 = 0;
    for &l in lengths.iter() {
        if l > max_len {
            excess_bits += (l as i32) - (max_len as i32);
        }
    }

    // Redistribute: move symbols from longest lengths to shorter ones
    // Simple approach: increase counts at max_len, decrease at max_len-1
    while excess_bits > 0 {
        // Find a non-zero count below max_len to donate
        for j in (1..max_len as usize).rev() {
            if len_count[j] > 0 {
                len_count[j] -= 1;
                len_count[j + 1] += 2;
                excess_bits -= 1;
                break;
            }
        }
        if excess_bits <= 0 {
            break;
        }
    }

    // Reassign lengths based on new counts (sort symbols by original length, then symbol)
    let mut syms: Vec<usize> = (0..lengths.len())
        .filter(|&i| lengths[i] > 0)
        .collect();
    syms.sort_by_key(|&i| (lengths[i], i));

    // Clamp all to max_len first
    for &s in &syms {
        if lengths[s] > max_len {
            lengths[s] = max_len;
        }
    }

    // Assign new lengths from len_count
    let mut sym_idx = 0;
    for l in 1..=max_len as usize {
        for _ in 0..len_count[l] {
            if sym_idx < syms.len() {
                lengths[syms[sym_idx]] = l as u8;
                sym_idx += 1;
            }
        }
    }
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

/// Encode a tile of quantized coefficients using canonical Huffman with ZRL.
///
/// Token encoding (per stream):
///   bit=0 → zero run: Rice(run_length-1, k_zrl)
///   bit=1 → non-zero: sign bit + Huffman(min(|val|-1, 255))
///            if |val|-1 >= 255: followed by raw 12-bit magnitude
pub fn huffman_encode_tile(coefficients: &[i32], tile_size: u32, num_levels: u32) -> HuffmanTile {
    let num_coefficients = coefficients.len();
    let num_groups = (num_levels * 2) as usize;
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
            let coeff_idx = stream_id + s * HUFFMAN_STREAMS_PER_TILE;
            if coefficients[coeff_idx] == 0 {
                if run == 0 {
                    let y = (coeff_idx / tile_size as usize) as u32;
                    let x = (coeff_idx % tile_size as usize) as u32;
                    zrl_group = compute_subband_group(x, y, tile_size, num_levels);
                }
                run += 1;
            } else {
                if run > 0 {
                    group_run_lengths[zrl_group].push(run - 1);
                    run = 0;
                }
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
            let coeff_idx = stream_id + s * HUFFMAN_STREAMS_PER_TILE;
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
                    let next_idx = stream_id + (s + run as usize) * HUFFMAN_STREAMS_PER_TILE;
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

                // Escape: append raw 12-bit magnitude for large values
                if sym == HUFFMAN_ESCAPE_SYM {
                    writer.write_bits(magnitude.min(4095), 12);
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
                let first_idx = stream_id + s * HUFFMAN_STREAMS_PER_TILE;
                let zy = (first_idx / tile.tile_size as usize) as u32;
                let zx = (first_idx % tile.tile_size as usize) as u32;
                let g_zrl = compute_subband_group(zx, zy, tile.tile_size, tile.num_levels);
                let run = reader.read_rice(tile.k_zrl_values[g_zrl]) + 1;
                // Zeros are already 0 in the output buffer
                s += run as usize;
            } else {
                // Non-zero coefficient
                let coeff_idx = stream_id + s * HUFFMAN_STREAMS_PER_TILE;
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
                    sym = (entry >> 16) as u32;
                    reader.consume_bits(code_len);
                } else {
                    // Slow path: code > 8 bits, bit-by-bit scan
                    sym = decode_slow(&mut reader, code_lengths_g, &codebooks[g].0);
                }

                let magnitude = if sym >= HUFFMAN_ESCAPE_SYM as u32 {
                    // Escape code: read raw 12-bit magnitude
                    reader.read_bits(12)
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
        for i in 0..coefficients.len() {
            let v = (i % 256) as i32;
            coefficients[i] = if v < 128 { 0 } else { (v - 128) * if i % 3 == 0 { -1 } else { 1 } };
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
        for i in 0..coefficients.len() {
            let r = (i * 7 + 13) % 256;
            if r < 40 {
                coefficients[i] = (r as i32 % 10 + 1) * if r % 2 == 0 { 1 } else { -1 };
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
}
