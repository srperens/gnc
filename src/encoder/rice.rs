//! Significance map + Golomb-Rice entropy coding.
//!
//! Fully parallel alternative to rANS: every coefficient encodes independently.
//! Uses 256 interleaved streams per tile (vs rANS's 32) for maximum GPU parallelism.
//!
//! Per coefficient:
//!   - Zero: 1 bit (0)
//!   - Non-zero: 1 bit (1) + 1 sign bit + Rice(|val|-1, k)
//!
//! Rice parameter `k` is chosen per subband group to match the geometric distribution.

use super::rans::compute_subband_group;

/// Number of interleaved streams per tile — 8x more than rANS for better GPU utilization.
pub const RICE_STREAMS_PER_TILE: usize = 256;

/// Max output bytes per stream (generous; typically much less).
pub const RICE_MAX_STREAM_BYTES: usize = 4096;

/// Encoded tile using significance map + Golomb-Rice coding with zero-run-length.
#[derive(Debug, Clone)]
pub struct RiceTile {
    pub num_coefficients: u32,
    pub tile_size: u32,
    pub num_levels: u32,
    pub num_groups: u32,
    /// Per-subband Rice parameter k for magnitudes (0..15).
    pub k_values: Vec<u8>,
    /// Per-subband Rice parameter k for zero-run lengths (0..15).
    pub k_zrl_values: Vec<u8>,
    /// Skip bitmap: bit g = 1 means all coefficients in group g are zero.
    pub skip_bitmap: u8,
    /// Bytes used by each of the 256 streams.
    pub stream_lengths: Vec<u32>,
    /// Concatenated stream data (all 256 streams).
    pub stream_data: Vec<u8>,
}

impl RiceTile {
    pub fn byte_size(&self) -> usize {
        let fixed_header = 4 + 4 + 4 + 4; // num_coefficients, tile_size, num_levels, num_groups
        let flags = 1; // flags byte

        // Check if all-skip
        let all_empty = self.stream_data.is_empty()
            && self.stream_lengths.iter().all(|&l| l == 0);
        let ng = self.num_groups.min(8) as u32;
        let all_mask = if ng >= 8 { 0xFFu8 } else { (1u8 << ng) - 1 };
        let all_skip = all_empty && (self.skip_bitmap & all_mask == all_mask);

        if all_skip {
            // flags + skip_bitmap
            fixed_header + flags + 1
        } else {
            let k_params = self.k_values.len() + self.k_zrl_values.len() + 1;
            let stream_len_bytes: usize = self.stream_lengths.iter()
                .map(|&l| varint_size(l as u16))
                .sum();
            fixed_header + flags + k_params + stream_len_bytes + self.stream_data.len()
        }
    }
}

/// Helper: write bits into a byte buffer (MSB-first packing).
struct BitWriter {
    buf: Vec<u8>,
    current_byte: u8,
    bits_in_byte: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(RICE_MAX_STREAM_BYTES),
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

    /// Write Golomb-Rice code for `value` with parameter `k`.
    #[inline]
    fn write_rice(&mut self, value: u32, k: u8) {
        let quotient = value >> k;
        let remainder = value & ((1u32 << k) - 1);

        // Unary: quotient 1-bits + one 0-bit.
        for _ in 0..quotient {
            self.write_bit(1);
        }
        self.write_bit(0);

        // Fixed-length remainder: k bits
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

/// Helper: read bits from a byte buffer (MSB-first packing).
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

    /// Read a Golomb-Rice coded value with parameter `k`.
    #[inline]
    fn read_rice(&mut self, k: u8) -> u32 {
        // Unary: count 1-bits until 0-bit
        let mut quotient = 0u32;
        while self.read_bit() == 1 {
            quotient += 1;
        }
        let remainder = if k > 0 { self.read_bits(k) } else { 0 };
        (quotient << k) | remainder
    }
}

/// Compute optimal Rice parameter k for a set of absolute values.
/// k = floor(log2(mean)) when mean > 0, else 0.
fn optimal_k(values: &[u32]) -> u8 {
    if values.is_empty() {
        return 0;
    }
    let sum: u64 = values.iter().map(|&v| v as u64).sum();
    let mean = sum / values.len() as u64;
    if mean == 0 {
        return 0;
    }
    // floor(log2(mean)), clamped to 0..15
    (63 - mean.leading_zeros()).min(15) as u8
}

/// Encode a tile of quantized coefficients using Golomb-Rice with zero-run-length.
///
/// Token encoding (per stream):
///   bit=1 → non-zero coefficient: sign bit + Rice(|val|-1, k_mag)
///   bit=0 → zero run: Rice(run_length-1, k_zrl)
///
/// Uses 256 interleaved streams for maximum GPU parallelism.
pub fn rice_encode_tile(coefficients: &[i32], tile_size: u32, num_levels: u32) -> RiceTile {
    let num_coefficients = coefficients.len();
    let num_groups = (num_levels * 2).max(1) as usize;
    let symbols_per_stream = num_coefficients / RICE_STREAMS_PER_TILE;

    // Phase 1: Compute per-subband k for magnitudes
    let mut group_abs_values: Vec<Vec<u32>> = vec![Vec::new(); num_groups];
    for (idx, &coeff) in coefficients.iter().enumerate() {
        let y = (idx / tile_size as usize) as u32;
        let x = (idx % tile_size as usize) as u32;
        let g = compute_subband_group(x, y, tile_size, num_levels);
        if coeff != 0 {
            group_abs_values[g].push(coeff.unsigned_abs() - 1);
        }
    }
    let k_values: Vec<u8> = group_abs_values.iter().map(|v| optimal_k(v)).collect();

    // Phase 1b: Compute per-subband k_zrl from zero-run statistics
    let mut group_run_lengths: Vec<Vec<u32>> = vec![Vec::new(); num_groups];
    for stream_id in 0..RICE_STREAMS_PER_TILE {
        let mut run = 0u32;
        let mut zrl_group = 0usize;
        for s in 0..symbols_per_stream {
            let coeff_idx = stream_id + s * RICE_STREAMS_PER_TILE;
            if coefficients[coeff_idx] == 0 {
                if run == 0 {
                    // First zero in run — record its subband
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
    let k_zrl_values: Vec<u8> = group_run_lengths.iter().map(|v| optimal_k(v)).collect();

    // Compute skip bitmap: bit g = 1 means all coefficients in group g are zero
    let mut skip_bitmap: u8 = 0;
    for g in 0..num_groups {
        if group_abs_values[g].is_empty() {
            skip_bitmap |= 1 << g;
        }
    }

    // Phase 2: Encode 256 interleaved streams with ZRL
    let mut all_stream_data = Vec::new();
    let mut stream_lengths = Vec::with_capacity(RICE_STREAMS_PER_TILE);

    for stream_id in 0..RICE_STREAMS_PER_TILE {
        let mut writer = BitWriter::new();
        let mut s = 0;

        // Initialize EMA from static k seeds (per-stream, 8 groups)
        let mut ema = [0u32; 8];
        for g in 0..num_groups {
            ema[g] = (1u32 << k_values[g]).max(1) << 4;
        }

        while s < symbols_per_stream {
            let coeff_idx = stream_id + s * RICE_STREAMS_PER_TILE;
            let y = (coeff_idx / tile_size as usize) as u32;
            let x = (coeff_idx % tile_size as usize) as u32;
            let g = compute_subband_group(x, y, tile_size, num_levels);

            // Skip bitmap: if this position's group is fully zero, advance without encoding
            if (skip_bitmap >> g) & 1 == 1 {
                s += 1;
                continue;
            }

            let coeff = coefficients[coeff_idx];

            if coeff == 0 {
                // Count zero run (only count non-skipped positions)
                let k = k_zrl_values[g];
                let mut run = 1u32;
                let mut ns = s + 1;
                while ns < symbols_per_stream {
                    let next_idx = stream_id + ns * RICE_STREAMS_PER_TILE;
                    let ny = (next_idx / tile_size as usize) as u32;
                    let nx = (next_idx % tile_size as usize) as u32;
                    let ng = compute_subband_group(nx, ny, tile_size, num_levels);
                    // Skip past bitmap-skipped positions
                    if (skip_bitmap >> ng) & 1 == 1 {
                        ns += 1;
                        continue;
                    }
                    if coefficients[next_idx] != 0 {
                        break;
                    }
                    run += 1;
                    ns += 1;
                }
                writer.write_bit(0); // token: zero run
                writer.write_rice(run - 1, k);
                s = ns;
            } else {
                writer.write_bit(1); // token: non-zero
                writer.write_bit(if coeff < 0 { 1 } else { 0 });
                let magnitude = coeff.unsigned_abs() - 1;

                // Derive adaptive k from EMA
                let ema_mean = ema[g] >> 4;
                let k = if ema_mean > 0 {
                    (31 - ema_mean.leading_zeros()).min(15) as u8
                } else {
                    0
                };
                writer.write_rice(magnitude, k);

                // Update EMA
                ema[g] = ema[g] - (ema[g] >> 3) + (magnitude << 1);
                s += 1;
            }
        }

        let stream_bytes = writer.flush();
        stream_lengths.push(stream_bytes.len() as u32);
        all_stream_data.extend_from_slice(&stream_bytes);
    }

    RiceTile {
        num_coefficients: num_coefficients as u32,
        tile_size,
        num_levels,
        num_groups: num_groups as u32,
        k_values,
        k_zrl_values,
        skip_bitmap,
        stream_lengths,
        stream_data: all_stream_data,
    }
}

/// Decode a Rice-coded tile (with ZRL) back to quantized coefficients.
pub fn rice_decode_tile(tile: &RiceTile) -> Vec<i32> {
    let num_coefficients = tile.num_coefficients as usize;
    let symbols_per_stream = num_coefficients / RICE_STREAMS_PER_TILE;
    let mut coefficients = vec![0i32; num_coefficients];

    let num_groups = tile.num_groups as usize;
    let mut data_offset = 0usize;
    for stream_id in 0..RICE_STREAMS_PER_TILE {
        let stream_len = tile.stream_lengths[stream_id] as usize;
        let stream_data = &tile.stream_data[data_offset..data_offset + stream_len];
        let mut reader = BitReader::new(stream_data);

        // Initialize EMA from static k seeds (per-stream, 8 groups)
        let mut ema = [0u32; 8];
        for g in 0..num_groups {
            ema[g] = (1u32 << tile.k_values[g]).max(1) << 4;
        }

        let mut s = 0usize;
        let skip_bitmap = tile.skip_bitmap;
        while s < symbols_per_stream {
            let coeff_idx = stream_id + s * RICE_STREAMS_PER_TILE;
            let cy = (coeff_idx / tile.tile_size as usize) as u32;
            let cx = (coeff_idx % tile.tile_size as usize) as u32;
            let cur_g = compute_subband_group(cx, cy, tile.tile_size, tile.num_levels);

            // Skip bitmap: if this position's group is fully zero, write 0 and advance
            if (skip_bitmap >> cur_g) & 1 == 1 {
                coefficients[coeff_idx] = 0;
                s += 1;
                continue;
            }

            let token = reader.read_bit();
            if token == 0 {
                // Zero run
                let run = reader.read_rice(tile.k_zrl_values[cur_g]) + 1;
                // Write zeros, skipping past bitmap-skipped positions
                let mut written = 0u32;
                let mut ws = s;
                while written < run && ws < symbols_per_stream {
                    let wi = stream_id + ws * RICE_STREAMS_PER_TILE;
                    let wy = (wi / tile.tile_size as usize) as u32;
                    let wx = (wi % tile.tile_size as usize) as u32;
                    let wg = compute_subband_group(wx, wy, tile.tile_size, tile.num_levels);
                    if (skip_bitmap >> wg) & 1 == 1 {
                        // Bitmap-skipped: write zero but don't count toward run
                        coefficients[wi] = 0;
                        ws += 1;
                        continue;
                    }
                    coefficients[wi] = 0;
                    written += 1;
                    ws += 1;
                }
                s = ws;
            } else {
                // Non-zero coefficient
                let sign = reader.read_bit();
                let g = cur_g;

                // Derive adaptive k from EMA
                let ema_mean = ema[g] >> 4;
                let k = if ema_mean > 0 {
                    (31 - ema_mean.leading_zeros()).min(15) as u8
                } else {
                    0
                };
                let rice_val = reader.read_rice(k);
                let magnitude = rice_val + 1;

                coefficients[coeff_idx] = if sign == 1 {
                    -(magnitude as i32)
                } else {
                    magnitude as i32
                };

                // Update EMA
                ema[g] = ema[g] - (ema[g] >> 3) + (rice_val << 1);
                s += 1;
            }
        }

        data_offset += stream_len;
    }

    coefficients
}

/// Serialize a RiceTile to bytes.
/// Tile format flags byte (after fixed 16-byte header).
const TILE_FLAG_COMPACT_STREAMS: u8 = 0x01; // varint-encoded stream lengths
const TILE_FLAG_ALL_SKIP: u8 = 0x02; // all subbands zero, no stream data

/// Write unsigned varint (u16 range: max 3 bytes).
fn write_tile_varint(out: &mut Vec<u8>, val: u16) {
    let mut v = val as u32;
    while v >= 0x80 {
        out.push((v & 0x7F) as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Read unsigned varint → u16.
fn read_tile_varint(data: &[u8], pos: &mut usize) -> u16 {
    let mut result = 0u32;
    let mut shift = 0;
    loop {
        let b = data[*pos] as u32;
        *pos += 1;
        result |= (b & 0x7F) << shift;
        if b < 0x80 {
            break;
        }
        shift += 7;
    }
    result as u16
}

/// Compute varint byte size for a u16 value.
fn varint_size(val: u16) -> usize {
    if val < 0x80 { 1 } else if val < 0x4000 { 2 } else { 3 }
}

pub fn serialize_tile_rice(tile: &RiceTile) -> Vec<u8> {
    let mut out = Vec::with_capacity(tile.byte_size());
    // Fixed header (16 bytes)
    out.extend_from_slice(&tile.num_coefficients.to_le_bytes());
    out.extend_from_slice(&tile.tile_size.to_le_bytes());
    out.extend_from_slice(&tile.num_levels.to_le_bytes());
    out.extend_from_slice(&tile.num_groups.to_le_bytes());

    // Check if tile is all-skip (all streams empty)
    let all_empty = tile.stream_data.is_empty()
        && tile.stream_lengths.iter().all(|&l| l == 0);
    let ng = tile.num_groups.min(8) as u32;
    let all_mask = if ng >= 8 { 0xFFu8 } else { (1u8 << ng) - 1 };
    let all_skip = all_empty && (tile.skip_bitmap & all_mask == all_mask);

    if all_skip {
        // Compact all-skip: flags + skip_bitmap only (2 bytes)
        out.push(TILE_FLAG_ALL_SKIP | TILE_FLAG_COMPACT_STREAMS);
        out.push(tile.skip_bitmap);
    } else {
        // Flags byte: always use varint stream lengths
        out.push(TILE_FLAG_COMPACT_STREAMS);

        // k values + k_zrl values + skip_bitmap
        out.extend_from_slice(&tile.k_values);
        out.extend_from_slice(&tile.k_zrl_values);
        out.push(tile.skip_bitmap);

        // Varint stream lengths (256 entries, 1-3 bytes each)
        for &len in &tile.stream_lengths {
            write_tile_varint(&mut out, len as u16);
        }

        // Stream data
        out.extend_from_slice(&tile.stream_data);
    }
    out
}

/// Deserialize a RiceTile from bytes. Returns (tile, bytes_consumed).
pub fn deserialize_tile_rice(data: &[u8]) -> (RiceTile, usize) {
    let mut pos = 0;

    // Fixed header (16 bytes)
    let num_coefficients = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let tile_size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let num_levels = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let num_groups = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    // Flags byte
    let flags = data[pos];
    pos += 1;
    let compact_streams = flags & TILE_FLAG_COMPACT_STREAMS != 0;
    let all_skip = flags & TILE_FLAG_ALL_SKIP != 0;

    if all_skip {
        // All-skip tile: just skip_bitmap, everything else is zero
        let skip_bitmap = data[pos];
        pos += 1;
        return (
            RiceTile {
                num_coefficients,
                tile_size,
                num_levels,
                num_groups,
                k_values: vec![0; num_groups as usize],
                k_zrl_values: vec![0; num_groups as usize],
                skip_bitmap,
                stream_lengths: vec![0; RICE_STREAMS_PER_TILE],
                stream_data: Vec::new(),
            },
            pos,
        );
    }

    // k_values + k_zrl_values + skip_bitmap
    let k_values = data[pos..pos + num_groups as usize].to_vec();
    pos += num_groups as usize;
    let k_zrl_values = data[pos..pos + num_groups as usize].to_vec();
    pos += num_groups as usize;
    let skip_bitmap = data[pos];
    pos += 1;

    // Stream lengths
    let mut stream_lengths = vec![0u32; RICE_STREAMS_PER_TILE];
    if compact_streams {
        // Varint stream lengths (256 entries)
        for i in 0..RICE_STREAMS_PER_TILE {
            stream_lengths[i] = read_tile_varint(data, &mut pos) as u32;
        }
    } else {
        // Legacy: fixed 256 × u16
        for i in 0..RICE_STREAMS_PER_TILE {
            stream_lengths[i] =
                u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as u32;
            pos += 2;
        }
    }

    let total_data: usize = stream_lengths.iter().map(|&l| l as usize).sum();
    let stream_data = data[pos..pos + total_data].to_vec();
    pos += total_data;

    (
        RiceTile {
            num_coefficients,
            tile_size,
            num_levels,
            num_groups,
            k_values,
            k_zrl_values,
            skip_bitmap,
            stream_lengths,
            stream_data,
        },
        pos,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rice_roundtrip_zeros() {
        let coefficients = vec![0i32; 65536];
        let tile = rice_encode_tile(&coefficients, 256, 3);
        let decoded = rice_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_rice_roundtrip_simple() {
        let mut coefficients = vec![0i32; 65536];
        // LL subband: larger values
        for y in 0..32 {
            for x in 0..32 {
                coefficients[y * 256 + x] = ((x + y) % 20) as i32 + 5;
            }
        }
        // Scatter some detail coefficients
        for i in (8192..65536).step_by(7) {
            coefficients[i] = (i % 11) as i32 - 5;
        }

        let tile = rice_encode_tile(&coefficients, 256, 3);
        assert_eq!(tile.num_groups, 6);
        let decoded = rice_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_rice_roundtrip_varied() {
        let mut coefficients = vec![0i32; 65536];
        for i in 0..65536 {
            let y = i / 256;
            let x = i % 256;
            let g = compute_subband_group(x as u32, y as u32, 256, 3);
            coefficients[i] = match g {
                0 => ((x + y) % 40) as i32 + 10,
                1 => {
                    if i % 5 == 0 {
                        (i % 7) as i32 - 3
                    } else {
                        0
                    }
                }
                _ => {
                    if i % 3 == 0 {
                        (i % 9) as i32 - 4
                    } else {
                        0
                    }
                }
            };
        }

        let tile = rice_encode_tile(&coefficients, 256, 3);
        let decoded = rice_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_rice_serialize_roundtrip() {
        let mut coefficients = vec![0i32; 65536];
        for i in (0..65536).step_by(3) {
            coefficients[i] = (i % 15) as i32 - 7;
        }

        let tile = rice_encode_tile(&coefficients, 256, 3);
        let serialized = serialize_tile_rice(&tile);
        let (deserialized, consumed) = deserialize_tile_rice(&serialized);
        assert_eq!(consumed, serialized.len());
        assert_eq!(tile.k_values, deserialized.k_values);
        assert_eq!(tile.stream_lengths, deserialized.stream_lengths);
        assert_eq!(tile.stream_data, deserialized.stream_data);

        let decoded = rice_decode_tile(&deserialized);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_rice_negative_values() {
        let mut coefficients = vec![0i32; 65536];
        for i in 0..65536 {
            coefficients[i] = -((i % 10) as i32);
        }

        let tile = rice_encode_tile(&coefficients, 256, 3);
        let decoded = rice_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_optimal_k() {
        assert_eq!(optimal_k(&[]), 0);
        assert_eq!(optimal_k(&[0, 0, 0]), 0);
        assert_eq!(optimal_k(&[1, 1, 1]), 0); // mean=1, log2(1)=0
        assert_eq!(optimal_k(&[2, 2, 2]), 1); // mean=2, log2(2)=1
        assert_eq!(optimal_k(&[7, 8, 9]), 3); // mean=8, log2(8)=3
    }

    #[test]
    fn test_rice_compression_vs_raw() {
        // Typical wavelet output: mostly zeros, small non-zero values
        let mut coefficients = vec![0i32; 65536];
        for i in (0..65536).step_by(5) {
            coefficients[i] = (i % 7) as i32 - 3;
        }

        let tile = rice_encode_tile(&coefficients, 256, 3);
        let raw_size = 65536 * 4; // i32 per coeff
        let rice_size = tile.byte_size();
        // Rice should be much smaller than raw
        assert!(
            rice_size < raw_size / 2,
            "Rice size {rice_size} should be much smaller than raw {raw_size}"
        );
    }
}
