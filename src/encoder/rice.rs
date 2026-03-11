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

/// Max output bytes per Rice stream for a given tile size and quantisation step.
///
/// Adapts the staging buffer to actual data volume, reducing memcpy cost at higher quality
/// settings where most coefficients are small.
///
/// Formula: estimate ~bpp based on qstep, convert to bytes/stream with 3× safety margin,
/// round up to next power-of-two, then clamp to [256, size_cap] where size_cap is the
/// tile-size-dependent ceiling (1024 for ≤128 px, 4096 for 256 px).
///
/// At q=75 (qstep≈4.0)  → 1024 bytes/stream vs. old fixed 4096 (4× saving)
/// At q=25 (qstep≈16.0) → 256  bytes/stream vs. old fixed 4096 (16× saving)
/// At q=95 (qstep≈2.0)  → 1024 bytes/stream vs. old fixed 4096 (4× saving)
/// At qstep=1.0          → 2048 bytes/stream vs. old fixed 4096 (2× saving)
///
/// Must be divisible by 4 (GPU stores u32 words) and kept in sync with the shader.
pub fn max_stream_bytes_for_tile(tile_size: u32, qstep: f32) -> usize {
    // Tile-size-dependent hard cap (preserves existing ≤128 px behaviour).
    let size_cap: usize = match tile_size {
        t if t <= 128 => 1024,
        _ => 4096,
    };

    // Rough bpp model: high quality (qstep≈1) → ~15 bpp; low quality (qstep≈16) → ~1.5 bpp.
    let qstep_clamped = qstep.clamp(0.5, 64.0);
    let bpp_estimate = 15.0_f32 * (1.0 / (1.0 + 0.3 * qstep_clamped));
    // Bytes per stream = bpp × tile_px / 8 × 3× safety margin, divided by 256 streams.
    // (tile_size × tile_size pixels, 256 streams → tile_size²/256 coefficients per stream)
    let tile_px = tile_size as f32;
    let bytes_per_stream = (bpp_estimate * (tile_px * tile_px) / 8.0 * 3.0 / 256.0) as usize;

    // Round up to next power-of-two, with a floor of 256 and the tile-size cap.
    bytes_per_stream.next_power_of_two().max(256).min(size_cap)
}

/// Max output bytes per stream (generous; typically much less).
/// Kept for use by the CPU-only encode path (rice_encode_tile).
pub const RICE_MAX_STREAM_BYTES: usize = 4096;

/// Encoded tile using significance map + Golomb-Rice coding with zero-run-length.
///
/// #53: k_zrl is split into two context-dependent parameters:
///   k_zrl_nz — used for zero runs that follow a nonzero coefficient
///   k_zrl_z  — used for zero runs that follow another zero run or start-of-stream
///
/// #checkerboard-ctx: k_stream_odd holds the EMA warm-start k for the 128 odd streams
/// (streams 1, 3, 5, ..., 255). Indexed by odd_idx = (stream_id - 1) / 2.
/// Absent in legacy tiles (empty Vec → treat as global k).
#[derive(Debug, Clone)]
pub struct RiceTile {
    pub num_coefficients: u32,
    pub tile_size: u32,
    pub num_levels: u32,
    pub num_groups: u32,
    /// Per-subband Rice parameter k for magnitudes (0..15).
    pub k_values: Vec<u8>,
    /// Per-subband Rice k for zero runs following a nonzero coefficient (0..15).
    pub k_zrl_nz_values: Vec<u8>,
    /// Per-subband Rice k for zero runs following another zero run / start-of-stream (0..15).
    pub k_zrl_z_values: Vec<u8>,
    /// Skip bitmap: bit g = 1 means all coefficients in group g are zero.
    pub skip_bitmap: u8,
    /// Checkerboard-context EMA warm-start k for odd streams (128 entries).
    /// Entry i → stream 2i+1. Empty when checkerboard context is not used.
    pub k_stream_odd: Vec<u8>,
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
        let ng = self.num_groups.min(8);
        let all_mask = if ng >= 8 { 0xFFu8 } else { (1u8 << ng) - 1 };
        let all_skip = all_empty && (self.skip_bitmap & all_mask == all_mask);

        if all_skip {
            // flags + skip_bitmap
            fixed_header + flags + 1
        } else {
            let k_params = self.k_values.len() + self.k_zrl_nz_values.len() + self.k_zrl_z_values.len() + 1;
            let stream_len_bytes: usize = self.stream_lengths.iter()
                .map(|&l| varint_size(l as u16))
                .sum();
            // checkerboard ctx: 1024 bytes for odd-stream initial k (if present)
            let ck_bytes = self.k_stream_odd.len(); // 0 or 1024
            fixed_header + flags + k_params + ck_bytes + stream_len_bytes + self.stream_data.len()
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

    // Phase 1b: Compute 2-state k_zrl from zero-run statistics (#53).
    // Context: was the preceding nonzero "large" (|coeff|>=2, i.e., magnitude>=1)?
    //   k_zrl_nz — runs after large nonzero (clustered signal → shorter runs expected)
    //   k_zrl_z  — runs after small nonzero (|coeff|==1) or start-of-stream (longer runs)
    let mut group_run_lengths_nz: Vec<Vec<u32>> = vec![Vec::new(); num_groups];
    let mut group_run_lengths_z: Vec<Vec<u32>> = vec![Vec::new(); num_groups];
    for stream_id in 0..RICE_STREAMS_PER_TILE {
        let mut run = 0u32;
        let mut zrl_group = 0usize;
        let mut zrl_ctx_large = false; // context when this zero run started
        let mut last_mag_large = false; // true if preceding nonzero had |coeff|>=2
        for s in 0..symbols_per_stream {
            let coeff_idx = stream_id + s * RICE_STREAMS_PER_TILE;
            if coefficients[coeff_idx] == 0 {
                if run == 0 {
                    let y = (coeff_idx / tile_size as usize) as u32;
                    let x = (coeff_idx % tile_size as usize) as u32;
                    zrl_group = compute_subband_group(x, y, tile_size, num_levels);
                    zrl_ctx_large = last_mag_large;
                }
                run += 1;
            } else {
                if run > 0 {
                    if zrl_ctx_large {
                        group_run_lengths_nz[zrl_group].push(run - 1);
                    } else {
                        group_run_lengths_z[zrl_group].push(run - 1);
                    }
                    run = 0;
                }
                // |coeff|>=2 ↔ unsigned_abs()-1 >= 1
                last_mag_large = coefficients[coeff_idx].unsigned_abs() >= 2;
            }
        }
        if run > 0 {
            if zrl_ctx_large {
                group_run_lengths_nz[zrl_group].push(run - 1);
            } else {
                group_run_lengths_z[zrl_group].push(run - 1);
            }
        }
    }
    let k_zrl_nz_values: Vec<u8> = group_run_lengths_nz.iter().map(|v| optimal_k(v)).collect();
    let k_zrl_z_values: Vec<u8> = group_run_lengths_z.iter().map(|v| optimal_k(v)).collect();

    // Compute skip bitmap: bit g = 1 means all coefficients in group g are zero
    let mut skip_bitmap: u8 = 0;
    for (g, group) in group_abs_values.iter().enumerate().take(num_groups) {
        if group.is_empty() {
            skip_bitmap |= 1 << g;
        }
    }

    // Phase 2: Encode 256 interleaved streams with ZRL.
    // Checkerboard k-context: even streams encode first, recording their final EMA means.
    // Odd streams are then warm-started with a 50/50 blend of global k and neighbor k.
    let mut all_stream_data = Vec::new();
    let mut stream_lengths = Vec::with_capacity(RICE_STREAMS_PER_TILE);

    // Even streams encode first; store their final EMA means for odd-stream warm-start.
    let mut even_final_ema = [[0u32; 8]; 128]; // indexed by stream_id / 2
    let mut even_stream_data: Vec<Vec<u8>> = vec![Vec::new(); 128];
    let mut even_stream_lengths: Vec<u32> = vec![0; 128];

    // --- Step 2a: even streams ---
    for even_id in (0..RICE_STREAMS_PER_TILE).step_by(2) {
        let stream_id = even_id;
        let mut writer = BitWriter::new();
        let mut s = 0;

        let mut ema = [0u32; 8];
        for g in 0..num_groups {
            ema[g] = (1u32 << k_values[g]).max(1) << 4;
        }

        let mut last_mag_large = false;
        while s < symbols_per_stream {
            let coeff_idx = stream_id + s * RICE_STREAMS_PER_TILE;
            let y = (coeff_idx / tile_size as usize) as u32;
            let x = (coeff_idx % tile_size as usize) as u32;
            let g = compute_subband_group(x, y, tile_size, num_levels);

            if (skip_bitmap >> g) & 1 == 1 {
                s += 1;
                continue;
            }

            let coeff = coefficients[coeff_idx];
            if coeff == 0 {
                let k = if last_mag_large { k_zrl_nz_values[g] } else { k_zrl_z_values[g] };
                let mut run = 1u32;
                let mut ns = s + 1;
                while ns < symbols_per_stream {
                    let next_idx = stream_id + ns * RICE_STREAMS_PER_TILE;
                    let ny = (next_idx / tile_size as usize) as u32;
                    let nx = (next_idx % tile_size as usize) as u32;
                    let ng = compute_subband_group(nx, ny, tile_size, num_levels);
                    if (skip_bitmap >> ng) & 1 == 1 { ns += 1; continue; }
                    if coefficients[next_idx] != 0 { break; }
                    run += 1;
                    ns += 1;
                }
                writer.write_bit(0);
                writer.write_rice(run - 1, k);
                s = ns;
                last_mag_large = false;
            } else {
                writer.write_bit(1);
                writer.write_bit(if coeff < 0 { 1 } else { 0 });
                let magnitude = coeff.unsigned_abs() - 1;
                let ema_mean = ema[g] >> 4;
                let k = if ema_mean > 0 { (31 - ema_mean.leading_zeros()).min(15) as u8 } else { 0 };
                writer.write_rice(magnitude, k);
                ema[g] = ema[g] - (ema[g] >> 3) + (magnitude << 1);
                s += 1;
                last_mag_large = magnitude >= 1;
            }
        }
        let even_idx = stream_id / 2;
        even_final_ema[even_idx] = ema;
        let bytes = writer.flush();
        even_stream_lengths[even_idx] = bytes.len() as u32;
        even_stream_data[even_idx] = bytes;
    }

    // Compute adjusted k for odd streams (50/50 blend of global k and neighbor k).
    // Layout: k_stream_odd[odd_idx * 8 + g] = blended k for odd stream (2*odd_idx+1), group g.
    // Always stores 128 * 8 = 1024 bytes; unused group slots (g >= num_groups) are 0.
    let mut k_stream_odd = vec![0u8; 128 * 8];
    for odd_idx in 0..128usize {
        let even_idx = odd_idx; // neighbor even stream index
        for g in 0..8usize {
            let global_k = if g < num_groups { k_values[g] as u32 } else { 0 };
            let neighbor_mean = even_final_ema[even_idx][g] >> 4;
            let neighbor_k = if neighbor_mean > 0 {
                (31 - neighbor_mean.leading_zeros()).min(15)
            } else {
                0
            };
            let adjusted = (global_k + neighbor_k).div_ceil(2).min(15);
            k_stream_odd[odd_idx * 8 + g] = adjusted as u8;
        }
    }

    // --- Step 2b: odd streams with per-group neighbor-context warm-start ---
    let mut odd_stream_data: Vec<Vec<u8>> = vec![Vec::new(); 128];
    let mut odd_stream_lengths: Vec<u32> = vec![0; 128];
    for odd_rank in 0..128usize {
        let stream_id = 2 * odd_rank + 1;
        let even_idx = odd_rank; // neighbor even stream
        let mut writer = BitWriter::new();
        let mut s = 0;

        // Per-group blend of global k and neighbor's final EMA k
        let mut ema = [0u32; 8];
        for g in 0..num_groups {
            let neighbor_mean = even_final_ema[even_idx][g] >> 4;
            let neighbor_k = if neighbor_mean > 0 {
                (31 - neighbor_mean.leading_zeros()).min(15)
            } else {
                0
            };
            let global_k = k_values[g] as u32;
            let adjusted_k = (global_k + neighbor_k).div_ceil(2).min(15);
            ema[g] = (1u32 << adjusted_k).max(1) << 4;
        }

        let mut last_mag_large = false;
        while s < symbols_per_stream {
            let coeff_idx = stream_id + s * RICE_STREAMS_PER_TILE;
            let y = (coeff_idx / tile_size as usize) as u32;
            let x = (coeff_idx % tile_size as usize) as u32;
            let g = compute_subband_group(x, y, tile_size, num_levels);

            if (skip_bitmap >> g) & 1 == 1 {
                s += 1;
                continue;
            }

            let coeff = coefficients[coeff_idx];
            if coeff == 0 {
                let k = if last_mag_large { k_zrl_nz_values[g] } else { k_zrl_z_values[g] };
                let mut run = 1u32;
                let mut ns = s + 1;
                while ns < symbols_per_stream {
                    let next_idx = stream_id + ns * RICE_STREAMS_PER_TILE;
                    let ny = (next_idx / tile_size as usize) as u32;
                    let nx = (next_idx % tile_size as usize) as u32;
                    let ng = compute_subband_group(nx, ny, tile_size, num_levels);
                    if (skip_bitmap >> ng) & 1 == 1 { ns += 1; continue; }
                    if coefficients[next_idx] != 0 { break; }
                    run += 1;
                    ns += 1;
                }
                writer.write_bit(0);
                writer.write_rice(run - 1, k);
                s = ns;
                last_mag_large = false;
            } else {
                writer.write_bit(1);
                writer.write_bit(if coeff < 0 { 1 } else { 0 });
                let magnitude = coeff.unsigned_abs() - 1;
                let ema_mean = ema[g] >> 4;
                let k = if ema_mean > 0 { (31 - ema_mean.leading_zeros()).min(15) as u8 } else { 0 };
                writer.write_rice(magnitude, k);
                ema[g] = ema[g] - (ema[g] >> 3) + (magnitude << 1);
                s += 1;
                last_mag_large = magnitude >= 1;
            }
        }
        let bytes = writer.flush();
        odd_stream_lengths[odd_rank] = bytes.len() as u32;
        odd_stream_data[odd_rank] = bytes;
    }

    // Interleave even and odd streams back into the canonical 0..255 order
    for stream_id in 0..RICE_STREAMS_PER_TILE {
        let (data, len) = if stream_id % 2 == 0 {
            let idx = stream_id / 2;
            (&even_stream_data[idx], even_stream_lengths[idx])
        } else {
            let idx = stream_id / 2;
            (&odd_stream_data[idx], odd_stream_lengths[idx])
        };
        stream_lengths.push(len);
        all_stream_data.extend_from_slice(data);
    }

    RiceTile {
        num_coefficients: num_coefficients as u32,
        tile_size,
        num_levels,
        num_groups: num_groups as u32,
        k_values,
        k_zrl_nz_values,
        k_zrl_z_values,
        skip_bitmap,
        k_stream_odd,
        stream_lengths,
        stream_data: all_stream_data,
    }
}

/// Decode a Rice-coded tile (with ZRL) back to quantized coefficients.
///
/// Uses two-pass checkerboard context:
/// - Pass 1: decode even streams (0,2,...,254), collect final EMA states.
/// - Derive adjusted_k per group for odd streams from even-stream neighbor EMA
///   (same 50/50 blend formula as encoder). If k_stream_odd is non-empty (CPU-encoded
///   tiles), those pre-computed values are used instead of re-deriving.
/// - Pass 2: decode odd streams (1,3,...,255) with adjusted initial EMA.
pub fn rice_decode_tile(tile: &RiceTile) -> Vec<i32> {
    let num_coefficients = tile.num_coefficients as usize;
    let symbols_per_stream = num_coefficients / RICE_STREAMS_PER_TILE;
    let mut coefficients = vec![0i32; num_coefficients];
    let num_groups = tile.num_groups as usize;
    let skip_bitmap = tile.skip_bitmap;
    let tile_size = tile.tile_size as usize;

    // Pre-compute stream byte offsets for random access
    let mut stream_offsets = vec![0usize; RICE_STREAMS_PER_TILE + 1];
    for s in 0..RICE_STREAMS_PER_TILE {
        stream_offsets[s + 1] = stream_offsets[s] + tile.stream_lengths[s] as usize;
    }

    // Decode one stream into coefficients[], return final EMA state.
    let decode_stream = |stream_id: usize, initial_ema: [u32; 8], coefficients: &mut Vec<i32>| -> [u32; 8] {
        let start = stream_offsets[stream_id];
        let end = stream_offsets[stream_id + 1];
        let stream_data = &tile.stream_data[start..end];
        let mut reader = BitReader::new(stream_data);
        let mut ema = initial_ema;
        let mut s = 0usize;
        let mut last_mag_large = false;
        while s < symbols_per_stream {
            let coeff_idx = stream_id + s * RICE_STREAMS_PER_TILE;
            let cy = (coeff_idx / tile_size) as u32;
            let cx = (coeff_idx % tile_size) as u32;
            let cur_g = compute_subband_group(cx, cy, tile.tile_size, tile.num_levels);
            if (skip_bitmap >> cur_g) & 1 == 1 {
                coefficients[coeff_idx] = 0;
                s += 1;
                continue;
            }
            let token = reader.read_bit();
            if token == 0 {
                let k = if last_mag_large {
                    tile.k_zrl_nz_values[cur_g]
                } else {
                    tile.k_zrl_z_values[cur_g]
                };
                let run = reader.read_rice(k) + 1;
                let mut written = 0u32;
                let mut ws = s;
                while written < run && ws < symbols_per_stream {
                    let wi = stream_id + ws * RICE_STREAMS_PER_TILE;
                    let wy = (wi / tile_size) as u32;
                    let wx = (wi % tile_size) as u32;
                    let wg = compute_subband_group(wx, wy, tile.tile_size, tile.num_levels);
                    if (skip_bitmap >> wg) & 1 == 1 {
                        coefficients[wi] = 0;
                        ws += 1;
                        continue;
                    }
                    coefficients[wi] = 0;
                    written += 1;
                    ws += 1;
                }
                s = ws;
                last_mag_large = false;
            } else {
                let sign = reader.read_bit();
                let g = cur_g;
                let ema_mean = ema[g] >> 4;
                let k = if ema_mean > 0 {
                    (31 - ema_mean.leading_zeros()).min(15) as u8
                } else {
                    0
                };
                let rice_val = reader.read_rice(k);
                let magnitude = rice_val + 1;
                coefficients[coeff_idx] = if sign == 1 { -(magnitude as i32) } else { magnitude as i32 };
                ema[g] = ema[g] - (ema[g] >> 3) + (rice_val << 1);
                s += 1;
                last_mag_large = rice_val >= 1;
            }
        }
        ema
    };

    // Pass 1: decode even streams, collect final EMAs
    let mut even_final_ema = [[0u32; 8]; 128];
    for (even_rank, ema_slot) in even_final_ema.iter_mut().enumerate() {
        let stream_id = even_rank * 2;
        let mut initial_ema = [0u32; 8];
        for (g, e) in initial_ema.iter_mut().enumerate().take(num_groups) {
            let k = tile.k_values[g] as u32;
            *e = (1u32 << k).max(1) << 4;
        }
        *ema_slot = decode_stream(stream_id, initial_ema, &mut coefficients);
    }

    // Derive adjusted_k for odd streams: 50/50 blend of global k and neighbor (even) k.
    // If k_stream_odd is present (CPU-encoded tiles), use pre-computed values directly.
    let mut adjusted_k_for_odd = [[0u32; 8]; 128];
    for (odd_rank, adj) in adjusted_k_for_odd.iter_mut().enumerate() {
        for g in 0..num_groups {
            if !tile.k_stream_odd.is_empty() {
                adj[g] = tile.k_stream_odd[odd_rank * 8 + g] as u32;
            } else {
                let global_k = tile.k_values[g] as u32;
                let neighbor_mean = even_final_ema[odd_rank][g] >> 4;
                let neighbor_k = if neighbor_mean > 0 {
                    (31 - neighbor_mean.leading_zeros()).min(15)
                } else {
                    0
                };
                adj[g] = (global_k + neighbor_k).div_ceil(2);
            }
        }
    }

    // Pass 2: decode odd streams with adjusted initial EMA
    for (odd_rank, adj) in adjusted_k_for_odd.iter().enumerate() {
        let stream_id = odd_rank * 2 + 1;
        let mut initial_ema = [0u32; 8];
        for (g, e) in initial_ema.iter_mut().enumerate().take(num_groups) {
            let k = adj[g];
            *e = (1u32 << k).max(1) << 4;
        }
        decode_stream(stream_id, initial_ema, &mut coefficients);
    }

    coefficients
}

/// Serialize a RiceTile to bytes.
/// Tile format flags byte (after fixed 16-byte header).
const TILE_FLAG_COMPACT_STREAMS: u8 = 0x01; // varint-encoded stream lengths
const TILE_FLAG_ALL_SKIP: u8 = 0x02; // all subbands zero, no stream data
const TILE_FLAG_CHECKERBOARD_K: u8 = 0x04; // 128-byte per-odd-stream EMA warm-start k block present

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
    let ng = tile.num_groups.min(8);
    let all_mask = if ng >= 8 { 0xFFu8 } else { (1u8 << ng) - 1 };
    let all_skip = all_empty && (tile.skip_bitmap & all_mask == all_mask);

    if all_skip {
        // Compact all-skip: flags + skip_bitmap only (2 bytes)
        out.push(TILE_FLAG_ALL_SKIP | TILE_FLAG_COMPACT_STREAMS);
        out.push(tile.skip_bitmap);
    } else {
        let has_ck = !tile.k_stream_odd.is_empty();
        // Flags byte: always use varint stream lengths; optionally checkerboard ctx
        let flags_byte = TILE_FLAG_COMPACT_STREAMS
            | if has_ck { TILE_FLAG_CHECKERBOARD_K } else { 0 };
        out.push(flags_byte);

        // k values + k_zrl_nz + k_zrl_z + skip_bitmap
        out.extend_from_slice(&tile.k_values);
        out.extend_from_slice(&tile.k_zrl_nz_values);
        out.extend_from_slice(&tile.k_zrl_z_values);
        out.push(tile.skip_bitmap);

        // Checkerboard per-odd-stream k (1024 bytes = 128 odd streams × 8 groups,
        // only when TILE_FLAG_CHECKERBOARD_K set)
        if has_ck {
            debug_assert_eq!(tile.k_stream_odd.len(), 128 * 8,
                "k_stream_odd must have exactly 1024 entries (128 odd streams × 8 groups)");
            out.extend_from_slice(&tile.k_stream_odd);
        }

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
    let has_ck = flags & TILE_FLAG_CHECKERBOARD_K != 0;

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
                k_zrl_nz_values: vec![0; num_groups as usize],
                k_zrl_z_values: vec![0; num_groups as usize],
                skip_bitmap,
                k_stream_odd: Vec::new(),
                stream_lengths: vec![0; RICE_STREAMS_PER_TILE],
                stream_data: Vec::new(),
            },
            pos,
        );
    }

    // k_values + k_zrl_nz_values + k_zrl_z_values + skip_bitmap
    let k_values = data[pos..pos + num_groups as usize].to_vec();
    pos += num_groups as usize;
    let k_zrl_nz_values = data[pos..pos + num_groups as usize].to_vec();
    pos += num_groups as usize;
    let k_zrl_z_values = data[pos..pos + num_groups as usize].to_vec();
    pos += num_groups as usize;
    let skip_bitmap = data[pos];
    pos += 1;

    // Checkerboard per-odd-stream k (1024 bytes = 128 odd streams × 8 groups,
    // present only when TILE_FLAG_CHECKERBOARD_K set)
    let k_stream_odd = if has_ck {
        let v = data[pos..pos + 1024].to_vec();
        pos += 1024;
        v
    } else {
        Vec::new()
    };

    // Stream lengths
    let mut stream_lengths = vec![0u32; RICE_STREAMS_PER_TILE];
    if compact_streams {
        // Varint stream lengths (256 entries)
        for sl in stream_lengths.iter_mut() {
            *sl = read_tile_varint(data, &mut pos) as u32;
        }
    } else {
        // Legacy: fixed 256 × u16
        for sl in stream_lengths.iter_mut() {
            *sl = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as u32;
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
            k_zrl_nz_values,
            k_zrl_z_values,
            skip_bitmap,
            k_stream_odd,
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
