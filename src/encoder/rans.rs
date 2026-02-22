/// rANS (range Asymmetric Numeral Systems) entropy coder.
///
/// Per-tile encoding: each tile is independently encoded with its own
/// frequency table. This preserves tile independence for GPU parallelism
/// and random access.
///
/// Based on Fabian Giesen's ryg_rans implementation.
/// Patent-free — ANS was placed in the public domain by Jarek Duda.

const RANS_BYTE_L: u32 = 1 << 23;
const RANS_PRECISION: u32 = 12;
const RANS_M: u32 = 1 << RANS_PRECISION;

/// A compressed tile: frequency table + rANS-encoded data.
#[derive(Debug, Clone)]
pub struct RansTile {
    /// Minimum coefficient value (offset for symbol mapping)
    pub min_val: i32,
    /// Number of symbols in alphabet
    pub alphabet_size: u32,
    /// Normalized frequency table (sum = RANS_M = 4096)
    pub freqs: Vec<u32>,
    /// rANS encoded byte stream
    pub data: Vec<u8>,
    /// Number of coefficients encoded
    pub num_coefficients: u32,
}

impl RansTile {
    /// Total size in bytes (data + freq table overhead)
    pub fn byte_size(&self) -> usize {
        // data + freq table (2 bytes per entry) + header (min_val, alphabet_size, num_coefficients, data_len)
        self.data.len() + self.alphabet_size as usize * 2 + 16
    }
}

/// Encode a tile's quantized coefficients using rANS.
/// `coefficients` are integer-valued (from quantization).
pub fn rans_encode_tile(coefficients: &[i32]) -> RansTile {
    if coefficients.is_empty() {
        return RansTile {
            min_val: 0,
            alphabet_size: 0,
            freqs: vec![],
            data: vec![],
            num_coefficients: 0,
        };
    }

    // Find range
    let min_val = *coefficients.iter().min().unwrap();
    let max_val = *coefficients.iter().max().unwrap();
    let alphabet_size = (max_val - min_val + 1) as usize;

    // Build histogram
    let mut hist = vec![0u32; alphabet_size];
    for &c in coefficients {
        hist[(c - min_val) as usize] += 1;
    }

    // Normalize histogram to sum = RANS_M
    let freqs = normalize_histogram(&hist, RANS_M);

    // Build cumulative frequencies
    let mut cum_freqs = vec![0u32; alphabet_size + 1];
    for i in 0..alphabet_size {
        cum_freqs[i + 1] = cum_freqs[i] + freqs[i];
    }
    debug_assert_eq!(cum_freqs[alphabet_size], RANS_M);

    // Allocate output buffer (conservative: worst case ~2 bytes per symbol + overhead)
    let buf_size = coefficients.len() * 2 + 64;
    let mut buf = vec![0u8; buf_size];
    let mut ptr = buf_size;

    // Initialize rANS state
    let mut state: u32 = RANS_BYTE_L;

    // Encode symbols in reverse order
    for &c in coefficients.iter().rev() {
        let sym = (c - min_val) as usize;
        let start = cum_freqs[sym];
        let freq = freqs[sym];

        // Renormalize: push bytes until state is small enough
        let x_max = ((RANS_BYTE_L >> RANS_PRECISION) << 8) * freq;
        while state >= x_max {
            ptr -= 1;
            buf[ptr] = (state & 0xff) as u8;
            state >>= 8;
        }

        // Encode
        state = ((state / freq) << RANS_PRECISION) + (state % freq) + start;
    }

    // Flush final state (little-endian)
    ptr -= 4;
    buf[ptr] = (state & 0xff) as u8;
    buf[ptr + 1] = ((state >> 8) & 0xff) as u8;
    buf[ptr + 2] = ((state >> 16) & 0xff) as u8;
    buf[ptr + 3] = ((state >> 24) & 0xff) as u8;

    let data = buf[ptr..].to_vec();

    RansTile {
        min_val,
        alphabet_size: alphabet_size as u32,
        freqs,
        data,
        num_coefficients: coefficients.len() as u32,
    }
}

/// Decode a rANS-encoded tile back to integer coefficients.
pub fn rans_decode_tile(tile: &RansTile) -> Vec<i32> {
    if tile.num_coefficients == 0 {
        return vec![];
    }

    let alphabet_size = tile.alphabet_size as usize;

    // Build cumulative frequencies
    let mut cum_freqs = vec![0u32; alphabet_size + 1];
    for i in 0..alphabet_size {
        cum_freqs[i + 1] = cum_freqs[i] + tile.freqs[i];
    }

    // Build decode lookup table: slot -> symbol
    let mut slot_to_sym = vec![0u16; RANS_M as usize];
    for sym in 0..alphabet_size {
        for j in cum_freqs[sym]..cum_freqs[sym] + tile.freqs[sym] {
            slot_to_sym[j as usize] = sym as u16;
        }
    }

    let buf = &tile.data;
    let mut ptr: usize = 0;

    // Read initial state
    let mut state = (buf[ptr] as u32)
        | ((buf[ptr + 1] as u32) << 8)
        | ((buf[ptr + 2] as u32) << 16)
        | ((buf[ptr + 3] as u32) << 24);
    ptr += 4;

    // Decode symbols
    let mut output = Vec::with_capacity(tile.num_coefficients as usize);
    let mask = RANS_M - 1;

    for _ in 0..tile.num_coefficients {
        // Get slot
        let slot = state & mask;
        let sym = slot_to_sym[slot as usize] as usize;

        // Map back to coefficient
        output.push(sym as i32 + tile.min_val);

        // Advance state
        let start = cum_freqs[sym];
        let freq = tile.freqs[sym];
        state = freq * (state >> RANS_PRECISION) + (state & mask) - start;

        // Renormalize
        while state < RANS_BYTE_L {
            if ptr < buf.len() {
                state = (state << 8) | buf[ptr] as u32;
                ptr += 1;
            } else {
                break;
            }
        }
    }

    output
}

/// Normalize a histogram so that:
/// - Sum of all frequencies = target_sum (power of 2)
/// - Every non-zero entry stays >= 1
fn normalize_histogram(hist: &[u32], target_sum: u32) -> Vec<u32> {
    let total: u64 = hist.iter().map(|&h| h as u64).sum();
    if total == 0 {
        // All zeros — shouldn't happen but handle gracefully
        let mut freqs = vec![0u32; hist.len()];
        if !freqs.is_empty() {
            freqs[0] = target_sum;
        }
        return freqs;
    }

    let mut freqs = vec![0u32; hist.len()];
    let mut assigned = 0u32;
    let mut nonzero_count = 0u32;

    // First pass: scale, ensuring minimum of 1 for non-zero entries
    for (i, &h) in hist.iter().enumerate() {
        if h > 0 {
            let scaled = (h as u64 * target_sum as u64 / total).max(1) as u32;
            freqs[i] = scaled;
            assigned += scaled;
            nonzero_count += 1;
        }
    }

    // Adjust to hit exact target_sum
    // Find the entries we can adjust (prefer adjusting the most frequent)
    let mut adjustable: Vec<usize> = (0..hist.len())
        .filter(|&i| hist[i] > 0)
        .collect();
    adjustable.sort_by(|&a, &b| hist[b].cmp(&hist[a]));

    if assigned < target_sum {
        // Need to add more
        let mut deficit = target_sum - assigned;
        for &idx in &adjustable {
            if deficit == 0 {
                break;
            }
            let add = deficit.min(target_sum / nonzero_count).max(1);
            freqs[idx] += add;
            deficit -= add;
        }
        // If still deficit (rounding), add to the first adjustable
        if deficit > 0 {
            freqs[adjustable[0]] += deficit;
        }
    } else if assigned > target_sum {
        // Need to remove
        let mut surplus = assigned - target_sum;
        for &idx in &adjustable {
            if surplus == 0 {
                break;
            }
            if freqs[idx] > 1 {
                let remove = surplus.min(freqs[idx] - 1);
                freqs[idx] -= remove;
                surplus -= remove;
            }
        }
    }

    debug_assert_eq!(
        freqs.iter().sum::<u32>(),
        target_sum,
        "Frequency normalization failed: sum={}, target={}",
        freqs.iter().sum::<u32>(),
        target_sum
    );

    freqs
}

/// Serialize a RansTile to bytes.
pub fn serialize_tile(tile: &RansTile) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&tile.min_val.to_le_bytes());
    out.extend_from_slice(&tile.alphabet_size.to_le_bytes());
    out.extend_from_slice(&tile.num_coefficients.to_le_bytes());
    let data_len = tile.data.len() as u32;
    out.extend_from_slice(&data_len.to_le_bytes());
    // Frequency table as u16 (max value = 4096, fits in u16)
    for &f in &tile.freqs {
        out.extend_from_slice(&(f as u16).to_le_bytes());
    }
    out.extend_from_slice(&tile.data);
    out
}

/// Deserialize a RansTile from bytes. Returns (tile, bytes_consumed).
pub fn deserialize_tile(data: &[u8]) -> (RansTile, usize) {
    let mut pos = 0;
    let min_val = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let alphabet_size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let num_coefficients = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let data_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    let mut freqs = Vec::with_capacity(alphabet_size as usize);
    for _ in 0..alphabet_size {
        let f = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
        freqs.push(f as u32);
        pos += 2;
    }

    let encoded_data = data[pos..pos + data_len].to_vec();
    pos += data_len;

    (
        RansTile {
            min_val,
            alphabet_size,
            freqs,
            data: encoded_data,
            num_coefficients,
        },
        pos,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_simple() {
        let coefficients = vec![0, 1, -1, 0, 2, -2, 0, 0, 1, -1, 0, 3];
        let tile = rans_encode_tile(&coefficients);
        let decoded = rans_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_roundtrip_zeros() {
        let coefficients = vec![0; 1000];
        let tile = rans_encode_tile(&coefficients);
        let decoded = rans_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
        // All zeros should compress very well
        assert!(tile.data.len() < 100);
    }

    #[test]
    fn test_roundtrip_large() {
        // Simulate wavelet coefficients: mostly small, some larger
        let mut coefficients = Vec::new();
        for i in 0..65536 {
            let v = if i % 7 == 0 {
                (i % 50) as i32 - 25
            } else if i % 3 == 0 {
                (i % 10) as i32 - 5
            } else {
                0
            };
            coefficients.push(v);
        }
        let tile = rans_encode_tile(&coefficients);
        let decoded = rans_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_roundtrip_serialize() {
        let coefficients = vec![5, -3, 0, 0, 1, 100, -50, 0];
        let tile = rans_encode_tile(&coefficients);
        let serialized = serialize_tile(&tile);
        let (deserialized, consumed) = deserialize_tile(&serialized);
        assert_eq!(consumed, serialized.len());
        let decoded = rans_decode_tile(&deserialized);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_compression_ratio() {
        // Mostly zeros should compress much better than 16 bits/coefficient
        let mut coefficients = vec![0i32; 65536];
        // Sprinkle in a few non-zero values
        for i in (0..65536).step_by(100) {
            coefficients[i] = (i % 5) as i32 - 2;
        }
        let tile = rans_encode_tile(&coefficients);
        let raw_size = coefficients.len() * 2; // 16 bits per coeff
        let rans_size = tile.byte_size();
        // Should be significantly smaller
        assert!(
            rans_size < raw_size / 2,
            "rANS size {} should be much less than raw size {}",
            rans_size,
            raw_size
        );
    }
}
