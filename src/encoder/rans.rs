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
const MAX_ZERO_RUN: i32 = 256;

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
pub fn normalize_histogram(hist: &[u32], target_sum: u32) -> Vec<u32> {
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
    let mut adjustable: Vec<usize> = (0..hist.len()).filter(|&i| hist[i] > 0).collect();
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

pub const STREAMS_PER_TILE: usize = 32;

/// A compressed tile using 32 interleaved rANS streams.
/// Coefficients are split stride-32 across streams sharing one frequency table.
/// Designed for GPU decode: one thread per stream, one workgroup per tile.
#[derive(Debug, Clone)]
pub struct InterleavedRansTile {
    /// Minimum coefficient value (offset for symbol mapping)
    pub min_val: i32,
    /// Number of symbols in alphabet
    pub alphabet_size: u32,
    /// Number of coefficients encoded
    pub num_coefficients: u32,
    /// Zero-run-length base symbol (0 = no ZRL for backward compat)
    pub zrun_base: i32,
    /// Normalized frequency table (sum = RANS_M = 4096)
    pub freqs: Vec<u32>,
    /// Cumulative frequency table (length = alphabet_size + 1)
    pub cumfreqs: Vec<u32>,
    /// Per-stream encoded byte data (renormalization bytes, excludes initial state)
    pub stream_data: Vec<Vec<u8>>,
    /// Per-stream initial rANS state
    pub stream_initial_state: Vec<u32>,
}

impl InterleavedRansTile {
    pub fn byte_size(&self) -> usize {
        // header: 16 + 32*4 (stream lengths) + 32*4 (initial states) = 272
        // freq table: alphabet_size * 2
        // stream data: sum of all stream lengths
        272 + self.alphabet_size as usize * 2
            + self.stream_data.iter().map(|s| s.len()).sum::<usize>()
    }
}

/// Encode a tile's quantized coefficients using 32 interleaved rANS streams.
/// All streams share one frequency table built from all coefficients.
/// Stream i gets coefficients at indices i, i+32, i+64, ... (stride-32).
pub fn rans_encode_tile_interleaved(coefficients: &[i32]) -> InterleavedRansTile {
    if coefficients.is_empty() {
        return InterleavedRansTile {
            min_val: 0,
            alphabet_size: 0,
            num_coefficients: 0,
            zrun_base: 0,
            freqs: vec![],
            cumfreqs: vec![0],
            stream_data: vec![vec![]; STREAMS_PER_TILE],
            stream_initial_state: vec![RANS_BYTE_L; STREAMS_PER_TILE],
        };
    }

    // Build shared frequency table from ALL coefficients
    let min_val = *coefficients.iter().min().unwrap();
    let max_val = *coefficients.iter().max().unwrap();
    let alphabet_size = (max_val - min_val + 1) as usize;

    let mut hist = vec![0u32; alphabet_size];
    for &c in coefficients {
        hist[(c - min_val) as usize] += 1;
    }

    let freqs = normalize_histogram(&hist, RANS_M);

    let mut cumfreqs = vec![0u32; alphabet_size + 1];
    for i in 0..alphabet_size {
        cumfreqs[i + 1] = cumfreqs[i] + freqs[i];
    }
    debug_assert_eq!(cumfreqs[alphabet_size], RANS_M);

    // Encode each of the 32 streams independently
    let num_coefficients = coefficients.len();
    let mut stream_data = Vec::with_capacity(STREAMS_PER_TILE);
    let mut stream_initial_state = Vec::with_capacity(STREAMS_PER_TILE);

    for s in 0..STREAMS_PER_TILE {
        let stream_coeffs: Vec<i32> = coefficients
            .iter()
            .skip(s)
            .step_by(STREAMS_PER_TILE)
            .copied()
            .collect();

        if stream_coeffs.is_empty() {
            stream_data.push(vec![]);
            stream_initial_state.push(RANS_BYTE_L);
            continue;
        }

        let buf_size = stream_coeffs.len() * 2 + 64;
        let mut buf = vec![0u8; buf_size];
        let mut ptr = buf_size;
        let mut state: u32 = RANS_BYTE_L;

        for &c in stream_coeffs.iter().rev() {
            let sym = (c - min_val) as usize;
            let start = cumfreqs[sym];
            let freq = freqs[sym];

            let x_max = ((RANS_BYTE_L >> RANS_PRECISION) << 8) * freq;
            while state >= x_max {
                ptr -= 1;
                buf[ptr] = (state & 0xff) as u8;
                state >>= 8;
            }

            state = ((state / freq) << RANS_PRECISION) + (state % freq) + start;
        }

        // Store final encoder state as the initial decoder state
        stream_initial_state.push(state);
        // Remaining bytes are renormalization data for the decoder
        stream_data.push(buf[ptr..].to_vec());
    }

    InterleavedRansTile {
        min_val,
        alphabet_size: alphabet_size as u32,
        num_coefficients: num_coefficients as u32,
        zrun_base: 0,
        freqs,
        cumfreqs,
        stream_data,
        stream_initial_state,
    }
}

/// Zero-run-length encode: replace consecutive zeros with run symbols.
/// Run symbol value = zrun_base + (run_length - 1). Non-zero values pass through.
pub fn zero_run_encode(coefficients: &[i32], zrun_base: i32) -> Vec<i32> {
    let mut symbols = Vec::with_capacity(coefficients.len());
    let mut i = 0;
    while i < coefficients.len() {
        if coefficients[i] == 0 {
            let mut run_len = 0i32;
            while i < coefficients.len() && coefficients[i] == 0 && run_len < MAX_ZERO_RUN {
                run_len += 1;
                i += 1;
            }
            symbols.push(zrun_base + run_len - 1);
        } else {
            symbols.push(coefficients[i]);
            i += 1;
        }
    }
    symbols
}

/// Zero-run-length decode: expand run symbols back to zeros.
pub fn zero_run_decode(symbols: &[i32], zrun_base: i32, output_len: usize) -> Vec<i32> {
    let mut output = Vec::with_capacity(output_len);
    for &sym in symbols {
        if sym >= zrun_base {
            let run_len = (sym - zrun_base + 1) as usize;
            output.extend(std::iter::repeat_n(0, run_len));
        } else {
            output.push(sym);
        }
    }
    debug_assert_eq!(output.len(), output_len, "ZRL decode length mismatch");
    output
}

/// Adaptive ZRL encoder: tries both ZRL and plain rANS, returns whichever
/// produces a smaller tile. Skips ZRL entirely when zero density is too low
/// (< 60%) since the alphabet expansion overhead would dominate.
pub fn rans_encode_tile_interleaved_zrl(coefficients: &[i32]) -> InterleavedRansTile {
    if coefficients.is_empty() {
        return rans_encode_tile_interleaved(coefficients);
    }

    // Quick zero-density check: ZRL can only help if there are enough consecutive zeros
    let zero_count = coefficients.iter().filter(|&&c| c == 0).count();
    let zero_fraction = zero_count as f64 / coefficients.len() as f64;
    if zero_fraction < 0.6 {
        return rans_encode_tile_interleaved(coefficients);
    }

    let tile_plain = rans_encode_tile_interleaved(coefficients);
    let tile_zrl = rans_encode_tile_interleaved_zrl_inner(coefficients);
    if tile_zrl.byte_size() < tile_plain.byte_size() {
        tile_zrl
    } else {
        tile_plain
    }
}

/// Core ZRL encoder: always applies zero-run-length coding before rANS.
fn rans_encode_tile_interleaved_zrl_inner(coefficients: &[i32]) -> InterleavedRansTile {
    if coefficients.is_empty() {
        return InterleavedRansTile {
            min_val: 0,
            alphabet_size: 0,
            num_coefficients: 0,
            zrun_base: 0,
            freqs: vec![],
            cumfreqs: vec![0],
            stream_data: vec![vec![]; STREAMS_PER_TILE],
            stream_initial_state: vec![RANS_BYTE_L; STREAMS_PER_TILE],
        };
    }

    // zrun_base = max(|non-zero coeff|) + 1, so run symbols never collide with coefficients
    let max_abs = coefficients
        .iter()
        .filter(|&&c| c != 0)
        .map(|&c| c.abs())
        .max()
        .unwrap_or(0);
    let zrun_base = max_abs + 1;

    // Extract stride-32 per-stream coefficients and apply ZRL per stream
    let num_coefficients = coefficients.len();
    let mut all_zrl_streams: Vec<Vec<i32>> = Vec::with_capacity(STREAMS_PER_TILE);

    for s in 0..STREAMS_PER_TILE {
        let stream_coeffs: Vec<i32> = coefficients
            .iter()
            .skip(s)
            .step_by(STREAMS_PER_TILE)
            .copied()
            .collect();
        all_zrl_streams.push(zero_run_encode(&stream_coeffs, zrun_base));
    }

    // Build shared histogram from ALL ZRL-transformed symbols across all streams
    let min_val = all_zrl_streams
        .iter()
        .flat_map(|s| s.iter())
        .copied()
        .min()
        .unwrap();
    let max_val = all_zrl_streams
        .iter()
        .flat_map(|s| s.iter())
        .copied()
        .max()
        .unwrap();
    let alphabet_size = (max_val - min_val + 1) as usize;

    let mut hist = vec![0u32; alphabet_size];
    for stream in &all_zrl_streams {
        for &sym in stream {
            hist[(sym - min_val) as usize] += 1;
        }
    }

    let freqs = normalize_histogram(&hist, RANS_M);

    let mut cumfreqs = vec![0u32; alphabet_size + 1];
    for i in 0..alphabet_size {
        cumfreqs[i + 1] = cumfreqs[i] + freqs[i];
    }
    debug_assert_eq!(cumfreqs[alphabet_size], RANS_M);

    // rANS encode each ZRL-transformed stream
    let mut stream_data = Vec::with_capacity(STREAMS_PER_TILE);
    let mut stream_initial_state = Vec::with_capacity(STREAMS_PER_TILE);

    for zrl_stream in &all_zrl_streams {
        if zrl_stream.is_empty() {
            stream_data.push(vec![]);
            stream_initial_state.push(RANS_BYTE_L);
            continue;
        }

        let buf_size = zrl_stream.len() * 2 + 64;
        let mut buf = vec![0u8; buf_size];
        let mut ptr = buf_size;
        let mut state: u32 = RANS_BYTE_L;

        for &c in zrl_stream.iter().rev() {
            let sym = (c - min_val) as usize;
            let start = cumfreqs[sym];
            let freq = freqs[sym];

            let x_max = ((RANS_BYTE_L >> RANS_PRECISION) << 8) * freq;
            while state >= x_max {
                ptr -= 1;
                buf[ptr] = (state & 0xff) as u8;
                state >>= 8;
            }

            state = ((state / freq) << RANS_PRECISION) + (state % freq) + start;
        }

        stream_initial_state.push(state);
        stream_data.push(buf[ptr..].to_vec());
    }

    InterleavedRansTile {
        min_val,
        alphabet_size: alphabet_size as u32,
        num_coefficients: num_coefficients as u32,
        zrun_base,
        freqs,
        cumfreqs,
        stream_data,
        stream_initial_state,
    }
}

/// Decode an interleaved rANS tile back to integer coefficients (CPU reference).
pub fn rans_decode_tile_interleaved(tile: &InterleavedRansTile) -> Vec<i32> {
    if tile.num_coefficients == 0 {
        return vec![];
    }

    let alphabet_size = tile.alphabet_size as usize;
    let num_coefficients = tile.num_coefficients as usize;
    let mut output = vec![0i32; num_coefficients];

    for s in 0..STREAMS_PER_TILE {
        if s >= num_coefficients {
            break;
        }
        let stream_output_len = 1 + (num_coefficients - 1 - s) / STREAMS_PER_TILE;

        let mut state = tile.stream_initial_state[s];
        let buf = &tile.stream_data[s];
        let mut ptr: usize = 0;
        let mask = RANS_M - 1;

        if tile.zrun_base != 0 {
            // ZRL-aware decode: while-loop tracking output position
            let mut output_i = 0;
            while output_i < stream_output_len {
                let slot = state & mask;
                let sym = binary_search_cumfreq(&tile.cumfreqs, slot, alphabet_size);
                let value = sym as i32 + tile.min_val;

                let start = tile.cumfreqs[sym];
                let freq = tile.cumfreqs[sym + 1] - tile.cumfreqs[sym];
                state = freq * (state >> RANS_PRECISION) + (state & mask) - start;

                while state < RANS_BYTE_L {
                    if ptr < buf.len() {
                        state = (state << 8) | buf[ptr] as u32;
                        ptr += 1;
                    } else {
                        break;
                    }
                }

                if value >= tile.zrun_base {
                    let run_len = (value - tile.zrun_base + 1) as usize;
                    for j in 0..run_len {
                        output[s + (output_i + j) * STREAMS_PER_TILE] = 0;
                    }
                    output_i += run_len;
                } else {
                    output[s + output_i * STREAMS_PER_TILE] = value;
                    output_i += 1;
                }
            }
        } else {
            // Original decode path (no ZRL)
            for i in 0..stream_output_len {
                let slot = state & mask;
                let sym = binary_search_cumfreq(&tile.cumfreqs, slot, alphabet_size);

                output[s + i * STREAMS_PER_TILE] = sym as i32 + tile.min_val;

                let start = tile.cumfreqs[sym];
                let freq = tile.cumfreqs[sym + 1] - tile.cumfreqs[sym];
                state = freq * (state >> RANS_PRECISION) + (state & mask) - start;

                while state < RANS_BYTE_L {
                    if ptr < buf.len() {
                        state = (state << 8) | buf[ptr] as u32;
                        ptr += 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    output
}

/// Binary search on cumfreq: find sym where cumfreq[sym] <= slot < cumfreq[sym+1].
fn binary_search_cumfreq(cumfreqs: &[u32], slot: u32, alphabet_size: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = alphabet_size;
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cumfreqs[mid + 1] <= slot {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
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

/// Serialize an InterleavedRansTile to bytes.
pub fn serialize_tile_interleaved(tile: &InterleavedRansTile) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&tile.min_val.to_le_bytes());
    out.extend_from_slice(&tile.alphabet_size.to_le_bytes());
    out.extend_from_slice(&tile.num_coefficients.to_le_bytes());
    out.extend_from_slice(&tile.zrun_base.to_le_bytes());
    // 32 stream data lengths
    for s in 0..STREAMS_PER_TILE {
        let len = tile.stream_data[s].len() as u32;
        out.extend_from_slice(&len.to_le_bytes());
    }
    // 32 initial states
    for s in 0..STREAMS_PER_TILE {
        out.extend_from_slice(&tile.stream_initial_state[s].to_le_bytes());
    }
    // Frequency table as u16
    for &f in &tile.freqs {
        out.extend_from_slice(&(f as u16).to_le_bytes());
    }
    // Stream data (concatenated)
    for s in 0..STREAMS_PER_TILE {
        out.extend_from_slice(&tile.stream_data[s]);
    }
    out
}

/// Deserialize an InterleavedRansTile from bytes. Returns (tile, bytes_consumed).
pub fn deserialize_tile_interleaved(data: &[u8]) -> (InterleavedRansTile, usize) {
    let mut pos = 0;

    let min_val = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let alphabet_size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let num_coefficients = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let zrun_base = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    let mut stream_lengths = Vec::with_capacity(STREAMS_PER_TILE);
    for _ in 0..STREAMS_PER_TILE {
        let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        stream_lengths.push(len);
        pos += 4;
    }

    let mut stream_initial_state = Vec::with_capacity(STREAMS_PER_TILE);
    for _ in 0..STREAMS_PER_TILE {
        let state = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        stream_initial_state.push(state);
        pos += 4;
    }

    let mut freqs = Vec::with_capacity(alphabet_size as usize);
    for _ in 0..alphabet_size {
        let f = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
        freqs.push(f as u32);
        pos += 2;
    }

    // Compute cumfreqs
    let mut cumfreqs = vec![0u32; alphabet_size as usize + 1];
    for i in 0..alphabet_size as usize {
        cumfreqs[i + 1] = cumfreqs[i] + freqs[i];
    }

    let mut stream_data = Vec::with_capacity(STREAMS_PER_TILE);
    for len in &stream_lengths {
        let len = *len as usize;
        stream_data.push(data[pos..pos + len].to_vec());
        pos += len;
    }

    (
        InterleavedRansTile {
            min_val,
            alphabet_size,
            num_coefficients,
            zrun_base,
            freqs,
            cumfreqs,
            stream_data,
            stream_initial_state,
        },
        pos,
    )
}

// --- Per-subband entropy coding ---

/// Compute which subband group (0..num_levels) a tile-local position belongs to.
///
/// Group 0 = LL (DC subband).
/// Group k (1..=num_levels) = Level (k-1) detail subbands (LH + HL + HH combined).
///
/// This mirrors `compute_subband_index` from cfl.rs but collapses LH/HL/HH
/// within each level into a single group, since their statistical distributions
/// are similar enough to share one frequency table.
pub fn compute_subband_group(lx: u32, ly: u32, tile_size: u32, num_levels: u32) -> usize {
    let mut region = tile_size;
    for level in 0..num_levels {
        let half = region / 2;
        if lx >= half || ly >= half {
            return (level + 1) as usize;
        }
        region = half;
    }
    0 // LL
}

/// Frequency table for one subband group within a tile.
#[derive(Debug, Clone)]
pub struct SubbandGroupFreqs {
    pub min_val: i32,
    pub alphabet_size: u32,
    /// Zero-run-length base symbol (0 = no ZRL for this group)
    pub zrun_base: i32,
    pub freqs: Vec<u32>,
    pub cumfreqs: Vec<u32>,
}

/// A compressed tile using per-subband frequency tables + 32 interleaved rANS streams.
///
/// Each of the `num_groups` (= 1 + num_levels) subband groups gets its own frequency table.
/// During encoding/decoding, the table selection for each coefficient is determined by its
/// 2D position within the tile (which subband group it falls in).
///
/// Detail subband groups (groups > 0) use per-group zero-run-length encoding to collapse
/// consecutive zero runs into single symbols. The LL subband (group 0) does not use ZRL
/// since it has few zeros. Each group's `zrun_base` is stored in its `SubbandGroupFreqs`.
#[derive(Debug, Clone)]
pub struct SubbandRansTile {
    pub num_coefficients: u32,
    pub tile_size: u32,
    pub num_levels: u32,
    pub num_groups: u32,
    pub groups: Vec<SubbandGroupFreqs>,
    pub stream_data: Vec<Vec<u8>>,
    pub stream_initial_state: Vec<u32>,
}

impl SubbandRansTile {
    pub fn byte_size(&self) -> usize {
        // header: num_coefficients(4) + tile_size(4) + num_levels(4) + num_groups(4) = 16
        // per-group: min_val(4) + alphabet_size(4) + zrun_base(4) + freqs(alphabet_size * 2)
        // streams: 32 stream_lengths(4 each) + 32 initial_states(4 each) = 256
        // stream data: sum of all stream bytes
        let group_overhead: usize = self
            .groups
            .iter()
            .map(|g| 12 + g.alphabet_size as usize * 2)
            .sum();
        16 + group_overhead + 256 + self.stream_data.iter().map(|s| s.len()).sum::<usize>()
    }
}

/// Encode a tile's quantized coefficients using per-subband frequency tables
/// and 32 interleaved rANS streams, with per-group zero-run-length encoding.
///
/// Each subband group (LL, and one per wavelet detail level) gets its own
/// frequency table, allowing rANS to model each distribution tightly.
/// Detail subband groups (g > 0) also get ZRL encoding: consecutive zeros
/// within the same group in a stream are collapsed into run-length symbols.
/// The LL subband (group 0) does not use ZRL since it has few zeros.
pub fn rans_encode_tile_interleaved_subband(
    coefficients: &[i32],
    tile_size: u32,
    num_levels: u32,
) -> SubbandRansTile {
    let num_groups = 1 + num_levels as usize;

    if coefficients.is_empty() {
        return SubbandRansTile {
            num_coefficients: 0,
            tile_size,
            num_levels,
            num_groups: num_groups as u32,
            groups: (0..num_groups)
                .map(|_| SubbandGroupFreqs {
                    min_val: 0,
                    alphabet_size: 0,
                    zrun_base: 0,
                    freqs: vec![],
                    cumfreqs: vec![0],
                })
                .collect(),
            stream_data: vec![vec![]; STREAMS_PER_TILE],
            stream_initial_state: vec![RANS_BYTE_L; STREAMS_PER_TILE],
        };
    }

    let ts = tile_size as usize;

    // Precompute subband group for each coefficient position
    let group_map: Vec<usize> = (0..coefficients.len())
        .map(|i| {
            let lx = (i % ts) as u32;
            let ly = (i / ts) as u32;
            compute_subband_group(lx, ly, tile_size, num_levels)
        })
        .collect();

    // Step 1: Compute per-group zrun_base for detail subbands (g > 0).
    // zrun_base = max(|non-zero coeff|) + 1 for that group, so run symbols never collide.
    // Group 0 (LL) gets zrun_base = 0 (no ZRL).
    let mut group_max_abs = vec![0i32; num_groups];
    let mut group_zero_count = vec![0usize; num_groups];
    let mut group_total_count = vec![0usize; num_groups];
    let mut group_has_data = vec![false; num_groups];

    for (i, &c) in coefficients.iter().enumerate() {
        let g = group_map[i];
        group_has_data[g] = true;
        group_total_count[g] += 1;
        if c == 0 {
            group_zero_count[g] += 1;
        } else {
            group_max_abs[g] = group_max_abs[g].max(c.abs());
        }
    }

    let mut group_zrun_base = vec![0i32; num_groups];
    for g in 1..num_groups {
        // Apply ZRL only when it is likely to help: high zero density AND the
        // existing coefficient alphabet is large enough that the overhead of
        // 256 extra run-length symbols in the frequency table is proportionally
        // small. For tiny alphabets (e.g. 5 symbols), rANS already gives the
        // zero symbol a very high frequency — ZRL can only hurt by bloating
        // the frequency table from 5 to 261 entries.
        if group_has_data[g] && group_total_count[g] > 0 {
            let zero_frac = group_zero_count[g] as f64 / group_total_count[g] as f64;
            let non_zrl_alphabet = (2 * group_max_abs[g] + 1) as usize; // rough: -max..+max + 0
            if zero_frac >= 0.6 && non_zrl_alphabet >= 16 {
                group_zrun_base[g] = group_max_abs[g] + 1;
            }
        }
    }

    // Step 2: Build per-stream ZRL-transformed symbol sequences.
    // For each stream, walk through its coefficient positions and apply per-group ZRL.
    let num_coefficients = coefficients.len();
    let mut stream_zrl_symbols: Vec<Vec<(i32, usize)>> = Vec::with_capacity(STREAMS_PER_TILE);

    for s in 0..STREAMS_PER_TILE {
        let indices: Vec<usize> = (s..num_coefficients).step_by(STREAMS_PER_TILE).collect();
        let mut symbols: Vec<(i32, usize)> = Vec::with_capacity(indices.len()); // (symbol, group)

        let mut i = 0;
        while i < indices.len() {
            let idx = indices[i];
            let g = group_map[idx];
            let c = coefficients[idx];
            let zrun_base = group_zrun_base[g];

            if zrun_base != 0 && c == 0 {
                // Count consecutive zeros in the same group within this stream
                let mut run_len = 0i32;
                while i < indices.len()
                    && coefficients[indices[i]] == 0
                    && group_map[indices[i]] == g
                    && run_len < MAX_ZERO_RUN
                {
                    run_len += 1;
                    i += 1;
                }
                symbols.push((zrun_base + run_len - 1, g));
            } else {
                symbols.push((c, g));
                i += 1;
            }
        }

        stream_zrl_symbols.push(symbols);
    }

    // Step 3: Build per-group frequency tables from ZRL-transformed symbols
    let mut group_min = vec![i32::MAX; num_groups];
    let mut group_max = vec![i32::MIN; num_groups];
    // Reset group_has_data based on ZRL symbols
    group_has_data = vec![false; num_groups];

    for stream_syms in &stream_zrl_symbols {
        for &(sym, g) in stream_syms {
            group_has_data[g] = true;
            group_min[g] = group_min[g].min(sym);
            group_max[g] = group_max[g].max(sym);
        }
    }

    let mut groups: Vec<SubbandGroupFreqs> = Vec::with_capacity(num_groups);

    for g in 0..num_groups {
        if !group_has_data[g] {
            groups.push(SubbandGroupFreqs {
                min_val: 0,
                alphabet_size: 1,
                zrun_base: 0,
                freqs: vec![RANS_M],
                cumfreqs: vec![0, RANS_M],
            });
            continue;
        }

        let min_val = group_min[g];
        let max_val = group_max[g];
        let alphabet_size = (max_val - min_val + 1) as usize;

        let mut hist = vec![0u32; alphabet_size];
        for stream_syms in &stream_zrl_symbols {
            for &(sym, sg) in stream_syms {
                if sg == g {
                    hist[(sym - min_val) as usize] += 1;
                }
            }
        }

        let freqs = normalize_histogram(&hist, RANS_M);

        let mut cumfreqs = vec![0u32; alphabet_size + 1];
        for i in 0..alphabet_size {
            cumfreqs[i + 1] = cumfreqs[i] + freqs[i];
        }
        debug_assert_eq!(cumfreqs[alphabet_size], RANS_M);

        groups.push(SubbandGroupFreqs {
            min_val,
            alphabet_size: alphabet_size as u32,
            zrun_base: group_zrun_base[g],
            freqs,
            cumfreqs,
        });
    }

    // Step 4: rANS encode each stream's ZRL-transformed symbols
    let mut stream_data = Vec::with_capacity(STREAMS_PER_TILE);
    let mut stream_initial_state = Vec::with_capacity(STREAMS_PER_TILE);

    for stream_syms in &stream_zrl_symbols {
        if stream_syms.is_empty() {
            stream_data.push(vec![]);
            stream_initial_state.push(RANS_BYTE_L);
            continue;
        }

        let buf_size = stream_syms.len() * 2 + 64;
        let mut buf = vec![0u8; buf_size];
        let mut ptr = buf_size;
        let mut state: u32 = RANS_BYTE_L;

        // Encode in reverse order
        for &(sym_val, g) in stream_syms.iter().rev() {
            let group = &groups[g];

            let sym = (sym_val - group.min_val) as usize;
            let start = group.cumfreqs[sym];
            let freq = group.freqs[sym];

            let x_max = ((RANS_BYTE_L >> RANS_PRECISION) << 8) * freq;
            while state >= x_max {
                ptr -= 1;
                buf[ptr] = (state & 0xff) as u8;
                state >>= 8;
            }

            state = ((state / freq) << RANS_PRECISION) + (state % freq) + start;
        }

        stream_initial_state.push(state);
        stream_data.push(buf[ptr..].to_vec());
    }

    SubbandRansTile {
        num_coefficients: num_coefficients as u32,
        tile_size,
        num_levels,
        num_groups: num_groups as u32,
        groups,
        stream_data,
        stream_initial_state,
    }
}

/// CPU reference decoder for per-subband interleaved rANS tiles.
/// Supports per-group ZRL: detail subbands with zrun_base > 0 have their
/// run-length symbols expanded back to consecutive zeros.
pub fn rans_decode_tile_interleaved_subband(tile: &SubbandRansTile) -> Vec<i32> {
    if tile.num_coefficients == 0 {
        return vec![];
    }

    let num_coefficients = tile.num_coefficients as usize;
    let ts = tile.tile_size as usize;
    let mut output = vec![0i32; num_coefficients];

    // Check if any group uses ZRL
    let has_zrl = tile.groups.iter().any(|g| g.zrun_base != 0);

    // Build per-group slot-to-symbol lookup tables
    let slot_tables: Vec<Vec<u16>> = tile
        .groups
        .iter()
        .map(|g| {
            let mut table = vec![0u16; RANS_M as usize];
            let asize = g.alphabet_size as usize;
            for sym in 0..asize {
                for j in g.cumfreqs[sym]..g.cumfreqs[sym + 1] {
                    table[j as usize] = sym as u16;
                }
            }
            table
        })
        .collect();

    for s in 0..STREAMS_PER_TILE {
        if s >= num_coefficients {
            break;
        }
        let stream_output_count = 1 + (num_coefficients - 1 - s) / STREAMS_PER_TILE;

        let mut state = tile.stream_initial_state[s];
        let buf = &tile.stream_data[s];
        let mut ptr: usize = 0;
        let mask = RANS_M - 1;

        if has_zrl {
            // ZRL-aware decode: output_i tracks position, may advance by more than 1
            let mut output_i = 0;
            while output_i < stream_output_count {
                let coeff_idx = s + output_i * STREAMS_PER_TILE;
                let lx = (coeff_idx % ts) as u32;
                let ly = (coeff_idx / ts) as u32;
                let g = compute_subband_group(lx, ly, tile.tile_size, tile.num_levels);
                let group = &tile.groups[g];

                let slot = state & mask;
                let sym = slot_tables[g][slot as usize] as usize;
                let value = sym as i32 + group.min_val;

                let start = group.cumfreqs[sym];
                let freq = group.cumfreqs[sym + 1] - group.cumfreqs[sym];
                state = freq * (state >> RANS_PRECISION) + (state & mask) - start;

                while state < RANS_BYTE_L {
                    if ptr < buf.len() {
                        state = (state << 8) | buf[ptr] as u32;
                        ptr += 1;
                    } else {
                        break;
                    }
                }

                if group.zrun_base != 0 && value >= group.zrun_base {
                    // ZRL run symbol: expand to consecutive zeros
                    let run_len = (value - group.zrun_base + 1) as usize;
                    for j in 0..run_len {
                        output[s + (output_i + j) * STREAMS_PER_TILE] = 0;
                    }
                    output_i += run_len;
                } else {
                    output[coeff_idx] = value;
                    output_i += 1;
                }
            }
        } else {
            // Original decode path (no ZRL in any group)
            for i in 0..stream_output_count {
                let coeff_idx = s + i * STREAMS_PER_TILE;
                let lx = (coeff_idx % ts) as u32;
                let ly = (coeff_idx / ts) as u32;
                let g = compute_subband_group(lx, ly, tile.tile_size, tile.num_levels);
                let group = &tile.groups[g];

                let slot = state & mask;
                let sym = slot_tables[g][slot as usize] as usize;

                output[coeff_idx] = sym as i32 + group.min_val;

                let start = group.cumfreqs[sym];
                let freq = group.cumfreqs[sym + 1] - group.cumfreqs[sym];
                state = freq * (state >> RANS_PRECISION) + (state & mask) - start;

                while state < RANS_BYTE_L {
                    if ptr < buf.len() {
                        state = (state << 8) | buf[ptr] as u32;
                        ptr += 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    output
}

/// Compute the "above-neighbor" context for a coefficient at tile position (lx, ly).
///
/// Returns 0 if the coefficient directly above in the same subband is zero or out-of-bounds.
/// Returns 1 if the above coefficient is nonzero.
/// LL subband (group 0) always returns context 0.
///
/// For a coefficient at (lx, ly):
/// - Find which subband it belongs to (within a wavelet level)
/// - Find the top-left corner and height of that subband region
/// - If this is the first row of the subband region, context = 0
/// - Otherwise, look at (lx, ly - 1) in the coefficient buffer
fn compute_above_context(
    coefficients: &[i32],
    lx: u32,
    ly: u32,
    tile_size: u32,
    num_levels: u32,
) -> u32 {
    let group = compute_subband_group(lx, ly, tile_size, num_levels);
    if group == 0 {
        return 0; // LL: always context 0
    }

    // Find the subband region boundaries for this coefficient.
    // Subband group `g` (1..=num_levels) corresponds to wavelet level (g-1).
    // At level L, the detail subbands occupy the region outside the (tile_size >> (L+1)) square
    // but inside the (tile_size >> L) square.
    let level = group - 1;
    let region = tile_size >> level;
    let half = region / 2;

    // Determine which sub-subband (LH, HL, or HH) and its top-left corner
    let subband_y_start = if ly < half {
        // HL subband: rows [0, half), cols [half, region)
        0
    } else if lx < half {
        // LH subband: rows [half, region), cols [0, half)
        half
    } else {
        // HH subband: rows [half, region), cols [half, region)
        half
    };

    // If this is the first row of the subband, no above neighbor
    if ly == subband_y_start {
        return 0;
    }

    // Look at the coefficient directly above
    let above_idx = ((ly - 1) * tile_size + lx) as usize;
    if coefficients[above_idx] == 0 {
        0
    } else {
        1
    }
}

/// Map a subband group index and context to the expanded group index for context-adaptive mode.
///
/// Layout: groups[0] = LL (context 0 only)
///         groups[1 + (g-1)*2] = detail group g, context 0
///         groups[1 + (g-1)*2 + 1] = detail group g, context 1
///
/// Total expanded groups = 1 + num_detail_groups * 2
fn context_group_index(subband_group: usize, context: u32) -> usize {
    if subband_group == 0 {
        0
    } else {
        1 + (subband_group - 1) * 2 + context as usize
    }
}

/// Number of expanded groups in context-adaptive mode.
fn num_context_groups(num_levels: u32) -> usize {
    1 + num_levels as usize * 2
}

/// Encode a tile using per-subband frequency tables with context-adaptive coding.
///
/// Like `rans_encode_tile_interleaved_subband` but each detail subband group gets
/// 2 frequency tables: one for coefficients whose above neighbor is zero (context 0)
/// and one for coefficients whose above neighbor is nonzero (context 1).
pub fn rans_encode_tile_interleaved_subband_ctx(
    coefficients: &[i32],
    tile_size: u32,
    num_levels: u32,
) -> SubbandRansTile {
    let num_groups = num_context_groups(num_levels);

    if coefficients.is_empty() {
        return SubbandRansTile {
            num_coefficients: 0,
            tile_size,
            num_levels,
            num_groups: num_groups as u32,
            groups: (0..num_groups)
                .map(|_| SubbandGroupFreqs {
                    min_val: 0,
                    alphabet_size: 0,
                    zrun_base: 0,
                    freqs: vec![],
                    cumfreqs: vec![0],
                })
                .collect(),
            stream_data: vec![vec![]; STREAMS_PER_TILE],
            stream_initial_state: vec![RANS_BYTE_L; STREAMS_PER_TILE],
        };
    }

    let ts = tile_size as usize;

    // Precompute subband group AND context for each coefficient position
    let group_map: Vec<usize> = (0..coefficients.len())
        .map(|i| {
            let lx = (i % ts) as u32;
            let ly = (i / ts) as u32;
            let g = compute_subband_group(lx, ly, tile_size, num_levels);
            let ctx = compute_above_context(coefficients, lx, ly, tile_size, num_levels);
            context_group_index(g, ctx)
        })
        .collect();

    // Compute per-expanded-group stats for ZRL decision
    let mut group_max_abs = vec![0i32; num_groups];
    let mut group_zero_count = vec![0usize; num_groups];
    let mut group_total_count = vec![0usize; num_groups];
    let mut group_has_data = vec![false; num_groups];

    for (i, &c) in coefficients.iter().enumerate() {
        let eg = group_map[i];
        group_has_data[eg] = true;
        group_total_count[eg] += 1;
        if c == 0 {
            group_zero_count[eg] += 1;
        } else {
            group_max_abs[eg] = group_max_abs[eg].max(c.abs());
        }
    }

    // ZRL decision: apply to detail context groups (expanded groups > 0)
    // Both context 0 and context 1 of the same original group share the same zrun_base
    // because they share the same coefficient value range.
    let mut group_zrun_base = vec![0i32; num_groups];
    for orig_g in 1..=num_levels as usize {
        // Check both context sub-groups; use shared zrun decision
        let eg0 = context_group_index(orig_g, 0);
        let eg1 = context_group_index(orig_g, 1);
        let total_max_abs = group_max_abs[eg0].max(group_max_abs[eg1]);
        let total_zeros = group_zero_count[eg0] + group_zero_count[eg1];
        let total_count = group_total_count[eg0] + group_total_count[eg1];

        if total_count > 0 {
            let zero_frac = total_zeros as f64 / total_count as f64;
            let non_zrl_alphabet = (2 * total_max_abs + 1) as usize;
            if zero_frac >= 0.6 && non_zrl_alphabet >= 16 {
                let zb = total_max_abs + 1;
                group_zrun_base[eg0] = zb;
                group_zrun_base[eg1] = zb;
            }
        }
    }

    // Build per-stream ZRL-transformed symbol sequences with expanded group indices
    let num_coefficients = coefficients.len();
    let mut stream_zrl_symbols: Vec<Vec<(i32, usize)>> = Vec::with_capacity(STREAMS_PER_TILE);

    for s in 0..STREAMS_PER_TILE {
        let indices: Vec<usize> = (s..num_coefficients).step_by(STREAMS_PER_TILE).collect();
        let mut symbols: Vec<(i32, usize)> = Vec::with_capacity(indices.len());

        let mut i = 0;
        while i < indices.len() {
            let idx = indices[i];
            let eg = group_map[idx];
            let c = coefficients[idx];
            let zrun_base = group_zrun_base[eg];

            if zrun_base != 0 && c == 0 {
                let mut run_len = 0i32;
                while i < indices.len()
                    && coefficients[indices[i]] == 0
                    && group_map[indices[i]] == eg
                    && run_len < MAX_ZERO_RUN
                {
                    run_len += 1;
                    i += 1;
                }
                symbols.push((zrun_base + run_len - 1, eg));
            } else {
                symbols.push((c, eg));
                i += 1;
            }
        }

        stream_zrl_symbols.push(symbols);
    }

    // Build per-expanded-group frequency tables from ZRL-transformed symbols
    let mut group_min = vec![i32::MAX; num_groups];
    let mut group_max = vec![i32::MIN; num_groups];
    group_has_data = vec![false; num_groups];

    for stream_syms in &stream_zrl_symbols {
        for &(sym, eg) in stream_syms {
            group_has_data[eg] = true;
            group_min[eg] = group_min[eg].min(sym);
            group_max[eg] = group_max[eg].max(sym);
        }
    }

    let mut groups: Vec<SubbandGroupFreqs> = Vec::with_capacity(num_groups);

    for eg in 0..num_groups {
        if !group_has_data[eg] {
            groups.push(SubbandGroupFreqs {
                min_val: 0,
                alphabet_size: 1,
                zrun_base: 0,
                freqs: vec![RANS_M],
                cumfreqs: vec![0, RANS_M],
            });
            continue;
        }

        let min_val = group_min[eg];
        let max_val = group_max[eg];
        let alphabet_size = (max_val - min_val + 1) as usize;

        let mut hist = vec![0u32; alphabet_size];
        for stream_syms in &stream_zrl_symbols {
            for &(sym, sg) in stream_syms {
                if sg == eg {
                    hist[(sym - min_val) as usize] += 1;
                }
            }
        }

        let freqs = normalize_histogram(&hist, RANS_M);

        let mut cumfreqs = vec![0u32; alphabet_size + 1];
        for i in 0..alphabet_size {
            cumfreqs[i + 1] = cumfreqs[i] + freqs[i];
        }
        debug_assert_eq!(cumfreqs[alphabet_size], RANS_M);

        groups.push(SubbandGroupFreqs {
            min_val,
            alphabet_size: alphabet_size as u32,
            zrun_base: group_zrun_base[eg],
            freqs,
            cumfreqs,
        });
    }

    // rANS encode each stream's ZRL-transformed symbols
    let mut stream_data = Vec::with_capacity(STREAMS_PER_TILE);
    let mut stream_initial_state = Vec::with_capacity(STREAMS_PER_TILE);

    for stream_syms in &stream_zrl_symbols {
        if stream_syms.is_empty() {
            stream_data.push(vec![]);
            stream_initial_state.push(RANS_BYTE_L);
            continue;
        }

        let buf_size = stream_syms.len() * 2 + 64;
        let mut buf = vec![0u8; buf_size];
        let mut ptr = buf_size;
        let mut state: u32 = RANS_BYTE_L;

        for &(sym_val, eg) in stream_syms.iter().rev() {
            let group = &groups[eg];
            let sym = (sym_val - group.min_val) as usize;
            let start = group.cumfreqs[sym];
            let freq = group.freqs[sym];

            let x_max = ((RANS_BYTE_L >> RANS_PRECISION) << 8) * freq;
            while state >= x_max {
                ptr -= 1;
                buf[ptr] = (state & 0xff) as u8;
                state >>= 8;
            }

            state = ((state / freq) << RANS_PRECISION) + (state % freq) + start;
        }

        stream_initial_state.push(state);
        stream_data.push(buf[ptr..].to_vec());
    }

    SubbandRansTile {
        num_coefficients: num_coefficients as u32,
        tile_size,
        num_levels,
        num_groups: num_groups as u32,
        groups,
        stream_data,
        stream_initial_state,
    }
}

/// CPU reference decoder for context-adaptive per-subband interleaved rANS tiles.
///
/// Each detail subband group has 2 frequency tables (context 0 and context 1).
/// Context is reconstructed on-the-fly: for each coefficient, check if the
/// coefficient directly above (same column, previous row within the same subband)
/// is zero or nonzero.
pub fn rans_decode_tile_interleaved_subband_ctx(tile: &SubbandRansTile) -> Vec<i32> {
    if tile.num_coefficients == 0 {
        return vec![];
    }

    let num_coefficients = tile.num_coefficients as usize;
    let ts = tile.tile_size as usize;
    let mut output = vec![0i32; num_coefficients];
    let num_ctx_groups = num_context_groups(tile.num_levels);

    // Build per-expanded-group slot-to-symbol lookup tables
    let slot_tables: Vec<Vec<u16>> = tile
        .groups
        .iter()
        .map(|g| {
            let mut table = vec![0u16; RANS_M as usize];
            let asize = g.alphabet_size as usize;
            for sym in 0..asize {
                for j in g.cumfreqs[sym]..g.cumfreqs[sym + 1] {
                    table[j as usize] = sym as u16;
                }
            }
            table
        })
        .collect();

    // Check if any group uses ZRL
    let has_zrl = tile.groups.iter().any(|g| g.zrun_base != 0);

    for s in 0..STREAMS_PER_TILE {
        if s >= num_coefficients {
            break;
        }
        let stream_output_count = 1 + (num_coefficients - 1 - s) / STREAMS_PER_TILE;

        let mut state = tile.stream_initial_state[s];
        let buf = &tile.stream_data[s];
        let mut ptr: usize = 0;
        let mask = RANS_M - 1;

        if has_zrl {
            let mut output_i = 0;
            while output_i < stream_output_count {
                let coeff_idx = s + output_i * STREAMS_PER_TILE;
                let lx = (coeff_idx % ts) as u32;
                let ly = (coeff_idx / ts) as u32;
                let g = compute_subband_group(lx, ly, tile.tile_size, tile.num_levels);

                // Reconstruct context from already-decoded output
                let ctx = compute_above_context(&output, lx, ly, tile.tile_size, tile.num_levels);
                let eg = context_group_index(g, ctx);
                let eg = eg.min(num_ctx_groups - 1);

                let group = &tile.groups[eg];

                let slot = state & mask;
                let sym = slot_tables[eg][slot as usize] as usize;
                let value = sym as i32 + group.min_val;

                let start = group.cumfreqs[sym];
                let freq = group.cumfreqs[sym + 1] - group.cumfreqs[sym];
                state = freq * (state >> RANS_PRECISION) + (state & mask) - start;

                while state < RANS_BYTE_L {
                    if ptr < buf.len() {
                        state = (state << 8) | buf[ptr] as u32;
                        ptr += 1;
                    } else {
                        break;
                    }
                }

                if group.zrun_base != 0 && value >= group.zrun_base {
                    let run_len = (value - group.zrun_base + 1) as usize;
                    for j in 0..run_len {
                        output[s + (output_i + j) * STREAMS_PER_TILE] = 0;
                    }
                    output_i += run_len;
                } else {
                    output[coeff_idx] = value;
                    output_i += 1;
                }
            }
        } else {
            for i in 0..stream_output_count {
                let coeff_idx = s + i * STREAMS_PER_TILE;
                let lx = (coeff_idx % ts) as u32;
                let ly = (coeff_idx / ts) as u32;
                let g = compute_subband_group(lx, ly, tile.tile_size, tile.num_levels);

                let ctx = compute_above_context(&output, lx, ly, tile.tile_size, tile.num_levels);
                let eg = context_group_index(g, ctx);
                let eg = eg.min(num_ctx_groups - 1);

                let group = &tile.groups[eg];

                let slot = state & mask;
                let sym = slot_tables[eg][slot as usize] as usize;

                output[coeff_idx] = sym as i32 + group.min_val;

                let start = group.cumfreqs[sym];
                let freq = group.cumfreqs[sym + 1] - group.cumfreqs[sym];
                state = freq * (state >> RANS_PRECISION) + (state & mask) - start;

                while state < RANS_BYTE_L {
                    if ptr < buf.len() {
                        state = (state << 8) | buf[ptr] as u32;
                        ptr += 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    output
}

/// Serialize a SubbandRansTile to bytes.
pub fn serialize_tile_subband(tile: &SubbandRansTile) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&tile.num_coefficients.to_le_bytes());
    out.extend_from_slice(&tile.tile_size.to_le_bytes());
    out.extend_from_slice(&tile.num_levels.to_le_bytes());
    out.extend_from_slice(&tile.num_groups.to_le_bytes());
    // Per-group frequency tables (with zrun_base)
    for g in &tile.groups {
        out.extend_from_slice(&g.min_val.to_le_bytes());
        out.extend_from_slice(&g.alphabet_size.to_le_bytes());
        out.extend_from_slice(&g.zrun_base.to_le_bytes());
        for &f in &g.freqs {
            out.extend_from_slice(&(f as u16).to_le_bytes());
        }
    }
    // 32 stream data lengths
    for s in 0..STREAMS_PER_TILE {
        let len = tile.stream_data[s].len() as u32;
        out.extend_from_slice(&len.to_le_bytes());
    }
    // 32 initial states
    for s in 0..STREAMS_PER_TILE {
        out.extend_from_slice(&tile.stream_initial_state[s].to_le_bytes());
    }
    // Stream data (concatenated)
    for s in 0..STREAMS_PER_TILE {
        out.extend_from_slice(&tile.stream_data[s]);
    }
    out
}

/// Deserialize a SubbandRansTile from bytes. Returns (tile, bytes_consumed).
pub fn deserialize_tile_subband(data: &[u8]) -> (SubbandRansTile, usize) {
    let mut pos = 0;

    let num_coefficients = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let tile_size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let num_levels = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let num_groups = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    let mut groups = Vec::with_capacity(num_groups as usize);
    for _ in 0..num_groups {
        let min_val = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let alphabet_size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let zrun_base = i32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;

        let mut freqs = Vec::with_capacity(alphabet_size as usize);
        for _ in 0..alphabet_size {
            let f = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
            freqs.push(f as u32);
            pos += 2;
        }

        let mut cumfreqs = vec![0u32; alphabet_size as usize + 1];
        for i in 0..alphabet_size as usize {
            cumfreqs[i + 1] = cumfreqs[i] + freqs[i];
        }

        groups.push(SubbandGroupFreqs {
            min_val,
            alphabet_size,
            zrun_base,
            freqs,
            cumfreqs,
        });
    }

    let mut stream_lengths = Vec::with_capacity(STREAMS_PER_TILE);
    for _ in 0..STREAMS_PER_TILE {
        let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        stream_lengths.push(len);
        pos += 4;
    }

    let mut stream_initial_state = Vec::with_capacity(STREAMS_PER_TILE);
    for _ in 0..STREAMS_PER_TILE {
        let state = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        stream_initial_state.push(state);
        pos += 4;
    }

    let mut stream_data = Vec::with_capacity(STREAMS_PER_TILE);
    for len in &stream_lengths {
        let len = *len as usize;
        stream_data.push(data[pos..pos + len].to_vec());
        pos += len;
    }

    (
        SubbandRansTile {
            num_coefficients,
            tile_size,
            num_levels,
            num_groups,
            groups,
            stream_data,
            stream_initial_state,
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

    #[test]
    fn test_interleaved_roundtrip_simple() {
        let coefficients = vec![0, 1, -1, 0, 2, -2, 0, 0, 1, -1, 0, 3];
        let tile = rans_encode_tile_interleaved(&coefficients);
        let decoded = rans_decode_tile_interleaved(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_interleaved_roundtrip_large() {
        // 65536 coefficients = standard tile size, simulating wavelet output
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
        let tile = rans_encode_tile_interleaved(&coefficients);
        let decoded = rans_decode_tile_interleaved(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_interleaved_matches_standard() {
        // Verify interleaved decode produces same output as standard decode
        let mut coefficients = vec![0i32; 65536];
        for i in (0..65536).step_by(3) {
            coefficients[i] = (i % 20) as i32 - 10;
        }
        let standard_tile = rans_encode_tile(&coefficients);
        let standard_decoded = rans_decode_tile(&standard_tile);

        let interleaved_tile = rans_encode_tile_interleaved(&coefficients);
        let interleaved_decoded = rans_decode_tile_interleaved(&interleaved_tile);

        assert_eq!(standard_decoded, interleaved_decoded);
    }

    #[test]
    fn test_interleaved_serialize_roundtrip() {
        let mut coefficients = Vec::new();
        for i in 0..65536 {
            coefficients.push(if i % 4 == 0 { (i % 10) as i32 - 5 } else { 0 });
        }
        let tile = rans_encode_tile_interleaved(&coefficients);
        let serialized = serialize_tile_interleaved(&tile);
        let (deserialized, consumed) = deserialize_tile_interleaved(&serialized);
        assert_eq!(consumed, serialized.len());
        let decoded = rans_decode_tile_interleaved(&deserialized);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_interleaved_zeros() {
        let coefficients = vec![0; 65536];
        let tile = rans_encode_tile_interleaved(&coefficients);
        let decoded = rans_decode_tile_interleaved(&tile);
        assert_eq!(coefficients, decoded);
        // All zeros should compress very well even with 32 stream overhead
        let total_stream_bytes: usize = tile.stream_data.iter().map(|s| s.len()).sum();
        assert!(
            total_stream_bytes < 200,
            "Stream data {} should be small for all-zero input",
            total_stream_bytes
        );
    }

    // --- Zero-run-length coding tests ---

    #[test]
    fn test_zrl_encode_decode_basic() {
        let coefficients = vec![0, 0, 0, 5, -3, 0, 0, 7, 0];
        let zrun_base = 8; // max_abs=7, zrun_base=8
        let encoded = zero_run_encode(&coefficients, zrun_base);
        // run(3)=10, 5, -3, run(2)=9, 7, run(1)=8
        assert_eq!(encoded, vec![10, 5, -3, 9, 7, 8]);
        let decoded = zero_run_decode(&encoded, zrun_base, coefficients.len());
        assert_eq!(decoded, coefficients);
    }

    #[test]
    fn test_zrl_all_zeros() {
        let coefficients = vec![0; 1000];
        let zrun_base = 1; // max_abs=0, zrun_base=1
        let encoded = zero_run_encode(&coefficients, zrun_base);
        // 1000 zeros = 3 runs of 256 + 1 run of 232
        assert_eq!(encoded.len(), 4);
        let decoded = zero_run_decode(&encoded, zrun_base, 1000);
        assert_eq!(decoded, coefficients);
    }

    #[test]
    fn test_zrl_no_zeros() {
        let coefficients = vec![1, -2, 3, -4, 5];
        let zrun_base = 6;
        let encoded = zero_run_encode(&coefficients, zrun_base);
        assert_eq!(encoded, coefficients); // no change
        let decoded = zero_run_decode(&encoded, zrun_base, coefficients.len());
        assert_eq!(decoded, coefficients);
    }

    #[test]
    fn test_zrl_interleaved_roundtrip_simple() {
        // Adaptive wrapper: may pick plain or ZRL, but must roundtrip correctly
        let coefficients = vec![0, 1, -1, 0, 2, -2, 0, 0, 1, -1, 0, 3];
        let tile = rans_encode_tile_interleaved_zrl(&coefficients);
        let decoded = rans_decode_tile_interleaved(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_zrl_inner_roundtrip() {
        // Test the inner ZRL encoder directly (always applies ZRL)
        let coefficients = vec![0, 1, -1, 0, 2, -2, 0, 0, 1, -1, 0, 3];
        let tile = rans_encode_tile_interleaved_zrl_inner(&coefficients);
        assert_ne!(tile.zrun_base, 0);
        let decoded = rans_decode_tile_interleaved(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_zrl_interleaved_roundtrip_large() {
        // 65536 coefficients = standard tile, mostly zeros (simulating wavelet output)
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
        let tile = rans_encode_tile_interleaved_zrl(&coefficients);
        let decoded = rans_decode_tile_interleaved(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_zrl_all_zeros_roundtrip() {
        // Inner ZRL encoder on all-zeros data
        let coefficients = vec![0; 65536];
        let tile = rans_encode_tile_interleaved_zrl_inner(&coefficients);
        assert_eq!(tile.zrun_base, 1);
        let decoded = rans_decode_tile_interleaved(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_zrl_serialize_roundtrip() {
        let mut coefficients = Vec::new();
        for i in 0..65536 {
            coefficients.push(if i % 4 == 0 { (i % 10) as i32 - 5 } else { 0 });
        }
        let tile = rans_encode_tile_interleaved_zrl(&coefficients);
        let serialized = serialize_tile_interleaved(&tile);
        let (deserialized, consumed) = deserialize_tile_interleaved(&serialized);
        assert_eq!(consumed, serialized.len());
        assert_eq!(deserialized.zrun_base, tile.zrun_base);
        let decoded = rans_decode_tile_interleaved(&deserialized);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_zrl_compression_improvement() {
        // Zero-heavy data: ZRL stream data should be smaller (test inner encoder)
        let mut coefficients = vec![0i32; 65536];
        for i in (0..65536).step_by(100) {
            coefficients[i] = (i % 5) as i32 - 2;
        }
        let tile_no_zrl = rans_encode_tile_interleaved(&coefficients);
        let tile_zrl = rans_encode_tile_interleaved_zrl_inner(&coefficients);

        let no_zrl_bytes: usize = tile_no_zrl.stream_data.iter().map(|s| s.len()).sum();
        let zrl_bytes: usize = tile_zrl.stream_data.iter().map(|s| s.len()).sum();

        assert!(
            zrl_bytes < no_zrl_bytes,
            "ZRL stream data {} should be smaller than non-ZRL {}",
            zrl_bytes,
            no_zrl_bytes
        );

        // Verify both decode correctly
        assert_eq!(
            rans_decode_tile_interleaved(&tile_no_zrl),
            rans_decode_tile_interleaved(&tile_zrl)
        );
    }

    // --- Per-subband entropy coding tests ---

    #[test]
    fn test_subband_group_mapping() {
        // 256x256 tile, 3 levels → groups: 0=LL, 1=level0, 2=level1, 3=level2
        // (0,0) = LL = group 0
        assert_eq!(compute_subband_group(0, 0, 256, 3), 0);
        // (128,128) = level 0 HH = group 1
        assert_eq!(compute_subband_group(128, 128, 256, 3), 1);
        // (128,0) = level 0 HL = group 1
        assert_eq!(compute_subband_group(128, 0, 256, 3), 1);
        // (0,128) = level 0 LH = group 1
        assert_eq!(compute_subband_group(0, 128, 256, 3), 1);
        // (64,0) = level 1 HL = group 2
        assert_eq!(compute_subband_group(64, 0, 256, 3), 2);
        // (32,0) = level 2 HL = group 3
        assert_eq!(compute_subband_group(32, 0, 256, 3), 3);
        // (0,0) still LL with 1 level
        assert_eq!(compute_subband_group(0, 0, 8, 1), 0);
        // (4,0) = level 0 HL with tile_size=8, 1 level = group 1
        assert_eq!(compute_subband_group(4, 0, 8, 1), 1);
    }

    #[test]
    fn test_subband_roundtrip_simple() {
        // 256x256 tile (65536 coefficients), 3 levels
        let mut coefficients = vec![0i32; 65536];
        // Set some LL coefficients (top-left 32x32) to larger values
        for y in 0..32 {
            for x in 0..32 {
                coefficients[y * 256 + x] = ((x + y) % 20) as i32 + 5;
            }
        }
        // Scatter some detail coefficients
        for i in (8192..65536).step_by(7) {
            coefficients[i] = (i % 11) as i32 - 5;
        }

        let tile = rans_encode_tile_interleaved_subband(&coefficients, 256, 3);
        assert_eq!(tile.num_groups, 4);
        let decoded = rans_decode_tile_interleaved_subband(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_subband_roundtrip_all_zeros() {
        let coefficients = vec![0i32; 65536];
        let tile = rans_encode_tile_interleaved_subband(&coefficients, 256, 3);
        let decoded = rans_decode_tile_interleaved_subband(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_subband_roundtrip_varied() {
        // Simulate realistic wavelet output: LL large, detail mostly zeros
        let mut coefficients = vec![0i32; 65536];
        for i in 0..65536 {
            let y = i / 256;
            let x = i % 256;
            let g = compute_subband_group(x as u32, y as u32, 256, 3);
            coefficients[i] = match g {
                0 => ((x + y) % 40) as i32 + 10, // LL: large positive
                1 => {
                    // Level 0 detail: mostly zeros
                    if i % 5 == 0 {
                        (i % 7) as i32 - 3
                    } else {
                        0
                    }
                }
                2 => {
                    // Level 1 detail: moderate zeros
                    if i % 3 == 0 {
                        (i % 9) as i32 - 4
                    } else {
                        0
                    }
                }
                _ => {
                    // Level 2 detail: fewer zeros
                    if i % 2 == 0 {
                        (i % 5) as i32 - 2
                    } else {
                        0
                    }
                }
            };
        }

        let tile = rans_encode_tile_interleaved_subband(&coefficients, 256, 3);
        let decoded = rans_decode_tile_interleaved_subband(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_subband_serialize_roundtrip() {
        let mut coefficients = vec![0i32; 65536];
        for i in (0..65536).step_by(3) {
            coefficients[i] = (i % 15) as i32 - 7;
        }
        let tile = rans_encode_tile_interleaved_subband(&coefficients, 256, 3);
        let serialized = serialize_tile_subband(&tile);
        let (deserialized, consumed) = deserialize_tile_subband(&serialized);
        assert_eq!(consumed, serialized.len());
        assert_eq!(deserialized.num_groups, tile.num_groups);
        assert_eq!(deserialized.tile_size, tile.tile_size);
        assert_eq!(deserialized.num_levels, tile.num_levels);
        let decoded = rans_decode_tile_interleaved_subband(&deserialized);
        assert_eq!(coefficients, decoded, "serialize roundtrip failed");
    }

    #[test]
    fn test_subband_matches_single_table() {
        // Per-subband and single-table should produce the same decoded output
        // (both lossless for the same integer coefficients)
        let mut coefficients = vec![0i32; 65536];
        for i in (0..65536).step_by(5) {
            coefficients[i] = (i % 20) as i32 - 10;
        }
        let single = rans_encode_tile_interleaved(&coefficients);
        let subband = rans_encode_tile_interleaved_subband(&coefficients, 256, 3);
        let decoded_single = rans_decode_tile_interleaved(&single);
        let decoded_subband = rans_decode_tile_interleaved_subband(&subband);
        assert_eq!(decoded_single, decoded_subband);
    }

    #[test]
    fn test_subband_compression_improvement() {
        // Per-subband should compress better than single-table for wavelet-like data
        let mut coefficients = vec![0i32; 65536];
        // LL: large positive narrow range
        for y in 0..32 {
            for x in 0..32 {
                coefficients[y * 256 + x] = 100 + ((x + y) % 10) as i32;
            }
        }
        // Detail: mostly zeros with occasional small values
        for i in 0..65536 {
            let y = i / 256;
            let x = i % 256;
            if compute_subband_group(x as u32, y as u32, 256, 3) > 0 && i % 10 == 0 {
                coefficients[i] = (i % 5) as i32 - 2;
            }
        }

        let single = rans_encode_tile_interleaved(&coefficients);
        let subband = rans_encode_tile_interleaved_subband(&coefficients, 256, 3);

        let single_bytes = single.byte_size();
        let subband_bytes = subband.byte_size();
        assert!(
            subband_bytes < single_bytes,
            "Per-subband {} should be smaller than single-table {} for wavelet-like data",
            subband_bytes,
            single_bytes
        );
    }

    // --- Per-subband ZRL tests ---

    #[test]
    fn test_subband_zrl_roundtrip_with_large_detail() {
        // Create data where detail subbands have large value ranges and high zero density,
        // which triggers ZRL encoding (alphabet >= 16 and zero_frac >= 0.6).
        let mut coefficients = vec![0i32; 65536];
        // LL: narrow range, no zeros
        for y in 0..32 {
            for x in 0..32 {
                coefficients[y * 256 + x] = 50 + ((x + y) % 10) as i32;
            }
        }
        // Detail subbands: wide range (-15..15) but mostly zeros (95%)
        for i in 0..65536 {
            let y = i / 256;
            let x = i % 256;
            if compute_subband_group(x as u32, y as u32, 256, 3) > 0 && i % 20 == 0 {
                coefficients[i] = ((i as i32 * 7) % 31) - 15;
            }
        }

        let tile = rans_encode_tile_interleaved_subband(&coefficients, 256, 3);

        // Verify ZRL is active in at least one detail group
        let zrl_active = tile.groups.iter().skip(1).any(|g| g.zrun_base != 0);
        assert!(
            zrl_active,
            "Expected ZRL to be active in at least one detail subband group"
        );

        // Verify LL group does NOT use ZRL
        assert_eq!(tile.groups[0].zrun_base, 0, "LL group should not use ZRL");

        // Roundtrip
        let decoded = rans_decode_tile_interleaved_subband(&tile);
        assert_eq!(coefficients, decoded, "ZRL subband roundtrip failed");
    }

    #[test]
    fn test_subband_zrl_serialize_roundtrip() {
        // Test that serialization preserves per-group zrun_base
        let mut coefficients = vec![0i32; 65536];
        for y in 0..32 {
            for x in 0..32 {
                coefficients[y * 256 + x] = 50 + ((x + y) % 10) as i32;
            }
        }
        // Wide-range, high zero density detail coefficients
        for i in 0..65536 {
            let y = i / 256;
            let x = i % 256;
            if compute_subband_group(x as u32, y as u32, 256, 3) > 0 && i % 20 == 0 {
                coefficients[i] = ((i as i32 * 7) % 31) - 15;
            }
        }

        let tile = rans_encode_tile_interleaved_subband(&coefficients, 256, 3);
        let serialized = serialize_tile_subband(&tile);
        let (deserialized, consumed) = deserialize_tile_subband(&serialized);
        assert_eq!(consumed, serialized.len());

        // Verify per-group zrun_base is preserved
        for (g, (orig, deser)) in tile
            .groups
            .iter()
            .zip(deserialized.groups.iter())
            .enumerate()
        {
            assert_eq!(
                orig.zrun_base, deser.zrun_base,
                "Group {} zrun_base mismatch after serialization",
                g
            );
        }

        let decoded = rans_decode_tile_interleaved_subband(&deserialized);
        assert_eq!(
            coefficients, decoded,
            "ZRL subband serialize roundtrip failed"
        );
    }

    #[test]
    fn test_subband_zrl_no_zrl_for_small_alphabet() {
        // Detail subbands with small value range (< 16 distinct values) should NOT use ZRL
        let mut coefficients = vec![0i32; 65536];
        for y in 0..32 {
            for x in 0..32 {
                coefficients[y * 256 + x] = 50 + ((x + y) % 10) as i32;
            }
        }
        // Detail: values -2..2 (alphabet=5, below 16 threshold)
        for i in 0..65536 {
            let y = i / 256;
            let x = i % 256;
            if compute_subband_group(x as u32, y as u32, 256, 3) > 0 && i % 10 == 0 {
                coefficients[i] = (i % 5) as i32 - 2;
            }
        }

        let tile = rans_encode_tile_interleaved_subband(&coefficients, 256, 3);

        // No group should use ZRL with such a small alphabet
        for (g, group) in tile.groups.iter().enumerate() {
            assert_eq!(
                group.zrun_base, 0,
                "Group {} should not use ZRL with small alphabet (alphabet_size={})",
                g, group.alphabet_size
            );
        }

        // Roundtrip should still work
        let decoded = rans_decode_tile_interleaved_subband(&tile);
        assert_eq!(coefficients, decoded);
    }

    // --- Context-adaptive entropy coding tests ---

    #[test]
    fn test_ctx_adaptive_roundtrip_simple() {
        // 256x256 tile, 3 levels
        let mut coefficients = vec![0i32; 65536];
        for y in 0..32 {
            for x in 0..32 {
                coefficients[y * 256 + x] = ((x + y) % 20) as i32 + 5;
            }
        }
        for i in (8192..65536).step_by(7) {
            coefficients[i] = (i % 11) as i32 - 5;
        }

        let tile = rans_encode_tile_interleaved_subband_ctx(&coefficients, 256, 3);
        // Should have 1 + 3*2 = 7 expanded groups
        assert_eq!(tile.num_groups, 7);
        let decoded = rans_decode_tile_interleaved_subband_ctx(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_ctx_adaptive_roundtrip_all_zeros() {
        let coefficients = vec![0i32; 65536];
        let tile = rans_encode_tile_interleaved_subband_ctx(&coefficients, 256, 3);
        let decoded = rans_decode_tile_interleaved_subband_ctx(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_ctx_adaptive_roundtrip_varied() {
        // Realistic wavelet output
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
                2 => {
                    if i % 3 == 0 {
                        (i % 9) as i32 - 4
                    } else {
                        0
                    }
                }
                _ => {
                    if i % 2 == 0 {
                        (i % 15) as i32 - 7
                    } else {
                        0
                    }
                }
            };
        }

        let tile = rans_encode_tile_interleaved_subband_ctx(&coefficients, 256, 3);
        let decoded = rans_decode_tile_interleaved_subband_ctx(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_ctx_adaptive_serialize_roundtrip() {
        let mut coefficients = vec![0i32; 65536];
        for y in 0..32 {
            for x in 0..32 {
                coefficients[y * 256 + x] = 50 + ((x + y) % 10) as i32;
            }
        }
        for i in 0..65536 {
            let y = i / 256;
            let x = i % 256;
            if compute_subband_group(x as u32, y as u32, 256, 3) > 0 && i % 10 == 0 {
                coefficients[i] = ((i as i32 * 3) % 21) - 10;
            }
        }

        let tile = rans_encode_tile_interleaved_subband_ctx(&coefficients, 256, 3);
        let serialized = serialize_tile_subband(&tile);
        let (deserialized, consumed) = deserialize_tile_subband(&serialized);
        assert_eq!(consumed, serialized.len());
        let decoded = rans_decode_tile_interleaved_subband_ctx(&deserialized);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_ctx_adaptive_compression_improvement() {
        // Realistic wavelet-like data with spatial correlation (above-neighbor context should help)
        let mut coefficients = vec![0i32; 65536];
        // LL: smooth slowly-varying values
        for y in 0..32 {
            for x in 0..32 {
                coefficients[y * 256 + x] = 100 + ((x + y) % 20) as i32;
            }
        }
        // Detail subbands: spatially correlated sparsity — rows of zeros followed by rows
        // with values. This pattern benefits from above-neighbor context.
        for i in 0..65536 {
            let y = i / 256;
            let x = i % 256;
            let g = compute_subband_group(x as u32, y as u32, 256, 3);
            if g > 0 {
                // Create spatially correlated patterns: zero rows alternate with value rows
                let local_y = y % 32; // position within subband region
                if local_y % 4 < 2 {
                    // Zero rows
                    coefficients[i] = 0;
                } else {
                    // Value rows: small coefficients
                    coefficients[i] = ((x + y) % 7) as i32 - 3;
                }
            }
        }

        let plain = rans_encode_tile_interleaved_subband(&coefficients, 256, 3);
        let ctx = rans_encode_tile_interleaved_subband_ctx(&coefficients, 256, 3);

        let plain_bytes = plain.byte_size();
        let ctx_bytes = ctx.byte_size();

        // Context-adaptive should be smaller (or at least not worse) for correlated data
        let savings_pct = 100.0 * (1.0 - ctx_bytes as f64 / plain_bytes as f64);
        eprintln!(
            "Context-adaptive: {} bytes vs plain: {} bytes ({:.1}% savings)",
            ctx_bytes, plain_bytes, savings_pct
        );

        // Verify roundtrip
        let decoded = rans_decode_tile_interleaved_subband_ctx(&ctx);
        assert_eq!(coefficients, decoded);

        // We expect some improvement for spatially correlated data
        assert!(
            ctx_bytes <= plain_bytes,
            "Context-adaptive ({}) should not be worse than plain ({}) for correlated data",
            ctx_bytes,
            plain_bytes
        );
    }
}
