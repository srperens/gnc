// GPU Rice encoder — 256 interleaved streams per tile.
//
// Significance map + Golomb-Rice coding. Fully parallel: no state chain.
// Each thread encodes one of 256 independent bit streams.
//
// Phase 1: Cooperatively compute optimal Rice parameter k per subband group.
// Phase 2: Each thread encodes its stream's coefficients.
//
// Per coefficient:
//   Zero:     1 bit (0)
//   Non-zero: 1 bit (1) + 1 sign bit + unary(quotient) + k bits (remainder)
//   where quotient = (|val|-1) >> k, remainder = (|val|-1) & ((1<<k)-1)

const STREAMS_PER_TILE: u32 = 256u;
const MAX_STREAM_BYTES: u32 = 2048u;
const MAX_STREAM_WORDS: u32 = 512u;  // MAX_STREAM_BYTES / 4
const MAX_GROUPS: u32 = 8u;

struct Params {
    num_tiles: u32,
    coefficients_per_tile: u32,
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    num_levels: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> stream_output: array<u32>;
@group(0) @binding(3) var<storage, read_write> stream_lengths: array<u32>;
@group(0) @binding(4) var<storage, read_write> k_output: array<u32>;

// Shared memory for Phase 1: k computation
var<workgroup> group_sum: array<atomic<u32>, 8>;
var<workgroup> group_count: array<atomic<u32>, 8>;
var<workgroup> shared_k: array<u32, 8>;

// Per-thread bit-packing state
var<private> p_bit_buffer: u32;
var<private> p_bits_in_buffer: u32;
var<private> p_word_buffer: u32;
var<private> p_bytes_in_word: u32;
var<private> p_word_pos: u32;
var<private> p_stream_word_base: u32;
var<private> p_total_bytes: u32;

// Directional subband grouping — matches CPU compute_subband_group exactly.
fn compute_subband_group(lx: u32, ly: u32) -> u32 {
    var region = params.tile_size;
    for (var level = 0u; level < params.num_levels; level++) {
        let half = region / 2u;
        if (lx >= half || ly >= half) {
            let lfd = params.num_levels - 1u - level;
            if (lfd == 0u) {
                return 1u;
            }
            let is_hh = (lx >= half) && (ly >= half);
            let base = 2u + (lfd - 1u) * 2u;
            return select(base, base + 1u, is_hh);
        }
        region = half;
    }
    return 0u;
}

// Emit a complete byte to the stream output buffer.
fn emit_byte(byte_val: u32) {
    p_word_buffer = p_word_buffer | ((byte_val & 0xFFu) << (p_bytes_in_word * 8u));
    p_bytes_in_word += 1u;
    p_total_bytes += 1u;
    if (p_bytes_in_word == 4u) {
        stream_output[p_stream_word_base + p_word_pos] = p_word_buffer;
        p_word_pos += 1u;
        p_word_buffer = 0u;
        p_bytes_in_word = 0u;
    }
}

// Emit a single bit (MSB-first packing). Flushes bytes automatically.
fn emit_bit(bit: u32) {
    p_bit_buffer = (p_bit_buffer << 1u) | (bit & 1u);
    p_bits_in_buffer += 1u;
    if (p_bits_in_buffer == 8u) {
        emit_byte(p_bit_buffer);
        p_bits_in_buffer = 0u;
        p_bit_buffer = 0u;
    }
}

// Emit multiple bits at once (MSB-first). count must be <= 15.
// Safe because bits_in_buffer <= 7, so total <= 22 < 32.
fn emit_bits(value: u32, count: u32) {
    p_bit_buffer = (p_bit_buffer << count) | (value & ((1u << count) - 1u));
    p_bits_in_buffer += count;
    while (p_bits_in_buffer >= 8u) {
        p_bits_in_buffer -= 8u;
        let bv = (p_bit_buffer >> p_bits_in_buffer) & 0xFFu;
        emit_byte(bv);
    }
    if (p_bits_in_buffer > 0u) {
        p_bit_buffer = p_bit_buffer & ((1u << p_bits_in_buffer) - 1u);
    } else {
        p_bit_buffer = 0u;
    }
}

// Flush remaining bits (pad to byte boundary) and partial word.
fn flush_remaining() {
    if (p_bits_in_buffer > 0u) {
        let byte_val = (p_bit_buffer << (8u - p_bits_in_buffer)) & 0xFFu;
        emit_byte(byte_val);
    }
    if (p_bytes_in_word > 0u) {
        stream_output[p_stream_word_base + p_word_pos] = p_word_buffer;
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let thread_id = lid.x;
    let tile_id = wid.x;

    if (tile_id >= params.num_tiles) {
        return;
    }

    let tile_x = tile_id % params.tiles_x;
    let tile_y = tile_id / params.tiles_x;
    let tile_origin_x = tile_x * params.tile_size;
    let tile_origin_y = tile_y * params.tile_size;

    let symbols_per_stream = params.coefficients_per_tile / STREAMS_PER_TILE;
    let num_groups = params.num_levels * 2u;

    // === Phase 1: Compute optimal k per subband group ===

    // Initialize shared accumulators
    if (thread_id < MAX_GROUPS) {
        atomicStore(&group_sum[thread_id], 0u);
        atomicStore(&group_count[thread_id], 0u);
    }
    workgroupBarrier();

    // Each thread scans its stream's coefficients, accumulates per-group stats
    for (var s = 0u; s < symbols_per_stream; s++) {
        let coeff_idx = thread_id + s * STREAMS_PER_TILE;
        let tile_row = coeff_idx / params.tile_size;
        let tile_col = coeff_idx % params.tile_size;
        let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                      + (tile_origin_x + tile_col);
        let coeff = i32(round(input[plane_idx]));

        if (coeff != 0) {
            let abs_val = u32(abs(coeff)) - 1u;
            let g = compute_subband_group(tile_col, tile_row);
            // Clamp atomic adds to avoid overflow on extreme content
            atomicAdd(&group_sum[g], min(abs_val, 65535u));
            atomicAdd(&group_count[g], 1u);
        }
    }
    workgroupBarrier();

    // First num_groups threads compute k = floor(log2(mean))
    if (thread_id < num_groups) {
        let count = atomicLoad(&group_count[thread_id]);
        if (count > 0u) {
            let mean = atomicLoad(&group_sum[thread_id]) / count;
            if (mean == 0u) {
                shared_k[thread_id] = 0u;
            } else {
                shared_k[thread_id] = min(31u - countLeadingZeros(mean), 15u);
            }
        } else {
            shared_k[thread_id] = 0u;
        }
    }
    workgroupBarrier();

    // Write k values to output (one thread per group)
    if (thread_id < num_groups) {
        k_output[tile_id * MAX_GROUPS + thread_id] = shared_k[thread_id];
    }

    // === Phase 2: Encode stream ===

    // Initialize per-thread state
    p_stream_word_base = (tile_id * STREAMS_PER_TILE + thread_id) * MAX_STREAM_WORDS;
    p_bit_buffer = 0u;
    p_bits_in_buffer = 0u;
    p_word_buffer = 0u;
    p_bytes_in_word = 0u;
    p_word_pos = 0u;
    p_total_bytes = 0u;

    for (var s = 0u; s < symbols_per_stream; s++) {
        let coeff_idx = thread_id + s * STREAMS_PER_TILE;
        let tile_row = coeff_idx / params.tile_size;
        let tile_col = coeff_idx % params.tile_size;
        let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                      + (tile_origin_x + tile_col);
        let coeff = i32(round(input[plane_idx]));

        if (coeff == 0) {
            emit_bit(0u);
        } else {
            // Significance: non-zero
            emit_bit(1u);

            // Sign bit: 1 = negative, 0 = positive
            emit_bit(select(0u, 1u, coeff < 0));

            let magnitude = u32(abs(coeff)) - 1u;
            let g = compute_subband_group(tile_col, tile_row);
            let k = shared_k[g];

            // Unary code: quotient 1-bits + terminating 0-bit (bulk emit)
            let quotient = min(magnitude >> k, 31u);
            var q_remaining = quotient;
            while (q_remaining > 0u) {
                let chunk = min(q_remaining, 15u);
                emit_bits((1u << chunk) - 1u, chunk);
                q_remaining -= chunk;
            }
            emit_bit(0u);

            // Fixed-length remainder: k bits
            if (k > 0u) {
                let remainder = magnitude & ((1u << k) - 1u);
                emit_bits(remainder, k);
            }
        }
    }

    flush_remaining();

    // Write stream byte count
    stream_lengths[tile_id * STREAMS_PER_TILE + thread_id] = p_total_bytes;
}
