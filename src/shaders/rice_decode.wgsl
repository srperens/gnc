// GPU Rice decoder — 256 interleaved streams per tile.
//
// Inverse of rice_encode.wgsl. Each thread decodes one independent bit stream.
// No state chain — fully parallel across all 256 streams.
//
// Per coefficient:
//   bit=0 → zero
//   bit=1 → sign bit + Rice(magnitude-1, k) → ±magnitude

const STREAMS_PER_TILE: u32 = 256u;
const MAX_GROUPS: u32 = 8u;
const K_STRIDE: u32 = 16u;  // MAX_GROUPS * 2: stride per tile in k_values (mag k + zrl k per group)

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
@group(0) @binding(1) var<storage, read> k_values: array<u32>;
@group(0) @binding(2) var<storage, read> stream_data: array<u32>;
@group(0) @binding(3) var<storage, read> stream_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Shared k values for this tile (magnitude k per group + zrl k per group)
var<workgroup> shared_k: array<u32, 8>;
var<workgroup> shared_k_zrl: array<u32, 8>;

// Per-thread bit-reader state
var<private> p_current_byte: u32;
var<private> p_bit_pos: u32;     // 0..8, position within current byte
var<private> p_byte_offset: u32; // absolute byte offset in stream_data

// Per-thread EMA state for adaptive k (fixed-point ×16, window ≈ 8 coefficients)
var<private> p_ema: array<u32, 8>;

// Directional subband grouping — must match encoder exactly.
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

// Read one byte from the packed u32 stream data buffer.
fn load_byte(byte_off: u32) -> u32 {
    let word_idx = byte_off >> 2u;
    let byte_pos = byte_off & 3u;
    return (stream_data[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

// Read a single bit (MSB-first within each byte).
fn read_bit() -> u32 {
    if (p_bit_pos == 8u) {
        p_current_byte = load_byte(p_byte_offset);
        p_byte_offset += 1u;
        p_bit_pos = 0u;
    }
    let bit = (p_current_byte >> (7u - p_bit_pos)) & 1u;
    p_bit_pos += 1u;
    return bit;
}

// Read multiple bits (MSB-first). count must be <= 15.
// Bulk extraction: grabs remaining bits from current byte before loading next.
fn read_bits(count: u32) -> u32 {
    var value = 0u;
    var remaining = count;
    while (remaining > 0u) {
        if (p_bit_pos == 8u) {
            p_current_byte = load_byte(p_byte_offset);
            p_byte_offset += 1u;
            p_bit_pos = 0u;
        }
        let avail = 8u - p_bit_pos;
        let take = min(remaining, avail);
        // Extract 'take' MSB-first bits from current position
        let shift = avail - take;
        let bits = (p_current_byte >> shift) & ((1u << take) - 1u);
        value = (value << take) | bits;
        p_bit_pos += take;
        remaining -= take;
    }
    return value;
}

// Read a Golomb-Rice coded value: unary quotient + k-bit remainder.
fn read_rice(k: u32) -> u32 {
    var quotient = 0u;
    while (read_bit() == 1u) {
        quotient += 1u;
    }
    let remainder = select(0u, read_bits(k), k > 0u);
    return (quotient << k) | remainder;
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
    let num_groups = max(1u, params.num_levels * 2u);

    // Cooperatively load k values + per-subband k_zrl into shared memory (stride K_STRIDE)
    if (thread_id < num_groups) {
        shared_k[thread_id] = k_values[tile_id * K_STRIDE + thread_id];
        shared_k_zrl[thread_id] = k_values[tile_id * K_STRIDE + MAX_GROUPS + thread_id];
    }
    workgroupBarrier();

    // Initialize EMA from static k seeds (per-thread private, 8 groups)
    for (var gi = 0u; gi < MAX_GROUPS; gi++) {
        p_ema[gi] = max(1u, 1u << shared_k[gi]) << 4u;
    }

    // Initialize bit reader with this stream's byte offset
    let stream_idx = tile_id * STREAMS_PER_TILE + thread_id;
    p_byte_offset = stream_offsets[stream_idx];
    p_bit_pos = 8u;  // force load on first read_bit()
    p_current_byte = 0u;

    // Decode tokens with ZRL
    var s = 0u;
    while (s < symbols_per_stream) {
        let token = read_bit();
        if (token == 0u) {
            // Zero run: use subband of first zero for k_zrl
            let zi = thread_id + s * STREAMS_PER_TILE;
            let zr = zi / params.tile_size;
            let zc = zi % params.tile_size;
            let g_zrl = compute_subband_group(zc, zr);
            let run = read_rice(shared_k_zrl[g_zrl]) + 1u;
            for (var j = 0u; j < run; j++) {
                let ci = thread_id + (s + j) * STREAMS_PER_TILE;
                let cr = ci / params.tile_size;
                let cc = ci % params.tile_size;
                let pi = (tile_origin_y + cr) * params.plane_width + (tile_origin_x + cc);
                output[pi] = 0.0;
            }
            s += run;
        } else {
            // Non-zero coefficient
            let coeff_idx = thread_id + s * STREAMS_PER_TILE;
            let tile_row = coeff_idx / params.tile_size;
            let tile_col = coeff_idx % params.tile_size;
            let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                          + (tile_origin_x + tile_col);

            let sign = read_bit();
            let g = compute_subband_group(tile_col, tile_row);

            // Derive adaptive k from EMA (context-adaptive Rice parameter)
            let ema_mean = p_ema[g] >> 4u;
            let k = select(min(31u - countLeadingZeros(ema_mean), 15u), 0u, ema_mean == 0u);

            let rice_val = read_rice(k);
            let magnitude = rice_val + 1u;
            let value = select(i32(magnitude), -i32(magnitude), sign == 1u);
            output[plane_idx] = f32(value);

            // Update EMA with decoded magnitude
            p_ema[g] = p_ema[g] - (p_ema[g] >> 3u) + (rice_val << 1u);

            s += 1u;
        }
    }
}
