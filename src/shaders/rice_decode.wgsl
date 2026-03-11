// GPU Rice decoder — 256 interleaved streams per tile.
//
// Inverse of rice_encode.wgsl. Each thread decodes one independent bit stream.
// No state chain — fully parallel across all 256 streams.
//
// Per coefficient:
//   bit=0 → zero run: Rice(run_length-1, k_zrl) where k_zrl selected by 2-state context
//   bit=1 → sign bit + Rice(magnitude-1, k_mag) → ±magnitude
//
// #53: 2-state k_zrl context based on magnitude of the preceding nonzero coefficient.
// k_zrl_nz is used after a "large" nonzero (|coeff| >= 2 → clustered signal, short runs);
// k_zrl_z is used after a "small" nonzero (|coeff| == 1) or at start-of-stream.

const STREAMS_PER_TILE: u32 = 256u;
const MAX_GROUPS: u32 = 8u;
// K_STRIDE per tile: [k_mag ×8][k_zrl_nz ×8][k_zrl_z ×8][skip_bitmap ×1] = 25
const K_STRIDE: u32 = 25u;

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

// Shared k values for this tile
var<workgroup> shared_k: array<u32, 8>;
var<workgroup> shared_k_zrl_nz: array<u32, 8>; // k_zrl after a large nonzero (|coeff|>=2)
var<workgroup> shared_k_zrl_z: array<u32, 8>;  // k_zrl after a small nonzero (|coeff|==1) or start
// Subband skip bitmap: bit g = 1 means all coefficients in group g are zero
var<workgroup> shared_skip_bitmap: u32;

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

    // Cooperatively load k values into shared memory (stride K_STRIDE = 25)
    // Layout: [k_mag ×8][k_zrl_nz ×8][k_zrl_z ×8][skip_bitmap]
    if (thread_id < num_groups) {
        shared_k[thread_id]        = k_values[tile_id * K_STRIDE + thread_id];
        shared_k_zrl_nz[thread_id] = k_values[tile_id * K_STRIDE + MAX_GROUPS + thread_id];
        shared_k_zrl_z[thread_id]  = k_values[tile_id * K_STRIDE + MAX_GROUPS * 2u + thread_id];
    }
    if (thread_id == 0u) {
        shared_skip_bitmap = k_values[tile_id * K_STRIDE + K_STRIDE - 1u];
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

    // Context state: was the preceding nonzero coefficient |coeff| >= 2?
    // Must mirror rice_encode.wgsl Phase 2 context tracking exactly.
    var last_mag_large: bool = false;

    // Decode tokens with ZRL
    var s = 0u;
    while (s < symbols_per_stream) {
        let ci0 = thread_id + s * STREAMS_PER_TILE;
        let cr0 = ci0 / params.tile_size;
        let cc0 = ci0 % params.tile_size;

        // Skip bitmap: if this position's group is fully zero, write 0 and advance
        let skip_g = compute_subband_group(cc0, cr0);
        if ((shared_skip_bitmap >> skip_g) & 1u) == 1u {
            let pi0 = (tile_origin_y + cr0) * params.plane_width + (tile_origin_x + cc0);
            output[pi0] = 0.0;
            s += 1u;
            continue;
        }

        let token = read_bit();
        if (token == 0u) {
            // Zero run: select k_zrl based on magnitude context of preceding nonzero
            let g_zrl = skip_g;
            let k_zrl = select(shared_k_zrl_z[g_zrl], shared_k_zrl_nz[g_zrl], last_mag_large);
            let run = read_rice(k_zrl) + 1u;
            // Write zeros, skipping past bitmap-skipped positions
            var written = 0u;
            var ws = s;
            while (written < run && ws < symbols_per_stream) {
                let ci = thread_id + ws * STREAMS_PER_TILE;
                let cr = ci / params.tile_size;
                let cc = ci % params.tile_size;
                let pi = (tile_origin_y + cr) * params.plane_width + (tile_origin_x + cc);
                let ws_g = compute_subband_group(cc, cr);
                if ((shared_skip_bitmap >> ws_g) & 1u) == 1u {
                    // Bitmap-skipped position: write zero but don't count toward run
                    output[pi] = 0.0;
                    ws += 1u;
                    continue;
                }
                output[pi] = 0.0;
                written += 1u;
                ws += 1u;
            }
            s = ws;
            last_mag_large = false;
        } else {
            // Non-zero coefficient
            let coeff_idx = ci0;
            let tile_row = cr0;
            let tile_col = cc0;
            let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                          + (tile_origin_x + tile_col);

            let sign = read_bit();
            let g = skip_g;

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
            // Mirror encoder: large = |coeff| >= 2, i.e., rice_val (= |coeff|-1) >= 1
            last_mag_large = (rice_val >= 1u);
        }
    }
}
