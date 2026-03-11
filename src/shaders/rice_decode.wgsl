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
//
// Checkerboard k-context (#checkerboard-ctx):
// Even streams (0,2,...,254) decode first, write final EMA means to shared_ctx_even.
// workgroupBarrier() — at top level (not inside branch) per Metal/M1 rule.
// Odd streams (1,3,...,255) derive adjusted k from even neighbor's decoded EMA state,
// then decode with that warm-start. No extra bitstream data — context from decoded data.

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
// Checkerboard context: even threads write final EMA means here after decoding.
// Odd threads read their left neighbor's EMA to derive adjusted k warm-start.
var<workgroup> shared_ctx_even: array<array<u32, 8>, 128>;

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

// Decode all symbols for the current thread's stream into the output plane.
// Uses per-thread private state: p_ema, p_byte_offset, p_bit_pos, p_current_byte.
// Also reads shared: shared_k_zrl_nz, shared_k_zrl_z, shared_skip_bitmap.
fn decode_stream_body(
    thread_id: u32,
    tile_origin_x: u32,
    tile_origin_y: u32,
    symbols_per_stream: u32,
) {
    var last_mag_large: bool = false;
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

    // Cooperatively load k values into shared memory.
    // Layout: [k_mag ×8][k_zrl_nz ×8][k_zrl_z ×8][skip_bitmap @K_STRIDE-1] = 25 entries
    if (thread_id < num_groups) {
        shared_k[thread_id]        = k_values[tile_id * K_STRIDE + thread_id];
        shared_k_zrl_nz[thread_id] = k_values[tile_id * K_STRIDE + MAX_GROUPS + thread_id];
        shared_k_zrl_z[thread_id]  = k_values[tile_id * K_STRIDE + MAX_GROUPS * 2u + thread_id];
    }
    if (thread_id == 0u) {
        shared_skip_bitmap = k_values[tile_id * K_STRIDE + K_STRIDE - 1u];
    }
    workgroupBarrier(); // top-level: all threads synchronize

    // === Checkerboard two-pass decode ===
    // Pass 1: even threads decode, write final EMA to shared_ctx_even.
    // workgroupBarrier() — at workgroup top level (not inside any branch per Metal/M1 rule).
    // Pass 2: odd threads derive adjusted k from neighbor EMA, then decode.

    // --- Pass 1: even threads ---
    if (thread_id % 2u == 0u) {
        // Initialize EMA from global k seeds (standard warm-start)
        for (var gi = 0u; gi < MAX_GROUPS; gi++) {
            p_ema[gi] = max(1u, 1u << shared_k[gi]) << 4u;
        }
        // Initialize bit reader
        p_byte_offset = stream_offsets[tile_id * STREAMS_PER_TILE + thread_id];
        p_bit_pos = 8u;
        p_current_byte = 0u;
        // Decode this stream
        decode_stream_body(thread_id, tile_origin_x, tile_origin_y, symbols_per_stream);
        // Expose final EMA to odd neighbor
        let even_idx = thread_id / 2u;
        for (var gi2 = 0u; gi2 < MAX_GROUPS; gi2++) {
            shared_ctx_even[even_idx][gi2] = p_ema[gi2] >> 4u;
        }
    }

    workgroupBarrier(); // top-level: even done, barrier before odd reads context

    // --- Pass 2: odd threads ---
    if (thread_id % 2u == 1u) {
        // Derive adjusted k from even neighbor's decoded EMA (same formula as encoder)
        let even_idx_o = (thread_id - 1u) / 2u;
        for (var gi3 = 0u; gi3 < MAX_GROUPS; gi3++) {
            let neighbor_mean = shared_ctx_even[even_idx_o][gi3];
            let global_k = shared_k[gi3];
            let neighbor_k = select(
                min(31u - countLeadingZeros(neighbor_mean), 15u),
                0u,
                neighbor_mean == 0u
            );
            let adjusted_k = clamp((global_k + neighbor_k + 1u) / 2u, 0u, 15u);
            p_ema[gi3] = max(1u, 1u << adjusted_k) << 4u;
        }
        // Initialize bit reader
        p_byte_offset = stream_offsets[tile_id * STREAMS_PER_TILE + thread_id];
        p_bit_pos = 8u;
        p_current_byte = 0u;
        // Decode this stream
        decode_stream_body(thread_id, tile_origin_x, tile_origin_y, symbols_per_stream);
    }
}
