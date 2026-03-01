// GPU Huffman decoder — 256 interleaved streams per tile.
//
// Inverse of huffman_encode.wgsl. Each thread decodes one independent bit stream.
// Uses 8-bit prefix decode table for fast Huffman symbol lookup.
//
// Optimized bit reader: 32-bit accumulator refilled in bulk.
// Single table lookup + advance (no peek/restore/consume).

const STREAMS_PER_TILE: u32 = 256u;
const ESCAPE_SYM: u32 = 63u;
const MAX_GROUPS: u32 = 8u;
const DT_STRIDE: u32 = 2048u;  // MAX_GROUPS * 256 (8-bit prefix table)
const MAX_CODE_LEN: u32 = 8u;

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
@group(0) @binding(1) var<storage, read> decode_table: array<u32>;
@group(0) @binding(2) var<storage, read> k_zrl_buf: array<u32>;
@group(0) @binding(3) var<storage, read> stream_data: array<u32>;
@group(0) @binding(4) var<storage, read> stream_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

// Shared decode table: MAX_GROUPS * 256 = 2048 entries = 8KB
var<workgroup> shared_decode: array<u32, 2048>;
// Shared k_zrl: MAX_GROUPS = 8 entries
var<workgroup> shared_k_zrl: array<u32, 8>;

// Per-thread 32-bit accumulator bit reader
var<private> p_acc: u32;       // accumulator — valid bits are at the top
var<private> p_bits: u32;      // number of valid bits in accumulator
var<private> p_byte_off: u32;  // next byte offset to load from stream_data

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

fn load_byte(byte_off: u32) -> u32 {
    let word_idx = byte_off >> 2u;
    let byte_pos = byte_off & 3u;
    return (stream_data[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

// Ensure accumulator has at least n valid bits.
fn ensure_bits(n: u32) {
    while (p_bits < n) {
        p_acc = (p_acc << 8u) | load_byte(p_byte_off);
        p_byte_off += 1u;
        p_bits += 8u;
    }
}

// Peek at top n bits of accumulator without consuming.
fn peek(n: u32) -> u32 {
    return (p_acc >> (p_bits - n)) & ((1u << n) - 1u);
}

// Consume n bits from accumulator.
fn drop_bits(n: u32) {
    p_bits -= n;
}

// Read n bits: ensure + peek + drop.
fn read_bits(n: u32) -> u32 {
    ensure_bits(n);
    let v = peek(n);
    drop_bits(n);
    return v;
}

// Read unary code: count 1-bits until 0-bit.
fn read_unary() -> u32 {
    var count = 0u;
    while (count < 31u) {
        ensure_bits(1u);
        if (peek(1u) == 0u) {
            drop_bits(1u);
            return count;
        }
        drop_bits(1u);
        count += 1u;
    }
    // Safety: consume the terminator
    ensure_bits(1u);
    drop_bits(1u);
    return count;
}

// Read Golomb-Rice coded value: unary quotient + k-bit remainder.
fn read_rice(k: u32) -> u32 {
    let quotient = read_unary();
    let remainder = select(0u, read_bits(k), k > 0u);
    return (quotient << k) | remainder;
}

// Read exp-Golomb coded value.
fn read_exp_golomb() -> u32 {
    // Count leading zeros
    var leading_zeros = 0u;
    while (leading_zeros < 20u) {
        ensure_bits(1u);
        if (peek(1u) == 1u) {
            drop_bits(1u);
            break;
        }
        drop_bits(1u);
        leading_zeros += 1u;
    }
    // Read remaining bits
    if (leading_zeros == 0u) {
        return 0u;
    }
    let rest = read_bits(leading_zeros);
    return (1u << leading_zeros) - 1u + rest;
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

    // Load decode table into shared memory cooperatively
    for (var i = thread_id; i < DT_STRIDE; i += STREAMS_PER_TILE) {
        shared_decode[i] = decode_table[tile_id * DT_STRIDE + i];
    }
    if (thread_id < MAX_GROUPS) {
        shared_k_zrl[thread_id] = k_zrl_buf[tile_id * MAX_GROUPS + thread_id];
    }
    workgroupBarrier();

    // Initialize accumulator bit reader
    let stream_idx = tile_id * STREAMS_PER_TILE + thread_id;
    p_byte_off = stream_offsets[stream_idx];
    p_acc = 0u;
    p_bits = 0u;

    // Decode loop
    var s = 0u;
    while (s < symbols_per_stream) {
        let token = read_bits(1u);
        if (token == 0u) {
            // Zero run
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

            let sign = read_bits(1u);
            let g = compute_subband_group(tile_col, tile_row);

            // Fast Huffman decode via 8-bit prefix table
            ensure_bits(8u);
            let peek8 = peek(8u);
            let entry = shared_decode[g * 256u + peek8];
            let sym = entry >> 16u;
            let code_len = entry & 0xFFFFu;

            // All codes are <= 8 bits (HUFFMAN_MAX_CODE_LEN=8), so a single
            // 8-bit prefix table lookup always resolves the symbol.
            drop_bits(code_len);
            var magnitude: u32 = sym;

            // Escape: read exp-Golomb magnitude for large values
            if (magnitude >= ESCAPE_SYM) {
                magnitude = ESCAPE_SYM + read_exp_golomb();
            }

            let value = select(i32(magnitude + 1u), -i32(magnitude + 1u), sign == 1u);
            output[plane_idx] = f32(value);
            s += 1u;
        }
    }
}
