// GPU Huffman encoder — 256 interleaved streams per tile.
//
// Uses pre-built canonical Huffman codebooks (uploaded from CPU).
// Same significance map + ZRL scheme as Rice, but replaces Rice magnitude
// coding with table-driven Huffman codewords.
//
// Codebook layout in buffer (per tile):
//   codebook[tile_id * CB_STRIDE + group * ALPHABET_SIZE + sym] = (code_length << 16) | codeword
//   k_zrl[tile_id * MAX_GROUPS + group] = Rice k parameter for ZRL

const STREAMS_PER_TILE: u32 = 256u;
const MAX_STREAM_BYTES: u32 = 512u;
const MAX_STREAM_WORDS: u32 = 128u;
const ALPHABET_SIZE: u32 = 32u;
const ESCAPE_SYM: u32 = 31u;
const MAX_GROUPS: u32 = 8u;
const CB_STRIDE: u32 = 256u;  // MAX_GROUPS * ALPHABET_SIZE

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
@group(0) @binding(2) var<storage, read> codebook: array<u32>;
@group(0) @binding(3) var<storage, read> k_zrl_buf: array<u32>;
@group(0) @binding(4) var<storage, read_write> stream_output: array<u32>;
@group(0) @binding(5) var<storage, read_write> stream_lengths: array<u32>;

// Shared codebook: MAX_GROUPS * ALPHABET_SIZE = 256 entries = 1KB
var<workgroup> shared_code: array<u32, 256>;
// Shared k_zrl: MAX_GROUPS = 8 entries
var<workgroup> shared_k_zrl: array<u32, 8>;

// Per-thread bit-packing state (identical to Rice)
var<private> p_bit_buffer: u32;
var<private> p_bits_in_buffer: u32;
var<private> p_word_buffer: u32;
var<private> p_bytes_in_word: u32;
var<private> p_word_pos: u32;
var<private> p_stream_word_base: u32;
var<private> p_total_bytes: u32;

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

fn emit_bit(bit: u32) {
    p_bit_buffer = (p_bit_buffer << 1u) | (bit & 1u);
    p_bits_in_buffer += 1u;
    if (p_bits_in_buffer == 8u) {
        emit_byte(p_bit_buffer);
        p_bits_in_buffer = 0u;
        p_bit_buffer = 0u;
    }
}

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

    // Phase 1: Load codebook + k_zrl into shared memory
    // 256 threads, 256 codebook entries — one per thread
    shared_code[thread_id] = codebook[tile_id * CB_STRIDE + thread_id];
    if (thread_id < MAX_GROUPS) {
        shared_k_zrl[thread_id] = k_zrl_buf[tile_id * MAX_GROUPS + thread_id];
    }
    workgroupBarrier();

    // Phase 2: Encode stream
    p_stream_word_base = (tile_id * STREAMS_PER_TILE + thread_id) * MAX_STREAM_WORDS;
    p_bit_buffer = 0u;
    p_bits_in_buffer = 0u;
    p_word_buffer = 0u;
    p_bytes_in_word = 0u;
    p_word_pos = 0u;
    p_total_bytes = 0u;

    var s = 0u;
    while (s < symbols_per_stream) {
        let coeff_idx = thread_id + s * STREAMS_PER_TILE;
        let tile_row = coeff_idx / params.tile_size;
        let tile_col = coeff_idx % params.tile_size;
        let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                      + (tile_origin_x + tile_col);
        let coeff = i32(round(input[plane_idx]));

        if (coeff == 0) {
            // Zero run with Rice ZRL (same as Rice coder)
            let g_zrl = compute_subband_group(tile_col, tile_row);
            let k_zrl = shared_k_zrl[g_zrl];
            let max_run = 32u << k_zrl;
            var run = 1u;
            var ns = s + 1u;
            while (ns < symbols_per_stream && run < max_run) {
                let ni = thread_id + ns * STREAMS_PER_TILE;
                let nr = ni / params.tile_size;
                let nc = ni % params.tile_size;
                let np = (tile_origin_y + nr) * params.plane_width + (tile_origin_x + nc);
                if (i32(round(input[np])) != 0) {
                    break;
                }
                run += 1u;
                ns += 1u;
            }

            emit_bit(0u);
            let run_val = run - 1u;
            let rq = run_val >> k_zrl;
            var rq_rem = rq;
            while (rq_rem > 0u) {
                let chunk = min(rq_rem, 15u);
                emit_bits((1u << chunk) - 1u, chunk);
                rq_rem -= chunk;
            }
            emit_bit(0u);
            if (k_zrl > 0u) {
                emit_bits(run_val & ((1u << k_zrl) - 1u), k_zrl);
            }
            s += run;
        } else {
            // Non-zero: significance + sign + Huffman code
            emit_bit(1u);
            emit_bit(select(0u, 1u, coeff < 0));

            let magnitude = u32(abs(coeff)) - 1u;
            let g = compute_subband_group(tile_col, tile_row);
            let sym = min(magnitude, ESCAPE_SYM);

            // Lookup Huffman code from shared memory
            let packed = shared_code[g * ALPHABET_SIZE + sym];
            let code_len = packed >> 16u;
            let codeword = packed & 0xFFFFu;

            if (code_len > 0u) {
                emit_bits(codeword, code_len);
            }

            // Escape: append raw 12-bit magnitude for large values
            if (sym == ESCAPE_SYM) {
                emit_bits(min(magnitude, 4095u), 12u);
            }

            s += 1u;
        }
    }

    flush_remaining();
    stream_lengths[tile_id * STREAMS_PER_TILE + thread_id] = p_total_bytes;
}
