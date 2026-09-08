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
// The per-stream output slot is sized by the host from `symbols_per_stream` and arrives in
// `params.max_stream_words` (BUG-22). It used to be this constant, a fixed 128 words = 512 bytes,
// with nothing checking it: a stream needing more wrote straight into its neighbour's slot and the
// host packed those bytes back out as data — 7.8-10.9 dB at q=90 on all four stills at tile 512,
// and nothing reported. Both writes below are now bounded, so a slot that is somehow too small
// truncates one stream instead of corrupting the next one.
const ALPHABET_SIZE: u32 = 64u;
const ESCAPE_SYM: u32 = 63u;
const MAX_GROUPS: u32 = 8u;
const CB_STRIDE: u32 = 512u;  // MAX_GROUPS * ALPHABET_SIZE

struct Params {
    num_tiles: u32,
    coefficients_per_tile: u32,
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    num_levels: u32,
    max_stream_words: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> codebook: array<u32>;
@group(0) @binding(3) var<storage, read> k_zrl_buf: array<u32>;
@group(0) @binding(4) var<storage, read_write> stream_output: array<u32>;
@group(0) @binding(5) var<storage, read_write> stream_lengths: array<u32>;

// Shared codebook: MAX_GROUPS * ALPHABET_SIZE = 512 entries = 2KB
var<workgroup> shared_code: array<u32, 512>;
// Shared k_zrl: MAX_GROUPS = 8 entries
var<workgroup> shared_k_zrl: array<u32, 8>;

// Per-thread bit-packing state
var<private> p_bit_buffer: u32;
var<private> p_bits_in_buffer: u32;
var<private> p_word_buffer: u32;
var<private> p_bytes_in_word: u32;
var<private> p_word_pos: u32;
var<private> p_stream_word_base: u32;
var<private> p_total_bytes: u32;

// Tile-local raster index of symbol `s` in stream `stream_id`.
//
// Streams walk the tile in column-major order, cut into STREAMS_PER_TILE contiguous segments, so
// the previous symbol in a stream is the coefficient directly above it. At the default 256 px tile
// a segment is exactly one column and this equals the old `thread_id + s * 256` coefficient for
// coefficient; at any other width that modulus interleaved distant columns into one stream and
// both the zero runs and their k_zrl tracked a mixture (BUG-14, BUG-11's twin). Must stay in sync
// with huffman.rs stream_coeff_index().
fn stream_coeff_index(stream_id: u32, s: u32, symbols_per_stream: u32) -> u32 {
    let j = stream_id * symbols_per_stream + s;
    return (j % params.tile_size) * params.tile_size + j / params.tile_size;
}

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
        if (p_word_pos < params.max_stream_words) {
            stream_output[p_stream_word_base + p_word_pos] = p_word_buffer;
        }
        p_word_pos += 1u;
        p_word_buffer = 0u;
        p_bytes_in_word = 0u;
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

fn emit_bit(bit: u32) {
    emit_bits(bit, 1u);
}

// Write exp-Golomb coded value.
fn emit_exp_golomb(value: u32) {
    if (value == 0u) {
        emit_bit(1u);
        return;
    }
    let v = value + 1u;
    let bits = 31u - countLeadingZeros(v);
    // Write 'bits' zeros then the binary representation of v
    for (var i = 0u; i < bits; i++) {
        emit_bit(0u);
    }
    emit_bits(v, bits + 1u);
}

fn flush_remaining() {
    if (p_bits_in_buffer > 0u) {
        let byte_val = (p_bit_buffer << (8u - p_bits_in_buffer)) & 0xFFu;
        emit_byte(byte_val);
    }
    if (p_bytes_in_word > 0u) {
        if (p_word_pos < params.max_stream_words) {
            stream_output[p_stream_word_base + p_word_pos] = p_word_buffer;
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

    // Phase 1: Load codebook + k_zrl into shared memory
    // 256 threads, 512 codebook entries — 2 loads per thread
    shared_code[thread_id] = codebook[tile_id * CB_STRIDE + thread_id];
    if (thread_id + 256u < CB_STRIDE) {
        shared_code[thread_id + 256u] = codebook[tile_id * CB_STRIDE + thread_id + 256u];
    }
    if (thread_id < MAX_GROUPS) {
        shared_k_zrl[thread_id] = k_zrl_buf[tile_id * MAX_GROUPS + thread_id];
    }
    workgroupBarrier();

    // Phase 2: Encode stream
    p_stream_word_base = (tile_id * STREAMS_PER_TILE + thread_id) * params.max_stream_words;
    p_bit_buffer = 0u;
    p_bits_in_buffer = 0u;
    p_word_buffer = 0u;
    p_bytes_in_word = 0u;
    p_word_pos = 0u;
    p_total_bytes = 0u;

    var s = 0u;
    while (s < symbols_per_stream) {
        let coeff_idx = stream_coeff_index(thread_id, s, symbols_per_stream);
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
                let ni = stream_coeff_index(thread_id, ns, symbols_per_stream);
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

            // Escape: append exp-Golomb for large magnitudes
            if (sym == ESCAPE_SYM) {
                emit_exp_golomb(magnitude - ESCAPE_SYM);
            }

            s += 1u;
        }
    }

    flush_remaining();
    stream_lengths[tile_id * STREAMS_PER_TILE + thread_id] = p_total_bytes;
}
