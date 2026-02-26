// GPU rANS encoder — lean variant WITHOUT ZRL support.
//
// Identical to rans_encode.wgsl but without zero-run-length encoding.
// This avoids large per-thread local arrays (16 KB/thread) that cause
// register spilling and ~10x performance regression on Apple M1 GPU.
//
// Use this shader when the histogram shader determines no group needs ZRL.

const RANS_BYTE_L: u32 = 8388608u;  // 1 << 23
const RANS_PRECISION: u32 = 12u;
const STREAMS_PER_TILE: u32 = 32u;
const MAX_ALPHABET: u32 = 4096u;
const MAX_STREAM_BYTES: u32 = 4096u;
const ENCODE_TILE_INFO_STRIDE: u32 = 32u;

struct Params {
    num_tiles: u32,
    coefficients_per_tile: u32,
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    per_subband: u32,
    num_levels: u32,
    flags: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> cumfreq_data: array<u32>;
@group(0) @binding(3) var<storage, read> tile_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> stream_output: array<u32>;
@group(0) @binding(5) var<storage, read_write> stream_metadata: array<u32>;

var<workgroup> shared_cumfreq: array<u32, 4096>;

fn write_byte(byte_offset: u32, value: u32) {
    let word_idx = byte_offset >> 2u;
    let byte_pos = byte_offset & 3u;
    stream_output[word_idx] = stream_output[word_idx] | ((value & 0xFFu) << (byte_pos * 8u));
}

fn compute_subband_group(lx: u32, ly: u32) -> u32 {
    var region = params.tile_size;
    for (var level = 0u; level < params.num_levels; level++) {
        let half = region / 2u;
        if (lx >= half || ly >= half) {
            return level + 1u;
        }
        region = half;
    }
    return 0u;
}

fn rans_encode_sym(
    state_in: u32, write_ptr_in: u32,
    stream_base_byte: u32,
    start: u32, freq: u32
) -> vec2<u32> {
    var state = state_in;
    var write_ptr = write_ptr_in;

    let x_max = ((RANS_BYTE_L >> RANS_PRECISION) << 8u) * freq;
    for (var r = 0u; r < 4u; r++) {
        if (state < x_max) { break; }
        write_ptr -= 1u;
        write_byte(stream_base_byte + write_ptr, state & 0xFFu);
        state >>= 8u;
    }

    state = ((state / freq) << RANS_PRECISION) + (state % freq) + start;
    return vec2<u32>(state, write_ptr);
}

@compute @workgroup_size(32)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let thread_id = lid.x;
    let tile_id = wid.x;

    if (tile_id >= params.num_tiles) {
        return;
    }

    let base = tile_id * ENCODE_TILE_INFO_STRIDE;

    let tile_x = tile_id % params.tiles_x;
    let tile_y = tile_id / params.tiles_x;
    let tile_origin_x = tile_x * params.tile_size;
    let tile_origin_y = tile_y * params.tile_size;

    let symbols_per_stream = params.coefficients_per_tile / STREAMS_PER_TILE;

    let stream_base_byte = (tile_id * STREAMS_PER_TILE + thread_id) * MAX_STREAM_BYTES;

    var write_ptr = MAX_STREAM_BYTES;
    var state: u32 = RANS_BYTE_L;

    if (params.per_subband != 0u) {
        // --- Per-subband encode (no ZRL) ---
        let num_groups = tile_info[base];

        var group_min: array<i32, 8>;
        var group_asize: array<u32, 8>;
        var group_cf_start: array<u32, 8>;
        var total_cf_entries = 0u;

        for (var g = 0u; g < num_groups; g++) {
            let gi = base + 1u + g * 4u;
            group_min[g] = bitcast<i32>(tile_info[gi]);
            group_asize[g] = tile_info[gi + 1u];
            let cf_global = tile_info[gi + 2u];
            group_cf_start[g] = total_cf_entries;
            let entries = group_asize[g] + 1u;

            for (var i = thread_id; i < entries; i += STREAMS_PER_TILE) {
                shared_cumfreq[total_cf_entries + i] = cumfreq_data[cf_global + i];
            }
            total_cf_entries += entries;
        }

        workgroupBarrier();

        // Standard encode in reverse order (no ZRL)
        for (var countdown = 0u; countdown < symbols_per_stream; countdown++) {
            let i = symbols_per_stream - 1u - countdown;
            let coeff_idx = thread_id + i * STREAMS_PER_TILE;

            let tile_row = coeff_idx / params.tile_size;
            let tile_col = coeff_idx % params.tile_size;

            let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                          + (tile_origin_x + tile_col);
            let coeff = input[plane_idx];

            let g = compute_subband_group(tile_col, tile_row);
            let gmin = group_min[g];
            var sym = u32(i32(round(coeff)) - gmin);
            if (sym >= group_asize[g]) {
                sym = group_asize[g] - 1u;
            }

            let cf_start = group_cf_start[g];
            let start = shared_cumfreq[cf_start + sym];
            let freq = shared_cumfreq[cf_start + sym + 1u] - start;

            let result = rans_encode_sym(state, write_ptr, stream_base_byte, start, freq);
            state = result.x;
            write_ptr = result.y;
        }

    } else {
        // --- Single-table encode (no ZRL) ---
        let min_val = bitcast<i32>(tile_info[base]);
        let alphabet_size = tile_info[base + 1u];
        let cumfreq_offset = tile_info[base + 2u];
        let alphabet_size_plus_one = alphabet_size + 1u;

        for (var i = thread_id; i < alphabet_size_plus_one; i += STREAMS_PER_TILE) {
            shared_cumfreq[i] = cumfreq_data[cumfreq_offset + i];
        }

        workgroupBarrier();

        // Standard encode in reverse order
        for (var countdown = 0u; countdown < symbols_per_stream; countdown++) {
            let i = symbols_per_stream - 1u - countdown;
            let coeff_idx = thread_id + i * STREAMS_PER_TILE;

            let tile_row = coeff_idx / params.tile_size;
            let tile_col = coeff_idx % params.tile_size;

            let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                          + (tile_origin_x + tile_col);
            let coeff = input[plane_idx];
            let sym = u32(i32(round(coeff)) - min_val);

            let start = shared_cumfreq[sym];
            let freq = shared_cumfreq[sym + 1u] - start;

            let result = rans_encode_sym(state, write_ptr, stream_base_byte, start, freq);
            state = result.x;
            write_ptr = result.y;
        }
    }

    // Write final state and write_ptr
    let meta_base = (tile_id * STREAMS_PER_TILE + thread_id) * 2u;
    stream_metadata[meta_base] = write_ptr;
    stream_metadata[meta_base + 1u] = state;
}
