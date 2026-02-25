// GPU rANS encoder — 32 interleaved streams per tile, with ZRL support.
//
// Each workgroup encodes one tile. 32 threads = 32 independent rANS streams.
// Cumfreq table is loaded cooperatively into workgroup shared memory.
// Encoding proceeds in reverse symbol order (last symbol first).
//
// ZRL (zero-run-length) encoding: consecutive zeros within each stream are
// replaced by run-length symbols (zrun_base + run_length - 1) before rANS.
// Each stream's ZRL is independent — runs don't cross stream boundaries.
//
// Supports two modes (selected by params.per_subband):
//   0 = Single frequency table per tile (with optional ZRL)
//   1 = Per-subband frequency tables (with per-group ZRL)

const RANS_BYTE_L: u32 = 8388608u;  // 1 << 23
const RANS_PRECISION: u32 = 12u;
const STREAMS_PER_TILE: u32 = 32u;
const MAX_ALPHABET: u32 = 2048u;
const MAX_STREAM_BYTES: u32 = 4096u;
const MAX_ZERO_RUN: u32 = 256u;

// Per-tile encode info stride in u32s.
// Single-table: [0]=min_val, [1]=alphabet_size, [2]=cumfreq_offset, [3]=zrun_base
// Per-subband:  [0]=num_groups, [1+g*4]=min_val, [2+g*4]=alphabet_size,
//               [3+g*4]=cumfreq_offset, [4+g*4]=zrun_base
const ENCODE_TILE_INFO_STRIDE: u32 = 32u;

// Max ZRL-encoded symbols per stream. With ZRL, the symbol count can only
// decrease (runs of zeros become single symbols), so this is a safe upper bound.
const MAX_ZRL_SYMBOLS: u32 = 2048u;

struct Params {
    num_tiles: u32,
    coefficients_per_tile: u32,  // tile_size * tile_size
    plane_width: u32,            // padded plane width in pixels
    tile_size: u32,
    tiles_x: u32,
    per_subband: u32,            // 0 = single table, 1 = per-subband
    num_levels: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> cumfreq_data: array<u32>;
@group(0) @binding(3) var<storage, read> tile_info: array<u32>;
@group(0) @binding(4) var<storage, read_write> stream_output: array<u32>;
@group(0) @binding(5) var<storage, read_write> stream_metadata: array<u32>;

// Shared cumfreq table for the current tile (loaded cooperatively)
var<workgroup> shared_cumfreq: array<u32, 4096>;

// Write one byte into the packed u32 output array.
// Each stream writes to its own non-overlapping region, so no atomics needed.
// Buffer must be zero-initialized; we use OR to set individual bytes within u32 words.
fn write_byte(byte_offset: u32, value: u32) {
    let word_idx = byte_offset >> 2u;
    let byte_pos = byte_offset & 3u;
    stream_output[word_idx] = stream_output[word_idx] | ((value & 0xFFu) << (byte_pos * 8u));
}

// Compute subband group for a tile-local position.
// Group 0 = LL, Group k (1..num_levels) = Level (k-1) detail subbands.
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

// Encode a single rANS symbol given its cumfreq start and freq.
// Returns updated (state, write_ptr).
fn rans_encode_sym(
    state_in: u32, write_ptr_in: u32,
    stream_base_byte: u32,
    start: u32, freq: u32
) -> vec2<u32> {
    var state = state_in;
    var write_ptr = write_ptr_in;

    // Renormalize: emit bytes while state is too large
    let x_max = ((RANS_BYTE_L >> RANS_PRECISION) << 8u) * freq;
    for (var r = 0u; r < 4u; r++) {
        if (state < x_max) { break; }
        write_ptr -= 1u;
        write_byte(stream_base_byte + write_ptr, state & 0xFFu);
        state >>= 8u;
    }

    // Encode
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

    // Compute tile position in the plane
    let tile_x = tile_id % params.tiles_x;
    let tile_y = tile_id / params.tiles_x;
    let tile_origin_x = tile_x * params.tile_size;
    let tile_origin_y = tile_y * params.tile_size;

    let symbols_per_stream = params.coefficients_per_tile / STREAMS_PER_TILE;

    // Output region for this stream (byte offset into stream_output)
    let stream_base_byte = (tile_id * STREAMS_PER_TILE + thread_id) * MAX_STREAM_BYTES;

    // write_ptr starts at end, grows backward (renormalization bytes written right-to-left)
    var write_ptr = MAX_STREAM_BYTES;
    var state: u32 = RANS_BYTE_L;

    if (params.per_subband != 0u) {
        // --- Per-subband encode path with ZRL ---
        let num_groups = tile_info[base];

        // Load per-group metadata into local arrays
        var group_min: array<i32, 8>;
        var group_asize: array<u32, 8>;
        var group_cf_start: array<u32, 8>;
        var group_zrun: array<i32, 8>;
        var total_cf_entries = 0u;
        var has_any_zrl = false;

        for (var g = 0u; g < num_groups; g++) {
            let gi = base + 1u + g * 4u;
            group_min[g] = bitcast<i32>(tile_info[gi]);
            group_asize[g] = tile_info[gi + 1u];
            let cf_global = tile_info[gi + 2u];
            group_zrun[g] = bitcast<i32>(tile_info[gi + 3u]);
            if (group_zrun[g] != 0) {
                has_any_zrl = true;
            }
            group_cf_start[g] = total_cf_entries;
            let entries = group_asize[g] + 1u;

            // Cooperatively load this group's cumfreqs into shared memory
            for (var i = thread_id; i < entries; i += STREAMS_PER_TILE) {
                shared_cumfreq[total_cf_entries + i] = cumfreq_data[cf_global + i];
            }
            total_cf_entries += entries;
        }

        workgroupBarrier();

        if (has_any_zrl) {
            // ZRL-aware encode: first build the forward ZRL symbol sequence,
            // then encode in reverse order.
            //
            // We store (symbol_value, group_index) pairs in local arrays.
            // The symbol_value is the ZRL-transformed value (run symbols for zeros).
            var zrl_sym_vals: array<i32, 2048>;
            var zrl_sym_groups: array<u32, 2048>;
            var zrl_count = 0u;

            // Forward pass: build ZRL symbol sequence for this stream
            var i = 0u;
            while (i < symbols_per_stream) {
                let coeff_idx = thread_id + i * STREAMS_PER_TILE;
                let tile_row = coeff_idx / params.tile_size;
                let tile_col = coeff_idx % params.tile_size;
                let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                              + (tile_origin_x + tile_col);
                let coeff = i32(round(input[plane_idx]));
                let g = compute_subband_group(tile_col, tile_row);
                let zrun = group_zrun[g];

                if (zrun != 0 && coeff == 0) {
                    // Count consecutive zeros in same group within this stream
                    var run_len = 0u;
                    while (i < symbols_per_stream && run_len < MAX_ZERO_RUN) {
                        let ci = thread_id + i * STREAMS_PER_TILE;
                        let cr = ci / params.tile_size;
                        let cc = ci % params.tile_size;
                        let pi = (tile_origin_y + cr) * params.plane_width
                               + (tile_origin_x + cc);
                        let cv = i32(round(input[pi]));
                        let cg = compute_subband_group(cc, cr);
                        if (cv != 0 || cg != g) { break; }
                        run_len += 1u;
                        i += 1u;
                    }
                    // Run symbol value = zrun_base + (run_len - 1)
                    zrl_sym_vals[zrl_count] = zrun + i32(run_len) - 1;
                    zrl_sym_groups[zrl_count] = g;
                    zrl_count += 1u;
                } else {
                    zrl_sym_vals[zrl_count] = coeff;
                    zrl_sym_groups[zrl_count] = g;
                    zrl_count += 1u;
                    i += 1u;
                }
            }

            // Reverse pass: rANS encode ZRL symbols
            for (var countdown = 0u; countdown < zrl_count; countdown++) {
                let si = zrl_count - 1u - countdown;
                let sym_val = zrl_sym_vals[si];
                let g = zrl_sym_groups[si];
                let gmin = group_min[g];

                var sym = u32(sym_val - gmin);
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
            // No ZRL: standard encode in reverse order
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
        }

    } else {
        // --- Single-table encode path with optional ZRL ---
        let min_val = bitcast<i32>(tile_info[base]);
        let alphabet_size = tile_info[base + 1u];
        let cumfreq_offset = tile_info[base + 2u];
        let zrun_base = bitcast<i32>(tile_info[base + 3u]);
        let alphabet_size_plus_one = alphabet_size + 1u;

        // Cooperatively load cumfreq into shared memory
        for (var i = thread_id; i < alphabet_size_plus_one; i += STREAMS_PER_TILE) {
            shared_cumfreq[i] = cumfreq_data[cumfreq_offset + i];
        }

        workgroupBarrier();

        if (zrun_base != 0) {
            // ZRL-aware encode: build forward ZRL sequence, then encode in reverse
            var zrl_sym_vals: array<i32, 2048>;
            var zrl_count = 0u;

            var i = 0u;
            while (i < symbols_per_stream) {
                let coeff_idx = thread_id + i * STREAMS_PER_TILE;
                let tile_row = coeff_idx / params.tile_size;
                let tile_col = coeff_idx % params.tile_size;
                let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                              + (tile_origin_x + tile_col);
                let coeff = i32(round(input[plane_idx]));

                if (coeff == 0) {
                    // Count consecutive zeros in this stream
                    var run_len = 0u;
                    while (i < symbols_per_stream && run_len < MAX_ZERO_RUN) {
                        let ci = thread_id + i * STREAMS_PER_TILE;
                        let cr = ci / params.tile_size;
                        let cc = ci % params.tile_size;
                        let pi = (tile_origin_y + cr) * params.plane_width
                               + (tile_origin_x + cc);
                        let cv = i32(round(input[pi]));
                        if (cv != 0) { break; }
                        run_len += 1u;
                        i += 1u;
                    }
                    zrl_sym_vals[zrl_count] = zrun_base + i32(run_len) - 1;
                    zrl_count += 1u;
                } else {
                    zrl_sym_vals[zrl_count] = coeff;
                    zrl_count += 1u;
                    i += 1u;
                }
            }

            // Reverse pass: rANS encode
            for (var countdown = 0u; countdown < zrl_count; countdown++) {
                let si = zrl_count - 1u - countdown;
                var sym = u32(zrl_sym_vals[si] - min_val);
                if (sym >= alphabet_size) { sym = alphabet_size - 1u; }

                let start = shared_cumfreq[sym];
                let freq = shared_cumfreq[sym + 1u] - start;

                let result = rans_encode_sym(state, write_ptr, stream_base_byte, start, freq);
                state = result.x;
                write_ptr = result.y;
            }

        } else {
            // No ZRL: standard encode in reverse order
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
    }

    // Write final state and write_ptr to metadata buffer
    let meta_base = (tile_id * STREAMS_PER_TILE + thread_id) * 2u;
    stream_metadata[meta_base] = write_ptr;
    stream_metadata[meta_base + 1u] = state;
}
