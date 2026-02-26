// Fused normalize + encode shader — single dispatch replaces two.
//
// Combines rans_normalize.wgsl + rans_encode_lean.wgsl into one kernel.
// Benefits:
//   1. Eliminates dispatch boundary between normalize and encode
//   2. Cumfreq written to global for CPU readback, read back from L1/L2 cache for encode
//   3. No global cumfreq_buf round-trip through DRAM (stays in cache)
//
// 256 threads per workgroup, 1 workgroup per tile.
// Phase 1: Threads 0-255 normalize histograms and build cumfreq (all groups)
// Phase 2: Threads 0-31 load cumfreq from cache and encode rANS streams

const WG_SIZE: u32 = 256u;
const RANS_M: u32 = 4096u;
const RANS_BYTE_L: u32 = 8388608u;
const RANS_PRECISION: u32 = 12u;
const STREAMS_PER_TILE: u32 = 32u;
const MAX_ALPHABET: u32 = 4096u;
const MAX_GROUP_ALPHABET: u32 = 4096u;
const MAX_GROUPS: u32 = 8u;
const MAX_STREAM_BYTES: u32 = 4096u;
const HIST_TILE_STRIDE: u32 = 32793u;
const ENCODE_TILE_INFO_STRIDE: u32 = 36u;

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
@group(0) @binding(1) var<storage, read> hist_input: array<u32>;
@group(0) @binding(2) var<storage, read> coeff_input: array<f32>;
@group(0) @binding(3) var<storage, read_write> cumfreq_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> tile_info_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> stream_output: array<u32>;
@group(0) @binding(6) var<storage, read_write> stream_metadata: array<u32>;

// Shared memory for normalization (reused per group)
var<workgroup> shared_freq: array<u32, 4096>;
var<workgroup> shared_sum: array<u32, 256>;
var<workgroup> w_total: u32;
var<workgroup> w_assigned: u32;
var<workgroup> w_max_idx: u32;

// Workgroup-shared scalars for histogram parsing
var<workgroup> w_min_val: u32;
var<workgroup> w_asize: u32;
var<workgroup> w_zrun_base: u32;
var<workgroup> w_hist_data_off: u32;
var<workgroup> w_hist_read_off: u32;

// Shared cumfreq for encode phase (loaded from global after all groups normalized)
var<workgroup> shared_cumfreq: array<u32, 4096>;

// ---- Normalize ----

fn normalize_inplace(lid: u32, asize: u32) {
    // Step 1: total
    var local_sum = 0u;
    for (var i = lid; i < asize; i += WG_SIZE) {
        local_sum += shared_freq[i];
    }
    shared_sum[lid] = local_sum;
    workgroupBarrier();

    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) { w_total = shared_sum[0]; }
    workgroupBarrier();

    let total = w_total;
    if (total == 0u) {
        if (lid == 0u) { shared_freq[0] = RANS_M; }
        workgroupBarrier();
        return;
    }

    // Step 2: scale
    for (var i = lid; i < asize; i += WG_SIZE) {
        let h = shared_freq[i];
        if (h > 0u) {
            var scaled = (h * RANS_M) / total;
            if (scaled == 0u) { scaled = 1u; }
            shared_freq[i] = scaled;
        }
    }
    workgroupBarrier();

    // Step 3: assigned sum
    var local_assigned = 0u;
    for (var i = lid; i < asize; i += WG_SIZE) {
        local_assigned += shared_freq[i];
    }
    shared_sum[lid] = local_assigned;
    workgroupBarrier();

    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        workgroupBarrier();
    }

    // Step 4: adjust
    if (lid == 0u) {
        let assigned = shared_sum[0];
        var max_val = 0u;
        var max_i = 0u;
        for (var i = 0u; i < asize; i++) {
            if (shared_freq[i] > max_val) {
                max_val = shared_freq[i];
                max_i = i;
            }
        }
        if (assigned < RANS_M) {
            shared_freq[max_i] += RANS_M - assigned;
        } else if (assigned > RANS_M) {
            var surplus = assigned - RANS_M;
            if (shared_freq[max_i] > surplus + 1u) {
                shared_freq[max_i] -= surplus;
            } else {
                let can_remove = shared_freq[max_i] - 1u;
                shared_freq[max_i] = 1u;
                surplus -= can_remove;
                for (var i = 0u; i < asize; i++) {
                    if (surplus == 0u) { break; }
                    if (shared_freq[i] > 1u) {
                        let r = min(surplus, shared_freq[i] - 1u);
                        shared_freq[i] -= r;
                        surplus -= r;
                    }
                }
            }
        }
    }
    workgroupBarrier();
}

// ---- Encode helpers ----

fn write_byte(byte_offset: u32, value: u32) {
    let word_idx = byte_offset >> 2u;
    let byte_pos = byte_offset & 3u;
    stream_output[word_idx] = stream_output[word_idx] | ((value & 0xFFu) << (byte_pos * 8u));
}

// Directional subband grouping: separates HH from LH+HL at each level.
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

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tile_id = wid.x;
    if (tile_id >= params.num_tiles) {
        return;
    }

    let hist_base = tile_id * HIST_TILE_STRIDE;
    let info_base = tile_id * ENCODE_TILE_INFO_STRIDE;

    let tile_x = tile_id % params.tiles_x;
    let tile_y = tile_id / params.tiles_x;
    let tile_origin_x = tile_x * params.tile_size;
    let tile_origin_y = tile_y * params.tile_size;
    let symbols_per_stream = params.coefficients_per_tile / STREAMS_PER_TILE;

    if (params.per_subband != 0u) {
        // ===== Per-subband: Phase 1 — Normalize all groups (256 threads) =====
        let num_groups = hist_input[hist_base];
        let cf_group_stride = MAX_GROUP_ALPHABET + 1u;
        let cf_tile_stride = MAX_GROUPS * cf_group_stride;
        let cf_tile_base = tile_id * cf_tile_stride;

        if (lid == 0u) {
            tile_info_out[info_base] = num_groups;
            w_hist_read_off = 1u;
        }
        workgroupBarrier();

        for (var g = 0u; g < num_groups; g++) {
            // Thread 0 reads group metadata
            if (lid == 0u) {
                let off = w_hist_read_off;
                w_min_val = hist_input[hist_base + off];
                w_asize = hist_input[hist_base + off + 1u];
                w_zrun_base = hist_input[hist_base + off + 2u];
                w_hist_data_off = off + 3u;
            }
            workgroupBarrier();

            let asize = w_asize;
            let hist_data_off = w_hist_data_off;

            // Load histogram into shared_freq
            for (var i = lid; i < asize; i += WG_SIZE) {
                shared_freq[i] = hist_input[hist_base + hist_data_off + i];
            }
            workgroupBarrier();

            // Normalize
            normalize_inplace(lid, asize);

            // Build cumfreq and write to global (sequential prefix sum by thread 0)
            let cf_base = cf_tile_base + g * cf_group_stride;
            if (lid == 0u) {
                cumfreq_out[cf_base] = 0u;
                for (var i = 0u; i < asize; i++) {
                    cumfreq_out[cf_base + i + 1u] = cumfreq_out[cf_base + i] + shared_freq[i];
                }

                // Write tile_info
                let gi = info_base + 1u + g * 4u;
                tile_info_out[gi] = w_min_val;
                tile_info_out[gi + 1u] = asize;
                tile_info_out[gi + 2u] = cf_base;
                tile_info_out[gi + 3u] = w_zrun_base;

                // Advance read offset
                w_hist_read_off = w_hist_data_off + asize;
            }
            workgroupBarrier();
        }

        // ===== Phase 2 — Load cumfreq from global into shared (L1/L2 cache hit) =====
        // This is the same load pattern as rans_encode_lean.wgsl but data is cache-hot
        var group_min: array<i32, 8>;
        var group_asize: array<u32, 8>;
        var group_cf_start: array<u32, 8>;
        var total_cf_entries = 0u;

        for (var g = 0u; g < num_groups; g++) {
            let gi = info_base + 1u + g * 4u;
            group_min[g] = bitcast<i32>(tile_info_out[gi]);
            group_asize[g] = tile_info_out[gi + 1u];
            let cf_global = tile_info_out[gi + 2u];
            group_cf_start[g] = total_cf_entries;
            let entries = group_asize[g] + 1u;

            for (var i = lid; i < entries; i += WG_SIZE) {
                shared_cumfreq[total_cf_entries + i] = cumfreq_out[cf_global + i];
            }
            total_cf_entries += entries;
        }

        workgroupBarrier();

        // ===== Phase 3 — Encode (threads 0-31) =====
        if (lid < STREAMS_PER_TILE) {
            let thread_id = lid;
            let stream_base_byte = (tile_id * STREAMS_PER_TILE + thread_id) * MAX_STREAM_BYTES;
            var write_ptr = MAX_STREAM_BYTES;
            var state: u32 = RANS_BYTE_L;

            for (var countdown = 0u; countdown < symbols_per_stream; countdown++) {
                let i = symbols_per_stream - 1u - countdown;
                let coeff_idx = thread_id + i * STREAMS_PER_TILE;
                let tile_row = coeff_idx / params.tile_size;
                let tile_col = coeff_idx % params.tile_size;
                let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                              + (tile_origin_x + tile_col);
                let coeff = coeff_input[plane_idx];

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

            let meta_base = (tile_id * STREAMS_PER_TILE + thread_id) * 2u;
            stream_metadata[meta_base] = write_ptr;
            stream_metadata[meta_base + 1u] = state;
        }

    } else {
        // ===== Single-table: Phase 1 — Normalize (256 threads) =====
        let min_val_u = hist_input[hist_base];
        let asize = hist_input[hist_base + 1u];
        let zrun_base = hist_input[hist_base + 2u];

        for (var i = lid; i < asize; i += WG_SIZE) {
            shared_freq[i] = hist_input[hist_base + 3u + i];
        }
        workgroupBarrier();

        normalize_inplace(lid, asize);

        // Build cumfreq and write to global
        let cf_stride = MAX_ALPHABET + 1u;
        let cf_base = tile_id * cf_stride;
        if (lid == 0u) {
            cumfreq_out[cf_base] = 0u;
            for (var i = 0u; i < asize; i++) {
                cumfreq_out[cf_base + i + 1u] = cumfreq_out[cf_base + i] + shared_freq[i];
            }
        }

        // Write tile_info
        if (lid == 0u) {
            tile_info_out[info_base] = min_val_u;
            tile_info_out[info_base + 1u] = asize;
            tile_info_out[info_base + 2u] = cf_base;
            tile_info_out[info_base + 3u] = zrun_base;
        }

        workgroupBarrier();

        // ===== Phase 2 — Load cumfreq into shared (cache-hot) + Encode =====
        let alphabet_size_plus_one = asize + 1u;
        for (var i = lid; i < alphabet_size_plus_one; i += WG_SIZE) {
            shared_cumfreq[i] = cumfreq_out[cf_base + i];
        }

        workgroupBarrier();

        if (lid < STREAMS_PER_TILE) {
            let thread_id = lid;
            let stream_base_byte = (tile_id * STREAMS_PER_TILE + thread_id) * MAX_STREAM_BYTES;
            var write_ptr = MAX_STREAM_BYTES;
            var state: u32 = RANS_BYTE_L;
            let min_val = bitcast<i32>(min_val_u);

            for (var countdown = 0u; countdown < symbols_per_stream; countdown++) {
                let i = symbols_per_stream - 1u - countdown;
                let coeff_idx = thread_id + i * STREAMS_PER_TILE;
                let tile_row = coeff_idx / params.tile_size;
                let tile_col = coeff_idx % params.tile_size;
                let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                              + (tile_origin_x + tile_col);
                let coeff = coeff_input[plane_idx];
                let sym = u32(i32(round(coeff)) - min_val);

                let start = shared_cumfreq[sym];
                let freq = shared_cumfreq[sym + 1u] - start;

                let result = rans_encode_sym(state, write_ptr, stream_base_byte, start, freq);
                state = result.x;
                write_ptr = result.y;
            }

            let meta_base = (tile_id * STREAMS_PER_TILE + thread_id) * 2u;
            stream_metadata[meta_base] = write_ptr;
            stream_metadata[meta_base + 1u] = state;
        }
    }
}
