// GPU histogram builder for rANS entropy coding with ZRL support.
//
// 256 threads per workgroup, 1 workgroup per tile.
// Three phases within one dispatch:
//   Phase 1: Find min/max, zero count, max_abs (parallel reduction)
//   Phase 2: Decide ZRL eligibility, compute extended alphabet
//   Phase 3: Build histogram (ZRL-aware for eligible groups)
//
// ZRL (zero-run-length) replaces consecutive zeros in each rANS stream
// with run-length symbols. Only applied to detail subbands (group > 0)
// with high zero density (>= 60%) and sufficient alphabet size (>= 16).
//
// Supports two modes (selected by params.per_subband):
//   0 = Single histogram per tile (with optional ZRL)
//   1 = Per-subband histograms (one per wavelet level group, with per-group ZRL)

const WG_SIZE: u32 = 256u;
const MAX_ALPHABET: u32 = 2048u;
const MAX_GROUP_ALPHABET: u32 = 2048u;
const MAX_GROUPS: u32 = 8u;
const MAX_ZERO_RUN: u32 = 256u;
const STREAMS_PER_TILE: u32 = 32u;

// Output stride per tile in u32s.
// Single-table: [min_val, alphabet_size, zrun_base, hist[0..MAX_ALPHABET]]
// Per-subband:  [num_groups, {min_val, alphabet_size, zrun_base, hist[0..MAX_GROUP_ALPHABET]} x groups]
const HIST_TILE_STRIDE: u32 = 16409u;  // 1 + MAX_GROUPS*(3+MAX_GROUP_ALPHABET)

struct Params {
    num_tiles: u32,
    coefficients_per_tile: u32,  // tile_size * tile_size
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    per_subband: u32,
    num_levels: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> hist_output: array<u32>;

// Shared memory for parallel reduction (min/max/zero_count/max_abs)
var<workgroup> shared_min: array<i32, 256>;
var<workgroup> shared_max: array<i32, 256>;
var<workgroup> shared_reduce_u: array<u32, 256>;

// Shared memory for histogram building (atomic)
// Size must fit sum of all group alphabets: up to MAX_GROUPS * MAX_GROUP_ALPHABET
var<workgroup> shared_hist: array<atomic<u32>, 5120>;

// Per-group metadata broadcast from thread 0
var<workgroup> shared_group_min: array<i32, 8>;
var<workgroup> shared_group_asize: array<u32, 8>;
var<workgroup> shared_group_hist_off: array<u32, 8>;
var<workgroup> shared_group_zrun: array<i32, 8>;
var<workgroup> shared_num_groups: u32;
// Flag: does any group have ZRL enabled?
var<workgroup> shared_any_zrl: u32;

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

// Read one coefficient from the plane buffer given tile-local coordinates.
fn read_coeff(tile_origin_x: u32, tile_origin_y: u32, local_idx: u32) -> i32 {
    let tile_row = local_idx / params.tile_size;
    let tile_col = local_idx % params.tile_size;
    let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                  + (tile_origin_x + tile_col);
    return i32(round(input[plane_idx]));
}

// Parallel reduction: compute sum across 256 threads using shared_reduce_u.
fn reduce_sum(lid: u32, val: u32) -> u32 {
    shared_reduce_u[lid] = val;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            shared_reduce_u[lid] += shared_reduce_u[lid + stride];
        }
        workgroupBarrier();
    }
    return shared_reduce_u[0];
}

// Parallel reduction: compute max(i32) across 256 threads using shared_min.
fn reduce_max_i32(lid: u32, val: i32) -> i32 {
    shared_min[lid] = val;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            if (shared_min[lid + stride] > shared_min[lid]) {
                shared_min[lid] = shared_min[lid + stride];
            }
        }
        workgroupBarrier();
    }
    return shared_min[0];
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

    let tile_x = tile_id % params.tiles_x;
    let tile_y = tile_id / params.tiles_x;
    let tile_origin_x = tile_x * params.tile_size;
    let tile_origin_y = tile_y * params.tile_size;

    let coeffs_per_thread = params.coefficients_per_tile / WG_SIZE;
    let out_base = tile_id * HIST_TILE_STRIDE;

    if (params.per_subband != 0u) {
        // --- Per-subband mode with ZRL ---
        let num_groups = params.num_levels + 1u;

        // Phase 1: Find per-group min/max, zero_count, max_abs_nonzero.
        // Each thread tracks local stats per group in registers.
        var local_min: array<i32, 8>;
        var local_max: array<i32, 8>;
        var local_has: array<bool, 8>;
        var local_zero_count: array<u32, 8>;
        var local_total_count: array<u32, 8>;
        var local_max_abs_nz: array<i32, 8>;

        for (var g = 0u; g < num_groups; g++) {
            local_min[g] = 2147483647;
            local_max[g] = -2147483647;
            local_has[g] = false;
            local_zero_count[g] = 0u;
            local_total_count[g] = 0u;
            local_max_abs_nz[g] = 0;
        }

        for (var j = 0u; j < coeffs_per_thread; j++) {
            let local_idx = lid + j * WG_SIZE;
            let c = read_coeff(tile_origin_x, tile_origin_y, local_idx);
            let tile_col = local_idx % params.tile_size;
            let tile_row = local_idx / params.tile_size;
            let g = compute_subband_group(tile_col, tile_row);
            local_has[g] = true;
            local_total_count[g] += 1u;
            if (c < local_min[g]) { local_min[g] = c; }
            if (c > local_max[g]) { local_max[g] = c; }
            if (c == 0) {
                local_zero_count[g] += 1u;
            } else {
                let a = abs(c);
                if (a > local_max_abs_nz[g]) { local_max_abs_nz[g] = a; }
            }
        }

        // Reduce per group sequentially (reuse shared arrays each iteration)
        for (var g = 0u; g < num_groups; g++) {
            // --- min/max reduction ---
            if (local_has[g]) {
                shared_min[lid] = local_min[g];
                shared_max[lid] = local_max[g];
            } else {
                shared_min[lid] = 2147483647;
                shared_max[lid] = -2147483647;
            }
            workgroupBarrier();

            for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
                if (lid < stride) {
                    if (shared_min[lid + stride] < shared_min[lid]) {
                        shared_min[lid] = shared_min[lid + stride];
                    }
                    if (shared_max[lid + stride] > shared_max[lid]) {
                        shared_max[lid] = shared_max[lid + stride];
                    }
                }
                workgroupBarrier();
            }

            // Save gmin/gmax BEFORE reductions that reuse shared_min.
            // reduce_max_i32() overwrites shared_min, so we must capture these first.
            let gmin_saved = shared_min[0];
            let gmax_saved = shared_max[0];

            // --- zero_count reduction ---
            let total_zeros = reduce_sum(lid, local_zero_count[g]);

            // --- total_count reduction ---
            let total_count = reduce_sum(lid, local_total_count[g]);

            // --- max_abs_nonzero reduction (overwrites shared_min!) ---
            let max_abs_nz = reduce_max_i32(lid, local_max_abs_nz[g]);

            // Thread 0 computes group metadata including ZRL decision
            if (lid == 0u) {
                let gmin = gmin_saved;
                let gmax = gmax_saved;

                if (gmax < gmin) {
                    // No data for this group — dummy single-symbol alphabet
                    shared_group_min[g] = 0;
                    shared_group_asize[g] = 1u;
                    shared_group_zrun[g] = 0;
                } else {
                    // ZRL decision: detail subbands only (g > 0), high zero density,
                    // sufficient non-ZRL alphabet size
                    var zrun_base_val = 0;
                    if (g > 0u && total_count > 0u) {
                        let non_zrl_asize = u32(2 * max_abs_nz + 1);
                        // Zero fraction check: >= 60% zeros
                        // (multiply to avoid float: total_zeros * 5 >= total_count * 3)
                        let zrl_eligible = (total_zeros * 5u >= total_count * 3u)
                                        && (non_zrl_asize >= 16u);
                        if (zrl_eligible) {
                            let candidate_zrun = max_abs_nz + 1;
                            // Check expanded alphabet fits in MAX_GROUP_ALPHABET
                            let expanded_max = candidate_zrun + i32(MAX_ZERO_RUN) - 1;
                            let expanded_asize = u32(expanded_max - gmin) + 1u;
                            if (expanded_asize <= MAX_GROUP_ALPHABET) {
                                zrun_base_val = candidate_zrun;
                            }
                        }
                    }
                    shared_group_zrun[g] = zrun_base_val;

                    if (zrun_base_val != 0) {
                        // ZRL: alphabet extends to include run symbols
                        // New max symbol = zrun_base + MAX_ZERO_RUN - 1
                        let new_max = zrun_base_val + i32(MAX_ZERO_RUN) - 1;
                        let new_asize = u32(new_max - gmin) + 1u;
                        shared_group_min[g] = gmin;
                        shared_group_asize[g] = min(new_asize, MAX_GROUP_ALPHABET);
                    } else {
                        // No ZRL: standard alphabet
                        shared_group_min[g] = gmin;
                        var asize = u32(gmax - gmin) + 1u;
                        if (asize > MAX_GROUP_ALPHABET) {
                            asize = MAX_GROUP_ALPHABET;
                        }
                        shared_group_asize[g] = asize;
                    }
                }
            }
            workgroupBarrier();
        }

        // Compute histogram offsets within shared_hist, check any_zrl
        if (lid == 0u) {
            shared_num_groups = num_groups;
            shared_any_zrl = 0u;
            var off = 0u;
            for (var g = 0u; g < num_groups; g++) {
                shared_group_hist_off[g] = off;
                off += shared_group_asize[g];
                if (shared_group_zrun[g] != 0) {
                    shared_any_zrl = 1u;
                }
            }
        }
        workgroupBarrier();

        // Initialize histogram to zero
        let total_hist_entries = shared_group_hist_off[num_groups - 1u]
                               + shared_group_asize[num_groups - 1u];
        for (var i = lid; i < total_hist_entries; i += WG_SIZE) {
            atomicStore(&shared_hist[i], 0u);
        }
        workgroupBarrier();

        // Phase 3: Build histogram
        if (shared_any_zrl != 0u) {
            // ZRL-aware histogram: 32 threads (one per rANS stream) do sequential
            // stride-32 scanning with per-group ZRL encoding.
            // Remaining threads idle — acceptable since ZRL scanning is inherently sequential.
            if (lid < STREAMS_PER_TILE) {
                let stream_id = lid;
                let symbols_per_stream = params.coefficients_per_tile / STREAMS_PER_TILE;

                var i = 0u;
                while (i < symbols_per_stream) {
                    let coeff_idx = stream_id + i * STREAMS_PER_TILE;
                    let c = read_coeff(tile_origin_x, tile_origin_y, coeff_idx);
                    let tile_col = coeff_idx % params.tile_size;
                    let tile_row = coeff_idx / params.tile_size;
                    let g = compute_subband_group(tile_col, tile_row);
                    let zrun = shared_group_zrun[g];

                    if (zrun != 0 && c == 0) {
                        // Count consecutive zeros in the same group within this stream
                        var run_len = 0u;
                        while (i < symbols_per_stream && run_len < MAX_ZERO_RUN) {
                            let ci = stream_id + i * STREAMS_PER_TILE;
                            let cc = read_coeff(tile_origin_x, tile_origin_y, ci);
                            let tc = ci % params.tile_size;
                            let tr = ci / params.tile_size;
                            let cg = compute_subband_group(tc, tr);
                            // Run continues only if zero AND same group
                            if (cc != 0 || cg != g) { break; }
                            run_len += 1u;
                            i += 1u;
                        }
                        // Emit run symbol: zrun_base + (run_len - 1)
                        let run_sym_val = zrun + i32(run_len) - 1;
                        var sym = u32(run_sym_val - shared_group_min[g]);
                        if (sym >= shared_group_asize[g]) {
                            sym = shared_group_asize[g] - 1u;
                        }
                        let hist_idx = shared_group_hist_off[g] + sym;
                        atomicAdd(&shared_hist[hist_idx], 1u);
                    } else {
                        // Normal coefficient
                        var sym = u32(c - shared_group_min[g]);
                        if (sym >= shared_group_asize[g]) {
                            sym = shared_group_asize[g] - 1u;
                        }
                        let hist_idx = shared_group_hist_off[g] + sym;
                        atomicAdd(&shared_hist[hist_idx], 1u);
                        i += 1u;
                    }
                }
            }
        } else {
            // No ZRL in any group: fast parallel histogram (256 threads)
            for (var j = 0u; j < coeffs_per_thread; j++) {
                let local_idx = lid + j * WG_SIZE;
                let c = read_coeff(tile_origin_x, tile_origin_y, local_idx);
                let tile_col = local_idx % params.tile_size;
                let tile_row = local_idx / params.tile_size;
                let g = compute_subband_group(tile_col, tile_row);

                var sym = u32(c - shared_group_min[g]);
                if (sym >= shared_group_asize[g]) {
                    sym = shared_group_asize[g] - 1u;
                }
                let hist_idx = shared_group_hist_off[g] + sym;
                atomicAdd(&shared_hist[hist_idx], 1u);
            }
        }
        workgroupBarrier();

        // Write output: [num_groups, {min_val, alphabet_size, zrun_base, hist[]} x groups]
        if (lid == 0u) {
            hist_output[out_base] = num_groups;
        }
        // Per-group output (cooperative write)
        var write_off = 1u;
        for (var g = 0u; g < num_groups; g++) {
            let asize = shared_group_asize[g];
            if (lid == 0u) {
                hist_output[out_base + write_off] = bitcast<u32>(shared_group_min[g]);
                hist_output[out_base + write_off + 1u] = asize;
                hist_output[out_base + write_off + 2u] = bitcast<u32>(shared_group_zrun[g]);
            }
            // Write histogram entries cooperatively
            let hist_base = shared_group_hist_off[g];
            for (var i = lid; i < asize; i += WG_SIZE) {
                hist_output[out_base + write_off + 3u + i] = atomicLoad(&shared_hist[hist_base + i]);
            }
            write_off += 3u + asize;
        }

    } else {
        // --- Single-table mode with optional ZRL ---

        // Phase 1: Find min/max, zero count, max_abs via parallel reduction
        var local_min: i32 = 2147483647;
        var local_max: i32 = -2147483647;
        var local_zero_count: u32 = 0u;
        var local_max_abs_nz: i32 = 0;
        var local_total: u32 = 0u;

        for (var j = 0u; j < coeffs_per_thread; j++) {
            let local_idx = lid + j * WG_SIZE;
            let c = read_coeff(tile_origin_x, tile_origin_y, local_idx);
            if (c < local_min) { local_min = c; }
            if (c > local_max) { local_max = c; }
            local_total += 1u;
            if (c == 0) {
                local_zero_count += 1u;
            } else {
                let a = abs(c);
                if (a > local_max_abs_nz) { local_max_abs_nz = a; }
            }
        }

        // min/max reduction
        shared_min[lid] = local_min;
        shared_max[lid] = local_max;
        workgroupBarrier();

        for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
            if (lid < stride) {
                if (shared_min[lid + stride] < shared_min[lid]) {
                    shared_min[lid] = shared_min[lid + stride];
                }
                if (shared_max[lid + stride] > shared_max[lid]) {
                    shared_max[lid] = shared_max[lid + stride];
                }
            }
            workgroupBarrier();
        }

        let gmin = shared_min[0];
        let gmax = shared_max[0];

        // zero count reduction
        let total_zeros = reduce_sum(lid, local_zero_count);
        // total count reduction
        let total_count = reduce_sum(lid, local_total);
        // max_abs_nonzero reduction
        let max_abs_nz = reduce_max_i32(lid, local_max_abs_nz);

        // Thread 0 decides ZRL and broadcasts
        if (lid == 0u) {
            var zrun_base_val = 0;
            if (total_count > 0u) {
                let non_zrl_asize = u32(2 * max_abs_nz + 1);
                let zrl_eligible = (total_zeros * 5u >= total_count * 3u)
                                && (non_zrl_asize >= 16u);
                if (zrl_eligible) {
                    let candidate_zrun = max_abs_nz + 1;
                    let expanded_max = candidate_zrun + i32(MAX_ZERO_RUN) - 1;
                    let expanded_asize = u32(expanded_max - gmin) + 1u;
                    if (expanded_asize <= MAX_ALPHABET) {
                        zrun_base_val = candidate_zrun;
                    }
                }
            }
            shared_group_zrun[0] = zrun_base_val;

            if (zrun_base_val != 0) {
                let new_max = zrun_base_val + i32(MAX_ZERO_RUN) - 1;
                let new_asize = min(u32(new_max - gmin) + 1u, MAX_ALPHABET);
                shared_group_min[0] = gmin;
                shared_group_asize[0] = new_asize;
            } else {
                shared_group_min[0] = gmin;
                var asize = u32(gmax - gmin) + 1u;
                if (asize > MAX_ALPHABET) { asize = MAX_ALPHABET; }
                shared_group_asize[0] = asize;
            }
        }
        workgroupBarrier();

        let min_val = shared_group_min[0];
        let alphabet_size = shared_group_asize[0];
        let zrun_base = shared_group_zrun[0];

        // Initialize histogram to zero
        for (var i = lid; i < alphabet_size; i += WG_SIZE) {
            atomicStore(&shared_hist[i], 0u);
        }
        workgroupBarrier();

        // Phase 3: Build histogram
        if (zrun_base != 0) {
            // ZRL-aware histogram: 32 threads, one per rANS stream
            if (lid < STREAMS_PER_TILE) {
                let stream_id = lid;
                let symbols_per_stream = params.coefficients_per_tile / STREAMS_PER_TILE;

                var i = 0u;
                while (i < symbols_per_stream) {
                    let coeff_idx = stream_id + i * STREAMS_PER_TILE;
                    let c = read_coeff(tile_origin_x, tile_origin_y, coeff_idx);

                    if (c == 0) {
                        // Count consecutive zeros in this stream
                        var run_len = 0u;
                        while (i < symbols_per_stream && run_len < MAX_ZERO_RUN) {
                            let ci = stream_id + i * STREAMS_PER_TILE;
                            let cc = read_coeff(tile_origin_x, tile_origin_y, ci);
                            if (cc != 0) { break; }
                            run_len += 1u;
                            i += 1u;
                        }
                        let run_sym_val = zrun_base + i32(run_len) - 1;
                        var sym = u32(run_sym_val - min_val);
                        if (sym >= alphabet_size) { sym = alphabet_size - 1u; }
                        atomicAdd(&shared_hist[sym], 1u);
                    } else {
                        var sym = u32(c - min_val);
                        if (sym >= alphabet_size) { sym = alphabet_size - 1u; }
                        atomicAdd(&shared_hist[sym], 1u);
                        i += 1u;
                    }
                }
            }
        } else {
            // No ZRL: fast parallel histogram (256 threads)
            for (var j = 0u; j < coeffs_per_thread; j++) {
                let local_idx = lid + j * WG_SIZE;
                let c = read_coeff(tile_origin_x, tile_origin_y, local_idx);
                var sym = u32(c - min_val);
                if (sym >= alphabet_size) { sym = alphabet_size - 1u; }
                atomicAdd(&shared_hist[sym], 1u);
            }
        }
        workgroupBarrier();

        // Write output: [min_val, alphabet_size, zrun_base, hist[0..alphabet_size]]
        if (lid == 0u) {
            hist_output[out_base] = bitcast<u32>(min_val);
            hist_output[out_base + 1u] = alphabet_size;
            hist_output[out_base + 2u] = bitcast<u32>(zrun_base);
        }
        for (var i = lid; i < alphabet_size; i += WG_SIZE) {
            hist_output[out_base + 3u + i] = atomicLoad(&shared_hist[i]);
        }
    }
}
