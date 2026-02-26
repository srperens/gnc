// Fused quantize + histogram shader.
//
// Combines scalar quantization (from quantize.wgsl) with per-tile histogram
// building (from rans_histogram.wgsl) in a single dispatch pass. This
// eliminates one full read+write of the coefficient buffer that would
// otherwise occur between separate quantize and histogram dispatches.
//
// Dispatch model: 1 workgroup (256 threads) per tile, like rans_histogram.
// Each thread:
//   1. Reads wavelet coefficients from the input buffer
//   2. Applies subband-aware quantization (dead zone, subband weights, adaptive QP)
//   3. Writes quantized values (as f32) to the output buffer
//   4. Accumulates histogram counts in workgroup shared memory via atomics
//
// The histogram output format matches rans_histogram.wgsl exactly so that
// downstream normalize + encode passes work unchanged.
//
// Supports per-subband mode only (per_subband == 1) since that is the
// production path used with GPU entropy encoding. The single-table mode
// could be added later if needed.

const WG_SIZE: u32 = 256u;
const MAX_ALPHABET: u32 = 4096u;
const MAX_GROUP_ALPHABET: u32 = 4096u;
const MAX_GROUPS: u32 = 8u;
const MAX_ZERO_RUN: u32 = 256u;
const STREAMS_PER_TILE: u32 = 32u;

// Output stride per tile in u32s (matches rans_histogram.wgsl).
// Per-subband: [num_groups, {min_val, alphabet_size, zrun_base, hist[0..MAX_GROUP_ALPHABET]} x groups]
const HIST_TILE_STRIDE: u32 = 32793u;  // 1 + MAX_GROUPS*(3+MAX_GROUP_ALPHABET)

struct Params {
    // -- Histogram params (first, to match rans_histogram layout) --
    num_tiles: u32,
    coefficients_per_tile: u32,  // tile_size * tile_size
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    per_subband: u32,
    num_levels: u32,
    flags: u32,        // bit 0: disable ZRL

    // -- Quantize params --
    total_count: u32,
    step_size: f32,
    dead_zone: f32,
    _pad0: u32,
    // Packed subband weights: [LL, L0_LH, L0_HL, L0_HH, L1_LH, ...]
    weights0: vec4<f32>,
    weights1: vec4<f32>,
    weights2: vec4<f32>,
    weights3: vec4<f32>,
    // Adaptive quantization params
    aq_enabled: u32,
    aq_ll_block_size: u32,
    aq_ll_blocks_per_tile_x: u32,
    aq_tiles_x: u32,
    // Plane height (needed for 2D position calculation in quantize)
    plane_height: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Input: wavelet coefficients (f32, pre-quantization)
@group(0) @binding(1) var<storage, read> input: array<f32>;
// Output: quantized coefficients (f32, post-quantization)
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
// Histogram output (same format as rans_histogram.wgsl)
@group(0) @binding(3) var<storage, read_write> hist_output: array<u32>;
// Weight map for adaptive quantization (read when aq_enabled == 1)
@group(0) @binding(4) var<storage, read> weight_map: array<f32>;

// ---- Shared memory for parallel reduction ----
var<workgroup> shared_min: array<i32, 256>;
var<workgroup> shared_max: array<i32, 256>;
var<workgroup> shared_reduce_u: array<u32, 256>;

// ---- Shared memory for histogram building (atomic) ----
var<workgroup> shared_hist: array<atomic<u32>, 5120>;

// Per-group metadata broadcast from thread 0
var<workgroup> shared_group_min: array<i32, 8>;
var<workgroup> shared_group_asize: array<u32, 8>;
var<workgroup> shared_group_hist_off: array<u32, 8>;
var<workgroup> shared_group_zrun: array<i32, 8>;
var<workgroup> shared_num_groups: u32;
var<workgroup> shared_any_zrl: u32;

// ---- Quantization helpers (from quantize.wgsl) ----

fn get_weight(index: u32) -> f32 {
    let vec_idx = index / 4u;
    let comp_idx = index % 4u;
    var v: vec4<f32>;
    switch vec_idx {
        case 0u: { v = params.weights0; }
        case 1u: { v = params.weights1; }
        case 2u: { v = params.weights2; }
        default: { v = params.weights3; }
    }
    return v[comp_idx];
}

fn compute_subband_index(lx: u32, ly: u32) -> u32 {
    var region = params.tile_size;
    for (var level = 0u; level < params.num_levels; level = level + 1u) {
        let half = region / 2u;
        let in_right = lx >= half;
        let in_bottom = ly >= half;
        if in_right || in_bottom {
            if in_right && in_bottom {
                return 1u + level * 3u + 2u; // HH
            } else if in_right {
                return 1u + level * 3u + 1u; // HL
            } else {
                return 1u + level * 3u;      // LH
            }
        }
        region = half;
    }
    return 0u; // LL
}

fn get_ll_block_xy(lx: u32, ly: u32) -> vec2<u32> {
    let ll_size = params.tile_size >> params.num_levels;
    let max_b = params.aq_ll_blocks_per_tile_x - 1u;

    var region = params.tile_size;
    for (var level = 0u; level < params.num_levels; level = level + 1u) {
        let half = region / 2u;
        let in_right = lx >= half;
        let in_bottom = ly >= half;
        if in_right || in_bottom {
            var sub_x = lx;
            var sub_y = ly;
            if in_right { sub_x = lx - half; }
            if in_bottom { sub_y = ly - half; }

            let ll_x = (sub_x * ll_size) / half;
            let ll_y = (sub_y * ll_size) / half;
            let bx = min(ll_x / params.aq_ll_block_size, max_b);
            let by = min(ll_y / params.aq_ll_block_size, max_b);
            return vec2<u32>(bx, by);
        }
        region = half;
    }
    let bx = min(lx / params.aq_ll_block_size, max_b);
    let by = min(ly / params.aq_ll_block_size, max_b);
    return vec2<u32>(bx, by);
}

fn get_spatial_weight(x: u32, y: u32) -> f32 {
    if params.aq_enabled == 0u {
        return 1.0;
    }
    let tile_x = x / params.tile_size;
    let tile_y = y / params.tile_size;
    let lx = x % params.tile_size;
    let ly = y % params.tile_size;
    let lb = get_ll_block_xy(lx, ly);
    let global_bx = tile_x * params.aq_ll_blocks_per_tile_x + lb.x;
    let global_by = tile_y * params.aq_ll_blocks_per_tile_x + lb.y;
    let global_blocks_x = params.aq_ll_blocks_per_tile_x * params.aq_tiles_x;
    let block_idx = global_by * global_blocks_x + global_bx;
    return weight_map[block_idx];
}

// Quantize a single coefficient value. Returns the quantized integer.
fn quantize_coeff(val: f32, effective_step: f32) -> i32 {
    let abs_val = abs(val);
    let sign_val = sign(val);
    let threshold = params.dead_zone * effective_step;
    if abs_val < threshold {
        return 0;
    }
    return i32(sign_val * floor(abs_val / effective_step + 0.5));
}

// ---- Histogram helpers (from rans_histogram.wgsl) ----

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

// Read one coefficient from the plane, quantize it, and write to output.
// Returns the quantized integer value.
fn quantize_and_read(tile_origin_x: u32, tile_origin_y: u32, local_idx: u32) -> i32 {
    let tile_row = local_idx / params.tile_size;
    let tile_col = local_idx % params.tile_size;
    let x = tile_origin_x + tile_col;
    let y = tile_origin_y + tile_row;
    let plane_idx = y * params.plane_width + x;

    // Quantize
    let lx = tile_col;
    let ly = tile_row;
    let subband_weight = get_weight(compute_subband_index(lx, ly));
    let spatial_weight = get_spatial_weight(x, y);
    let effective_step = params.step_size * subband_weight * spatial_weight;

    let val = input[plane_idx];
    let q = quantize_coeff(val, effective_step);

    // Write quantized value to output buffer (as f32, matching original pipeline)
    output[plane_idx] = f32(q);

    return q;
}

// Parallel reduction: sum across 256 threads using shared_reduce_u.
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

// Parallel reduction: max(i32) across 256 threads using shared_min.
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
        // --- Per-subband mode ---
        let num_groups = params.num_levels + 1u;

        // Phase 1: Quantize all coefficients, track per-group min/max/zeros/max_abs
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
            // Quantize + write + return quantized integer
            let c = quantize_and_read(tile_origin_x, tile_origin_y, local_idx);
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

            let gmin_saved = shared_min[0];
            let gmax_saved = shared_max[0];

            let total_zeros = reduce_sum(lid, local_zero_count[g]);
            let total_count = reduce_sum(lid, local_total_count[g]);
            let max_abs_nz = reduce_max_i32(lid, local_max_abs_nz[g]);

            if (lid == 0u) {
                let gmin = gmin_saved;
                let gmax = gmax_saved;

                if (gmax < gmin) {
                    shared_group_min[g] = 0;
                    shared_group_asize[g] = 1u;
                    shared_group_zrun[g] = 0;
                } else {
                    var zrun_base_val = 0;
                    if ((params.flags & 1u) == 0u && g > 0u && total_count > 0u) {
                        let non_zrl_asize = u32(2 * max_abs_nz + 1);
                        let zrl_eligible = (total_zeros * 5u >= total_count * 3u)
                                        && (non_zrl_asize >= 16u);
                        if (zrl_eligible) {
                            let candidate_zrun = max_abs_nz + 1;
                            let expanded_max = candidate_zrun + i32(MAX_ZERO_RUN) - 1;
                            let expanded_asize = u32(expanded_max - gmin) + 1u;
                            if (expanded_asize <= MAX_GROUP_ALPHABET) {
                                zrun_base_val = candidate_zrun;
                            }
                        }
                    }
                    shared_group_zrun[g] = zrun_base_val;

                    if (zrun_base_val != 0) {
                        let new_max = zrun_base_val + i32(MAX_ZERO_RUN) - 1;
                        let new_asize = u32(new_max - gmin) + 1u;
                        shared_group_min[g] = gmin;
                        shared_group_asize[g] = min(new_asize, MAX_GROUP_ALPHABET);
                    } else {
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

        // Compute histogram offsets, check any_zrl
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

        // Phase 2: Build histogram from already-quantized output buffer
        // (Re-read the quantized values we just wrote; this is an L1 cache hit
        //  since we wrote them moments ago, much cheaper than a full VRAM read.)
        if (shared_any_zrl != 0u) {
            // ZRL-aware: 32 threads sequential scanning
            if (lid < STREAMS_PER_TILE) {
                let stream_id = lid;
                let symbols_per_stream = params.coefficients_per_tile / STREAMS_PER_TILE;

                var i = 0u;
                while (i < symbols_per_stream) {
                    let coeff_idx = stream_id + i * STREAMS_PER_TILE;
                    let tile_row = coeff_idx / params.tile_size;
                    let tile_col = coeff_idx % params.tile_size;
                    let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                                  + (tile_origin_x + tile_col);
                    let c = i32(round(output[plane_idx]));
                    let g = compute_subband_group(tile_col, tile_row);
                    let zrun = shared_group_zrun[g];

                    if (zrun != 0 && c == 0) {
                        var run_len = 0u;
                        while (i < symbols_per_stream && run_len < MAX_ZERO_RUN) {
                            let ci = stream_id + i * STREAMS_PER_TILE;
                            let tr = ci / params.tile_size;
                            let tc = ci % params.tile_size;
                            let pi = (tile_origin_y + tr) * params.plane_width
                                   + (tile_origin_x + tc);
                            let cc = i32(round(output[pi]));
                            let cg = compute_subband_group(tc, tr);
                            if (cc != 0 || cg != g) { break; }
                            run_len += 1u;
                            i += 1u;
                        }
                        let run_sym_val = zrun + i32(run_len) - 1;
                        var sym = u32(run_sym_val - shared_group_min[g]);
                        if (sym >= shared_group_asize[g]) {
                            sym = shared_group_asize[g] - 1u;
                        }
                        let hist_idx = shared_group_hist_off[g] + sym;
                        atomicAdd(&shared_hist[hist_idx], 1u);
                    } else {
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
            // No ZRL: fast parallel histogram (all 256 threads)
            for (var j = 0u; j < coeffs_per_thread; j++) {
                let local_idx = lid + j * WG_SIZE;
                let tile_row = local_idx / params.tile_size;
                let tile_col = local_idx % params.tile_size;
                let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                              + (tile_origin_x + tile_col);
                let c = i32(round(output[plane_idx]));
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
        var write_off = 1u;
        for (var g = 0u; g < num_groups; g++) {
            let asize = shared_group_asize[g];
            if (lid == 0u) {
                hist_output[out_base + write_off] = bitcast<u32>(shared_group_min[g]);
                hist_output[out_base + write_off + 1u] = asize;
                hist_output[out_base + write_off + 2u] = bitcast<u32>(shared_group_zrun[g]);
            }
            let hist_base = shared_group_hist_off[g];
            for (var i = lid; i < asize; i += WG_SIZE) {
                hist_output[out_base + write_off + 3u + i] = atomicLoad(&shared_hist[hist_base + i]);
            }
            write_off += 3u + asize;
        }

    } else {
        // --- Single-table mode ---

        // Phase 1: Quantize + track stats
        var local_min: i32 = 2147483647;
        var local_max: i32 = -2147483647;
        var local_zero_count: u32 = 0u;
        var local_max_abs_nz: i32 = 0;
        var local_total: u32 = 0u;

        for (var j = 0u; j < coeffs_per_thread; j++) {
            let local_idx = lid + j * WG_SIZE;
            let c = quantize_and_read(tile_origin_x, tile_origin_y, local_idx);
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

        let total_zeros = reduce_sum(lid, local_zero_count);
        let total_count = reduce_sum(lid, local_total);
        let max_abs_nz = reduce_max_i32(lid, local_max_abs_nz);

        if (lid == 0u) {
            var zrun_base_val = 0;
            if ((params.flags & 1u) == 0u && total_count > 0u) {
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

        // Build histogram from quantized output
        if (zrun_base != 0) {
            if (lid < STREAMS_PER_TILE) {
                let stream_id = lid;
                let symbols_per_stream = params.coefficients_per_tile / STREAMS_PER_TILE;

                var i = 0u;
                while (i < symbols_per_stream) {
                    let coeff_idx = stream_id + i * STREAMS_PER_TILE;
                    let tile_row = coeff_idx / params.tile_size;
                    let tile_col = coeff_idx % params.tile_size;
                    let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                                  + (tile_origin_x + tile_col);
                    let c = i32(round(output[plane_idx]));

                    if (c == 0) {
                        var run_len = 0u;
                        while (i < symbols_per_stream && run_len < MAX_ZERO_RUN) {
                            let ci = stream_id + i * STREAMS_PER_TILE;
                            let tr = ci / params.tile_size;
                            let tc = ci % params.tile_size;
                            let pi = (tile_origin_y + tr) * params.plane_width
                                   + (tile_origin_x + tc);
                            let cc = i32(round(output[pi]));
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
            for (var j = 0u; j < coeffs_per_thread; j++) {
                let local_idx = lid + j * WG_SIZE;
                let tile_row = local_idx / params.tile_size;
                let tile_col = local_idx % params.tile_size;
                let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                              + (tile_origin_x + tile_col);
                let c = i32(round(output[plane_idx]));
                var sym = u32(c - min_val);
                if (sym >= alphabet_size) { sym = alphabet_size - 1u; }
                atomicAdd(&shared_hist[sym], 1u);
            }
        }
        workgroupBarrier();

        // Write output
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
