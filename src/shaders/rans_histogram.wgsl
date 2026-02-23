// GPU histogram builder for rANS entropy coding.
//
// 256 threads per workgroup, 1 workgroup per tile.
// Two phases within one dispatch:
//   Phase 1: Find min/max per tile (parallel reduction in shared memory)
//   Phase 2: Build histogram using workgroup-shared atomics
//
// Supports two modes (selected by params.per_subband):
//   0 = Single histogram per tile
//   1 = Per-subband histograms (one per wavelet level group)

const WG_SIZE: u32 = 256u;
const MAX_ALPHABET: u32 = 2048u;
const MAX_GROUP_ALPHABET: u32 = 512u;
const MAX_GROUPS: u32 = 8u;

// Output stride per tile in u32s.
// Single-table: [min_val, alphabet_size, hist[0..MAX_ALPHABET]]
// Per-subband:  [num_groups, {min_val, alphabet_size, hist[0..MAX_GROUP_ALPHABET]} x groups]
const HIST_TILE_STRIDE: u32 = 2060u;

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

// Shared memory for parallel reduction (min/max)
var<workgroup> shared_min: array<i32, 256>;
var<workgroup> shared_max: array<i32, 256>;

// Shared memory for histogram building (atomic)
var<workgroup> shared_hist: array<atomic<u32>, 2048>;

// Per-group metadata broadcast from thread 0
var<workgroup> shared_group_min: array<i32, 8>;
var<workgroup> shared_group_asize: array<u32, 8>;
var<workgroup> shared_group_hist_off: array<u32, 8>;
var<workgroup> shared_num_groups: u32;

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

        // Phase 1: Find per-group min/max via sequential group reductions.
        // Each thread tracks local min/max per group in registers.
        var local_min: array<i32, 8>;
        var local_max: array<i32, 8>;
        var local_has: array<bool, 8>;
        for (var g = 0u; g < num_groups; g++) {
            local_min[g] = 2147483647; // i32 MAX
            local_max[g] = -2147483647; // ~i32 MIN
            local_has[g] = false;
        }

        for (var j = 0u; j < coeffs_per_thread; j++) {
            let local_idx = lid + j * WG_SIZE;
            let c = read_coeff(tile_origin_x, tile_origin_y, local_idx);
            let tile_col = local_idx % params.tile_size;
            let tile_row = local_idx / params.tile_size;
            let g = compute_subband_group(tile_col, tile_row);
            local_has[g] = true;
            if (c < local_min[g]) { local_min[g] = c; }
            if (c > local_max[g]) { local_max[g] = c; }
        }

        // Reduce per group sequentially (reuse shared_min/shared_max each iteration)
        for (var g = 0u; g < num_groups; g++) {
            if (local_has[g]) {
                shared_min[lid] = local_min[g];
                shared_max[lid] = local_max[g];
            } else {
                shared_min[lid] = 2147483647;
                shared_max[lid] = -2147483647;
            }
            workgroupBarrier();

            // Tree reduction
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

            // Thread 0 broadcasts group metadata
            if (lid == 0u) {
                let gmin = shared_min[0];
                let gmax = shared_max[0];
                var asize: u32;
                if (gmax < gmin) {
                    // No data for this group — use dummy single-symbol alphabet
                    shared_group_min[g] = 0;
                    shared_group_asize[g] = 1u;
                } else {
                    shared_group_min[g] = gmin;
                    asize = u32(gmax - gmin) + 1u;
                    if (asize > MAX_GROUP_ALPHABET) {
                        asize = MAX_GROUP_ALPHABET;
                    }
                    shared_group_asize[g] = asize;
                }
            }
            workgroupBarrier();
        }

        // Compute histogram offsets within shared_hist
        if (lid == 0u) {
            shared_num_groups = num_groups;
            var off = 0u;
            for (var g = 0u; g < num_groups; g++) {
                shared_group_hist_off[g] = off;
                off += shared_group_asize[g];
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

        // Phase 2: Build per-group histograms
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
        workgroupBarrier();

        // Write output: [num_groups, {min_val, alphabet_size, hist[]} x groups]
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
            }
            // Write histogram entries cooperatively
            let hist_base = shared_group_hist_off[g];
            for (var i = lid; i < asize; i += WG_SIZE) {
                hist_output[out_base + write_off + 2u + i] = atomicLoad(&shared_hist[hist_base + i]);
            }
            write_off += 2u + asize;
        }

    } else {
        // --- Single-table mode ---

        // Phase 1: Find min/max via parallel reduction
        var local_min: i32 = 2147483647;
        var local_max: i32 = -2147483647;

        for (var j = 0u; j < coeffs_per_thread; j++) {
            let local_idx = lid + j * WG_SIZE;
            let c = read_coeff(tile_origin_x, tile_origin_y, local_idx);
            if (c < local_min) { local_min = c; }
            if (c > local_max) { local_max = c; }
        }

        shared_min[lid] = local_min;
        shared_max[lid] = local_max;
        workgroupBarrier();

        // Tree reduction
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

        // Thread 0 computes alphabet_size and broadcasts
        if (lid == 0u) {
            let gmin = shared_min[0];
            let gmax = shared_max[0];
            var asize = u32(gmax - gmin) + 1u;
            if (asize > MAX_ALPHABET) {
                asize = MAX_ALPHABET;
            }
            shared_group_min[0] = gmin;
            shared_group_asize[0] = asize;
        }
        workgroupBarrier();

        let min_val = shared_group_min[0];
        let alphabet_size = shared_group_asize[0];

        // Initialize histogram to zero
        for (var i = lid; i < alphabet_size; i += WG_SIZE) {
            atomicStore(&shared_hist[i], 0u);
        }
        workgroupBarrier();

        // Phase 2: Build histogram
        for (var j = 0u; j < coeffs_per_thread; j++) {
            let local_idx = lid + j * WG_SIZE;
            let c = read_coeff(tile_origin_x, tile_origin_y, local_idx);
            var sym = u32(c - min_val);
            if (sym >= alphabet_size) {
                sym = alphabet_size - 1u;
            }
            atomicAdd(&shared_hist[sym], 1u);
        }
        workgroupBarrier();

        // Write output: [min_val, alphabet_size, hist[0..alphabet_size]]
        if (lid == 0u) {
            hist_output[out_base] = bitcast<u32>(min_val);
            hist_output[out_base + 1u] = alphabet_size;
        }
        for (var i = lid; i < alphabet_size; i += WG_SIZE) {
            hist_output[out_base + 2u + i] = atomicLoad(&shared_hist[i]);
        }
    }
}
