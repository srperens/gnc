// GPU histogram normalization + cumfreq builder for rANS entropy coding.
//
// Reads raw histograms from the histogram shader output (including ZRL metadata),
// normalizes frequencies to sum exactly to RANS_M, and builds cumulative frequency
// tables.  This eliminates the CPU roundtrip between histogram and encode
// dispatches — the entire histogram→normalize→encode chain runs on GPU
// in a single command encoder submission.
//
// 256 threads per workgroup, 1 workgroup per tile.

const WG_SIZE: u32 = 256u;
const RANS_M: u32 = 4096u;           // 1 << 12, must match RANS_PRECISION
const MAX_ALPHABET: u32 = 4096u;
const MAX_GROUP_ALPHABET: u32 = 4096u;
const MAX_GROUPS: u32 = 8u;
const HIST_TILE_STRIDE: u32 = 32793u;  // 1 + MAX_GROUPS*(3+MAX_GROUP_ALPHABET)
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
@group(0) @binding(2) var<storage, read_write> cumfreq_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> tile_info_out: array<u32>;

// Shared memory for normalization
var<workgroup> shared_freq: array<u32, 4096>;
var<workgroup> shared_sum: array<u32, 256>;
var<workgroup> w_total: u32;
var<workgroup> w_assigned: u32;
var<workgroup> w_max_idx: u32;

// Per-subband group metadata (broadcast by thread 0)
var<workgroup> w_min_val: u32;
var<workgroup> w_asize: u32;
var<workgroup> w_zrun_base: u32;
var<workgroup> w_hist_data_off: u32;
var<workgroup> w_hist_read_off: u32;

// Normalize shared_freq[0..asize] in-place so the entries sum to exactly RANS_M.
// Non-zero histogram entries get at least freq 1.  The deficit/surplus is
// applied to the largest entry (simple, GPU-friendly, negligible quality impact).
fn normalize_inplace(lid: u32, asize: u32) {
    // --- Step 1: compute total = sum of raw histogram ---
    var local_sum = 0u;
    for (var i = lid; i < asize; i += WG_SIZE) {
        local_sum += shared_freq[i];
    }
    shared_sum[lid] = local_sum;
    workgroupBarrier();

    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        w_total = shared_sum[0];
    }
    workgroupBarrier();

    let total = w_total;
    if (total == 0u) {
        // Degenerate: no data — put all mass on symbol 0
        if (lid == 0u) {
            shared_freq[0] = RANS_M;
        }
        workgroupBarrier();
        return;
    }

    // --- Step 2: scale, ensuring minimum 1 for non-zero entries ---
    for (var i = lid; i < asize; i += WG_SIZE) {
        let h = shared_freq[i];
        if (h > 0u) {
            var scaled = (h * RANS_M) / total;
            if (scaled == 0u) { scaled = 1u; }
            shared_freq[i] = scaled;
        }
    }
    workgroupBarrier();

    // --- Step 3: compute assigned = sum of scaled freqs ---
    var local_assigned = 0u;
    for (var i = lid; i < asize; i += WG_SIZE) {
        local_assigned += shared_freq[i];
    }
    shared_sum[lid] = local_assigned;
    workgroupBarrier();

    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        workgroupBarrier();
    }

    // --- Step 4: adjust largest entry to hit exact target ---
    if (lid == 0u) {
        let assigned = shared_sum[0];

        // Find the max entry
        var max_val = 0u;
        var max_i = 0u;
        for (var i = 0u; i < asize; i++) {
            if (shared_freq[i] > max_val) {
                max_val = shared_freq[i];
                max_i = i;
            }
        }

        if (assigned < RANS_M) {
            // Deficit — add to the largest entry
            shared_freq[max_i] += RANS_M - assigned;
        } else if (assigned > RANS_M) {
            // Surplus — subtract from largest, keeping >= 1
            var surplus = assigned - RANS_M;
            if (shared_freq[max_i] > surplus + 1u) {
                shared_freq[max_i] -= surplus;
            } else {
                // Rare: need to distribute across multiple entries
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

    if (params.per_subband != 0u) {
        // ===== Per-subband mode =====
        // Histogram layout per group: [min_val, alphabet_size, zrun_base, hist[0..asize]]
        let num_groups = hist_input[hist_base];
        let cf_group_stride = MAX_GROUP_ALPHABET + 1u;
        let cf_tile_stride = MAX_GROUPS * cf_group_stride;
        let cf_tile_base = tile_id * cf_tile_stride;

        // Write num_groups to tile_info
        if (lid == 0u) {
            tile_info_out[info_base] = num_groups;
            w_hist_read_off = 1u;  // past num_groups
        }
        workgroupBarrier();

        for (var g = 0u; g < num_groups; g++) {
            // Thread 0 reads and broadcasts group metadata
            if (lid == 0u) {
                let off = w_hist_read_off;
                w_min_val = hist_input[hist_base + off];
                w_asize = hist_input[hist_base + off + 1u];
                w_zrun_base = hist_input[hist_base + off + 2u];
                w_hist_data_off = off + 3u;  // histogram data starts after min_val, asize, zrun_base
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

            // Build cumfreq (sequential prefix sum by thread 0)
            let cf_base = cf_tile_base + g * cf_group_stride;
            if (lid == 0u) {
                cumfreq_out[cf_base] = 0u;
                for (var i = 0u; i < asize; i++) {
                    cumfreq_out[cf_base + i + 1u] = cumfreq_out[cf_base + i] + shared_freq[i];
                }

                // Write tile_info for this group (4 u32s per group)
                let gi = info_base + 1u + g * 4u;
                tile_info_out[gi] = w_min_val;
                tile_info_out[gi + 1u] = asize;
                tile_info_out[gi + 2u] = cf_base;
                tile_info_out[gi + 3u] = w_zrun_base;  // pass through ZRL base from histogram

                // Advance read offset for next group
                w_hist_read_off = w_hist_data_off + asize;
            }
            workgroupBarrier();
        }

    } else {
        // ===== Single-table mode =====
        // Histogram layout: [min_val, alphabet_size, zrun_base, hist[0..asize]]
        let min_val = hist_input[hist_base];
        let asize = hist_input[hist_base + 1u];
        let zrun_base = hist_input[hist_base + 2u];

        // Load histogram into shared_freq (data starts at offset 3)
        for (var i = lid; i < asize; i += WG_SIZE) {
            shared_freq[i] = hist_input[hist_base + 3u + i];
        }
        workgroupBarrier();

        // Normalize
        normalize_inplace(lid, asize);

        // Build cumfreq (sequential prefix sum by thread 0)
        let cf_stride = MAX_ALPHABET + 1u;
        let cf_base = tile_id * cf_stride;
        if (lid == 0u) {
            cumfreq_out[cf_base] = 0u;
            for (var i = 0u; i < asize; i++) {
                cumfreq_out[cf_base + i + 1u] = cumfreq_out[cf_base + i] + shared_freq[i];
            }
        }

        // Write tile_info: [min_val, alphabet_size, cumfreq_offset, zrun_base]
        if (lid == 0u) {
            tile_info_out[info_base] = min_val;
            tile_info_out[info_base + 1u] = asize;
            tile_info_out[info_base + 2u] = cf_base;
            tile_info_out[info_base + 3u] = zrun_base;
        }
    }
}
