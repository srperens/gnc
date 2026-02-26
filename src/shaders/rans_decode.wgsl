// GPU rANS decoder — 32 interleaved streams per tile.
//
// Each workgroup decodes one tile. 32 threads = 32 independent rANS streams.
// Cumfreq table is loaded cooperatively into workgroup shared memory.
// Symbol lookup uses binary search on cumfreq (~7 iterations for typical alphabets).
//
// Supports two modes (selected by params.per_subband):
//   0 = Single frequency table per tile (with optional ZRL)
//   1 = Per-subband frequency tables (no ZRL; one table per wavelet level group)

const RANS_BYTE_L: u32 = 8388608u;  // 1 << 23
const RANS_PRECISION: u32 = 12u;
const RANS_MASK: u32 = 4095u;       // (1 << 12) - 1
const STREAMS_PER_TILE: u32 = 32u;
const MAX_ALPHABET: u32 = 4096u;
const MAX_GROUPS: u32 = 8u;

// Per-tile info stride in u32s (must match host TILE_INFO_STRIDE)
const TILE_INFO_STRIDE: u32 = 100u;

struct Params {
    num_tiles: u32,
    coefficients_per_tile: u32,  // tile_size * tile_size (65536)
    plane_width: u32,            // padded plane width in pixels
    tile_size: u32,              // 256
    tiles_x: u32,                // number of tiles per row
    per_subband: u32,            // 0 = single table, 1 = per-subband tables
    num_levels: u32,             // wavelet decomposition levels (only used when per_subband=1)
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tile_info: array<u32>;
@group(0) @binding(2) var<storage, read> cumfreq_data: array<u32>;
@group(0) @binding(3) var<storage, read> stream_data: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Shared cumfreq table for the current tile (loaded cooperatively)
// Per-subband: up to MAX_GROUPS * (MAX_GROUP_ALPHABET + 1) entries
// Single-table: up to MAX_ALPHABET + 1 = 4097 entries
var<workgroup> shared_cumfreq: array<u32, 4096>;

// Read one byte from the packed u32 stream data array.
// Bytes are packed little-endian: byte 0 is bits [0:7] of u32[0].
fn read_byte(byte_offset: u32) -> u32 {
    let word_idx = byte_offset >> 2u;
    let byte_pos = byte_offset & 3u;
    return (stream_data[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

// Binary search: find sym where cumfreq[sym] <= slot < cumfreq[sym+1].
// Searches the entire shared_cumfreq (used for single-table mode).
fn binary_search(slot: u32, alphabet_size: u32) -> u32 {
    var lo = 0u;
    var hi = alphabet_size;
    // Max 12 iterations covers alphabet up to 4096
    for (var iter = 0u; iter < 12u; iter++) {
        if (lo >= hi) { break; }
        let mid = (lo + hi) >> 1u;
        if (shared_cumfreq[mid + 1u] <= slot) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Binary search within a sub-range of shared_cumfreq starting at `start`.
// Used for per-subband mode where each group's cumfreqs are at a different offset.
fn binary_search_at(slot: u32, start: u32, alphabet_size: u32) -> u32 {
    var lo = 0u;
    var hi = alphabet_size;
    for (var iter = 0u; iter < 12u; iter++) {
        if (lo >= hi) { break; }
        let mid = (lo + hi) >> 1u;
        if (shared_cumfreq[start + mid + 1u] <= slot) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Directional subband grouping: separates HH from LH+HL at each level.
// Group 0 = LL, Group 1 = deepest detail (merged), then pairs of (LH+HL, HH)
// for remaining levels from deep to shallow. Total groups = num_levels * 2.
fn compute_subband_group(lx: u32, ly: u32) -> u32 {
    var region = params.tile_size;
    for (var level = 0u; level < params.num_levels; level++) {
        let half = region / 2u;
        if (lx >= half || ly >= half) {
            let lfd = params.num_levels - 1u - level;
            if (lfd == 0u) {
                return 1u;  // Deepest detail level: merged LH+HL+HH
            }
            let is_hh = (lx >= half) && (ly >= half);
            let base = 2u + (lfd - 1u) * 2u;
            return select(base, base + 1u, is_hh);
        }
        region = half;
    }
    return 0u;  // LL
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

    let base = tile_id * TILE_INFO_STRIDE;

    // Compute tile position in the plane
    let tile_x = tile_id % params.tiles_x;
    let tile_y = tile_id / params.tiles_x;
    let tile_origin_x = tile_x * params.tile_size;
    let tile_origin_y = tile_y * params.tile_size;

    let target_outputs = params.coefficients_per_tile / STREAMS_PER_TILE;

    if (params.per_subband != 0u) {
        // --- Per-subband decode path (with per-group ZRL) ---
        // Tile-info layout (4 u32s per group):
        //   [0]: num_groups
        //   [1+g*4+0]: group g min_val
        //   [1+g*4+1]: group g alphabet_size
        //   [1+g*4+2]: group g cumfreq_offset (into cumfreq_data)
        //   [1+g*4+3]: group g zrun_base (0 = no ZRL)
        //   [33]: stream_data_byte_base
        //   [34..66]: 32 initial states
        //   [66..98]: 32 stream byte offsets

        let num_groups = tile_info[base];

        // Load per-group metadata into local arrays
        var group_min: array<i32, 8>;
        var group_asize: array<u32, 8>;
        var group_cf_start: array<u32, 8>;  // start index in shared_cumfreq
        var group_zrun: array<i32, 8>;      // per-group zrun_base
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

        // Per-stream decode setup
        let stream_byte_base = tile_info[base + 33u];
        let initial_state = tile_info[base + 34u + thread_id];
        let stream_offset = tile_info[base + 66u + thread_id];

        var state = initial_state;
        var byte_ptr = stream_byte_base + stream_offset;

        if (has_any_zrl) {
            // ZRL-aware decode: output_i may advance by more than 1 per symbol
            var output_i = 0u;
            while (output_i < target_outputs) {
                let coeff_idx = thread_id + output_i * STREAMS_PER_TILE;
                let tile_row = coeff_idx / params.tile_size;
                let tile_col = coeff_idx % params.tile_size;
                let g = compute_subband_group(tile_col, tile_row);

                let cf_start = group_cf_start[g];
                let asize = group_asize[g];
                let gmin = group_min[g];
                let zrun = group_zrun[g];

                let slot = state & RANS_MASK;
                let sym = binary_search_at(slot, cf_start, asize);

                let start = shared_cumfreq[cf_start + sym];
                let freq = shared_cumfreq[cf_start + sym + 1u] - start;
                state = freq * (state >> RANS_PRECISION) + (state & RANS_MASK) - start;

                for (var r = 0u; r < 3u; r++) {
                    if (state >= RANS_BYTE_L) { break; }
                    let byte_val = read_byte(byte_ptr);
                    state = (state << 8u) | byte_val;
                    byte_ptr++;
                }

                let value = i32(sym) + gmin;

                if (zrun != 0 && value >= zrun) {
                    // ZRL run symbol: expand to consecutive zeros
                    var run_len = u32(value - zrun) + 1u;
                    if (output_i + run_len > target_outputs) {
                        run_len = target_outputs - output_i;
                    }
                    for (var j = 0u; j < run_len; j++) {
                        let ri = thread_id + (output_i + j) * STREAMS_PER_TILE;
                        let rrow = ri / params.tile_size;
                        let rcol = ri % params.tile_size;
                        let pidx = (tile_origin_y + rrow) * params.plane_width
                                 + (tile_origin_x + rcol);
                        output[pidx] = 0.0;
                    }
                    output_i += run_len;
                } else {
                    let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                                  + (tile_origin_x + tile_col);
                    output[plane_idx] = f32(value);
                    output_i += 1u;
                }
            }
        } else {
            // No ZRL in any group: simple for-loop decode
            for (var output_i = 0u; output_i < target_outputs; output_i++) {
                let coeff_idx = thread_id + output_i * STREAMS_PER_TILE;
                let tile_row = coeff_idx / params.tile_size;
                let tile_col = coeff_idx % params.tile_size;
                let g = compute_subband_group(tile_col, tile_row);

                let cf_start = group_cf_start[g];
                let asize = group_asize[g];
                let gmin = group_min[g];

                let slot = state & RANS_MASK;
                let sym = binary_search_at(slot, cf_start, asize);

                let start = shared_cumfreq[cf_start + sym];
                let freq = shared_cumfreq[cf_start + sym + 1u] - start;
                state = freq * (state >> RANS_PRECISION) + (state & RANS_MASK) - start;

                for (var r = 0u; r < 3u; r++) {
                    if (state >= RANS_BYTE_L) { break; }
                    let byte_val = read_byte(byte_ptr);
                    state = (state << 8u) | byte_val;
                    byte_ptr++;
                }

                let value = i32(sym) + gmin;
                let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                              + (tile_origin_x + tile_col);
                output[plane_idx] = f32(value);
            }
        }
    } else {
        // --- Single-table decode path (original, with optional ZRL) ---
        let alphabet_size = tile_info[base + 1u];
        let cumfreq_offset = tile_info[base + 2u];
        let alphabet_size_plus_one = alphabet_size + 1u;

        // Cooperatively load cumfreq into shared memory
        for (var i = thread_id; i < alphabet_size_plus_one; i += STREAMS_PER_TILE) {
            shared_cumfreq[i] = cumfreq_data[cumfreq_offset + i];
        }

        workgroupBarrier();

        let min_val = bitcast<i32>(tile_info[base]);
        let zrun_base = bitcast<i32>(tile_info[base + 3u]);

        // Per-stream: initial state, byte offset
        let stream_byte_base = tile_info[base + 4u];
        let initial_state = tile_info[base + 5u + thread_id];
        let stream_offset = tile_info[base + 37u + thread_id];

        var state = initial_state;
        var byte_ptr = stream_byte_base + stream_offset;

        // Decode symbols with zero-run expansion.
        var output_i = 0u;
        while (output_i < target_outputs) {
            let slot = state & RANS_MASK;
            let sym = binary_search(slot, alphabet_size);

            let start = shared_cumfreq[sym];
            let freq = shared_cumfreq[sym + 1u] - start;
            state = freq * (state >> RANS_PRECISION) + (state & RANS_MASK) - start;

            for (var r = 0u; r < 3u; r++) {
                if (state >= RANS_BYTE_L) { break; }
                let byte_val = read_byte(byte_ptr);
                state = (state << 8u) | byte_val;
                byte_ptr++;
            }

            let value = i32(sym) + min_val;

            if (zrun_base != 0 && value >= zrun_base) {
                var run_len = u32(value - zrun_base) + 1u;
                if (output_i + run_len > target_outputs) {
                    run_len = target_outputs - output_i;
                }
                for (var j = 0u; j < run_len; j++) {
                    let coeff_idx = thread_id + (output_i + j) * STREAMS_PER_TILE;
                    let tile_row = coeff_idx / params.tile_size;
                    let tile_col = coeff_idx % params.tile_size;
                    let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                                  + (tile_origin_x + tile_col);
                    output[plane_idx] = 0.0;
                }
                output_i += run_len;
            } else {
                let coeff_idx = thread_id + output_i * STREAMS_PER_TILE;
                let tile_row = coeff_idx / params.tile_size;
                let tile_col = coeff_idx % params.tile_size;
                let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                              + (tile_origin_x + tile_col);
                output[plane_idx] = f32(value);
                output_i += 1u;
            }
        }
    }
}
