// LeGall 5/3 Wavelet Transform (lifting scheme)
// Operates on one plane at a time, tile by tile.
// Two passes: row transform, then column transform.
//
// Lifting steps (forward):
//   Predict: d[n] = odd[n] - floor((even[n] + even[n+1]) / 2)
//   Update:  s[n] = even[n] + floor((d[n-1] + d[n] + 2) / 4)
//
// For GPU: each workgroup processes one row (or column) of a tile.
// We use shared memory for the lifting operations.

struct Params {
    width: u32,         // plane width
    height: u32,        // plane height
    tile_size: u32,     // full tile dimension (for computing tile origins)
    direction: u32,     // 0 = forward, 1 = inverse
    pass_mode: u32,     // 0 = row pass, 1 = column pass
    tiles_x: u32,       // number of tiles horizontally
    region_size: u32,   // sub-region size for this level (tile_size >> level)
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Shared memory for one row/column of a tile (max tile_size = 256)
var<workgroup> shared_data: array<f32, 256>;
var<workgroup> shared_low: array<f32, 128>;
var<workgroup> shared_high: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // wg_id.x = which row/column within the tile
    // wg_id.y = which tile (flattened tile index)
    let line_in_tile = wg_id.x;
    let tile_idx = wg_id.y;
    let thread_id = lid.x;

    let tile_x = tile_idx % params.tiles_x;
    let tile_y = tile_idx / params.tiles_x;

    let tile_origin_x = tile_x * params.tile_size;
    let tile_origin_y = tile_y * params.tile_size;

    // Determine the length of this row/column (region_size for multi-level)
    let ts = params.region_size;
    let half = ts / 2u;

    if params.direction == 0u {
        // ---- FORWARD TRANSFORM ----
        if params.pass_mode == 0u {
            // Row pass: process one row of the tile
            let row = tile_origin_y + line_in_tile;
            if row >= params.height { return; }

            // Load row into shared memory (2 elements per thread)
            let col0 = tile_origin_x + thread_id * 2u;
            let col1 = col0 + 1u;
            if col0 < params.width && col0 < tile_origin_x + ts {
                shared_data[thread_id * 2u] = input[row * params.width + col0];
            }
            if col1 < params.width && col1 < tile_origin_x + ts {
                shared_data[thread_id * 2u + 1u] = input[row * params.width + col1];
            }
            workgroupBarrier();

            // Split into even and odd
            if thread_id < half {
                shared_low[thread_id] = shared_data[thread_id * 2u];
                shared_high[thread_id] = shared_data[thread_id * 2u + 1u];
            }
            workgroupBarrier();

            // Predict step: high[n] -= floor((low[n] + low[n+1]) / 2)
            if thread_id < half {
                let left = shared_low[thread_id];
                var right: f32;
                if thread_id + 1u < half {
                    right = shared_low[thread_id + 1u];
                } else {
                    right = shared_low[thread_id]; // mirror at boundary
                }
                shared_high[thread_id] = shared_high[thread_id] - floor((left + right) * 0.5);
            }
            workgroupBarrier();

            // Update step: low[n] += floor((high[n-1] + high[n] + 2) / 4)
            if thread_id < half {
                var left_h: f32;
                if thread_id > 0u {
                    left_h = shared_high[thread_id - 1u];
                } else {
                    left_h = shared_high[0u]; // mirror at boundary
                }
                let right_h = shared_high[thread_id];
                shared_low[thread_id] = shared_low[thread_id] + floor((left_h + right_h + 2.0) * 0.25);
            }
            workgroupBarrier();

            // Write output: low frequencies first, then high frequencies
            // Layout: [L0 L1 ... L_{half-1} H0 H1 ... H_{half-1}]
            if thread_id < half {
                let out_col_l = tile_origin_x + thread_id;
                let out_col_h = tile_origin_x + half + thread_id;
                if out_col_l < params.width {
                    output[row * params.width + out_col_l] = shared_low[thread_id];
                }
                if out_col_h < params.width {
                    output[row * params.width + out_col_h] = shared_high[thread_id];
                }
            }
        } else {
            // Column pass: process one column of the tile
            let col = tile_origin_x + line_in_tile;
            if col >= params.width { return; }

            // Load column into shared memory
            if thread_id < half {
                let row0 = tile_origin_y + thread_id * 2u;
                let row1 = row0 + 1u;
                if row0 < params.height {
                    shared_data[thread_id * 2u] = input[row0 * params.width + col];
                }
                if row1 < params.height {
                    shared_data[thread_id * 2u + 1u] = input[row1 * params.width + col];
                }
            }
            workgroupBarrier();

            // Split into even and odd
            if thread_id < half {
                shared_low[thread_id] = shared_data[thread_id * 2u];
                shared_high[thread_id] = shared_data[thread_id * 2u + 1u];
            }
            workgroupBarrier();

            // Predict: high[n] -= floor((low[n] + low[n+1]) / 2)
            if thread_id < half {
                let left = shared_low[thread_id];
                var right: f32;
                if thread_id + 1u < half {
                    right = shared_low[thread_id + 1u];
                } else {
                    right = shared_low[thread_id];
                }
                shared_high[thread_id] = shared_high[thread_id] - floor((left + right) * 0.5);
            }
            workgroupBarrier();

            // Update: low[n] += floor((high[n-1] + high[n] + 2) / 4)
            if thread_id < half {
                var left_h: f32;
                if thread_id > 0u {
                    left_h = shared_high[thread_id - 1u];
                } else {
                    left_h = shared_high[0u];
                }
                let right_h = shared_high[thread_id];
                shared_low[thread_id] = shared_low[thread_id] + floor((left_h + right_h + 2.0) * 0.25);
            }
            workgroupBarrier();

            // Write: low rows first, then high rows
            if thread_id < half {
                let out_row_l = tile_origin_y + thread_id;
                let out_row_h = tile_origin_y + half + thread_id;
                if out_row_l < params.height {
                    output[out_row_l * params.width + col] = shared_low[thread_id];
                }
                if out_row_h < params.height {
                    output[out_row_h * params.width + col] = shared_high[thread_id];
                }
            }
        }
    } else {
        // ---- INVERSE TRANSFORM ----
        if params.pass_mode == 1u {
            // Inverse column pass (run before inverse row pass)
            let col = tile_origin_x + line_in_tile;
            if col >= params.width { return; }

            // Load: low rows, then high rows
            if thread_id < half {
                let row_l = tile_origin_y + thread_id;
                let row_h = tile_origin_y + half + thread_id;
                if row_l < params.height {
                    shared_low[thread_id] = input[row_l * params.width + col];
                }
                if row_h < params.height {
                    shared_high[thread_id] = input[row_h * params.width + col];
                }
            }
            workgroupBarrier();

            // Inverse update: low[n] -= floor((high[n-1] + high[n] + 2) / 4)
            if thread_id < half {
                var left_h: f32;
                if thread_id > 0u {
                    left_h = shared_high[thread_id - 1u];
                } else {
                    left_h = shared_high[0u];
                }
                let right_h = shared_high[thread_id];
                shared_low[thread_id] = shared_low[thread_id] - floor((left_h + right_h + 2.0) * 0.25);
            }
            workgroupBarrier();

            // Inverse predict: high[n] += floor((low[n] + low[n+1]) / 2)
            if thread_id < half {
                let left = shared_low[thread_id];
                var right: f32;
                if thread_id + 1u < half {
                    right = shared_low[thread_id + 1u];
                } else {
                    right = shared_low[thread_id];
                }
                shared_high[thread_id] = shared_high[thread_id] + floor((left + right) * 0.5);
            }
            workgroupBarrier();

            // Interleave back: even = low, odd = high
            if thread_id < half {
                let row0 = tile_origin_y + thread_id * 2u;
                let row1 = row0 + 1u;
                if row0 < params.height {
                    output[row0 * params.width + col] = shared_low[thread_id];
                }
                if row1 < params.height {
                    output[row1 * params.width + col] = shared_high[thread_id];
                }
            }
        } else {
            // Inverse row pass
            let row = tile_origin_y + line_in_tile;
            if row >= params.height { return; }

            // Load: low cols, then high cols
            if thread_id < half {
                let col_l = tile_origin_x + thread_id;
                let col_h = tile_origin_x + half + thread_id;
                if col_l < params.width {
                    shared_low[thread_id] = input[row * params.width + col_l];
                }
                if col_h < params.width {
                    shared_high[thread_id] = input[row * params.width + col_h];
                }
            }
            workgroupBarrier();

            // Inverse update: low[n] -= floor((high[n-1] + high[n] + 2) / 4)
            if thread_id < half {
                var left_h: f32;
                if thread_id > 0u {
                    left_h = shared_high[thread_id - 1u];
                } else {
                    left_h = shared_high[0u];
                }
                let right_h = shared_high[thread_id];
                shared_low[thread_id] = shared_low[thread_id] - floor((left_h + right_h + 2.0) * 0.25);
            }
            workgroupBarrier();

            // Inverse predict: high[n] += floor((low[n] + low[n+1]) / 2)
            if thread_id < half {
                let left = shared_low[thread_id];
                var right: f32;
                if thread_id + 1u < half {
                    right = shared_low[thread_id + 1u];
                } else {
                    right = shared_low[thread_id];
                }
                shared_high[thread_id] = shared_high[thread_id] + floor((left + right) * 0.5);
            }
            workgroupBarrier();

            // Interleave back
            if thread_id < half {
                let col0 = tile_origin_x + thread_id * 2u;
                let col1 = col0 + 1u;
                if col0 < params.width {
                    output[row * params.width + col0] = shared_low[thread_id];
                }
                if col1 < params.width {
                    output[row * params.width + col1] = shared_high[thread_id];
                }
            }
        }
    }
}
