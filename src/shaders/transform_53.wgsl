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
    region_size: u32,   // physical region size: tile_size + 2*overlap for level-0 forward;
                        // tile_size >> level for higher levels; overlap=0 for inverse.
    overlap: u32,       // per-side overlap pixels (encoder forward only; 0 for inverse)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Shared memory — sized for max physical_tile_size = tile_size + 2*MAX_OVERLAP.
// MAX_OVERLAP = tile_size/2 - 1 = 127; physical_tile_size_max = 510; half_max = 255.
// Workgroup size 256 covers half_max=255 with one thread to spare.
// For overlap=0 (standard path), only threads 0..half-1 do work; the rest are idle.
var<workgroup> shared_data: array<f32, 512>;
var<workgroup> shared_low: array<f32, 256>;
var<workgroup> shared_high: array<f32, 256>;

@compute @workgroup_size(256)
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

    // Physical region size (= tile_size + 2*overlap for level-0 forward; tile_size>>level otherwise).
    let ts = params.region_size;
    let half = ts / 2u;

    if params.direction == 0u {
        // ---- FORWARD TRANSFORM ----
        if params.pass_mode == 0u {
            // Row pass: process one row of the tile
            let row = tile_origin_y + line_in_tile;
            if row >= params.height { return; }

            // Load physical_tile_size = ts elements into shared memory.
            // With overlap > 0: read from [tile_origin_x - overlap, tile_origin_x + tile_size + overlap).
            // Clamp to [0, width-1] to handle image edges (boundary replication).
            let overlap = params.overlap;
            let col_start = i32(tile_origin_x) - i32(overlap);
            if thread_id * 2u < ts {
                let c0 = u32(clamp(col_start + i32(thread_id * 2u), 0, i32(params.width) - 1));
                shared_data[thread_id * 2u] = input[row * params.width + c0];
            }
            if thread_id * 2u + 1u < ts {
                let c1 = u32(clamp(col_start + i32(thread_id * 2u + 1u), 0, i32(params.width) - 1));
                shared_data[thread_id * 2u + 1u] = input[row * params.width + c1];
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

            // Write output: the central (tile_size) coefficients, discarding the halo.
            //
            // skip = overlap/2: halo elements at start of each sub-band (low and high).
            // out_cnt = half - overlap: number of output elements per sub-band.
            //   - level 0 fwd: half = tile_size/2 + overlap, out_cnt = tile_size/2. ✓
            //   - level 1+ fwd (overlap=0): half = ts/2, out_cnt = half. ✓ (same as original)
            //
            // Output layout in the image buffer: [low | high] within each tile region:
            //   low  → tile_origin_x + 0 .. tile_origin_x + out_cnt - 1
            //   high → tile_origin_x + out_cnt .. tile_origin_x + 2*out_cnt - 1
            let skip = overlap / 2u;
            let out_cnt = half - overlap;
            if thread_id < out_cnt {
                let out_col_l = tile_origin_x + thread_id;
                let out_col_h = tile_origin_x + out_cnt + thread_id;
                if out_col_l < params.width {
                    output[row * params.width + out_col_l] = shared_low[skip + thread_id];
                }
                if out_col_h < params.width {
                    output[row * params.width + out_col_h] = shared_high[skip + thread_id];
                }
            }
        } else {
            // Column pass: process one column of the tile
            let col = tile_origin_x + line_in_tile;
            if col >= params.width { return; }

            // Load column into shared memory with vertical overlap.
            let overlap = params.overlap;
            let row_start = i32(tile_origin_y) - i32(overlap);
            if thread_id < half {
                let r0 = u32(clamp(row_start + i32(thread_id * 2u),     0, i32(params.height) - 1));
                let r1 = u32(clamp(row_start + i32(thread_id * 2u + 1u), 0, i32(params.height) - 1));
                shared_data[thread_id * 2u]     = input[r0 * params.width + col];
                shared_data[thread_id * 2u + 1u] = input[r1 * params.width + col];
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

            // Write: central tile_size rows (skip halo on each end of each sub-band)
            let skip = overlap / 2u;
            let out_cnt = half - overlap;
            if thread_id < out_cnt {
                let out_row_l = tile_origin_y + thread_id;
                let out_row_h = tile_origin_y + out_cnt + thread_id;
                if out_row_l < params.height {
                    output[out_row_l * params.width + col] = shared_low[skip + thread_id];
                }
                if out_row_h < params.height {
                    output[out_row_h * params.width + col] = shared_high[skip + thread_id];
                }
            }
        }
    } else {
        // ---- INVERSE TRANSFORM (overlap is always 0 on decode) ----
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
