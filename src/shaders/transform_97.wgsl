// CDF 9/7 Wavelet Transform (four-step lifting scheme)
// Same tile structure as LeGall 5/3: one plane at a time, row then column pass.
//
// CDF 9/7 lifting coefficients (Daubechies factorization, public domain):
//   Forward:
//     Step 1 (predict):  high[n] += α * (low[n] + low[n+1])       α = -1.586134342
//     Step 2 (update):   low[n]  += β * (high[n-1] + high[n])      β = -0.052980118
//     Step 3 (predict):  high[n] += γ * (low[n] + low[n+1])        γ =  0.882911075
//     Step 4 (update):   low[n]  += δ * (high[n-1] + high[n])      δ =  0.443506852
//     Normalize:         low[n]  *= K,  high[n] *= 1/K             K =  1.149604398
//
//   Inverse: reverse order, negate coefficients, swap K and 1/K.

const ALPHA: f32 = -1.586134342;
const BETA: f32 = -0.052980118;
const GAMMA: f32 = 0.882911075;
const DELTA: f32 = 0.443506852;
const K: f32 = 1.149604398;
const INV_K: f32 = 0.869864452; // 1.0 / K

struct Params {
    width: u32,
    height: u32,
    tile_size: u32,
    direction: u32,
    pass_mode: u32,
    tiles_x: u32,
    region_size: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Shared memory — same layout as transform.wgsl
var<workgroup> shared_data: array<f32, 256>;
var<workgroup> shared_low: array<f32, 128>;
var<workgroup> shared_high: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let line_in_tile = wg_id.x;
    let tile_idx = wg_id.y;
    let thread_id = lid.x;

    let tile_x = tile_idx % params.tiles_x;
    let tile_y = tile_idx / params.tiles_x;

    let tile_origin_x = tile_x * params.tile_size;
    let tile_origin_y = tile_y * params.tile_size;

    let ts = params.region_size;
    let half = ts / 2u;

    if params.direction == 0u {
        // ---- FORWARD TRANSFORM ----
        if params.pass_mode == 0u {
            // Row pass
            let row = tile_origin_y + line_in_tile;
            if row >= params.height { return; }

            // Load row into shared memory
            let col0 = tile_origin_x + thread_id * 2u;
            let col1 = col0 + 1u;
            if col0 < params.width && col0 < tile_origin_x + ts {
                shared_data[thread_id * 2u] = input[row * params.width + col0];
            }
            if col1 < params.width && col1 < tile_origin_x + ts {
                shared_data[thread_id * 2u + 1u] = input[row * params.width + col1];
            }
            workgroupBarrier();

            // Split into even/odd
            if thread_id < half {
                shared_low[thread_id] = shared_data[thread_id * 2u];
                shared_high[thread_id] = shared_data[thread_id * 2u + 1u];
            }
            workgroupBarrier();

            // Step 1: high[n] += α * (low[n] + low[n+1])
            if thread_id < half {
                let left = shared_low[thread_id];
                var right: f32;
                if thread_id + 1u < half {
                    right = shared_low[thread_id + 1u];
                } else {
                    right = shared_low[thread_id];
                }
                shared_high[thread_id] = shared_high[thread_id] + ALPHA * (left + right);
            }
            workgroupBarrier();

            // Step 2: low[n] += β * (high[n-1] + high[n])
            if thread_id < half {
                var left_h: f32;
                if thread_id > 0u {
                    left_h = shared_high[thread_id - 1u];
                } else {
                    left_h = shared_high[0u];
                }
                let right_h = shared_high[thread_id];
                shared_low[thread_id] = shared_low[thread_id] + BETA * (left_h + right_h);
            }
            workgroupBarrier();

            // Step 3: high[n] += γ * (low[n] + low[n+1])
            if thread_id < half {
                let left = shared_low[thread_id];
                var right: f32;
                if thread_id + 1u < half {
                    right = shared_low[thread_id + 1u];
                } else {
                    right = shared_low[thread_id];
                }
                shared_high[thread_id] = shared_high[thread_id] + GAMMA * (left + right);
            }
            workgroupBarrier();

            // Step 4: low[n] += δ * (high[n-1] + high[n])
            if thread_id < half {
                var left_h: f32;
                if thread_id > 0u {
                    left_h = shared_high[thread_id - 1u];
                } else {
                    left_h = shared_high[0u];
                }
                let right_h = shared_high[thread_id];
                shared_low[thread_id] = shared_low[thread_id] + DELTA * (left_h + right_h);
            }
            workgroupBarrier();

            // Normalize: low *= K, high *= 1/K
            if thread_id < half {
                shared_low[thread_id] = shared_low[thread_id] * K;
                shared_high[thread_id] = shared_high[thread_id] * INV_K;
            }
            workgroupBarrier();

            // Write output: low then high
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
            // Column pass
            let col = tile_origin_x + line_in_tile;
            if col >= params.width { return; }

            // Load column
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

            // Split even/odd
            if thread_id < half {
                shared_low[thread_id] = shared_data[thread_id * 2u];
                shared_high[thread_id] = shared_data[thread_id * 2u + 1u];
            }
            workgroupBarrier();

            // Step 1: high[n] += α * (low[n] + low[n+1])
            if thread_id < half {
                let left = shared_low[thread_id];
                var right: f32;
                if thread_id + 1u < half {
                    right = shared_low[thread_id + 1u];
                } else {
                    right = shared_low[thread_id];
                }
                shared_high[thread_id] = shared_high[thread_id] + ALPHA * (left + right);
            }
            workgroupBarrier();

            // Step 2: low[n] += β * (high[n-1] + high[n])
            if thread_id < half {
                var left_h: f32;
                if thread_id > 0u {
                    left_h = shared_high[thread_id - 1u];
                } else {
                    left_h = shared_high[0u];
                }
                let right_h = shared_high[thread_id];
                shared_low[thread_id] = shared_low[thread_id] + BETA * (left_h + right_h);
            }
            workgroupBarrier();

            // Step 3: high[n] += γ * (low[n] + low[n+1])
            if thread_id < half {
                let left = shared_low[thread_id];
                var right: f32;
                if thread_id + 1u < half {
                    right = shared_low[thread_id + 1u];
                } else {
                    right = shared_low[thread_id];
                }
                shared_high[thread_id] = shared_high[thread_id] + GAMMA * (left + right);
            }
            workgroupBarrier();

            // Step 4: low[n] += δ * (high[n-1] + high[n])
            if thread_id < half {
                var left_h: f32;
                if thread_id > 0u {
                    left_h = shared_high[thread_id - 1u];
                } else {
                    left_h = shared_high[0u];
                }
                let right_h = shared_high[thread_id];
                shared_low[thread_id] = shared_low[thread_id] + DELTA * (left_h + right_h);
            }
            workgroupBarrier();

            // Normalize
            if thread_id < half {
                shared_low[thread_id] = shared_low[thread_id] * K;
                shared_high[thread_id] = shared_high[thread_id] * INV_K;
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
            // Inverse column pass
            let col = tile_origin_x + line_in_tile;
            if col >= params.width { return; }

            // Load: low rows then high rows
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

            // Denormalize: low *= 1/K, high *= K
            if thread_id < half {
                shared_low[thread_id] = shared_low[thread_id] * INV_K;
                shared_high[thread_id] = shared_high[thread_id] * K;
            }
            workgroupBarrier();

            // Inverse step 4: low[n] -= δ * (high[n-1] + high[n])
            if thread_id < half {
                var left_h: f32;
                if thread_id > 0u {
                    left_h = shared_high[thread_id - 1u];
                } else {
                    left_h = shared_high[0u];
                }
                let right_h = shared_high[thread_id];
                shared_low[thread_id] = shared_low[thread_id] - DELTA * (left_h + right_h);
            }
            workgroupBarrier();

            // Inverse step 3: high[n] -= γ * (low[n] + low[n+1])
            if thread_id < half {
                let left = shared_low[thread_id];
                var right: f32;
                if thread_id + 1u < half {
                    right = shared_low[thread_id + 1u];
                } else {
                    right = shared_low[thread_id];
                }
                shared_high[thread_id] = shared_high[thread_id] - GAMMA * (left + right);
            }
            workgroupBarrier();

            // Inverse step 2: low[n] -= β * (high[n-1] + high[n])
            if thread_id < half {
                var left_h: f32;
                if thread_id > 0u {
                    left_h = shared_high[thread_id - 1u];
                } else {
                    left_h = shared_high[0u];
                }
                let right_h = shared_high[thread_id];
                shared_low[thread_id] = shared_low[thread_id] - BETA * (left_h + right_h);
            }
            workgroupBarrier();

            // Inverse step 1: high[n] -= α * (low[n] + low[n+1])
            if thread_id < half {
                let left = shared_low[thread_id];
                var right: f32;
                if thread_id + 1u < half {
                    right = shared_low[thread_id + 1u];
                } else {
                    right = shared_low[thread_id];
                }
                shared_high[thread_id] = shared_high[thread_id] - ALPHA * (left + right);
            }
            workgroupBarrier();

            // Interleave back
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

            // Load: low cols then high cols
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

            // Denormalize: low *= 1/K, high *= K
            if thread_id < half {
                shared_low[thread_id] = shared_low[thread_id] * INV_K;
                shared_high[thread_id] = shared_high[thread_id] * K;
            }
            workgroupBarrier();

            // Inverse step 4: low[n] -= δ * (high[n-1] + high[n])
            if thread_id < half {
                var left_h: f32;
                if thread_id > 0u {
                    left_h = shared_high[thread_id - 1u];
                } else {
                    left_h = shared_high[0u];
                }
                let right_h = shared_high[thread_id];
                shared_low[thread_id] = shared_low[thread_id] - DELTA * (left_h + right_h);
            }
            workgroupBarrier();

            // Inverse step 3: high[n] -= γ * (low[n] + low[n+1])
            if thread_id < half {
                let left = shared_low[thread_id];
                var right: f32;
                if thread_id + 1u < half {
                    right = shared_low[thread_id + 1u];
                } else {
                    right = shared_low[thread_id];
                }
                shared_high[thread_id] = shared_high[thread_id] - GAMMA * (left + right);
            }
            workgroupBarrier();

            // Inverse step 2: low[n] -= β * (high[n-1] + high[n])
            if thread_id < half {
                var left_h: f32;
                if thread_id > 0u {
                    left_h = shared_high[thread_id - 1u];
                } else {
                    left_h = shared_high[0u];
                }
                let right_h = shared_high[thread_id];
                shared_low[thread_id] = shared_low[thread_id] - BETA * (left_h + right_h);
            }
            workgroupBarrier();

            // Inverse step 1: high[n] -= α * (low[n] + low[n+1])
            if thread_id < half {
                let left = shared_low[thread_id];
                var right: f32;
                if thread_id + 1u < half {
                    right = shared_low[thread_id + 1u];
                } else {
                    right = shared_low[thread_id];
                }
                shared_high[thread_id] = shared_high[thread_id] - ALPHA * (left + right);
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
