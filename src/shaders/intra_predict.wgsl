// Intra prediction encoder shader — sequential raster scan within each tile.
//
// Computes prediction from already-reconstructed neighbors (closed-loop).
// This ensures encoder and decoder agree on prediction values.
// 4 modes: DC (0), Horizontal (1), Vertical (2), Diagonal-down-left (3).
// Boundary: tile-local. Missing left/top neighbors → predict 128.0.
//
// @workgroup_size(1), 1 workgroup per tile.

struct Params {
    plane_width: u32,   // padded plane width in pixels
    plane_height: u32,  // padded plane height in pixels
    tile_size: u32,     // tile size in pixels (e.g. 256)
    tiles_x: u32,       // number of tiles in x
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> modes: array<u32>;

const BLOCK_SIZE: u32 = 8u;

fn in_pixel(x: u32, y: u32) -> f32 {
    return input[y * params.plane_width + x];
}

// Prediction reference: uses original input pixels (open-loop).
// Since we're before the wavelet transform, encoder's local recon = input (no loss).
// Decoder uses its own progressive reconstruction, causing prediction drift proportional
// to wavelet quantization error. INTRA_TILE_SIZE in intra.rs limits drift accumulation.
// See intra.rs doc comment for architectural discussion.

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let tile_id = wid.x;
    let tiles_x = params.tiles_x;
    let tiles_y = params.plane_height / params.tile_size;
    let total_tiles = tiles_x * tiles_y;
    if (tile_id >= total_tiles) {
        return;
    }

    let tile_x = tile_id % tiles_x;
    let tile_y = tile_id / tiles_x;
    let tile_ox = tile_x * params.tile_size;
    let tile_oy = tile_y * params.tile_size;

    let tile_blocks = params.tile_size / BLOCK_SIZE;
    let blocks_x = params.plane_width / BLOCK_SIZE;

    // Raster scan blocks within this tile
    for (var lby = 0u; lby < tile_blocks; lby = lby + 1u) {
        for (var lbx = 0u; lbx < tile_blocks; lbx = lbx + 1u) {
            let bx = tile_ox / BLOCK_SIZE + lbx;
            let by = tile_oy / BLOCK_SIZE + lby;
            let block_id = by * blocks_x + bx;

            let ox = bx * BLOCK_SIZE;
            let oy = by * BLOCK_SIZE;

            let has_left = lbx > 0u;
            let has_top = lby > 0u;

            // Fetch left column and top row from original input
            // (The encoder's local recon = input since no loss yet)
            var left_col: array<f32, 8>;
            var top_row: array<f32, 8>;

            for (var i = 0u; i < BLOCK_SIZE; i = i + 1u) {
                if (has_left) {
                    left_col[i] = in_pixel(ox - 1u, oy + i);
                } else {
                    left_col[i] = 128.0;
                }
                if (has_top) {
                    top_row[i] = in_pixel(ox + i, oy - 1u);
                } else {
                    top_row[i] = 128.0;
                }
            }

            // DC value
            var dc_sum = 0.0;
            if (has_left && has_top) {
                for (var i = 0u; i < BLOCK_SIZE; i = i + 1u) {
                    dc_sum += left_col[i] + top_row[i];
                }
                dc_sum = dc_sum / f32(BLOCK_SIZE * 2u);
            } else if (has_left) {
                for (var i = 0u; i < BLOCK_SIZE; i = i + 1u) {
                    dc_sum += left_col[i];
                }
                dc_sum = dc_sum / f32(BLOCK_SIZE);
            } else if (has_top) {
                for (var i = 0u; i < BLOCK_SIZE; i = i + 1u) {
                    dc_sum += top_row[i];
                }
                dc_sum = dc_sum / f32(BLOCK_SIZE);
            } else {
                dc_sum = 128.0;
            }

            // Top-right pixel for diagonal
            var top_right: f32;
            if (has_top && ox + BLOCK_SIZE < params.plane_width) {
                top_right = in_pixel(ox + BLOCK_SIZE, oy - 1u);
            } else if (has_top) {
                top_right = top_row[BLOCK_SIZE - 1u];
            } else {
                top_right = 128.0;
            }

            // Compute SAD for each mode
            var sad: array<f32, 4>;
            sad[0] = 0.0;
            sad[1] = 0.0;
            sad[2] = 0.0;
            sad[3] = 0.0;

            for (var y = 0u; y < BLOCK_SIZE; y = y + 1u) {
                for (var x = 0u; x < BLOCK_SIZE; x = x + 1u) {
                    let val = in_pixel(ox + x, oy + y);

                    let pred_dc = dc_sum;
                    sad[0] += abs(val - pred_dc);

                    let pred_h = left_col[y];
                    sad[1] += abs(val - pred_h);

                    let pred_v = top_row[x];
                    sad[2] += abs(val - pred_v);

                    let diag_idx = x + y + 1u;
                    var pred_d: f32;
                    if (diag_idx < BLOCK_SIZE) {
                        pred_d = top_row[diag_idx];
                    } else {
                        pred_d = top_right;
                    }
                    sad[3] += abs(val - pred_d);
                }
            }

            // Pick mode with minimum SAD
            var best_mode = 0u;
            var best_sad = sad[0];
            for (var m = 1u; m < 4u; m = m + 1u) {
                if (sad[m] < best_sad) {
                    best_sad = sad[m];
                    best_mode = m;
                }
            }

            modes[block_id] = best_mode;

            // Write residual = input - prediction
            for (var y = 0u; y < BLOCK_SIZE; y = y + 1u) {
                for (var x = 0u; x < BLOCK_SIZE; x = x + 1u) {
                    let val = in_pixel(ox + x, oy + y);
                    var pred: f32;

                    switch best_mode {
                        case 0u: {
                            pred = dc_sum;
                        }
                        case 1u: {
                            pred = left_col[y];
                        }
                        case 2u: {
                            pred = top_row[x];
                        }
                        case 3u: {
                            let diag_idx = x + y + 1u;
                            if (diag_idx < BLOCK_SIZE) {
                                pred = top_row[diag_idx];
                            } else {
                                pred = top_right;
                            }
                        }
                        default: {
                            pred = 128.0;
                        }
                    }

                    let idx = (oy + y) * params.plane_width + (ox + x);
                    residual[idx] = val - pred;
                }
            }
        }
    }
}
