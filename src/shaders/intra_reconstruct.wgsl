// Intra prediction decoder shader — sequential raster scan within each tile.
//
// 1 workgroup per tile, sequential raster scan of 8×8 blocks within tile.
// Each block's prediction depends on already-reconstructed left/top neighbors.
//
// @workgroup_size(1) — parallelism comes from dispatching multiple tiles.

struct Params {
    plane_width: u32,   // padded plane width in pixels
    plane_height: u32,  // padded plane height in pixels
    tile_size: u32,     // tile size in pixels (e.g. 256)
    tiles_x: u32,       // number of tiles in x
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> residual: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> modes: array<u32>;

const BLOCK_SIZE: u32 = 8u;

fn out_pixel(x: u32, y: u32) -> f32 {
    return output[y * params.plane_width + x];
}

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
            let mode = modes[block_id];

            let ox = bx * BLOCK_SIZE;
            let oy = by * BLOCK_SIZE;

            let has_left = lbx > 0u;
            let has_top = lby > 0u;

            // Fetch left column and top row from already-reconstructed output
            var left_col: array<f32, 8>;
            var top_row: array<f32, 8>;

            for (var i = 0u; i < BLOCK_SIZE; i = i + 1u) {
                if (has_left) {
                    left_col[i] = out_pixel(ox - 1u, oy + i);
                } else {
                    left_col[i] = 128.0;
                }
                if (has_top) {
                    top_row[i] = out_pixel(ox + i, oy - 1u);
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
                top_right = out_pixel(ox + BLOCK_SIZE, oy - 1u);
            } else if (has_top) {
                top_right = top_row[BLOCK_SIZE - 1u];
            } else {
                top_right = 128.0;
            }

            // Reconstruct: output = residual + prediction
            for (var y = 0u; y < BLOCK_SIZE; y = y + 1u) {
                for (var x = 0u; x < BLOCK_SIZE; x = x + 1u) {
                    let idx = (oy + y) * params.plane_width + (ox + x);
                    let res = residual[idx];
                    var pred: f32;

                    switch mode {
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

                    output[idx] = res + pred;
                }
            }
        }
    }
}
