// MV median smoothing — 3×3 median filter on the 8×8-resolution split MV buffer.
//
// Reduces sharp inter-block MV discontinuities, which can cause high residual energy
// at block boundaries.  Each component (dx, dy) is filtered independently.
//
// Gated by GNC_MV_SMOOTH=1 env var — off by default.
//
// Buffer layout: flat row-major array<i32>, two elements per block.
//   mvs_in[block_idx * 2]     = dx_qpel
//   mvs_in[block_idx * 2 + 1] = dy_qpel
//   block_idx = block_y * blocks_total_x + block_x
//
// Dispatch: dispatch_workgroups(tiles_x, tiles_y, 1)
// Workgroup: 256 threads, one workgroup per tile.
//   A 256×256 tile at 8px resolution = 32×32 = 1024 blocks per tile.
//   Each thread handles 4 blocks.

struct Params {
    padded_w:   u32,  // image width in pixels
    padded_h:   u32,  // image height in pixels
    tile_size:  u32,  // tile size in pixels (256)
    block_size: u32,  // block size in pixels (8)
}

@group(0) @binding(0) var<uniform>             params:  Params;
@group(0) @binding(1) var<storage, read>       mvs_in:  array<i32>;  // original split MVs
@group(0) @binding(2) var<storage, read_write> mvs_out: array<i32>;  // smoothed output

// Sort 9 i32 values and return the median (5th element, 0-indexed).
// Uses a simple bubble sort — 9 elements is small enough that the constant-time
// sorting network would only be marginally faster, and this keeps the code readable.
fn median9(v: array<i32, 9>) -> i32 {
    var a = v;
    for (var i = 0u; i < 9u; i++) {
        for (var j = 0u; j < 8u - i; j++) {
            if a[j] > a[j + 1u] {
                let t   = a[j];
                a[j]       = a[j + 1u];
                a[j + 1u]  = t;
            }
        }
    }
    return a[4]; // median
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)           wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid:   u32,
) {
    let tile_x = wg_id.x;
    let tile_y = wg_id.y;

    let tiles_x          = params.padded_w / params.tile_size;
    let tiles_y          = params.padded_h / params.tile_size;
    let blocks_total_x   = params.padded_w / params.block_size;
    let blocks_per_tile_row = params.tile_size / params.block_size;   // 32
    let blocks_per_tile     = blocks_per_tile_row * blocks_per_tile_row; // 1024
    let blocks_per_thread   = (blocks_per_tile + 255u) / 256u;          // 4

    // Guard: skip out-of-bounds tiles (shouldn't happen with padded dims, but be safe).
    if tile_x >= tiles_x || tile_y >= tiles_y {
        return;
    }

    // Each thread handles `blocks_per_thread` consecutive blocks within the tile.
    for (var b = 0u; b < blocks_per_thread; b++) {
        let block_in_tile = lid * blocks_per_thread + b;
        if block_in_tile >= blocks_per_tile {
            break;
        }

        // Global block coordinates of this block.
        let bx_in_tile = block_in_tile % blocks_per_tile_row;
        let by_in_tile = block_in_tile / blocks_per_tile_row;
        let block_gx   = tile_x * blocks_per_tile_row + bx_in_tile;
        let block_gy   = tile_y * blocks_per_tile_row + by_in_tile;
        let block_idx  = block_gy * blocks_total_x + block_gx;

        // Collect 3×3 neighbourhood MVs, clamping at frame boundaries.
        var dx_nbr: array<i32, 9>;
        var dy_nbr: array<i32, 9>;
        var n = 0u;
        for (var dy = -1i; dy <= 1i; dy++) {
            for (var dx = -1i; dx <= 1i; dx++) {
                // Clamp neighbour coordinates to [0, blocks_total_x/y - 1].
                let nx = clamp(i32(block_gx) + dx, 0i, i32(blocks_total_x) - 1i);
                let ny = clamp(i32(block_gy) + dy, 0i, i32(params.padded_h / params.block_size) - 1i);
                let nidx = u32(ny) * blocks_total_x + u32(nx);
                dx_nbr[n] = mvs_in[nidx * 2u     ];
                dy_nbr[n] = mvs_in[nidx * 2u + 1u];
                n++;
            }
        }

        mvs_out[block_idx * 2u     ] = median9(dx_nbr);
        mvs_out[block_idx * 2u + 1u] = median9(dy_nbr);
    }
}
