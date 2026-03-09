// Tile skip mode for inter-frame (P/B) residual coding.
//
// For each tile in the coefficient buffer, computes mean |coeff|.
// If mean < skip_threshold, zeros all tile coefficients.
// Zeroed tiles produce compact all-skip RiceTiles (saves bits with minimal quality loss).
//
// Dispatch: dispatch_workgroups(tiles_x, tiles_y, 1)
// One workgroup per tile. Workgroup size 256 threads.
// Each thread handles (tile_pixels + 255) / 256 pixels.
//
// Shared memory: 256 f32 (partial abs sums) = 1KB << 32KB limit.

struct Params {
    padded_w:       u32,
    padded_h:       u32,
    tile_size:      u32,
    skip_threshold: f32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> coeffs: array<f32>;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)           wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid:   u32,
) {
    let tiles_x = params.padded_w / params.tile_size;
    let tiles_y = params.padded_h / params.tile_size;
    let tile_x  = wg_id.x;
    let tile_y  = wg_id.y;

    let out_of_bounds = tile_x >= tiles_x || tile_y >= tiles_y;

    // Initialize shared memory — all threads participate regardless of bounds.
    shared_sum[lid] = 0.0;
    workgroupBarrier();

    let tile_pixels = params.tile_size * params.tile_size;
    let pixels_per_thread = (tile_pixels + 255u) / 256u;

    // Accumulate abs values for in-bounds tiles only.
    if !out_of_bounds {
        let tile_origin_x = tile_x * params.tile_size;
        let tile_origin_y = tile_y * params.tile_size;
        var local_sum: f32 = 0.0;
        for (var i = 0u; i < pixels_per_thread; i = i + 1u) {
            let pixel_idx = lid * pixels_per_thread + i;
            if pixel_idx < tile_pixels {
                let local_y = pixel_idx / params.tile_size;
                let local_x = pixel_idx % params.tile_size;
                let px = tile_origin_x + local_x;
                let py = tile_origin_y + local_y;
                local_sum = local_sum + abs(coeffs[py * params.padded_w + px]);
            }
        }
        shared_sum[lid] = local_sum;
    }
    workgroupBarrier();

    // Parallel reduction over 256 threads.
    // IMPORTANT: workgroupBarrier() must be called unconditionally by ALL threads
    // every iteration — Metal/M1 miscompiles barriers inside divergent branches.
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            shared_sum[lid] = shared_sum[lid] + shared_sum[lid + stride];
        }
        workgroupBarrier();
    }

    // After reduction, shared_sum[0] holds the total sum for this tile.
    // All threads check the mean and zero their assigned pixels if below threshold.
    if !out_of_bounds {
        let mean = shared_sum[0] / f32(tile_pixels);
        if mean < params.skip_threshold {
            let tile_origin_x = tile_x * params.tile_size;
            let tile_origin_y = tile_y * params.tile_size;
            for (var i = 0u; i < pixels_per_thread; i = i + 1u) {
                let pixel_idx = lid * pixels_per_thread + i;
                if pixel_idx < tile_pixels {
                    let local_y = pixel_idx / params.tile_size;
                    let local_x = pixel_idx % params.tile_size;
                    let px = tile_origin_x + local_x;
                    let py = tile_origin_y + local_y;
                    coeffs[py * params.padded_w + px] = 0.0;
                }
            }
        }
    }
}
