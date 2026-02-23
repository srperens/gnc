// Per-block luminance variance computation.
//
// Each workgroup processes one 32x32 block of the Y (luma) plane.
// Uses workgroup shared memory for a parallel two-pass reduction:
//   1. Each of 256 threads accumulates 4 pixels (1024 / 256 = 4)
//   2. Parallel reduction for sum and sum-of-squares across 256 threads
//   3. Compute variance = E[x^2] - E[x]^2
//
// Output: one f32 variance value per block, stored in row-major order
// in the variance map buffer.
//
// Workgroup size 256 is compatible with WebGPU default limits.

struct Params {
    // Plane dimensions (padded to tile boundaries)
    width: u32,
    height: u32,
    // Number of blocks in x direction
    blocks_x: u32,
    // Total number of blocks
    total_blocks: u32,
}

const BLOCK_SIZE: u32 = 32u;
const BLOCK_PIXELS: u32 = 1024u; // 32 * 32
const WG_SIZE: u32 = 256u;
const PIXELS_PER_THREAD: u32 = 4u; // 1024 / 256

@group(0) @binding(0) var<uniform> params: Params;
// Y (luma) plane, one f32 per pixel
@group(0) @binding(1) var<storage, read> y_plane: array<f32>;
// Output: one f32 variance per block
@group(0) @binding(2) var<storage, read_write> variance_map: array<f32>;

// Shared memory for parallel reduction (256 entries, one per thread).
var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_sum_sq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let block_idx = wg_id.x;
    if block_idx >= params.total_blocks {
        return;
    }

    // Map block index to 2D block coordinates
    let block_x = block_idx % params.blocks_x;
    let block_y = block_idx / params.blocks_x;

    // Each thread accumulates 4 pixels
    var local_sum = 0.0;
    var local_sum_sq = 0.0;

    for (var i = 0u; i < PIXELS_PER_THREAD; i = i + 1u) {
        let pixel_idx = lid * PIXELS_PER_THREAD + i;
        let local_y = pixel_idx / BLOCK_SIZE;
        let local_x = pixel_idx % BLOCK_SIZE;

        let px = block_x * BLOCK_SIZE + local_x;
        let py = block_y * BLOCK_SIZE + local_y;

        var val = 0.0;
        if px < params.width && py < params.height {
            val = y_plane[py * params.width + px];
        }

        local_sum = local_sum + val;
        local_sum_sq = local_sum_sq + val * val;
    }

    // Store partial sums in shared memory
    shared_sum[lid] = local_sum;
    shared_sum_sq[lid] = local_sum_sq;
    workgroupBarrier();

    // Parallel reduction over 256 elements
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if lid < stride {
            shared_sum[lid] = shared_sum[lid] + shared_sum[lid + stride];
            shared_sum_sq[lid] = shared_sum_sq[lid] + shared_sum_sq[lid + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 computes and writes the variance
    if lid == 0u {
        let n = f32(BLOCK_PIXELS);
        let mean = shared_sum[0] / n;
        let mean_sq = shared_sum_sq[0] / n;
        // Variance = E[x^2] - E[x]^2, clamped to avoid numerical noise
        let variance = max(mean_sq - mean * mean, 0.0);
        variance_map[block_idx] = variance;
    }
}
