// Per-block variance computation on the LL (lowpass) wavelet subband.
//
// After the wavelet transform, the LL subband at each decomposition level
// is a downsampled version of the image. Within each tile, the LL subband
// occupies the top-left corner: [0..tile_size/(2^L), 0..tile_size/(2^L)].
//
// This shader reads the LL subband coefficients from the wavelet-domain
// buffer, computing variance over ll_block_size x ll_block_size blocks
// within each tile's LL region. Blocks are output in global 2D row-major
// order: global_block_y * global_blocks_x + global_block_x, where the
// global coordinates combine tile position and local block position.
//
// Each workgroup processes one LL-block. The output is one f32 variance
// value per block.
//
// Workgroup size 256 is compatible with WebGPU default limits.

struct Params {
    // Padded plane dimensions (full wavelet-domain buffer)
    width: u32,
    height: u32,
    // Tile parameters
    tile_size: u32,
    num_levels: u32,
    // LL subband size per tile = tile_size / (1 << num_levels)
    ll_size: u32,
    // AQ block size in LL-subband coordinates
    ll_block_size: u32,
    // Number of LL-blocks per tile in x and y
    ll_blocks_per_tile_x: u32,
    ll_blocks_per_tile_y: u32,
    // Number of tiles in x direction
    tiles_x: u32,
    // Total number of LL-blocks across all tiles
    total_blocks: u32,
    // Global grid dimensions: ll_blocks_per_tile_x * tiles_x, ll_blocks_per_tile_y * tiles_y
    global_blocks_x: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Wavelet-domain buffer (all planes laid out as f32 per coefficient)
@group(0) @binding(1) var<storage, read> wavelet_buf: array<f32>;
// Output: one f32 variance per LL-block, in global 2D row-major order
@group(0) @binding(2) var<storage, read_write> variance_map: array<f32>;

// Shared memory for parallel reduction.
var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_sum_sq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let block_idx = wg_id.x;
    if block_idx >= params.total_blocks {
        shared_sum[lid] = 0.0;
        shared_sum_sq[lid] = 0.0;
        // Still need to participate in barriers
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
            workgroupBarrier();
        }
        return;
    }

    // Decompose block_idx (dispatch order) into tile and local LL-block coordinates.
    // Dispatch is tile-major: iterate through tiles, then blocks within each tile.
    let ll_blocks_per_tile = params.ll_blocks_per_tile_x * params.ll_blocks_per_tile_y;
    let tile_idx = block_idx / ll_blocks_per_tile;
    let block_in_tile = block_idx % ll_blocks_per_tile;

    // Tile 2D coordinates
    let tile_x = tile_idx % params.tiles_x;
    let tile_y = tile_idx / params.tiles_x;

    // LL-block 2D coordinates within the tile's LL region
    let lb_x = block_in_tile % params.ll_blocks_per_tile_x;
    let lb_y = block_in_tile / params.ll_blocks_per_tile_x;

    // Pixel range in the wavelet-domain buffer for this LL-block.
    // The LL subband within a tile starts at the tile's top-left corner.
    let tile_origin_x = tile_x * params.tile_size;
    let tile_origin_y = tile_y * params.tile_size;
    let block_origin_x = tile_origin_x + lb_x * params.ll_block_size;
    let block_origin_y = tile_origin_y + lb_y * params.ll_block_size;

    let block_pixels = params.ll_block_size * params.ll_block_size;
    // Each thread accumulates ceil(block_pixels / 256) pixels
    let pixels_per_thread = (block_pixels + 255u) / 256u;

    var local_sum = 0.0;
    var local_sum_sq = 0.0;

    for (var i = 0u; i < pixels_per_thread; i = i + 1u) {
        let pixel_idx = lid * pixels_per_thread + i;
        if pixel_idx < block_pixels {
            let local_y = pixel_idx / params.ll_block_size;
            let local_x = pixel_idx % params.ll_block_size;

            let px = block_origin_x + local_x;
            let py = block_origin_y + local_y;

            var val = 0.0;
            if px < params.width && py < params.height {
                val = wavelet_buf[py * params.width + px];
            }

            local_sum = local_sum + val;
            local_sum_sq = local_sum_sq + val * val;
        }
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

    // Thread 0 computes the variance and writes to global 2D output position
    if lid == 0u {
        let n = f32(block_pixels);
        let mean = shared_sum[0] / n;
        let mean_sq = shared_sum_sq[0] / n;
        // Variance = E[x^2] - E[x]^2, clamped to avoid numerical noise
        let variance = max(mean_sq - mean * mean, 0.0);

        // Output in global 2D row-major order
        let global_bx = tile_x * params.ll_blocks_per_tile_x + lb_x;
        let global_by = tile_y * params.ll_blocks_per_tile_y + lb_y;
        let out_idx = global_by * params.global_blocks_x + global_bx;
        variance_map[out_idx] = variance;
    }
}
