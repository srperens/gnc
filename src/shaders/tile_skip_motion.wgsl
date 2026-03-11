// Tile skip mode — zero motion vectors for spatially-stable P-frame tiles.
//
// For each tile, computes the zero-MV prediction error:
//   mean_sad = mean |current_pixel - reference_pixel_at_same_position|
//
// If mean_sad < skip_threshold, the tile is a "skip tile": all 8×8 split MVs
// covering that tile are zeroed.  After this pass:
//   MC produces residual = current − ref_same_pos (the actual temporal change).
//   For correctly identified skip tiles that residual is small, quantisation
//   drives it to zero, and the Rice encoder produces a compact all-skip RiceTile.
//
// Encoder / decoder symmetry: decoder sees MVs=0 + zero residual → outputs
// ref_same_pos tile unchanged.  Quality = PSNR(current, ref_same_pos), which
// is high by construction (we only skip tiles whose mean_sad < threshold).
//
// Motion-vector buffer is at 8×8-block resolution (quarter-pel units).
// Tiles are tile_size × tile_size pixels (256×256).  Each tile contains
// (tile_size / block_size_8)² = 32×32 = 1024 8×8 blocks.
//
// Dispatch: dispatch_workgroups(tiles_x, tiles_y, 1)
// Workgroup: 256 threads.
//   Phase 1  – each thread accumulates |cur−ref| for tile_pixels/256 pixels.
//   Phase 2  – parallel reduction → total SAD in shared_sum[0].
//   Phase 3  – skip decision (stored back in shared_sum[0] as 1.0 = skip).
//   Phase 4  – each thread zeroes its 4 assigned 8×8 blocks if tile is skip.
//   Phase 5  – (if block_skip_enabled) for non-skip tiles: per-8×8-block SAD.
//              Each thread independently evaluates its 4 blocks; zeroes any
//              block where mean_sad < skip_threshold. No reduction needed.

struct Params {
    padded_w:          u32,  // image width in pixels
    padded_h:          u32,  // image height in pixels
    tile_size:         u32,  // 256
    block_size_8:      u32,  // 8  (split-MV resolution)
    skip_threshold:    f32,  // mean per-pixel SAD below which tile/block is skipped
    block_skip_enabled: u32, // 1 = also apply per-8×8-block skip in non-skip tiles
}

@group(0) @binding(0) var<uniform>             params:         Params;
@group(0) @binding(1) var<storage, read>       current_plane:  array<f32>;
@group(0) @binding(2) var<storage, read>       ref_plane:      array<f32>;
@group(0) @binding(3) var<storage, read_write> motion_vectors: array<i32>;
// binding 4: per-tile skip decisions (1 = skip, 0 = not skip), one u32 per tile.
// Written by thread 0 after the skip decision in phase 3.
// Indexed as tile_y * tiles_x + tile_x.
@group(0) @binding(4) var<storage, read_write> tile_skip_out:  array<u32>;

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

    // Unconditional initialisation (Metal/M1 requires barriers outside divergent branches).
    shared_sum[lid] = 0.0;
    workgroupBarrier();

    // ── Phase 1: accumulate |current − reference| over all tile pixels ───────
    let tile_pixels        = params.tile_size * params.tile_size;
    let pixels_per_thread  = (tile_pixels + 255u) / 256u;  // = 256 for 256×256 tile

    if !out_of_bounds {
        let tile_origin_x = tile_x * params.tile_size;
        let tile_origin_y = tile_y * params.tile_size;
        var local_sum: f32 = 0.0;
        for (var i = 0u; i < pixels_per_thread; i++) {
            let pixel_idx = lid * pixels_per_thread + i;
            if pixel_idx < tile_pixels {
                let local_x = pixel_idx % params.tile_size;
                let local_y = pixel_idx / params.tile_size;
                let gx      = tile_origin_x + local_x;
                let gy      = tile_origin_y + local_y;
                let idx     = gy * params.padded_w + gx;
                local_sum  += abs(current_plane[idx] - ref_plane[idx]);
            }
        }
        shared_sum[lid] = local_sum;
    }
    workgroupBarrier();

    // ── Phase 2: parallel reduction — total SAD in shared_sum[0] ─────────────
    // Barriers must be unconditional (Metal/M1 requirement).
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        workgroupBarrier();
    }

    // ── Phase 3: skip decision — reuse shared_sum[0] as flag (1.0 = skip) ───
    if lid == 0u && !out_of_bounds {
        let mean_sad     = shared_sum[0] / f32(tile_pixels);
        shared_sum[0]    = select(0.0, 1.0, mean_sad < params.skip_threshold);
        // Export skip decision to per-tile output buffer so the encoder can zero
        // the quantised coefficient buffer for skip tiles after quantize (post-quant
        // zeroing avoids wavelet-filter bleed that plagued the spatial-domain approach).
        let tiles_x = params.padded_w / params.tile_size;
        tile_skip_out[tile_y * tiles_x + tile_x] = u32(shared_sum[0]);
    }
    workgroupBarrier();

    let blocks_per_tile_row = params.tile_size / params.block_size_8;    // 32
    let blocks_per_tile     = blocks_per_tile_row * blocks_per_tile_row; // 1024
    let blocks_per_thread   = (blocks_per_tile + 255u) / 256u;           // 4
    let blocks_total_x      = params.padded_w / params.block_size_8;

    // ── Phase 4: zero motion vectors for skip tiles ───────────────────────────
    if !out_of_bounds && shared_sum[0] > 0.5 {
        for (var b = 0u; b < blocks_per_thread; b++) {
            let block_in_tile = lid * blocks_per_thread + b;
            if block_in_tile < blocks_per_tile {
                let bx_in_tile = block_in_tile % blocks_per_tile_row;
                let by_in_tile = block_in_tile / blocks_per_tile_row;
                let block_gx   = tile_x * blocks_per_tile_row + bx_in_tile;
                let block_gy   = tile_y * blocks_per_tile_row + by_in_tile;
                let block_idx  = block_gy * blocks_total_x + block_gx;
                motion_vectors[block_idx * 2u     ] = 0;
                motion_vectors[block_idx * 2u + 1u] = 0;
            }
        }
    }

    // ── Phase 5: per-8×8-block skip in non-skip tiles ────────────────────────
    // For non-skip tiles, independently evaluate each 8×8 block.  Each thread
    // handles 4 blocks (no reduction needed — each block's SAD is independent).
    // Zeroes blocks whose zero-MV mean_sad < skip_threshold: the quantiser will
    // drive their near-zero residuals to zero anyway, saving MV + residual bits.
    if !out_of_bounds && shared_sum[0] < 0.5 && params.block_skip_enabled != 0u {
        for (var b = 0u; b < blocks_per_thread; b++) {
            let block_in_tile = lid * blocks_per_thread + b;
            if block_in_tile < blocks_per_tile {
                let bx_in_tile    = block_in_tile % blocks_per_tile_row;
                let by_in_tile    = block_in_tile / blocks_per_tile_row;
                let block_gx      = tile_x * blocks_per_tile_row + bx_in_tile;
                let block_gy      = tile_y * blocks_per_tile_row + by_in_tile;
                let block_orig_x  = block_gx * params.block_size_8;
                let block_orig_y  = block_gy * params.block_size_8;

                // Accumulate |current − ref| over all 64 pixels in this 8×8 block.
                var block_sad: f32 = 0.0;
                for (var dy = 0u; dy < params.block_size_8; dy++) {
                    for (var dx = 0u; dx < params.block_size_8; dx++) {
                        let px  = block_orig_x + dx;
                        let py  = block_orig_y + dy;
                        let idx = py * params.padded_w + px;
                        block_sad += abs(current_plane[idx] - ref_plane[idx]);
                    }
                }
                let mean_block_sad = block_sad / f32(params.block_size_8 * params.block_size_8);
                if mean_block_sad < params.skip_threshold {
                    let block_idx = block_gy * blocks_total_x + block_gx;
                    motion_vectors[block_idx * 2u     ] = 0;
                    motion_vectors[block_idx * 2u + 1u] = 0;
                }
            }
        }
    }
}
