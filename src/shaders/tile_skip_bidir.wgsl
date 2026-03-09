// Tile skip mode — zero motion vectors for spatially-stable B-frame tiles.
//
// For each tile, computes the zero-MV bidirectional prediction error:
//   mean_sad = mean |current_pixel - (fwd_ref[same_pos] + bwd_ref[same_pos]) * 0.5|
//
// If mean_sad < skip_threshold, the tile is a "skip tile": all 16×16 block MVs
// covering that tile are zeroed (both fwd and bwd) and block_modes is set to 2
// (bidirectional).  After this pass:
//   MC produces residual = current − bidir_pred_same_pos (the actual temporal change).
//   For correctly identified skip tiles that residual is small, quantisation
//   drives it to zero, and the Rice encoder produces a compact all-skip RiceTile.
//
// Encoder / decoder symmetry: decoder sees MVs=0 + bidir_mode + zero residual →
// outputs bidir_avg(fwd_ref, bwd_ref) tile unchanged.  Quality = PSNR(current,
// bidir_avg), which is high by construction (we only skip tiles below threshold).
//
// Motion-vector buffers are at 16×16-block resolution (quarter-pel units).
// Tiles are tile_size × tile_size pixels (256×256).  Each tile contains
// (tile_size / block_size)² = 16×16 = 256 16×16 blocks.
//
// Dispatch: dispatch_workgroups(tiles_x, tiles_y, 1)
// Workgroup: 256 threads.
//   Phase 1  – each thread accumulates |cur − bidir_pred| for tile_pixels/256 pixels.
//   Phase 2  – parallel reduction → total SAD in shared_sum[0].
//   Phase 3  – skip decision (stored back in shared_sum[0] as 1.0 = skip).
//   Phase 4  – each thread zeroes its assigned 16×16 blocks if skip is set.

struct Params {
    padded_w:       u32,  // image width in pixels (padded to tile alignment)
    padded_h:       u32,  // image height in pixels (padded to tile alignment)
    tile_size:      u32,  // 256
    block_size:     u32,  // 16 (16×16 block ME resolution for B-frames)
    skip_threshold: f32,  // mean per-pixel SAD below which tile is skipped
    _pad0:          u32,
    _pad1:          u32,
    _pad2:          u32,
}

@group(0) @binding(0) var<uniform>             params:             Params;
@group(0) @binding(1) var<storage, read>       current_plane:      array<f32>;
@group(0) @binding(2) var<storage, read>       fwd_ref_plane:      array<f32>;
@group(0) @binding(3) var<storage, read>       bwd_ref_plane:      array<f32>;
@group(0) @binding(4) var<storage, read_write> fwd_motion_vectors: array<i32>;
@group(0) @binding(5) var<storage, read_write> bwd_motion_vectors: array<i32>;
@group(0) @binding(6) var<storage, read_write> block_modes:        array<u32>;

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

    // ── Phase 1: accumulate |current − bidir_pred| over all tile pixels ──────
    let tile_pixels       = params.tile_size * params.tile_size;
    let pixels_per_thread = (tile_pixels + 255u) / 256u;  // = 256 for 256×256 tile

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
                // Zero-MV bidir prediction: average of forward and backward references
                // at the same pixel position (no displacement).
                let bidir_pred = (fwd_ref_plane[idx] + bwd_ref_plane[idx]) * 0.5;
                local_sum += abs(current_plane[idx] - bidir_pred);
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
        let mean_sad  = shared_sum[0] / f32(tile_pixels);
        shared_sum[0] = select(0.0, 1.0, mean_sad < params.skip_threshold);
    }
    workgroupBarrier();

    // ── Phase 4: zero motion vectors + set bidir mode for skip tiles ─────────
    // A 256×256 tile at 16×16 resolution has 16×16 = 256 blocks.
    // With 256 threads: 1 block per thread (blocks_per_thread = 1).
    if !out_of_bounds && shared_sum[0] > 0.5 {
        let blocks_per_tile_row = params.tile_size / params.block_size;    // 16
        let blocks_per_tile     = blocks_per_tile_row * blocks_per_tile_row; // 256
        let blocks_per_thread   = max(1u, (blocks_per_tile + 255u) / 256u); // 1
        let blocks_total_x      = params.padded_w / params.block_size;

        for (var b = 0u; b < blocks_per_thread; b++) {
            let block_in_tile = lid * blocks_per_thread + b;
            if block_in_tile < blocks_per_tile {
                let bx_in_tile = block_in_tile % blocks_per_tile_row;
                let by_in_tile = block_in_tile / blocks_per_tile_row;
                let block_gx   = tile_x * blocks_per_tile_row + bx_in_tile;
                let block_gy   = tile_y * blocks_per_tile_row + by_in_tile;
                let block_idx  = block_gy * blocks_total_x + block_gx;
                // Zero both forward and backward MVs.
                fwd_motion_vectors[block_idx * 2u     ] = 0;
                fwd_motion_vectors[block_idx * 2u + 1u] = 0;
                bwd_motion_vectors[block_idx * 2u     ] = 0;
                bwd_motion_vectors[block_idx * 2u + 1u] = 0;
                // Force bidir mode so MC averages fwd+bwd refs (both at zero MV).
                block_modes[block_idx] = 2u;
            }
        }
    }
}
