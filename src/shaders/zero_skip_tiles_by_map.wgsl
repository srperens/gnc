// Zero quantised coefficient tiles according to a pre-computed per-tile skip map.
//
// Called AFTER wavelet + quantize and BEFORE entropy encode.
// For each tile where tile_skip_map[tile_y * tiles_x + tile_x] == 1, all
// coefficients in that tile are zeroed.
//
// This produces compact all-skip RiceTiles at entropy encode time.
// The decoder reconstructs skip tiles from MC prediction only (zero residual),
// which is correct because the skip decision was made on the zero-MV SAD before
// MC compensation (tile_skip_motion.wgsl), ensuring that ref_same_pos ≈ current.
//
// Correctness vs the broken spatial-domain approach (#59):
//   Old (#59): zeroed mc_out (spatial residual) BEFORE wavelet.
//              → wavelet filter spread energy from non-zero neighbouring tiles into
//                the skip tile, making the coefficients non-zero → bleed.
//   New (this shader): zeros AFTER quantize, when coefficients are already
//              tile-independent. No bleed possible.
//
// Dispatch: dispatch_workgroups(tiles_x, tiles_y, 1)
// One workgroup per tile. Workgroup size 256 threads.
// Each thread zeros (tile_pixels + 255) / 256 coefficients.
//
// Shared memory: none needed (per-tile skip bit read from uniform map).

struct Params {
    padded_w:  u32,
    padded_h:  u32,
    tile_size: u32,
    _pad:      u32,
}

@group(0) @binding(0) var<uniform>            params:        Params;
@group(0) @binding(1) var<storage, read>      tile_skip_map: array<u32>;
@group(0) @binding(2) var<storage, read_write> coeffs:       array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)           wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid:   u32,
) {
    let tiles_x = params.padded_w / params.tile_size;
    let tiles_y = params.padded_h / params.tile_size;
    let tile_x  = wg_id.x;
    let tile_y  = wg_id.y;

    if tile_x >= tiles_x || tile_y >= tiles_y {
        return;
    }

    // Read skip decision for this tile.
    let skip = tile_skip_map[tile_y * tiles_x + tile_x];
    if skip == 0u {
        return; // Not a skip tile — leave coefficients unchanged.
    }

    // Zero all coefficients in this tile.
    let tile_pixels       = params.tile_size * params.tile_size;
    let pixels_per_thread = (tile_pixels + 255u) / 256u;
    let tile_origin_x     = tile_x * params.tile_size;
    let tile_origin_y     = tile_y * params.tile_size;

    for (var i = 0u; i < pixels_per_thread; i++) {
        let pixel_idx = lid * pixels_per_thread + i;
        if pixel_idx < tile_pixels {
            let local_x = pixel_idx % params.tile_size;
            let local_y = pixel_idx / params.tile_size;
            let px      = tile_origin_x + local_x;
            let py      = tile_origin_y + local_y;
            coeffs[py * params.padded_w + px] = 0.0;
        }
    }
}
