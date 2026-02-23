// Motion compensation shader.
// Two modes:
//   Forward (encoder): residual[x,y] = current[x,y] - reference[x+dx, y+dy]
//   Inverse (decoder): reconstructed[x,y] = residual[x,y] + reference[x+dx, y+dy]
//
// One thread per pixel. Each thread reads its block's MV and applies it.
// Operates on a single plane — dispatch 3 times for Y, Co, Cg.

struct Params {
    width: u32,
    height: u32,
    block_size: u32,
    mode: u32,       // 0 = forward, 1 = inverse
    blocks_x: u32,
    total_pixels: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_plane: array<f32>;
@group(0) @binding(2) var<storage, read> reference_plane: array<f32>;
@group(0) @binding(3) var<storage, read> motion_vectors: array<i32>;
@group(0) @binding(4) var<storage, read_write> output_plane: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_idx = global_id.x;
    if pixel_idx >= params.total_pixels {
        return;
    }

    let x = pixel_idx % params.width;
    let y = pixel_idx / params.width;

    // Determine which block this pixel belongs to
    let bx = x / params.block_size;
    let by = y / params.block_size;
    let block_idx = by * params.blocks_x + bx;

    // Read motion vector for this block
    let dx = motion_vectors[block_idx * 2u];
    let dy = motion_vectors[block_idx * 2u + 1u];

    // Reference coordinate (clamped to frame bounds)
    let ref_x = clamp(i32(x) + dx, 0, i32(params.width) - 1);
    let ref_y = clamp(i32(y) + dy, 0, i32(params.height) - 1);
    let ref_idx = u32(ref_y) * params.width + u32(ref_x);

    let input_val = input_plane[pixel_idx];
    let ref_val = reference_plane[ref_idx];

    if params.mode == 0u {
        // Forward: residual = current - predicted
        output_plane[pixel_idx] = input_val - ref_val;
    } else {
        // Inverse: reconstructed = residual + predicted
        output_plane[pixel_idx] = input_val + ref_val;
    }
}
