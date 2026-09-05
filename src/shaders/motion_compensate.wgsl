// Motion compensation shader with quarter-pel bilinear interpolation.
// Two modes:
//   Forward (encoder): residual[x,y] = current[x,y] - predicted[x,y]
//   Inverse (decoder): reconstructed[x,y] = residual[x,y] + predicted[x,y]
//
// Motion vectors are in quarter-pel units (value 4 = 1 pixel). Fractional
// components (non-multiple of 4) use bilinear interpolation of the reference.
//
// One thread per pixel. Each thread reads its block's MV and applies it.
// Operates on a single plane — dispatch 3 times for Y, Co, Cg.

// blocks_x describes THIS plane's block grid; mv_blocks_{x,y} describe the motion-vector
// field's grid. They coincide for luma, but not for 4:2:0 chroma: luma and chroma planes are
// padded to a tile multiple independently, so chroma_padded_w/4 can exceed padded_w/8 (e.g.
// 720p: 768/4 = 192 chroma blocks against 1280/8 = 160 MV columns). Indexing the MV field with
// this plane's stride then shifts every row. See docs/decisions/0007.
struct Params {
    width: u32,
    height: u32,
    block_size: u32,
    mode: u32,       // 0 = forward, 1 = inverse
    blocks_x: u32,
    total_pixels: u32,
    mv_blocks_x: u32,  // row stride of the MV field
    mv_blocks_y: u32,  // rows in the MV field
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_plane: array<f32>;
@group(0) @binding(2) var<storage, read> reference_plane: array<f32>;
@group(0) @binding(3) var<storage, read> motion_vectors: array<i32>;
@group(0) @binding(4) var<storage, read_write> output_plane: array<f32>;

// Bilinear sample from reference plane at quarter-pel position (qx, qy).
// qx, qy are in quarter-pel units (pixel * 4 + subpel_offset, range 0..3).
fn bilinear_ref(qx: i32, qy: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);

    let ix = qx >> 2;        // integer part
    let iy = qy >> 2;
    let fx = qx & 3;         // fractional quarter-pel (0..3)
    let fy = qy & 3;

    if fx == 0 && fy == 0 {
        // On integer grid — direct lookup (most common case)
        let cx = clamp(ix, 0, w - 1);
        let cy = clamp(iy, 0, h - 1);
        return reference_plane[u32(cy) * params.width + u32(cx)];
    }

    let x0 = clamp(ix, 0, w - 1);
    let x1 = clamp(ix + 1, 0, w - 1);
    let y0 = clamp(iy, 0, h - 1);
    let y1 = clamp(iy + 1, 0, h - 1);

    let p00 = reference_plane[u32(y0) * params.width + u32(x0)];
    let p10 = reference_plane[u32(y0) * params.width + u32(x1)];
    let p01 = reference_plane[u32(y1) * params.width + u32(x0)];
    let p11 = reference_plane[u32(y1) * params.width + u32(x1)];

    let ffx = f32(fx) * 0.25;
    let ffy = f32(fy) * 0.25;

    let top = p00 * (1.0 - ffx) + p10 * ffx;
    let bot = p01 * (1.0 - ffx) + p11 * ffx;
    return top * (1.0 - ffy) + bot * ffy;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_idx = global_id.x;
    if pixel_idx >= params.total_pixels {
        return;
    }

    let x = pixel_idx % params.width;
    let y = pixel_idx / params.width;

    // Determine which block this pixel belongs to, then index the MV field with ITS stride.
    // Blocks past the end of the MV field are padding; clamp so encoder and decoder resolve
    // them identically instead of reading out of bounds.
    let bx = x / params.block_size;
    let by = y / params.block_size;
    let block_idx = min(by, params.mv_blocks_y - 1u) * params.mv_blocks_x
                  + min(bx, params.mv_blocks_x - 1u);

    // Read motion vector for this block (in quarter-pel units)
    let dx_qp = motion_vectors[block_idx * 2u];
    let dy_qp = motion_vectors[block_idx * 2u + 1u];

    // Reference position in quarter-pel units
    let ref_qx = i32(x) * 4 + dx_qp;
    let ref_qy = i32(y) * 4 + dy_qp;

    let input_val = input_plane[pixel_idx];
    let ref_val = bilinear_ref(ref_qx, ref_qy);

    if params.mode == 0u {
        // Forward: residual = current - predicted
        output_plane[pixel_idx] = input_val - ref_val;
    } else {
        // Inverse: reconstructed = residual + predicted
        output_plane[pixel_idx] = input_val + ref_val;
    }
}
