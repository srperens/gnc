// Bidirectional motion compensation shader.
// Three prediction modes per block:
//   Mode 0 (forward only):  pred = MC(fwd_ref, fwd_mv)
//   Mode 1 (backward only): pred = MC(bwd_ref, bwd_mv)
//   Mode 2 (bidir average): pred = (MC(fwd_ref, fwd_mv) + MC(bwd_ref, bwd_mv)) / 2
//
// Two operation modes:
//   Forward (encoder): output = current - pred
//   Inverse (decoder): output = residual + pred
//
// MVs are in quarter-pel units (value 4 = 1 pixel). Bilinear interpolation for sub-pel samples.
// One thread per pixel. Dispatch 3 times for Y, Co, Cg.

struct Params {
    width: u32,
    height: u32,
    block_size: u32,
    mode: u32,       // 0 = forward (residual), 1 = inverse (reconstruct)
    blocks_x: u32,
    total_pixels: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_plane: array<f32>;
@group(0) @binding(2) var<storage, read> fwd_reference: array<f32>;
@group(0) @binding(3) var<storage, read> bwd_reference: array<f32>;
@group(0) @binding(4) var<storage, read> fwd_motion_vectors: array<i32>;
@group(0) @binding(5) var<storage, read> bwd_motion_vectors: array<i32>;
@group(0) @binding(6) var<storage, read> block_modes: array<u32>;
@group(0) @binding(7) var<storage, read_write> output_plane: array<f32>;

// Quarter-pel sample from forward reference (inlined because WGSL cannot pass storage ptrs).
// x4, y4 are in quarter-pel units (pixel * 4 + subpel, range 0..3).
fn sample_hp_fwd(x4: i32, y4: i32, w: u32, h: u32) -> f32 {
    let fx = x4 >> 2;
    let fy = y4 >> 2;
    let frac_x = x4 & 3;
    let frac_y = y4 & 3;
    let x0 = clamp(fx, 0, i32(w) - 1);
    let y0 = clamp(fy, 0, i32(h) - 1);
    let x1 = clamp(fx + 1, 0, i32(w) - 1);
    let y1 = clamp(fy + 1, 0, i32(h) - 1);
    let p00 = fwd_reference[u32(y0) * w + u32(x0)];
    let p10 = fwd_reference[u32(y0) * w + u32(x1)];
    let p01 = fwd_reference[u32(y1) * w + u32(x0)];
    let p11 = fwd_reference[u32(y1) * w + u32(x1)];
    if frac_x == 0 && frac_y == 0 {
        return p00;
    }
    let ffx = f32(frac_x) * 0.25;
    let ffy = f32(frac_y) * 0.25;
    let top = p00 * (1.0 - ffx) + p10 * ffx;
    let bot = p01 * (1.0 - ffx) + p11 * ffx;
    return top * (1.0 - ffy) + bot * ffy;
}

// Quarter-pel sample from backward reference.
fn sample_hp_bwd(x4: i32, y4: i32, w: u32, h: u32) -> f32 {
    let fx = x4 >> 2;
    let fy = y4 >> 2;
    let frac_x = x4 & 3;
    let frac_y = y4 & 3;
    let x0 = clamp(fx, 0, i32(w) - 1);
    let y0 = clamp(fy, 0, i32(h) - 1);
    let x1 = clamp(fx + 1, 0, i32(w) - 1);
    let y1 = clamp(fy + 1, 0, i32(h) - 1);
    let p00 = bwd_reference[u32(y0) * w + u32(x0)];
    let p10 = bwd_reference[u32(y0) * w + u32(x1)];
    let p01 = bwd_reference[u32(y1) * w + u32(x0)];
    let p11 = bwd_reference[u32(y1) * w + u32(x1)];
    if frac_x == 0 && frac_y == 0 {
        return p00;
    }
    let ffx = f32(frac_x) * 0.25;
    let ffy = f32(frac_y) * 0.25;
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

    // Determine which block this pixel belongs to
    let bx = x / params.block_size;
    let by = y / params.block_size;
    let block_idx = by * params.blocks_x + bx;

    // Read MVs (in half-pel units) and block mode
    let fwd_dx = fwd_motion_vectors[block_idx * 2u];
    let fwd_dy = fwd_motion_vectors[block_idx * 2u + 1u];
    let bwd_dx = bwd_motion_vectors[block_idx * 2u];
    let bwd_dy = bwd_motion_vectors[block_idx * 2u + 1u];
    let bmode = block_modes[block_idx];

    // Compute prediction value based on mode
    var pred: f32 = 0.0;

    // Quarter-pel coordinates: pixel_pos * 4 + mv_quarterpel
    let px2 = i32(x) * 4;
    let py2 = i32(y) * 4;

    if bmode == 0u {
        // Forward only
        pred = sample_hp_fwd(px2 + fwd_dx, py2 + fwd_dy, params.width, params.height);
    } else if bmode == 1u {
        // Backward only
        pred = sample_hp_bwd(px2 + bwd_dx, py2 + bwd_dy, params.width, params.height);
    } else {
        // Bidirectional average
        let fwd_val = sample_hp_fwd(px2 + fwd_dx, py2 + fwd_dy, params.width, params.height);
        let bwd_val = sample_hp_bwd(px2 + bwd_dx, py2 + bwd_dy, params.width, params.height);
        pred = (fwd_val + bwd_val) * 0.5;
    }

    let input_val = input_plane[pixel_idx];

    if params.mode == 0u {
        // Forward: residual = current - predicted
        output_plane[pixel_idx] = input_val - pred;
    } else {
        // Inverse: reconstructed = residual + predicted
        output_plane[pixel_idx] = input_val + pred;
    }
}
