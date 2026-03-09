// Bidirectional motion compensation shader for chroma dimensions (4:2:0).
// Same as motion_compensate_bidir.wgsl but operates at chroma resolution.
//
// Three prediction modes per block:
//   Mode 0 (forward only):  pred = MC(fwd_ref, fwd_mv)
//   Mode 1 (backward only): pred = MC(bwd_ref, bwd_mv)
//   Mode 2 (bidir average): pred = (MC(fwd_ref, fwd_mv) + MC(bwd_ref, bwd_mv)) / 2
//
// Two operation modes:
//   Forward (encoder): output = current - pred
//   Inverse (decoder): output = residual + pred
//
// MVs are already scaled for chroma (÷2 from luma). Half-pel units.
// One thread per chroma pixel. Dispatch for each chroma plane (Co, Cg) separately.

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

fn sample_hp_fwd(x2: i32, y2: i32, w: u32, h: u32) -> f32 {
    let fx = x2 >> 1;
    let fy = y2 >> 1;
    let frac_x = x2 & 1;
    let frac_y = y2 & 1;
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
    } else if frac_x == 1 && frac_y == 0 {
        return (p00 + p10) * 0.5;
    } else if frac_x == 0 && frac_y == 1 {
        return (p00 + p01) * 0.5;
    } else {
        return (p00 + p10 + p01 + p11) * 0.25;
    }
}

fn sample_hp_bwd(x2: i32, y2: i32, w: u32, h: u32) -> f32 {
    let fx = x2 >> 1;
    let fy = y2 >> 1;
    let frac_x = x2 & 1;
    let frac_y = y2 & 1;
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
    } else if frac_x == 1 && frac_y == 0 {
        return (p00 + p10) * 0.5;
    } else if frac_x == 0 && frac_y == 1 {
        return (p00 + p01) * 0.5;
    } else {
        return (p00 + p10 + p01 + p11) * 0.25;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_idx = global_id.x;
    if pixel_idx >= params.total_pixels {
        return;
    }

    let x = pixel_idx % params.width;
    let y = pixel_idx / params.width;

    let bx = x / params.block_size;
    let by = y / params.block_size;
    let block_idx = by * params.blocks_x + bx;

    let fwd_dx = fwd_motion_vectors[block_idx * 2u];
    let fwd_dy = fwd_motion_vectors[block_idx * 2u + 1u];
    let bwd_dx = bwd_motion_vectors[block_idx * 2u];
    let bwd_dy = bwd_motion_vectors[block_idx * 2u + 1u];
    let bmode = block_modes[block_idx];

    var pred: f32 = 0.0;

    let px2 = i32(x) * 2;
    let py2 = i32(y) * 2;

    if bmode == 0u {
        pred = sample_hp_fwd(px2 + fwd_dx, py2 + fwd_dy, params.width, params.height);
    } else if bmode == 1u {
        pred = sample_hp_bwd(px2 + bwd_dx, py2 + bwd_dy, params.width, params.height);
    } else {
        let fwd_val = sample_hp_fwd(px2 + fwd_dx, py2 + fwd_dy, params.width, params.height);
        let bwd_val = sample_hp_bwd(px2 + bwd_dx, py2 + bwd_dy, params.width, params.height);
        pred = (fwd_val + bwd_val) * 0.5;
    }

    let input_val = input_plane[pixel_idx];

    if params.mode == 0u {
        output_plane[pixel_idx] = input_val - pred;
    } else {
        output_plane[pixel_idx] = input_val + pred;
    }
}
