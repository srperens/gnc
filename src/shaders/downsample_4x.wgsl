// 4× average downscale — produces a 1/4-resolution luma plane by averaging 4×4 blocks.
//
// Used as the first stage of pyramid motion estimation: full-res planes are downscaled
// before the coarse ME pass, extending the effective search range from ±32 to ±96 pixels
// without the O(range²) compute cost of a larger search window.
//
// Output dimensions: ceil(in_w/4) × ceil(in_h/4), clamped to grid boundary.
// Dispatch: (ceil(out_w/8), ceil(out_h/8), 1)

struct Params {
    in_w:  u32,
    in_h:  u32,
    out_w: u32,
    out_h: u32,
}

@group(0) @binding(0) var<uniform>              params: Params;
@group(0) @binding(1) var<storage, read>        input:  array<f32>;
@group(0) @binding(2) var<storage, read_write>  output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    if ox >= params.out_w || oy >= params.out_h {
        return;
    }

    // Average a 4×4 pixel window; clamp at image boundary.
    var sum: f32 = 0.0;
    let in_x = ox * 4u;
    let in_y = oy * 4u;
    for (var dy = 0u; dy < 4u; dy++) {
        for (var dx = 0u; dx < 4u; dx++) {
            let px = min(in_x + dx, params.in_w - 1u);
            let py = min(in_y + dy, params.in_h - 1u);
            sum += input[py * params.in_w + px];
        }
    }
    output[oy * params.out_w + ox] = sum * 0.0625; // / 16.0
}
