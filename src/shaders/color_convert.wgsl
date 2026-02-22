// Color space conversion: RGB <-> YCoCg-R
// YCoCg-R is a reversible integer color transform used for lossless/near-lossless coding.
// We operate on f32 for GPU convenience; values are in [0, 1] range for 8-bit
// or [0, 1023] for 10-bit (normalized later).

struct Params {
    width: u32,
    height: u32,
    // 0 = forward (RGB -> YCoCg), 1 = inverse (YCoCg -> RGB)
    direction: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Input: 3 planes interleaved as [R0,G0,B0, R1,G1,B1, ...] or [Y,Co,Cg,...]
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.x;
    let total_pixels = params.width * params.height;

    if pixel_idx >= total_pixels {
        return;
    }

    let base = pixel_idx * 3u;

    if params.direction == 0u {
        // Forward: RGB -> YCoCg-R
        let r = input[base + 0u];
        let g = input[base + 1u];
        let b = input[base + 2u];

        let co = r - b;
        let t = b + co * 0.5;
        let cg = g - t;
        let y = t + cg * 0.5;

        output[base + 0u] = y;
        output[base + 1u] = co;
        output[base + 2u] = cg;
    } else {
        // Inverse: YCoCg-R -> RGB
        let y = input[base + 0u];
        let co = input[base + 1u];
        let cg = input[base + 2u];

        let t = y - cg * 0.5;
        let g = cg + t;
        let b = t - co * 0.5;
        let r = b + co;

        output[base + 0u] = r;
        output[base + 1u] = g;
        output[base + 2u] = b;
    }
}
