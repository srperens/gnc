// Convert f32 RGB buffer to rgba8unorm storage texture.
// Each thread writes one pixel: reads 3 f32s (R,G,B), writes rgba with alpha=1.0.

struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let pixel_idx = y * params.width + x;
    let base = pixel_idx * 3u;

    let r = clamp(input[base + 0u] / 255.0, 0.0, 1.0);
    let g = clamp(input[base + 1u] / 255.0, 0.0, 1.0);
    let b = clamp(input[base + 2u] / 255.0, 0.0, 1.0);

    textureStore(output_tex, vec2<u32>(x, y), vec4<f32>(r, g, b, 1.0));
}
