// Crop padded RGB buffer to original dimensions.
// Each thread copies one pixel (3 f32 channels).

struct Params {
    src_width: u32,  // padded width
    dst_width: u32,  // original width
    dst_height: u32, // original height
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.x;
    let total_pixels = params.dst_width * params.dst_height;

    if (pixel_idx >= total_pixels) {
        return;
    }

    let y = pixel_idx / params.dst_width;
    let x = pixel_idx % params.dst_width;

    let src_base = (y * params.src_width + x) * 3u;
    let dst_base = pixel_idx * 3u;

    output[dst_base + 0u] = input[src_base + 0u];
    output[dst_base + 1u] = input[src_base + 1u];
    output[dst_base + 2u] = input[src_base + 2u];
}
