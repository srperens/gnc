// GPU frame padding — edge-replicate extension to tile-aligned dimensions.
//
// Replaces CPU pad_frame() to avoid ~10ms CPU allocation+copy at 1080p.
// Reads from unpadded input buffer (width*height*3 f32) and writes to
// padded output buffer (padded_w*padded_h*3 f32) with edge replication.
//
// 256 threads per workgroup, 1 thread per output pixel.

struct Params {
    width: u32,       // original image width
    height: u32,      // original image height
    padded_w: u32,    // padded (tile-aligned) width
    padded_h: u32,    // padded (tile-aligned) height
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_pixels = params.padded_w * params.padded_h;
    if (idx >= total_pixels) {
        return;
    }

    let x = idx % params.padded_w;
    let y = idx / params.padded_w;

    // Clamp to original image bounds (edge replication)
    let sx = min(x, params.width - 1u);
    let sy = min(y, params.height - 1u);

    let src = (sy * params.width + sx) * 3u;
    let dst = (y * params.padded_w + x) * 3u;

    output[dst] = input[src];
    output[dst + 1u] = input[src + 1u];
    output[dst + 2u] = input[src + 2u];
}
