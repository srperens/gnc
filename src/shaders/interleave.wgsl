// Interleave 3 separate plane buffers into one interleaved buffer.
// output[i*3+0] = plane0[i], output[i*3+1] = plane1[i], output[i*3+2] = plane2[i]

struct Params {
    total_pixels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> plane0: array<f32>;
@group(0) @binding(2) var<storage, read> plane1: array<f32>;
@group(0) @binding(3) var<storage, read> plane2: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_pixels) {
        return;
    }

    let base = idx * 3u;
    output[base + 0u] = plane0[idx];
    output[base + 1u] = plane1[idx];
    output[base + 2u] = plane2[idx];
}
