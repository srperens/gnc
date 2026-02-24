// Deinterleave one interleaved buffer into 3 separate plane buffers.
// plane0[i] = input[i*3+0], plane1[i] = input[i*3+1], plane2[i] = input[i*3+2]
// Inverse of interleave.wgsl.

struct Params {
    total_pixels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> plane0: array<f32>;
@group(0) @binding(3) var<storage, read_write> plane1: array<f32>;
@group(0) @binding(4) var<storage, read_write> plane2: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.total_pixels) {
        return;
    }

    let base = idx * 3u;
    plane0[idx] = input[base + 0u];
    plane1[idx] = input[base + 1u];
    plane2[idx] = input[base + 2u];
}
