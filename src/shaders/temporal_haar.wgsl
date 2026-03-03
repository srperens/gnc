// Temporal Haar Wavelet Lifting (per-element)
//
// Applies Haar lifting between two frames in the wavelet domain.
// Pure per-element math — no shared memory, no cross-thread deps.
// Dispatch: (ceil(count / 256), 1, 1)
//
// Forward (direction=0): frame_a, frame_b → low, high
//   high = b - a
//   low  = a + 0.5 * high
//
// Inverse (direction=1): low, high → frame_a, frame_b
//   a = low - 0.5 * high
//   b = high + a

struct Params {
    count: u32,      // total number of f32 elements (padded_w * padded_h)
    direction: u32,  // 0 = forward, 1 = inverse
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> frame_a: array<f32>;
@group(0) @binding(2) var<storage, read> frame_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_low: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_high: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count {
        return;
    }

    let a = frame_a[idx];
    let b = frame_b[idx];

    if params.direction == 0u {
        // Forward: split into low-frequency and high-frequency
        let high = b - a;
        let low = a + 0.5 * high;
        out_low[idx] = low;
        out_high[idx] = high;
    } else {
        // Inverse: reconstruct from low (frame_a) and high (frame_b)
        let low = a;
        let high = b;
        let ra = low - 0.5 * high;
        let rb = high + ra;
        out_low[idx] = ra;
        out_high[idx] = rb;
    }
}
