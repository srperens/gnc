// Uniform scalar quantization / dequantization
// Each thread processes one coefficient.

struct Params {
    total_count: u32,
    step_size: f32,
    // 0 = forward (quantize), 1 = inverse (dequantize)
    direction: u32,
    // Dead-zone width as fraction of step_size (0.0 for uniform, 0.5 typical)
    dead_zone: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.total_count {
        return;
    }

    let val = input[idx];

    if params.direction == 0u {
        // Forward: quantize
        // Uniform: round(val / step_size)
        // With dead zone: values within [-dead_zone * step, dead_zone * step] map to 0
        let abs_val = abs(val);
        let sign_val = sign(val);
        let threshold = params.dead_zone * params.step_size;

        if abs_val < threshold {
            output[idx] = 0.0;
        } else {
            output[idx] = sign_val * floor(abs_val / params.step_size + 0.5);
        }
    } else {
        // Inverse: dequantize
        output[idx] = val * params.step_size;
    }
}
