// Convert f32 RGB values to packed u8 (4 bytes per u32).
// Each thread packs 4 consecutive f32 values into one u32.

struct Params {
    total_f32s: u32, // total number of f32 values (width * height * 3)
    // peak: max signal value; 255.0 for 8-bit, 1023.0 for 10-bit
    peak: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;  // each thread handles one u32 = 4 bytes
    let base = idx * 4u;

    if (base >= params.total_f32s) {
        return;
    }

    var packed: u32 = 0u;
    for (var i = 0u; i < 4u; i++) {
        let fi = base + i;
        var val: u32 = 0u;
        if (fi < params.total_f32s) {
            let f = input[fi];
            val = u32(clamp(f + 0.5, 0.0, params.peak));
        }
        packed = packed | (val << (i * 8u));
    }
    output[idx] = packed;
}
