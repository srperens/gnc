// Convert f32 RGB values to packed u16 (2 bytes per component, 2 components per u32).
// Used for 9–16-bit output (e.g. 10-bit content where 8-bit packing overflows).
// Each thread packs 2 consecutive f32 values into one u32:
//   bits  0–15: component[base+0] clamped to [0, peak]
//   bits 16–31: component[base+1] clamped to [0, peak]

struct Params {
    total_f32s: u32, // total number of f32 values (width * height * 3)
    peak: f32,       // max signal value; 1023.0 for 10-bit, 65535.0 for 16-bit
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;  // each thread handles one u32 = 2 u16 components
    let base = idx * 2u;

    if (base >= params.total_f32s) {
        return;
    }

    var lo: u32 = 0u;
    var hi: u32 = 0u;

    if (base < params.total_f32s) {
        lo = u32(clamp(input[base] + 0.5, 0.0, params.peak)) & 0xFFFFu;
    }
    if (base + 1u < params.total_f32s) {
        hi = u32(clamp(input[base + 1u] + 0.5, 0.0, params.peak)) & 0xFFFFu;
    }

    output[idx] = lo | (hi << 16u);
}
