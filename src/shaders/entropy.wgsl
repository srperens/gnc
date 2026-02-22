// Phase 1 entropy coding: Simple coefficient packing
// For the baseline, we just convert quantized f32 coefficients to i16 values
// and pack them into a u32 output buffer (2 coefficients per u32).
//
// This is NOT real entropy coding — it's just packing for measurement.
// Real entropy coding (rANS, tANS, Huffman) comes in Phase 2.

struct Params {
    total_count: u32,
    // 0 = encode (f32 -> packed i16), 1 = decode (packed i16 -> f32)
    direction: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_packed: array<u32>;
// For decode direction:
@group(0) @binding(3) var<storage, read_write> output_f32: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if params.direction == 0u {
        // Encode: pack pairs of f32 quantized values into u32
        let pair_idx = idx;
        let src_idx = pair_idx * 2u;
        if src_idx >= params.total_count {
            return;
        }

        // Clamp to i16 range and pack two i16 into one u32
        var val0 = i32(clamp(input_f32[src_idx], -32768.0, 32767.0));
        var val1: i32 = 0;
        if src_idx + 1u < params.total_count {
            val1 = i32(clamp(input_f32[src_idx + 1u], -32768.0, 32767.0));
        }

        // Pack: low 16 bits = val0, high 16 bits = val1
        let packed = (u32(val0) & 0xFFFFu) | ((u32(val1) & 0xFFFFu) << 16u);
        output_packed[pair_idx] = packed;
    } else {
        // Decode: unpack u32 into pairs of f32
        let pair_idx = idx;
        let dst_idx = pair_idx * 2u;
        if dst_idx >= params.total_count {
            return;
        }

        let packed = output_packed[pair_idx];
        // Extract as signed i16
        var low = i32(packed & 0xFFFFu);
        if low >= 32768 {
            low = low - 65536;
        }
        var high = i32((packed >> 16u) & 0xFFFFu);
        if high >= 32768 {
            high = high - 65536;
        }

        output_f32[dst_idx] = f32(low);
        if dst_idx + 1u < params.total_count {
            output_f32[dst_idx + 1u] = f32(high);
        }
    }
}
