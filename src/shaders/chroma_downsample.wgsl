// Chroma downsampling: box-filter average for 4:2:2 (2:1 horiz) and 4:2:0 (2-2 box).
// shift_x and shift_y control the subsampling ratio (0 = no change, 1 = half).
//
// The shader processes the full dst_stride × dst_height_padded region (tile-aligned).
// For pixels in the valid chroma region (col < dst_width && row < dst_height) it
// computes a box-filter average from the source.  For pixels in the padding zone
// (col >= dst_width or row >= dst_height) it replicates the nearest valid edge pixel.
// This ensures the entire padded buffer is initialised before the wavelet transform
// runs, eliminating the garbage-data corruption that caused 28-36 dB PSNR loss.

struct Params {
    src_width:         u32,
    src_height:        u32,
    dst_width:         u32,   // valid chroma width  (before tile rounding)
    dst_height:        u32,   // valid chroma height (before tile rounding)
    dst_stride:        u32,   // = chroma_padded_width  (tile-aligned row stride)
    dst_height_padded: u32,   // = chroma_padded_height (tile-aligned)
    shift_x:           u32,
    shift_y:           u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       src:    array<f32>;
@group(0) @binding(2) var<storage, read_write> dst:    array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.dst_stride * params.dst_height_padded;
    if idx >= total { return; }

    let col = idx % params.dst_stride;
    let row = idx / params.dst_stride;

    // Clamp to the valid chroma region for edge replication in the padding zone.
    let clamped_col = min(col, params.dst_width  - 1u);
    let clamped_row = min(row, params.dst_height - 1u);

    // Map clamped dst pixel back to the top-left source pixel of its block.
    let sx_start = clamped_col << params.shift_x;
    let sy_start = clamped_row << params.shift_y;

    // Box-filter average over the source block (clamped to source bounds).
    var sum   = 0.0f;
    var count = 0u;
    let kx = 1u << params.shift_x;
    let ky = 1u << params.shift_y;
    for (var oy = 0u; oy < ky; oy++) {
        for (var ox = 0u; ox < kx; ox++) {
            let px = min(sx_start + ox, params.src_width  - 1u);
            let py = min(sy_start + oy, params.src_height - 1u);
            sum   += src[py * params.src_width + px];
            count += 1u;
        }
    }
    // idx == row * dst_stride + col, which is the correct padded-layout offset.
    dst[idx] = sum / f32(count);
}
