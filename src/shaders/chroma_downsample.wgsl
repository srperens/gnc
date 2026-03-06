// Chroma downsampling: box-filter average for 4:2:2 (2:1 horiz) and 4:2:0 (2:2 box).
// shift_x and shift_y control the subsampling ratio (0 = no change, 1 = half).
//
// dst_stride is the row stride used when writing output — may differ from dst_width
// when the destination buffer uses a padded layout (e.g. chroma data in a padded plane).
// Set dst_stride = dst_width for compact layout, or dst_stride = padded_dst_width for
// padded layout (so the output is compatible with shaders that read using padded_width).

struct Params {
    src_width:  u32,
    src_height: u32,
    dst_width:  u32,
    dst_height: u32,
    shift_x:    u32,
    shift_y:    u32,
    dst_stride: u32,  // row stride for dst writes; use dst_width for compact layout
    _pad1:      u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       src:    array<f32>;
@group(0) @binding(2) var<storage, read_write> dst:    array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.dst_width * params.dst_height { return; }

    let dx = idx % params.dst_width;
    let dy = idx / params.dst_width;

    let sx_start = dx << params.shift_x;
    let sy_start = dy << params.shift_y;

    var sum   = 0.0f;
    var count = 0u;
    let kx = 1u << params.shift_x;
    let ky = 1u << params.shift_y;
    for (var oy = 0u; oy < ky; oy++) {
        for (var ox = 0u; ox < kx; ox++) {
            let px = sx_start + ox;
            let py = sy_start + oy;
            if px < params.src_width && py < params.src_height {
                sum   += src[py * params.src_width + px];
                count += 1u;
            }
        }
    }
    // Write using dst_stride (padded row width) so output layout matches
    // the padded plane format expected by subsequent wavelet shaders.
    dst[dy * params.dst_stride + dx] = sum / f32(count);
}
