// Chroma nearest-neighbor upsampling (4:2:2 or 4:2:0 -> 4:4:4).
// Each destination pixel maps back to the nearest source pixel.
//
// Params layout must match ChromaResampleParams in chroma_resample.rs.

struct Params {
    src_width:         u32,
    src_height:        u32,
    dst_width:         u32,
    dst_height:        u32,
    dst_stride:        u32,   // unused by upsample; kept for struct alignment
    dst_height_padded: u32,   // unused by upsample; kept for struct alignment
    shift_x:           u32,
    shift_y:           u32,
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

    // Map dst pixel back to the nearest src pixel (simple nearest-neighbour)
    let sx = dx >> params.shift_x;
    let sy = dy >> params.shift_y;

    dst[idx] = src[sy * params.src_width + sx];
}
