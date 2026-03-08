// Scale motion vectors by arithmetic right-shift for chroma subsampling.
//
// Luma MVs are in half-luma-pel units.  For 4:2:0 chroma (2× both axes)
// the equivalent chroma MV is half as large: dx_c = dx_luma >> 1.
//
// The block grid for 4×4 chroma blocks and 8×8 luma blocks is identical
// (both have padded_w/8 × padded_h/8 blocks), so the same MV index maps
// to the same spatial block — only the values need scaling.

struct Params {
    total_blocks: u32,
    shift_x: u32,   // 1 for 4:2:0 and 4:2:2 horizontal
    shift_y: u32,   // 1 for 4:2:0, 0 for 4:2:2
    _pad: u32,
}

@group(0) @binding(0) var<uniform>             params:  Params;
@group(0) @binding(1) var<storage, read>       src_mvs: array<i32>;
@group(0) @binding(2) var<storage, read_write> dst_mvs: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if b >= params.total_blocks { return; }

    // MVs stored as interleaved i32 pairs: [dx0, dy0, dx1, dy1, ...]
    // Arithmetic right-shift preserves sign for negative MVs.
    let dx = src_mvs[b * 2u];
    let dy = src_mvs[b * 2u + 1u];
    dst_mvs[b * 2u]      = dx >> params.shift_x;
    dst_mvs[b * 2u + 1u] = dy >> params.shift_y;
}
