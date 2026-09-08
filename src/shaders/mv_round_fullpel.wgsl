// Round motion vectors to full-pel, in place.
//
// Why this exists (BUG-39 cause 4): `motion_compensate.wgsl` interpolates the reference
// bilinearly at quarter-pel positions, so a sub-pel motion vector makes the prediction
// *fractional*. The residual `current - prediction` is then fractional too, and a lossless
// configuration quantises it at step 1.0 — which rounds it, and rounding is exactly the thing a
// lossless configuration must not do. At a full-pel vector `bilinear_ref` takes its
// `fx == 0 && fy == 0` early-out and returns a reference sample unchanged, so the prediction is
// an integer, the residual is an integer, and step 1.0 is an identity.
//
// The decoder needs no counterpart: it reads whatever vectors the bitstream carries and runs the
// same interpolation, so full-pel vectors take the same exact path there.
//
// MVs are in quarter-pel units (value 4 = one pixel), stored as interleaved i32 pairs
// [dx0, dy0, dx1, dy1, ...] — the same layout `motion_mv_scale.wgsl` reads.

struct Params {
    total_blocks: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> mvs:    array<i32>;

// Nearest multiple of 4, ties away from zero. WGSL integer division truncates toward zero, so
// the bias has to carry the sign: (v + 2) / 4 * 4 for v >= 0, (v - 2) / 4 * 4 below it.
fn round_to_fullpel(v: i32) -> i32 {
    let bias = select(-2, 2, v >= 0);
    return ((v + bias) / 4) * 4;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if b >= params.total_blocks { return; }
    mvs[b * 2u]      = round_to_fullpel(mvs[b * 2u]);
    mvs[b * 2u + 1u] = round_to_fullpel(mvs[b * 2u + 1u]);
}
