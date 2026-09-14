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
//
// `quantum` is 4 for a full-pel luma vector, and **8 on any axis the chroma planes subsample**
// (LOSSLESS-5) — so (8,8) at 4:2:0 and (8,4) at 4:2:2. On a subsampled axis the chroma
// displacement is the luma one halved, so a multiple of 4 becomes a multiple of 2, which is
// half-pel *there*, and the interpolation comes back. Restricting the luma vector to even pixels
// on that axis is what makes the chroma displacement full-pel too, and it does it without the
// decoder needing to know: the decoder halves whatever the bitstream carries, so a multiple of 8
// arrives as a multiple of 4 on both sides. Rounding the encoder's chroma MV buffer instead was
// tried and is wrong for exactly that reason — the decoder does no rounding, so the two drift
// apart (46-56 dB and falling, measured).

struct Params {
    total_blocks: u32,
    quantum_x: i32,
    quantum_y: i32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read_write> mvs:    array<i32>;

// Nearest multiple of `q`, ties away from zero. WGSL integer division truncates toward zero, so
// the bias has to carry the sign: (v + q/2) / q * q for v >= 0, (v - q/2) / q * q below it.
fn round_to_quantum(v: i32, q: i32) -> i32 {
    let half = q / 2;
    let bias = select(-half, half, v >= 0);
    return ((v + bias) / q) * q;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if b >= params.total_blocks { return; }
    mvs[b * 2u]      = round_to_quantum(mvs[b * 2u],      params.quantum_x);
    mvs[b * 2u + 1u] = round_to_quantum(mvs[b * 2u + 1u], params.quantum_y);
}
