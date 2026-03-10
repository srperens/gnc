// Zero residual pixels for skip-mode 16×16 blocks.
//
// A "skip" block has SAD (sum-of-absolute-differences) below a threshold,
// meaning the reference block is already a good predictor. Zeroing the
// residual tells the wavelet + entropy coder there is nothing to encode —
// the decoder will copy the reference block directly. Cost: ~1 bit overhead
// from the existing MV skip bitmap. Benefit: zero residual coefficients
// compress extremely well with Rice (ZRL coding).
//
// Input:  mc_out  — MC residual buffer (f32, linearised row-major, size = padded_w * padded_h)
//         sad_buf — 16×16-block SAD values from block_match (u32, one per 16×16 block)
// Output: mc_out  — skip-block regions zeroed in place (other regions unchanged)
//
// Dispatch: one thread per pixel, total = padded_w * padded_h.

struct Params {
    padded_w:      u32,
    padded_h:      u32,
    blocks_x:      u32,   // padded_w / 16
    total_pixels:  u32,   // padded_w * padded_h
    skip_threshold: u32,  // SAD < this → skip (per 16×16 block)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform>             params:  Params;
@group(0) @binding(1) var<storage, read>       sad_buf: array<u32>;
@group(0) @binding(2) var<storage, read_write> mc_out:  array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.x;
    if pixel_idx >= params.total_pixels { return; }

    let px = pixel_idx % params.padded_w;
    let py = pixel_idx / params.padded_w;

    // Which 16×16 block does this pixel belong to?
    let bx = px / 16u;
    let by = py / 16u;
    let block_idx = by * params.blocks_x + bx;

    // Zero only if SAD is below the skip threshold.
    if sad_buf[block_idx] < params.skip_threshold {
        mc_out[pixel_idx] = 0.0;
    }
}
