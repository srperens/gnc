// MED / LOCO-I prediction, forward. Replaces the wavelet on the lossless path (LOSSLESS-1).
//
// The residual is `pixel - median_edge_predictor(left, above, above-left)`, the predictor FFV1
// and JPEG-LS both use. It is what beats GNC at q=100: prediction against the neighbour rather
// than against the scale.
//
// Forward is embarrassingly parallel even though the predictor is a serial dependency in
// principle: at lossless the encoder's reconstruction *is* its input, so every thread can read
// the neighbours it needs straight from `src`. Only the decoder has to march the wavefront
// (med_reconstruct.wgsl), and that was measured at 4.9x one pass — 201 fps on 1080p 4:4:4.
//
// Prediction resets at every tile boundary, so tiles stay independently decodable.

struct Params {
    width: u32,
    height: u32,
    tile_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

// Median edge detector: picks min or max when the neighbourhood looks like an edge, and the
// planar extrapolation a+b-c when it looks smooth.
fn med(a: f32, b: f32, c: f32) -> f32 {
    let mx = max(a, b);
    let mn = min(a, b);
    if (c >= mx) { return mn; }
    if (c <= mn) { return mx; }
    return a + b - c;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.width * params.height) {
        return;
    }
    let x = idx % params.width;
    let y = idx / params.width;
    let lx = x % params.tile_size;
    let ly = y % params.tile_size;

    var p = 0.0;
    if (lx == 0u && ly == 0u) {
        p = 0.0;                                    // tile origin: nothing to predict from
    } else if (ly == 0u) {
        p = src[idx - 1u];                          // top row: left only
    } else if (lx == 0u) {
        p = src[idx - params.width];                // left column: above only
    } else {
        p = med(src[idx - 1u], src[idx - params.width], src[idx - params.width - 1u]);
    }
    dst[idx] = src[idx] - p;
}
