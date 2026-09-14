// GPU frame padding to tile-aligned dimensions.
//
// Replaces CPU pad_frame() to avoid ~10ms CPU allocation+copy at 1080p.
// Reads from unpadded input buffer (width*height*3 f32) and writes to
// padded output buffer (padded_w*padded_h*3 f32).
//
// 256 threads per workgroup, 1 thread per output pixel.
//
// ---------------------------------------------------------------------------
// What goes in the padding, and why it is not plain replication (PAD-1)
// ---------------------------------------------------------------------------
//
// A 1920x1080 frame is coded as 2048x1280, so **20.9% of the coded samples are outside the
// picture** and the decoder crops every one of them. They are therefore *don't-care*: what goes
// there is a free choice, and it was measured to be worth 6.6 of the 27.1-point intra gap to
// JPEG 2000 (INTRA-1 step 3, decision 0034).
//
// Plain edge replication is flat along the direction it extends — the cheapest possible in that
// axis — but it carries the edge line's full *transverse* detail across every padded row, and
// that detail is coded at every decomposition level. So the padding is made flat in **both** axes
// instead: replicate, then fade over RAMP pixels to one value per channel. Beyond the ramp the
// padding is constant and its detail bands go to zero, and because the fade starts from the
// replicated value there is no step at the picture seam to pollute the coefficients that also
// reconstruct visible pixels.
//
// Measured on four stills, q=80..94, `--abac`, BD-rate against replication at *identical visible
// quality*: **-4.63% RGB / -4.60% Y**. Mirroring the picture into the padding instead costs
// +11.4%, so replication was already the better of the two textbook extensions — what matters is
// being flat in the direction of extension, not smooth at the seam.
//
// **The fade target is EDGE_SAMPLES strided samples of the edge line being extended**, which
// measured identical to that line's exact mean (-4.63% either way) while costing nothing: the
// sample positions are a pure function of the plane dimensions, so this needs no reduction, no
// per-frame uniform and no host-side pass, and all threads in a strip read the same few
// addresses. One sample is worse (-4.38%) and a hardcoded mid-grey is worse and content-blind
// (-4.44%).
//
// `round()` is deliberate: the input is f32 on a 0-255 scale, every term below is an exact
// multiple of 1/64 for 8-bit input, and rounding to the input's own integer grid keeps the fill
// at least as cheap to code while making the encoder's padding exactly reproducible by the
// offline model in `scripts/meas_intra1_padding.py` — which is the canary for this shader.
//
// `fill_mode == 0` restores plain replication, which is what the A/B in that harness needs.

struct Params {
    width: u32,       // original image width
    height: u32,      // original image height
    padded_w: u32,    // padded (tile-aligned) width
    padded_h: u32,    // padded (tile-aligned) height
    fill_mode: u32,   // 0 = replicate, 1 = replicate then fade flat, 2 = zero (PAD-2 / Dirac)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Fade length in pixels. 8 measured best of {8, 32}; 32 leaves more of the edge line's detail in
// the padding and gives back half a point (-4.08% against -4.63%).
const RAMP: f32 = 8.0;

// Samples of the edge line averaged to get the fade target. 8 reproduces the exact line mean.
const EDGE_SAMPLES: u32 = 8u;

fn sample_pos(i: u32, extent: u32) -> u32 {
    // The positions the offline model uses: (2i+1) * extent / (2n), integer division.
    return (2u * i + 1u) * extent / (2u * EDGE_SAMPLES);
}

// Mean of EDGE_SAMPLES samples along the bottom edge row, for one channel.
fn bottom_target(ch: u32) -> f32 {
    let row = (params.height - 1u) * params.width;
    var acc = 0.0;
    for (var i = 0u; i < EDGE_SAMPLES; i = i + 1u) {
        acc = acc + input[(row + sample_pos(i, params.width)) * 3u + ch];
    }
    return acc / f32(EDGE_SAMPLES);
}

// Mean of EDGE_SAMPLES samples down the right edge column, for one channel.
fn right_target(ch: u32) -> f32 {
    let col = params.width - 1u;
    var acc = 0.0;
    for (var i = 0u; i < EDGE_SAMPLES; i = i + 1u) {
        acc = acc + input[(sample_pos(i, params.height) * params.width + col) * 3u + ch];
    }
    return acc / f32(EDGE_SAMPLES);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_pixels = params.padded_w * params.padded_h;
    if (idx >= total_pixels) {
        return;
    }

    let x = idx % params.padded_w;
    let y = idx / params.padded_w;

    // Clamp to original image bounds (edge replication)
    let sx = min(x, params.width - 1u);
    let sy = min(y, params.height - 1u);

    let src = (sy * params.width + sx) * 3u;
    let dst = (y * params.padded_w + x) * 3u;

    let below = y >= params.height;
    let right = x >= params.width;

    if (!(below || right)) {
        output[dst] = input[src];
        output[dst + 1u] = input[src + 1u];
        output[dst + 2u] = input[src + 2u];
        return;
    }

    // PAD-2 candidate (Dirac/Schroedinger inter): zero-extend so a zero-padded current
    // against a zero-padded reference differences to nothing in the padding itself.
    if (params.fill_mode == 2u) {
        output[dst] = 0.0;
        output[dst + 1u] = 0.0;
        output[dst + 2u] = 0.0;
        return;
    }

    if (params.fill_mode == 0u) {
        // Visible pixels are copied unchanged on both paths, and so is everything when the fill
        // is switched back to plain replication.
        output[dst] = input[src];
        output[dst + 1u] = input[src + 1u];
        output[dst + 2u] = input[src + 2u];
        return;
    }

    // Fade the two strips in the same order the offline model does — bottom across every column
    // first, then right across every row, so a corner pixel is faded twice. Getting this order
    // wrong is invisible in quality and shows up only as a byte-count mismatch in the canary.
    let t_below = min((f32(y - params.height) + 1.0) / RAMP, 1.0);
    let t_right = min((f32(x - params.width) + 1.0) / RAMP, 1.0);

    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        var v = input[src + ch];
        if (below) {
            v = v * (1.0 - t_below) + bottom_target(ch) * t_below;
        }
        if (right) {
            v = v * (1.0 - t_right) + right_target(ch) * t_right;
        }
        output[dst + ch] = round(v);
    }
}
