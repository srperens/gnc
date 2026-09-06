// MED / LOCO-I reconstruction, inverse. The wavefront half of LOSSLESS-1.
//
// Pixel (x,y) cannot be reconstructed before (x-1,y) and (x,y-1), so the tile is marched as
// anti-diagonals: every pixel on diagonal d depends only on d-1 and d-2, so a whole diagonal
// goes in parallel. One workgroup per tile, 2*tile_size-1 diagonals, a storage barrier between
// each. Measured cost against the same arithmetic with independent neighbours: 4.9x
// (1.02 ms -> 4.98 ms on 1080p 4:4:4), which is 201 fps and about 25% of a 20 ms frame budget.
// The cost is occupancy — the short diagonals at the corners idle most of the workgroup — not
// synchronisation: the weaker barrier measured 4.79 ms, barely different.

struct Params {
    width: u32,
    height: u32,
    tile_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;        // residuals
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;  // reconstruction, read back as context

fn med(a: f32, b: f32, c: f32) -> f32 {
    let mx = max(a, b);
    let mn = min(a, b);
    if (c >= mx) { return mn; }
    if (c <= mn) { return mx; }
    return a + b - c;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let ts = params.tile_size;
    let tiles_x = (params.width + ts - 1u) / ts;
    let ox = (wg.x % tiles_x) * ts;
    let oy = (wg.x / tiles_x) * ts;
    let ndiag = 2u * ts - 1u;

    for (var d = 0u; d < ndiag; d = d + 1u) {
        var lo = 0u;
        if (d >= ts) {
            lo = d - ts + 1u;
        }
        let hi = min(d, ts - 1u);
        var k = lid.x;
        loop {
            if (k > hi - lo) { break; }
            let lx = lo + k;
            let ly = d - lx;
            let gx = ox + lx;
            let gy = oy + ly;
            if (gx < params.width && gy < params.height) {
                let idx = gy * params.width + gx;
                var p = 0.0;
                if (lx == 0u && ly == 0u) {
                    p = 0.0;
                } else if (ly == 0u) {
                    p = dst[idx - 1u];
                } else if (lx == 0u) {
                    p = dst[idx - params.width];
                } else {
                    // Neighbours come from dst — written by earlier diagonals of this workgroup.
                    p = med(dst[idx - 1u], dst[idx - params.width], dst[idx - params.width - 1u]);
                }
                dst[idx] = src[idx] + p;
            }
            k = k + 256u;
        }
        // storageBarrier(), not workgroupBarrier(): the dependency runs through a storage buffer.
        storageBarrier();
    }
}
