// Deblock reference frames at tile boundaries (encoder-internal, no bitstream impact).
//
// GNC tiles are 256×256 pixels. Quantization introduces discontinuities at tile
// boundaries. Applying a mild smoothing filter at tile boundaries BEFORE using
// the frame as a motion-estimation reference reduces the reference error and
// therefore lowers residual energy for P/B-frames.
//
// This filter is ENCODER-ONLY. The decoder never sees it.  It is NOT post-processing
// (#36, closed) — it only touches encoder-internal reference buffers.
//
// Filter (4-tap, applied at boundary between rows R-1 and R, or cols C-1 and C):
//   p2 = pixel two steps inside the first tile  (row R-2 or col C-2)
//   p1 = pixel one step inside the first tile   (row R-1 or col C-1)
//   p0 = pixel one step inside the second tile  (row R   or col C)
//   pn1= pixel two steps inside the second tile (row R+1 or col C+1)
//
//   new_p1 = (p2 + 2*p1 + p0) / 4
//   new_p0 = (p1 + 2*p0 + pn1) / 4
//
// Adaptive gate: skip when |p1 - p0| > 2 * qstep.  A large cross-boundary gradient
// is likely a real edge, not a compression artifact; do not smooth it.
//
// Two entry points share the same params struct:
//   - main_h: process one horizontal tile boundary row (boundary between row R-1 and R).
//             Dispatch: (padded_w, n_horiz_boundaries, 1) workgroups of size (1, 1, 1).
//             Thread (x, y, _): x = pixel column, y = boundary index → R = (y+1)*256.
//   - main_v: process one vertical tile boundary column.
//             Dispatch: (n_vert_boundaries, padded_h, 1) workgroups of size (1, 1, 1).
//             Thread (x, y, _): x = boundary index → C = (x+1)*256, y = pixel row.
//
// Both entry points are combined in one dispatch with larger workgroups to keep
// Rust dispatch code simple.  Actually, we use the following layout:
//
//   main_h: dispatch_workgroups(tiles_x, horiz_boundaries, 1)
//           workgroup_size(256, 1, 1)
//           thread i = column within segment of 256 pixels for that wg.
//           wg_id.x = column_tile_idx (0..tiles_x)
//           wg_id.y = boundary_idx   (0..horiz_boundaries)
//           → column c = wg_id.x * 256 + lid
//           → boundary row R = (wg_id.y + 1) * 256
//
//   main_v: dispatch_workgroups(vert_boundaries, tiles_y, 1)
//           workgroup_size(1, 256, 1)
//           thread i = row within segment of 256 pixels for that wg.
//           wg_id.x = boundary_idx  (0..vert_boundaries)
//           wg_id.y = row_tile_idx  (0..tiles_y)
//           → row r = wg_id.y * 256 + lid
//           → boundary col C = (wg_id.x + 1) * 256

struct DeblockParams {
    padded_w:  u32,
    padded_h:  u32,
    tile_size: u32,   // always 256 but parameterized for correctness
    qstep:     f32,   // quantization step; gate threshold = 2 * qstep
}

@group(0) @binding(0) var<uniform>             params: DeblockParams;
@group(0) @binding(1) var<storage, read_write> plane:  array<f32>;

// Inline index helper — row-major, row r, column c.
fn idx(r: u32, c: u32) -> u32 {
    return r * params.padded_w + c;
}

// Horizontal boundary: smooth between row (R-1) and row R.
// Called for each pixel column c.
fn smooth_horiz(c: u32, R: u32) {
    // Guard: need two pixels on each side — rows R-2, R-1, R, R+1 must exist.
    if R < 2u || R + 1u >= params.padded_h {
        return;
    }
    let p2  = plane[idx(R - 2u, c)];
    let p1  = plane[idx(R - 1u, c)];
    let p0  = plane[idx(R,      c)];
    let pn1 = plane[idx(R + 1u, c)];

    // Adaptive gate: skip real edges.
    if abs(p1 - p0) > 2.0 * params.qstep {
        return;
    }

    plane[idx(R - 1u, c)] = (p2 + 2.0 * p1 + p0) * 0.25;
    plane[idx(R,      c)] = (p1 + 2.0 * p0 + pn1) * 0.25;
}

// Vertical boundary: smooth between column (C-1) and column C.
// Called for each pixel row r.
fn smooth_vert(r: u32, C: u32) {
    // Guard: need two pixels on each side — cols C-2, C-1, C, C+1 must exist.
    if C < 2u || C + 1u >= params.padded_w {
        return;
    }
    let p2  = plane[idx(r, C - 2u)];
    let p1  = plane[idx(r, C - 1u)];
    let p0  = plane[idx(r, C     )];
    let pn1 = plane[idx(r, C + 1u)];

    // Adaptive gate: skip real edges.
    if abs(p1 - p0) > 2.0 * params.qstep {
        return;
    }

    plane[idx(r, C - 1u)] = (p2 + 2.0 * p1 + p0) * 0.25;
    plane[idx(r, C     )] = (p1 + 2.0 * p0 + pn1) * 0.25;
}

// Horizontal tile boundary pass.
// dispatch_workgroups(tiles_x, horiz_boundaries, 1), workgroup_size(256, 1, 1)
// Each workgroup covers one row of 256 pixels at one horizontal boundary.
@compute @workgroup_size(256, 1, 1)
fn main_h(
    @builtin(workgroup_id)           wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid:   u32,
) {
    let col_tile_idx   = wg_id.x;  // which 256-pixel column tile segment
    let boundary_idx   = wg_id.y;  // 0-based index of horizontal boundary

    let c = col_tile_idx * params.tile_size + lid;
    let R = (boundary_idx + 1u) * params.tile_size;  // boundary row

    if c >= params.padded_w || R >= params.padded_h {
        return;
    }
    smooth_horiz(c, R);
}

// Vertical tile boundary pass.
// dispatch_workgroups(vert_boundaries, tiles_y, 1), workgroup_size(1, 256, 1)
// Each workgroup covers one column of 256 pixels at one vertical boundary.
@compute @workgroup_size(1, 256, 1)
fn main_v(
    @builtin(workgroup_id)           wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid:   u32,
) {
    let boundary_idx = wg_id.x;  // 0-based index of vertical boundary
    let row_tile_idx = wg_id.y;  // which 256-pixel row tile segment

    let C = (boundary_idx + 1u) * params.tile_size;  // boundary column
    let r = row_tile_idx * params.tile_size + lid;

    if C >= params.padded_w || r >= params.padded_h {
        return;
    }
    smooth_vert(r, C);
}
