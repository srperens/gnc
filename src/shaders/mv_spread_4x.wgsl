// Spread pyramid MVs to full-resolution predictor buffer.
//
// Each 16×16 block at 4× downscaled resolution corresponds to a 4×4 grid of 16×16 blocks
// at full resolution.  This shader scales the pyramid MV by 4 (coordinate transform) and
// writes it into all 16 corresponding full-resolution predictor entries.
//
// Coordinate mapping:
//   pyramid block (px, py) → full-res blocks (4*px .. 4*px+3,  4*py .. 4*py+3)
//   MV scale: full_mv_qp = pyramid_mv_qp × 4
//             (1 pyramid pixel = 4 full-res pixels = 16 quarter-pel units;
//              pyramid MV in qp units → full-res qp units via ×4)
//
// Dispatch: (pyr_blocks_x, pyr_blocks_y, 1)
// One thread per pyramid block — writes 16 full-resolution entries.

struct Params {
    pyr_blocks_x:  u32,  // blocks in pyramid X direction
    pyr_blocks_y:  u32,  // blocks in pyramid Y direction
    full_blocks_x: u32,  // blocks in full-res X direction
    full_blocks_y: u32,  // blocks in full-res Y direction
}

@group(0) @binding(0) var<uniform>             params:       Params;
@group(0) @binding(1) var<storage, read>       pyramid_mvs:  array<i32>;  // qp units
@group(0) @binding(2) var<storage, read_write> full_pred:    array<i32>;  // qp units

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if px >= params.pyr_blocks_x || py >= params.pyr_blocks_y {
        return;
    }

    // Read pyramid MV (quarter-pel units at pyramid resolution).
    let pyr_idx = py * params.pyr_blocks_x + px;
    let mv_dx = pyramid_mvs[pyr_idx * 2u    ] * 4;  // scale to full-res qp
    let mv_dy = pyramid_mvs[pyr_idx * 2u + 1u] * 4;

    // Write to the 4×4 grid of full-resolution blocks.
    for (var dy = 0u; dy < 4u; dy++) {
        for (var dx = 0u; dx < 4u; dx++) {
            let fx = px * 4u + dx;
            let fy = py * 4u + dy;
            if fx < params.full_blocks_x && fy < params.full_blocks_y {
                let full_idx = fy * params.full_blocks_x + fx;
                full_pred[full_idx * 2u    ] = mv_dx;
                full_pred[full_idx * 2u + 1u] = mv_dy;
            }
        }
    }
}
