// CfL (Chroma-from-Luma) forward prediction shader (encoder side).
//
// Computes the residual: output[i] = chroma[i] - alpha[tile][subband] * luma_ref[i]
// Mirror of cfl_predict.wgsl (decoder) which does addition instead of subtraction.

struct Params {
    total_count: u32,
    width: u32,
    height: u32,
    tile_size: u32,
    num_levels: u32,
    num_subbands: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> chroma: array<f32>;
@group(0) @binding(2) var<storage, read> luma_ref: array<f32>;
@group(0) @binding(3) var<storage, read> alphas: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Subband index computation — mirrors quantize.wgsl / cfl_predict.wgsl.
fn compute_subband_index(lx: u32, ly: u32) -> u32 {
    var region = params.tile_size;
    for (var level = 0u; level < params.num_levels; level = level + 1u) {
        let half = region / 2u;
        let in_right = lx >= half;
        let in_bottom = ly >= half;
        if in_right || in_bottom {
            if in_right && in_bottom {
                return 1u + level * 3u + 2u; // HH
            } else if in_right {
                return 1u + level * 3u + 1u; // HL
            } else {
                return 1u + level * 3u;      // LH
            }
        }
        region = half;
    }
    return 0u; // LL
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.total_count {
        return;
    }

    // 2D position in the plane
    let y = idx / params.width;
    let x = idx % params.width;

    // Tile index
    let tiles_x = (params.width + params.tile_size - 1u) / params.tile_size;
    let tx = x / params.tile_size;
    let ty = y / params.tile_size;
    let tile_idx = ty * tiles_x + tx;

    // Tile-local position and subband
    let lx = x % params.tile_size;
    let ly = y % params.tile_size;
    let sb = compute_subband_index(lx, ly);

    // Alpha lookup
    let alpha = alphas[tile_idx * params.num_subbands + sb];

    // Forward prediction: residual = chroma - alpha * luma
    output[idx] = chroma[idx] - alpha * luma_ref[idx];
}
