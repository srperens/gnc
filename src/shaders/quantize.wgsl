// Subband-aware scalar quantization / dequantization with optional
// spatial adaptive weighting.
//
// Each thread processes one coefficient. The effective step size is:
//   base_step * subband_weight * spatial_weight
//
// The subband_weight comes from the wavelet decomposition level/orientation.
// The spatial_weight comes from a per-block variance-derived weight map
// (when adaptive quantization is enabled; otherwise spatial_weight = 1.0).

struct Params {
    total_count: u32,
    step_size: f32,
    // 0 = forward (quantize), 1 = inverse (dequantize)
    direction: u32,
    // Dead-zone width as fraction of step_size (0.0 for uniform, 0.5 typical)
    dead_zone: f32,
    // Plane dimensions (padded to tile boundaries)
    width: u32,
    height: u32,
    tile_size: u32,
    num_levels: u32,
    // Packed subband weights: [LL, L0_LH, L0_HL, L0_HH, L1_LH, ...]
    weights0: vec4<f32>,
    weights1: vec4<f32>,
    weights2: vec4<f32>,
    weights3: vec4<f32>,
    // Adaptive quantization params
    // 0 = disabled, 1 = enabled
    aq_enabled: u32,
    // Block size for the spatial weight map (e.g. 32)
    aq_block_size: u32,
    // Number of weight-map blocks in x direction
    aq_blocks_x: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
// Spatial weight map: one f32 per block. Only read when aq_enabled == 1.
@group(0) @binding(3) var<storage, read> weight_map: array<f32>;

// Look up a packed weight by flat index (0..15).
fn get_weight(index: u32) -> f32 {
    let vec_idx = index / 4u;
    let comp_idx = index % 4u;
    var v: vec4<f32>;
    switch vec_idx {
        case 0u: { v = params.weights0; }
        case 1u: { v = params.weights1; }
        case 2u: { v = params.weights2; }
        default: { v = params.weights3; }
    }
    return v[comp_idx];
}

// Determine the flat weight index for a coefficient at tile-local position (lx, ly).
// Walks from the outermost decomposition level inward:
//   Level 0 (outermost): region = tile_size, half = tile_size/2
//     Top-right quadrant -> HL (index 1+0*3+1=2)
//     Bottom-left quadrant -> LH (index 1+0*3+0=1)
//     Bottom-right quadrant -> HH (index 1+0*3+2=3)
//   Coefficients in the top-left (LL) quadrant continue to the next level.
//   If all levels exhausted -> LL (index 0).
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

// Look up the spatial weight for a given pixel position.
// Returns 1.0 when adaptive quantization is disabled.
fn get_spatial_weight(x: u32, y: u32) -> f32 {
    if params.aq_enabled == 0u {
        return 1.0;
    }
    let bx = x / params.aq_block_size;
    let by = y / params.aq_block_size;
    let block_idx = by * params.aq_blocks_x + bx;
    return weight_map[block_idx];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.total_count {
        return;
    }

    // Compute 2D position in the plane and tile-local coordinates
    let y = idx / params.width;
    let x = idx % params.width;
    let lx = x % params.tile_size;
    let ly = y % params.tile_size;

    let subband_weight = get_weight(compute_subband_index(lx, ly));
    let spatial_weight = get_spatial_weight(x, y);
    let effective_step = params.step_size * subband_weight * spatial_weight;

    let val = input[idx];

    if params.direction == 0u {
        // Forward: quantize with per-subband + spatial step
        let abs_val = abs(val);
        let sign_val = sign(val);
        let threshold = params.dead_zone * effective_step;

        if abs_val < threshold {
            output[idx] = 0.0;
        } else {
            output[idx] = sign_val * floor(abs_val / effective_step + 0.5);
        }
    } else {
        // Inverse: dequantize with per-subband + spatial step
        output[idx] = val * effective_step;
    }
}
