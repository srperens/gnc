// Subband-aware scalar quantization / dequantization with optional
// wavelet-domain adaptive weighting.
//
// Each thread processes one coefficient. The effective step size is:
//   base_step * subband_weight * spatial_weight
//
// The subband_weight comes from the wavelet decomposition level/orientation.
// The spatial_weight comes from a per-LL-block variance-derived weight map
// (when adaptive quantization is enabled; otherwise spatial_weight = 1.0).
//
// The weight map is computed on the LL (lowpass) subband after the wavelet
// transform. Each entry corresponds to a block of LL coefficients, which
// maps to a spatial region. The quantizer maps each wavelet coefficient
// (regardless of subband) to the correct LL-block weight by converting
// tile-local wavelet coordinates to a global 2D block position.
//
// Weight map layout: global 2D row-major order, where
// global_bx = tile_x * ll_blocks_per_tile_x + local_bx
// global_by = tile_y * ll_blocks_per_tile_y + local_by
// index = global_by * (ll_blocks_per_tile_x * tiles_x) + global_bx

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
    // LL-block size (in LL-subband coordinates)
    aq_ll_block_size: u32,
    // Number of LL-blocks per tile in x direction
    aq_ll_blocks_per_tile_x: u32,
    // Number of tiles in x direction
    aq_tiles_x: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
// Weight map: one f32 per LL-block, in global 2D row-major order.
// Only read when aq_enabled == 1.
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

// Map a wavelet-domain tile-local coordinate (lx, ly) to the corresponding
// LL-block (bx, by) coordinates within the tile.
//
// For the LL subband (top-left ll_size x ll_size), the mapping is direct:
// (lx / ll_block_size, ly / ll_block_size).
//
// For detail subbands at level k, each coefficient represents a spatial
// region. We map it to the LL-block covering the same spatial region by
// scaling the subband-local position: ll_coord = sub_local * ll_size / half.
fn get_ll_block_xy(lx: u32, ly: u32) -> vec2<u32> {
    let ll_size = params.tile_size >> params.num_levels;
    let max_b = params.aq_ll_blocks_per_tile_x - 1u;

    // Walk the subband hierarchy
    var region = params.tile_size;
    for (var level = 0u; level < params.num_levels; level = level + 1u) {
        let half = region / 2u;
        let in_right = lx >= half;
        let in_bottom = ly >= half;
        if in_right || in_bottom {
            // Detail subband at this level.
            var sub_x = lx;
            var sub_y = ly;
            if in_right { sub_x = lx - half; }
            if in_bottom { sub_y = ly - half; }

            // Scale to LL coordinates: ll_coord = sub * ll_size / half
            let ll_x = (sub_x * ll_size) / half;
            let ll_y = (sub_y * ll_size) / half;
            let bx = min(ll_x / params.aq_ll_block_size, max_b);
            let by = min(ll_y / params.aq_ll_block_size, max_b);
            return vec2<u32>(bx, by);
        }
        region = half;
    }
    // LL subband: direct mapping
    let bx = min(lx / params.aq_ll_block_size, max_b);
    let by = min(ly / params.aq_ll_block_size, max_b);
    return vec2<u32>(bx, by);
}

// Look up the spatial weight for a given wavelet-domain position.
// Returns 1.0 when adaptive quantization is disabled.
fn get_spatial_weight(x: u32, y: u32) -> f32 {
    if params.aq_enabled == 0u {
        return 1.0;
    }
    // Tile coordinates
    let tile_x = x / params.tile_size;
    let tile_y = y / params.tile_size;

    // Tile-local wavelet coordinates
    let lx = x % params.tile_size;
    let ly = y % params.tile_size;

    // Map to LL-block (bx, by) within this tile
    let lb = get_ll_block_xy(lx, ly);

    // Global 2D block coordinates
    let global_bx = tile_x * params.aq_ll_blocks_per_tile_x + lb.x;
    let global_by = tile_y * params.aq_ll_blocks_per_tile_x + lb.y;
    let global_blocks_x = params.aq_ll_blocks_per_tile_x * params.aq_tiles_x;
    let block_idx = global_by * global_blocks_x + global_bx;
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
