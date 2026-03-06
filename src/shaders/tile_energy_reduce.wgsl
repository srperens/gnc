// Per-tile energy reduction for temporal highpass adaptive quantization.
//
// For each tile in the Y-plane coefficient buffer, computes:
//   mean_abs = sum(|coeff|) / tile_pixel_count
// then maps this energy to a qstep multiplier via log-space interpolation:
//   energy <= low_thresh  → max_mul (aggressive quantization, near-static tile)
//   energy >= high_thresh → MIN_MUL (1.2, preserve detail; always coarser than lowpass)
//   between              → log-linear interpolation
//
// Also maintains a global max_abs across all pixels via atomicMax on a u32
// buffer (works for positive IEEE 754 floats: bit patterns preserve order).
//
// Dispatch: dispatch_workgroups(tiles_x, tiles_y, 1)
// One workgroup per tile. Workgroup size 256 threads.
// Each thread handles tile_pixels/256 = 65536/256 = 256 pixels (for 256×256 tiles).
//
// Shared memory: 256 f32 (partial abs sums) + 256 f32 (partial max) = 2KB << 32KB limit.

struct Params {
    padded_w:   u32,
    padded_h:   u32,
    tile_size:  u32,
    // tiles_x is derived from padded_w / tile_size in the shader
    low_thresh: f32,
    high_thresh: f32,
    max_mul:    f32,
    _pad:       u32,
}

@group(0) @binding(0) var<uniform>            params:           Params;
@group(0) @binding(1) var<storage, read>      y_plane:          array<f32>;
@group(0) @binding(2) var<storage, read_write> tile_muls:       array<f32>;
@group(0) @binding(3) var<storage, read_write> global_max_bits: array<atomic<u32>>;
/// Raw mean_abs energy per tile (pre-mapping). Used by CPU to identify tiles whose
/// highpass is too energetic to benefit from temporal wavelet coding.
@group(0) @binding(4) var<storage, read_write> tile_energies:  array<f32>;

var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_max: array<f32, 256>;

// Log-space interpolation between max_mul and MIN_MUL.
// Mirrors EncoderPipeline::map_energy_to_mul on the CPU side.
// MIN_MUL ensures temporal highpass is always at least slightly coarser
// than lowpass quantisation, even for high-motion tiles.
const MIN_MUL: f32 = 1.2;
fn map_energy_to_mul(energy: f32) -> f32 {
    if energy <= params.low_thresh {
        return params.max_mul;
    }
    if energy >= params.high_thresh {
        return MIN_MUL;
    }
    // log-linear: t = log(energy/low) / log(high/low), result = max_mul + t*(MIN_MUL-max_mul)
    let t = log(energy / params.low_thresh) / log(params.high_thresh / params.low_thresh);
    return params.max_mul + t * (MIN_MUL - params.max_mul);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id)          wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid:  u32,
) {
    let tiles_x = params.padded_w / params.tile_size;
    let tiles_y = params.padded_h / params.tile_size;
    let tile_x  = wg_id.x;
    let tile_y  = wg_id.y;

    // Guard: skip out-of-bounds workgroups (shouldn't happen with correct dispatch)
    if tile_x >= tiles_x || tile_y >= tiles_y {
        shared_sum[lid] = 0.0;
        shared_max[lid] = 0.0;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            workgroupBarrier();
        }
        return;
    }

    let tile_pixels = params.tile_size * params.tile_size;
    // Each thread processes tile_pixels / 256 pixels (exact for tile_size = 256)
    let pixels_per_thread = (tile_pixels + 255u) / 256u;

    let tile_origin_x = tile_x * params.tile_size;
    let tile_origin_y = tile_y * params.tile_size;

    var local_sum: f32 = 0.0;
    var local_max: f32 = 0.0;

    for (var i = 0u; i < pixels_per_thread; i = i + 1u) {
        let pixel_idx = lid * pixels_per_thread + i;
        if pixel_idx < tile_pixels {
            let local_y = pixel_idx / params.tile_size;
            let local_x = pixel_idx % params.tile_size;
            let px = tile_origin_x + local_x;
            let py = tile_origin_y + local_y;
            let abs_val = abs(y_plane[py * params.padded_w + px]);
            local_sum = local_sum + abs_val;
            local_max = max(local_max, abs_val);
        }
    }

    // Store partial results in shared memory
    shared_sum[lid] = local_sum;
    shared_max[lid] = local_max;
    workgroupBarrier();

    // Parallel reduction: sum and max over 256 threads
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            shared_sum[lid] = shared_sum[lid] + shared_sum[lid + stride];
            shared_max[lid] = max(shared_max[lid], shared_max[lid + stride]);
        }
        workgroupBarrier();
    }

    // Thread 0: write tile multiplier, raw energy, and update global max
    if lid == 0u {
        let tile_idx = tile_y * tiles_x + tile_x;
        let mean_abs = shared_sum[0] / f32(tile_pixels);
        tile_muls[tile_idx]    = map_energy_to_mul(mean_abs);
        tile_energies[tile_idx] = mean_abs;

        // Update global max using atomic on u32 reinterpretation.
        // IEEE 754 positive floats preserve order under integer comparison.
        let tile_max_bits = bitcast<u32>(shared_max[0]);
        atomicMax(&global_max_bits[0], tile_max_bits);
    }
}
