// GPU weight map normalization: variance -> raw weights -> normalize -> smooth -> re-normalize.
//
// Single-workgroup (256 threads) shader replacing the CPU compute_weight_map().
// Dispatched as (1, 1, 1) — one workgroup handles all blocks (up to 8K resolution).
//
// 5 phases with barriers:
//   1. Map variance -> raw weights via clamp(1 + scale*(8 - log2(1+v)), 0.5, 2.0)
//   2. Reduce sum of log(w) for geometric mean
//   3. Normalize by geometric mean, clamp
//   4. 3x3 box filter smooth -> weight_map output
//   5. Re-normalize by geometric mean (bits ∝ -log₂(step), so geo-mean=1 ≈ iso-bitrate)

struct Params {
    blocks_x: u32,
    blocks_y: u32,
    total_blocks: u32,
    aq_strength: f32,
}

const MIN_WEIGHT: f32 = 0.5;
const MAX_WEIGHT: f32 = 2.0;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> variance: array<f32>;
@group(0) @binding(2) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(3) var<storage, read_write> weight_map: array<f32>;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
    let scale = 0.15 * params.aq_strength;
    let blocks_per_thread = (params.total_blocks + 255u) / 256u;
    let start = lid * blocks_per_thread;
    let end = min(start + blocks_per_thread, params.total_blocks);

    // Phase 1: Map variance -> raw weights
    for (var i = start; i < end; i++) {
        let log_var = log2(1.0 + variance[i]);
        let w = 1.0 + scale * (8.0 - log_var);
        scratch[i] = clamp(w, MIN_WEIGHT, MAX_WEIGHT);
    }
    storageBarrier();

    // Phase 2: Reduce sum of log(w) for geometric mean
    var local_log_sum = 0.0;
    for (var i = start; i < end; i++) {
        local_log_sum += log(max(scratch[i], 0.001));
    }
    shared_sum[lid] = local_log_sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        workgroupBarrier();
    }

    let log_mean = shared_sum[0] / f32(params.total_blocks);
    let geo_factor = exp(-log_mean);

    // Phase 3: Normalize by geometric mean, clamp
    for (var i = start; i < end; i++) {
        scratch[i] = clamp(scratch[i] * geo_factor, MIN_WEIGHT, MAX_WEIGHT);
    }
    storageBarrier();

    // Phase 4: 3x3 box filter smooth -> weight_map
    if params.blocks_x >= 3u && params.blocks_y >= 3u {
        for (var i = start; i < end; i++) {
            let bx = i % params.blocks_x;
            let by = i / params.blocks_x;

            var s = 0.0;
            var count = 0.0;
            for (var dy = -1i; dy <= 1i; dy++) {
                for (var dx = -1i; dx <= 1i; dx++) {
                    let nx = i32(bx) + dx;
                    let ny = i32(by) + dy;
                    if nx >= 0 && nx < i32(params.blocks_x) && ny >= 0 && ny < i32(params.blocks_y) {
                        s += scratch[u32(ny) * params.blocks_x + u32(nx)];
                        count += 1.0;
                    }
                }
            }
            weight_map[i] = s / count;
        }
    } else {
        for (var i = start; i < end; i++) {
            weight_map[i] = scratch[i];
        }
    }
    storageBarrier();

    // Phase 5: Re-normalize after smoothing using geometric mean.
    // Geometric mean = 1.0 ensures E[log(w)] = 0, which preserves total
    // bitrate since bits ∝ -log2(step) and step ∝ weight.
    var local_log_sum2 = 0.0;
    for (var i = start; i < end; i++) {
        local_log_sum2 += log(max(weight_map[i], 0.001));
    }
    shared_sum[lid] = local_log_sum2;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if lid < stride {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        workgroupBarrier();
    }

    let log_mean2 = shared_sum[0] / f32(params.total_blocks);
    let geo_factor2 = exp(-log_mean2);
    for (var i = start; i < end; i++) {
        weight_map[i] *= geo_factor2;
    }
}
