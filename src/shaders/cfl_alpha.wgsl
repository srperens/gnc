// CfL (Chroma-from-Luma) alpha computation shader.
//
// One workgroup (256 threads) per tile. Each thread accumulates local
// sum_yc and sum_yy per subband, then a sequential per-subband parallel
// reduction computes alpha = sum_yc / sum_yy.
//
// Outputs both raw alphas (for CPU u8 serialization readback) and
// dequantized alphas (for GPU forward prediction — stays on device).

const MAX_SUBBANDS: u32 = 16u; // 1 + 3*5 for up to 5 wavelet levels

struct Params {
    tile_pixels: u32,   // tile_size * tile_size
    tile_size: u32,
    num_levels: u32,
    num_subbands: u32,
    width: u32,         // padded plane width
    height: u32,        // padded plane height
    tiles_x: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> recon_y: array<f32>;
@group(0) @binding(2) var<storage, read> chroma: array<f32>;
@group(0) @binding(3) var<storage, read_write> raw_alphas: array<f32>;
@group(0) @binding(4) var<storage, read_write> dq_alphas: array<f32>;

var<workgroup> shared_yc: array<f32, 256>;
var<workgroup> shared_yy: array<f32, 256>;

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
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let tile_idx = wg_id.x;
    let tx = tile_idx % params.tiles_x;
    let ty = tile_idx / params.tiles_x;
    let tile_origin_x = tx * params.tile_size;
    let tile_origin_y = ty * params.tile_size;

    // Each thread accumulates partial sums for all subbands
    var local_yc: array<f32, MAX_SUBBANDS>;
    var local_yy: array<f32, MAX_SUBBANDS>;
    for (var s = 0u; s < MAX_SUBBANDS; s++) {
        local_yc[s] = 0.0;
        local_yy[s] = 0.0;
    }

    let pixels_per_thread = params.tile_pixels / 256u;
    for (var i = 0u; i < pixels_per_thread; i++) {
        let pixel_idx = lid * pixels_per_thread + i;
        let ly = pixel_idx / params.tile_size;
        let lx = pixel_idx % params.tile_size;

        let gx = tile_origin_x + lx;
        let gy = tile_origin_y + ly;
        let global_idx = gy * params.width + gx;

        let y_val = recon_y[global_idx];
        let c_val = chroma[global_idx];
        let sb = compute_subband_index(lx, ly);

        local_yc[sb] += y_val * c_val;
        local_yy[sb] += y_val * y_val;
    }

    // Reduce per subband: sequential over subbands, parallel over threads
    let alpha_base = tile_idx * params.num_subbands;

    for (var s = 0u; s < params.num_subbands; s++) {
        shared_yc[lid] = local_yc[s];
        shared_yy[lid] = local_yy[s];
        workgroupBarrier();

        // Parallel reduction
        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            if lid < stride {
                shared_yc[lid] += shared_yc[lid + stride];
                shared_yy[lid] += shared_yy[lid + stride];
            }
            workgroupBarrier();
        }

        // Thread 0: compute alpha, quantize+dequantize, store both outputs
        if lid == 0u {
            var alpha = 0.0;
            if shared_yy[0] > 1e-10 {
                alpha = shared_yc[0] / shared_yy[0];
            }
            raw_alphas[alpha_base + s] = alpha;

            // Quantize + dequantize (matches Rust quantize_alpha/dequantize_alpha)
            let clamped = clamp(alpha, -2.0, 2.0);
            let normalized = (clamped + 2.0) / 4.0;
            let q = u32(round(normalized * 255.0));
            let dq = (f32(q) / 255.0) * 4.0 - 2.0;
            dq_alphas[alpha_base + s] = dq;
        }
        workgroupBarrier();
    }
}
