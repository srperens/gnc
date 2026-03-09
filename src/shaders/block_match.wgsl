// Hierarchical block matching motion estimation shader.
// One workgroup (256 threads) per 16×16 block.
//
// Three-phase coarse-to-fine search:
//   Phase 1: Coarse search over full ±search_range using subsampled 4×4 SAD
//            (every 4th pixel in each dimension = 16 samples per candidate).
//            14.8x fewer memory reads than full-resolution search.
//   Phase 2: Fine search ±4 around coarse winner with full 16×16 SAD.
//   Phase 3: Quarter-pel refinement in two stages:
//            Stage A: 8-point diamond at ±2 QP units (= ±0.5 pixel, half-pel positions)
//            Stage B: 8-point diamond at ±1 QP unit  (= ±0.25 pixel, quarter-pel positions)
//
// Output MVs are in quarter-pel units (value 4 = 1 pixel).
// Parallel min-reduction across workgroup to find best MV in each phase.

struct Params {
    width: u32,
    height: u32,
    block_size: u32,
    search_range: u32,
    blocks_x: u32,
    total_blocks: u32,
    // Non-zero: skip coarse search, use predictor_mvs as starting point
    use_predictor: u32,
    // Fine search range when using predictor (in pixels)
    pred_fine_range: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> current_y: array<f32>;
@group(0) @binding(2) var<storage, read> reference_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> motion_vectors: array<i32>;
@group(0) @binding(4) var<storage, read_write> sad_values: array<u32>;
@group(0) @binding(5) var<storage, read> predictor_mvs: array<i32>;

var<workgroup> shared_sad: array<u32, 256>;
var<workgroup> shared_mv: array<u32, 256>;
// Tracking best half-pel candidate across iterations (parallel half-pel)
var<workgroup> hp_track_sad: u32;
var<workgroup> hp_track_mv: u32;

// Subsample stride for coarse search phase
const COARSE_STRIDE: u32 = 4u;
// Fine search range (±FINE_RANGE pixels around coarse winner)
const FINE_RANGE: i32 = 4;

// Pack (dx, dy) as (dx+32768) << 16 | (dy+32768)
fn pack_mv(dx: i32, dy: i32) -> u32 {
    return (u32(dx + 32768) << 16u) | u32(dy + 32768);
}

fn unpack_dx(packed: u32) -> i32 {
    return i32(packed >> 16u) - 32768;
}

fn unpack_dy(packed: u32) -> i32 {
    return i32(packed & 0xFFFFu) - 32768;
}

// Parallel min-reduction over shared_sad/shared_mv (256 entries, log2 = 8 steps)
fn min_reduce(tid: u32) {
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            if shared_sad[tid + stride] < shared_sad[tid] {
                shared_sad[tid] = shared_sad[tid + stride];
                shared_mv[tid] = shared_mv[tid + stride];
            }
        }
        workgroupBarrier();
    }
}

// Bilinear sample at quarter-pel position (qx, qy in quarter-pel units).
// Integer part = qx >> 2, fractional (0..3) = qx & 3, weight = frac * 0.25.
fn bilinear_sample(qx: i32, qy: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);

    let ix = qx >> 2;
    let iy = qy >> 2;
    let fx = qx & 3;
    let fy = qy & 3;

    if fx == 0 && fy == 0 {
        let cx = clamp(ix, 0, w - 1);
        let cy = clamp(iy, 0, h - 1);
        return reference_y[u32(cy) * params.width + u32(cx)];
    }

    let x0 = clamp(ix, 0, w - 1);
    let x1 = clamp(ix + 1, 0, w - 1);
    let y0 = clamp(iy, 0, h - 1);
    let y1 = clamp(iy + 1, 0, h - 1);

    let p00 = reference_y[u32(y0) * params.width + u32(x0)];
    let p10 = reference_y[u32(y0) * params.width + u32(x1)];
    let p01 = reference_y[u32(y1) * params.width + u32(x0)];
    let p11 = reference_y[u32(y1) * params.width + u32(x1)];

    let ffx = f32(fx) * 0.25;
    let ffy = f32(fy) * 0.25;

    let top = p00 * (1.0 - ffx) + p10 * ffx;
    let bot = p01 * (1.0 - ffx) + p11 * ffx;
    return top * (1.0 - ffy) + bot * ffy;
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let block_idx = group_id.x;
    if block_idx >= params.total_blocks {
        return;
    }

    let tid = local_id.x;
    let by = block_idx / params.blocks_x;
    let bx = block_idx % params.blocks_x;
    let block_origin_x = bx * params.block_size;
    let block_origin_y = by * params.block_size;

    let sr = i32(params.search_range);
    let search_side = 2u * params.search_range + 1u;
    let total_candidates = search_side * search_side;
    let bs = i32(params.block_size);
    let w = i32(params.width);
    let h = i32(params.height);

    // ========== Phase 1: Coarse search or temporal predictor ==========
    var coarse_dx: i32 = 0;
    var coarse_dy: i32 = 0;
    var fine_range: i32 = FINE_RANGE;

    // Phase 1 uses either temporal prediction or full coarse search.
    // Both paths write to shared_sad/shared_mv and need barriers, so we
    // keep all barriers at the top level for uniform control flow (WebGPU/Metal).
    if params.use_predictor != 0u {
        // Temporal MV prediction: use previous frame's MV as starting point.
        // Predictor MVs are in quarter-pel units — convert to integer-pel.
        if tid == 0u {
            let pred_hx = predictor_mvs[block_idx * 2u];
            let pred_hy = predictor_mvs[block_idx * 2u + 1u];
            // Round quarter-pel to nearest integer-pel
            coarse_dx = (pred_hx + select(0, 2, pred_hx > 0)) / 4;
            coarse_dy = (pred_hy + select(0, 2, pred_hy > 0)) / 4;
            shared_mv[0] = pack_mv(coarse_dx, coarse_dy);
        }
        // Dummy writes for the else-path shared arrays (compiler may optimize away)
        shared_sad[tid] = 0u;
    } else {
        // Full coarse search with subsampled SAD
        // Every COARSE_STRIDE-th pixel in each dimension: 4×4 = 16 samples from 16×16 block.
        var best_sad: u32 = 0xFFFFFFFFu;
        var best_dx: i32 = 0;
        var best_dy: i32 = 0;

        var cand_idx = tid;
        loop {
            if cand_idx >= total_candidates {
                break;
            }

            let cand_dy = i32(cand_idx / search_side) - sr;
            let cand_dx = i32(cand_idx % search_side) - sr;
            let ref_x = i32(block_origin_x) + cand_dx;
            let ref_y = i32(block_origin_y) + cand_dy;

            if ref_x >= 0 && ref_y >= 0 &&
               ref_x + bs <= w && ref_y + bs <= h {
                var sad: u32 = 0u;
                for (var py = 0u; py < params.block_size; py += COARSE_STRIDE) {
                    for (var px = 0u; px < params.block_size; px += COARSE_STRIDE) {
                        let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                        let r_idx = u32(ref_y + i32(py)) * params.width + u32(ref_x + i32(px));
                        sad += u32(abs(current_y[cur_idx] - reference_y[r_idx]));
                    }
                }

                if sad < best_sad {
                    best_sad = sad;
                    best_dx = cand_dx;
                    best_dy = cand_dy;
                }
            }

            cand_idx += 256u;
        }

        shared_sad[tid] = best_sad;
        shared_mv[tid] = pack_mv(best_dx, best_dy);
    }
    // Barriers unconditionally at top level (WebGPU/Metal require uniform control flow)
    workgroupBarrier();
    min_reduce(tid);
    // For predictor path: shared_sad is all zeros so min_reduce is a no-op,
    // shared_mv[0] keeps the predictor value set by thread 0.

    let coarse_result = shared_mv[0];
    coarse_dx = unpack_dx(coarse_result);
    coarse_dy = unpack_dy(coarse_result);
    if params.use_predictor != 0u {
        fine_range = i32(params.pred_fine_range);
    }
    workgroupBarrier();

    // ========== Phase 2: Fine search with full 16×16 SAD ==========
    // ±fine_range around coarse winner (or predicted MV).
    let fine_side = u32(2 * fine_range + 1);
    let fine_total = fine_side * fine_side;

    var fine_best_sad: u32 = 0xFFFFFFFFu;
    var fine_best_dx: i32 = coarse_dx;
    var fine_best_dy: i32 = coarse_dy;

    var fine_idx = tid;
    loop {
        if fine_idx >= fine_total {
            break;
        }

        let dy_off = i32(fine_idx / fine_side) - fine_range;
        let dx_off = i32(fine_idx % fine_side) - fine_range;
        let test_dx = coarse_dx + dx_off;
        let test_dy = coarse_dy + dy_off;
        let ref_x = i32(block_origin_x) + test_dx;
        let ref_y = i32(block_origin_y) + test_dy;

        if ref_x >= 0 && ref_y >= 0 &&
           ref_x + bs <= w && ref_y + bs <= h {
            var sad: u32 = 0u;
            for (var py = 0u; py < params.block_size; py++) {
                for (var px = 0u; px < params.block_size; px++) {
                    let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                    let r_idx = u32(ref_y + i32(py)) * params.width + u32(ref_x + i32(px));
                    sad += u32(abs(current_y[cur_idx] - reference_y[r_idx]));
                }
            }
            if sad < fine_best_sad {
                fine_best_sad = sad;
                fine_best_dx = test_dx;
                fine_best_dy = test_dy;
            }
        }

        fine_idx += 256u;
    }

    // Min-reduction for fine winner
    shared_sad[tid] = fine_best_sad;
    shared_mv[tid] = pack_mv(fine_best_dx, fine_best_dy);
    workgroupBarrier();
    min_reduce(tid);

    // ========== Phase 2b: Zero-MV competition (skip mode) ==========
    // Evaluate (0,0) at full precision and compare against the fine winner.
    // The coarse search uses subsampled SAD (16 samples) which can pick wrong
    // winners for low-texture/static content. Re-evaluate both candidates with
    // full 16×16 SAD at 8× fixed-point precision to properly distinguish small
    // differences, with a small bias toward zero-MV (lower MV coding cost).
    let fine_packed_pre = shared_mv[0];
    let fine_dx_pre = unpack_dx(fine_packed_pre);
    let fine_dy_pre = unpack_dy(fine_packed_pre);
    workgroupBarrier();

    let need_zero_check = (fine_dx_pre != 0 || fine_dy_pre != 0);
    let zpx = tid % params.block_size;
    let zpy = tid / params.block_size;
    let zcur_idx = (block_origin_y + zpy) * params.width + block_origin_x + zpx;
    let zcur_val = current_y[zcur_idx];

    // Zero-MV SAD at 8× precision
    shared_sad[tid] = select(0u, u32(abs(zcur_val - reference_y[zcur_idx]) * 8.0), need_zero_check);
    workgroupBarrier();
    for (var zs = 128u; zs > 0u; zs >>= 1u) {
        if tid < zs { shared_sad[tid] += shared_sad[tid + zs]; }
        workgroupBarrier();
    }
    let zero_sad_8x = shared_sad[0];

    // Fine winner SAD at same 8× precision
    let fref_x = clamp(i32(block_origin_x + zpx) + fine_dx_pre, 0, w - 1);
    let fref_y = clamp(i32(block_origin_y + zpy) + fine_dy_pre, 0, h - 1);
    shared_sad[tid] = select(0u, u32(abs(zcur_val - reference_y[u32(fref_y) * params.width + u32(fref_x)]) * 8.0), need_zero_check);
    workgroupBarrier();
    for (var zs = 128u; zs > 0u; zs >>= 1u) {
        if tid < zs { shared_sad[tid] += shared_sad[tid + zs]; }
        workgroupBarrier();
    }
    let fine_sad_8x = shared_sad[0];

    // Prefer zero-MV unless fine winner is meaningfully better.
    // Bias of ~0.25 per pixel (2 at 8× scale) prevents random MVs in
    // low-texture regions while allowing real motion (which typically has
    // SAD deltas of 10+ per pixel).
    let zero_bias = params.block_size * params.block_size * 2u;
    if tid == 0u && need_zero_check && zero_sad_8x <= fine_sad_8x + zero_bias {
        shared_sad[0] = zero_sad_8x / 8u;
        shared_mv[0] = pack_mv(0, 0);
    } else if tid == 0u {
        shared_sad[0] = fine_sad_8x / 8u;
        shared_mv[0] = fine_packed_pre;
    }
    workgroupBarrier();

    // ========== Phase 3: Quarter-pel refinement (two stages, all 256 threads parallel) ==========
    // Stage A: 8-point diamond at ±2 QP units (= ±0.5 pixel, half-pel positions).
    // Stage B: 8-point diamond at ±1 QP unit  (= ±0.25 pixel, quarter-pel positions).
    // All threads participate unconditionally to keep workgroupBarrier() in uniform
    // control flow (required by WebGPU/Metal). Skipped when fine SAD is zero.
    let fine_packed = shared_mv[0];
    let fine_sad = shared_sad[0];
    let int_dx = unpack_dx(fine_packed);
    let int_dy = unpack_dy(fine_packed);
    workgroupBarrier();

    // Integer MV scaled to quarter-pel units (value 4 = 1 pixel).
    let center_qx = int_dx * 4;
    let center_qy = int_dy * 4;

    if tid == 0u {
        hp_track_sad = fine_sad;
        hp_track_mv = pack_mv(center_qx, center_qy);
    }
    workgroupBarrier();

    let do_subpel = fine_sad > 0u;
    let qp_px = tid % params.block_size;
    let qp_py = tid / params.block_size;
    let qp_cur_val = select(0.0, current_y[(block_origin_y + qp_py) * params.width + block_origin_x + qp_px], do_subpel);

    // Stage A: half-pel search (step = 2 QP units, 8-point diamond).
    for (var cand = 0u; cand < 8u; cand++) {
        var off_x: i32 = 0;
        var off_y: i32 = 0;
        switch cand {
            case 0u: { off_x =  0; off_y = -2; }  // N
            case 1u: { off_x = -2; off_y =  0; }  // W
            case 2u: { off_x =  2; off_y =  0; }  // E
            case 3u: { off_x =  0; off_y =  2; }  // S
            case 4u: { off_x = -2; off_y = -2; }  // NW
            case 5u: { off_x =  2; off_y = -2; }  // NE
            case 6u: { off_x = -2; off_y =  2; }  // SW
            case 7u: { off_x =  2; off_y =  2; }  // SE
            default: {}
        }

        let test_qx = center_qx + off_x;
        let test_qy = center_qy + off_y;

        let ref_base_qx = i32(block_origin_x) * 4 + test_qx;
        let ref_base_qy = i32(block_origin_y) * 4 + test_qy;
        let valid = do_subpel && ref_base_qx >= -3 && ref_base_qy >= -3 &&
                    ref_base_qx + bs * 4 <= w * 4 + 3 &&
                    ref_base_qy + bs * 4 <= h * 4 + 3;

        var pixel_diff: u32 = 0u;
        if valid {
            let rqx = i32(block_origin_x + qp_px) * 4 + test_qx;
            let rqy = i32(block_origin_y + qp_py) * 4 + test_qy;
            let ref_val = bilinear_sample(rqx, rqy);
            pixel_diff = u32(abs(qp_cur_val - ref_val));
        }

        shared_sad[tid] = pixel_diff;
        workgroupBarrier();

        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            if tid < stride {
                shared_sad[tid] += shared_sad[tid + stride];
            }
            workgroupBarrier();
        }

        if tid == 0u && valid {
            if shared_sad[0] < hp_track_sad {
                hp_track_sad = shared_sad[0];
                hp_track_mv = pack_mv(test_qx, test_qy);
            }
        }
        workgroupBarrier();
    }

    // Stage B: quarter-pel search (step = 1 QP unit) around Stage A winner.
    let stageA_qx = unpack_dx(hp_track_mv);
    let stageA_qy = unpack_dy(hp_track_mv);

    for (var cand = 0u; cand < 8u; cand++) {
        var off_x: i32 = 0;
        var off_y: i32 = 0;
        switch cand {
            case 0u: { off_x =  0; off_y = -1; }  // N
            case 1u: { off_x = -1; off_y =  0; }  // W
            case 2u: { off_x =  1; off_y =  0; }  // E
            case 3u: { off_x =  0; off_y =  1; }  // S
            case 4u: { off_x = -1; off_y = -1; }  // NW
            case 5u: { off_x =  1; off_y = -1; }  // NE
            case 6u: { off_x = -1; off_y =  1; }  // SW
            case 7u: { off_x =  1; off_y =  1; }  // SE
            default: {}
        }

        let test_qx = stageA_qx + off_x;
        let test_qy = stageA_qy + off_y;

        let ref_base_qx = i32(block_origin_x) * 4 + test_qx;
        let ref_base_qy = i32(block_origin_y) * 4 + test_qy;
        let valid = do_subpel && ref_base_qx >= -3 && ref_base_qy >= -3 &&
                    ref_base_qx + bs * 4 <= w * 4 + 3 &&
                    ref_base_qy + bs * 4 <= h * 4 + 3;

        var pixel_diff: u32 = 0u;
        if valid {
            let rqx = i32(block_origin_x + qp_px) * 4 + test_qx;
            let rqy = i32(block_origin_y + qp_py) * 4 + test_qy;
            let ref_val = bilinear_sample(rqx, rqy);
            pixel_diff = u32(abs(qp_cur_val - ref_val));
        }

        shared_sad[tid] = pixel_diff;
        workgroupBarrier();

        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            if tid < stride {
                shared_sad[tid] += shared_sad[tid + stride];
            }
            workgroupBarrier();
        }

        if tid == 0u && valid {
            if shared_sad[0] < hp_track_sad {
                hp_track_sad = shared_sad[0];
                hp_track_mv = pack_mv(test_qx, test_qy);
            }
        }
        workgroupBarrier();
    }

    // Thread 0 writes the final quarter-pel MV result.
    if tid == 0u {
        let final_dx = unpack_dx(hp_track_mv);
        let final_dy = unpack_dy(hp_track_mv);
        motion_vectors[block_idx * 2u] = final_dx;
        motion_vectors[block_idx * 2u + 1u] = final_dy;
        sad_values[block_idx] = hp_track_sad;
    }
}
