// Bidirectional hierarchical block matching motion estimation shader.
// One workgroup (256 threads) per 16x16 block.
//
// Uses coarse-to-fine search for both forward and backward references:
//   Phase 1:  Coarse forward search with subsampled 4×4 SAD, then fine ±4 refinement.
//   Phase 2:  Coarse backward search with subsampled 4×4 SAD, then fine ±4 refinement.
//   Phase 3a: Parallel bidir SAD — all 256 threads, 1 pixel each, sum-reduced.
//   Phase 3b: Thread 0 picks best mode (fwd/bwd/bidir), broadcasts via shared memory.
//   Phase 3c: 8 threads do forward half-pel refinement (modes 0,2).
//   Phase 3d: 8 threads do backward half-pel refinement (modes 1,2; uses refined fwd MV).
//   Phase 3e: Thread 0 writes results.
//
// Note: WGSL does not allow passing storage pointers as function parameters,
// so reference accesses are inlined where needed.

struct Params {
    width: u32,
    height: u32,
    block_size: u32,
    search_range: u32,
    blocks_x: u32,
    total_blocks: u32,
    use_predictor: u32,
    pred_fine_range: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> current_y: array<f32>;
@group(0) @binding(2) var<storage, read> reference_fwd_y: array<f32>;
@group(0) @binding(3) var<storage, read> reference_bwd_y: array<f32>;
@group(0) @binding(4) var<storage, read_write> fwd_motion_vectors: array<i32>;
@group(0) @binding(5) var<storage, read_write> bwd_motion_vectors: array<i32>;
@group(0) @binding(6) var<storage, read_write> block_modes: array<u32>;
@group(0) @binding(7) var<storage, read_write> sad_values: array<u32>;
@group(0) @binding(8) var<storage, read> predictor_fwd_mvs: array<i32>;
@group(0) @binding(9) var<storage, read> predictor_bwd_mvs: array<i32>;

var<workgroup> shared_sad: array<u32, 256>;
var<workgroup> shared_mv: array<u32, 256>;
// Tracking best half-pel candidate across iterations (parallel half-pel)
var<workgroup> hp_track_sad: u32;
var<workgroup> hp_track_mv: u32;

// Subsample stride for coarse search
const COARSE_STRIDE: u32 = 4u;
// Fine search range (±FINE_RANGE pixels around coarse winner)
const FINE_RANGE: i32 = 4;

fn pack_mv(dx: i32, dy: i32) -> u32 {
    return (u32(dx + 32768) << 16u) | u32(dy + 32768);
}

fn unpack_dx(packed: u32) -> i32 {
    return i32(packed >> 16u) - 32768;
}

fn unpack_dy(packed: u32) -> i32 {
    return i32(packed & 0xFFFFu) - 32768;
}

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

// Quarter-pel sample functions (x4, y4 in quarter-pel units; integer = >> 2, frac = & 3).
fn sample_hp_fwd(x4: i32, y4: i32, w: u32, h: u32) -> f32 {
    let fx = x4 >> 2;
    let fy = y4 >> 2;
    let frac_x = x4 & 3;
    let frac_y = y4 & 3;
    let x0 = clamp(fx, 0, i32(w) - 1);
    let y0 = clamp(fy, 0, i32(h) - 1);
    let x1 = clamp(fx + 1, 0, i32(w) - 1);
    let y1 = clamp(fy + 1, 0, i32(h) - 1);
    let p00 = reference_fwd_y[u32(y0) * w + u32(x0)];
    let p10 = reference_fwd_y[u32(y0) * w + u32(x1)];
    let p01 = reference_fwd_y[u32(y1) * w + u32(x0)];
    let p11 = reference_fwd_y[u32(y1) * w + u32(x1)];
    if frac_x == 0 && frac_y == 0 {
        return p00;
    }
    let ffx = f32(frac_x) * 0.25;
    let ffy = f32(frac_y) * 0.25;
    let top = p00 * (1.0 - ffx) + p10 * ffx;
    let bot = p01 * (1.0 - ffx) + p11 * ffx;
    return top * (1.0 - ffy) + bot * ffy;
}

fn sample_hp_bwd(x4: i32, y4: i32, w: u32, h: u32) -> f32 {
    let fx = x4 >> 2;
    let fy = y4 >> 2;
    let frac_x = x4 & 3;
    let frac_y = y4 & 3;
    let x0 = clamp(fx, 0, i32(w) - 1);
    let y0 = clamp(fy, 0, i32(h) - 1);
    let x1 = clamp(fx + 1, 0, i32(w) - 1);
    let y1 = clamp(fy + 1, 0, i32(h) - 1);
    let p00 = reference_bwd_y[u32(y0) * w + u32(x0)];
    let p10 = reference_bwd_y[u32(y0) * w + u32(x1)];
    let p01 = reference_bwd_y[u32(y1) * w + u32(x0)];
    let p11 = reference_bwd_y[u32(y1) * w + u32(x1)];
    if frac_x == 0 && frac_y == 0 {
        return p00;
    }
    let ffx = f32(frac_x) * 0.25;
    let ffy = f32(frac_y) * 0.25;
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

    // ========== Phase 1: Forward search (coarse or predicted) ==========
    var coarse_fwd_dx: i32 = 0;
    var coarse_fwd_dy: i32 = 0;
    var fine_range: i32 = FINE_RANGE;

    if params.use_predictor != 0u {
        // Temporal MV prediction: use previous B-frame's forward MV as starting point.
        // Predictor MVs are in quarter-pel units — convert to integer-pel.
        if tid == 0u {
            let pred_hx = predictor_fwd_mvs[block_idx * 2u];
            let pred_hy = predictor_fwd_mvs[block_idx * 2u + 1u];
            coarse_fwd_dx = (pred_hx + select(0, 2, pred_hx > 0)) / 4;
            coarse_fwd_dy = (pred_hy + select(0, 2, pred_hy > 0)) / 4;
        }
        shared_mv[0] = pack_mv(coarse_fwd_dx, coarse_fwd_dy);
        workgroupBarrier();
        let pred_packed = shared_mv[0];
        coarse_fwd_dx = unpack_dx(pred_packed);
        coarse_fwd_dy = unpack_dy(pred_packed);
        workgroupBarrier();
        fine_range = i32(params.pred_fine_range);
    } else {
        // Full coarse search with subsampled SAD
        var best_sad: u32 = 0xFFFFFFFFu;
        var best_dx: i32 = 0;
        var best_dy: i32 = 0;

        var cand_idx = tid;
        loop {
            if cand_idx >= total_candidates { break; }
            let cand_dy = i32(cand_idx / search_side) - sr;
            let cand_dx = i32(cand_idx % search_side) - sr;
            let ref_x = i32(block_origin_x) + cand_dx;
            let ref_y = i32(block_origin_y) + cand_dy;

            if ref_x >= 0 && ref_y >= 0 && ref_x + bs <= w && ref_y + bs <= h {
                var sad: u32 = 0u;
                for (var py = 0u; py < params.block_size; py += COARSE_STRIDE) {
                    for (var px = 0u; px < params.block_size; px += COARSE_STRIDE) {
                        let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                        let r_idx = u32(ref_y + i32(py)) * params.width + u32(ref_x + i32(px));
                        sad += u32(abs(current_y[cur_idx] - reference_fwd_y[r_idx]));
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
        workgroupBarrier();
        min_reduce(tid);

        coarse_fwd_dx = unpack_dx(shared_mv[0]);
        coarse_fwd_dy = unpack_dy(shared_mv[0]);
        workgroupBarrier();
    }

    // ========== Phase 1b: Fine forward search (full SAD) ==========
    let fine_side = u32(2 * fine_range + 1);
    let fine_total = fine_side * fine_side;

    var fine_best_sad: u32 = 0xFFFFFFFFu;
    var fine_best_dx: i32 = coarse_fwd_dx;
    var fine_best_dy: i32 = coarse_fwd_dy;

    var fine_idx = tid;
    loop {
        if fine_idx >= fine_total { break; }
        let dy_off = i32(fine_idx / fine_side) - fine_range;
        let dx_off = i32(fine_idx % fine_side) - fine_range;
        let test_dx = coarse_fwd_dx + dx_off;
        let test_dy = coarse_fwd_dy + dy_off;
        let ref_x = i32(block_origin_x) + test_dx;
        let ref_y = i32(block_origin_y) + test_dy;

        if ref_x >= 0 && ref_y >= 0 && ref_x + bs <= w && ref_y + bs <= h {
            var sad: u32 = 0u;
            for (var py = 0u; py < params.block_size; py++) {
                for (var px = 0u; px < params.block_size; px++) {
                    let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                    let r_idx = u32(ref_y + i32(py)) * params.width + u32(ref_x + i32(px));
                    sad += u32(abs(current_y[cur_idx] - reference_fwd_y[r_idx]));
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

    shared_sad[tid] = fine_best_sad;
    shared_mv[tid] = pack_mv(fine_best_dx, fine_best_dy);
    workgroupBarrier();
    min_reduce(tid);

    var final_fwd_dx: i32;
    var final_fwd_dy: i32;
    var final_fwd_sad: u32;
    let fwd_packed = shared_mv[0];
    final_fwd_dx = unpack_dx(fwd_packed);
    final_fwd_dy = unpack_dy(fwd_packed);
    final_fwd_sad = shared_sad[0];
    workgroupBarrier();

    // ========== Phase 2: Backward search (coarse or predicted) ==========
    var coarse_bwd_dx: i32 = 0;
    var coarse_bwd_dy: i32 = 0;

    if params.use_predictor != 0u {
        // Temporal MV prediction: use previous B-frame's backward MV as starting point.
        if tid == 0u {
            let pred_hx = predictor_bwd_mvs[block_idx * 2u];
            let pred_hy = predictor_bwd_mvs[block_idx * 2u + 1u];
            coarse_bwd_dx = (pred_hx + select(0, 2, pred_hx > 0)) / 4;
            coarse_bwd_dy = (pred_hy + select(0, 2, pred_hy > 0)) / 4;
        }
        shared_mv[0] = pack_mv(coarse_bwd_dx, coarse_bwd_dy);
        workgroupBarrier();
        let pred_packed = shared_mv[0];
        coarse_bwd_dx = unpack_dx(pred_packed);
        coarse_bwd_dy = unpack_dy(pred_packed);
        workgroupBarrier();
    } else {
        var best_sad: u32 = 0xFFFFFFFFu;
        var best_dx: i32 = 0;
        var best_dy: i32 = 0;

        var cand_idx = tid;
        loop {
            if cand_idx >= total_candidates { break; }
            let cand_dy = i32(cand_idx / search_side) - sr;
            let cand_dx = i32(cand_idx % search_side) - sr;
            let ref_x = i32(block_origin_x) + cand_dx;
            let ref_y = i32(block_origin_y) + cand_dy;

            if ref_x >= 0 && ref_y >= 0 && ref_x + bs <= w && ref_y + bs <= h {
                var sad: u32 = 0u;
                for (var py = 0u; py < params.block_size; py += COARSE_STRIDE) {
                    for (var px = 0u; px < params.block_size; px += COARSE_STRIDE) {
                        let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                        let r_idx = u32(ref_y + i32(py)) * params.width + u32(ref_x + i32(px));
                        sad += u32(abs(current_y[cur_idx] - reference_bwd_y[r_idx]));
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
        workgroupBarrier();
        min_reduce(tid);

        coarse_bwd_dx = unpack_dx(shared_mv[0]);
        coarse_bwd_dy = unpack_dy(shared_mv[0]);
        workgroupBarrier();
    }

    // ========== Phase 2b: Fine backward search (full SAD) ==========
    fine_best_sad = 0xFFFFFFFFu;
    fine_best_dx = coarse_bwd_dx;
    fine_best_dy = coarse_bwd_dy;

    fine_idx = tid;
    loop {
        if fine_idx >= fine_total { break; }
        let dy_off = i32(fine_idx / fine_side) - fine_range;
        let dx_off = i32(fine_idx % fine_side) - fine_range;
        let test_dx = coarse_bwd_dx + dx_off;
        let test_dy = coarse_bwd_dy + dy_off;
        let ref_x = i32(block_origin_x) + test_dx;
        let ref_y = i32(block_origin_y) + test_dy;

        if ref_x >= 0 && ref_y >= 0 && ref_x + bs <= w && ref_y + bs <= h {
            var sad: u32 = 0u;
            for (var py = 0u; py < params.block_size; py++) {
                for (var px = 0u; px < params.block_size; px++) {
                    let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                    let r_idx = u32(ref_y + i32(py)) * params.width + u32(ref_x + i32(px));
                    sad += u32(abs(current_y[cur_idx] - reference_bwd_y[r_idx]));
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

    shared_sad[tid] = fine_best_sad;
    shared_mv[tid] = pack_mv(fine_best_dx, fine_best_dy);
    workgroupBarrier();
    min_reduce(tid);

    var final_bwd_dx: i32;
    var final_bwd_dy: i32;
    var final_bwd_sad: u32;
    let bwd_packed = shared_mv[0];
    final_bwd_dx = unpack_dx(bwd_packed);
    final_bwd_dy = unpack_dy(bwd_packed);
    final_bwd_sad = shared_sad[0];
    workgroupBarrier();

    // ========== Phase 3a: Parallel bidir SAD computation (all 256 threads) ==========
    // 16x16 = 256 pixels; thread tid handles pixel (tid % 16, tid / 16).
    // This replaces the old serial loop on thread 0 (256 iterations -> 1 per thread).
    {
        let px = tid % params.block_size;
        let py = tid / params.block_size;

        var pixel_diff: u32 = 0u;
        if px < params.block_size && py < params.block_size {
            let fwd_ref_x = i32(block_origin_x) + final_fwd_dx;
            let fwd_ref_y = i32(block_origin_y) + final_fwd_dy;
            let bwd_ref_x = i32(block_origin_x) + final_bwd_dx;
            let bwd_ref_y = i32(block_origin_y) + final_bwd_dy;

            // Both references must be fully in-bounds for bidir to be valid
            if fwd_ref_x >= 0 && fwd_ref_y >= 0 &&
               fwd_ref_x + bs <= w && fwd_ref_y + bs <= h &&
               bwd_ref_x >= 0 && bwd_ref_y >= 0 &&
               bwd_ref_x + bs <= w && bwd_ref_y + bs <= h {
                let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                let f_idx = u32(fwd_ref_y + i32(py)) * params.width + u32(fwd_ref_x + i32(px));
                let b_idx = u32(bwd_ref_y + i32(py)) * params.width + u32(bwd_ref_x + i32(px));
                let pred = (reference_fwd_y[f_idx] + reference_bwd_y[b_idx]) * 0.5;
                let diff = current_y[cur_idx] - pred;
                pixel_diff = u32(abs(diff));
            } else {
                // Out of bounds: poison with large per-pixel value so bidir loses
                pixel_diff = 0xFFFFu;
            }
        }

        shared_sad[tid] = pixel_diff;
        workgroupBarrier();

        // Sum reduction across 256 threads -> shared_sad[0] = bidir_sad
        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            if tid < stride {
                shared_sad[tid] += shared_sad[tid + stride];
            }
            workgroupBarrier();
        }
    }

    let bidir_sad = shared_sad[0];
    workgroupBarrier();

    // ========== Phase 3b: Mode selection (thread 0), broadcast to all ==========
    // Layout: shared_mv[0]=mode, shared_mv[1]=packed fwd MV, shared_mv[2]=packed bwd MV
    //         shared_sad[0]=best_total_sad
    if tid == 0u {
        var best_mode: u32 = 0u;
        var best_total_sad = final_fwd_sad;
        if final_bwd_sad < best_total_sad {
            best_total_sad = final_bwd_sad;
            best_mode = 1u;
        }
        if bidir_sad < best_total_sad {
            best_total_sad = bidir_sad;
            best_mode = 2u;
        }

        shared_mv[0] = best_mode;
        shared_mv[1] = pack_mv(final_fwd_dx, final_fwd_dy);
        shared_mv[2] = pack_mv(final_bwd_dx, final_bwd_dy);
        shared_sad[0] = best_total_sad;
    }
    workgroupBarrier();

    let best_mode = shared_mv[0];
    let mode_fwd_dx = unpack_dx(shared_mv[1]);
    let mode_fwd_dy = unpack_dy(shared_mv[1]);
    let mode_bwd_dx = unpack_dx(shared_mv[2]);
    let mode_bwd_dy = unpack_dy(shared_mv[2]);
    let mode_sad = shared_sad[0];
    workgroupBarrier();

    // Integer MVs scaled to quarter-pel units (value 4 = 1 pixel).
    let center_fwd_qx = mode_fwd_dx * 4;
    let center_fwd_qy = mode_fwd_dy * 4;
    let center_bwd_qx = mode_bwd_dx * 4;
    let center_bwd_qy = mode_bwd_dy * 4;

    // ========== Phase 3c: Forward quarter-pel refinement (two stages, all 256 threads) ==========
    // Stage A: half-pel positions (step ±2 QP units). Stage B: quarter-pel (step ±1).
    // Active for mode 0 (fwd-only) and mode 2 (bidir). Skipped if SAD is zero.

    var hp_fwd_dx = center_fwd_qx;
    var hp_fwd_dy = center_fwd_qy;
    var hp_fwd_sad = mode_sad;

    let do_fwd_hp = (best_mode == 0u || best_mode == 2u) && mode_sad > 0u;
    if tid == 0u {
        hp_track_sad = mode_sad;
        hp_track_mv = pack_mv(center_fwd_qx, center_fwd_qy);
    }
    workgroupBarrier();

    let hp_px = tid % params.block_size;
    let hp_py = tid / params.block_size;
    let hp_cur_val = select(0.0, current_y[(block_origin_y + hp_py) * params.width + block_origin_x + hp_px], do_fwd_hp);

    // For bidir, precompute backward sample at integer bwd position (constant across fwd candidates).
    var hp_bwd_sample: f32 = 0.0;
    if do_fwd_hp && best_mode == 2u {
        let bqx = i32(block_origin_x + hp_px) * 4 + center_bwd_qx;
        let bqy = i32(block_origin_y + hp_py) * 4 + center_bwd_qy;
        hp_bwd_sample = sample_hp_bwd(bqx, bqy, params.width, params.height);
    }

    // Stage A: half-pel (step 2) for forward.
    for (var cand = 0u; cand < 8u; cand++) {
        var off_x: i32 = 0;
        var off_y: i32 = 0;
        switch cand {
            case 0u: { off_x =  0; off_y = -2; }
            case 1u: { off_x = -2; off_y =  0; }
            case 2u: { off_x =  2; off_y =  0; }
            case 3u: { off_x =  0; off_y =  2; }
            case 4u: { off_x = -2; off_y = -2; }
            case 5u: { off_x =  2; off_y = -2; }
            case 6u: { off_x = -2; off_y =  2; }
            case 7u: { off_x =  2; off_y =  2; }
            default: {}
        }

        let test_qx = center_fwd_qx + off_x;
        let test_qy = center_fwd_qy + off_y;

        let ref_base_qx = i32(block_origin_x) * 4 + test_qx;
        let ref_base_qy = i32(block_origin_y) * 4 + test_qy;
        let valid = do_fwd_hp && ref_base_qx >= -3 && ref_base_qy >= -3 &&
                    ref_base_qx + bs * 4 <= w * 4 + 3 &&
                    ref_base_qy + bs * 4 <= h * 4 + 3;

        var pixel_diff: u32 = 0u;
        if valid {
            let rqx = i32(block_origin_x + hp_px) * 4 + test_qx;
            let rqy = i32(block_origin_y + hp_py) * 4 + test_qy;
            let fwd_val = sample_hp_fwd(rqx, rqy, params.width, params.height);
            var pred: f32;
            if best_mode == 0u { pred = fwd_val; } else { pred = (fwd_val + hp_bwd_sample) * 0.5; }
            pixel_diff = u32(abs(hp_cur_val - pred));
        }

        shared_sad[tid] = pixel_diff;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            if tid < stride { shared_sad[tid] += shared_sad[tid + stride]; }
            workgroupBarrier();
        }
        if tid == 0u && valid {
            if shared_sad[0] < hp_track_sad { hp_track_sad = shared_sad[0]; hp_track_mv = pack_mv(test_qx, test_qy); }
        }
        workgroupBarrier();
    }

    // Stage B: quarter-pel (step 1) for forward.
    let stageA_fwd_qx = unpack_dx(hp_track_mv);
    let stageA_fwd_qy = unpack_dy(hp_track_mv);

    for (var cand = 0u; cand < 8u; cand++) {
        var off_x: i32 = 0;
        var off_y: i32 = 0;
        switch cand {
            case 0u: { off_x =  0; off_y = -1; }
            case 1u: { off_x = -1; off_y =  0; }
            case 2u: { off_x =  1; off_y =  0; }
            case 3u: { off_x =  0; off_y =  1; }
            case 4u: { off_x = -1; off_y = -1; }
            case 5u: { off_x =  1; off_y = -1; }
            case 6u: { off_x = -1; off_y =  1; }
            case 7u: { off_x =  1; off_y =  1; }
            default: {}
        }

        let test_qx = stageA_fwd_qx + off_x;
        let test_qy = stageA_fwd_qy + off_y;

        let ref_base_qx = i32(block_origin_x) * 4 + test_qx;
        let ref_base_qy = i32(block_origin_y) * 4 + test_qy;
        let valid = do_fwd_hp && ref_base_qx >= -3 && ref_base_qy >= -3 &&
                    ref_base_qx + bs * 4 <= w * 4 + 3 &&
                    ref_base_qy + bs * 4 <= h * 4 + 3;

        var pixel_diff: u32 = 0u;
        if valid {
            let rqx = i32(block_origin_x + hp_px) * 4 + test_qx;
            let rqy = i32(block_origin_y + hp_py) * 4 + test_qy;
            let fwd_val = sample_hp_fwd(rqx, rqy, params.width, params.height);
            var pred: f32;
            if best_mode == 0u { pred = fwd_val; } else { pred = (fwd_val + hp_bwd_sample) * 0.5; }
            pixel_diff = u32(abs(hp_cur_val - pred));
        }

        shared_sad[tid] = pixel_diff;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            if tid < stride { shared_sad[tid] += shared_sad[tid + stride]; }
            workgroupBarrier();
        }
        if tid == 0u && valid {
            if shared_sad[0] < hp_track_sad { hp_track_sad = shared_sad[0]; hp_track_mv = pack_mv(test_qx, test_qy); }
        }
        workgroupBarrier();
    }

    if do_fwd_hp {
        hp_fwd_dx = unpack_dx(hp_track_mv);
        hp_fwd_dy = unpack_dy(hp_track_mv);
        hp_fwd_sad = hp_track_sad;
    }
    workgroupBarrier();

    // ========== Phase 3d: Backward quarter-pel refinement (two stages, all 256 threads) ==========
    // Stage A: half-pel (step ±2). Stage B: quarter-pel (step ±1).
    // Active for mode 1 (bwd-only) and mode 2 (bidir).

    var hp_bwd_dx = center_bwd_qx;
    var hp_bwd_dy = center_bwd_qy;

    let do_bwd_hp = (best_mode == 1u || best_mode == 2u) && mode_sad > 0u;

    // Broadcast refined forward QP MV to all threads.
    if tid == 0u {
        shared_mv[0] = pack_mv(hp_fwd_dx, hp_fwd_dy);
    }
    workgroupBarrier();
    let refined_fwd_qx = unpack_dx(shared_mv[0]);
    let refined_fwd_qy = unpack_dy(shared_mv[0]);

    var center_sad_for_bwd: u32 = mode_sad;
    if best_mode == 2u {
        center_sad_for_bwd = hp_fwd_sad;
    }

    if tid == 0u {
        hp_track_sad = center_sad_for_bwd;
        hp_track_mv = pack_mv(center_bwd_qx, center_bwd_qy);
    }
    workgroupBarrier();

    let hp2_px = tid % params.block_size;
    let hp2_py = tid / params.block_size;
    let hp2_cur_val = select(0.0, current_y[(block_origin_y + hp2_py) * params.width + block_origin_x + hp2_px], do_bwd_hp);

    // For bidir, precompute refined forward sample (constant across bwd candidates).
    var hp_fwd_sample: f32 = 0.0;
    if do_bwd_hp && best_mode == 2u {
        let fqx = i32(block_origin_x + hp2_px) * 4 + refined_fwd_qx;
        let fqy = i32(block_origin_y + hp2_py) * 4 + refined_fwd_qy;
        hp_fwd_sample = sample_hp_fwd(fqx, fqy, params.width, params.height);
    }

    // Stage A: half-pel (step 2) for backward.
    for (var cand = 0u; cand < 8u; cand++) {
        var off_x: i32 = 0;
        var off_y: i32 = 0;
        switch cand {
            case 0u: { off_x =  0; off_y = -2; }
            case 1u: { off_x = -2; off_y =  0; }
            case 2u: { off_x =  2; off_y =  0; }
            case 3u: { off_x =  0; off_y =  2; }
            case 4u: { off_x = -2; off_y = -2; }
            case 5u: { off_x =  2; off_y = -2; }
            case 6u: { off_x = -2; off_y =  2; }
            case 7u: { off_x =  2; off_y =  2; }
            default: {}
        }

        let test_qx = center_bwd_qx + off_x;
        let test_qy = center_bwd_qy + off_y;

        let ref_base_qx = i32(block_origin_x) * 4 + test_qx;
        let ref_base_qy = i32(block_origin_y) * 4 + test_qy;
        let valid = do_bwd_hp && ref_base_qx >= -3 && ref_base_qy >= -3 &&
                    ref_base_qx + bs * 4 <= w * 4 + 3 &&
                    ref_base_qy + bs * 4 <= h * 4 + 3;

        var pixel_diff: u32 = 0u;
        if valid {
            let rqx = i32(block_origin_x + hp2_px) * 4 + test_qx;
            let rqy = i32(block_origin_y + hp2_py) * 4 + test_qy;
            let bwd_val = sample_hp_bwd(rqx, rqy, params.width, params.height);
            var pred: f32;
            if best_mode == 1u { pred = bwd_val; } else { pred = (hp_fwd_sample + bwd_val) * 0.5; }
            pixel_diff = u32(abs(hp2_cur_val - pred));
        }

        shared_sad[tid] = pixel_diff;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            if tid < stride { shared_sad[tid] += shared_sad[tid + stride]; }
            workgroupBarrier();
        }
        if tid == 0u && valid {
            if shared_sad[0] < hp_track_sad { hp_track_sad = shared_sad[0]; hp_track_mv = pack_mv(test_qx, test_qy); }
        }
        workgroupBarrier();
    }

    // Stage B: quarter-pel (step 1) for backward.
    let stageA_bwd_qx = unpack_dx(hp_track_mv);
    let stageA_bwd_qy = unpack_dy(hp_track_mv);

    for (var cand = 0u; cand < 8u; cand++) {
        var off_x: i32 = 0;
        var off_y: i32 = 0;
        switch cand {
            case 0u: { off_x =  0; off_y = -1; }
            case 1u: { off_x = -1; off_y =  0; }
            case 2u: { off_x =  1; off_y =  0; }
            case 3u: { off_x =  0; off_y =  1; }
            case 4u: { off_x = -1; off_y = -1; }
            case 5u: { off_x =  1; off_y = -1; }
            case 6u: { off_x = -1; off_y =  1; }
            case 7u: { off_x =  1; off_y =  1; }
            default: {}
        }

        let test_qx = stageA_bwd_qx + off_x;
        let test_qy = stageA_bwd_qy + off_y;

        let ref_base_qx = i32(block_origin_x) * 4 + test_qx;
        let ref_base_qy = i32(block_origin_y) * 4 + test_qy;
        let valid = do_bwd_hp && ref_base_qx >= -3 && ref_base_qy >= -3 &&
                    ref_base_qx + bs * 4 <= w * 4 + 3 &&
                    ref_base_qy + bs * 4 <= h * 4 + 3;

        var pixel_diff: u32 = 0u;
        if valid {
            let rqx = i32(block_origin_x + hp2_px) * 4 + test_qx;
            let rqy = i32(block_origin_y + hp2_py) * 4 + test_qy;
            let bwd_val = sample_hp_bwd(rqx, rqy, params.width, params.height);
            var pred: f32;
            if best_mode == 1u { pred = bwd_val; } else { pred = (hp_fwd_sample + bwd_val) * 0.5; }
            pixel_diff = u32(abs(hp2_cur_val - pred));
        }

        shared_sad[tid] = pixel_diff;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            if tid < stride { shared_sad[tid] += shared_sad[tid + stride]; }
            workgroupBarrier();
        }
        if tid == 0u && valid {
            if shared_sad[0] < hp_track_sad { hp_track_sad = shared_sad[0]; hp_track_mv = pack_mv(test_qx, test_qy); }
        }
        workgroupBarrier();
    }

    if do_bwd_hp {
        hp_bwd_dx = unpack_dx(hp_track_mv);
        hp_bwd_dy = unpack_dy(hp_track_mv);
    }
    workgroupBarrier();

    // ========== Phase 3e: Write final results (thread 0) ==========
    if tid == 0u {
        fwd_motion_vectors[block_idx * 2u] = hp_fwd_dx;
        fwd_motion_vectors[block_idx * 2u + 1u] = hp_fwd_dy;
        bwd_motion_vectors[block_idx * 2u] = hp_bwd_dx;
        bwd_motion_vectors[block_idx * 2u + 1u] = hp_bwd_dy;
        block_modes[block_idx] = best_mode;
        sad_values[block_idx] = shared_sad[0];
    }
}
