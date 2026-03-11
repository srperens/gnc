// Variable block-size motion estimation: 16x16 → 8x8 split decision.
// One workgroup (256 threads) per 16x16 macroblock.
//
// Takes 16x16 MVs + SADs from dispatch 1 (block_match.wgsl), runs 4× 8x8
// sub-block refinement (fine search + 8-point half-pel), then makes an RD
// split decision per macroblock.
//
// Thread assignment within workgroup:
//   sub_id    = tid / 64  (0..3 → TL/TR/BL/BR 8x8 sub-block)
//   local_tid = tid % 64  (index within sub-block, maps to pixel)
//
// Output: 8x8-resolution MV buffer (4 entries per macroblock).

struct SplitParams {
    width: u32,
    height: u32,
    blocks_x: u32,       // 16x16 macroblock grid width
    blocks_y: u32,       // 16x16 macroblock grid height
    total_macroblocks: u32,
    lambda_sad: u32,      // RD split penalty (higher = fewer splits)
    use_predictor: u32,   // Non-zero: use 8x8 predictor MVs
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: SplitParams;
@group(0) @binding(1) var<storage, read> current_y: array<f32>;
@group(0) @binding(2) var<storage, read> reference_y: array<f32>;
@group(0) @binding(3) var<storage, read> parent_mvs: array<i32>;     // 16x16 MVs (half-pel)
@group(0) @binding(4) var<storage, read> parent_sads: array<u32>;    // 16x16 SADs
@group(0) @binding(5) var<storage, read> predictor_mvs: array<i32>;  // optional 8x8 predictors
@group(0) @binding(6) var<storage, read_write> output_mvs: array<i32>; // 8x8 resolution output
// Binding 7: 4 sub-block SADs per macroblock for diagnostic readback.
// Always written (cheap: ~32 KB/frame at 1080p). CPU readback only when GNC_BLOCKSIZE_DIAG=1.
@group(0) @binding(7) var<storage, read_write> sub_sad_out: array<u32>;

// Shared memory: 256 entries for SAD/MV parallel operations
var<workgroup> shared_sad: array<u32, 256>;
var<workgroup> shared_mv: array<u32, 256>;
// Per sub-block half-pel tracking (4 sub-blocks)
var<workgroup> hp_track_sad: array<u32, 4>;
var<workgroup> hp_track_mv: array<u32, 4>;
// Sub-block SADs for RD decision
var<workgroup> sub_sads: array<u32, 4>;
// Broadcast: parent MV in integer-pel
var<workgroup> parent_int_dx: i32;
var<workgroup> parent_int_dy: i32;
// Merged-block SADs for 4-way RD decision (computed serially by thread 0)
var<workgroup> merged_sad_hband_top: u32;
var<workgroup> merged_sad_hband_bot: u32;
var<workgroup> merged_sad_vband_left: u32;
var<workgroup> merged_sad_vband_right: u32;
// Best MV index for each merged partition (0 or 1 → which sub-block MV was chosen)
var<workgroup> hband_top_mv_idx: u32;
var<workgroup> hband_bot_mv_idx: u32;
var<workgroup> vband_left_mv_idx: u32;
var<workgroup> vband_right_mv_idx: u32;

// ±2px around parent MV (parent 16×16 is already a good predictor for 8×8).
// Equivalent to the predictor fine-range used in block_match.wgsl.
const FINE_RANGE: i32 = 2;
const SUB_BS: u32 = 8u;
const SUB_PIXELS: u32 = 64u; // 8x8
const ZERO_BIAS_8X8: u32 = 128u; // 8x8 equivalent of 512 for 16x16

fn pack_mv(dx: i32, dy: i32) -> u32 {
    return (u32(dx + 32768) << 16u) | u32(dy + 32768);
}

fn unpack_dx(packed: u32) -> i32 {
    return i32(packed >> 16u) - 32768;
}

fn unpack_dy(packed: u32) -> i32 {
    return i32(packed & 0xFFFFu) - 32768;
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

// 64-thread min-reduction within a sub-block arena [base..base+64)
fn sub_min_reduce(tid: u32, base: u32) {
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            if shared_sad[base + tid + stride] < shared_sad[base + tid] {
                shared_sad[base + tid] = shared_sad[base + tid + stride];
                shared_mv[base + tid] = shared_mv[base + tid + stride];
            }
        }
        workgroupBarrier();
    }
}

// 64-thread sum-reduction within a sub-block arena [base..base+64)
fn sub_sum_reduce(tid: u32, base: u32) {
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            shared_sad[base + tid] += shared_sad[base + tid + stride];
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let mb_idx = group_id.x;
    if mb_idx >= params.total_macroblocks {
        return;
    }

    let tid = local_id.x;
    let sub_id = tid / SUB_PIXELS;   // 0..3: TL, TR, BL, BR
    let local_tid = tid % SUB_PIXELS; // 0..63 within sub-block

    let mb_y = mb_idx / params.blocks_x;
    let mb_x = mb_idx % params.blocks_x;
    let mb_origin_x = mb_x * 16u;
    let mb_origin_y = mb_y * 16u;

    // Sub-block origin within macroblock
    let sub_col = sub_id % 2u;  // 0=left, 1=right
    let sub_row = sub_id / 2u;  // 0=top, 1=bottom
    let sub_origin_x = mb_origin_x + sub_col * SUB_BS;
    let sub_origin_y = mb_origin_y + sub_row * SUB_BS;

    let w = i32(params.width);
    let h = i32(params.height);
    let base = sub_id * SUB_PIXELS; // shared memory base for this sub-block

    // ========== Phase 1: Load parent MV (quarter-pel → integer-pel) ==========
    if tid == 0u {
        let phx = parent_mvs[mb_idx * 2u];
        let phy = parent_mvs[mb_idx * 2u + 1u];
        parent_int_dx = (phx + select(0, 2, phx > 0)) / 4;
        parent_int_dy = (phy + select(0, 2, phy > 0)) / 4;
    }
    workgroupBarrier();
    let center_dx = parent_int_dx;
    let center_dy = parent_int_dy;

    // ========== Phase 2: Fine search per sub-block ==========
    // ±FINE_RANGE around parent MV, full 8x8 SAD per candidate.
    let fine_side = u32(2 * FINE_RANGE + 1);
    let fine_total = fine_side * fine_side; // 81 candidates

    var best_sad: u32 = 0xFFFFFFFFu;
    var best_dx: i32 = center_dx;
    var best_dy: i32 = center_dy;

    // Each of 64 threads handles candidates in round-robin
    var cand_idx = local_tid;
    loop {
        if cand_idx >= fine_total {
            break;
        }

        let dy_off = i32(cand_idx / fine_side) - FINE_RANGE;
        let dx_off = i32(cand_idx % fine_side) - FINE_RANGE;
        let test_dx = center_dx + dx_off;
        let test_dy = center_dy + dy_off;
        let ref_x = i32(sub_origin_x) + test_dx;
        let ref_y = i32(sub_origin_y) + test_dy;

        if ref_x >= 0 && ref_y >= 0 &&
           ref_x + i32(SUB_BS) <= w && ref_y + i32(SUB_BS) <= h {
            var sad: u32 = 0u;
            for (var py = 0u; py < SUB_BS; py++) {
                for (var px = 0u; px < SUB_BS; px++) {
                    let cur_idx = (sub_origin_y + py) * params.width + sub_origin_x + px;
                    let r_idx = u32(ref_y + i32(py)) * params.width + u32(ref_x + i32(px));
                    sad += u32(abs(current_y[cur_idx] - reference_y[r_idx]));
                }
            }
            if sad < best_sad {
                best_sad = sad;
                best_dx = test_dx;
                best_dy = test_dy;
            }
        }

        cand_idx += SUB_PIXELS;
    }

    shared_sad[base + local_tid] = best_sad;
    shared_mv[base + local_tid] = pack_mv(best_dx, best_dy);
    workgroupBarrier();
    sub_min_reduce(local_tid, base);

    let fine_packed = shared_mv[base];
    let fine_dx = unpack_dx(fine_packed);
    let fine_dy = unpack_dy(fine_packed);
    let fine_sad = shared_sad[base];
    workgroupBarrier();

    // ========== Phase 3: Zero-MV competition ==========
    let need_zero = (fine_dx != 0 || fine_dy != 0);
    let px = local_tid % SUB_BS;
    let py = local_tid / SUB_BS;
    let cur_idx = (sub_origin_y + py) * params.width + sub_origin_x + px;
    let cur_val = current_y[cur_idx];

    // Zero SAD
    shared_sad[base + local_tid] = select(0u, u32(abs(cur_val - reference_y[cur_idx]) * 8.0), need_zero);
    workgroupBarrier();
    sub_sum_reduce(local_tid, base);
    let zero_sad_8x = shared_sad[base];

    // Fine winner SAD at 8x precision
    let fref_x = clamp(i32(sub_origin_x + px) + fine_dx, 0, w - 1);
    let fref_y = clamp(i32(sub_origin_y + py) + fine_dy, 0, h - 1);
    shared_sad[base + local_tid] = select(0u, u32(abs(cur_val - reference_y[u32(fref_y) * params.width + u32(fref_x)]) * 8.0), need_zero);
    workgroupBarrier();
    sub_sum_reduce(local_tid, base);
    let fine_sad_8x = shared_sad[base];

    if local_tid == 0u && need_zero && zero_sad_8x <= fine_sad_8x + ZERO_BIAS_8X8 {
        shared_sad[base] = zero_sad_8x / 8u;
        shared_mv[base] = pack_mv(0, 0);
    } else if local_tid == 0u {
        shared_sad[base] = fine_sad_8x / 8u;
        shared_mv[base] = fine_packed;
    }
    workgroupBarrier();

    // ========== Phase 4: Quarter-pel refinement (two stages) ==========
    // Stage A: half-pel (step ±2 QP units). Stage B: quarter-pel (step ±1 QP unit).
    let int_packed = shared_mv[base];
    let int_dx = unpack_dx(int_packed);
    let int_dy = unpack_dy(int_packed);
    let int_sad = shared_sad[base];
    workgroupBarrier();

    // Integer MV to quarter-pel (value 4 = 1 pixel).
    let qp_center_qx = int_dx * 4;
    let qp_center_qy = int_dy * 4;

    if local_tid == 0u {
        hp_track_sad[sub_id] = int_sad;
        hp_track_mv[sub_id] = pack_mv(qp_center_qx, qp_center_qy);
    }
    workgroupBarrier();

    let do_subpel = int_sad > 0u;
    let qp_cur_val = select(0.0, current_y[cur_idx], do_subpel);

    // Stage A: half-pel search (step 2 QP units, 8-point diamond).
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

        let test_qx = qp_center_qx + off_x;
        let test_qy = qp_center_qy + off_y;

        let ref_base_qx = i32(sub_origin_x) * 4 + test_qx;
        let ref_base_qy = i32(sub_origin_y) * 4 + test_qy;
        let valid = do_subpel && ref_base_qx >= -3 && ref_base_qy >= -3 &&
                    ref_base_qx + i32(SUB_BS) * 4 <= w * 4 + 3 &&
                    ref_base_qy + i32(SUB_BS) * 4 <= h * 4 + 3;

        var pixel_diff: u32 = 0u;
        if valid {
            let rqx = i32(sub_origin_x + px) * 4 + test_qx;
            let rqy = i32(sub_origin_y + py) * 4 + test_qy;
            let ref_val = bilinear_sample(rqx, rqy);
            pixel_diff = u32(abs(qp_cur_val - ref_val));
        }

        shared_sad[base + local_tid] = pixel_diff;
        workgroupBarrier();
        sub_sum_reduce(local_tid, base);

        if local_tid == 0u && valid {
            if shared_sad[base] < hp_track_sad[sub_id] {
                hp_track_sad[sub_id] = shared_sad[base];
                hp_track_mv[sub_id] = pack_mv(test_qx, test_qy);
            }
        }
        workgroupBarrier();
    }

    // Stage B: quarter-pel search (step 1 QP unit) around Stage A winner.
    let stageA_qx = unpack_dx(hp_track_mv[sub_id]);
    let stageA_qy = unpack_dy(hp_track_mv[sub_id]);

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

        let test_qx = stageA_qx + off_x;
        let test_qy = stageA_qy + off_y;

        let ref_base_qx = i32(sub_origin_x) * 4 + test_qx;
        let ref_base_qy = i32(sub_origin_y) * 4 + test_qy;
        let valid = do_subpel && ref_base_qx >= -3 && ref_base_qy >= -3 &&
                    ref_base_qx + i32(SUB_BS) * 4 <= w * 4 + 3 &&
                    ref_base_qy + i32(SUB_BS) * 4 <= h * 4 + 3;

        var pixel_diff: u32 = 0u;
        if valid {
            let rqx = i32(sub_origin_x + px) * 4 + test_qx;
            let rqy = i32(sub_origin_y + py) * 4 + test_qy;
            let ref_val = bilinear_sample(rqx, rqy);
            pixel_diff = u32(abs(qp_cur_val - ref_val));
        }

        shared_sad[base + local_tid] = pixel_diff;
        workgroupBarrier();
        sub_sum_reduce(local_tid, base);

        if local_tid == 0u && valid {
            if shared_sad[base] < hp_track_sad[sub_id] {
                hp_track_sad[sub_id] = shared_sad[base];
                hp_track_mv[sub_id] = pack_mv(test_qx, test_qy);
            }
        }
        workgroupBarrier();
    }

    // Store sub-block SAD for RD decision
    if local_tid == 0u {
        sub_sads[sub_id] = hp_track_sad[sub_id];
    }
    workgroupBarrier();

    // ========== Phase 5a: Merged-block SAD computation (thread 0, serial) ==========
    // Computes SADs for HBAND and VBAND partitions over 128-pixel regions.
    // For each merged partition, two sub-block MV candidates are tried and the
    // better one is chosen.  Results written to workgroup vars for Phase 5b.
    //
    // Sub-block layout within macroblock (pixel coordinates relative to mb_origin):
    //   sub[0]=TL  x:0-7,  y:0-7      sub[1]=TR  x:8-15, y:0-7
    //   sub[2]=BL  x:0-7,  y:8-15     sub[3]=BR  x:8-15, y:8-15
    //
    // HBAND-top  (16×8): pixels with y=0..7  (sub 0+1), two MV candidates: mv[0], mv[1]
    // HBAND-bot  (16×8): pixels with y=8..15 (sub 2+3), two MV candidates: mv[2], mv[3]
    // VBAND-left (8×16): pixels with x=0..7  (sub 0+2), two MV candidates: mv[0], mv[2]
    // VBAND-right(8×16): pixels with x=8..15 (sub 1+3), two MV candidates: mv[1], mv[3]
    if tid == 0u {
        // --- HBAND-top ---
        var sad0: u32 = 0u;
        var sad1: u32 = 0u;
        let mv0_qx = unpack_dx(hp_track_mv[0]);
        let mv0_qy = unpack_dy(hp_track_mv[0]);
        let mv1_qx = unpack_dx(hp_track_mv[1]);
        let mv1_qy = unpack_dy(hp_track_mv[1]);
        for (var ry = 0u; ry < 8u; ry++) {
            for (var rx = 0u; rx < 16u; rx++) {
                let cur_x = mb_origin_x + rx;
                let cur_y = mb_origin_y + ry;
                let cv = current_y[cur_y * params.width + cur_x];
                let rqx0 = i32(cur_x) * 4 + mv0_qx;
                let rqy0 = i32(cur_y) * 4 + mv0_qy;
                let rqx1 = i32(cur_x) * 4 + mv1_qx;
                let rqy1 = i32(cur_y) * 4 + mv1_qy;
                sad0 += u32(abs(cv - bilinear_sample(rqx0, rqy0)));
                sad1 += u32(abs(cv - bilinear_sample(rqx1, rqy1)));
            }
        }
        if sad0 <= sad1 {
            merged_sad_hband_top = sad0;
            hband_top_mv_idx = 0u;
        } else {
            merged_sad_hband_top = sad1;
            hband_top_mv_idx = 1u;
        }

        // --- HBAND-bot ---
        var sad2: u32 = 0u;
        var sad3: u32 = 0u;
        let mv2_qx = unpack_dx(hp_track_mv[2]);
        let mv2_qy = unpack_dy(hp_track_mv[2]);
        let mv3_qx = unpack_dx(hp_track_mv[3]);
        let mv3_qy = unpack_dy(hp_track_mv[3]);
        for (var ry = 8u; ry < 16u; ry++) {
            for (var rx = 0u; rx < 16u; rx++) {
                let cur_x = mb_origin_x + rx;
                let cur_y = mb_origin_y + ry;
                let cv = current_y[cur_y * params.width + cur_x];
                let rqx2 = i32(cur_x) * 4 + mv2_qx;
                let rqy2 = i32(cur_y) * 4 + mv2_qy;
                let rqx3 = i32(cur_x) * 4 + mv3_qx;
                let rqy3 = i32(cur_y) * 4 + mv3_qy;
                sad2 += u32(abs(cv - bilinear_sample(rqx2, rqy2)));
                sad3 += u32(abs(cv - bilinear_sample(rqx3, rqy3)));
            }
        }
        if sad2 <= sad3 {
            merged_sad_hband_bot = sad2;
            hband_bot_mv_idx = 2u;
        } else {
            merged_sad_hband_bot = sad3;
            hband_bot_mv_idx = 3u;
        }

        // --- VBAND-left ---
        var sadA: u32 = 0u;
        var sadB: u32 = 0u;
        for (var ry = 0u; ry < 16u; ry++) {
            for (var rx = 0u; rx < 8u; rx++) {
                let cur_x = mb_origin_x + rx;
                let cur_y = mb_origin_y + ry;
                let cv = current_y[cur_y * params.width + cur_x];
                let rqxA = i32(cur_x) * 4 + mv0_qx;
                let rqyA = i32(cur_y) * 4 + mv0_qy;
                let rqxB = i32(cur_x) * 4 + mv2_qx;
                let rqyB = i32(cur_y) * 4 + mv2_qy;
                sadA += u32(abs(cv - bilinear_sample(rqxA, rqyA)));
                sadB += u32(abs(cv - bilinear_sample(rqxB, rqyB)));
            }
        }
        if sadA <= sadB {
            merged_sad_vband_left = sadA;
            vband_left_mv_idx = 0u;
        } else {
            merged_sad_vband_left = sadB;
            vband_left_mv_idx = 2u;
        }

        // --- VBAND-right ---
        var sadC: u32 = 0u;
        var sadD: u32 = 0u;
        for (var ry = 0u; ry < 16u; ry++) {
            for (var rx = 8u; rx < 16u; rx++) {
                let cur_x = mb_origin_x + rx;
                let cur_y = mb_origin_y + ry;
                let cv = current_y[cur_y * params.width + cur_x];
                let rqxC = i32(cur_x) * 4 + mv1_qx;
                let rqyC = i32(cur_y) * 4 + mv1_qy;
                let rqxD = i32(cur_x) * 4 + mv3_qx;
                let rqyD = i32(cur_y) * 4 + mv3_qy;
                sadC += u32(abs(cv - bilinear_sample(rqxC, rqyC)));
                sadD += u32(abs(cv - bilinear_sample(rqxD, rqyD)));
            }
        }
        if sadC <= sadD {
            merged_sad_vband_right = sadC;
            vband_right_mv_idx = 1u;
        } else {
            merged_sad_vband_right = sadD;
            vband_right_mv_idx = 3u;
        }
    }
    workgroupBarrier();

    // ========== Phase 5b: 4-way RD decision (thread 0) + output ==========
    // RD cost = SAD + mode-specific overhead term.
    // Actual costs (see GP12-calibrated code below):
    //   16×16: rd = parent_sad                   (zero extra cost; parent SAD already paid)
    //   HBAND:  rd = hband_sad + lambda * 4       (2 MVs, flat lambda*4 overhead)
    //   VBAND:  rd = vband_sad + lambda * 4       (2 MVs, flat lambda*4 overhead)
    //   8×8:    rd = sum_sub + max(lambda, parent_sad/4)  (GP12-calibrated adaptive threshold)
    let blocks_x_8 = params.blocks_x * 2u;

    if tid == 0u {
        // Write 4 sub-block SADs for diagnostic readback (GNC_BLOCKSIZE_DIAG).
        // Always written; CPU readback only happens when env var is set.
        let diag_base = mb_idx * 4u;
        sub_sad_out[diag_base + 0u] = sub_sads[0];
        sub_sad_out[diag_base + 1u] = sub_sads[1];
        sub_sad_out[diag_base + 2u] = sub_sads[2];
        sub_sad_out[diag_base + 3u] = sub_sads[3];

        let sum_sub  = sub_sads[0] + sub_sads[1] + sub_sads[2] + sub_sads[3];
        let parent_sad = parent_sads[mb_idx];

        // GP12-calibrated threshold: 8×8 must beat 16×16 by max(lambda, parent/4).
        // This preserves the original split aggressiveness that was validated on crowd_run.
        // HBAND/VBAND (2 MVs each) use lambda*4 overhead as additional RD candidates.
        let threshold = max(params.lambda_sad, parent_sad / 4u);

        let rd_16x16 = parent_sad;
        // 8×8 uses the GP12-calibrated adaptive threshold (not flat lambda*8) to preserve
        // the split aggressiveness tuned empirically. At high parent_sad (fast motion),
        // threshold = parent_sad/4 prevents over-splitting. HBAND/VBAND use flat lambda*4
        // (2 MVs each), which is intentionally on a different scale — they compete against
        // 16×16 and 8×8 but are not required to be dimensionally equivalent.
        let rd_8x8   = sum_sub + threshold;
        let rd_hband = (merged_sad_hband_top + merged_sad_hband_bot) + params.lambda_sad * 4u;
        let rd_vband = (merged_sad_vband_left + merged_sad_vband_right) + params.lambda_sad * 4u;

        // Find best mode: 0=16×16, 1=HBAND, 2=VBAND, 3=8×8
        var best_rd = rd_16x16;
        var best_mode = 0u;
        if rd_hband < best_rd { best_rd = rd_hband; best_mode = 1u; }
        if rd_vband < best_rd { best_rd = rd_vband; best_mode = 2u; }
        if rd_8x8   < best_rd { best_rd = rd_8x8;  best_mode = 3u; }

        if best_mode == 3u {
            // 8×8: write all 4 independent sub-block MVs
            for (var s = 0u; s < 4u; s++) {
                let sc = s % 2u;
                let sr = s / 2u;
                let out_idx = (mb_y * 2u + sr) * blocks_x_8 + (mb_x * 2u + sc);
                let mv = hp_track_mv[s];
                output_mvs[out_idx * 2u]      = unpack_dx(mv);
                output_mvs[out_idx * 2u + 1u] = unpack_dy(mv);
            }
        } else if best_mode == 1u {
            // HBAND: sub[0]+sub[1] share top MV; sub[2]+sub[3] share bottom MV
            let top_mv = hp_track_mv[hband_top_mv_idx];
            let bot_mv = hp_track_mv[hband_bot_mv_idx];
            // TL (sub 0, row 0, col 0)
            let out00 = (mb_y * 2u + 0u) * blocks_x_8 + (mb_x * 2u + 0u);
            output_mvs[out00 * 2u]      = unpack_dx(top_mv);
            output_mvs[out00 * 2u + 1u] = unpack_dy(top_mv);
            // TR (sub 1, row 0, col 1)
            let out01 = (mb_y * 2u + 0u) * blocks_x_8 + (mb_x * 2u + 1u);
            output_mvs[out01 * 2u]      = unpack_dx(top_mv);
            output_mvs[out01 * 2u + 1u] = unpack_dy(top_mv);
            // BL (sub 2, row 1, col 0)
            let out10 = (mb_y * 2u + 1u) * blocks_x_8 + (mb_x * 2u + 0u);
            output_mvs[out10 * 2u]      = unpack_dx(bot_mv);
            output_mvs[out10 * 2u + 1u] = unpack_dy(bot_mv);
            // BR (sub 3, row 1, col 1)
            let out11 = (mb_y * 2u + 1u) * blocks_x_8 + (mb_x * 2u + 1u);
            output_mvs[out11 * 2u]      = unpack_dx(bot_mv);
            output_mvs[out11 * 2u + 1u] = unpack_dy(bot_mv);
        } else if best_mode == 2u {
            // VBAND: sub[0]+sub[2] share left MV; sub[1]+sub[3] share right MV
            let left_mv  = hp_track_mv[vband_left_mv_idx];
            let right_mv = hp_track_mv[vband_right_mv_idx];
            // TL (sub 0, row 0, col 0)
            let out00v = (mb_y * 2u + 0u) * blocks_x_8 + (mb_x * 2u + 0u);
            output_mvs[out00v * 2u]      = unpack_dx(left_mv);
            output_mvs[out00v * 2u + 1u] = unpack_dy(left_mv);
            // TR (sub 1, row 0, col 1)
            let out01v = (mb_y * 2u + 0u) * blocks_x_8 + (mb_x * 2u + 1u);
            output_mvs[out01v * 2u]      = unpack_dx(right_mv);
            output_mvs[out01v * 2u + 1u] = unpack_dy(right_mv);
            // BL (sub 2, row 1, col 0)
            let out10v = (mb_y * 2u + 1u) * blocks_x_8 + (mb_x * 2u + 0u);
            output_mvs[out10v * 2u]      = unpack_dx(left_mv);
            output_mvs[out10v * 2u + 1u] = unpack_dy(left_mv);
            // BR (sub 3, row 1, col 1)
            let out11v = (mb_y * 2u + 1u) * blocks_x_8 + (mb_x * 2u + 1u);
            output_mvs[out11v * 2u]      = unpack_dx(right_mv);
            output_mvs[out11v * 2u + 1u] = unpack_dy(right_mv);
        } else {
            // 16×16: replicate parent half-pel MV to all 4 output slots
            let pdx = parent_mvs[mb_idx * 2u];
            let pdy = parent_mvs[mb_idx * 2u + 1u];
            for (var s = 0u; s < 4u; s++) {
                let sc = s % 2u;
                let sr = s / 2u;
                let out_idx = (mb_y * 2u + sr) * blocks_x_8 + (mb_x * 2u + sc);
                output_mvs[out_idx * 2u]      = pdx;
                output_mvs[out_idx * 2u + 1u] = pdy;
            }
        }
    }
}
