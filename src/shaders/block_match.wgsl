// Hierarchical block matching motion estimation shader.
// One workgroup (256 threads) per 16×16 block.
//
// Three-phase coarse-to-fine search:
//   Phase 1: Coarse search over full ±search_range using subsampled 4×4 SAD
//            (every 4th pixel in each dimension = 16 samples per candidate).
//            14.8x fewer memory reads than full-resolution search.
//   Phase 2: Fine search ±4 around coarse winner with full 16×16 SAD.
//   Phase 3: Half-pel refinement around fine winner using bilinear interpolation.
//
// Output MVs are in half-pel units.
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

// Bilinear sample at half-pel position (hx, hy are in half-pel units).
fn bilinear_sample(hx: i32, hy: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);

    let ix = hx >> 1;
    let iy = hy >> 1;
    let fx = hx & 1;
    let fy = hy & 1;

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

    let ffx = f32(fx) * 0.5;
    let ffy = f32(fy) * 0.5;

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

    if params.use_predictor != 0u {
        // Temporal MV prediction: use previous frame's MV as starting point.
        // Predictor MVs are in half-pel units — convert to integer-pel.
        if tid == 0u {
            let pred_hx = predictor_mvs[block_idx * 2u];
            let pred_hy = predictor_mvs[block_idx * 2u + 1u];
            // Round half-pel to nearest integer-pel
            coarse_dx = (pred_hx + select(0, 1, pred_hx > 0)) / 2;
            coarse_dy = (pred_hy + select(0, 1, pred_hy > 0)) / 2;
        }
        shared_mv[0] = pack_mv(coarse_dx, coarse_dy);
        workgroupBarrier();
        let pred_packed = shared_mv[0];
        coarse_dx = unpack_dx(pred_packed);
        coarse_dy = unpack_dy(pred_packed);
        workgroupBarrier();
        fine_range = i32(params.pred_fine_range);
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

        // Min-reduction for coarse winner
        shared_sad[tid] = best_sad;
        shared_mv[tid] = pack_mv(best_dx, best_dy);
        workgroupBarrier();
        min_reduce(tid);

        let coarse_packed = shared_mv[0];
        coarse_dx = unpack_dx(coarse_packed);
        coarse_dy = unpack_dy(coarse_packed);
        workgroupBarrier();
    }

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

    // ========== Phase 3: Half-pel refinement ==========
    // 8 half-pel offsets around the fine winner (same as before).
    let fine_packed = shared_mv[0];
    let fine_sad = shared_sad[0];
    let int_dx = unpack_dx(fine_packed);
    let int_dy = unpack_dy(fine_packed);
    workgroupBarrier();

    let center_hx = int_dx * 2;
    let center_hy = int_dy * 2;

    var hp_sad: u32 = 0xFFFFFFFFu;
    var hp_dx: i32 = center_hx;
    var hp_dy: i32 = center_hy;

    if tid < 8u {
        var off_x: i32 = 0;
        var off_y: i32 = 0;
        switch tid {
            case 0u: { off_x = -1; off_y = -1; }
            case 1u: { off_x =  0; off_y = -1; }
            case 2u: { off_x =  1; off_y = -1; }
            case 3u: { off_x = -1; off_y =  0; }
            case 4u: { off_x =  1; off_y =  0; }
            case 5u: { off_x = -1; off_y =  1; }
            case 6u: { off_x =  0; off_y =  1; }
            case 7u: { off_x =  1; off_y =  1; }
            default: {}
        }

        let test_hx = center_hx + off_x;
        let test_hy = center_hy + off_y;

        var sad: u32 = 0u;
        var valid = true;

        let ref_base_hx = i32(block_origin_x) * 2 + test_hx;
        let ref_base_hy = i32(block_origin_y) * 2 + test_hy;
        if ref_base_hx < -1 || ref_base_hy < -1 ||
           ref_base_hx + bs * 2 > w * 2 + 1 ||
           ref_base_hy + bs * 2 > h * 2 + 1 {
            valid = false;
        }

        if valid {
            for (var py = 0u; py < params.block_size; py++) {
                for (var px = 0u; px < params.block_size; px++) {
                    let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                    let cur_val = current_y[cur_idx];

                    let rhx = i32(block_origin_x + px) * 2 + test_hx;
                    let rhy = i32(block_origin_y + py) * 2 + test_hy;
                    let ref_val = bilinear_sample(rhx, rhy);

                    let diff = cur_val - ref_val;
                    sad += u32(abs(diff));
                }
            }
            hp_sad = sad;
            hp_dx = test_hx;
            hp_dy = test_hy;
        }
    }

    // Thread 8 holds the integer result
    if tid == 8u {
        hp_sad = fine_sad;
        hp_dx = center_hx;
        hp_dy = center_hy;
    }
    if tid <= 8u {
        shared_sad[tid] = hp_sad;
        shared_mv[tid] = pack_mv(hp_dx, hp_dy);
    } else {
        shared_sad[tid] = 0xFFFFFFFFu;
    }
    workgroupBarrier();
    min_reduce(tid);

    // Thread 0 writes the final half-pel MV result
    if tid == 0u {
        let packed_mv = shared_mv[0];
        let final_dx = unpack_dx(packed_mv);
        let final_dy = unpack_dy(packed_mv);
        motion_vectors[block_idx * 2u] = final_dx;
        motion_vectors[block_idx * 2u + 1u] = final_dy;
        sad_values[block_idx] = shared_sad[0];
    }
}
