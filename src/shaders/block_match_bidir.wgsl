// Bidirectional block matching motion estimation shader.
// One workgroup (256 threads) per 16x16 block.
// Tests forward-only, backward-only, and average (bidir) prediction.
// Picks whichever mode gives lowest SAD, with half-pel refinement for the winner.
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
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> current_y: array<f32>;
@group(0) @binding(2) var<storage, read> reference_fwd_y: array<f32>;
@group(0) @binding(3) var<storage, read> reference_bwd_y: array<f32>;
@group(0) @binding(4) var<storage, read_write> fwd_motion_vectors: array<i32>;
@group(0) @binding(5) var<storage, read_write> bwd_motion_vectors: array<i32>;
@group(0) @binding(6) var<storage, read_write> block_modes: array<u32>;
@group(0) @binding(7) var<storage, read_write> sad_values: array<u32>;

// Shared memory for parallel min-reduction
var<workgroup> shared_sad: array<u32, 256>;
// Pack (dx, dy) as (dx+32768) << 16 | (dy+32768)
var<workgroup> shared_mv: array<u32, 256>;

// Half-pel sample from forward reference (inlined because WGSL cannot pass storage ptrs)
fn sample_hp_fwd(x2: i32, y2: i32, w: u32, h: u32) -> f32 {
    let fx = x2 >> 1;
    let fy = y2 >> 1;
    let frac_x = x2 & 1;
    let frac_y = y2 & 1;
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
    } else if frac_x == 1 && frac_y == 0 {
        return (p00 + p10) * 0.5;
    } else if frac_x == 0 && frac_y == 1 {
        return (p00 + p01) * 0.5;
    } else {
        return (p00 + p10 + p01 + p11) * 0.25;
    }
}

// Half-pel sample from backward reference
fn sample_hp_bwd(x2: i32, y2: i32, w: u32, h: u32) -> f32 {
    let fx = x2 >> 1;
    let fy = y2 >> 1;
    let frac_x = x2 & 1;
    let frac_y = y2 & 1;
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
    } else if frac_x == 1 && frac_y == 0 {
        return (p00 + p10) * 0.5;
    } else if frac_x == 0 && frac_y == 1 {
        return (p00 + p01) * 0.5;
    } else {
        return (p00 + p10 + p01 + p11) * 0.25;
    }
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

    // Phase 1: Find best forward MV (against forward reference)
    var best_fwd_sad: u32 = 0xFFFFFFFFu;
    var best_fwd_dx: i32 = 0;
    var best_fwd_dy: i32 = 0;

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
           ref_x + i32(params.block_size) <= i32(params.width) &&
           ref_y + i32(params.block_size) <= i32(params.height) {

            var sad: u32 = 0u;
            for (var py = 0u; py < params.block_size; py++) {
                for (var px = 0u; px < params.block_size; px++) {
                    let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                    let r_idx = u32(ref_y + i32(py)) * params.width + u32(ref_x + i32(px));
                    let diff = current_y[cur_idx] - reference_fwd_y[r_idx];
                    sad += u32(abs(diff));
                }
            }
            if sad < best_fwd_sad {
                best_fwd_sad = sad;
                best_fwd_dx = cand_dx;
                best_fwd_dy = cand_dy;
            }
        }
        cand_idx += 256u;
    }

    // Parallel reduction for forward MV
    shared_sad[tid] = best_fwd_sad;
    shared_mv[tid] = (u32(best_fwd_dx + 32768) << 16u) | u32(best_fwd_dy + 32768);
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            if shared_sad[tid + stride] < shared_sad[tid] {
                shared_sad[tid] = shared_sad[tid + stride];
                shared_mv[tid] = shared_mv[tid + stride];
            }
        }
        workgroupBarrier();
    }

    // Broadcast forward winner via shared memory
    var final_fwd_dx: i32;
    var final_fwd_dy: i32;
    var final_fwd_sad: u32;
    // shared_mv[0] and shared_sad[0] already have the winner after reduction
    workgroupBarrier();
    let fwd_packed = shared_mv[0];
    final_fwd_dx = i32(fwd_packed >> 16u) - 32768;
    final_fwd_dy = i32(fwd_packed & 0xFFFFu) - 32768;
    final_fwd_sad = shared_sad[0];
    workgroupBarrier();

    // Phase 2: Find best backward MV (against backward reference)
    var best_bwd_sad: u32 = 0xFFFFFFFFu;
    var best_bwd_dx: i32 = 0;
    var best_bwd_dy: i32 = 0;

    cand_idx = tid;
    loop {
        if cand_idx >= total_candidates {
            break;
        }
        let cand_dy = i32(cand_idx / search_side) - sr;
        let cand_dx = i32(cand_idx % search_side) - sr;

        let ref_x = i32(block_origin_x) + cand_dx;
        let ref_y = i32(block_origin_y) + cand_dy;

        if ref_x >= 0 && ref_y >= 0 &&
           ref_x + i32(params.block_size) <= i32(params.width) &&
           ref_y + i32(params.block_size) <= i32(params.height) {

            var sad: u32 = 0u;
            for (var py = 0u; py < params.block_size; py++) {
                for (var px = 0u; px < params.block_size; px++) {
                    let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                    let r_idx = u32(ref_y + i32(py)) * params.width + u32(ref_x + i32(px));
                    let diff = current_y[cur_idx] - reference_bwd_y[r_idx];
                    sad += u32(abs(diff));
                }
            }
            if sad < best_bwd_sad {
                best_bwd_sad = sad;
                best_bwd_dx = cand_dx;
                best_bwd_dy = cand_dy;
            }
        }
        cand_idx += 256u;
    }

    // Parallel reduction for backward MV
    shared_sad[tid] = best_bwd_sad;
    shared_mv[tid] = (u32(best_bwd_dx + 32768) << 16u) | u32(best_bwd_dy + 32768);
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            if shared_sad[tid + stride] < shared_sad[tid] {
                shared_sad[tid] = shared_sad[tid + stride];
                shared_mv[tid] = shared_mv[tid + stride];
            }
        }
        workgroupBarrier();
    }

    var final_bwd_dx: i32;
    var final_bwd_dy: i32;
    var final_bwd_sad: u32;
    let bwd_packed = shared_mv[0];
    final_bwd_dx = i32(bwd_packed >> 16u) - 32768;
    final_bwd_dy = i32(bwd_packed & 0xFFFFu) - 32768;
    final_bwd_sad = shared_sad[0];
    workgroupBarrier();

    // Phase 3: Thread 0 computes bidir SAD and picks best mode
    if tid == 0u {
        // Compute bidirectional SAD: current - (fwd + bwd) / 2
        var bidir_sad: u32 = 0xFFFFFFFFu;
        let fwd_x = i32(block_origin_x) + final_fwd_dx;
        let fwd_y = i32(block_origin_y) + final_fwd_dy;
        let bwd_x = i32(block_origin_x) + final_bwd_dx;
        let bwd_y = i32(block_origin_y) + final_bwd_dy;

        if fwd_x >= 0 && fwd_y >= 0 && fwd_x + i32(params.block_size) <= i32(params.width) && fwd_y + i32(params.block_size) <= i32(params.height) &&
           bwd_x >= 0 && bwd_y >= 0 && bwd_x + i32(params.block_size) <= i32(params.width) && bwd_y + i32(params.block_size) <= i32(params.height) {
            bidir_sad = 0u;
            for (var py = 0u; py < params.block_size; py++) {
                for (var px = 0u; px < params.block_size; px++) {
                    let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                    let f_idx = u32(fwd_y + i32(py)) * params.width + u32(fwd_x + i32(px));
                    let b_idx = u32(bwd_y + i32(py)) * params.width + u32(bwd_x + i32(px));
                    let pred = (reference_fwd_y[f_idx] + reference_bwd_y[b_idx]) * 0.5;
                    let diff = current_y[cur_idx] - pred;
                    bidir_sad += u32(abs(diff));
                }
            }
        }

        // Pick best mode: 0=fwd, 1=bwd, 2=bidir
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

        // Convert winning integer MVs to half-pel units (multiply by 2)
        var hp_fwd_dx = final_fwd_dx * 2;
        var hp_fwd_dy = final_fwd_dy * 2;
        var hp_bwd_dx = final_bwd_dx * 2;
        var hp_bwd_dy = final_bwd_dy * 2;

        // Half-pel refinement for forward MV (if used by winning mode)
        if best_mode == 0u || best_mode == 2u {
            var hp_best_sad = best_total_sad;
            let base_fx2 = final_fwd_dx * 2;
            let base_fy2 = final_fwd_dy * 2;
            for (var offy = -1; offy <= 1; offy++) {
                for (var offx = -1; offx <= 1; offx++) {
                    if offx == 0 && offy == 0 { continue; }
                    let test_fx2 = base_fx2 + offx;
                    let test_fy2 = base_fy2 + offy;

                    var test_sad: u32 = 0u;
                    for (var py = 0u; py < params.block_size; py++) {
                        for (var px = 0u; px < params.block_size; px++) {
                            let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                            let rx2 = i32(block_origin_x + px) * 2 + test_fx2;
                            let ry2 = i32(block_origin_y + py) * 2 + test_fy2;
                            let fwd_val = sample_hp_fwd(rx2, ry2, params.width, params.height);

                            var pred: f32;
                            if best_mode == 0u {
                                pred = fwd_val;
                            } else {
                                let bx2 = i32(block_origin_x + px) * 2 + hp_bwd_dx;
                                let by2 = i32(block_origin_y + py) * 2 + hp_bwd_dy;
                                let bwd_val = sample_hp_bwd(bx2, by2, params.width, params.height);
                                pred = (fwd_val + bwd_val) * 0.5;
                            }
                            let diff = current_y[cur_idx] - pred;
                            test_sad += u32(abs(diff));
                        }
                    }
                    if test_sad < hp_best_sad {
                        hp_best_sad = test_sad;
                        hp_fwd_dx = test_fx2;
                        hp_fwd_dy = test_fy2;
                    }
                }
            }
            best_total_sad = hp_best_sad;
        }

        // Half-pel refinement for backward MV (if used by winning mode)
        if best_mode == 1u || best_mode == 2u {
            var hp_best_sad = best_total_sad;
            let base_bx2 = final_bwd_dx * 2;
            let base_by2 = final_bwd_dy * 2;
            for (var offy = -1; offy <= 1; offy++) {
                for (var offx = -1; offx <= 1; offx++) {
                    if offx == 0 && offy == 0 { continue; }
                    let test_bx2 = base_bx2 + offx;
                    let test_by2 = base_by2 + offy;

                    var test_sad: u32 = 0u;
                    for (var py = 0u; py < params.block_size; py++) {
                        for (var px = 0u; px < params.block_size; px++) {
                            let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                            let rx2 = i32(block_origin_x + px) * 2 + test_bx2;
                            let ry2 = i32(block_origin_y + py) * 2 + test_by2;
                            let bwd_val = sample_hp_bwd(rx2, ry2, params.width, params.height);

                            var pred: f32;
                            if best_mode == 1u {
                                pred = bwd_val;
                            } else {
                                let fx2 = i32(block_origin_x + px) * 2 + hp_fwd_dx;
                                let fy2 = i32(block_origin_y + py) * 2 + hp_fwd_dy;
                                let fwd_val = sample_hp_fwd(fx2, fy2, params.width, params.height);
                                pred = (fwd_val + bwd_val) * 0.5;
                            }
                            let diff = current_y[cur_idx] - pred;
                            test_sad += u32(abs(diff));
                        }
                    }
                    if test_sad < hp_best_sad {
                        hp_best_sad = test_sad;
                        hp_bwd_dx = test_bx2;
                        hp_bwd_dy = test_by2;
                    }
                }
            }
        }

        // Write results: half-pel MVs
        fwd_motion_vectors[block_idx * 2u] = hp_fwd_dx;
        fwd_motion_vectors[block_idx * 2u + 1u] = hp_fwd_dy;
        bwd_motion_vectors[block_idx * 2u] = hp_bwd_dx;
        bwd_motion_vectors[block_idx * 2u + 1u] = hp_bwd_dy;
        block_modes[block_idx] = best_mode;
        sad_values[block_idx] = best_total_sad;
    }
}
