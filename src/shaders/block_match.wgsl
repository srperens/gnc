// Block matching motion estimation shader with half-pel refinement.
// One workgroup (256 threads) per 16×16 block.
// Phase 1: Full-pel search ±search_range pixels, SAD on Y (luma) plane.
// Phase 2: Half-pel refinement around integer winner using bilinear interpolation.
// Output MVs are in half-pel units (integer MV * 2, then refined by ±1 half-pel).
// Parallel min-reduction across workgroup to find best MV.

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
@group(0) @binding(2) var<storage, read> reference_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> motion_vectors: array<i32>;
@group(0) @binding(4) var<storage, read_write> sad_values: array<u32>;

// Shared memory for parallel min-reduction
var<workgroup> shared_sad: array<u32, 256>;
// Pack (dx, dy) as (dx+32768) << 16 | (dy+32768) for atomic-free reduction
var<workgroup> shared_mv: array<u32, 256>;

// Bilinear sample at half-pel position (hx, hy are in half-pel units).
// Returns interpolated value or clamped-edge value for out-of-bounds.
fn bilinear_sample(hx: i32, hy: i32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);

    // Integer pixel coordinates (floor of half-pel / 2)
    let ix = hx >> 1;
    let iy = hy >> 1;
    let fx = hx & 1; // 0 = on grid, 1 = half-pel offset
    let fy = hy & 1;

    if fx == 0 && fy == 0 {
        // On integer grid — direct lookup
        let cx = clamp(ix, 0, w - 1);
        let cy = clamp(iy, 0, h - 1);
        return reference_y[u32(cy) * params.width + u32(cx)];
    }

    // Bilinear interpolation from 4 neighbors
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

    // ========== Phase 1: Integer-pel full search ==========
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

        // Reference block origin
        let ref_x = i32(block_origin_x) + cand_dx;
        let ref_y = i32(block_origin_y) + cand_dy;

        // Check if entire reference block is within bounds
        if ref_x >= 0 && ref_y >= 0 &&
           ref_x + i32(params.block_size) <= i32(params.width) &&
           ref_y + i32(params.block_size) <= i32(params.height) {

            var sad: u32 = 0u;
            for (var py = 0u; py < params.block_size; py++) {
                for (var px = 0u; px < params.block_size; px++) {
                    let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                    let r_idx = u32(ref_y + i32(py)) * params.width + u32(ref_x + i32(px));
                    let diff = current_y[cur_idx] - reference_y[r_idx];
                    sad += u32(abs(diff));
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

    // Store in shared memory for min-reduction
    shared_sad[tid] = best_sad;
    shared_mv[tid] = (u32(best_dx + 32768) << 16u) | u32(best_dy + 32768);
    workgroupBarrier();

    // Parallel min-reduction (log2(256) = 8 steps)
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            if shared_sad[tid + stride] < shared_sad[tid] {
                shared_sad[tid] = shared_sad[tid + stride];
                shared_mv[tid] = shared_mv[tid + stride];
            }
        }
        workgroupBarrier();
    }

    // ========== Phase 2: Half-pel refinement ==========
    // Thread 0 reads integer winner, then first 8 threads each test one half-pel offset.
    // Half-pel offsets around the integer winner (8 neighbors):
    //   (-1,-1) (0,-1) (1,-1)
    //   (-1, 0)        (1, 0)
    //   (-1, 1) (0, 1) (1, 1)
    // These are in half-pel units, so the integer winner is at (dx*2, dy*2).

    // Broadcast integer winner to all threads via shared memory
    workgroupBarrier();

    // Read the integer-pel winner from reduction result
    let int_packed = shared_mv[0];
    let int_sad = shared_sad[0];
    let int_dx = i32(int_packed >> 16u) - 32768;
    let int_dy = i32(int_packed & 0xFFFFu) - 32768;

    // Half-pel center in half-pel units
    let center_hx = int_dx * 2;
    let center_hy = int_dy * 2;

    // 8 half-pel offsets (thread 0..7 each tests one)
    // offset_table: dx_hp, dy_hp for each of 8 neighbors
    var hp_sad: u32 = 0xFFFFFFFFu;
    var hp_dx: i32 = center_hx;
    var hp_dy: i32 = center_hy;

    if tid < 8u {
        // Map tid to half-pel offset
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

        // Compute SAD with bilinear interpolation at half-pel position
        var sad: u32 = 0u;
        var valid = true;

        // Quick bounds check: reference block center must be roughly in frame
        let ref_base_hx = i32(block_origin_x) * 2 + test_hx;
        let ref_base_hy = i32(block_origin_y) * 2 + test_hy;
        if ref_base_hx < -1 || ref_base_hy < -1 ||
           ref_base_hx + i32(params.block_size) * 2 > i32(params.width) * 2 + 1 ||
           ref_base_hy + i32(params.block_size) * 2 > i32(params.height) * 2 + 1 {
            valid = false;
        }

        if valid {
            for (var py = 0u; py < params.block_size; py++) {
                for (var px = 0u; px < params.block_size; px++) {
                    let cur_idx = (block_origin_y + py) * params.width + block_origin_x + px;
                    let cur_val = current_y[cur_idx];

                    // Half-pel reference position for this pixel
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

    // Write half-pel results + integer winner into shared for final reduction
    // Thread 8 holds the integer result (already at center_hx, center_hy)
    if tid == 8u {
        hp_sad = int_sad;
        hp_dx = center_hx;
        hp_dy = center_hy;
    }
    if tid <= 8u {
        shared_sad[tid] = hp_sad;
        shared_mv[tid] = (u32(hp_dx + 32768) << 16u) | u32(hp_dy + 32768);
    } else {
        shared_sad[tid] = 0xFFFFFFFFu;
    }
    workgroupBarrier();

    // Min-reduction over first 16 slots (more than enough for 9 candidates)
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            if shared_sad[tid + stride] < shared_sad[tid] {
                shared_sad[tid] = shared_sad[tid + stride];
                shared_mv[tid] = shared_mv[tid + stride];
            }
        }
        workgroupBarrier();
    }

    // Thread 0 writes the final half-pel MV result
    if tid == 0u {
        let packed_mv = shared_mv[0];
        let final_dx = i32(packed_mv >> 16u) - 32768;
        let final_dy = i32(packed_mv & 0xFFFFu) - 32768;
        // Output is in half-pel units
        motion_vectors[block_idx * 2u] = final_dx;
        motion_vectors[block_idx * 2u + 1u] = final_dy;
        sad_values[block_idx] = shared_sad[0];
    }
}
