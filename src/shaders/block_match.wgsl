// Block matching motion estimation shader.
// One workgroup (256 threads) per 16×16 block.
// Full search ±search_range pixels, SAD computed on Y (luma) plane only.
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

    // Each thread evaluates a subset of candidates
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

    // Thread 0 writes the result
    if tid == 0u {
        let packed_mv = shared_mv[0];
        let dx = i32(packed_mv >> 16u) - 32768;
        let dy = i32(packed_mv & 0xFFFFu) - 32768;
        motion_vectors[block_idx * 2u] = dx;
        motion_vectors[block_idx * 2u + 1u] = dy;
        sad_values[block_idx] = shared_sad[0];
    }
}
