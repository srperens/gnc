// 16×16 Block DCT Transform (Type-II / Type-III)
//
// One workgroup (256 threads) processes one 16×16 block.
// Separable: row DCT → barrier → column DCT, all in smem memory.
// Dispatch: (padded_width / 16, padded_height / 16, 1)
//
// Better energy compaction than 8×8 at the cost of more compute per block
// and coarser spatial adaptation. Good candidate for high-quality modes.

struct Params {
    width: u32,
    height: u32,
    direction: u32, // 0 = forward, 1 = inverse
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> smem: array<f32, 256>;

const PI: f32 = 3.14159265359;

// Orthonormal DCT-II basis for N=16
// alpha(0) = 1/sqrt(16) = 0.25, alpha(k>0) = sqrt(2/16) = 0.353553
fn dct_basis(k: u32, n: u32) -> f32 {
    let scale = select(0.353553390593, 0.25, k == 0u);
    return scale * cos(PI * f32(2u * n + 1u) * f32(k) / 32.0);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let thread_id = lid.x;
    let row = thread_id / 16u;
    let col = thread_id % 16u;

    let px = wg_id.x * 16u + col;
    let py = wg_id.y * 16u + row;

    // Load block into smem memory
    if px < params.width && py < params.height {
        smem[thread_id] = input[py * params.width + px];
    } else {
        smem[thread_id] = 0.0;
    }
    workgroupBarrier();

    if params.direction == 0u {
        // ---- FORWARD DCT ----

        // Row DCT
        var sum = 0.0;
        for (var n = 0u; n < 16u; n = n + 1u) {
            sum += smem[row * 16u + n] * dct_basis(col, n);
        }
        workgroupBarrier();
        smem[thread_id] = sum;
        workgroupBarrier();

        // Column DCT
        sum = 0.0;
        for (var n = 0u; n < 16u; n = n + 1u) {
            sum += smem[n * 16u + col] * dct_basis(row, n);
        }

        if px < params.width && py < params.height {
            output[py * params.width + px] = sum;
        }
    } else {
        // ---- INVERSE DCT ----

        // Row IDCT
        var sum = 0.0;
        for (var k = 0u; k < 16u; k = k + 1u) {
            sum += smem[row * 16u + k] * dct_basis(k, col);
        }
        workgroupBarrier();
        smem[thread_id] = sum;
        workgroupBarrier();

        // Column IDCT
        sum = 0.0;
        for (var k = 0u; k < 16u; k = k + 1u) {
            sum += smem[k * 16u + col] * dct_basis(k, row);
        }

        if px < params.width && py < params.height {
            output[py * params.width + px] = sum;
        }
    }
}
