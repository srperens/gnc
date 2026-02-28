// 8×8 Block DCT Transform (Type-II / Type-III)
//
// One workgroup (64 threads) processes one 8×8 block.
// Separable: row DCT → barrier → column DCT, all in smem memory.
// Dispatch: (padded_width / 8, padded_height / 8, 1)
//
// Output layout: coefficients stored in-place (same position as input pixel).
// Position (x % 8, y % 8) within each block gives the frequency index.
// (0,0) = DC, (7,7) = highest horizontal + vertical frequency.

struct Params {
    width: u32,
    height: u32,
    direction: u32, // 0 = forward, 1 = inverse
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> smem: array<f32, 64>;

const PI: f32 = 3.14159265359;

// Orthonormal DCT-II basis: C[k][n] = alpha(k) * cos(pi * (2n+1) * k / 16)
// alpha(0) = 1/sqrt(8), alpha(k>0) = sqrt(2/8) = 0.5
fn dct_basis(k: u32, n: u32) -> f32 {
    let scale = select(0.5, 0.353553390593, k == 0u);
    return scale * cos(PI * f32(2u * n + 1u) * f32(k) / 16.0);
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let thread_id = lid.x;
    let row = thread_id / 8u;
    let col = thread_id % 8u;

    let px = wg_id.x * 8u + col;
    let py = wg_id.y * 8u + row;

    // Load block into smem memory
    if px < params.width && py < params.height {
        smem[thread_id] = input[py * params.width + px];
    } else {
        smem[thread_id] = 0.0;
    }
    workgroupBarrier();

    if params.direction == 0u {
        // ---- FORWARD DCT ----

        // Row DCT: thread (row, col) computes frequency 'col' of row 'row'
        var sum = 0.0;
        for (var n = 0u; n < 8u; n = n + 1u) {
            sum += smem[row * 8u + n] * dct_basis(col, n);
        }
        workgroupBarrier();
        smem[thread_id] = sum;
        workgroupBarrier();

        // Column DCT: thread (row, col) computes frequency 'row' of column 'col'
        sum = 0.0;
        for (var n = 0u; n < 8u; n = n + 1u) {
            sum += smem[n * 8u + col] * dct_basis(row, n);
        }

        if px < params.width && py < params.height {
            output[py * params.width + px] = sum;
        }
    } else {
        // ---- INVERSE DCT (IDCT) ----
        // Orthonormal DCT is inverted by transposing the basis matrix.
        // x[n] = sum_k X[k] * C[k][n]  (same basis values, sum over freq index)

        // Row IDCT: recover spatial sample 'col' from frequency-domain row 'row'
        var sum = 0.0;
        for (var k = 0u; k < 8u; k = k + 1u) {
            sum += smem[row * 8u + k] * dct_basis(k, col);
        }
        workgroupBarrier();
        smem[thread_id] = sum;
        workgroupBarrier();

        // Column IDCT: recover spatial sample 'row' from frequency-domain column 'col'
        sum = 0.0;
        for (var k = 0u; k < 8u; k = k + 1u) {
            sum += smem[k * 8u + col] * dct_basis(k, row);
        }

        if px < params.width && py < params.height {
            output[py * params.width + px] = sum;
        }
    }
}
