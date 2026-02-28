// Fused 8×8 DCT + Quantize + Dequantize + IDCT (Mega-Kernel Stage 1)
//
// One workgroup (64 threads) processes one 8×8 block.
// In shared memory: forward DCT → quantize → dequantize → inverse DCT.
// Outputs: quantized indices (f32) AND reconstructed pixels (f32).
// Dispatch: (padded_width / 8, padded_height / 8, 1)
//
// This replaces 3 separate dispatches (transform + quantize + local decode)
// with a single dispatch. Shared memory never leaves the workgroup.

struct Params {
    width: u32,
    height: u32,
    step_size: f32,
    dead_zone: f32,  // dead zone as fraction of step_size (0.0 = no dead zone)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> quant_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> recon_out: array<f32>;

var<workgroup> smem: array<f32, 64>;

const PI: f32 = 3.14159265359;

// Orthonormal DCT-II basis: C[k][n] = alpha(k) * cos(pi * (2n+1) * k / 16)
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

    // ---- Step 1: Load block into shared memory ----
    if px < params.width && py < params.height {
        smem[thread_id] = input[py * params.width + px];
    } else {
        smem[thread_id] = 0.0;
    }
    workgroupBarrier();

    // ---- Step 2: Forward DCT (row pass) ----
    var sum = 0.0;
    for (var n = 0u; n < 8u; n = n + 1u) {
        sum += smem[row * 8u + n] * dct_basis(col, n);
    }
    workgroupBarrier();
    smem[thread_id] = sum;
    workgroupBarrier();

    // ---- Step 3: Forward DCT (col pass) ----
    sum = 0.0;
    for (var n = 0u; n < 8u; n = n + 1u) {
        sum += smem[n * 8u + col] * dct_basis(row, n);
    }
    // sum now holds the DCT coefficient for this (row, col) frequency

    // ---- Step 4: Quantize ----
    let threshold = params.dead_zone * params.step_size;
    var quant_idx: f32;
    if abs(sum) < threshold {
        quant_idx = 0.0;
    } else {
        quant_idx = sign(sum) * floor(abs(sum) / params.step_size + 0.5);
    }

    // Write quantized index to output
    if px < params.width && py < params.height {
        quant_out[py * params.width + px] = quant_idx;
    }

    // ---- Step 5: Dequantize ----
    let recon_coeff = quant_idx * params.step_size;

    // ---- Step 6: Inverse DCT (row pass, from dequantized coefficients) ----
    // Load dequantized coefficients into shared memory
    workgroupBarrier();
    smem[thread_id] = recon_coeff;
    workgroupBarrier();

    // Row IDCT: recover spatial sample 'col' from frequency-domain row
    sum = 0.0;
    for (var k = 0u; k < 8u; k = k + 1u) {
        sum += smem[row * 8u + k] * dct_basis(k, col);
    }
    workgroupBarrier();
    smem[thread_id] = sum;
    workgroupBarrier();

    // ---- Step 7: Inverse DCT (col pass) ----
    sum = 0.0;
    for (var k = 0u; k < 8u; k = k + 1u) {
        sum += smem[k * 8u + col] * dct_basis(k, row);
    }

    // Write reconstructed pixel
    if px < params.width && py < params.height {
        recon_out[py * params.width + px] = sum;
    }
}
