// 4×4 Walsh-Hadamard Transform
//
// One workgroup (256 threads) processes 16 blocks of 4×4 (a 16×16 pixel area).
// No multiplications — only additions and subtractions (+ final scaling by 0.5).
// Dispatch: (padded_width / 16, padded_height / 16, 1)
//
// The WHT is self-inverse (orthonormal symmetric): forward and inverse are
// identical operations. direction param is accepted for API compatibility
// but both paths run the same code.
//
// Extremely fast, minimal smem memory (1 KB). Good candidate for
// low-latency scenarios where compression quality can be traded for speed.

struct Params {
    width: u32,
    height: u32,
    direction: u32, // 0 = forward, 1 = inverse (same operation for WHT)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> smem: array<f32, 256>;

// 4-point normalized WHT basis (rows of H4 / 2):
//   [+1 +1 +1 +1] / 2
//   [+1 -1 +1 -1] / 2   (highest sequency)
//   [+1 +1 -1 -1] / 2
//   [+1 -1 -1 +1] / 2
//
// Computed via butterfly: a=x0+x1, b=x0-x1, c=x2+x3, d=x2-x3
//   y0 = (a+c)/2, y1 = (b-d)/2, y2 = (a-c)/2, y3 = (b+d)/2

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let thread_id = lid.x;

    // Map thread to position within 16×16 area (4×4 grid of 4×4 blocks)
    let block_in_wg = thread_id / 16u;   // which of the 16 blocks (0..15)
    let local_id = thread_id % 16u;       // position within 4×4 block
    let local_row = local_id / 4u;
    let local_col = local_id % 4u;

    let block_in_wg_x = block_in_wg % 4u;
    let block_in_wg_y = block_in_wg / 4u;

    let px = wg_id.x * 16u + block_in_wg_x * 4u + local_col;
    let py = wg_id.y * 16u + block_in_wg_y * 4u + local_row;

    // Load pixel into smem memory
    if px < params.width && py < params.height {
        smem[thread_id] = input[py * params.width + px];
    } else {
        smem[thread_id] = 0.0;
    }
    workgroupBarrier();

    // ---- Row WHT (4-point butterfly) ----
    let row_base = block_in_wg * 16u + local_row * 4u;
    let x0 = smem[row_base + 0u];
    let x1 = smem[row_base + 1u];
    let x2 = smem[row_base + 2u];
    let x3 = smem[row_base + 3u];

    let a = x0 + x1;
    let b = x0 - x1;
    let c = x2 + x3;
    let d = x2 - x3;

    // Symmetric H4 ordering: rows = [1,1,1,1], [1,1,-1,-1], [1,-1,-1,1], [1,-1,1,-1]
    // This makes the matrix self-inverse (W*W = I) since H4 is symmetric + orthogonal.
    var row_result: f32;
    switch local_col {
        case 0u: { row_result = (a + c) * 0.5; }
        case 1u: { row_result = (a - c) * 0.5; }
        case 2u: { row_result = (b - d) * 0.5; }
        default: { row_result = (b + d) * 0.5; }
    }

    workgroupBarrier();
    smem[thread_id] = row_result;
    workgroupBarrier();

    // ---- Column WHT (4-point butterfly) ----
    let col_base = block_in_wg * 16u + local_col;
    let y0 = smem[col_base + 0u];
    let y1 = smem[col_base + 4u];
    let y2 = smem[col_base + 8u];
    let y3 = smem[col_base + 12u];

    let ea = y0 + y1;
    let eb = y0 - y1;
    let ec = y2 + y3;
    let ed = y2 - y3;

    var col_result: f32;
    switch local_row {
        case 0u: { col_result = (ea + ec) * 0.5; }
        case 1u: { col_result = (ea - ec) * 0.5; }
        case 2u: { col_result = (eb - ed) * 0.5; }
        default: { col_result = (eb + ed) * 0.5; }
    }

    // Write output
    if px < params.width && py < params.height {
        output[py * params.width + px] = col_result;
    }
}
