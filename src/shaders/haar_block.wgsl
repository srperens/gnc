// Block-local Haar Wavelet Transform (2 levels, 16×16 block)
//
// One workgroup (256 threads) processes one 16×16 block.
// Two decomposition levels entirely within the block (no cross-block deps).
// Dispatch: (padded_width / 16, padded_height / 16, 1)
//
// Output layout (wavelet subband structure within each 16×16 block):
//   Level 1:  [LL1(8×8) | HL1(8×8)]
//             [LH1(8×8) | HH1(8×8)]
//
//   Level 2 (within LL1):
//             [LL2(4×4) | HL2(4×4) | HL1(8×8)]
//             [LH2(4×4) | HH2(4×4) |         ]
//             [LH1(8×8) |          | HH1(8×8)]
//
// This is identical to the standard wavelet subband layout, making it
// compatible with existing subband-aware quantization and entropy coding.

struct Params {
    width: u32,
    height: u32,
    direction: u32, // 0 = forward, 1 = inverse
    levels: u32,    // 1 or 2 decomposition levels (default 2)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> smem: array<f32, 256>;

const INV_SQRT2: f32 = 0.707106781187;

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

    if params.direction == 0u {
        // ---- FORWARD HAAR (2 levels) ----

        // Load block into smem memory
        if px < params.width && py < params.height {
            smem[thread_id] = input[py * params.width + px];
        } else {
            smem[thread_id] = 0.0;
        }
        workgroupBarrier();

        // == Level 0: full 16×16 ==

        // Row Haar: 16 elements → [8 low, 8 high]
        var val: f32;
        if col < 8u {
            let even = smem[row * 16u + col * 2u];
            let odd = smem[row * 16u + col * 2u + 1u];
            val = (even + odd) * INV_SQRT2;
        } else {
            let idx = col - 8u;
            let even = smem[row * 16u + idx * 2u];
            let odd = smem[row * 16u + idx * 2u + 1u];
            val = (even - odd) * INV_SQRT2;
        }
        workgroupBarrier();
        smem[thread_id] = val;
        workgroupBarrier();

        // Column Haar: 16 rows → [8 low rows, 8 high rows]
        if row < 8u {
            let even = smem[row * 2u * 16u + col];
            let odd = smem[(row * 2u + 1u) * 16u + col];
            val = (even + odd) * INV_SQRT2;
        } else {
            let idx = row - 8u;
            let even = smem[idx * 2u * 16u + col];
            let odd = smem[(idx * 2u + 1u) * 16u + col];
            val = (even - odd) * INV_SQRT2;
        }
        workgroupBarrier();
        smem[thread_id] = val;
        workgroupBarrier();

        // == Level 1: top-left 8×8 (LL subband) only ==
        // Threads outside the 8×8 region are idle (detail subbands are final).

        if params.levels >= 2u {
            // Level 1 row Haar: 8 elements → [4 low, 4 high]
            // Barriers unconditional — only 8×8 threads compute + write.
            var l1r_fwd = val; // preserve for non-active threads
            if row < 8u && col < 8u {
                if col < 4u {
                    let even = smem[row * 16u + col * 2u];
                    let odd = smem[row * 16u + col * 2u + 1u];
                    l1r_fwd = (even + odd) * INV_SQRT2;
                } else {
                    let idx = col - 4u;
                    let even = smem[row * 16u + idx * 2u];
                    let odd = smem[row * 16u + idx * 2u + 1u];
                    l1r_fwd = (even - odd) * INV_SQRT2;
                }
            }
            workgroupBarrier();
            if row < 8u && col < 8u {
                smem[thread_id] = l1r_fwd;
            }
            workgroupBarrier();

            // Level 1 col Haar: 8 rows → [4 low rows, 4 high rows]
            var l1c_fwd = smem[thread_id];
            if row < 8u && col < 8u {
                if row < 4u {
                    let even = smem[row * 2u * 16u + col];
                    let odd = smem[(row * 2u + 1u) * 16u + col];
                    l1c_fwd = (even + odd) * INV_SQRT2;
                } else {
                    let idx = row - 4u;
                    let even = smem[idx * 2u * 16u + col];
                    let odd = smem[(idx * 2u + 1u) * 16u + col];
                    l1c_fwd = (even - odd) * INV_SQRT2;
                }
            }
            workgroupBarrier();
            if row < 8u && col < 8u {
                smem[thread_id] = l1c_fwd;
            }
            workgroupBarrier();
        }

        // Write output
        if px < params.width && py < params.height {
            output[py * params.width + px] = smem[thread_id];
        }

    } else {
        // ---- INVERSE HAAR (2 levels) ----
        //
        // All barriers are at the top level (unconditional) to avoid any
        // WGSL compiler issues with barriers in divergent branches.

        // Load coefficients into smem memory
        if px < params.width && py < params.height {
            smem[thread_id] = input[py * params.width + px];
        } else {
            smem[thread_id] = 0.0;
        }
        workgroupBarrier();

        // == Inverse Level 1: reconstruct 8×8 LL from 4×4 subbands ==

        if params.levels >= 2u {
            // Level 1 col inverse: read before barrier, write after barrier
            // Only threads in the 8×8 region compute; others preserve smem.
            var l1c = 0.0;
            if row < 8u && col < 8u {
                let src_i = row / 2u;
                let lo = smem[src_i * 16u + col];
                let hi = smem[(src_i + 4u) * 16u + col];
                if row % 2u == 0u {
                    l1c = (lo + hi) * INV_SQRT2;
                } else {
                    l1c = (lo - hi) * INV_SQRT2;
                }
            }
            workgroupBarrier();
            if row < 8u && col < 8u {
                smem[thread_id] = l1c;
            }
            workgroupBarrier();

            // Level 1 row inverse
            var l1r = 0.0;
            if row < 8u && col < 8u {
                let src_j = col / 2u;
                let lo = smem[row * 16u + src_j];
                let hi = smem[row * 16u + src_j + 4u];
                if col % 2u == 0u {
                    l1r = (lo + hi) * INV_SQRT2;
                } else {
                    l1r = (lo - hi) * INV_SQRT2;
                }
            }
            workgroupBarrier();
            if row < 8u && col < 8u {
                smem[thread_id] = l1r;
            }
            workgroupBarrier();
        }

        // == Inverse Level 0: reconstruct full 16×16 ==

        // Level 0 col inverse
        let src_i0 = row / 2u;
        let low_v0 = smem[src_i0 * 16u + col];
        let high_v0 = smem[(src_i0 + 8u) * 16u + col];
        var l0c: f32;
        if row % 2u == 0u {
            l0c = (low_v0 + high_v0) * INV_SQRT2;
        } else {
            l0c = (low_v0 - high_v0) * INV_SQRT2;
        }
        workgroupBarrier();
        smem[thread_id] = l0c;
        workgroupBarrier();

        // Level 0 row inverse
        let src_j0 = col / 2u;
        let low_r0 = smem[row * 16u + src_j0];
        let high_r0 = smem[row * 16u + src_j0 + 8u];
        var result: f32;
        if col % 2u == 0u {
            result = (low_r0 + high_r0) * INV_SQRT2;
        } else {
            result = (low_r0 - high_r0) * INV_SQRT2;
        }

        // Write reconstructed pixel
        if px < params.width && py < params.height {
            output[py * params.width + px] = result;
        }
    }
}
