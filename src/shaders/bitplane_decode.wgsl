// GPU bitplane decoder — one workgroup per 32x32 block.
//
// Each thread in the workgroup is responsible for one coefficient in the block.
// The bitplane data is read cooperatively: thread 0 reads the all-zero flags
// and broadcasts via shared memory, then each thread reads its own bit from
// the significance bitmap.
//
// This is fully parallel with no serial dependencies between coefficients.
// Patent-free: bitplane coding predates all modern codec patents.

const BLOCK_SIZE: u32 = 32u;
const COEFFS_PER_BLOCK: u32 = 1024u; // 32 * 32

// Must match host-side BITPLANE_TILE_INFO_STRIDE
const TILE_INFO_STRIDE: u32 = 8u;
// Must match host-side BITPLANE_BLOCK_INFO_STRIDE
const BLOCK_INFO_STRIDE: u32 = 4u;

struct Params {
    num_tiles: u32,
    coefficients_per_tile: u32,
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    block_size: u32,
    blocks_per_tile_side: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tile_info: array<u32>;
@group(0) @binding(2) var<storage, read> block_info: array<u32>;
@group(0) @binding(3) var<storage, read> bitplane_data: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Shared memory for broadcasting bitplane metadata and sign bit counts.
// We use shared memory to broadcast per-bitplane all-zero flags efficiently.
var<workgroup> shared_max_bitplane: u32;
var<workgroup> shared_data_byte_offset: u32;

// Read a single bit from the packed bitplane data.
// Bits are MSB-first within each byte, bytes are little-endian packed into u32s.
// global_bit_offset = byte_offset * 8 + bit_within_byte
fn read_bit(byte_offset: u32, bit_index: u32) -> u32 {
    let abs_byte = byte_offset + (bit_index >> 3u);
    let bit_in_byte = bit_index & 7u;
    // Bytes are packed LE into u32s
    let word_idx = abs_byte >> 2u;
    let byte_pos = abs_byte & 3u;
    let byte_val = (bitplane_data[word_idx] >> (byte_pos * 8u)) & 0xFFu;
    // MSB-first within byte
    return (byte_val >> (7u - bit_in_byte)) & 1u;
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let thread_id = lid.x;
    let global_block_id = wid.x;

    // Each thread handles 4 coefficients (256 threads * 4 = 1024 coefficients per block)
    // This keeps workgroup size at 256 for broad GPU compatibility.
    let coeff_base = thread_id * 4u;

    // Load block metadata (all threads read from global memory — it's cached)
    let block_base = global_block_id * BLOCK_INFO_STRIDE;
    let max_bitplane = block_info[block_base];
    let data_byte_offset = block_info[block_base + 1u];

    // Early exit for all-zero blocks
    if max_bitplane == 0u {
        // Determine which tile this block belongs to and write zeros
        // (output buffer may already be zeroed, but we write explicitly for correctness)
        let blocks_per_tile = params.blocks_per_tile_side * params.blocks_per_tile_side;
        let tile_id = global_block_id / blocks_per_tile;
        let block_in_tile = global_block_id % blocks_per_tile;
        let tile_base = tile_id * TILE_INFO_STRIDE;
        let tile_x = tile_info[tile_base + 2u];
        let tile_y = tile_info[tile_base + 3u];
        let block_bx = block_in_tile % params.blocks_per_tile_side;
        let block_by = block_in_tile / params.blocks_per_tile_side;
        let block_origin_x = tile_x * params.tile_size + block_bx * BLOCK_SIZE;
        let block_origin_y = tile_y * params.tile_size + block_by * BLOCK_SIZE;

        for (var c = 0u; c < 4u; c++) {
            let coeff_idx = coeff_base + c;
            if coeff_idx >= COEFFS_PER_BLOCK { break; }
            let ly = coeff_idx / BLOCK_SIZE;
            let lx = coeff_idx % BLOCK_SIZE;
            let plane_idx = (block_origin_y + ly) * params.plane_width + (block_origin_x + lx);
            output[plane_idx] = 0.0;
        }
        return;
    }

    // Determine tile and block position
    let blocks_per_tile = params.blocks_per_tile_side * params.blocks_per_tile_side;
    let tile_id = global_block_id / blocks_per_tile;
    let block_in_tile = global_block_id % blocks_per_tile;
    let tile_base = tile_id * TILE_INFO_STRIDE;
    let tile_x = tile_info[tile_base + 2u];
    let tile_y = tile_info[tile_base + 3u];
    let block_bx = block_in_tile % params.blocks_per_tile_side;
    let block_by = block_in_tile / params.blocks_per_tile_side;
    let block_origin_x = tile_x * params.tile_size + block_bx * BLOCK_SIZE;
    let block_origin_y = tile_y * params.tile_size + block_by * BLOCK_SIZE;

    // Decode bitplanes for this thread's 4 coefficients.
    // Each thread reads its own bits from the bitplane data.
    //
    // Bitplane data layout (bit stream, MSB-first):
    //   For each bitplane p from (max_bitplane-1) down to 0:
    //     [all_zero_flag: 1 bit]
    //     If not all-zero:
    //       [significance_bitmap: 1024 bits]
    //   [sign_bits: one per non-zero coefficient, in raster order]

    var magnitudes: array<u32, 4>;
    for (var c = 0u; c < 4u; c++) {
        magnitudes[c] = 0u;
    }

    // First pass: scan through bitplanes to reconstruct magnitudes.
    // We need to walk the bit stream sequentially to find where each bitplane's
    // data starts, since all-zero flags cause variable-length encoding.
    var bit_cursor = 0u;

    for (var plane_idx = 0u; plane_idx < max_bitplane; plane_idx++) {
        let p = max_bitplane - 1u - plane_idx; // MSB to LSB
        let bit_value = 1u << p;

        // Read the all-zero flag
        let all_zero = read_bit(data_byte_offset, bit_cursor);
        bit_cursor += 1u;

        if all_zero == 1u {
            continue; // skip — all zeros in this bitplane
        }

        // Read this thread's bits from the significance bitmap
        for (var c = 0u; c < 4u; c++) {
            let coeff_idx = coeff_base + c;
            if coeff_idx >= COEFFS_PER_BLOCK { break; }
            let bit = read_bit(data_byte_offset, bit_cursor + coeff_idx);
            if bit == 1u {
                magnitudes[c] |= bit_value;
            }
        }

        bit_cursor += COEFFS_PER_BLOCK; // advance past the full bitmap
    }

    // Second pass: count total non-zero coefficients before our positions
    // to find where our sign bits are in the sign section.
    // We need to count all non-zero magnitudes before each of our coefficients.
    //
    // Since we can't efficiently do a prefix-sum across all 1024 coefficients
    // without shared memory, we use a simpler approach: each thread reads the
    // sign bits by counting non-zero coefficients before its own positions.
    //
    // We walk the full coefficient range and count non-zeros. This is O(1024)
    // per thread but simple and correct. For a 256-thread workgroup, each thread
    // handles 4 coefficients, so the total work is 4*1024 reads from bitplane data,
    // which is actually quite fast since it's all in the same cache lines.
    //
    // Optimization: we can compute the sign bit offset by scanning the
    // significance bitmaps (which we already have in memory) rather than
    // re-reading magnitudes.

    // Re-scan to count non-zero coefficients before each of our 4 positions.
    // We need the total count of non-zero coefficients in raster order up to
    // but not including each of our coefficients.
    var signs: array<u32, 4>;
    for (var c = 0u; c < 4u; c++) {
        signs[c] = 0u;
    }

    // Count non-zero coefficients before our first coefficient by reading
    // from the bitplane data. We reconstruct whether each prior coefficient
    // is non-zero by OR-ing all its bitplane bits.
    //
    // Actually, it's simpler to count from the bitplane data: a coefficient
    // is non-zero if ANY of its bitplane bits is 1. We can check this by
    // re-reading the significance bitmaps.

    // Precompute: for each of our 4 coefficients, count how many non-zero
    // coefficients come before it in raster order.
    var nz_before: array<u32, 4>;
    for (var c = 0u; c < 4u; c++) {
        nz_before[c] = 0u;
    }

    // Walk through all coefficients 0..1023, checking if each is non-zero
    // by testing all bitplanes.
    for (var i = 0u; i < COEFFS_PER_BLOCK; i++) {
        // Check if coefficient i is non-zero by scanning bitplanes
        var is_nonzero = false;
        var scan_cursor = 0u;

        for (var plane_idx = 0u; plane_idx < max_bitplane; plane_idx++) {
            let all_zero = read_bit(data_byte_offset, scan_cursor);
            scan_cursor += 1u;

            if all_zero == 1u {
                continue;
            }

            // Read bit for coefficient i
            let bit = read_bit(data_byte_offset, scan_cursor + i);
            if bit == 1u {
                is_nonzero = true;
            }
            scan_cursor += COEFFS_PER_BLOCK;
        }

        // Update nz_before for each of our coefficients that comes after i
        if is_nonzero {
            for (var c = 0u; c < 4u; c++) {
                let coeff_idx = coeff_base + c;
                if i < coeff_idx {
                    nz_before[c] += 1u;
                }
            }
        }
    }

    // Now read sign bits from the sign section
    // The sign section starts after all bitplane data, at bit_cursor
    for (var c = 0u; c < 4u; c++) {
        let coeff_idx = coeff_base + c;
        if coeff_idx >= COEFFS_PER_BLOCK { break; }

        if magnitudes[c] > 0u {
            let sign_bit_idx = bit_cursor + nz_before[c];
            signs[c] = read_bit(data_byte_offset, sign_bit_idx);
        }
    }

    // Write output
    for (var c = 0u; c < 4u; c++) {
        let coeff_idx = coeff_base + c;
        if coeff_idx >= COEFFS_PER_BLOCK { break; }

        let ly = coeff_idx / BLOCK_SIZE;
        let lx = coeff_idx % BLOCK_SIZE;
        let plane_idx = (block_origin_y + ly) * params.plane_width + (block_origin_x + lx);

        if magnitudes[c] == 0u {
            output[plane_idx] = 0.0;
        } else if signs[c] == 1u {
            output[plane_idx] = -f32(magnitudes[c]);
        } else {
            output[plane_idx] = f32(magnitudes[c]);
        }
    }
}
