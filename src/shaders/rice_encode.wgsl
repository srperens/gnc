// GPU Rice encoder — 256 interleaved streams per tile.
//
// Significance map + Golomb-Rice coding. Fully parallel: no state chain.
// Each thread encodes one of 256 independent bit streams.
//
// Phase 1: Cooperatively compute optimal Rice parameter k per subband group.
// Phase 2: Each thread encodes its stream's coefficients.
//
// Per coefficient:
//   Zero:     1 bit (0)
//   Non-zero: 1 bit (1) + 1 sign bit + unary(quotient) + k bits (remainder)
//   where quotient = (|val|-1) >> k, remainder = (|val|-1) & ((1<<k)-1)

const STREAMS_PER_TILE: u32 = 256u;
const MAX_STREAM_BYTES: u32 = 512u;
const MAX_STREAM_WORDS: u32 = 128u;  // MAX_STREAM_BYTES / 4
const MAX_GROUPS: u32 = 8u;
const K_STRIDE: u32 = 16u;  // MAX_GROUPS * 2: stride per tile in k_output (mag k + zrl k per group)

struct Params {
    num_tiles: u32,
    coefficients_per_tile: u32,
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    num_levels: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> stream_output: array<u32>;
@group(0) @binding(3) var<storage, read_write> stream_lengths: array<u32>;
@group(0) @binding(4) var<storage, read_write> k_output: array<u32>;

// Shared memory for Phase 1: k computation
var<workgroup> group_sum: array<atomic<u32>, 8>;
var<workgroup> group_count: array<atomic<u32>, 8>;
var<workgroup> shared_k: array<u32, 8>;
// Per-subband ZRL stats
var<workgroup> zrl_sum: array<atomic<u32>, 8>;
var<workgroup> zrl_count: array<atomic<u32>, 8>;
var<workgroup> shared_k_zrl: array<u32, 8>;

// Per-thread local accumulators for Phase 1 (reduces atomic contention 32×)
var<private> p_local_sum: array<u32, 8>;
var<private> p_local_count: array<u32, 8>;
var<private> p_local_zrl_sum: array<u32, 8>;
var<private> p_local_zrl_count: array<u32, 8>;

// Per-thread bit-packing state
var<private> p_bit_buffer: u32;
var<private> p_bits_in_buffer: u32;
var<private> p_word_buffer: u32;
var<private> p_bytes_in_word: u32;
var<private> p_word_pos: u32;
var<private> p_stream_word_base: u32;
var<private> p_total_bytes: u32;

// Directional subband grouping — matches CPU compute_subband_group exactly.
fn compute_subband_group(lx: u32, ly: u32) -> u32 {
    var region = params.tile_size;
    for (var level = 0u; level < params.num_levels; level++) {
        let half = region / 2u;
        if (lx >= half || ly >= half) {
            let lfd = params.num_levels - 1u - level;
            if (lfd == 0u) {
                return 1u;
            }
            let is_hh = (lx >= half) && (ly >= half);
            let base = 2u + (lfd - 1u) * 2u;
            return select(base, base + 1u, is_hh);
        }
        region = half;
    }
    return 0u;
}

// Emit a complete byte to the stream output buffer.
fn emit_byte(byte_val: u32) {
    p_word_buffer = p_word_buffer | ((byte_val & 0xFFu) << (p_bytes_in_word * 8u));
    p_bytes_in_word += 1u;
    p_total_bytes += 1u;
    if (p_bytes_in_word == 4u) {
        if (p_word_pos < MAX_STREAM_WORDS) {
            stream_output[p_stream_word_base + p_word_pos] = p_word_buffer;
        }
        p_word_pos += 1u;
        p_word_buffer = 0u;
        p_bytes_in_word = 0u;
    }
}

// Emit a single bit (MSB-first packing). Flushes bytes automatically.
fn emit_bit(bit: u32) {
    p_bit_buffer = (p_bit_buffer << 1u) | (bit & 1u);
    p_bits_in_buffer += 1u;
    if (p_bits_in_buffer == 8u) {
        emit_byte(p_bit_buffer);
        p_bits_in_buffer = 0u;
        p_bit_buffer = 0u;
    }
}

// Emit multiple bits at once (MSB-first). count must be <= 15.
// Safe because bits_in_buffer <= 7, so total <= 22 < 32.
fn emit_bits(value: u32, count: u32) {
    p_bit_buffer = (p_bit_buffer << count) | (value & ((1u << count) - 1u));
    p_bits_in_buffer += count;
    while (p_bits_in_buffer >= 8u) {
        p_bits_in_buffer -= 8u;
        let bv = (p_bit_buffer >> p_bits_in_buffer) & 0xFFu;
        emit_byte(bv);
    }
    if (p_bits_in_buffer > 0u) {
        p_bit_buffer = p_bit_buffer & ((1u << p_bits_in_buffer) - 1u);
    } else {
        p_bit_buffer = 0u;
    }
}

// Flush remaining bits (pad to byte boundary) and partial word.
fn flush_remaining() {
    if (p_bits_in_buffer > 0u) {
        let byte_val = (p_bit_buffer << (8u - p_bits_in_buffer)) & 0xFFu;
        emit_byte(byte_val);
    }
    if (p_bytes_in_word > 0u && p_word_pos < MAX_STREAM_WORDS) {
        stream_output[p_stream_word_base + p_word_pos] = p_word_buffer;
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let thread_id = lid.x;
    let tile_id = wid.x;

    if (tile_id >= params.num_tiles) {
        return;
    }

    let tile_x = tile_id % params.tiles_x;
    let tile_y = tile_id / params.tiles_x;
    let tile_origin_x = tile_x * params.tile_size;
    let tile_origin_y = tile_y * params.tile_size;

    let symbols_per_stream = params.coefficients_per_tile / STREAMS_PER_TILE;
    let num_groups = max(1u, params.num_levels * 2u);

    // === Phase 1: Compute optimal k per subband group + k_zrl ===

    // Initialize shared accumulators
    if (thread_id < MAX_GROUPS) {
        atomicStore(&group_sum[thread_id], 0u);
        atomicStore(&group_count[thread_id], 0u);
        atomicStore(&zrl_sum[thread_id], 0u);
        atomicStore(&zrl_count[thread_id], 0u);
    }

    // Initialize per-thread local accumulators
    for (var g = 0u; g < MAX_GROUPS; g++) {
        p_local_sum[g] = 0u;
        p_local_count[g] = 0u;
        p_local_zrl_sum[g] = 0u;
        p_local_zrl_count[g] = 0u;
    }
    workgroupBarrier();

    // Each thread scans its stream using LOCAL accumulators (no atomic contention)
    {
        var zero_run = 0u;
        var zrl_group = 0u;
        for (var s = 0u; s < symbols_per_stream; s++) {
            let coeff_idx = thread_id + s * STREAMS_PER_TILE;
            let tile_row = coeff_idx / params.tile_size;
            let tile_col = coeff_idx % params.tile_size;
            let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                          + (tile_origin_x + tile_col);
            let coeff = i32(round(input[plane_idx]));

            if (coeff != 0) {
                let abs_val = u32(abs(coeff)) - 1u;
                let g = compute_subband_group(tile_col, tile_row);
                p_local_sum[g] += min(abs_val, 65535u);
                p_local_count[g] += 1u;
                if (zero_run > 0u) {
                    p_local_zrl_sum[zrl_group] += min(zero_run - 1u, 65535u);
                    p_local_zrl_count[zrl_group] += 1u;
                    zero_run = 0u;
                }
            } else {
                if (zero_run == 0u) {
                    zrl_group = compute_subband_group(tile_col, tile_row);
                }
                zero_run += 1u;
            }
        }
        if (zero_run > 0u) {
            p_local_zrl_sum[zrl_group] += min(zero_run - 1u, 65535u);
            p_local_zrl_count[zrl_group] += 1u;
        }
    }

    // Reduce per-thread locals to shared memory (one atomicAdd per group per thread)
    for (var g = 0u; g < MAX_GROUPS; g++) {
        if (p_local_count[g] > 0u) {
            atomicAdd(&group_sum[g], p_local_sum[g]);
            atomicAdd(&group_count[g], p_local_count[g]);
        }
        if (p_local_zrl_count[g] > 0u) {
            atomicAdd(&zrl_sum[g], p_local_zrl_sum[g]);
            atomicAdd(&zrl_count[g], p_local_zrl_count[g]);
        }
    }
    workgroupBarrier();

    // Compute k values
    if (thread_id < num_groups) {
        let count = atomicLoad(&group_count[thread_id]);
        if (count > 0u) {
            let mean = atomicLoad(&group_sum[thread_id]) / count;
            if (mean == 0u) {
                shared_k[thread_id] = 0u;
            } else {
                shared_k[thread_id] = min(31u - countLeadingZeros(mean), 15u);
            }
        } else {
            shared_k[thread_id] = 0u;
        }
    }
    // Compute per-subband k_zrl values
    if (thread_id < num_groups) {
        let zc = atomicLoad(&zrl_count[thread_id]);
        if (zc > 0u) {
            let zmean = atomicLoad(&zrl_sum[thread_id]) / zc;
            if (zmean == 0u) {
                shared_k_zrl[thread_id] = 0u;
            } else {
                shared_k_zrl[thread_id] = min(31u - countLeadingZeros(zmean), 15u);
            }
        } else {
            shared_k_zrl[thread_id] = 0u;
        }
    }
    workgroupBarrier();

    // Write k values (mag + zrl) to output (stride K_STRIDE = MAX_GROUPS * 2)
    if (thread_id < num_groups) {
        k_output[tile_id * K_STRIDE + thread_id] = shared_k[thread_id];
        k_output[tile_id * K_STRIDE + MAX_GROUPS + thread_id] = shared_k_zrl[thread_id];
    }

    // === Phase 2: Encode stream with per-subband ZRL ===

    // Initialize per-thread state
    p_stream_word_base = (tile_id * STREAMS_PER_TILE + thread_id) * MAX_STREAM_WORDS;
    p_bit_buffer = 0u;
    p_bits_in_buffer = 0u;
    p_word_buffer = 0u;
    p_bytes_in_word = 0u;
    p_word_pos = 0u;
    p_total_bytes = 0u;

    var s = 0u;
    while (s < symbols_per_stream) {
        let coeff_idx = thread_id + s * STREAMS_PER_TILE;
        let tile_row = coeff_idx / params.tile_size;
        let tile_col = coeff_idx % params.tile_size;
        let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                      + (tile_origin_x + tile_col);
        let coeff = i32(round(input[plane_idx]));

        if (coeff == 0) {
            let g_zrl = compute_subband_group(tile_col, tile_row);
            let k_zrl = shared_k_zrl[g_zrl];

            // Count zero run
            let max_run = 32u << k_zrl;
            var run = 1u;
            var ns = s + 1u;
            while (ns < symbols_per_stream && run < max_run) {
                let ni = thread_id + ns * STREAMS_PER_TILE;
                let nr = ni / params.tile_size;
                let nc = ni % params.tile_size;
                let np = (tile_origin_y + nr) * params.plane_width + (tile_origin_x + nc);
                if (i32(round(input[np])) != 0) {
                    break;
                }
                run += 1u;
                ns += 1u;
            }

            // Encode zero run: [0] [unary(rq)] [0] [remainder, k_zrl bits]
            let run_val = run - 1u;
            let rq = run_val >> k_zrl;
            let run_remainder = run_val & ((1u << k_zrl) - 1u);

            // Batch emit when total bits fit in 15 (rq <= 12 with k_zrl <= 2)
            let total_zrl_bits = 2u + rq + k_zrl;
            if (total_zrl_bits <= 15u) {
                // Combined: [0] [rq 1-bits] [0] [k_zrl remainder bits]
                var combined = ((1u << rq) - 1u) << (k_zrl + 1u);
                combined |= run_remainder;
                emit_bits(combined, total_zrl_bits);
            } else {
                // Fallback for large quotients
                emit_bit(0u);
                var rq_rem = rq;
                while (rq_rem > 0u) {
                    let chunk = min(rq_rem, 15u);
                    emit_bits((1u << chunk) - 1u, chunk);
                    rq_rem -= chunk;
                }
                emit_bit(0u);
                if (k_zrl > 0u) {
                    emit_bits(run_remainder, k_zrl);
                }
            }

            s += run;
        } else {
            let sign_bit = select(0u, 1u, coeff < 0);
            let magnitude = u32(abs(coeff)) - 1u;
            let g = compute_subband_group(tile_col, tile_row);
            let k = shared_k[g];
            let quotient = min(magnitude >> k, 31u);
            let remainder = magnitude & ((1u << k) - 1u);

            // Batch emit when total bits fit in 15
            let total_bits = 3u + quotient + k;
            if (total_bits <= 15u) {
                // Combined: [1] [sign] [quotient 1-bits] [0] [k remainder bits]
                var combined = 1u << (2u + quotient + k); // significance bit
                combined |= sign_bit << (1u + quotient + k);
                combined |= ((1u << quotient) - 1u) << (k + 1u); // unary
                // stop bit at position k is 0 (implicit)
                combined |= remainder;
                emit_bits(combined, total_bits);
            } else {
                // Fallback for large quotients
                emit_bit(1u);
                emit_bit(sign_bit);
                var q_remaining = quotient;
                while (q_remaining > 0u) {
                    let chunk = min(q_remaining, 15u);
                    emit_bits((1u << chunk) - 1u, chunk);
                    q_remaining -= chunk;
                }
                emit_bit(0u);
                if (k > 0u) {
                    emit_bits(remainder, k);
                }
            }

            s += 1u;
        }
    }

    flush_remaining();

    // Write stream byte count
    stream_lengths[tile_id * STREAMS_PER_TILE + thread_id] = p_total_bytes;
}
