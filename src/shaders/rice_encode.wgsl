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
//
// #53: 2-state k_zrl context based on MAGNITUDE of the preceding nonzero coefficient.
// Zero-run lengths are encoded with one of two Rice parameters:
//   k_zrl_large — preceding nonzero had |coeff| >= 2 (large → clustered, short runs expected)
//   k_zrl_small — preceding nonzero had |coeff| == 1 (small/isolated → long runs expected)
// Note: context "after zero run / start" maps to k_zrl_small since no magnitude is
// available. The encoder gathers two separate run-length histograms in Phase 1 and
// selects the appropriate k in Phase 2 via a per-stream boolean state.

const STREAMS_PER_TILE: u32 = 256u;
const MAX_GROUPS: u32 = 8u;
// K_STRIDE per tile: [k_mag ×8][k_zrl_nz ×8][k_zrl_z ×8][skip_bitmap ×1] = 25
const K_STRIDE: u32 = 25u;

// Field order must stay in sync with rice_gpu.rs RiceParams (bytemuck::Pod).
struct Params {
    num_tiles: u32,
    coefficients_per_tile: u32,
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    num_levels: u32,
    // Per-stream byte ceiling; must be divisible by 4.
    // Set by CPU via max_stream_bytes_for_tile(): 1024 for 128×128, 4096 for 256×256.
    max_stream_bytes: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> stream_output: array<u32>;
@group(0) @binding(3) var<storage, read_write> stream_lengths: array<u32>;
@group(0) @binding(4) var<storage, read_write> k_output: array<u32>;
// One u32 per tile; set to 1 if any stream in that tile overflows max_stream_bytes.
@group(0) @binding(5) var<storage, read_write> overflow_flags: array<atomic<u32>>;

// Shared memory for Phase 1: k computation
var<workgroup> group_sum: array<atomic<u32>, 8>;
var<workgroup> group_count: array<atomic<u32>, 8>;
var<workgroup> shared_k: array<u32, 8>;
// Per-subband ZRL stats — two contexts: after-nonzero (nz) and after-zero/start (z)
var<workgroup> zrl_sum_nz: array<atomic<u32>, 8>;
var<workgroup> zrl_count_nz: array<atomic<u32>, 8>;
var<workgroup> zrl_sum_z: array<atomic<u32>, 8>;
var<workgroup> zrl_count_z: array<atomic<u32>, 8>;
var<workgroup> shared_k_zrl_nz: array<u32, 8>;
var<workgroup> shared_k_zrl_z: array<u32, 8>;
// Subband skip bitmap: bit g = 1 means all coefficients in group g are zero
var<workgroup> shared_skip_bitmap: u32;
// #checkerboard-ctx: even threads expose their final EMA mean to adjacent odd threads.
// shared_ctx_even[even_idx][group] = EMA mean (p_ema[g] >> 4) after encoding.
// Size: 128 × 8 × 4 = 4096 bytes. Total workgroup mem stays well below 32 KB.
var<workgroup> shared_ctx_even: array<array<u32, 8>, 128>;

// Per-thread local accumulators for Phase 1 (reduces atomic contention 32×)
var<private> p_local_sum: array<u32, 8>;
var<private> p_local_count: array<u32, 8>;
var<private> p_local_zrl_sum_nz: array<u32, 8>;
var<private> p_local_zrl_count_nz: array<u32, 8>;
var<private> p_local_zrl_sum_z: array<u32, 8>;
var<private> p_local_zrl_count_z: array<u32, 8>;

// Per-thread bit-packing state
var<private> p_bit_buffer: u32;
var<private> p_bits_in_buffer: u32;
var<private> p_word_buffer: u32;
var<private> p_bytes_in_word: u32;
var<private> p_word_pos: u32;
var<private> p_stream_word_base: u32;
var<private> p_total_bytes: u32;

// Per-thread EMA state for adaptive k (fixed-point ×16, window ≈ 8 coefficients)
var<private> p_ema: array<u32, 8>;

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
// p_tile_id must be set before calling (used for overflow signaling).
var<private> p_tile_id: u32;

fn emit_byte(byte_val: u32) {
    p_word_buffer = p_word_buffer | ((byte_val & 0xFFu) << (p_bytes_in_word * 8u));
    p_bytes_in_word += 1u;
    p_total_bytes += 1u;
    if (p_bytes_in_word == 4u) {
        let max_words = params.max_stream_bytes / 4u;
        if (p_word_pos < max_words) {
            stream_output[p_stream_word_base + p_word_pos] = p_word_buffer;
        } else {
            // Overflow: stream exceeded max_stream_bytes. Signal the tile.
            atomicStore(&overflow_flags[p_tile_id], 1u);
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
    let max_words = params.max_stream_bytes / 4u;
    if (p_bytes_in_word > 0u && p_word_pos < max_words) {
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

    // === Phase 1: Compute optimal k per subband group + 2-state k_zrl ===

    // Initialize shared accumulators
    if (thread_id < MAX_GROUPS) {
        atomicStore(&group_sum[thread_id], 0u);
        atomicStore(&group_count[thread_id], 0u);
        atomicStore(&zrl_sum_nz[thread_id], 0u);
        atomicStore(&zrl_count_nz[thread_id], 0u);
        atomicStore(&zrl_sum_z[thread_id], 0u);
        atomicStore(&zrl_count_z[thread_id], 0u);
    }

    // Initialize per-thread local accumulators
    for (var g = 0u; g < MAX_GROUPS; g++) {
        p_local_sum[g] = 0u;
        p_local_count[g] = 0u;
        p_local_zrl_sum_nz[g] = 0u;
        p_local_zrl_count_nz[g] = 0u;
        p_local_zrl_sum_z[g] = 0u;
        p_local_zrl_count_z[g] = 0u;
    }
    workgroupBarrier();

    // Each thread scans its stream using LOCAL accumulators (no atomic contention).
    // Context for k_zrl selection: was the preceding nonzero coefficient |coeff| >= 2?
    // "Large" (|coeff|>=2) → spatially clustered → shorter zero runs → k_zrl_nz (smaller k).
    // "Small" (|coeff|==1) or start-of-stream → isolated → longer zero runs → k_zrl_z (larger k).
    {
        var zero_run = 0u;
        var zrl_group = 0u;
        // zrl_ctx_large: true if the nonzero that preceded this run had abs_val >= 1 (|coeff|>=2)
        var zrl_ctx_large: bool = false; // context at start of current zero run
        var last_mag_large: bool = false; // true after a "large" nonzero coeff
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
                    // Zero run ended; attribute to the correct context bucket.
                    let rv = min(zero_run - 1u, 65535u);
                    if (zrl_ctx_large) {
                        p_local_zrl_sum_nz[zrl_group] += rv;
                        p_local_zrl_count_nz[zrl_group] += 1u;
                    } else {
                        p_local_zrl_sum_z[zrl_group] += rv;
                        p_local_zrl_count_z[zrl_group] += 1u;
                    }
                    zero_run = 0u;
                }
                // Update context: |coeff| >= 2 ↔ abs_val >= 1
                last_mag_large = (abs_val >= 1u);
            } else {
                if (zero_run == 0u) {
                    // First zero of a new run: record subband and magnitude context.
                    zrl_group = compute_subband_group(tile_col, tile_row);
                    zrl_ctx_large = last_mag_large;
                }
                zero_run += 1u;
            }
        }
        // Trailing zero run at end of stream
        if (zero_run > 0u) {
            let rv = min(zero_run - 1u, 65535u);
            if (zrl_ctx_large) {
                p_local_zrl_sum_nz[zrl_group] += rv;
                p_local_zrl_count_nz[zrl_group] += 1u;
            } else {
                p_local_zrl_sum_z[zrl_group] += rv;
                p_local_zrl_count_z[zrl_group] += 1u;
            }
        }
    }

    // Reduce per-thread locals to shared memory (one atomicAdd per group per thread)
    for (var g = 0u; g < MAX_GROUPS; g++) {
        if (p_local_count[g] > 0u) {
            atomicAdd(&group_sum[g], p_local_sum[g]);
            atomicAdd(&group_count[g], p_local_count[g]);
        }
        if (p_local_zrl_count_nz[g] > 0u) {
            atomicAdd(&zrl_sum_nz[g], p_local_zrl_sum_nz[g]);
            atomicAdd(&zrl_count_nz[g], p_local_zrl_count_nz[g]);
        }
        if (p_local_zrl_count_z[g] > 0u) {
            atomicAdd(&zrl_sum_z[g], p_local_zrl_sum_z[g]);
            atomicAdd(&zrl_count_z[g], p_local_zrl_count_z[g]);
        }
    }
    workgroupBarrier();

    // Compute magnitude k values
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
    // Compute k_zrl_nz (zero runs that follow a nonzero coefficient)
    if (thread_id < num_groups) {
        let zc = atomicLoad(&zrl_count_nz[thread_id]);
        if (zc > 0u) {
            let zmean = atomicLoad(&zrl_sum_nz[thread_id]) / zc;
            shared_k_zrl_nz[thread_id] = select(0u, min(31u - countLeadingZeros(zmean), 15u), zmean > 0u);
        } else {
            shared_k_zrl_nz[thread_id] = 0u;
        }
    }
    // Compute k_zrl_z (zero runs that follow another zero run or start-of-stream)
    if (thread_id < num_groups) {
        let zc = atomicLoad(&zrl_count_z[thread_id]);
        if (zc > 0u) {
            let zmean = atomicLoad(&zrl_sum_z[thread_id]) / zc;
            shared_k_zrl_z[thread_id] = select(0u, min(31u - countLeadingZeros(zmean), 15u), zmean > 0u);
        } else {
            shared_k_zrl_z[thread_id] = 0u;
        }
    }
    workgroupBarrier();

    // Compute subband skip bitmap: bit g = 1 → all coefficients in group g are zero
    if (thread_id == 0u) {
        var bitmap = 0u;
        for (var g = 0u; g < num_groups; g++) {
            if (atomicLoad(&group_count[g]) == 0u) {
                bitmap |= (1u << g);
            }
        }
        shared_skip_bitmap = bitmap;
    }

    // Write k values to output.
    // Layout: [k_mag ×8][k_zrl_nz ×8][k_zrl_z ×8][skip_bitmap] = K_STRIDE=25
    if (thread_id < num_groups) {
        k_output[tile_id * K_STRIDE + thread_id] = shared_k[thread_id];
        k_output[tile_id * K_STRIDE + MAX_GROUPS + thread_id] = shared_k_zrl_nz[thread_id];
        k_output[tile_id * K_STRIDE + MAX_GROUPS * 2u + thread_id] = shared_k_zrl_z[thread_id];
    }
    if (thread_id == 0u) {
        k_output[tile_id * K_STRIDE + K_STRIDE - 1u] = shared_skip_bitmap;
    }
    workgroupBarrier();

    // === Phase 2: Encode stream with 2-state per-subband ZRL ===
    //
    // Checkerboard k-context (#checkerboard-ctx):
    //   Step 2a — even threads (0,2,...,254) encode first using the standard EMA init.
    //             After encoding, they write their final EMA means to shared_ctx_even.
    //   workgroupBarrier() — at top level (never inside a branch) per Metal/M1 rule.
    //   Step 2b — odd threads (1,3,...,255) read their left-neighbor (even thread t-1)
    //             EMA means and warm-start their own EMA with a 50/50 blend of the
    //             global k (Phase 1) and the neighbor k (derived from neighbor mean).
    //             This captures cross-stream spatial correlation without any extra dispatch.
    //
    // The barrier is always executed by ALL threads. Only the encoding work is guarded
    // by the even/odd if-blocks so the barrier is unconditionally at the top level.

    // Initialize per-thread output state (both even and odd)
    p_tile_id = tile_id;
    p_stream_word_base = (tile_id * STREAMS_PER_TILE + thread_id) * (params.max_stream_bytes / 4u);
    p_bit_buffer = 0u;
    p_bits_in_buffer = 0u;
    p_word_buffer = 0u;
    p_bytes_in_word = 0u;
    p_word_pos = 0u;
    p_total_bytes = 0u;

    // --- Step 2a: even threads encode ---
    if (thread_id % 2u == 0u) {
        // Initialize EMA from global k seeds (standard warm-start)
        for (var gi = 0u; gi < MAX_GROUPS; gi++) {
            p_ema[gi] = max(1u, 1u << shared_k[gi]) << 4u;
        }

        var last_mag_large_e: bool = false;
        var se = 0u;
        while (se < symbols_per_stream) {
            let coeff_idx_e = thread_id + se * STREAMS_PER_TILE;
            let tile_row_e = coeff_idx_e / params.tile_size;
            let tile_col_e = coeff_idx_e % params.tile_size;

            let skip_ge = compute_subband_group(tile_col_e, tile_row_e);
            if ((shared_skip_bitmap >> skip_ge) & 1u) == 1u {
                se += 1u;
                continue;
            }

            let plane_idx_e = (tile_origin_y + tile_row_e) * params.plane_width
                            + (tile_origin_x + tile_col_e);
            let coeff_e = i32(round(input[plane_idx_e]));

            if (coeff_e == 0) {
                let g_zrl_e = skip_ge;
                let k_zrl_e = select(shared_k_zrl_z[g_zrl_e], shared_k_zrl_nz[g_zrl_e], last_mag_large_e);

                var run_e = 1u;
                var ns_e = se + 1u;
                while (ns_e < symbols_per_stream) {
                    let ni_e = thread_id + ns_e * STREAMS_PER_TILE;
                    let nr_e = ni_e / params.tile_size;
                    let nc_e = ni_e % params.tile_size;
                    let ns_ge = compute_subband_group(nc_e, nr_e);
                    if ((shared_skip_bitmap >> ns_ge) & 1u) == 1u {
                        ns_e += 1u;
                        continue;
                    }
                    let np_e = (tile_origin_y + nr_e) * params.plane_width + (tile_origin_x + nc_e);
                    if (i32(round(input[np_e])) != 0) {
                        break;
                    }
                    run_e += 1u;
                    ns_e += 1u;
                }

                let run_val_e = run_e - 1u;
                let rq_e = run_val_e >> k_zrl_e;
                let run_rem_e = run_val_e & ((1u << k_zrl_e) - 1u);
                let total_zrl_e = 2u + rq_e + k_zrl_e;
                if (total_zrl_e <= 15u) {
                    var combined_e = ((1u << rq_e) - 1u) << (k_zrl_e + 1u);
                    combined_e |= run_rem_e;
                    emit_bits(combined_e, total_zrl_e);
                } else {
                    emit_bit(0u);
                    var rq_rem_e = rq_e;
                    while (rq_rem_e > 0u) {
                        let chunk_e = min(rq_rem_e, 15u);
                        emit_bits((1u << chunk_e) - 1u, chunk_e);
                        rq_rem_e -= chunk_e;
                    }
                    emit_bit(0u);
                    if (k_zrl_e > 0u) {
                        emit_bits(run_rem_e, k_zrl_e);
                    }
                }

                se = ns_e;
                last_mag_large_e = false;
            } else {
                let sign_bit_e = select(0u, 1u, coeff_e < 0);
                let magnitude_e = u32(abs(coeff_e)) - 1u;
                let ge = skip_ge;

                let ema_mean_e = p_ema[ge] >> 4u;
                let k_e = select(min(31u - countLeadingZeros(ema_mean_e), 15u), 0u, ema_mean_e == 0u);
                let quotient_e = magnitude_e >> k_e;
                let remainder_e = magnitude_e & ((1u << k_e) - 1u);

                let total_bits_e = 3u + quotient_e + k_e;
                if (total_bits_e <= 15u) {
                    var combined_e2 = 1u << (2u + quotient_e + k_e);
                    combined_e2 |= sign_bit_e << (1u + quotient_e + k_e);
                    combined_e2 |= ((1u << quotient_e) - 1u) << (k_e + 1u);
                    combined_e2 |= remainder_e;
                    emit_bits(combined_e2, total_bits_e);
                } else {
                    emit_bit(1u);
                    emit_bit(sign_bit_e);
                    var q_rem_e = quotient_e;
                    while (q_rem_e > 0u) {
                        let chunk_e2 = min(q_rem_e, 15u);
                        emit_bits((1u << chunk_e2) - 1u, chunk_e2);
                        q_rem_e -= chunk_e2;
                    }
                    emit_bit(0u);
                    if (k_e > 0u) {
                        emit_bits(remainder_e, k_e);
                    }
                }

                p_ema[ge] = p_ema[ge] - (p_ema[ge] >> 3u) + (magnitude_e << 1u);
                se += 1u;
                last_mag_large_e = (magnitude_e >= 1u);
            }
        }

        flush_remaining();
        stream_lengths[tile_id * STREAMS_PER_TILE + thread_id] = p_total_bytes;

        // Expose final EMA means to odd neighbor threads via shared memory.
        let even_idx = thread_id / 2u;
        for (var gi2 = 0u; gi2 < MAX_GROUPS; gi2++) {
            shared_ctx_even[even_idx][gi2] = p_ema[gi2] >> 4u;
        }
    }

    // Barrier at workgroup top level — NEVER inside a branch (Metal/M1 requirement).
    workgroupBarrier();

    // --- Step 2b: odd threads encode with neighbor-context warm-start ---
    if (thread_id % 2u == 1u) {
        // Read left-neighbor (even thread t-1) EMA means and blend with global k.
        // Blending: adjusted_k = (global_k + neighbor_k + 1) / 2  (round-up 50/50).
        // This biases odd streams toward the actual signal statistics of their
        // immediately adjacent even stream, reducing initial EMA overshoot.
        let even_idx_o = (thread_id - 1u) / 2u;
        for (var gi3 = 0u; gi3 < MAX_GROUPS; gi3++) {
            let neighbor_mean = shared_ctx_even[even_idx_o][gi3];
            let global_k = shared_k[gi3];
            let neighbor_k = select(
                min(31u - countLeadingZeros(neighbor_mean), 15u),
                0u,
                neighbor_mean == 0u
            );
            let adjusted_k = clamp((global_k + neighbor_k + 1u) / 2u, 0u, 15u);
            p_ema[gi3] = max(1u, 1u << adjusted_k) << 4u;
            // Decoder derives the same adjusted_k from its decoded even-stream EMA states.
            // No bitstream storage needed — context is reproduced from decoded data.
        }

        // Reset output state (was initialized above but p_ema is now adjusted)
        p_bit_buffer = 0u;
        p_bits_in_buffer = 0u;
        p_word_buffer = 0u;
        p_bytes_in_word = 0u;
        p_word_pos = 0u;
        p_total_bytes = 0u;

        var last_mag_large_o: bool = false;
        var so = 0u;
        while (so < symbols_per_stream) {
            let coeff_idx_o = thread_id + so * STREAMS_PER_TILE;
            let tile_row_o = coeff_idx_o / params.tile_size;
            let tile_col_o = coeff_idx_o % params.tile_size;

            let skip_go = compute_subband_group(tile_col_o, tile_row_o);
            if ((shared_skip_bitmap >> skip_go) & 1u) == 1u {
                so += 1u;
                continue;
            }

            let plane_idx_o = (tile_origin_y + tile_row_o) * params.plane_width
                            + (tile_origin_x + tile_col_o);
            let coeff_o = i32(round(input[plane_idx_o]));

            if (coeff_o == 0) {
                let g_zrl_o = skip_go;
                let k_zrl_o = select(shared_k_zrl_z[g_zrl_o], shared_k_zrl_nz[g_zrl_o], last_mag_large_o);

                var run_o = 1u;
                var ns_o = so + 1u;
                while (ns_o < symbols_per_stream) {
                    let ni_o = thread_id + ns_o * STREAMS_PER_TILE;
                    let nr_o = ni_o / params.tile_size;
                    let nc_o = ni_o % params.tile_size;
                    let ns_go = compute_subband_group(nc_o, nr_o);
                    if ((shared_skip_bitmap >> ns_go) & 1u) == 1u {
                        ns_o += 1u;
                        continue;
                    }
                    let np_o = (tile_origin_y + nr_o) * params.plane_width + (tile_origin_x + nc_o);
                    if (i32(round(input[np_o])) != 0) {
                        break;
                    }
                    run_o += 1u;
                    ns_o += 1u;
                }

                let run_val_o = run_o - 1u;
                let rq_o = run_val_o >> k_zrl_o;
                let run_rem_o = run_val_o & ((1u << k_zrl_o) - 1u);
                let total_zrl_o = 2u + rq_o + k_zrl_o;
                if (total_zrl_o <= 15u) {
                    var combined_o = ((1u << rq_o) - 1u) << (k_zrl_o + 1u);
                    combined_o |= run_rem_o;
                    emit_bits(combined_o, total_zrl_o);
                } else {
                    emit_bit(0u);
                    var rq_rem_o = rq_o;
                    while (rq_rem_o > 0u) {
                        let chunk_o = min(rq_rem_o, 15u);
                        emit_bits((1u << chunk_o) - 1u, chunk_o);
                        rq_rem_o -= chunk_o;
                    }
                    emit_bit(0u);
                    if (k_zrl_o > 0u) {
                        emit_bits(run_rem_o, k_zrl_o);
                    }
                }

                so = ns_o;
                last_mag_large_o = false;
            } else {
                let sign_bit_o = select(0u, 1u, coeff_o < 0);
                let magnitude_o = u32(abs(coeff_o)) - 1u;
                let go = skip_go;

                let ema_mean_o = p_ema[go] >> 4u;
                let k_o = select(min(31u - countLeadingZeros(ema_mean_o), 15u), 0u, ema_mean_o == 0u);
                let quotient_o = magnitude_o >> k_o;
                let remainder_o = magnitude_o & ((1u << k_o) - 1u);

                let total_bits_o = 3u + quotient_o + k_o;
                if (total_bits_o <= 15u) {
                    var combined_o2 = 1u << (2u + quotient_o + k_o);
                    combined_o2 |= sign_bit_o << (1u + quotient_o + k_o);
                    combined_o2 |= ((1u << quotient_o) - 1u) << (k_o + 1u);
                    combined_o2 |= remainder_o;
                    emit_bits(combined_o2, total_bits_o);
                } else {
                    emit_bit(1u);
                    emit_bit(sign_bit_o);
                    var q_rem_o = quotient_o;
                    while (q_rem_o > 0u) {
                        let chunk_o2 = min(q_rem_o, 15u);
                        emit_bits((1u << chunk_o2) - 1u, chunk_o2);
                        q_rem_o -= chunk_o2;
                    }
                    emit_bit(0u);
                    if (k_o > 0u) {
                        emit_bits(remainder_o, k_o);
                    }
                }

                p_ema[go] = p_ema[go] - (p_ema[go] >> 3u) + (magnitude_o << 1u);
                so += 1u;
                last_mag_large_o = (magnitude_o >= 1u);
            }
        }

        flush_remaining();
        stream_lengths[tile_id * STREAMS_PER_TILE + thread_id] = p_total_bytes;
    }
}
