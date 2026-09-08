// GPU Rice decoder — 256 interleaved streams per tile.
//
// Inverse of rice_encode.wgsl. Each thread decodes one independent bit stream.
// No state chain — fully parallel across all 256 streams.
//
// Per coefficient:
//   bit=0 → zero run: Rice(run_length-1, k_zrl) where k_zrl selected by 2-state context
//   bit=1 → sign bit + Rice(magnitude-1, k_mag) → ±magnitude
//
// #53: 2-state k_zrl context based on magnitude of the preceding nonzero coefficient.
// k_zrl_nz is used after a "large" nonzero (|coeff| >= 2 → clustered signal, short runs);
// k_zrl_z is used after a "small" nonzero (|coeff| == 1) or at start-of-stream.
//
// Checkerboard k-context (#checkerboard-ctx):
// Even streams (0,2,...,254) decode first, write final EMA means to shared_ctx_even.
// workgroupBarrier() — at top level (not inside branch) per Metal/M1 rule.
// Odd streams (1,3,...,255) derive adjusted k from even neighbor's decoded EMA state,
// then decode with that warm-start. No extra bitstream data — context from decoded data.

const STREAMS_PER_TILE: u32 = 256u;
const MAX_GROUPS: u32 = 12u;
// K_STRIDE per tile: [k_mag][k_zrl_nz][k_zrl_z] × MAX_GROUPS, then skip_bitmap
const K_STRIDE: u32 = MAX_GROUPS * 3u + 1u;

struct Params {
    num_tiles: u32,
    coefficients_per_tile: u32,
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    num_levels: u32,
    _pad0: u32, // encode's max_stream_bytes; unused here
    plane_height: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> k_values: array<u32>;
@group(0) @binding(2) var<storage, read> stream_data: array<u32>;
@group(0) @binding(3) var<storage, read> stream_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Shared k values for this tile
var<workgroup> shared_k: array<u32, 12>;
var<workgroup> shared_k_zrl_nz: array<u32, 12>; // k_zrl after a large nonzero (|coeff|>=2)
var<workgroup> shared_k_zrl_z: array<u32, 12>;  // k_zrl after a small nonzero (|coeff|==1) or start
// Subband skip bitmap: bit g = 1 means all coefficients in group g are zero
var<workgroup> shared_skip_bitmap: u32;
// Checkerboard context: even threads write final EMA means here after decoding.
// Odd threads read their left neighbor's EMA to derive adjusted k warm-start.
var<workgroup> shared_ctx_even: array<array<u32, 12>, 128>;

// Per-thread bit-reader state.
//
// A 32-bit window, MSB-first: bit 31 of `p_window` is the next bit out, and
// `p_win_bits` counts how many of the high bits are still valid. Refilling takes a
// whole `u32` of `stream_data` at once instead of one byte at a time, which is the
// point — the unary Rice loop is `while (read_bit())` inside a stage that is ~47% of
// I-frame decode, and the byte-at-a-time reader issued a storage load per 8 bits.
//
// The refill deliberately takes only the bytes remaining in the word that holds
// `p_byte_offset`, never the next word. That keeps the set of words this shader
// touches **exactly** what the byte-at-a-time version touched: `load_byte` already
// read the whole containing `u32` to extract one byte, so caching the rest of it is
// free, while reading ahead into the following word would be a new access past the
// end of the last stream. 1..4 bytes per refill, so `p_win_bits` is 8..32 after one.
var<private> p_window: u32;      // next bits, left-aligned at bit 31
var<private> p_win_bits: u32;    // valid high bits in p_window, 0..32
var<private> p_byte_offset: u32; // absolute byte offset in stream_data

// Per-thread EMA state for adaptive k (fixed-point ×16, window ≈ 8 coefficients)
var<private> p_ema: array<u32, 12>;

// Directional subband grouping — must match encoder exactly.
// Tile-local raster index of symbol `s` in stream `stream_id`.
//
// Streams walk the tile in column-major order, cut into STREAMS_PER_TILE contiguous segments, so
// the previous symbol in a stream is the coefficient directly above it. At the default 256 px tile
// a segment is exactly one column and this equals the old `stream_id + s * 256` coefficient for
// coefficient; at any other width that modulus interleaved distant columns into one stream and the
// adaptive k tracked the mixture (BUG-11). Must stay in sync with rice.rs stream_coeff_index().
fn stream_coeff_index(stream_id: u32, s: u32, symbols_per_stream: u32, th: u32) -> u32 {
    let j = stream_id * symbols_per_stream + s;
    let col = j / th;
    let row = j % th;
    return row * params.tile_size + col;
}

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

// Refill the window from the one `u32` of `stream_data` that holds `p_byte_offset`.
//
// `stream_data` packs bytes little-endian within each word (byte 0 is the low 8 bits)
// while bits run MSB-first within a byte, so the word has to be byte-swapped to put
// the earliest byte at the top of the window. Shifting by `byte_pos * 8` then drops
// the bytes of this word that were already consumed. Never reads the next word: see
// the note on `p_window`.
fn refill() {
    let word_idx = p_byte_offset >> 2u;
    let byte_pos = p_byte_offset & 3u;
    let word = stream_data[word_idx];
    let swapped = ((word & 0xFFu) << 24u)
                | (((word >> 8u) & 0xFFu) << 16u)
                | (((word >> 16u) & 0xFFu) << 8u)
                | ((word >> 24u) & 0xFFu);
    let avail = 4u - byte_pos;              // 1..4 bytes left in this word
    p_window = swapped << (byte_pos * 8u);  // shift is 0, 8, 16 or 24
    p_win_bits = avail * 8u;
    p_byte_offset += avail;
}

// Read a single bit (MSB-first).
fn read_bit() -> u32 {
    if (p_win_bits == 0u) {
        refill();
    }
    let bit = p_window >> 31u;
    p_window = p_window << 1u;
    p_win_bits -= 1u;
    return bit;
}

// Read multiple bits (MSB-first). A well-formed stream always asks for <= 15: every k
// in the bitstream comes from `optimal_k`, which clamps to 0..15.
//
// `MAX_TAKE` does not trust that. `k` is deserialised straight out of the bitstream as
// a byte and can be up to 255 in a corrupt or hostile one, and a WGSL shift by >= 32 is
// undefined — so an unbounded `take` would turn a bad `k` into undefined behaviour
// inside the decoder. Capping it keeps `32u - take` and `p_window << take` in range for
// any input at all: a bad `k` then reads garbage bits and terminates, which is what the
// per-tile CRC is there to catch. The byte-at-a-time reader this replaced had the same
// property for free, because its `take` could never exceed the 8 bits left in a byte.
const MAX_TAKE: u32 = 16u;

fn read_bits(count: u32) -> u32 {
    var value = 0u;
    var remaining = count;
    while (remaining > 0u) {
        if (p_win_bits == 0u) {
            refill();
        }
        let take = min(min(remaining, p_win_bits), MAX_TAKE);
        let bits = p_window >> (32u - take);
        value = (value << take) | bits;
        p_window = p_window << take;
        p_win_bits -= take;
        remaining -= take;
    }
    return value;
}

// Read a Golomb-Rice coded value: unary quotient + k-bit remainder.
fn read_rice(k: u32) -> u32 {
    var quotient = 0u;
    while (read_bit() == 1u) {
        quotient += 1u;
    }
    let remainder = select(0u, read_bits(k), k > 0u);
    return (quotient << k) | remainder;
}

// Decode all symbols for the current thread's stream into the output plane.
// Uses per-thread private state: p_ema, p_byte_offset, p_window, p_win_bits.
// Also reads shared: shared_k_zrl_nz, shared_k_zrl_z, shared_skip_bitmap.
fn decode_stream_body(
    thread_id: u32,
    tile_origin_x: u32,
    tile_origin_y: u32,
    symbols_per_stream: u32,
    th: u32,
) {
    var last_mag_large: bool = false;
    var s = 0u;
    while (s < symbols_per_stream) {
        let ci0 = stream_coeff_index(thread_id, s, symbols_per_stream, th);
        let cr0 = ci0 / params.tile_size;
        let cc0 = ci0 % params.tile_size;

        // Skip bitmap: if this position's group is fully zero, write 0 and advance
        let skip_g = compute_subband_group(cc0, cr0);
        if ((shared_skip_bitmap >> skip_g) & 1u) == 1u {
            let pi0 = (tile_origin_y + cr0) * params.plane_width + (tile_origin_x + cc0);
            output[pi0] = 0.0;
            s += 1u;
            continue;
        }

        let token = read_bit();
        if (token == 0u) {
            // Zero run: select k_zrl based on magnitude context of preceding nonzero
            let g_zrl = skip_g;
            let k_zrl = select(shared_k_zrl_z[g_zrl], shared_k_zrl_nz[g_zrl], last_mag_large);
            let run = read_rice(k_zrl) + 1u;
            // Write zeros, skipping past bitmap-skipped positions
            var written = 0u;
            var ws = s;
            while (written < run && ws < symbols_per_stream) {
                let ci = stream_coeff_index(thread_id, ws, symbols_per_stream, th);
                let cr = ci / params.tile_size;
                let cc = ci % params.tile_size;
                let pi = (tile_origin_y + cr) * params.plane_width + (tile_origin_x + cc);
                let ws_g = compute_subband_group(cc, cr);
                if ((shared_skip_bitmap >> ws_g) & 1u) == 1u {
                    // Bitmap-skipped position: write zero but don't count toward run
                    output[pi] = 0.0;
                    ws += 1u;
                    continue;
                }
                output[pi] = 0.0;
                written += 1u;
                ws += 1u;
            }
            s = ws;
            last_mag_large = false;
        } else {
            // Non-zero coefficient
            let tile_row = cr0;
            let tile_col = cc0;
            let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                          + (tile_origin_x + tile_col);

            let sign = read_bit();
            let g = skip_g;

            // Derive adaptive k from EMA (context-adaptive Rice parameter)
            let ema_mean = p_ema[g] >> 4u;
            let k = select(min(31u - countLeadingZeros(ema_mean), 15u), 0u, ema_mean == 0u);

            let rice_val = read_rice(k);
            let magnitude = rice_val + 1u;
            let value = select(i32(magnitude), -i32(magnitude), sign == 1u);
            output[plane_idx] = f32(value);

            // Update EMA with decoded magnitude
            p_ema[g] = p_ema[g] - (p_ema[g] >> 3u) + (rice_val << 1u);

            s += 1u;
            // Mirror encoder: large = |coeff| >= 2, i.e., rice_val (= |coeff|-1) >= 1
            last_mag_large = (rice_val >= 1u);
        }
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
    let tw = min(params.tile_size, params.plane_width - tile_origin_x);
    let th = min(params.tile_size, params.plane_height - tile_origin_y);

    let symbols_per_stream = (tw * th) / STREAMS_PER_TILE;
    let num_groups = max(1u, params.num_levels * 2u);

    // Cooperatively load k values into shared memory.
    // Layout: [k_mag][k_zrl_nz][k_zrl_z] × MAX_GROUPS, skip_bitmap @K_STRIDE-1
    if (thread_id < num_groups) {
        shared_k[thread_id]        = k_values[tile_id * K_STRIDE + thread_id];
        shared_k_zrl_nz[thread_id] = k_values[tile_id * K_STRIDE + MAX_GROUPS + thread_id];
        shared_k_zrl_z[thread_id]  = k_values[tile_id * K_STRIDE + MAX_GROUPS * 2u + thread_id];
    }
    if (thread_id == 0u) {
        shared_skip_bitmap = k_values[tile_id * K_STRIDE + K_STRIDE - 1u];
    }
    workgroupBarrier(); // top-level: all threads synchronize

    // === Checkerboard two-pass decode ===
    // Pass 1: even threads decode, write final EMA to shared_ctx_even.
    // workgroupBarrier() — at workgroup top level (not inside any branch per Metal/M1 rule).
    // Pass 2: odd threads derive adjusted k from neighbor EMA, then decode.

    // --- Pass 1: even threads ---
    if (thread_id % 2u == 0u) {
        // Initialize EMA from global k seeds (standard warm-start)
        for (var gi = 0u; gi < MAX_GROUPS; gi++) {
            p_ema[gi] = max(1u, 1u << shared_k[gi]) << 4u;
        }
        // Initialize bit reader
        p_byte_offset = stream_offsets[tile_id * STREAMS_PER_TILE + thread_id];
        p_window = 0u;
        p_win_bits = 0u;
        // Decode this stream
        decode_stream_body(thread_id, tile_origin_x, tile_origin_y, symbols_per_stream, th);
        // Expose final EMA to odd neighbor
        let even_idx = thread_id / 2u;
        for (var gi2 = 0u; gi2 < MAX_GROUPS; gi2++) {
            shared_ctx_even[even_idx][gi2] = p_ema[gi2] >> 4u;
        }
    }

    workgroupBarrier(); // top-level: even done, barrier before odd reads context

    // --- Pass 2: odd threads ---
    if (thread_id % 2u == 1u) {
        // Derive adjusted k from even neighbor's decoded EMA (same formula as encoder)
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
        }
        // Initialize bit reader
        p_byte_offset = stream_offsets[tile_id * STREAMS_PER_TILE + thread_id];
        p_window = 0u;
        p_win_bits = 0u;
        // Decode this stream
        decode_stream_body(thread_id, tile_origin_x, tile_origin_y, symbols_per_stream, th);
    }
}
