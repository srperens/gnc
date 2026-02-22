// GPU rANS decoder — 32 interleaved streams per tile.
//
// Each workgroup decodes one tile. 32 threads = 32 independent rANS streams.
// Cumfreq table is loaded cooperatively into workgroup shared memory.
// Symbol lookup uses binary search on cumfreq (~7 iterations for typical alphabets).

const RANS_BYTE_L: u32 = 8388608u;  // 1 << 23
const RANS_PRECISION: u32 = 12u;
const RANS_MASK: u32 = 4095u;       // (1 << 12) - 1
const STREAMS_PER_TILE: u32 = 32u;
const MAX_ALPHABET: u32 = 1024u;

// Per-tile info stride in u32s (must match host TILE_INFO_STRIDE)
const TILE_INFO_STRIDE: u32 = 100u;

struct Params {
    num_tiles: u32,
    coefficients_per_tile: u32,  // tile_size * tile_size (65536)
    plane_width: u32,            // padded plane width in pixels
    tile_size: u32,              // 256
    tiles_x: u32,                // number of tiles per row
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tile_info: array<u32>;
@group(0) @binding(2) var<storage, read> cumfreq_data: array<u32>;
@group(0) @binding(3) var<storage, read> stream_data: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Shared cumfreq table for the current tile (loaded cooperatively)
var<workgroup> shared_cumfreq: array<u32, 1025>;  // MAX_ALPHABET + 1

// Read one byte from the packed u32 stream data array.
// Bytes are packed little-endian: byte 0 is bits [0:7] of u32[0].
fn read_byte(byte_offset: u32) -> u32 {
    let word_idx = byte_offset >> 2u;
    let byte_pos = byte_offset & 3u;
    return (stream_data[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

// Binary search: find sym where cumfreq[sym] <= slot < cumfreq[sym+1].
fn binary_search(slot: u32, alphabet_size: u32) -> u32 {
    var lo = 0u;
    var hi = alphabet_size;
    // Max 11 iterations covers alphabet up to 1024
    for (var iter = 0u; iter < 11u; iter++) {
        if (lo >= hi) { break; }
        let mid = (lo + hi) >> 1u;
        if (shared_cumfreq[mid + 1u] <= slot) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

@compute @workgroup_size(32)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let thread_id = lid.x;
    let tile_id = wid.x;

    if (tile_id >= params.num_tiles) {
        return;
    }

    let base = tile_id * TILE_INFO_STRIDE;

    // All threads read tile metadata from global memory
    let alphabet_size = tile_info[base + 1u];
    let cumfreq_offset = tile_info[base + 2u];
    let alphabet_size_plus_one = alphabet_size + 1u;

    // Cooperatively load cumfreq into shared memory
    for (var i = thread_id; i < alphabet_size_plus_one; i += STREAMS_PER_TILE) {
        shared_cumfreq[i] = cumfreq_data[cumfreq_offset + i];
    }

    workgroupBarrier();

    let min_val = bitcast<i32>(tile_info[base]);

    // Per-stream: initial state, byte offset, byte count
    let stream_byte_base = tile_info[base + 3u];
    let initial_state = tile_info[base + 4u + thread_id];
    let stream_offset = tile_info[base + 36u + thread_id];

    var state = initial_state;
    var byte_ptr = stream_byte_base + stream_offset;

    let symbols_per_thread = params.coefficients_per_tile / STREAMS_PER_TILE;

    // Compute tile position in the plane
    let tile_x = tile_id % params.tiles_x;
    let tile_y = tile_id / params.tiles_x;
    let tile_origin_x = tile_x * params.tile_size;
    let tile_origin_y = tile_y * params.tile_size;

    // Decode symbols for this stream
    for (var i = 0u; i < symbols_per_thread; i++) {
        // Extract slot from current state
        let slot = state & RANS_MASK;
        let sym = binary_search(slot, alphabet_size);

        // Compute output position: stride-32 deinterleave into tile, then into plane
        let coeff_idx = thread_id + i * STREAMS_PER_TILE;
        let tile_row = coeff_idx / params.tile_size;
        let tile_col = coeff_idx % params.tile_size;
        let plane_idx = (tile_origin_y + tile_row) * params.plane_width
                      + (tile_origin_x + tile_col);

        output[plane_idx] = f32(i32(sym) + min_val);

        // Advance rANS state
        let start = shared_cumfreq[sym];
        let freq = shared_cumfreq[sym + 1u] - start;
        state = freq * (state >> RANS_PRECISION) + (state & RANS_MASK) - start;

        // Renormalize: read bytes until state >= RANS_BYTE_L (at most 3 reads)
        for (var r = 0u; r < 3u; r++) {
            if (state >= RANS_BYTE_L) { break; }
            let byte_val = read_byte(byte_ptr);
            state = (state << 8u) | byte_val;
            byte_ptr++;
        }
    }
}
