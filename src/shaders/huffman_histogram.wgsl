// GPU Huffman histogram — 256 threads per tile, cooperative frequency counting.
//
// Counts magnitude frequencies per subband group per tile, and accumulates
// ZRL (zero-run-length) statistics for k_zrl computation.
//
// Output:
//   hist_output[tile * HIST_STRIDE + group * ALPHABET_SIZE + symbol] = frequency
//   zrl_output[tile * ZRL_STRIDE + group * 2 + 0] = sum of (run_length - 1)
//   zrl_output[tile * ZRL_STRIDE + group * 2 + 1] = count of runs

const STREAMS_PER_TILE: u32 = 256u;
const ALPHABET_SIZE: u32 = 64u;
const ESCAPE_SYM: u32 = 63u;
const MAX_GROUPS: u32 = 8u;
const HIST_STRIDE: u32 = 512u;  // MAX_GROUPS * ALPHABET_SIZE
const ZRL_STRIDE: u32 = 16u;    // MAX_GROUPS * 2

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
@group(0) @binding(2) var<storage, read_write> hist_output: array<u32>;
@group(0) @binding(3) var<storage, read_write> zrl_output: array<u32>;

// Shared histogram: MAX_GROUPS * ALPHABET_SIZE = 512 entries = 2KB
var<workgroup> shared_hist: array<atomic<u32>, 512>;
// Shared ZRL stats: MAX_GROUPS * 2 = 16 entries = 64B
var<workgroup> zrl_sum: array<atomic<u32>, 8>;
var<workgroup> zrl_count: array<atomic<u32>, 8>;

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

    // Clear shared memory cooperatively: 512 entries, 256 threads → 2 per thread
    atomicStore(&shared_hist[thread_id], 0u);
    atomicStore(&shared_hist[thread_id + 256u], 0u);
    if (thread_id < MAX_GROUPS) {
        atomicStore(&zrl_sum[thread_id], 0u);
        atomicStore(&zrl_count[thread_id], 0u);
    }
    workgroupBarrier();

    // Each thread scans its stream: accumulate magnitude histogram + ZRL stats
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
            let magnitude = u32(abs(coeff)) - 1u;
            let g = compute_subband_group(tile_col, tile_row);
            let sym = min(magnitude, ESCAPE_SYM);
            atomicAdd(&shared_hist[g * ALPHABET_SIZE + sym], 1u);

            if (zero_run > 0u) {
                atomicAdd(&zrl_sum[zrl_group], min(zero_run - 1u, 65535u));
                atomicAdd(&zrl_count[zrl_group], 1u);
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
        atomicAdd(&zrl_sum[zrl_group], min(zero_run - 1u, 65535u));
        atomicAdd(&zrl_count[zrl_group], 1u);
    }
    workgroupBarrier();

    // Write histograms to output buffer cooperatively: 512 entries, 256 threads → 2 per thread
    hist_output[tile_id * HIST_STRIDE + thread_id] = atomicLoad(&shared_hist[thread_id]);
    hist_output[tile_id * HIST_STRIDE + thread_id + 256u] = atomicLoad(&shared_hist[thread_id + 256u]);

    // Write ZRL stats (first 8 threads)
    if (thread_id < MAX_GROUPS) {
        zrl_output[tile_id * ZRL_STRIDE + thread_id * 2u] = atomicLoad(&zrl_sum[thread_id]);
        zrl_output[tile_id * ZRL_STRIDE + thread_id * 2u + 1u] = atomicLoad(&zrl_count[thread_id]);
    }
}
