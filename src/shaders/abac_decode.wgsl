// Adaptive binary arithmetic decode over code-blocks — one thread per block.
//
// Direct port of `src/encoder/abac.rs`. The two must stay bit-exact; `abac_gpu` asserts that
// against the CPU decoder on real coefficients, which is the only thing that makes a port of a
// serial entropy coder safe to touch.
//
// Why one thread per block rather than a workgroup per block: the coder is inherently serial —
// every symbol's interval depends on the previous one, and every context depends on already
// decoded neighbours. There is nothing to parallelise *inside* a block. The parallelism is
// across blocks, and there are about 3000 of them in a 1080p frame at 64px, which is what makes
// this viable on a GPU at all.
//
// Why 16-bit coder state: WGSL has no 64-bit integers, and the interval split needs
// range * probability. At 16 bits that product is under 2^28 and fits a u32. See abac.rs.

const PROB_BITS: u32 = 12u;
const PROB_ONE: u32 = 4096u;      // 1 << PROB_BITS
const PROB_HALF: u32 = 2048u;
const ADAPT_SHIFT: u32 = 5u;

const STATE_BITS: u32 = 16u;
const STATE_MASK: u32 = 0xFFFFu;
const HALF: u32 = 0x8000u;
const QUARTER: u32 = 0x4000u;
const THREE_QUARTER: u32 = 0xC000u;

const NUM_BUCKETS: u32 = 6u;
const NUM_CONTEXTS: u32 = 18u;    // NUM_BUCKETS * 3

// Per-block geometry. `byte_offset` is into `stream` counted in bytes; `out_offset`, `stride`
// place the block inside its plane.
struct BlockInfo {
    byte_offset: u32,
    byte_len: u32,
    out_offset: u32,
    width: u32,
    height: u32,
    stride: u32,
    _pad0: u32,
    _pad1: u32,
}

// Four scalars rather than u32 + vec3<u32>: a vec3 forces 16-byte alignment, which makes the
// struct 32 bytes and no longer matches the 16-byte host struct.
struct Params {
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> stream: array<u32>;
@group(0) @binding(2) var<storage, read> blocks: array<BlockInfo>;
@group(0) @binding(3) var<storage, read_write> out: array<i32>;

// Decoder state, per thread. Kept in one struct so the helpers read like the Rust ones.
struct Dec {
    low: u32,
    high: u32,
    code: u32,
    byte_pos: u32,   // absolute byte index into `stream`
    bit_pos: u32,    // 0..7, MSB-first within the byte
    byte_end: u32,   // one past this block's last byte
}

// Bytes are packed little-endian into u32 words, matching `bytemuck` on the host.
fn get_byte(i: u32) -> u32 {
    let w = stream[i >> 2u];
    return (w >> ((i & 3u) * 8u)) & 0xFFu;
}

// Past the end of the block reads as zero, exactly as the Rust BitReader does, so a truncated
// or corrupt block terminates instead of running away.
fn get_bit(d: ptr<function, Dec>) -> u32 {
    if ((*d).byte_pos >= (*d).byte_end) {
        return 0u;
    }
    let b = (get_byte((*d).byte_pos) >> (7u - (*d).bit_pos)) & 1u;
    (*d).bit_pos = (*d).bit_pos + 1u;
    if ((*d).bit_pos == 8u) {
        (*d).bit_pos = 0u;
        (*d).byte_pos = (*d).byte_pos + 1u;
    }
    return b;
}

fn renormalise(d: ptr<function, Dec>) {
    loop {
        if ((*d).high < HALF) {
            // nothing to subtract
        } else if ((*d).low >= HALF) {
            (*d).code = (*d).code - HALF;
            (*d).low = (*d).low - HALF;
            (*d).high = (*d).high - HALF;
        } else if ((*d).low >= QUARTER && (*d).high < THREE_QUARTER) {
            (*d).code = (*d).code - QUARTER;
            (*d).low = (*d).low - QUARTER;
            (*d).high = (*d).high - QUARTER;
        } else {
            break;
        }
        (*d).low = ((*d).low << 1u) & STATE_MASK;
        (*d).high = (((*d).high << 1u) | 1u) & STATE_MASK;
        (*d).code = (((*d).code << 1u) | get_bit(d)) & STATE_MASK;
    }
}

// Takes an *index* into `probs`, not a pointer to it: WGSL forbids passing a `ptr<workgroup, T>`
// as a function argument (naga rejects it outright), so the callee indexes the module-scope
// workgroup array itself. Reads like the pointer version and compiles.
fn decode_bit(d: ptr<function, Dec>, pi: u32) -> u32 {
    var p = probs[pi];
    let range = (*d).high - (*d).low + 1u;
    let mid = (*d).low + ((range * p) >> PROB_BITS) - 1u;
    var bit = 0u;
    if ((*d).code > mid) {
        bit = 1u;
        (*d).low = mid + 1u;
    } else {
        (*d).high = mid;
    }
    // Adapt. Matches Prob::update in abac.rs including the clamp, which keeps the interval alive.
    if (bit == 1u) {
        p = p - (p >> ADAPT_SHIFT);
    } else {
        p = p + ((PROB_ONE - p) >> ADAPT_SHIFT);
    }
    probs[pi] = clamp(p, 1u, PROB_ONE - 1u);
    renormalise(d);
    return bit;
}

fn decode_bypass(d: ptr<function, Dec>) -> u32 {
    let range = (*d).high - (*d).low + 1u;
    let mid = (*d).low + ((range * PROB_HALF) >> PROB_BITS) - 1u;
    var bit = 0u;
    if ((*d).code > mid) {
        bit = 1u;
        (*d).low = mid + 1u;
    } else {
        (*d).high = mid;
    }
    renormalise(d);
    return bit;
}

// Must match `bucket` in abac.rs, which is `32 - leading_zeros(nb)` clamped. WGSL has
// firstLeadingBit rather than leading_zeros, and 32 - leading_zeros == firstLeadingBit + 1.
// Getting this off by one silently shifts every context and costs rate without failing anything
// except the bit-exactness assertion in abac_gpu.
fn bucket(nb: u32) -> u32 {
    if (nb == 0u) {
        return 0u;
    }
    return min(firstLeadingBit(nb) + 1u, NUM_BUCKETS - 1u);
}

// Per-thread scratch in workgroup memory rather than device memory.
//
// The first working version kept the 18 context probabilities in a function-scope array and read
// the neighbour magnitudes back out of the output buffer. Both live in device memory on Metal —
// a dynamically indexed function array spills, and the output buffer is uncached — so every
// coefficient cost about five device-memory round trips. That measured 53 Mcoeff/s at 64px
// blocks. Moving both into workgroup memory is the difference between "a GPU port exists" and
// "a GPU port is worth shipping".
//
// Budget on an M1 (32 KB per workgroup): 32 threads x 18 probabilities = 2.3 KB, plus two rows
// of magnitudes per thread at MAX_BLOCK_W = 64 => 32 x 2 x 64 x 4 B = 16 KB. Total 18.3 KB.
// This is why the code-block width is capped at 64 and the host asserts it.
//
// **Both arrays are thread-interleaved, and that is the whole performance story.** The obvious
// layout gives thread `t` a contiguous slice — `rows[t * 128 + x]` — which puts adjacent threads
// 128 words apart. Metal threadgroup memory has 32 banks of 4 bytes, and 128 is a multiple of 32,
// so all 32 lanes of a SIMD group hit the *same bank* on every single neighbour read: a 32x
// serialisation on the hottest access in the decoder. Indexing as `[x * WG + t]` instead puts
// lane `t` in bank `t`. Getting this wrong is invisible — the decode is still bit-exact, just
// dramatically slower.
const WG: u32 = 32u;
const MAX_BLOCK_W: u32 = 64u;

var<workgroup> probs: array<u32, 576>;              // WG * NUM_CONTEXTS
var<workgroup> rows: array<u32, 4096>;              // WG * 2 * MAX_BLOCK_W

@compute @workgroup_size(32)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let blk = gid.x;
    let tid = lid.x;
    if (blk >= params.num_blocks) {
        return;
    }
    let info = blocks[blk];

    // Thread-interleaved indexing: context `i` for this thread lives at `i * WG + tid`.
    for (var i = 0u; i < NUM_CONTEXTS; i++) {
        probs[i * WG + tid] = PROB_HALF;
    }
    // Two row buffers per thread, alternated by row parity so no copy is needed between rows.
    for (var i = 0u; i < 2u * MAX_BLOCK_W; i++) {
        rows[i * WG + tid] = 0u;
    }

    var d: Dec;
    d.low = 0u;
    d.high = STATE_MASK;
    d.code = 0u;
    d.byte_pos = info.byte_offset;
    d.bit_pos = 0u;
    // Explicit length rather than "where the next block starts", so the host is free to order
    // blocks however it likes — which it needs, to group equal-sized blocks into a SIMD group.
    d.byte_end = info.byte_offset + info.byte_len;

    for (var i = 0u; i < STATE_BITS; i++) {
        d.code = (d.code << 1u) | get_bit(&d);
    }

    for (var y = 0u; y < info.height; y++) {
        // `cur` is the row being decoded, `prev` the one above. Positions outside the block read
        // as zero, so blocks stay independent — a block must decode without reference to any
        // other, which is what lets them run concurrently.
        let cur = (y & 1u) * MAX_BLOCK_W;
        let prev = ((y + 1u) & 1u) * MAX_BLOCK_W;
        for (var x = 0u; x < info.width; x++) {
            var nb = 0u;
            if (x > 0u) {
                nb = nb + rows[(cur + x - 1u) * WG + tid];
            }
            if (y > 0u) {
                nb = nb + rows[(prev + x) * WG + tid];
                if (x > 0u) {
                    nb = nb + rows[(prev + x - 1u) * WG + tid];
                }
                if (x + 1u < info.width) {
                    nb = nb + rows[(prev + x + 1u) * WG + tid];
                }
            }
            let ctx = bucket(nb);

            var a = 0u;
            var v = 0i;
            if (decode_bit(&d, ctx * WG + tid) == 1u) {
                a = 1u;
                if (decode_bit(&d, (NUM_BUCKETS + ctx) * WG + tid) == 1u) {
                    a = 2u;
                    if (decode_bit(&d, (2u * NUM_BUCKETS + ctx) * WG + tid) == 1u) {
                        // Exp-Golomb order 0, bypass-coded.
                        var zeros = 0u;
                        loop {
                            if (decode_bypass(&d) == 1u) {
                                break;
                            }
                            zeros = zeros + 1u;
                            if (zeros > 32u) {
                                break; // corrupt or truncated
                            }
                        }
                        var n = 1u;
                        for (var k = 0u; k < zeros; k++) {
                            n = (n << 1u) | decode_bypass(&d);
                        }
                        a = n - 1u + 3u;
                    }
                }
                let neg = decode_bypass(&d);
                if (neg == 1u) {
                    v = -i32(a);
                } else {
                    v = i32(a);
                }
            }
            out[info.out_offset + y * info.stride + x] = v;
            rows[(cur + x) * WG + tid] = a;
        }
    }
}
