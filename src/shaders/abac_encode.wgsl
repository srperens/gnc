// Adaptive binary arithmetic *encode* over code-blocks — one thread per block.
//
// The mirror of `abac_decode.wgsl`, and a direct port of the encoders in
// `src/encoder/abac.rs`. Both engines are here, one entry point each, exactly as on the decode
// side: `main` is the bit-renormalising interval coder, `main_rc` the byte-renormalising range
// coder (which is what a shipped encode uses — see `abac_coder_from_env`).
//
// **These must be bit-exact against the CPU encoder, not merely decodable.** A GPU encoder that
// produced a different-but-valid stream would move every rate figure abac has, and would make the
// GPU and CPU encoders two coders rather than one implementation. `abac_gpu_encode` asserts
// byte-identity per block against `abac.rs` on the same geometries the decoder is verified on.
//
// # Why one thread per block, again
//
// Encode is as serial within a block as decode is — every symbol's interval depends on the
// previous one and every context on already-*coded* neighbours — and as parallel across blocks.
// A padded 1080p 4:4:4 frame holds roughly 3000 blocks at cb=64. CLAUDE.md's bounded-dependency
// rule prices this: the chain is confined to a block, blocks are independent.
//
// # A block's coded size is not known before it is coded
//
// That is the one structural problem an encoder has and a decoder does not, and this shader
// supports both answers to it (see `abac_gpu_encode.rs`, which measures them):
//
//   - `params.emit == 0` — **count**. Runs the whole coder and writes nothing but the byte count.
//     The host prefix-sums those counts into exact offsets and dispatches again with emit == 1.
//     Two coder passes, exact packing, no bound to get wrong.
//   - `params.emit == 1` — **emit**. Writes bytes into the slot `[dst_byte, dst_byte + cap_bytes)`
//     the host assigned, and still records the length. With slots from `bound` below this is one
//     coder pass plus a cheap scan and a compaction copy.
//
// `bound` computes a *provable* per-block ceiling from the coefficients, so the slot mode is not
// a heuristic with a panic behind it: BUG-22 is what an unbounded per-stream output slot does
// (7.8-10.9 dB at q=90, a stream spilling into its neighbour's). `flags[0]` is still raised if a
// slot is ever exceeded, because a bound that is merely argued is a bound that is wrong later.
//
// Every block's slot starts on a 4-byte boundary and this shader only ever writes whole u32
// words. That is not tidiness: bytes are packed little-endian into u32 words (the decoder unpacks
// them the same way), so two blocks sharing a word would be two threads read-modify-writing it.
// Alignment makes every word the property of exactly one thread and removes the race by
// construction rather than by atomics.

const PROB_BITS: u32 = 12u;
const PROB_ONE: u32 = 4096u;
const PROB_HALF: u32 = 2048u;
const ADAPT_SHIFT: u32 = 5u;

const STATE_BITS: u32 = 16u;
const STATE_MASK: u32 = 0xFFFFu;
const HALF: u32 = 0x8000u;
const QUARTER: u32 = 0x4000u;
const THREE_QUARTER: u32 = 0xC000u;

const NUM_BUCKETS: u32 = 6u;
const NUM_CONTEXTS: u32 = 18u;
// ENT-9 candidate A: the Exp-Golomb unary prefix is context-coded on (position, bucket) rather
// than bypassed. Must match `abac.rs`'s PREFIX_POSITIONS / PREFIX_BASE / NUM_CONTEXTS_ALL, and
// `prefix_ctx` there is `PREFIX_BASE + min(position, 3) * NUM_BUCKETS + bucket`.
const PREFIX_POSITIONS: u32 = 4u;
const PREFIX_BASE: u32 = NUM_CONTEXTS;
const NUM_CONTEXTS_ALL: u32 = NUM_CONTEXTS + NUM_BUCKETS * PREFIX_POSITIONS;   // 42

fn prefix_ctx(position: u32, ctx: u32) -> u32 {
    return PREFIX_BASE + min(position, PREFIX_POSITIONS - 1u) * NUM_BUCKETS + ctx;
}

const RC_PROB_BITS: u32 = 11u;
const RC_PROB_ONE: u32 = 2048u;
const RC_PROB_HALF: u32 = 1024u;
const RC_ADAPT_SHIFT: u32 = 5u;
const RC_TOP: u32 = 16777216u;   // 1 << 24

// Per-block geometry and output slot. Layout must match `EncBlock` in `abac_gpu_encode.rs`.
struct EncBlock {
    // Index of the block's top-left coefficient in the (padded) input plane.
    in_offset: u32,
    width: u32,
    height: u32,
    // Row stride of the input plane, in coefficients.
    stride: u32,
    // Byte offset of this block's output slot. Always a multiple of 4 — see the header.
    dst_byte: u32,
    // Bytes available at `dst_byte`. Also a multiple of 4. Unused when counting.
    cap_bytes: u32,
    // Where this block's length goes in `lengths`, i.e. its position in `code_blocks` order.
    // Dispatch order is *not* that order: blocks are sorted by area so a SIMD group holds
    // equal-sized work, the same reason the decoder sorts.
    index: u32,
    _pad0: u32,
}

struct Params {
    num_blocks: u32,
    // 0 = count bytes only, 1 = write bytes into the slot.
    emit: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> blocks: array<EncBlock>;
@group(0) @binding(3) var<storage, read_write> out_words: array<u32>;
// Bytes this block coded to, indexed by `EncBlock.index`. Written in both modes: in emit mode it
// is the canary that the second pass agreed with the first.
@group(0) @binding(4) var<storage, read_write> lengths: array<u32>;
// [0] slot overflow, [1] a coefficient magnitude at or above 2^29. Both are "this cannot happen"
// conditions, which is exactly why they are counted rather than assumed.
@group(0) @binding(5) var<storage, read_write> flags: array<atomic<u32>>;

const WG: u32 = 32u;
const MAX_BLOCK_W: u32 = 64u;

// Same thread-interleaved workgroup scratch as the decoder, and for the same reason: the obvious
// per-thread-contiguous layout puts all 32 lanes of a SIMD group in one of Metal's 32 banks on
// every neighbour read. `[i * WG + tid]` puts lane `tid` in bank `tid`. Budget: 576 + 4096 words
// = 18.3 KB of the M1's 32 KB.
var<workgroup> probs: array<u32, 1344>;
var<workgroup> rows: array<u32, 1024>;              // WG * ROW_WORDS

// `rows` packs four magnitudes per word, one byte each, because the full u32 was 16 KB on its own
// and put the shader over the workgroup budget the device is created with (BUG-31).
//
// **Clamping at ROW_CLAMP cannot change a single context, which is why this is bit-exact and not
// an approximation.** `bucket` saturates: any `nb >= 1 << (NUM_BUCKETS - 2)` returns
// `NUM_BUCKETS - 1`. So for a contributor `a`:
//   * `a < ROW_CLAMP` is stored exactly, and a sum of such contributors is exact;
//   * `a >= ROW_CLAMP` stores ROW_CLAMP, and both the clamped and the true sum are then
//     `>= ROW_CLAMP` and both bucket to `NUM_BUCKETS - 1`.
// Either way `bucket(nb)` is identical, so the coder sees the same context sequence and emits the
// same bytes. Asserted rather than argued: the abac identity gates compare whole files against the
// CPU coder in `abac.rs`, which stores true magnitudes.
const ROW_CLAMP: u32 = 1u << (NUM_BUCKETS - 2u);
const ROW_WORDS: u32 = 2u * MAX_BLOCK_W / 4u;       // 32 words of 4 magnitudes per thread

fn row_get(i: u32, tid: u32) -> u32 {
    let w = rows[(i >> 2u) * WG + tid];
    return (w >> ((i & 3u) * 8u)) & 0xFFu;
}

fn row_set(i: u32, tid: u32, a: u32) {
    let idx = (i >> 2u) * WG + tid;
    let sh = (i & 3u) * 8u;
    rows[idx] = (rows[idx] & ~(0xFFu << sh)) | (min(a, ROW_CLAMP) << sh);
}


// `unsigned_abs`, exactly: negating in u32 wraps, so i32::MIN maps to 2^31 rather than to itself.
fn mag_of(v: i32) -> u32 {
    let u = bitcast<u32>(v);
    if (v < 0) {
        return 0u - u;
    }
    return u;
}

// Must match `bucket` in abac.rs: `32 - leading_zeros(nb)` clamped. WGSL has firstLeadingBit,
// and 32 - leading_zeros == firstLeadingBit + 1.
fn bucket(nb: u32) -> u32 {
    if (nb == 0u) {
        return 0u;
    }
    return min(firstLeadingBit(nb) + 1u, NUM_BUCKETS - 1u);
}

// ---------------------------------------------------------------------------
// Interval coder (Witten-Neal-Cleary), bit-renormalising
// ---------------------------------------------------------------------------

struct Enc {
    low: u32,
    high: u32,
    // u32 where abac.rs has u64. `pending` counts renormalisation steps, so it is bounded by the
    // block's output in bits — under 2^20 for a 64x64 block. u64 there is belt and braces.
    pending: u32,
    // Bit writer: MSB-first into `cur`, like abac.rs's BitWriter.
    cur: u32,
    nbits: u32,
    // Byte writer: little-endian into `word`, one whole u32 store per four bytes.
    word: u32,
    nbytes: u32,
    word_pos: u32,
    total: u32,
    base_word: u32,
    cap_words: u32,
    emit: u32,
}

fn e_put_byte(e: ptr<function, Enc>, b: u32) {
    (*e).word = (*e).word | ((b & 0xFFu) << ((*e).nbytes * 8u));
    (*e).nbytes = (*e).nbytes + 1u;
    (*e).total = (*e).total + 1u;
    if ((*e).nbytes == 4u) {
        if ((*e).emit != 0u) {
            if ((*e).word_pos < (*e).cap_words) {
                out_words[(*e).base_word + (*e).word_pos] = (*e).word;
            } else {
                atomicOr(&flags[0], 1u);
            }
        }
        (*e).word_pos = (*e).word_pos + 1u;
        (*e).word = 0u;
        (*e).nbytes = 0u;
    }
}

fn e_put_bit(e: ptr<function, Enc>, bit: u32) {
    (*e).cur = ((*e).cur << 1u) | (bit & 1u);
    (*e).nbits = (*e).nbits + 1u;
    if ((*e).nbits == 8u) {
        e_put_byte(e, (*e).cur);
        (*e).cur = 0u;
        (*e).nbits = 0u;
    }
}

// One bit, then the pending run inverted — the E3 mapping of the WNC coder.
fn e_emit(e: ptr<function, Enc>, bit: u32) {
    e_put_bit(e, bit);
    loop {
        if ((*e).pending == 0u) {
            break;
        }
        e_put_bit(e, 1u - bit);
        (*e).pending = (*e).pending - 1u;
    }
}

fn e_renormalise(e: ptr<function, Enc>) {
    loop {
        if ((*e).high < HALF) {
            e_emit(e, 0u);
        } else if ((*e).low >= HALF) {
            e_emit(e, 1u);
            (*e).low = (*e).low - HALF;
            (*e).high = (*e).high - HALF;
        } else if ((*e).low >= QUARTER && (*e).high < THREE_QUARTER) {
            (*e).pending = (*e).pending + 1u;
            (*e).low = (*e).low - QUARTER;
            (*e).high = (*e).high - QUARTER;
        } else {
            break;
        }
        (*e).low = ((*e).low << 1u) & STATE_MASK;
        (*e).high = (((*e).high << 1u) | 1u) & STATE_MASK;
    }
}

// `pi` is an index into `probs`, not a pointer: WGSL forbids passing ptr<workgroup, T>.
fn e_encode(e: ptr<function, Enc>, bit: u32, pi: u32) {
    var p = probs[pi];
    let range = (*e).high - (*e).low + 1u;
    // range * p stays under 2^28 by construction — see STATE_BITS in abac.rs.
    let mid = (*e).low + ((range * p) >> PROB_BITS) - 1u;
    if (bit == 1u) {
        (*e).low = mid + 1u;
    } else {
        (*e).high = mid;
    }
    if (bit == 1u) {
        p = p - (p >> ADAPT_SHIFT);
    } else {
        p = p + ((PROB_ONE - p) >> ADAPT_SHIFT);
    }
    probs[pi] = clamp(p, 1u, PROB_ONE - 1u);
    e_renormalise(e);
}

fn e_encode_bypass(e: ptr<function, Enc>, bit: u32) {
    let range = (*e).high - (*e).low + 1u;
    let mid = (*e).low + ((range * PROB_HALF) >> PROB_BITS) - 1u;
    if (bit == 1u) {
        (*e).low = mid + 1u;
    } else {
        (*e).high = mid;
    }
    e_renormalise(e);
}

fn e_finish(e: ptr<function, Enc>) {
    // Two bits plus the pending run disambiguate the final interval.
    (*e).pending = (*e).pending + 1u;
    if ((*e).low < QUARTER) {
        e_emit(e, 0u);
    } else {
        e_emit(e, 1u);
    }
    // BitWriter::finish — zero-pad to the byte boundary.
    loop {
        if ((*e).nbits == 0u) {
            break;
        }
        e_put_bit(e, 0u);
    }
    // Flush a partial word. Its unused high bytes are never read: the decoder is bounded by
    // `byte_len`, and the next block starts at the next 4-byte boundary.
    if ((*e).nbytes != 0u) {
        if ((*e).emit != 0u) {
            if ((*e).word_pos < (*e).cap_words) {
                out_words[(*e).base_word + (*e).word_pos] = (*e).word;
            } else {
                atomicOr(&flags[0], 1u);
            }
        }
    }
}

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

    for (var i = 0u; i < NUM_CONTEXTS_ALL; i++) {
        probs[i * WG + tid] = PROB_HALF;
    }
    for (var i = 0u; i < ROW_WORDS; i++) {
        rows[i * WG + tid] = 0u;
    }

    var e: Enc;
    e.low = 0u;
    e.high = STATE_MASK;
    e.pending = 0u;
    e.cur = 0u;
    e.nbits = 0u;
    e.word = 0u;
    e.nbytes = 0u;
    e.word_pos = 0u;
    e.total = 0u;
    e.base_word = info.dst_byte >> 2u;
    e.cap_words = info.cap_bytes >> 2u;
    e.emit = params.emit;

    for (var y = 0u; y < info.height; y++) {
        let cur = (y & 1u) * MAX_BLOCK_W;
        let prev = ((y + 1u) & 1u) * MAX_BLOCK_W;
        for (var x = 0u; x < info.width; x++) {
            var nb = 0u;
            if (x > 0u) {
                nb = nb + row_get(cur + x - 1u, tid);
            }
            if (y > 0u) {
                nb = nb + row_get(prev + x, tid);
                if (x > 0u) {
                    nb = nb + row_get(prev + x - 1u, tid);
                }
                if (x + 1u < info.width) {
                    nb = nb + row_get(prev + x + 1u, tid);
                }
            }
            let ctx = bucket(nb);

            // The quantiser emits integral f32, so this round is a no-op that matches the host's
            // `.round() as i32` exactly; every other GPU entropy encoder here reads it the same way.
            let v = i32(round(input[info.in_offset + y * info.stride + x]));
            let a = mag_of(v);
            if (a >= 0x20000000u) {
                atomicOr(&flags[1], 1u);
            }

            e_encode(&e, u32(a > 0u), ctx * WG + tid);
            if (a > 0u) {
                e_encode(&e, u32(a > 1u), (NUM_BUCKETS + ctx) * WG + tid);
                if (a > 1u) {
                    e_encode(&e, u32(a > 2u), (2u * NUM_BUCKETS + ctx) * WG + tid);
                    if (a > 2u) {
                        // Exp-Golomb order 0 of (a - 3): context-coded unary prefix, bypassed mantissa.
                        let n = a - 2u;
                        let len = 32u - countLeadingZeros(n);
                        // Unary prefix: `len - 1` "keep going" then one "stop", each in its own
                        // (position, bucket) context instead of at p = 1/2.
                        for (var k = 0u; k < len; k++) {
                            e_encode(&e, u32(k + 1u == len), prefix_ctx(k, ctx) * WG + tid);
                        }
                        // Mantissa: the low `len - 1` bits of `n`, MSB-first, still bypassed.
                        for (var k = len - 1u; k > 0u; k--) {
                            e_encode_bypass(&e, (n >> (k - 1u)) & 1u);
                        }
                    }
                }
                e_encode_bypass(&e, u32(v < 0));
            }
            row_set(cur + x, tid, a);
        }
    }
    e_finish(&e);
    lengths[info.index] = e.total;
}

// ---------------------------------------------------------------------------
// Variant B: byte-renormalising range coder — the one a shipped encode uses
// ---------------------------------------------------------------------------
//
// `low` is u64 in abac.rs and WGSL has no u64, which is the only real porting problem in this
// file. It is emulated exactly rather than approximately: `low` is kept as the pair
// (`low` = the low 32 bits, `carry` = the high bits), and every use in abac.rs maps to it
// term for term —
//
//   `low += bound`              -> add with an overflow test that increments `carry`
//   `low < 0xFF000000`          -> `low < 0xFF000000 && carry == 0`
//   `low > 0xFFFFFFFF`          -> `carry != 0`
//   `(low >> 32) as u8`         -> `carry & 0xFF`
//   `(low >> 24) & 0xFF`        -> `(low >> 24) & 0xFF`   (bits 24..31 live in the low word)
//   `low = (low << 8) & 0xFFFFFFFF` -> `low <<= 8; carry = 0`
//
// `carry` is a *count*, not a flag. It is tempting to argue that at most one carry can ever be
// pending — `shift_low` clears it, and one `low += bound` can only produce one bit — but that
// argument needs `shift_low` to run between two adds, and the renormalisation loop does not run
// when `range` stays above RC_TOP. Counting costs one instruction and makes the emulation exact
// whether the argument holds or not, which is the difference between a port and a rewrite.

struct REnc {
    low: u32,
    carry: u32,
    range: u32,
    cache: u32,
    cache_size: u32,
    word: u32,
    nbytes: u32,
    word_pos: u32,
    total: u32,
    base_word: u32,
    cap_words: u32,
    emit: u32,
}

fn r_put_byte(e: ptr<function, REnc>, b: u32) {
    (*e).word = (*e).word | ((b & 0xFFu) << ((*e).nbytes * 8u));
    (*e).nbytes = (*e).nbytes + 1u;
    (*e).total = (*e).total + 1u;
    if ((*e).nbytes == 4u) {
        if ((*e).emit != 0u) {
            if ((*e).word_pos < (*e).cap_words) {
                out_words[(*e).base_word + (*e).word_pos] = (*e).word;
            } else {
                atomicOr(&flags[0], 1u);
            }
        }
        (*e).word_pos = (*e).word_pos + 1u;
        (*e).word = 0u;
        (*e).nbytes = 0u;
    }
}

// Emit the top byte of `low`, propagating a carry back through any pending 0xFF run.
fn r_shift_low(e: ptr<function, REnc>) {
    if ((*e).low < 0xFF000000u || (*e).carry != 0u) {
        let c = (*e).carry & 0xFFu;
        loop {
            r_put_byte(e, ((*e).cache + c) & 0xFFu);
            (*e).cache = 0xFFu;
            (*e).cache_size = (*e).cache_size - 1u;
            if ((*e).cache_size == 0u) {
                break;
            }
        }
        (*e).cache = ((*e).low >> 24u) & 0xFFu;
    }
    (*e).cache_size = (*e).cache_size + 1u;
    // `(low << 8) & 0xFFFFFFFF` in abac.rs drops bit 32, which is this carry.
    (*e).low = (*e).low << 8u;
    (*e).carry = 0u;
}

fn r_add_low(e: ptr<function, REnc>, v: u32) {
    let s = (*e).low + v;
    if (s < (*e).low) {
        (*e).carry = (*e).carry + 1u;
    }
    (*e).low = s;
}

fn r_renormalise(e: ptr<function, REnc>) {
    loop {
        if ((*e).range >= RC_TOP) {
            break;
        }
        (*e).range = (*e).range << 8u;
        r_shift_low(e);
    }
}

fn r_encode(e: ptr<function, REnc>, bit: u32, pi: u32) {
    var p = probs[pi];
    // (range >> 11) * p peaks at 2^21 * 2047 < 2^32, so no split multiply is needed.
    let bound = ((*e).range >> RC_PROB_BITS) * p;
    if (bit == 1u) {
        r_add_low(e, bound);
        (*e).range = (*e).range - bound;
        p = p - (p >> RC_ADAPT_SHIFT);
    } else {
        (*e).range = bound;
        p = p + ((RC_PROB_ONE - p) >> RC_ADAPT_SHIFT);
    }
    probs[pi] = clamp(p, 1u, RC_PROB_ONE - 1u);
    r_renormalise(e);
}

fn r_encode_bypass(e: ptr<function, REnc>, bit: u32) {
    (*e).range = (*e).range >> 1u;
    if (bit == 1u) {
        r_add_low(e, (*e).range);
    }
    r_renormalise(e);
}

fn r_finish(e: ptr<function, REnc>) {
    for (var i = 0u; i < 5u; i++) {
        r_shift_low(e);
    }
    if ((*e).nbytes != 0u) {
        if ((*e).emit != 0u) {
            if ((*e).word_pos < (*e).cap_words) {
                out_words[(*e).base_word + (*e).word_pos] = (*e).word;
            } else {
                atomicOr(&flags[0], 1u);
            }
        }
    }
}

@compute @workgroup_size(32)
fn main_rc(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let blk = gid.x;
    let tid = lid.x;
    if (blk >= params.num_blocks) {
        return;
    }
    let info = blocks[blk];

    for (var i = 0u; i < NUM_CONTEXTS_ALL; i++) {
        probs[i * WG + tid] = RC_PROB_HALF;
    }
    for (var i = 0u; i < ROW_WORDS; i++) {
        rows[i * WG + tid] = 0u;
    }

    var e: REnc;
    e.low = 0u;
    e.carry = 0u;
    e.range = 0xFFFFFFFFu;
    e.cache = 0u;
    e.cache_size = 1u;
    e.word = 0u;
    e.nbytes = 0u;
    e.word_pos = 0u;
    e.total = 0u;
    e.base_word = info.dst_byte >> 2u;
    e.cap_words = info.cap_bytes >> 2u;
    e.emit = params.emit;

    for (var y = 0u; y < info.height; y++) {
        let cur = (y & 1u) * MAX_BLOCK_W;
        let prev = ((y + 1u) & 1u) * MAX_BLOCK_W;
        for (var x = 0u; x < info.width; x++) {
            var nb = 0u;
            if (x > 0u) {
                nb = nb + row_get(cur + x - 1u, tid);
            }
            if (y > 0u) {
                nb = nb + row_get(prev + x, tid);
                if (x > 0u) {
                    nb = nb + row_get(prev + x - 1u, tid);
                }
                if (x + 1u < info.width) {
                    nb = nb + row_get(prev + x + 1u, tid);
                }
            }
            let ctx = bucket(nb);

            let v = i32(round(input[info.in_offset + y * info.stride + x]));
            let a = mag_of(v);
            if (a >= 0x20000000u) {
                atomicOr(&flags[1], 1u);
            }

            r_encode(&e, u32(a > 0u), ctx * WG + tid);
            if (a > 0u) {
                r_encode(&e, u32(a > 1u), (NUM_BUCKETS + ctx) * WG + tid);
                if (a > 1u) {
                    r_encode(&e, u32(a > 2u), (2u * NUM_BUCKETS + ctx) * WG + tid);
                    if (a > 2u) {
                        let n = a - 2u;
                        let len = 32u - countLeadingZeros(n);
                        // Unary prefix: `len - 1` "keep going" then one "stop", each in its own
                        // (position, bucket) context instead of at p = 1/2.
                        for (var k = 0u; k < len; k++) {
                            r_encode(&e, u32(k + 1u == len), prefix_ctx(k, ctx) * WG + tid);
                        }
                        // Mantissa: the low `len - 1` bits of `n`, MSB-first, still bypassed.
                        for (var k = len - 1u; k > 0u; k--) {
                            r_encode_bypass(&e, (n >> (k - 1u)) & 1u);
                        }
                    }
                }
                r_encode_bypass(&e, u32(v < 0));
            }
            row_set(cur + x, tid, a);
        }
    }
    r_finish(&e);
    lengths[info.index] = e.total;
}

// ---------------------------------------------------------------------------
// Provable per-block output ceiling, for the one-coder-pass mode
// ---------------------------------------------------------------------------
//
// Counts the *decisions* a block will code, which needs no arithmetic coder — one pass over the
// coefficients with a handful of integer ops each — and multiplies by the worst-case output per
// decision. That is what makes this cheap enough to be worth a pass; running the coder to find
// the exact size is the other mode.
//
// **Where the constants come from.** Renormalisation keeps the interval above QUARTER = 2^14, and
// a context probability is clamped into [1, 4095], so one context-coded decision can shrink the
// interval by at most a factor 2^-13 and therefore emit at most 13 bits. For the range coder the
// interval is kept above 2^24 and may be up to 2^32, and a clamped probability can drop it to
// 2^13 — three renormalisation bytes, 24 bits. A bypass decision halves the interval exactly: one
// bit for the interval coder, at most one whole byte of renormalisation for the range coder. So
// 24 bits per context decision and 8 per bypass decision covers both engines with slack, and the
// 256-bit tail covers `finish` (five bytes for the range coder, two bits plus a pending run and
// byte padding for the interval one).
//
// It is loose — a zero coefficient costs 3 bytes here and about a fiftieth of a byte in reality —
// and that is the price of the mode: the slot buffer is roughly 3 bytes per coefficient where the
// output is nearer 0.5. It is a *bound*, not an estimate, which is the whole point: BUG-22's
// 512-byte Huffman slot was an estimate with a panic behind it.
const BOUND_CTX_BITS: u32 = 24u;
const BOUND_BYPASS_BITS: u32 = 8u;
const BOUND_TAIL_BITS: u32 = 256u;

@compute @workgroup_size(64)
fn bound(@builtin(global_invocation_id) gid: vec3<u32>) {
    let blk = gid.x;
    if (blk >= params.num_blocks) {
        return;
    }
    let info = blocks[blk];
    var n_ctx = 0u;
    var n_byp = 0u;
    for (var y = 0u; y < info.height; y++) {
        for (var x = 0u; x < info.width; x++) {
            let a = mag_of(i32(round(input[info.in_offset + y * info.stride + x])));
            n_ctx = n_ctx + 1u;
            if (a > 0u) {
                n_byp = n_byp + 1u;          // sign
                n_ctx = n_ctx + 1u;          // |v| > 1
                if (a > 1u) {
                    n_ctx = n_ctx + 1u;      // |v| > 2
                    if (a > 2u) {
                        let len = 32u - countLeadingZeros(a - 2u);
                        n_byp = n_byp + 2u * len - 1u;
                    }
                }
            }
        }
    }
    let bits = BOUND_CTX_BITS * n_ctx + BOUND_BYPASS_BITS * n_byp + BOUND_TAIL_BITS;
    // Round the slot up to a whole word: every block's slot must start 4-byte aligned.
    lengths[info.index] = ((bits + 7u) / 8u + 3u) & ~3u;
}

// ---------------------------------------------------------------------------
// Compaction, for the one-coder-pass mode
// ---------------------------------------------------------------------------
//
// Bounded slots leave the streams scattered with large gaps, and reading that whole buffer back
// would cost several times what the compressed frame costs. This copies each block's stream to
// the packed offset the host computed from the lengths, so only the packed bytes are read back.
//
// One workgroup per block, threads striding over the block's words — a word copy, not a byte
// copy, which is why both source and destination offsets are word-aligned.

struct CopyJob {
    src_byte: u32,
    dst_byte: u32,
    len_bytes: u32,
    _pad0: u32,
}

@group(0) @binding(6) var<storage, read> jobs: array<CopyJob>;
@group(0) @binding(7) var<storage, read_write> packed: array<u32>;

@compute @workgroup_size(64)
fn compact(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let b = wid.x;
    if (b >= params.num_blocks) {
        return;
    }
    let j = jobs[b];
    let words = (j.len_bytes + 3u) / 4u;
    let src = j.src_byte >> 2u;
    let dst = j.dst_byte >> 2u;
    for (var i = lid.x; i < words; i = i + 64u) {
        packed[dst + i] = out_words[src + i];
    }
}
