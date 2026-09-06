//! Adaptive binary arithmetic coding over code-blocks — CPU reference implementation.
//!
//! This is the coder the EBCOT evaluation settled on (RESEARCH_LOG 2026-09-06). It is a
//! *reference*: correctness and bitrate first, speed later. The arithmetic coder is the textbook
//! Witten-Neal-Cleary construction, chosen because it is short enough to verify by exhaustive
//! roundtrip rather than because it is fast.
//!
//! # Why code-blocks and not GNC's 256 streams
//!
//! GNC's Rice and rANS backends give each tile 256 independent streams, mapping coefficient *i*
//! to stream `i % 256`. That is what makes the decode embarrassingly parallel, and for Rice it
//! costs almost nothing (measured: under 1% against one stream per subband).
//!
//! It does not survive context modelling. A context-adaptive coder has to *learn* its
//! probabilities, and a stream of ~256 symbols is far too little data to learn 18 context
//! probabilities on: measured, the gain collapses from −6.6% to −0.7%, and to +2.4% without a
//! signalled initial-probability table. A 64×64 code-block gives one coder 4096 symbols, which is
//! enough — and a 1080p luma plane still holds about 450 independent code-blocks, so there is
//! ample GPU parallelism at frame scale even though it is not 256 per tile.
//!
//! Measured ceiling against GNC's Rice at its operating point: **−7.6% to −18.3%, mean −13.7%**
//! across four images, with cold-start adaptation and per-block overheads charged.
//!
//! # Scan and context
//!
//! Coefficient-major raster scan inside each block, so left, up, up-left and up-right are all
//! fully decoded when the current coefficient is coded. That is a richer context than EBCOT's
//! own plane-major significance state, and it is affordable here because embedded truncatability
//! — the only thing plane-major order buys — measured 0.00 dB on this codec.

/// Probability precision. 12 bits keeps the arithmetic in u32 without renormalisation surprises.
const PROB_BITS: u32 = 12;
const PROB_ONE: u32 = 1 << PROB_BITS;
/// Adaptation rate. 5 is the usual compromise: fast enough to track a 4096-symbol block, slow
/// enough not to thrash on noise.
const ADAPT_SHIFT: u32 = 5;

const HALF: u32 = 0x8000_0000;
const QUARTER: u32 = 0x4000_0000;
const THREE_QUARTER: u32 = 0xC000_0000;

/// Number of neighbourhood buckets a context is drawn from.
const NUM_BUCKETS: usize = 6;
/// Contexts: one bucket set per binary decision (significant, >1, >2).
const NUM_CONTEXTS: usize = NUM_BUCKETS * 3;

/// One adaptive binary probability, as P(bit == 0) scaled to `PROB_ONE`.
#[derive(Clone, Copy)]
struct Prob(u32);

impl Prob {
    fn new() -> Self {
        Prob(PROB_ONE / 2)
    }

    fn update(&mut self, bit: bool) {
        if bit {
            self.0 -= self.0 >> ADAPT_SHIFT;
        } else {
            self.0 += (PROB_ONE - self.0) >> ADAPT_SHIFT;
        }
        // Keep strictly inside (0, PROB_ONE) so the interval never collapses.
        self.0 = self.0.clamp(1, PROB_ONE - 1);
    }
}

struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    nbits: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self { bytes: Vec::new(), cur: 0, nbits: 0 }
    }

    fn put(&mut self, bit: bool) {
        self.cur = (self.cur << 1) | u8::from(bit);
        self.nbits += 1;
        if self.nbits == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        while self.nbits != 0 {
            self.put(false);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    nbits: u8,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0, nbits: 0 }
    }

    /// Past the end reads as zero, so a truncated stream terminates instead of panicking —
    /// the same rule the Rice tile parser follows.
    fn get(&mut self) -> bool {
        if self.pos >= self.bytes.len() {
            return false;
        }
        let b = (self.bytes[self.pos] >> (7 - self.nbits)) & 1;
        self.nbits += 1;
        if self.nbits == 8 {
            self.nbits = 0;
            self.pos += 1;
        }
        b != 0
    }
}

/// Witten-Neal-Cleary arithmetic encoder over adaptive binary contexts.
struct Encoder {
    low: u32,
    high: u32,
    pending: u64,
    out: BitWriter,
}

impl Encoder {
    fn new() -> Self {
        Self { low: 0, high: u32::MAX, pending: 0, out: BitWriter::new() }
    }

    fn emit(&mut self, bit: bool) {
        self.out.put(bit);
        while self.pending > 0 {
            self.out.put(!bit);
            self.pending -= 1;
        }
    }

    fn encode(&mut self, bit: bool, p: &mut Prob) {
        let range = (self.high - self.low) as u64 + 1;
        // Split proportional to P(bit == 0); -1 keeps `mid` strictly below `high`.
        let mid = self.low + ((range * p.0 as u64) >> PROB_BITS) as u32 - 1;
        if bit {
            self.low = mid + 1;
        } else {
            self.high = mid;
        }
        p.update(bit);
        loop {
            if self.high < HALF {
                self.emit(false);
            } else if self.low >= HALF {
                self.emit(true);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTER {
                self.pending += 1;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
        }
    }

    /// Bypass bit: coded at a fixed half probability, as CABAC does for suffix and sign bits.
    fn encode_bypass(&mut self, bit: bool) {
        let mut half = Prob(PROB_ONE / 2);
        let range = (self.high - self.low) as u64 + 1;
        let mid = self.low + ((range * half.0 as u64) >> PROB_BITS) as u32 - 1;
        if bit {
            self.low = mid + 1;
        } else {
            self.high = mid;
        }
        let _ = &mut half; // bypass bits do not adapt
        loop {
            if self.high < HALF {
                self.emit(false);
            } else if self.low >= HALF {
                self.emit(true);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTER {
                self.pending += 1;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        // Two bits plus the pending run disambiguate the final interval.
        self.pending += 1;
        if self.low < QUARTER {
            self.emit(false);
        } else {
            self.emit(true);
        }
        self.out.finish()
    }
}

struct Decoder<'a> {
    low: u32,
    high: u32,
    code: u32,
    input: BitReader<'a>,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        let mut input = BitReader::new(bytes);
        let mut code = 0u32;
        for _ in 0..32 {
            code = (code << 1) | u32::from(input.get());
        }
        Self { low: 0, high: u32::MAX, code, input }
    }

    fn renormalise(&mut self) {
        loop {
            if self.high < HALF {
                // nothing to subtract
            } else if self.low >= HALF {
                self.code -= HALF;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTER {
                self.code -= QUARTER;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.code = (self.code << 1) | u32::from(self.input.get());
        }
    }

    fn decode(&mut self, p: &mut Prob) -> bool {
        let range = (self.high - self.low) as u64 + 1;
        let mid = self.low + ((range * p.0 as u64) >> PROB_BITS) as u32 - 1;
        let bit = self.code > mid;
        if bit {
            self.low = mid + 1;
        } else {
            self.high = mid;
        }
        p.update(bit);
        self.renormalise();
        bit
    }

    fn decode_bypass(&mut self) -> bool {
        let range = (self.high - self.low) as u64 + 1;
        let mid = self.low + ((range * (PROB_ONE / 2) as u64) >> PROB_BITS) as u32 - 1;
        let bit = self.code > mid;
        if bit {
            self.low = mid + 1;
        } else {
            self.high = mid;
        }
        self.renormalise();
        bit
    }
}

/// Bucket the neighbourhood magnitude sum into a context index.
fn bucket(nb: u32) -> usize {
    if nb == 0 {
        0
    } else {
        ((32 - nb.leading_zeros()) as usize).min(NUM_BUCKETS - 1)
    }
}

/// Neighbourhood sum for position (y, x) from already-coded coefficients.
///
/// Left, up, up-left and up-right — all decoded before (y, x) in a raster scan, so a decoder has
/// them. Out-of-block positions count as zero rather than reaching into a neighbouring block,
/// which keeps blocks genuinely independent and decodable in any order.
fn neighbour_sum(mag: &[u32], w: usize, y: usize, x: usize) -> u32 {
    let at = |yy: usize, xx: usize| -> u32 {
        if xx >= w {
            0
        } else {
            mag[yy * w + xx]
        }
    };
    let mut s = 0u32;
    if x > 0 {
        s = s.saturating_add(at(y, x - 1));
    }
    if y > 0 {
        s = s.saturating_add(at(y - 1, x));
        if x > 0 {
            s = s.saturating_add(at(y - 1, x - 1));
        }
        s = s.saturating_add(at(y - 1, x + 1));
    }
    s
}

/// Encode one code-block of quantised coefficients.
///
/// Binarisation per coefficient: significant?, |v|>1?, |v|>2?, then the remainder as Exp-Golomb
/// order 0 and the sign, both as bypass bits. Only the three decisions are context-coded, which is
/// where the measured gain sits; bypassing the rest keeps the context count at 18.
pub fn encode_block(coefficients: &[i32], width: usize) -> Vec<u8> {
    assert!(width > 0, "code-block width must be non-zero");
    assert_eq!(
        coefficients.len() % width,
        0,
        "code-block must be rectangular: {} coefficients at width {width}",
        coefficients.len()
    );
    let height = coefficients.len() / width;
    let mut probs = [Prob::new(); NUM_CONTEXTS];
    let mut enc = Encoder::new();
    let mut mag = vec![0u32; coefficients.len()];

    for y in 0..height {
        for x in 0..width {
            let v = coefficients[y * width + x];
            let a = v.unsigned_abs();
            let ctx = bucket(neighbour_sum(&mag, width, y, x));
            enc.encode(a > 0, &mut probs[ctx]);
            if a > 0 {
                enc.encode(a > 1, &mut probs[NUM_BUCKETS + ctx]);
                if a > 1 {
                    enc.encode(a > 2, &mut probs[2 * NUM_BUCKETS + ctx]);
                    if a > 2 {
                        // Exp-Golomb order 0 of (a - 3), MSB-first, as bypass bits.
                        let n = a - 3 + 1;
                        let len = 32 - n.leading_zeros();
                        for _ in 0..len - 1 {
                            enc.encode_bypass(false);
                        }
                        for i in (0..len).rev() {
                            enc.encode_bypass((n >> i) & 1 != 0);
                        }
                    }
                }
                enc.encode_bypass(v < 0);
            }
            mag[y * width + x] = a;
        }
    }
    enc.finish()
}

/// Inverse of `encode_block`. `count` and `width` come from the caller, as they do for Rice tiles.
pub fn decode_block(bytes: &[u8], count: usize, width: usize) -> Vec<i32> {
    assert!(width > 0, "code-block width must be non-zero");
    let height = count / width;
    let mut probs = [Prob::new(); NUM_CONTEXTS];
    let mut dec = Decoder::new(bytes);
    let mut out = vec![0i32; count];
    let mut mag = vec![0u32; count];

    for y in 0..height {
        for x in 0..width {
            let ctx = bucket(neighbour_sum(&mag, width, y, x));
            let mut a: u32 = 0;
            if dec.decode(&mut probs[ctx]) {
                a = 1;
                if dec.decode(&mut probs[NUM_BUCKETS + ctx]) {
                    a = 2;
                    if dec.decode(&mut probs[2 * NUM_BUCKETS + ctx]) {
                        // Exp-Golomb order 0
                        let mut zeros = 0u32;
                        while !dec.decode_bypass() {
                            zeros += 1;
                            if zeros > 32 {
                                return out; // corrupt or truncated
                            }
                        }
                        let mut n = 1u32;
                        for _ in 0..zeros {
                            n = (n << 1) | u32::from(dec.decode_bypass());
                        }
                        a = n - 1 + 3;
                    }
                }
                let neg = dec.decode_bypass();
                out[y * width + x] = if neg { -(a as i32) } else { a as i32 };
            }
            mag[y * width + x] = a;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(coefficients: &[i32], width: usize, label: &str) {
        let bytes = encode_block(coefficients, width);
        let back = decode_block(&bytes, coefficients.len(), width);
        assert_eq!(back, coefficients, "{label}: roundtrip must be exact");
    }

    #[test]
    fn roundtrip_shapes_and_distributions() {
        roundtrip(&vec![0i32; 64 * 64], 64, "all zero");
        roundtrip(&vec![1i32; 64 * 64], 64, "all one");
        roundtrip(&vec![-1i32; 64 * 64], 64, "all minus one");
        roundtrip(&(0..64 * 64).map(|i| (i % 7) as i32 - 3).collect::<Vec<_>>(), 64, "small cycle");
        // Sparse with a large outlier: exercises the Exp-Golomb suffix and the significance path.
        let mut sparse = vec![0i32; 64 * 64];
        sparse[0] = 1;
        sparse[37] = -9001;
        sparse[64 * 64 - 1] = 4095;
        roundtrip(&sparse, 64, "sparse with outliers");
        // Non-square blocks happen at subband edges.
        roundtrip(&(0..64 * 17).map(|i| ((i * 31) % 11) as i32 - 5).collect::<Vec<_>>(), 17, "17 wide");
        roundtrip(&(0..5).map(|i| i as i32).collect::<Vec<_>>(), 5, "single row");
    }

    #[test]
    fn roundtrip_extremes() {
        // i32::MIN has no positive counterpart; `unsigned_abs` is why this must be tested.
        roundtrip(&[i32::MIN, i32::MAX, 0, -1, 1, 0, 0, 0], 4, "i32 extremes");
    }

    #[test]
    fn corrupt_stream_does_not_panic() {
        let coefficients: Vec<i32> = (0..64 * 64).map(|i| (i % 13) as i32 - 6).collect();
        let bytes = encode_block(&coefficients, 64);
        for cut in [0, 1, 7, bytes.len() / 3, bytes.len() / 2] {
            let _ = decode_block(&bytes[..cut], coefficients.len(), 64);
        }
        for pos in [0, 3, bytes.len() / 2, bytes.len() - 1] {
            let mut bad = bytes.clone();
            bad[pos] ^= 0xFF;
            let _ = decode_block(&bad, coefficients.len(), 64);
        }
    }

    /// The point of the whole exercise: on a sparse, clustered field of the kind a wavelet
    /// subband produces, the context-coded stream must be materially smaller than the
    /// zeroth-order entropy of the same symbols. If this fails the contexts are not working.
    #[test]
    fn beats_zeroth_order_entropy_on_clustered_data() {
        // Clustered significance: a few busy regions in an otherwise empty band.
        let w = 64usize;
        let mut c = vec![0i32; w * w];
        for y in 0..w {
            for x in 0..w {
                let busy = (y / 8 + x / 8) % 3 == 0;
                if busy {
                    c[y * w + x] = (((y * 31 + x * 17) % 9) as i32) - 4;
                }
            }
        }
        let coded_bits = encode_block(&c, w).len() * 8;

        let mut counts = std::collections::HashMap::new();
        for &v in &c {
            *counts.entry(v).or_insert(0usize) += 1;
        }
        let n = c.len() as f64;
        let h0: f64 = counts
            .values()
            .map(|&k| {
                let p = k as f64 / n;
                -p * p.log2()
            })
            .sum::<f64>()
            * n;

        assert!(
            (coded_bits as f64) < h0 * 0.9,
            "context coding should beat H0 by more than 10% on clustered data: \
             {coded_bits} bits vs H0 {h0:.0}"
        );
    }
}
