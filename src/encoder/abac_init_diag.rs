//! ENT-6: what does abac's cold start actually cost, and what would a warm one buy?
//!
//! Every column in [`super::coef_entropy_diag`] is a *bound* — an ideal coder's conditional
//! entropy, with the probabilities pooled over a whole plane's worth of a subband. Bounds cannot
//! answer ENT-6, because ENT-6 is not about the model at all: abac's model is within 4.1% of the
//! bound on the 82% of rate that lives in full 64×64 blocks (`0024`). What it is about is that
//! abac opens **every** code-block at p = 1/2, and the LL and level-3/4/5 bands at tile 256 with
//! 5 levels are 8, 16 and 32 px square, so each is one short block with 64 to 1024 coefficients
//! to learn 18 context probabilities on. A bound pooled over the plane cannot see that, by
//! construction: pooling *is* the thing being paid for.
//!
//! So this module simulates the coder instead of bounding it. It walks each code-block exactly as
//! [`super::abac::encode_block`] does, drives the **shipped** [`Prob`] update rule, and charges
//! `−log2 p` per context-coded decision plus one bit per bypassed decision. Two initialisations,
//! same walk:
//!
//! | column | initialisation |
//! |---|---|
//! | `Ahalf` | p = 1/2 for every context, which is what abac does today |
//! | `Awarm` | the empirical P(bit = 0) of that (plane, subband, context), quantised to one byte |
//!
//! **`Ahalf` is the canary, and it is a strong one.** It must reproduce the bytes the bitstream
//! actually spent, because it is modelling the real coder with the real update rule on the real
//! coefficients — the only differences left are the arithmetic coder's rounding and the per-block
//! length field. If `Ahalf` and `shipped` disagree by more than about a percent, this diagnostic
//! is wrong and `Awarm` means nothing.
//!
//! **What `Awarm` costs to ship is not in the column.** `Awarm`'s table is 18 bytes per (plane,
//! subband) — 864 bytes for a 5-level frame, about 0.05% of a 1080p frame at q=90 — *if it is
//! signalled once per frame*. BACKLOG's candidate 1 says "per subband per tile", which is 288
//! bytes × 120 tiles = 34 KB, about 1.9% of the same frame, against a target win of 2%. The
//! header cost is therefore not a footnote here, it decides the item, and the two designs differ
//! by a factor of 40. The `hdr` column prints the per-frame cost so the net is visible next to
//! the gross.
//!
//! Read-only on data the encoder has already produced — it cannot move the bitstream
//! (`docs/decisions/0010`).

use super::abac::{bucket, neighbour_sum, Prob, NUM_BUCKETS, PROB_ONE};
use super::coef_entropy_diag::BinCount;

/// Contexts abac carries: one bucket set per binary decision.
pub(crate) const NUM_CONTEXTS: usize = NUM_BUCKETS * 3;

/// Bytes a signalled initialisation table costs per (plane, subband): one per context.
pub(crate) const TABLE_BYTES_PER_BAND: f64 = NUM_CONTEXTS as f64;

/// Quantise an empirical P(bit = 0) to the byte a signalled table would carry, and expand it back
/// to the coder's scale. Doing both here is deliberate: the diagnostic must pay the quantisation
/// a real header would pay, or it is pricing a table nobody can send.
fn quantised_p_zero(counts: &BinCount) -> u32 {
    if counts.n == 0 {
        return PROB_ONE / 2;
    }
    let p0 = 1.0 - counts.ones as f64 / counts.n as f64;
    // One byte, never 0 or 255 — the coder's own clamp forbids a certain probability, and a
    // signalled 0 would be a decoder that cannot code the symbol it is about to see.
    let byte = (p0 * 255.0).round().clamp(1.0, 254.0) as u32;
    byte * PROB_ONE / 255
}

/// The initialisation table for one (plane, subband): one probability per context.
pub(crate) fn warm_init(ctx: &[BinCount]) -> Vec<u32> {
    debug_assert_eq!(ctx.len(), NUM_CONTEXTS);
    ctx.iter().map(quantised_p_zero).collect()
}

/// The cold table abac uses today.
pub(crate) fn cold_init() -> Vec<u32> {
    vec![PROB_ONE / 2; NUM_CONTEXTS]
}

/// Which order the coder visits a code-block's coefficients in.
///
/// The distinction is ENT-8's whole question. `Raster` is what abac does: one thread per
/// code-block walking rows left to right, so all four of `neighbour_sum`'s causal neighbours —
/// left, up, up-left, up-right — are always available. `Lockstep` is BPC-PaCo's schedule ported
/// to abac's template: the block is cut into two-column stripes with a thread each, and every
/// thread codes the left column of row *y* before any thread codes a right column, which is what
/// makes 32 threads per block possible.
///
/// **abac does not inherit BPC-PaCo's "the parallelism is free" result, and this is why.**
/// BPC-PaCo reads all eight neighbours, so its stripe schedule still averages 4 already-coded
/// ones (3 on a left column, 5 on a right), the same as a raster scan. abac reads four and they
/// are all on the causal side: a left-column coefficient loses the *left* neighbour, because
/// `x - 1` is a right column visited later in the same row, and a right-column one keeps all
/// four. So the average falls from 4 of 4 to **3.5 of 4** and the cost has to be measured on
/// abac's own template.
/// The stripe width is a dial, not a constant. With stripes `k` columns wide, only the *first*
/// column of each stripe loses its left neighbour — every other column's left neighbour sits in
/// an earlier phase of the same stripe and is already coded — so **1 in k** coefficients pays,
/// against `w / k` threads per block. `k = 2` is BPC-PaCo's own width and the most parallel;
/// `k = w` is a single stripe and is exactly [`Scan::Raster`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Scan {
    Raster,
    Lockstep { stripe_cols: usize },
}

impl Scan {
    /// The visit order for a `w x h` block. Nothing else about the walk changes between scans:
    /// `mag` is zero-initialised and written only when a position is visited, so which neighbours
    /// `neighbour_sum` can see follows from the order alone — exactly as it would for a decoder.
    fn order(self, w: usize, h: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            match self {
                Scan::Raster => out.extend((0..w).map(|x| (y, x))),
                Scan::Lockstep { stripe_cols } => {
                    let k = stripe_cols.max(1);
                    for phase in 0..k {
                        let mut x = phase;
                        while x < w {
                            out.push((y, x));
                            x += k;
                        }
                    }
                }
            }
        }
        out
    }

    /// Threads a `w`-wide code-block would get under this scan — one per stripe.
    pub(crate) fn threads_per_block(self, w: usize) -> usize {
        match self {
            Scan::Raster => 1,
            Scan::Lockstep { stripe_cols } => w.div_ceil(stripe_cols.max(1)),
        }
    }
}

/// Bits abac's real engine spends on one code-block, started from `init`.
///
/// The walk is [`super::abac::encode_block`]'s, decision for decision, and the probability update
/// is the shipped [`Prob::update`] — imported rather than reimplemented, so the simulation cannot
/// drift away from the coder it is modelling. Bypassed bits (the Exp-Golomb remainder and the
/// sign) are charged at one bit each, which is exactly what `encode_bypass` costs.
pub(crate) fn adapt_bits(coefficients: &[i32], width: usize, init: &[u32]) -> f64 {
    adapt_bits_scan(coefficients, width, init, Scan::Raster)
}

/// [`adapt_bits`] with the visit order named explicitly — ENT-8's step 1.
///
/// **This models the pre-ENT-9 binarisation**, with the Exp-Golomb prefix bypassed at `2*len - 1`
/// bits. It is deliberately not updated to the shipped coder: ENT-8's published scan figures
/// (`k=2 +0.74%/+0.65%`, and the rest of that table) were taken on this binarisation, and they
/// stay reproducible only while it does. For a model of what the coder does *now*, use
/// [`adapt_bits_prefix_ctx`].
pub(crate) fn adapt_bits_scan(coefficients: &[i32], width: usize, init: &[u32], scan: Scan) -> f64 {
    let height = coefficients.len() / width;
    let mut probs: Vec<Prob> = init.iter().map(|&p| Prob::from_p_zero(p)).collect();
    let mut mag = vec![0u32; coefficients.len()];
    let mut bits = 0.0;

    for (y, x) in scan.order(width, height) {
        let v = coefficients[y * width + x];
        let a = v.unsigned_abs();
        let ctx = bucket(neighbour_sum(&mag, width, y, x));
        code(&mut bits, &mut probs, ctx, a > 0);
        if a > 0 {
            code(&mut bits, &mut probs, NUM_BUCKETS + ctx, a > 1);
            if a > 1 {
                code(&mut bits, &mut probs, 2 * NUM_BUCKETS + ctx, a > 2);
                if a > 2 {
                    // Exp-Golomb order 0 of (a - 3) as bypass bits: 2*len - 1 of them.
                    let n = a - 3 + 1;
                    let len = 32 - n.leading_zeros();
                    bits += f64::from(2 * len - 1);
                }
            }
            bits += 1.0; // sign, bypassed
        }
        mag[y * width + x] = a;
    }
    bits
}

/// The shipped binarisation since ENT-9 candidate A: the Exp-Golomb unary prefix
/// **context-coded** rather than bypassed. Was step 2 milestone 1's instrument; it is now the
/// model of the real coder, which is why the canary below compares against it.
///
/// This is the check `0063` said had to come before any shader work. Step 1b priced candidate A
/// on statistics pooled per plane and subband, which is generous by construction: it charges no
/// adaptation and lets every block share one set of counts. Here the 24 new contexts are
/// **cold-started per code-block**, exactly like the 18 they join, and learn on the same 4096
/// symbols. That is the effect that collapsed abac's 256-stream variant from −6.6% to −0.7%, so
/// a pooled bound is not evidence about it either way.
///
/// The mantissa and the sign stay bypassed, as in the shipped coder — candidate B is not modelled
/// here.
pub(crate) fn adapt_bits_prefix_ctx(coefficients: &[i32], width: usize, init: &[u32]) -> f64 {
    let height = coefficients.len() / width;
    // 18 shipped contexts, then 24 for the prefix: (bucket, min(position, 3)).
    let mut probs: Vec<Prob> = init.iter().map(|&p| Prob::from_p_zero(p)).collect();
    probs.extend((0..NUM_BUCKETS * 4).map(|_| Prob::from_p_zero(PROB_ONE / 2)));
    let base = init.len();
    let mut mag = vec![0u32; coefficients.len()];
    let mut bits = 0.0;

    for (y, x) in Scan::Raster.order(width, height) {
        let v = coefficients[y * width + x];
        let a = v.unsigned_abs();
        let ctx = bucket(neighbour_sum(&mag, width, y, x));
        code(&mut bits, &mut probs, ctx, a > 0);
        if a > 0 {
            code(&mut bits, &mut probs, NUM_BUCKETS + ctx, a > 1);
            if a > 1 {
                code(&mut bits, &mut probs, 2 * NUM_BUCKETS + ctx, a > 2);
                if a > 2 {
                    let n = a - 3 + 1;
                    let len = 32 - n.leading_zeros();
                    // The prefix is `len - 1` "keep going" bits then one "stop", each coded in
                    // its own (bucket, position) context instead of at p = 1/2.
                    for i in 0..len {
                        let slot = (i as usize).min(3);
                        code(
                            &mut bits,
                            &mut probs,
                            base + slot * NUM_BUCKETS + ctx,
                            i == len - 1,
                        );
                    }
                    // Mantissa: `len - 1` bypass bits, unchanged.
                    bits += f64::from(len - 1);
                }
            }
            bits += 1.0; // sign, bypassed
        }
        mag[y * width + x] = a;
    }
    bits
}

/// Charge one context-coded decision at `−log2 p` and advance the probability, exactly as the
/// coder does. Split out rather than inlined so the accumulator and the probability array can be
/// borrowed separately.
fn code(bits: &mut f64, probs: &mut [Prob], c: usize, bit: bool) {
    let p0 = probs[c].p_zero();
    *bits += -(if bit { 1.0 - p0 } else { p0 }).log2();
    probs[c].update(bit);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::abac::{encode_block, encode_block_rc, Coder};

    /// The instrument's canary, and it is measured rather than asserted from taste.
    ///
    /// [`adapt_bits`] models the real coder's probability update but charges an ideal `−log2 p`
    /// per decision, so the real encoder must always spend **more** — the range coder's
    /// truncation, its flush, and byte alignment. This test states the direction as a hard
    /// assertion and the *size* as a bound derived from what the coder actually does here, so
    /// that `coef_entropy_diag`'s consistency check has a number behind it instead of a guess.
    #[test]
    fn simulation_is_a_lower_bound_on_the_real_coder() {
        let cold = cold_init();
        let mut worst = 0.0f64;
        let mut worst_case = String::new();
        // Sparsity is the axis that matters and the one convenience omits. A real level-1 subband
        // at q=90 is mostly zeros, and that is where the coder's 12-bit probability resolution
        // costs most: the ideal cost of the frequent symbol falls below what a quantised
        // probability can express, so the gap to `−log2 p` widens. Dense content hides it —
        // measured here at 8 bits per block dense against ~10x that at 1-in-64 density.
        for &(w, h) in &[
            (8usize, 8usize),
            (16, 16),
            (32, 32),
            (64, 64),
            (64, 17),
            (5, 3),
        ] {
            for &spread in &[1i32, 7, 64, 1000] {
                for &density in &[1usize, 4, 11, 64, 512] {
                    let coefficients: Vec<i32> = (0..w * h)
                        .map(|i| {
                            // Deterministic, structured, and not all one sign: a plausible subband.
                            let v =
                                ((i * 2654435761usize) % (spread as usize * 2 + 1)) as i32 - spread;
                            if i % density != 1 % density.max(2) {
                                0
                            } else {
                                v
                            }
                        })
                        .collect();
                    // **The model must be of the *shipped* binarisation, which since ENT-9 is
                    // `adapt_bits_prefix_ctx`.** `adapt_bits` still models the pre-ENT-9 coder — it
                    // is ENT-8's instrument and its published scan figures were taken on that
                    // binarisation, so it stays as it was. Pointing this canary at it after the
                    // prefix became context-coded made it fail with `real 344 < simulated 366`,
                    // which is the canary doing its job: the coder had got *cheaper* than the model
                    // of a binarisation it no longer uses.
                    let sim_bits = adapt_bits_prefix_ctx(&coefficients, w, &cold);
                    // **Both engines.** They share this binarisation and this probability model but
                    // not a bitstream, and their per-block flush differs — the range coder's is
                    // several bytes where the interval coder's is one. Testing only one is how the
                    // first version of this canary came to disagree with the diagnostic by 5 bytes
                    // per block: the shipped tiles are `Coder::Range` and the test was measuring
                    // `Coder::Interval`.
                    for (engine, real) in [
                        (Coder::Interval, encode_block(&coefficients, w)),
                        (Coder::Range, encode_block_rc(&coefficients, w)),
                    ] {
                        let real_bits = (real.len() * 8) as f64;
                        assert!(
                            real_bits >= sim_bits,
                            "{engine:?} {w}x{h} spread {spread} density 1/{density}: real \
                         {real_bits} < simulated {sim_bits} — the simulation is supposed to be a \
                         lower bound, so the walk has diverged from the coder"
                        );
                        if real_bits - sim_bits > worst {
                            worst = real_bits - sim_bits;
                            worst_case =
                                format!("{engine:?} {w}x{h} spread {spread} density 1/{density}");
                        }
                    }
                }
            }
        }
        // Measured here, not chosen: the largest gap over 240 engine/geometry/spread/density
        // combinations. The band is generous by roughly 2x so content variation cannot make it
        // flap, and it is what justifies `coef_entropy_diag`'s per-block consistency check.
        eprintln!(
            "worst per-block overhead over 240 combinations: {worst:.1} bits, at {worst_case}"
        );
        assert!(
            worst < 400.0,
            "worst per-block overhead is {worst} bits at {worst_case}, above the 400-bit band \
             this test establishes — re-derive the band before trusting the ENT-6 columns"
        );
    }

    /// ENT-8's premise, asserted rather than reasoned about: under the lockstep scan abac's
    /// **four**-neighbour causal template loses the left neighbour on even columns and keeps all
    /// four on odd ones, averaging **3.5 of 4** — not the 4 of 4 that BPC-PaCo's eight-neighbour
    /// template keeps under the same schedule. If this ever reads 4.0 for `Lockstep`, the scan
    /// being priced is not the one that makes 32 threads per block possible.
    #[test]
    fn lockstep_costs_abacs_template_half_a_neighbour() {
        let (w, h) = (16usize, 16usize);
        for (scan, want_even, want_odd) in [
            (Scan::Raster, 4usize, 4usize),
            (Scan::Lockstep { stripe_cols: 2 }, 3, 4),
        ] {
            let order = scan.order(w, h);
            assert_eq!(
                order.len(),
                w * h,
                "{scan:?} must visit each position exactly once"
            );
            let mut visited = vec![false; w * h];
            let mut per_parity = [Vec::new(), Vec::new()];
            for &(y, x) in &order {
                // Interior only; the block edge legitimately has fewer neighbours. abac's
                // template is left, up, up-left, up-right.
                if y > 0 && x > 0 && x + 1 < w {
                    let n = usize::from(visited[y * w + x - 1])
                        + usize::from(visited[(y - 1) * w + x])
                        + usize::from(visited[(y - 1) * w + x - 1])
                        + usize::from(visited[(y - 1) * w + x + 1]);
                    per_parity[x % 2].push(n);
                }
                visited[y * w + x] = true;
            }
            assert_eq!(
                scan.threads_per_block(64),
                if scan == Scan::Raster { 1 } else { 32 },
                "{scan:?} thread count"
            );
            for (parity, want) in [(0usize, want_even), (1, want_odd)] {
                assert!(
                    per_parity[parity].iter().all(|&n| n == want),
                    "{scan:?} parity {parity}: wanted {want} coded neighbours, got {:?}",
                    per_parity[parity]
                );
            }
        }
    }
}
