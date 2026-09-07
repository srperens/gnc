//! INTRA-1 step 1: what GNC spends against the entropy of GNC's own coefficients.
//!
//! Gated behind `GNC_COEF_ENTROPY=1`, zero cost when unset. Reads the **shipped** abac tiles —
//! not a re-encode, not a simulation — decodes them back to the coefficients the bitstream
//! actually carries, and prices those coefficients three ways per subband:
//!
//! * `H0` — zeroth-order entropy of the signed quantised symbols, pooled per plane and subband.
//!   The floor for any coder that ignores context. Above the shipped rate is the normal result
//!   and is what says the context model is doing something.
//! * `Hctx` — conditional entropy of abac's **own** binarisation under abac's **own** context,
//!   pooled per plane and subband. Same three binary decisions, same 6 magnitude buckets, same
//!   bypassed Exp-Golomb suffix and sign. The difference from the shipped rate is therefore
//!   purely adaptation loss: cold-started per-block probabilities, the arithmetic coder's own
//!   rounding, and the per-block length field.
//! * `Hnb`, `Hnb0`, `Hbig` — the same coefficients under *richer* models: the whole magnitude
//!   coded as one symbol conditioned on a finer causal neighbourhood context. `Hnb` uses 50
//!   contexts per subband with the model cost charged, `Hnb0` is the same without that charge,
//!   and `Hbig` widens the template to 200 contexts. Together they bound what any
//!   neighbourhood-context entropy coder over these coefficients could reach, which is what
//!   bounds the headroom left in the coder.
//!
//! **Why this decides the item.** ENT-4 measured GNC at +27.1% of rate against JPEG 2000 in 9/7
//! mode at matched RGB PSNR, with abac on. If `Hnb` sits close to the shipped rate, no entropy
//! coder over these coefficients can recover 27 points and the gap is upstream — quantisation,
//! lifting normalisation, tiling. If `Hnb` sits far below it, there is coder headroom left and
//! the offline context estimate understated it for a third time.
//!
//! The bound is deliberately generous to the coder: contexts are pooled across the whole plane
//! rather than learned per block, so `Hnb` is a *lower* bound on any real implementation. A
//! generous bound that still cannot close the gap is the strong form of the conclusion.
//!
//! Read-only on data the encoder has already produced — it cannot move the bitstream
//! (`docs/decisions/0010`).

use std::collections::HashMap;

use super::abac::{bucket, neighbour_sum, NUM_BUCKETS};
use super::abac_tile::{abac_decode_tile, band_name, code_blocks_banded, AbacTile};

/// Binary counts for one adaptive context: how many decisions, how many of them were 1.
#[derive(Default, Clone, Copy)]
struct BinCount {
    n: u64,
    ones: u64,
}

impl BinCount {
    fn push(&mut self, bit: bool) {
        self.n += 1;
        self.ones += u64::from(bit);
    }

    /// n * H(p) — what an ideal adaptive coder converges to on this context.
    fn bits(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let p = self.ones as f64 / self.n as f64;
        if p <= 0.0 || p >= 1.0 {
            return 0.0;
        }
        self.n as f64 * -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
    }
}

/// Entropy of a symbol histogram, and the KT model cost of signalling it, separately.
///
/// `(A - 1)/2 * log2(n)` is the standard asymptotic redundancy of learning an `A`-symbol
/// distribution from `n` observations. Charging it keeps a context split from looking free: a
/// model with one context per coefficient reaches zero entropy and infinite model cost. It is
/// returned separately because the charge is itself an assumption, and the conclusion here has to
/// survive dropping it — a bound that only holds because of its own overhead term is not a bound.
fn hist_bits(h: &HashMap<i64, u64>) -> (f64, f64) {
    let n: u64 = h.values().sum();
    if n == 0 {
        return (0.0, 0.0);
    }
    let nf = n as f64;
    let entropy: f64 = h
        .values()
        .map(|&c| {
            let p = c as f64 / nf;
            -(c as f64) * p.log2()
        })
        .sum();
    let model = (h.len().saturating_sub(1)) as f64 * 0.5 * nf.log2();
    (entropy, model)
}

/// Finer neighbourhood context for the `Hnb` bound.
///
/// Two parts, because the shipped bucket collapses them into one sum: the two nearest causal
/// neighbours (left, up) carry most of the information, the two diagonals carry the rest, and
/// separating them lets the model see an edge that the sum cannot distinguish from a texture.
/// 10 x 5 = 50 contexts per subband, against the shipped 6 — affordable here because the
/// statistics are pooled over a whole plane's worth of a band, hundreds of thousands of symbols.
fn rich_context(mag: &[u32], w: usize, y: usize, x: usize) -> (usize, usize) {
    let at = |yy: usize, xx: usize| -> u32 {
        if xx >= w {
            0
        } else {
            mag[yy * w + xx]
        }
    };
    let mut near = 0u32;
    let mut diag = 0u32;
    let mut far = 0u32;
    if x > 0 {
        near = near.saturating_add(at(y, x - 1));
        if x > 1 {
            far = far.saturating_add(at(y, x - 2));
        }
    }
    if y > 0 {
        near = near.saturating_add(at(y - 1, x));
        if x > 0 {
            diag = diag.saturating_add(at(y - 1, x - 1));
        }
        diag = diag.saturating_add(at(y - 1, x + 1));
        if y > 1 {
            far = far.saturating_add(at(y - 2, x));
        }
    }
    let lb = |v: u32, cap: usize| -> usize {
        if v == 0 {
            0
        } else {
            ((32 - v.leading_zeros()) as usize).min(cap - 1)
        }
    };
    let nb = lb(near, 10) * 5 + lb(diag, 5);
    (nb, nb * 4 + lb(far, 4))
}

/// Everything accumulated for one (plane, subband) cell.
#[derive(Default)]
struct BandStats {
    coefficients: u64,
    /// Bytes the shipped bitstream actually spends here, including each block's length field.
    shipped_bytes: f64,
    blocks: u64,
    /// Signed-symbol histogram for `H0`.
    h0: HashMap<i64, u64>,
    /// abac's 18 contexts, for `Hctx`.
    ctx: Vec<BinCount>,
    /// Bits abac bypasses: Exp-Golomb suffix and sign, charged at one bit each as the coder does.
    bypass_bits: f64,
    /// Magnitude histogram per rich context, for `Hnb`.
    rich: HashMap<usize, HashMap<i64, u64>>,
    /// The same, under a 4x larger context that also sees the two coefficients two steps away.
    big: HashMap<usize, HashMap<i64, u64>>,
    /// One sign bit per significant coefficient; signs are not modelled by either bound.
    sign_bits: f64,
}

impl BandStats {
    fn new() -> Self {
        Self {
            ctx: vec![BinCount::default(); NUM_BUCKETS * 3],
            ..Default::default()
        }
    }

    fn h0_bits(&self) -> f64 {
        let (e, m) = hist_bits(&self.h0);
        e + m
    }

    fn hctx_bits(&self) -> f64 {
        self.ctx.iter().map(BinCount::bits).sum::<f64>() + self.bypass_bits
    }

    /// `(with model cost, without)`. The second is the pure conditional entropy — an absolute
    /// floor no coder with this context can go below, model cost or not.
    fn hnb_bits(&self) -> (f64, f64) {
        let (e, m) = self
            .rich
            .values()
            .map(hist_bits)
            .fold((0.0, 0.0), |(a, b), (e, m)| (a + e, b + m));
        (e + m + self.sign_bits, e + self.sign_bits)
    }

    fn hbig_bits(&self) -> f64 {
        let (e, m) = self
            .big
            .values()
            .map(hist_bits)
            .fold((0.0, 0.0), |(a, b), (e, m)| (a + e, b + m));
        e + m + self.sign_bits
    }
}

/// Walk one code-block exactly as `abac::encode_block` does, accumulating both models.
fn accumulate_block(st: &mut BandStats, coefficients: &[i32], width: usize) {
    let height = coefficients.len() / width;
    let mut mag = vec![0u32; coefficients.len()];
    for y in 0..height {
        for x in 0..width {
            let v = coefficients[y * width + x];
            let a = v.unsigned_abs();

            *st.h0.entry(v as i64).or_insert(0) += 1;

            // --- abac's own model, decision for decision ---
            let c = bucket(neighbour_sum(&mag, width, y, x));
            st.ctx[c].push(a > 0);
            if a > 0 {
                st.ctx[NUM_BUCKETS + c].push(a > 1);
                if a > 1 {
                    st.ctx[2 * NUM_BUCKETS + c].push(a > 2);
                    if a > 2 {
                        // Exp-Golomb order 0 of (a - 3), as bypass bits: 2*len - 1 of them.
                        let n = a - 3 + 1;
                        let len = 32 - n.leading_zeros();
                        st.bypass_bits += f64::from(2 * len - 1);
                    }
                }
                st.bypass_bits += 1.0; // sign
                st.sign_bits += 1.0;
            }

            // --- the richer bounds, on the same causal information ---
            let (rc, bc) = rich_context(&mag, width, y, x);
            *st.rich.entry(rc).or_default().entry(i64::from(a)).or_insert(0) += 1;
            *st.big.entry(bc).or_default().entry(i64::from(a)).or_insert(0) += 1;

            mag[y * width + x] = a;
        }
    }
    st.coefficients += coefficients.len() as u64;
}

/// Bytes the tile header spends on one block's length field, as `serialize_tile_abac` writes it.
fn length_field_bytes(len: u32) -> f64 {
    match len {
        0..=127 => 1.0,
        128..=16383 => 2.0,
        16384..=2_097_151 => 3.0,
        _ => 4.0,
    }
}

/// Price the shipped abac tiles against the entropy of the coefficients they carry.
///
/// `plane_tile_counts` gives how many of `tiles` belong to Y, Co and Cg, in that order — the
/// order `encode_entropy` appends them in.
pub fn run(tiles: &[AbacTile], plane_tile_counts: [usize; 3], qstep: f32) {
    if tiles.is_empty() {
        eprintln!("[coef-entropy] no abac tiles — run with --abac");
        return;
    }
    let num_levels = tiles[0].num_levels;
    let cb = tiles[0].cb_size as usize;
    let nbands = if num_levels == 0 { 1 } else { 1 + 3 * num_levels as usize };
    eprintln!(
        "[coef-entropy] GNC_COEF_ENTROPY active: qstep={qstep}, {} tiles, tile {}px, {num_levels} \
         levels, cb {cb}px, coder {:?}",
        tiles.len(),
        tiles[0].tile_size,
        tiles[0].coder
    );

    let planes = ["Y", "Co", "Cg"];
    let mut stats: Vec<Vec<BandStats>> = (0..3)
        .map(|_| (0..nbands).map(|_| BandStats::new()).collect())
        .collect();

    let mut idx = 0usize;
    for (p, &count) in plane_tile_counts.iter().enumerate() {
        for _ in 0..count {
            if idx >= tiles.len() {
                break;
            }
            let tile = &tiles[idx];
            idx += 1;
            let ts = tile.tile_size as usize;
            let coefficients = abac_decode_tile(tile);
            let blocks = code_blocks_banded(ts, tile.num_levels, tile.cb_size as usize);
            debug_assert_eq!(blocks.len(), tile.block_lengths.len());
            for (b, &(bx, by, bw, bh, band)) in blocks.iter().enumerate() {
                let st = &mut stats[p][band];
                let len = tile.block_lengths[b];
                st.shipped_bytes += f64::from(len) + length_field_bytes(len);
                st.blocks += 1;
                let mut blk = Vec::with_capacity(bw * bh);
                for y in 0..bh {
                    let row = (by + y) * ts + bx;
                    blk.extend_from_slice(&coefficients[row..row + bw]);
                }
                accumulate_block(st, &blk, bw);
            }
        }
    }
    if idx != tiles.len() {
        eprintln!(
            "[coef-entropy] WARNING: plane tile counts cover {idx} of {} tiles — the split is \
             wrong and the per-plane rows below are not trustworthy",
            tiles.len()
        );
    }

    eprintln!(
        "  {:>5} {:>5} {:>10} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>8} {:>8}",
        "plane", "band", "coeffs", "shipped B", "H0 B", "Hctx B", "Hnb B", "Hnb0 B", "Hbig B",
        "vs Hnb", "vs Hbig"
    );
    let mut t = [0.0f64; 6];
    let mut t_n = 0u64;
    for (p, plane) in planes.iter().enumerate() {
        let mut pt = [0.0f64; 6];
        let mut pcnt = 0u64;
        for (band, st) in stats[p].iter().enumerate() {
            if st.coefficients == 0 {
                continue;
            }
            let (hnb, hnb0) = st.hnb_bits();
            let row = [
                st.shipped_bytes,
                st.h0_bits() / 8.0,
                st.hctx_bits() / 8.0,
                hnb / 8.0,
                hnb0 / 8.0,
                st.hbig_bits() / 8.0,
            ];
            eprintln!(
                "  {plane:>5} {:>5} {:>10} {:>11.0} {:>11.0} {:>11.0} {:>11.0} {:>11.0} \
                 {:>11.0} {:>+7.1}% {:>+7.1}%",
                band_name(band, num_levels),
                st.coefficients,
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                (row[0] / row[3].max(1e-9) - 1.0) * 100.0,
                (row[0] / row[5].max(1e-9) - 1.0) * 100.0,
            );
            for i in 0..6 {
                pt[i] += row[i];
            }
            pcnt += st.coefficients;
        }
        if pcnt == 0 {
            continue;
        }
        eprintln!(
            "  {plane:>5} {:>5} {pcnt:>10} {:>11.0} {:>11.0} {:>11.0} {:>11.0} {:>11.0} \
             {:>11.0} {:>+7.1}% {:>+7.1}%",
            "ALL", pt[0], pt[1], pt[2], pt[3], pt[4], pt[5],
            (pt[0] / pt[3].max(1e-9) - 1.0) * 100.0,
            (pt[0] / pt[5].max(1e-9) - 1.0) * 100.0,
        );
        for i in 0..6 {
            t[i] += pt[i];
        }
        t_n += pcnt;
    }
    eprintln!(
        "  {:>5} {:>5} {t_n:>10} {:>11.0} {:>11.0} {:>11.0} {:>11.0} {:>11.0} {:>11.0} \
         {:>+7.1}% {:>+7.1}%",
        "TOTAL", "", t[0], t[1], t[2], t[3], t[4], t[5],
        (t[0] / t[3].max(1e-9) - 1.0) * 100.0,
        (t[0] / t[5].max(1e-9) - 1.0) * 100.0,
    );

    // The decisive comparison. ENT-4: GNC with abac needs +27.1% of JPEG 2000's bits at matched
    // RGB PSNR. Split that gap at the entropy coder: what would this frame have to cost to match
    // J2K, and can *any* neighbourhood-context coder over these coefficients get there?
    //
    // `best` is the smallest of the three bounds, including the one with no model cost charged at
    // all, so the headroom below is the most generous reading of the coder's remaining slack.
    const J2K_GAP: f64 = 0.271;
    let (shipped, best) = (t[0], t[3].min(t[4]).min(t[5]));
    let target = shipped / (1.0 + J2K_GAP);
    let headroom = (shipped - best).max(0.0);
    let serialized: usize = tiles.iter().map(AbacTile::byte_size).sum();
    eprintln!(
        "  serialized abac tiles {serialized} B, of which the rows above account for \
         {shipped:.0} B ({:.2}%); the rest is the seven-byte per-tile header.",
        shipped / serialized as f64 * 100.0
    );
    eprintln!(
        "  J2K 9/7 target at matched RGB PSNR (ENT-4, +27.1%): {target:.0} B. Best bound on \
         these coefficients: {best:.0} B ({:+.1}% of shipped).",
        (best / shipped - 1.0) * 100.0
    );
    eprintln!(
        "  => entropy coding can recover at most {headroom:.0} B of the {:.0} B gap ({:.1}% of \
         it); {:.0} B ({:.1}%) is upstream of the coder.",
        shipped - target,
        headroom / (shipped - target).max(1e-9) * 100.0,
        (shipped - target - headroom).max(0.0),
        (shipped - target - headroom).max(0.0) / (shipped - target).max(1e-9) * 100.0,
    );
}
