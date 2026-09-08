//! ENT-7 step 3: what would BPC-PaCo's coder cost on the coefficients GNC already ships?
//!
//! BPC-PaCo — bitplane coding with parallel coefficient processing; Aulí-Llinàs, Enfedaque,
//! Moure, Sanchez, *IEEE TIP* 25(1), 2016, and the GPU implementation in *IEEE TPDS* 28(8), 2017
//! — differs from abac in three places, and it matters a great deal which one is being priced.
//!
//! **1. The scan, and it is not what a first reading suggests.** BPC-PaCo splits a code-block into
//! vertical stripes **two columns wide**, one thread per stripe, and steps them in lockstep: every
//! thread codes the left column of row *y*, then every thread codes the right column of row *y*.
//! The significance context therefore *does* include neighbours coded in the **current** bitplane
//! — a left-column coefficient has 3 already-visited neighbours, a right-column one has 5, and
//! the average of 4 is **exactly what JPEG 2000's sequential raster scan gets** (TIP 2016 §III-A).
//! No information is given up for the parallelism; what makes it safe is the schedule, not a
//! weaker context. The paper's own ablation confirms it: swap the coder for the MQ coder and
//! BPC-PaCo lands "almost the same as JPEG 2000".
//!
//! That scan is reproduced here exactly, and cheaply, because a sequential visit in the order
//! "row *y* even columns ascending, then row *y* odd columns ascending" produces the same
//! visited-neighbour sets the lockstep does (TIP 2016 Fig. 1(b) makes the same point in order to
//! get a bit-identical serial reference).
//!
//! **2. Stationary probabilities.** One estimate per (context, **bitplane**, subband), trained
//! offline, in a table both ends hold and nobody transmits. 14 contexts: 9 from the plain sum of
//! the 8 neighbours' significance flags, 4 for sign coding, 1 for magnitude refinement. That
//! deletes abac's cold start — ENT-6 measured the short blocks at +25.9% over the bound at q=90
//! because abac opens every context at p = 1/2 — and gives up adaptation everywhere else.
//!
//! **3. Many fixed-length codewords per block.** Each stripe's arithmetic coder emits W-bit
//! codewords into slots it reserves in the block's bitstream, so a 64-wide block carries 32
//! independent coders. TIP 2016 §IV attributes essentially all of BPC-PaCo's rate penalty to
//! this: forcing a single codeword stream *improves* on JPEG 2000 for every corpus but natural
//! images. It is charged here as its own column, because it is the part that scales with GNC's
//! block geometry rather than with its content.
//!
//! ## The columns
//!
//! | column | what it prices |
//! |---|---|
//! | `Hbpc` | BPC-PaCo's scan and contexts, probabilities pooled per subband *of this image* |
//! | `Hbpcn` | the same, but with the context frozen at the plane boundary — a WGSL port with **no** cross-lane exchange, which is the shape a naive port takes |
//! | `Hbpcf` | `Hbpc`'s model against a table trained on **other images** — the honest stationary number |
//! | `flw` | the excess bits in each stripe's final codeword, `streams × W/2` per block |
//!
//! `Hbpc` pools its probabilities over a whole plane's worth of a subband of the image being
//! coded, which is the same generous convention `Hctx` and `Hnb` already use in
//! [`super::coef_entropy_diag`] — an oracle table, and deliberately so. `Hbpcf` is what a shipped
//! table reaches, and it needs statistics from elsewhere:
//!
//! ```text
//! GNC_COEF_ENTROPY=1 GNC_BPC_DUMP=<file>  gnc benchmark -i <png> -q 90 --abac   # collect
//! GNC_COEF_ENTROPY=1 GNC_BPC_TABLE=<file> gnc benchmark -i <png> -q 90 --abac   # price
//! ```
//!
//! `scripts/meas_ent7_bpc.py` runs that leave-one-image-out over the four stills of decision
//! `0024`. A table trained on the image being priced is not a stationary model, it is an oracle,
//! and that is what `Hbpc` is for.
//!
//! **Two deviations from the papers, both stated so the number can be read honestly.** TPDS 2017
//! uses separate significance tables for the significance-propagation and cleanup passes; here
//! the two passes share one table, which costs almost nothing because `φ_sig = 0` already
//! identifies the cleanup case. And the sign is coded as "does it disagree with the neighbourhood
//! prediction", which is how the pooled estimate stays meaningful — pooling a raw sign bit over a
//! context that predicts the sign would read 1 bit and hide the model.
//!
//! Read-only on data the encoder has already produced — it cannot move the bitstream
//! (`docs/decisions/0010`).

use std::collections::HashMap;
use std::fmt::Write as _;

use super::coef_entropy_diag::BinCount;

/// Significance contexts: the plain sum of the eight neighbours' significance flags, 0..=8
/// (TIP 2016 eq. (1) — position-agnostic, unlike JPEG 2000's Table D.1).
const NUM_SIG: usize = 9;
/// Sign contexts (TIP 2016 eq. (2)): vertical and horizontal neighbour agreement, four classes.
const NUM_SIGN: usize = 4;
/// Magnitude refinement gets exactly one context — the paper's own choice, on the grounds that
/// more of them buy nothing.
const NUM_REF: usize = 1;
/// Contexts per bitplane, per subband.
const CTX_PER_PLANE: usize = NUM_SIG + NUM_SIGN + NUM_REF;
/// Bitplanes the tables cover. GNC's coefficients are `i32` but quantisation keeps them far below
/// this; anything above is folded into the top plane rather than silently dropped.
const MAX_PLANES: usize = 24;
/// One flat array per (plane, subband) cell: a stationary table is per bitplane *and* per subband.
const NUM_BPC_CONTEXTS: usize = MAX_PLANES * CTX_PER_PLANE;

const SIG: usize = 0;
const SIGN: usize = NUM_SIG;
const REF: usize = NUM_SIG + NUM_SIGN;

/// Codeword width of BPC-PaCo's fixed-length arithmetic coder (TIP 2016 §III-D: W = 16, P̂ = 7).
const FLW_W: f64 = 16.0;
/// Columns per stripe — one arithmetic coder each (TIP 2016 §III-A).
const STRIPE_COLS: usize = 2;

fn ctx(plane: u32, slot: usize) -> usize {
    (plane as usize).min(MAX_PLANES - 1) * CTX_PER_PLANE + slot
}

/// Counts for one (plane, subband) cell.
#[derive(Clone)]
pub(crate) struct BpcStats {
    /// BPC-PaCo's own two-column lockstep scan.
    pub(crate) lockstep: Vec<BinCount>,
    /// The same contexts with the neighbourhood frozen at the plane boundary: no cross-lane
    /// exchange at all, which is what a WGSL port without shuffle or barriers would have.
    pub(crate) frozen: Vec<BinCount>,
    /// Excess bits in the final codeword of each stripe's coder, summed over blocks.
    pub(crate) flw_bits: f64,
    pub(crate) blocks: u64,
}

impl Default for BpcStats {
    fn default() -> Self {
        Self {
            lockstep: vec![BinCount::default(); NUM_BPC_CONTEXTS],
            frozen: vec![BinCount::default(); NUM_BPC_CONTEXTS],
            flw_bits: 0.0,
            blocks: 0,
        }
    }
}

impl BpcStats {
    fn pooled_bits(counts: &[BinCount]) -> f64 {
        counts.iter().map(BinCount::bits).sum()
    }

    /// BPC-PaCo's model with an oracle table trained on this image.
    pub(crate) fn lockstep_bits(&self) -> f64 {
        Self::pooled_bits(&self.lockstep)
    }

    /// The no-communication variant, same tables, same passes.
    pub(crate) fn frozen_bits(&self) -> f64 {
        Self::pooled_bits(&self.frozen)
    }

    /// How many binary decisions the lockstep variant coded — the miss-rate denominator.
    pub(crate) fn decisions(&self) -> u64 {
        self.lockstep.iter().map(|c| c.n).sum()
    }

    /// Cross-entropy of this cell's decisions against a table trained elsewhere, and how many
    /// decisions fell in a context the table never saw. A miss is charged at p = 1/2 and
    /// *reported*, never quietly replaced by a self-trained probability: a stationary table
    /// meeting statistics it was not trained on is the failure mode, not an inconvenience.
    pub(crate) fn fixed_bits(&self, table: &BpcTable, key: (usize, usize)) -> (f64, u64) {
        let mut bits = 0.0;
        let mut misses = 0u64;
        for (c, bc) in self.lockstep.iter().enumerate() {
            if bc.n == 0 {
                continue;
            }
            match table.p.get(&(key.0, key.1, c)) {
                Some(&p) => {
                    let p = p.clamp(1.0 / 65536.0, 1.0 - 1.0 / 65536.0);
                    bits += bc.ones as f64 * -p.log2()
                        + (bc.n - bc.ones) as f64 * -(1.0 - p).log2();
                }
                None => {
                    bits += bc.n as f64;
                    misses += bc.n;
                }
            }
        }
        (bits, misses)
    }
}

/// A stationary probability table: P(bit == 1) per (plane, subband, context).
#[derive(Default)]
pub(crate) struct BpcTable {
    p: HashMap<(usize, usize, usize), f64>,
    pub(crate) source: String,
}

impl BpcTable {
    /// Parse the format [`dump`] writes: `plane band ctx n ones`. Several dumps concatenated
    /// train on the union, which is how the leave-one-image-out tables are built.
    pub(crate) fn load(path: &str) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut acc: HashMap<(usize, usize, usize), (u64, u64)> = HashMap::new();
        for line in text.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() != 5 {
                continue;
            }
            let num = |s: &str| s.parse::<u64>().unwrap_or(0);
            let key = (num(f[0]) as usize, num(f[1]) as usize, num(f[2]) as usize);
            let e = acc.entry(key).or_insert((0, 0));
            e.0 += num(f[3]);
            e.1 += num(f[4]);
        }
        let p = acc
            .into_iter()
            .filter(|(_, (n, _))| *n > 0)
            .map(|(k, (n, ones))| (k, ones as f64 / n as f64))
            .collect();
        Ok(Self { p, source: path.to_string() })
    }
}

/// Serialize the lockstep counts so another run can train on them.
pub(crate) fn dump(stats: &[Vec<BpcStats>]) -> String {
    let mut out =
        String::from("# plane band ctx n ones — ENT-7 BPC-PaCo lockstep-scan context counts\n");
    for (p, bands) in stats.iter().enumerate() {
        for (band, st) in bands.iter().enumerate() {
            for (c, bc) in st.lockstep.iter().enumerate() {
                if bc.n == 0 {
                    continue;
                }
                let _ = writeln!(out, "{p} {band} {c} {} {}", bc.n, bc.ones);
            }
        }
    }
    out
}

/// Walk one code-block as BPC-PaCo would, twice — once with its lockstep scan, once with the
/// neighbourhood frozen at each plane boundary.
///
/// The neighbourhood stops at the block edge; outside counts as insignificant, as in JPEG 2000
/// and as `abac` already does, so blocks stay independently decodable.
pub(crate) fn accumulate_block(st: &mut BpcStats, coefficients: &[i32], width: usize) {
    let height = coefficients.len() / width;
    if height == 0 || width == 0 {
        return;
    }
    let mag: Vec<u32> = coefficients.iter().map(|v| v.unsigned_abs()).collect();
    let neg: Vec<bool> = coefficients.iter().map(|v| *v < 0).collect();
    let max = mag.iter().copied().max().unwrap_or(0);
    st.blocks += 1;
    if max == 0 {
        return;
    }
    let nplanes = 32 - max.leading_zeros();

    walk(&mut st.lockstep, &mag, &neg, width, height, nplanes, false);
    walk(&mut st.frozen, &mag, &neg, width, height, nplanes, true);

    // One arithmetic coder per stripe, each wasting on average half a codeword at the end of the
    // block (TIP 2016 §III-C: the coder reserves W bits and fills them when its interval is
    // exhausted, so only the last reservation of each stripe is partly unused).
    let streams = width.div_ceil(STRIPE_COLS);
    st.flw_bits += streams as f64 * FLW_W / 2.0;
}

/// BPC-PaCo's scan order within one bitplane: for each row, every stripe's left column, then
/// every stripe's right column (TIP 2016 §III-A and Fig. 1(b)).
///
/// A sequential visit in this order sees exactly the neighbours the parallel lockstep schedule
/// does — the parallel and serial coders are bit-identical, which is the property the paper
/// relies on for its reference implementation and the reason this diagnostic can model a
/// massively parallel coder with a single-threaded walk.
fn visit_order(w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut order = Vec::with_capacity(w * h);
    for y in 0..h {
        for phase in 0..STRIPE_COLS {
            let mut x = phase;
            while x < w {
                order.push((y, x));
                x += STRIPE_COLS;
            }
        }
    }
    order
}

/// One bitplane walk. `frozen` selects what the context may read: the significance state as the
/// scan updates it (BPC-PaCo's lockstep, 3 or 5 visited neighbours, average 4) or the state as it
/// stood at the end of the previous plane (no cross-lane exchange at all).
///
/// Everything else is identical between the two, so the difference between them is the price of
/// giving up the exchange and nothing else.
fn walk(
    counts: &mut [BinCount],
    mag: &[u32],
    neg: &[bool],
    w: usize,
    h: usize,
    nplanes: u32,
    frozen_ctx: bool,
) {
    let order = &visit_order(w, h);
    // 0 = insignificant, +1 / −1 = significant with that sign. Carries both the significance
    // flags the zero-coding context needs and the signs the sign context needs.
    let mut state = vec![0i8; mag.len()];
    let mut refined = vec![false; mag.len()];
    let mut snapshot = vec![0i8; mag.len()];

    for plane in (0..nplanes).rev() {
        if frozen_ctx {
            snapshot.copy_from_slice(&state);
        }
        for &(y, x) in order {
            let i = y * w + x;
            let bit = (mag[i] >> plane) & 1 != 0;

            // Read the context before the update, so the borrow ends first.
            let src: &[i8] = if frozen_ctx { &snapshot } else { &state };
            let was_significant = src[i] != 0;
            let phi = significant_neighbours(src, w, h, y, x);
            let (sign_class, predicted_negative) = sign_context(src, w, h, y, x);

            if was_significant {
                counts[ctx(plane, REF)].push(bit);
                refined[i] = true;
            } else {
                counts[ctx(plane, SIG + phi)].push(bit);
                if bit {
                    counts[ctx(plane, SIGN + sign_class)].push(neg[i] != predicted_negative);
                    state[i] = if neg[i] { -1 } else { 1 };
                }
            }
        }
    }
    // `refined` exists to mirror the three-pass structure; BPC-PaCo uses one refinement context,
    // so nothing reads it. Kept as the canary that the refinement pass ran at all.
    debug_assert!(nplanes < 2 || refined.iter().any(|&r| r) || mag.iter().all(|&m| m == 0));
}

/// How many of the eight neighbours are significant, 0..=8 (TIP 2016 eq. (1)).
fn significant_neighbours(state: &[i8], w: usize, h: usize, y: usize, x: usize) -> usize {
    let at = |yy: isize, xx: isize| -> usize {
        if yy < 0 || xx < 0 || yy as usize >= h || xx as usize >= w {
            0
        } else {
            usize::from(state[yy as usize * w + xx as usize] != 0)
        }
    };
    let (yi, xi) = (y as isize, x as isize);
    at(yi, xi - 1)
        + at(yi, xi + 1)
        + at(yi - 1, xi)
        + at(yi + 1, xi)
        + at(yi - 1, xi - 1)
        + at(yi - 1, xi + 1)
        + at(yi + 1, xi - 1)
        + at(yi + 1, xi + 1)
}

/// BPC-PaCo's sign context (TIP 2016 eq. (2)) and the sign it predicts.
///
/// Returns `(class, predicted_negative)`. The class is the four-way split on whether the vertical
/// and horizontal neighbourhoods agree; the prediction is what the coded bit is measured against,
/// which is what stops a pooled estimate from washing the correlation out.
fn sign_context(state: &[i8], w: usize, h: usize, y: usize, x: usize) -> (usize, bool) {
    let at = |yy: isize, xx: isize| -> i32 {
        if yy < 0 || xx < 0 || yy as usize >= h || xx as usize >= w {
            0
        } else {
            i32::from(state[yy as usize * w + xx as usize])
        }
    };
    let (yi, xi) = (y as isize, x as isize);
    let v = (at(yi - 1, xi) + at(yi + 1, xi)).clamp(-1, 1);
    let hz = (at(yi, xi - 1) + at(yi, xi + 1)).clamp(-1, 1);
    let class = match (v, hz) {
        (0, 0) => 3,
        (0, _) => 1,
        (_, 0) => 2,
        (a, b) if a == b => 0,
        _ => 3,
    };
    let predicted = if v != 0 { v } else { hz };
    (class, predicted < 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The instrument's canary. BPC-PaCo's claim — the one that decides whether its parallelism
    /// is free — is that its scan sees as many already-coded neighbours as a sequential raster
    /// scan does: 3 for a left-column coefficient, 5 for a right-column one, **average 4, the
    /// same as JPEG 2000** (TIP 2016 §III-A). If this test fails, every rate figure this module
    /// produces is measuring some other coder.
    #[test]
    fn lockstep_scan_sees_three_and_five_neighbours() {
        let (w, h) = (16usize, 16usize);
        let order = visit_order(w, h);
        assert_eq!(order.len(), w * h, "every coefficient is visited exactly once");

        let mut visited = vec![false; w * h];
        let mut per_column = [Vec::new(), Vec::new()];
        for &(y, x) in &order {
            // Interior only: the block edge legitimately has fewer neighbours.
            if y > 0 && y + 1 < h && x > 0 && x + 1 < w {
                let mut n = 0;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dy == 0 && dx == 0 {
                            continue;
                        }
                        let yy = (y as i32 + dy) as usize;
                        let xx = (x as i32 + dx) as usize;
                        n += usize::from(visited[yy * w + xx]);
                    }
                }
                per_column[x % STRIPE_COLS].push(n);
            }
            visited[y * w + x] = true;
        }

        assert!(per_column[0].iter().all(|&n| n == 3), "left column: {:?}", per_column[0]);
        assert!(per_column[1].iter().all(|&n| n == 5), "right column: {:?}", per_column[1]);
        let all: Vec<usize> = per_column.concat();
        let avnp = all.iter().sum::<usize>() as f64 / all.len() as f64;
        assert!((avnp - 4.0).abs() < 1e-9, "AVNP is {avnp}, JPEG 2000's raster scan gets 4");
    }

    /// The no-exchange variant must be strictly less informed, so it cannot cost less. This is
    /// the direction check on the `Hbpcn` column: a port that drops the cross-lane exchange
    /// cannot come out ahead of one that keeps it.
    #[test]
    fn dropping_the_exchange_never_helps() {
        // A block with structure, so the neighbourhood carries information at all.
        let (w, h) = (16usize, 16usize);
        let coefficients: Vec<i32> = (0..w * h)
            .map(|i| {
                let (y, x) = (i / w, i % w);
                let v = ((x as i32 - 8).abs() + (y as i32 - 8).abs()) * 3 - 12;
                if (x + y) % 5 == 0 {
                    -v
                } else {
                    v
                }
            })
            .collect();
        let mut st = BpcStats::default();
        accumulate_block(&mut st, &coefficients, w);
        assert!(st.decisions() > 0, "canary: the walk coded nothing");
        assert!(
            st.frozen_bits() >= st.lockstep_bits(),
            "frozen {} < lockstep {} — the less informed context cannot be cheaper",
            st.frozen_bits(),
            st.lockstep_bits()
        );
    }

    /// One arithmetic coder per two columns, each wasting half a codeword at the end.
    #[test]
    fn codeword_excess_scales_with_stripe_count() {
        let mut st = BpcStats::default();
        accumulate_block(&mut st, &vec![7i32; 64 * 64], 64);
        assert_eq!(st.flw_bits, 32.0 * FLW_W / 2.0);
        let mut narrow = BpcStats::default();
        accumulate_block(&mut narrow, &vec![7i32; 8 * 8], 8);
        assert_eq!(narrow.flw_bits, 4.0 * FLW_W / 2.0);
    }
}
