//! Compare the adaptive-binary code-block coder against Rice on real quantised coefficients.
//!
//! Gated behind `GNC_ABAC_COMPARE=1`, zero cost when unset. Reads the quantised planes back from
//! the GPU — the same readback `sig_context_diag` uses — and codes every tile twice: once with
//! the shipping Rice coder, once with `abac` over 64×64 code-blocks cut from each subband.
//!
//! This is the number that decides whether the EBCOT work is worth doing. The offline ceiling
//! measurement (`scripts/meas_ebcot_context.py`) put it at −13.7% mean on four images, but that
//! was conditional entropy on a Python DWT. This is real coefficients, a real arithmetic coder,
//! and Rice's real serialised size including its headers.
//!
//! It reports rate only. Decode throughput is the other half of the decision and this says
//! nothing about it.
//!
//! The baseline is the **shipped** Rice size, taken from the frame's own entropy data. An earlier
//! version compared against `rice::rice_encode_tile`, the CPU reference implementation, and got
//! −35%: the CPU reference produced more bytes for the coefficients alone than the real encoder
//! produced for the whole file, because the GPU path has per-stream k and the checkerboard
//! k-context that the reference lacks. A result better than its own offline ceiling (−13.7%) is
//! how that showed up.

use super::abac::Coder;

/// Enumerate a tile's subbands as `(x0, y0, w, h)` in tile-local coordinates.
///
/// Mallat layout, matching `subband_region` in `sig_context_diag`: LL in the top-left corner at
/// `tile_size >> num_levels`, then HL/LH/HH at each level outward.
fn subbands(tile_size: usize, num_levels: u32) -> Vec<(usize, usize, usize, usize)> {
    let ll = tile_size >> num_levels;
    let mut out = vec![(0, 0, ll, ll)];
    let mut region = tile_size;
    for _ in 0..num_levels {
        let half = region / 2;
        out.push((half, 0, half, half)); // HL
        out.push((0, half, half, half)); // LH
        out.push((half, half, half, half)); // HH
        region = half;
    }
    out
}

/// Code every 64×64 code-block of every subband of one tile, returning total bytes.
/// Wall-clock time spent in abac encode and decode across the whole comparison, in seconds.
/// Rate is only half the decision; an adaptive binary coder is serial per symbol and this is the
/// first signal on the other half.
#[derive(Default)]
pub struct Timing {
    pub encode_s: f64,
    pub decode_s: f64,
    pub coefficients: usize,
}

#[allow(clippy::too_many_arguments)] // tile geometry plus the variant under test
/// The frame's real code-blocks, collected so the GPU can decode exactly what the CPU just did.
///
/// `tests/abac_bench.rs` times the same grid on *synthesised* planes, and that answers a slightly
/// different question: significance density drives how many binary decisions each coefficient
/// costs, so a synthetic plane is not the same workload. Measured, the synthetic proxy runs 6%
/// fast at cb=64. More importantly, only a real frame can be compared against the codec's own
/// whole-frame decode figure in the same process — which is the comparison that decides whether
/// any of this is shippable.
#[derive(Default)]
struct GpuWorkload {
    packed: Vec<u8>,
    infos: Vec<crate::encoder::abac_gpu::BlockInfo>,
    /// The CPU-decoded coefficients, laid out block-contiguously to match `infos`.
    expected: Vec<i32>,
}

impl GpuWorkload {
    /// Blocks are laid out contiguously rather than at their plane positions. That is sound
    /// because the shader takes neighbour magnitudes from workgroup memory, never from the output
    /// buffer, so the output layout cannot change what it decodes — only where it lands.
    fn push(&mut self, bytes: &[u8], block: &[i32], bw: usize, bh: usize) {
        self.infos.push(crate::encoder::abac_gpu::BlockInfo::new(
            self.packed.len() as u32,
            bytes.len() as u32,
            self.expected.len() as u32,
            bw as u32,
            bh as u32,
            bw as u32,
        ));
        self.packed.extend_from_slice(bytes);
        self.expected.extend_from_slice(block);
    }
}

#[allow(clippy::too_many_arguments)] // a diagnostic's plane geometry; splitting it would obscure
fn abac_tile_bytes(
    tile: &[i32],
    tile_size: usize,
    num_levels: u32,
    cb: usize,
    coder: Coder,
    timing: &mut Timing,
    work: &mut GpuWorkload,
) -> usize {
    let mut total = 0usize;
    let mut blocks = 0usize;
    // Coverage canary. A subband layout that misses coefficients would make abac look smaller
    // while every individual block still round-trips, which is the one way this comparison could
    // be wrong without any test failing.
    let mut covered = 0usize;
    for (sx, sy, sw, sh) in subbands(tile_size, num_levels) {
        let mut by = 0;
        while by < sh {
            let bh = cb.min(sh - by);
            let mut bx = 0;
            while bx < sw {
                let bw = cb.min(sw - bx);
                let mut blk = Vec::with_capacity(bw * bh);
                for y in 0..bh {
                    let row = (sy + by + y) * tile_size + sx + bx;
                    blk.extend_from_slice(&tile[row..row + bw]);
                }
                let t0 = std::time::Instant::now();
                let bytes = coder.encode_block(&blk, bw);
                timing.encode_s += t0.elapsed().as_secs_f64();
                let t1 = std::time::Instant::now();
                // Canary: a size comparison against a coder that does not reproduce its input
                // is meaningless. Verify every block, not a sample.
                let back = coder.decode_block(&bytes, blk.len(), bw);
                timing.decode_s += t1.elapsed().as_secs_f64();
                timing.coefficients += blk.len();
                assert_eq!(
                    back, blk,
                    "abac roundtrip failed on a real {bw}x{bh} code-block — the size comparison \
                     below would be meaningless"
                );
                work.push(&bytes, &blk, bw, bh);
                total += bytes.len();
                covered += blk.len();
                blocks += 1;
                bx += cb;
            }
            by += cb;
        }
    }
    assert_eq!(
        covered,
        tile.len(),
        "subband cutting covered {covered} of {} coefficients — the size comparison would be \
         meaningless and no roundtrip test would catch it",
        tile.len()
    );
    // Each block needs its length in the bitstream. Two bytes covers a 64×64 block of
    // quantised coefficients at any realistic rate; charge it so the comparison is honest.
    total + blocks * 2
}

#[allow(clippy::too_many_arguments)] // a diagnostic's plane geometry; splitting it would obscure
fn abac_plane_bytes(
    label: &str,
    plane: &[f32],
    width: u32,
    height: u32,
    tile_size: u32,
    num_levels: u32,
    cb: usize,
    coder: Coder,
    timing: &mut Timing,
    work: &mut GpuWorkload,
) -> usize {
    let ts = tile_size as usize;
    let tiles_x = (width as usize) / ts;
    let tiles_y = (height as usize) / ts;
    let mut abac_bytes = 0usize;

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let mut tile = Vec::with_capacity(ts * ts);
            for y in 0..ts {
                let row = (ty * ts + y) * (width as usize) + tx * ts;
                tile.extend(plane[row..row + ts].iter().map(|&v| v.round() as i32));
            }
            abac_bytes += abac_tile_bytes(&tile, ts, num_levels, cb, coder, timing, work);
        }
    }
    eprintln!("  [abac] {label:3} {tiles_x}x{tiles_y} tiles: abac {abac_bytes:>9} B");
    abac_bytes
}

/// Read the quantised planes back and report Rice vs abac bytes per plane and in total.
#[allow(clippy::too_many_arguments)]
pub fn run_multi_plane(
    ctx: &crate::GpuContext,
    y_buf: &wgpu::Buffer,
    co_buf: &wgpu::Buffer,
    cg_buf: &wgpu::Buffer,
    padded_w: u32,
    padded_h: u32,
    chroma_w: u32,
    chroma_h: u32,
    tile_size: u32,
    num_levels: u32,
    rice_reference_bytes: usize,
) {
    // 128 means one code-block per subband at 5 levels in a 256px tile, which measured best.
    // Swept on bbb at q=55: 16px +1.1% (worse than Rice), 32px −15.1%, 64px −19.2%,
    // 128px −20.0%, 256px −20.0% (identical, since no subband exceeds 128). Bigger is better
    // because a context-adaptive coder needs symbols to learn on — the same mechanism that made
    // the 256-stream variant fail, one scale down. 16px blocks hold 256 coefficients, which is
    // not enough to learn 18 context probabilities and lands worse than Rice.
    let cb: usize = std::env::var("GNC_ABAC_CB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let coder = Coder::from_env();
    eprintln!(
        "[abac] comparison active (GNC_ABAC_COMPARE), coder {coder:?} (GNC_ABAC_CODER), \
         code-block {cb}px (GNC_ABAC_CB), {num_levels} levels"
    );

    let y = crate::gpu_util::read_buffer_f32(ctx, y_buf, (padded_w * padded_h) as usize);
    let co = crate::gpu_util::read_buffer_f32(ctx, co_buf, (chroma_w * chroma_h) as usize);
    let cg = crate::gpu_util::read_buffer_f32(ctx, cg_buf, (chroma_w * chroma_h) as usize);

    let mut a_tot = 0usize;
    let mut timing = Timing::default();
    let mut work = GpuWorkload::default();
    for (label, plane, w, h) in [
        ("Y", &y, padded_w, padded_h),
        ("Co", &co, chroma_w, chroma_h),
        ("Cg", &cg, chroma_w, chroma_h),
    ] {
        a_tot += abac_plane_bytes(
            label, plane, w, h, tile_size, num_levels, cb, coder, &mut timing, &mut work,
        );
    }
    if rice_reference_bytes > 0 {
        eprintln!(
            "  [abac] TOTAL: shipped rice {rice_reference_bytes} B   abac {a_tot} B   {:+.1}%  \
             (rate only — says nothing about decode throughput)",
            (a_tot as f64 / rice_reference_bytes as f64 - 1.0) * 100.0
        );
    } else {
        eprintln!("  [abac] TOTAL: abac {a_tot} B (no Rice reference — not a Rice frame)");
    }
    let mc = timing.coefficients as f64 / 1e6;
    eprintln!(
        "  [abac] single-thread CPU: encode {:.0} ms, decode {:.0} ms for {:.2} Mcoeff \
         ({:.1} Mcoeff/s decode). Reference coder, not optimised, no SIMD, no GPU.",
        timing.encode_s * 1e3,
        timing.decode_s * 1e3,
        mc,
        mc / timing.decode_s.max(1e-9)
    );

    gpu_decode_report(ctx, &work, coder, cb);
}

/// Decode the frame's real code-blocks on the GPU, verify them against the CPU coder, then time
/// the dispatch.
///
/// Scope is the entropy stage only, so it is a *lower bound* on any abac frame decode: the
/// inverse wavelet and the colour transform come on top, and both are shared with Rice. That is
/// what makes it comparable against the codec's own `Decode:` figure printed by `benchmark` —
/// since the shared stages cancel,
///
/// ```text
/// abac_frame − rice_frame = abac_entropy − rice_entropy ≥ abac_entropy − rice_frame
/// ```
///
/// so this number minus the whole-frame Rice figure is a conservative floor on what abac costs.
fn gpu_decode_report(ctx: &crate::GpuContext, work: &GpuWorkload, coder: Coder, cb: usize) {
    use crate::encoder::abac_gpu::{GpuAbacDecoder, MAX_BLOCK_W};
    if work.infos.is_empty() {
        return;
    }
    if work.infos.iter().any(|i| i.width > MAX_BLOCK_W) {
        eprintln!(
            "  [abac] GPU decode skipped: code-block {cb}px exceeds the shader's {MAX_BLOCK_W}px \
             cap. The rate figures above are unaffected."
        );
        return;
    }
    let decoder = GpuAbacDecoder::new(ctx);
    let out_len = work.expected.len();

    // Repeat rather than trust one reading: five sessions share this Mac and the same dispatch
    // has read 48% apart across runs. The first call also carries pipeline setup.
    const REPEATS: usize = 7;
    let mut times = Vec::with_capacity(REPEATS);
    for i in 0..=REPEATS {
        let (got, gpu_s) = decoder.decode(ctx, &work.packed, &work.infos, out_len, coder);
        if i == 0 {
            // A timing number for a decode that is wrong is worth nothing, so check before timing.
            let bad = got
                .iter()
                .zip(work.expected.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                bad, 0,
                "GPU abac decode diverged from the CPU {coder:?} coder on {bad} of {out_len} \
                 real coefficients"
            );
        } else {
            times.push(gpu_s);
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let mc = out_len as f64 / 1e6;
    eprintln!(
        "  [abac] GPU decode, {coder:?} coder, the SAME real blocks: {} blocks, {:.2} Mcoeff, \
         median {:.1} ms ({:.1} Mcoeff/s) over {} dispatches [{:.1}-{:.1} ms], all {} \
         coefficients bit-exact vs the CPU coder. ENTROPY STAGE ONLY — a lower bound; compare \
         against the whole-frame `Decode:` figure below, which already includes Rice's entropy \
         stage plus the inverse wavelet and colour transform.",
        work.infos.len(),
        mc,
        median * 1e3,
        mc / median,
        times.len(),
        times[0] * 1e3,
        times[times.len() - 1] * 1e3,
        out_len
    );
}
