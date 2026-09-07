//! Tile container for the adaptive binary code-block coder (`EntropyCoder::Abac`).
//!
//! The coder itself is in [`super::abac`]; this module is what puts it in the bitstream. A tile is
//! cut into code-blocks, each block is coded independently, and the blob carries a small header
//! plus one length per block.
//!
//! **Why code-blocks and not GNC's usual 256 streams per tile.** An adaptive binary coder learns
//! its probabilities as it goes, so it needs symbols to learn on. Splitting a tile 256 ways gives
//! each coder ~256 symbols, and the measured gain collapses from −6.6% to −0.7% (warm start) or
//! +2.4% (cold) — worse than Rice. A 64×64 block gives one coder 4096 symbols and its raster scan
//! makes the full neighbourhood available. Parallelism survives: a padded 1080p 4:4:4 frame holds
//! roughly 3000 independent blocks, which is ample GPU work even though it is not 256 per tile.
//! See BACKLOG "EBCOT — evaluating in halves", parts 2-6.
//!
//! **Geometry is derived, never transmitted.** Both sides compute the same block list from
//! `(tile_size, num_levels, cb_size)`, so the blob carries lengths only. [`code_blocks`] is the
//! single definition and the encoder, the CPU decoder and the GPU decoder all call it — a second
//! copy of this loop is exactly how a coverage bug would get in, and a coverage bug here shrinks
//! the file while every individual block still round-trips.

use super::abac::Coder;

/// Code-block edge in pixels. 64 is the measured optimum once decode throughput is priced in:
/// against Rice on real coefficients at q=90 it is −13.8% rate at 33.0 ms of entropy decode,
/// where cb=32 is −10.9% at 31.4 ms — cb=64 dominates. Larger blocks code better still (cb=128 is
/// −20.0% at q=55) but `abac_decode.wgsl` keeps two rows of neighbour magnitudes per thread in
/// workgroup memory and is sized for 64.
pub const DEFAULT_CB: u32 = 64;

/// One tile's worth of code-block streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbacTile {
    pub tile_size: u32,
    pub num_levels: u32,
    pub cb_size: u32,
    /// Which arithmetic engine coded this tile. Recorded per tile rather than taken from the
    /// environment at decode time: the two engines share a binarisation but not a bitstream, and
    /// a decoder that guesses wrong does not fail — it produces a plausible wrong image.
    pub coder: Coder,
    /// Bytes in each block's stream, in [`code_blocks`] order.
    pub block_lengths: Vec<u32>,
    /// The blocks' streams concatenated in the same order.
    pub block_data: Vec<u8>,
}

impl AbacTile {
    pub fn byte_size(&self) -> usize {
        serialize_tile_abac(self).len()
    }
}

/// Enumerate a tile's subbands as `(x0, y0, w, h)` in tile-local coordinates.
///
/// Mallat layout: LL in the top-left at `tile_size >> num_levels`, then HL/LH/HH at each level
/// outward. `num_levels = 0` yields the whole tile as one region, which is what the MED lossless
/// path needs.
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

/// Every code-block of a tile as `(x0, y0, w, h)` in tile-local coordinates.
///
/// Blocks never straddle a subband boundary — coefficients either side of one have different
/// statistics and different orientations, and the context model is built on the assumption that a
/// block is homogeneous. Deep subbands smaller than `cb` become one short block each; at tile 256
/// with 5 levels that is the 8×8 LL plus three 8×8 level-5 bands, 0.4% of the tile.
pub fn code_blocks(tile_size: usize, num_levels: u32, cb: usize) -> Vec<(usize, usize, usize, usize)> {
    code_blocks_banded(tile_size, num_levels, cb)
        .into_iter()
        .map(|(x, y, w, h, _)| (x, y, w, h))
        .collect()
}

/// The same enumeration, with each block tagged by the index of the subband it was cut from.
///
/// The index is the position in the Mallat order [`subbands`] returns — 0 = LL, then HL/LH/HH per
/// level from the coarsest outward. Diagnostics need the tag to attribute bytes to a band; the
/// codec itself does not, which is why [`code_blocks`] stays the narrower signature.
pub fn code_blocks_banded(
    tile_size: usize,
    num_levels: u32,
    cb: usize,
) -> Vec<(usize, usize, usize, usize, usize)> {
    let mut out = Vec::new();
    for (band, (sx, sy, sw, sh)) in subbands(tile_size, num_levels).into_iter().enumerate() {
        let mut by = 0;
        while by < sh {
            let bh = cb.min(sh - by);
            let mut bx = 0;
            while bx < sw {
                let bw = cb.min(sw - bx);
                out.push((sx + bx, sy + by, bw, bh, band));
                bx += cb;
            }
            by += cb;
        }
    }
    out
}

/// Human-readable name of subband `band` at `num_levels`, in [`subbands`] order.
pub fn band_name(band: usize, num_levels: u32) -> String {
    if num_levels == 0 {
        return "ALL".to_string();
    }
    if band == 0 {
        return "LL".to_string();
    }
    // `subbands` pushes the finest level first (offset tile/2, size tile/2), so band 1..3 are
    // level 1 — the highest-frequency bands — and the index grows toward the coarsest.
    let level = (band - 1) / 3 + 1;
    let orient = ["HL", "LH", "HH"][(band - 1) % 3];
    format!("{orient}{level}")
}

/// Encode one tile's coefficients (row-major, `tile_size × tile_size`).
pub fn abac_encode_tile(
    coefficients: &[i32],
    tile_size: u32,
    num_levels: u32,
    cb_size: u32,
    coder: Coder,
) -> AbacTile {
    let ts = tile_size as usize;
    assert_eq!(
        coefficients.len(),
        ts * ts,
        "abac tile expects {ts}x{ts} coefficients, got {}",
        coefficients.len()
    );
    let blocks = code_blocks(ts, num_levels, cb_size as usize);
    let mut block_lengths = Vec::with_capacity(blocks.len());
    let mut block_data = Vec::new();
    // Coverage canary. Blocks that miss coefficients make the file smaller while every block
    // still round-trips on its own, so no roundtrip test would catch it.
    let mut covered = 0usize;
    for (bx, by, bw, bh) in blocks {
        let mut blk = Vec::with_capacity(bw * bh);
        for y in 0..bh {
            let row = (by + y) * ts + bx;
            blk.extend_from_slice(&coefficients[row..row + bw]);
        }
        let bytes = coder.encode_block(&blk, bw);
        block_lengths.push(bytes.len() as u32);
        block_data.extend_from_slice(&bytes);
        covered += blk.len();
    }
    assert_eq!(
        covered,
        coefficients.len(),
        "code-block cutting covered {covered} of {} coefficients",
        coefficients.len()
    );
    AbacTile {
        tile_size,
        num_levels,
        cb_size,
        coder,
        block_lengths,
        block_data,
    }
}

/// CPU decode of one tile back to row-major coefficients.
///
/// The GPU shader is the shipping path; this is the reference the shader is verified against, and
/// the fallback for callers with no device.
pub fn abac_decode_tile(tile: &AbacTile) -> Vec<i32> {
    let ts = tile.tile_size as usize;
    let mut out = vec![0i32; ts * ts];
    let blocks = code_blocks(ts, tile.num_levels, tile.cb_size as usize);
    assert_eq!(
        blocks.len(),
        tile.block_lengths.len(),
        "tile header describes {} blocks but its geometry yields {}",
        tile.block_lengths.len(),
        blocks.len()
    );
    let mut off = 0usize;
    for (i, (bx, by, bw, bh)) in blocks.into_iter().enumerate() {
        let len = tile.block_lengths[i] as usize;
        let blk = tile
            .coder
            .decode_block(&tile.block_data[off..off + len], bw * bh, bw);
        off += len;
        for y in 0..bh {
            let row = (by + y) * ts + bx;
            out[row..row + bw].copy_from_slice(&blk[y * bw..(y + 1) * bw]);
        }
    }
    out
}

// ---- Serialisation ----
//
// Tile blob layout:
//   u16  tile_size
//   u8   num_levels
//   u8   cb_log2
//   u8   coder            0 = interval, 1 = range
//   u16  num_blocks
//   ...  num_blocks LEB128 block lengths
//   ...  concatenated block streams
//
// Lengths are varints rather than a fixed width because block size varies by three orders of
// magnitude across a tile: the 8×8 deep subbands cost a handful of bytes where a busy 64×64 HL
// block costs thousands. A flat u32 would spend 4 bytes on every one of ~25 blocks per tile per
// plane, which at 1080p 4:4:4 is 120 KB a frame of pure header.

fn put_uvarint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_uvarint(data: &[u8], pos: &mut usize) -> u32 {
    let mut v = 0u32;
    let mut shift = 0;
    loop {
        let b = data[*pos];
        *pos += 1;
        v |= ((b & 0x7f) as u32) << shift;
        if b & 0x80 == 0 {
            return v;
        }
        shift += 7;
    }
}

pub fn serialize_tile_abac(tile: &AbacTile) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + tile.block_lengths.len() * 2 + tile.block_data.len());
    out.extend_from_slice(&(tile.tile_size as u16).to_le_bytes());
    out.push(tile.num_levels as u8);
    out.push(tile.cb_size.trailing_zeros() as u8);
    out.push(tile.coder as u8);
    out.extend_from_slice(&(tile.block_lengths.len() as u16).to_le_bytes());
    for &l in &tile.block_lengths {
        put_uvarint(&mut out, l);
    }
    out.extend_from_slice(&tile.block_data);
    out
}

/// Returns the tile and how many bytes it consumed.
pub fn deserialize_tile_abac(data: &[u8]) -> (AbacTile, usize) {
    let mut pos = 0usize;
    let tile_size = u16::from_le_bytes(data[0..2].try_into().unwrap()) as u32;
    pos += 2;
    let num_levels = data[pos] as u32;
    pos += 1;
    let cb_size = 1u32 << data[pos];
    pos += 1;
    let coder = match data[pos] {
        0 => Coder::Interval,
        1 => Coder::Range,
        other => panic!("unknown abac coder id {other} in tile header"),
    };
    pos += 1;
    let num_blocks = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
    pos += 2;
    let mut block_lengths = Vec::with_capacity(num_blocks);
    let mut total = 0usize;
    for _ in 0..num_blocks {
        let l = get_uvarint(data, &mut pos);
        total += l as usize;
        block_lengths.push(l);
    }
    let block_data = data[pos..pos + total].to_vec();
    pos += total;
    (
        AbacTile {
            tile_size,
            num_levels,
            cb_size,
            coder,
            block_lengths,
            block_data,
        },
        pos,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(ts: usize, levels: u32) -> Vec<i32> {
        // Wavelet-like: a dense LL corner and increasingly sparse detail bands.
        let mut v = vec![0i32; ts * ts];
        let mut state = 0x1234_5678u32;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for (sx, sy, sw, sh) in subbands(ts, levels) {
            let dense = sx == 0 && sy == 0;
            for y in 0..sh {
                for x in 0..sw {
                    let r = rnd();
                    let val = if dense {
                        (r % 400) as i32 - 200
                    } else if r % 8 == 0 {
                        (r % 20) as i32 - 10
                    } else {
                        0
                    };
                    v[(sy + y) * ts + sx + x] = val;
                }
            }
        }
        v
    }

    #[test]
    fn code_blocks_cover_every_coefficient_exactly_once() {
        for &ts in &[64usize, 128, 256] {
            for levels in 0..=5u32 {
                if ts >> levels == 0 {
                    continue;
                }
                for &cb in &[32usize, 64] {
                    let mut hits = vec![0u8; ts * ts];
                    for (bx, by, bw, bh) in code_blocks(ts, levels, cb) {
                        assert!(bw <= cb && bh <= cb);
                        for y in 0..bh {
                            for x in 0..bw {
                                hits[(by + y) * ts + bx + x] += 1;
                            }
                        }
                    }
                    assert!(
                        hits.iter().all(|&h| h == 1),
                        "ts={ts} levels={levels} cb={cb}: coverage is not exactly 1 everywhere"
                    );
                }
            }
        }
    }

    #[test]
    fn tile_roundtrips_through_both_coders() {
        for coder in [Coder::Interval, Coder::Range] {
            for &(ts, levels) in &[(64u32, 3u32), (256, 5)] {
                let coeffs = synth(ts as usize, levels);
                let tile = abac_encode_tile(&coeffs, ts, levels, DEFAULT_CB, coder);
                assert_eq!(abac_decode_tile(&tile), coeffs, "{coder:?} ts={ts}");
            }
        }
    }

    #[test]
    fn tile_survives_serialisation() {
        let coeffs = synth(256, 5);
        for coder in [Coder::Interval, Coder::Range] {
            let tile = abac_encode_tile(&coeffs, 256, 5, DEFAULT_CB, coder);
            let blob = serialize_tile_abac(&tile);
            assert_eq!(blob.len(), tile.byte_size());
            let (back, consumed) = deserialize_tile_abac(&blob);
            assert_eq!(consumed, blob.len(), "deserialiser must consume the whole blob");
            assert_eq!(back, tile);
            assert_eq!(abac_decode_tile(&back), coeffs);
        }
    }

    #[test]
    fn all_zero_tile_is_cheap() {
        let coeffs = vec![0i32; 256 * 256];
        let tile = abac_encode_tile(&coeffs, 256, 5, DEFAULT_CB, Coder::Range);
        assert_eq!(abac_decode_tile(&tile), coeffs);
        // 25 blocks, each a handful of bytes. Anything near the coefficient count means the
        // significance contexts are not learning.
        assert!(
            tile.byte_size() < 512,
            "an all-zero 256x256 tile cost {} bytes",
            tile.byte_size()
        );
    }
}
