//! ENT-14 — an all-zero abac code-block is coded as no bytes at all.
//!
//! Before this, every code-block paid a terminated arithmetic interval whether or not it held a
//! single significant coefficient — ~11.5 bytes measured — and on sparse content that floor *was*
//! the file. A 512x512 gradient came out **2.3x larger than Rice** at q=25, all of it 840 empty
//! blocks, and `regression_gradient_q50` refused abac as the default because of it.
//!
//! The two properties below are what make the fix a pure **encoder-side** change, and they are
//! tested rather than assumed because the whole argument for not bumping a bitstream generation
//! rests on the first one.

use gnc::encoder::abac::Coder;
use gnc::encoder::abac_tile::{abac_decode_tile, abac_encode_tile, code_blocks, DEFAULT_CB};

/// **A zero-length stream already decodes to zeros, on both engines.**
///
/// This is why ENT-14 needed no new format and no generation bump: a decoder that predates the
/// change reads a length-0 block correctly, because it reads past the end as zero bytes and the
/// initial contexts then resolve every significance decision to "not significant". Emitting
/// nothing for an empty block is therefore a choice the *encoder* makes inside a format that
/// already expressed it.
///
/// If this ever stops holding, ENT-14 becomes a format change and the generation must move.
#[test]
fn empty_stream_decodes_to_zeros() {
    type Decode = fn(&[u8], usize, usize) -> Vec<i32>;
    let engines: [(&str, Decode); 2] = [
        ("range", gnc::encoder::abac::decode_block_rc),
        ("interval", gnc::encoder::abac::decode_block),
    ];
    for (name, decode) in engines {
        for &(w, h) in &[(8usize, 8usize), (32, 32), (64, 64)] {
            let out = decode(&[], w * h, w);
            assert!(
                out.iter().all(|&v| v == 0),
                "{name} {w}x{h}: a zero-length stream decoded to {} non-zero coefficients — \
                 ENT-14's empty-block encoding is only format-compatible while this holds",
                out.iter().filter(|&&v| v != 0).count()
            );
        }
    }
}

/// An all-zero tile costs one varint per block, and round-trips.
///
/// The size assertion is the point: 100 blocks at cb=32 used to cost 763 bytes and now cost the
/// length words alone. A regression here reads as "the empty path stopped firing", which is
/// exactly the silent failure ENT-14 was filed against — the bytes would still be correct.
#[test]
fn all_zero_tile_costs_only_its_lengths() {
    let ts = 256usize;
    let coeffs = vec![0i32; ts * ts];
    let blocks = code_blocks(ts, 5, DEFAULT_CB as usize).len();

    for coder in [Coder::Range, Coder::Interval] {
        let tile = abac_encode_tile(&coeffs, ts as u32, 5, DEFAULT_CB, coder);

        assert!(
            tile.block_lengths.iter().all(|&l| l == 0),
            "{coder:?}: an all-zero tile must code every one of its {blocks} blocks to zero bytes, \
             got lengths {:?}",
            &tile.block_lengths[..tile.block_lengths.len().min(8)]
        );
        assert!(
            tile.block_data.is_empty(),
            "{coder:?}: an all-zero tile must carry no block data, got {} bytes",
            tile.block_data.len()
        );
        // Header is 7 bytes plus one varint per block, and a zero varint is one byte.
        assert!(
            tile.byte_size() <= 8 + blocks,
            "{coder:?}: an all-zero tile cost {} bytes over {blocks} blocks",
            tile.byte_size()
        );
        assert_eq!(
            abac_decode_tile(&tile),
            coeffs,
            "{coder:?}: an all-zero tile must decode back to zeros"
        );
    }
}

/// A tile with one significant coefficient codes exactly one block, and decodes exactly.
///
/// The complement of the test above: the empty path must not swallow a block that has something
/// in it. A bug that coded *everything* as empty would pass `all_zero_tile_costs_only_its_lengths`
/// and shrink every file.
#[test]
fn one_live_coefficient_codes_exactly_one_block() {
    let ts = 256usize;
    let mut coeffs = vec![0i32; ts * ts];
    // Middle of the tile, so it lands in a real block rather than the LL corner.
    coeffs[(ts / 2) * ts + ts / 2] = -7;

    for coder in [Coder::Range, Coder::Interval] {
        let tile = abac_encode_tile(&coeffs, ts as u32, 5, DEFAULT_CB, coder);
        let live = tile.block_lengths.iter().filter(|&&l| l > 0).count();
        assert_eq!(
            live, 1,
            "{coder:?}: one significant coefficient should leave exactly one non-empty block, \
             got {live}"
        );
        assert_eq!(
            abac_decode_tile(&tile),
            coeffs,
            "{coder:?}: the coefficient must survive the round trip"
        );
    }
}
