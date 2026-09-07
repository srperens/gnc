use super::abac::Coder;
use super::abac_gpu_encode::GpuAbacEncoder;
use super::abac_tile::{self, AbacTile};
use super::bitplane;
use super::huffman;
use super::rans;
use super::rans_gpu_encode::GpuRansEncoder;
use super::rice;
use crate::gpu_util::read_buffer_f32;
use crate::{CodecConfig, EntropyCoder, EntropyData, FrameInfo, GpuContext};

/// Which entropy path to use (resolved from CodecConfig).
pub(super) enum EntropyMode {
    Rans,
    SubbandRans,
    /// Context-adaptive per-subband rANS (2 tables per detail group)
    SubbandRansCtx,
    Bitplane,
    Rice,
    Huffman,
    /// Adaptive binary arithmetic coding over code-blocks. GPU encode (ENT-5) and GPU decode,
    /// one thread per block on both sides; the CPU coder in `abac.rs` is the reference both are
    /// verified byte-exact against, and the fallback for callers with no device.
    Abac,
}

impl EntropyMode {
    pub(super) fn from_config(config: &CodecConfig) -> Self {
        if config.entropy_coder == EntropyCoder::Abac {
            EntropyMode::Abac
        } else if config.entropy_coder == EntropyCoder::Huffman {
            EntropyMode::Huffman
        } else if config.entropy_coder == EntropyCoder::Rice {
            EntropyMode::Rice
        } else if config.entropy_coder == EntropyCoder::Bitplane {
            EntropyMode::Bitplane
        } else if config.per_subband_entropy && config.context_adaptive {
            EntropyMode::SubbandRansCtx
        } else if config.per_subband_entropy {
            EntropyMode::SubbandRans
        } else {
            EntropyMode::Rans
        }
    }
}

/// Whether the inter (P/B) pipeline has a GPU *encoder* for this configuration's entropy stage.
///
/// This answers "where does the entropy stage run", and nothing else. It must never be used to
/// pick a frame encoder, a motion estimator or a local decode — ARCH-3 is the record of what
/// happens when it is. Until 2026-09-07, choosing a coder with no GPU entropy encoder swapped the
/// whole P/B frame encoder for a second implementation that encoded every P-frame wrong (BUG-18).
///
/// **More precisely, this asks whether the *batched three-plane* GPU dispatch supports the
/// configuration**, which is narrower than "a GPU encoder exists". Rice and rANS have encode
/// shaders wired into that batch (`rice_encode.wgsl`, `rans_encode.wgsl`). Huffman has a shader
/// but only `pipeline.rs` (intra) dispatches it, so on the inter path it codes on the CPU — which
/// is also why P-frames used to come out of the batched pipeline with an empty Huffman tile
/// vector. Context-adaptive rANS has no GPU encoder at all; the same exclusion is in
/// `pipeline.rs`, and without it the flag would silently drop the context modelling the caller
/// asked for.
///
/// **abac is the case where the two readings come apart, and it is `false` here on purpose.**
/// Since ENT-5 abac *does* have a GPU encoder (`abac_encode.wgsl`), but it is a per-plane
/// dispatch, not part of this batch — so returning `true` would route abac into a batched path
/// with no abac arm. It reaches the GPU from inside [`encode_entropy`], which tests
/// `config.gpu_entropy_encode` directly and branches on abac before it ever looks at the
/// `use_gpu_encode` this function feeds. So a `false` here means "not in the batch", not "on the
/// CPU": abac's entropy stage runs on the GPU either way. Deliberately kept separate — this flag
/// also switches the fused quantize+histogram shader, and BUG-16 records that shader moving
/// Rice's *pixels* at q=25. `docs/decisions/0024`.
pub(super) fn inter_gpu_entropy_available(config: &CodecConfig) -> bool {
    matches!(
        config.entropy_coder,
        EntropyCoder::Rice | EntropyCoder::Rans
    ) && !config.context_adaptive
}

/// Helper: entropy-encode a quantized plane buffer (GPU or CPU path).
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_entropy(
    gpu_encoder: &mut GpuRansEncoder,
    abac_encoder: &mut GpuAbacEncoder,
    ctx: &GpuContext,
    quantized_buf: &wgpu::Buffer,
    padded_pixels: usize,
    padded_w: usize,
    tiles_x: usize,
    tiles_y: usize,
    tile_size: usize,
    entropy_mode: &EntropyMode,
    config: &CodecConfig,
    use_gpu_encode: bool,
    info: &FrameInfo,
    entropy_levels: u32,
    rans_tiles: &mut Vec<rans::InterleavedRansTile>,
    subband_tiles: &mut Vec<rans::SubbandRansTile>,
    bp_tiles: &mut Vec<bitplane::BitplaneTile>,
    rice_tiles: &mut Vec<rice::RiceTile>,
    huffman_tiles: &mut Vec<huffman::HuffmanTile>,
    abac_tiles: &mut Vec<AbacTile>,
) {
    // abac's own GPU encoder (ENT-5). Deliberately *not* folded into `use_gpu_encode`: that
    // flag switches on the rANS histogram machinery and, in `sequence.rs`, which whole-frame
    // P-frame pipeline runs — the conflation ARCH-3 exists to separate. abac needs neither, so
    // it is routed on `gpu_entropy_encode` directly and nothing else moves.
    if matches!(entropy_mode, EntropyMode::Abac) && config.gpu_entropy_encode {
        assert_eq!(
            padded_pixels,
            padded_w * tiles_y * tile_size,
            "abac GPU encode: plane is {padded_pixels} coefficients but its geometry says \
             {padded_w} x {} — the encoder reads straight out of the quantised buffer and a \
             mismatch would read past the plane",
            tiles_y * tile_size,
        );
        let mut t = abac_encoder.encode_plane_to_tiles(
            ctx,
            quantized_buf,
            padded_w,
            tiles_x,
            tiles_y,
            config.tile_size,
            entropy_levels,
            config.abac_code_block,
            config.abac_coder,
            config.abac_gpu_sizing,
        );
        abac_tiles.append(&mut t);
    } else if use_gpu_encode
        && !matches!(
            entropy_mode,
            EntropyMode::Rice | EntropyMode::Huffman | EntropyMode::Abac
        )
    {
        let (mut rt, mut st) = gpu_encoder.encode_plane_to_tiles(
            ctx,
            quantized_buf,
            info,
            config.per_subband_entropy,
            entropy_levels,
        );
        rans_tiles.append(&mut rt);
        subband_tiles.append(&mut st);
    } else {
        let quantized = read_buffer_f32(ctx, quantized_buf, padded_pixels);
        entropy_encode_tiles(
            &quantized,
            padded_w,
            tiles_x,
            tiles_y,
            tile_size,
            entropy_mode,
            config.tile_size,
            entropy_levels,
            rans_tiles,
            subband_tiles,
            bp_tiles,
            rice_tiles,
            huffman_tiles,
            abac_tiles,
            config.abac_coder,
            config.abac_code_block,
        );
    }
}

/// CPU entropy decode for a single plane: reconstruct quantized f32 coefficients from tiles.
///
/// `tile_offset` is the index of the first tile for this plane in the flat tile vector.
/// For 4:4:4 this equals `plane_idx * tiles_per_plane`; for non-444 use the explicit offset.
pub(crate) fn entropy_decode_plane(
    entropy: &EntropyData,
    tile_offset: usize,
    tiles_per_plane: usize,
    tile_size: usize,
    padded_w: usize,
) -> Vec<f32> {
    let tile_start = tile_offset;
    let tiles_x = padded_w / tile_size;
    let padded_h_tiles = tiles_per_plane / tiles_x;
    let padded_h = padded_h_tiles * tile_size;
    let total_pixels = padded_w * padded_h;
    let mut plane = vec![0.0f32; total_pixels];

    for t in 0..tiles_per_plane {
        let tx = t % tiles_x;
        let ty = t / tiles_x;

        let coeffs: Vec<i32> = match entropy {
            EntropyData::Rans(tiles) => rans::rans_decode_tile_interleaved(&tiles[tile_start + t]),
            EntropyData::SubbandRans(tiles) => {
                let tile = &tiles[tile_start + t];
                // Detect context-adaptive mode: plain subbands have num_levels*2 groups
                let expected_plain = tile.num_levels * 2;
                if tile.num_groups > expected_plain {
                    rans::rans_decode_tile_interleaved_subband_ctx(tile)
                } else {
                    rans::rans_decode_tile_interleaved_subband(tile)
                }
            }
            EntropyData::Bitplane(tiles) => bitplane::bitplane_decode_tile(&tiles[tile_start + t]),
            EntropyData::Rice(tiles) => rice::rice_decode_tile(&tiles[tile_start + t]),
            EntropyData::Huffman(tiles) => huffman::huffman_decode_tile(&tiles[tile_start + t]),
            EntropyData::Abac(tiles) => abac_tile::abac_decode_tile(&tiles[tile_start + t]),
        };

        // Scatter tile coefficients back into flat plane
        for row in 0..tile_size {
            for col in 0..tile_size {
                let py = ty * tile_size + row;
                let px = tx * tile_size + col;
                plane[py * padded_w + px] = coeffs[row * tile_size + col] as f32;
            }
        }
    }

    plane
}

/// Entropy-encode all tiles from a quantized plane (CPU path).
#[allow(clippy::too_many_arguments)]
pub(super) fn entropy_encode_tiles(
    quantized: &[f32],
    plane_width: usize,
    tiles_x: usize,
    tiles_y: usize,
    tile_size: usize,
    mode: &EntropyMode,
    tile_size_u32: u32,
    num_levels: u32,
    rans_tiles: &mut Vec<rans::InterleavedRansTile>,
    subband_tiles: &mut Vec<rans::SubbandRansTile>,
    bp_tiles: &mut Vec<bitplane::BitplaneTile>,
    rice_tiles: &mut Vec<rice::RiceTile>,
    huffman_tiles: &mut Vec<huffman::HuffmanTile>,
    abac_tiles: &mut Vec<AbacTile>,
    // One engine for the whole encode, from the config: the two share a binarisation but not a
    // bitstream, and it is recorded per tile so the decoder never has to guess.
    abac_coder: Coder,
    abac_cb: u32,
) {
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let coeffs = extract_tile_coefficients(quantized, plane_width, tx, ty, tile_size);
            match mode {
                EntropyMode::Bitplane => {
                    bp_tiles.push(bitplane::bitplane_encode_tile(&coeffs, tile_size_u32));
                }
                EntropyMode::SubbandRans => {
                    subband_tiles.push(rans::rans_encode_tile_interleaved_subband(
                        &coeffs,
                        tile_size_u32,
                        num_levels,
                    ));
                }
                EntropyMode::SubbandRansCtx => {
                    subband_tiles.push(rans::rans_encode_tile_interleaved_subband_ctx(
                        &coeffs,
                        tile_size_u32,
                        num_levels,
                    ));
                }
                EntropyMode::Rans => {
                    rans_tiles.push(rans::rans_encode_tile_interleaved_zrl(&coeffs));
                }
                EntropyMode::Rice => {
                    rice_tiles.push(rice::rice_encode_tile(
                        &coeffs,
                        tile_size_u32,
                        num_levels,
                    ));
                }
                EntropyMode::Huffman => {
                    huffman_tiles.push(huffman::huffman_encode_tile(
                        &coeffs,
                        tile_size_u32,
                        num_levels,
                    ));
                }
                EntropyMode::Abac => {
                    abac_tiles.push(abac_tile::abac_encode_tile(
                        &coeffs,
                        tile_size_u32,
                        num_levels,
                        abac_cb,
                        abac_coder,
                    ));
                }
            }
        }
    }

    // Canary. A silently-inactive entropy coder shows up as "no rate change", which reads as a
    // null result rather than a bug; this prints what actually ran on real coefficients.
    if matches!(mode, EntropyMode::Abac) && super::diagnostics::enabled() {
        let coded = &abac_tiles[abac_tiles.len() - tiles_x * tiles_y..];
        let blocks: usize = coded.iter().map(|t| t.block_lengths.len()).sum();
        let bytes: usize = coded.iter().map(|t| t.block_data.len()).sum();
        let empty = coded
            .iter()
            .flat_map(|t| t.block_lengths.iter())
            .filter(|&&l| l == 0)
            .count();
        eprintln!(
            "  [abac] plane {tiles_x}x{tiles_y} tiles: abac_blocks={blocks} \
             (empty={empty}) bytes={bytes} coder={:?} cb={}",
            abac_coder, abac_cb,
        );
    }
}

/// Extract a tile's worth of coefficients from a flat plane array, converting f32 to i32.
fn extract_tile_coefficients(
    plane: &[f32],
    plane_width: usize,
    tile_x: usize,
    tile_y: usize,
    tile_size: usize,
) -> Vec<i32> {
    let mut coeffs = Vec::with_capacity(tile_size * tile_size);
    let origin_x = tile_x * tile_size;
    let origin_y = tile_y * tile_size;

    for y in 0..tile_size {
        for x in 0..tile_size {
            let idx = (origin_y + y) * plane_width + (origin_x + x);
            coeffs.push(plane[idx].round() as i32);
        }
    }
    coeffs
}

/// Which arithmetic engine `EntropyCoder::Abac` encodes with.
///
/// **Range by default**, unlike [`Coder::from_env`], which the standalone diagnostics use and
/// which defaults to Interval because that is what their −19% to −25% figures were measured with.
/// For a shipped encode the choice is settled: on real coefficients at q=90 Range costs 33.0 ms
/// of entropy decode against Interval's 96.3 ms — 2.9x — for 0.7 points of rate (−13.8% vs
/// −14.5% on bbb). Range at cb=64 dominates every other cell measured.
pub fn abac_coder_from_env() -> Coder {
    match std::env::var("GNC_ABAC_CODER").as_deref() {
        Ok("interval") => Coder::Interval,
        _ => Coder::Range,
    }
}

/// Code-block edge, `GNC_ABAC_CB`. Clamped to what `abac_decode.wgsl` can address: it keeps two
/// rows of neighbour magnitudes per thread in workgroup memory, sized for 64. A larger value
/// would code better and then fail to decode on the GPU, so it is refused here rather than at
/// dispatch.
pub fn abac_cb_from_env() -> u32 {
    let cb = std::env::var("GNC_ABAC_CB")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(abac_tile::DEFAULT_CB);
    assert!(
        cb.is_power_of_two() && (4..=crate::encoder::abac_gpu::MAX_BLOCK_W).contains(&cb),
        "GNC_ABAC_CB must be a power of two in 4..={}, got {cb}",
        crate::encoder::abac_gpu::MAX_BLOCK_W
    );
    cb
}
