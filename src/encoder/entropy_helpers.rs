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
}

impl EntropyMode {
    pub(super) fn from_config(config: &CodecConfig) -> Self {
        if config.entropy_coder == EntropyCoder::Huffman {
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

/// Helper: entropy-encode a quantized plane buffer (GPU or CPU path).
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_entropy(
    gpu_encoder: &mut GpuRansEncoder,
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
) {
    if use_gpu_encode && !matches!(entropy_mode, EntropyMode::Rice | EntropyMode::Huffman) {
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
        );
    }
}

/// CPU entropy decode for a single plane: reconstruct quantized f32 coefficients from tiles.
pub(crate) fn entropy_decode_plane(
    entropy: &EntropyData,
    plane_idx: usize,
    tiles_per_plane: usize,
    tile_size: usize,
    padded_w: usize,
) -> Vec<f32> {
    let tile_start = plane_idx * tiles_per_plane;
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
            }
        }
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
