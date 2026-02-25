use crate::encoder::{bitplane, rans};

/// Serialize a CompressedFrame to the GP10 binary format.
/// GP10 format supports temporal coding (frame_type + motion vectors).
pub fn serialize_compressed(frame: &crate::CompressedFrame) -> Vec<u8> {
    let mut out = Vec::new();
    // Header
    out.extend_from_slice(b"GP10"); // version 10 = temporal coding
    out.extend_from_slice(&frame.info.width.to_le_bytes());
    out.extend_from_slice(&frame.info.height.to_le_bytes());
    out.extend_from_slice(&frame.info.bit_depth.to_le_bytes());
    out.extend_from_slice(&frame.info.tile_size.to_le_bytes());
    out.extend_from_slice(&frame.config.quantization_step.to_le_bytes());
    out.extend_from_slice(&frame.config.dead_zone.to_le_bytes());
    out.extend_from_slice(&frame.config.wavelet_levels.to_le_bytes());
    // Wavelet type: 0 = LeGall53, 1 = CDF97
    let wavelet_byte: u8 = match frame.config.wavelet_type {
        crate::WaveletType::LeGall53 => 0,
        crate::WaveletType::CDF97 => 1,
    };
    out.push(wavelet_byte);
    // Per-subband entropy: 0 = off, 1 = on
    let per_subband_byte: u8 = if frame.config.per_subband_entropy {
        1
    } else {
        0
    };
    out.push(per_subband_byte);
    // Subband weights: ll, num_detail_levels, per-level [LH, HL, HH], chroma_weight
    let sw = &frame.config.subband_weights;
    out.extend_from_slice(&sw.ll.to_le_bytes());
    let num_detail = sw.detail.len() as u32;
    out.extend_from_slice(&num_detail.to_le_bytes());
    for level in &sw.detail {
        out.extend_from_slice(&level[0].to_le_bytes()); // LH
        out.extend_from_slice(&level[1].to_le_bytes()); // HL
        out.extend_from_slice(&level[2].to_le_bytes()); // HH
    }
    out.extend_from_slice(&sw.chroma_weight.to_le_bytes());
    // CfL alpha side info
    let cfl_enabled: u8 = if frame.cfl_alphas.is_some() { 1 } else { 0 };
    out.push(cfl_enabled);
    if let Some(ref cfl) = frame.cfl_alphas {
        out.extend_from_slice(&cfl.num_subbands.to_le_bytes());
        let tiles_x = frame.info.width.div_ceil(frame.info.tile_size);
        let tiles_y = frame.info.height.div_ceil(frame.info.tile_size);
        let num_cfl_tiles = tiles_x * tiles_y;
        out.extend_from_slice(&num_cfl_tiles.to_le_bytes());
        // CfL alphas stored as i16 LE (2 bytes each)
        for &a in &cfl.alphas {
            out.extend_from_slice(&a.to_le_bytes());
        }
    }
    // Adaptive quantization config + weight map
    let aq_flag: u32 = if frame.config.adaptive_quantization {
        1
    } else {
        0
    };
    out.extend_from_slice(&aq_flag.to_le_bytes());
    out.extend_from_slice(&frame.config.aq_strength.to_le_bytes());
    if let Some(ref wm) = frame.weight_map {
        let wm_len = wm.len() as u32;
        out.extend_from_slice(&wm_len.to_le_bytes());
        for &w in wm {
            out.extend_from_slice(&w.to_le_bytes());
        }
    } else {
        out.extend_from_slice(&0u32.to_le_bytes());
    }
    // Frame type: 0 = Intra, 1 = Predicted (GP10)
    let frame_type_byte: u8 = match frame.frame_type {
        crate::FrameType::Intra => 0,
        crate::FrameType::Predicted => 1,
    };
    out.push(frame_type_byte);
    // Motion field (only for P-frames)
    if let Some(ref mf) = frame.motion_field {
        out.extend_from_slice(&(mf.block_size as u16).to_le_bytes());
        let num_blocks = mf.vectors.len() as u32;
        out.extend_from_slice(&num_blocks.to_le_bytes());
        for mv in &mf.vectors {
            out.extend_from_slice(&mv[0].to_le_bytes());
            out.extend_from_slice(&mv[1].to_le_bytes());
        }
    }
    // Entropy coder type: 0 = rANS, 1 = bitplane, 2 = per-subband rANS
    let entropy_type: u32 = match &frame.entropy {
        crate::EntropyData::Rans(_) => 0,
        crate::EntropyData::SubbandRans(_) => 2,
        crate::EntropyData::Bitplane(_) => 1,
    };
    out.extend_from_slice(&entropy_type.to_le_bytes());
    // Tile data
    match &frame.entropy {
        crate::EntropyData::Rans(tiles) => {
            let num_tiles = tiles.len() as u32;
            out.extend_from_slice(&num_tiles.to_le_bytes());
            for tile in tiles {
                let tile_bytes = rans::serialize_tile_interleaved(tile);
                out.extend_from_slice(&tile_bytes);
            }
        }
        crate::EntropyData::SubbandRans(tiles) => {
            let num_tiles = tiles.len() as u32;
            out.extend_from_slice(&num_tiles.to_le_bytes());
            for tile in tiles {
                let tile_bytes = rans::serialize_tile_subband(tile);
                out.extend_from_slice(&tile_bytes);
            }
        }
        crate::EntropyData::Bitplane(tiles) => {
            let num_tiles = tiles.len() as u32;
            out.extend_from_slice(&num_tiles.to_le_bytes());
            for tile in tiles {
                let tile_bytes = bitplane::serialize_tile_bitplane(tile);
                out.extend_from_slice(&tile_bytes);
            }
        }
    }
    out
}

/// Deserialize a CompressedFrame from the GP10/GPC9/GPC8 binary format.
pub fn deserialize_compressed(data: &[u8]) -> crate::CompressedFrame {
    assert!(data.len() >= 37, "File too small");
    let magic = &data[0..4];
    let is_gpc9 = magic == b"GPC9";
    let is_gp10 = magic == b"GP10";
    assert!(
        magic == b"GPC8" || is_gpc9 || is_gp10,
        "Invalid magic (expected GPC8, GPC9 or GP10; older files must be re-encoded)"
    );

    let width = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let height = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let bit_depth = u32::from_le_bytes(data[12..16].try_into().unwrap());
    let tile_size = u32::from_le_bytes(data[16..20].try_into().unwrap());
    let qstep = f32::from_le_bytes(data[20..24].try_into().unwrap());
    let dead_zone = f32::from_le_bytes(data[24..28].try_into().unwrap());
    let wavelet_levels = u32::from_le_bytes(data[28..32].try_into().unwrap());

    // Wavelet type
    let wavelet_type = match data[32] {
        0 => crate::WaveletType::LeGall53,
        1 => crate::WaveletType::CDF97,
        w => panic!("Unknown wavelet type: {}", w),
    };

    // Per-subband entropy flag (GPC9 and GP10)
    let (per_subband_entropy, subband_weights_start) = if is_gpc9 || is_gp10 {
        (data[33] != 0, 34)
    } else {
        (false, 33)
    };

    // Subband weights
    let ll = f32::from_le_bytes(
        data[subband_weights_start..subband_weights_start + 4]
            .try_into()
            .unwrap(),
    );
    let num_detail = u32::from_le_bytes(
        data[subband_weights_start + 4..subband_weights_start + 8]
            .try_into()
            .unwrap(),
    ) as usize;
    let mut pos = subband_weights_start + 8;
    let mut detail = Vec::with_capacity(num_detail);
    for _ in 0..num_detail {
        let lh = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        let hl = f32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
        let hh = f32::from_le_bytes(data[pos + 8..pos + 12].try_into().unwrap());
        detail.push([lh, hl, hh]);
        pos += 12;
    }
    let chroma_weight = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    // CfL alpha side info
    let cfl_flag = data[pos];
    pos += 1;
    let (cfl_enabled, cfl_alphas) = if cfl_flag != 0 {
        let nsb = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let num_cfl_tiles = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        // 2 chroma planes * num_tiles * num_subbands i16 values
        let alpha_count = (2 * num_cfl_tiles * nsb) as usize;
        let mut alphas = Vec::with_capacity(alpha_count);
        for _ in 0..alpha_count {
            let v = i16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
            alphas.push(v);
            pos += 2;
        }
        (
            true,
            Some(crate::CflAlphas {
                alphas,
                num_subbands: nsb,
            }),
        )
    } else {
        (false, None)
    };

    // Adaptive quantization config + weight map
    let aq_flag = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let aq_strength = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let adaptive_quantization = aq_flag != 0;

    let wm_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let weight_map = if wm_len > 0 {
        let mut wm = Vec::with_capacity(wm_len);
        for _ in 0..wm_len {
            wm.push(f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        Some(wm)
    } else {
        None
    };

    // Frame type + motion field (GP10 only)
    let (frame_type, motion_field) = if is_gp10 {
        let ft = match data[pos] {
            0 => crate::FrameType::Intra,
            1 => crate::FrameType::Predicted,
            f => panic!("Unknown frame type: {}", f),
        };
        pos += 1;
        let mf = if ft == crate::FrameType::Predicted {
            let block_size = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as u32;
            pos += 2;
            let num_blocks = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let mut vectors = Vec::with_capacity(num_blocks);
            for _ in 0..num_blocks {
                let dx = i16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
                let dy = i16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap());
                vectors.push([dx, dy]);
                pos += 4;
            }
            Some(crate::MotionField {
                vectors,
                block_size,
            })
        } else {
            None
        };
        (ft, mf)
    } else {
        (crate::FrameType::Intra, None)
    };

    // Entropy coder type: 0 = rANS, 1 = bitplane, 2 = per-subband rANS
    let entropy_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;

    let num_tiles = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    let (entropy_coder, entropy, per_subband) = match entropy_type {
        0 => {
            let mut tiles = Vec::with_capacity(num_tiles);
            for _ in 0..num_tiles {
                let (tile, consumed) = rans::deserialize_tile_interleaved(&data[pos..]);
                tiles.push(tile);
                pos += consumed;
            }
            (
                crate::EntropyCoder::Rans,
                crate::EntropyData::Rans(tiles),
                false,
            )
        }
        1 => {
            let mut tiles = Vec::with_capacity(num_tiles);
            for _ in 0..num_tiles {
                let (tile, consumed) = bitplane::deserialize_tile_bitplane(&data[pos..]);
                tiles.push(tile);
                pos += consumed;
            }
            (
                crate::EntropyCoder::Bitplane,
                crate::EntropyData::Bitplane(tiles),
                false,
            )
        }
        2 => {
            let mut tiles = Vec::with_capacity(num_tiles);
            for _ in 0..num_tiles {
                let (tile, consumed) = rans::deserialize_tile_subband(&data[pos..]);
                tiles.push(tile);
                pos += consumed;
            }
            (
                crate::EntropyCoder::Rans,
                crate::EntropyData::SubbandRans(tiles),
                true,
            )
        }
        _ => panic!("Unknown entropy coder type: {}", entropy_type),
    };

    crate::CompressedFrame {
        info: crate::FrameInfo {
            width,
            height,
            bit_depth,
            tile_size,
        },
        config: crate::CodecConfig {
            tile_size,
            quantization_step: qstep,
            dead_zone,
            wavelet_levels,
            subband_weights: crate::SubbandWeights {
                ll,
                detail,
                chroma_weight,
            },
            cfl_enabled,
            entropy_coder,
            wavelet_type,
            adaptive_quantization,
            aq_strength,
            per_subband_entropy: per_subband_entropy || per_subband,
            ..Default::default()
        },
        entropy,
        cfl_alphas,
        weight_map,
        frame_type,
        motion_field,
    }
}
