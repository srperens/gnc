use crate::encoder::{bitplane, rans};
use crate::FrameType;

// ---- CRC-32 (ISO 3309 / ITU-T V.42, same as zlib/gzip/PNG) ----

/// CRC-32 lookup table (polynomial 0xEDB88320, reflected).
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
};

/// Compute CRC-32 checksum over a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[index];
    }
    crc ^ 0xFFFF_FFFF
}

// ---- GNV1 Sequence Container Format ----
//
// File layout:
//   File Header (28 bytes, fixed):
//     magic: "GNV1" (4 bytes)
//     version: u32 LE           (currently 1)
//     width: u32 LE
//     height: u32 LE
//     frame_count: u32 LE
//     framerate_num: u32 LE     (e.g. 30000)
//     framerate_den: u32 LE     (e.g. 1001)
//
//   Frame Index Table (frame_count entries, 21 bytes each):
//     offset: u64 LE            (byte offset from start of file to frame data)
//     size: u32 LE              (frame data size in bytes)
//     frame_type: u8            (0 = Intra, 1 = Predicted)
//     pts: u64 LE               (presentation timestamp, frame number)
//
//   Frame Data:
//     [frame 0 GP11 data] [frame 1 GP11 data] ... [frame N-1 GP11 data]

const GNV1_MAGIC: &[u8; 4] = b"GNV1";
const GNV1_VERSION: u32 = 1;
const GNV1_HEADER_SIZE: usize = 28;
const GNV1_INDEX_ENTRY_SIZE: usize = 21; // 8 + 4 + 1 + 8

/// Entry in the GNV1 frame index table.
#[derive(Debug, Clone, Copy)]
pub struct FrameIndexEntry {
    /// Byte offset from start of file to frame data
    pub offset: u64,
    /// Size of this frame's serialized data in bytes
    pub size: u32,
    /// Frame type: 0 = Intra, 1 = Predicted
    pub frame_type: u8,
    /// Presentation timestamp (frame number)
    pub pts: u64,
}

/// Parsed GNV1 sequence header with frame index for random access.
#[derive(Debug, Clone)]
pub struct SequenceHeader {
    pub version: u32,
    pub width: u32,
    pub height: u32,
    pub frame_count: u32,
    pub framerate_num: u32,
    pub framerate_den: u32,
    /// Frame index table for random access
    pub index: Vec<FrameIndexEntry>,
}

impl SequenceHeader {
    /// Duration in seconds (frame_count / framerate)
    pub fn duration_secs(&self) -> f64 {
        if self.framerate_num == 0 {
            return 0.0;
        }
        self.frame_count as f64 * self.framerate_den as f64 / self.framerate_num as f64
    }

    /// Frames per second
    pub fn fps(&self) -> f64 {
        if self.framerate_den == 0 {
            return 0.0;
        }
        self.framerate_num as f64 / self.framerate_den as f64
    }
}

/// Serialize a sequence of CompressedFrames into the GNV1 container format.
///
/// Each frame is serialized using the GP11 format and wrapped in
/// a container with a file header and frame index table for random access.
pub fn serialize_sequence(frames: &[crate::CompressedFrame], framerate: (u32, u32)) -> Vec<u8> {
    assert!(!frames.is_empty(), "Cannot serialize empty sequence");

    let frame_count = frames.len() as u32;
    let first = &frames[0];

    // Serialize each frame to GP10 bytes
    let frame_blobs: Vec<Vec<u8>> = frames.iter().map(serialize_compressed).collect();

    // Compute offsets: header + index table + frame data
    let index_table_size = frames.len() * GNV1_INDEX_ENTRY_SIZE;
    let data_start = GNV1_HEADER_SIZE + index_table_size;

    let mut offsets = Vec::with_capacity(frames.len());
    let mut current_offset = data_start as u64;
    for blob in &frame_blobs {
        offsets.push(current_offset);
        current_offset += blob.len() as u64;
    }
    let total_size = current_offset as usize;

    let mut out = Vec::with_capacity(total_size);

    // File header (28 bytes)
    out.extend_from_slice(GNV1_MAGIC);
    out.extend_from_slice(&GNV1_VERSION.to_le_bytes());
    out.extend_from_slice(&first.info.width.to_le_bytes());
    out.extend_from_slice(&first.info.height.to_le_bytes());
    out.extend_from_slice(&frame_count.to_le_bytes());
    out.extend_from_slice(&framerate.0.to_le_bytes());
    out.extend_from_slice(&framerate.1.to_le_bytes());

    // Frame index table
    for (i, blob) in frame_blobs.iter().enumerate() {
        out.extend_from_slice(&offsets[i].to_le_bytes());
        out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        let ft: u8 = match frames[i].frame_type {
            FrameType::Intra => 0,
            FrameType::Predicted => 1,
            FrameType::Bidirectional => 2,
        };
        out.push(ft);
        // PTS = frame number
        out.extend_from_slice(&(i as u64).to_le_bytes());
    }

    // Frame data
    for blob in &frame_blobs {
        out.extend_from_slice(blob);
    }

    out
}

/// Parse the GNV1 file header and frame index table from raw bytes.
///
/// Returns a `SequenceHeader` that can be used with `deserialize_sequence_frame`
/// for random access to individual frames.
pub fn deserialize_sequence_header(data: &[u8]) -> SequenceHeader {
    assert!(
        data.len() >= GNV1_HEADER_SIZE,
        "GNV1 data too small for header ({} bytes, need at least {})",
        data.len(),
        GNV1_HEADER_SIZE
    );
    assert!(
        &data[0..4] == GNV1_MAGIC,
        "Invalid GNV1 magic (expected GNV1, got {:?})",
        &data[0..4]
    );

    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let width = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let height = u32::from_le_bytes(data[12..16].try_into().unwrap());
    let frame_count = u32::from_le_bytes(data[16..20].try_into().unwrap());
    let framerate_num = u32::from_le_bytes(data[20..24].try_into().unwrap());
    let framerate_den = u32::from_le_bytes(data[24..28].try_into().unwrap());

    let index_table_size = frame_count as usize * GNV1_INDEX_ENTRY_SIZE;
    assert!(
        data.len() >= GNV1_HEADER_SIZE + index_table_size,
        "GNV1 data too small for index table"
    );

    let mut index = Vec::with_capacity(frame_count as usize);
    let mut pos = GNV1_HEADER_SIZE;
    for _ in 0..frame_count {
        let offset = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let frame_type = data[pos];
        pos += 1;
        let pts = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        index.push(FrameIndexEntry {
            offset,
            size,
            frame_type,
            pts,
        });
    }

    SequenceHeader {
        version,
        width,
        height,
        frame_count,
        framerate_num,
        framerate_den,
        index,
    }
}

/// Deserialize a single frame from a GNV1 container by index.
///
/// Uses the frame index table for O(1) random access — no need to parse
/// preceding frames.
pub fn deserialize_sequence_frame(
    data: &[u8],
    header: &SequenceHeader,
    frame_idx: usize,
) -> crate::CompressedFrame {
    assert!(
        frame_idx < header.frame_count as usize,
        "Frame index {} out of range (sequence has {} frames)",
        frame_idx,
        header.frame_count
    );

    let entry = &header.index[frame_idx];
    let start = entry.offset as usize;
    let end = start + entry.size as usize;
    assert!(
        end <= data.len(),
        "Frame {} data extends beyond file (offset {}..{}, file size {})",
        frame_idx,
        start,
        end,
        data.len()
    );

    deserialize_compressed(&data[start..end])
}

/// Find the nearest preceding I-frame (keyframe) for a target PTS.
///
/// Scans the frame index backwards from the frame at or after `target_pts`
/// to find the most recent Intra frame. Returns the frame index of that
/// keyframe. Used for seek operations.
pub fn seek_to_keyframe(header: &SequenceHeader, target_pts: u64) -> usize {
    if header.index.is_empty() {
        return 0;
    }

    // Find the frame at or just before target_pts
    let target_frame = header
        .index
        .iter()
        .rposition(|e| e.pts <= target_pts)
        .unwrap_or(0);

    // Scan backwards to find the nearest preceding I-frame
    for i in (0..=target_frame).rev() {
        if header.index[i].frame_type == 0 {
            // Intra
            return i;
        }
    }

    // Fallback: first frame (should always be an I-frame)
    0
}

/// Serialize frame header fields common to GP10 and GP11.
fn serialize_frame_header(frame: &crate::CompressedFrame, out: &mut Vec<u8>) {
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
    out.push(u8::from(frame.config.per_subband_entropy));
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
    let cfl_enabled: u8 = u8::from(frame.cfl_alphas.is_some());
    out.push(cfl_enabled);
    if let Some(ref cfl) = frame.cfl_alphas {
        out.extend_from_slice(&cfl.num_subbands.to_le_bytes());
        let tiles_x = frame.info.width.div_ceil(frame.info.tile_size);
        let tiles_y = frame.info.height.div_ceil(frame.info.tile_size);
        let num_cfl_tiles = tiles_x * tiles_y;
        out.extend_from_slice(&num_cfl_tiles.to_le_bytes());
        for &a in &cfl.alphas {
            out.extend_from_slice(&a.to_le_bytes());
        }
    }
    // Adaptive quantization config + weight map
    let aq_flag: u32 = u32::from(frame.config.adaptive_quantization);
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
    // Frame type: 0 = Intra, 1 = Predicted, 2 = Bidirectional
    let frame_type_byte: u8 = match frame.frame_type {
        crate::FrameType::Intra => 0,
        crate::FrameType::Predicted => 1,
        crate::FrameType::Bidirectional => 2,
    };
    out.push(frame_type_byte);
}

/// Serialize tile data blobs from entropy data.
fn serialize_tile_blobs(entropy: &crate::EntropyData) -> Vec<Vec<u8>> {
    match entropy {
        crate::EntropyData::Rans(tiles) => {
            tiles.iter().map(rans::serialize_tile_interleaved).collect()
        }
        crate::EntropyData::SubbandRans(tiles) => {
            tiles.iter().map(rans::serialize_tile_subband).collect()
        }
        crate::EntropyData::Bitplane(tiles) => {
            tiles.iter().map(bitplane::serialize_tile_bitplane).collect()
        }
    }
}

/// Serialize a CompressedFrame to the GP11 binary format.
///
/// GP11 adds over GP10:
/// - Per-tile CRC-32 checksums for error detection
/// - Tile index table (sizes + CRCs) for random tile access
/// - Full B-frame motion serialization (backward vectors + block modes)
pub fn serialize_compressed(frame: &crate::CompressedFrame) -> Vec<u8> {
    let mut out = Vec::new();
    // Magic: GP11
    out.extend_from_slice(b"GP11");
    // Common header fields
    serialize_frame_header(frame, &mut out);
    // Motion field — GP11 serializes full motion for P and B frames
    if let Some(ref mf) = frame.motion_field {
        out.extend_from_slice(&(mf.block_size as u16).to_le_bytes());
        let num_blocks = mf.vectors.len() as u32;
        out.extend_from_slice(&num_blocks.to_le_bytes());
        // Forward motion vectors
        for mv in &mf.vectors {
            out.extend_from_slice(&mv[0].to_le_bytes());
            out.extend_from_slice(&mv[1].to_le_bytes());
        }
        // B-frame backward vectors + block modes (GP11 extension)
        if frame.frame_type == crate::FrameType::Bidirectional {
            if let Some(ref bwd) = mf.backward_vectors {
                out.extend_from_slice(&(bwd.len() as u32).to_le_bytes());
                for mv in bwd {
                    out.extend_from_slice(&mv[0].to_le_bytes());
                    out.extend_from_slice(&mv[1].to_le_bytes());
                }
            } else {
                out.extend_from_slice(&0u32.to_le_bytes());
            }
            if let Some(ref modes) = mf.block_modes {
                out.extend_from_slice(&(modes.len() as u32).to_le_bytes());
                out.extend_from_slice(modes);
            } else {
                out.extend_from_slice(&0u32.to_le_bytes());
            }
        }
    }
    // Entropy coder type: 0 = rANS, 1 = bitplane, 2 = per-subband rANS
    let entropy_type: u32 = match &frame.entropy {
        crate::EntropyData::Rans(_) => 0,
        crate::EntropyData::SubbandRans(_) => 2,
        crate::EntropyData::Bitplane(_) => 1,
    };
    out.extend_from_slice(&entropy_type.to_le_bytes());
    // Serialize each tile to bytes, compute CRC-32
    let tile_blobs = serialize_tile_blobs(&frame.entropy);
    let num_tiles = tile_blobs.len() as u32;
    out.extend_from_slice(&num_tiles.to_le_bytes());
    // Tile index table: [size: u32, crc32: u32] × num_tiles
    for blob in &tile_blobs {
        out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        out.extend_from_slice(&crc32(blob).to_le_bytes());
    }
    // Tile data (concatenated, sizes from index)
    for blob in &tile_blobs {
        out.extend_from_slice(blob);
    }
    out
}

/// Tile CRC validation result returned by GP11 deserialization.
#[derive(Debug, Clone)]
pub struct TileCrcResult {
    /// Index of the tile
    pub tile_index: usize,
    /// Expected CRC from the bitstream
    pub expected: u32,
    /// Actual CRC computed from tile data
    pub actual: u32,
}

impl TileCrcResult {
    pub fn is_valid(&self) -> bool {
        self.expected == self.actual
    }
}

/// Result of deserializing a GP11 frame with CRC validation.
#[derive(Debug, Clone)]
pub struct DeserializeResult {
    pub frame: crate::CompressedFrame,
    /// Per-tile CRC validation (empty for GP10/GPC9/GPC8 which have no CRCs)
    pub tile_crcs: Vec<TileCrcResult>,
}

impl DeserializeResult {
    /// Returns indices of tiles that failed CRC validation.
    pub fn corrupt_tiles(&self) -> Vec<usize> {
        self.tile_crcs
            .iter()
            .filter(|c| !c.is_valid())
            .map(|c| c.tile_index)
            .collect()
    }

    /// True if all tile CRCs are valid (or no CRCs present).
    pub fn all_valid(&self) -> bool {
        self.tile_crcs.iter().all(|c| c.is_valid())
    }

    /// Replace corrupt tiles with zero-data tiles (decode to mid-gray).
    /// Returns the list of tile indices that were substituted.
    pub fn substitute_corrupt_tiles(&mut self) -> Vec<usize> {
        let corrupt = self.corrupt_tiles();
        if corrupt.is_empty() {
            return corrupt;
        }
        substitute_tiles(&mut self.frame, &corrupt);
        corrupt
    }
}

/// Replace specified tile indices with trivial zero tiles in-place.
/// Zero tiles decode to zero-valued coefficients, which produce mid-gray
/// after inverse wavelet and inverse color conversion.
pub fn substitute_tiles(frame: &mut crate::CompressedFrame, tile_indices: &[usize]) {
    for &idx in tile_indices {
        match &mut frame.entropy {
            crate::EntropyData::Rans(ref mut tiles) => {
                if idx < tiles.len() {
                    tiles[idx] = make_zero_interleaved_tile(tiles[idx].num_coefficients);
                }
            }
            crate::EntropyData::SubbandRans(ref mut tiles) => {
                if idx < tiles.len() {
                    let t = &tiles[idx];
                    tiles[idx] = make_zero_subband_tile(
                        t.num_coefficients,
                        t.tile_size,
                        t.num_levels,
                    );
                }
            }
            crate::EntropyData::Bitplane(ref mut tiles) => {
                if idx < tiles.len() {
                    let t = &tiles[idx];
                    tiles[idx] = crate::encoder::bitplane::BitplaneTile {
                        num_coefficients: t.num_coefficients,
                        tile_size: t.tile_size,
                        block_offsets: vec![0; t.block_offsets.len()],
                        block_data: Vec::new(),
                    };
                }
            }
        }
    }
}

/// Create a trivial InterleavedRansTile that decodes to all zeros.
fn make_zero_interleaved_tile(num_coefficients: u32) -> crate::encoder::rans::InterleavedRansTile {
    // Single-symbol alphabet: just symbol 0 with probability 1.0
    crate::encoder::rans::InterleavedRansTile {
        min_val: 0,
        alphabet_size: 1,
        num_coefficients,
        zrun_base: 0,
        freqs: vec![4096], // full probability on symbol 0
        cumfreqs: vec![0, 4096],
        stream_data: vec![Vec::new(); 32],
        stream_initial_state: vec![4096; 32], // initial state = RANS_M for single symbol
    }
}

/// Create a trivial SubbandRansTile that decodes to all zeros.
fn make_zero_subband_tile(
    num_coefficients: u32,
    tile_size: u32,
    num_levels: u32,
) -> crate::encoder::rans::SubbandRansTile {
    let num_groups = 1 + num_levels;
    let groups = (0..num_groups)
        .map(|_| crate::encoder::rans::SubbandGroupFreqs {
            min_val: 0,
            alphabet_size: 1,
            zrun_base: 0,
            freqs: vec![4096],
            cumfreqs: vec![0, 4096],
        })
        .collect();
    crate::encoder::rans::SubbandRansTile {
        num_coefficients,
        tile_size,
        num_levels,
        num_groups,
        groups,
        stream_data: vec![Vec::new(); 32],
        stream_initial_state: vec![4096; 32],
    }
}

/// Deserialize a CompressedFrame from GP11/GP10/GPC9/GPC8 binary format.
/// Returns the frame without CRC validation. Use `deserialize_compressed_validated`
/// for GP11 CRC checking.
pub fn deserialize_compressed(data: &[u8]) -> crate::CompressedFrame {
    deserialize_compressed_validated(data).frame
}

/// Deserialize a CompressedFrame with per-tile CRC validation (GP11).
/// For GP10/GPC9/GPC8, returns empty `tile_crcs`.
pub fn deserialize_compressed_validated(data: &[u8]) -> DeserializeResult {
    assert!(data.len() >= 37, "File too small");
    let magic = &data[0..4];
    let is_gpc9 = magic == b"GPC9";
    let is_gp10 = magic == b"GP10";
    let is_gp11 = magic == b"GP11";
    assert!(
        magic == b"GPC8" || is_gpc9 || is_gp10 || is_gp11,
        "Invalid magic (expected GPC8, GPC9, GP10 or GP11; older files must be re-encoded)"
    );

    // --- Common header (same layout for GPC9/GP10/GP11; GPC8 lacks per-subband flag) ---
    let width = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let height = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let bit_depth = u32::from_le_bytes(data[12..16].try_into().unwrap());
    let tile_size = u32::from_le_bytes(data[16..20].try_into().unwrap());
    let qstep = f32::from_le_bytes(data[20..24].try_into().unwrap());
    let dead_zone = f32::from_le_bytes(data[24..28].try_into().unwrap());
    let wavelet_levels = u32::from_le_bytes(data[28..32].try_into().unwrap());
    let wavelet_type = match data[32] {
        0 => crate::WaveletType::LeGall53,
        1 => crate::WaveletType::CDF97,
        w => panic!("Unknown wavelet type: {w}"),
    };

    // Per-subband entropy flag (GPC9/GP10/GP11 only)
    let (per_subband_entropy, mut pos) = if is_gpc9 || is_gp10 || is_gp11 {
        (data[33] != 0, 34)
    } else {
        (false, 33)
    };

    // --- Subband weights ---
    let ll = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    let num_detail = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
    pos += 8;
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

    // --- CfL alpha side info ---
    let cfl_flag = data[pos];
    pos += 1;
    let (cfl_enabled, cfl_alphas) = if cfl_flag != 0 {
        let nsb = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let num_cfl_tiles = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
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

    // --- Adaptive quantization ---
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

    // --- Frame type + motion field (GP10/GP11 only) ---
    let (frame_type, motion_field) = if is_gp10 || is_gp11 {
        let ft = match data[pos] {
            0 => crate::FrameType::Intra,
            1 => crate::FrameType::Predicted,
            2 => crate::FrameType::Bidirectional,
            f => panic!("Unknown frame type: {f}"),
        };
        pos += 1;
        let mf = if ft == crate::FrameType::Predicted || ft == crate::FrameType::Bidirectional {
            let block_size = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as u32;
            pos += 2;
            let num_blocks =
                u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let mut vectors = Vec::with_capacity(num_blocks);
            for _ in 0..num_blocks {
                let dx = i16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
                let dy = i16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap());
                vectors.push([dx, dy]);
                pos += 4;
            }
            // GP11 B-frames: backward vectors + block modes
            let (backward_vectors, block_modes) =
                if is_gp11 && ft == crate::FrameType::Bidirectional {
                    let bwd_count =
                        u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                    pos += 4;
                    let bwd = if bwd_count > 0 {
                        let mut bv = Vec::with_capacity(bwd_count);
                        for _ in 0..bwd_count {
                            let dx =
                                i16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
                            let dy =
                                i16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap());
                            bv.push([dx, dy]);
                            pos += 4;
                        }
                        Some(bv)
                    } else {
                        None
                    };
                    let modes_count =
                        u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                    pos += 4;
                    let modes = if modes_count > 0 {
                        let m = data[pos..pos + modes_count].to_vec();
                        pos += modes_count;
                        Some(m)
                    } else {
                        None
                    };
                    (bwd, modes)
                } else {
                    (None, None)
                };
            Some(crate::MotionField {
                vectors,
                block_size,
                backward_vectors,
                block_modes,
            })
        } else {
            None
        };
        (ft, mf)
    } else {
        // GPC8/GPC9: always Intra, no motion
        (crate::FrameType::Intra, None)
    };

    // --- Entropy tiles ---
    let entropy_type = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let num_tiles = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    // GP11: tile index table with sizes + CRC-32s
    let (tile_sizes, tile_crcs) = if is_gp11 {
        let mut sizes = Vec::with_capacity(num_tiles);
        let mut expected_crcs = Vec::with_capacity(num_tiles);
        for _ in 0..num_tiles {
            sizes.push(u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
            pos += 4;
            expected_crcs.push(u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        // Validate CRCs
        let mut crcs = Vec::with_capacity(num_tiles);
        let mut offset = pos;
        for i in 0..num_tiles {
            let sz = sizes[i] as usize;
            let actual = crc32(&data[offset..offset + sz]);
            crcs.push(TileCrcResult {
                tile_index: i,
                expected: expected_crcs[i],
                actual,
            });
            offset += sz;
        }
        (Some(sizes), crcs)
    } else {
        (None, Vec::new())
    };

    // Deserialize tile data
    let (entropy_coder, entropy, per_subband) = match entropy_type {
        0 => {
            let mut tiles = Vec::with_capacity(num_tiles);
            for i in 0..num_tiles {
                let slice = if let Some(ref sizes) = tile_sizes {
                    &data[pos..pos + sizes[i] as usize]
                } else {
                    &data[pos..]
                };
                let (tile, consumed) = rans::deserialize_tile_interleaved(slice);
                tiles.push(tile);
                pos += consumed;
            }
            (crate::EntropyCoder::Rans, crate::EntropyData::Rans(tiles), false)
        }
        1 => {
            let mut tiles = Vec::with_capacity(num_tiles);
            for i in 0..num_tiles {
                let slice = if let Some(ref sizes) = tile_sizes {
                    &data[pos..pos + sizes[i] as usize]
                } else {
                    &data[pos..]
                };
                let (tile, consumed) = bitplane::deserialize_tile_bitplane(slice);
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
            for i in 0..num_tiles {
                let slice = if let Some(ref sizes) = tile_sizes {
                    &data[pos..pos + sizes[i] as usize]
                } else {
                    &data[pos..]
                };
                let (tile, consumed) = rans::deserialize_tile_subband(slice);
                tiles.push(tile);
                pos += consumed;
            }
            (
                crate::EntropyCoder::Rans,
                crate::EntropyData::SubbandRans(tiles),
                true,
            )
        }
        _ => panic!("Unknown entropy coder type: {entropy_type}"),
    };

    DeserializeResult {
        frame: crate::CompressedFrame {
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
        },
        tile_crcs,
    }
}
