use crate::encoder::{bitplane, huffman, rans, rice};
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
    // Transform type: 0 = Wavelet, 1 = BlockDCT8
    let transform_byte: u8 = match frame.config.transform_type {
        crate::TransformType::Wavelet => 0,
        crate::TransformType::BlockDCT8 => 1,
    };
    out.push(transform_byte);
    // Per-subband entropy: 0 = off, 1 = on
    out.push(u8::from(frame.config.per_subband_entropy));
    // Chroma format byte (GP13 — always written here to maintain correct byte alignment)
    out.push(frame.info.chroma_format.to_u8());
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
    // Intra prediction modes (packed 2-bit, 4 modes per byte)
    if let Some(ref modes) = frame.intra_modes {
        out.push(1u8); // intra_flag
        let num_blocks = modes.len() as u32 * 4; // approximate: 4 modes per byte
        // Store exact block count from dimensions
        let blocks_x = frame.info.padded_width() / 8;
        let blocks_y = frame.info.padded_height() / 8;
        let exact_blocks = blocks_x * blocks_y;
        out.extend_from_slice(&exact_blocks.to_le_bytes());
        let _ = num_blocks; // suppress warning
        out.extend_from_slice(&(modes.len() as u32).to_le_bytes());
        out.extend_from_slice(modes);
    } else {
        out.push(0u8); // intra_flag = 0
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
        crate::EntropyData::Rice(tiles) => {
            tiles.iter().map(rice::serialize_tile_rice).collect()
        }
        crate::EntropyData::Huffman(tiles) => {
            tiles.iter().map(huffman::serialize_tile_huffman).collect()
        }
    }
}

// ---- Delta MV coding helpers (GP12) ----

/// Zigzag encode i16 → u16: maps 0→0, -1→1, 1→2, -2→3, ...
#[inline]
fn zigzag_encode(val: i16) -> u16 {
    ((val << 1) ^ (val >> 15)) as u16
}

/// Zigzag decode u16 → i16
#[inline]
fn zigzag_decode(val: u16) -> i16 {
    ((val >> 1) as i16) ^ -((val & 1) as i16)
}

/// Write unsigned varint (u16 range: max 3 bytes)
fn write_varint(out: &mut Vec<u8>, val: u16) {
    let mut v = val as u32;
    while v >= 0x80 {
        out.push((v & 0x7F) as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Read unsigned varint → u16
fn read_varint(data: &[u8], pos: &mut usize) -> u16 {
    let mut result = 0u32;
    let mut shift = 0;
    loop {
        let b = data[*pos] as u32;
        *pos += 1;
        result |= (b & 0x7F) << shift;
        if b < 0x80 {
            break;
        }
        shift += 7;
    }
    result as u16
}

/// Median of three i16 values
#[inline]
fn median3(a: i16, b: i16, c: i16) -> i16 {
    a.max(b).min(c).max(a.min(b))
}

/// Compute MV predictor for block at (bx, by) using median of (left, above, above-right).
/// `vectors` must contain already-reconstructed absolute MVs for all prior blocks in raster order.
#[inline]
fn mv_predictor(vectors: &[[i16; 2]], bx: usize, by: usize, blocks_x: usize) -> [i16; 2] {
    let idx = by * blocks_x + bx;
    let left = if bx > 0 { vectors[idx - 1] } else { [0, 0] };
    let above = if by > 0 { vectors[idx - blocks_x] } else { [0, 0] };
    let above_right = if by > 0 && bx + 1 < blocks_x {
        vectors[idx - blocks_x + 1]
    } else {
        above // fallback to above when above-right unavailable
    };
    [
        median3(left[0], above[0], above_right[0]),
        median3(left[1], above[1], above_right[1]),
    ]
}

/// Serialize motion vectors as delta-coded zigzag varints with skip bitmap.
///
/// Format: [skip_bitmap: ceil(N/8) bytes] [delta MVs for non-skip blocks]
/// Skip bit = 1 means MV is (0,0) — no delta bytes written for that block.
/// Skip bit = 0 means delta MV follows (2 zigzag varints: dx, dy).
fn serialize_mvs_delta(out: &mut Vec<u8>, vectors: &[[i16; 2]], blocks_x: usize) {
    let n = vectors.len();
    // Build skip bitmap: bit=1 for MV=(0,0) blocks
    let bitmap_bytes = n.div_ceil(8);
    let bitmap_start = out.len();
    out.resize(bitmap_start + bitmap_bytes, 0);
    for (i, mv) in vectors.iter().enumerate() {
        if mv[0] == 0 && mv[1] == 0 {
            out[bitmap_start + i / 8] |= 1 << (i % 8);
        }
    }
    // Write delta MVs only for non-skip blocks
    let blocks_y = if blocks_x > 0 { n / blocks_x } else { 0 };
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let idx = by * blocks_x + bx;
            if vectors[idx][0] == 0 && vectors[idx][1] == 0 {
                continue; // skip — no MV data needed
            }
            let pred = mv_predictor(vectors, bx, by, blocks_x);
            let dx = vectors[idx][0] - pred[0];
            let dy = vectors[idx][1] - pred[1];
            write_varint(out, zigzag_encode(dx));
            write_varint(out, zigzag_encode(dy));
        }
    }
}

/// Deserialize delta-coded zigzag varint MVs with skip bitmap back to absolute MVs.
fn deserialize_mvs_delta(
    data: &[u8],
    pos: &mut usize,
    num_blocks: usize,
    blocks_x: usize,
) -> Vec<[i16; 2]> {
    // Read skip bitmap
    let bitmap_bytes = num_blocks.div_ceil(8);
    let bitmap = &data[*pos..*pos + bitmap_bytes];
    *pos += bitmap_bytes;
    let mut vectors = Vec::with_capacity(num_blocks);
    let blocks_y = if blocks_x > 0 { num_blocks / blocks_x } else { 0 };
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let idx = by * blocks_x + bx;
            let is_skip = (bitmap[idx / 8] >> (idx % 8)) & 1 != 0;
            if is_skip {
                vectors.push([0, 0]);
            } else {
                let pred = mv_predictor(&vectors, bx, by, blocks_x);
                let dx = zigzag_decode(read_varint(data, pos));
                let dy = zigzag_decode(read_varint(data, pos));
                vectors.push([pred[0] + dx, pred[1] + dy]);
            }
        }
    }
    vectors
}

/// Serialize a CompressedFrame to the GP12 binary format.
///
/// GP12 is identical to GP11 except motion vectors use delta-coded
/// zigzag varint encoding with a median spatial predictor, reducing
/// MV overhead by 50-80% for typical content.
pub fn serialize_compressed(frame: &crate::CompressedFrame) -> Vec<u8> {
    let mut out = Vec::new();
    // Magic: GP15 (splits Rice k_zrl into k_zrl_nz + k_zrl_z per subband for 2-state
    // magnitude-conditioned zero-run context, K_STRIDE 17→25 per tile, #53).
    // GP14 adds fwd_ref_idx + bwd_ref_idx for hierarchical pyramid B-frames.
    // GP13 is GP12 + chroma_format byte.
    out.extend_from_slice(b"GP15");
    // Common header fields (includes chroma_format byte for GP13)
    serialize_frame_header(frame, &mut out);
    // Motion field — GP12 uses delta-coded varint MVs
    if let Some(ref mf) = frame.motion_field {
        out.extend_from_slice(&(mf.block_size as u16).to_le_bytes());
        let num_blocks = mf.vectors.len() as u32;
        out.extend_from_slice(&num_blocks.to_le_bytes());
        // Compute blocks_x from frame dimensions and block_size
        let padded_w = frame.info.padded_width();
        let blocks_x = (padded_w / mf.block_size) as usize;
        // Forward motion vectors (delta-coded zigzag varint)
        serialize_mvs_delta(&mut out, &mf.vectors, blocks_x);
        // B-frame backward vectors + block modes + ref indices (GP14)
        if frame.frame_type == crate::FrameType::Bidirectional {
            if let Some(ref bwd) = mf.backward_vectors {
                out.extend_from_slice(&(bwd.len() as u32).to_le_bytes());
                let bwd_blocks_x = (padded_w / 16) as usize; // B-frames use 16×16
                serialize_mvs_delta(&mut out, bwd, bwd_blocks_x);
            } else {
                out.extend_from_slice(&0u32.to_le_bytes());
            }
            if let Some(ref modes) = mf.block_modes {
                out.extend_from_slice(&(modes.len() as u32).to_le_bytes());
                out.extend_from_slice(modes);
            } else {
                out.extend_from_slice(&0u32.to_le_bytes());
            }
            // GP14: ref pool indices (1 byte each; 0xFF = use default: fwd=0, bwd=1)
            out.push(mf.fwd_ref_idx.unwrap_or(0));
            out.push(mf.bwd_ref_idx.unwrap_or(1));
        }
    }
    // Entropy coder type: 0 = rANS, 1 = bitplane, 2 = per-subband rANS, 3 = Rice, 4 = Huffman
    let entropy_type: u32 = match &frame.entropy {
        crate::EntropyData::Rans(_) => 0,
        crate::EntropyData::SubbandRans(_) => 2,
        crate::EntropyData::Bitplane(_) => 1,
        crate::EntropyData::Rice(_) => 3,
        crate::EntropyData::Huffman(_) => 4,
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
            crate::EntropyData::Rice(ref mut tiles) => {
                if idx < tiles.len() {
                    let t = &tiles[idx];
                    tiles[idx] = crate::encoder::rice::RiceTile {
                        num_coefficients: t.num_coefficients,
                        tile_size: t.tile_size,
                        num_levels: t.num_levels,
                        num_groups: t.num_groups,
                        k_values: vec![0; t.num_groups as usize],
                        k_zrl_nz_values: vec![0; t.num_groups as usize],
                        k_zrl_z_values: vec![0; t.num_groups as usize],
                        skip_bitmap: 0xFF, // all groups skipped
                        stream_lengths: vec![0; rice::RICE_STREAMS_PER_TILE],
                        stream_data: Vec::new(),
                    };
                }
            }
            crate::EntropyData::Huffman(ref mut tiles) => {
                if idx < tiles.len() {
                    let t = &tiles[idx];
                    tiles[idx] = crate::encoder::huffman::HuffmanTile {
                        num_coefficients: t.num_coefficients,
                        tile_size: t.tile_size,
                        num_levels: t.num_levels,
                        num_groups: t.num_groups,
                        code_lengths: vec![vec![0u8; huffman::HUFFMAN_ALPHABET_SIZE]; t.num_groups as usize],
                        k_zrl_values: vec![0; t.num_groups as usize],
                        stream_lengths: vec![0; huffman::HUFFMAN_STREAMS_PER_TILE],
                        stream_data: Vec::new(),
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
    let num_groups = num_levels * 2;
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

/// Deserialize a CompressedFrame from GP12/GP11/GP10/GPC9/GPC8 binary format.
/// Returns the frame without CRC validation. Use `deserialize_compressed_validated`
/// for GP12/GP11 CRC checking.
pub fn deserialize_compressed(data: &[u8]) -> crate::CompressedFrame {
    deserialize_compressed_validated(data).frame
}

/// Deserialize a CompressedFrame with per-tile CRC validation (GP12/GP11).
/// For GP10/GPC9/GPC8, returns empty `tile_crcs`.
pub fn deserialize_compressed_validated(data: &[u8]) -> DeserializeResult {
    assert!(data.len() >= 37, "File too small");
    let magic = &data[0..4];
    let is_gpc9 = magic == b"GPC9";
    let is_gp10 = magic == b"GP10";
    let is_gp11 = magic == b"GP11";
    let is_gp12 = magic == b"GP12";
    let is_gp13 = magic == b"GP13";
    let is_gp14 = magic == b"GP14";
    let is_gp15 = magic == b"GP15";
    assert!(
        magic == b"GPC8" || is_gpc9 || is_gp10 || is_gp11 || is_gp12 || is_gp13 || is_gp14 || is_gp15,
        "Invalid magic (expected GPC8, GPC9, GP10, GP11, GP12, GP13, GP14 or GP15; older files must be re-encoded)"
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

    // Transform type byte (added alongside wavelet_type)
    let transform_type = match data[33] {
        0 => crate::TransformType::Wavelet,
        1 => crate::TransformType::BlockDCT8,
        t => panic!("Unknown transform type: {t}"),
    };

    // Per-subband entropy flag (GPC9/GP10/GP11/GP12/GP13/GP14)
    let (per_subband_entropy, mut pos) = if is_gpc9 || is_gp10 || is_gp11 || is_gp12 || is_gp13 || is_gp14 || is_gp15 {
        (data[34] != 0, 35)
    } else {
        (false, 34)
    };

    // Chroma format byte (GP13/GP14; older formats default to 4:4:4)
    let chroma_format_decoded = if is_gp13 || is_gp14 || is_gp15 {
        let cf = crate::ChromaFormat::from_u8(data[pos])
            .unwrap_or(crate::ChromaFormat::Yuv444);
        pos += 1;
        cf
    } else {
        crate::ChromaFormat::Yuv444
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

    // --- Intra prediction modes (GP11/GP12/GP13/GP14) ---
    let intra_modes = if (is_gp11 || is_gp12 || is_gp13 || is_gp14 || is_gp15) && data[pos] != 0 {
        pos += 1; // skip intra_flag
        let _num_blocks = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let packed_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let modes = data[pos..pos + packed_len].to_vec();
        pos += packed_len;
        Some(modes)
    } else if is_gp11 || is_gp12 || is_gp13 || is_gp14 || is_gp15 {
        pos += 1; // skip intra_flag = 0
        None
    } else {
        None
    };

    // --- Frame type + motion field (GP10/GP11/GP12/GP13/GP14) ---
    let (frame_type, motion_field) = if is_gp10 || is_gp11 || is_gp12 || is_gp13 || is_gp14 || is_gp15 {
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
            let vectors = if is_gp12 || is_gp13 || is_gp14 || is_gp15 {
                // GP12/GP13/GP14: delta-coded zigzag varint MVs
                let padded_w = width.div_ceil(tile_size) * tile_size;
                let blocks_x = (padded_w / block_size) as usize;
                deserialize_mvs_delta(data, &mut pos, num_blocks, blocks_x)
            } else {
                // GP10/GP11: raw i16 MVs
                let mut vecs = Vec::with_capacity(num_blocks);
                for _ in 0..num_blocks {
                    let dx = i16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
                    let dy = i16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap());
                    vecs.push([dx, dy]);
                    pos += 4;
                }
                vecs
            };
            // GP11/GP12/GP13/GP14 B-frames: backward vectors + block modes
            let (backward_vectors, block_modes, fwd_ref_idx, bwd_ref_idx) =
                if (is_gp11 || is_gp12 || is_gp13 || is_gp14 || is_gp15) && ft == crate::FrameType::Bidirectional {
                    let bwd_count =
                        u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                    pos += 4;
                    let bwd = if bwd_count > 0 {
                        if is_gp12 || is_gp13 || is_gp14 || is_gp15 {
                            let padded_w = width.div_ceil(tile_size) * tile_size;
                            let bwd_blocks_x = (padded_w / 16) as usize;
                            Some(deserialize_mvs_delta(data, &mut pos, bwd_count, bwd_blocks_x))
                        } else {
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
                        }
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
                    // GP14+: ref pool indices (1 byte each); older formats default to 0/1
                    let (fwd_idx, bwd_idx) = if is_gp14 || is_gp15 {
                        let f = data[pos];
                        let b = data[pos + 1];
                        pos += 2;
                        (Some(f), Some(b))
                    } else {
                        (None, None)
                    };
                    (bwd, modes, fwd_idx, bwd_idx)
                } else {
                    (None, None, None, None)
                };
            Some(crate::MotionField {
                vectors,
                block_size,
                backward_vectors,
                block_modes,
                fwd_ref_idx,
                bwd_ref_idx,
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

    // GP11/GP12/GP13/GP14: tile index table with sizes + CRC-32s
    let (tile_sizes, tile_crcs) = if is_gp11 || is_gp12 || is_gp13 || is_gp14 || is_gp15 {
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
        3 => {
            let mut tiles = Vec::with_capacity(num_tiles);
            for i in 0..num_tiles {
                let slice = if let Some(ref sizes) = tile_sizes {
                    &data[pos..pos + sizes[i] as usize]
                } else {
                    &data[pos..]
                };
                let (tile, consumed) = rice::deserialize_tile_rice(slice);
                tiles.push(tile);
                pos += consumed;
            }
            (
                crate::EntropyCoder::Rice,
                crate::EntropyData::Rice(tiles),
                false,
            )
        }
        4 => {
            let mut tiles = Vec::with_capacity(num_tiles);
            for i in 0..num_tiles {
                let slice = if let Some(ref sizes) = tile_sizes {
                    &data[pos..pos + sizes[i] as usize]
                } else {
                    &data[pos..]
                };
                let (tile, consumed) = huffman::deserialize_tile_huffman(slice);
                tiles.push(tile);
                pos += consumed;
            }
            (
                crate::EntropyCoder::Huffman,
                crate::EntropyData::Huffman(tiles),
                false,
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
                chroma_format: chroma_format_decoded,
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
                transform_type,
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
            intra_modes,
            residual_stats: None,
            residual_stats_co: None,
            residual_stats_cg: None,
        },
        tile_crcs,
    }
}

// ---- GNV2 Temporal Wavelet Sequence Container Format ----
//
// File layout:
//   File Header (34 bytes, fixed):
//     magic: "GNV2" (4 bytes)
//     version: u32 LE           (currently 1)
//     width: u32 LE
//     height: u32 LE
//     frame_count: u32 LE       (total original frames)
//     framerate_num: u32 LE
//     framerate_den: u32 LE
//     temporal_transform: u8    (0=none, 1=haar, 2=legall53)
//     gop_size: u8              (2, 4, 8)
//     highpass_qstep_mul: f32   (e.g. 2.0)
//
//   Frame Index Table (N entries, 22 bytes each):
//     offset: u64 LE            (byte offset from file start)
//     size: u32 LE              (frame data size)
//     frame_role: u8            (0=lowpass/seekable, 1=highpass, 2=tail-iframe)
//     temporal_level: u8        (for highpass: which level; for lowpass/tail: 0)
//     gop_index: u16 LE         (which GOP, 0-based)
//     frame_index_in_gop: u16 LE (position within GOP)
//     pts: u32 LE               (presentation timestamp, display-order frame number)
//
//   Frame Data:
//     [GP12 frame blobs in index order]
//
// Frame ordering per GOP: lowpass first, then highpass from deepest level to finest.
// After all GOPs: tail I-frames.

const GNV2_MAGIC: &[u8; 4] = b"GNV2";
const GNV2_VERSION: u32 = 1;
const GNV2_HEADER_SIZE: usize = 34;
const GNV2_INDEX_ENTRY_SIZE: usize = 22; // 8 + 4 + 1 + 1 + 2 + 2 + 4

/// Entry in the GNV2 frame index table.
#[derive(Debug, Clone, Copy)]
pub struct TemporalFrameIndexEntry {
    /// Byte offset from start of file to frame data
    pub offset: u64,
    /// Size of this frame's serialized data in bytes
    pub size: u32,
    /// Frame role: 0 = lowpass (seekable keyframe), 1 = highpass, 2 = tail I-frame
    pub frame_role: u8,
    /// Temporal wavelet level (0 = finest). 0 for lowpass and tail frames.
    pub temporal_level: u8,
    /// GOP index (0-based)
    pub gop_index: u16,
    /// Frame position within its GOP
    pub frame_index_in_gop: u16,
    /// Presentation timestamp (display-order frame number)
    pub pts: u32,
}

/// Parsed GNV2 temporal sequence header with frame index.
#[derive(Debug, Clone)]
pub struct TemporalSequenceHeader {
    pub version: u32,
    pub width: u32,
    pub height: u32,
    pub frame_count: u32,
    pub framerate_num: u32,
    pub framerate_den: u32,
    pub temporal_transform: crate::TemporalTransform,
    pub gop_size: u8,
    pub highpass_qstep_mul: f32,
    /// Frame index table for random access
    pub index: Vec<TemporalFrameIndexEntry>,
}

impl TemporalSequenceHeader {
    /// Duration in seconds
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

    /// Number of GOPs in the sequence
    pub fn num_groups(&self) -> usize {
        // Count distinct gop_index values with frame_role != 2 (tail)
        let max_gop = self
            .index
            .iter()
            .filter(|e| e.frame_role != 2)
            .map(|e| e.gop_index)
            .max();
        match max_gop {
            Some(g) => g as usize + 1,
            None => 0,
        }
    }

    /// Number of tail I-frames
    pub fn num_tail_iframes(&self) -> usize {
        self.index.iter().filter(|e| e.frame_role == 2).count()
    }
}

/// Encode temporal transform enum to byte.
fn temporal_transform_to_byte(t: crate::TemporalTransform) -> u8 {
    match t {
        crate::TemporalTransform::None => 0,
        crate::TemporalTransform::Haar => 1,
        crate::TemporalTransform::LeGall53 => 2,
    }
}

/// Decode byte to temporal transform enum.
fn byte_to_temporal_transform(b: u8) -> crate::TemporalTransform {
    match b {
        0 => crate::TemporalTransform::None,
        1 => crate::TemporalTransform::Haar,
        2 => crate::TemporalTransform::LeGall53,
        _ => panic!("Unknown temporal transform byte: {b}"),
    }
}

/// Serialize a `TemporalEncodedSequence` into the GNV2 container format.
///
/// Frame ordering within each GOP: lowpass first, then highpass from deepest
/// level to finest. After all GOPs: tail I-frames.
pub fn serialize_temporal_sequence(
    seq: &crate::TemporalEncodedSequence,
    framerate: (u32, u32),
) -> Vec<u8> {
    // Get width/height from first available frame
    let first_frame = if !seq.groups.is_empty() {
        &seq.groups[0].low_frame
    } else if !seq.tail_iframes.is_empty() {
        &seq.tail_iframes[0]
    } else {
        panic!("Cannot serialize empty temporal sequence");
    };
    let width = first_frame.info.width;
    let height = first_frame.info.height;

    // Compute highpass_qstep_mul from first frame's config
    let highpass_qstep_mul = first_frame.config.temporal_highpass_qstep_mul;

    // --- Serialize all GP12 frame blobs and build index entries ---
    let mut frame_blobs: Vec<Vec<u8>> = Vec::new();
    let mut index_entries: Vec<TemporalFrameIndexEntry> = Vec::new();

    // PTS assignment: within each GOP, the lowpass represents the temporal
    // average, and highpass frames represent detail. For display order,
    // GOP g covers frames [g*gop_size .. (g+1)*gop_size).
    let gop_size = seq.gop_size;

    // Use group_gop_indices to map each group to its actual GOP index.
    // This handles scene cuts: some GOP indices may be skipped (their frames are tail_iframes).
    let fallback_gop_indices: Vec<usize>;
    let group_gop_indices: &[usize] = if seq.group_gop_indices.len() == seq.groups.len() {
        &seq.group_gop_indices
    } else {
        // Backwards-compat: no scene-cut support, groups[g] → GOP g
        fallback_gop_indices = (0..seq.groups.len()).collect();
        &fallback_gop_indices
    };

    for (g, group) in seq.groups.iter().enumerate() {
        let actual_gop_idx = group_gop_indices[g];
        let gop_base_pts = (actual_gop_idx * gop_size) as u32;
        let mut frame_pos_in_gop: u16 = 0;

        // Lowpass frame (seekable keyframe)
        let blob = serialize_compressed(&group.low_frame);
        frame_blobs.push(blob);
        index_entries.push(TemporalFrameIndexEntry {
            offset: 0, // filled in later
            size: 0,   // filled in later
            frame_role: 0,
            temporal_level: 0,
            gop_index: actual_gop_idx as u16,
            frame_index_in_gop: frame_pos_in_gop,
            pts: gop_base_pts,
        });
        frame_pos_in_gop += 1;

        // Highpass frames: deepest level first (last in high_frames vec),
        // then progressively finer levels
        // high_frames[0] = finest (level 0), high_frames[last] = deepest
        let num_levels = group.high_frames.len();
        for level_rev in 0..num_levels {
            // level_rev=0 → deepest level, level_rev=num_levels-1 → finest
            let level_idx = num_levels - 1 - level_rev;
            let temporal_level = level_idx as u8;
            for (fi, hp_frame) in group.high_frames[level_idx].iter().enumerate() {
                let blob = serialize_compressed(hp_frame);
                frame_blobs.push(blob);
                index_entries.push(TemporalFrameIndexEntry {
                    offset: 0,
                    size: 0,
                    frame_role: 1,
                    temporal_level,
                    gop_index: actual_gop_idx as u16,
                    frame_index_in_gop: frame_pos_in_gop,
                    pts: gop_base_pts + frame_pos_in_gop as u32,
                });
                let _ = fi; // suppress unused warning
                frame_pos_in_gop += 1;
            }
        }
    }

    // Tail I-frames (includes scene-cut I-frames with arbitrary PTS).
    // Use tail_iframe_pts if provided; otherwise fall back to sequential PTS after last group.
    let last_group_gop = group_gop_indices.last().copied().unwrap_or(0);
    let tail_base_pts_fallback = if seq.groups.is_empty() {
        0u32
    } else {
        ((last_group_gop + 1) * gop_size) as u32
    };
    for (ti, tail_frame) in seq.tail_iframes.iter().enumerate() {
        let pts = if ti < seq.tail_iframe_pts.len() {
            seq.tail_iframe_pts[ti]
        } else {
            tail_base_pts_fallback + ti as u32
        };
        let blob = serialize_compressed(tail_frame);
        frame_blobs.push(blob);
        index_entries.push(TemporalFrameIndexEntry {
            offset: 0,
            size: 0,
            frame_role: 2,
            temporal_level: 0,
            gop_index: (pts / gop_size as u32) as u16,
            frame_index_in_gop: (pts % gop_size as u32) as u16,
            pts,
        });
    }

    // --- Compute byte offsets ---
    let total_index_entries = index_entries.len();
    let index_table_size = total_index_entries * GNV2_INDEX_ENTRY_SIZE;
    let data_start = GNV2_HEADER_SIZE + index_table_size;

    let mut current_offset = data_start as u64;
    for (i, blob) in frame_blobs.iter().enumerate() {
        index_entries[i].offset = current_offset;
        index_entries[i].size = blob.len() as u32;
        current_offset += blob.len() as u64;
    }
    let total_size = current_offset as usize;

    // --- Write output ---
    let mut out = Vec::with_capacity(total_size);

    // File header (40 bytes)
    out.extend_from_slice(GNV2_MAGIC);
    out.extend_from_slice(&GNV2_VERSION.to_le_bytes());
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&(seq.frame_count as u32).to_le_bytes());
    out.extend_from_slice(&framerate.0.to_le_bytes());
    out.extend_from_slice(&framerate.1.to_le_bytes());
    out.push(temporal_transform_to_byte(seq.mode));
    out.push(seq.gop_size as u8);
    out.extend_from_slice(&highpass_qstep_mul.to_le_bytes());
    debug_assert_eq!(out.len(), GNV2_HEADER_SIZE);

    // Frame index table
    for entry in &index_entries {
        out.extend_from_slice(&entry.offset.to_le_bytes());
        out.extend_from_slice(&entry.size.to_le_bytes());
        out.push(entry.frame_role);
        out.push(entry.temporal_level);
        out.extend_from_slice(&entry.gop_index.to_le_bytes());
        out.extend_from_slice(&entry.frame_index_in_gop.to_le_bytes());
        out.extend_from_slice(&entry.pts.to_le_bytes());
    }

    // Frame data
    for blob in &frame_blobs {
        out.extend_from_slice(blob);
    }

    debug_assert_eq!(out.len(), total_size);
    out
}

/// Parse the GNV2 file header and frame index table from raw bytes.
///
/// Returns a `TemporalSequenceHeader` for random access to groups and frames.
pub fn deserialize_temporal_sequence_header(data: &[u8]) -> TemporalSequenceHeader {
    assert!(
        data.len() >= GNV2_HEADER_SIZE,
        "GNV2 data too small for header ({} bytes, need at least {})",
        data.len(),
        GNV2_HEADER_SIZE
    );
    assert!(
        &data[0..4] == GNV2_MAGIC,
        "Invalid GNV2 magic (expected GNV2, got {:?})",
        &data[0..4]
    );

    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let width = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let height = u32::from_le_bytes(data[12..16].try_into().unwrap());
    let frame_count = u32::from_le_bytes(data[16..20].try_into().unwrap());
    let framerate_num = u32::from_le_bytes(data[20..24].try_into().unwrap());
    let framerate_den = u32::from_le_bytes(data[24..28].try_into().unwrap());
    let temporal_transform = byte_to_temporal_transform(data[28]);
    let gop_size = data[29];
    let highpass_qstep_mul = f32::from_le_bytes(data[30..34].try_into().unwrap());

    // Determine number of index entries by scanning ahead.
    // We know each entry is GNV2_INDEX_ENTRY_SIZE bytes, and the index table
    // immediately follows the header. We need to figure out how many entries
    // there are. We can compute this from the first entry's offset (which
    // tells us where data starts, hence the size of header + index table).
    //
    // But we need at least one entry to read the first offset. Handle the
    // edge case of zero frames.
    if frame_count == 0 {
        return TemporalSequenceHeader {
            version,
            width,
            height,
            frame_count,
            framerate_num,
            framerate_den,
            temporal_transform,
            gop_size,
            highpass_qstep_mul,
            index: Vec::new(),
        };
    }

    // Read the first entry's offset to determine total index entries
    assert!(
        data.len() >= GNV2_HEADER_SIZE + 8,
        "GNV2 data too small to read first index entry offset"
    );
    let first_offset = u64::from_le_bytes(
        data[GNV2_HEADER_SIZE..GNV2_HEADER_SIZE + 8]
            .try_into()
            .unwrap(),
    );
    let index_table_total_size = first_offset as usize - GNV2_HEADER_SIZE;
    let num_entries = index_table_total_size / GNV2_INDEX_ENTRY_SIZE;

    assert!(
        data.len() >= GNV2_HEADER_SIZE + num_entries * GNV2_INDEX_ENTRY_SIZE,
        "GNV2 data too small for index table"
    );

    let mut index = Vec::with_capacity(num_entries);
    let mut pos = GNV2_HEADER_SIZE;
    for _ in 0..num_entries {
        let offset = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let frame_role = data[pos];
        pos += 1;
        let temporal_level = data[pos];
        pos += 1;
        let gop_index = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
        pos += 2;
        let frame_index_in_gop = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
        pos += 2;
        let pts = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        index.push(TemporalFrameIndexEntry {
            offset,
            size,
            frame_role,
            temporal_level,
            gop_index,
            frame_index_in_gop,
            pts,
        });
    }

    TemporalSequenceHeader {
        version,
        width,
        height,
        frame_count,
        framerate_num,
        framerate_den,
        temporal_transform,
        gop_size,
        highpass_qstep_mul,
        index,
    }
}

/// Deserialize a single temporal group (GOP) from a GNV2 container.
///
/// Uses the frame index for O(1) access to the correct byte ranges.
pub fn deserialize_temporal_group(
    data: &[u8],
    header: &TemporalSequenceHeader,
    group_idx: usize,
) -> crate::TemporalGroup {
    // Collect all index entries belonging to this GOP (exclude tail I-frames)
    let gop_entries: Vec<&TemporalFrameIndexEntry> = header
        .index
        .iter()
        .filter(|e| e.gop_index == group_idx as u16 && e.frame_role != 2)
        .collect();

    assert!(
        !gop_entries.is_empty(),
        "No frames found for GOP index {group_idx}"
    );

    // The first entry must be the lowpass (frame_role=0)
    assert_eq!(
        gop_entries[0].frame_role, 0,
        "First frame in GOP must be lowpass (frame_role=0)"
    );

    // Deserialize lowpass frame
    let lp_entry = gop_entries[0];
    let lp_start = lp_entry.offset as usize;
    let lp_end = lp_start + lp_entry.size as usize;
    assert!(
        lp_end <= data.len(),
        "Lowpass frame data extends beyond file"
    );
    let low_frame = deserialize_compressed(&data[lp_start..lp_end]);

    // Collect highpass entries (frame_role=1), grouped by temporal_level
    // Entries are stored deepest-first in the file, but high_frames vec
    // is indexed [0] = finest level, so we need to reconstruct that ordering.
    let hp_entries: Vec<&TemporalFrameIndexEntry> =
        gop_entries.iter().filter(|e| e.frame_role == 1).copied().collect();

    if hp_entries.is_empty() {
        return crate::TemporalGroup {
            low_frame,
            high_frames: Vec::new(),
        };
    }

    // Find the number of temporal levels
    let max_level = hp_entries.iter().map(|e| e.temporal_level).max().unwrap();
    let num_levels = max_level as usize + 1;

    // Build high_frames: Vec<Vec<CompressedFrame>> indexed by level (0=finest)
    let mut high_frames: Vec<Vec<crate::CompressedFrame>> = vec![Vec::new(); num_levels];

    for entry in &hp_entries {
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        assert!(
            end <= data.len(),
            "Highpass frame data extends beyond file (GOP {}, level {}, offset {}..{})",
            group_idx,
            entry.temporal_level,
            start,
            end
        );
        let frame = deserialize_compressed(&data[start..end]);
        high_frames[entry.temporal_level as usize].push(frame);
    }

    crate::TemporalGroup {
        low_frame,
        high_frames,
    }
}

/// Convenience: deserialize an entire GNV2 container into a `TemporalEncodedSequence`.
///
/// Reads all GOPs and tail I-frames.
pub fn deserialize_temporal_sequence(data: &[u8]) -> crate::TemporalEncodedSequence {
    let header = deserialize_temporal_sequence_header(data);

    // Collect actual temporal GOP indices from the index (lowpass entries, sorted).
    let mut temporal_gop_set: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for entry in &header.index {
        if entry.frame_role == 0 {
            temporal_gop_set.insert(entry.gop_index as usize);
        }
    }
    let temporal_gop_indices: Vec<usize> = temporal_gop_set.into_iter().collect();

    let mut groups = Vec::with_capacity(temporal_gop_indices.len());
    let mut group_gop_indices = Vec::with_capacity(temporal_gop_indices.len());
    for &gop_idx in &temporal_gop_indices {
        groups.push(deserialize_temporal_group(data, &header, gop_idx));
        group_gop_indices.push(gop_idx);
    }

    // Deserialize tail I-frames, sorted by PTS (handles scene-cut I-frames at arbitrary PTS).
    let mut tail_entries: Vec<_> = header.index.iter().filter(|e| e.frame_role == 2).collect();
    tail_entries.sort_by_key(|e| e.pts);
    let mut tail_iframes = Vec::with_capacity(tail_entries.len());
    let mut tail_iframe_pts = Vec::with_capacity(tail_entries.len());
    for entry in &tail_entries {
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        assert!(end <= data.len(), "Tail I-frame data extends beyond file");
        tail_iframes.push(deserialize_compressed(&data[start..end]));
        tail_iframe_pts.push(entry.pts);
    }

    crate::TemporalEncodedSequence {
        mode: header.temporal_transform,
        groups,
        group_gop_indices,
        tail_iframes,
        tail_iframe_pts,
        frame_count: header.frame_count as usize,
        gop_size: header.gop_size as usize,
    }
}

/// Find the GOP containing a target PTS for keyframe seeking.
///
/// Returns the GOP index. The lowpass frame of that GOP is the seekable entry point.
/// To decode frame at target_pts, decode the entire GOP and extract the frame.
pub fn seek_to_temporal_keyframe(header: &TemporalSequenceHeader, target_pts: u32) -> usize {
    // Each GOP covers gop_size frames: GOP g covers PTS [g*gop_size, (g+1)*gop_size)
    let gop_size = header.gop_size as u32;
    if gop_size == 0 {
        return 0;
    }
    let num_groups = header.num_groups();
    let gop_idx = (target_pts / gop_size) as usize;
    gop_idx.min(num_groups.saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::rice;

    /// Create a minimal CompressedFrame with trivial Rice entropy data.
    /// The frame is small (16x16) with a single tile containing all-zero coefficients.
    fn make_test_frame(width: u32, height: u32) -> crate::CompressedFrame {
        let tile_size = 256u32;
        let num_levels = 3u32;
        let num_groups = num_levels * 2; // directional subband splitting: 2 groups/level
        let tiles_x = width.div_ceil(tile_size);
        let tiles_y = height.div_ceil(tile_size);
        let num_tiles = (tiles_x * tiles_y * 3) as usize; // 3 planes

        let tile = rice::RiceTile {
            num_coefficients: tile_size * tile_size,
            tile_size,
            num_levels,
            num_groups,
            k_values: vec![0; num_groups as usize],
            k_zrl_nz_values: vec![0; num_groups as usize],
            k_zrl_z_values: vec![0; num_groups as usize],
            skip_bitmap: 0xFF, // all groups skipped (all zeros)
            stream_lengths: vec![0; rice::RICE_STREAMS_PER_TILE],
            stream_data: Vec::new(),
        };

        let tiles = vec![tile; num_tiles];

        crate::CompressedFrame {
            info: crate::FrameInfo {
                width,
                height,
                bit_depth: 8,
                tile_size,
                chroma_format: crate::ChromaFormat::Yuv444,
            },
            config: crate::CodecConfig {
                tile_size,
                quantization_step: 4.0,
                dead_zone: 0.5,
                wavelet_levels: num_levels,
                subband_weights: crate::SubbandWeights::uniform(num_levels),
                entropy_coder: crate::EntropyCoder::Rice,
                temporal_highpass_qstep_mul: 2.0,
                ..Default::default()
            },
            entropy: crate::EntropyData::Rice(tiles),
            cfl_alphas: None,
            weight_map: None,
            frame_type: crate::FrameType::Intra,
            motion_field: None,
            intra_modes: None,
            residual_stats: None,
            residual_stats_co: None,
            residual_stats_cg: None,
        }
    }

    /// Test GNV2 roundtrip: serialize a TemporalEncodedSequence, deserialize it,
    /// and verify all structure and frame data matches.
    #[test]
    fn test_gnv2_roundtrip() {
        let width = 64;
        let height = 64;
        let gop_size = 4; // 2 levels: level 0 has 2 HP frames, level 1 has 1 HP frame

        // Build a TemporalEncodedSequence with 2 GOPs + 1 tail I-frame
        // GOP layout for gop_size=4:
        //   1 lowpass + 1 highpass(level 1) + 2 highpass(level 0) = 4 frames/GOP
        let make_group = || crate::TemporalGroup {
            low_frame: make_test_frame(width, height),
            high_frames: vec![
                // level 0 (finest): 2 frames
                vec![make_test_frame(width, height), make_test_frame(width, height)],
                // level 1 (deepest): 1 frame
                vec![make_test_frame(width, height)],
            ],
        };

        let seq = crate::TemporalEncodedSequence {
            mode: crate::TemporalTransform::Haar,
            groups: vec![make_group(), make_group()],
            group_gop_indices: vec![0, 1],
            tail_iframes: vec![make_test_frame(width, height)],
            tail_iframe_pts: vec![8],
            frame_count: 9, // 2 GOPs × 4 + 1 tail
            gop_size,
        };

        let framerate = (30000u32, 1001u32);

        // Serialize
        let data = serialize_temporal_sequence(&seq, framerate);

        // Verify magic
        assert_eq!(&data[0..4], b"GNV2");

        // Deserialize header
        let header = deserialize_temporal_sequence_header(&data);
        assert_eq!(header.version, 1);
        assert_eq!(header.width, width);
        assert_eq!(header.height, height);
        assert_eq!(header.frame_count, 9);
        assert_eq!(header.framerate_num, 30000);
        assert_eq!(header.framerate_den, 1001);
        assert_eq!(header.temporal_transform, crate::TemporalTransform::Haar);
        assert_eq!(header.gop_size, 4);
        assert!((header.highpass_qstep_mul - 2.0).abs() < f32::EPSILON);
        assert_eq!(header.num_groups(), 2);
        assert_eq!(header.num_tail_iframes(), 1);

        // Verify index table structure
        // Each GOP: 1 lowpass + 1 hp(L1) + 2 hp(L0) = 4 entries
        // 2 GOPs + 1 tail = 9 entries total
        assert_eq!(header.index.len(), 9);

        // Check GOP 0 entries
        let gop0: Vec<_> = header
            .index
            .iter()
            .filter(|e| e.gop_index == 0 && e.frame_role != 2)
            .collect();
        assert_eq!(gop0.len(), 4);
        assert_eq!(gop0[0].frame_role, 0); // lowpass
        assert_eq!(gop0[1].frame_role, 1); // highpass
        assert_eq!(gop0[1].temporal_level, 1); // deepest level first in file
        assert_eq!(gop0[2].frame_role, 1);
        assert_eq!(gop0[2].temporal_level, 0); // finest level
        assert_eq!(gop0[3].frame_role, 1);
        assert_eq!(gop0[3].temporal_level, 0);

        // Check tail entry
        let tail: Vec<_> = header.index.iter().filter(|e| e.frame_role == 2).collect();
        assert_eq!(tail.len(), 1);

        // Deserialize individual group
        let group0 = deserialize_temporal_group(&data, &header, 0);
        assert_eq!(group0.low_frame.info.width, width);
        assert_eq!(group0.low_frame.info.height, height);
        assert_eq!(group0.high_frames.len(), 2); // 2 levels
        assert_eq!(group0.high_frames[0].len(), 2); // level 0: 2 frames
        assert_eq!(group0.high_frames[1].len(), 1); // level 1: 1 frame

        // Deserialize full sequence
        let decoded = deserialize_temporal_sequence(&data);
        assert_eq!(decoded.mode, crate::TemporalTransform::Haar);
        assert_eq!(decoded.groups.len(), 2);
        assert_eq!(decoded.tail_iframes.len(), 1);
        assert_eq!(decoded.frame_count, 9);
        assert_eq!(decoded.gop_size, 4);

        // Verify all frames roundtrip: re-serialize each frame and compare bytes
        for (gi, group) in decoded.groups.iter().enumerate() {
            let orig_group = &seq.groups[gi];
            let lp_bytes = serialize_compressed(&group.low_frame);
            let orig_lp_bytes = serialize_compressed(&orig_group.low_frame);
            assert_eq!(
                lp_bytes, orig_lp_bytes,
                "Lowpass frame mismatch in GOP {gi}"
            );

            for (li, level_frames) in group.high_frames.iter().enumerate() {
                for (fi, frame) in level_frames.iter().enumerate() {
                    let hp_bytes = serialize_compressed(frame);
                    let orig_hp_bytes =
                        serialize_compressed(&orig_group.high_frames[li][fi]);
                    assert_eq!(
                        hp_bytes, orig_hp_bytes,
                        "Highpass frame mismatch in GOP {gi}, level {li}, frame {fi}"
                    );
                }
            }
        }

        // Verify tail I-frame roundtrip
        for (ti, tail_frame) in decoded.tail_iframes.iter().enumerate() {
            let t_bytes = serialize_compressed(tail_frame);
            let orig_t_bytes = serialize_compressed(&seq.tail_iframes[ti]);
            assert_eq!(t_bytes, orig_t_bytes, "Tail I-frame {ti} mismatch");
        }
    }

    /// Test GNV2 with a single GOP of size 2 (simplest case: 1 level).
    #[test]
    fn test_gnv2_gop2_roundtrip() {
        let width = 32;
        let height = 32;

        let seq = crate::TemporalEncodedSequence {
            mode: crate::TemporalTransform::Haar,
            groups: vec![crate::TemporalGroup {
                low_frame: make_test_frame(width, height),
                high_frames: vec![
                    // level 0: 1 frame
                    vec![make_test_frame(width, height)],
                ],
            }],
            group_gop_indices: vec![0],
            tail_iframes: Vec::new(),
            tail_iframe_pts: Vec::new(),
            frame_count: 2,
            gop_size: 2,
        };

        let data = serialize_temporal_sequence(&seq, (24, 1));
        let header = deserialize_temporal_sequence_header(&data);
        assert_eq!(header.gop_size, 2);
        assert_eq!(header.index.len(), 2); // 1 lowpass + 1 highpass
        assert_eq!(header.num_groups(), 1);
        assert_eq!(header.num_tail_iframes(), 0);

        let decoded = deserialize_temporal_sequence(&data);
        assert_eq!(decoded.groups.len(), 1);
        assert_eq!(decoded.groups[0].high_frames.len(), 1);
        assert_eq!(decoded.groups[0].high_frames[0].len(), 1);
        assert!(decoded.tail_iframes.is_empty());
    }

    /// Test GNV2 with GOP size 8 (3 levels).
    #[test]
    fn test_gnv2_gop8_roundtrip() {
        let width = 64;
        let height = 64;

        // GOP size 8: 3 levels
        // level 2 (deepest): 1 HP frame
        // level 1: 2 HP frames
        // level 0 (finest): 4 HP frames
        // total: 1 LP + 1 + 2 + 4 = 8
        let seq = crate::TemporalEncodedSequence {
            mode: crate::TemporalTransform::Haar,
            groups: vec![crate::TemporalGroup {
                low_frame: make_test_frame(width, height),
                high_frames: vec![
                    // level 0: 4 frames
                    vec![
                        make_test_frame(width, height),
                        make_test_frame(width, height),
                        make_test_frame(width, height),
                        make_test_frame(width, height),
                    ],
                    // level 1: 2 frames
                    vec![
                        make_test_frame(width, height),
                        make_test_frame(width, height),
                    ],
                    // level 2: 1 frame
                    vec![make_test_frame(width, height)],
                ],
            }],
            group_gop_indices: vec![0],
            tail_iframes: Vec::new(),
            tail_iframe_pts: Vec::new(),
            frame_count: 8,
            gop_size: 8,
        };

        let data = serialize_temporal_sequence(&seq, (30, 1));
        let header = deserialize_temporal_sequence_header(&data);
        assert_eq!(header.gop_size, 8);
        assert_eq!(header.index.len(), 8); // 1 + 1 + 2 + 4

        // Verify ordering: LP, then L2, then L1×2, then L0×4
        assert_eq!(header.index[0].frame_role, 0); // lowpass
        assert_eq!(header.index[1].frame_role, 1);
        assert_eq!(header.index[1].temporal_level, 2); // deepest first
        assert_eq!(header.index[2].frame_role, 1);
        assert_eq!(header.index[2].temporal_level, 1);
        assert_eq!(header.index[3].frame_role, 1);
        assert_eq!(header.index[3].temporal_level, 1);
        assert_eq!(header.index[4].frame_role, 1);
        assert_eq!(header.index[4].temporal_level, 0); // finest last
        assert_eq!(header.index[7].temporal_level, 0);

        let decoded = deserialize_temporal_sequence(&data);
        assert_eq!(decoded.groups[0].high_frames.len(), 3);
        assert_eq!(decoded.groups[0].high_frames[0].len(), 4); // level 0
        assert_eq!(decoded.groups[0].high_frames[1].len(), 2); // level 1
        assert_eq!(decoded.groups[0].high_frames[2].len(), 1); // level 2
    }

    /// Test GNV2 header helper methods.
    #[test]
    fn test_gnv2_header_helpers() {
        let header = TemporalSequenceHeader {
            version: 1,
            width: 1920,
            height: 1080,
            frame_count: 240,
            framerate_num: 30000,
            framerate_den: 1001,
            temporal_transform: crate::TemporalTransform::Haar,
            gop_size: 8,
            highpass_qstep_mul: 2.0,
            index: Vec::new(),
        };
        let fps = header.fps();
        assert!((fps - 29.97).abs() < 0.01);
        let dur = header.duration_secs();
        assert!((dur - 8.008).abs() < 0.01);
    }
}
