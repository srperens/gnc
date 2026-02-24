/// Bitplane entropy coder — GPU-friendly alternative to rANS.
///
/// Bitplane coding decomposes each coefficient into sign + magnitude bits,
/// then encodes bitplanes from MSB to LSB within 32x32 blocks. Each bitplane
/// can be skipped entirely if all-zero (common in high-frequency subbands),
/// saving 1024 bits per skipped plane. Sign bits are emitted only once per
/// coefficient, at the bitplane where it first becomes significant.
///
/// This approach is fully GPU-parallelizable on the decode side: one workgroup
/// per 32x32 block, one thread per coefficient. No serial dependencies.
/// Patent-free: bitplane coding predates all modern codec patents.
use crate::{FrameInfo, GpuContext};
use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

/// Block size for bitplane coding (32x32 = 1024 coefficients)
const BLOCK_SIZE: usize = 32;
const COEFFS_PER_BLOCK: usize = BLOCK_SIZE * BLOCK_SIZE;

/// A bitplane-coded tile (256x256), composed of 8x8 = 64 blocks of 32x32.
#[derive(Debug, Clone)]
pub struct BitplaneTile {
    /// Per-block encoded data, concatenated.
    pub block_data: Vec<u8>,
    /// Byte offset where each block's data starts within `block_data`.
    pub block_offsets: Vec<u32>,
    /// Number of coefficients in the tile (tile_size * tile_size).
    pub num_coefficients: u32,
    /// Tile size (e.g. 256).
    pub tile_size: u32,
}

impl BitplaneTile {
    /// Total compressed size in bytes.
    pub fn byte_size(&self) -> usize {
        // data + offsets table (4 bytes each) + header (num_coefficients, tile_size, num_blocks)
        self.block_data.len() + self.block_offsets.len() * 4 + 12
    }
}

/// Encode a tile's quantized coefficients using bitplane coding.
///
/// Input: i32 coefficients in raster order within the tile (tile_size x tile_size).
/// The tile is divided into 32x32 blocks. Each block is independently coded.
pub fn bitplane_encode_tile(coefficients: &[i32], tile_size: u32) -> BitplaneTile {
    let ts = tile_size as usize;
    assert_eq!(
        coefficients.len(),
        ts * ts,
        "Coefficient count must match tile_size^2"
    );

    let blocks_per_side = ts / BLOCK_SIZE;
    let num_blocks = blocks_per_side * blocks_per_side;

    let mut block_data = Vec::new();
    let mut block_offsets = Vec::with_capacity(num_blocks);

    for by in 0..blocks_per_side {
        for bx in 0..blocks_per_side {
            block_offsets.push(block_data.len() as u32);
            encode_block(coefficients, ts, bx, by, &mut block_data);
        }
    }

    BitplaneTile {
        block_data,
        block_offsets,
        num_coefficients: (ts * ts) as u32,
        tile_size,
    }
}

/// Encode a single 32x32 block using bitplane coding.
///
/// Format:
///   [max_bitplane: u8]
///   For each bitplane p from (max_bitplane-1) down to 0:
///     [all_zero_flag: 1 bit]
///     If not all-zero:
///       [significance_bitmap: 1024 bits = 128 bytes]
///   [sign_bits: N bits, one per non-zero coefficient]
///
/// Bits are packed into bytes, MSB first within each byte.
fn encode_block(coefficients: &[i32], tile_width: usize, bx: usize, by: usize, out: &mut Vec<u8>) {
    // Extract block coefficients and convert to (sign, magnitude)
    let mut magnitudes = [0u32; COEFFS_PER_BLOCK];
    let mut signs = [0u8; COEFFS_PER_BLOCK]; // 0 = positive or zero, 1 = negative

    let block_origin_x = bx * BLOCK_SIZE;
    let block_origin_y = by * BLOCK_SIZE;

    let mut max_mag: u32 = 0;

    for ly in 0..BLOCK_SIZE {
        for lx in 0..BLOCK_SIZE {
            let idx = (block_origin_y + ly) * tile_width + (block_origin_x + lx);
            let coeff = coefficients[idx];
            let mag = coeff.unsigned_abs();
            let local = ly * BLOCK_SIZE + lx;
            magnitudes[local] = mag;
            signs[local] = if coeff < 0 { 1 } else { 0 };
            if mag > max_mag {
                max_mag = mag;
            }
        }
    }

    // Determine number of bitplanes needed
    let max_bitplane = if max_mag == 0 {
        0
    } else {
        32 - max_mag.leading_zeros()
    } as u8;

    out.push(max_bitplane);

    if max_bitplane == 0 {
        // All zeros — no bitplane data, no sign bits
        return;
    }

    // Use a bit writer for the variable-length bitplane + sign data
    let mut writer = BitWriter::new();

    // Encode bitplanes from MSB to LSB
    for p in (0..max_bitplane).rev() {
        let bit_mask = 1u32 << p;

        // Check if entire bitplane is all-zero
        let mut all_zero = true;
        for &mag in &magnitudes {
            if mag & bit_mask != 0 {
                all_zero = false;
                break;
            }
        }

        if all_zero {
            writer.write_bit(1); // all-zero flag = 1
        } else {
            writer.write_bit(0); // all-zero flag = 0
                                 // Write significance bitmap: 1024 bits, one per coefficient
            for &mag in &magnitudes {
                writer.write_bit(if mag & bit_mask != 0 { 1 } else { 0 });
            }
        }
    }

    // Write sign bits for all non-zero coefficients (in raster order)
    for i in 0..COEFFS_PER_BLOCK {
        if magnitudes[i] > 0 {
            writer.write_bit(signs[i]);
        }
    }

    writer.flush_to(out);
}

/// Decode a bitplane-coded tile back to i32 coefficients.
///
/// Returns coefficients in raster order within the tile (tile_size x tile_size).
pub fn bitplane_decode_tile(tile: &BitplaneTile) -> Vec<i32> {
    let ts = tile.tile_size as usize;
    let blocks_per_side = ts / BLOCK_SIZE;
    let num_blocks = blocks_per_side * blocks_per_side;

    let mut output = vec![0i32; ts * ts];

    for block_idx in 0..num_blocks {
        let bx = block_idx % blocks_per_side;
        let by = block_idx / blocks_per_side;
        let data_start = tile.block_offsets[block_idx] as usize;
        let data_end = if block_idx + 1 < num_blocks {
            tile.block_offsets[block_idx + 1] as usize
        } else {
            tile.block_data.len()
        };
        decode_block(
            &tile.block_data[data_start..data_end],
            &mut output,
            ts,
            bx,
            by,
        );
    }

    output
}

/// Decode a single 32x32 block from bitplane data.
fn decode_block(data: &[u8], output: &mut [i32], tile_width: usize, bx: usize, by: usize) {
    if data.is_empty() {
        return;
    }

    let max_bitplane = data[0];

    if max_bitplane == 0 {
        // All zeros — output is already initialized to 0
        return;
    }

    let mut reader = BitReader::new(&data[1..]);
    let mut magnitudes = [0u32; COEFFS_PER_BLOCK];

    // Read bitplanes from MSB to LSB
    for p in (0..max_bitplane).rev() {
        let all_zero = reader.read_bit();
        if all_zero == 1 {
            continue; // skip this bitplane
        }
        // Read significance bitmap
        let bit_val = 1u32 << p;
        for mag in magnitudes.iter_mut() {
            if reader.read_bit() == 1 {
                *mag |= bit_val;
            }
        }
    }

    // Read sign bits for non-zero coefficients
    let block_origin_x = bx * BLOCK_SIZE;
    let block_origin_y = by * BLOCK_SIZE;

    for (i, &mag) in magnitudes.iter().enumerate() {
        let ly = i / BLOCK_SIZE;
        let lx = i % BLOCK_SIZE;
        let out_idx = (block_origin_y + ly) * tile_width + (block_origin_x + lx);

        if mag > 0 {
            let sign = reader.read_bit();
            output[out_idx] = if sign == 1 { -(mag as i32) } else { mag as i32 };
        }
        // else: output already 0
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/// Serialize a BitplaneTile to bytes.
pub fn serialize_tile_bitplane(tile: &BitplaneTile) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&tile.num_coefficients.to_le_bytes());
    out.extend_from_slice(&tile.tile_size.to_le_bytes());
    let num_blocks = tile.block_offsets.len() as u32;
    out.extend_from_slice(&num_blocks.to_le_bytes());
    // Block offsets
    for &off in &tile.block_offsets {
        out.extend_from_slice(&off.to_le_bytes());
    }
    // Block data length + data
    let data_len = tile.block_data.len() as u32;
    out.extend_from_slice(&data_len.to_le_bytes());
    out.extend_from_slice(&tile.block_data);
    out
}

/// Deserialize a BitplaneTile from bytes. Returns (tile, bytes_consumed).
pub fn deserialize_tile_bitplane(data: &[u8]) -> (BitplaneTile, usize) {
    let mut pos = 0;
    let num_coefficients = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let tile_size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    pos += 4;
    let num_blocks = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;

    let mut block_offsets = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        let off = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        block_offsets.push(off);
        pos += 4;
    }

    let data_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let block_data = data[pos..pos + data_len].to_vec();
    pos += data_len;

    (
        BitplaneTile {
            block_data,
            block_offsets,
            num_coefficients,
            tile_size,
        },
        pos,
    )
}

// ---------------------------------------------------------------------------
// GPU decode support
// ---------------------------------------------------------------------------

/// Per-tile info stride in u32s for the GPU bitplane decode shader.
/// Layout: [num_blocks, block_info_offset, plane_tile_x, plane_tile_y, padding...]
pub(crate) const BITPLANE_TILE_INFO_STRIDE: usize = 8;

/// Per-block info stride in u32s.
/// Layout: [max_bitplane, data_byte_offset, padding...]
pub(crate) const BITPLANE_BLOCK_INFO_STRIDE: usize = 4;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct BitplaneDecodeParams {
    num_tiles: u32,
    coefficients_per_tile: u32,
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    block_size: u32,
    blocks_per_tile_side: u32,
    _pad: u32,
}

/// CPU-packed bitplane plane data, ready for writing into pre-allocated GPU buffers.
pub(crate) struct PackedBitplanePlane {
    pub params: BitplaneDecodeParams,
    pub tile_info: Vec<u32>,
    pub block_info: Vec<u32>,
    pub bitplane_data: Vec<u32>,
}

/// GPU bitplane decoder.
pub struct GpuBitplaneDecoder {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuBitplaneDecoder {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bitplane_decode"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/bitplane_decode.wgsl").into(),
                ),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("bitplane_decode_bgl"),
                    entries: &[
                        // 0: uniform params
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 1: tile_info (storage, read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 2: block_info (storage, read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 3: bitplane_data (storage, read) — packed u32
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 4: output (storage, read_write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bitplane_decode_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bitplane_decode_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Pack bitplane tile data into CPU arrays for one plane.
    pub fn pack_decode_data(tiles: &[BitplaneTile], info: &FrameInfo) -> PackedBitplanePlane {
        let num_tiles = tiles.len();
        let tile_size = info.tile_size;
        let coefficients_per_tile = tile_size * tile_size;
        let padded_w = info.padded_width();
        let tiles_x = info.tiles_x();
        let blocks_per_side = tile_size as usize / BLOCK_SIZE;
        let blocks_per_tile = blocks_per_side * blocks_per_side;

        let params = BitplaneDecodeParams {
            num_tiles: num_tiles as u32,
            coefficients_per_tile,
            plane_width: padded_w,
            tile_size,
            tiles_x,
            block_size: BLOCK_SIZE as u32,
            blocks_per_tile_side: blocks_per_side as u32,
            _pad: 0,
        };

        let total_blocks = num_tiles * blocks_per_tile;
        let mut tile_info_data = vec![0u32; num_tiles * BITPLANE_TILE_INFO_STRIDE];
        let mut block_info_data = vec![0u32; total_blocks * BITPLANE_BLOCK_INFO_STRIDE];
        let mut all_bitplane_bytes: Vec<u8> = Vec::new();

        let mut global_block_idx = 0usize;

        for (t, tile) in tiles.iter().enumerate() {
            let tile_x = t % tiles_x as usize;
            let tile_y = t / tiles_x as usize;
            let base = t * BITPLANE_TILE_INFO_STRIDE;

            tile_info_data[base] = blocks_per_tile as u32;
            tile_info_data[base + 1] = global_block_idx as u32;
            tile_info_data[base + 2] = tile_x as u32;
            tile_info_data[base + 3] = tile_y as u32;

            let num_blocks_in_tile = tile.block_offsets.len();
            for b in 0..num_blocks_in_tile {
                let block_base = global_block_idx * BITPLANE_BLOCK_INFO_STRIDE;

                let data_start = tile.block_offsets[b] as usize;
                let data_end = if b + 1 < num_blocks_in_tile {
                    tile.block_offsets[b + 1] as usize
                } else {
                    tile.block_data.len()
                };
                let block_slice = &tile.block_data[data_start..data_end];

                // max_bitplane is the first byte
                let max_bitplane = if block_slice.is_empty() {
                    0
                } else {
                    block_slice[0]
                };
                block_info_data[block_base] = max_bitplane as u32;

                // Byte offset of the bitplane data (after max_bitplane byte) in the global buffer
                let bitplane_data_offset = all_bitplane_bytes.len() as u32;
                block_info_data[block_base + 1] = bitplane_data_offset;

                // Append the bitplane data (everything after the max_bitplane byte)
                if block_slice.len() > 1 {
                    all_bitplane_bytes.extend_from_slice(&block_slice[1..]);
                }

                global_block_idx += 1;
            }
        }

        // Pad to u32 boundary and pack
        while all_bitplane_bytes.len() % 4 != 0 {
            all_bitplane_bytes.push(0);
        }
        let bitplane_data_u32: Vec<u32> = all_bitplane_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // wgpu requires non-zero buffer sizes
        let bitplane_data = if bitplane_data_u32.is_empty() {
            vec![0u32]
        } else {
            bitplane_data_u32
        };

        PackedBitplanePlane {
            params,
            tile_info: tile_info_data,
            block_info: block_info_data,
            bitplane_data,
        }
    }

    /// Pack bitplane tiles into GPU buffers for a single plane.
    /// Returns (params_buf, tile_info_buf, block_info_buf, bitplane_data_buf).
    pub fn prepare_decode_buffers(
        &self,
        ctx: &GpuContext,
        tiles: &[BitplaneTile],
        info: &FrameInfo,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let packed = Self::pack_decode_data(tiles, info);

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bitplane_decode_params"),
                contents: bytemuck::bytes_of(&packed.params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let tile_info_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bitplane_tile_info"),
                contents: bytemuck::cast_slice(&packed.tile_info),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let block_info_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bitplane_block_info"),
                contents: bytemuck::cast_slice(&packed.block_info),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let bitplane_data_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bitplane_data"),
                contents: bytemuck::cast_slice(&packed.bitplane_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        (params_buf, tile_info_buf, block_info_buf, bitplane_data_buf)
    }

    /// Dispatch GPU bitplane decode for one plane.
    /// Uses one workgroup per block (not per tile), since each 32x32 block is independent.
    pub fn dispatch_decode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        params_buf: &wgpu::Buffer,
        tile_info_buf: &wgpu::Buffer,
        block_info_buf: &wgpu::Buffer,
        bitplane_data_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        total_blocks: u32,
    ) {
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitplane_decode_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tile_info_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: block_info_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bitplane_data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bitplane_decode_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(total_blocks, 1, 1);
    }
}

// ---------------------------------------------------------------------------
// Bit-level I/O helpers
// ---------------------------------------------------------------------------

/// Simple MSB-first bit writer that packs bits into bytes.
struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    bits_in_current: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            bits_in_current: 0,
        }
    }

    /// Write a single bit (0 or 1), MSB-first packing.
    fn write_bit(&mut self, bit: u8) {
        self.current = (self.current << 1) | (bit & 1);
        self.bits_in_current += 1;
        if self.bits_in_current == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.bits_in_current = 0;
        }
    }

    /// Flush remaining bits (padded with zeros on the right) and append to output.
    fn flush_to(mut self, out: &mut Vec<u8>) {
        if self.bits_in_current > 0 {
            // Pad remaining bits to fill the byte (shift left)
            self.current <<= 8 - self.bits_in_current;
            self.bytes.push(self.current);
        }
        out.extend_from_slice(&self.bytes);
    }
}

/// Simple MSB-first bit reader.
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8, // 0..8, next bit to read within current byte
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read a single bit (0 or 1), MSB-first.
    fn read_bit(&mut self) -> u8 {
        if self.byte_pos >= self.data.len() {
            return 0; // past end, return 0
        }
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        bit
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitplane_roundtrip_simple() {
        let tile_size = 32u32; // Smallest possible: one block
        let mut coefficients = vec![0i32; (tile_size * tile_size) as usize];
        coefficients[0] = 5;
        coefficients[1] = -3;
        coefficients[100] = 15;
        coefficients[500] = -1;
        coefficients[1023] = 7;

        let tile = bitplane_encode_tile(&coefficients, tile_size);
        let decoded = bitplane_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_bitplane_roundtrip_zeros() {
        let tile_size = 32u32;
        let coefficients = vec![0i32; (tile_size * tile_size) as usize];
        let tile = bitplane_encode_tile(&coefficients, tile_size);
        let decoded = bitplane_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
        // All zeros should be very compact: just 1 byte (max_bitplane=0) per block
        assert!(tile.block_data.len() <= 4, "All-zero block should be tiny");
    }

    #[test]
    fn test_bitplane_roundtrip_256_tile() {
        let tile_size = 256u32;
        let n = (tile_size * tile_size) as usize;
        let mut coefficients = vec![0i32; n];
        // Simulate wavelet-like distribution: mostly zeros, some small values
        for i in 0..n {
            coefficients[i] = if i % 7 == 0 {
                (i % 50) as i32 - 25
            } else if i % 3 == 0 {
                (i % 10) as i32 - 5
            } else {
                0
            };
        }
        let tile = bitplane_encode_tile(&coefficients, tile_size);
        let decoded = bitplane_decode_tile(&tile);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_bitplane_smaller_than_raw() {
        let tile_size = 256u32;
        let n = (tile_size * tile_size) as usize;
        // Mostly zeros with a few non-zero values — common after wavelet + quantization
        let mut coefficients = vec![0i32; n];
        for i in (0..n).step_by(100) {
            coefficients[i] = (i % 5) as i32 - 2;
        }
        let tile = bitplane_encode_tile(&coefficients, tile_size);
        let raw_size = n * 2; // 16 bits per coefficient
        let bitplane_size = tile.byte_size();
        assert!(
            bitplane_size < raw_size,
            "Bitplane size {} should be less than raw i16 size {}",
            bitplane_size,
            raw_size
        );
    }

    #[test]
    fn test_bitplane_serialize_roundtrip() {
        let tile_size = 256u32;
        let n = (tile_size * tile_size) as usize;
        let mut coefficients = vec![0i32; n];
        for i in 0..n {
            coefficients[i] = if i % 4 == 0 { (i % 10) as i32 - 5 } else { 0 };
        }
        let tile = bitplane_encode_tile(&coefficients, tile_size);
        let serialized = serialize_tile_bitplane(&tile);
        let (deserialized, consumed) = deserialize_tile_bitplane(&serialized);
        assert_eq!(consumed, serialized.len());
        let decoded = bitplane_decode_tile(&deserialized);
        assert_eq!(coefficients, decoded);
    }

    #[test]
    fn test_bit_writer_reader_roundtrip() {
        let mut writer = BitWriter::new();
        let bits = [1u8, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1];
        for &b in &bits {
            writer.write_bit(b);
        }
        let mut output = Vec::new();
        writer.flush_to(&mut output);

        let mut reader = BitReader::new(&output);
        for &expected in &bits {
            assert_eq!(reader.read_bit(), expected);
        }
    }
}
