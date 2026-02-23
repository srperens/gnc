/// GPU rANS decoder — dispatches the rans_decode.wgsl compute shader.
///
/// Packs interleaved tile data into GPU buffers and decodes all tiles
/// for a plane in a single dispatch. Output stays on GPU as f32 coefficients
/// ready for dequantization, eliminating the CPU↔GPU round-trip.
use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use super::rans::{InterleavedRansTile, STREAMS_PER_TILE};
use crate::{FrameInfo, GpuContext};

const TILE_INFO_STRIDE: usize = 100;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RansDecodeParams {
    num_tiles: u32,
    coefficients_per_tile: u32,
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct GpuRansDecoder {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuRansDecoder {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rans_decode"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/rans_decode.wgsl").into(),
                ),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("rans_decode_bgl"),
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
                        // 2: cumfreq_data (storage, read)
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
                        // 3: stream_data (storage, read)
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
                label: Some("rans_decode_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rans_decode_pipeline"),
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

    /// Pack interleaved tiles into GPU buffers for a single plane.
    /// Returns (params_buf, tile_info_buf, cumfreq_buf, stream_data_buf).
    pub fn prepare_decode_buffers(
        &self,
        ctx: &GpuContext,
        tiles: &[InterleavedRansTile],
        info: &FrameInfo,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let num_tiles = tiles.len();
        let tile_size = info.tile_size;
        let coefficients_per_tile = tile_size * tile_size;
        let padded_w = info.padded_width();
        let tiles_x = info.tiles_x();

        let params = RansDecodeParams {
            num_tiles: num_tiles as u32,
            coefficients_per_tile,
            plane_width: padded_w,
            tile_size,
            tiles_x,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rans_decode_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Build packed buffers
        let mut tile_info_data = vec![0u32; num_tiles * TILE_INFO_STRIDE];
        let mut cumfreq_all: Vec<u32> = Vec::new();
        let mut stream_bytes_all: Vec<u8> = Vec::new();

        for (t, tile) in tiles.iter().enumerate() {
            assert!(
                tile.alphabet_size <= 2048,
                "Alphabet size {} exceeds GPU max 2048",
                tile.alphabet_size
            );

            let base = t * TILE_INFO_STRIDE;

            // [0]: min_val (bit-reinterpret i32 → u32)
            tile_info_data[base] = tile.min_val as u32;
            // [1]: alphabet_size
            tile_info_data[base + 1] = tile.alphabet_size;
            // [2]: cumfreq_offset
            tile_info_data[base + 2] = cumfreq_all.len() as u32;

            cumfreq_all.extend_from_slice(&tile.cumfreqs);

            // [3]: zrun_base (0 = no ZRL)
            tile_info_data[base + 3] = tile.zrun_base as u32;

            // [4]: stream_data_byte_base
            let tile_stream_byte_base = stream_bytes_all.len() as u32;
            tile_info_data[base + 4] = tile_stream_byte_base;

            // Per-stream info (shifted +1 for zrun_base field)
            let mut stream_byte_offset = 0u32;
            for s in 0..STREAMS_PER_TILE {
                // [5+s]: initial_state
                tile_info_data[base + 5 + s] = tile.stream_initial_state[s];
                // [37+s]: byte offset within tile's stream region
                tile_info_data[base + 37 + s] = stream_byte_offset;

                stream_byte_offset += tile.stream_data[s].len() as u32;
            }

            // Append all stream bytes for this tile
            for s in 0..STREAMS_PER_TILE {
                stream_bytes_all.extend_from_slice(&tile.stream_data[s]);
            }
        }

        // Pad to u32 boundary and pack as u32 array (little-endian)
        while !stream_bytes_all.len().is_multiple_of(4) {
            stream_bytes_all.push(0);
        }
        let stream_data_u32: Vec<u32> = stream_bytes_all
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // wgpu requires non-zero buffer sizes
        if cumfreq_all.is_empty() {
            cumfreq_all.push(0);
        }
        let stream_data_final = if stream_data_u32.is_empty() {
            vec![0u32]
        } else {
            stream_data_u32
        };

        let tile_info_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rans_tile_info"),
                contents: bytemuck::cast_slice(&tile_info_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let cumfreq_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rans_cumfreq"),
                contents: bytemuck::cast_slice(&cumfreq_all),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let stream_data_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rans_stream_data"),
                contents: bytemuck::cast_slice(&stream_data_final),
                usage: wgpu::BufferUsages::STORAGE,
            });

        (params_buf, tile_info_buf, cumfreq_buf, stream_data_buf)
    }

    /// Dispatch GPU rANS decode for one plane. Output is written directly
    /// to `output_buf` as f32 coefficients at the correct plane positions.
    pub fn dispatch_decode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        params_buf: &wgpu::Buffer,
        tile_info_buf: &wgpu::Buffer,
        cumfreq_buf: &wgpu::Buffer,
        stream_data_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_tiles: u32,
    ) {
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rans_decode_bg"),
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
                    resource: cumfreq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: stream_data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rans_decode_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_tiles, 1, 1);
    }
}
