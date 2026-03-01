/// GPU spatial intra prediction — predicts each 8×8 block from spatial neighbors
/// (left column, top row) and computes residual.
///
/// 4 modes: DC (0), Horizontal (1), Vertical (2), Diagonal-down-left (3).
/// Mode selection minimizes SAD. 2 bits/block overhead (~9.6 KB for 1080p).
/// Y plane only (initial implementation).
///
/// **Known limitation**: With the wavelet transform path, intra prediction hurts
/// quality because block-level prediction creates boundary discontinuities that
/// the tile-level wavelet handles poorly, plus prediction drift (encoder uses
/// original pixels, decoder uses lossy reconstruction). Feature is designed for
/// future BlockDCT8 integration where transform and prediction operate at the
/// same 8×8 block scale. INTRA_TILE_SIZE limits drift accumulation.
use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

const INTRA_BLOCK_SIZE: u32 = 8;

/// Tile size for intra prediction boundaries (independent of wavelet tile size).
/// Smaller = less drift between encoder/decoder predictions, more boundary blocks.
/// Must be a multiple of INTRA_BLOCK_SIZE and a divisor of the wavelet tile size.
pub const INTRA_TILE_SIZE: u32 = 32;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct IntraParams {
    plane_width: u32,
    plane_height: u32,
    tile_size: u32,
    tiles_x: u32,
}

pub struct IntraPredictor {
    forward_pipeline: wgpu::ComputePipeline,
    forward_bgl: wgpu::BindGroupLayout,
    inverse_pipeline: wgpu::ComputePipeline,
    inverse_bgl: wgpu::BindGroupLayout,
}

impl IntraPredictor {
    pub fn new(ctx: &GpuContext) -> Self {
        // Forward (encoder) pipeline
        let fwd_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("intra_predict"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/intra_predict.wgsl").into(),
                ),
            });

        let forward_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("intra_predict_bgl"),
                    entries: &[
                        // binding 0: params (uniform)
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
                        // binding 1: input plane (read-only)
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
                        // binding 2: residual output (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 3: modes output (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
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

        let fwd_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("intra_predict_pl"),
                bind_group_layouts: &[&forward_bgl],
                push_constant_ranges: &[],
            });

        let forward_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("intra_predict_pipeline"),
                    layout: Some(&fwd_pl),
                    module: &fwd_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Inverse (decoder) pipeline
        let inv_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("intra_reconstruct"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/intra_reconstruct.wgsl").into(),
                ),
            });

        let inverse_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("intra_reconstruct_bgl"),
                    entries: &[
                        // binding 0: params (uniform)
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
                        // binding 1: residual input (read-only)
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
                        // binding 2: output (read-write, for sequential reconstruction)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 3: modes input (read-only)
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
                    ],
                });

        let inv_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("intra_reconstruct_pl"),
                bind_group_layouts: &[&inverse_bgl],
                push_constant_ranges: &[],
            });

        let inverse_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("intra_reconstruct_pipeline"),
                    layout: Some(&inv_pl),
                    module: &inv_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            forward_pipeline,
            forward_bgl,
            inverse_pipeline,
            inverse_bgl,
        }
    }

    /// Encoder: select best intra mode per 8×8 block, write residual.
    /// input → residual (same size), modes (1 u32 per block).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        residual: &wgpu::Buffer,
        modes_buf: &wgpu::Buffer,
        padded_w: u32,
        padded_h: u32,
        tile_size: u32,
    ) {
        let tiles_x = padded_w / tile_size;
        let params = IntraParams {
            plane_width: padded_w,
            plane_height: padded_h,
            tile_size,
            tiles_x,
        };
        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("intra_predict_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("intra_predict_bg"),
            layout: &self.forward_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: residual.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: modes_buf.as_entire_binding(),
                },
            ],
        });

        let total_tiles = (padded_w / tile_size) * (padded_h / tile_size);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("intra_predict_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.forward_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(total_tiles, 1, 1);
    }

    /// Decoder: reconstruct spatial pixels from residual + modes.
    /// residual + modes → output (sequential raster scan per tile).
    #[allow(clippy::too_many_arguments)]
    pub fn inverse(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        residual: &wgpu::Buffer,
        output: &wgpu::Buffer,
        modes_buf: &wgpu::Buffer,
        padded_w: u32,
        padded_h: u32,
        tile_size: u32,
    ) {
        let tiles_x = padded_w / tile_size;
        let tiles_y = padded_h / tile_size;
        let params = IntraParams {
            plane_width: padded_w,
            plane_height: padded_h,
            tile_size,
            tiles_x,
        };
        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("intra_reconstruct_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("intra_reconstruct_bg"),
            layout: &self.inverse_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: residual.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: modes_buf.as_entire_binding(),
                },
            ],
        });

        let total_tiles = tiles_x * tiles_y;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("intra_reconstruct_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.inverse_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(total_tiles, 1, 1);
    }

    /// Number of 8×8 blocks for given padded dimensions.
    pub fn num_blocks(padded_w: u32, padded_h: u32) -> u32 {
        (padded_w / INTRA_BLOCK_SIZE) * (padded_h / INTRA_BLOCK_SIZE)
    }

    /// Pack u32 modes (one per block) into 2-bit packed bytes (4 modes per byte).
    pub fn pack_modes(modes: &[u32]) -> Vec<u8> {
        let byte_count = modes.len().div_ceil(4);
        let mut packed = vec![0u8; byte_count];
        for (i, &m) in modes.iter().enumerate() {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            packed[byte_idx] |= ((m & 0x3) as u8) << bit_offset;
        }
        packed
    }

    /// Unpack 2-bit packed bytes to u32 modes.
    pub fn unpack_modes(packed: &[u8], num_blocks: usize) -> Vec<u32> {
        let mut modes = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let m = (packed[byte_idx] >> bit_offset) & 0x3;
            modes.push(m as u32);
        }
        modes
    }
}
