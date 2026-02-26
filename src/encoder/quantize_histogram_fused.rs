//! Fused quantize + histogram GPU pipeline.
//!
//! Combines the quantization pass (quantize.wgsl) and the histogram pass
//! (rans_histogram.wgsl) into a single compute dispatch. This eliminates
//! one full read + write of the quantized coefficient buffer per plane,
//! saving significant memory bandwidth (~24MB per plane at 1080p).
//!
//! The shader dispatches one workgroup (256 threads) per tile. Each thread
//! reads wavelet coefficients, quantizes them, writes the quantized values
//! back, and accumulates histogram counts in workgroup shared memory.
//!
//! The histogram output format is identical to rans_histogram.wgsl, so
//! downstream normalize + encode passes work unchanged.

use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

/// Uniform buffer layout matching the fused shader's `Params` struct.
/// Must be kept in sync with `quantize_histogram_fused.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FusedQuantizeHistogramParams {
    // Histogram params (first block)
    pub num_tiles: u32,
    pub coefficients_per_tile: u32,
    pub plane_width: u32,
    pub tile_size: u32,
    pub tiles_x: u32,
    pub per_subband: u32,
    pub num_levels: u32,
    pub flags: u32, // bit 0: disable ZRL

    // Quantize params
    pub total_count: u32,
    pub step_size: f32,
    pub dead_zone: f32,
    pub _pad0: u32,
    pub weights0: [f32; 4],
    pub weights1: [f32; 4],
    pub weights2: [f32; 4],
    pub weights3: [f32; 4],
    // Adaptive quantization
    pub aq_enabled: u32,
    pub aq_ll_block_size: u32,
    pub aq_ll_blocks_per_tile_x: u32,
    pub aq_tiles_x: u32,
    // Plane height
    pub plane_height: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

/// GPU pipeline for the fused quantize + histogram shader.
pub struct FusedQuantizeHistogram {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Dummy 1-element buffer used when adaptive quantization is disabled.
    dummy_weight_buf: wgpu::Buffer,
}

impl FusedQuantizeHistogram {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("quantize_histogram_fused"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/quantize_histogram_fused.wgsl").into(),
                ),
            });

        // Bind group layout:
        //   binding 0: uniform (params)
        //   binding 1: storage read (input wavelet coefficients)
        //   binding 2: storage read_write (output quantized coefficients)
        //   binding 3: storage read_write (histogram output)
        //   binding 4: storage read (weight map for adaptive QP)
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("fused_qh_bgl"),
                    entries: &[
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fused_qh_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fused_qh_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let dummy_weight_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fused_qh_dummy_wm"),
                contents: bytemuck::bytes_of(&1.0f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        Self {
            pipeline,
            bind_group_layout,
            dummy_weight_buf,
        }
    }

    /// Dispatch the fused quantize + histogram shader.
    ///
    /// Reads wavelet coefficients from `input_buf`, writes quantized values to
    /// `output_buf`, and writes histogram data to `hist_buf`.
    ///
    /// The `hist_buf` must be pre-allocated with at least
    /// `num_tiles * HIST_TILE_STRIDE * 4` bytes (same as rans_histogram).
    ///
    /// `weight_map`: if Some, provides (buffer, ll_block_size, ll_blocks_per_tile_x, tiles_x)
    /// for adaptive quantization. If None, uniform quantization is used.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        hist_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
        tile_size: u32,
        num_levels: u32,
        step_size: f32,
        dead_zone: f32,
        weights: &[f32; 16],
        per_subband: bool,
        flags: u32,
        weight_map: Option<(&wgpu::Buffer, u32, u32, u32)>,
    ) {
        let tiles_x = width / tile_size;
        let tiles_y = height / tile_size;
        let num_tiles = tiles_x * tiles_y;
        let total_count = width * height;
        let coefficients_per_tile = tile_size * tile_size;

        let (aq_enabled, aq_ll_block_size, aq_ll_blocks_per_tile_x, aq_tiles_x, wm_buf) =
            match weight_map {
                Some((buf, ll_block_size, ll_blocks_per_tile_x, tiles_x_val)) => {
                    (1u32, ll_block_size, ll_blocks_per_tile_x, tiles_x_val, buf)
                }
                None => (0u32, 8, 1, 1, &self.dummy_weight_buf),
            };

        let params = FusedQuantizeHistogramParams {
            num_tiles,
            coefficients_per_tile,
            plane_width: width,
            tile_size,
            tiles_x,
            per_subband: u32::from(per_subband),
            num_levels,
            flags,
            total_count,
            step_size,
            dead_zone,
            _pad0: 0,
            weights0: [weights[0], weights[1], weights[2], weights[3]],
            weights1: [weights[4], weights[5], weights[6], weights[7]],
            weights2: [weights[8], weights[9], weights[10], weights[11]],
            weights3: [weights[12], weights[13], weights[14], weights[15]],
            aq_enabled,
            aq_ll_block_size,
            aq_ll_blocks_per_tile_x,
            aq_tiles_x,
            plane_height: height,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fused_qh_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_qh_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: hist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wm_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_qh_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_tiles, 1, 1);
    }
}
