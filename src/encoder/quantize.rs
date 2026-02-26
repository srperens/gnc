use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QuantizeParams {
    total_count: u32,
    step_size: f32,
    direction: u32,
    dead_zone: f32,
    width: u32,
    height: u32,
    tile_size: u32,
    num_levels: u32,
    weights0: [f32; 4],
    weights1: [f32; 4],
    weights2: [f32; 4],
    weights3: [f32; 4],
    // Adaptive quantization parameters
    aq_enabled: u32,
    aq_ll_block_size: u32,
    aq_ll_blocks_per_tile_x: u32,
    aq_tiles_x: u32,
}

pub struct Quantizer {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Dummy 1-element buffer used when adaptive quantization is disabled.
    /// Avoids needing separate pipelines for adaptive vs non-adaptive.
    dummy_weight_buf: wgpu::Buffer,
}

impl Quantizer {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("quantize"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/quantize.wgsl").into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("quantize_bgl"),
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
                label: Some("quantize_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quantize_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Dummy weight map buffer (single 1.0) for when AQ is disabled
        let dummy_weight_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dummy_weight_map"),
                contents: bytemuck::bytes_of(&1.0f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        Self {
            pipeline,
            bind_group_layout,
            dummy_weight_buf,
        }
    }

    /// Dispatch quantization/dequantization without adaptive spatial weighting.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        total_count: u32,
        step_size: f32,
        dead_zone: f32,
        forward: bool,
        width: u32,
        height: u32,
        tile_size: u32,
        num_levels: u32,
        weights: &[f32; 16],
    ) {
        self.dispatch_adaptive(
            ctx,
            encoder,
            input_buf,
            output_buf,
            total_count,
            step_size,
            dead_zone,
            forward,
            width,
            height,
            tile_size,
            num_levels,
            weights,
            None,
        );
    }

    /// Dispatch quantization/dequantization with optional adaptive spatial weighting.
    ///
    /// `weight_map`: if Some, provides (buffer, ll_block_size, ll_blocks_per_tile_x, tiles_x).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_adaptive(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        total_count: u32,
        step_size: f32,
        dead_zone: f32,
        forward: bool,
        width: u32,
        height: u32,
        tile_size: u32,
        num_levels: u32,
        weights: &[f32; 16],
        weight_map: Option<(&wgpu::Buffer, u32, u32, u32)>,
    ) {
        let (aq_enabled, aq_ll_block_size, aq_ll_blocks_per_tile_x, aq_tiles_x, wm_buf) =
            match weight_map {
                Some((buf, ll_block_size, ll_blocks_per_tile_x, tiles_x)) => {
                    (1u32, ll_block_size, ll_blocks_per_tile_x, tiles_x, buf)
                }
                None => (0u32, 8, 1, 1, &self.dummy_weight_buf),
            };

        let params = QuantizeParams {
            total_count,
            step_size,
            direction: if forward { 0 } else { 1 },
            dead_zone,
            width,
            height,
            tile_size,
            num_levels,
            weights0: [weights[0], weights[1], weights[2], weights[3]],
            weights1: [weights[4], weights[5], weights[6], weights[7]],
            weights2: [weights[8], weights[9], weights[10], weights[11]],
            weights3: [weights[12], weights[13], weights[14], weights[15]],
            aq_enabled,
            aq_ll_block_size,
            aq_ll_blocks_per_tile_x,
            aq_tiles_x,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quantize_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quantize_bg"),
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
                    resource: wm_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = total_count.div_ceil(256);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("quantize_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}
