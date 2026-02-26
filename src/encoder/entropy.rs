use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EntropyParams {
    total_count: u32,
    direction: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Phase 1 entropy: simple i16 packing (no real entropy coding yet).
pub struct EntropyCoder {
    encode_pipeline: wgpu::ComputePipeline,
    decode_pipeline: wgpu::ComputePipeline,
    encode_bgl: wgpu::BindGroupLayout,
    decode_bgl: wgpu::BindGroupLayout,
}

impl EntropyCoder {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("entropy"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/entropy.wgsl").into()),
            });

        // Encode bind group layout: params, input_f32, output_packed, (unused output_f32)
        let encode_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("entropy_encode_bgl"),
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
                ],
            });

        let decode_bgl = encode_bgl.clone();

        let encode_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("entropy_encode_pl"),
                bind_group_layouts: &[&encode_bgl],
                push_constant_ranges: &[],
            });

        let encode_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("entropy_encode_pipeline"),
                    layout: Some(&encode_pl),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let decode_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("entropy_decode_pl"),
                bind_group_layouts: &[&encode_bgl],
                push_constant_ranges: &[],
            });

        let decode_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("entropy_decode_pipeline"),
                    layout: Some(&decode_pl),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            encode_pipeline,
            decode_pipeline,
            encode_bgl,
            decode_bgl,
        }
    }

    /// Number of u32 words needed to pack `count` f32 coefficients as i16 pairs
    pub fn packed_size(count: u32) -> u32 {
        count.div_ceil(2)
    }

    /// Encode: quantized f32 -> packed u32 (i16 pairs)
    pub fn dispatch_encode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input_f32_buf: &wgpu::Buffer,
        output_packed_buf: &wgpu::Buffer,
        dummy_f32_buf: &wgpu::Buffer,
        total_count: u32,
    ) {
        let params = EntropyParams {
            total_count,
            direction: 0,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("entropy_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entropy_encode_bg"),
            layout: &self.encode_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_f32_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_packed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dummy_f32_buf.as_entire_binding(),
                },
            ],
        });

        let pairs = Self::packed_size(total_count);
        let workgroups = pairs.div_ceil(256);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("entropy_encode_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.encode_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Decode: packed u32 (i16 pairs) -> f32
    pub fn dispatch_decode(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        dummy_input_f32: &wgpu::Buffer,
        packed_buf: &wgpu::Buffer,
        output_f32_buf: &wgpu::Buffer,
        total_count: u32,
    ) {
        let params = EntropyParams {
            total_count,
            direction: 1,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("entropy_decode_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entropy_decode_bg"),
            layout: &self.decode_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dummy_input_f32.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: packed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_f32_buf.as_entire_binding(),
                },
            ],
        });

        let pairs = Self::packed_size(total_count);
        let workgroups = pairs.div_ceil(256);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("entropy_decode_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.decode_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}
