use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FusedBlockParams {
    width: u32,
    height: u32,
    step_size: f32,
    dead_zone: f32,
    freq_strength: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Fused DCT-8×8 + quantize + local decode pipeline.
///
/// One dispatch: forward DCT → quantize → dequantize → inverse DCT,
/// all in workgroup shared memory. Outputs both quantized indices
/// and reconstructed pixels.
pub struct FusedBlock {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl FusedBlock {
    pub fn new(ctx: &GpuContext) -> Self {
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("fused_block_bgl"),
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
                        // binding 1: input pixels (read-only storage)
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
                        // binding 2: quantized output (read-write storage)
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
                        // binding 3: reconstructed output (read-write storage)
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

        let pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("fused_block_pl"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("dct8_fused"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/dct8_fused.wgsl").into(),
                ),
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("dct8_fused"),
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

    /// Dispatch the fused DCT-8×8 + quantize + local decode.
    ///
    /// Inputs: `input` buffer of f32 pixel values.
    /// Outputs:
    ///   - `quant_out`: quantized indices (f32, same layout as input)
    ///   - `recon_out`: reconstructed pixels after quantize → dequantize → IDCT
    ///
    /// `width` and `height` must be multiples of 8.
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        quant_out: &wgpu::Buffer,
        recon_out: &wgpu::Buffer,
        width: u32,
        height: u32,
        step_size: f32,
        dead_zone: f32,
        freq_strength: f32,
    ) {
        let params = FusedBlockParams {
            width,
            height,
            step_size,
            dead_zone,
            freq_strength,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fused_block_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused_block_bg"),
            layout: &self.bind_group_layout,
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
                    resource: quant_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: recon_out.as_entire_binding(),
                },
            ],
        });

        // Use ceil-div to cover non-8-aligned dimensions (shader bounds-checks).
        let wg_x = (width + 7) / 8;
        let wg_y = (height + 7) / 8;

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_block"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}
