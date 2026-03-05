use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Temporal53Params {
    count: u32,
    direction: u32, // 0 = forward, 1 = inverse
    pass_idx: u32,  // 0 = first pass, 1 = second pass
    _pad: u32,
}

pub struct Temporal53Gpu {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Temporal53Gpu {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("temporal_53"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/temporal_53.wgsl").into(),
                ),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("temporal_53_bgl"),
                    entries: &[
                        // binding 0: uniform params
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
                        // bindings 1-4: read-only inputs
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        // bindings 5-6: read_write outputs
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
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
                label: Some("temporal_53_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("temporal_53_pipeline"),
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

    /// Dispatch one pass of the temporal 5/3 wavelet.
    ///
    /// `forward`: true = forward transform, false = inverse
    /// `pass`: 0 = first pass (predict / undo-update), 1 = second pass (update / undo-predict)
    ///
    /// Buffer assignments depend on direction and pass — see shader comments.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        in0: &wgpu::Buffer,
        in1: &wgpu::Buffer,
        in2: &wgpu::Buffer,
        in3: &wgpu::Buffer,
        out0: &wgpu::Buffer,
        out1: &wgpu::Buffer,
        count: u32,
        forward: bool,
        pass: u32,
    ) {
        let params = Temporal53Params {
            count,
            direction: if forward { 0 } else { 1 },
            pass_idx: pass,
            _pad: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("temporal_53_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("temporal_53_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: in0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: in1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: in2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: in3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out1.as_entire_binding(),
                },
            ],
        });

        let workgroups = count.div_ceil(256);

        let mut pass_enc = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("temporal_53_pass"),
            timestamp_writes: None,
        });
        pass_enc.set_pipeline(&self.pipeline);
        pass_enc.set_bind_group(0, &bind_group, &[]);
        pass_enc.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Forward transform: 4 input frames → 2 lowpass (s0, s1) + 2 highpass (d0, d1).
    ///
    /// Requires 2 dispatches (predict then update) with a queue.submit between them
    /// to ensure predict results are visible to the update pass.
    ///
    /// `scratch0`, `scratch1` are temporary buffers for intermediate d0, d1 values.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_4(
        &self,
        ctx: &GpuContext,
        f0: &wgpu::Buffer,
        f1: &wgpu::Buffer,
        f2: &wgpu::Buffer,
        f3: &wgpu::Buffer,
        s0: &wgpu::Buffer,
        s1: &wgpu::Buffer,
        d0: &wgpu::Buffer,
        d1: &wgpu::Buffer,
        count: u32,
    ) {
        // Pass 1: predict — compute d0, d1 from f0..f3
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("t53_fwd_predict"),
            });
        self.dispatch(ctx, &mut cmd, f0, f1, f2, f3, d0, d1, count, true, 0);
        ctx.queue.submit(Some(cmd.finish()));

        // Pass 2: update — compute s0, s1 from f0, d0, f2, d1
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("t53_fwd_update"),
            });
        self.dispatch(ctx, &mut cmd, f0, d0, f2, d1, s0, s1, count, true, 1);
        ctx.queue.submit(Some(cmd.finish()));
    }

    /// Inverse transform: 2 lowpass (s0, s1) + 2 highpass (d0, d1) → 4 output frames.
    ///
    /// `f0_out`..`f3_out` receive the reconstructed frames.
    /// Reuses `f0_out` and `f2_out` as intermediate storage for x0/x2.
    #[allow(clippy::too_many_arguments)]
    pub fn inverse_4(
        &self,
        ctx: &GpuContext,
        s0: &wgpu::Buffer,
        s1: &wgpu::Buffer,
        d0: &wgpu::Buffer,
        d1: &wgpu::Buffer,
        f0_out: &wgpu::Buffer,
        f1_out: &wgpu::Buffer,
        f2_out: &wgpu::Buffer,
        f3_out: &wgpu::Buffer,
        count: u32,
    ) {
        // Pass 1: undo update — compute x0, x2 from s0, d0, s1, d1
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("t53_inv_update"),
            });
        self.dispatch(
            ctx, &mut cmd, s0, d0, s1, d1, f0_out, f2_out, count, false, 0,
        );
        ctx.queue.submit(Some(cmd.finish()));

        // Pass 2: undo predict — compute x1, x3 from x0, d0, x2, d1
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("t53_inv_predict"),
            });
        self.dispatch(
            ctx, &mut cmd, f0_out, d0, f2_out, d1, f1_out, f3_out, count, false, 1,
        );
        ctx.queue.submit(Some(cmd.finish()));
    }
}
