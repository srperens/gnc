use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::{FrameInfo, GpuContext, WaveletType};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TransformParams {
    width: u32,
    height: u32,
    tile_size: u32,
    direction: u32,
    pass_mode: u32,
    tiles_x: u32,
    region_size: u32,
    _pad0: u32,
}

pub struct WaveletTransform {
    pipeline_53: wgpu::ComputePipeline,
    pipeline_97: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl WaveletTransform {
    pub fn new(ctx: &GpuContext) -> Self {
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("transform_bgl"),
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
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("transform_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // LeGall 5/3 pipeline
        let shader_53 = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("transform_53"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/transform.wgsl").into()),
            });

        let pipeline_53 = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("transform_pipeline_53"),
                layout: Some(&pipeline_layout),
                module: &shader_53,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // CDF 9/7 pipeline
        let shader_97 = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("transform_97"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/transform_97.wgsl").into(),
                ),
            });

        let pipeline_97 = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("transform_pipeline_97"),
                layout: Some(&pipeline_layout),
                module: &shader_97,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            pipeline_53,
            pipeline_97,
            bind_group_layout,
        }
    }

    /// Dispatch one pass of the wavelet transform for a single plane.
    /// `pass_mode`: 0 = rows, 1 = columns
    /// `region_size`: the sub-region width/height for this decomposition level.
    fn dispatch_pass(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        info: &FrameInfo,
        forward: bool,
        pass_mode: u32,
        region_size: u32,
        wavelet_type: WaveletType,
    ) {
        let params = TransformParams {
            width: info.padded_width(),
            height: info.padded_height(),
            tile_size: info.tile_size,
            direction: if forward { 0 } else { 1 },
            pass_mode,
            tiles_x: info.tiles_x(),
            region_size,
            _pad0: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("transform_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transform_bg"),
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
            ],
        });

        let pipeline = match wavelet_type {
            WaveletType::LeGall53 => &self.pipeline_53,
            WaveletType::CDF97 => &self.pipeline_97,
        };

        // Dispatch: one workgroup per line (row or column) per tile
        let lines_per_tile = region_size;
        let total_tiles = info.total_tiles();

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("transform_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(lines_per_tile, total_tiles, 1);
    }

    /// Run multi-level 2D forward wavelet transform.
    /// Level 0 transforms the full tile, level 1 transforms the LL subband, etc.
    pub fn forward(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        temp_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        info: &FrameInfo,
        levels: u32,
        wavelet_type: WaveletType,
    ) {
        let mut region = info.tile_size;

        for level in 0..levels {
            // Each level reads from output_buf (or input_buf for level 0),
            // does rows into temp_buf, then columns into output_buf.
            let src = if level == 0 { input_buf } else { output_buf };

            // Row pass: src -> temp_buf
            self.dispatch_pass(
                ctx,
                encoder,
                src,
                temp_buf,
                info,
                true,
                0,
                region,
                wavelet_type,
            );

            // Column pass: temp_buf -> output_buf
            self.dispatch_pass(
                ctx,
                encoder,
                temp_buf,
                output_buf,
                info,
                true,
                1,
                region,
                wavelet_type,
            );

            region /= 2;
        }
    }

    /// Run multi-level 2D inverse wavelet transform.
    /// Levels are processed in reverse order (smallest region first).
    pub fn inverse(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        temp_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        info: &FrameInfo,
        levels: u32,
        wavelet_type: WaveletType,
    ) {
        let buf_size = (info.padded_width() * info.padded_height()) as u64 * 4;

        // Copy input_buf -> output_buf first so we have all subbands in output_buf.
        // Each inverse level reads from output_buf and writes back to output_buf,
        // preserving the detail subbands from higher levels.
        encoder.copy_buffer_to_buffer(input_buf, 0, output_buf, 0, buf_size);

        let min_region = info.tile_size >> (levels - 1);
        let mut region = min_region;

        for _level in 0..levels {
            // Inverse column pass: output_buf -> temp_buf
            self.dispatch_pass(
                ctx,
                encoder,
                output_buf,
                temp_buf,
                info,
                false,
                1,
                region,
                wavelet_type,
            );

            // Inverse row pass: temp_buf -> output_buf
            self.dispatch_pass(
                ctx,
                encoder,
                temp_buf,
                output_buf,
                info,
                false,
                0,
                region,
                wavelet_type,
            );

            region *= 2;
        }
    }
}
