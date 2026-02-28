use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlockTransformParams {
    width: u32,
    height: u32,
    direction: u32,
    levels: u32, // Haar: 1 or 2 decomposition levels (ignored by other transforms)
}

/// Block-based transform types for the mega-kernel pipeline exploration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockTransformType {
    /// 8×8 Type-II DCT (JPEG/H.264 style). 64 threads/block, 1 dispatch.
    DCT8,
    /// 16×16 Type-II DCT. Better energy compaction, 256 threads/block, 1 dispatch.
    DCT16,
    /// 4×4 Walsh-Hadamard. No multiplications, 16 blocks/workgroup, 1 dispatch.
    Hadamard4,
    /// Block-local Haar wavelet (2 levels in 16×16 block). 1 dispatch.
    HaarBlock,
}

impl BlockTransformType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::DCT8 => "DCT-8x8",
            Self::DCT16 => "DCT-16x16",
            Self::Hadamard4 => "WHT-4x4",
            Self::HaarBlock => "Haar-16x16",
        }
    }

    /// Dispatch grid cell size (pixels). Determines workgroup count.
    pub fn dispatch_block_size(&self) -> u32 {
        match self {
            Self::DCT8 => 8,
            Self::DCT16 | Self::Hadamard4 | Self::HaarBlock => 16,
        }
    }

    pub fn all() -> &'static [BlockTransformType] {
        &[Self::DCT8, Self::DCT16, Self::Hadamard4, Self::HaarBlock]
    }
}

/// GPU pipelines for all block-based transform candidates.
pub struct BlockTransform {
    pipeline_dct8: wgpu::ComputePipeline,
    pipeline_dct16: wgpu::ComputePipeline,
    pipeline_had4: wgpu::ComputePipeline,
    pipeline_haar: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl BlockTransform {
    pub fn new(ctx: &GpuContext) -> Self {
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("block_transform_bgl"),
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

        let pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("block_transform_pl"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        fn make_pipeline(
            ctx: &GpuContext,
            layout: &wgpu::PipelineLayout,
            label: &str,
            source: &str,
        ) -> wgpu::ComputePipeline {
            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
        }

        Self {
            pipeline_dct8: make_pipeline(
                ctx,
                &pipeline_layout,
                "dct8",
                include_str!("../shaders/dct8.wgsl"),
            ),
            pipeline_dct16: make_pipeline(
                ctx,
                &pipeline_layout,
                "dct16",
                include_str!("../shaders/dct16.wgsl"),
            ),
            pipeline_had4: make_pipeline(
                ctx,
                &pipeline_layout,
                "hadamard4",
                include_str!("../shaders/hadamard4.wgsl"),
            ),
            pipeline_haar: make_pipeline(
                ctx,
                &pipeline_layout,
                "haar_block",
                include_str!("../shaders/haar_block.wgsl"),
            ),
            bind_group_layout,
        }
    }

    /// Dispatch a block transform (forward or inverse) into an existing command encoder.
    /// `width` and `height` must be aligned to the transform's dispatch_block_size.
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        width: u32,
        height: u32,
        forward: bool,
        transform_type: BlockTransformType,
    ) {
        self.dispatch_with_levels(ctx, encoder, input, output, width, height, forward, transform_type, 2)
    }

    /// Dispatch with explicit level count (only meaningful for HaarBlock).
    pub fn dispatch_with_levels(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        width: u32,
        height: u32,
        forward: bool,
        transform_type: BlockTransformType,
        levels: u32,
    ) {
        let params = BlockTransformParams {
            width,
            height,
            direction: if forward { 0 } else { 1 },
            levels,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("block_transform_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("block_transform_bg"),
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
                    resource: output.as_entire_binding(),
                },
            ],
        });

        let pipeline = match transform_type {
            BlockTransformType::DCT8 => &self.pipeline_dct8,
            BlockTransformType::DCT16 => &self.pipeline_dct16,
            BlockTransformType::Hadamard4 => &self.pipeline_had4,
            BlockTransformType::HaarBlock => &self.pipeline_haar,
        };

        let bs = transform_type.dispatch_block_size();
        let wg_x = width / bs;
        let wg_y = height / bs;

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("block_transform"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}
