// Chroma resampling: GPU-accelerated downsample (444→422/420) and
// upsample (422/420→444) using compute shaders.
//
// Both operations share the same Params uniform layout; the shader source
// differs (box-filter average for downsample vs. nearest-neighbour for upsample).

use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ChromaResampleParams {
    src_width:         u32,
    src_height:        u32,
    dst_width:         u32,
    dst_height:        u32,
    dst_stride:        u32,  // tile-aligned row stride (= chroma_padded_width for downsample)
    dst_height_padded: u32,  // tile-aligned height     (= chroma_padded_height for downsample)
    shift_x:           u32,
    shift_y:           u32,
}

/// GPU pipeline for chroma resampling (either downsample or upsample).
///
/// Create via [`ChromaResampler::new_downsample`] or
/// [`ChromaResampler::new_upsample`].  Both share the same bind group
/// layout (uniform params + two storage buffers) and differ only in the
/// shader used.
pub struct ChromaResampler {
    pipeline:          wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ChromaResampler {
    fn new_with_shader(ctx: &GpuContext, shader_src: &str, label: &str) -> Self {
        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{label}_bgl")),
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
                    label: Some(&format!("{label}_pl")),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{label}_pipeline")),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self { pipeline, bind_group_layout }
    }

    /// Create a downsampler (box-filter average).
    pub fn new_downsample(ctx: &GpuContext) -> Self {
        Self::new_with_shader(
            ctx,
            include_str!("../shaders/chroma_downsample.wgsl"),
            "chroma_downsample",
        )
    }

    /// Create an upsampler (nearest-neighbour).
    pub fn new_upsample(ctx: &GpuContext) -> Self {
        Self::new_with_shader(
            ctx,
            include_str!("../shaders/chroma_upsample.wgsl"),
            "chroma_upsample",
        )
    }

    /// Dispatch downsample.
    ///
    /// - `src_buf` / `dst_buf` must be `STORAGE` f32 buffers.
    /// - `src_w` / `src_h` are the source (luma padded) dimensions.
    /// - `shift_x` / `shift_y` are the log2 scale factors.
    /// - `dst_stride` is the tile-aligned chroma row stride (= `chroma_padded_width`).
    /// - `dst_height_padded` is the tile-aligned chroma height (= `chroma_padded_height`).
    ///
    /// The shader writes the full `dst_stride × dst_height_padded` region, padding
    /// columns/rows beyond the valid chroma area with edge-replicated values so the
    /// wavelet transform finds no uninitialised garbage in the buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        src_buf: &wgpu::Buffer,
        dst_buf: &wgpu::Buffer,
        src_w: u32,
        src_h: u32,
        shift_x: u32,
        shift_y: u32,
        dst_stride: u32,
        dst_height_padded: u32,
    ) {
        // Valid dst dims are derived from the source dims and shift factors.
        self.dispatch_raw(ctx, cmd, src_buf, dst_buf, src_w, src_h,
            src_w >> shift_x, src_h >> shift_y, shift_x, shift_y,
            dst_stride, dst_height_padded);
    }

    /// Dispatch upsample: src is chroma-sized (padded), dst is luma-sized.
    /// `src_w`/`src_h` = chroma padded dimensions; `dst_w`/`dst_h` = luma padded dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_upsample(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        src_buf: &wgpu::Buffer,
        dst_buf: &wgpu::Buffer,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        shift_x: u32,
        shift_y: u32,
    ) {
        // Upsample shader guards on idx >= dst_width * dst_height and uses no
        // stride/padded-height fields.  Pass them as 0 so there are no dummy
        // sentinel values, and dispatch exactly dst_w * dst_h invocations.
        let params = ChromaResampleParams {
            src_width:         src_w,
            src_height:        src_h,
            dst_width:         dst_w,
            dst_height:        dst_h,
            dst_stride:        0, // unused by upsample shader
            dst_height_padded: 0, // unused by upsample shader
            shift_x,
            shift_y,
        };
        let total_dst = dst_w * dst_h;
        self.dispatch_with_params(ctx, cmd, src_buf, dst_buf, &params, total_dst);
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_raw(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        src_buf: &wgpu::Buffer,
        dst_buf: &wgpu::Buffer,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        shift_x: u32,
        shift_y: u32,
        dst_stride: u32,
        dst_height_padded: u32,
    ) {
        let params = ChromaResampleParams {
            src_width:  src_w,
            src_height: src_h,
            dst_width:  dst_w,
            dst_height: dst_h,
            dst_stride,
            dst_height_padded,
            shift_x,
            shift_y,
        };
        // Dispatch over the full padded output region (dst_stride × dst_height_padded)
        // so every element — including padding — is written before the wavelet runs.
        let total_dst = dst_stride * dst_height_padded;
        self.dispatch_with_params(ctx, cmd, src_buf, dst_buf, &params, total_dst);
    }

    /// Create a uniform params buffer, bind group, and compute pass, then dispatch
    /// `total_dst` invocations (rounded up to 256-thread workgroups).
    fn dispatch_with_params(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        src_buf: &wgpu::Buffer,
        dst_buf: &wgpu::Buffer,
        params: &ChromaResampleParams,
        total_dst: u32,
    ) {
        let params_buf =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("chroma_resample_params"),
                    contents: bytemuck::bytes_of(params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chroma_resample_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = total_dst.div_ceil(256);
        let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("chroma_resample_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}
