use std::cell::RefCell;
use wgpu;

use super::buffer_cache::CachedBuffers;
use crate::encoder::bitplane::GpuBitplaneDecoder;
use crate::encoder::cfl::CflPredictor;
use crate::encoder::color::ColorConverter;
use crate::encoder::interleave::PlaneInterleaver;
use crate::encoder::motion::MotionEstimator;
use crate::encoder::quantize::Quantizer;
use crate::encoder::rans_gpu::GpuRansDecoder;
use crate::encoder::transform::WaveletTransform;
use crate::{CompressedFrame, GpuContext};

/// Handle returned by `decode_to_texture` with metadata about the decoded frame.
/// The actual texture view is accessible via `DecoderPipeline::output_texture_view()`.
pub struct TextureHandle {
    pub width: u32,
    pub height: u32,
}

/// Full decoding pipeline: GPU rANS Decode -> Dequantize -> Inverse Wavelet -> Interleave -> Inverse Color -> Crop
///
/// All GPU stages run without CPU readback until the final RGB output.
/// GPU buffers are cached across frames for zero-allocation steady-state decode.
pub struct DecoderPipeline {
    pub(super) color: ColorConverter,
    pub(super) transform: WaveletTransform,
    pub(super) quantize: Quantizer,
    pub(super) rans_decoder: GpuRansDecoder,
    pub(super) bitplane_decoder: GpuBitplaneDecoder,
    pub(super) interleaver: PlaneInterleaver,
    pub(super) cfl_predictor: CflPredictor,
    pub(super) motion: MotionEstimator,
    pub(super) crop_pipeline: wgpu::ComputePipeline,
    pub(super) crop_bgl: wgpu::BindGroupLayout,
    pub(super) pack_pipeline: wgpu::ComputePipeline,
    pub(super) pack_bgl: wgpu::BindGroupLayout,
    pub(super) buf_to_tex_pipeline: wgpu::ComputePipeline,
    pub(super) buf_to_tex_bgl: wgpu::BindGroupLayout,
    pub(super) cached: RefCell<Option<CachedBuffers>>,
    pub(super) pending_rx:
        RefCell<Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
}

impl DecoderPipeline {
    pub fn new(ctx: &GpuContext) -> Self {
        let crop_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("crop"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/crop.wgsl").into()),
            });

        let crop_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("crop_bgl"),
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

        let crop_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("crop_pl"),
                bind_group_layouts: &[&crop_bgl],
                push_constant_ranges: &[],
            });

        let crop_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("crop_pipeline"),
                layout: Some(&crop_pl),
                module: &crop_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Pack u8 shader (f32 → packed u8)
        let pack_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pack_u8"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pack_u8.wgsl").into()),
            });

        let pack_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pack_bgl"),
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

        let pack_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pack_pl"),
                bind_group_layouts: &[&pack_bgl],
                push_constant_ranges: &[],
            });

        let pack_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pack_pipeline"),
                layout: Some(&pack_pl),
                module: &pack_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Buffer-to-texture shader (f32 RGB buffer → rgba8unorm texture)
        let buf_to_tex_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("buffer_to_texture"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/buffer_to_texture.wgsl").into(),
                ),
            });

        let buf_to_tex_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("buf_to_tex_bgl"),
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
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

        let buf_to_tex_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("buf_to_tex_pl"),
                bind_group_layouts: &[&buf_to_tex_bgl],
                push_constant_ranges: &[],
            });

        let buf_to_tex_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("buf_to_tex_pipeline"),
                    layout: Some(&buf_to_tex_pl),
                    module: &buf_to_tex_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            color: ColorConverter::new(ctx),
            transform: WaveletTransform::new(ctx),
            quantize: Quantizer::new(ctx),
            rans_decoder: GpuRansDecoder::new(ctx),
            bitplane_decoder: GpuBitplaneDecoder::new(ctx),
            interleaver: PlaneInterleaver::new(ctx),
            cfl_predictor: CflPredictor::new(ctx),
            motion: MotionEstimator::new(ctx),
            crop_pipeline,
            crop_bgl,
            pack_pipeline,
            pack_bgl,
            buf_to_tex_pipeline,
            buf_to_tex_bgl,
            cached: RefCell::new(None),
            pending_rx: RefCell::new(None),
        }
    }

    pub(super) fn ensure_cached(
        &self,
        ctx: &GpuContext,
        padded_w: u32,
        padded_h: u32,
        width: u32,
        height: u32,
        tile_size: u32,
    ) {
        let mut cached = self.cached.borrow_mut();
        if let Some(ref c) = *cached {
            if c.padded_w == padded_w
                && c.padded_h == padded_h
                && c.width == width
                && c.height == height
                && c.tile_size == tile_size
            {
                return;
            }
        }

        *cached = Some(CachedBuffers::new(
            ctx,
            padded_w,
            padded_h,
            width,
            height,
            tile_size,
            &self.buf_to_tex_bgl,
        ));
    }

    /// Decode a compressed frame back to RGB f32 data.
    /// Returns Vec<f32> of length width * height * 3 (interleaved R,G,B).
    pub fn decode(&self, ctx: &GpuContext, frame: &CompressedFrame) -> Vec<f32> {
        let profile = std::env::var("GNC_PROFILE").is_ok();
        let t_start = std::time::Instant::now();

        let info = &frame.info;
        let w = info.width;
        let h = info.height;
        let output_pixels = w * h;
        let output_size = (output_pixels as u64) * 3 * 4;

        self.ensure_cached(
            ctx,
            info.padded_width(),
            info.padded_height(),
            w,
            h,
            info.tile_size,
        );

        let t_alloc = t_start.elapsed();

        self.prepare_frame_data(ctx, frame);

        let t_prepare = t_start.elapsed();

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let mut cmd = self.encode_gpu_work(ctx, frame, bufs);

        // Copy f32 cropped output to staging
        cmd.copy_buffer_to_buffer(&bufs.cropped_buf, 0, &bufs.staging, 0, output_size);

        let t_encode_cmd = t_start.elapsed();

        ctx.queue.submit(Some(cmd.finish()));

        let t_submit = t_start.elapsed();

        // Map staging and read back
        let slice = bufs.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        bufs.staging.unmap();
        drop(cached);

        if profile {
            let t_total = t_start.elapsed();
            eprintln!(
                "[decode profile] alloc={:.2}ms prepare={:.2}ms cmd={:.2}ms submit={:.2}ms readback={:.2}ms total={:.2}ms",
                t_alloc.as_secs_f64() * 1000.0,
                (t_prepare - t_alloc).as_secs_f64() * 1000.0,
                (t_encode_cmd - t_prepare).as_secs_f64() * 1000.0,
                (t_submit - t_encode_cmd).as_secs_f64() * 1000.0,
                (t_total - t_submit).as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
            );
        }

        result
    }

    /// Decode a compressed frame to packed RGB u8 data.
    /// Returns Vec<u8> of length width * height * 3. This is 4x smaller to read back
    /// from the GPU than the f32 path, significantly reducing readback latency.
    pub fn decode_u8(&self, ctx: &GpuContext, frame: &CompressedFrame) -> Vec<u8> {
        let profile = std::env::var("GNC_PROFILE").is_ok();
        let t_start = std::time::Instant::now();

        let info = &frame.info;
        let w = info.width;
        let h = info.height;
        let total_f32s = w * h * 3;
        let packed_u32s = total_f32s.div_ceil(4);
        let packed_byte_size = (packed_u32s as u64) * 4;

        self.ensure_cached(
            ctx,
            info.padded_width(),
            info.padded_height(),
            w,
            h,
            info.tile_size,
        );

        let t_alloc = t_start.elapsed();

        self.prepare_frame_data(ctx, frame);

        let t_prepare = t_start.elapsed();

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let mut cmd = self.encode_gpu_work(ctx, frame, bufs);

        // GPU pack: f32 → packed u8 (using cached pack_params_buf)
        {
            let pack_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pack_bg"),
                layout: &self.pack_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bufs.pack_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bufs.cropped_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bufs.packed_u8_buf.as_entire_binding(),
                    },
                ],
            });

            let workgroups = packed_u32s.div_ceil(256);
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pack_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &pack_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy packed u8 to staging
        cmd.copy_buffer_to_buffer(
            &bufs.packed_u8_buf,
            0,
            &bufs.staging_u8,
            0,
            packed_byte_size,
        );

        let t_encode_cmd = t_start.elapsed();

        ctx.queue.submit(Some(cmd.finish()));

        let t_submit = t_start.elapsed();

        // Map staging and read back
        let slice = bufs.staging_u8.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let bytes: &[u8] = &data;
        let result = bytes[..total_f32s as usize].to_vec();
        drop(data);
        bufs.staging_u8.unmap();
        drop(cached);

        if profile {
            let t_total = t_start.elapsed();
            eprintln!(
                "[decode_u8 profile] alloc={:.2}ms prepare={:.2}ms cmd={:.2}ms submit={:.2}ms readback={:.2}ms total={:.2}ms",
                t_alloc.as_secs_f64() * 1000.0,
                (t_prepare - t_alloc).as_secs_f64() * 1000.0,
                (t_encode_cmd - t_prepare).as_secs_f64() * 1000.0,
                (t_submit - t_encode_cmd).as_secs_f64() * 1000.0,
                (t_total - t_submit).as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
            );
        }

        result
    }

    /// Decode a compressed frame to an on-GPU rgba8unorm texture.
    /// Zero readback: data stays entirely on the GPU. Returns a reference to the
    /// texture view that can be used directly for rendering (e.g. as a sampled texture
    /// in a render pass). The texture is owned by the cached buffers and reused across
    /// frames — the caller must use or copy it before the next decode call.
    pub fn decode_to_texture(&self, ctx: &GpuContext, frame: &CompressedFrame) -> TextureHandle {
        let info = &frame.info;
        let w = info.width;
        let h = info.height;

        self.ensure_cached(
            ctx,
            info.padded_width(),
            info.padded_height(),
            w,
            h,
            info.tile_size,
        );

        self.prepare_frame_data(ctx, frame);

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let mut cmd = self.encode_gpu_work(ctx, frame, bufs);

        // Buffer-to-texture: cropped f32 RGB → rgba8unorm texture
        {
            let wg_x = w.div_ceil(16);
            let wg_y = h.div_ceil(16);
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("buf_to_tex_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.buf_to_tex_pipeline);
            pass.set_bind_group(0, &bufs.buf_to_tex_bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        ctx.queue.submit(Some(cmd.finish()));

        TextureHandle {
            width: w,
            height: h,
        }
    }

    /// Get a reference to the output texture view from the most recent decode_to_texture call.
    /// Returns None if no frame has been decoded yet.
    pub fn output_texture_view(&self) -> Option<std::cell::Ref<'_, wgpu::TextureView>> {
        let cached = self.cached.borrow();
        if cached.is_none() {
            return None;
        }
        Some(std::cell::Ref::map(cached, |c| {
            &c.as_ref().unwrap().output_texture_view
        }))
    }

    /// Submit GPU decode work without waiting for the result.
    /// Returns an opaque token that can be used with `finish_decode_u8` to get the result.
    /// This enables pipelined decode: submit frame N, do CPU work for frame N+1,
    /// then finish frame N's readback.
    pub fn submit_decode_u8(&self, ctx: &GpuContext, frame: &CompressedFrame) {
        let info = &frame.info;
        let w = info.width;
        let h = info.height;
        let total_f32s = w * h * 3;
        let packed_u32s = total_f32s.div_ceil(4);
        let packed_byte_size = (packed_u32s as u64) * 4;

        self.ensure_cached(
            ctx,
            info.padded_width(),
            info.padded_height(),
            w,
            h,
            info.tile_size,
        );

        self.prepare_frame_data(ctx, frame);

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let mut cmd = self.encode_gpu_work(ctx, frame, bufs);

        // GPU pack: f32 → packed u8 (using cached pack_params_buf)
        {
            let pack_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pack_bg"),
                layout: &self.pack_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bufs.pack_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bufs.cropped_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bufs.packed_u8_buf.as_entire_binding(),
                    },
                ],
            });

            let workgroups = packed_u32s.div_ceil(256);
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pack_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &pack_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy packed u8 to staging
        cmd.copy_buffer_to_buffer(
            &bufs.packed_u8_buf,
            0,
            &bufs.staging_u8,
            0,
            packed_byte_size,
        );

        ctx.queue.submit(Some(cmd.finish()));

        // Request map (non-blocking — will be ready after poll)
        let slice = bufs.staging_u8.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        drop(cached);

        // Store the receiver for later retrieval
        *self.pending_rx.borrow_mut() = Some(rx);
    }

    /// Finish a previously submitted decode_u8 operation.
    /// Blocks until the GPU work is complete and returns the u8 result.
    pub fn finish_decode_u8(&self, ctx: &GpuContext, width: u32, height: u32) -> Vec<u8> {
        let rx = self
            .pending_rx
            .borrow_mut()
            .take()
            .expect("finish_decode_u8 called without prior submit_decode_u8");

        let total_bytes = (width * height * 3) as usize;

        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let slice = bufs.staging_u8.slice(..);
        let data = slice.get_mapped_range();
        let bytes: &[u8] = &data;
        let result = bytes[..total_bytes].to_vec();
        drop(data);
        bufs.staging_u8.unmap();
        drop(cached);

        result
    }
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
