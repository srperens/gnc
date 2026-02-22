use std::cell::RefCell;
use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::encoder::color::ColorConverter;
use crate::encoder::interleave::PlaneInterleaver;
use crate::encoder::quantize::Quantizer;
use crate::encoder::rans_gpu::GpuRansDecoder;
use crate::encoder::transform::WaveletTransform;
use crate::{CompressedFrame, GpuContext};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CropParams {
    src_width: u32,
    dst_width: u32,
    dst_height: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PackParams {
    total_f32s: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Cached GPU buffers for decode — allocated once, reused across frames.
struct CachedBuffers {
    padded_w: u32,
    padded_h: u32,
    width: u32,
    height: u32,
    scratch_a: wgpu::Buffer,
    scratch_b: wgpu::Buffer,
    scratch_c: wgpu::Buffer,
    plane_results: [wgpu::Buffer; 3],
    ycocg_buf: wgpu::Buffer,
    rgb_out_buf: wgpu::Buffer,
    cropped_buf: wgpu::Buffer,
    packed_u8_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    staging_u8: wgpu::Buffer,
}

/// Full decoding pipeline: GPU rANS Decode -> Dequantize -> Inverse Wavelet -> Interleave -> Inverse Color -> Crop
///
/// All GPU stages run without CPU readback until the final RGB output.
/// GPU buffers are cached across frames for zero-allocation steady-state decode.
pub struct DecoderPipeline {
    color: ColorConverter,
    transform: WaveletTransform,
    quantize: Quantizer,
    rans_decoder: GpuRansDecoder,
    interleaver: PlaneInterleaver,
    crop_pipeline: wgpu::ComputePipeline,
    crop_bgl: wgpu::BindGroupLayout,
    pack_pipeline: wgpu::ComputePipeline,
    pack_bgl: wgpu::BindGroupLayout,
    cached: RefCell<Option<CachedBuffers>>,
}

impl DecoderPipeline {
    pub fn new(ctx: &GpuContext) -> Self {
        let crop_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("crop"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/crop.wgsl").into(),
                ),
            });

        let crop_bgl =
            ctx.device
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

        let crop_pl =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("crop_pl"),
                    bind_group_layouts: &[&crop_bgl],
                    push_constant_ranges: &[],
                });

        let crop_pipeline =
            ctx.device
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
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/pack_u8.wgsl").into(),
                ),
            });

        let pack_bgl =
            ctx.device
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

        let pack_pl =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pack_pl"),
                    bind_group_layouts: &[&pack_bgl],
                    push_constant_ranges: &[],
                });

        let pack_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("pack_pipeline"),
                    layout: Some(&pack_pl),
                    module: &pack_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            color: ColorConverter::new(ctx),
            transform: WaveletTransform::new(ctx),
            quantize: Quantizer::new(ctx),
            rans_decoder: GpuRansDecoder::new(ctx),
            interleaver: PlaneInterleaver::new(ctx),
            crop_pipeline,
            crop_bgl,
            pack_pipeline,
            pack_bgl,
            cached: RefCell::new(None),
        }
    }

    fn ensure_cached(&self, ctx: &GpuContext, padded_w: u32, padded_h: u32, width: u32, height: u32) {
        let mut cached = self.cached.borrow_mut();
        if let Some(ref c) = *cached {
            if c.padded_w == padded_w && c.padded_h == padded_h && c.width == width && c.height == height {
                return;
            }
        }

        let padded_pixels = (padded_w * padded_h) as usize;
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let buf_size_3 = (padded_pixels * 3 * std::mem::size_of::<f32>()) as u64;
        let output_pixels = (width * height) as usize;
        let output_size = (output_pixels * 3 * std::mem::size_of::<f32>()) as u64;

        let scratch_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let scratch_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_scratch_a"),
            size: plane_size,
            usage: scratch_usage,
            mapped_at_creation: false,
        });
        let scratch_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_scratch_b"),
            size: plane_size,
            usage: scratch_usage,
            mapped_at_creation: false,
        });
        let scratch_c = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_scratch_c"),
            size: plane_size,
            usage: scratch_usage,
            mapped_at_creation: false,
        });

        let plane_results = std::array::from_fn(|p| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dec_plane_result_{p}")),
                size: plane_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let ycocg_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_ycocg"),
            size: buf_size_3,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let rgb_out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_rgb_out"),
            size: buf_size_3,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let cropped_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_cropped"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Packed u8 output: ceil(total_f32s / 4) u32s
        let total_f32s = output_pixels * 3;
        let packed_u32s = total_f32s.div_ceil(4) as u64;
        let packed_byte_size = packed_u32s * 4;
        let packed_u8_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_packed_u8"),
            size: packed_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_u8 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_staging_u8"),
            size: packed_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        *cached = Some(CachedBuffers {
            padded_w,
            padded_h,
            width,
            height,
            scratch_a,
            scratch_b,
            scratch_c,
            plane_results,
            ycocg_buf,
            rgb_out_buf,
            cropped_buf,
            packed_u8_buf,
            staging,
            staging_u8,
        });
    }

    /// Encode GPU commands for the full decode pipeline up to and including crop.
    /// Returns the command encoder with all work recorded, plus the crop params buffer
    /// (which must stay alive until submission).
    #[allow(clippy::too_many_arguments)]
    fn encode_gpu_work(
        &self,
        ctx: &GpuContext,
        frame: &CompressedFrame,
        bufs: &CachedBuffers,
        rans_bufs: &[(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)],
    ) -> (wgpu::CommandEncoder, wgpu::Buffer) {
        let info = &frame.info;
        let config = &frame.config;
        let padded_w = info.padded_width();
        let padded_h = info.padded_height();
        let padded_pixels = (padded_w * padded_h) as usize;
        let tiles_per_plane = info.tiles_x() as usize * info.tiles_y() as usize;

        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let w = info.width;
        let h = info.height;
        let output_pixels = (w * h) as u32;

        let crop_params = CropParams {
            src_width: padded_w,
            dst_width: w,
            dst_height: h,
            _pad: 0,
        };
        let crop_params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("crop_params"),
                contents: bytemuck::bytes_of(&crop_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("decode_full"),
            });

        // Per-plane: rANS → dequantize → inverse wavelet → copy to result buffer
        for p in 0..3 {
            let (ref params_buf, ref tile_info_buf, ref cumfreq_buf, ref stream_data_buf) =
                rans_bufs[p];

            self.rans_decoder.dispatch_decode(
                ctx,
                &mut cmd,
                params_buf,
                tile_info_buf,
                cumfreq_buf,
                stream_data_buf,
                &bufs.scratch_a,
                tiles_per_plane as u32,
            );

            let weights = if p == 0 { &weights_luma } else { &weights_chroma };
            self.quantize.dispatch(
                ctx,
                &mut cmd,
                &bufs.scratch_a,
                &bufs.scratch_b,
                padded_pixels as u32,
                config.quantization_step,
                config.dead_zone,
                false,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
                weights,
            );

            self.transform.inverse(
                ctx,
                &mut cmd,
                &bufs.scratch_b,
                &bufs.scratch_c,
                &bufs.scratch_a,
                info,
                config.wavelet_levels,
            );

            cmd.copy_buffer_to_buffer(
                &bufs.scratch_a,
                0,
                &bufs.plane_results[p],
                0,
                plane_size,
            );
        }

        // GPU interleave: 3 planes → interleaved YCoCg
        self.interleaver.dispatch(
            ctx,
            &mut cmd,
            &bufs.plane_results[0],
            &bufs.plane_results[1],
            &bufs.plane_results[2],
            &bufs.ycocg_buf,
            padded_pixels as u32,
        );

        // Inverse color (YCoCg-R → RGB)
        self.color.dispatch(
            ctx,
            &mut cmd,
            &bufs.ycocg_buf,
            &bufs.rgb_out_buf,
            padded_w,
            padded_h,
            false,
        );

        // GPU crop: padded RGB → compact cropped output
        {
            let crop_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("crop_bg"),
                layout: &self.crop_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: crop_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bufs.rgb_out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bufs.cropped_buf.as_entire_binding(),
                    },
                ],
            });

            let workgroups = output_pixels.div_ceil(256);
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("crop_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.crop_pipeline);
            pass.set_bind_group(0, &crop_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        (cmd, crop_params_buf)
    }

    /// Prepare rANS GPU buffers for all 3 planes.
    fn prepare_rans_bufs(
        &self,
        ctx: &GpuContext,
        frame: &CompressedFrame,
    ) -> Vec<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)> {
        let info = &frame.info;
        let tiles_per_plane = info.tiles_x() as usize * info.tiles_y() as usize;
        let mut rans_bufs = Vec::with_capacity(3);
        for p in 0..3 {
            let plane_tiles =
                &frame.tiles[p * tiles_per_plane..(p + 1) * tiles_per_plane];
            rans_bufs.push(
                self.rans_decoder
                    .prepare_decode_buffers(ctx, plane_tiles, info),
            );
        }
        rans_bufs
    }

    /// Decode a compressed frame back to RGB f32 data.
    /// Returns Vec<f32> of length width * height * 3 (interleaved R,G,B).
    pub fn decode(&self, ctx: &GpuContext, frame: &CompressedFrame) -> Vec<f32> {
        let profile = std::env::var("GNC_PROFILE").is_ok();
        let t_start = std::time::Instant::now();

        let info = &frame.info;
        let w = info.width;
        let h = info.height;
        let output_pixels = (w * h) as u32;
        let output_size = (output_pixels as u64) * 3 * 4;

        self.ensure_cached(ctx, info.padded_width(), info.padded_height(), w, h);

        let t_alloc = t_start.elapsed();

        let rans_bufs = self.prepare_rans_bufs(ctx, frame);

        let t_prepare = t_start.elapsed();

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let (mut cmd, _crop_params_buf) = self.encode_gpu_work(ctx, frame, bufs, &rans_bufs);

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
        let total_f32s = (w * h * 3) as u32;
        let packed_u32s = total_f32s.div_ceil(4);
        let packed_byte_size = (packed_u32s as u64) * 4;

        self.ensure_cached(ctx, info.padded_width(), info.padded_height(), w, h);

        let t_alloc = t_start.elapsed();

        let rans_bufs = self.prepare_rans_bufs(ctx, frame);

        let t_prepare = t_start.elapsed();

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let (mut cmd, _crop_params_buf) = self.encode_gpu_work(ctx, frame, bufs, &rans_bufs);

        // GPU pack: f32 → packed u8
        let pack_params = PackParams {
            total_f32s,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let pack_params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pack_params"),
                contents: bytemuck::bytes_of(&pack_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        {
            let pack_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pack_bg"),
                layout: &self.pack_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pack_params_buf.as_entire_binding(),
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
        cmd.copy_buffer_to_buffer(&bufs.packed_u8_buf, 0, &bufs.staging_u8, 0, packed_byte_size);

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
        let t_poll = t_start.elapsed();
        rx.recv().unwrap().unwrap();
        let t_recv = t_start.elapsed();

        let data = slice.get_mapped_range();
        let bytes: &[u8] = &data;
        let result = bytes[..total_f32s as usize].to_vec();
        drop(data);
        let t_copy = t_start.elapsed();
        bufs.staging_u8.unmap();
        drop(cached);

        if profile {
            let t_total = t_start.elapsed();
            eprintln!(
                "[decode_u8 profile] alloc={:.2}ms prepare={:.2}ms cmd={:.2}ms submit={:.2}ms poll={:.2}ms recv={:.2}ms copy={:.2}ms cleanup={:.2}ms total={:.2}ms",
                t_alloc.as_secs_f64() * 1000.0,
                (t_prepare - t_alloc).as_secs_f64() * 1000.0,
                (t_encode_cmd - t_prepare).as_secs_f64() * 1000.0,
                (t_submit - t_encode_cmd).as_secs_f64() * 1000.0,
                (t_poll - t_submit).as_secs_f64() * 1000.0,
                (t_recv - t_poll).as_secs_f64() * 1000.0,
                (t_copy - t_recv).as_secs_f64() * 1000.0,
                (t_total - t_copy).as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
            );
        }

        result
    }
}
