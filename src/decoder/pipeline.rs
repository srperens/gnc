use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;
use wgpu;
use wgpu::util::DeviceExt;

use crate::encoder::adaptive::AQ_BLOCK_SIZE;
use crate::encoder::bitplane::GpuBitplaneDecoder;
use crate::encoder::cfl::{self, CflPredictor};
use crate::encoder::color::ColorConverter;
use crate::encoder::interleave::PlaneInterleaver;
use crate::encoder::motion::MotionEstimator;
use crate::encoder::quantize::Quantizer;
use crate::encoder::rans_gpu::GpuRansDecoder;
use crate::encoder::transform::WaveletTransform;
use crate::{CompressedFrame, EntropyData, FrameType, GpuContext};

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
    /// Dequantized Y wavelet coefficients for CfL prediction
    y_ref_wavelet_buf: wgpu::Buffer,
    /// Reference planes from previous decoded frame (for temporal prediction)
    reference_planes: [wgpu::Buffer; 3],
}

/// Prepared entropy decode buffers — either rANS or bitplane, ready for GPU dispatch.
enum PreparedEntropyBufs {
    /// Per-plane: (params, tile_info, cumfreq, stream_data)
    Rans(Vec<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer)>),
    /// Per-plane: ((params, tile_info, block_info, bitplane_data), total_blocks)
    Bitplane(
        Vec<(
            (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer),
            u32,
        )>,
    ),
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
    bitplane_decoder: GpuBitplaneDecoder,
    interleaver: PlaneInterleaver,
    cfl_predictor: CflPredictor,
    motion: MotionEstimator,
    crop_pipeline: wgpu::ComputePipeline,
    crop_bgl: wgpu::BindGroupLayout,
    pack_pipeline: wgpu::ComputePipeline,
    pack_bgl: wgpu::BindGroupLayout,
    cached: RefCell<Option<CachedBuffers>>,
    pending_rx: RefCell<Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
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
            cached: RefCell::new(None),
            pending_rx: RefCell::new(None),
        }
    }

    fn ensure_cached(
        &self,
        ctx: &GpuContext,
        padded_w: u32,
        padded_h: u32,
        width: u32,
        height: u32,
    ) {
        let mut cached = self.cached.borrow_mut();
        if let Some(ref c) = *cached {
            if c.padded_w == padded_w
                && c.padded_h == padded_h
                && c.width == width
                && c.height == height
            {
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
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
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

        let y_ref_wavelet_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_y_ref_wavelet"),
            size: plane_size,
            usage: scratch_usage,
            mapped_at_creation: false,
        });

        let reference_planes = std::array::from_fn(|p| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dec_reference_{p}")),
                size: plane_size,
                usage: scratch_usage,
                mapped_at_creation: false,
            })
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
            y_ref_wavelet_buf,
            reference_planes,
        });
    }

    /// Upload the weight map from the compressed frame to GPU, if present.
    fn upload_weight_map(
        &self,
        ctx: &GpuContext,
        frame: &CompressedFrame,
    ) -> Option<(wgpu::Buffer, u32)> {
        let wm = frame.weight_map.as_ref()?;
        let padded_w = frame.info.padded_width();
        let blocks_x = (padded_w + AQ_BLOCK_SIZE - 1) / AQ_BLOCK_SIZE;

        let buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dec_weight_map"),
                contents: bytemuck::cast_slice(wm),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Some((buf, blocks_x))
    }

    /// Upload motion vectors from compressed frame to GPU, if present.
    fn upload_motion_vectors(
        &self,
        ctx: &GpuContext,
        frame: &CompressedFrame,
    ) -> Option<wgpu::Buffer> {
        let mf = frame.motion_field.as_ref()?;
        Some(MotionEstimator::upload_motion_vectors(ctx, &mf.vectors))
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
        entropy_bufs: &PreparedEntropyBufs,
        weight_map_gpu: &Option<(wgpu::Buffer, u32)>,
        mv_buf: &Option<wgpu::Buffer>,
    ) -> (wgpu::CommandEncoder, wgpu::Buffer) {
        let info = &frame.info;
        let config = &frame.config;
        let padded_w = info.padded_width();
        let padded_h = info.padded_height();
        let padded_pixels = (padded_w * padded_h) as usize;
        let is_pframe = frame.frame_type == FrameType::Predicted;
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

        // Prepare CfL alpha buffer if needed
        let has_cfl = frame.cfl_alphas.is_some();
        let cfl_alpha_buf = if has_cfl {
            let cfl_data = frame.cfl_alphas.as_ref().unwrap();
            let nsb = cfl_data.num_subbands;
            let total_tiles = info.tiles_x() as usize * info.tiles_y() as usize;
            // Dequantize all alphas to f32 for the GPU shader
            // Layout: [Co alphas for all tiles][Cg alphas for all tiles]
            // Each chroma plane's alphas = total_tiles * nsb values
            let alphas_per_plane = total_tiles * nsb as usize;
            let all_f32: Vec<f32> = cfl_data
                .alphas
                .iter()
                .map(|&q| cfl::dequantize_alpha(q))
                .collect();
            // We'll index into this per plane during dispatch
            Some((cfl::upload_alpha_buffer(ctx, &all_f32), alphas_per_plane))
        } else {
            None
        };

        // Per-plane: entropy decode → dequantize → (CfL inverse predict) → inverse wavelet → copy to result buffer
        for p in 0..3 {
            match entropy_bufs {
                PreparedEntropyBufs::Rans(ref rans_bufs) => {
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
                }
                PreparedEntropyBufs::Bitplane(ref bp_bufs) => {
                    let (ref params_buf, ref tile_info_buf, ref block_info_buf, ref data_buf) =
                        bp_bufs[p].0;
                    self.bitplane_decoder.dispatch_decode(
                        ctx,
                        &mut cmd,
                        params_buf,
                        tile_info_buf,
                        block_info_buf,
                        data_buf,
                        &bufs.scratch_a,
                        bp_bufs[p].1,
                    );
                }
            }

            let weights = if p == 0 {
                &weights_luma
            } else {
                &weights_chroma
            };
            let wm_param = weight_map_gpu
                .as_ref()
                .map(|(buf, blocks_x)| (buf, AQ_BLOCK_SIZE, *blocks_x));
            self.quantize.dispatch_adaptive(
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
                wm_param,
            );

            if p == 0 && has_cfl {
                // Save dequantized Y wavelet for CfL chroma prediction
                cmd.copy_buffer_to_buffer(
                    &bufs.scratch_b,
                    0,
                    &bufs.y_ref_wavelet_buf,
                    0,
                    plane_size,
                );
            }

            if p > 0 && has_cfl {
                // CfL inverse prediction: scratch_b (dequantized residual) + alpha * y_ref → scratch_c
                let (ref alpha_buf, alphas_per_plane) = *cfl_alpha_buf.as_ref().unwrap();
                // Plane 1 (Co) alphas start at offset 0, plane 2 (Cg) at offset alphas_per_plane
                let plane_alpha_offset = (p - 1) * alphas_per_plane;
                let plane_alpha_byte_offset =
                    (plane_alpha_offset * std::mem::size_of::<f32>()) as u64;
                let plane_alpha_byte_size = (alphas_per_plane * std::mem::size_of::<f32>()) as u64;

                // Create a sub-buffer view for this plane's alphas
                let plane_alpha_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cfl_plane_alpha"),
                    size: plane_alpha_byte_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                cmd.copy_buffer_to_buffer(
                    alpha_buf,
                    plane_alpha_byte_offset,
                    &plane_alpha_buf,
                    0,
                    plane_alpha_byte_size,
                );

                self.cfl_predictor.dispatch_inverse(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_b,         // dequantized residual
                    &bufs.y_ref_wavelet_buf, // reconstructed Y wavelet
                    &plane_alpha_buf,        // per-tile per-subband alphas
                    &bufs.scratch_c,         // output: reconstructed chroma wavelet
                    padded_pixels as u32,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                );

                // Inverse wavelet: scratch_c → (temp scratch_b) → scratch_a
                self.transform.inverse(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_c,
                    &bufs.scratch_b,
                    &bufs.scratch_a,
                    info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );
            } else {
                // Standard path: inverse wavelet from scratch_b
                self.transform.inverse(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_b,
                    &bufs.scratch_c,
                    &bufs.scratch_a,
                    info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );
            }

            if is_pframe {
                // P-frame: scratch_a has residual, add MC prediction from reference
                // recon = residual + MC(reference)
                // scratch_a (residual) + reference_planes[p] → plane_results[p]
                self.motion.compensate(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_a,
                    &bufs.reference_planes[p],
                    mv_buf.as_ref().unwrap(),
                    &bufs.plane_results[p],
                    padded_w,
                    padded_h,
                    false, // inverse: recon = residual + predicted
                );
            } else {
                // I-frame: scratch_a has reconstructed spatial data
                cmd.copy_buffer_to_buffer(
                    &bufs.scratch_a,
                    0,
                    &bufs.plane_results[p],
                    0,
                    plane_size,
                );
            }

            // Copy reconstructed plane to reference buffer for next frame
            cmd.copy_buffer_to_buffer(
                &bufs.plane_results[p],
                0,
                &bufs.reference_planes[p],
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

    /// Prepare entropy decode GPU buffers for all 3 planes.
    fn prepare_entropy_bufs(
        &self,
        ctx: &GpuContext,
        frame: &CompressedFrame,
    ) -> PreparedEntropyBufs {
        let info = &frame.info;
        let tiles_per_plane = info.tiles_x() as usize * info.tiles_y() as usize;
        let blocks_per_tile_side = info.tile_size as usize / 32;
        let blocks_per_tile = blocks_per_tile_side * blocks_per_tile_side;

        match &frame.entropy {
            EntropyData::Rans(tiles) => {
                let mut rans_bufs = Vec::with_capacity(3);
                for p in 0..3 {
                    let plane_tiles = &tiles[p * tiles_per_plane..(p + 1) * tiles_per_plane];
                    rans_bufs.push(self.rans_decoder.prepare_decode_buffers(
                        ctx,
                        plane_tiles,
                        info,
                    ));
                }
                PreparedEntropyBufs::Rans(rans_bufs)
            }
            EntropyData::SubbandRans(tiles) => {
                let mut rans_bufs = Vec::with_capacity(3);
                for p in 0..3 {
                    let plane_tiles = &tiles[p * tiles_per_plane..(p + 1) * tiles_per_plane];
                    rans_bufs.push(self.rans_decoder.prepare_decode_buffers_subband(
                        ctx,
                        plane_tiles,
                        info,
                    ));
                }
                PreparedEntropyBufs::Rans(rans_bufs)
            }
            EntropyData::Bitplane(tiles) => {
                let mut bp_bufs = Vec::with_capacity(3);
                for p in 0..3 {
                    let plane_tiles = &tiles[p * tiles_per_plane..(p + 1) * tiles_per_plane];
                    let total_blocks = (plane_tiles.len() * blocks_per_tile) as u32;
                    let bufs = self
                        .bitplane_decoder
                        .prepare_decode_buffers(ctx, plane_tiles, info);
                    bp_bufs.push((bufs, total_blocks));
                }
                PreparedEntropyBufs::Bitplane(bp_bufs)
            }
        }
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

        let entropy_bufs = self.prepare_entropy_bufs(ctx, frame);
        let weight_map_gpu = self.upload_weight_map(ctx, frame);
        let mv_buf = self.upload_motion_vectors(ctx, frame);

        let t_prepare = t_start.elapsed();

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let (mut cmd, _crop_params_buf) =
            self.encode_gpu_work(ctx, frame, bufs, &entropy_bufs, &weight_map_gpu, &mv_buf);

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

        let entropy_bufs = self.prepare_entropy_bufs(ctx, frame);
        let weight_map_gpu = self.upload_weight_map(ctx, frame);
        let mv_buf = self.upload_motion_vectors(ctx, frame);

        let t_prepare = t_start.elapsed();

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let (mut cmd, _crop_params_buf) =
            self.encode_gpu_work(ctx, frame, bufs, &entropy_bufs, &weight_map_gpu, &mv_buf);

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

    /// Submit GPU decode work without waiting for the result.
    /// Returns an opaque token that can be used with `finish_decode_u8` to get the result.
    /// This enables pipelined decode: submit frame N, do CPU work for frame N+1,
    /// then finish frame N's readback.
    pub fn submit_decode_u8(&self, ctx: &GpuContext, frame: &CompressedFrame) {
        let info = &frame.info;
        let w = info.width;
        let h = info.height;
        let total_f32s = (w * h * 3) as u32;
        let packed_u32s = total_f32s.div_ceil(4);
        let packed_byte_size = (packed_u32s as u64) * 4;

        self.ensure_cached(ctx, info.padded_width(), info.padded_height(), w, h);

        let entropy_bufs = self.prepare_entropy_bufs(ctx, frame);
        let weight_map_gpu = self.upload_weight_map(ctx, frame);
        let mv_buf = self.upload_motion_vectors(ctx, frame);

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let (mut cmd, _crop_params_buf) =
            self.encode_gpu_work(ctx, frame, bufs, &entropy_bufs, &weight_map_gpu, &mv_buf);

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
