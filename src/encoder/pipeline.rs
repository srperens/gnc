use wgpu;
use wgpu::util::DeviceExt;

use super::adaptive::{self, VarianceAnalyzer, WeightMapNormalizer, AQ_LL_BLOCK_SIZE};
use super::bitplane;
use super::buffer_cache::CachedEncodeBuffers;
use super::cfl::{self, CflAlphaComputer, CflForwardPredictor, CflPredictor};
use super::color::ColorConverter;
use super::entropy_helpers::{encode_entropy, EntropyMode};
use super::interleave::PlaneDeinterleaver;
use super::quantize::Quantizer;
use super::quantize_histogram_fused::FusedQuantizeHistogram;
use super::rans;
use super::rans_gpu_encode::GpuRansEncoder;
use super::rice;
use super::rice_gpu::GpuRiceEncoder;
use super::transform::WaveletTransform;
use crate::gpu_util::ensure_var_buf;
use crate::{
    CflAlphas, CodecConfig, CompressedFrame, EntropyCoder, EntropyData, FrameInfo, FrameType,
    GpuContext,
};

// Temporal coding (encode_sequence, encode_pframe, local_decode_iframe)
// is in the `sequence` sibling module which adds an `impl EncoderPipeline` block.

/// Full encoding pipeline: Color -> Wavelet -> (LL Variance Analysis) -> Quantize -> rANS Entropy
pub struct EncoderPipeline {
    pub(super) color: ColorConverter,
    pub(super) transform: WaveletTransform,
    pub(super) quantize: Quantizer,
    pub(super) variance: VarianceAnalyzer,
    pub(super) motion: super::motion::MotionEstimator,
    pub(super) gpu_encoder: GpuRansEncoder,
    pub(super) gpu_rice_encoder: GpuRiceEncoder,
    pub(super) deinterleaver: PlaneDeinterleaver,
    pub(super) weight_normalizer: WeightMapNormalizer,
    pub(super) cfl_alpha: CflAlphaComputer,
    pub(super) cfl_forward: CflForwardPredictor,
    pub(super) cfl_inverse: CflPredictor,
    pub(super) fused_qh: FusedQuantizeHistogram,
    pad_pipeline: wgpu::ComputePipeline,
    pad_bgl: wgpu::BindGroupLayout,
    pub(super) cached: Option<CachedEncodeBuffers>,
}

impl EncoderPipeline {
    pub fn new(ctx: &GpuContext) -> Self {
        // GPU pad pipeline
        let pad_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pad"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/pad.wgsl").into(),
                ),
            });
        let pad_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("pad_bgl"),
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
        let pad_pl =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pad_pl"),
                    bind_group_layouts: &[&pad_bgl],
                    push_constant_ranges: &[],
                });
        let pad_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("pad_pipeline"),
                    layout: Some(&pad_pl),
                    module: &pad_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            color: ColorConverter::new(ctx),
            transform: WaveletTransform::new(ctx),
            quantize: super::quantize::Quantizer::new(ctx),
            variance: VarianceAnalyzer::new(ctx),
            motion: super::motion::MotionEstimator::new(ctx),
            gpu_encoder: GpuRansEncoder::new(ctx),
            gpu_rice_encoder: GpuRiceEncoder::new(ctx),
            deinterleaver: PlaneDeinterleaver::new(ctx),
            weight_normalizer: WeightMapNormalizer::new(ctx),
            cfl_alpha: CflAlphaComputer::new(ctx),
            cfl_forward: CflForwardPredictor::new(ctx),
            cfl_inverse: CflPredictor::new(ctx),
            fused_qh: FusedQuantizeHistogram::new(ctx),
            pad_pipeline,
            pad_bgl,
            cached: None,
        }
    }

    /// Ensure cached buffers exist and match the given resolution.
    pub(super) fn ensure_cached(
        &mut self,
        ctx: &GpuContext,
        padded_w: u32,
        padded_h: u32,
        orig_w: u32,
        orig_h: u32,
    ) {
        let needs_alloc = match &self.cached {
            Some(c) => {
                c.padded_w != padded_w
                    || c.padded_h != padded_h
                    || c.orig_w != orig_w
                    || c.orig_h != orig_h
            }
            None => true,
        };
        if !needs_alloc {
            return;
        }
        self.cached = Some(CachedEncodeBuffers::new(
            ctx, padded_w, padded_h, orig_w, orig_h,
        ));
    }

    /// Encode an RGB frame.
    /// Input: &[f32] of length width * height * 3 (interleaved R,G,B).
    /// Values in [0, 255] for 8-bit or [0, 1023] for 10-bit.
    pub fn encode(
        &mut self,
        ctx: &GpuContext,
        rgb_data: &[f32],
        width: u32,
        height: u32,
        config: &CodecConfig,
    ) -> CompressedFrame {
        let profile = std::env::var("GNC_PROFILE").is_ok();
        let t_start = std::time::Instant::now();

        let info = FrameInfo {
            width,
            height,
            bit_depth: 8,
            tile_size: config.tile_size,
        };

        let padded_w = info.padded_width();
        let padded_h = info.padded_height();
        let padded_pixels = (padded_w * padded_h) as usize;
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;

        // Ensure cached buffers exist for this resolution
        self.ensure_cached(ctx, padded_w, padded_h, width, height);
        let bufs = self.cached.as_ref().unwrap();

        let t_setup = t_start.elapsed();

        // Upload raw (unpadded) input directly to GPU — GPU shader handles padding
        ctx.queue
            .write_buffer(&bufs.raw_input_buf, 0, bytemuck::cast_slice(rgb_data));

        let t_pad = t_start.elapsed();

        // ---- Submit 1: GPU pad + color convert + deinterleave ----
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encode_preprocess"),
            });

        // GPU pad: raw_input_buf -> input_buf (edge-replicate to tile alignment)
        {
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct PadParams {
                width: u32,
                height: u32,
                padded_w: u32,
                padded_h: u32,
            }
            let pad_params = PadParams {
                width,
                height,
                padded_w,
                padded_h,
            };
            let pad_params_buf =
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("pad_params"),
                        contents: bytemuck::bytes_of(&pad_params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            let pad_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pad_bg"),
                layout: &self.pad_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pad_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bufs.raw_input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bufs.input_buf.as_entire_binding(),
                    },
                ],
            });
            let total_padded_pixels = padded_w * padded_h;
            let workgroups = total_padded_pixels.div_ceil(256);
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pad_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pad_pipeline);
            pass.set_bind_group(0, &pad_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Color convert (RGB -> YCoCg-R): input_buf -> color_out (interleaved)
        self.color.dispatch(
            ctx,
            &mut cmd,
            &bufs.input_buf,
            &bufs.color_out,
            padded_w,
            padded_h,
            true,
            config.is_lossless(),
        );

        // GPU deinterleave: color_out -> plane_a(Y), co_plane(Co), cg_plane(Cg)
        self.deinterleaver.dispatch(
            ctx,
            &mut cmd,
            &bufs.color_out,
            &bufs.plane_a,
            &bufs.co_plane,
            &bufs.cg_plane,
            padded_pixels as u32,
        );

        ctx.queue.submit(Some(cmd.finish()));

        let t_preprocess = t_start.elapsed();

        // ---- Per-plane encoding ----
        let mut rans_tiles: Vec<rans::InterleavedRansTile> = Vec::new();
        let mut subband_tiles: Vec<rans::SubbandRansTile> = Vec::new();
        let mut bp_tiles: Vec<bitplane::BitplaneTile> = Vec::new();
        let mut rice_tiles: Vec<rice::RiceTile> = Vec::new();
        let entropy_mode = EntropyMode::from_config(config);
        let tile_size = config.tile_size as usize;
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;
        let use_gpu_encode = config.gpu_entropy_encode
            && config.entropy_coder != EntropyCoder::Bitplane
            && !config.context_adaptive;
        let use_gpu_rice = use_gpu_encode && config.entropy_coder == EntropyCoder::Rice;

        // Fused quantize+histogram: saves one full buffer read+write per plane.
        // Only applicable when GPU entropy encoding is active and CfL is off
        // (CfL needs separate quantize+dequantize for Y reconstruction).
        let use_fused_qh = config.use_fused_quantize_histogram
            && use_gpu_encode
            && !config.cfl_enabled;

        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();

        let nsb = cfl::num_subbands(config.wavelet_levels);
        let mut cfl_alphas_all: Vec<i16> = Vec::new();

        // Ensure CfL alpha buffers are large enough
        if config.cfl_enabled {
            let total_tiles = (tiles_x * tiles_y) as u32;
            let alpha_buf_size = (total_tiles * nsb) as u64 * std::mem::size_of::<f32>() as u64;
            let bufs = self.cached.as_mut().unwrap();
            ensure_var_buf(
                ctx,
                &mut bufs.raw_alpha,
                &mut bufs.raw_alpha_cap,
                alpha_buf_size,
                "enc_raw_alpha",
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            ensure_var_buf(
                ctx,
                &mut bufs.dq_alpha,
                &mut bufs.dq_alpha_cap,
                alpha_buf_size,
                "enc_dq_alpha",
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
        }

        // Ensure per-plane histogram buffers for fused quantize+histogram path
        if use_fused_qh {
            let total_tiles_for_hist = info.tiles_x() * info.tiles_y();
            let bufs = self.cached.as_mut().unwrap();
            bufs.ensure_fused_hist_bufs(ctx, total_tiles_for_hist);
        }

        let bufs = self.cached.as_ref().unwrap();

        // ---- Single command encoder for all 3 planes: wavelet + AQ + quantize ----
        // Dispatches execute sequentially within the encoder, so CfL dependencies
        // (chroma needs reconstructed Y) are naturally satisfied.
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encode_3plane"),
            });

        // CfL alpha staging buffers (created on demand, tiny ~2KB each)
        let total_tiles_u32 = (tiles_x * tiles_y) as u32;
        let alpha_count = if config.cfl_enabled {
            (total_tiles_u32 * nsb) as usize
        } else {
            0
        };
        let alpha_bytes = (alpha_count * std::mem::size_of::<f32>()) as u64;
        let mr = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let alpha_staging: [wgpu::Buffer; 2] = if config.cfl_enabled {
            std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(["alpha_stg_co", "alpha_stg_cg"][i]),
                    size: alpha_bytes.max(4),
                    usage: mr,
                    mapped_at_creation: false,
                })
            })
        } else {
            // Dummy buffers (never used)
            std::array::from_fn(|_| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("alpha_stg_dummy"),
                    size: 4,
                    usage: mr,
                    mapped_at_creation: false,
                })
            })
        };

        // --- Y plane: wavelet transform ---
        self.transform.forward(
            ctx,
            &mut cmd,
            &bufs.plane_a,
            &bufs.plane_b,
            &bufs.plane_c,
            &info,
            config.wavelet_levels,
            config.wavelet_type,
        );
        // After wavelet: plane_c has Y wavelet coefficients

        // --- Adaptive quantization: variance analysis on Y's LL subband ---
        // Must run AFTER wavelet transform so we read from wavelet-domain data.
        // The LL subband at the deepest level naturally represents spatial content.
        let aq_active = config.adaptive_quantization && config.aq_strength > 0.0;
        if aq_active {
            // Variance analysis reads from Y wavelet buffer (plane_c)
            self.variance.dispatch(
                ctx,
                &mut cmd,
                &bufs.plane_c,
                &bufs.variance_buf,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
            );

            // Weight map normalization on GPU.
            // Pass global grid dimensions (tiles_x * ll_blocks_per_tile) so the
            // 3x3 smoothing filter sees a coherent 2D layout across all tiles.
            let (ll_bx, ll_by, total_blocks, _, tiles_x_u32, tiles_y_u32) = adaptive::ll_block_dims(
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
            );
            let global_bx = ll_bx * tiles_x_u32;
            let global_by = ll_by * tiles_y_u32;
            self.weight_normalizer.dispatch(
                ctx,
                &mut cmd,
                &bufs.variance_buf,
                &bufs.wm_scratch,
                &bufs.weight_map_buf,
                global_bx,
                global_by,
                total_blocks,
                config.aq_strength,
            );
        }

        // --- Y plane: quantize ---
        // Precompute AQ dimensions (used for all 3 planes when AQ is active)
        let aq_dims = if aq_active {
            let (_, ll_bx, _, tx) = adaptive::weight_map_dims(
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
            );
            let ll_size = config.tile_size >> config.wavelet_levels;
            let ll_block_size = AQ_LL_BLOCK_SIZE.min(ll_size);
            Some((ll_block_size, ll_bx, tx))
        } else {
            None
        };

        // Build AQ weight map parameter for quantizer dispatch.
        // Uses the precomputed aq_dims and the cached weight_map_buf.
        let wm_param = aq_dims
            .as_ref()
            .map(|&(ll_bs, ll_bx, tx)| (&bufs.weight_map_buf, ll_bs, ll_bx, tx));

        if config.cfl_enabled {
            // Quantize + dequantize to get reconstructed Y wavelet for CfL
            // Must use dispatch_adaptive to match decoder's dequantization path
            self.quantize.dispatch_adaptive(
                ctx,
                &mut cmd,
                &bufs.plane_c,
                &bufs.plane_b,
                padded_pixels as u32,
                config.quantization_step,
                config.dead_zone,
                true,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
                &weights_luma,
                wm_param,
            );
            self.quantize.dispatch_adaptive(
                ctx,
                &mut cmd,
                &bufs.plane_b,
                &bufs.plane_a,
                padded_pixels as u32,
                config.quantization_step,
                config.dead_zone,
                false,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
                &weights_luma,
                wm_param,
            );
            cmd.copy_buffer_to_buffer(&bufs.plane_a, 0, &bufs.recon_y, 0, plane_size);
        } else if use_fused_qh {
            // Fused quantize + histogram: writes quantized to plane_b AND histogram
            // to fused_hist_bufs[0] in a single dispatch.
            let hist_bufs = bufs.fused_hist_bufs.as_ref().unwrap();
            self.fused_qh.dispatch(
                ctx,
                &mut cmd,
                &bufs.plane_c,
                &bufs.plane_b,
                &hist_bufs[0],
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
                config.quantization_step,
                config.dead_zone,
                &weights_luma,
                config.per_subband_entropy,
                1, // flags: disable ZRL (lean encoder)
                wm_param,
            );
        } else {
            self.quantize.dispatch_adaptive(
                ctx,
                &mut cmd,
                &bufs.plane_c,
                &bufs.plane_b,
                padded_pixels as u32,
                config.quantization_step,
                config.dead_zone,
                true,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
                &weights_luma,
                wm_param,
            );
        }
        // Persist Y quantized (plane_b) before Co's wavelet overwrites it
        cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.mc_out, 0, plane_size);

        // --- Co plane: wavelet + (CfL) + quantize ---
        self.transform.forward(
            ctx,
            &mut cmd,
            &bufs.co_plane,
            &bufs.plane_b,
            &bufs.plane_c,
            &info,
            config.wavelet_levels,
            config.wavelet_type,
        );

        if config.cfl_enabled {
            self.cfl_alpha.dispatch(
                ctx,
                &mut cmd,
                &bufs.recon_y,
                &bufs.plane_c,
                &bufs.raw_alpha,
                &bufs.dq_alpha,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
            );
            self.cfl_forward.dispatch(
                ctx,
                &mut cmd,
                &bufs.plane_c,
                &bufs.recon_y,
                &bufs.dq_alpha,
                &bufs.plane_a,
                padded_pixels as u32,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
            );
            self.quantize.dispatch_adaptive(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.plane_b,
                padded_pixels as u32,
                config.quantization_step,
                config.dead_zone,
                true,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
                &weights_chroma,
                wm_param,
            );
            // Preserve Co alpha before Cg overwrites raw_alpha
            cmd.copy_buffer_to_buffer(&bufs.raw_alpha, 0, &alpha_staging[0], 0, alpha_bytes);
        } else if use_fused_qh {
            let hist_bufs = bufs.fused_hist_bufs.as_ref().unwrap();
            self.fused_qh.dispatch(
                ctx,
                &mut cmd,
                &bufs.plane_c,
                &bufs.plane_b,
                &hist_bufs[1],
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
                config.quantization_step,
                config.dead_zone,
                &weights_chroma,
                config.per_subband_entropy,
                1, // flags: disable ZRL
                wm_param,
            );
        } else {
            self.quantize.dispatch_adaptive(
                ctx,
                &mut cmd,
                &bufs.plane_c,
                &bufs.plane_b,
                padded_pixels as u32,
                config.quantization_step,
                config.dead_zone,
                true,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
                &weights_chroma,
                wm_param,
            );
        }
        // Persist Co quantized (plane_b) before Cg's wavelet overwrites it
        cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.ref_upload, 0, plane_size);

        // --- Cg plane: wavelet + (CfL) + quantize ---
        self.transform.forward(
            ctx,
            &mut cmd,
            &bufs.cg_plane,
            &bufs.plane_b,
            &bufs.plane_c,
            &info,
            config.wavelet_levels,
            config.wavelet_type,
        );

        if config.cfl_enabled {
            self.cfl_alpha.dispatch(
                ctx,
                &mut cmd,
                &bufs.recon_y,
                &bufs.plane_c,
                &bufs.raw_alpha,
                &bufs.dq_alpha,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
            );
            self.cfl_forward.dispatch(
                ctx,
                &mut cmd,
                &bufs.plane_c,
                &bufs.recon_y,
                &bufs.dq_alpha,
                &bufs.plane_a,
                padded_pixels as u32,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
            );
            self.quantize.dispatch_adaptive(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.plane_b,
                padded_pixels as u32,
                config.quantization_step,
                config.dead_zone,
                true,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
                &weights_chroma,
                wm_param,
            );
            // Copy Cg alpha to staging for deferred readback
            cmd.copy_buffer_to_buffer(&bufs.raw_alpha, 0, &alpha_staging[1], 0, alpha_bytes);
        } else if use_fused_qh {
            let hist_bufs = bufs.fused_hist_bufs.as_ref().unwrap();
            self.fused_qh.dispatch(
                ctx,
                &mut cmd,
                &bufs.plane_c,
                &bufs.plane_b,
                &hist_bufs[2],
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
                config.quantization_step,
                config.dead_zone,
                &weights_chroma,
                config.per_subband_entropy,
                1, // flags: disable ZRL
                wm_param,
            );
        } else {
            self.quantize.dispatch_adaptive(
                ctx,
                &mut cmd,
                &bufs.plane_c,
                &bufs.plane_b,
                padded_pixels as u32,
                config.quantization_step,
                config.dead_zone,
                true,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
                &weights_chroma,
                wm_param,
            );
        }
        // Cg quantized stays in plane_b (last plane, no overwrite)

        // Add weight map copy to staging WITHIN this command encoder (deferred readback)
        let wm_total_blocks = if aq_active {
            let (total_blocks, _, _, _) = adaptive::weight_map_dims(
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
            );
            let wm_bytes = (total_blocks as u64) * std::mem::size_of::<f32>() as u64;
            cmd.copy_buffer_to_buffer(
                &bufs.weight_map_buf,
                0,
                &bufs.weight_map_staging,
                0,
                wm_bytes,
            );
            total_blocks as usize
        } else {
            0
        };

        let t_wavelet_quant = t_start.elapsed();

        // Single submit for all 3 planes + weight map copy (no blocking poll!)
        ctx.queue.submit(Some(cmd.finish()));

        let t_submit_wq = t_start.elapsed();

        // CPU entropy encode path: each plane reads from its persisted buffer
        if !use_gpu_encode {
            // Y from mc_out, Co from ref_upload, Cg from plane_b
            for (p, qbuf) in [&bufs.mc_out, &bufs.ref_upload, &bufs.plane_b]
                .iter()
                .enumerate()
            {
                let _ = p;
                encode_entropy(
                    &mut self.gpu_encoder,
                    ctx,
                    qbuf,
                    padded_pixels,
                    padded_w as usize,
                    tiles_x,
                    tiles_y,
                    tile_size,
                    &entropy_mode,
                    config,
                    use_gpu_encode,
                    &info,
                    &mut rans_tiles,
                    &mut subband_tiles,
                    &mut bp_tiles,
                    &mut rice_tiles,
                );
            }
        }

        // Batched 3-plane GPU entropy encode: single submit + single poll for all planes.
        // GPU queue guarantees sequential execution: wavelet+quantize finishes before
        // entropy starts reading quantized buffers. No intermediate poll needed.
        if use_gpu_rice {
            // GPU Rice encode: single-pass, 256 streams per tile
            let mut rt = self.gpu_rice_encoder.encode_3planes_to_tiles(
                ctx,
                [&bufs.mc_out, &bufs.ref_upload, &bufs.plane_b],
                &info,
                config.wavelet_levels,
            );
            rice_tiles.append(&mut rt);
        } else if use_gpu_encode && use_fused_qh {
            // Fused path: histograms already computed by fused quantize+histogram shader.
            // Skip histogram pass, go straight to normalize + encode.
            let hist_bufs = bufs.fused_hist_bufs.as_ref().unwrap();
            let (mut rt, mut st) = self.gpu_encoder.encode_3planes_skip_histogram(
                ctx,
                [&bufs.mc_out, &bufs.ref_upload, &bufs.plane_b],
                [&hist_bufs[0], &hist_bufs[1], &hist_bufs[2]],
                &info,
                config.per_subband_entropy,
                config.wavelet_levels,
            );
            rans_tiles.append(&mut rt);
            subband_tiles.append(&mut st);
        } else if use_gpu_encode {
            // Y quantized in mc_out, Co in ref_upload, Cg in plane_b
            let (mut rt, mut st) = self.gpu_encoder.encode_3planes_to_tiles(
                ctx,
                [&bufs.mc_out, &bufs.ref_upload, &bufs.plane_b],
                &info,
                config.per_subband_entropy,
                config.wavelet_levels,
            );
            rans_tiles.append(&mut rt);
            subband_tiles.append(&mut st);
        }

        let t_entropy = t_start.elapsed();

        // Deferred weight map readback (data was copied to staging in cmd2, already
        // submitted and completed as part of the wavelet+quantize + entropy GPU work)
        let weight_map = if aq_active && wm_total_blocks > 0 {
            let bufs = self.cached.as_ref().unwrap();
            let wm_bytes = (wm_total_blocks * std::mem::size_of::<f32>()) as u64;
            let (tx, rx) = std::sync::mpsc::channel();
            let tx_c = tx.clone();
            bufs.weight_map_staging
                .slice(..wm_bytes)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx_c.send(result).unwrap();
                });
            drop(tx);
            ctx.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let view = bufs.weight_map_staging.slice(..wm_bytes).get_mapped_range();
            let wm: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
            drop(view);
            bufs.weight_map_staging.unmap();
            Some(wm)
        } else {
            None
        };

        // Deferred CfL alpha readback
        if config.cfl_enabled {
            let (tx, rx) = std::sync::mpsc::channel();
            for stg in &alpha_staging {
                let tx_c = tx.clone();
                stg.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    tx_c.send(result).unwrap();
                });
            }
            drop(tx);
            ctx.device.poll(wgpu::Maintain::Wait);
            for _ in 0..2 {
                rx.recv().unwrap().unwrap();
            }
            for stg in &alpha_staging {
                let view = stg.slice(..).get_mapped_range();
                let raw_alphas: &[i32] = bytemuck::cast_slice(&view);
                let q_alphas: Vec<i16> = raw_alphas.iter().map(|&a| a as i16).collect();
                cfl_alphas_all.extend_from_slice(&q_alphas);
                drop(view);
                stg.unmap();
            }
        }

        let cfl_alphas = if config.cfl_enabled {
            Some(CflAlphas {
                alphas: cfl_alphas_all,
                num_subbands: nsb,
            })
        } else {
            None
        };

        let entropy = match entropy_mode {
            EntropyMode::Bitplane => EntropyData::Bitplane(bp_tiles),
            EntropyMode::SubbandRans | EntropyMode::SubbandRansCtx => {
                EntropyData::SubbandRans(subband_tiles)
            }
            EntropyMode::Rans => EntropyData::Rans(rans_tiles),
            EntropyMode::Rice => EntropyData::Rice(rice_tiles),
        };

        if profile {
            let t_total = t_start.elapsed();
            eprintln!(
                "[encode profile] setup={:.2}ms pad={:.2}ms preprocess={:.2}ms wq_cmd={:.2}ms wq_submit={:.2}ms entropy={:.2}ms total={:.2}ms",
                t_setup.as_secs_f64() * 1000.0,
                (t_pad - t_setup).as_secs_f64() * 1000.0,
                (t_preprocess - t_pad).as_secs_f64() * 1000.0,
                (t_wavelet_quant - t_preprocess).as_secs_f64() * 1000.0,
                (t_submit_wq - t_wavelet_quant).as_secs_f64() * 1000.0,
                (t_entropy - t_submit_wq).as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
            );
        }

        CompressedFrame {
            info,
            config: config.clone(),
            entropy,
            cfl_alphas,
            weight_map,
            frame_type: FrameType::Intra,
            motion_field: None,
        }
    }
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
