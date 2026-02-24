use wgpu;

use super::adaptive::{self, VarianceAnalyzer, WeightMapNormalizer, AQ_BLOCK_SIZE};
use super::bitplane;
use super::cfl::{self, CflAlphaComputer, CflForwardPredictor};
use super::color::ColorConverter;
use super::interleave::PlaneDeinterleaver;
use super::motion::{MotionEstimator, ME_BLOCK_SIZE};
use super::quantize::Quantizer;
use super::rans;
use super::rans_gpu_encode::GpuRansEncoder;
use super::transform::WaveletTransform;
use crate::{
    CflAlphas, CodecConfig, CompressedFrame, EntropyCoder, EntropyData, FrameInfo, FrameType,
    GpuContext, MotionField,
};

/// Which entropy path to use (resolved from CodecConfig).
enum EntropyMode {
    Rans,
    SubbandRans,
    Bitplane,
}

/// Cached GPU buffers reused across encode() calls to avoid per-frame allocation.
struct CachedEncodeBuffers {
    // Resolution these buffers were allocated for
    padded_w: u32,
    padded_h: u32,

    // 3-channel buffers (size = 3 * plane_size)
    input_buf: wgpu::Buffer,
    color_out: wgpu::Buffer,

    // Single-plane work buffers (size = plane_size each)
    plane_a: wgpu::Buffer,
    plane_b: wgpu::Buffer,
    plane_c: wgpu::Buffer,
    co_plane: wgpu::Buffer,
    cg_plane: wgpu::Buffer,
    recon_y: wgpu::Buffer,

    // Adaptive quantization (size = wm_buf_size, fixed for given resolution)
    variance_buf: wgpu::Buffer,
    wm_scratch: wgpu::Buffer,
    weight_map_buf: wgpu::Buffer,

    // CfL alpha buffers (variable size, 2x growth)
    raw_alpha: wgpu::Buffer,
    dq_alpha: wgpu::Buffer,
    alpha_cap: u64,

    // P-frame work buffers (size = plane_size each)
    mc_out: wgpu::Buffer,
    ref_upload: wgpu::Buffer,
    recon_out: wgpu::Buffer,
}

/// Full encoding pipeline: Color -> (Variance Analysis) -> Wavelet -> Quantize -> rANS Entropy
pub struct EncoderPipeline {
    color: ColorConverter,
    transform: WaveletTransform,
    quantize: Quantizer,
    variance: VarianceAnalyzer,
    motion: MotionEstimator,
    gpu_encoder: GpuRansEncoder,
    deinterleaver: PlaneDeinterleaver,
    weight_normalizer: WeightMapNormalizer,
    cfl_alpha: CflAlphaComputer,
    cfl_forward: CflForwardPredictor,
    cached: Option<CachedEncodeBuffers>,
}

impl EncoderPipeline {
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            color: ColorConverter::new(ctx),
            transform: WaveletTransform::new(ctx),
            quantize: Quantizer::new(ctx),
            variance: VarianceAnalyzer::new(ctx),
            motion: MotionEstimator::new(ctx),
            gpu_encoder: GpuRansEncoder::new(ctx),
            deinterleaver: PlaneDeinterleaver::new(ctx),
            weight_normalizer: WeightMapNormalizer::new(ctx),
            cfl_alpha: CflAlphaComputer::new(ctx),
            cfl_forward: CflForwardPredictor::new(ctx),
            cached: None,
        }
    }

    /// Ensure cached buffers exist and match the given padded resolution.
    /// Returns a mutable reference to the cached buffers.
    fn ensure_cached(&mut self, ctx: &GpuContext, padded_w: u32, padded_h: u32) {
        let needs_alloc = match &self.cached {
            Some(c) => c.padded_w != padded_w || c.padded_h != padded_h,
            None => true,
        };
        if !needs_alloc {
            return;
        }

        let padded_pixels = (padded_w * padded_h) as usize;
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let buf_size_3 = (padded_pixels * 3 * std::mem::size_of::<f32>()) as u64;

        let (blocks_x, blocks_y, total_blocks) = adaptive::weight_map_dims(padded_w, padded_h);
        let _ = (blocks_x, blocks_y);
        let wm_buf_size = (total_blocks as usize * std::mem::size_of::<f32>()) as u64;

        // Initial alpha capacity: generous default
        let alpha_init_cap = 4096u64;

        let plane_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        self.cached = Some(CachedEncodeBuffers {
            padded_w,
            padded_h,

            input_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_input"),
                size: buf_size_3,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            color_out: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_color_out"),
                size: buf_size_3,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),

            plane_a: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_plane_a"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),
            plane_b: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_plane_b"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),
            plane_c: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_plane_c"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),
            co_plane: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_co_plane"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),
            cg_plane: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_cg_plane"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),
            recon_y: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_recon_y"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),

            variance_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_variance"),
                size: wm_buf_size.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            wm_scratch: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_wm_scratch"),
                size: wm_buf_size.max(4),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            weight_map_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_weight_map"),
                size: wm_buf_size.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),

            raw_alpha: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_raw_alpha"),
                size: alpha_init_cap,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            dq_alpha: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_dq_alpha"),
                size: alpha_init_cap,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            alpha_cap: alpha_init_cap,

            mc_out: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_mc_out"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),
            ref_upload: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_ref_upload"),
                size: plane_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            recon_out: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_recon_out"),
                size: plane_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        });
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
        self.ensure_cached(ctx, padded_w, padded_h);
        let bufs = self.cached.as_ref().unwrap();

        // Pad input to tile-aligned dimensions
        let padded_rgb = pad_frame(rgb_data, width, height, padded_w, padded_h);

        // Upload input
        ctx.queue
            .write_buffer(&bufs.input_buf, 0, bytemuck::cast_slice(&padded_rgb));

        // ---- Submit 1: color convert + deinterleave + (variance + weight normalize) ----
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encode_preprocess"),
            });

        // Color convert (RGB -> YCoCg-R): input_buf -> color_out (interleaved)
        self.color.dispatch(
            ctx,
            &mut cmd,
            &bufs.input_buf,
            &bufs.color_out,
            padded_w,
            padded_h,
            true,
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

        // Adaptive quantization: variance + weight normalization on GPU
        let weight_map = if config.adaptive_quantization && config.aq_strength > 0.0 {
            let (blocks_x, blocks_y, total_blocks) = adaptive::weight_map_dims(padded_w, padded_h);

            // Variance analysis reads Y from plane_a (already on GPU from deinterleave)
            self.variance.dispatch(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.variance_buf,
                padded_w,
                padded_h,
            );

            // Weight map normalization on GPU (replaces CPU compute_weight_map)
            self.weight_normalizer.dispatch(
                ctx,
                &mut cmd,
                &bufs.variance_buf,
                &bufs.wm_scratch,
                &bufs.weight_map_buf,
                blocks_x,
                blocks_y,
                config.aq_strength,
            );

            ctx.queue.submit(Some(cmd.finish()));

            // Small readback (~8KB for 1080p) for CompressedFrame serialization only
            let wm = read_buffer_f32_cached(ctx, &bufs.weight_map_buf, total_blocks as usize);
            Some(wm)
        } else {
            ctx.queue.submit(Some(cmd.finish()));
            None
        };

        // ---- Per-plane encoding ----
        let mut rans_tiles: Vec<rans::InterleavedRansTile> = Vec::new();
        let mut subband_tiles: Vec<rans::SubbandRansTile> = Vec::new();
        let mut bp_tiles: Vec<bitplane::BitplaneTile> = Vec::new();
        let entropy_mode = if config.entropy_coder == EntropyCoder::Bitplane {
            EntropyMode::Bitplane
        } else if config.per_subband_entropy {
            EntropyMode::SubbandRans
        } else {
            EntropyMode::Rans
        };
        let tile_size = config.tile_size as usize;
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;
        let use_gpu_encode =
            config.gpu_entropy_encode && config.entropy_coder != EntropyCoder::Bitplane;

        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();

        let nsb = cfl::num_subbands(config.wavelet_levels);
        let mut cfl_alphas_all: Vec<u8> = Vec::new();

        // Ensure CfL alpha buffers are large enough
        if config.cfl_enabled {
            let total_tiles = (tiles_x * tiles_y) as u32;
            let alpha_buf_size = (total_tiles * nsb) as u64 * std::mem::size_of::<f32>() as u64;
            let bufs = self.cached.as_mut().unwrap();
            ensure_var_buf(
                ctx,
                &mut bufs.raw_alpha,
                &mut bufs.alpha_cap,
                alpha_buf_size,
                "enc_raw_alpha",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            ensure_var_buf(
                ctx,
                &mut bufs.dq_alpha,
                &mut bufs.alpha_cap,
                alpha_buf_size,
                "enc_dq_alpha",
                wgpu::BufferUsages::STORAGE,
            );
        }
        let bufs = self.cached.as_ref().unwrap();

        for p in 0..3 {
            let is_chroma = p > 0;
            let use_cfl = config.cfl_enabled && is_chroma;

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encode_plane"),
                });

            if p == 0 {
                // Y plane: data already in plane_a from deinterleave

                // Wavelet forward: plane_a -> plane_b(scratch) -> plane_c
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

                if config.cfl_enabled {
                    // Y with CfL: quantize + dequantize to get reconstructed Y wavelet
                    self.quantize.dispatch(
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
                    );
                    self.quantize.dispatch(
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
                    );

                    // Copy recon Y to persistent buffer (stays on GPU for chroma CfL)
                    cmd.copy_buffer_to_buffer(&bufs.plane_a, 0, &bufs.recon_y, 0, plane_size);

                    // Save Y quantized for batched rANS encode
                    if use_gpu_encode {
                        cmd.copy_buffer_to_buffer(
                            &bufs.plane_b, 0, &bufs.mc_out, 0, plane_size,
                        );
                    }

                    ctx.queue.submit(Some(cmd.finish()));

                    if !use_gpu_encode {
                        Self::encode_entropy(
                            &mut self.gpu_encoder,
                            ctx,
                            &bufs.plane_b,
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
                        );
                    }
                } else {
                    // Non-CfL Y: adaptive quantize
                    let wm_param = if config.adaptive_quantization && config.aq_strength > 0.0 {
                        let (_, _, blocks_x) = adaptive::weight_map_dims(padded_w, padded_h);
                        Some((&bufs.weight_map_buf, AQ_BLOCK_SIZE, blocks_x))
                    } else {
                        None
                    };
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

                    if use_gpu_encode {
                        cmd.copy_buffer_to_buffer(
                            &bufs.plane_b, 0, &bufs.mc_out, 0, plane_size,
                        );
                    }

                    ctx.queue.submit(Some(cmd.finish()));

                    if !use_gpu_encode {
                        Self::encode_entropy(
                            &mut self.gpu_encoder,
                            ctx,
                            &bufs.plane_b,
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
                        );
                    }
                }
            } else if use_cfl {
                // Chroma plane with CfL: wavelet + alpha + forward + quantize all on GPU
                let chroma_source = if p == 1 {
                    &bufs.co_plane
                } else {
                    &bufs.cg_plane
                };

                // Wavelet forward: chroma_source -> plane_b(scratch) -> plane_c
                self.transform.forward(
                    ctx,
                    &mut cmd,
                    chroma_source,
                    &bufs.plane_b,
                    &bufs.plane_c,
                    &info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );

                // GPU CfL alpha computation
                let total_tiles = (tiles_x * tiles_y) as u32;
                let alpha_count = (total_tiles * nsb) as usize;

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

                // GPU CfL forward: residual = chroma_wavelet - alpha * recon_y -> plane_a
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

                // Quantize residual: plane_a -> plane_b
                self.quantize.dispatch(
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
                );

                // Save Co quantized for batched rANS encode
                if use_gpu_encode && p == 1 {
                    cmd.copy_buffer_to_buffer(
                        &bufs.plane_b, 0, &bufs.ref_upload, 0, plane_size,
                    );
                }

                ctx.queue.submit(Some(cmd.finish()));

                // Tiny readback of raw alphas (~few hundred bytes) for u8 serialization
                let raw_alphas = read_buffer_f32_cached(ctx, &bufs.raw_alpha, alpha_count);
                let q_alphas: Vec<u8> =
                    raw_alphas.iter().map(|&a| cfl::quantize_alpha(a)).collect();
                cfl_alphas_all.extend_from_slice(&q_alphas);

                if !use_gpu_encode {
                    Self::encode_entropy(
                        &mut self.gpu_encoder,
                        ctx,
                        &bufs.plane_b,
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
                    );
                }
            } else {
                // Non-CfL chroma: wavelet + quantize
                let chroma_source = if p == 1 {
                    &bufs.co_plane
                } else {
                    &bufs.cg_plane
                };

                // Wavelet forward: chroma_source -> plane_b(scratch) -> plane_c
                self.transform.forward(
                    ctx,
                    &mut cmd,
                    chroma_source,
                    &bufs.plane_b,
                    &bufs.plane_c,
                    &info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );

                let wm_param = if config.adaptive_quantization && config.aq_strength > 0.0 {
                    let (_, _, blocks_x) = adaptive::weight_map_dims(padded_w, padded_h);
                    Some((&bufs.weight_map_buf, AQ_BLOCK_SIZE, blocks_x))
                } else {
                    None
                };
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

                if use_gpu_encode && p == 1 {
                    cmd.copy_buffer_to_buffer(
                        &bufs.plane_b, 0, &bufs.ref_upload, 0, plane_size,
                    );
                }

                ctx.queue.submit(Some(cmd.finish()));

                if !use_gpu_encode {
                    Self::encode_entropy(
                        &mut self.gpu_encoder,
                        ctx,
                        &bufs.plane_b,
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
                    );
                }
            }
        }

        // Batched 3-plane rANS encode: single submit + single poll for all planes
        if use_gpu_encode {
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
            EntropyMode::SubbandRans => EntropyData::SubbandRans(subband_tiles),
            EntropyMode::Rans => EntropyData::Rans(rans_tiles),
        };

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

    /// Helper: entropy-encode a quantized plane buffer (GPU or CPU path).
    #[allow(clippy::too_many_arguments)]
    fn encode_entropy(
        gpu_encoder: &mut GpuRansEncoder,
        ctx: &GpuContext,
        quantized_buf: &wgpu::Buffer,
        padded_pixels: usize,
        padded_w: usize,
        tiles_x: usize,
        tiles_y: usize,
        tile_size: usize,
        entropy_mode: &EntropyMode,
        config: &CodecConfig,
        use_gpu_encode: bool,
        info: &FrameInfo,
        rans_tiles: &mut Vec<rans::InterleavedRansTile>,
        subband_tiles: &mut Vec<rans::SubbandRansTile>,
        bp_tiles: &mut Vec<bitplane::BitplaneTile>,
    ) {
        if use_gpu_encode {
            let (mut rt, mut st) = gpu_encoder.encode_plane_to_tiles(
                ctx,
                quantized_buf,
                info,
                config.per_subband_entropy,
                config.wavelet_levels,
            );
            rans_tiles.append(&mut rt);
            subband_tiles.append(&mut st);
        } else {
            let quantized = read_buffer_f32_cached(ctx, quantized_buf, padded_pixels);
            entropy_encode_tiles(
                &quantized,
                padded_w,
                tiles_x,
                tiles_y,
                tile_size,
                entropy_mode,
                config.tile_size,
                config.wavelet_levels,
                rans_tiles,
                subband_tiles,
                bp_tiles,
            );
        }
    }

    /// Encode a sequence of RGB frames with temporal prediction.
    ///
    /// Frame 0 (and every `keyframe_interval` frames) is encoded as an I-frame.
    /// Other frames are encoded as P-frames (residual from previous decoded frame).
    /// The encoder maintains a local decode loop so it uses the same reference
    /// as the decoder would.
    pub fn encode_sequence(
        &mut self,
        ctx: &GpuContext,
        frames: &[&[f32]],
        width: u32,
        height: u32,
        config: &CodecConfig,
    ) -> Vec<CompressedFrame> {
        let ki = config.keyframe_interval;
        let mut results = Vec::with_capacity(frames.len());
        let mut reference_planes: Option<[Vec<f32>; 3]> = None;

        let info = FrameInfo {
            width,
            height,
            bit_depth: 8,
            tile_size: config.tile_size,
        };
        let padded_w = info.padded_width();
        let padded_h = info.padded_height();
        let padded_pixels = (padded_w * padded_h) as usize;

        for (i, rgb_data) in frames.iter().enumerate() {
            let is_keyframe = ki <= 1 || i % ki as usize == 0 || reference_planes.is_none();

            if is_keyframe {
                let mut compressed = self.encode(ctx, rgb_data, width, height, config);
                compressed.frame_type = FrameType::Intra;

                // Local decode to get reference planes for subsequent P-frames
                let planes =
                    self.local_decode_iframe(ctx, &compressed, padded_w, padded_h, padded_pixels);
                reference_planes = Some(planes);
                results.push(compressed);
            } else {
                let ref_planes = reference_planes.as_ref().unwrap();
                let (compressed, new_ref) = self.encode_pframe(
                    ctx,
                    rgb_data,
                    ref_planes,
                    width,
                    height,
                    padded_w,
                    padded_h,
                    padded_pixels,
                    &info,
                    config,
                );
                reference_planes = Some(new_ref);
                results.push(compressed);
            }
        }

        results
    }

    /// I-frame local decode: entropy decode → dequantize → inverse wavelet → spatial planes.
    /// Reuses cached plane_a/b/c buffers (safe since encode() is done when this runs).
    fn local_decode_iframe(
        &self,
        ctx: &GpuContext,
        frame: &CompressedFrame,
        padded_w: u32,
        padded_h: u32,
        padded_pixels: usize,
    ) -> [Vec<f32>; 3] {
        let config = &frame.config;
        let info = &frame.info;
        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;
        let tiles_per_plane = tiles_x * tiles_y;

        let bufs = self.cached.as_ref().unwrap();

        let mut result_planes: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];

        for p in 0..3 {
            let quantized = entropy_decode_plane(
                &frame.entropy,
                p,
                tiles_per_plane,
                config.tile_size as usize,
                padded_w as usize,
            );

            ctx.queue
                .write_buffer(&bufs.plane_a, 0, bytemuck::cast_slice(&quantized));

            let weights = if p == 0 {
                &weights_luma
            } else {
                &weights_chroma
            };

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("local_decode"),
                });

            self.quantize.dispatch(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.plane_b,
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
                &bufs.plane_b,
                &bufs.plane_c,
                &bufs.plane_a,
                info,
                config.wavelet_levels,
                config.wavelet_type,
            );

            ctx.queue.submit(Some(cmd.finish()));
            result_planes[p] = read_buffer_f32_cached(ctx, &bufs.plane_a, padded_pixels);
        }

        result_planes
    }

    /// Encode a P-frame: ME -> MC -> encode residual -> local decode loop.
    #[allow(clippy::too_many_arguments)]
    fn encode_pframe(
        &mut self,
        ctx: &GpuContext,
        rgb_data: &[f32],
        ref_planes: &[Vec<f32>; 3],
        width: u32,
        height: u32,
        padded_w: u32,
        padded_h: u32,
        padded_pixels: usize,
        info: &FrameInfo,
        config: &CodecConfig,
    ) -> (CompressedFrame, [Vec<f32>; 3]) {
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;

        // Ensure cached buffers exist for this resolution
        self.ensure_cached(ctx, padded_w, padded_h);
        let bufs = self.cached.as_ref().unwrap();

        // Color convert + deinterleave on GPU
        let padded_rgb = pad_frame(rgb_data, width, height, padded_w, padded_h);
        ctx.queue
            .write_buffer(&bufs.input_buf, 0, bytemuck::cast_slice(&padded_rgb));

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pf_preprocess"),
            });
        self.color.dispatch(
            ctx,
            &mut cmd,
            &bufs.input_buf,
            &bufs.color_out,
            padded_w,
            padded_h,
            true,
        );
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

        // Block matching on Y plane (already on GPU as plane_a)
        // Upload reference Y to ref_upload buffer
        ctx.queue
            .write_buffer(&bufs.ref_upload, 0, bytemuck::cast_slice(&ref_planes[0]));

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pf_me"),
            });
        let (mv_buf, _sad_buf) = self.motion.estimate(
            ctx,
            &mut cmd,
            &bufs.plane_a,
            &bufs.ref_upload,
            padded_w,
            padded_h,
        );
        ctx.queue.submit(Some(cmd.finish()));

        let me_blocks_x = padded_w / ME_BLOCK_SIZE;
        let me_blocks_y = padded_h / ME_BLOCK_SIZE;
        let me_total_blocks = me_blocks_x * me_blocks_y;
        let mvs = MotionEstimator::read_motion_vectors(ctx, &mv_buf, me_total_blocks);
        let mv_buf_ro = MotionEstimator::upload_motion_vectors(ctx, &mvs);

        // MC + wavelet + quantize per plane, keeping data on GPU
        let entropy_mode = if config.entropy_coder == EntropyCoder::Bitplane {
            EntropyMode::Bitplane
        } else if config.per_subband_entropy {
            EntropyMode::SubbandRans
        } else {
            EntropyMode::Rans
        };
        let tile_size = config.tile_size as usize;
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;
        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();
        let use_gpu_encode =
            config.gpu_entropy_encode && config.entropy_coder != EntropyCoder::Bitplane;

        let mut rans_tiles: Vec<rans::InterleavedRansTile> = Vec::new();
        let mut subband_tiles: Vec<rans::SubbandRansTile> = Vec::new();
        let mut bp_tiles: Vec<bitplane::BitplaneTile> = Vec::new();

        for p in 0..3 {
            let weights = if p == 0 {
                &weights_luma
            } else {
                &weights_chroma
            };

            // Upload reference plane to cached ref_upload buffer
            ctx.queue
                .write_buffer(&bufs.ref_upload, 0, bytemuck::cast_slice(&ref_planes[p]));

            let cur_plane = if p == 0 {
                &bufs.plane_a
            } else if p == 1 {
                &bufs.co_plane
            } else {
                &bufs.cg_plane
            };

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_enc"),
                });

            // MC: current - reference -> mc_out (residual)
            self.motion.compensate(
                ctx,
                &mut cmd,
                cur_plane,
                &bufs.ref_upload,
                &mv_buf_ro,
                &bufs.mc_out,
                padded_w,
                padded_h,
                true,
            );

            // Copy residual to plane_a for wavelet input
            cmd.copy_buffer_to_buffer(&bufs.mc_out, 0, &bufs.plane_a, 0, plane_size);

            // Wavelet forward: plane_a -> plane_b(scratch) -> plane_c
            self.transform.forward(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.plane_b,
                &bufs.plane_c,
                info,
                config.wavelet_levels,
                config.wavelet_type,
            );

            // Quantize: plane_c -> plane_b
            self.quantize.dispatch(
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
                weights,
            );

            // Save quantized for batched rANS (Y→recon_y, Co→co_plane, Cg stays in plane_b)
            if use_gpu_encode && p == 0 {
                cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.recon_y, 0, plane_size);
            } else if use_gpu_encode && p == 1 {
                cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.co_plane, 0, plane_size);
            }

            ctx.queue.submit(Some(cmd.finish()));

            if !use_gpu_encode {
                Self::encode_entropy(
                    &mut self.gpu_encoder,
                    ctx,
                    &bufs.plane_b,
                    padded_pixels,
                    padded_w as usize,
                    tiles_x,
                    tiles_y,
                    tile_size,
                    &entropy_mode,
                    config,
                    use_gpu_encode,
                    info,
                    &mut rans_tiles,
                    &mut subband_tiles,
                    &mut bp_tiles,
                );
            }
        }

        // Batched 3-plane rANS encode for P-frame
        if use_gpu_encode {
            let (mut rt, mut st) = self.gpu_encoder.encode_3planes_to_tiles(
                ctx,
                [&bufs.recon_y, &bufs.co_plane, &bufs.plane_b],
                info,
                config.per_subband_entropy,
                config.wavelet_levels,
            );
            rans_tiles.append(&mut rt);
            subband_tiles.append(&mut st);
        }

        let entropy = match entropy_mode {
            EntropyMode::Bitplane => EntropyData::Bitplane(bp_tiles),
            EntropyMode::SubbandRans => EntropyData::SubbandRans(subband_tiles),
            EntropyMode::Rans => EntropyData::Rans(rans_tiles),
        };

        let compressed = CompressedFrame {
            info: *info,
            config: config.clone(),
            entropy,
            cfl_alphas: None,
            weight_map: None,
            frame_type: FrameType::Predicted,
            motion_field: Some(MotionField {
                vectors: mvs,
                block_size: ME_BLOCK_SIZE,
            }),
        };

        // Local decode loop: dequant -> inv wavelet -> MC inverse -> new reference
        // Keeps decoded residual on GPU (plane_a) to avoid readback+reupload.
        let tiles_per_plane = tiles_x * tiles_y;
        let mut recon_planes: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];

        for p in 0..3 {
            let quantized = entropy_decode_plane(
                &compressed.entropy,
                p,
                tiles_per_plane,
                tile_size,
                padded_w as usize,
            );

            ctx.queue
                .write_buffer(&bufs.plane_a, 0, bytemuck::cast_slice(&quantized));

            let weights = if p == 0 {
                &weights_luma
            } else {
                &weights_chroma
            };

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_ld"),
                });

            self.quantize.dispatch(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.plane_b,
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
                &bufs.plane_b,
                &bufs.plane_c,
                &bufs.plane_a,
                info,
                config.wavelet_levels,
                config.wavelet_type,
            );

            // Upload reference plane for inverse MC
            ctx.queue
                .write_buffer(&bufs.ref_upload, 0, bytemuck::cast_slice(&ref_planes[p]));

            // Inverse MC: recon = decoded_residual(plane_a) + MC(ref_upload) -> recon_out
            // Residual stays on GPU in plane_a — no readback+reupload needed.
            self.motion.compensate(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.ref_upload,
                &mv_buf_ro,
                &bufs.recon_out,
                padded_w,
                padded_h,
                false,
            );
            ctx.queue.submit(Some(cmd.finish()));
            recon_planes[p] = read_buffer_f32_cached(ctx, &bufs.recon_out, padded_pixels);
        }

        (compressed, recon_planes)
    }
}

/// Grow a variable-size cached buffer if the required size exceeds capacity (2× growth).
fn ensure_var_buf(
    ctx: &GpuContext,
    buf: &mut wgpu::Buffer,
    cap: &mut u64,
    required: u64,
    label: &str,
    usage: wgpu::BufferUsages,
) {
    if required > *cap {
        let new_cap = (required * 2).max(4);
        *buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: new_cap,
            usage,
            mapped_at_creation: false,
        });
        *cap = new_cap;
    }
}

/// Read a GPU buffer back to CPU as Vec<f32>, using a mapped_at_creation staging buffer
/// to avoid requiring a separate copy command submission.
fn read_buffer_f32_cached(ctx: &GpuContext, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let size = (count * std::mem::size_of::<f32>()) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_read"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut cmd = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_to_staging"),
        });
    cmd.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    ctx.queue.submit(Some(cmd.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

/// CPU entropy decode for a single plane: reconstruct quantized f32 coefficients from tiles.
fn entropy_decode_plane(
    entropy: &EntropyData,
    plane_idx: usize,
    tiles_per_plane: usize,
    tile_size: usize,
    padded_w: usize,
) -> Vec<f32> {
    let tile_start = plane_idx * tiles_per_plane;
    let tiles_x = padded_w / tile_size;
    let padded_h_tiles = tiles_per_plane / tiles_x;
    let padded_h = padded_h_tiles * tile_size;
    let total_pixels = padded_w * padded_h;
    let mut plane = vec![0.0f32; total_pixels];

    for t in 0..tiles_per_plane {
        let tx = t % tiles_x;
        let ty = t / tiles_x;

        let coeffs: Vec<i32> = match entropy {
            EntropyData::Rans(tiles) => rans::rans_decode_tile_interleaved(&tiles[tile_start + t]),
            EntropyData::SubbandRans(tiles) => {
                rans::rans_decode_tile_interleaved_subband(&tiles[tile_start + t])
            }
            EntropyData::Bitplane(tiles) => bitplane::bitplane_decode_tile(&tiles[tile_start + t]),
        };

        // Scatter tile coefficients back into flat plane
        for row in 0..tile_size {
            for col in 0..tile_size {
                let py = ty * tile_size + row;
                let px = tx * tile_size + col;
                plane[py * padded_w + px] = coeffs[row * tile_size + col] as f32;
            }
        }
    }

    plane
}

/// Entropy-encode all tiles from a quantized plane (CPU path).
#[allow(clippy::too_many_arguments)]
fn entropy_encode_tiles(
    quantized: &[f32],
    plane_width: usize,
    tiles_x: usize,
    tiles_y: usize,
    tile_size: usize,
    mode: &EntropyMode,
    tile_size_u32: u32,
    num_levels: u32,
    rans_tiles: &mut Vec<rans::InterleavedRansTile>,
    subband_tiles: &mut Vec<rans::SubbandRansTile>,
    bp_tiles: &mut Vec<bitplane::BitplaneTile>,
) {
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let coeffs = extract_tile_coefficients(quantized, plane_width, tx, ty, tile_size);
            match mode {
                EntropyMode::Bitplane => {
                    bp_tiles.push(bitplane::bitplane_encode_tile(&coeffs, tile_size_u32));
                }
                EntropyMode::SubbandRans => {
                    subband_tiles.push(rans::rans_encode_tile_interleaved_subband(
                        &coeffs,
                        tile_size_u32,
                        num_levels,
                    ));
                }
                EntropyMode::Rans => {
                    rans_tiles.push(rans::rans_encode_tile_interleaved_zrl(&coeffs));
                }
            }
        }
    }
}

/// Extract a tile's worth of coefficients from a flat plane array, converting f32 to i32.
fn extract_tile_coefficients(
    plane: &[f32],
    plane_width: usize,
    tile_x: usize,
    tile_y: usize,
    tile_size: usize,
) -> Vec<i32> {
    let mut coeffs = Vec::with_capacity(tile_size * tile_size);
    let origin_x = tile_x * tile_size;
    let origin_y = tile_y * tile_size;

    for y in 0..tile_size {
        for x in 0..tile_size {
            let idx = (origin_y + y) * plane_width + (origin_x + x);
            coeffs.push(plane[idx].round() as i32);
        }
    }
    coeffs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::pipeline::DecoderPipeline;

    fn compute_psnr(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mse: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = *x as f64 - *y as f64;
                d * d
            })
            .sum::<f64>()
            / a.len() as f64;
        if mse < 1e-10 {
            return 100.0;
        }
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    }

    /// Generate a synthetic RGB frame: smooth gradient with some detail.
    fn make_gradient_frame(w: u32, h: u32, offset: f32) -> Vec<f32> {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = ((x as f32 + offset) / w as f32 * 255.0).clamp(0.0, 255.0);
                let g = ((y as f32 + offset) / h as f32 * 255.0).clamp(0.0, 255.0);
                let b = (((x + y) as f32 + offset) / (w + h) as f32 * 255.0).clamp(0.0, 255.0);
                data.push(r);
                data.push(g);
                data.push(b);
            }
        }
        data
    }

    #[test]
    fn test_encode_sequence_all_iframes() {
        let ctx = GpuContext::new();
        let mut enc = EncoderPipeline::new(&ctx);

        let w = 256;
        let h = 256;
        let f0 = make_gradient_frame(w, h, 0.0);
        let f1 = make_gradient_frame(w, h, 5.0);
        let f2 = make_gradient_frame(w, h, 10.0);

        let mut config = CodecConfig::default();
        config.tile_size = 256;
        config.keyframe_interval = 1; // all I-frames

        let frames: Vec<&[f32]> = vec![&f0, &f1, &f2];
        let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);

        assert_eq!(compressed.len(), 3);
        for (i, cf) in compressed.iter().enumerate() {
            assert_eq!(
                cf.frame_type,
                FrameType::Intra,
                "frame {i} should be Intra with ki=1"
            );
            assert!(cf.motion_field.is_none(), "I-frame should have no MVs");
        }
    }

    #[test]
    fn test_encode_sequence_ip_pattern() {
        let ctx = GpuContext::new();
        let mut enc = EncoderPipeline::new(&ctx);

        let w = 256;
        let h = 256;
        let f0 = make_gradient_frame(w, h, 0.0);
        let f1 = make_gradient_frame(w, h, 3.0);
        let f2 = make_gradient_frame(w, h, 6.0);
        let f3 = make_gradient_frame(w, h, 9.0);

        let mut config = CodecConfig::default();
        config.tile_size = 256;
        config.keyframe_interval = 4; // I P P P

        let frames: Vec<&[f32]> = vec![&f0, &f1, &f2, &f3];
        let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);

        assert_eq!(compressed.len(), 4);
        assert_eq!(compressed[0].frame_type, FrameType::Intra);
        assert_eq!(compressed[1].frame_type, FrameType::Predicted);
        assert_eq!(compressed[2].frame_type, FrameType::Predicted);
        assert_eq!(compressed[3].frame_type, FrameType::Predicted);

        // P-frames must have motion fields
        for i in 1..4 {
            assert!(
                compressed[i].motion_field.is_some(),
                "P-frame {i} should have motion field"
            );
        }
    }

    #[test]
    fn test_pframe_roundtrip_quality() {
        let ctx = GpuContext::new();
        let mut enc = EncoderPipeline::new(&ctx);
        let dec = DecoderPipeline::new(&ctx);

        let w = 256;
        let h = 256;
        let f0 = make_gradient_frame(w, h, 0.0);
        let f1 = make_gradient_frame(w, h, 2.0); // slight shift

        let mut config = CodecConfig::default();
        config.tile_size = 256;
        config.keyframe_interval = 8;

        let frames: Vec<&[f32]> = vec![&f0, &f1];
        let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);

        // Decode both frames through the decoder (which maintains reference state)
        let dec0 = dec.decode(&ctx, &compressed[0]);
        let dec1 = dec.decode(&ctx, &compressed[1]);

        // Both frames should reconstruct with good quality
        let psnr0 = compute_psnr(&f0, &dec0);
        let psnr1 = compute_psnr(&f1, &dec1);

        eprintln!("I-frame PSNR: {psnr0:.2} dB, P-frame PSNR: {psnr1:.2} dB");

        assert!(
            psnr0 > 30.0,
            "I-frame PSNR should be > 30 dB, got {psnr0:.2}"
        );
        assert!(
            psnr1 > 25.0,
            "P-frame PSNR should be > 25 dB, got {psnr1:.2}"
        );
    }

    #[test]
    fn test_pframe_identical_frames_correct_decode() {
        let ctx = GpuContext::new();
        let mut enc = EncoderPipeline::new(&ctx);
        let dec = DecoderPipeline::new(&ctx);

        let w = 256;
        let h = 256;
        let f0 = make_gradient_frame(w, h, 0.0);
        let f1 = f0.clone(); // identical frame

        let mut config = CodecConfig::default();
        config.tile_size = 256;
        config.keyframe_interval = 8;

        let frames: Vec<&[f32]> = vec![&f0, &f1];
        let compressed = enc.encode_sequence(&ctx, &frames, w, h, &config);

        // Decode both frames
        let dec0 = dec.decode(&ctx, &compressed[0]);
        let dec1 = dec.decode(&ctx, &compressed[1]);

        // Both should decode with good quality
        let psnr0 = compute_psnr(&f0, &dec0);
        let psnr1 = compute_psnr(&f1, &dec1);
        eprintln!("I-frame PSNR: {psnr0:.2} dB, P-frame (identical input) PSNR: {psnr1:.2} dB");

        assert!(psnr0 > 30.0, "I-frame PSNR too low: {psnr0:.2}");
        assert!(psnr1 > 30.0, "P-frame PSNR too low: {psnr1:.2}");

        // Motion vectors should be near-zero for identical content
        let mf = compressed[1].motion_field.as_ref().unwrap();
        let max_mv: i16 = mf
            .vectors
            .iter()
            .flat_map(|v| v.iter())
            .map(|v| v.abs())
            .max()
            .unwrap_or(0);
        eprintln!("Max MV magnitude for identical frames: {max_mv}");

        // Decoded frame 1 should be very close to decoded frame 0 (identical source)
        let inter_psnr = compute_psnr(&dec0, &dec1);
        eprintln!("Inter-frame PSNR (dec0 vs dec1): {inter_psnr:.2} dB");
        assert!(
            inter_psnr > 30.0,
            "Identical frames should decode similarly: {inter_psnr:.2} dB"
        );
    }

    #[test]
    fn test_sequence_decode_all_frames() {
        let ctx = GpuContext::new();
        let mut enc = EncoderPipeline::new(&ctx);
        let dec = DecoderPipeline::new(&ctx);

        let w = 256;
        let h = 256;
        let frames_rgb: Vec<Vec<f32>> = (0..5)
            .map(|i| make_gradient_frame(w, h, i as f32 * 3.0))
            .collect();
        let frame_refs: Vec<&[f32]> = frames_rgb.iter().map(|f| f.as_slice()).collect();

        let mut config = CodecConfig::default();
        config.tile_size = 256;
        config.keyframe_interval = 3; // I P P I P

        let compressed = enc.encode_sequence(&ctx, &frame_refs, w, h, &config);

        assert_eq!(compressed[0].frame_type, FrameType::Intra);
        assert_eq!(compressed[1].frame_type, FrameType::Predicted);
        assert_eq!(compressed[2].frame_type, FrameType::Predicted);
        assert_eq!(compressed[3].frame_type, FrameType::Intra);
        assert_eq!(compressed[4].frame_type, FrameType::Predicted);

        // Decode all frames in order (decoder maintains reference state)
        for (i, cf) in compressed.iter().enumerate() {
            let decoded = dec.decode(&ctx, cf);
            let psnr = compute_psnr(&frames_rgb[i], &decoded);
            eprintln!(
                "Frame {i} ({:?}): PSNR={psnr:.2} dB, size={} bytes",
                cf.frame_type,
                cf.byte_size()
            );
            assert!(psnr > 25.0, "Frame {i} PSNR too low: {psnr:.2} dB");
        }
    }
}

/// Pad frame to tile-aligned dimensions by edge extension.
fn pad_frame(data: &[f32], w: u32, h: u32, pw: u32, ph: u32) -> Vec<f32> {
    let w = w as usize;
    let h = h as usize;
    let pw = pw as usize;
    let ph = ph as usize;

    let mut padded = vec![0.0f32; pw * ph * 3];
    for y in 0..ph {
        let sy = y.min(h - 1);
        for x in 0..pw {
            let sx = x.min(w - 1);
            let src = (sy * w + sx) * 3;
            let dst = (y * pw + x) * 3;
            padded[dst] = data[src];
            padded[dst + 1] = data[src + 1];
            padded[dst + 2] = data[src + 2];
        }
    }
    padded
}
