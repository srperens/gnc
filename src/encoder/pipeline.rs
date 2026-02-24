use wgpu;

use super::adaptive::{self, VarianceAnalyzer, WeightMapNormalizer, AQ_BLOCK_SIZE};
use super::bitplane;
use super::buffer_cache::{pad_frame, CachedEncodeBuffers};
use super::cfl::{self, CflAlphaComputer, CflForwardPredictor};
use super::color::ColorConverter;
use super::entropy_helpers::{encode_entropy, EntropyMode};
use super::interleave::PlaneDeinterleaver;
use super::quantize::Quantizer;
use super::rans;
use super::rans_gpu_encode::GpuRansEncoder;
use super::transform::WaveletTransform;
use crate::gpu_util::{ensure_var_buf, read_buffer_f32};
use crate::{
    CflAlphas, CodecConfig, CompressedFrame, EntropyCoder, EntropyData, FrameInfo, FrameType,
    GpuContext,
};

// Temporal coding (encode_sequence, encode_pframe, local_decode_iframe)
// is in the `sequence` sibling module which adds an `impl EncoderPipeline` block.

/// Full encoding pipeline: Color -> (Variance Analysis) -> Wavelet -> Quantize -> rANS Entropy
pub struct EncoderPipeline {
    pub(super) color: ColorConverter,
    pub(super) transform: WaveletTransform,
    pub(super) quantize: Quantizer,
    pub(super) variance: VarianceAnalyzer,
    pub(super) motion: super::motion::MotionEstimator,
    pub(super) gpu_encoder: GpuRansEncoder,
    pub(super) deinterleaver: PlaneDeinterleaver,
    pub(super) weight_normalizer: WeightMapNormalizer,
    pub(super) cfl_alpha: CflAlphaComputer,
    pub(super) cfl_forward: CflForwardPredictor,
    pub(super) cached: Option<CachedEncodeBuffers>,
}

impl EncoderPipeline {
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            color: ColorConverter::new(ctx),
            transform: WaveletTransform::new(ctx),
            quantize: super::quantize::Quantizer::new(ctx),
            variance: VarianceAnalyzer::new(ctx),
            motion: super::motion::MotionEstimator::new(ctx),
            gpu_encoder: GpuRansEncoder::new(ctx),
            deinterleaver: PlaneDeinterleaver::new(ctx),
            weight_normalizer: WeightMapNormalizer::new(ctx),
            cfl_alpha: CflAlphaComputer::new(ctx),
            cfl_forward: CflForwardPredictor::new(ctx),
            cached: None,
        }
    }

    /// Ensure cached buffers exist and match the given padded resolution.
    pub(super) fn ensure_cached(&mut self, ctx: &GpuContext, padded_w: u32, padded_h: u32) {
        let needs_alloc = match &self.cached {
            Some(c) => c.padded_w != padded_w || c.padded_h != padded_h,
            None => true,
        };
        if !needs_alloc {
            return;
        }
        self.cached = Some(CachedEncodeBuffers::new(ctx, padded_w, padded_h));
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
            let wm = read_buffer_f32(ctx, &bufs.weight_map_buf, total_blocks as usize);
            Some(wm)
        } else {
            ctx.queue.submit(Some(cmd.finish()));
            None
        };

        // ---- Per-plane encoding ----
        let mut rans_tiles: Vec<rans::InterleavedRansTile> = Vec::new();
        let mut subband_tiles: Vec<rans::SubbandRansTile> = Vec::new();
        let mut bp_tiles: Vec<bitplane::BitplaneTile> = Vec::new();
        let entropy_mode = EntropyMode::from_config(config);
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
                        cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.mc_out, 0, plane_size);
                    }

                    ctx.queue.submit(Some(cmd.finish()));

                    if !use_gpu_encode {
                        encode_entropy(
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
                        cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.mc_out, 0, plane_size);
                    }

                    ctx.queue.submit(Some(cmd.finish()));

                    if !use_gpu_encode {
                        encode_entropy(
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
                    cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.ref_upload, 0, plane_size);
                }

                ctx.queue.submit(Some(cmd.finish()));

                // Tiny readback of raw alphas (~few hundred bytes) for u8 serialization
                let raw_alphas = read_buffer_f32(ctx, &bufs.raw_alpha, alpha_count);
                let q_alphas: Vec<u8> =
                    raw_alphas.iter().map(|&a| cfl::quantize_alpha(a)).collect();
                cfl_alphas_all.extend_from_slice(&q_alphas);

                if !use_gpu_encode {
                    encode_entropy(
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
                    cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.ref_upload, 0, plane_size);
                }

                ctx.queue.submit(Some(cmd.finish()));

                if !use_gpu_encode {
                    encode_entropy(
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
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
