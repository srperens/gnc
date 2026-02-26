use wgpu;

use super::adaptive::{self, VarianceAnalyzer, WeightMapNormalizer, AQ_LL_BLOCK_SIZE};
use super::bitplane;
use super::buffer_cache::{pad_frame, CachedEncodeBuffers};
use super::cfl::{self, CflAlphaComputer, CflForwardPredictor, CflPredictor};
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

/// Full encoding pipeline: Color -> Wavelet -> (LL Variance Analysis) -> Quantize -> rANS Entropy
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
    pub(super) cfl_inverse: CflPredictor,
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
            cfl_inverse: CflPredictor::new(ctx),
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

        // ---- Submit 1: color convert + deinterleave ----
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

        // ---- Per-plane encoding ----
        let mut rans_tiles: Vec<rans::InterleavedRansTile> = Vec::new();
        let mut subband_tiles: Vec<rans::SubbandRansTile> = Vec::new();
        let mut bp_tiles: Vec<bitplane::BitplaneTile> = Vec::new();
        let entropy_mode = EntropyMode::from_config(config);
        let tile_size = config.tile_size as usize;
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;
        let use_gpu_encode = config.gpu_entropy_encode
            && config.entropy_coder != EntropyCoder::Bitplane
            && !config.context_adaptive;

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
            // (CfL path currently does not use AQ; can be extended later)
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
            cmd.copy_buffer_to_buffer(&bufs.plane_a, 0, &bufs.recon_y, 0, plane_size);
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
            // Preserve Co alpha before Cg overwrites raw_alpha
            cmd.copy_buffer_to_buffer(&bufs.raw_alpha, 0, &alpha_staging[0], 0, alpha_bytes);
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
            // Copy Cg alpha to staging for deferred readback
            cmd.copy_buffer_to_buffer(&bufs.raw_alpha, 0, &alpha_staging[1], 0, alpha_bytes);
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

        // Single submit for all 3 planes
        ctx.queue.submit(Some(cmd.finish()));

        // Weight map readback (for serialization into CompressedFrame)
        let weight_map = if aq_active {
            let (total_blocks, _, _, _) = adaptive::weight_map_dims(
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
            );
            let wm = read_buffer_f32(ctx, &bufs.weight_map_buf, total_blocks as usize);
            Some(wm)
        } else {
            None
        };

        // Deferred CfL alpha readback (single poll for both chroma planes)
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
                // Shader writes i32 values quantized to i16 range [-16384, 16384]
                let raw_alphas: &[i32] = bytemuck::cast_slice(&view);
                let q_alphas: Vec<i16> = raw_alphas.iter().map(|&a| a as i16).collect();
                cfl_alphas_all.extend_from_slice(&q_alphas);
                drop(view);
                stg.unmap();
            }
        }

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
                );
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
            EntropyMode::SubbandRans | EntropyMode::SubbandRansCtx => {
                EntropyData::SubbandRans(subband_tiles)
            }
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
