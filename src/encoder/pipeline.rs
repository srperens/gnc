use wgpu;
use wgpu::util::DeviceExt;

use super::adaptive::{self, VarianceAnalyzer, AQ_BLOCK_SIZE};
use super::bitplane;
use super::cfl;
use super::color::ColorConverter;
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

/// Full encoding pipeline: Color -> (Variance Analysis) -> Wavelet -> Quantize -> rANS Entropy
pub struct EncoderPipeline {
    color: ColorConverter,
    transform: WaveletTransform,
    quantize: Quantizer,
    variance: VarianceAnalyzer,
    motion: MotionEstimator,
    gpu_encoder: GpuRansEncoder,
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
        }
    }

    /// Encode an RGB frame.
    /// Input: &[f32] of length width * height * 3 (interleaved R,G,B).
    /// Values in [0, 255] for 8-bit or [0, 1023] for 10-bit.
    pub fn encode(
        &self,
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

        // Pad input to tile-aligned dimensions
        let padded_rgb = pad_frame(rgb_data, width, height, padded_w, padded_h);

        let pixel_count_3 = padded_pixels * 3;
        let buf_size = (pixel_count_3 * std::mem::size_of::<f32>()) as u64;

        // Create GPU buffers
        let input_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_input"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let color_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_color_out"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Per-plane buffers for wavelet + quantize
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let plane_buf_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_plane_a"),
            size: plane_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let plane_buf_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_plane_b"),
            size: plane_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let plane_buf_c = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_plane_c"),
            size: plane_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload input
        ctx.queue
            .write_buffer(&input_buf, 0, bytemuck::cast_slice(&padded_rgb));

        // Step 1: Color convert (RGB -> YCoCg-R) on GPU
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encode_color"),
            });
        self.color.dispatch(
            ctx, &mut cmd, &input_buf, &color_out, padded_w, padded_h, true,
        );
        ctx.queue.submit(Some(cmd.finish()));

        // Read back and deinterleave into 3 planes
        let color_data = read_buffer_f32(ctx, &color_out, pixel_count_3);
        let mut planes: [Vec<f32>; 3] = [
            vec![0.0; padded_pixels],
            vec![0.0; padded_pixels],
            vec![0.0; padded_pixels],
        ];
        for i in 0..padded_pixels {
            planes[0][i] = color_data[i * 3];
            planes[1][i] = color_data[i * 3 + 1];
            planes[2][i] = color_data[i * 3 + 2];
        }

        // Step 1.5: Adaptive quantization — compute variance from Y plane, build weight map
        let (weight_map, weight_map_gpu) = if config.adaptive_quantization
            && config.aq_strength > 0.0
        {
            let (blocks_x, blocks_y, total_blocks) = adaptive::weight_map_dims(padded_w, padded_h);

            let y_buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("enc_y_plane"),
                    contents: bytemuck::cast_slice(&planes[0]),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let variance_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_variance"),
                size: (total_blocks as usize * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encode_variance"),
                });
            self.variance
                .dispatch(ctx, &mut cmd, &y_buf, &variance_buf, padded_w, padded_h);
            ctx.queue.submit(Some(cmd.finish()));

            let raw_variance = read_buffer_f32(ctx, &variance_buf, total_blocks as usize);
            let wm =
                adaptive::compute_weight_map(&raw_variance, blocks_x, blocks_y, config.aq_strength);

            let wm_gpu = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("enc_weight_map"),
                    contents: bytemuck::cast_slice(&wm),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            (Some(wm), Some((wm_gpu, blocks_x)))
        } else {
            (None, None)
        };

        // Step 2 & 3: Wavelet transform + quantize per plane on GPU,
        // then entropy encode per tile
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
        // GPU encode is used for rANS modes when enabled (not for bitplane)
        let use_gpu_encode =
            config.gpu_entropy_encode && config.entropy_coder != EntropyCoder::Bitplane;

        // Pre-pack subband weights: luma and chroma variants
        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();

        // CfL state: reconstructed Y wavelet (dequantized) for chroma prediction
        let mut recon_y_wavelet: Option<Vec<f32>> = None;
        let mut cfl_alphas_all: Vec<u8> = Vec::new();
        let nsb = cfl::num_subbands(config.wavelet_levels);

        for p in 0..3 {
            let is_chroma = p > 0;
            let use_cfl = config.cfl_enabled && is_chroma;

            ctx.queue
                .write_buffer(&plane_buf_a, 0, bytemuck::cast_slice(&planes[p]));

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encode_plane"),
                });

            // Wavelet forward: plane_buf_a → (temp plane_buf_b) → plane_buf_c (wavelet coefficients)
            self.transform.forward(
                ctx,
                &mut cmd,
                &plane_buf_a,
                &plane_buf_b,
                &plane_buf_c,
                &info,
                config.wavelet_levels,
                config.wavelet_type,
            );

            if p == 0 && config.cfl_enabled {
                // Y plane with CfL: quantize, then dequantize to get reconstructed Y wavelet.
                // Quantize: plane_buf_c → plane_buf_b (quantized)
                self.quantize.dispatch(
                    ctx,
                    &mut cmd,
                    &plane_buf_c,
                    &plane_buf_b,
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
                // Dequantize: plane_buf_b → plane_buf_a (reconstructed Y wavelet)
                self.quantize.dispatch(
                    ctx,
                    &mut cmd,
                    &plane_buf_b,
                    &plane_buf_a,
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
                ctx.queue.submit(Some(cmd.finish()));

                // Read back reconstructed Y wavelet (needed for CfL chroma prediction)
                recon_y_wavelet = Some(read_buffer_f32(ctx, &plane_buf_a, padded_pixels));

                if use_gpu_encode {
                    // GPU encode: plane_buf_b stays on GPU, no readback needed
                    let (mut rt, mut st) = self.gpu_encoder.encode_plane_to_tiles(
                        ctx,
                        &plane_buf_b,
                        &info,
                        config.per_subband_entropy,
                        config.wavelet_levels,
                    );
                    rans_tiles.append(&mut rt);
                    subband_tiles.append(&mut st);
                } else {
                    // CPU fallback: read back quantized data
                    let quantized = read_buffer_f32(ctx, &plane_buf_b, padded_pixels);
                    entropy_encode_tiles(
                        &quantized,
                        padded_w as usize,
                        tiles_x,
                        tiles_y,
                        tile_size,
                        &entropy_mode,
                        config.tile_size,
                        config.wavelet_levels,
                        &mut rans_tiles,
                        &mut subband_tiles,
                        &mut bp_tiles,
                    );
                }
            } else if use_cfl {
                // Chroma plane with CfL prediction
                ctx.queue.submit(Some(cmd.finish()));

                // Read back chroma wavelet coefficients from plane_buf_c
                let chroma_wavelet = read_buffer_f32(ctx, &plane_buf_c, padded_pixels);
                let recon_y = recon_y_wavelet.as_ref().unwrap();

                // Compute per-tile per-subband alpha
                let alphas_f32 = cfl::compute_cfl_alphas(
                    recon_y,
                    &chroma_wavelet,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                );

                // Quantize then dequantize alphas (encoder must use same values as decoder)
                let q_alphas: Vec<u8> =
                    alphas_f32.iter().map(|&a| cfl::quantize_alpha(a)).collect();
                let dq_alphas: Vec<f32> =
                    q_alphas.iter().map(|&q| cfl::dequantize_alpha(q)).collect();

                // Compute residual on CPU: residual = chroma_wavelet - alpha * recon_y
                let residual = cfl::apply_cfl_predict_cpu(
                    &chroma_wavelet,
                    recon_y,
                    &dq_alphas,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                );

                // Upload residual, quantize on GPU
                ctx.queue
                    .write_buffer(&plane_buf_c, 0, bytemuck::cast_slice(&residual));
                let mut cmd = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("encode_cfl_residual"),
                    });
                self.quantize.dispatch(
                    ctx,
                    &mut cmd,
                    &plane_buf_c,
                    &plane_buf_b,
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
                ctx.queue.submit(Some(cmd.finish()));

                if use_gpu_encode {
                    // GPU encode: plane_buf_b stays on GPU
                    let (mut rt, mut st) = self.gpu_encoder.encode_plane_to_tiles(
                        ctx,
                        &plane_buf_b,
                        &info,
                        config.per_subband_entropy,
                        config.wavelet_levels,
                    );
                    rans_tiles.append(&mut rt);
                    subband_tiles.append(&mut st);
                } else {
                    let quantized = read_buffer_f32(ctx, &plane_buf_b, padded_pixels);
                    entropy_encode_tiles(
                        &quantized,
                        padded_w as usize,
                        tiles_x,
                        tiles_y,
                        tile_size,
                        &entropy_mode,
                        config.tile_size,
                        config.wavelet_levels,
                        &mut rans_tiles,
                        &mut subband_tiles,
                        &mut bp_tiles,
                    );
                }

                // Store quantized alphas for serialization
                cfl_alphas_all.extend_from_slice(&q_alphas);
            } else {
                // Non-CfL path (original flow) — with optional adaptive quantization
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
                    &plane_buf_c,
                    &plane_buf_b,
                    padded_pixels as u32,
                    config.quantization_step,
                    config.dead_zone,
                    true,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                    weights,
                    wm_param,
                );
                ctx.queue.submit(Some(cmd.finish()));

                if use_gpu_encode {
                    // GPU encode: plane_buf_b stays on GPU
                    let (mut rt, mut st) = self.gpu_encoder.encode_plane_to_tiles(
                        ctx,
                        &plane_buf_b,
                        &info,
                        config.per_subband_entropy,
                        config.wavelet_levels,
                    );
                    rans_tiles.append(&mut rt);
                    subband_tiles.append(&mut st);
                } else {
                    let quantized = read_buffer_f32(ctx, &plane_buf_b, padded_pixels);
                    entropy_encode_tiles(
                        &quantized,
                        padded_w as usize,
                        tiles_x,
                        tiles_y,
                        tile_size,
                        &entropy_mode,
                        config.tile_size,
                        config.wavelet_levels,
                        &mut rans_tiles,
                        &mut subband_tiles,
                        &mut bp_tiles,
                    );
                }
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

    /// Encode a sequence of RGB frames with temporal prediction.
    ///
    /// Frame 0 (and every `keyframe_interval` frames) is encoded as an I-frame.
    /// Other frames are encoded as P-frames (residual from previous decoded frame).
    /// The encoder maintains a local decode loop so it uses the same reference
    /// as the decoder would.
    pub fn encode_sequence(
        &self,
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
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;
        let tiles_per_plane = tiles_x * tiles_y;

        let buf_a = create_plane_buffer(ctx, plane_size, "ld_a");
        let buf_b = create_plane_buffer(ctx, plane_size, "ld_b");
        let buf_c = create_plane_buffer(ctx, plane_size, "ld_c");

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
                .write_buffer(&buf_a, 0, bytemuck::cast_slice(&quantized));

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
                ctx, &mut cmd, &buf_a, &buf_b, padded_pixels as u32,
                config.quantization_step, config.dead_zone, false,
                padded_w, padded_h, config.tile_size, config.wavelet_levels, weights,
            );

            self.transform.inverse(
                ctx, &mut cmd, &buf_b, &buf_c, &buf_a,
                info, config.wavelet_levels, config.wavelet_type,
            );

            ctx.queue.submit(Some(cmd.finish()));
            result_planes[p] = read_buffer_f32(ctx, &buf_a, padded_pixels);
        }

        result_planes
    }

    /// Encode a P-frame: ME → MC → encode residual → local decode loop.
    #[allow(clippy::too_many_arguments)]
    fn encode_pframe(
        &self,
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
        let pixel_count_3 = padded_pixels * 3;
        let buf_size = (pixel_count_3 * std::mem::size_of::<f32>()) as u64;

        // Color convert to YCoCg-R
        let padded_rgb = pad_frame(rgb_data, width, height, padded_w, padded_h);
        let input_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pf_input"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let color_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pf_color_out"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        ctx.queue
            .write_buffer(&input_buf, 0, bytemuck::cast_slice(&padded_rgb));
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pf_color"),
            });
        self.color
            .dispatch(ctx, &mut cmd, &input_buf, &color_out, padded_w, padded_h, true);
        ctx.queue.submit(Some(cmd.finish()));

        let color_data = read_buffer_f32(ctx, &color_out, pixel_count_3);
        let mut current_planes: [Vec<f32>; 3] = [
            vec![0.0; padded_pixels],
            vec![0.0; padded_pixels],
            vec![0.0; padded_pixels],
        ];
        for i in 0..padded_pixels {
            current_planes[0][i] = color_data[i * 3];
            current_planes[1][i] = color_data[i * 3 + 1];
            current_planes[2][i] = color_data[i * 3 + 2];
        }

        // Block matching on Y plane
        let current_y_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pf_cur_y"),
                contents: bytemuck::cast_slice(&current_planes[0]),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let ref_y_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pf_ref_y"),
                contents: bytemuck::cast_slice(&ref_planes[0]),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pf_me"),
            });
        let (mv_buf, _sad_buf) =
            self.motion
                .estimate(ctx, &mut cmd, &current_y_buf, &ref_y_buf, padded_w, padded_h);
        ctx.queue.submit(Some(cmd.finish()));

        let blocks_x = padded_w / ME_BLOCK_SIZE;
        let blocks_y = padded_h / ME_BLOCK_SIZE;
        let total_blocks = blocks_x * blocks_y;
        let mvs = MotionEstimator::read_motion_vectors(ctx, &mv_buf, total_blocks);
        let mv_buf_ro = MotionEstimator::upload_motion_vectors(ctx, &mvs);

        // Compute residual planes: residual = current - MC(reference, MVs)
        let mut residual_planes: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for p in 0..3 {
            let cur_buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pf_cur"),
                    contents: bytemuck::cast_slice(&current_planes[p]),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let ref_buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pf_ref"),
                    contents: bytemuck::cast_slice(&ref_planes[p]),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let residual_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pf_residual"),
                size: plane_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_mc"),
                });
            self.motion.compensate(
                ctx, &mut cmd, &cur_buf, &ref_buf, &mv_buf_ro, &residual_buf,
                padded_w, padded_h, true,
            );
            ctx.queue.submit(Some(cmd.finish()));
            residual_planes[p] = read_buffer_f32(ctx, &residual_buf, padded_pixels);
        }

        // Encode residual: wavelet → quantize → entropy
        let plane_buf_a = create_plane_buffer(ctx, plane_size, "pf_a");
        let plane_buf_b = create_plane_buffer(ctx, plane_size, "pf_b");
        let plane_buf_c = create_plane_buffer(ctx, plane_size, "pf_c");

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

        let mut rans_tiles: Vec<rans::InterleavedRansTile> = Vec::new();
        let mut subband_tiles: Vec<rans::SubbandRansTile> = Vec::new();
        let mut bp_tiles: Vec<bitplane::BitplaneTile> = Vec::new();

        for p in 0..3 {
            ctx.queue
                .write_buffer(&plane_buf_a, 0, bytemuck::cast_slice(&residual_planes[p]));

            let weights = if p == 0 { &weights_luma } else { &weights_chroma };

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_enc"),
                });

            self.transform.forward(
                ctx, &mut cmd, &plane_buf_a, &plane_buf_b, &plane_buf_c,
                info, config.wavelet_levels, config.wavelet_type,
            );

            self.quantize.dispatch(
                ctx, &mut cmd, &plane_buf_c, &plane_buf_b, padded_pixels as u32,
                config.quantization_step, config.dead_zone, true,
                padded_w, padded_h, config.tile_size, config.wavelet_levels, weights,
            );
            ctx.queue.submit(Some(cmd.finish()));

            let quantized = read_buffer_f32(ctx, &plane_buf_b, padded_pixels);
            entropy_encode_tiles(
                &quantized, padded_w as usize, tiles_x, tiles_y, tile_size,
                &entropy_mode, config.tile_size, config.wavelet_levels,
                &mut rans_tiles, &mut subband_tiles, &mut bp_tiles,
            );
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

        // Local decode loop: dequant → inv wavelet → add prediction → new reference
        let tiles_per_plane = tiles_x * tiles_y;
        let mut recon_planes: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];

        for p in 0..3 {
            let quantized = entropy_decode_plane(
                &compressed.entropy, p, tiles_per_plane, tile_size, padded_w as usize,
            );

            ctx.queue
                .write_buffer(&plane_buf_a, 0, bytemuck::cast_slice(&quantized));

            let weights = if p == 0 { &weights_luma } else { &weights_chroma };

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_ld"),
                });

            self.quantize.dispatch(
                ctx, &mut cmd, &plane_buf_a, &plane_buf_b, padded_pixels as u32,
                config.quantization_step, config.dead_zone, false,
                padded_w, padded_h, config.tile_size, config.wavelet_levels, weights,
            );

            self.transform.inverse(
                ctx, &mut cmd, &plane_buf_b, &plane_buf_c, &plane_buf_a,
                info, config.wavelet_levels, config.wavelet_type,
            );
            ctx.queue.submit(Some(cmd.finish()));

            let decoded_residual = read_buffer_f32(ctx, &plane_buf_a, padded_pixels);

            // Inverse MC: recon = decoded_residual + MC(reference)
            let ref_buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pf_ld_ref"),
                    contents: bytemuck::cast_slice(&ref_planes[p]),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let res_buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pf_ld_res"),
                    contents: bytemuck::cast_slice(&decoded_residual),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let recon_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pf_ld_recon"),
                size: plane_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_ld_mc"),
                });
            self.motion.compensate(
                ctx, &mut cmd, &res_buf, &ref_buf, &mv_buf_ro, &recon_buf,
                padded_w, padded_h, false,
            );
            ctx.queue.submit(Some(cmd.finish()));
            recon_planes[p] = read_buffer_f32(ctx, &recon_buf, padded_pixels);
        }

        (compressed, recon_planes)
    }
}

/// Create a plane-sized GPU buffer with typical read/write/copy usage.
fn create_plane_buffer(ctx: &GpuContext, size: u64, label: &str) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
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
            EntropyData::Rans(tiles) => {
                rans::rans_decode_tile_interleaved(&tiles[tile_start + t])
            }
            EntropyData::SubbandRans(tiles) => {
                rans::rans_decode_tile_interleaved_subband(&tiles[tile_start + t])
            }
            EntropyData::Bitplane(tiles) => {
                bitplane::bitplane_decode_tile(&tiles[tile_start + t])
            }
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

/// Read a GPU buffer back to CPU as Vec<f32>.
fn read_buffer_f32(ctx: &GpuContext, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
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
