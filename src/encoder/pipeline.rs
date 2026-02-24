use wgpu;
use wgpu::util::DeviceExt;

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
        }
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
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Per-plane buffers for wavelet + quantize
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let plane_buf_a = create_plane_buffer(ctx, plane_size, "enc_plane_a");
        let plane_buf_b = create_plane_buffer(ctx, plane_size, "enc_plane_b");
        let plane_buf_c = create_plane_buffer(ctx, plane_size, "enc_plane_c");

        // Separate Co/Cg buffers so deinterleave writes all 3 planes at once
        let co_plane_buf = create_plane_buffer(ctx, plane_size, "enc_co_plane");
        let cg_plane_buf = create_plane_buffer(ctx, plane_size, "enc_cg_plane");

        // Upload input
        ctx.queue
            .write_buffer(&input_buf, 0, bytemuck::cast_slice(&padded_rgb));

        // ---- Submit 1: color convert + deinterleave + (variance + weight normalize) ----
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encode_preprocess"),
            });

        // Color convert (RGB -> YCoCg-R): input_buf -> color_out (interleaved)
        self.color.dispatch(
            ctx, &mut cmd, &input_buf, &color_out, padded_w, padded_h, true,
        );

        // GPU deinterleave: color_out -> plane_buf_a(Y), co_plane_buf(Co), cg_plane_buf(Cg)
        self.deinterleaver.dispatch(
            ctx,
            &mut cmd,
            &color_out,
            &plane_buf_a,
            &co_plane_buf,
            &cg_plane_buf,
            padded_pixels as u32,
        );

        // Adaptive quantization: variance + weight normalization on GPU
        let (weight_map, weight_map_gpu) = if config.adaptive_quantization
            && config.aq_strength > 0.0
        {
            let (blocks_x, blocks_y, total_blocks) = adaptive::weight_map_dims(padded_w, padded_h);
            let wm_buf_size = (total_blocks as usize * std::mem::size_of::<f32>()) as u64;

            let variance_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_variance"),
                size: wm_buf_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let scratch_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_wm_scratch"),
                size: wm_buf_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let weight_map_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_weight_map"),
                size: wm_buf_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Variance analysis reads Y from plane_buf_a (already on GPU from deinterleave)
            self.variance.dispatch(
                ctx,
                &mut cmd,
                &plane_buf_a,
                &variance_buf,
                padded_w,
                padded_h,
            );

            // Weight map normalization on GPU (replaces CPU compute_weight_map)
            self.weight_normalizer.dispatch(
                ctx,
                &mut cmd,
                &variance_buf,
                &scratch_buf,
                &weight_map_buf,
                blocks_x,
                blocks_y,
                config.aq_strength,
            );

            ctx.queue.submit(Some(cmd.finish()));

            // Small readback (~8KB for 1080p) for CompressedFrame serialization only
            let wm = read_buffer_f32(ctx, &weight_map_buf, total_blocks as usize);
            (Some(wm), Some((weight_map_buf, blocks_x)))
        } else {
            ctx.queue.submit(Some(cmd.finish()));
            (None, None)
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

        // Persistent recon_y buffer for CfL (stays on GPU across plane loop)
        let mut recon_y_buf: Option<wgpu::Buffer> = None;

        // Chroma source buffers indexed by plane
        let chroma_sources = [&co_plane_buf, &cg_plane_buf];

        for p in 0..3 {
            let is_chroma = p > 0;
            let use_cfl = config.cfl_enabled && is_chroma;

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encode_plane"),
                });

            if p == 0 {
                // Y plane: data already in plane_buf_a from deinterleave

                // Wavelet forward: plane_buf_a -> plane_buf_b(scratch) -> plane_buf_c
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

                if config.cfl_enabled {
                    // Y with CfL: quantize + dequantize to get reconstructed Y wavelet
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

                    // Copy recon Y to persistent buffer (stays on GPU for chroma CfL)
                    let ry_buf = create_plane_buffer(ctx, plane_size, "enc_recon_y");
                    cmd.copy_buffer_to_buffer(&plane_buf_a, 0, &ry_buf, 0, plane_size);
                    recon_y_buf = Some(ry_buf);

                    ctx.queue.submit(Some(cmd.finish()));

                    // Entropy encode from plane_buf_b (quantized Y)
                    Self::encode_entropy(
                        &mut self.gpu_encoder,
                        ctx,
                        &plane_buf_b,
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
                } else {
                    // Non-CfL Y: adaptive quantize
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
                        &weights_luma,
                        wm_param,
                    );
                    ctx.queue.submit(Some(cmd.finish()));

                    Self::encode_entropy(
                        &mut self.gpu_encoder,
                        ctx,
                        &plane_buf_b,
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
            } else if use_cfl {
                // Chroma plane with CfL: wavelet + alpha + forward + quantize all on GPU
                let chroma_source = chroma_sources[p - 1];

                // Wavelet forward: chroma_source -> plane_buf_b(scratch) -> plane_buf_c
                self.transform.forward(
                    ctx,
                    &mut cmd,
                    chroma_source,
                    &plane_buf_b,
                    &plane_buf_c,
                    &info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );

                // GPU CfL alpha computation
                let total_tiles = (tiles_x * tiles_y) as u32;
                let alpha_count = (total_tiles * nsb) as usize;
                let alpha_buf_size = (alpha_count * std::mem::size_of::<f32>()) as u64;

                let raw_alpha_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("enc_raw_alpha"),
                    size: alpha_buf_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let dq_alpha_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("enc_dq_alpha"),
                    size: alpha_buf_size,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });

                let ry_buf = recon_y_buf.as_ref().unwrap();
                self.cfl_alpha.dispatch(
                    ctx,
                    &mut cmd,
                    ry_buf,
                    &plane_buf_c,
                    &raw_alpha_buf,
                    &dq_alpha_buf,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                );

                // GPU CfL forward: residual = chroma_wavelet - alpha * recon_y -> plane_buf_a
                self.cfl_forward.dispatch(
                    ctx,
                    &mut cmd,
                    &plane_buf_c,
                    ry_buf,
                    &dq_alpha_buf,
                    &plane_buf_a,
                    padded_pixels as u32,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                );

                // Quantize residual: plane_buf_a -> plane_buf_b
                self.quantize.dispatch(
                    ctx,
                    &mut cmd,
                    &plane_buf_a,
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

                // Tiny readback of raw alphas (~few hundred bytes) for u8 serialization
                let raw_alphas = read_buffer_f32(ctx, &raw_alpha_buf, alpha_count);
                let q_alphas: Vec<u8> =
                    raw_alphas.iter().map(|&a| cfl::quantize_alpha(a)).collect();
                cfl_alphas_all.extend_from_slice(&q_alphas);

                Self::encode_entropy(
                    &mut self.gpu_encoder,
                    ctx,
                    &plane_buf_b,
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
            } else {
                // Non-CfL chroma: copy from deinterleaved buffer, wavelet + quantize
                let chroma_source = chroma_sources[p - 1];

                // Wavelet forward: chroma_source -> plane_buf_b(scratch) -> plane_buf_c
                self.transform.forward(
                    ctx,
                    &mut cmd,
                    chroma_source,
                    &plane_buf_b,
                    &plane_buf_c,
                    &info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );

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
                    &weights_chroma,
                    wm_param,
                );
                ctx.queue.submit(Some(cmd.finish()));

                Self::encode_entropy(
                    &mut self.gpu_encoder,
                    ctx,
                    &plane_buf_b,
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
            let quantized = read_buffer_f32(ctx, quantized_buf, padded_pixels);
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
                ctx,
                &mut cmd,
                &buf_a,
                &buf_b,
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
                &buf_b,
                &buf_c,
                &buf_a,
                info,
                config.wavelet_levels,
                config.wavelet_type,
            );

            ctx.queue.submit(Some(cmd.finish()));
            result_planes[p] = read_buffer_f32(ctx, &buf_a, padded_pixels);
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
        let pixel_count_3 = padded_pixels * 3;
        let buf_size = (pixel_count_3 * std::mem::size_of::<f32>()) as u64;

        // Create GPU buffers
        let input_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pf_input"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let color_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pf_color_out"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Plane buffers: Y goes to plane_buf_a, Co/Cg to separate buffers
        let plane_buf_a = create_plane_buffer(ctx, plane_size, "pf_a");
        let plane_buf_b = create_plane_buffer(ctx, plane_size, "pf_b");
        let plane_buf_c = create_plane_buffer(ctx, plane_size, "pf_c");
        let co_plane_buf = create_plane_buffer(ctx, plane_size, "pf_co");
        let cg_plane_buf = create_plane_buffer(ctx, plane_size, "pf_cg");

        // Color convert + deinterleave on GPU
        let padded_rgb = pad_frame(rgb_data, width, height, padded_w, padded_h);
        ctx.queue
            .write_buffer(&input_buf, 0, bytemuck::cast_slice(&padded_rgb));

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pf_preprocess"),
            });
        self.color.dispatch(
            ctx, &mut cmd, &input_buf, &color_out, padded_w, padded_h, true,
        );
        self.deinterleaver.dispatch(
            ctx,
            &mut cmd,
            &color_out,
            &plane_buf_a,
            &co_plane_buf,
            &cg_plane_buf,
            padded_pixels as u32,
        );
        ctx.queue.submit(Some(cmd.finish()));

        // Block matching on Y plane (already on GPU as plane_buf_a)
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
                .estimate(ctx, &mut cmd, &plane_buf_a, &ref_y_buf, padded_w, padded_h);
        ctx.queue.submit(Some(cmd.finish()));

        let blocks_x = padded_w / ME_BLOCK_SIZE;
        let blocks_y = padded_h / ME_BLOCK_SIZE;
        let total_blocks = blocks_x * blocks_y;
        let mvs = MotionEstimator::read_motion_vectors(ctx, &mv_buf, total_blocks);
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

        // Current plane sources on GPU: [plane_buf_a(Y), co_plane_buf, cg_plane_buf]
        let cur_plane_bufs = [&plane_buf_a, &co_plane_buf, &cg_plane_buf];

        for p in 0..3 {
            let weights = if p == 0 {
                &weights_luma
            } else {
                &weights_chroma
            };

            // Upload reference plane (from local decode of previous frame)
            let ref_buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("pf_ref"),
                    contents: bytemuck::cast_slice(&ref_planes[p]),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            // MC residual buffer: can't write to same buffer we read from
            let mc_out = create_plane_buffer(ctx, plane_size, "pf_mc_out");

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_enc"),
                });

            // MC: current - reference -> mc_out (residual)
            self.motion.compensate(
                ctx,
                &mut cmd,
                cur_plane_bufs[p],
                &ref_buf,
                &mv_buf_ro,
                &mc_out,
                padded_w,
                padded_h,
                true,
            );

            // Copy residual to plane_buf_a for wavelet input
            cmd.copy_buffer_to_buffer(&mc_out, 0, &plane_buf_a, 0, plane_size);

            // Wavelet forward: plane_buf_a -> plane_buf_b(scratch) -> plane_buf_c
            self.transform.forward(
                ctx,
                &mut cmd,
                &plane_buf_a,
                &plane_buf_b,
                &plane_buf_c,
                info,
                config.wavelet_levels,
                config.wavelet_type,
            );

            // Quantize: plane_buf_c -> plane_buf_b
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
                weights,
            );
            ctx.queue.submit(Some(cmd.finish()));

            // Entropy encode from plane_buf_b
            Self::encode_entropy(
                &mut self.gpu_encoder,
                ctx,
                &plane_buf_b,
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

        // Local decode loop: dequant -> inv wavelet -> add prediction -> new reference
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
                .write_buffer(&plane_buf_a, 0, bytemuck::cast_slice(&quantized));

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
                &plane_buf_a,
                &plane_buf_b,
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
                &plane_buf_b,
                &plane_buf_c,
                &plane_buf_a,
                info,
                config.wavelet_levels,
                config.wavelet_type,
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
                ctx, &mut cmd, &res_buf, &ref_buf, &mv_buf_ro, &recon_buf, padded_w, padded_h,
                false,
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
