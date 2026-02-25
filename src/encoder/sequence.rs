use wgpu;

use super::adaptive::{self, AQ_LL_BLOCK_SIZE};
use super::bitplane;
use super::buffer_cache::pad_frame;
use super::cfl;
use super::entropy_helpers::{encode_entropy, entropy_decode_plane, EntropyMode};
use super::motion::{MotionEstimator, ME_BLOCK_SIZE};
use super::pipeline::EncoderPipeline;
use super::rans;
use super::rate_control::RateController;
use crate::gpu_util::read_buffer_f32;
use crate::{
    CodecConfig, CompressedFrame, EntropyCoder, EntropyData, FrameInfo, FrameType, GpuContext,
    MotionField,
};

/// Default frame rate assumed when rate control is active but no explicit fps is set.
const DEFAULT_FPS: f64 = 30.0;

impl EncoderPipeline {
    /// Encode a sequence of RGB frames with temporal prediction.
    ///
    /// Frame 0 (and every `keyframe_interval` frames) is encoded as an I-frame.
    /// Other frames are encoded as P-frames (residual from previous decoded frame).
    /// The encoder maintains a local decode loop so it uses the same reference
    /// as the decoder would.
    ///
    /// When `config.target_bitrate` is set, a rate controller adjusts the
    /// quantization step per frame to hit the target bitrate (CBR or VBR).
    pub fn encode_sequence(
        &mut self,
        ctx: &GpuContext,
        frames: &[&[f32]],
        width: u32,
        height: u32,
        config: &CodecConfig,
    ) -> Vec<CompressedFrame> {
        self.encode_sequence_with_fps(ctx, frames, width, height, config, DEFAULT_FPS)
    }

    /// Like [`encode_sequence`] but with an explicit frame rate for rate control.
    pub fn encode_sequence_with_fps(
        &mut self,
        ctx: &GpuContext,
        frames: &[&[f32]],
        width: u32,
        height: u32,
        config: &CodecConfig,
        fps: f64,
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

        // Create rate controller if target bitrate is set
        let mut rate_ctrl = config
            .target_bitrate
            .map(|bitrate| RateController::new(bitrate, fps, width, height, config.rate_mode));

        for (i, rgb_data) in frames.iter().enumerate() {
            let is_keyframe = ki <= 1 || i % ki as usize == 0 || reference_planes.is_none();

            // Build per-frame config: override qstep if rate control is active
            let frame_config = if let Some(ref rc) = rate_ctrl {
                let mut cfg = config.clone();
                cfg.quantization_step = rc.estimate_qstep();
                cfg
            } else {
                config.clone()
            };

            if is_keyframe {
                let mut compressed = self.encode(ctx, rgb_data, width, height, &frame_config);
                compressed.frame_type = FrameType::Intra;

                // Update rate controller with actual encoded result
                if let Some(ref mut rc) = rate_ctrl {
                    rc.update(frame_config.quantization_step, compressed.bpp());
                }

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
                    &frame_config,
                );

                // Update rate controller with actual encoded result
                if let Some(ref mut rc) = rate_ctrl {
                    rc.update(frame_config.quantization_step, compressed.bpp());
                }

                reference_planes = Some(new_ref);
                results.push(compressed);
            }
        }

        results
    }

    /// I-frame local decode: entropy decode → dequantize (with AQ) → CfL inverse → inverse wavelet.
    /// Must match the standalone decoder exactly so P-frame references don't drift.
    fn local_decode_iframe(
        &mut self,
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
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;

        // Ensure CfL alpha buffers are large enough for ALL planes' alphas
        let has_cfl = frame.cfl_alphas.is_some();
        let cfl_alphas_per_plane;
        let all_cfl_f32: Vec<f32>;
        if has_cfl {
            let cfl_data = frame.cfl_alphas.as_ref().unwrap();
            cfl_alphas_per_plane = tiles_per_plane * cfl_data.num_subbands as usize;
            all_cfl_f32 = cfl_data
                .alphas
                .iter()
                .map(|&q| cfl::dequantize_alpha(q))
                .collect();
            let total_alpha_bytes = (all_cfl_f32.len() * std::mem::size_of::<f32>()) as u64;
            let bufs = self.cached.as_mut().unwrap();
            crate::gpu_util::ensure_var_buf(
                ctx,
                &mut bufs.dq_alpha,
                &mut bufs.dq_alpha_cap,
                total_alpha_bytes,
                "enc_dq_alpha",
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            crate::gpu_util::ensure_var_buf(
                ctx,
                &mut bufs.raw_alpha,
                &mut bufs.raw_alpha_cap,
                (cfl_alphas_per_plane * std::mem::size_of::<f32>()) as u64,
                "enc_raw_alpha",
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
        } else {
            cfl_alphas_per_plane = 0;
            all_cfl_f32 = Vec::new();
        }

        let bufs = self.cached.as_ref().unwrap();

        // Build weight-map parameter (matching decoder's dispatch_adaptive call)
        let wm_param = if frame.weight_map.is_some() {
            let (_, ll_bx, _, tx) = adaptive::weight_map_dims(
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
            );
            let ll_size = config.tile_size >> config.wavelet_levels;
            let ll_block_size = AQ_LL_BLOCK_SIZE.min(ll_size);
            // Upload weight map to cached weight_map_buf
            ctx.queue.write_buffer(
                &bufs.weight_map_buf,
                0,
                bytemuck::cast_slice(frame.weight_map.as_ref().unwrap()),
            );
            Some((&bufs.weight_map_buf, ll_block_size, ll_bx, tx))
        } else {
            None
        };

        // Upload CfL alphas if present
        if has_cfl {
            ctx.queue
                .write_buffer(&bufs.dq_alpha, 0, bytemuck::cast_slice(&all_cfl_f32));
        }

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

            // Dequantize with AQ weight map (matching decoder)
            self.quantize.dispatch_adaptive(
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
                wm_param,
            );

            if p == 0 && has_cfl {
                // Save dequantized Y wavelet for CfL chroma prediction
                cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.recon_y, 0, plane_size);
            }

            if p > 0 && has_cfl {
                // CfL inverse prediction: plane_b (dequantized residual) + alpha * Y → plane_c
                let plane_alpha_offset = (p - 1) * cfl_alphas_per_plane;
                let plane_alpha_byte_offset =
                    (plane_alpha_offset * std::mem::size_of::<f32>()) as u64;
                let plane_alpha_byte_size =
                    (cfl_alphas_per_plane * std::mem::size_of::<f32>()) as u64;

                // Copy this plane's alphas from full alpha buffer to raw_alpha (temp)
                cmd.copy_buffer_to_buffer(
                    &bufs.dq_alpha,
                    plane_alpha_byte_offset,
                    &bufs.raw_alpha,
                    0,
                    plane_alpha_byte_size,
                );

                self.cfl_inverse.dispatch_inverse(
                    ctx,
                    &mut cmd,
                    &bufs.plane_b,
                    &bufs.recon_y,
                    &bufs.raw_alpha,
                    &bufs.plane_c,
                    padded_pixels as u32,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                );

                // Inverse wavelet: plane_c → plane_b(scratch) → plane_a
                self.transform.inverse(
                    ctx,
                    &mut cmd,
                    &bufs.plane_c,
                    &bufs.plane_b,
                    &bufs.plane_a,
                    info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );
            } else {
                // Standard path (Y plane, or no CfL): plane_b → plane_c(scratch) → plane_a
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
            }

            ctx.queue.submit(Some(cmd.finish()));
            result_planes[p] = read_buffer_f32(ctx, &bufs.plane_a, padded_pixels);
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
        let (mv_buf, sad_buf) = self.motion.estimate(
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
        let mut mvs = MotionEstimator::read_motion_vectors(ctx, &mv_buf, me_total_blocks);
        let sads = MotionEstimator::read_sad_values(ctx, &sad_buf, me_total_blocks);

        // Per-tile adaptive I/P decision:
        // Compare residual energy (SAD) vs original energy per tile.
        // If residual >= original, zero out the tile's MVs.
        let current_y = crate::gpu_util::read_buffer_f32(ctx, &bufs.plane_a, padded_pixels);
        let tile_size_usize = config.tile_size as usize;
        let blocks_per_tile_x = tile_size_usize / ME_BLOCK_SIZE as usize;
        let blocks_per_tile_y = tile_size_usize / ME_BLOCK_SIZE as usize;
        let tiles_x_count = info.tiles_x() as usize;
        let tiles_y_count = info.tiles_y() as usize;

        for ty in 0..tiles_y_count {
            for tx in 0..tiles_x_count {
                let mut tile_sad: u64 = 0;
                for bty in 0..blocks_per_tile_y {
                    for btx in 0..blocks_per_tile_x {
                        let global_bx = tx * blocks_per_tile_x + btx;
                        let global_by = ty * blocks_per_tile_y + bty;
                        let block_idx = global_by * me_blocks_x as usize + global_bx;
                        if block_idx < sads.len() {
                            tile_sad += sads[block_idx] as u64;
                        }
                    }
                }

                let mut tile_energy: u64 = 0;
                let tile_px_x = tx * tile_size_usize;
                let tile_px_y = ty * tile_size_usize;
                for py in 0..tile_size_usize {
                    for px in 0..tile_size_usize {
                        let gx = tile_px_x + px;
                        let gy = tile_px_y + py;
                        if gx < padded_w as usize && gy < padded_h as usize {
                            let idx = gy * padded_w as usize + gx;
                            tile_energy += current_y[idx].abs() as u64;
                        }
                    }
                }

                if tile_sad >= tile_energy {
                    for bty in 0..blocks_per_tile_y {
                        for btx in 0..blocks_per_tile_x {
                            let global_bx = tx * blocks_per_tile_x + btx;
                            let global_by = ty * blocks_per_tile_y + bty;
                            let block_idx = global_by * me_blocks_x as usize + global_bx;
                            if block_idx < mvs.len() {
                                mvs[block_idx] = [0, 0];
                            }
                        }
                    }
                }
            }
        }

        let mv_buf_ro = MotionEstimator::upload_motion_vectors(ctx, &mvs);

        // MC + wavelet + quantize per plane, keeping data on GPU
        let entropy_mode = EntropyMode::from_config(config);
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
            EntropyMode::SubbandRans | EntropyMode::SubbandRansCtx => {
                EntropyData::SubbandRans(subband_tiles)
            }
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
            recon_planes[p] = read_buffer_f32(ctx, &bufs.recon_out, padded_pixels);
        }

        (compressed, recon_planes)
    }
}
