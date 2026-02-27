use wgpu;
use wgpu::util::DeviceExt;

use super::adaptive::{self, AQ_LL_BLOCK_SIZE};
use super::bitplane;
use super::cfl;
use super::entropy_helpers::{encode_entropy, EntropyMode};
use super::motion::{MotionEstimator, ME_BLOCK_SIZE};
use super::pipeline::EncoderPipeline;
use super::rans;
use super::rice;
use super::rate_control::RateController;
use crate::{
    CodecConfig, CompressedFrame, EntropyCoder, EntropyData, FrameInfo, FrameType, GpuContext,
    MotionField,
};

/// Default frame rate assumed when rate control is active but no explicit fps is set.
const DEFAULT_FPS: f64 = 30.0;

/// Number of consecutive B-frames between anchor frames in a GOP.
/// A value of 2 gives groups of [B B P] — standard for moderate latency.
const B_FRAMES_PER_GROUP: usize = 2;

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
    ///
    /// When `keyframe_interval >= 4`, B-frames are used in groups of
    /// [`B_FRAMES_PER_GROUP`] between anchor (I/P) frames. The encoding order
    /// differs from display order: within each group `[B₁ B₂ P]`, the P-frame
    /// is encoded first to provide the backward reference for B-frames.
    /// Results are returned in **display order**.
    pub fn encode_sequence_with_fps(
        &mut self,
        ctx: &GpuContext,
        frames: &[&[f32]],
        width: u32,
        height: u32,
        config: &CodecConfig,
        fps: f64,
    ) -> Vec<CompressedFrame> {
        let ki = config.keyframe_interval as usize;
        let use_bframes = ki >= 4;
        let b_count = if use_bframes { B_FRAMES_PER_GROUP } else { 0 };
        let group_size = b_count + 1;

        let n = frames.len();
        let mut results: Vec<Option<CompressedFrame>> = (0..n).map(|_| None).collect();
        let mut has_reference = false;

        let info = FrameInfo {
            width,
            height,
            bit_depth: 8,
            tile_size: config.tile_size,
        };
        let padded_w = info.padded_width();
        let padded_h = info.padded_height();
        let padded_pixels = (padded_w * padded_h) as usize;

        // Ensure cached buffers (including gpu_ref_planes) exist for this resolution
        self.ensure_cached(ctx, padded_w, padded_h, width, height);

        // Create rate controller if target bitrate is set
        let mut rate_ctrl = config
            .target_bitrate
            .map(|bitrate| RateController::new(bitrate, fps, width, height, config.rate_mode));

        // Previous P-frame MV buffer for temporal prediction. When available, the ME
        // shader skips the expensive coarse search and only does fine refinement around
        // the predicted MV. Initialized to zero-MVs so even the first P-frame uses
        // fast predicted mode (fine ±4 search around (0,0)).
        let me_blocks_x = padded_w / super::motion::ME_BLOCK_SIZE;
        let me_blocks_y = padded_h / super::motion::ME_BLOCK_SIZE;
        let me_total_blocks = me_blocks_x * me_blocks_y;
        let zero_mv_data = vec![0i32; (me_total_blocks * 2) as usize];
        let zero_mv_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("zero_mvs"),
                contents: bytemuck::cast_slice(&zero_mv_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let mut prev_mv_buf: Option<wgpu::Buffer> = None;

        let mut display_idx = 0;
        while display_idx < n {
            let is_keyframe = ki <= 1 || display_idx % ki == 0 || !has_reference;

            // Build per-frame config: override qstep if rate control is active
            let frame_config = if let Some(ref rc) = rate_ctrl {
                let mut cfg = config.clone();
                cfg.quantization_step = rc.estimate_qstep();
                cfg
            } else {
                config.clone()
            };

            if is_keyframe {
                let _t_iframe = std::time::Instant::now();
                let mut compressed =
                    self.encode(ctx, frames[display_idx], width, height, &frame_config);
                compressed.frame_type = FrameType::Intra;

                if let Some(ref mut rc) = rate_ctrl {
                    rc.update(frame_config.quantization_step, compressed.bpp());
                }

                // GPU local decode: quantized data is on GPU from encode()
                // (Y→mc_out, Co→ref_upload, Cg→plane_b). No CPU entropy decode needed.
                self.local_decode_iframe_gpu(ctx, &compressed, padded_w, padded_h, padded_pixels);
                if std::env::var("GNC_PROFILE").is_ok() {
                    eprintln!("  I-frame total: {:.1}ms", _t_iframe.elapsed().as_secs_f64() * 1000.0);
                }
                has_reference = true;
                // Keep prev_mv_buf across keyframes: motion patterns are usually
                // similar across GOP boundaries (no scene-change detection).
                // This avoids a full ±32 ME search on the first P after each I-frame.
                results[display_idx] = Some(compressed);
                display_idx += 1;
                continue;
            }

            if !use_bframes {
                // P-frame only mode
                let (compressed, new_mv_buf) = self.encode_pframe(
                    ctx,
                    frames[display_idx],
                    width,
                    height,
                    padded_w,
                    padded_h,
                    padded_pixels,
                    &info,
                    &frame_config,
                    prev_mv_buf.as_ref().or(Some(&zero_mv_buf)),
                    false,
                );
                prev_mv_buf = Some(new_mv_buf);
                if let Some(ref mut rc) = rate_ctrl {
                    rc.update(frame_config.quantization_step, compressed.bpp());
                }
                results[display_idx] = Some(compressed);
                display_idx += 1;
                continue;
            }

            // --- B-frame group processing ---
            // Determine how many inter frames remain until next keyframe
            let next_key = (((display_idx / ki) + 1) * ki).min(n);
            let remaining = next_key - display_idx;
            let full_groups = remaining / group_size;
            let _remainder = remaining % group_size;

            for g in 0..full_groups {
                let group_start = display_idx + g * group_size;
                let p_display = group_start + b_count; // P-frame is last in display order

                // 1+2. Encode P-frame anchor with save_bwd_ref=true: copies
                // current ref → bwd_ref at the START of the same command encoder,
                // then does ME/MC/encode/local-decode. This eliminates a separate
                // GPU submit for the reference copy.
                let p_config = if let Some(ref rc) = rate_ctrl {
                    let mut cfg = config.clone();
                    cfg.quantization_step = rc.estimate_qstep();
                    cfg
                } else {
                    config.clone()
                };
                let (compressed, new_mv_buf) = self.encode_pframe(
                    ctx,
                    frames[p_display],
                    width,
                    height,
                    padded_w,
                    padded_h,
                    padded_pixels,
                    &info,
                    &p_config,
                    prev_mv_buf.as_ref().or(Some(&zero_mv_buf)),
                    true,
                );
                prev_mv_buf = Some(new_mv_buf);
                if let Some(ref mut rc) = rate_ctrl {
                    rc.update(p_config.quantization_step, compressed.bpp());
                }
                results[p_display] = Some(compressed);

                // 3. Swap: gpu_ref_planes = past anchor, gpu_bwd_ref_planes = future P
                self.swap_ref_planes();

                // 4. Encode B-frames between past and future anchors.
                // First B-frame uses zero MVs as predictor (fast ±4 search around (0,0)).
                // Subsequent B-frames use previous B's MVs for temporal prediction.
                let mut prev_bidir_fwd_mv: Option<wgpu::Buffer> = None;
                let mut prev_bidir_bwd_mv: Option<wgpu::Buffer> = None;
                for b in 0..b_count {
                    let b_display = group_start + b;
                    let b_config = if let Some(ref rc) = rate_ctrl {
                        let mut cfg = config.clone();
                        cfg.quantization_step = rc.estimate_qstep();
                        cfg
                    } else {
                        config.clone()
                    };
                    // Use zero_mv_buf as fallback predictor for the first B-frame
                    let fwd_pred = prev_bidir_fwd_mv.as_ref().or(Some(&zero_mv_buf));
                    let bwd_pred = prev_bidir_bwd_mv.as_ref().or(Some(&zero_mv_buf));
                    let (compressed, new_fwd_mv, new_bwd_mv) = self.encode_bframe(
                        ctx,
                        frames[b_display],
                        width,
                        height,
                        padded_w,
                        padded_h,
                        padded_pixels,
                        &info,
                        &b_config,
                        fwd_pred,
                        bwd_pred,
                    );
                    prev_bidir_fwd_mv = Some(new_fwd_mv);
                    prev_bidir_bwd_mv = Some(new_bwd_mv);
                    if let Some(ref mut rc) = rate_ctrl {
                        rc.update(b_config.quantization_step, compressed.bpp());
                    }
                    results[b_display] = Some(compressed);
                }

                // 5. Swap back: gpu_ref_planes = decoded P (for next group's forward ref)
                self.swap_ref_planes();
            }

            // Remainder frames (< group_size) encoded as P-frames
            let rem_start = display_idx + full_groups * group_size;
            for j in rem_start..next_key {
                let p_config = if let Some(ref rc) = rate_ctrl {
                    let mut cfg = config.clone();
                    cfg.quantization_step = rc.estimate_qstep();
                    cfg
                } else {
                    config.clone()
                };
                let (compressed, new_mv_buf) = self.encode_pframe(
                    ctx,
                    frames[j],
                    width,
                    height,
                    padded_w,
                    padded_h,
                    padded_pixels,
                    &info,
                    &p_config,
                    prev_mv_buf.as_ref().or(Some(&zero_mv_buf)),
                    false,
                );
                prev_mv_buf = Some(new_mv_buf);
                if let Some(ref mut rc) = rate_ctrl {
                    rc.update(p_config.quantization_step, compressed.bpp());
                }
                results[j] = Some(compressed);
            }

            display_idx = next_key;
        }

        results.into_iter().map(|o| o.unwrap()).collect()
    }

    /// I-frame local decode using quantized data already on GPU.
    ///
    /// After `encode()`, quantized planes are persisted in:
    ///   Y → `mc_out`, Co → `ref_upload`, Cg → `plane_b`
    /// and the AQ weight map is in `weight_map_buf`.
    ///
    /// This avoids the expensive CPU entropy decode + re-upload that
    /// `local_decode_iframe` performs. All work is purely GPU.
    fn local_decode_iframe_gpu(
        &mut self,
        ctx: &GpuContext,
        frame: &CompressedFrame,
        padded_w: u32,
        padded_h: u32,
        padded_pixels: usize,
    ) {
        let config = &frame.config;
        let info = &frame.info;
        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;
        let tiles_per_plane = tiles_x * tiles_y;
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;

        let has_cfl = frame.cfl_alphas.is_some();
        let cfl_alphas_per_plane;

        // Upload CfL alphas if present (tiny: ~2KB, negligible vs 30MB entropy decode)
        if has_cfl {
            let cfl_data = frame.cfl_alphas.as_ref().unwrap();
            cfl_alphas_per_plane = tiles_per_plane * cfl_data.num_subbands as usize;
            let all_cfl_f32: Vec<f32> = cfl_data
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
            let bufs = self.cached.as_ref().unwrap();
            ctx.queue
                .write_buffer(&bufs.dq_alpha, 0, bytemuck::cast_slice(&all_cfl_f32));
        } else {
            cfl_alphas_per_plane = 0;
        }

        let bufs = self.cached.as_ref().unwrap();

        // Weight map is already in weight_map_buf from encode()'s forward pass
        let wm_param = if frame.weight_map.is_some() {
            let (_, ll_bx, _, tx) = adaptive::weight_map_dims(
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
            );
            let ll_size = config.tile_size >> config.wavelet_levels;
            let ll_block_size = AQ_LL_BLOCK_SIZE.min(ll_size);
            Some((&bufs.weight_map_buf, ll_block_size, ll_bx, tx))
        } else {
            None
        };

        // Quantized planes persisted by encode():
        //   Y → mc_out, Co → ref_upload, Cg → plane_b
        let quant_bufs: [&wgpu::Buffer; 3] = [&bufs.mc_out, &bufs.ref_upload, &bufs.plane_b];

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("local_decode_gpu"),
            });

        for p in 0..3 {
            let weights = if p == 0 {
                &weights_luma
            } else {
                &weights_chroma
            };

            // Dequantize: quant_buf → cg_plane (scratch — preserves plane_b for Cg)
            self.quantize.dispatch_adaptive(
                ctx,
                &mut cmd,
                quant_bufs[p],
                &bufs.cg_plane,
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
                cmd.copy_buffer_to_buffer(&bufs.cg_plane, 0, &bufs.recon_y, 0, plane_size);
            }

            if p > 0 && has_cfl {
                // CfL inverse prediction: cg_plane (dequant residual) + alpha * Y → plane_c
                let plane_alpha_offset = (p - 1) * cfl_alphas_per_plane;
                let plane_alpha_byte_offset =
                    (plane_alpha_offset * std::mem::size_of::<f32>()) as u64;
                let plane_alpha_byte_size =
                    (cfl_alphas_per_plane * std::mem::size_of::<f32>()) as u64;

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
                    &bufs.cg_plane,
                    &bufs.recon_y,
                    &bufs.raw_alpha,
                    &bufs.plane_c,
                    padded_pixels as u32,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                );

                // Inverse wavelet: plane_c → cg_plane(scratch) → plane_a
                // Uses cg_plane as scratch (not plane_b) to preserve Cg quantized data
                self.transform.inverse(
                    ctx,
                    &mut cmd,
                    &bufs.plane_c,
                    &bufs.cg_plane,
                    &bufs.plane_a,
                    info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );
            } else {
                // Standard path: cg_plane → plane_c(scratch) → plane_a
                self.transform.inverse(
                    ctx,
                    &mut cmd,
                    &bufs.cg_plane,
                    &bufs.plane_c,
                    &bufs.plane_a,
                    info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );
            }

            // Copy decoded plane to persistent GPU reference buffer
            cmd.copy_buffer_to_buffer(&bufs.plane_a, 0, &bufs.gpu_ref_planes[p], 0, plane_size);
        }

        ctx.queue.submit(Some(cmd.finish()));
    }

    /// Encode a P-frame: ME -> MC -> encode residual -> local decode loop.
    /// Reference planes are read from and written to `gpu_ref_planes` — no CPU roundtrip.
    ///
    /// Optimized pipeline: MV buffer stays on GPU for MC (no readback/re-upload roundtrip).
    /// GPU work is batched into minimal command encoder submits. MV readback is deferred
    /// to the end for bitstream serialization only.
    /// Encode a P-frame. Returns `(compressed_frame, mv_buffer)`.
    /// The `mv_buffer` can be passed as `predictor_mvs` to the next P-frame for temporal
    /// MV prediction (skip coarse search, only fine-refine around the predicted MV).
    #[allow(clippy::too_many_arguments)]
    /// Encode a P-frame.
    ///
    /// `save_bwd_ref`: when true, copies gpu_ref_planes → gpu_bwd_ref_planes at
    /// the start of the command encoder (before ME overwrites anything). This folds
    /// the reference copy into the same GPU submission, eliminating a separate submit.
    #[allow(clippy::too_many_arguments)]
    fn encode_pframe(
        &mut self,
        ctx: &GpuContext,
        rgb_data: &[f32],
        width: u32,
        height: u32,
        padded_w: u32,
        padded_h: u32,
        padded_pixels: usize,
        info: &FrameInfo,
        config: &CodecConfig,
        predictor_mvs: Option<&wgpu::Buffer>,
        save_bwd_ref: bool,
    ) -> (CompressedFrame, wgpu::Buffer) {
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;

        self.ensure_cached(ctx, padded_w, padded_h, width, height);
        let bufs = self.cached.as_ref().unwrap();

        // Upload raw (unpadded) frame — GPU shader handles padding
        ctx.queue
            .write_buffer(&bufs.raw_input_buf, 0, bytemuck::cast_slice(rgb_data));

        let entropy_mode = EntropyMode::from_config(config);
        let tile_size = config.tile_size as usize;
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;
        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();
        let use_gpu_encode =
            config.gpu_entropy_encode && config.entropy_coder != EntropyCoder::Bitplane;

        let me_blocks_x = padded_w / ME_BLOCK_SIZE;
        let me_blocks_y = padded_h / ME_BLOCK_SIZE;
        let me_total_blocks = me_blocks_x * me_blocks_y;

        let mut rans_tiles: Vec<rans::InterleavedRansTile> = Vec::new();
        let mut subband_tiles: Vec<rans::SubbandRansTile> = Vec::new();
        let mut bp_tiles: Vec<bitplane::BitplaneTile> = Vec::new();
        let mut rice_tiles: Vec<rice::RiceTile> = Vec::new();

        // === Batched GPU pipeline: preprocess + ME + forward encode ===
        // MV buffer stays on GPU — used directly by MC without readback/re-upload.
        let mv_buf;

        if use_gpu_encode {
            // === Fully batched GPU pipeline: forward + entropy + local decode ===
            // Single command encoder, single submit, single poll.
            // Eliminates GPU pipeline stalls between forward/entropy/decode phases.
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_batch_all"),
                });

            // Phase 0a: Copy ref planes to bwd_ref if requested (for B-frame group)
            if save_bwd_ref {
                for p in 0..3 {
                    cmd.copy_buffer_to_buffer(
                        &bufs.gpu_ref_planes[p],
                        0,
                        &bufs.gpu_bwd_ref_planes[p],
                        0,
                        plane_size,
                    );
                }
            }

            // Phase 0b: GPU padding (raw → padded, edge-replicate)
            self.dispatch_gpu_pad_cached(ctx, &mut cmd, padded_w, padded_h);

            // Phase 1: Preprocess + ME + MC + transform + quantize (all 3 planes)
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
            self.deinterleaver.dispatch(
                ctx,
                &mut cmd,
                &bufs.color_out,
                &bufs.plane_a,
                &bufs.co_plane,
                &bufs.cg_plane,
                padded_pixels as u32,
            );

            let me_params = if predictor_mvs.is_some() {
                &bufs.me_params_pred
            } else {
                &bufs.me_params_nopred
            };
            mv_buf = self.motion.estimate_cached(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.gpu_ref_planes[0],
                padded_w,
                padded_h,
                predictor_mvs,
                me_params,
                &bufs.me_sad_buf,
                &bufs.me_dummy_pred,
            );

            for p in 0..3 {
                let weights = if p == 0 {
                    &weights_luma
                } else {
                    &weights_chroma
                };
                let cur_plane = match p {
                    0 => &bufs.plane_a,
                    1 => &bufs.co_plane,
                    _ => &bufs.cg_plane,
                };

                self.motion.compensate_cached(
                    ctx,
                    &mut cmd,
                    cur_plane,
                    &bufs.gpu_ref_planes[p],
                    &mv_buf,
                    &bufs.mc_out,
                    padded_w,
                    padded_h,
                    &bufs.mc_fwd_params,
                );
                cmd.copy_buffer_to_buffer(&bufs.mc_out, 0, &bufs.plane_a, 0, plane_size);

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

                if p == 0 {
                    cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.recon_y, 0, plane_size);
                } else if p == 1 {
                    cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.co_plane, 0, plane_size);
                }
            }

            // Phase 2: GPU entropy encode dispatches + staging copies (same cmd encoder)
            let use_rice = matches!(entropy_mode, EntropyMode::Rice);
            if use_rice {
                self.gpu_rice_encoder.dispatch_3planes_to_cmd(
                    ctx,
                    &mut cmd,
                    [&bufs.recon_y, &bufs.co_plane, &bufs.plane_b],
                    info,
                    config.wavelet_levels,
                );
            } else {
                self.gpu_encoder.dispatch_3planes_to_cmd(
                    ctx,
                    &mut cmd,
                    [&bufs.recon_y, &bufs.co_plane, &bufs.plane_b],
                    info,
                    config.per_subband_entropy,
                    config.wavelet_levels,
                );
            }

            // Phase 3: Local decode dispatches (same cmd encoder)
            let quant_bufs: [&wgpu::Buffer; 3] =
                [&bufs.recon_y, &bufs.co_plane, &bufs.plane_b];
            for (p, quant_buf) in quant_bufs.iter().enumerate() {
                let weights = if p == 0 {
                    &weights_luma
                } else {
                    &weights_chroma
                };

                self.quantize.dispatch(
                    ctx,
                    &mut cmd,
                    quant_buf,
                    &bufs.cg_plane,
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
                    &bufs.cg_plane,
                    &bufs.plane_c,
                    &bufs.plane_a,
                    info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );
                self.motion.compensate_cached(
                    ctx,
                    &mut cmd,
                    &bufs.plane_a,
                    &bufs.gpu_ref_planes[p],
                    &mv_buf,
                    &bufs.recon_out,
                    padded_w,
                    padded_h,
                    &bufs.mc_inv_params,
                );
                cmd.copy_buffer_to_buffer(
                    &bufs.recon_out,
                    0,
                    &bufs.gpu_ref_planes[p],
                    0,
                    plane_size,
                );
            }

            // MV staging copy using cached staging buffer
            cmd.copy_buffer_to_buffer(&mv_buf, 0, &bufs.mv_staging_buf, 0, bufs.mv_staging_size);

            // === Single submit for everything ===
            let _t_submit = std::time::Instant::now();
            ctx.queue.submit(Some(cmd.finish()));

            // Single poll drains forward + entropy + local decode + MV copy
            if use_rice {
                rice_tiles = self.gpu_rice_encoder.finish_3planes_readback(
                    ctx,
                    info,
                    config.wavelet_levels,
                );
            } else {
                let (mut rt, mut st) = self.gpu_encoder.finish_3planes_readback(
                    ctx,
                    info,
                    config.per_subband_entropy,
                    config.wavelet_levels,
                );
                rans_tiles.append(&mut rt);
                subband_tiles.append(&mut st);
            }
            if std::env::var("GNC_PROFILE").is_ok() {
                eprintln!("  P-frame GPU+readback: {:.1}ms", _t_submit.elapsed().as_secs_f64() * 1000.0);
            }

            let mvs = MotionEstimator::finish_mv_readback_cached(ctx, &bufs.mv_staging_buf, bufs.mv_staging_size, bufs.me_total_blocks);

            let entropy = match entropy_mode {
                EntropyMode::Bitplane => EntropyData::Bitplane(bp_tiles),
                EntropyMode::SubbandRans | EntropyMode::SubbandRansCtx => {
                    EntropyData::SubbandRans(subband_tiles)
                }
                EntropyMode::Rans => EntropyData::Rans(rans_tiles),
                EntropyMode::Rice => EntropyData::Rice(rice_tiles),
            };

            return (CompressedFrame {
                info: *info,
                config: config.clone(),
                entropy,
                cfl_alphas: None,
                weight_map: None,
                frame_type: FrameType::Predicted,
                motion_field: Some(MotionField {
                    vectors: mvs,
                    block_size: ME_BLOCK_SIZE,
                    backward_vectors: None,
                    block_modes: None,
                }),
            }, mv_buf);
        } else {
            // CPU entropy path: preprocess + ME batched, then per-plane submits
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_preprocess_me"),
                });
            if save_bwd_ref {
                for p in 0..3 {
                    cmd.copy_buffer_to_buffer(
                        &bufs.gpu_ref_planes[p],
                        0,
                        &bufs.gpu_bwd_ref_planes[p],
                        0,
                        plane_size,
                    );
                }
            }
            self.dispatch_gpu_pad_cached(ctx, &mut cmd, padded_w, padded_h);
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
            self.deinterleaver.dispatch(
                ctx,
                &mut cmd,
                &bufs.color_out,
                &bufs.plane_a,
                &bufs.co_plane,
                &bufs.cg_plane,
                padded_pixels as u32,
            );
            let (mb, _sad) = self.motion.estimate(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.gpu_ref_planes[0],
                padded_w,
                padded_h,
                predictor_mvs,
            );
            mv_buf = mb;
            ctx.queue.submit(Some(cmd.finish()));

            for p in 0..3 {
                let weights = if p == 0 {
                    &weights_luma
                } else {
                    &weights_chroma
                };
                let cur_plane = match p {
                    0 => &bufs.plane_a,
                    1 => &bufs.co_plane,
                    _ => &bufs.cg_plane,
                };

                let mut cmd = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("pf_enc"),
                    });
                self.motion.compensate(
                    ctx,
                    &mut cmd,
                    cur_plane,
                    &bufs.gpu_ref_planes[p],
                    &mv_buf,
                    &bufs.mc_out,
                    padded_w,
                    padded_h,
                    true,
                );
                cmd.copy_buffer_to_buffer(&bufs.mc_out, 0, &bufs.plane_a, 0, plane_size);
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
                if p == 0 {
                    cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.recon_y, 0, plane_size);
                } else if p == 1 {
                    cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.co_plane, 0, plane_size);
                }
                ctx.queue.submit(Some(cmd.finish()));

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
                    &mut rice_tiles,
                );
            }
        }

        let entropy = match entropy_mode {
            EntropyMode::Bitplane => EntropyData::Bitplane(bp_tiles),
            EntropyMode::SubbandRans | EntropyMode::SubbandRansCtx => {
                EntropyData::SubbandRans(subband_tiles)
            }
            EntropyMode::Rans => EntropyData::Rans(rans_tiles),
            EntropyMode::Rice => EntropyData::Rice(rice_tiles),
        };

        // === Batched local decode + MV copy: single command encoder ===
        // Quantized data on GPU: Y in recon_y, Co in co_plane, Cg in plane_b.
        // Uses cg_plane as scratch (original Cg spatial data no longer needed).
        // MV staging copy piggybacks on this batch to avoid an extra submit.
        let mv_staging =
            MotionEstimator::create_mv_staging(ctx, &mv_buf, me_total_blocks);
        let quant_bufs: [&wgpu::Buffer; 3] = [&bufs.recon_y, &bufs.co_plane, &bufs.plane_b];
        {
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_local_decode"),
                });

            for (p, quant_buf) in quant_bufs.iter().enumerate() {
                let weights = if p == 0 {
                    &weights_luma
                } else {
                    &weights_chroma
                };

                self.quantize.dispatch(
                    ctx,
                    &mut cmd,
                    quant_buf,
                    &bufs.cg_plane,
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
                    &bufs.cg_plane,
                    &bufs.plane_c,
                    &bufs.plane_a,
                    info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );
                self.motion.compensate(
                    ctx,
                    &mut cmd,
                    &bufs.plane_a,
                    &bufs.gpu_ref_planes[p],
                    &mv_buf,
                    &bufs.recon_out,
                    padded_w,
                    padded_h,
                    false,
                );
                cmd.copy_buffer_to_buffer(
                    &bufs.recon_out,
                    0,
                    &bufs.gpu_ref_planes[p],
                    0,
                    plane_size,
                );
            }

            // MV copy in same batch — no extra submit round-trip
            cmd.copy_buffer_to_buffer(&mv_buf, 0, &mv_staging.buffer, 0, mv_staging.size);

            ctx.queue.submit(Some(cmd.finish()));
        }

        // Single poll drains local decode + MV copy together
        let mvs = MotionEstimator::finish_mv_readback(ctx, &mv_staging);

        (CompressedFrame {
            info: *info,
            config: config.clone(),
            entropy,
            cfl_alphas: None,
            weight_map: None,
            frame_type: FrameType::Predicted,
            motion_field: Some(MotionField {
                vectors: mvs,
                block_size: ME_BLOCK_SIZE,
                backward_vectors: None,
                block_modes: None,
            }),
        }, mv_buf)
    }

    /// Encode a B-frame with bidirectional prediction.
    /// Forward reference in `gpu_ref_planes`, backward reference in `gpu_bwd_ref_planes`.
    /// B-frames do NOT update gpu_ref_planes (they are non-reference frames).
    ///
    /// Optimized: GPU buffers used directly for MC (no readback/re-upload roundtrip).
    /// Bidir data (fwd MVs, bwd MVs, block modes) read back in single batched poll.
    ///
    #[allow(clippy::too_many_arguments)]
    fn encode_bframe(
        &mut self,
        ctx: &GpuContext,
        rgb_data: &[f32],
        width: u32,
        height: u32,
        padded_w: u32,
        padded_h: u32,
        padded_pixels: usize,
        info: &FrameInfo,
        config: &CodecConfig,
        predictor_fwd_mvs: Option<&wgpu::Buffer>,
        predictor_bwd_mvs: Option<&wgpu::Buffer>,
    ) -> (CompressedFrame, wgpu::Buffer, wgpu::Buffer) {
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;

        self.ensure_cached(ctx, padded_w, padded_h, width, height);
        let bufs = self.cached.as_ref().unwrap();

        // Upload raw (unpadded) frame — GPU shader handles padding
        ctx.queue
            .write_buffer(&bufs.raw_input_buf, 0, bytemuck::cast_slice(rgb_data));

        let entropy_mode = EntropyMode::from_config(config);
        let tile_size = config.tile_size as usize;
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;
        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();
        let use_gpu_encode =
            config.gpu_entropy_encode && config.entropy_coder != EntropyCoder::Bitplane;

        let me_blocks_x = padded_w / ME_BLOCK_SIZE;
        let me_blocks_y = padded_h / ME_BLOCK_SIZE;
        let me_total_blocks = me_blocks_x * me_blocks_y;

        let mut rans_tiles: Vec<rans::InterleavedRansTile> = Vec::new();
        let mut subband_tiles: Vec<rans::SubbandRansTile> = Vec::new();
        let mut bp_tiles: Vec<bitplane::BitplaneTile> = Vec::new();
        let mut rice_tiles: Vec<rice::RiceTile> = Vec::new();

        // === Batched GPU pipeline: preprocess + bidir ME + forward encode ===
        // MV/mode buffers stay on GPU — used directly by bidir MC.
        let fwd_mv_buf;
        let bwd_mv_buf;
        let modes_buf_owned: wgpu::Buffer; // only used in CPU fallback path

        if use_gpu_encode {
            // === Fully batched B-frame: forward + entropy + bidir staging ===
            // Single command encoder, single submit, single poll.
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bf_batch_all"),
                });

            // Phase 0: GPU padding (raw → padded, edge-replicate)
            self.dispatch_gpu_pad_cached(ctx, &mut cmd, padded_w, padded_h);

            // Phase 1: Preprocess + bidir ME + MC + transform + quantize
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
            self.deinterleaver.dispatch(
                ctx,
                &mut cmd,
                &bufs.color_out,
                &bufs.plane_a,
                &bufs.co_plane,
                &bufs.cg_plane,
                padded_pixels as u32,
            );

            let have_bidir_pred = predictor_fwd_mvs.is_some() && predictor_bwd_mvs.is_some();
            let bidir_params = if have_bidir_pred {
                &bufs.bidir_params_pred
            } else {
                &bufs.bidir_params_nopred
            };
            let (fmb, bmb) = self.motion.estimate_bidir_cached(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.gpu_ref_planes[0],
                &bufs.gpu_bwd_ref_planes[0],
                padded_w,
                padded_h,
                predictor_fwd_mvs,
                predictor_bwd_mvs,
                bidir_params,
                &bufs.bidir_sad_buf,
                &bufs.bidir_modes_scratch,
                &bufs.me_dummy_pred,
            );
            fwd_mv_buf = fmb;
            bwd_mv_buf = bmb;

            for p in 0..3 {
                let weights = if p == 0 {
                    &weights_luma
                } else {
                    &weights_chroma
                };
                let cur_plane = match p {
                    0 => &bufs.plane_a,
                    1 => &bufs.co_plane,
                    _ => &bufs.cg_plane,
                };

                self.motion.compensate_bidir_cached(
                    ctx,
                    &mut cmd,
                    cur_plane,
                    &bufs.gpu_ref_planes[p],
                    &bufs.gpu_bwd_ref_planes[p],
                    &fwd_mv_buf,
                    &bwd_mv_buf,
                    &bufs.bidir_modes_scratch,
                    &bufs.mc_out,
                    padded_w,
                    padded_h,
                    &bufs.mc_bidir_fwd_params,
                );
                cmd.copy_buffer_to_buffer(&bufs.mc_out, 0, &bufs.plane_a, 0, plane_size);

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

                if p == 0 {
                    cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.recon_y, 0, plane_size);
                } else if p == 1 {
                    cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.co_plane, 0, plane_size);
                }
            }

            // Phase 2: GPU entropy encode dispatches (same cmd)
            let use_rice = matches!(entropy_mode, EntropyMode::Rice);
            if use_rice {
                self.gpu_rice_encoder.dispatch_3planes_to_cmd(
                    ctx,
                    &mut cmd,
                    [&bufs.recon_y, &bufs.co_plane, &bufs.plane_b],
                    info,
                    config.wavelet_levels,
                );
            } else {
                self.gpu_encoder.dispatch_3planes_to_cmd(
                    ctx,
                    &mut cmd,
                    [&bufs.recon_y, &bufs.co_plane, &bufs.plane_b],
                    info,
                    config.per_subband_entropy,
                    config.wavelet_levels,
                );
            }

            // Phase 3: Bidir MV + modes staging copies using cached staging buffers
            let modes_size = (bufs.me_total_blocks as u64) * 4;
            cmd.copy_buffer_to_buffer(
                &fwd_mv_buf, 0, &bufs.bidir_fwd_staging, 0, bufs.mv_staging_size,
            );
            cmd.copy_buffer_to_buffer(
                &bwd_mv_buf, 0, &bufs.bidir_bwd_staging, 0, bufs.mv_staging_size,
            );
            cmd.copy_buffer_to_buffer(
                &bufs.bidir_modes_scratch, 0, &bufs.bidir_modes_staging, 0, modes_size,
            );

            // Single submit
            let _t_submit = std::time::Instant::now();
            ctx.queue.submit(Some(cmd.finish()));

            // Single poll drains everything
            if use_rice {
                rice_tiles = self.gpu_rice_encoder.finish_3planes_readback(
                    ctx,
                    info,
                    config.wavelet_levels,
                );
            } else {
                let (mut rt, mut st) = self.gpu_encoder.finish_3planes_readback(
                    ctx,
                    info,
                    config.per_subband_entropy,
                    config.wavelet_levels,
                );
                rans_tiles.append(&mut rt);
                subband_tiles.append(&mut st);
            }
            if std::env::var("GNC_PROFILE").is_ok() {
                eprintln!("  B-frame GPU+readback: {:.1}ms", _t_submit.elapsed().as_secs_f64() * 1000.0);
            }

            let (fwd_mvs, bwd_mvs, block_modes) =
                MotionEstimator::finish_bidir_readback_cached(
                    ctx,
                    &bufs.bidir_fwd_staging,
                    &bufs.bidir_bwd_staging,
                    &bufs.bidir_modes_staging,
                    bufs.mv_staging_size,
                    modes_size,
                    bufs.me_total_blocks,
                );

            let entropy = match entropy_mode {
                EntropyMode::Bitplane => EntropyData::Bitplane(bp_tiles),
                EntropyMode::SubbandRans | EntropyMode::SubbandRansCtx => {
                    EntropyData::SubbandRans(subband_tiles)
                }
                EntropyMode::Rans => EntropyData::Rans(rans_tiles),
                EntropyMode::Rice => EntropyData::Rice(rice_tiles),
            };

            return (CompressedFrame {
                info: *info,
                config: config.clone(),
                entropy,
                cfl_alphas: None,
                weight_map: None,
                frame_type: FrameType::Bidirectional,
                motion_field: Some(MotionField {
                    vectors: fwd_mvs,
                    block_size: ME_BLOCK_SIZE,
                    backward_vectors: Some(bwd_mvs),
                    block_modes: Some(block_modes),
                }),
            }, fwd_mv_buf, bwd_mv_buf);
        } else {
            // CPU entropy path: preprocess + bidir ME batched, then per-plane submits
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bf_preprocess_me"),
                });
            self.dispatch_gpu_pad_cached(ctx, &mut cmd, padded_w, padded_h);
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
            self.deinterleaver.dispatch(
                ctx,
                &mut cmd,
                &bufs.color_out,
                &bufs.plane_a,
                &bufs.co_plane,
                &bufs.cg_plane,
                padded_pixels as u32,
            );
            let (fmb, bmb, mmb, _sad) = self.motion.estimate_bidir(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.gpu_ref_planes[0],
                &bufs.gpu_bwd_ref_planes[0],
                padded_w,
                padded_h,
                predictor_fwd_mvs,
                predictor_bwd_mvs,
            );
            fwd_mv_buf = fmb;
            bwd_mv_buf = bmb;
            modes_buf_owned = mmb;
            ctx.queue.submit(Some(cmd.finish()));

            for p in 0..3 {
                let weights = if p == 0 {
                    &weights_luma
                } else {
                    &weights_chroma
                };
                let cur_plane = match p {
                    0 => &bufs.plane_a,
                    1 => &bufs.co_plane,
                    _ => &bufs.cg_plane,
                };

                let mut cmd = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("bf_enc"),
                    });
                self.motion.compensate_bidir(
                    ctx,
                    &mut cmd,
                    cur_plane,
                    &bufs.gpu_ref_planes[p],
                    &bufs.gpu_bwd_ref_planes[p],
                    &fwd_mv_buf,
                    &bwd_mv_buf,
                    &modes_buf_owned,
                    &bufs.mc_out,
                    padded_w,
                    padded_h,
                    true,
                );
                cmd.copy_buffer_to_buffer(&bufs.mc_out, 0, &bufs.plane_a, 0, plane_size);
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
                if p == 0 {
                    cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.recon_y, 0, plane_size);
                } else if p == 1 {
                    cmd.copy_buffer_to_buffer(&bufs.plane_b, 0, &bufs.co_plane, 0, plane_size);
                }
                ctx.queue.submit(Some(cmd.finish()));

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
                    &mut rice_tiles,
                );
            }
        }

        let entropy = match entropy_mode {
            EntropyMode::Bitplane => EntropyData::Bitplane(bp_tiles),
            EntropyMode::SubbandRans | EntropyMode::SubbandRansCtx => {
                EntropyData::SubbandRans(subband_tiles)
            }
            EntropyMode::Rans => EntropyData::Rans(rans_tiles),
            EntropyMode::Rice => EntropyData::Rice(rice_tiles),
        };

        // === Deferred batched readback: single submit + poll for all bidir data ===
        let (fwd_mvs, bwd_mvs, block_modes) = MotionEstimator::read_bidir_data(
            ctx,
            &fwd_mv_buf,
            &bwd_mv_buf,
            &modes_buf_owned,
            me_total_blocks,
        );

        (CompressedFrame {
            info: *info,
            config: config.clone(),
            entropy,
            cfl_alphas: None,
            weight_map: None,
            frame_type: FrameType::Bidirectional,
            motion_field: Some(MotionField {
                vectors: fwd_mvs,
                block_size: ME_BLOCK_SIZE,
                backward_vectors: Some(bwd_mvs),
                block_modes: Some(block_modes),
            }),
        }, fwd_mv_buf, bwd_mv_buf)
    }

    /// Swap gpu_ref_planes ↔ gpu_bwd_ref_planes at the Rust level (zero GPU cost).
    /// Used during B-frame encoding to toggle between past/future references.
    fn swap_ref_planes(&mut self) {
        let bufs = self.cached.as_mut().unwrap();
        std::mem::swap(&mut bufs.gpu_ref_planes, &mut bufs.gpu_bwd_ref_planes);
    }
}
