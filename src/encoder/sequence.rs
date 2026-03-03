use wgpu;

use super::adaptive::{self, AQ_LL_BLOCK_SIZE};
use super::bitplane;
use super::cfl;
use super::entropy_helpers::{self, encode_entropy, EntropyMode};
use super::motion::{MotionEstimator, ME_BLOCK_SIZE, ME_SPLIT_BLOCK_SIZE};
use super::pipeline::EncoderPipeline;
use super::rans;
use super::rice;
use super::huffman;
use super::diagnostics;
use super::rate_control::RateController;
use crate::{
    CodecConfig, CompressedFrame, EntropyCoder, EntropyData, FrameInfo, FrameType, GpuContext,
    MotionField, TemporalEncodedSequence, TemporalGroup, TemporalTransform,
};
use crate::temporal;

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
        self.encode_sequence_streaming(ctx, frames.len(), |i| frames[i].to_vec(), width, height, config, fps)
    }

    /// Streaming variant that loads frames on-demand via a closure.
    /// Only keeps a small window of frames in memory at any time,
    /// enabling encoding of long sequences without OOM.
    #[allow(clippy::too_many_arguments)] // sequence encode needs all these params
    pub fn encode_sequence_streaming(
        &mut self,
        ctx: &GpuContext,
        frame_count: usize,
        mut load_frame: impl FnMut(usize) -> Vec<f32>,
        width: u32,
        height: u32,
        config: &CodecConfig,
        fps: f64,
    ) -> Vec<CompressedFrame> {
        let ki = config.keyframe_interval as usize;
        let use_bframes = ki >= 4;
        let b_count = if use_bframes { B_FRAMES_PER_GROUP } else { 0 };
        let group_size = b_count + 1;

        let n = frame_count;
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
        // the predicted MV. First P-frame after a keyframe uses None (full ±32 coarse
        // search) since there's no reliable predictor — using zero-MVs with the
        // predictor path would limit search to ±2 pixels and miss real motion.
        let mut prev_mv_buf: Option<wgpu::Buffer> = None;

        let diag_enabled = diagnostics::enabled();
        let mut last_iframe_bytes: Option<usize> = None;

        // Temporal wavelet diagnostic: staging buffers + previous frame state
        let diag_twav_staging: Option<[wgpu::Buffer; 3]> = if diag_enabled {
            Some(std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(
                        ["diag_twav_y", "diag_twav_co", "diag_twav_cg"][i],
                    ),
                    size: (padded_pixels * std::mem::size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }))
        } else {
            None
        };
        let mut prev_twav_coeffs: Option<[Vec<f32>; 3]> = None;

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
                let frame_data = load_frame(display_idx);
                let mut compressed =
                    self.encode(ctx, &frame_data, width, height, &frame_config);
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
                prev_mv_buf = None;
                if diag_enabled {
                    last_iframe_bytes = Some(compressed.byte_size());
                    let d = diagnostics::collect(display_idx, &compressed, frame_config.quantization_step, last_iframe_bytes);
                    diagnostics::print(&d);

                    // Temporal wavelet diagnostic: capture original-signal coefficients
                    if let Some(ref stg) = diag_twav_staging {
                        let coeffs = self.diag_original_wavelet_coefficients(
                            ctx, &frame_data, padded_w, padded_h, padded_pixels,
                            &info, &frame_config, stg,
                        );
                        if let Some(ref prev) = prev_twav_coeffs {
                            let y_stats = diagnostics::compute_temporal_wavelet(
                                &coeffs[0], &prev[0], padded_w, padded_h,
                                config.tile_size, config.wavelet_levels, frame_config.quantization_step,
                            );
                            let co_stats = diagnostics::compute_temporal_wavelet(
                                &coeffs[1], &prev[1], padded_w, padded_h,
                                config.tile_size, config.wavelet_levels, frame_config.quantization_step,
                            );
                            let cg_stats = diagnostics::compute_temporal_wavelet(
                                &coeffs[2], &prev[2], padded_w, padded_h,
                                config.tile_size, config.wavelet_levels, frame_config.quantization_step,
                            );
                            diagnostics::print_temporal_wavelet(
                                display_idx, FrameType::Intra, &y_stats, &co_stats, &cg_stats,
                            );
                        }
                        prev_twav_coeffs = Some(coeffs);
                    }
                }
                results[display_idx] = Some(compressed);
                display_idx += 1;
                continue;
            }

            if !use_bframes {
                // P-frame only mode — skip local decode if next frame is keyframe or EOSequence
                let next_is_key_or_end = display_idx + 1 >= n
                    || (ki > 1 && (display_idx + 1) % ki == 0);
                let frame_data = load_frame(display_idx);
                let (compressed, new_mv_buf) = self.encode_pframe(
                    ctx,
                    &frame_data,
                    width,
                    height,
                    padded_w,
                    padded_h,
                    padded_pixels,
                    &info,
                    &frame_config,
                    prev_mv_buf.as_ref(),
                    false,
                    !next_is_key_or_end,
                );
                prev_mv_buf = Some(new_mv_buf);
                if let Some(ref mut rc) = rate_ctrl {
                    rc.update(frame_config.quantization_step, compressed.bpp());
                }
                if diag_enabled {
                    let d = diagnostics::collect(display_idx, &compressed, frame_config.quantization_step, last_iframe_bytes);
                    diagnostics::print(&d);

                    if let Some(ref stg) = diag_twav_staging {
                        let coeffs = self.diag_original_wavelet_coefficients(
                            ctx, &frame_data, padded_w, padded_h, padded_pixels,
                            &info, config, stg,
                        );
                        if let Some(ref prev) = prev_twav_coeffs {
                            let y_stats = diagnostics::compute_temporal_wavelet(
                                &coeffs[0], &prev[0], padded_w, padded_h,
                                config.tile_size, config.wavelet_levels, config.quantization_step,
                            );
                            let co_stats = diagnostics::compute_temporal_wavelet(
                                &coeffs[1], &prev[1], padded_w, padded_h,
                                config.tile_size, config.wavelet_levels, config.quantization_step,
                            );
                            let cg_stats = diagnostics::compute_temporal_wavelet(
                                &coeffs[2], &prev[2], padded_w, padded_h,
                                config.tile_size, config.wavelet_levels, config.quantization_step,
                            );
                            diagnostics::print_temporal_wavelet(
                                display_idx, FrameType::Predicted, &y_stats, &co_stats, &cg_stats,
                            );
                        }
                        prev_twav_coeffs = Some(coeffs);
                    }
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
                let p_frame_data = load_frame(p_display);
                let (compressed, new_mv_buf) = self.encode_pframe(
                    ctx,
                    &p_frame_data,
                    width,
                    height,
                    padded_w,
                    padded_h,
                    padded_pixels,
                    &info,
                    &p_config,
                    prev_mv_buf.as_ref(),
                    true,
                    true, // anchor P always needs decode (bwd ref for B-frames)
                );
                prev_mv_buf = Some(new_mv_buf);
                if let Some(ref mut rc) = rate_ctrl {
                    rc.update(p_config.quantization_step, compressed.bpp());
                }
                if diag_enabled {
                    let d = diagnostics::collect(p_display, &compressed, p_config.quantization_step, last_iframe_bytes);
                    diagnostics::print(&d);

                    if let Some(ref stg) = diag_twav_staging {
                        let coeffs = self.diag_original_wavelet_coefficients(
                            ctx, &p_frame_data, padded_w, padded_h, padded_pixels,
                            &info, config, stg,
                        );
                        if let Some(ref prev) = prev_twav_coeffs {
                            let y_stats = diagnostics::compute_temporal_wavelet(
                                &coeffs[0], &prev[0], padded_w, padded_h,
                                config.tile_size, config.wavelet_levels, config.quantization_step,
                            );
                            let co_stats = diagnostics::compute_temporal_wavelet(
                                &coeffs[1], &prev[1], padded_w, padded_h,
                                config.tile_size, config.wavelet_levels, config.quantization_step,
                            );
                            let cg_stats = diagnostics::compute_temporal_wavelet(
                                &coeffs[2], &prev[2], padded_w, padded_h,
                                config.tile_size, config.wavelet_levels, config.quantization_step,
                            );
                            diagnostics::print_temporal_wavelet(
                                p_display, FrameType::Predicted, &y_stats, &co_stats, &cg_stats,
                            );
                        }
                        prev_twav_coeffs = Some(coeffs);
                    }
                }
                results[p_display] = Some(compressed);

                // 3. Swap: gpu_ref_planes = past anchor, gpu_bwd_ref_planes = future P
                self.swap_ref_planes();

                // 4. Encode B-frames between past and future anchors.
                // First B-frame uses None predictor (full coarse search).
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
                    let fwd_pred = prev_bidir_fwd_mv.as_ref();
                    let bwd_pred = prev_bidir_bwd_mv.as_ref();
                    let b_frame_data = load_frame(b_display);
                    let (compressed, new_fwd_mv, new_bwd_mv) = self.encode_bframe(
                        ctx,
                        &b_frame_data,
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
                    if diag_enabled {
                        let d = diagnostics::collect(b_display, &compressed, b_config.quantization_step, last_iframe_bytes);
                        diagnostics::print(&d);

                        if let Some(ref stg) = diag_twav_staging {
                            let coeffs = self.diag_original_wavelet_coefficients(
                                ctx, &b_frame_data, padded_w, padded_h, padded_pixels,
                                &info, config, stg,
                            );
                            if let Some(ref prev) = prev_twav_coeffs {
                                let y_stats = diagnostics::compute_temporal_wavelet(
                                    &coeffs[0], &prev[0], padded_w, padded_h,
                                    config.tile_size, config.wavelet_levels, config.quantization_step,
                                );
                                let co_stats = diagnostics::compute_temporal_wavelet(
                                    &coeffs[1], &prev[1], padded_w, padded_h,
                                    config.tile_size, config.wavelet_levels, config.quantization_step,
                                );
                                let cg_stats = diagnostics::compute_temporal_wavelet(
                                    &coeffs[2], &prev[2], padded_w, padded_h,
                                    config.tile_size, config.wavelet_levels, config.quantization_step,
                                );
                                diagnostics::print_temporal_wavelet(
                                    b_display, FrameType::Bidirectional, &y_stats, &co_stats, &cg_stats,
                                );
                            }
                            prev_twav_coeffs = Some(coeffs);
                        }
                    }
                    results[b_display] = Some(compressed);
                }

                // 5. Swap back: gpu_ref_planes = decoded P (for next group's forward ref)
                self.swap_ref_planes();
            }

            // Remainder frames (< group_size) encoded as P-frames
            let rem_start = display_idx + full_groups * group_size;
            #[allow(clippy::needless_range_loop)] // j used as frame index for load_frame, results, and diagnostics
            for j in rem_start..next_key {
                let p_config = if let Some(ref rc) = rate_ctrl {
                    let mut cfg = config.clone();
                    cfg.quantization_step = rc.estimate_qstep();
                    cfg
                } else {
                    config.clone()
                };
                // Skip decode if next frame is keyframe or end of sequence
                let rem_needs_decode = j + 1 < next_key && j + 1 < n;
                let rem_frame_data = load_frame(j);
                let (compressed, new_mv_buf) = self.encode_pframe(
                    ctx,
                    &rem_frame_data,
                    width,
                    height,
                    padded_w,
                    padded_h,
                    padded_pixels,
                    &info,
                    &p_config,
                    prev_mv_buf.as_ref(),
                    false,
                    rem_needs_decode,
                );
                prev_mv_buf = Some(new_mv_buf);
                if let Some(ref mut rc) = rate_ctrl {
                    rc.update(p_config.quantization_step, compressed.bpp());
                }
                if diag_enabled {
                    let d = diagnostics::collect(j, &compressed, p_config.quantization_step, last_iframe_bytes);
                    diagnostics::print(&d);

                    if let Some(ref stg) = diag_twav_staging {
                        let coeffs = self.diag_original_wavelet_coefficients(
                            ctx, &rem_frame_data, padded_w, padded_h, padded_pixels,
                            &info, config, stg,
                        );
                        if let Some(ref prev) = prev_twav_coeffs {
                            let y_stats = diagnostics::compute_temporal_wavelet(
                                &coeffs[0], &prev[0], padded_w, padded_h,
                                config.tile_size, config.wavelet_levels, config.quantization_step,
                            );
                            let co_stats = diagnostics::compute_temporal_wavelet(
                                &coeffs[1], &prev[1], padded_w, padded_h,
                                config.tile_size, config.wavelet_levels, config.quantization_step,
                            );
                            let cg_stats = diagnostics::compute_temporal_wavelet(
                                &coeffs[2], &prev[2], padded_w, padded_h,
                                config.tile_size, config.wavelet_levels, config.quantization_step,
                            );
                            diagnostics::print_temporal_wavelet(
                                j, FrameType::Predicted, &y_stats, &co_stats, &cg_stats,
                            );
                        }
                        prev_twav_coeffs = Some(coeffs);
                    }
                }
                results[j] = Some(compressed);
            }

            display_idx = next_key;
        }

        results.into_iter().map(|o| o.unwrap()).collect()
    }

    /// Encode a sequence using a temporal wavelet transform (in-memory only).
    /// This bypasses ME/MC entirely: temporal transform → spatial encode for each coeff frame.
    pub fn encode_sequence_temporal_wavelet(
        &mut self,
        ctx: &GpuContext,
        frames: &[&[f32]],
        width: u32,
        height: u32,
        config: &CodecConfig,
        mode: TemporalTransform,
        gop_size: usize,
    ) -> TemporalEncodedSequence {
        assert!(
            mode != TemporalTransform::None,
            "temporal wavelet encode requires a non-None mode"
        );
        let group_size = match mode {
            TemporalTransform::Haar => gop_size,
            _ => temporal::group_size(mode),
        };
        if mode == TemporalTransform::Haar {
            assert!(
                temporal::is_power_of_two(group_size),
                "temporal Haar requires GOP size to be a power of two"
            );
        }
        let mut groups: Vec<TemporalGroup> = Vec::new();
        let mut tail_iframes: Vec<CompressedFrame> = Vec::new();

        let mut cfg = config.clone();
        cfg.keyframe_interval = 1;
        cfg.temporal_transform = TemporalTransform::None;
        cfg.cfl_enabled = false;

        let mut i = 0usize;
        let info = FrameInfo {
            width,
            height,
            bit_depth: 8,
            tile_size: cfg.tile_size,
        };
        let padded_w = info.padded_width();
        let padded_h = info.padded_height();
        let padded_pixels = (padded_w * padded_h) as usize;
        self.ensure_cached(ctx, padded_w, padded_h, width, height);
        let staging_prequant: [wgpu::Buffer; 3] = std::array::from_fn(|idx| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(["tw_prequant_y", "tw_prequant_co", "tw_prequant_cg"][idx]),
                size: (padded_pixels * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        while i + group_size <= frames.len() {
            match mode {
                TemporalTransform::Haar => {
                    // 1) Spatial wavelet per frame (pre-quantized coeffs)
                    let mut per_plane_coeffs: [Vec<Vec<f32>>; 3] =
                        [Vec::new(), Vec::new(), Vec::new()];
                    for j in 0..group_size {
                        let coeffs = self.diag_original_wavelet_prequant(
                            ctx,
                            frames[i + j],
                            padded_w,
                            padded_h,
                            padded_pixels,
                            &info,
                            &cfg,
                            &staging_prequant,
                        );
                        for p in 0..3 {
                            per_plane_coeffs[p].push(coeffs[p].clone());
                        }
                    }

                    // 2) Temporal Haar in wavelet domain, per plane
                    let (low_y, highs_y) = temporal::haar_multilevel_forward(
                        &per_plane_coeffs[0].iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                    );
                    let (low_co, highs_co) = temporal::haar_multilevel_forward(
                        &per_plane_coeffs[1].iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                    );
                    let (low_cg, highs_cg) = temporal::haar_multilevel_forward(
                        &per_plane_coeffs[2].iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
                    );

                    if diagnostics::enabled() {
                        let group_idx = groups.len();
                        let base = i;
                        let end = base + group_size - 1;
                        eprintln!(
                            "TW ENC group {} frames [{}..{}]",
                            group_idx, base, end
                        );
                        eprintln!("  low -> [{}..{}]", base, end);
                        for lvl in 0..highs_y.len() {
                            let span = 1usize << (lvl + 1);
                            for idx in 0..highs_y[lvl].len() {
                                let s = base + idx * span;
                                let e = s + span - 1;
                                eprintln!("  high L{}[{}] -> [{}..{}]", lvl, idx, s, e);
                            }
                        }
                    }

                    // 3) Encode low frame
                    let low_cf = self.encode_from_wavelet_coeffs(
                        ctx,
                        [&low_y, &low_co, &low_cg],
                        &cfg,
                        &info,
                        padded_w,
                        padded_h,
                        padded_pixels,
                    );

                    // 4) Encode high frames per level (coarser qstep)
                    let mut high_cfg = cfg.clone();
                    high_cfg.quantization_step *= config.temporal_highpass_qstep_mul;
                    let mut high_cfs: Vec<Vec<CompressedFrame>> = Vec::new();
                    for lvl in 0..highs_y.len() {
                        let mut lvl_frames: Vec<CompressedFrame> = Vec::new();
                        for idx in 0..highs_y[lvl].len() {
                            let cf = self.encode_from_wavelet_coeffs(
                                ctx,
                                [
                                    &highs_y[lvl][idx],
                                    &highs_co[lvl][idx],
                                    &highs_cg[lvl][idx],
                                ],
                                &high_cfg,
                                &info,
                                padded_w,
                                padded_h,
                                padded_pixels,
                            );
                            lvl_frames.push(cf);
                        }
                        high_cfs.push(lvl_frames);
                    }
                    groups.push(TemporalGroup {
                        low_frame: low_cf,
                        high_frames: high_cfs,
                    });
                }
                TemporalTransform::LeGall53 => {
                    panic!("temporal LeGall53 not supported in multilevel mode yet");
                }
                TemporalTransform::None => unreachable!(),
            }
            i += group_size;
        }

        // Tail: encode remaining frames as I-frames (no temporal transform)
        while i < frames.len() {
            let cf = self.encode(ctx, frames[i], width, height, &cfg);
            tail_iframes.push(cf);
            i += 1;
        }

        TemporalEncodedSequence {
            mode,
            groups,
            tail_iframes,
            frame_count: frames.len(),
            gop_size: group_size,
        }
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

        for (p, &quant_buf) in quant_bufs.iter().enumerate() {
            let weights = if p == 0 {
                &weights_luma
            } else {
                &weights_chroma
            };

            // Dequantize: quant_buf → cg_plane (scratch — preserves plane_b for Cg)
            self.quantize.dispatch_adaptive(
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
                wm_param,
                0.0,
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
    /// Encode a P-frame.
    ///
    /// `save_bwd_ref`: when true, copies gpu_ref_planes → gpu_bwd_ref_planes at
    /// the start of the command encoder (before ME overwrites anything).
    ///
    /// `needs_decode`: when false, skips the local decode loop (dequant + inverse
    /// wavelet + inverse MC). Use false when this P-frame's decoded output won't
    /// be used as a reference (e.g. last P before a keyframe or end of sequence).
    /// Saves ~42 GPU dispatches per skipped frame.
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
        needs_decode: bool,
    ) -> (CompressedFrame, wgpu::Buffer) {
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;

        self.ensure_cached(ctx, padded_w, padded_h, width, height);
        let bufs = self.cached.as_ref().unwrap();

        // Upload raw (unpadded) frame — GPU shader handles padding
        ctx.queue
            .write_buffer(&bufs.raw_input_buf, 0, bytemuck::cast_slice(rgb_data));

        // --- Residual-adapted quantization for P-frames ---
        // MC residuals are noise-like: energy spread uniformly across wavelet subbands.
        // Perceptual weights (designed for natural images) are counterproductive here.
        // Use uniform weights so all subbands get equal treatment, and increase dead_zone
        // to aggressively zero small coefficients that don't contribute to quality.
        let uniform_weights = crate::SubbandWeights::uniform(config.wavelet_levels);
        let weights_luma = uniform_weights.pack_weights();
        let weights_chroma = uniform_weights.pack_weights_chroma();
        let res_dead_zone = config.dead_zone * 2.0;

        // Config stored in CompressedFrame must match encoder parameters so decoder
        // uses the same dequantization.
        let mut res_config = config.clone();
        res_config.subband_weights = uniform_weights;
        res_config.dead_zone = res_dead_zone;

        let entropy_mode = EntropyMode::from_config(config);
        let tile_size = config.tile_size as usize;
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;
        let use_gpu_encode =
            config.gpu_entropy_encode && config.entropy_coder != EntropyCoder::Bitplane;

        let mut rans_tiles: Vec<rans::InterleavedRansTile> = Vec::new();
        let mut subband_tiles: Vec<rans::SubbandRansTile> = Vec::new();
        let mut bp_tiles: Vec<bitplane::BitplaneTile> = Vec::new();
        let mut rice_tiles: Vec<rice::RiceTile> = Vec::new();
        let mut huffman_tiles: Vec<huffman::HuffmanTile> = Vec::new();

        // Diagnostics: staging buffers for per-channel residual readback
        let diag_enabled = diagnostics::enabled();
        let diag_residual_staging = if diag_enabled {
            Some([
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("diag_residual_y_staging"),
                    size: plane_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("diag_residual_co_staging"),
                    size: plane_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("diag_residual_cg_staging"),
                    size: plane_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            ])
        } else {
            None
        };

        // === Batched GPU pipeline: preprocess + ME + forward encode ===
        // MV buffer stays on GPU — used directly by MC without readback/re-upload.
        let mv_buf;
        let split_mv_buf;

        if use_gpu_encode {
            // === Fully batched GPU pipeline: forward + entropy + local decode ===
            // Single command encoder, single submit, single poll.
            // Eliminates GPU pipeline stalls between forward/entropy/decode phases.
            let _t_pf = std::time::Instant::now();
            let profile = std::env::var("GNC_PROFILE").is_ok();
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

            // Profiling: flush preprocess to isolate ME timing
            if profile {
                ctx.queue.submit(Some(cmd.finish()));
                ctx.device.poll(wgpu::Maintain::Wait);
                eprintln!("    P preprocess: {:.1}ms", _t_pf.elapsed().as_secs_f64() * 1000.0);
                cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_me"),
                });
            }

            let _t_me = std::time::Instant::now();
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

            // Variable block size: 8x8 split decision
            // Lambda must be high enough to prevent unnecessary splits on easy content.
            // Each split adds 3 extra MVs (12 bytes raw); must outweigh residual savings.
            let lambda_sad = (config.quantization_step * 16.0 + 128.0).round() as u32;
            split_mv_buf = self.motion.estimate_split(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.gpu_ref_planes[0],
                &mv_buf,
                &bufs.me_sad_buf,
                None, // 8x8 temporal predictor (future enhancement)
                padded_w,
                padded_h,
                lambda_sad,
            );

            // Profiling: flush ME to isolate MC+wavelet+quantize timing
            if profile {
                ctx.queue.submit(Some(cmd.finish()));
                ctx.device.poll(wgpu::Maintain::Wait);
                eprintln!("    P ME+split: {:.1}ms", _t_me.elapsed().as_secs_f64() * 1000.0);
                cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_mcwq"),
                });
            }

            let _t_mcwq = std::time::Instant::now();
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
                // Quantize output: Y→recon_y, Co→co_plane, Cg→plane_b (no copy needed)
                let quant_out = match p {
                    0 => &bufs.recon_y,
                    1 => &bufs.co_plane,
                    _ => &bufs.plane_b,
                };

                self.motion.compensate_cached(
                    ctx,
                    &mut cmd,
                    cur_plane,
                    &bufs.gpu_ref_planes[p],
                    &split_mv_buf,
                    &bufs.mc_out,
                    padded_w,
                    padded_h,
                    &bufs.mc_fwd_params_8,
                );

                // Diagnostics: copy per-channel residual before wavelet overwrites mc_out
                if let Some(ref stg) = diag_residual_staging {
                    cmd.copy_buffer_to_buffer(&bufs.mc_out, 0, &stg[p], 0, plane_size);
                }

                // mc_out feeds directly into wavelet (read-only at level 0)
                self.transform.forward(
                    ctx,
                    &mut cmd,
                    &bufs.mc_out,
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
                    quant_out,
                    padded_pixels as u32,
                    config.quantization_step,
                    res_dead_zone,
                    true,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                    weights,
                );
            }

            // Profiling: flush forward phase to measure GPU time
            if profile {
                ctx.queue.submit(Some(cmd.finish()));
                ctx.device.poll(wgpu::Maintain::Wait);
                eprintln!("    P MC+wavelet+quant: {:.1}ms", _t_mcwq.elapsed().as_secs_f64() * 1000.0);
                eprintln!("    P fwd total: {:.1}ms", _t_pf.elapsed().as_secs_f64() * 1000.0);
                cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_entropy"),
                });
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

            // Profiling: flush entropy phase
            let _t_after_entropy;
            if profile {
                ctx.queue.submit(Some(cmd.finish()));
                ctx.device.poll(wgpu::Maintain::Wait);
                _t_after_entropy = _t_pf.elapsed();
                eprintln!("    P entropy+stg: {:.1}ms", _t_after_entropy.as_secs_f64() * 1000.0);
                cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_local_decode"),
                });
            } else {
                _t_after_entropy = std::time::Duration::ZERO;
            }

            // Phase 3: Local decode dispatches (same cmd encoder)
            // Skipped when this P-frame won't be used as a reference
            // (saves ~42 GPU dispatches = ~6-8ms per frame).
            if needs_decode {
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
                        res_dead_zone,
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
                        &split_mv_buf,
                        &bufs.recon_out,
                        padded_w,
                        padded_h,
                        &bufs.mc_inv_params_8,
                    );
                    cmd.copy_buffer_to_buffer(
                        &bufs.recon_out,
                        0,
                        &bufs.gpu_ref_planes[p],
                        0,
                        plane_size,
                    );
                }
            }

            // MV staging copy: 8x8 split MVs
            cmd.copy_buffer_to_buffer(&split_mv_buf, 0, &bufs.split_mv_staging_buf, 0, bufs.split_mv_staging_size);

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

            // Read back 8x8-resolution MVs from split staging
            let mvs = MotionEstimator::finish_mv_readback_cached(ctx, &bufs.split_mv_staging_buf, bufs.split_mv_staging_size, bufs.split_total_blocks);

            // Diagnostics: read back per-channel residual and compute stats
            let (residual_stats, residual_stats_co, residual_stats_cg) =
                if let Some(ref stg) = diag_residual_staging {
                    (
                        Some(diagnostics::compute_residual_stats(ctx, &stg[0], plane_size, padded_pixels)),
                        Some(diagnostics::compute_residual_stats(ctx, &stg[1], plane_size, padded_pixels)),
                        Some(diagnostics::compute_residual_stats(ctx, &stg[2], plane_size, padded_pixels)),
                    )
                } else {
                    (None, None, None)
                };

            let entropy = match entropy_mode {
                EntropyMode::Bitplane => EntropyData::Bitplane(bp_tiles),
                EntropyMode::SubbandRans | EntropyMode::SubbandRansCtx => {
                    EntropyData::SubbandRans(subband_tiles)
                }
                EntropyMode::Rans => EntropyData::Rans(rans_tiles),
                EntropyMode::Rice => EntropyData::Rice(rice_tiles),
                EntropyMode::Huffman => EntropyData::Huffman(huffman_tiles),
            };

            return (CompressedFrame {
                info: *info,
                config: res_config.clone(),
                entropy,
                cfl_alphas: None,
                weight_map: None,
                frame_type: FrameType::Predicted,
                motion_field: Some(MotionField {
                    vectors: mvs,
                    block_size: ME_SPLIT_BLOCK_SIZE,
                    backward_vectors: None,
                    block_modes: None,
                }),
                intra_modes: None,
                residual_stats,
                residual_stats_co,
                residual_stats_cg,
            }, mv_buf); // Return 16x16 mv_buf as temporal predictor for next frame
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
            let (mb, sad_buf) = self.motion.estimate(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.gpu_ref_planes[0],
                padded_w,
                padded_h,
                predictor_mvs,
            );
            mv_buf = mb;

            // Variable block size: 8x8 split decision
            let lambda_sad = (config.quantization_step * 16.0 + 128.0).round() as u32;
            split_mv_buf = self.motion.estimate_split(
                ctx,
                &mut cmd,
                &bufs.plane_a,
                &bufs.gpu_ref_planes[0],
                &mv_buf,
                &sad_buf,
                None,
                padded_w,
                padded_h,
                lambda_sad,
            );
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
                // Quantize output: Y→recon_y, Co→co_plane, Cg→plane_b
                let quant_out = match p {
                    0 => &bufs.recon_y,
                    1 => &bufs.co_plane,
                    _ => &bufs.plane_b,
                };

                self.motion.compensate_cached(
                    ctx,
                    &mut cmd,
                    cur_plane,
                    &bufs.gpu_ref_planes[p],
                    &split_mv_buf,
                    &bufs.mc_out,
                    padded_w,
                    padded_h,
                    &bufs.mc_fwd_params_8,
                );
                // mc_out feeds directly into wavelet (read-only at level 0)
                self.transform.forward(
                    ctx,
                    &mut cmd,
                    &bufs.mc_out,
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
                    quant_out,
                    padded_pixels as u32,
                    config.quantization_step,
                    res_dead_zone,
                    true,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                    weights,
                );
                ctx.queue.submit(Some(cmd.finish()));

                encode_entropy(
                    &mut self.gpu_encoder,
                    ctx,
                    quant_out,
                    padded_pixels,
                    padded_w as usize,
                    tiles_x,
                    tiles_y,
                    tile_size,
                    &entropy_mode,
                    config,
                    use_gpu_encode,
                    info,
                    config.wavelet_levels,
                    &mut rans_tiles,
                    &mut subband_tiles,
                    &mut bp_tiles,
                    &mut rice_tiles,
                    &mut huffman_tiles,
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
            EntropyMode::Huffman => EntropyData::Huffman(huffman_tiles),
        };

        // === Batched local decode + MV copy: single command encoder ===
        let split_blocks_8 = (padded_w / ME_SPLIT_BLOCK_SIZE) * (padded_h / ME_SPLIT_BLOCK_SIZE);
        let mv_staging =
            MotionEstimator::create_mv_staging(ctx, &split_mv_buf, split_blocks_8);
        {
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pf_local_decode"),
                });

            if needs_decode {
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
                        res_dead_zone,
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
                        &split_mv_buf,
                        &bufs.recon_out,
                        padded_w,
                        padded_h,
                        &bufs.mc_inv_params_8,
                    );
                    cmd.copy_buffer_to_buffer(
                        &bufs.recon_out,
                        0,
                        &bufs.gpu_ref_planes[p],
                        0,
                        plane_size,
                    );
                }
            }

            // MV copy in same batch — 8x8 split MVs
            cmd.copy_buffer_to_buffer(&split_mv_buf, 0, &mv_staging.buffer, 0, mv_staging.size);

            ctx.queue.submit(Some(cmd.finish()));
        }

        // Single poll drains local decode + MV copy together
        let mvs = MotionEstimator::finish_mv_readback(ctx, &mv_staging);

        (CompressedFrame {
            info: *info,
            config: res_config,
            entropy,
            cfl_alphas: None,
            weight_map: None,
            frame_type: FrameType::Predicted,
            motion_field: Some(MotionField {
                vectors: mvs,
                block_size: ME_SPLIT_BLOCK_SIZE,
                backward_vectors: None,
                block_modes: None,
            }),
            intra_modes: None,
            residual_stats: None,
            residual_stats_co: None,
            residual_stats_cg: None,
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

        // --- Residual-adapted quantization for B-frames ---
        // Same rationale as P-frames: MC residuals need uniform weights + higher dead_zone.
        let uniform_weights = crate::SubbandWeights::uniform(config.wavelet_levels);
        let weights_luma = uniform_weights.pack_weights();
        let weights_chroma = uniform_weights.pack_weights_chroma();
        let res_dead_zone = config.dead_zone * 2.0;

        let mut res_config = config.clone();
        res_config.subband_weights = uniform_weights;
        res_config.dead_zone = res_dead_zone;

        let entropy_mode = EntropyMode::from_config(config);
        let tile_size = config.tile_size as usize;
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;
        let use_gpu_encode =
            config.gpu_entropy_encode && config.entropy_coder != EntropyCoder::Bitplane;

        let me_total_blocks = (padded_w / ME_BLOCK_SIZE) * (padded_h / ME_BLOCK_SIZE);

        let mut rans_tiles: Vec<rans::InterleavedRansTile> = Vec::new();
        let mut subband_tiles: Vec<rans::SubbandRansTile> = Vec::new();
        let mut bp_tiles: Vec<bitplane::BitplaneTile> = Vec::new();
        let mut rice_tiles: Vec<rice::RiceTile> = Vec::new();
        let mut huffman_tiles: Vec<huffman::HuffmanTile> = Vec::new();

        // Diagnostics: staging buffers for per-channel residual readback
        let diag_enabled = diagnostics::enabled();
        let diag_residual_staging = if diag_enabled {
            Some([
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("diag_bframe_residual_y_staging"),
                    size: plane_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("diag_bframe_residual_co_staging"),
                    size: plane_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("diag_bframe_residual_cg_staging"),
                    size: plane_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            ])
        } else {
            None
        };

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
                // Quantize output: Y→recon_y, Co→co_plane, Cg→plane_b (no copy needed)
                let quant_out = match p {
                    0 => &bufs.recon_y,
                    1 => &bufs.co_plane,
                    _ => &bufs.plane_b,
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

                // Diagnostics: copy per-channel residual before wavelet overwrites mc_out
                if let Some(ref stg) = diag_residual_staging {
                    cmd.copy_buffer_to_buffer(&bufs.mc_out, 0, &stg[p], 0, plane_size);
                }

                // mc_out feeds directly into wavelet (read-only at level 0)
                self.transform.forward(
                    ctx,
                    &mut cmd,
                    &bufs.mc_out,
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
                    quant_out,
                    padded_pixels as u32,
                    config.quantization_step,
                    res_dead_zone,
                    true,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                    weights,
                );
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

            // Diagnostics: read back per-channel residual and compute stats
            let (residual_stats, residual_stats_co, residual_stats_cg) =
                if let Some(ref stg) = diag_residual_staging {
                    (
                        Some(diagnostics::compute_residual_stats(ctx, &stg[0], plane_size, padded_pixels)),
                        Some(diagnostics::compute_residual_stats(ctx, &stg[1], plane_size, padded_pixels)),
                        Some(diagnostics::compute_residual_stats(ctx, &stg[2], plane_size, padded_pixels)),
                    )
                } else {
                    (None, None, None)
                };

            let entropy = match entropy_mode {
                EntropyMode::Bitplane => EntropyData::Bitplane(bp_tiles),
                EntropyMode::SubbandRans | EntropyMode::SubbandRansCtx => {
                    EntropyData::SubbandRans(subband_tiles)
                }
                EntropyMode::Rans => EntropyData::Rans(rans_tiles),
                EntropyMode::Rice => EntropyData::Rice(rice_tiles),
                EntropyMode::Huffman => EntropyData::Huffman(huffman_tiles),
            };

            return (CompressedFrame {
                info: *info,
                config: res_config.clone(),
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
                intra_modes: None,
                residual_stats,
                residual_stats_co,
                residual_stats_cg,
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
                // Quantize output: Y→recon_y, Co→co_plane, Cg→plane_b
                let quant_out = match p {
                    0 => &bufs.recon_y,
                    1 => &bufs.co_plane,
                    _ => &bufs.plane_b,
                };

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
                    ME_BLOCK_SIZE, // B-frames still use 16x16 blocks
                );
                // mc_out feeds directly into wavelet (read-only at level 0)
                self.transform.forward(
                    ctx,
                    &mut cmd,
                    &bufs.mc_out,
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
                    quant_out,
                    padded_pixels as u32,
                    config.quantization_step,
                    res_dead_zone,
                    true,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                    weights,
                );
                ctx.queue.submit(Some(cmd.finish()));

                encode_entropy(
                    &mut self.gpu_encoder,
                    ctx,
                    quant_out,
                    padded_pixels,
                    padded_w as usize,
                    tiles_x,
                    tiles_y,
                    tile_size,
                    &entropy_mode,
                    config,
                    use_gpu_encode,
                    info,
                    config.wavelet_levels,
                    &mut rans_tiles,
                    &mut subband_tiles,
                    &mut bp_tiles,
                    &mut rice_tiles,
                    &mut huffman_tiles,
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
            EntropyMode::Huffman => EntropyData::Huffman(huffman_tiles),
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
            config: res_config,
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
            intra_modes: None,
            residual_stats: None,
            residual_stats_co: None,
            residual_stats_cg: None,
        }, fwd_mv_buf, bwd_mv_buf)
    }

    /// Swap gpu_ref_planes ↔ gpu_bwd_ref_planes at the Rust level (zero GPU cost).
    /// Used during B-frame encoding to toggle between past/future references.
    fn swap_ref_planes(&mut self) {
        let bufs = self.cached.as_mut().unwrap();
        std::mem::swap(&mut bufs.gpu_ref_planes, &mut bufs.gpu_bwd_ref_planes);
    }

    /// Diagnostic: compute quantized wavelet coefficients of the original signal.
    ///
    /// Runs a separate GPU pass: pad → color → deinterleave → wavelet → quantize
    /// → staging copy for all 3 planes. Returns [Y, Co, Cg] quantized coefficients.
    ///
    /// Uses I-frame quantization parameters (perceptual weights, normal dead_zone)
    /// for consistent frame-to-frame comparison regardless of frame type.
    #[allow(clippy::too_many_arguments)]
    fn diag_original_wavelet_coefficients(
        &mut self,
        ctx: &GpuContext,
        rgb_data: &[f32],
        padded_w: u32,
        padded_h: u32,
        padded_pixels: usize,
        info: &FrameInfo,
        config: &CodecConfig,
        staging: &[wgpu::Buffer; 3],
    ) -> [Vec<f32>; 3] {
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let bufs = self.cached.as_ref().unwrap();

        ctx.queue
            .write_buffer(&bufs.raw_input_buf, 0, bytemuck::cast_slice(rgb_data));

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("diag_temporal_wavelet"),
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

        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();

        let planes: [&wgpu::Buffer; 3] = [&bufs.plane_a, &bufs.co_plane, &bufs.cg_plane];
        for (p, &cur_plane) in planes.iter().enumerate() {
            let weights = if p == 0 {
                &weights_luma
            } else {
                &weights_chroma
            };

            // wavelet forward: cur_plane(read-only) → plane_b(scratch) → plane_c(output)
            self.transform.forward(
                ctx,
                &mut cmd,
                cur_plane,
                &bufs.plane_b,
                &bufs.plane_c,
                info,
                config.wavelet_levels,
                config.wavelet_type,
            );

            // quantize: plane_c → recon_y
            self.quantize.dispatch(
                ctx,
                &mut cmd,
                &bufs.plane_c,
                &bufs.recon_y,
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

            // copy to staging
            cmd.copy_buffer_to_buffer(&bufs.recon_y, 0, &staging[p], 0, plane_size);
        }

        ctx.queue.submit(Some(cmd.finish()));

        [
            diagnostics::read_plane_f32(ctx, &staging[0], plane_size, padded_pixels),
            diagnostics::read_plane_f32(ctx, &staging[1], plane_size, padded_pixels),
            diagnostics::read_plane_f32(ctx, &staging[2], plane_size, padded_pixels),
        ]
    }

    /// Diagnostic: compute pre-quantized wavelet coefficients of the original signal.
    ///
    /// Runs a separate GPU pass: pad → color → deinterleave → wavelet → staging copy.
    /// Returns [Y, Co, Cg] wavelet coefficients (pre-quantize).
    #[allow(clippy::too_many_arguments)]
    fn diag_original_wavelet_prequant(
        &mut self,
        ctx: &GpuContext,
        rgb_data: &[f32],
        padded_w: u32,
        padded_h: u32,
        padded_pixels: usize,
        info: &FrameInfo,
        config: &CodecConfig,
        staging: &[wgpu::Buffer; 3],
    ) -> [Vec<f32>; 3] {
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let bufs = self.cached.as_ref().unwrap();

        ctx.queue
            .write_buffer(&bufs.raw_input_buf, 0, bytemuck::cast_slice(rgb_data));

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("diag_temporal_wavelet_prequant"),
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

        let planes: [&wgpu::Buffer; 3] = [&bufs.plane_a, &bufs.co_plane, &bufs.cg_plane];
        for (p, &cur_plane) in planes.iter().enumerate() {
            // wavelet forward: cur_plane(read-only) → plane_b(scratch) → plane_c(output)
            self.transform.forward(
                ctx,
                &mut cmd,
                cur_plane,
                &bufs.plane_b,
                &bufs.plane_c,
                info,
                config.wavelet_levels,
                config.wavelet_type,
            );

            // copy wavelet coeffs to staging
            cmd.copy_buffer_to_buffer(&bufs.plane_c, 0, &staging[p], 0, plane_size);
        }

        ctx.queue.submit(Some(cmd.finish()));

        [
            diagnostics::read_plane_f32(ctx, &staging[0], plane_size, padded_pixels),
            diagnostics::read_plane_f32(ctx, &staging[1], plane_size, padded_pixels),
            diagnostics::read_plane_f32(ctx, &staging[2], plane_size, padded_pixels),
        ]
    }

    /// Expose pre-quantized wavelet coeffs for diagnostics from main.
    pub fn debug_wavelet_prequant(
        &mut self,
        ctx: &GpuContext,
        rgb_data: &[f32],
        info: &FrameInfo,
        config: &CodecConfig,
    ) -> [Vec<f32>; 3] {
        let padded_w = info.padded_width();
        let padded_h = info.padded_height();
        let padded_pixels = (padded_w * padded_h) as usize;
        self.ensure_cached(ctx, padded_w, padded_h, info.width, info.height);
        let staging: [wgpu::Buffer; 3] = std::array::from_fn(|idx| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(["tw_dbg_prequant_y", "tw_dbg_prequant_co", "tw_dbg_prequant_cg"][idx]),
                size: (padded_pixels * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        self.diag_original_wavelet_prequant(
            ctx,
            rgb_data,
            padded_w,
            padded_h,
            padded_pixels,
            info,
            config,
            &staging,
        )
    }

    /// Diagnostic: quantize precomputed wavelet coefficients and return quantized planes.
    pub fn debug_quantize_wavelet_coeffs(
        &mut self,
        ctx: &GpuContext,
        coeffs: [&[f32]; 3],
        info: &FrameInfo,
        config: &CodecConfig,
    ) -> [Vec<f32>; 3] {
        let padded_w = info.padded_width();
        let padded_h = info.padded_height();
        let padded_pixels = (padded_w * padded_h) as usize;
        self.ensure_cached(ctx, padded_w, padded_h, info.width, info.height);

        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let bufs = self.cached.as_ref().unwrap();
        let staging: [wgpu::Buffer; 3] = std::array::from_fn(|idx| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(["tw_dbg_quant_y", "tw_dbg_quant_co", "tw_dbg_quant_cg"][idx]),
                size: plane_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();

        // Submit per-plane: write_buffer data is only flushed at submit-time,
        // so reusing the same buffer across planes requires separate submits.
        for p in 0..3 {
            let weights = if p == 0 {
                &weights_luma
            } else {
                &weights_chroma
            };
            ctx.queue
                .write_buffer(&bufs.plane_c, 0, bytemuck::cast_slice(coeffs[p]));
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tw_dbg_quantize"),
                });
            self.quantize.dispatch(
                ctx,
                &mut cmd,
                &bufs.plane_c,
                &bufs.recon_y,
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
            cmd.copy_buffer_to_buffer(&bufs.recon_y, 0, &staging[p], 0, plane_size);
            ctx.queue.submit(Some(cmd.finish()));
        }

        std::array::from_fn(|p| {
            diagnostics::read_plane_f32(ctx, &staging[p], plane_size, padded_pixels)
        })
    }

    /// Encode a frame from pre-quantized wavelet coefficients (CPU entropy path).
    #[allow(clippy::too_many_arguments)]
    fn encode_from_wavelet_coeffs(
        &mut self,
        ctx: &GpuContext,
        coeffs: [&[f32]; 3],
        config: &CodecConfig,
        info: &FrameInfo,
        padded_w: u32,
        padded_h: u32,
        padded_pixels: usize,
    ) -> CompressedFrame {
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;
        let tile_size = config.tile_size as usize;
        let entropy_mode = EntropyMode::from_config(config);
        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();

        let mut rans_tiles: Vec<rans::InterleavedRansTile> = Vec::new();
        let mut subband_tiles: Vec<rans::SubbandRansTile> = Vec::new();
        let mut bp_tiles: Vec<bitplane::BitplaneTile> = Vec::new();
        let mut rice_tiles: Vec<rice::RiceTile> = Vec::new();
        let mut huffman_tiles: Vec<huffman::HuffmanTile> = Vec::new();

        let bufs = self.cached.as_ref().unwrap();
        let staging: [wgpu::Buffer; 3] = std::array::from_fn(|idx| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(["tw_quant_y", "tw_quant_co", "tw_quant_cg"][idx]),
                size: plane_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // Submit per-plane to ensure each write_buffer is flushed before its
        // dispatch.  A single submit would only see the last write_buffer to
        // plane_c (wgpu flushes all pending writes at the start of submit).
        for p in 0..3 {
            let weights = if p == 0 {
                &weights_luma
            } else {
                &weights_chroma
            };
            // Upload coefficients to plane_c
            ctx.queue
                .write_buffer(&bufs.plane_c, 0, bytemuck::cast_slice(coeffs[p]));

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tw_quantize"),
                });

            // Quantize: plane_c → recon_y (quantized coeffs)
            self.quantize.dispatch(
                ctx,
                &mut cmd,
                &bufs.plane_c,
                &bufs.recon_y,
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

            cmd.copy_buffer_to_buffer(&bufs.recon_y, 0, &staging[p], 0, plane_size);
            ctx.queue.submit(Some(cmd.finish()));
        }

        // Read back quantized coeffs per plane and CPU-encode tiles
        for p in 0..3 {
            let quantized =
                diagnostics::read_plane_f32(ctx, &staging[p], plane_size, padded_pixels);
            entropy_helpers::entropy_encode_tiles(
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
                &mut rice_tiles,
                &mut huffman_tiles,
            );
        }

        let entropy = match entropy_mode {
            EntropyMode::Bitplane => EntropyData::Bitplane(bp_tiles),
            EntropyMode::SubbandRans | EntropyMode::SubbandRansCtx => {
                EntropyData::SubbandRans(subband_tiles)
            }
            EntropyMode::Rans => EntropyData::Rans(rans_tiles),
            EntropyMode::Rice => EntropyData::Rice(rice_tiles),
            EntropyMode::Huffman => EntropyData::Huffman(huffman_tiles),
        };

        CompressedFrame {
            info: *info,
            config: config.clone(),
            entropy,
            cfl_alphas: None,
            weight_map: None,
            frame_type: FrameType::Intra,
            motion_field: None,
            intra_modes: None,
            residual_stats: None,
            residual_stats_co: None,
            residual_stats_cg: None,
        }
    }
}
