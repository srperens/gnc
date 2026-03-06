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

/// Per-frame metadata used when batching temporal highpass frame encoding.
/// Collected during Pass A (adaptive-mul computation) and consumed in Pass C
/// (batched GPU quantize + Rice encode).
struct HighFrameInfo {
    tile_muls: Vec<f32>,
    buf_idx: usize,
    is_zero: bool,
    wm_data: Option<Vec<f32>>,
}

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
    #[allow(clippy::too_many_arguments)]
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
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        self.ensure_cached(ctx, padded_w, padded_h, width, height);
        while i + group_size <= frames.len() {
            match mode {
                TemporalTransform::Haar => {
                    // Allocate per-frame GPU buffers for wavelet coefficients: [frame][plane]
                    let tw_frame_bufs: Vec<[wgpu::Buffer; 3]> = (0..group_size)
                        .map(|j| {
                            std::array::from_fn(|p| {
                                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                                    label: Some(&format!("tw_frame_{}_{}", j, p)),
                                    size: plane_size,
                                    usage: wgpu::BufferUsages::STORAGE
                                        | wgpu::BufferUsages::COPY_DST
                                        | wgpu::BufferUsages::COPY_SRC,
                                    mapped_at_creation: false,
                                })
                            })
                        })
                        .collect();
                    // Snapshot buffers to avoid read-after-write aliasing in multilevel Haar.
                    // Each level snapshots its inputs before processing, so output writes
                    // don't corrupt inputs needed by later pairs within the same level.
                    let tw_snapshot: Vec<wgpu::Buffer> = (0..group_size)
                        .map(|s| {
                            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some(&format!("tw_snap_{}", s)),
                                size: plane_size,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_DST
                                    | wgpu::BufferUsages::COPY_SRC,
                                mapped_at_creation: false,
                            })
                        })
                        .collect();

                    // 1) Upload all GOP frames to per-frame GPU buffers (avoids write_buffer race)
                    let raw_input_size = std::mem::size_of_val(frames[i]) as u64;
                    let per_frame_input: Vec<wgpu::Buffer> = (0..group_size)
                        .map(|j| {
                            let buf = ctx
                                .device
                                .create_buffer(&wgpu::BufferDescriptor {
                                    label: Some(&format!("tw_raw_input_{}", j)),
                                    size: raw_input_size,
                                    usage: wgpu::BufferUsages::COPY_SRC
                                        | wgpu::BufferUsages::COPY_DST,
                                    mapped_at_creation: false,
                                });
                            ctx.queue
                                .write_buffer(&buf, 0, bytemuck::cast_slice(frames[i + j]));
                            buf
                        })
                        .collect();

                    // Spatial wavelet per frame — single command encoder to prevent
                    // intermediate buffer races (separate submits can overlap on GPU)
                    {
                        let bufs = self.cached.as_ref().unwrap();
                        let mut cmd = ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("tw_spatial_all"),
                            });
                        for j in 0..group_size {
                            cmd.copy_buffer_to_buffer(
                                &per_frame_input[j],
                                0,
                                &bufs.raw_input_buf,
                                0,
                                raw_input_size,
                            );
                            self.dispatch_gpu_pad_cached(ctx, &mut cmd, padded_w, padded_h);
                            let bufs = self.cached.as_ref().unwrap();
                            self.color.dispatch(
                                ctx,
                                &mut cmd,
                                &bufs.input_buf,
                                &bufs.color_out,
                                padded_w,
                                padded_h,
                                true,
                                cfg.is_lossless(),
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

                            let planes: [&wgpu::Buffer; 3] =
                                [&bufs.plane_a, &bufs.co_plane, &bufs.cg_plane];
                            for (p, &cur_plane) in planes.iter().enumerate() {
                                self.transform.forward(
                                    ctx,
                                    &mut cmd,
                                    cur_plane,
                                    &bufs.plane_b,
                                    &bufs.plane_c,
                                    &info,
                                    cfg.wavelet_levels,
                                    cfg.wavelet_type,
                                );
                                cmd.copy_buffer_to_buffer(
                                    &bufs.plane_c,
                                    0,
                                    &tw_frame_bufs[j][p],
                                    0,
                                    plane_size,
                                );
                            }
                        }
                        ctx.queue.submit(Some(cmd.finish()));
                    }

                    // 2) Temporal Haar on GPU, per plane — multilevel dyadic decomposition
                    // After completion, buffer layout is:
                    //   buf[0] = final lowpass
                    //   buf[1] = high L(num_levels-1) (coarsest)
                    //   buf[2..4] = high L(num_levels-2)
                    //   buf[gop/2..gop] = high L0 (finest)
                    let num_levels = (group_size as f64).log2() as usize;
                    #[allow(clippy::needless_range_loop)] // p indexes 2nd dim of tw_frame_bufs[frame][p]
                    for p in 0..3 {
                        let mut current_count = group_size;
                        let mut cmd = ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("tw_temporal_haar"),
                            });
                        for _level in 0..num_levels {
                            let pairs = current_count / 2;
                            // Snapshot inputs to avoid read-after-write aliasing
                            for j in 0..current_count {
                                cmd.copy_buffer_to_buffer(
                                    &tw_frame_bufs[j][p], 0, &tw_snapshot[j], 0, plane_size,
                                );
                            }
                            // Read from snapshot, write directly to tw_frame_bufs
                            for pair in 0..pairs {
                                self.temporal_haar.dispatch(
                                    ctx,
                                    &mut cmd,
                                    &tw_snapshot[pair * 2],
                                    &tw_snapshot[pair * 2 + 1],
                                    &tw_frame_bufs[pair][p],
                                    &tw_frame_bufs[pairs + pair][p],
                                    padded_pixels as u32,
                                    true,
                                );
                            }
                            current_count = pairs;
                        }
                        ctx.queue.submit(Some(cmd.finish()));
                    }

                    // Debug logging
                    if diagnostics::enabled() {
                        let group_idx = groups.len();
                        let base = i;
                        let end = base + group_size - 1;
                        eprintln!(
                            "TW ENC group {} frames [{}..{}]",
                            group_idx, base, end
                        );
                    }

                    // 3) Encode low frame from GPU buffer
                    let low_cf = self.encode_from_gpu_wavelet_planes(
                        ctx,
                        [
                            &tw_frame_bufs[0][0],
                            &tw_frame_bufs[0][1],
                            &tw_frame_bufs[0][2],
                        ],
                        &cfg,
                        &info,
                        padded_w,
                        padded_h,
                        padded_pixels,
                    );

                    // 4) Encode high frames per level with per-tile adaptive muls
                    let mut high_cfg = cfg.clone();
                    high_cfg.cfl_enabled = false; // temporal highpass: CfL alpha regression is unreliable on temporal residuals
                    let max_mul = config.temporal_highpass_qstep_mul;
                    let mut high_cfs: Vec<Vec<CompressedFrame>> = Vec::new();
                    for lvl in 0..num_levels {
                        let count = group_size >> (lvl + 1);
                        let start = group_size >> (lvl + 1);
                        let mut lvl_frames: Vec<CompressedFrame> = Vec::new();
                        for idx in 0..count {
                            let buf_idx = start + idx;
                            let (tile_muls, max_abs) = if config.adaptive_temporal_mul {
                                Self::compute_temporal_tile_muls(
                                    ctx,
                                    &tw_frame_bufs[buf_idx][0],
                                    padded_w,
                                    padded_h,
                                    cfg.tile_size,
                                    max_mul,
                                )
                            } else {
                                // Fixed mul: uniform weight for all tiles
                                let tiles_x = padded_w / cfg.tile_size;
                                let tiles_y = padded_h / cfg.tile_size;
                                (vec![max_mul; (tiles_x * tiles_y) as usize], f32::MAX)
                            };
                            if std::env::var("GNC_TW_DIAG").is_ok() {
                                let mut sorted = tile_muls.clone();
                                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                let n = sorted.len();
                                let pct = |p: usize| sorted[(p * n / 100).min(n - 1)];
                                let all_zero = max_abs < high_cfg.dead_zone * high_cfg.quantization_step;
                                eprintln!(
                                    "  TW L{} H{}: {} tiles  mul p10={:.3} p50={:.3} p90={:.3}  eff_qstep=[{:.2}..{:.2}]  skip={}",
                                    lvl, idx, n, pct(10), pct(50), pct(90),
                                    high_cfg.quantization_step * sorted[0],
                                    high_cfg.quantization_step * sorted[n - 1],
                                    all_zero,
                                );
                            }
                            if max_abs < high_cfg.dead_zone * high_cfg.quantization_step {
                                // Frame is all-zero after quantization — push synthetic zero frame
                                lvl_frames.push(Self::make_zero_compressed_frame(&high_cfg, &info));
                                continue;
                            }
                            let cf = self.encode_from_gpu_wavelet_planes_weighted(
                                ctx,
                                [
                                    &tw_frame_bufs[buf_idx][0],
                                    &tw_frame_bufs[buf_idx][1],
                                    &tw_frame_bufs[buf_idx][2],
                                ],
                                &high_cfg,
                                &info,
                                padded_w,
                                padded_h,
                                padded_pixels,
                                Some(&tile_muls),
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
                    // LeGall 5/3 operates on groups of exactly 4 frames.
                    // Output: 2 lowpass (s0, s1) + 2 highpass (d0, d1).
                    assert_eq!(group_size, 4, "LeGall53 requires group_size=4");

                    // Allocate per-frame GPU buffers [frame][plane]
                    let tw_frame_bufs: Vec<[wgpu::Buffer; 3]> = (0..4)
                        .map(|j| {
                            std::array::from_fn(|p| {
                                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                                    label: Some(&format!("tw53_frame_{}_{}", j, p)),
                                    size: plane_size,
                                    usage: wgpu::BufferUsages::STORAGE
                                        | wgpu::BufferUsages::COPY_DST
                                        | wgpu::BufferUsages::COPY_SRC,
                                    mapped_at_creation: false,
                                })
                            })
                        })
                        .collect();

                    // Output buffers: s0, s1, d0, d1 — separate per plane
                    let tw_out_bufs: Vec<[wgpu::Buffer; 3]> = (0..4)
                        .map(|j| {
                            std::array::from_fn(|p| {
                                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                                    label: Some(&format!("tw53_out_{}_{}", j, p)),
                                    size: plane_size,
                                    usage: wgpu::BufferUsages::STORAGE
                                        | wgpu::BufferUsages::COPY_DST
                                        | wgpu::BufferUsages::COPY_SRC,
                                    mapped_at_creation: false,
                                })
                            })
                        })
                        .collect();

                    // 1) Spatial wavelet per frame (same as Haar path)
                    for j in 0..4 {
                        let bufs = self.cached.as_ref().unwrap();
                        ctx.queue.write_buffer(
                            &bufs.raw_input_buf,
                            0,
                            bytemuck::cast_slice(frames[i + j]),
                        );
                        let mut cmd = ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("tw53_spatial"),
                            });
                        self.dispatch_gpu_pad_cached(ctx, &mut cmd, padded_w, padded_h);
                        let bufs = self.cached.as_ref().unwrap();
                        self.color.dispatch(
                            ctx,
                            &mut cmd,
                            &bufs.input_buf,
                            &bufs.color_out,
                            padded_w,
                            padded_h,
                            true,
                            cfg.is_lossless(),
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

                        let planes: [&wgpu::Buffer; 3] =
                            [&bufs.plane_a, &bufs.co_plane, &bufs.cg_plane];
                        for (p, &cur_plane) in planes.iter().enumerate() {
                            self.transform.forward(
                                ctx,
                                &mut cmd,
                                cur_plane,
                                &bufs.plane_b,
                                &bufs.plane_c,
                                &info,
                                cfg.wavelet_levels,
                                cfg.wavelet_type,
                            );
                            cmd.copy_buffer_to_buffer(
                                &bufs.plane_c,
                                0,
                                &tw_frame_bufs[j][p],
                                0,
                                plane_size,
                            );
                        }
                        ctx.queue.submit(Some(cmd.finish()));
                    }

                    // 2) Temporal 5/3 on GPU, per plane — two passes (predict then update)
                    #[allow(clippy::needless_range_loop)]
                    for p in 0..3 {
                        self.temporal_53.forward_4(
                            ctx,
                            &tw_frame_bufs[0][p], // f0
                            &tw_frame_bufs[1][p], // f1
                            &tw_frame_bufs[2][p], // f2
                            &tw_frame_bufs[3][p], // f3
                            &tw_out_bufs[0][p],   // s0
                            &tw_out_bufs[1][p],   // s1
                            &tw_out_bufs[2][p],   // d0
                            &tw_out_bufs[3][p],   // d1
                            padded_pixels as u32,
                        );
                    }

                    if diagnostics::enabled() {
                        let group_idx = groups.len();
                        let base = i;
                        let end = base + group_size - 1;
                        eprintln!(
                            "TW53 ENC group {} frames [{}..{}]",
                            group_idx, base, end
                        );
                    }

                    // 3) Encode lowpass frames (s0, s1) with base qstep
                    let low_s0 = self.encode_from_gpu_wavelet_planes(
                        ctx,
                        [&tw_out_bufs[0][0], &tw_out_bufs[0][1], &tw_out_bufs[0][2]],
                        &cfg,
                        &info,
                        padded_w,
                        padded_h,
                        padded_pixels,
                    );

                    // 4) Encode highpass frames (d0, d1) with per-tile adaptive muls
                    let mut high_cfg = cfg.clone();
                    high_cfg.cfl_enabled = false; // temporal highpass: CfL alpha regression is unreliable on temporal residuals
                    let max_mul = config.temporal_highpass_qstep_mul;

                    // TemporalGroup format: low_frame + high_frames[level][idx]
                    // For 5/3 with 4 frames: 1 level, with s1 as second lowpass
                    // We store: low_frame = s0, high_frames = [[s1, d0, d1]]
                    // This way the decoder knows: first frame of highpass level is actually
                    // the second lowpass, remaining are true highpass.
                    let s1_cf = self.encode_from_gpu_wavelet_planes(
                        ctx,
                        [&tw_out_bufs[1][0], &tw_out_bufs[1][1], &tw_out_bufs[1][2]],
                        &cfg,
                        &info,
                        padded_w,
                        padded_h,
                        padded_pixels,
                    );
                    let (d0_muls, d0_max_abs) = if config.adaptive_temporal_mul {
                        Self::compute_temporal_tile_muls(
                            ctx, &tw_out_bufs[2][0], padded_w, padded_h, cfg.tile_size, max_mul,
                        )
                    } else {
                        let tiles_x = padded_w / cfg.tile_size;
                        let tiles_y = padded_h / cfg.tile_size;
                        (vec![max_mul; (tiles_x * tiles_y) as usize], f32::MAX)
                    };
                    let d0_cf = if d0_max_abs < high_cfg.dead_zone * high_cfg.quantization_step {
                        Self::make_zero_compressed_frame(&high_cfg, &info)
                    } else {
                        self.encode_from_gpu_wavelet_planes_weighted(
                            ctx,
                            [&tw_out_bufs[2][0], &tw_out_bufs[2][1], &tw_out_bufs[2][2]],
                            &high_cfg,
                            &info,
                            padded_w,
                            padded_h,
                            padded_pixels,
                            Some(&d0_muls),
                        )
                    };
                    let (d1_muls, d1_max_abs) = if config.adaptive_temporal_mul {
                        Self::compute_temporal_tile_muls(
                            ctx, &tw_out_bufs[3][0], padded_w, padded_h, cfg.tile_size, max_mul,
                        )
                    } else {
                        let tiles_x = padded_w / cfg.tile_size;
                        let tiles_y = padded_h / cfg.tile_size;
                        (vec![max_mul; (tiles_x * tiles_y) as usize], f32::MAX)
                    };
                    let d1_cf = if d1_max_abs < high_cfg.dead_zone * high_cfg.quantization_step {
                        Self::make_zero_compressed_frame(&high_cfg, &info)
                    } else {
                        self.encode_from_gpu_wavelet_planes_weighted(
                            ctx,
                            [&tw_out_bufs[3][0], &tw_out_bufs[3][1], &tw_out_bufs[3][2]],
                            &high_cfg,
                            &info,
                            padded_w,
                            padded_h,
                            padded_pixels,
                            Some(&d1_muls),
                        )
                    };

                    groups.push(TemporalGroup {
                        low_frame: low_s0,
                        high_frames: vec![vec![s1_cf, d0_cf, d1_cf]],
                    });
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

    /// Encode a single GOP of frames using temporal Haar wavelet.
    ///
    /// This is the streaming-friendly version: pass in exactly `gop_size` frames,
    /// get back one `TemporalGroup`. Call repeatedly for each GOP, then use
    /// `encode()` for any tail frames.
    ///
    /// `next_gop_frames`: optional slice of the NEXT GOP's raw frames. When provided,
    /// these are written to the GPU staging buffers WHILE the GPU runs the high-frame
    /// Rice encode (~100ms), hiding the ~22ms upload cost. The following GOP's encode
    /// call detects the pre-upload and skips its write_buffer step.
    pub fn encode_temporal_wavelet_gop_haar(
        &mut self,
        ctx: &GpuContext,
        gop_frames: &[&[f32]],
        width: u32,
        height: u32,
        config: &CodecConfig,
        next_gop_frames: Option<&[&[f32]]>,
    ) -> TemporalGroup {
        let group_size = gop_frames.len();
        assert!(
            temporal::is_power_of_two(group_size) && group_size >= 2,
            "temporal Haar requires GOP size to be a power of two >= 2, got {}",
            group_size
        );

        let mut cfg = config.clone();
        cfg.keyframe_interval = 1;
        cfg.temporal_transform = TemporalTransform::None;

        let info = FrameInfo {
            width,
            height,
            bit_depth: 8,
            tile_size: cfg.tile_size,
        };
        let padded_w = info.padded_width();
        let padded_h = info.padded_height();
        let padded_pixels = (padded_w * padded_h) as usize;
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        self.ensure_cached(ctx, padded_w, padded_h, width, height);

        // Reuse cached temporal wavelet buffers across GOPs (avoids ~22ms per-GOP allocation).
        // We take() the cache out of self to avoid holding an immutable borrow on self
        // while calling &mut self methods (encode_from_gpu_wavelet_planes etc.).
        let raw_input_size = std::mem::size_of_val(gop_frames[0]) as u64;
        self.ensure_tw_cached(ctx, padded_w, padded_h, group_size, raw_input_size);
        // Ensure second buffer set for GOP pipelining (pre-compute next GOP during high_enc).
        self.ensure_tw_cached_b(ctx, padded_w, padded_h, group_size, raw_input_size);
        self.ensure_sp_cached_b(ctx, padded_w, padded_h, width, height);
        let mut tw = self.tw_cached.take().unwrap();

        // Per-stage wall-clock profiling (gated behind GNC_HAAR_PROFILE env var).
        let prof = std::env::var("GNC_HAAR_PROFILE").is_ok();
        let t_start = if prof { Some(std::time::Instant::now()) } else { None };

        // Skip spatial wavelet + temporal Haar if pre-computed during previous GOP's high_enc.
        // This overlaps ~72ms of GPU compute with the previous GOP's Rice encoding (~100ms),
        // saving those phases from the critical path and pushing throughput toward 60 fps.
        let skip_spatial_haar = tw.spatial_haar_precomputed;
        tw.spatial_haar_precomputed = false; // consume flag

        // Skip upload + spatial wavelet + temporal Haar when pre-computed during previous GOP.
        // When skip_spatial_haar is true, tw.frame_bufs already contains the post-spatial+Haar
        // wavelet data for all group_size frames (computed concurrently with prev GOP's high_enc).
        let num_levels = (group_size as f64).log2() as usize;
        let (t_after_upload, t_after_spatial, t_after_haar);
        if skip_spatial_haar {
            let t = if prof { Some(std::time::Instant::now()) } else { None };
            t_after_upload = t;
            t_after_spatial = t;
            t_after_haar = t;
        } else {
            // Upload all frames to per-frame GPU buffers (avoids write_buffer race).
            // Skip if the previous GOP pre-uploaded these frames during its high_enc phase.
            if tw.next_gop_pre_uploaded {
                tw.next_gop_pre_uploaded = false; // consume the pre-upload flag
            } else {
                for (input_buf, frame_data) in tw.per_frame_input.iter().zip(gop_frames.iter()) {
                    ctx.queue.write_buffer(input_buf, 0, bytemuck::cast_slice(frame_data));
                }
            }

            t_after_upload = if prof { Some(std::time::Instant::now()) } else { None };

            // 1) Spatial wavelet per frame — single command encoder to prevent
            // intermediate buffer races (separate submits can overlap on GPU)
            {
                let bufs = self.cached.as_ref().unwrap();
                let mut cmd = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("tw_spatial_all"),
                    });
                for j in 0..group_size {
                    cmd.copy_buffer_to_buffer(
                        &tw.per_frame_input[j],
                        0,
                        &bufs.raw_input_buf,
                        0,
                        raw_input_size,
                    );
                    self.dispatch_gpu_pad_cached(ctx, &mut cmd, padded_w, padded_h);
                    let bufs = self.cached.as_ref().unwrap();
                    self.color.dispatch(
                        ctx,
                        &mut cmd,
                        &bufs.input_buf,
                        &bufs.color_out,
                        padded_w,
                        padded_h,
                        true,
                        cfg.is_lossless(),
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

                    let planes: [&wgpu::Buffer; 3] =
                        [&bufs.plane_a, &bufs.co_plane, &bufs.cg_plane];
                    for (p, &cur_plane) in planes.iter().enumerate() {
                        self.transform.forward(
                            ctx,
                            &mut cmd,
                            cur_plane,
                            &bufs.plane_b,
                            &bufs.plane_c,
                            &info,
                            cfg.wavelet_levels,
                            cfg.wavelet_type,
                        );
                        cmd.copy_buffer_to_buffer(
                            &bufs.plane_c,
                            0,
                            &tw.frame_bufs[j][p],
                            0,
                            plane_size,
                        );
                    }
                }
                ctx.queue.submit(Some(cmd.finish()));
                if prof { ctx.device.poll(wgpu::Maintain::Wait); }
            }

            t_after_spatial = if prof { Some(std::time::Instant::now()) } else { None };

            // 2) Temporal Haar on GPU, per plane — multilevel dyadic decomposition
            #[allow(clippy::needless_range_loop)]
            for p in 0..3 {
                let mut current_count = group_size;
                let mut cmd = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("tw_temporal_haar"),
                    });
                for _level in 0..num_levels {
                    let pairs = current_count / 2;
                    for j in 0..current_count {
                        cmd.copy_buffer_to_buffer(
                            &tw.frame_bufs[j][p], 0, &tw.snapshot[j], 0, plane_size,
                        );
                    }
                    for pair in 0..pairs {
                        self.temporal_haar.dispatch(
                            ctx,
                            &mut cmd,
                            &tw.snapshot[pair * 2],
                            &tw.snapshot[pair * 2 + 1],
                            &tw.frame_bufs[pair][p],
                            &tw.frame_bufs[pairs + pair][p],
                            padded_pixels as u32,
                            true,
                        );
                    }
                    current_count = pairs;
                }
                ctx.queue.submit(Some(cmd.finish()));
                if prof { ctx.device.poll(wgpu::Maintain::Wait); }
            }

            t_after_haar = if prof { Some(std::time::Instant::now()) } else { None };
        } // end if !skip_spatial_haar

        // 3) Encode low frame from GPU buffer
        let low_cf = self.encode_from_gpu_wavelet_planes(
            ctx,
            [
                &tw.frame_bufs[0][0],
                &tw.frame_bufs[0][1],
                &tw.frame_bufs[0][2],
            ],
            &cfg,
            &info,
            padded_w,
            padded_h,
            padded_pixels,
        );

        let t_after_low_enc = if prof { Some(std::time::Instant::now()) } else { None };

        // 4) Encode high frames per level with per-tile adaptive muls.
        //    CfL is disabled for highpass frames (alpha regression unreliable on residuals).
        //    Batched: all non-zero frames are quantized+rice-encoded in ONE command encoder
        //    (single submit+poll) to eliminate per-frame CPU-GPU sync overhead.
        let mut high_cfg = cfg.clone();
        high_cfg.cfl_enabled = false; // temporal highpass: CfL alpha regression is unreliable on temporal residuals
        let max_mul = config.temporal_highpass_qstep_mul;

        // --- Pass A: Compute adaptive muls for all high frames in ONE batched readback ---
        // Build the ordered list of (buf_idx, lvl, idx) for all high frames
        // Collect buf_idx order (matches level/idx nested loop order)
        let mut hframe_buf_indices: Vec<(usize, usize, usize)> = Vec::new(); // (buf_idx, lvl, idx)
        for lvl in 0..num_levels {
            let count = group_size >> (lvl + 1);
            let start = group_size >> (lvl + 1);
            for idx in 0..count {
                hframe_buf_indices.push((start + idx, lvl, idx));
            }
        }
        let total_high_frames = hframe_buf_indices.len();

        // Note: hard-zeroing of high-energy tiles (TILE_ENERGY_ZERO_THRESH) was removed.
        // The adaptive mul system (tile_energy_reduce.wgsl, MIN_MUL=1.2) already handles
        // high-motion tiles gracefully by quantizing them coarser. Zeroing at 17.0
        // mean_abs (6.7% of peak) is too aggressive and causes ghosting artifacts
        // (decoder falls back to LL-only = temporal average) in affected tiles.

        let mul_results: Vec<(Vec<f32>, Vec<f32>, f32)> = if config.adaptive_temporal_mul {
            // GPU-side per-tile energy reduction — eliminates 58MB CPU readback.
            //
            // For each high frame: dispatch tile_energy_reduce shader which computes
            // mean_abs per tile and maps it to a mul via log-space interpolation.
            // Only max_abs (4 bytes per frame) is read back to CPU for zero-skip detection.
            // tile_muls (~160 bytes per frame) are read back as a small separate readback.

            let tiles_x = padded_w / cfg.tile_size;
            let tiles_y = padded_h / cfg.tile_size;
            let num_tiles = (tiles_x * tiles_y) as usize;
            let tile_muls_bytes = (num_tiles * std::mem::size_of::<f32>()) as u64;

            // Allocate per-frame staging buffers for tile_muls and max_abs readback.
            // These are small (~160 bytes + 4 bytes per frame) — negligible allocation cost.
            let n = total_high_frames;
            let tile_muls_staging: Vec<wgpu::Buffer> = (0..n)
                .map(|j| {
                    ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("tw_ter_muls_stg_{}", j)),
                        size: tile_muls_bytes.max(4),
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })
                .collect();

            // Pre-clear max_abs_bufs to 0 (atomicMax requires 0 as identity).
            // write_buffer is staged — these writes land before the GPU dispatch.
            let zero_bytes: [u8; 4] = [0u8; 4];
            for j in 0..n {
                ctx.queue.write_buffer(&tw.max_abs_bufs[j], 0, &zero_bytes);
            }

            // Single command encoder: all tile_energy_reduce dispatches + copies to staging.
            let mut cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tw_tile_energy_reduce_batch"),
            });
            for (j, &(buf_idx, _, _)) in hframe_buf_indices.iter().enumerate() {
                self.dispatch_tile_energy_reduce(
                    ctx,
                    &mut cmd,
                    &tw.frame_bufs[buf_idx][0],
                    &tw.tile_muls_bufs[j],
                    &tw.max_abs_bufs[j],
                    &tw.tile_energies_bufs[j],
                    &tw.ter_params_buf,
                    padded_w,
                    padded_h,
                    cfg.tile_size,
                    max_mul,
                );
                // Copy tile_muls, tile_energies, and max_abs outputs to staging (CPU-readable)
                cmd.copy_buffer_to_buffer(
                    &tw.tile_muls_bufs[j], 0,
                    &tile_muls_staging[j], 0,
                    tile_muls_bytes.max(4),
                );
                cmd.copy_buffer_to_buffer(
                    &tw.tile_energies_bufs[j], 0,
                    &tw.tile_energies_staging_bufs[j], 0,
                    tile_muls_bytes.max(4),
                );
                cmd.copy_buffer_to_buffer(
                    &tw.max_abs_bufs[j], 0,
                    &tw.max_abs_staging_bufs[j], 0,
                    4,
                );
            }
            ctx.queue.submit(Some(cmd.finish()));

            // Map all staging buffers, single poll.
            let (tx, rx) = std::sync::mpsc::channel();
            for (j, muls_stg) in tile_muls_staging.iter().enumerate() {
                let tx2 = tx.clone();
                muls_stg
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| { tx2.send(('m', j, r)).unwrap(); });
                let tx3 = tx.clone();
                tw.tile_energies_staging_bufs[j]
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| { tx3.send(('e', j, r)).unwrap(); });
                let tx4 = tx.clone();
                tw.max_abs_staging_bufs[j]
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| { tx4.send(('x', j, r)).unwrap(); });
            }
            drop(tx);
            ctx.device.poll(wgpu::Maintain::Wait);
            // Drain channel (3 callbacks per frame: tile_muls + tile_energies + max_abs)
            for _ in 0..(n * 3) {
                rx.recv().unwrap().2.unwrap();
            }

            // Collect results: tile_muls (Vec<f32>) + tile_energies (Vec<f32>) + max_abs (f32) per frame
            let mut results: Vec<(Vec<f32>, Vec<f32>, f32)> = Vec::with_capacity(n);
            for (j, muls_stg) in tile_muls_staging.iter().enumerate() {
                let muls_view = muls_stg.slice(..).get_mapped_range();
                let muls: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&muls_view[..tile_muls_bytes as usize])
                    .to_vec();
                drop(muls_view);
                muls_stg.unmap();

                let energies_view = tw.tile_energies_staging_bufs[j].slice(..).get_mapped_range();
                let energies: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&energies_view[..tile_muls_bytes as usize])
                    .to_vec();
                drop(energies_view);
                tw.tile_energies_staging_bufs[j].unmap();

                let max_view = tw.max_abs_staging_bufs[j].slice(..).get_mapped_range();
                let max_bits = u32::from_le_bytes(max_view[..4].try_into().unwrap());
                drop(max_view);
                tw.max_abs_staging_bufs[j].unmap();
                let max_abs = f32::from_bits(max_bits);

                results.push((muls, energies, max_abs));
            }
            results
        } else {
            let tiles_x = padded_w / cfg.tile_size;
            let tiles_y = padded_h / cfg.tile_size;
            let n_tiles = (tiles_x * tiles_y) as usize;
            // Non-adaptive path: all tiles get max_mul, energies treated as zero (no zeroing).
            vec![(vec![max_mul; n_tiles], vec![0.0f32; n_tiles], f32::MAX); total_high_frames]
        };

        let t_after_aq = if prof { Some(std::time::Instant::now()) } else { None };

        let mut hframe_infos: Vec<HighFrameInfo> = Vec::with_capacity(total_high_frames);
        for (i, &(buf_idx, lvl, idx)) in hframe_buf_indices.iter().enumerate() {
            let (tile_muls, tile_energies, max_abs) = mul_results[i].clone();
            if std::env::var("GNC_TW_DIAG").is_ok() {
                let mut sorted = tile_muls.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = sorted.len();
                let pct = |p: usize| sorted[(p * n / 100).min(n - 1)];
                let all_zero = max_abs < high_cfg.dead_zone * high_cfg.quantization_step;
                let mut esorted = tile_energies.clone();
                esorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let en = esorted.len();
                let epct = |p: usize| esorted[(p * en / 100).min(en - 1)];
                eprintln!(
                    "  TW L{} H{}: {} tiles  mul p10={:.3} p50={:.3} p90={:.3}  eff_qstep=[{:.2}..{:.2}]  skip={}  energy p50={:.1} p90={:.1} p99={:.1}",
                    lvl, idx, n, pct(10), pct(50), pct(90),
                    high_cfg.quantization_step * sorted[0],
                    high_cfg.quantization_step * sorted[n - 1],
                    all_zero, epct(50), epct(90), epct(99),
                );
            }
            let is_zero = max_abs < high_cfg.dead_zone * high_cfg.quantization_step;
            hframe_infos.push(HighFrameInfo { tile_muls, buf_idx, is_zero, wm_data: None });
        }

        // --- Pass B: Expand weight maps for non-zero frames (CPU work, no GPU sync) ---
        let tiles_x = padded_w / cfg.tile_size;
        let tiles_y = padded_h / cfg.tile_size;
        let ll_size = high_cfg.tile_size >> high_cfg.wavelet_levels;
        let ll_block_size = AQ_LL_BLOCK_SIZE.min(ll_size);
        let ll_bx = ll_size.div_ceil(ll_block_size);
        let ll_by = ll_bx;
        let blocks_per_tile = (ll_bx * ll_by) as usize;
        let total_blocks = (tiles_x * tiles_y) as usize * blocks_per_tile;

        for hf in &mut hframe_infos {
            if hf.is_zero {
                continue;
            }
            let mut expanded = vec![1.0f32; total_blocks];
            for ty_idx in 0..tiles_y as usize {
                for tx_idx in 0..tiles_x as usize {
                    let tile_idx = ty_idx * tiles_x as usize + tx_idx;
                    // Use the adaptive mul directly. High-energy tiles already get
                    // MIN_MUL=1.2 (minimal quantization) from tile_energy_reduce.wgsl,
                    // which is far preferable to hard-zeroing (which causes decoder ghosting).
                    let w = hf.tile_muls[tile_idx];
                    for by in 0..ll_by as usize {
                        for bx in 0..ll_bx as usize {
                            let global_bx = tx_idx * ll_bx as usize + bx;
                            let global_by = ty_idx * ll_bx as usize + by;
                            let global_blocks_x = ll_bx as usize * tiles_x as usize;
                            expanded[global_by * global_blocks_x + global_bx] = w;
                        }
                    }
                }
            }
            hf.wm_data = Some(expanded);
        }

        // Count non-zero frames for batch sizing
        let non_zero_frames: Vec<usize> = hframe_infos
            .iter()
            .enumerate()
            .filter(|(_, hf)| !hf.is_zero)
            .map(|(i, _)| i)
            .collect();
        let batch_size = non_zero_frames.len();

        // --- Pass C: Batch encode all non-zero high frames in one command encoder ---
        let mut high_cfs: Vec<Vec<CompressedFrame>> = Vec::new();

        if batch_size > 0 {
            use wgpu::util::DeviceExt;

            let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;
            self.gpu_rice_encoder.prepare_batch_staging(ctx, num_tiles, batch_size);
            // Write params_buf ONCE before the batch loop.  On Metal/wgpu write_buffer is
            // staged: only the last write before queue.submit takes effect.  All high frames
            // in a GOP share the same FrameInfo (same tile layout, same wavelet_levels),
            // so a single pre-write is both correct and sufficient.
            self.gpu_rice_encoder
                .write_params_buf_for_batch(ctx, &info, high_cfg.wavelet_levels);

            let weights_luma = high_cfg.subband_weights.pack_weights();
            let weights_chroma = high_cfg.subband_weights.pack_weights_chroma();
            let bufs = self.cached.as_ref().unwrap();
            let quant_dests: [&wgpu::Buffer; 3] = [&bufs.mc_out, &bufs.ref_upload, &bufs.plane_b];

            // Build weight-map GPU buffers (created per-frame, owned here)
            let mut wm_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(batch_size);
            for &fi in &non_zero_frames {
                let expanded = hframe_infos[fi].wm_data.as_ref().unwrap();
                wm_bufs.push(ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("tw_aq_weight_map_batch"),
                    contents: bytemuck::cast_slice(expanded),
                    usage: wgpu::BufferUsages::STORAGE,
                }));
            }

            let mut cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tw_high_frames_batch"),
            });

            for (slot, &fi) in non_zero_frames.iter().enumerate() {
                let hf = &hframe_infos[fi];
                let plane_bufs: [&wgpu::Buffer; 3] = [
                    &tw.frame_bufs[hf.buf_idx][0],
                    &tw.frame_bufs[hf.buf_idx][1],
                    &tw.frame_bufs[hf.buf_idx][2],
                ];

                let wm_buf = &wm_bufs[slot];
                let wm_param = Some((
                    wm_buf as &wgpu::Buffer,
                    ll_block_size,
                    ll_bx,
                    tiles_x,
                ));

                // Quantize 3 planes (non-CfL path: just dispatch_adaptive for each)
                for p in 0..3 {
                    let weights = if p == 0 { &weights_luma } else { &weights_chroma };
                    cmd.copy_buffer_to_buffer(plane_bufs[p], 0, &bufs.plane_c, 0, plane_size);
                    self.quantize.dispatch_adaptive(
                        ctx, &mut cmd, &bufs.plane_c, quant_dests[p],
                        padded_pixels as u32, high_cfg.quantization_step, high_cfg.dead_zone,
                        true, padded_w, padded_h, high_cfg.tile_size, high_cfg.wavelet_levels,
                        weights, wm_param, 0.0,
                    );
                }

                // Rice-encode 3 quantized planes, copying to per-frame batch staging slot
                self.gpu_rice_encoder.dispatch_3planes_to_cmd_batch(
                    ctx,
                    &mut cmd,
                    quant_dests,
                    &info,
                    slot,
                );
            }

            // Single submit + single poll for ALL high frames
            ctx.queue.submit(Some(cmd.finish()));

            // GOP pipelining: pre-compute next GOP's spatial wavelet + temporal Haar into the B
            // buffer set while the current GOP's Rice encoding runs on GPU (~100ms window).
            // The spatial+haar pre-compute takes ~72ms, fitting within the high_enc window.
            // device.poll(Wait) in finish_batch_readback waits for ALL submitted GPU work,
            // so both high_enc (A set) and spatial+haar (B set) complete before we proceed.
            //
            // Buffer isolation: high_enc reads tw.frame_bufs[A]; spatial+haar writes
            // tw_b.frame_bufs[B] and uses sp_cached_b intermediate buffers — no conflicts.
            if let Some(next_frames) = next_gop_frames {
                let mut tw_b = self.tw_cached_b.take().unwrap();
                // Write next GOP's frames to tw_b.per_frame_input (staged; consumed at submit).
                for (input_buf, frame_data) in tw_b.per_frame_input.iter().zip(next_frames.iter()) {
                    ctx.queue.write_buffer(input_buf, 0, bytemuck::cast_slice(frame_data));
                }
                // Build single command encoder: spatial_wl + temporal_haar for next GOP.
                // Both stages share intermediate sp_b buffers and write to tw_b.frame_bufs.
                {
                    let sp_b = self.sp_cached_b.as_ref().unwrap();
                    let pad_params = &self.cached.as_ref().unwrap().pad_params_buf;
                    let mut cmd_pre = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("tw_precompute_spatial_haar"),
                    });
                    // Spatial wavelet for all frames into tw_b.frame_bufs
                    for j in 0..group_size {
                        cmd_pre.copy_buffer_to_buffer(
                            &tw_b.per_frame_input[j], 0, &sp_b.raw_input_buf, 0, raw_input_size,
                        );
                        self.dispatch_gpu_pad_with(
                            ctx, &mut cmd_pre,
                            pad_params, &sp_b.raw_input_buf, &sp_b.input_buf,
                            padded_w, padded_h,
                        );
                        self.color.dispatch(
                            ctx, &mut cmd_pre,
                            &sp_b.input_buf, &sp_b.color_out,
                            padded_w, padded_h, true, cfg.is_lossless(),
                        );
                        self.deinterleaver.dispatch(
                            ctx, &mut cmd_pre,
                            &sp_b.color_out, &sp_b.plane_a, &sp_b.co_plane, &sp_b.cg_plane,
                            padded_pixels as u32,
                        );
                        let planes_b: [&wgpu::Buffer; 3] =
                            [&sp_b.plane_a, &sp_b.co_plane, &sp_b.cg_plane];
                        for (p, &cur_plane) in planes_b.iter().enumerate() {
                            self.transform.forward(
                                ctx, &mut cmd_pre,
                                cur_plane, &sp_b.plane_b, &sp_b.plane_c,
                                &info, cfg.wavelet_levels, cfg.wavelet_type,
                            );
                            cmd_pre.copy_buffer_to_buffer(
                                &sp_b.plane_c, 0, &tw_b.frame_bufs[j][p], 0, plane_size,
                            );
                        }
                    }
                    // Temporal Haar for all 3 planes on tw_b.frame_bufs
                    #[allow(clippy::needless_range_loop)]
                    for p in 0..3 {
                        let mut current_count = group_size;
                        for _level in 0..num_levels {
                            let pairs = current_count / 2;
                            for j in 0..current_count {
                                cmd_pre.copy_buffer_to_buffer(
                                    &tw_b.frame_bufs[j][p], 0, &tw_b.snapshot[j], 0, plane_size,
                                );
                            }
                            for pair in 0..pairs {
                                self.temporal_haar.dispatch(
                                    ctx, &mut cmd_pre,
                                    &tw_b.snapshot[pair * 2], &tw_b.snapshot[pair * 2 + 1],
                                    &tw_b.frame_bufs[pair][p], &tw_b.frame_bufs[pairs + pair][p],
                                    padded_pixels as u32, true,
                                );
                            }
                            current_count = pairs;
                        }
                    }
                    // Submit pre-compute: runs concurrently with the current GOP's Rice encoding.
                    // No memory hazards — uses entirely separate buffers (tw_b + sp_b vs tw + cached).
                    ctx.queue.submit(Some(cmd_pre.finish()));
                }
                tw_b.spatial_haar_precomputed = true;
                self.tw_cached_b = Some(tw_b);
            }

            let all_frame_tiles = self.gpu_rice_encoder.finish_batch_readback(
                ctx,
                &info,
                high_cfg.wavelet_levels,
                batch_size,
            );

            // Assemble CompressedFrames for all frames (zero and non-zero) per level
            let mut slot_iter = all_frame_tiles.into_iter();
            let mut per_frame_tiles: Vec<Option<Vec<rice::RiceTile>>> =
                vec![None; hframe_infos.len()];
            for &fi in &non_zero_frames {
                per_frame_tiles[fi] = Some(slot_iter.next().unwrap());
            }

            let mut frame_idx = 0;
            for lvl in 0..num_levels {
                let count = group_size >> (lvl + 1);
                let mut lvl_frames: Vec<CompressedFrame> = Vec::new();
                for _idx in 0..count {
                    let hf = &hframe_infos[frame_idx];
                    if hf.is_zero {
                        lvl_frames.push(Self::make_zero_compressed_frame(&high_cfg, &info));
                    } else {
                        let rice_tiles = per_frame_tiles[frame_idx].take().unwrap();
                        lvl_frames.push(CompressedFrame {
                            info,
                            config: high_cfg.clone(),
                            entropy: EntropyData::Rice(rice_tiles),
                            cfl_alphas: None,
                            weight_map: hframe_infos[frame_idx].wm_data.clone(),
                            frame_type: FrameType::Intra,
                            motion_field: None,
                            intra_modes: None,
                            residual_stats: None,
                            residual_stats_co: None,
                            residual_stats_cg: None,
                        });
                    }
                    frame_idx += 1;
                }
                high_cfs.push(lvl_frames);
            }
        } else {
            // All high frames are zero — just push synthetic zero frames
            for lvl in 0..num_levels {
                let count = group_size >> (lvl + 1);
                let lvl_frames: Vec<CompressedFrame> = (0..count)
                    .map(|_| Self::make_zero_compressed_frame(&high_cfg, &info))
                    .collect();
                high_cfs.push(lvl_frames);
            }
        }

        let t_after_high_enc = if prof { Some(std::time::Instant::now()) } else { None };

        // Ping-pong buffer swap for GOP pipelining:
        // If we pre-computed next GOP's spatial+haar into tw_b, promote tw_b → tw_cached
        // (ready for next call) and recycle tw (done) → tw_cached_b (for next pre-compute).
        // If no pre-compute happened (last GOP or no next_gop_frames), keep tw as tw_cached.
        if self.tw_cached_b.as_ref().map_or(false, |b| b.spatial_haar_precomputed) {
            let tw_b = self.tw_cached_b.take().unwrap();
            self.tw_cached_b = Some(tw);      // recycle current A → becomes next B slot
            self.tw_cached = Some(tw_b);       // pre-computed B → becomes next A (current)
        } else {
            self.tw_cached = Some(tw);
        }

        // Print per-stage profiling summary if GNC_HAAR_PROFILE is set.
        if prof {
            let t0 = t_start.unwrap();
            let t1 = t_after_upload.unwrap();
            let t2 = t_after_spatial.unwrap();
            let t3 = t_after_haar.unwrap();
            let t4 = t_after_low_enc.unwrap();
            let t5 = t_after_aq.unwrap();
            let t6 = t_after_high_enc.unwrap();
            let ms = |a: std::time::Instant, b: std::time::Instant| -> f64 {
                b.duration_since(a).as_secs_f64() * 1000.0
            };
            let upload_ms    = ms(t0, t1);
            let spatial_ms   = ms(t1, t2);
            let haar_ms      = ms(t2, t3);
            let low_enc_ms   = ms(t3, t4);
            let aq_ms        = ms(t4, t5);
            let high_enc_ms  = ms(t5, t6);
            let total_ms     = ms(t0, t6);
            let other_ms     = total_ms - upload_ms - spatial_ms - haar_ms
                               - low_enc_ms - aq_ms - high_enc_ms;
            eprintln!(
                "[HAAR_PROF] upload={:.1}ms spatial_wl={:.1}ms temporal_haar={:.1}ms \
                 low_enc={:.1}ms aq_readback={:.1}ms high_enc={:.1}ms \
                 other={:.1}ms TOTAL={:.1}ms",
                upload_ms, spatial_ms, haar_ms, low_enc_ms, aq_ms, high_enc_ms,
                other_ms, total_ms,
            );
        }

        TemporalGroup {
            low_frame: low_cf,
            high_frames: high_cfs,
        }
    }

    /// Encode a single GOP using temporal wavelet (Haar or LeGall 5/3).
    /// Streaming-friendly: pass exactly `gop_size` frames, get one `TemporalGroup`.
    ///
    /// `next_gop_frames`: optional next GOP's frames for async pre-upload (Haar only).
    /// When provided, they are written to GPU staging during high_enc, hiding ~22ms upload cost.
    #[allow(clippy::too_many_arguments)] // temporal GOP encode genuinely needs all these params
    pub fn encode_temporal_wavelet_gop(
        &mut self,
        ctx: &GpuContext,
        gop_frames: &[&[f32]],
        width: u32,
        height: u32,
        config: &CodecConfig,
        mode: TemporalTransform,
        next_gop_frames: Option<&[&[f32]]>,
    ) -> TemporalGroup {
        match mode {
            TemporalTransform::Haar => {
                self.encode_temporal_wavelet_gop_haar(ctx, gop_frames, width, height, config, next_gop_frames)
            }
            TemporalTransform::LeGall53 => {
                assert_eq!(gop_frames.len(), 4, "LeGall53 requires exactly 4 frames per GOP");
                // Use encode_sequence_temporal_wavelet with a 4-frame slice
                let seq = self.encode_sequence_temporal_wavelet(
                    ctx, gop_frames, width, height, config, mode, 4,
                );
                assert_eq!(seq.groups.len(), 1);
                seq.groups.into_iter().next().unwrap()
            }
            TemporalTransform::None => unreachable!("None mode in temporal wavelet GOP encode"),
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
    /// Kept for potential future use (e.g. LeGall53 temporal transform).
    #[allow(dead_code)]
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
        for stg in &staging {
            let quantized =
                diagnostics::read_plane_f32(ctx, stg, plane_size, padded_pixels);
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

    /// Like `encode_from_wavelet_coeffs` but takes GPU buffers instead of CPU slices.
    /// This avoids the CPU readback + re-upload roundtrip for temporal wavelet.
    #[allow(clippy::too_many_arguments)]
    /// Encode wavelet coefficients from GPU buffers using the fast GPU entropy path.
    ///
    /// All 3 planes are quantized in a single command encoder (using mc_out, ref_upload,
    /// plane_b as per-plane output buffers — matching the normal encode() layout), then
    /// GPU Rice encode is dispatched in the same submit. This gives ~1 submit per frame
    /// instead of 3 submits + 3 full-plane readbacks for CPU entropy.
    fn encode_from_gpu_wavelet_planes(
        &mut self,
        ctx: &GpuContext,
        plane_bufs: [&wgpu::Buffer; 3],
        config: &CodecConfig,
        info: &FrameInfo,
        padded_w: u32,
        padded_h: u32,
        padded_pixels: usize,
    ) -> CompressedFrame {
        self.encode_from_gpu_wavelet_planes_weighted(
            ctx, plane_bufs, config, info, padded_w, padded_h, padded_pixels, None,
        )
    }

    /// Encode wavelet planes with optional per-tile adaptive weights for temporal highpass.
    ///
    /// When `tile_weights` is provided, it contains one weight per tile (tiles_x * tiles_y).
    /// Higher weight → coarser quantization (for static tiles where temporal highpass is near zero).
    /// Lower weight → finer quantization (for motion tiles with significant temporal detail).
    /// The weights are expanded to per-LL-block format for the existing AQ mechanism.
    ///
    /// When `config.cfl_enabled` is true, CfL prediction is applied to chroma planes:
    /// Y is quantized+dequantized to produce a reconstructed reference, then chroma planes
    /// are predicted from it. The residual (chroma - alpha * luma) is quantized instead.
    #[allow(clippy::too_many_arguments)] // GPU+quant context genuinely needs all 9 params
    fn encode_from_gpu_wavelet_planes_weighted(
        &mut self,
        ctx: &GpuContext,
        plane_bufs: [&wgpu::Buffer; 3],
        config: &CodecConfig,
        info: &FrameInfo,
        padded_w: u32,
        padded_h: u32,
        padded_pixels: usize,
        tile_weights: Option<&[f32]>,
    ) -> CompressedFrame {
        use crate::CflAlphas;
        use wgpu::util::DeviceExt;

        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();

        let tiles_x = padded_w / config.tile_size;
        let tiles_y = padded_h / config.tile_size;
        let nsb = cfl::num_subbands(config.wavelet_levels);

        // Ensure CfL alpha buffers are large enough
        if config.cfl_enabled {
            let total_tiles = tiles_x * tiles_y;
            let alpha_buf_size =
                (total_tiles * nsb) as u64 * std::mem::size_of::<f32>() as u64;
            let bufs = self.cached.as_mut().unwrap();
            crate::gpu_util::ensure_var_buf(
                ctx,
                &mut bufs.raw_alpha,
                &mut bufs.raw_alpha_cap,
                alpha_buf_size,
                "enc_raw_alpha",
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            crate::gpu_util::ensure_var_buf(
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
        // Quantized output destinations (same layout as normal encode):
        // Y → mc_out, Co → ref_upload, Cg → plane_b
        let quant_dests: [&wgpu::Buffer; 3] = [&bufs.mc_out, &bufs.ref_upload, &bufs.plane_b];

        // Expand per-tile weights to per-LL-block format if provided
        let (wm_buf_storage, wm_data) = if let Some(tw) = tile_weights {
            let ll_size = config.tile_size >> config.wavelet_levels;
            let ll_block_size = AQ_LL_BLOCK_SIZE.min(ll_size);
            let ll_bx = ll_size.div_ceil(ll_block_size);
            let ll_by = ll_bx;
            let blocks_per_tile = (ll_bx * ll_by) as usize;
            let total_blocks = (tiles_x * tiles_y) as usize * blocks_per_tile;

            // Expand: all LL-blocks within a tile share the tile's weight
            let mut expanded = vec![1.0f32; total_blocks];
            for ty_idx in 0..tiles_y as usize {
                for tx_idx in 0..tiles_x as usize {
                    let tile_idx = ty_idx * tiles_x as usize + tx_idx;
                    let w = tw[tile_idx];
                    for by in 0..ll_by as usize {
                        for bx in 0..ll_bx as usize {
                            let global_bx = tx_idx * ll_bx as usize + bx;
                            let global_by = ty_idx * ll_bx as usize + by;
                            let global_blocks_x = ll_bx as usize * tiles_x as usize;
                            expanded[global_by * global_blocks_x + global_bx] = w;
                        }
                    }
                }
            }

            let buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("tw_aq_weight_map"),
                    contents: bytemuck::cast_slice(&expanded),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            (Some(buf), Some(expanded))
        } else {
            (None, None)
        };

        let wm_param = if let Some(ref wm_buf) = wm_buf_storage {
            let ll_size = config.tile_size >> config.wavelet_levels;
            let ll_block_size = AQ_LL_BLOCK_SIZE.min(ll_size);
            let ll_bx = ll_size.div_ceil(ll_block_size);
            Some((wm_buf as &wgpu::Buffer, ll_block_size, ll_bx, tiles_x))
        } else {
            None
        };

        // CfL alpha staging buffers (for readback)
        let alpha_count = (tiles_x * tiles_y * nsb) as usize;
        let alpha_bytes = (alpha_count * std::mem::size_of::<f32>()) as u64;
        let alpha_staging: [wgpu::Buffer; 2] = if config.cfl_enabled {
            std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(["tw_alpha_stg_co", "tw_alpha_stg_cg"][i]),
                    size: alpha_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
        } else {
            std::array::from_fn(|_| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("tw_alpha_stg_dummy"),
                    size: 4,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
        };

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tw_gpu_quant_entropy"),
            });

        if config.cfl_enabled {
            // CfL path: Y must be quantized+dequantized first to produce recon_y reference

            // Y plane: quantize → dequantize → recon_y
            cmd.copy_buffer_to_buffer(plane_bufs[0], 0, &bufs.plane_c, 0, plane_size);
            self.quantize.dispatch_adaptive(
                ctx, &mut cmd, &bufs.plane_c, quant_dests[0],
                padded_pixels as u32, config.quantization_step, config.dead_zone,
                true, padded_w, padded_h, config.tile_size, config.wavelet_levels,
                &weights_luma, wm_param, 0.0,
            );
            // Dequantize Y to get reconstructed luma (matching what decoder will produce)
            self.quantize.dispatch_adaptive(
                ctx, &mut cmd, quant_dests[0], &bufs.plane_a,
                padded_pixels as u32, config.quantization_step, config.dead_zone,
                false, padded_w, padded_h, config.tile_size, config.wavelet_levels,
                &weights_luma, wm_param, 0.0,
            );
            cmd.copy_buffer_to_buffer(&bufs.plane_a, 0, &bufs.recon_y, 0, plane_size);

            // Co plane: CfL alpha → forward predict → quantize
            cmd.copy_buffer_to_buffer(plane_bufs[1], 0, &bufs.plane_c, 0, plane_size);
            self.cfl_alpha.dispatch(
                ctx, &mut cmd, &bufs.recon_y, &bufs.plane_c,
                &bufs.raw_alpha, &bufs.dq_alpha,
                padded_w, padded_h, config.tile_size, config.wavelet_levels,
            );
            self.cfl_forward.dispatch(
                ctx, &mut cmd, &bufs.plane_c, &bufs.recon_y, &bufs.dq_alpha, &bufs.plane_a,
                padded_pixels as u32, padded_w, padded_h, config.tile_size, config.wavelet_levels,
            );
            self.quantize.dispatch_adaptive(
                ctx, &mut cmd, &bufs.plane_a, quant_dests[1],
                padded_pixels as u32, config.quantization_step, config.dead_zone,
                true, padded_w, padded_h, config.tile_size, config.wavelet_levels,
                &weights_chroma, wm_param, 0.0,
            );
            cmd.copy_buffer_to_buffer(&bufs.raw_alpha, 0, &alpha_staging[0], 0, alpha_bytes);

            // Cg plane: CfL alpha → forward predict → quantize
            cmd.copy_buffer_to_buffer(plane_bufs[2], 0, &bufs.plane_c, 0, plane_size);
            self.cfl_alpha.dispatch(
                ctx, &mut cmd, &bufs.recon_y, &bufs.plane_c,
                &bufs.raw_alpha, &bufs.dq_alpha,
                padded_w, padded_h, config.tile_size, config.wavelet_levels,
            );
            self.cfl_forward.dispatch(
                ctx, &mut cmd, &bufs.plane_c, &bufs.recon_y, &bufs.dq_alpha, &bufs.plane_a,
                padded_pixels as u32, padded_w, padded_h, config.tile_size, config.wavelet_levels,
            );
            self.quantize.dispatch_adaptive(
                ctx, &mut cmd, &bufs.plane_a, quant_dests[2],
                padded_pixels as u32, config.quantization_step, config.dead_zone,
                true, padded_w, padded_h, config.tile_size, config.wavelet_levels,
                &weights_chroma, wm_param, 0.0,
            );
            cmd.copy_buffer_to_buffer(&bufs.raw_alpha, 0, &alpha_staging[1], 0, alpha_bytes);
        } else {
            // Non-CfL path: quantize all 3 planes directly
            for p in 0..3 {
                let weights = if p == 0 { &weights_luma } else { &weights_chroma };
                cmd.copy_buffer_to_buffer(plane_bufs[p], 0, &bufs.plane_c, 0, plane_size);
                self.quantize.dispatch_adaptive(
                    ctx, &mut cmd, &bufs.plane_c, quant_dests[p],
                    padded_pixels as u32, config.quantization_step, config.dead_zone,
                    true, padded_w, padded_h, config.tile_size, config.wavelet_levels,
                    weights, wm_param, 0.0,
                );
            }
        }

        self.gpu_rice_encoder.dispatch_3planes_to_cmd(
            ctx,
            &mut cmd,
            quant_dests,
            info,
            config.wavelet_levels,
        );

        ctx.queue.submit(Some(cmd.finish()));

        let rice_tiles = self.gpu_rice_encoder.finish_3planes_readback(
            ctx,
            info,
            config.wavelet_levels,
        );

        let cfl_alphas = if config.cfl_enabled {
            let mut cfl_alphas_all: Vec<i16> = Vec::new();
            let (tx, rx) = std::sync::mpsc::channel();
            for stg in &alpha_staging {
                let tx = tx.clone();
                stg.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).unwrap();
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
            Some(CflAlphas {
                alphas: cfl_alphas_all,
                num_subbands: nsb,
            })
        } else {
            None
        };

        CompressedFrame {
            info: *info,
            config: config.clone(),
            entropy: EntropyData::Rice(rice_tiles),
            cfl_alphas,
            weight_map: wm_data,
            frame_type: FrameType::Intra,
            motion_field: None,
            intra_modes: None,
            residual_stats: None,
            residual_stats_co: None,
            residual_stats_cg: None,
        }
    }

    /// Map temporal highpass energy to an adaptive qstep multiplier.
    ///
    /// Low energy (static content) → high mul (aggressive quantization, tiny highpass).
    /// High energy (motion) → low mul (preserve temporal detail).
    /// Returns a value in `[1.0, max_mul]`.
    ///
    /// The weight_map in the quantize shader multiplies step_size, so mul > 1.0
    /// means coarser quantization (fewer bits). We never go below 1.0 because
    /// highpass should never be finer than lowpass.
    ///
    /// Calibrated for real temporal highpass energy values (typically 0-20 for
    /// 1080p content at q=75). Uses log-space interpolation between thresholds.
    fn map_energy_to_mul(energy: f64, max_mul: f32) -> f32 {
        // Below low_thresh: near-zero energy → max_mul (aggressive)
        // Above high_thresh: high energy → 1.0 (preserve detail)
        // Between: log-linear interpolation
        let low_thresh: f64 = 0.5;
        let high_thresh: f64 = 10.0;

        if energy <= low_thresh {
            return max_mul;
        }
        if energy >= high_thresh {
            return 1.0;
        }

        // Log-linear interpolation: t=0 at low_thresh, t=1 at high_thresh
        let t = ((energy / low_thresh).ln() / (high_thresh / low_thresh).ln()) as f32;
        max_mul + t * (1.0 - max_mul)
    }

    /// Build an all-zero `CompressedFrame` for a highpass frame whose coefficients
    /// all quantize to zero. Avoids dispatching any GPU work.
    fn make_zero_compressed_frame(config: &CodecConfig, info: &FrameInfo) -> CompressedFrame {
        let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;
        let num_groups = (config.wavelet_levels * 2).max(1) as usize;
        let all_skip_mask = if num_groups >= 8 {
            0xFFu8
        } else {
            (1u8 << num_groups) - 1
        };
        let rice_tiles: Vec<crate::encoder::rice::RiceTile> = (0..num_tiles * 3)
            .map(|_| crate::encoder::rice::RiceTile {
                num_coefficients: info.tile_size * info.tile_size,
                tile_size: info.tile_size,
                num_levels: config.wavelet_levels,
                num_groups: num_groups as u32,
                k_values: vec![0u8; num_groups],
                k_zrl_values: vec![0u8; num_groups],
                skip_bitmap: all_skip_mask,
                stream_lengths: vec![0u32; crate::encoder::rice::RICE_STREAMS_PER_TILE],
                stream_data: Vec::new(),
            })
            .collect();
        CompressedFrame {
            info: *info,
            config: config.clone(),
            entropy: EntropyData::Rice(rice_tiles),
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

    /// Compute per-tile adaptive qstep multipliers for temporal highpass.
    ///
    /// Reads back the Y plane from GPU, computes per-tile mean_abs of temporal
    /// highpass coefficients. Maps each tile's energy to a qstep multiplier via
    /// `map_energy_to_mul`: low energy (static) → high mul (coarse quant),
    /// high energy (motion) → low mul (preserve detail).
    ///
    /// Returns `(muls, global_max_abs)`:
    /// - `muls`: per-tile multipliers in `[1.0, max_mul]`.
    /// - `global_max_abs`: maximum absolute Y coefficient. Compare against
    ///   `dead_zone * qstep` to detect all-zero frames without a second readback.
    fn compute_temporal_tile_muls(
        ctx: &GpuContext,
        y_plane_buf: &wgpu::Buffer,
        padded_w: u32,
        padded_h: u32,
        tile_size: u32,
        max_mul: f32,
    ) -> (Vec<f32>, f32) {
        let padded_pixels = (padded_w * padded_h) as usize;
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;

        // Read back Y plane
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tw_aq_staging"),
            size: plane_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tw_aq_readback"),
            });
        cmd.copy_buffer_to_buffer(y_plane_buf, 0, &staging, 0, plane_size);
        ctx.queue.submit(Some(cmd.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let coeffs: &[f32] = bytemuck::cast_slice(&data);

        let tiles_x = padded_w / tile_size;
        let tiles_y = padded_h / tile_size;
        let num_tiles = (tiles_x * tiles_y) as usize;
        let ts = tile_size as usize;
        let tile_pixel_count = (ts * ts) as f64;

        // Compute per-tile mean_abs and global max_abs in a single pass.
        // max_abs is used to detect fully-quantized-to-zero frames without a second readback.
        let mut tile_mean_abs = vec![0.0f64; num_tiles];
        let mut global_max_abs = 0.0f32;
        for ty in 0..tiles_y as usize {
            for tx_idx in 0..tiles_x as usize {
                let tile_idx = ty * tiles_x as usize + tx_idx;
                let mut sum_abs = 0.0f64;
                for row in 0..ts {
                    let y_pos = ty * ts + row;
                    let x_base = tx_idx * ts;
                    for col in 0..ts {
                        let idx = y_pos * padded_w as usize + x_base + col;
                        let abs_val = coeffs[idx].abs();
                        sum_abs += abs_val as f64;
                        if abs_val > global_max_abs {
                            global_max_abs = abs_val;
                        }
                    }
                }
                tile_mean_abs[tile_idx] = sum_abs / tile_pixel_count;
            }
        }

        drop(data);
        staging.unmap();

        // Map per-tile energy → qstep multiplier
        let muls: Vec<f32> = tile_mean_abs
            .iter()
            .map(|&energy| Self::map_energy_to_mul(energy, max_mul))
            .collect();

        if std::env::var("GNC_TW_DIAG").is_ok() {
            let mut sorted_e: Vec<f64> = tile_mean_abs.clone();
            sorted_e.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sorted_e.len();
            let pct_e = |p: usize| sorted_e[(p * n / 100).min(n - 1)];
            let mut sorted_m: Vec<f32> = muls.clone();
            sorted_m.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pct_m = |p: usize| sorted_m[(p * n / 100).min(n - 1)];
            eprintln!(
                "    tile energy: min={:.4} p25={:.4} p50={:.4} p75={:.4} max={:.4}  max_abs_y={:.4}",
                sorted_e[0], pct_e(25), pct_e(50), pct_e(75), sorted_e[n - 1],
                global_max_abs,
            );
            eprintln!(
                "    tile mul:    min={:.3} p25={:.3} p50={:.3} p75={:.3} max={:.3}  (max_mul={:.2})",
                sorted_m[0], pct_m(25), pct_m(50), pct_m(75), sorted_m[n - 1], max_mul,
            );
        }

        (muls, global_max_abs)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_energy_to_mul_bounds() {
        let max_mul = 2.0f32;
        // Very low energy → max_mul (aggressive quantization)
        let low = EncoderPipeline::map_energy_to_mul(0.001, max_mul);
        assert!((low - max_mul).abs() < 0.01, "low energy mul={low}");
        // Very high energy → 1.0 (preserve detail, never finer than lowpass)
        let high = EncoderPipeline::map_energy_to_mul(100.0, max_mul);
        assert!((high - 1.0).abs() < 0.01, "high energy mul={high}");
        // Medium energy → between 1.0 and max_mul
        let mid = EncoderPipeline::map_energy_to_mul(2.0, max_mul);
        assert!(mid > 1.0 && mid < max_mul, "mid energy mul={mid}");
    }

    #[test]
    fn test_map_energy_to_mul_monotonic() {
        let max_mul = 2.0f32;
        let energies = [0.01, 0.1, 0.5, 1.0, 5.0, 50.0];
        let muls: Vec<f32> = energies
            .iter()
            .map(|&e| EncoderPipeline::map_energy_to_mul(e, max_mul))
            .collect();
        // Higher energy → lower mul (monotonically decreasing)
        for i in 1..muls.len() {
            assert!(
                muls[i] <= muls[i - 1],
                "not monotonic: energy {:.2} → {:.3}, energy {:.2} → {:.3}",
                energies[i - 1], muls[i - 1], energies[i], muls[i],
            );
        }
    }

    #[test]
    fn test_map_energy_to_mul_respects_max() {
        // Different max_mul ceilings
        for max in [1.5f32, 2.0, 3.0] {
            let mul = EncoderPipeline::map_energy_to_mul(0.0, max);
            assert!(
                (mul - max).abs() < 0.01,
                "zero energy should give max_mul={max}, got {mul}"
            );
            let mul_high = EncoderPipeline::map_energy_to_mul(1000.0, max);
            assert!(
                (mul_high - 1.0).abs() < 0.01,
                "high energy should floor at 1.0, got {mul_high}"
            );
        }
    }

    #[test]
    fn test_map_energy_to_mul_never_below_one() {
        // Mul should never go below 1.0 — highpass should never be finer than lowpass
        for max in [1.5f32, 2.0, 3.0, 5.0] {
            for e in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0] {
                let mul = EncoderPipeline::map_energy_to_mul(e, max);
                assert!(
                    mul >= 1.0 - 0.001,
                    "mul={mul} < 1.0 at energy={e}, max_mul={max}"
                );
            }
        }
    }
}
