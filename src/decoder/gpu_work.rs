use wgpu;

use super::buffer_cache::CachedBuffers;
use super::pipeline::DecoderPipeline;
use crate::encoder::adaptive::{self, AQ_LL_BLOCK_SIZE};
use crate::encoder::block_transform::BlockTransformType;
use crate::{
    ChromaFormat, CompressedFrame, EntropyData, FrameInfo, FrameType, GpuContext, TransformType,
};

impl DecoderPipeline {
    /// Encode GPU commands for the full decode pipeline up to and including crop.
    /// All buffers are read from CachedBuffers (written by prepare_frame_data).
    pub(super) fn encode_gpu_work(
        &self,
        ctx: &GpuContext,
        frame: &CompressedFrame,
        bufs: &CachedBuffers,
    ) -> wgpu::CommandEncoder {
        let info = &frame.info;
        let config = &frame.config;
        let padded_w = info.padded_width();
        let padded_h = info.padded_height();
        let padded_pixels = (padded_w * padded_h) as usize;
        let luma_tiles_per_plane = info.luma_tiles_per_plane();
        let blocks_per_tile_side = info.tile_size as usize / 32;
        let blocks_per_tile = blocks_per_tile_side * blocks_per_tile_side;

        // Per-plane info for non-444 chroma decode. chroma_info uses Yuv444 so the
        // wavelet sees it as full-resolution within the subsampled dimensions.
        let chroma_info_storage;
        let (chroma_tiles_per_plane, plane_info_arr): (usize, [&FrameInfo; 3]) =
            if info.chroma_format == ChromaFormat::Yuv444 {
                (luma_tiles_per_plane, [info, info, info])
            } else {
                chroma_info_storage = info.make_chroma_info();
                (
                    info.chroma_tiles_per_plane(),
                    [info, &chroma_info_storage, &chroma_info_storage],
                )
            };
        // tiles_per_plane used for luma (p==0) and in the for loop via plane_tiles_per_plane(p)
        let tiles_per_plane = luma_tiles_per_plane;

        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let w = info.width;
        let h = info.height;
        let output_pixels = w * h;

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("decode_full"),
            });

        // CfL metadata
        let has_cfl = frame.cfl_alphas.is_some();
        let cfl_alphas_per_plane = if has_cfl {
            let cfl_data = frame.cfl_alphas.as_ref().unwrap();
            tiles_per_plane * cfl_data.num_subbands as usize
        } else {
            0
        };

        // 4:2:0 chroma-domain MC: pre-scale luma MVs → chroma MVs (shared by both chroma planes).
        // Must run once before the plane loop because both p=1 and p=2 use mv_chroma_buf.
        let is_pframe = frame.frame_type == FrameType::Predicted;
        let is_bframe = frame.frame_type == FrameType::Bidirectional;
        let is_420 = info.chroma_format == ChromaFormat::Yuv420;
        if (is_pframe || is_bframe) && is_420 {
            let chroma_shift_x = info.chroma_format.horiz_shift();
            let chroma_shift_y = info.chroma_format.vert_shift();
            let split_blocks_x = padded_w / crate::encoder::motion::ME_SPLIT_BLOCK_SIZE;
            let split_blocks_y = padded_h / crate::encoder::motion::ME_SPLIT_BLOCK_SIZE;
            let total_blocks = split_blocks_x * split_blocks_y;

            // Scale forward MVs to chroma dims
            self.motion.dispatch_mv_scale(
                ctx,
                &mut cmd,
                &bufs.mv_buf,
                &bufs.mv_chroma_buf,
                total_blocks,
                chroma_shift_x,
                chroma_shift_y,
            );

            // Scale backward MVs to chroma dims (for B-frame)
            if is_bframe {
                self.motion.dispatch_mv_scale(
                    ctx,
                    &mut cmd,
                    &bufs.bwd_mv_buf,
                    &bufs.bwd_mv_chroma_buf,
                    total_blocks,
                    chroma_shift_x,
                    chroma_shift_y,
                );
            }
        }

        // Per-plane: entropy decode → dequantize → (CfL inverse predict) → inverse wavelet → copy to result buffer
        // p is used for many array accesses beyond plane_info_arr — keep the index loop.
        #[allow(clippy::needless_range_loop)]
        for p in 0..3 {
            // Per-plane dimensions: chroma planes are smaller for non-444 formats.
            let p_info = plane_info_arr[p];
            let p_tiles = if p == 0 {
                luma_tiles_per_plane
            } else {
                chroma_tiles_per_plane
            };
            let p_padded_w = p_info.padded_width();
            let p_padded_h = p_info.padded_height();
            let p_padded_pixels = (p_padded_w * p_padded_h) as usize;
            let p_plane_size = (p_padded_pixels * std::mem::size_of::<f32>()) as u64;

            if bufs.ctx_adaptive_decode {
                // Context-adaptive tiles were CPU-decoded in prepare_frame_data;
                // copy the already-decoded coefficients into scratch_a.
                cmd.copy_buffer_to_buffer(
                    &bufs.cpu_decoded_planes[p],
                    0,
                    &bufs.scratch_a,
                    0,
                    p_plane_size,
                );
            } else {
                match &frame.entropy {
                    EntropyData::Rice(_) => {
                        // GPU Rice decode: 256 parallel streams per tile
                        self.rice_decoder.dispatch_decode(
                            ctx,
                            &mut cmd,
                            &bufs.entropy_params[p],
                            &bufs.entropy_tile_info[p],
                            &bufs.entropy_var_a[p],
                            &bufs.entropy_var_b[p],
                            &bufs.scratch_a,
                            p_tiles as u32,
                        );
                    }
                    EntropyData::Rans(_) | EntropyData::SubbandRans(_) => {
                        self.rans_decoder.dispatch_decode(
                            ctx,
                            &mut cmd,
                            &bufs.entropy_params[p],
                            &bufs.entropy_tile_info[p],
                            &bufs.entropy_var_a[p],
                            &bufs.entropy_var_b[p],
                            &bufs.scratch_a,
                            p_tiles as u32,
                        );
                    }
                    EntropyData::Bitplane(_) => {
                        let total_blocks = (p_tiles * blocks_per_tile) as u32;
                        self.bitplane_decoder.dispatch_decode(
                            ctx,
                            &mut cmd,
                            &bufs.entropy_params[p],
                            &bufs.entropy_tile_info[p],
                            &bufs.entropy_var_a[p],
                            &bufs.entropy_var_b[p],
                            &bufs.scratch_a,
                            total_blocks,
                        );
                    }
                    EntropyData::Huffman(_) => {
                        // GPU Huffman decode: 256 parallel streams per tile
                        // Bindings: params, decode_table (tile_info), k_zrl (cpu_decoded_planes),
                        //           stream_data (var_a), stream_offsets (var_b), output (scratch_a)
                        self.huffman_decoder.dispatch_decode(
                            ctx,
                            &mut cmd,
                            &bufs.entropy_params[p],
                            &bufs.entropy_tile_info[p],  // decode_table
                            &bufs.cpu_decoded_planes[p], // k_zrl
                            &bufs.entropy_var_a[p],      // stream_data
                            &bufs.entropy_var_b[p],      // stream_offsets
                            &bufs.scratch_a,
                            p_tiles as u32,
                        );
                    }
                }
            }

            if config.transform_type == TransformType::BlockDCT8 {
                // ---- Block DCT decode path ----
                // scratch_a has quantized DCT coefficients from entropy decode.
                // Dequantize (flat, no subband weights) → inverse DCT-8×8 → spatial pixels.
                let uniform_weights = [1.0f32; 16];
                self.quantize.dispatch_adaptive(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_a,
                    &bufs.scratch_b,
                    padded_pixels as u32,
                    config.quantization_step,
                    config.dead_zone,
                    false, // dequantize
                    padded_w,
                    padded_h,
                    config.tile_size,
                    0, // wavelet_levels=0 for flat data
                    &uniform_weights,
                    None, // no AQ weight map
                    config.dct_freq_strength,
                );

                // Inverse DCT-8×8: scratch_b (dequantized coefficients) → scratch_a (spatial pixels)
                self.block_transform.dispatch(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_b,
                    &bufs.scratch_a,
                    padded_w,
                    padded_h,
                    false, // inverse
                    BlockTransformType::DCT8,
                );
            } else {
                // ---- Wavelet decode path ----
                let weights = if p == 0 {
                    &weights_luma
                } else {
                    &weights_chroma
                };
                // AQ weight map: only applies to luma (chroma uses no AQ weight map)
                let wm_param = if frame.weight_map.is_some() && p == 0 {
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
                self.quantize.dispatch_adaptive(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_a,
                    &bufs.scratch_b,
                    p_padded_pixels as u32,
                    config.quantization_step,
                    config.dead_zone,
                    false,
                    p_padded_w,
                    p_padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                    weights,
                    wm_param,
                    0.0, // no DCT freq weighting for wavelet path
                );

                if p == 0 && has_cfl {
                    cmd.copy_buffer_to_buffer(
                        &bufs.scratch_b,
                        0,
                        &bufs.y_ref_wavelet_buf,
                        0,
                        plane_size, // CfL always 444 — luma plane size
                    );
                }

                if p > 0 && has_cfl {
                    // CfL only runs in 444 mode, so padded_w/h == p_padded_w/h here.
                    let plane_alpha_offset = (p - 1) * cfl_alphas_per_plane;
                    let plane_alpha_byte_offset =
                        (plane_alpha_offset * std::mem::size_of::<f32>()) as u64;
                    let plane_alpha_byte_size =
                        (cfl_alphas_per_plane * std::mem::size_of::<f32>()) as u64;

                    cmd.copy_buffer_to_buffer(
                        &bufs.cfl_alpha_buf,
                        plane_alpha_byte_offset,
                        &bufs.plane_alpha_bufs[p - 1],
                        0,
                        plane_alpha_byte_size,
                    );

                    self.cfl_predictor.dispatch_inverse(
                        ctx,
                        &mut cmd,
                        &bufs.scratch_b,
                        &bufs.y_ref_wavelet_buf,
                        &bufs.plane_alpha_bufs[p - 1],
                        &bufs.scratch_c,
                        padded_pixels as u32,
                        padded_w,
                        padded_h,
                        config.tile_size,
                        config.wavelet_levels,
                    );

                    self.transform.inverse(
                        ctx,
                        &mut cmd,
                        &bufs.scratch_c,
                        &bufs.scratch_b,
                        &bufs.scratch_a,
                        p_info,
                        config.wavelet_levels,
                        config.wavelet_type,
                        p, // plane_idx: avoids param slot collision in non-444 mode
                    );
                } else {
                    self.transform.inverse(
                        ctx,
                        &mut cmd,
                        &bufs.scratch_b,
                        &bufs.scratch_c,
                        &bufs.scratch_a,
                        p_info,
                        config.wavelet_levels,
                        config.wavelet_type,
                        p, // plane_idx: avoids param slot collision in non-444 mode
                    );
                }
            } // end wavelet decode path

            // Intra reconstruction: Y plane only (p==0), after inverse wavelet
            if p == 0 && frame.intra_modes.is_some() {
                // scratch_a has wavelet-decoded residual; reconstruct spatial pixels
                let intra_ts = crate::encoder::intra::INTRA_TILE_SIZE;
                self.intra.inverse(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_a, // residual in
                    &bufs.scratch_b, // reconstructed out
                    &bufs.intra_modes_buf,
                    padded_w,
                    padded_h,
                    intra_ts,
                );
                // Copy reconstructed back to scratch_a for I/P/B branching
                cmd.copy_buffer_to_buffer(&bufs.scratch_b, 0, &bufs.scratch_a, 0, plane_size);
            }

            // For non-444 chroma (p>0): upsample scratch_a from chroma size to luma size.
            // Exception: 4:2:0 P-frame chroma uses chroma-domain MC (see below).
            // For 444: scratch_a is already at luma size.
            let is_non444_chroma = p > 0 && info.chroma_format != ChromaFormat::Yuv444;
            // 4:2:0 P-frame chroma: perform MC at chroma dims to avoid NN-upsample HF artifacts.
            // The encoder stores gpu_ref_planes[p] at luma dims (NN-upsampled), so
            // box_filter(ref) recovers the correct chroma-resolution reference.
            let is_420_pframe_chroma = is_420 && is_pframe && p > 0;
            if is_non444_chroma && !is_420_pframe_chroma {
                // 4:2:2 or I/B-frame non-444: upsample chroma residual to luma dims first.
                let up_buf = if p == 1 {
                    &bufs.co_plane_up
                } else {
                    &bufs.cg_plane_up
                };
                self.chroma_up.dispatch_upsample(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_a,
                    up_buf,
                    p_padded_w,
                    p_padded_h,
                    padded_w,
                    padded_h,
                    info.chroma_format.horiz_shift(),
                    info.chroma_format.vert_shift(),
                );
                // After upsample, up_buf has luma-sized chroma data.
                // Copy to scratch_a so the I/P/B path below works uniformly.
                cmd.copy_buffer_to_buffer(up_buf, 0, &bufs.scratch_a, 0, plane_size);
            }

            if is_420_pframe_chroma {
                // 4:2:0 P-frame chroma-domain MC:
                //   scratch_a        = chroma-dims residual (from wavelet decode)
                //   reference_planes[p] = luma-sized NN-upsampled reconstructed chroma
                //   box_filter(reference_planes[p]) = chroma-dims reference (avoids HF artifacts)
                //
                // Step 1: box-filter luma-sized reference → chroma dims.
                // Re-use co_plane_up/cg_plane_up as scratch for the chroma-dims reference.
                let chroma_ref_scratch = if p == 1 {
                    &bufs.co_plane_up
                } else {
                    &bufs.cg_plane_up
                };
                self.chroma_down.dispatch(
                    ctx,
                    &mut cmd,
                    &bufs.reference_planes[p],
                    chroma_ref_scratch,
                    padded_w,
                    padded_h,
                    info.chroma_format.horiz_shift(),
                    info.chroma_format.vert_shift(),
                    p_padded_w,
                    p_padded_h,
                );
                // Step 2: inverse MC at chroma dims: residual + warp(chroma_ref) → chroma_recon_buf.
                // chroma_recon_buf is luma-sized; only the first chroma_pixels elements are written.
                self.motion.compensate(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_a,
                    chroma_ref_scratch,
                    &bufs.mv_chroma_buf,
                    &bufs.chroma_recon_buf,
                    p_padded_w,
                    p_padded_h,
                    false, // inverse: recon = residual + predicted
                    bufs.mc_block_size / 2,
                );
                // Step 3: NN-upsample chroma_recon_buf → plane_results[p] (luma dims).
                self.chroma_up.dispatch_upsample(
                    ctx,
                    &mut cmd,
                    &bufs.chroma_recon_buf,
                    &bufs.plane_results[p],
                    p_padded_w,
                    p_padded_h,
                    padded_w,
                    padded_h,
                    info.chroma_format.horiz_shift(),
                    info.chroma_format.vert_shift(),
                );
                // Step 4: store reconstructed luma-sized chroma → reference_planes[p] for next frame.
                cmd.copy_buffer_to_buffer(
                    &bufs.plane_results[p],
                    0,
                    &bufs.reference_planes[p],
                    0,
                    plane_size,
                );
            } else if is_bframe && p > 0 && is_420 {
                // 4:2.0 B-frame chroma-domain bidir MC:
                //   Same logic as P-frame but with both forward and backward references.
                //   Step 1: box-filter fwd ref → chroma dims
                let fwd_chroma_ref = &bufs.co_plane_up;
                self.chroma_down.dispatch(
                    ctx,
                    &mut cmd,
                    &bufs.reference_planes[p],
                    fwd_chroma_ref,
                    padded_w,
                    padded_h,
                    info.chroma_format.horiz_shift(),
                    info.chroma_format.vert_shift(),
                    p_padded_w,
                    p_padded_h,
                );
                // Step 2: box-filter bwd ref → chroma dims (reuse cg_plane_up as scratch)
                let bwd_chroma_ref = &bufs.cg_plane_up;
                self.chroma_down.dispatch(
                    ctx,
                    &mut cmd,
                    &bufs.bwd_reference_planes[p],
                    bwd_chroma_ref,
                    padded_w,
                    padded_h,
                    info.chroma_format.horiz_shift(),
                    info.chroma_format.vert_shift(),
                    p_padded_w,
                    p_padded_h,
                );
                // Step 3: bidir MC at chroma dims using scaled chroma MVs
                // - scratch_a has residual at chroma dims
                // - fwd_chroma_ref has fwd ref at chroma dims
                // - bwd_chroma_ref has bwd ref at chroma dims
                // - MVs: mv_chroma_buf, bwd_mv_chroma_buf (scaled)
                // Output → chroma_recon_buf (chroma dims), then upsample to plane_results[p].
                self.motion.compensate_bidir_chroma_cached(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_a,
                    fwd_chroma_ref,
                    bwd_chroma_ref,
                    &bufs.mv_chroma_buf,
                    &bufs.bwd_mv_chroma_buf,
                    &bufs.block_modes_buf,
                    &bufs.chroma_recon_buf,
                    p_padded_w,
                    p_padded_h,
                    false, // inverse: reconstruct = residual + pred
                );
                // Step 4: NN-upsample chroma_recon_buf → plane_results[p] (luma dims).
                self.chroma_up.dispatch_upsample(
                    ctx,
                    &mut cmd,
                    &bufs.chroma_recon_buf,
                    &bufs.plane_results[p],
                    p_padded_w,
                    p_padded_h,
                    padded_w,
                    padded_h,
                    info.chroma_format.horiz_shift(),
                    info.chroma_format.vert_shift(),
                );
            } else if is_bframe {
                // B-frame: use bidirectional compensation with both references
                self.motion.compensate_bidir(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_a,               // decoded residual
                    &bufs.reference_planes[p],     // forward ref
                    &bufs.bwd_reference_planes[p], // backward ref
                    &bufs.mv_buf,
                    &bufs.bwd_mv_buf,
                    &bufs.block_modes_buf,
                    &bufs.plane_results[p],
                    padded_w,
                    padded_h,
                    false, // inverse: recon = residual + predicted
                    bufs.mc_block_size,
                );
            } else if is_pframe {
                // P-frame luma or 4:4:4/4:2:2 chroma: scratch_a has residual at luma dims,
                // add MC prediction from reference.
                self.motion.compensate(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_a,
                    &bufs.reference_planes[p],
                    &bufs.mv_buf,
                    &bufs.plane_results[p],
                    padded_w,
                    padded_h,
                    false, // inverse: recon = residual + predicted
                    bufs.mc_block_size,
                );
            } else {
                // I-frame: scratch_a has reconstructed spatial data (luma-sized after any upsample)
                cmd.copy_buffer_to_buffer(
                    &bufs.scratch_a,
                    0,
                    &bufs.plane_results[p],
                    0,
                    plane_size,
                );
            }

            // Copy reconstructed plane to reference buffer for next frame.
            // B-frames don't update references (they are non-reference frames).
            // 4:2:0 P-frame chroma already updated reference_planes[p] above (step 4).
            if !is_bframe && !is_420_pframe_chroma {
                cmd.copy_buffer_to_buffer(
                    &bufs.plane_results[p],
                    0,
                    &bufs.reference_planes[p],
                    0,
                    plane_size,
                );
            }
        }

        // GPU interleave: 3 planes → interleaved YCoCg
        self.interleaver.dispatch(
            ctx,
            &mut cmd,
            &bufs.plane_results[0],
            &bufs.plane_results[1],
            &bufs.plane_results[2],
            &bufs.ycocg_buf,
            padded_pixels as u32,
        );

        // Inverse color (YCoCg-R → RGB)
        self.color.dispatch(
            ctx,
            &mut cmd,
            &bufs.ycocg_buf,
            &bufs.rgb_out_buf,
            padded_w,
            padded_h,
            false,
            config.is_lossless(),
        );

        // GPU crop: padded RGB → compact cropped output
        {
            let crop_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("crop_bg"),
                layout: &self.crop_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bufs.crop_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bufs.rgb_out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bufs.cropped_buf.as_entire_binding(),
                    },
                ],
            });

            let workgroups = output_pixels.div_ceil(256);
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("crop_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.crop_pipeline);
            pass.set_bind_group(0, &crop_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        cmd
    }
}
