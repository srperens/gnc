use wgpu;

use super::buffer_cache::CachedBuffers;
use super::pipeline::DecoderPipeline;
use crate::encoder::adaptive::{self, AQ_LL_BLOCK_SIZE};
use crate::{CompressedFrame, EntropyData, FrameType, GpuContext};

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
        let is_pframe = frame.frame_type == FrameType::Predicted;
        let tiles_per_plane = info.tiles_x() as usize * info.tiles_y() as usize;
        let blocks_per_tile_side = info.tile_size as usize / 32;
        let blocks_per_tile = blocks_per_tile_side * blocks_per_tile_side;

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

        // Per-plane: entropy decode → dequantize → (CfL inverse predict) → inverse wavelet → copy to result buffer
        for p in 0..3 {
            if bufs.ctx_adaptive_decode {
                // Context-adaptive tiles were CPU-decoded in prepare_frame_data;
                // copy the already-decoded coefficients into scratch_a.
                cmd.copy_buffer_to_buffer(
                    &bufs.cpu_decoded_planes[p],
                    0,
                    &bufs.scratch_a,
                    0,
                    plane_size,
                );
            } else {
                match &frame.entropy {
                    EntropyData::Rice(_) => {
                        // Rice uses CPU decode path; data already in cpu_decoded_planes
                        unreachable!("Rice frames use ctx_adaptive_decode path");
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
                            tiles_per_plane as u32,
                        );
                    }
                    EntropyData::Bitplane(_) => {
                        let total_blocks = (tiles_per_plane * blocks_per_tile) as u32;
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
                }
            }

            let weights = if p == 0 {
                &weights_luma
            } else {
                &weights_chroma
            };
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
            self.quantize.dispatch_adaptive(
                ctx,
                &mut cmd,
                &bufs.scratch_a,
                &bufs.scratch_b,
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
                cmd.copy_buffer_to_buffer(
                    &bufs.scratch_b,
                    0,
                    &bufs.y_ref_wavelet_buf,
                    0,
                    plane_size,
                );
            }

            if p > 0 && has_cfl {
                // CfL inverse prediction: scratch_b (dequantized residual) + alpha * y_ref → scratch_c
                let plane_alpha_offset = (p - 1) * cfl_alphas_per_plane;
                let plane_alpha_byte_offset =
                    (plane_alpha_offset * std::mem::size_of::<f32>()) as u64;
                let plane_alpha_byte_size =
                    (cfl_alphas_per_plane * std::mem::size_of::<f32>()) as u64;

                // Copy this plane's alphas from the full alpha buffer to the per-plane buffer
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
                    &bufs.scratch_b,               // dequantized residual
                    &bufs.y_ref_wavelet_buf,       // reconstructed Y wavelet
                    &bufs.plane_alpha_bufs[p - 1], // per-tile per-subband alphas
                    &bufs.scratch_c,               // output: reconstructed chroma wavelet
                    padded_pixels as u32,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                );

                // Inverse wavelet: scratch_c → (temp scratch_b) → scratch_a
                self.transform.inverse(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_c,
                    &bufs.scratch_b,
                    &bufs.scratch_a,
                    info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );
            } else {
                // Standard path: inverse wavelet from scratch_b
                self.transform.inverse(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_b,
                    &bufs.scratch_c,
                    &bufs.scratch_a,
                    info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );
            }

            let is_bframe = frame.frame_type == FrameType::Bidirectional;
            if is_bframe {
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
                );
            } else if is_pframe {
                // P-frame: scratch_a has residual, add MC prediction from reference
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
                );
            } else {
                // I-frame: scratch_a has reconstructed spatial data
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
            if !is_bframe {
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
