//! P-frame decode checkpoint instrumentation for divergence debugging.
//!
//! Runs the Y-plane decode pipeline step by step with GPU readbacks after each stage,
//! allowing comparison between encoder local decode and decoder at each checkpoint.

use super::pipeline::DecoderPipeline;
use crate::gpu_util::read_buffer_f32;
use crate::{CompressedFrame, EntropyData, FrameType, GpuContext};

/// Intermediate values at each stage of the P-frame Y-plane decode pipeline.
pub struct PFrameCheckpoints {
    /// Checkpoint 1: MC prediction signal (prediction = reconstructed - residual)
    pub mc_prediction: Vec<f32>,
    /// Checkpoint 3: Quantized coefficients from entropy decode
    pub quantized: Vec<f32>,
    /// Checkpoint 4: Dequantized wavelet coefficients
    pub dequantized: Vec<f32>,
    /// Checkpoint 5: Spatial residual after IDWT
    pub spatial_residual: Vec<f32>,
    /// Checkpoint 6: Reconstructed pixels after MC inverse (residual + prediction)
    pub reconstructed: Vec<f32>,
}

/// Diff statistics for a single checkpoint.
pub struct DiffStats {
    pub max_diff: f32,
    pub mean_diff: f32,
    pub nonzero_count: usize,
    pub total_count: usize,
}

pub fn compute_diff(a: &[f32], b: &[f32], count: usize) -> DiffStats {
    let n = count.min(a.len()).min(b.len());
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f64 = 0.0;
    let mut nonzero = 0usize;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
        sum_diff += d as f64;
        if d > 0.001 {
            nonzero += 1;
        }
    }
    DiffStats {
        max_diff,
        mean_diff: (sum_diff / n as f64) as f32,
        nonzero_count: nonzero,
        total_count: n,
    }
}

impl DecoderPipeline {
    /// Decode a P-frame's Y plane step by step with GPU readbacks at each stage.
    ///
    /// Prerequisite: I-frame must already be decoded (reference planes populated).
    /// This method updates reference_planes[0] at the end, so subsequent frames
    /// can still use the normal decode path.
    ///
    /// Returns intermediate values at 5 checkpoints (checkpoint 2 "After DWT" is
    /// encoder-only and not produced here).
    pub fn decode_pframe_checkpoints(
        &self,
        ctx: &GpuContext,
        frame: &CompressedFrame,
    ) -> PFrameCheckpoints {
        assert_eq!(
            frame.frame_type,
            FrameType::Predicted,
            "decode_pframe_checkpoints only works for P-frames"
        );

        let info = &frame.info;
        let config = &frame.config;
        let padded_w = info.padded_width();
        let padded_h = info.padded_height();
        let padded_pixels = (padded_w * padded_h) as usize;
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let tiles_per_plane = info.tiles_x() as usize * info.tiles_y() as usize;
        let blocks_per_tile_side = info.tile_size as usize / 32;
        let blocks_per_tile = blocks_per_tile_side * blocks_per_tile_side;

        let weights_luma = config.subband_weights.pack_weights();

        // Upload frame data (entropy data, MVs, etc.)
        self.ensure_cached(ctx, padded_w, padded_h, info.width, info.height, info.tile_size);
        self.prepare_frame_data(ctx, frame);

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        // ===== Step 1: Entropy decode (Y plane, p=0) =====
        {
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("checkpoint_entropy"),
                });

            if bufs.ctx_adaptive_decode {
                cmd.copy_buffer_to_buffer(
                    &bufs.cpu_decoded_planes[0],
                    0,
                    &bufs.scratch_a,
                    0,
                    plane_size,
                );
            } else {
                match &frame.entropy {
                    EntropyData::Rice(_) => {
                        self.rice_decoder.dispatch_decode(
                            ctx,
                            &mut cmd,
                            &bufs.entropy_params[0],
                            &bufs.entropy_tile_info[0],
                            &bufs.entropy_var_a[0],
                            &bufs.entropy_var_b[0],
                            &bufs.scratch_a,
                            tiles_per_plane as u32,
                        );
                    }
                    EntropyData::Rans(_) | EntropyData::SubbandRans(_) => {
                        self.rans_decoder.dispatch_decode(
                            ctx,
                            &mut cmd,
                            &bufs.entropy_params[0],
                            &bufs.entropy_tile_info[0],
                            &bufs.entropy_var_a[0],
                            &bufs.entropy_var_b[0],
                            &bufs.scratch_a,
                            tiles_per_plane as u32,
                        );
                    }
                    EntropyData::Bitplane(_) => {
                        let total_blocks = (tiles_per_plane * blocks_per_tile) as u32;
                        self.bitplane_decoder.dispatch_decode(
                            ctx,
                            &mut cmd,
                            &bufs.entropy_params[0],
                            &bufs.entropy_tile_info[0],
                            &bufs.entropy_var_a[0],
                            &bufs.entropy_var_b[0],
                            &bufs.scratch_a,
                            total_blocks,
                        );
                    }
                    EntropyData::Huffman(_) => {
                        self.huffman_decoder.dispatch_decode(
                            ctx,
                            &mut cmd,
                            &bufs.entropy_params[0],
                            &bufs.entropy_tile_info[0],
                            &bufs.cpu_decoded_planes[0],
                            &bufs.entropy_var_a[0],
                            &bufs.entropy_var_b[0],
                            &bufs.scratch_a,
                            tiles_per_plane as u32,
                        );
                    }
                }
            }

            ctx.queue.submit(Some(cmd.finish()));
            ctx.device.poll(wgpu::Maintain::Wait);
        }
        let quantized = read_buffer_f32(ctx, &bufs.scratch_a, padded_pixels);

        // ===== Step 2: Dequantize (scratch_a → scratch_b) =====
        {
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("checkpoint_dequant"),
                });

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
                config.wavelet_levels,
                &weights_luma,
                None, // P-frames have no weight map
                0.0,  // no DCT freq weighting
            );

            ctx.queue.submit(Some(cmd.finish()));
            ctx.device.poll(wgpu::Maintain::Wait);
        }
        let dequantized = read_buffer_f32(ctx, &bufs.scratch_b, padded_pixels);

        // ===== Step 3: Inverse wavelet (scratch_b → scratch_c → scratch_a) =====
        {
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("checkpoint_idwt"),
                });

            self.transform.inverse(
                ctx,
                &mut cmd,
                &bufs.scratch_b,
                &bufs.scratch_c,
                &bufs.scratch_a,
                info,
                config.wavelet_levels,
                config.wavelet_type,
                0, // single-plane cmd encoder
            );

            ctx.queue.submit(Some(cmd.finish()));
            ctx.device.poll(wgpu::Maintain::Wait);
        }
        let spatial_residual = read_buffer_f32(ctx, &bufs.scratch_a, padded_pixels);

        // ===== Step 4: MC inverse (scratch_a + reference → plane_results[0]) =====
        {
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("checkpoint_mc_inv"),
                });

            self.motion.compensate(
                ctx,
                &mut cmd,
                &bufs.scratch_a,
                &bufs.reference_planes[0],
                &bufs.mv_buf,
                &bufs.plane_results[0],
                padded_w,
                padded_h,
                false, // inverse: recon = residual + predicted
                bufs.mc_block_size,
            );

            ctx.queue.submit(Some(cmd.finish()));
            ctx.device.poll(wgpu::Maintain::Wait);
        }
        let reconstructed = read_buffer_f32(ctx, &bufs.plane_results[0], padded_pixels);

        // Derive MC prediction: prediction = reconstructed - residual
        let mc_prediction: Vec<f32> = reconstructed
            .iter()
            .zip(spatial_residual.iter())
            .map(|(&r, &res)| r - res)
            .collect();

        // Update reference planes for future frames (mirror normal decode path)
        {
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("checkpoint_update_ref"),
                });
            cmd.copy_buffer_to_buffer(
                &bufs.plane_results[0],
                0,
                &bufs.reference_planes[0],
                0,
                plane_size,
            );
            ctx.queue.submit(Some(cmd.finish()));
            ctx.device.poll(wgpu::Maintain::Wait);
        }

        drop(cached);

        PFrameCheckpoints {
            mc_prediction,
            quantized,
            dequantized,
            spatial_residual,
            reconstructed,
        }
    }
}
