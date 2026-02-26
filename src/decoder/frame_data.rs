use wgpu;

use super::pipeline::DecoderPipeline;
use crate::encoder::bitplane::GpuBitplaneDecoder;
use crate::encoder::cfl;
use crate::encoder::entropy_helpers;
use crate::encoder::motion::MotionEstimator;
use crate::encoder::rans_gpu::GpuRansDecoder;
use crate::gpu_util::ensure_var_buf;
use crate::{CompressedFrame, EntropyData, GpuContext};

impl DecoderPipeline {
    /// Write per-frame data into pre-allocated cached buffers.
    /// Handles entropy data, CfL alphas, weight map, and motion vectors.
    pub(super) fn prepare_frame_data(&self, ctx: &GpuContext, frame: &CompressedFrame) {
        let mut cached = self.cached.borrow_mut();
        let bufs = cached.as_mut().unwrap();
        let info = &frame.info;
        let tiles_per_plane = info.tiles_x() as usize * info.tiles_y() as usize;

        let storage_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        // --- Entropy data ---
        // Reset context-adaptive flag; only SubbandRans with expanded groups sets it true.
        bufs.ctx_adaptive_decode = false;

        match &frame.entropy {
            EntropyData::Rans(tiles) => {
                for p in 0..3 {
                    let plane_tiles = &tiles[p * tiles_per_plane..(p + 1) * tiles_per_plane];
                    let packed = GpuRansDecoder::pack_decode_data(plane_tiles, info);
                    let a_size = (packed.cumfreq.len() * 4) as u64;
                    let b_size = (packed.stream_data.len() * 4) as u64;

                    ensure_var_buf(
                        ctx,
                        &mut bufs.entropy_var_a[p],
                        &mut bufs.entropy_var_a_cap[p],
                        a_size,
                        "dec_entropy_var_a",
                        storage_dst,
                    );
                    ensure_var_buf(
                        ctx,
                        &mut bufs.entropy_var_b[p],
                        &mut bufs.entropy_var_b_cap[p],
                        b_size,
                        "dec_entropy_var_b",
                        storage_dst,
                    );

                    ctx.queue.write_buffer(
                        &bufs.entropy_params[p],
                        0,
                        bytemuck::bytes_of(&packed.params),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_tile_info[p],
                        0,
                        bytemuck::cast_slice(&packed.tile_info),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_var_a[p],
                        0,
                        bytemuck::cast_slice(&packed.cumfreq),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_var_b[p],
                        0,
                        bytemuck::cast_slice(&packed.stream_data),
                    );
                }
            }
            EntropyData::SubbandRans(tiles) => {
                // Detect context-adaptive tiles: num_groups > 1 + num_levels
                let is_ctx_adaptive = tiles
                    .first()
                    .is_some_and(|t| t.num_groups > t.num_levels * 2);
                bufs.ctx_adaptive_decode = is_ctx_adaptive;

                if is_ctx_adaptive {
                    // CPU decode for context-adaptive tiles; write decoded f32 coefficients
                    // directly into cpu_decoded_planes buffers for each plane.
                    let padded_w = info.padded_width() as usize;
                    let tile_size = info.tile_size as usize;
                    for p in 0..3 {
                        let plane_data = entropy_helpers::entropy_decode_plane(
                            &frame.entropy,
                            p,
                            tiles_per_plane,
                            tile_size,
                            padded_w,
                        );
                        ctx.queue.write_buffer(
                            &bufs.cpu_decoded_planes[p],
                            0,
                            bytemuck::cast_slice(&plane_data),
                        );
                    }
                } else {
                    for p in 0..3 {
                        let plane_tiles = &tiles[p * tiles_per_plane..(p + 1) * tiles_per_plane];
                        let packed = GpuRansDecoder::pack_decode_data_subband(plane_tiles, info);
                        let a_size = (packed.cumfreq.len() * 4) as u64;
                        let b_size = (packed.stream_data.len() * 4) as u64;

                        ensure_var_buf(
                            ctx,
                            &mut bufs.entropy_var_a[p],
                            &mut bufs.entropy_var_a_cap[p],
                            a_size,
                            "dec_entropy_var_a",
                            storage_dst,
                        );
                        ensure_var_buf(
                            ctx,
                            &mut bufs.entropy_var_b[p],
                            &mut bufs.entropy_var_b_cap[p],
                            b_size,
                            "dec_entropy_var_b",
                            storage_dst,
                        );

                        ctx.queue.write_buffer(
                            &bufs.entropy_params[p],
                            0,
                            bytemuck::bytes_of(&packed.params),
                        );
                        ctx.queue.write_buffer(
                            &bufs.entropy_tile_info[p],
                            0,
                            bytemuck::cast_slice(&packed.tile_info),
                        );
                        ctx.queue.write_buffer(
                            &bufs.entropy_var_a[p],
                            0,
                            bytemuck::cast_slice(&packed.cumfreq),
                        );
                        ctx.queue.write_buffer(
                            &bufs.entropy_var_b[p],
                            0,
                            bytemuck::cast_slice(&packed.stream_data),
                        );
                    }
                }
            }
            EntropyData::Bitplane(tiles) => {
                for p in 0..3 {
                    let plane_tiles = &tiles[p * tiles_per_plane..(p + 1) * tiles_per_plane];
                    let packed = GpuBitplaneDecoder::pack_decode_data(plane_tiles, info);
                    let a_size = (packed.block_info.len() * 4) as u64;
                    let b_size = (packed.bitplane_data.len() * 4) as u64;

                    ensure_var_buf(
                        ctx,
                        &mut bufs.entropy_var_a[p],
                        &mut bufs.entropy_var_a_cap[p],
                        a_size,
                        "dec_entropy_var_a",
                        storage_dst,
                    );
                    ensure_var_buf(
                        ctx,
                        &mut bufs.entropy_var_b[p],
                        &mut bufs.entropy_var_b_cap[p],
                        b_size,
                        "dec_entropy_var_b",
                        storage_dst,
                    );

                    ctx.queue.write_buffer(
                        &bufs.entropy_params[p],
                        0,
                        bytemuck::bytes_of(&packed.params),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_tile_info[p],
                        0,
                        bytemuck::cast_slice(&packed.tile_info),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_var_a[p],
                        0,
                        bytemuck::cast_slice(&packed.block_info),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_var_b[p],
                        0,
                        bytemuck::cast_slice(&packed.bitplane_data),
                    );
                }
            }
        }

        // --- CfL alphas ---
        if let Some(cfl_data) = &frame.cfl_alphas {
            // Dequantize i16 alphas to f32 for GPU prediction shaders
            let all_f32: Vec<f32> = cfl_data
                .alphas
                .iter()
                .map(|&q| cfl::dequantize_alpha(q))
                .collect();
            let alpha_size = (all_f32.len() * 4) as u64;
            ensure_var_buf(
                ctx,
                &mut bufs.cfl_alpha_buf,
                &mut bufs.cfl_alpha_cap,
                alpha_size,
                "dec_cfl_alpha",
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            cfl::write_alpha_buffer_into(ctx, &all_f32, &bufs.cfl_alpha_buf);

            // Ensure plane_alpha_bufs are large enough
            let nsb = cfl_data.num_subbands as usize;
            let alphas_per_plane = tiles_per_plane * nsb;
            let plane_alpha_size = (alphas_per_plane * 4) as u64;
            if plane_alpha_size > bufs.plane_alpha_cap {
                let new_cap = (plane_alpha_size * 2).max(4);
                for i in 0..2 {
                    bufs.plane_alpha_bufs[i] = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("dec_plane_alpha_{i}")),
                        size: new_cap,
                        usage: storage_dst,
                        mapped_at_creation: false,
                    });
                }
                bufs.plane_alpha_cap = new_cap;
            }
        }

        // --- Weight map ---
        if let Some(wm) = &frame.weight_map {
            let wm_size = (wm.len() * 4) as u64;
            ensure_var_buf(
                ctx,
                &mut bufs.weight_map_buf,
                &mut bufs.weight_map_cap,
                wm_size,
                "dec_weight_map",
                storage_dst,
            );
            ctx.queue
                .write_buffer(&bufs.weight_map_buf, 0, bytemuck::cast_slice(wm));
        }

        // --- Motion vectors ---
        if let Some(mf) = &frame.motion_field {
            let mv_size = (mf.vectors.len() * 2 * 4) as u64;
            ensure_var_buf(
                ctx,
                &mut bufs.mv_buf,
                &mut bufs.mv_cap,
                mv_size,
                "dec_mv",
                storage_dst,
            );
            MotionEstimator::write_motion_vectors_into(ctx, &mf.vectors, &bufs.mv_buf);
        }

        // --- Backward motion vectors + block modes (B-frames) ---
        if let Some(mf) = &frame.motion_field {
            if let Some(ref bwd_vecs) = mf.backward_vectors {
                let bwd_mv_size = (bwd_vecs.len() * 2 * 4) as u64;
                ensure_var_buf(
                    ctx,
                    &mut bufs.bwd_mv_buf,
                    &mut bufs.bwd_mv_cap,
                    bwd_mv_size,
                    "dec_bwd_mv",
                    storage_dst,
                );
                MotionEstimator::write_motion_vectors_into(ctx, bwd_vecs, &bufs.bwd_mv_buf);
            }
            if let Some(ref modes) = mf.block_modes {
                let modes_size = (modes.len() * 4) as u64; // u8 → u32 on GPU
                ensure_var_buf(
                    ctx,
                    &mut bufs.block_modes_buf,
                    &mut bufs.block_modes_cap,
                    modes_size,
                    "dec_block_modes",
                    storage_dst,
                );
                // Upload as u32 for shader compatibility
                let modes_u32: Vec<u32> = modes.iter().map(|&m| m as u32).collect();
                ctx.queue
                    .write_buffer(&bufs.block_modes_buf, 0, bytemuck::cast_slice(&modes_u32));
            }
        }
    }
}
