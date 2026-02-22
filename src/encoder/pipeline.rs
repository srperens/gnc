use wgpu;

use super::cfl;
use super::color::ColorConverter;
use super::quantize::Quantizer;
use super::rans::{self, InterleavedRansTile};
use super::transform::WaveletTransform;
use crate::{CflAlphas, CodecConfig, CompressedFrame, FrameInfo, GpuContext};

/// Full encoding pipeline: Color -> Wavelet -> Quantize -> rANS Entropy
pub struct EncoderPipeline {
    color: ColorConverter,
    transform: WaveletTransform,
    quantize: Quantizer,
}

impl EncoderPipeline {
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            color: ColorConverter::new(ctx),
            transform: WaveletTransform::new(ctx),
            quantize: Quantizer::new(ctx),
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let plane_buf_c = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_plane_c"),
            size: plane_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
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
        self.color
            .dispatch(ctx, &mut cmd, &input_buf, &color_out, padded_w, padded_h, true);
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

        // Step 2 & 3: Wavelet transform + quantize per plane on GPU,
        // then rANS encode per tile on CPU
        let mut all_tiles: Vec<InterleavedRansTile> = Vec::new();
        let tile_size = config.tile_size as usize;
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;

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
            self.transform
                .forward(ctx, &mut cmd, &plane_buf_a, &plane_buf_b, &plane_buf_c, &info, config.wavelet_levels);

            if p == 0 && config.cfl_enabled {
                // Y plane with CfL: quantize, then dequantize to get reconstructed Y wavelet.
                // Quantize: plane_buf_c → plane_buf_b (quantized)
                self.quantize.dispatch(
                    ctx, &mut cmd, &plane_buf_c, &plane_buf_b,
                    padded_pixels as u32, config.quantization_step, config.dead_zone,
                    true, padded_w, padded_h, config.tile_size, config.wavelet_levels,
                    &weights_luma,
                );
                // Dequantize: plane_buf_b → plane_buf_a (reconstructed Y wavelet)
                self.quantize.dispatch(
                    ctx, &mut cmd, &plane_buf_b, &plane_buf_a,
                    padded_pixels as u32, config.quantization_step, config.dead_zone,
                    false, padded_w, padded_h, config.tile_size, config.wavelet_levels,
                    &weights_luma,
                );
                ctx.queue.submit(Some(cmd.finish()));

                // Read back both quantized Y and reconstructed Y wavelet
                let quantized = read_buffer_f32(ctx, &plane_buf_b, padded_pixels);
                recon_y_wavelet = Some(read_buffer_f32(ctx, &plane_buf_a, padded_pixels));

                // rANS encode Y tiles
                for ty in 0..tiles_y {
                    for tx in 0..tiles_x {
                        let coeffs = extract_tile_coefficients(
                            &quantized, padded_w as usize, tx, ty, tile_size,
                        );
                        let tile = rans::rans_encode_tile_interleaved_zrl(&coeffs);
                        all_tiles.push(tile);
                    }
                }
            } else if use_cfl {
                // Chroma plane with CfL prediction
                ctx.queue.submit(Some(cmd.finish()));

                // Read back chroma wavelet coefficients from plane_buf_c
                let chroma_wavelet = read_buffer_f32(ctx, &plane_buf_c, padded_pixels);
                let recon_y = recon_y_wavelet.as_ref().unwrap();

                // Compute per-tile per-subband alpha
                let alphas_f32 = cfl::compute_cfl_alphas(
                    recon_y, &chroma_wavelet,
                    padded_w, padded_h, config.tile_size, config.wavelet_levels,
                );

                // Quantize then dequantize alphas (encoder must use same values as decoder)
                let q_alphas: Vec<u8> = alphas_f32.iter().map(|&a| cfl::quantize_alpha(a)).collect();
                let dq_alphas: Vec<f32> = q_alphas.iter().map(|&q| cfl::dequantize_alpha(q)).collect();

                // Compute residual on CPU: residual = chroma_wavelet - alpha * recon_y
                let residual = cfl::apply_cfl_predict_cpu(
                    &chroma_wavelet, recon_y, &dq_alphas,
                    padded_w, padded_h, config.tile_size, config.wavelet_levels,
                );

                // Upload residual, quantize on GPU, read back
                ctx.queue.write_buffer(&plane_buf_c, 0, bytemuck::cast_slice(&residual));
                let mut cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encode_cfl_residual"),
                });
                self.quantize.dispatch(
                    ctx, &mut cmd, &plane_buf_c, &plane_buf_b,
                    padded_pixels as u32, config.quantization_step, config.dead_zone,
                    true, padded_w, padded_h, config.tile_size, config.wavelet_levels,
                    &weights_chroma,
                );
                ctx.queue.submit(Some(cmd.finish()));

                let quantized = read_buffer_f32(ctx, &plane_buf_b, padded_pixels);

                // rANS encode residual tiles
                for ty in 0..tiles_y {
                    for tx in 0..tiles_x {
                        let coeffs = extract_tile_coefficients(
                            &quantized, padded_w as usize, tx, ty, tile_size,
                        );
                        let tile = rans::rans_encode_tile_interleaved_zrl(&coeffs);
                        all_tiles.push(tile);
                    }
                }

                // Store quantized alphas for serialization
                cfl_alphas_all.extend_from_slice(&q_alphas);
            } else {
                // Non-CfL path (original flow)
                let weights = if p == 0 { &weights_luma } else { &weights_chroma };
                self.quantize.dispatch(
                    ctx, &mut cmd, &plane_buf_c, &plane_buf_b,
                    padded_pixels as u32, config.quantization_step, config.dead_zone,
                    true, padded_w, padded_h, config.tile_size, config.wavelet_levels,
                    weights,
                );
                ctx.queue.submit(Some(cmd.finish()));

                let quantized = read_buffer_f32(ctx, &plane_buf_b, padded_pixels);

                for ty in 0..tiles_y {
                    for tx in 0..tiles_x {
                        let coeffs = extract_tile_coefficients(
                            &quantized, padded_w as usize, tx, ty, tile_size,
                        );
                        let tile = rans::rans_encode_tile_interleaved_zrl(&coeffs);
                        all_tiles.push(tile);
                    }
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

        CompressedFrame {
            info,
            config: config.clone(),
            tiles: all_tiles,
            cfl_alphas,
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

// Make read helpers available to decoder
pub(crate) fn gpu_read_f32(ctx: &GpuContext, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
    read_buffer_f32(ctx, buffer, count)
}
