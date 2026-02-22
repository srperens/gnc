use wgpu;

use crate::encoder::color::ColorConverter;
use crate::encoder::pipeline::gpu_read_f32;
use crate::encoder::quantize::Quantizer;
use crate::encoder::rans;
use crate::encoder::transform::WaveletTransform;
use crate::{CompressedFrame, GpuContext};

/// Full decoding pipeline: rANS Decode -> Dequantize -> Inverse Wavelet -> Inverse Color
pub struct DecoderPipeline {
    color: ColorConverter,
    transform: WaveletTransform,
    quantize: Quantizer,
}

impl DecoderPipeline {
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            color: ColorConverter::new(ctx),
            transform: WaveletTransform::new(ctx),
            quantize: Quantizer::new(ctx),
        }
    }

    /// Decode a compressed frame back to RGB f32 data.
    /// Returns Vec<f32> of length width * height * 3 (interleaved R,G,B).
    pub fn decode(&self, ctx: &GpuContext, frame: &CompressedFrame) -> Vec<f32> {
        let info = &frame.info;
        let config = &frame.config;
        let padded_w = info.padded_width();
        let padded_h = info.padded_height();
        let padded_pixels = (padded_w * padded_h) as usize;
        let pixel_count_3 = padded_pixels * 3;

        let buf_size = (pixel_count_3 * std::mem::size_of::<f32>()) as u64;
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;

        let tile_size = config.tile_size as usize;
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;
        let tiles_per_plane = tiles_x * tiles_y;

        // Step 1: rANS decode each tile on CPU, reconstruct quantized planes
        let mut planes: [Vec<f32>; 3] = [
            vec![0.0; padded_pixels],
            vec![0.0; padded_pixels],
            vec![0.0; padded_pixels],
        ];

        for p in 0..3 {
            for ty in 0..tiles_y {
                for tx in 0..tiles_x {
                    let tile_idx = p * tiles_per_plane + ty * tiles_x + tx;
                    let tile = &frame.tiles[tile_idx];
                    let coeffs = rans::rans_decode_tile(tile);
                    scatter_tile_coefficients(
                        &mut planes[p],
                        padded_w as usize,
                        tx,
                        ty,
                        tile_size,
                        &coeffs,
                    );
                }
            }
        }

        // Plane buffers for GPU operations
        let plane_buf_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_plane_a"),
            size: plane_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let plane_buf_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_plane_b"),
            size: plane_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let plane_buf_c = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_plane_c"),
            size: plane_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Color output
        let color_in_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_color_in"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rgb_out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_rgb_out"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Step 2 & 3: Dequantize and inverse wavelet per plane on GPU
        let mut ycocg_interleaved = vec![0.0f32; pixel_count_3];

        for p in 0..3 {
            ctx.queue
                .write_buffer(&plane_buf_a, 0, bytemuck::cast_slice(&planes[p]));

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("decode_plane"),
                });

            // Dequantize
            self.quantize.dispatch(
                ctx,
                &mut cmd,
                &plane_buf_a,
                &plane_buf_b,
                padded_pixels as u32,
                config.quantization_step,
                config.dead_zone,
                false,
            );

            // Inverse wavelet
            self.transform
                .inverse(ctx, &mut cmd, &plane_buf_b, &plane_buf_c, &plane_buf_a, info, config.wavelet_levels);

            ctx.queue.submit(Some(cmd.finish()));

            let reconstructed_plane = gpu_read_f32(ctx, &plane_buf_a, padded_pixels);
            for i in 0..padded_pixels {
                ycocg_interleaved[i * 3 + p] = reconstructed_plane[i];
            }
        }

        // Step 4: Inverse color conversion (YCoCg-R -> RGB)
        ctx.queue
            .write_buffer(&color_in_buf, 0, bytemuck::cast_slice(&ycocg_interleaved));

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("color_decode"),
            });
        self.color.dispatch(
            ctx, &mut cmd, &color_in_buf, &rgb_out_buf, padded_w, padded_h, false,
        );
        ctx.queue.submit(Some(cmd.finish()));

        let padded_rgb = gpu_read_f32(ctx, &rgb_out_buf, pixel_count_3);

        // Crop back to original dimensions
        crop_frame(&padded_rgb, padded_w, padded_h, info.width, info.height)
    }
}

/// Write decoded tile coefficients back into a flat plane array.
fn scatter_tile_coefficients(
    plane: &mut [f32],
    plane_width: usize,
    tile_x: usize,
    tile_y: usize,
    tile_size: usize,
    coeffs: &[i32],
) {
    let origin_x = tile_x * tile_size;
    let origin_y = tile_y * tile_size;
    let mut ci = 0;

    for y in 0..tile_size {
        for x in 0..tile_size {
            let idx = (origin_y + y) * plane_width + (origin_x + x);
            plane[idx] = coeffs[ci] as f32;
            ci += 1;
        }
    }
}

/// Crop padded frame back to original dimensions.
fn crop_frame(padded: &[f32], pw: u32, _ph: u32, w: u32, h: u32) -> Vec<f32> {
    let pw = pw as usize;
    let w = w as usize;
    let h = h as usize;
    let mut output = vec![0.0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let src = (y * pw + x) * 3;
            let dst = (y * w + x) * 3;
            output[dst] = padded[src];
            output[dst + 1] = padded[src + 1];
            output[dst + 2] = padded[src + 2];
        }
    }
    output
}
