use wgpu;

use super::color::ColorConverter;
use super::quantize::Quantizer;
use super::rans::{self, RansTile};
use super::transform::WaveletTransform;
use crate::{CodecConfig, CompressedFrame, FrameInfo, GpuContext};

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
        let mut all_tiles: Vec<RansTile> = Vec::new();
        let tile_size = config.tile_size as usize;
        let tiles_x = info.tiles_x() as usize;
        let tiles_y = info.tiles_y() as usize;

        for p in 0..3 {
            ctx.queue
                .write_buffer(&plane_buf_a, 0, bytemuck::cast_slice(&planes[p]));

            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encode_plane"),
                });

            // Wavelet forward
            self.transform
                .forward(ctx, &mut cmd, &plane_buf_a, &plane_buf_b, &plane_buf_c, &info, config.wavelet_levels);

            // Quantize
            self.quantize.dispatch(
                ctx,
                &mut cmd,
                &plane_buf_c,
                &plane_buf_b,
                padded_pixels as u32,
                config.quantization_step,
                config.dead_zone,
                true,
            );

            ctx.queue.submit(Some(cmd.finish()));

            // Read back quantized plane
            let quantized = read_buffer_f32(ctx, &plane_buf_b, padded_pixels);

            // rANS encode each tile independently
            for ty in 0..tiles_y {
                for tx in 0..tiles_x {
                    let coeffs = extract_tile_coefficients(
                        &quantized,
                        padded_w as usize,
                        tx,
                        ty,
                        tile_size,
                    );
                    let tile = rans::rans_encode_tile(&coeffs);
                    all_tiles.push(tile);
                }
            }
        }

        CompressedFrame {
            info,
            config: config.clone(),
            tiles: all_tiles,
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
