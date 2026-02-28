//! Transform Shootout: Compare block-based transforms against wavelet baseline.
//!
//! Measures both speed (GPU dispatch time) and RD performance (PSNR vs bitrate proxy)
//! for each transform candidate on a single Y plane.

use std::time::Instant;

use crate::encoder::block_transform::{BlockTransform, BlockTransformType};
use crate::encoder::fused_block::FusedBlock;
use crate::encoder::transform::WaveletTransform;
use crate::{FrameInfo, GpuContext, WaveletType};

/// One RD measurement point for a (transform, qstep) combination.
#[derive(Debug, Clone)]
pub struct RDPoint {
    pub qstep: f32,
    pub psnr_db: f64,
    /// Fraction of nonzero coefficients after quantization (proxy for bitrate).
    pub nonzero_frac: f64,
    /// Estimated bits per pixel (simple entropy estimate).
    pub bpp_estimate: f64,
}

/// Full result for one transform candidate.
#[derive(Debug, Clone)]
pub struct TransformResult {
    pub name: String,
    /// Number of GPU dispatches for forward transform.
    pub dispatches: u32,
    /// Median forward transform time over N iterations (ms).
    pub forward_time_ms: f64,
    /// Median inverse transform time over N iterations (ms).
    pub inverse_time_ms: f64,
    /// RD points at various quality levels.
    pub rd_points: Vec<RDPoint>,
}

/// Extract Y (luma) plane from interleaved RGB f32 using YCoCg-R formula.
/// Y = (R + 2*G + B) / 4
fn extract_y_plane(rgb: &[f32], w: u32, h: u32) -> Vec<f32> {
    let mut y = Vec::with_capacity((w * h) as usize);
    for i in 0..(w * h) as usize {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        y.push((r + 2.0 * g + b) * 0.25);
    }
    y
}

/// Pad plane to target dimensions by edge-replicating.
fn pad_plane(src: &[f32], w: u32, h: u32, pad_w: u32, pad_h: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (pad_w * pad_h) as usize];
    for y in 0..pad_h {
        for x in 0..pad_w {
            let sx = x.min(w - 1);
            let sy = y.min(h - 1);
            out[(y * pad_w + x) as usize] = src[(sy * w + sx) as usize];
        }
    }
    out
}

/// Simple scalar quantize + dequantize on CPU. Returns (quantized_indices, reconstructed).
fn quantize_dequantize(coeffs: &[f32], qstep: f32) -> (Vec<i32>, Vec<f32>) {
    let mut indices = Vec::with_capacity(coeffs.len());
    let mut recon = Vec::with_capacity(coeffs.len());
    for &c in coeffs {
        let q = (c / qstep).round() as i32;
        indices.push(q);
        recon.push(q as f32 * qstep);
    }
    (indices, recon)
}

/// Compute PSNR between original and reconstructed plane (unpadded region only).
fn compute_psnr(orig: &[f32], recon: &[f32], w: u32, h: u32, pad_w: u32) -> f64 {
    let mut mse = 0.0f64;
    let n = (w * h) as f64;
    for y in 0..h {
        for x in 0..w {
            let o = orig[(y * w + x) as usize] as f64;
            let r = recon[(y * pad_w + x) as usize] as f64;
            let d = o - r;
            mse += d * d;
        }
    }
    mse /= n;
    if mse < 1e-10 {
        return 99.0;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Estimate bits per pixel from quantized coefficients.
/// Uses a simple model: nonzero_count * (1 + log2(mean_abs_value)) + zero_count * 0.1
fn estimate_bpp(indices: &[i32], total_pixels: u32) -> (f64, f64) {
    let mut nonzero_count = 0u64;
    let mut abs_sum = 0u64;
    for &q in indices {
        if q != 0 {
            nonzero_count += 1;
            abs_sum += q.unsigned_abs() as u64;
        }
    }
    let total = indices.len() as f64;
    let nz_frac = nonzero_count as f64 / total;

    // Rough entropy estimate
    let bits = if nonzero_count > 0 {
        let mean_abs = abs_sum as f64 / nonzero_count as f64;
        let bits_per_nz = 1.0 + mean_abs.log2().max(0.0); // sign + magnitude
        let zero_bits = (total - nonzero_count as f64) * 0.1; // ~0.1 bits per zero (run coding)
        nonzero_count as f64 * bits_per_nz + zero_bits
    } else {
        total * 0.1
    };
    let bpp = bits / total_pixels as f64;
    (bpp, nz_frac)
}

/// Create a storage buffer from f32 data.
fn create_storage_buf(ctx: &GpuContext, data: &[f32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shootout_buf"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
}

/// Create an empty storage buffer of given f32 count.
fn create_empty_buf(ctx: &GpuContext, count: usize) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shootout_buf"),
        size: (count * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Read back a GPU buffer to CPU f32 vec.
fn readback(ctx: &GpuContext, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback_staging"),
        size: (count * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, (count * 4) as u64);
    ctx.queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

/// Measure median time of a GPU operation over N iterations.
fn measure_gpu_time(ctx: &GpuContext, iterations: u32, mut submit_fn: impl FnMut()) -> f64 {
    // Warmup
    submit_fn();
    ctx.device.poll(wgpu::Maintain::Wait);

    let mut times = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let t = Instant::now();
        submit_fn();
        ctx.device.poll(wgpu::Maintain::Wait);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2] // median
}

/// Run the full transform shootout.
pub fn run_shootout(
    ctx: &GpuContext,
    rgb_data: &[f32],
    width: u32,
    height: u32,
    iterations: u32,
) -> Vec<TransformResult> {
    let y_orig = extract_y_plane(rgb_data, width, height);

    // Pad to multiple of 256 (tile size) so wavelet baseline works too.
    let pad_w = ((width + 255) / 256) * 256;
    let pad_h = ((height + 255) / 256) * 256;
    let y_padded = pad_plane(&y_orig, width, height, pad_w, pad_h);
    let pixel_count = (pad_w * pad_h) as usize;

    let input_buf = create_storage_buf(ctx, &y_padded);
    let output_buf = create_empty_buf(ctx, pixel_count);
    let temp_buf = create_empty_buf(ctx, pixel_count);

    let block_transform = BlockTransform::new(ctx);
    let wavelet_transform = WaveletTransform::new(ctx);

    let qsteps: &[f32] = &[0.001, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
    let total_px = width * height;

    let mut results = Vec::new();

    // ---- Block transform candidates ----
    for &tt in BlockTransformType::all() {
        println!("  Testing {}...", tt.name());

        // Measure forward transform time
        let fwd_time = measure_gpu_time(ctx, iterations, || {
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fwd"),
                });
            block_transform.dispatch(
                ctx, &mut enc, &input_buf, &output_buf, pad_w, pad_h, true, tt,
            );
            ctx.queue.submit(Some(enc.finish()));
        });

        // Measure inverse transform time
        let inv_time = measure_gpu_time(ctx, iterations, || {
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("inv"),
                });
            block_transform.dispatch(
                ctx, &mut enc, &output_buf, &temp_buf, pad_w, pad_h, false, tt,
            );
            ctx.queue.submit(Some(enc.finish()));
        });

        // Direct GPU roundtrip (no quantization) to verify transform correctness
        {
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("roundtrip"),
                });
            block_transform.dispatch(
                ctx, &mut enc, &input_buf, &output_buf, pad_w, pad_h, true, tt,
            );
            block_transform.dispatch(
                ctx, &mut enc, &output_buf, &temp_buf, pad_w, pad_h, false, tt,
            );
            ctx.queue.submit(Some(enc.finish()));
            ctx.device.poll(wgpu::Maintain::Wait);

            let roundtrip = readback(ctx, &temp_buf, pixel_count);
            let rt_psnr = compute_psnr(&y_orig, &roundtrip, width, height, pad_w);
            println!("    Direct GPU roundtrip PSNR: {:.2} dB", rt_psnr);
        }

        // Forward transform for RD analysis
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fwd_rd"),
            });
        block_transform.dispatch(
            ctx, &mut enc, &input_buf, &output_buf, pad_w, pad_h, true, tt,
        );
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);

        let coeffs = readback(ctx, &output_buf, pixel_count);

        // RD points
        let mut rd_points = Vec::new();
        for &qs in qsteps {
            let (indices, recon_coeffs) = quantize_dequantize(&coeffs, qs);

            // Upload reconstructed coefficients and inverse transform
            ctx.queue
                .write_buffer(&output_buf, 0, bytemuck::cast_slice(&recon_coeffs));
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("inv_rd"),
                });
            block_transform.dispatch(
                ctx, &mut enc, &output_buf, &temp_buf, pad_w, pad_h, false, tt,
            );
            ctx.queue.submit(Some(enc.finish()));
            ctx.device.poll(wgpu::Maintain::Wait);

            let recon_pixels = readback(ctx, &temp_buf, pixel_count);
            let psnr = compute_psnr(&y_orig, &recon_pixels, width, height, pad_w);
            let (bpp, nz_frac) = estimate_bpp(&indices, total_px);

            rd_points.push(RDPoint {
                qstep: qs,
                psnr_db: psnr,
                nonzero_frac: nz_frac,
                bpp_estimate: bpp,
            });
        }

        results.push(TransformResult {
            name: tt.name().to_string(),
            dispatches: 1,
            forward_time_ms: fwd_time,
            inverse_time_ms: inv_time,
            rd_points,
        });
    }

    // ---- Wavelet baseline (CDF 9/7, 4 levels) ----
    {
        let levels = 4u32;
        let wtype = WaveletType::CDF97;
        let info = FrameInfo {
            width: pad_w,
            height: pad_h,
            bit_depth: 8,
            tile_size: 256,
        };

        println!("  Testing CDF-9/7 wavelet (4 levels)...");

        let fwd_time = measure_gpu_time(ctx, iterations, || {
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wvl_fwd"),
                });
            wavelet_transform.forward(
                ctx,
                &mut enc,
                &input_buf,
                &temp_buf,
                &output_buf,
                &info,
                levels,
                wtype,
            );
            ctx.queue.submit(Some(enc.finish()));
        });

        let inv_time = measure_gpu_time(ctx, iterations, || {
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wvl_inv"),
                });
            wavelet_transform.inverse(
                ctx,
                &mut enc,
                &output_buf,
                &temp_buf,
                &input_buf,
                &info,
                levels,
                wtype,
            );
            ctx.queue.submit(Some(enc.finish()));
        });

        // Forward transform for RD
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("wvl_fwd_rd"),
            });
        wavelet_transform.forward(
            ctx,
            &mut enc,
            &input_buf,
            &temp_buf,
            &output_buf,
            &info,
            levels,
            wtype,
        );
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);

        let coeffs = readback(ctx, &output_buf, pixel_count);

        let mut rd_points = Vec::new();
        for &qs in qsteps {
            let (indices, recon_coeffs) = quantize_dequantize(&coeffs, qs);

            // Upload and inverse wavelet
            // Restore input_buf first (wavelet inverse reads from input_buf after copy)
            ctx.queue
                .write_buffer(&output_buf, 0, bytemuck::cast_slice(&recon_coeffs));
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("wvl_inv_rd"),
                });
            wavelet_transform.inverse(
                ctx,
                &mut enc,
                &output_buf,
                &temp_buf,
                &input_buf,
                &info,
                levels,
                wtype,
            );
            ctx.queue.submit(Some(enc.finish()));
            ctx.device.poll(wgpu::Maintain::Wait);

            let recon_pixels = readback(ctx, &input_buf, pixel_count);
            let psnr = compute_psnr(&y_orig, &recon_pixels, width, height, pad_w);
            let (bpp, nz_frac) = estimate_bpp(&indices, total_px);

            rd_points.push(RDPoint {
                qstep: qs,
                psnr_db: psnr,
                nonzero_frac: nz_frac,
                bpp_estimate: bpp,
            });
        }

        // Restore input_buf for any subsequent use
        ctx.queue
            .write_buffer(&input_buf, 0, bytemuck::cast_slice(&y_padded));

        results.push(TransformResult {
            name: "CDF-9/7 (4L)".to_string(),
            dispatches: levels * 2, // 2 passes per level (row + col)
            forward_time_ms: fwd_time,
            inverse_time_ms: inv_time,
            rd_points,
        });
    }

    results
}

/// Print shootout results as a formatted table.
pub fn print_results(results: &[TransformResult]) {
    println!("\n{}", "=".repeat(90));
    println!("TRANSFORM SHOOTOUT RESULTS");
    println!("{}", "=".repeat(90));

    // Speed table
    println!("\n--- Speed (median of N iterations) ---\n");
    println!(
        "{:<16} {:>10} {:>12} {:>12}",
        "Transform", "Dispatches", "Forward(ms)", "Inverse(ms)"
    );
    println!("{}", "-".repeat(54));
    for r in results {
        println!(
            "{:<16} {:>10} {:>12.3} {:>12.3}",
            r.name, r.dispatches, r.forward_time_ms, r.inverse_time_ms
        );
    }

    // RD table per qstep
    if let Some(first) = results.first() {
        for (i, pt) in first.rd_points.iter().enumerate() {
            println!(
                "\n--- RD @ qstep={:.0} ---\n",
                pt.qstep
            );
            println!(
                "{:<16} {:>10} {:>10} {:>12}",
                "Transform", "PSNR(dB)", "NZ(%)", "BPP(est)"
            );
            println!("{}", "-".repeat(52));
            for r in results {
                if let Some(rd) = r.rd_points.get(i) {
                    println!(
                        "{:<16} {:>10.2} {:>9.1}% {:>12.3}",
                        r.name,
                        rd.psnr_db,
                        rd.nonzero_frac * 100.0,
                        rd.bpp_estimate
                    );
                }
            }
        }
    }

    println!("\n{}", "=".repeat(90));
}

/// Benchmark the fused DCT-8×8 + quantize + local decode pipeline
/// against the separate-dispatch baseline (DCT-8×8 → quantize → IDCT).
pub fn run_fused_benchmark(
    ctx: &GpuContext,
    rgb_data: &[f32],
    width: u32,
    height: u32,
    iterations: u32,
) {
    let y_orig = extract_y_plane(rgb_data, width, height);
    let pad_w = ((width + 7) / 8) * 8; // Only need 8-alignment for DCT-8×8
    let pad_h = ((height + 7) / 8) * 8;
    let y_padded = pad_plane(&y_orig, width, height, pad_w, pad_h);
    let pixel_count = (pad_w * pad_h) as usize;
    let total_px = width * height;

    let input_buf = create_storage_buf(ctx, &y_padded);
    let quant_buf = create_empty_buf(ctx, pixel_count);
    let recon_buf = create_empty_buf(ctx, pixel_count);
    let temp_buf = create_empty_buf(ctx, pixel_count);

    let fused = FusedBlock::new(ctx);
    let block_transform = BlockTransform::new(ctx);

    let qsteps: &[f32] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0];

    println!("\n{}", "=".repeat(90));
    println!("FUSED MEGA-KERNEL BENCHMARK (DCT-8×8 + Quantize + Local Decode)");
    println!("{}", "=".repeat(90));

    // ---- Fused: 1 dispatch for all 3 stages ----
    println!("\n--- Fused pipeline (1 dispatch) ---");

    let fused_time = measure_gpu_time(ctx, iterations, || {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused"),
        });
        fused.dispatch(ctx, &mut enc, &input_buf, &quant_buf, &recon_buf,
            pad_w, pad_h, 4.0, 0.5, 7.0);
        ctx.queue.submit(Some(enc.finish()));
    });
    println!("  Fused time: {:.3} ms (1 dispatch)", fused_time);

    // ---- Separate: 3 dispatches (DCT fwd + quantize-on-CPU + DCT inv) ----
    // Note: we simulate the separate pipeline using existing block_transform
    println!("\n--- Separate pipeline (3 dispatches) ---");

    let separate_time = measure_gpu_time(ctx, iterations, || {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("separate"),
        });
        // Forward DCT
        block_transform.dispatch(ctx, &mut enc, &input_buf, &quant_buf,
            pad_w, pad_h, true, BlockTransformType::DCT8);
        // Note: in the real pipeline, quantize is another GPU dispatch.
        // Here we just show 2 dispatches (fwd + inv) as a lower bound.
        // The fused kernel also does quantize + dequantize + IDCT in one.
        block_transform.dispatch(ctx, &mut enc, &quant_buf, &recon_buf,
            pad_w, pad_h, false, BlockTransformType::DCT8);
        ctx.queue.submit(Some(enc.finish()));
    });
    println!("  Separate time: {:.3} ms (2 dispatches, no quant)", separate_time);
    println!("  Speedup: {:.2}x", separate_time / fused_time);

    // ---- RD comparison ----
    println!("\n--- RD Performance ---\n");
    println!("{:<10} {:>10} {:>10} {:>12}", "qstep", "PSNR(dB)", "NZ(%)", "BPP(est)");
    println!("{}", "-".repeat(46));

    for &qs in qsteps {
        // Run fused
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fused_rd"),
            });
            fused.dispatch(ctx, &mut enc, &input_buf, &quant_buf, &recon_buf,
                pad_w, pad_h, qs, 0.5, 7.0);
            ctx.queue.submit(Some(enc.finish()));
            ctx.device.poll(wgpu::Maintain::Wait);
        }

        let recon_pixels = readback(ctx, &recon_buf, pixel_count);
        let psnr = compute_psnr(&y_orig, &recon_pixels, width, height, pad_w);

        let quant_vals = readback(ctx, &quant_buf, pixel_count);
        let indices: Vec<i32> = quant_vals.iter().map(|&v| v as i32).collect();
        let (bpp, nz_frac) = estimate_bpp(&indices, total_px);

        println!("{:<10.1} {:>10.2} {:>9.1}% {:>12.3}", qs, psnr, nz_frac * 100.0, bpp);
    }

    // Verify roundtrip with near-lossless qstep
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused_rt"),
        });
        fused.dispatch(ctx, &mut enc, &input_buf, &quant_buf, &recon_buf,
            pad_w, pad_h, 0.001, 0.0, 0.0);
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);

        let recon_pixels = readback(ctx, &recon_buf, pixel_count);
        let psnr = compute_psnr(&y_orig, &recon_pixels, width, height, pad_w);
        println!("\n  Near-lossless roundtrip PSNR: {:.2} dB", psnr);
    }

    println!("\n{}", "=".repeat(90));
}

// ---- CPU Haar reference for debugging ----

const INV_SQRT2: f32 = 0.707106781187;

/// CPU reference: 1-level Haar forward on a 16×16 block (in-place, stride=16 within buffer).
fn cpu_haar_forward_level(data: &mut [f32], stride: u32, size: u32) {
    let s = stride as usize;
    let n = size as usize;
    let half = n / 2;

    // Row Haar
    let mut tmp = vec![0.0f32; n];
    for r in 0..n {
        for c in 0..half {
            let even = data[r * s + c * 2];
            let odd = data[r * s + c * 2 + 1];
            tmp[c] = (even + odd) * INV_SQRT2;
            tmp[c + half] = (even - odd) * INV_SQRT2;
        }
        for c in 0..n {
            data[r * s + c] = tmp[c];
        }
    }

    // Column Haar
    let mut col_buf = vec![0.0f32; n];
    for c in 0..n {
        for r in 0..half {
            let even = data[(r * 2) * s + c];
            let odd = data[(r * 2 + 1) * s + c];
            col_buf[r] = (even + odd) * INV_SQRT2;
            col_buf[r + half] = (even - odd) * INV_SQRT2;
        }
        for r in 0..n {
            data[r * s + c] = col_buf[r];
        }
    }
}

/// CPU reference: 1-level Haar inverse on a 16×16 block (in-place, stride within buffer).
fn cpu_haar_inverse_level(data: &mut [f32], stride: u32, size: u32) {
    let s = stride as usize;
    let n = size as usize;
    let half = n / 2;

    // Inverse column Haar
    let mut col_buf = vec![0.0f32; n];
    for c in 0..n {
        for i in 0..half {
            let low = data[i * s + c];
            let high = data[(i + half) * s + c];
            col_buf[i * 2] = (low + high) * INV_SQRT2;
            col_buf[i * 2 + 1] = (low - high) * INV_SQRT2;
        }
        for r in 0..n {
            data[r * s + c] = col_buf[r];
        }
    }

    // Inverse row Haar
    let mut tmp = vec![0.0f32; n];
    for r in 0..n {
        for i in 0..half {
            let low = data[r * s + i];
            let high = data[r * s + i + half];
            tmp[i * 2] = (low + high) * INV_SQRT2;
            tmp[i * 2 + 1] = (low - high) * INV_SQRT2;
        }
        for c in 0..n {
            data[r * s + c] = tmp[c];
        }
    }
}

/// CPU reference: full 2-level Haar forward (16×16 → level 0 → level 1 on 8×8 LL)
fn cpu_haar_forward_2level(block: &mut [f32]) {
    cpu_haar_forward_level(block, 16, 16); // Level 0: full 16×16
    cpu_haar_forward_level(block, 16, 8);  // Level 1: top-left 8×8
}

/// CPU reference: full 2-level Haar inverse
fn cpu_haar_inverse_2level(block: &mut [f32]) {
    cpu_haar_inverse_level(block, 16, 8);  // Level 1 inverse: top-left 8×8
    cpu_haar_inverse_level(block, 16, 16); // Level 0 inverse: full 16×16
}

/// Diagnostic: compare GPU Haar output against CPU reference for a single 16×16 block.
pub fn diagnose_haar(ctx: &GpuContext) {
    println!("\n--- Haar Diagnostic (single 16×16 block) ---\n");

    // Create a test block with a known pattern (gradient)
    let mut test_input = vec![0.0f32; 256];
    for r in 0..16u32 {
        for c in 0..16u32 {
            test_input[(r * 16 + c) as usize] = (r * 16 + c) as f32;
        }
    }

    let block_transform = BlockTransform::new(ctx);
    let input_buf = create_storage_buf(ctx, &test_input);
    let output_buf = create_empty_buf(ctx, 256);
    let inv_output_buf = create_empty_buf(ctx, 256);

    // ===== Test 1-level roundtrip =====
    println!("  == 1-level Haar roundtrip ==");
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("haar_1l_fwd"),
        });
        block_transform.dispatch_with_levels(
            ctx, &mut enc, &input_buf, &output_buf, 16, 16, true,
            BlockTransformType::HaarBlock, 1,
        );
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
    }
    let gpu_fwd_1l = readback(ctx, &output_buf, 256);

    let mut cpu_fwd_1l = test_input.clone();
    cpu_haar_forward_level(&mut cpu_fwd_1l, 16, 16);

    let fwd_1l_diff = max_abs_diff(&gpu_fwd_1l, &cpu_fwd_1l);
    println!("    1L forward GPU vs CPU max diff: {:.6}", fwd_1l_diff);

    // 1-level inverse
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("haar_1l_inv"),
        });
        block_transform.dispatch_with_levels(
            ctx, &mut enc, &output_buf, &inv_output_buf, 16, 16, false,
            BlockTransformType::HaarBlock, 1,
        );
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
    }
    let gpu_rt_1l = readback(ctx, &inv_output_buf, 256);
    let rt_1l_diff = max_abs_diff(&gpu_rt_1l, &test_input);
    println!("    1L roundtrip max diff: {:.6}", rt_1l_diff);

    // ===== Test 2-level roundtrip =====
    println!("  == 2-level Haar roundtrip ==");
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("haar_2l_fwd"),
        });
        block_transform.dispatch_with_levels(
            ctx, &mut enc, &input_buf, &output_buf, 16, 16, true,
            BlockTransformType::HaarBlock, 2,
        );
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
    }
    let gpu_fwd_2l = readback(ctx, &output_buf, 256);

    let mut cpu_fwd_2l = test_input.clone();
    cpu_haar_forward_2level(&mut cpu_fwd_2l);
    let fwd_2l_diff = max_abs_diff(&gpu_fwd_2l, &cpu_fwd_2l);
    println!("    2L forward GPU vs CPU max diff: {:.6}", fwd_2l_diff);

    // 2-level inverse
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("haar_2l_inv"),
        });
        block_transform.dispatch_with_levels(
            ctx, &mut enc, &output_buf, &inv_output_buf, 16, 16, false,
            BlockTransformType::HaarBlock, 2,
        );
        ctx.queue.submit(Some(enc.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
    }
    let gpu_rt_2l = readback(ctx, &inv_output_buf, 256);
    let rt_2l_diff = max_abs_diff(&gpu_rt_2l, &test_input);
    println!("    2L roundtrip max diff: {:.6}", rt_2l_diff);

    // If 1-level works but 2-level doesn't, isolate further:
    // Run 2-level forward, apply CPU level-1 inverse, check if that matches 1-level forward
    if rt_1l_diff < 0.01 && rt_2l_diff > 0.01 {
        println!("\n  == Isolating level 1 inverse bug ==");

        // GPU 2-level forward output is in gpu_fwd_2l (confirmed correct)
        // Apply CPU level 1 inverse to get what should be 1-level forward output
        let mut after_cpu_l1_inv = gpu_fwd_2l.clone();
        cpu_haar_inverse_level(&mut after_cpu_l1_inv, 16, 8);
        let l1_inv_diff = max_abs_diff(&after_cpu_l1_inv, &gpu_fwd_1l);
        println!("    CPU L1-inv of 2L-fwd vs 1L-fwd max diff: {:.6}", l1_inv_diff);

        // Now test: upload 2-level forward, run GPU inverse with levels=2
        // and compare with: CPU level-1-inverse followed by GPU level-0-inverse
        // We already have the GPU 2L roundtrip result (gpu_rt_2l), which is wrong.
        // Let's see what the GPU produces for individual coefficient patterns.

        // Test with only DC coefficient nonzero
        let mut dc_only = vec![0.0f32; 256];
        dc_only[0] = 100.0; // LL2 DC
        let dc_buf = create_storage_buf(ctx, &dc_only);
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("haar_dc_inv"),
            });
            block_transform.dispatch_with_levels(
                ctx, &mut enc, &dc_buf, &inv_output_buf, 16, 16, false,
                BlockTransformType::HaarBlock, 2,
            );
            ctx.queue.submit(Some(enc.finish()));
            ctx.device.poll(wgpu::Maintain::Wait);
        }
        let gpu_dc_inv = readback(ctx, &inv_output_buf, 256);

        let mut cpu_dc_inv = dc_only.clone();
        cpu_haar_inverse_2level(&mut cpu_dc_inv);

        println!("    DC-only inverse test:");
        println!("      GPU[0,0]={:.4}, CPU[0,0]={:.4}", gpu_dc_inv[0], cpu_dc_inv[0]);
        println!("      GPU[0,1]={:.4}, CPU[0,1]={:.4}", gpu_dc_inv[1], cpu_dc_inv[1]);
        println!("      GPU[1,0]={:.4}, CPU[1,0]={:.4}", gpu_dc_inv[16], cpu_dc_inv[16]);
        let dc_inv_diff = max_abs_diff(&gpu_dc_inv, &cpu_dc_inv);
        println!("      Max diff: {:.6}", dc_inv_diff);

        // Print the 8×8 LL region after GPU 2L inverse to see what level 1 produced
        println!("\n    GPU 2L inverse, first 8×8 region (should be LL0 reconstruction):");
        for r in 0..4 {
            print!("      row {}: ", r);
            for c in 0..8 {
                print!("{:8.3} ", gpu_rt_2l[r * 16 + c]);
            }
            println!();
        }
    }

    // Print first row of 2L results for visual inspection
    println!("\n  First row comparison (2-level):");
    print!("    Input:   ");
    for c in 0..16 { print!("{:7.2} ", test_input[c]); }
    println!();
    print!("    GPU fwd: ");
    for c in 0..16 { print!("{:7.2} ", gpu_fwd_2l[c]); }
    println!();
    print!("    CPU fwd: ");
    for c in 0..16 { print!("{:7.2} ", cpu_fwd_2l[c]); }
    println!();
    print!("    GPU inv: ");
    for c in 0..16 { print!("{:7.2} ", gpu_rt_2l[c]); }
    println!();

    // CPU roundtrip sanity check
    let mut cpu_inv = cpu_fwd_2l.clone();
    cpu_haar_inverse_2level(&mut cpu_inv);
    let cpu_rt_diff = max_abs_diff(&cpu_inv, &test_input);
    println!("    CPU 2L roundtrip max diff: {:.10}", cpu_rt_diff);
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}
