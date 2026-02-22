//! Chroma-from-Luma (CfL) prediction.
//!
//! Predicts chroma wavelet coefficients from reconstructed luma using a
//! per-tile per-subband linear scaling factor: `chroma ≈ alpha * luma`.
//! The encoder transmits only the residual `chroma - alpha * luma`, which
//! has lower energy and entropy than raw chroma.
//!
//! CPU-side: alpha computation, quantization, and forward prediction (residual).
//! GPU-side: `CflPredictor` dispatches the inverse prediction shader for decoding.

use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

// ---------------------------------------------------------------------------
// Alpha quantization: map f32 in [-2, 2] to u8 [0, 255] and back
// ---------------------------------------------------------------------------

const ALPHA_MIN: f32 = -2.0;
const ALPHA_MAX: f32 = 2.0;
const ALPHA_RANGE: f32 = ALPHA_MAX - ALPHA_MIN; // 4.0

pub fn quantize_alpha(alpha: f32) -> u8 {
    let clamped = alpha.clamp(ALPHA_MIN, ALPHA_MAX);
    let normalized = (clamped - ALPHA_MIN) / ALPHA_RANGE; // [0, 1]
    (normalized * 255.0).round() as u8
}

pub fn dequantize_alpha(q: u8) -> f32 {
    (q as f32 / 255.0) * ALPHA_RANGE + ALPHA_MIN
}

// ---------------------------------------------------------------------------
// Subband index computation (mirrors quantize.wgsl logic)
// ---------------------------------------------------------------------------

/// Compute the flat subband index for a tile-local position (lx, ly).
///
/// Layout matches the WGSL `compute_subband_index` in `quantize.wgsl`:
///   Index 0 = LL (DC)
///   Index 1 + level*3 + 0 = LH (bottom-left)
///   Index 1 + level*3 + 1 = HL (top-right)
///   Index 1 + level*3 + 2 = HH (bottom-right)
pub fn compute_subband_index(lx: u32, ly: u32, tile_size: u32, num_levels: u32) -> usize {
    let mut region = tile_size;
    for level in 0..num_levels {
        let half = region / 2;
        let in_right = lx >= half;
        let in_bottom = ly >= half;
        if in_right || in_bottom {
            if in_right && in_bottom {
                return (1 + level * 3 + 2) as usize; // HH
            } else if in_right {
                return (1 + level * 3 + 1) as usize; // HL
            } else {
                return (1 + level * 3) as usize; // LH
            }
        }
        region = half;
    }
    0 // LL
}

/// Number of subbands for a given number of wavelet decomposition levels.
pub fn num_subbands(num_levels: u32) -> u32 {
    1 + 3 * num_levels
}

// ---------------------------------------------------------------------------
// Per-tile per-subband alpha computation (least-squares)
// ---------------------------------------------------------------------------

/// Compute CfL alpha values for all tiles and subbands.
///
/// For each tile t and subband s, computes:
///   `alpha[t][s] = sum(Y[i]*C[i]) / sum(Y[i]*Y[i])`
/// where i ranges over coefficients in tile t, subband s.
///
/// Returns flat Vec of f32 alphas, layout: [tile0_sb0, tile0_sb1, ..., tile1_sb0, ...]
pub fn compute_cfl_alphas(
    recon_y: &[f32],
    chroma_wavelet: &[f32],
    width: u32,
    height: u32,
    tile_size: u32,
    num_levels: u32,
) -> Vec<f32> {
    let tiles_x = width.div_ceil(tile_size);
    let tiles_y = height.div_ceil(tile_size);
    let nsb = num_subbands(num_levels) as usize;
    let total_tiles = (tiles_x * tiles_y) as usize;

    let mut alphas = vec![0.0f32; total_tiles * nsb];

    // Accumulators per tile × subband
    let mut sum_yc = vec![0.0f64; total_tiles * nsb];
    let mut sum_yy = vec![0.0f64; total_tiles * nsb];

    for gy in 0..height {
        for gx in 0..width {
            let tx = gx / tile_size;
            let ty = gy / tile_size;
            let tile_idx = (ty * tiles_x + tx) as usize;
            let lx = gx % tile_size;
            let ly = gy % tile_size;
            let sb = compute_subband_index(lx, ly, tile_size, num_levels);

            let idx = (gy * width + gx) as usize;
            let y = recon_y[idx] as f64;
            let c = chroma_wavelet[idx] as f64;

            let k = tile_idx * nsb + sb;
            sum_yc[k] += y * c;
            sum_yy[k] += y * y;
        }
    }

    for k in 0..total_tiles * nsb {
        alphas[k] = if sum_yy[k] > 1e-10 {
            (sum_yc[k] / sum_yy[k]) as f32
        } else {
            0.0
        };
    }

    alphas
}

// ---------------------------------------------------------------------------
// CPU-side forward prediction (residual = chroma - alpha * luma)
// ---------------------------------------------------------------------------

/// Compute CfL residual on CPU: `residual[i] = chroma[i] - alpha[tile][sb] * luma[i]`
///
/// `alphas_f32` should be dequantized alphas (from quantize→dequantize roundtrip)
/// to match what the decoder will use.
pub fn apply_cfl_predict_cpu(
    chroma: &[f32],
    recon_y: &[f32],
    alphas_f32: &[f32],
    width: u32,
    height: u32,
    tile_size: u32,
    num_levels: u32,
) -> Vec<f32> {
    let tiles_x = width.div_ceil(tile_size);
    let nsb = num_subbands(num_levels) as usize;
    let count = (width * height) as usize;
    let mut residual = vec![0.0f32; count];

    for gy in 0..height {
        for gx in 0..width {
            let tx = gx / tile_size;
            let ty = gy / tile_size;
            let tile_idx = (ty * tiles_x + tx) as usize;
            let lx = gx % tile_size;
            let ly = gy % tile_size;
            let sb = compute_subband_index(lx, ly, tile_size, num_levels);

            let idx = (gy * width + gx) as usize;
            let alpha = alphas_f32[tile_idx * nsb + sb];
            residual[idx] = chroma[idx] - alpha * recon_y[idx];
        }
    }

    residual
}

// ---------------------------------------------------------------------------
// GPU-side inverse prediction (decoder): CflPredictor
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CflParams {
    total_count: u32,
    width: u32,
    height: u32,
    tile_size: u32,
    num_levels: u32,
    num_subbands: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU pipeline for CfL inverse prediction (used by decoder).
///
/// Computes `output[i] = residual[i] + alpha[tile][sb] * luma_ref[i]`
pub struct CflPredictor {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl CflPredictor {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cfl_predict"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/cfl_predict.wgsl").into(),
                ),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("cfl_bgl"),
                    entries: &[
                        // binding 0: uniform params
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 1: residual (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 2: luma_ref (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 3: alphas f32 array (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 4: output (write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cfl_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cfl_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Dispatch inverse CfL prediction on the GPU.
    ///
    /// Computes `output[i] = residual[i] + alpha[tile][sb] * luma_ref[i]`
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_inverse(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        residual_buf: &wgpu::Buffer,
        luma_ref_buf: &wgpu::Buffer,
        alpha_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        total_count: u32,
        width: u32,
        height: u32,
        tile_size: u32,
        num_levels: u32,
    ) {
        let nsb = num_subbands(num_levels);
        let params = CflParams {
            total_count,
            width,
            height,
            tile_size,
            num_levels,
            num_subbands: nsb,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cfl_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cfl_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: residual_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: luma_ref_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: alpha_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = total_count.div_ceil(256);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cfl_inverse_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

/// Upload dequantized alpha values as a GPU f32 buffer for the CfL shader.
pub fn upload_alpha_buffer(ctx: &GpuContext, alphas_f32: &[f32]) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cfl_alphas"),
            contents: bytemuck::cast_slice(alphas_f32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_quantize_roundtrip() {
        // Test exact center
        assert_eq!(quantize_alpha(0.0), 128); // (0 - (-2)) / 4 * 255 = 127.5 → 128
        assert_eq!(dequantize_alpha(128), 0.0 + (128.0 / 255.0) * 4.0 - 2.0);

        // Test boundaries
        assert_eq!(quantize_alpha(-2.0), 0);
        assert_eq!(quantize_alpha(2.0), 255);

        // Roundtrip precision: max error should be ALPHA_RANGE / 255 / 2 ≈ 0.0078
        for i in 0..=255u8 {
            let alpha = dequantize_alpha(i);
            let requantized = quantize_alpha(alpha);
            assert!(
                (requantized as i16 - i as i16).unsigned_abs() <= 1,
                "roundtrip failed for {}: {} -> {} -> {}",
                i,
                alpha,
                quantize_alpha(alpha),
                requantized
            );
        }
    }

    #[test]
    fn alpha_clamp() {
        assert_eq!(quantize_alpha(-5.0), 0);
        assert_eq!(quantize_alpha(5.0), 255);
    }

    #[test]
    fn subband_index_ll() {
        // Position (0,0) should always be in the innermost LL
        assert_eq!(compute_subband_index(0, 0, 256, 3), 0);
        assert_eq!(compute_subband_index(0, 0, 128, 2), 0);
    }

    #[test]
    fn subband_index_hh_level0() {
        // Bottom-right quadrant of outermost level = HH at level 0
        // For tile_size=256, level 0: half=128
        // (128, 128) is in_right && in_bottom → HH = 1 + 0*3 + 2 = 3
        assert_eq!(compute_subband_index(128, 128, 256, 3), 3);
    }

    #[test]
    fn subband_index_detail_levels() {
        // tile_size=8, 2 levels:
        // Level 0: half=4. (4,0) → HL = 1+0*3+1 = 2
        assert_eq!(compute_subband_index(4, 0, 8, 2), 2);
        // Level 0: (0,4) → LH = 1+0*3+0 = 1
        assert_eq!(compute_subband_index(0, 4, 8, 2), 1);
        // Level 1: (2,0) in [0..4) → recurse. half=2. (2,0) → HL = 1+1*3+1 = 5
        assert_eq!(compute_subband_index(2, 0, 8, 2), 5);
        // LL at (0,0)
        assert_eq!(compute_subband_index(0, 0, 8, 2), 0);
    }

    #[test]
    fn cfl_alpha_simple() {
        // 8x8 tile, 1 level → 4 subbands (LL, LH, HL, HH)
        let tile_size = 8u32;
        let num_levels = 1u32;
        let w = tile_size;
        let h = tile_size;
        let n = (w * h) as usize;

        // Luma = constant 2.0 everywhere, chroma = constant 1.0
        // alpha should be 0.5 for all subbands
        let recon_y = vec![2.0f32; n];
        let chroma = vec![1.0f32; n];

        let alphas = compute_cfl_alphas(&recon_y, &chroma, w, h, tile_size, num_levels);
        assert_eq!(alphas.len(), 4); // 1 tile × 4 subbands
        for &a in &alphas {
            assert!((a - 0.5).abs() < 1e-6, "expected 0.5, got {}", a);
        }
    }

    #[test]
    fn cfl_predict_roundtrip() {
        let tile_size = 8u32;
        let num_levels = 1u32;
        let w = tile_size;
        let h = tile_size;
        let n = (w * h) as usize;

        // Create some structured data
        let recon_y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let chroma: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 + 1.0).collect();

        let alphas = compute_cfl_alphas(&recon_y, &chroma, w, h, tile_size, num_levels);

        // Quantize then dequantize to simulate what encoder/decoder actually uses
        let q_alphas: Vec<u8> = alphas.iter().map(|&a| quantize_alpha(a)).collect();
        let dq_alphas: Vec<f32> = q_alphas.iter().map(|&q| dequantize_alpha(q)).collect();

        let residual =
            apply_cfl_predict_cpu(&chroma, &recon_y, &dq_alphas, w, h, tile_size, num_levels);

        // Inverse: reconstructed = residual + alpha * luma
        let tiles_x = (w + tile_size - 1) / tile_size;
        let nsb = num_subbands(num_levels) as usize;
        let mut reconstructed = vec![0.0f32; n];
        for gy in 0..h {
            for gx in 0..w {
                let tx = gx / tile_size;
                let ty = gy / tile_size;
                let tile_idx = (ty * tiles_x + tx) as usize;
                let lx = gx % tile_size;
                let ly = gy % tile_size;
                let sb = compute_subband_index(lx, ly, tile_size, num_levels);
                let idx = (gy * w + gx) as usize;
                let alpha = dq_alphas[tile_idx * nsb + sb];
                reconstructed[idx] = residual[idx] + alpha * recon_y[idx];
            }
        }

        // Check reconstruction matches original chroma within alpha quantization error
        for i in 0..n {
            let err = (reconstructed[i] - chroma[i]).abs();
            assert!(
                err < 0.1,
                "pixel {} reconstruction error {}: recon={}, orig={}",
                i,
                err,
                reconstructed[i],
                chroma[i]
            );
        }
    }
}
