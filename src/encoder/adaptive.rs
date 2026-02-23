//! Adaptive quantization: per-block variance analysis and weight map generation.
//!
//! Pipeline:
//! 1. GPU compute: measure per-32x32-block variance from the Y (luma) plane
//! 2. CPU: convert raw variance to log-domain weights, normalize to average 1.0
//!
//! The weight map is small (~2560 f32s for 1080p), so the CPU normalization step
//! is negligible compared to the GPU work.

use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

/// Block size used for variance analysis (matches well with 256x256 tile structure)
pub const AQ_BLOCK_SIZE: u32 = 32;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VarianceParams {
    width: u32,
    height: u32,
    blocks_x: u32,
    total_blocks: u32,
}

/// GPU pipeline for computing per-block variance from the Y (luma) plane.
pub struct VarianceAnalyzer {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl VarianceAnalyzer {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("variance_map"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/variance_map.wgsl").into(),
                ),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("variance_bgl"),
                    entries: &[
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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
                label: Some("variance_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("variance_pipeline"),
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

    /// Dispatch variance computation on the GPU.
    ///
    /// `y_plane_buf`: GPU buffer containing the Y (luma) plane as f32 values
    /// `variance_buf`: output GPU buffer for the variance map (one f32 per block)
    /// `width`, `height`: padded plane dimensions
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        y_plane_buf: &wgpu::Buffer,
        variance_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) {
        let blocks_x = (width + AQ_BLOCK_SIZE - 1) / AQ_BLOCK_SIZE;
        let blocks_y = (height + AQ_BLOCK_SIZE - 1) / AQ_BLOCK_SIZE;
        let total_blocks = blocks_x * blocks_y;

        let params = VarianceParams {
            width,
            height,
            blocks_x,
            total_blocks,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("variance_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("variance_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y_plane_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: variance_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("variance_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per block, each with 1024 threads (32x32)
        pass.dispatch_workgroups(total_blocks, 1, 1);
    }
}

/// Compute dimensions for the variance/weight map.
pub fn weight_map_dims(padded_w: u32, padded_h: u32) -> (u32, u32, u32) {
    let blocks_x = (padded_w + AQ_BLOCK_SIZE - 1) / AQ_BLOCK_SIZE;
    let blocks_y = (padded_h + AQ_BLOCK_SIZE - 1) / AQ_BLOCK_SIZE;
    (blocks_x, blocks_y, blocks_x * blocks_y)
}

/// Convert raw variance values to a normalized weight map on the CPU.
///
/// The mapping uses: `weight = clamp(base + scale * log2(1 + variance), min, max)`
/// where base/scale are derived from `aq_strength`.
///
/// Low variance (smooth regions) -> weight > 1.0 (coarser quantization, save bits)
/// High variance (textured regions) -> weight < 1.0 (finer quantization, preserve detail)
///
/// The weights are normalized so their average across the frame is 1.0,
/// preserving the target bitrate.
///
/// Finally, a simple 3x3 averaging is applied to smooth block boundaries.
pub fn compute_weight_map(
    raw_variance: &[f32],
    blocks_x: u32,
    blocks_y: u32,
    aq_strength: f32,
) -> Vec<f32> {
    let total = (blocks_x * blocks_y) as usize;
    assert_eq!(raw_variance.len(), total);

    if aq_strength <= 0.0 || total == 0 {
        return vec![1.0; total];
    }

    // Map variance to raw weights using log-domain mapping.
    // Higher variance -> lower weight (finer quantization).
    // We invert the mapping: base is the "smooth region" multiplier (>1),
    // and we subtract the log-variance contribution.
    let scale = 0.15 * aq_strength;
    let min_weight = 0.5_f32;
    let max_weight = 2.0_f32;

    let mut weights: Vec<f32> = raw_variance
        .iter()
        .map(|&var| {
            // log2(1 + var) maps [0, inf) -> [0, inf) monotonically
            let log_var = (1.0 + var).log2();
            // Invert: high variance -> small weight
            let w = 1.0 + scale * (8.0 - log_var);
            w.clamp(min_weight, max_weight)
        })
        .collect();

    // Normalize weights to average 1.0
    let mean: f32 = weights.iter().sum::<f32>() / total as f32;
    if mean > 0.0 {
        for w in &mut weights {
            *w /= mean;
        }
    }

    // Re-clamp after normalization
    for w in &mut weights {
        *w = w.clamp(min_weight, max_weight);
    }

    // Smooth the weight map with a 3x3 box filter to reduce block boundary artifacts.
    // Skip smoothing when either dimension is < 3 — the kernel can't preserve
    // spatial variation on grids that small.
    if blocks_x >= 3 && blocks_y >= 3 {
        smooth_weight_map(&weights, blocks_x, blocks_y)
    } else {
        weights
    }
}

/// 3x3 box-filter smoothing of the weight map.
/// Reduces visible block boundaries in the adaptive quantization.
fn smooth_weight_map(weights: &[f32], blocks_x: u32, blocks_y: u32) -> Vec<f32> {
    let bx = blocks_x as usize;
    let by = blocks_y as usize;
    let mut smoothed = vec![0.0f32; bx * by];

    for y in 0..by {
        for x in 0..bx {
            let mut sum = 0.0f32;
            let mut count = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < bx as i32 && ny >= 0 && ny < by as i32 {
                        sum += weights[ny as usize * bx + nx as usize];
                        count += 1.0;
                    }
                }
            }
            smoothed[y * bx + x] = sum / count;
        }
    }

    // Re-normalize after smoothing to maintain average ~1.0
    let total = smoothed.len() as f32;
    let mean: f32 = smoothed.iter().sum::<f32>() / total;
    if mean > 0.0 {
        for w in &mut smoothed {
            *w /= mean;
        }
    }

    smoothed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_map_all_zero_variance() {
        // Uniform image: all variance = 0 -> all weights = 1.0
        let variance = vec![0.0; 64];
        let weights = compute_weight_map(&variance, 8, 8, 1.0);
        for &w in &weights {
            assert!(
                (w - 1.0).abs() < 0.1,
                "Expected ~1.0 for uniform variance, got {}",
                w
            );
        }
    }

    #[test]
    fn test_weight_map_disabled() {
        // aq_strength = 0 -> all weights = 1.0 regardless of variance
        let variance = vec![100.0, 0.0, 50.0, 200.0];
        let weights = compute_weight_map(&variance, 2, 2, 0.0);
        for &w in &weights {
            assert_eq!(w, 1.0);
        }
    }

    #[test]
    fn test_weight_map_smooth_gets_higher_weight() {
        // Smooth region (low var) should get higher weight (coarser quant)
        // Textured region (high var) should get lower weight (finer quant)
        let variance = vec![0.0, 10000.0];
        let weights = compute_weight_map(&variance, 2, 1, 1.0);
        assert!(
            weights[0] > weights[1],
            "Smooth region weight ({}) should be > textured region weight ({})",
            weights[0],
            weights[1]
        );
    }

    #[test]
    fn test_weight_map_dims() {
        let (bx, by, total) = weight_map_dims(1920, 1088);
        assert_eq!(bx, 60);
        assert_eq!(by, 34);
        assert_eq!(total, 60 * 34);
    }
}
