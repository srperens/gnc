//! Adaptive quantization: per-block variance analysis on the wavelet LL subband.
//!
//! Pipeline:
//! 1. GPU compute: after wavelet transform, measure per-block variance on the LL
//!    (lowpass) subband within each tile. The LL subband at level L occupies the
//!    top-left corner of each tile: [0..tile_size/(2^L), 0..tile_size/(2^L)].
//! 2. GPU normalize: convert variance -> log-domain weights, normalize to average 1.0,
//!    smooth with 3x3 box filter (via WeightMapNormalizer shader).
//!
//! The weight map is indexed by tile + LL-block position. The quantizer maps each
//! wavelet coefficient back to the correct LL-block weight using subband-aware
//! coordinate conversion.

use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

/// Block size for variance analysis in LL-subband coordinates.
/// With tile_size=256 and 3 wavelet levels, LL is 32x32 per tile.
/// An 8x8 LL block means 4x4 = 16 blocks per tile, each covering
/// a 64x64 spatial region. This provides reasonable spatial granularity.
pub const AQ_LL_BLOCK_SIZE: u32 = 8;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VarianceParams {
    width: u32,
    height: u32,
    tile_size: u32,
    num_levels: u32,
    ll_size: u32,
    ll_block_size: u32,
    ll_blocks_per_tile_x: u32,
    ll_blocks_per_tile_y: u32,
    tiles_x: u32,
    total_blocks: u32,
    global_blocks_x: u32,
    _pad: u32,
}

/// Compute LL-subband block layout for a given resolution and wavelet config.
///
/// Returns (ll_blocks_per_tile_x, ll_blocks_per_tile_y, total_blocks, ll_size, tiles_x, tiles_y).
pub fn ll_block_dims(
    padded_w: u32,
    padded_h: u32,
    tile_size: u32,
    num_levels: u32,
) -> (u32, u32, u32, u32, u32, u32) {
    let ll_size = tile_size >> num_levels; // tile_size / 2^num_levels
    let ll_block_size = AQ_LL_BLOCK_SIZE.min(ll_size); // Don't exceed LL size
    let ll_blocks_per_tile_x = ll_size.div_ceil(ll_block_size);
    let ll_blocks_per_tile_y = ll_size.div_ceil(ll_block_size);
    let tiles_x = padded_w / tile_size;
    let tiles_y = padded_h / tile_size;
    let total_tiles = tiles_x * tiles_y;
    let blocks_per_tile = ll_blocks_per_tile_x * ll_blocks_per_tile_y;
    let total_blocks = total_tiles * blocks_per_tile;
    (
        ll_blocks_per_tile_x,
        ll_blocks_per_tile_y,
        total_blocks,
        ll_size,
        tiles_x,
        tiles_y,
    )
}

/// Compute the weight map dimensions for the normalized weight output.
/// The weight map has one entry per LL-block across all tiles.
///
/// Returns (total_blocks, ll_blocks_per_tile_x, ll_blocks_per_tile_y, tiles_x).
pub fn weight_map_dims(
    padded_w: u32,
    padded_h: u32,
    tile_size: u32,
    num_levels: u32,
) -> (u32, u32, u32, u32) {
    let (lbx, lby, total, _, tx, _) = ll_block_dims(padded_w, padded_h, tile_size, num_levels);
    (total, lbx, lby, tx)
}

/// GPU pipeline for computing per-block variance from the LL wavelet subband.
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

    /// Dispatch variance computation on the LL subband of the wavelet-domain buffer.
    ///
    /// `wavelet_buf`: GPU buffer containing wavelet-domain coefficients (after forward transform)
    /// `variance_buf`: output GPU buffer for the variance map (one f32 per LL-block)
    /// `width`, `height`: padded plane dimensions
    /// `tile_size`: tile size in pixels
    /// `num_levels`: number of wavelet decomposition levels
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        wavelet_buf: &wgpu::Buffer,
        variance_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
        tile_size: u32,
        num_levels: u32,
    ) {
        let (ll_blocks_per_tile_x, ll_blocks_per_tile_y, total_blocks, ll_size, tiles_x, _) =
            ll_block_dims(width, height, tile_size, num_levels);
        let ll_block_size = AQ_LL_BLOCK_SIZE.min(ll_size);
        let global_blocks_x = ll_blocks_per_tile_x * tiles_x;

        let params = VarianceParams {
            width,
            height,
            tile_size,
            num_levels,
            ll_size,
            ll_block_size,
            ll_blocks_per_tile_x,
            ll_blocks_per_tile_y,
            tiles_x,
            total_blocks,
            global_blocks_x,
            _pad: 0,
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
                    resource: wavelet_buf.as_entire_binding(),
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
        // One workgroup per LL-block
        pass.dispatch_workgroups(total_blocks, 1, 1);
    }
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

/// GPU pipeline for normalizing the variance map into a weight map.
///
/// Replaces the CPU `compute_weight_map()` for the hot path.
/// Single-workgroup dispatch handles all blocks.
pub struct WeightMapNormalizer {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WeightNormParams {
    blocks_x: u32,
    blocks_y: u32,
    total_blocks: u32,
    aq_strength: f32,
}

impl WeightMapNormalizer {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("weight_map_normalize"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/weight_map_normalize.wgsl").into(),
                ),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("weight_norm_bgl"),
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
                        // binding 1: variance (read)
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
                        // binding 2: scratch (rw)
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
                        // binding 3: weight_map (rw)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
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
                label: Some("weight_norm_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("weight_norm_pipeline"),
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

    /// Dispatch weight map normalization on the GPU.
    ///
    /// Reads raw variance from `variance_buf`, writes normalized + smoothed
    /// weights to `weight_map_buf`. `scratch_buf` is used as intermediate storage.
    ///
    /// `blocks_x` and `blocks_y` are the per-tile LL-block counts. The normalizer
    /// treats the flat array of all LL-blocks as a 2D grid for smoothing purposes.
    /// For multi-tile layouts, `blocks_x` should be `ll_blocks_per_tile_x * tiles_x`
    /// and `blocks_y` should be `ll_blocks_per_tile_y * tiles_y` for correct smoothing.
    /// However, since smoothing should NOT cross tile boundaries, we pass the per-tile
    /// dimensions and let the shader smooth within each tile's block grid.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        variance_buf: &wgpu::Buffer,
        scratch_buf: &wgpu::Buffer,
        weight_map_buf: &wgpu::Buffer,
        blocks_x: u32,
        blocks_y: u32,
        total_blocks: u32,
        aq_strength: f32,
    ) {
        let params = WeightNormParams {
            blocks_x,
            blocks_y,
            total_blocks,
            aq_strength,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("weight_norm_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("weight_norm_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: variance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scratch_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: weight_map_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("weight_norm_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
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
    fn test_ll_block_dims() {
        // 1920x1088 padded, tile_size=256, 3 levels
        // LL size = 256/8 = 32. ll_block_size = 8. ll_blocks_per_tile = 4x4 = 16.
        // tiles_x = 1920/256 = 7.5 -> need padded: 8 tiles. Let's use 2048x1280.
        let (lbx, lby, total, ll_size, tx, ty) = ll_block_dims(2048, 1280, 256, 3);
        assert_eq!(ll_size, 32);
        assert_eq!(lbx, 4);
        assert_eq!(lby, 4);
        assert_eq!(tx, 8);
        assert_eq!(ty, 5);
        assert_eq!(total, 8 * 5 * 16);
    }

    #[test]
    fn test_weight_map_dims_fn() {
        let (total, lbx, lby, tx) = weight_map_dims(2048, 1280, 256, 3);
        assert_eq!(lbx, 4);
        assert_eq!(lby, 4);
        assert_eq!(tx, 8);
        assert_eq!(total, 8 * 5 * 16);
    }
}
