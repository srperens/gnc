//! MED / LOCO-I prediction as an alternative to the wavelet on the lossless path.
//!
//! LOSSLESS-1. GNC at q=100 loses to FFV1 by 27% and to x264 `-qp 0` by 43%, and both win the
//! same way: predict each pixel from its neighbours and entropy-code the error directly, with no
//! transform at all. An offline gate over four images put 10–26% on the table against GNC's own
//! q=100, and the wavefront the decoder needs was measured at 4.9x one pass, still 201 fps.
//!
//! This is prediction *instead of* a transform. Prediction *before* one was measured separately
//! (BUG-13) and costs 4–8% at lossless, because it hands the wavelet a harder signal — that
//! result says nothing about this one.

use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MedParams {
    width: u32,
    height: u32,
    tile_size: u32,
    _pad: u32,
}

/// Forward (parallel) and inverse (wavefront) MED prediction pipelines.
pub struct MedTransform {
    forward_pipeline: wgpu::ComputePipeline,
    inverse_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl MedTransform {
    pub fn new(ctx: &GpuContext) -> Self {
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("med_bgl"),
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
                label: Some("med_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let make = |label: &str, source: &str| {
            let shader = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        Self {
            forward_pipeline: make("med_predict", include_str!("../shaders/med_predict.wgsl")),
            inverse_pipeline: make(
                "med_reconstruct",
                include_str!("../shaders/med_reconstruct.wgsl"),
            ),
            bind_group_layout,
        }
    }

    #[allow(clippy::too_many_arguments)] // GPU dispatch: buffers, dimensions, direction
    fn dispatch(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        width: u32,
        height: u32,
        tile_size: u32,
        forward: bool,
    ) {
        let params = MedParams {
            width,
            height,
            tile_size,
            _pad: 0,
        };
        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("med_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("med_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        // Forward: one thread per pixel. Inverse: one workgroup per tile, marching diagonals.
        let workgroups = if forward {
            (width * height).div_ceil(256)
        } else {
            width.div_ceil(tile_size) * height.div_ceil(tile_size)
        };

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(if forward { "med_forward" } else { "med_inverse" }),
            timestamp_writes: None,
        });
        pass.set_pipeline(if forward {
            &self.forward_pipeline
        } else {
            &self.inverse_pipeline
        });
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Pixels → prediction residuals. Fully parallel.
    #[allow(clippy::too_many_arguments)] // GPU dispatch
    pub fn forward(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        width: u32,
        height: u32,
        tile_size: u32,
    ) {
        self.dispatch(ctx, encoder, input, output, width, height, tile_size, true);
    }

    /// Prediction residuals → pixels. Wavefront; one workgroup per tile.
    #[allow(clippy::too_many_arguments)] // GPU dispatch
    pub fn inverse(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        width: u32,
        height: u32,
        tile_size: u32,
    ) {
        self.dispatch(ctx, encoder, input, output, width, height, tile_size, false);
    }
}

/// CPU reference, used by the roundtrip test to pin the GPU implementation to an exact
/// definition rather than to itself.
#[cfg(test)]
pub fn med_forward_cpu(src: &[f32], width: usize, height: usize, tile_size: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; src.len()];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let (lx, ly) = (x % tile_size, y % tile_size);
            let p = if lx == 0 && ly == 0 {
                0.0
            } else if ly == 0 {
                src[idx - 1]
            } else if lx == 0 {
                src[idx - width]
            } else {
                med(src[idx - 1], src[idx - width], src[idx - width - 1])
            };
            dst[idx] = src[idx] - p;
        }
    }
    dst
}

#[cfg(test)]
fn med(a: f32, b: f32, c: f32) -> f32 {
    let (mx, mn) = (a.max(b), a.min(b));
    if c >= mx {
        mn
    } else if c <= mn {
        mx
    } else {
        a + b - c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inverse of the CPU forward must be exact, including at tile boundaries where
    /// prediction resets. This pins the definition the shaders implement.
    #[test]
    fn med_cpu_roundtrip_is_exact() {
        let (w, h, ts) = (64usize, 48usize, 16usize);
        let src: Vec<f32> = (0..w * h)
            .map(|i| ((i * 37 + (i / w) * 11) % 256) as f32)
            .collect();
        let res = med_forward_cpu(&src, w, h, ts);

        let mut out = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let (lx, ly) = (x % ts, y % ts);
                let p = if lx == 0 && ly == 0 {
                    0.0
                } else if ly == 0 {
                    out[idx - 1]
                } else if lx == 0 {
                    out[idx - w]
                } else {
                    med(out[idx - 1], out[idx - w], out[idx - w - 1])
                };
                out[idx] = res[idx] + p;
            }
        }
        assert_eq!(out, src, "MED roundtrip must be bit-exact");
    }
}
