use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

pub const ME_BLOCK_SIZE: u32 = 16;
pub const ME_SEARCH_RANGE: u32 = 32;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlockMatchParams {
    width: u32,
    height: u32,
    block_size: u32,
    search_range: u32,
    blocks_x: u32,
    total_blocks: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MotionCompensateParams {
    width: u32,
    height: u32,
    block_size: u32,
    mode: u32, // 0 = forward (residual), 1 = inverse (reconstruct)
    blocks_x: u32,
    total_pixels: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU-based motion estimation and compensation.
pub struct MotionEstimator {
    match_pipeline: wgpu::ComputePipeline,
    match_bgl: wgpu::BindGroupLayout,
    compensate_pipeline: wgpu::ComputePipeline,
    compensate_bgl: wgpu::BindGroupLayout,
}

impl MotionEstimator {
    pub fn new(ctx: &GpuContext) -> Self {
        let match_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("block_match"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/block_match.wgsl").into(),
                ),
            });

        let compensate_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("motion_compensate"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/motion_compensate.wgsl").into(),
                ),
            });

        // Block match bind group layout: uniform, current_y, reference_y, mvs, sads
        let match_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("block_match_bgl"),
                    entries: &[
                        bgl_uniform(0),
                        bgl_storage_ro(1),
                        bgl_storage_ro(2),
                        bgl_storage_rw(3),
                        bgl_storage_rw(4),
                    ],
                });

        let match_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("block_match_pl"),
                bind_group_layouts: &[&match_bgl],
                push_constant_ranges: &[],
            });

        let match_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("block_match_pipeline"),
                    layout: Some(&match_pl),
                    module: &match_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Motion compensate bind group layout: uniform, input, reference, mvs, output
        let compensate_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("motion_compensate_bgl"),
                    entries: &[
                        bgl_uniform(0),
                        bgl_storage_ro(1),
                        bgl_storage_ro(2),
                        bgl_storage_ro(3),
                        bgl_storage_rw(4),
                    ],
                });

        let compensate_pl =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("motion_compensate_pl"),
                    bind_group_layouts: &[&compensate_bgl],
                    push_constant_ranges: &[],
                });

        let compensate_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("motion_compensate_pipeline"),
                    layout: Some(&compensate_pl),
                    module: &compensate_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            match_pipeline,
            match_bgl,
            compensate_pipeline,
            compensate_bgl,
        }
    }

    /// Dispatch block matching motion estimation on GPU.
    /// Returns (mv_buffer, sad_buffer). MVs are stored as i32 pairs (dx, dy).
    pub fn estimate(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        current_y: &wgpu::Buffer,
        reference_y: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let blocks_x = width / ME_BLOCK_SIZE;
        let blocks_y = height / ME_BLOCK_SIZE;
        let total_blocks = blocks_x * blocks_y;

        let params = BlockMatchParams {
            width,
            height,
            block_size: ME_BLOCK_SIZE,
            search_range: ME_SEARCH_RANGE,
            blocks_x,
            total_blocks,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("block_match_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let mv_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion_vectors"),
            size: (total_blocks as usize * 2 * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sad_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sad_values"),
            size: (total_blocks as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("block_match_bg"),
            layout: &self.match_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: current_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reference_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sad_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("block_match_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.match_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // One workgroup per block
            pass.dispatch_workgroups(total_blocks, 1, 1);
        }

        (mv_buf, sad_buf)
    }

    /// Dispatch motion compensation on GPU for a single plane.
    ///
    /// `forward=true`:  residual = current - predicted (encoder)
    /// `forward=false`: reconstructed = residual + predicted (decoder)
    #[allow(clippy::too_many_arguments)]
    pub fn compensate(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        input_plane: &wgpu::Buffer,
        reference_plane: &wgpu::Buffer,
        mv_buf: &wgpu::Buffer,
        output_plane: &wgpu::Buffer,
        width: u32,
        height: u32,
        forward: bool,
    ) {
        let blocks_x = width / ME_BLOCK_SIZE;
        let total_pixels = width * height;

        let params = MotionCompensateParams {
            width,
            height,
            block_size: ME_BLOCK_SIZE,
            mode: if forward { 0 } else { 1 },
            blocks_x,
            total_pixels,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mc_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mc_bg"),
            layout: &self.compensate_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_plane.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reference_plane.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_plane.as_entire_binding(),
                },
            ],
        });

        let workgroups = total_pixels.div_ceil(256);
        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mc_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compensate_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// Read motion vectors from GPU buffer back to CPU.
    pub fn read_motion_vectors(
        ctx: &GpuContext,
        mv_buf: &wgpu::Buffer,
        total_blocks: u32,
    ) -> Vec<[i16; 2]> {
        let count = total_blocks as usize * 2;
        let size = (count * std::mem::size_of::<i32>()) as u64;

        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mv_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_mv"),
            });
        cmd.copy_buffer_to_buffer(mv_buf, 0, &staging, 0, size);
        ctx.queue.submit(Some(cmd.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let i32_data: &[i32] = bytemuck::cast_slice(&data);

        let mut mvs = Vec::with_capacity(total_blocks as usize);
        for i in 0..total_blocks as usize {
            mvs.push([i32_data[i * 2] as i16, i32_data[i * 2 + 1] as i16]);
        }

        drop(data);
        staging.unmap();
        mvs
    }

    /// Upload motion vectors from CPU (i16 pairs) to GPU buffer (i32 pairs).
    pub fn upload_motion_vectors(
        ctx: &GpuContext,
        mvs: &[[i16; 2]],
    ) -> wgpu::Buffer {
        let i32_data: Vec<i32> = mvs
            .iter()
            .flat_map(|mv| [mv[0] as i32, mv[1] as i32])
            .collect();

        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mv_upload"),
                contents: bytemuck::cast_slice(&i32_data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }
}

// Helper functions for bind group layout entries
fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_estimation_zero_motion() {
        let ctx = GpuContext::new();
        let me = MotionEstimator::new(&ctx);

        let width = 64u32;
        let height = 64u32;
        let pixels = (width * height) as usize;

        // Create identical frames — MVs should be (0, 0)
        let frame_data: Vec<f32> = (0..pixels).map(|i| (i % 256) as f32).collect();

        let current_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("current"),
                contents: bytemuck::cast_slice(&frame_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let reference_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reference"),
                contents: bytemuck::cast_slice(&frame_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test_me"),
            });
        let (mv_buf, _sad_buf) =
            me.estimate(&ctx, &mut cmd, &current_buf, &reference_buf, width, height);
        ctx.queue.submit(Some(cmd.finish()));

        let blocks_x = width / ME_BLOCK_SIZE;
        let blocks_y = height / ME_BLOCK_SIZE;
        let total_blocks = blocks_x * blocks_y;
        let mvs = MotionEstimator::read_motion_vectors(&ctx, &mv_buf, total_blocks);

        for mv in &mvs {
            assert_eq!(mv[0], 0, "dx should be 0 for identical frames");
            assert_eq!(mv[1], 0, "dy should be 0 for identical frames");
        }
    }

    #[test]
    fn test_motion_estimation_known_shift() {
        let ctx = GpuContext::new();
        let me = MotionEstimator::new(&ctx);

        // Use a larger frame to ensure blocks away from edges can find the shift
        let width = 128u32;
        let height = 128u32;
        let pixels = (width * height) as usize;
        let shift_x: i32 = 5;
        let shift_y: i32 = 3;

        // Create a reference frame with unique per-pixel values (no modular aliasing)
        let mut reference_data = vec![0.0f32; pixels];
        for y in 0..height {
            for x in 0..width {
                reference_data[(y * width + x) as usize] = (x + y * width) as f32;
            }
        }

        // Create current frame = reference shifted by (shift_x, shift_y)
        // current[x,y] = reference[x + shift_x, y + shift_y]
        // So the MV should be (shift_x, shift_y) since reference[x+dx, y+dy] = current[x,y]
        let mut current_data = vec![0.0f32; pixels];
        for y in 0..height {
            for x in 0..width {
                let rx = (x as i32 + shift_x).clamp(0, width as i32 - 1) as u32;
                let ry = (y as i32 + shift_y).clamp(0, height as i32 - 1) as u32;
                current_data[(y * width + x) as usize] =
                    reference_data[(ry * width + rx) as usize];
            }
        }

        let current_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("current"),
                contents: bytemuck::cast_slice(&current_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let reference_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reference"),
                contents: bytemuck::cast_slice(&reference_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test_me_shift"),
            });
        let (mv_buf, _sad_buf) =
            me.estimate(&ctx, &mut cmd, &current_buf, &reference_buf, width, height);
        ctx.queue.submit(Some(cmd.finish()));

        let blocks_x = width / ME_BLOCK_SIZE;
        let blocks_y = height / ME_BLOCK_SIZE;
        let total_blocks = blocks_x * blocks_y;
        let mvs = MotionEstimator::read_motion_vectors(&ctx, &mv_buf, total_blocks);

        // Interior blocks (not near edges) should find the correct shift
        let mut correct_count = 0;
        let mut total_interior = 0;
        for by in 2..blocks_y - 2 {
            for bx in 2..blocks_x - 2 {
                let idx = (by * blocks_x + bx) as usize;
                total_interior += 1;
                if mvs[idx][0] == shift_x as i16 && mvs[idx][1] == shift_y as i16 {
                    correct_count += 1;
                }
            }
        }
        assert!(
            correct_count > total_interior / 2,
            "Most interior blocks should find shift ({},{}): {}/{} correct",
            shift_x,
            shift_y,
            correct_count,
            total_interior
        );
    }

    #[test]
    fn test_motion_compensate_roundtrip() {
        let ctx = GpuContext::new();
        let me = MotionEstimator::new(&ctx);

        let width = 64u32;
        let height = 64u32;
        let pixels = (width * height) as usize;
        let plane_size = (pixels * std::mem::size_of::<f32>()) as u64;

        // Create test data
        let current: Vec<f32> = (0..pixels).map(|i| (i % 200) as f32 + 10.0).collect();
        let reference: Vec<f32> = (0..pixels).map(|i| (i % 150) as f32 + 5.0).collect();

        let blocks_x = width / ME_BLOCK_SIZE;
        let blocks_y = height / ME_BLOCK_SIZE;
        let total_blocks = (blocks_x * blocks_y) as usize;

        // Zero motion vectors
        let mvs: Vec<i32> = vec![0i32; total_blocks * 2];

        let current_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("current"),
                contents: bytemuck::cast_slice(&current),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let reference_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reference"),
                contents: bytemuck::cast_slice(&reference),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let mv_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mvs"),
                contents: bytemuck::cast_slice(&mvs),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let residual_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("residual"),
            size: plane_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let recon_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("recon"),
            size: plane_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Forward: residual = current - reference
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mc_fwd"),
            });
        me.compensate(
            &ctx,
            &mut cmd,
            &current_buf,
            &reference_buf,
            &mv_buf,
            &residual_buf,
            width,
            height,
            true,
        );
        ctx.queue.submit(Some(cmd.finish()));

        // Inverse: recon = residual + reference
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mc_inv"),
            });
        me.compensate(
            &ctx,
            &mut cmd,
            &residual_buf,
            &reference_buf,
            &mv_buf,
            &recon_buf,
            width,
            height,
            false,
        );
        ctx.queue.submit(Some(cmd.finish()));

        // Read back reconstructed
        let recon = read_buffer_f32(&ctx, &recon_buf, pixels);

        // Should match original current within float precision
        let max_err: f32 = current
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.01,
            "MC roundtrip error too large: {}",
            max_err
        );
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
}
