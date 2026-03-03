use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use crate::GpuContext;

pub const ME_BLOCK_SIZE: u32 = 16;
pub const ME_SEARCH_RANGE: u32 = 32;
/// Fine search range when using temporal MV predictor (±pixels).
/// Temporal predictors are typically accurate within ~1 pixel, so ±2
/// provides sufficient refinement while saving ~67% of fine search work
/// (25 vs 81 candidates, fitting in 1 SIMD group instead of 3).
pub const ME_PRED_FINE_RANGE: u32 = 2;
/// Search range for bidirectional ME (B-frames). B-frames interpolate between
/// two references, so motion per direction is typically smaller than P-frames.
pub const ME_BIDIR_SEARCH_RANGE: u32 = 16;
/// Fine search range for bidir ME with temporal predictor (±pixels).
/// Smaller than P-frame (±4) because B-frame MVs between consecutive groups
/// are highly correlated and the temporal predictor is very accurate.
pub const ME_BIDIR_PRED_FINE_RANGE: u32 = 2;

/// Staging buffer for deferred MV readback. Created by `create_mv_staging`,
/// consumed by `finish_mv_readback`. Allows the GPU copy to piggyback on
/// another command encoder (e.g. local decode batch) instead of a separate submit.
pub struct MvStaging {
    pub buffer: wgpu::Buffer,
    pub size: u64,
    pub total_blocks: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlockMatchParams {
    width: u32,
    height: u32,
    block_size: u32,
    search_range: u32,
    blocks_x: u32,
    total_blocks: u32,
    /// Non-zero: skip coarse search, use predictor_mvs as starting point for fine search.
    use_predictor: u32,
    /// Fine search range when using predictor (in pixels, e.g. 8 for ±8).
    pred_fine_range: u32,
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

/// Staging buffers for deferred bidir data readback.
pub struct BidirStaging {
    pub fwd: wgpu::Buffer,
    pub bwd: wgpu::Buffer,
    pub modes: wgpu::Buffer,
    pub mv_size: u64,
    pub modes_size: u64,
    pub total_blocks: u32,
}

/// Block size for 8x8 sub-blocks used by split decision.
pub const ME_SPLIT_BLOCK_SIZE: u32 = 8;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlockMatchSplitParams {
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    total_macroblocks: u32,
    lambda_sad: u32,
    use_predictor: u32,
    _pad: u32,
}

/// GPU-based motion estimation and compensation.
pub struct MotionEstimator {
    match_pipeline: wgpu::ComputePipeline,
    match_bgl: wgpu::BindGroupLayout,
    compensate_pipeline: wgpu::ComputePipeline,
    compensate_bgl: wgpu::BindGroupLayout,
    match_bidir_pipeline: wgpu::ComputePipeline,
    match_bidir_bgl: wgpu::BindGroupLayout,
    compensate_bidir_pipeline: wgpu::ComputePipeline,
    compensate_bidir_bgl: wgpu::BindGroupLayout,
    split_pipeline: wgpu::ComputePipeline,
    split_bgl: wgpu::BindGroupLayout,
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

        // Block match bind group layout: uniform, current_y, reference_y, mvs, sads, predictor_mvs
        let match_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("block_match_bgl"),
                entries: &[
                    bgl_uniform(0),
                    bgl_storage_ro(1),
                    bgl_storage_ro(2),
                    bgl_storage_rw(3),
                    bgl_storage_rw(4),
                    bgl_storage_ro(5),
                ],
            });

        let match_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("block_match_pl"),
                bind_group_layouts: &[&match_bgl],
                push_constant_ranges: &[],
            });

        let match_pipeline = ctx
            .device
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

        let compensate_pl = ctx
            .device
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

        // --- Bidirectional block match ---
        let match_bidir_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("block_match_bidir"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/block_match_bidir.wgsl").into(),
                ),
            });

        // 10 bindings: uniform, current_y(ro), ref_fwd_y(ro), ref_bwd_y(ro),
        //              fwd_mvs(rw), bwd_mvs(rw), block_modes(rw), sads(rw),
        //              predictor_fwd_mvs(ro), predictor_bwd_mvs(ro)
        let match_bidir_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("block_match_bidir_bgl"),
                    entries: &[
                        bgl_uniform(0),
                        bgl_storage_ro(1),
                        bgl_storage_ro(2),
                        bgl_storage_ro(3),
                        bgl_storage_rw(4),
                        bgl_storage_rw(5),
                        bgl_storage_rw(6),
                        bgl_storage_rw(7),
                        bgl_storage_ro(8),
                        bgl_storage_ro(9),
                    ],
                });

        let match_bidir_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("block_match_bidir_pl"),
                bind_group_layouts: &[&match_bidir_bgl],
                push_constant_ranges: &[],
            });

        let match_bidir_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("block_match_bidir_pipeline"),
                    layout: Some(&match_bidir_pl),
                    module: &match_bidir_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // --- Bidirectional motion compensation ---
        let compensate_bidir_shader =
            ctx.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("motion_compensate_bidir"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("../shaders/motion_compensate_bidir.wgsl").into(),
                    ),
                });

        // 8 bindings: uniform, input(ro), fwd_ref(ro), bwd_ref(ro),
        //             fwd_mvs(ro), bwd_mvs(ro), block_modes(ro), output(rw)
        let compensate_bidir_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("motion_compensate_bidir_bgl"),
                    entries: &[
                        bgl_uniform(0),
                        bgl_storage_ro(1),
                        bgl_storage_ro(2),
                        bgl_storage_ro(3),
                        bgl_storage_ro(4),
                        bgl_storage_ro(5),
                        bgl_storage_ro(6),
                        bgl_storage_rw(7),
                    ],
                });

        let compensate_bidir_pl =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("motion_compensate_bidir_pl"),
                    bind_group_layouts: &[&compensate_bidir_bgl],
                    push_constant_ranges: &[],
                });

        let compensate_bidir_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("motion_compensate_bidir_pipeline"),
                    layout: Some(&compensate_bidir_pl),
                    module: &compensate_bidir_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // --- Split (variable block size) pipeline ---
        let split_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("block_match_split"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/block_match_split.wgsl").into(),
                ),
            });

        // 7 bindings: uniform, current_y(ro), reference_y(ro), parent_mvs(ro),
        //             parent_sads(ro), predictor_mvs(ro), output_mvs(rw)
        let split_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("block_match_split_bgl"),
                entries: &[
                    bgl_uniform(0),
                    bgl_storage_ro(1),
                    bgl_storage_ro(2),
                    bgl_storage_ro(3),
                    bgl_storage_ro(4),
                    bgl_storage_ro(5),
                    bgl_storage_rw(6),
                ],
            });

        let split_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("block_match_split_pl"),
                bind_group_layouts: &[&split_bgl],
                push_constant_ranges: &[],
            });

        let split_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("block_match_split_pipeline"),
                layout: Some(&split_pl),
                module: &split_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            match_pipeline,
            match_bgl,
            compensate_pipeline,
            compensate_bgl,
            match_bidir_pipeline,
            match_bidir_bgl,
            compensate_bidir_pipeline,
            compensate_bidir_bgl,
            split_pipeline,
            split_bgl,
        }
    }

    /// Dispatch block matching motion estimation on GPU.
    /// Returns (mv_buffer, sad_buffer). MVs are stored as i32 pairs (dx, dy) in half-pel units.
    ///
    /// When `predictor_mvs` is provided (from previous P-frame), the shader skips the
    /// expensive coarse search and uses the predicted MV as the starting point for a fine
    /// search with ±`ME_PRED_FINE_RANGE` pixels. This typically reduces ME cost by ~4x.
    #[allow(clippy::too_many_arguments)] // motion estimation needs current/reference frames + dimensions
    pub fn estimate(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        current_y: &wgpu::Buffer,
        reference_y: &wgpu::Buffer,
        width: u32,
        height: u32,
        predictor_mvs: Option<&wgpu::Buffer>,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let blocks_x = width / ME_BLOCK_SIZE;
        let blocks_y = height / ME_BLOCK_SIZE;
        let total_blocks = blocks_x * blocks_y;

        let has_predictor = predictor_mvs.is_some();
        let params = BlockMatchParams {
            width,
            height,
            block_size: ME_BLOCK_SIZE,
            search_range: ME_SEARCH_RANGE,
            blocks_x,
            total_blocks,
            use_predictor: u32::from(has_predictor),
            pred_fine_range: ME_PRED_FINE_RANGE,
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

        // Dummy predictor buffer when none provided (shader won't read it)
        let dummy_pred;
        let pred_buf = if let Some(pred) = predictor_mvs {
            pred
        } else {
            dummy_pred = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dummy_pred_mvs"),
                size: 8, // minimum valid size
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_pred
        };

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
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: pred_buf.as_entire_binding(),
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
    /// `block_size`: block size for MV grid (8 or 16)
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
        block_size: u32,
    ) {
        let blocks_x = width / block_size;
        let total_pixels = width * height;

        let params = MotionCompensateParams {
            width,
            height,
            block_size,
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

    /// Dispatch bidirectional block matching motion estimation on GPU.
    /// Tests forward-only, backward-only, and bidir average, picking the lowest SAD.
    /// Returns (fwd_mv_buf, bwd_mv_buf, block_modes_buf, sad_buf).
    /// MVs are stored as i32 pairs (dx, dy) in half-pel units.
    /// block_modes: 0 = forward, 1 = backward, 2 = bidirectional.
    #[allow(clippy::too_many_arguments)]
    pub fn estimate_bidir(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        current_y: &wgpu::Buffer,
        ref_fwd_y: &wgpu::Buffer,
        ref_bwd_y: &wgpu::Buffer,
        width: u32,
        height: u32,
        predictor_fwd_mvs: Option<&wgpu::Buffer>,
        predictor_bwd_mvs: Option<&wgpu::Buffer>,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let blocks_x = width / ME_BLOCK_SIZE;
        let blocks_y = height / ME_BLOCK_SIZE;
        let total_blocks = blocks_x * blocks_y;

        let have_predictor = predictor_fwd_mvs.is_some() && predictor_bwd_mvs.is_some();
        let params = BlockMatchParams {
            width,
            height,
            block_size: ME_BLOCK_SIZE,
            search_range: ME_BIDIR_SEARCH_RANGE,
            blocks_x,
            total_blocks,
            use_predictor: if have_predictor { 1 } else { 0 },
            pred_fine_range: if have_predictor {
                ME_BIDIR_PRED_FINE_RANGE
            } else {
                0
            },
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("block_match_bidir_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let mv_size = (total_blocks as usize * 2 * std::mem::size_of::<i32>()) as u64;
        let scalar_size = (total_blocks as usize * std::mem::size_of::<u32>()) as u64;

        let fwd_mv_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fwd_motion_vectors"),
            size: mv_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bwd_mv_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bwd_motion_vectors"),
            size: mv_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let block_modes_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block_modes"),
            size: scalar_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sad_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bidir_sad_values"),
            size: scalar_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Dummy 8-byte predictor buffers when no temporal predictor is available.
        let dummy_pred = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bidir_dummy_predictor"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let pred_fwd = predictor_fwd_mvs.unwrap_or(&dummy_pred);
        let pred_bwd = predictor_bwd_mvs.unwrap_or(&dummy_pred);

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("block_match_bidir_bg"),
            layout: &self.match_bidir_bgl,
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
                    resource: ref_fwd_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ref_bwd_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: fwd_mv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bwd_mv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: block_modes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: sad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: pred_fwd.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: pred_bwd.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("block_match_bidir_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.match_bidir_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // One workgroup per block
            pass.dispatch_workgroups(total_blocks, 1, 1);
        }

        (fwd_mv_buf, bwd_mv_buf, block_modes_buf, sad_buf)
    }

    /// Dispatch bidirectional motion compensation on GPU for a single plane.
    ///
    /// Uses per-block modes (0=fwd, 1=bwd, 2=bidir) to select prediction.
    /// `forward=true`:  residual = current - predicted (encoder)
    /// `forward=false`: reconstructed = residual + predicted (decoder)
    #[allow(clippy::too_many_arguments)]
    pub fn compensate_bidir(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        input_plane: &wgpu::Buffer,
        fwd_reference: &wgpu::Buffer,
        bwd_reference: &wgpu::Buffer,
        fwd_mv_buf: &wgpu::Buffer,
        bwd_mv_buf: &wgpu::Buffer,
        block_modes_buf: &wgpu::Buffer,
        output_plane: &wgpu::Buffer,
        width: u32,
        height: u32,
        forward: bool,
        block_size: u32,
    ) {
        let blocks_x = width / block_size;
        let total_pixels = width * height;

        let params = MotionCompensateParams {
            width,
            height,
            block_size,
            mode: if forward { 0 } else { 1 },
            blocks_x,
            total_pixels,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mc_bidir_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mc_bidir_bg"),
            layout: &self.compensate_bidir_bgl,
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
                    resource: fwd_reference.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bwd_reference.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: fwd_mv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bwd_mv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: block_modes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: output_plane.as_entire_binding(),
                },
            ],
        });

        let workgroups = total_pixels.div_ceil(256);
        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mc_bidir_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compensate_bidir_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    // --- Cached variants: use pre-allocated buffers from CachedEncodeBuffers ---

    /// Like `estimate` but uses pre-allocated params/sad/dummy buffers.
    /// Still creates MV buffer per call (needed for temporal prediction ping-pong).
    #[allow(clippy::too_many_arguments)]
    pub fn estimate_cached(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        current_y: &wgpu::Buffer,
        reference_y: &wgpu::Buffer,
        width: u32,
        height: u32,
        predictor_mvs: Option<&wgpu::Buffer>,
        params_buf: &wgpu::Buffer,
        sad_buf: &wgpu::Buffer,
        dummy_pred: &wgpu::Buffer,
    ) -> wgpu::Buffer {
        let blocks_x = width / ME_BLOCK_SIZE;
        let blocks_y = height / ME_BLOCK_SIZE;
        let total_blocks = blocks_x * blocks_y;

        let mv_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion_vectors"),
            size: (total_blocks as usize * 2 * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pred_buf = predictor_mvs.unwrap_or(dummy_pred);

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
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: pred_buf.as_entire_binding(),
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
            pass.dispatch_workgroups(total_blocks, 1, 1);
        }

        mv_buf
    }

    /// Like `compensate` but uses a pre-allocated params buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn compensate_cached(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        input_plane: &wgpu::Buffer,
        reference_plane: &wgpu::Buffer,
        mv_buf: &wgpu::Buffer,
        output_plane: &wgpu::Buffer,
        width: u32,
        height: u32,
        params_buf: &wgpu::Buffer,
    ) {
        let total_pixels = width * height;

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

    /// Dispatch 8x8 split decision shader.
    /// Takes 16x16 parent MVs + SADs, runs 4× 8x8 sub-block refinement with
    /// RD split decision. Returns 8x8-resolution MV buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn estimate_split(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        current_y: &wgpu::Buffer,
        reference_y: &wgpu::Buffer,
        parent_mvs: &wgpu::Buffer,
        parent_sads: &wgpu::Buffer,
        predictor_8x8_mvs: Option<&wgpu::Buffer>,
        width: u32,
        height: u32,
        lambda_sad: u32,
    ) -> wgpu::Buffer {
        let blocks_x = width / ME_BLOCK_SIZE;
        let blocks_y = height / ME_BLOCK_SIZE;
        let total_macroblocks = blocks_x * blocks_y;

        // Output at 8x8 resolution: 4x more blocks
        let blocks_x_8 = width / ME_SPLIT_BLOCK_SIZE;
        let blocks_y_8 = height / ME_SPLIT_BLOCK_SIZE;
        let total_blocks_8 = blocks_x_8 * blocks_y_8;

        let params = BlockMatchSplitParams {
            width,
            height,
            blocks_x,
            blocks_y,
            total_macroblocks,
            lambda_sad,
            use_predictor: u32::from(predictor_8x8_mvs.is_some()),
            _pad: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("split_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let output_mv_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("split_output_mvs"),
            size: (total_blocks_8 as usize * 2 * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Dummy predictor buffer when none provided
        let dummy_pred;
        let pred_buf = if let Some(pred) = predictor_8x8_mvs {
            pred
        } else {
            dummy_pred = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("split_dummy_pred"),
                size: 8,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            &dummy_pred
        };

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("split_bg"),
            layout: &self.split_bgl,
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
                    resource: parent_mvs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: parent_sads.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: pred_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output_mv_buf.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("block_match_split_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.split_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(total_macroblocks, 1, 1);
        }

        output_mv_buf
    }

    /// Downsample 8x8-resolution MVs to 16x16 by taking the top-left sub-block MV
    /// of each macroblock. Used for temporal predictor input to block_match.wgsl.
    pub fn downsample_8x8_to_16x16(
        mvs_8x8: &[[i16; 2]],
        blocks_x_8: u32,
        blocks_y_8: u32,
    ) -> Vec<[i16; 2]> {
        let blocks_x_16 = blocks_x_8 / 2;
        let blocks_y_16 = blocks_y_8 / 2;
        let mut mvs_16 = Vec::with_capacity((blocks_x_16 * blocks_y_16) as usize);
        for by in 0..blocks_y_16 {
            for bx in 0..blocks_x_16 {
                // Top-left sub-block of this macroblock
                let idx_8 = (by * 2) * blocks_x_8 + (bx * 2);
                mvs_16.push(mvs_8x8[idx_8 as usize]);
            }
        }
        mvs_16
    }

    /// Like `estimate_bidir` but uses pre-allocated params/sad/dummy/modes buffers.
    /// Still creates fwd/bwd MV buffers per call (needed for temporal prediction).
    /// Returns `(fwd_mv_buf, bwd_mv_buf)`. Block modes are written to `modes_buf`.
    #[allow(clippy::too_many_arguments)]
    pub fn estimate_bidir_cached(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        current_y: &wgpu::Buffer,
        ref_fwd_y: &wgpu::Buffer,
        ref_bwd_y: &wgpu::Buffer,
        width: u32,
        height: u32,
        predictor_fwd_mvs: Option<&wgpu::Buffer>,
        predictor_bwd_mvs: Option<&wgpu::Buffer>,
        params_buf: &wgpu::Buffer,
        sad_buf: &wgpu::Buffer,
        modes_buf: &wgpu::Buffer,
        dummy_pred: &wgpu::Buffer,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let blocks_x = width / ME_BLOCK_SIZE;
        let blocks_y = height / ME_BLOCK_SIZE;
        let total_blocks = blocks_x * blocks_y;

        let mv_size = (total_blocks as usize * 2 * std::mem::size_of::<i32>()) as u64;

        let fwd_mv_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fwd_motion_vectors"),
            size: mv_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bwd_mv_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bwd_motion_vectors"),
            size: mv_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pred_fwd = predictor_fwd_mvs.unwrap_or(dummy_pred);
        let pred_bwd = predictor_bwd_mvs.unwrap_or(dummy_pred);

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("block_match_bidir_bg"),
            layout: &self.match_bidir_bgl,
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
                    resource: ref_fwd_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ref_bwd_y.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: fwd_mv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bwd_mv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: modes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: sad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: pred_fwd.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: pred_bwd.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("block_match_bidir_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.match_bidir_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(total_blocks, 1, 1);
        }

        (fwd_mv_buf, bwd_mv_buf)
    }

    /// Like `compensate_bidir` but uses a pre-allocated params buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn compensate_bidir_cached(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        input_plane: &wgpu::Buffer,
        fwd_reference: &wgpu::Buffer,
        bwd_reference: &wgpu::Buffer,
        fwd_mv_buf: &wgpu::Buffer,
        bwd_mv_buf: &wgpu::Buffer,
        block_modes_buf: &wgpu::Buffer,
        output_plane: &wgpu::Buffer,
        width: u32,
        height: u32,
        params_buf: &wgpu::Buffer,
    ) {
        let total_pixels = width * height;

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mc_bidir_bg"),
            layout: &self.compensate_bidir_bgl,
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
                    resource: fwd_reference.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bwd_reference.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: fwd_mv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bwd_mv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: block_modes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: output_plane.as_entire_binding(),
                },
            ],
        });

        let workgroups = total_pixels.div_ceil(256);
        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mc_bidir_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compensate_bidir_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// Finish MV readback from a cached staging buffer.
    /// Unlike `finish_mv_readback`, this uses a pre-allocated staging buffer that
    /// must be unmapped before reuse (this function handles unmapping).
    pub fn finish_mv_readback_cached(
        ctx: &GpuContext,
        staging: &wgpu::Buffer,
        mv_size: u64,
        total_blocks: u32,
    ) -> Vec<[i16; 2]> {
        let slice = staging.slice(..mv_size);
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

    /// Finish bidir readback from cached staging buffers.
    pub fn finish_bidir_readback_cached(
        ctx: &GpuContext,
        fwd_staging: &wgpu::Buffer,
        bwd_staging: &wgpu::Buffer,
        modes_staging: &wgpu::Buffer,
        mv_size: u64,
        modes_size: u64,
        total_blocks: u32,
    ) -> (Vec<[i16; 2]>, Vec<[i16; 2]>, Vec<u8>) {
        let (tx, rx) = std::sync::mpsc::channel();
        for (buf, size) in [
            (fwd_staging, mv_size),
            (bwd_staging, mv_size),
            (modes_staging, modes_size),
        ] {
            let tx_clone = tx.clone();
            buf.slice(..size).map_async(wgpu::MapMode::Read, move |r| {
                tx_clone.send(r).unwrap();
            });
        }
        drop(tx);
        ctx.device.poll(wgpu::Maintain::Wait);
        for _ in 0..3 {
            rx.recv().unwrap().unwrap();
        }

        let fwd_data = fwd_staging.slice(..mv_size).get_mapped_range();
        let fwd_i32: &[i32] = bytemuck::cast_slice(&fwd_data);
        let mut fwd_mvs = Vec::with_capacity(total_blocks as usize);
        for i in 0..total_blocks as usize {
            fwd_mvs.push([fwd_i32[i * 2] as i16, fwd_i32[i * 2 + 1] as i16]);
        }
        drop(fwd_data);
        fwd_staging.unmap();

        let bwd_data = bwd_staging.slice(..mv_size).get_mapped_range();
        let bwd_i32: &[i32] = bytemuck::cast_slice(&bwd_data);
        let mut bwd_mvs = Vec::with_capacity(total_blocks as usize);
        for i in 0..total_blocks as usize {
            bwd_mvs.push([bwd_i32[i * 2] as i16, bwd_i32[i * 2 + 1] as i16]);
        }
        drop(bwd_data);
        bwd_staging.unmap();

        let modes_data = modes_staging.slice(..modes_size).get_mapped_range();
        let u32_modes: &[u32] = bytemuck::cast_slice(&modes_data);
        let block_modes: Vec<u8> = u32_modes.iter().map(|&v| v as u8).collect();
        drop(modes_data);
        modes_staging.unmap();

        (fwd_mvs, bwd_mvs, block_modes)
    }

    /// Read block modes from GPU buffer back to CPU.
    /// Block modes are stored as u32 in the shader (0=fwd, 1=bwd, 2=bidir),
    /// returned as Vec<u8> for compact storage.
    pub fn read_block_modes(
        ctx: &GpuContext,
        block_modes_buf: &wgpu::Buffer,
        total_blocks: u32,
    ) -> Vec<u8> {
        let count = total_blocks as usize;
        let size = (count * std::mem::size_of::<u32>()) as u64;

        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("block_modes_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_block_modes"),
            });
        cmd.copy_buffer_to_buffer(block_modes_buf, 0, &staging, 0, size);
        ctx.queue.submit(Some(cmd.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let u32_data: &[u32] = bytemuck::cast_slice(&data);
        let result: Vec<u8> = u32_data.iter().map(|&v| v as u8).collect();
        drop(data);
        staging.unmap();
        result
    }

    /// Upload block modes from CPU (Vec<u8>) to GPU buffer (u32 per mode).
    pub fn upload_block_modes(ctx: &GpuContext, modes: &[u8]) -> wgpu::Buffer {
        let u32_data: Vec<u32> = modes.iter().map(|&m| m as u32).collect();

        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("block_modes_upload"),
                contents: bytemuck::cast_slice(&u32_data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    /// Read motion vectors from GPU buffer back to CPU (self-contained, creates its own submit).
    pub fn read_motion_vectors(
        ctx: &GpuContext,
        mv_buf: &wgpu::Buffer,
        total_blocks: u32,
    ) -> Vec<[i16; 2]> {
        let staging = Self::create_mv_staging(ctx, mv_buf, total_blocks);

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_mv"),
            });
        cmd.copy_buffer_to_buffer(mv_buf, 0, &staging.buffer, 0, staging.size);
        ctx.queue.submit(Some(cmd.finish()));

        Self::finish_mv_readback(ctx, &staging)
    }

    /// Create MV staging buffer and return it for deferred copy.
    /// The caller adds `copy_buffer_to_buffer(mv_buf, 0, staging.buffer, 0, staging.size)`
    /// to their own command encoder, allowing the copy to piggyback on other GPU work.
    pub fn create_mv_staging(
        ctx: &GpuContext,
        _mv_buf: &wgpu::Buffer,
        total_blocks: u32,
    ) -> MvStaging {
        let count = total_blocks as usize * 2;
        let size = (count * std::mem::size_of::<i32>()) as u64;

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mv_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        MvStaging {
            buffer,
            size,
            total_blocks,
        }
    }

    /// Map and read MV data from a staging buffer. Call after the command encoder
    /// containing the copy has been submitted and the GPU work is complete (or will
    /// be drained by poll).
    pub fn finish_mv_readback(ctx: &GpuContext, staging: &MvStaging) -> Vec<[i16; 2]> {
        let slice = staging.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let i32_data: &[i32] = bytemuck::cast_slice(&data);

        let mut mvs = Vec::with_capacity(staging.total_blocks as usize);
        for i in 0..staging.total_blocks as usize {
            mvs.push([i32_data[i * 2] as i16, i32_data[i * 2 + 1] as i16]);
        }

        drop(data);
        staging.buffer.unmap();
        mvs
    }

    /// Create bidir staging buffers for deferred copy.
    /// The caller adds copy_buffer_to_buffer commands to their own command encoder.
    pub fn create_bidir_staging(ctx: &GpuContext, total_blocks: u32) -> BidirStaging {
        let mv_count = total_blocks as usize * 2;
        let mv_size = (mv_count * std::mem::size_of::<i32>()) as u64;
        let modes_size = (total_blocks as usize * std::mem::size_of::<u32>()) as u64;
        let mr = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        BidirStaging {
            fwd: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fwd_mv_staging"),
                size: mv_size,
                usage: mr,
                mapped_at_creation: false,
            }),
            bwd: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("bwd_mv_staging"),
                size: mv_size,
                usage: mr,
                mapped_at_creation: false,
            }),
            modes: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("modes_staging"),
                size: modes_size,
                usage: mr,
                mapped_at_creation: false,
            }),
            mv_size,
            modes_size,
            total_blocks,
        }
    }

    /// Read bidir data from staging buffers after GPU work is complete.
    pub fn finish_bidir_readback(
        ctx: &GpuContext,
        staging: &BidirStaging,
    ) -> (Vec<[i16; 2]>, Vec<[i16; 2]>, Vec<u8>) {
        let (tx, rx) = std::sync::mpsc::channel();
        for buf in [&staging.fwd, &staging.bwd, &staging.modes] {
            let tx_clone = tx.clone();
            buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                tx_clone.send(r).unwrap();
            });
        }
        drop(tx);
        ctx.device.poll(wgpu::Maintain::Wait);
        for _ in 0..3 {
            rx.recv().unwrap().unwrap();
        }

        let fwd_data = staging.fwd.slice(..).get_mapped_range();
        let fwd_i32: &[i32] = bytemuck::cast_slice(&fwd_data);
        let mut fwd_mvs = Vec::with_capacity(staging.total_blocks as usize);
        for i in 0..staging.total_blocks as usize {
            fwd_mvs.push([fwd_i32[i * 2] as i16, fwd_i32[i * 2 + 1] as i16]);
        }
        drop(fwd_data);
        staging.fwd.unmap();

        let bwd_data = staging.bwd.slice(..).get_mapped_range();
        let bwd_i32: &[i32] = bytemuck::cast_slice(&bwd_data);
        let mut bwd_mvs = Vec::with_capacity(staging.total_blocks as usize);
        for i in 0..staging.total_blocks as usize {
            bwd_mvs.push([bwd_i32[i * 2] as i16, bwd_i32[i * 2 + 1] as i16]);
        }
        drop(bwd_data);
        staging.bwd.unmap();

        let modes_data = staging.modes.slice(..).get_mapped_range();
        let u32_modes: &[u32] = bytemuck::cast_slice(&modes_data);
        let block_modes: Vec<u8> = u32_modes.iter().map(|&v| v as u8).collect();
        drop(modes_data);
        staging.modes.unmap();

        (fwd_mvs, bwd_mvs, block_modes)
    }

    /// Batched readback of bidirectional MVs and block modes in a single GPU submit + poll.
    /// Returns (fwd_mvs, bwd_mvs, block_modes) with only 1 blocking wait instead of 3.
    pub fn read_bidir_data(
        ctx: &GpuContext,
        fwd_mv_buf: &wgpu::Buffer,
        bwd_mv_buf: &wgpu::Buffer,
        block_modes_buf: &wgpu::Buffer,
        total_blocks: u32,
    ) -> (Vec<[i16; 2]>, Vec<[i16; 2]>, Vec<u8>) {
        let mv_count = total_blocks as usize * 2;
        let mv_size = (mv_count * std::mem::size_of::<i32>()) as u64;
        let modes_size = (total_blocks as usize * std::mem::size_of::<u32>()) as u64;

        let staging_fwd = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fwd_mv_staging"),
            size: mv_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_bwd = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bwd_mv_staging"),
            size: mv_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_modes = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("modes_staging"),
            size: modes_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Single command encoder for all 3 copies
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_bidir_data"),
            });
        cmd.copy_buffer_to_buffer(fwd_mv_buf, 0, &staging_fwd, 0, mv_size);
        cmd.copy_buffer_to_buffer(bwd_mv_buf, 0, &staging_bwd, 0, mv_size);
        cmd.copy_buffer_to_buffer(block_modes_buf, 0, &staging_modes, 0, modes_size);
        ctx.queue.submit(Some(cmd.finish()));

        // Map all 3 staging buffers, then single poll
        let (tx1, rx1) = std::sync::mpsc::channel();
        let (tx2, rx2) = std::sync::mpsc::channel();
        let (tx3, rx3) = std::sync::mpsc::channel();
        staging_fwd
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                tx1.send(r).unwrap();
            });
        staging_bwd
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                tx2.send(r).unwrap();
            });
        staging_modes
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                tx3.send(r).unwrap();
            });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx1.recv().unwrap().unwrap();
        rx2.recv().unwrap().unwrap();
        rx3.recv().unwrap().unwrap();

        // Read forward MVs
        let fwd_data = staging_fwd.slice(..).get_mapped_range();
        let fwd_i32: &[i32] = bytemuck::cast_slice(&fwd_data);
        let mut fwd_mvs = Vec::with_capacity(total_blocks as usize);
        for i in 0..total_blocks as usize {
            fwd_mvs.push([fwd_i32[i * 2] as i16, fwd_i32[i * 2 + 1] as i16]);
        }
        drop(fwd_data);
        staging_fwd.unmap();

        // Read backward MVs
        let bwd_data = staging_bwd.slice(..).get_mapped_range();
        let bwd_i32: &[i32] = bytemuck::cast_slice(&bwd_data);
        let mut bwd_mvs = Vec::with_capacity(total_blocks as usize);
        for i in 0..total_blocks as usize {
            bwd_mvs.push([bwd_i32[i * 2] as i16, bwd_i32[i * 2 + 1] as i16]);
        }
        drop(bwd_data);
        staging_bwd.unmap();

        // Read block modes
        let modes_data = staging_modes.slice(..).get_mapped_range();
        let u32_modes: &[u32] = bytemuck::cast_slice(&modes_data);
        let block_modes: Vec<u8> = u32_modes.iter().map(|&v| v as u8).collect();
        drop(modes_data);
        staging_modes.unmap();

        (fwd_mvs, bwd_mvs, block_modes)
    }

    /// Read SAD values from GPU buffer back to CPU.
    pub fn read_sad_values(
        ctx: &GpuContext,
        sad_buf: &wgpu::Buffer,
        total_blocks: u32,
    ) -> Vec<u32> {
        let count = total_blocks as usize;
        let size = (count * std::mem::size_of::<u32>()) as u64;

        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sad_staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_sad"),
            });
        cmd.copy_buffer_to_buffer(sad_buf, 0, &staging, 0, size);
        ctx.queue.submit(Some(cmd.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Upload motion vectors from CPU (i16 pairs) to GPU buffer (i32 pairs).
    pub fn upload_motion_vectors(ctx: &GpuContext, mvs: &[[i16; 2]]) -> wgpu::Buffer {
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

    /// Write motion vectors into a pre-allocated GPU buffer via queue.write_buffer().
    pub fn write_motion_vectors_into(ctx: &GpuContext, mvs: &[[i16; 2]], buf: &wgpu::Buffer) {
        let i32_data: Vec<i32> = mvs
            .iter()
            .flat_map(|mv| [mv[0] as i32, mv[1] as i32])
            .collect();
        ctx.queue
            .write_buffer(buf, 0, bytemuck::cast_slice(&i32_data));
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
        let (mv_buf, _sad_buf) = me.estimate(
            &ctx,
            &mut cmd,
            &current_buf,
            &reference_buf,
            width,
            height,
            None,
        );
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
                current_data[(y * width + x) as usize] = reference_data[(ry * width + rx) as usize];
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
        let (mv_buf, _sad_buf) = me.estimate(
            &ctx,
            &mut cmd,
            &current_buf,
            &reference_buf,
            width,
            height,
            None,
        );
        ctx.queue.submit(Some(cmd.finish()));

        let blocks_x = width / ME_BLOCK_SIZE;
        let blocks_y = height / ME_BLOCK_SIZE;
        let total_blocks = blocks_x * blocks_y;
        let mvs = MotionEstimator::read_motion_vectors(&ctx, &mv_buf, total_blocks);

        // MVs are now in half-pel units, so expected values are shift * 2
        let expected_dx = (shift_x * 2) as i16;
        let expected_dy = (shift_y * 2) as i16;

        // Interior blocks (not near edges) should find the correct shift
        let mut correct_count = 0;
        let mut total_interior = 0;
        for by in 2..blocks_y - 2 {
            for bx in 2..blocks_x - 2 {
                let idx = (by * blocks_x + bx) as usize;
                total_interior += 1;
                if mvs[idx][0] == expected_dx && mvs[idx][1] == expected_dy {
                    correct_count += 1;
                }
            }
        }
        assert!(
            correct_count > total_interior / 2,
            "Most interior blocks should find shift ({},{}) in half-pel units: {}/{} correct",
            expected_dx,
            expected_dy,
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
            ME_BLOCK_SIZE,
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
            ME_BLOCK_SIZE,
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
        assert!(max_err < 0.01, "MC roundtrip error too large: {}", max_err);
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
