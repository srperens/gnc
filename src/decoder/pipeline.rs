use bytemuck::{Pod, Zeroable};
use std::cell::RefCell;
use wgpu;

use crate::encoder::adaptive::AQ_BLOCK_SIZE;
use crate::encoder::bitplane::GpuBitplaneDecoder;
use crate::encoder::cfl::{self, CflPredictor};
use crate::encoder::color::ColorConverter;
use crate::encoder::interleave::PlaneInterleaver;
use crate::encoder::motion::{MotionEstimator, ME_BLOCK_SIZE};
use crate::encoder::quantize::Quantizer;
use crate::encoder::rans_gpu::GpuRansDecoder;
use crate::encoder::transform::WaveletTransform;
use crate::{CompressedFrame, EntropyData, FrameType, GpuContext};

/// Handle returned by `decode_to_texture` with metadata about the decoded frame.
/// The actual texture view is accessible via `DecoderPipeline::output_texture_view()`.
pub struct TextureHandle {
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CropParams {
    src_width: u32,
    dst_width: u32,
    dst_height: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PackParams {
    total_f32s: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TextureParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Cached GPU buffers for decode — allocated once, reused across frames.
/// Zero buffer allocations in steady-state decode: all buffers pre-allocated
/// on resolution init, data uploaded via queue.write_buffer().
struct CachedBuffers {
    padded_w: u32,
    padded_h: u32,
    width: u32,
    height: u32,
    tile_size: u32,
    scratch_a: wgpu::Buffer,
    scratch_b: wgpu::Buffer,
    scratch_c: wgpu::Buffer,
    plane_results: [wgpu::Buffer; 3],
    ycocg_buf: wgpu::Buffer,
    rgb_out_buf: wgpu::Buffer,
    cropped_buf: wgpu::Buffer,
    packed_u8_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    staging_u8: wgpu::Buffer,
    /// Dequantized Y wavelet coefficients for CfL prediction
    y_ref_wavelet_buf: wgpu::Buffer,
    /// Reference planes from previous decoded frame (for temporal prediction)
    reference_planes: [wgpu::Buffer; 3],

    // Per-plane entropy decode buffers (reused across frames)
    entropy_params: [wgpu::Buffer; 3],
    entropy_tile_info: [wgpu::Buffer; 3],
    entropy_var_a: [wgpu::Buffer; 3],
    entropy_var_b: [wgpu::Buffer; 3],
    entropy_var_a_cap: [u64; 3],
    entropy_var_b_cap: [u64; 3],

    // Fixed small params buffers (written once per resolution)
    crop_params_buf: wgpu::Buffer,
    pack_params_buf: wgpu::Buffer,

    // CfL buffers (reused when CfL enabled)
    cfl_alpha_buf: wgpu::Buffer,
    cfl_alpha_cap: u64,
    plane_alpha_bufs: [wgpu::Buffer; 2],
    plane_alpha_cap: u64,

    // Conditional per-frame uploads
    weight_map_buf: wgpu::Buffer,
    weight_map_cap: u64,
    mv_buf: wgpu::Buffer,
    mv_cap: u64,

    // Output texture for zero-readback decode path
    // (texture and params_buf kept alive for the view and bind group that reference them)
    #[allow(dead_code)]
    output_texture: wgpu::Texture,
    output_texture_view: wgpu::TextureView,
    buf_to_tex_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    buf_to_tex_params_buf: wgpu::Buffer,
}

/// Full decoding pipeline: GPU rANS Decode -> Dequantize -> Inverse Wavelet -> Interleave -> Inverse Color -> Crop
///
/// All GPU stages run without CPU readback until the final RGB output.
/// GPU buffers are cached across frames for zero-allocation steady-state decode.
pub struct DecoderPipeline {
    color: ColorConverter,
    transform: WaveletTransform,
    quantize: Quantizer,
    rans_decoder: GpuRansDecoder,
    bitplane_decoder: GpuBitplaneDecoder,
    interleaver: PlaneInterleaver,
    cfl_predictor: CflPredictor,
    motion: MotionEstimator,
    crop_pipeline: wgpu::ComputePipeline,
    crop_bgl: wgpu::BindGroupLayout,
    pack_pipeline: wgpu::ComputePipeline,
    pack_bgl: wgpu::BindGroupLayout,
    buf_to_tex_pipeline: wgpu::ComputePipeline,
    buf_to_tex_bgl: wgpu::BindGroupLayout,
    cached: RefCell<Option<CachedBuffers>>,
    pending_rx: RefCell<Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
}

impl DecoderPipeline {
    pub fn new(ctx: &GpuContext) -> Self {
        let crop_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("crop"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/crop.wgsl").into()),
            });

        let crop_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("crop_bgl"),
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

        let crop_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("crop_pl"),
                bind_group_layouts: &[&crop_bgl],
                push_constant_ranges: &[],
            });

        let crop_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("crop_pipeline"),
                layout: Some(&crop_pl),
                module: &crop_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Pack u8 shader (f32 → packed u8)
        let pack_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pack_u8"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/pack_u8.wgsl").into()),
            });

        let pack_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pack_bgl"),
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

        let pack_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pack_pl"),
                bind_group_layouts: &[&pack_bgl],
                push_constant_ranges: &[],
            });

        let pack_pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pack_pipeline"),
                layout: Some(&pack_pl),
                module: &pack_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Buffer-to-texture shader (f32 RGB buffer → rgba8unorm texture)
        let buf_to_tex_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("buffer_to_texture"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/buffer_to_texture.wgsl").into(),
                ),
            });

        let buf_to_tex_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("buf_to_tex_bgl"),
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
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

        let buf_to_tex_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("buf_to_tex_pl"),
                bind_group_layouts: &[&buf_to_tex_bgl],
                push_constant_ranges: &[],
            });

        let buf_to_tex_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("buf_to_tex_pipeline"),
                    layout: Some(&buf_to_tex_pl),
                    module: &buf_to_tex_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            color: ColorConverter::new(ctx),
            transform: WaveletTransform::new(ctx),
            quantize: Quantizer::new(ctx),
            rans_decoder: GpuRansDecoder::new(ctx),
            bitplane_decoder: GpuBitplaneDecoder::new(ctx),
            interleaver: PlaneInterleaver::new(ctx),
            cfl_predictor: CflPredictor::new(ctx),
            motion: MotionEstimator::new(ctx),
            crop_pipeline,
            crop_bgl,
            pack_pipeline,
            pack_bgl,
            buf_to_tex_pipeline,
            buf_to_tex_bgl,
            cached: RefCell::new(None),
            pending_rx: RefCell::new(None),
        }
    }

    fn ensure_cached(
        &self,
        ctx: &GpuContext,
        padded_w: u32,
        padded_h: u32,
        width: u32,
        height: u32,
        tile_size: u32,
    ) {
        let mut cached = self.cached.borrow_mut();
        if let Some(ref c) = *cached {
            if c.padded_w == padded_w
                && c.padded_h == padded_h
                && c.width == width
                && c.height == height
                && c.tile_size == tile_size
            {
                return;
            }
        }

        let padded_pixels = (padded_w * padded_h) as usize;
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let buf_size_3 = (padded_pixels * 3 * std::mem::size_of::<f32>()) as u64;
        let output_pixels = (width * height) as usize;
        let output_size = (output_pixels * 3 * std::mem::size_of::<f32>()) as u64;

        let scratch_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let scratch_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_scratch_a"),
            size: plane_size,
            usage: scratch_usage,
            mapped_at_creation: false,
        });
        let scratch_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_scratch_b"),
            size: plane_size,
            usage: scratch_usage,
            mapped_at_creation: false,
        });
        let scratch_c = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_scratch_c"),
            size: plane_size,
            usage: scratch_usage,
            mapped_at_creation: false,
        });

        let plane_results = std::array::from_fn(|p| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dec_plane_result_{p}")),
                size: plane_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });

        let ycocg_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_ycocg"),
            size: buf_size_3,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let rgb_out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_rgb_out"),
            size: buf_size_3,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let cropped_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_cropped"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Packed u8 output: ceil(total_f32s / 4) u32s
        let total_f32s = output_pixels * 3;
        let packed_u32s = total_f32s.div_ceil(4) as u64;
        let packed_byte_size = packed_u32s * 4;
        let packed_u8_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_packed_u8"),
            size: packed_byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_u8 = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_staging_u8"),
            size: packed_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let y_ref_wavelet_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_y_ref_wavelet"),
            size: plane_size,
            usage: scratch_usage,
            mapped_at_creation: false,
        });

        let reference_planes = std::array::from_fn(|p| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dec_reference_{p}")),
                size: plane_size,
                usage: scratch_usage,
                mapped_at_creation: false,
            })
        });

        // --- Pre-allocated entropy decode buffers ---
        let tiles_per_plane = ((padded_w / tile_size) * (padded_h / tile_size)) as usize;
        // rANS TILE_INFO_STRIDE (100) > bitplane (8), so use rANS stride for max
        let tile_info_size =
            (tiles_per_plane * crate::encoder::rans_gpu::TILE_INFO_STRIDE * 4).max(4) as u64;
        let var_a_init_cap = (tiles_per_plane * 2048 * 4).max(4) as u64;
        let var_b_init_cap = plane_size.max(4);

        let storage_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let uniform_dst = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;

        let entropy_params: [wgpu::Buffer; 3] = std::array::from_fn(|p| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dec_entropy_params_{p}")),
                size: 32,
                usage: uniform_dst,
                mapped_at_creation: false,
            })
        });
        let entropy_tile_info: [wgpu::Buffer; 3] = std::array::from_fn(|p| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dec_entropy_tile_info_{p}")),
                size: tile_info_size,
                usage: storage_dst,
                mapped_at_creation: false,
            })
        });
        let entropy_var_a: [wgpu::Buffer; 3] = std::array::from_fn(|p| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dec_entropy_var_a_{p}")),
                size: var_a_init_cap,
                usage: storage_dst,
                mapped_at_creation: false,
            })
        });
        let entropy_var_b: [wgpu::Buffer; 3] = std::array::from_fn(|p| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dec_entropy_var_b_{p}")),
                size: var_b_init_cap,
                usage: storage_dst,
                mapped_at_creation: false,
            })
        });

        // Crop params (constant per resolution)
        let crop_params = CropParams {
            src_width: padded_w,
            dst_width: width,
            dst_height: height,
            _pad: 0,
        };
        let crop_params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_crop_params"),
            size: std::mem::size_of::<CropParams>() as u64,
            usage: uniform_dst,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&crop_params_buf, 0, bytemuck::bytes_of(&crop_params));

        // Pack params (constant per resolution)
        let pack_params = PackParams {
            total_f32s: total_f32s as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let pack_params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_pack_params"),
            size: std::mem::size_of::<PackParams>() as u64,
            usage: uniform_dst,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&pack_params_buf, 0, bytemuck::bytes_of(&pack_params));

        // CfL alpha buffers
        let cfl_alpha_cap = (tiles_per_plane * 16 * 2 * 4).max(4) as u64;
        let cfl_alpha_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_cfl_alpha"),
            size: cfl_alpha_cap,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let plane_alpha_cap = (tiles_per_plane * 16 * 4).max(4) as u64;
        let plane_alpha_bufs: [wgpu::Buffer; 2] = std::array::from_fn(|i| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dec_plane_alpha_{i}")),
                size: plane_alpha_cap,
                usage: storage_dst,
                mapped_at_creation: false,
            })
        });

        // Weight map buffer
        let aq_blocks_x = padded_w.div_ceil(AQ_BLOCK_SIZE);
        let aq_blocks_y = padded_h.div_ceil(AQ_BLOCK_SIZE);
        let weight_map_cap = (aq_blocks_x * aq_blocks_y * 4).max(4) as u64;
        let weight_map_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_weight_map"),
            size: weight_map_cap,
            usage: storage_dst,
            mapped_at_creation: false,
        });

        // Motion vector buffer
        let mv_blocks_x = padded_w / ME_BLOCK_SIZE;
        let mv_blocks_y = padded_h / ME_BLOCK_SIZE;
        let mv_cap = (mv_blocks_x * mv_blocks_y * 2 * 4).max(4) as u64;
        let mv_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_mv"),
            size: mv_cap,
            usage: storage_dst,
            mapped_at_creation: false,
        });

        // Output texture for zero-readback decode path
        let output_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dec_output_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_texture_view =
            output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let buf_to_tex_params = TextureParams {
            width,
            height,
            _pad0: 0,
            _pad1: 0,
        };
        let buf_to_tex_params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_buf_to_tex_params"),
            size: std::mem::size_of::<TextureParams>() as u64,
            usage: uniform_dst,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(
            &buf_to_tex_params_buf,
            0,
            bytemuck::bytes_of(&buf_to_tex_params),
        );

        let buf_to_tex_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("buf_to_tex_bg"),
            layout: &self.buf_to_tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_to_tex_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cropped_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_texture_view),
                },
            ],
        });

        *cached = Some(CachedBuffers {
            padded_w,
            padded_h,
            width,
            height,
            tile_size,
            scratch_a,
            scratch_b,
            scratch_c,
            plane_results,
            ycocg_buf,
            rgb_out_buf,
            cropped_buf,
            packed_u8_buf,
            staging,
            staging_u8,
            y_ref_wavelet_buf,
            reference_planes,
            entropy_params,
            entropy_tile_info,
            entropy_var_a,
            entropy_var_b,
            entropy_var_a_cap: [var_a_init_cap; 3],
            entropy_var_b_cap: [var_b_init_cap; 3],
            crop_params_buf,
            pack_params_buf,
            cfl_alpha_buf,
            cfl_alpha_cap,
            plane_alpha_bufs,
            plane_alpha_cap,
            weight_map_buf,
            weight_map_cap,
            mv_buf,
            mv_cap,
            output_texture,
            output_texture_view,
            buf_to_tex_bind_group,
            buf_to_tex_params_buf,
        });
    }

    /// Grow a variable-size buffer if it's too small, using 2× growth strategy.
    fn ensure_var_buf(
        ctx: &GpuContext,
        buf: &mut wgpu::Buffer,
        cap: &mut u64,
        required: u64,
        label: &str,
        usage: wgpu::BufferUsages,
    ) {
        if required > *cap {
            let new_cap = (required * 2).max(4);
            *buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: new_cap,
                usage,
                mapped_at_creation: false,
            });
            *cap = new_cap;
        }
    }

    /// Write per-frame data into pre-allocated cached buffers.
    /// Handles entropy data, CfL alphas, weight map, and motion vectors.
    fn prepare_frame_data(&self, ctx: &GpuContext, frame: &CompressedFrame) {
        let mut cached = self.cached.borrow_mut();
        let bufs = cached.as_mut().unwrap();
        let info = &frame.info;
        let tiles_per_plane = info.tiles_x() as usize * info.tiles_y() as usize;

        let storage_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        // --- Entropy data ---
        match &frame.entropy {
            EntropyData::Rans(tiles) => {
                for p in 0..3 {
                    let plane_tiles = &tiles[p * tiles_per_plane..(p + 1) * tiles_per_plane];
                    let packed = GpuRansDecoder::pack_decode_data(plane_tiles, info);
                    let a_size = (packed.cumfreq.len() * 4) as u64;
                    let b_size = (packed.stream_data.len() * 4) as u64;

                    Self::ensure_var_buf(
                        ctx,
                        &mut bufs.entropy_var_a[p],
                        &mut bufs.entropy_var_a_cap[p],
                        a_size,
                        "dec_entropy_var_a",
                        storage_dst,
                    );
                    Self::ensure_var_buf(
                        ctx,
                        &mut bufs.entropy_var_b[p],
                        &mut bufs.entropy_var_b_cap[p],
                        b_size,
                        "dec_entropy_var_b",
                        storage_dst,
                    );

                    ctx.queue.write_buffer(
                        &bufs.entropy_params[p],
                        0,
                        bytemuck::bytes_of(&packed.params),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_tile_info[p],
                        0,
                        bytemuck::cast_slice(&packed.tile_info),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_var_a[p],
                        0,
                        bytemuck::cast_slice(&packed.cumfreq),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_var_b[p],
                        0,
                        bytemuck::cast_slice(&packed.stream_data),
                    );
                }
            }
            EntropyData::SubbandRans(tiles) => {
                for p in 0..3 {
                    let plane_tiles = &tiles[p * tiles_per_plane..(p + 1) * tiles_per_plane];
                    let packed = GpuRansDecoder::pack_decode_data_subband(plane_tiles, info);
                    let a_size = (packed.cumfreq.len() * 4) as u64;
                    let b_size = (packed.stream_data.len() * 4) as u64;

                    Self::ensure_var_buf(
                        ctx,
                        &mut bufs.entropy_var_a[p],
                        &mut bufs.entropy_var_a_cap[p],
                        a_size,
                        "dec_entropy_var_a",
                        storage_dst,
                    );
                    Self::ensure_var_buf(
                        ctx,
                        &mut bufs.entropy_var_b[p],
                        &mut bufs.entropy_var_b_cap[p],
                        b_size,
                        "dec_entropy_var_b",
                        storage_dst,
                    );

                    ctx.queue.write_buffer(
                        &bufs.entropy_params[p],
                        0,
                        bytemuck::bytes_of(&packed.params),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_tile_info[p],
                        0,
                        bytemuck::cast_slice(&packed.tile_info),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_var_a[p],
                        0,
                        bytemuck::cast_slice(&packed.cumfreq),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_var_b[p],
                        0,
                        bytemuck::cast_slice(&packed.stream_data),
                    );
                }
            }
            EntropyData::Bitplane(tiles) => {
                for p in 0..3 {
                    let plane_tiles = &tiles[p * tiles_per_plane..(p + 1) * tiles_per_plane];
                    let packed = GpuBitplaneDecoder::pack_decode_data(plane_tiles, info);
                    let a_size = (packed.block_info.len() * 4) as u64;
                    let b_size = (packed.bitplane_data.len() * 4) as u64;

                    Self::ensure_var_buf(
                        ctx,
                        &mut bufs.entropy_var_a[p],
                        &mut bufs.entropy_var_a_cap[p],
                        a_size,
                        "dec_entropy_var_a",
                        storage_dst,
                    );
                    Self::ensure_var_buf(
                        ctx,
                        &mut bufs.entropy_var_b[p],
                        &mut bufs.entropy_var_b_cap[p],
                        b_size,
                        "dec_entropy_var_b",
                        storage_dst,
                    );

                    ctx.queue.write_buffer(
                        &bufs.entropy_params[p],
                        0,
                        bytemuck::bytes_of(&packed.params),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_tile_info[p],
                        0,
                        bytemuck::cast_slice(&packed.tile_info),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_var_a[p],
                        0,
                        bytemuck::cast_slice(&packed.block_info),
                    );
                    ctx.queue.write_buffer(
                        &bufs.entropy_var_b[p],
                        0,
                        bytemuck::cast_slice(&packed.bitplane_data),
                    );
                }
            }
        }

        // --- CfL alphas ---
        if let Some(cfl_data) = &frame.cfl_alphas {
            let all_f32: Vec<f32> = cfl_data
                .alphas
                .iter()
                .map(|&q| cfl::dequantize_alpha(q))
                .collect();
            let alpha_size = (all_f32.len() * 4) as u64;
            Self::ensure_var_buf(
                ctx,
                &mut bufs.cfl_alpha_buf,
                &mut bufs.cfl_alpha_cap,
                alpha_size,
                "dec_cfl_alpha",
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            cfl::write_alpha_buffer_into(ctx, &all_f32, &bufs.cfl_alpha_buf);

            // Ensure plane_alpha_bufs are large enough
            let nsb = cfl_data.num_subbands as usize;
            let alphas_per_plane = tiles_per_plane * nsb;
            let plane_alpha_size = (alphas_per_plane * 4) as u64;
            if plane_alpha_size > bufs.plane_alpha_cap {
                let new_cap = (plane_alpha_size * 2).max(4);
                for i in 0..2 {
                    bufs.plane_alpha_bufs[i] = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("dec_plane_alpha_{i}")),
                        size: new_cap,
                        usage: storage_dst,
                        mapped_at_creation: false,
                    });
                }
                bufs.plane_alpha_cap = new_cap;
            }
        }

        // --- Weight map ---
        if let Some(wm) = &frame.weight_map {
            let wm_size = (wm.len() * 4) as u64;
            Self::ensure_var_buf(
                ctx,
                &mut bufs.weight_map_buf,
                &mut bufs.weight_map_cap,
                wm_size,
                "dec_weight_map",
                storage_dst,
            );
            ctx.queue
                .write_buffer(&bufs.weight_map_buf, 0, bytemuck::cast_slice(wm));
        }

        // --- Motion vectors ---
        if let Some(mf) = &frame.motion_field {
            let mv_size = (mf.vectors.len() * 2 * 4) as u64;
            Self::ensure_var_buf(
                ctx,
                &mut bufs.mv_buf,
                &mut bufs.mv_cap,
                mv_size,
                "dec_mv",
                storage_dst,
            );
            MotionEstimator::write_motion_vectors_into(ctx, &mf.vectors, &bufs.mv_buf);
        }
    }

    /// Encode GPU commands for the full decode pipeline up to and including crop.
    /// All buffers are read from CachedBuffers (written by prepare_frame_data).
    fn encode_gpu_work(
        &self,
        ctx: &GpuContext,
        frame: &CompressedFrame,
        bufs: &CachedBuffers,
    ) -> wgpu::CommandEncoder {
        let info = &frame.info;
        let config = &frame.config;
        let padded_w = info.padded_width();
        let padded_h = info.padded_height();
        let padded_pixels = (padded_w * padded_h) as usize;
        let is_pframe = frame.frame_type == FrameType::Predicted;
        let tiles_per_plane = info.tiles_x() as usize * info.tiles_y() as usize;
        let blocks_per_tile_side = info.tile_size as usize / 32;
        let blocks_per_tile = blocks_per_tile_side * blocks_per_tile_side;

        let weights_luma = config.subband_weights.pack_weights();
        let weights_chroma = config.subband_weights.pack_weights_chroma();
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let w = info.width;
        let h = info.height;
        let output_pixels = w * h;

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("decode_full"),
            });

        // CfL metadata
        let has_cfl = frame.cfl_alphas.is_some();
        let cfl_alphas_per_plane = if has_cfl {
            let cfl_data = frame.cfl_alphas.as_ref().unwrap();
            tiles_per_plane * cfl_data.num_subbands as usize
        } else {
            0
        };

        // Per-plane: entropy decode → dequantize → (CfL inverse predict) → inverse wavelet → copy to result buffer
        for p in 0..3 {
            match &frame.entropy {
                EntropyData::Rans(_) | EntropyData::SubbandRans(_) => {
                    self.rans_decoder.dispatch_decode(
                        ctx,
                        &mut cmd,
                        &bufs.entropy_params[p],
                        &bufs.entropy_tile_info[p],
                        &bufs.entropy_var_a[p],
                        &bufs.entropy_var_b[p],
                        &bufs.scratch_a,
                        tiles_per_plane as u32,
                    );
                }
                EntropyData::Bitplane(_) => {
                    let total_blocks = (tiles_per_plane * blocks_per_tile) as u32;
                    self.bitplane_decoder.dispatch_decode(
                        ctx,
                        &mut cmd,
                        &bufs.entropy_params[p],
                        &bufs.entropy_tile_info[p],
                        &bufs.entropy_var_a[p],
                        &bufs.entropy_var_b[p],
                        &bufs.scratch_a,
                        total_blocks,
                    );
                }
            }

            let weights = if p == 0 {
                &weights_luma
            } else {
                &weights_chroma
            };
            let wm_param = if frame.weight_map.is_some() {
                let blocks_x = padded_w.div_ceil(AQ_BLOCK_SIZE);
                Some((&bufs.weight_map_buf, AQ_BLOCK_SIZE, blocks_x))
            } else {
                None
            };
            self.quantize.dispatch_adaptive(
                ctx,
                &mut cmd,
                &bufs.scratch_a,
                &bufs.scratch_b,
                padded_pixels as u32,
                config.quantization_step,
                config.dead_zone,
                false,
                padded_w,
                padded_h,
                config.tile_size,
                config.wavelet_levels,
                weights,
                wm_param,
            );

            if p == 0 && has_cfl {
                // Save dequantized Y wavelet for CfL chroma prediction
                cmd.copy_buffer_to_buffer(
                    &bufs.scratch_b,
                    0,
                    &bufs.y_ref_wavelet_buf,
                    0,
                    plane_size,
                );
            }

            if p > 0 && has_cfl {
                // CfL inverse prediction: scratch_b (dequantized residual) + alpha * y_ref → scratch_c
                let plane_alpha_offset = (p - 1) * cfl_alphas_per_plane;
                let plane_alpha_byte_offset =
                    (plane_alpha_offset * std::mem::size_of::<f32>()) as u64;
                let plane_alpha_byte_size =
                    (cfl_alphas_per_plane * std::mem::size_of::<f32>()) as u64;

                // Copy this plane's alphas from the full alpha buffer to the per-plane buffer
                cmd.copy_buffer_to_buffer(
                    &bufs.cfl_alpha_buf,
                    plane_alpha_byte_offset,
                    &bufs.plane_alpha_bufs[p - 1],
                    0,
                    plane_alpha_byte_size,
                );

                self.cfl_predictor.dispatch_inverse(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_b,               // dequantized residual
                    &bufs.y_ref_wavelet_buf,       // reconstructed Y wavelet
                    &bufs.plane_alpha_bufs[p - 1], // per-tile per-subband alphas
                    &bufs.scratch_c,               // output: reconstructed chroma wavelet
                    padded_pixels as u32,
                    padded_w,
                    padded_h,
                    config.tile_size,
                    config.wavelet_levels,
                );

                // Inverse wavelet: scratch_c → (temp scratch_b) → scratch_a
                self.transform.inverse(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_c,
                    &bufs.scratch_b,
                    &bufs.scratch_a,
                    info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );
            } else {
                // Standard path: inverse wavelet from scratch_b
                self.transform.inverse(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_b,
                    &bufs.scratch_c,
                    &bufs.scratch_a,
                    info,
                    config.wavelet_levels,
                    config.wavelet_type,
                );
            }

            if is_pframe {
                // P-frame: scratch_a has residual, add MC prediction from reference
                self.motion.compensate(
                    ctx,
                    &mut cmd,
                    &bufs.scratch_a,
                    &bufs.reference_planes[p],
                    &bufs.mv_buf,
                    &bufs.plane_results[p],
                    padded_w,
                    padded_h,
                    false, // inverse: recon = residual + predicted
                );
            } else {
                // I-frame: scratch_a has reconstructed spatial data
                cmd.copy_buffer_to_buffer(
                    &bufs.scratch_a,
                    0,
                    &bufs.plane_results[p],
                    0,
                    plane_size,
                );
            }

            // Copy reconstructed plane to reference buffer for next frame
            cmd.copy_buffer_to_buffer(
                &bufs.plane_results[p],
                0,
                &bufs.reference_planes[p],
                0,
                plane_size,
            );
        }

        // GPU interleave: 3 planes → interleaved YCoCg
        self.interleaver.dispatch(
            ctx,
            &mut cmd,
            &bufs.plane_results[0],
            &bufs.plane_results[1],
            &bufs.plane_results[2],
            &bufs.ycocg_buf,
            padded_pixels as u32,
        );

        // Inverse color (YCoCg-R → RGB)
        self.color.dispatch(
            ctx,
            &mut cmd,
            &bufs.ycocg_buf,
            &bufs.rgb_out_buf,
            padded_w,
            padded_h,
            false,
        );

        // GPU crop: padded RGB → compact cropped output
        {
            let crop_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("crop_bg"),
                layout: &self.crop_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bufs.crop_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bufs.rgb_out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bufs.cropped_buf.as_entire_binding(),
                    },
                ],
            });

            let workgroups = output_pixels.div_ceil(256);
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("crop_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.crop_pipeline);
            pass.set_bind_group(0, &crop_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        cmd
    }

    /// Decode a compressed frame back to RGB f32 data.
    /// Returns Vec<f32> of length width * height * 3 (interleaved R,G,B).
    pub fn decode(&self, ctx: &GpuContext, frame: &CompressedFrame) -> Vec<f32> {
        let profile = std::env::var("GNC_PROFILE").is_ok();
        let t_start = std::time::Instant::now();

        let info = &frame.info;
        let w = info.width;
        let h = info.height;
        let output_pixels = w * h;
        let output_size = (output_pixels as u64) * 3 * 4;

        self.ensure_cached(
            ctx,
            info.padded_width(),
            info.padded_height(),
            w,
            h,
            info.tile_size,
        );

        let t_alloc = t_start.elapsed();

        self.prepare_frame_data(ctx, frame);

        let t_prepare = t_start.elapsed();

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let mut cmd = self.encode_gpu_work(ctx, frame, bufs);

        // Copy f32 cropped output to staging
        cmd.copy_buffer_to_buffer(&bufs.cropped_buf, 0, &bufs.staging, 0, output_size);

        let t_encode_cmd = t_start.elapsed();

        ctx.queue.submit(Some(cmd.finish()));

        let t_submit = t_start.elapsed();

        // Map staging and read back
        let slice = bufs.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        bufs.staging.unmap();
        drop(cached);

        if profile {
            let t_total = t_start.elapsed();
            eprintln!(
                "[decode profile] alloc={:.2}ms prepare={:.2}ms cmd={:.2}ms submit={:.2}ms readback={:.2}ms total={:.2}ms",
                t_alloc.as_secs_f64() * 1000.0,
                (t_prepare - t_alloc).as_secs_f64() * 1000.0,
                (t_encode_cmd - t_prepare).as_secs_f64() * 1000.0,
                (t_submit - t_encode_cmd).as_secs_f64() * 1000.0,
                (t_total - t_submit).as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
            );
        }

        result
    }

    /// Decode a compressed frame to packed RGB u8 data.
    /// Returns Vec<u8> of length width * height * 3. This is 4x smaller to read back
    /// from the GPU than the f32 path, significantly reducing readback latency.
    pub fn decode_u8(&self, ctx: &GpuContext, frame: &CompressedFrame) -> Vec<u8> {
        let profile = std::env::var("GNC_PROFILE").is_ok();
        let t_start = std::time::Instant::now();

        let info = &frame.info;
        let w = info.width;
        let h = info.height;
        let total_f32s = w * h * 3;
        let packed_u32s = total_f32s.div_ceil(4);
        let packed_byte_size = (packed_u32s as u64) * 4;

        self.ensure_cached(
            ctx,
            info.padded_width(),
            info.padded_height(),
            w,
            h,
            info.tile_size,
        );

        let t_alloc = t_start.elapsed();

        self.prepare_frame_data(ctx, frame);

        let t_prepare = t_start.elapsed();

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let mut cmd = self.encode_gpu_work(ctx, frame, bufs);

        // GPU pack: f32 → packed u8 (using cached pack_params_buf)
        {
            let pack_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pack_bg"),
                layout: &self.pack_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bufs.pack_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bufs.cropped_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bufs.packed_u8_buf.as_entire_binding(),
                    },
                ],
            });

            let workgroups = packed_u32s.div_ceil(256);
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pack_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &pack_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy packed u8 to staging
        cmd.copy_buffer_to_buffer(
            &bufs.packed_u8_buf,
            0,
            &bufs.staging_u8,
            0,
            packed_byte_size,
        );

        let t_encode_cmd = t_start.elapsed();

        ctx.queue.submit(Some(cmd.finish()));

        let t_submit = t_start.elapsed();

        // Map staging and read back
        let slice = bufs.staging_u8.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let bytes: &[u8] = &data;
        let result = bytes[..total_f32s as usize].to_vec();
        drop(data);
        bufs.staging_u8.unmap();
        drop(cached);

        if profile {
            let t_total = t_start.elapsed();
            eprintln!(
                "[decode_u8 profile] alloc={:.2}ms prepare={:.2}ms cmd={:.2}ms submit={:.2}ms readback={:.2}ms total={:.2}ms",
                t_alloc.as_secs_f64() * 1000.0,
                (t_prepare - t_alloc).as_secs_f64() * 1000.0,
                (t_encode_cmd - t_prepare).as_secs_f64() * 1000.0,
                (t_submit - t_encode_cmd).as_secs_f64() * 1000.0,
                (t_total - t_submit).as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
            );
        }

        result
    }

    /// Decode a compressed frame to an on-GPU rgba8unorm texture.
    /// Zero readback: data stays entirely on the GPU. Returns a reference to the
    /// texture view that can be used directly for rendering (e.g. as a sampled texture
    /// in a render pass). The texture is owned by the cached buffers and reused across
    /// frames — the caller must use or copy it before the next decode call.
    pub fn decode_to_texture(&self, ctx: &GpuContext, frame: &CompressedFrame) -> TextureHandle {
        let info = &frame.info;
        let w = info.width;
        let h = info.height;

        self.ensure_cached(
            ctx,
            info.padded_width(),
            info.padded_height(),
            w,
            h,
            info.tile_size,
        );

        self.prepare_frame_data(ctx, frame);

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let mut cmd = self.encode_gpu_work(ctx, frame, bufs);

        // Buffer-to-texture: cropped f32 RGB → rgba8unorm texture
        {
            let wg_x = w.div_ceil(16);
            let wg_y = h.div_ceil(16);
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("buf_to_tex_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.buf_to_tex_pipeline);
            pass.set_bind_group(0, &bufs.buf_to_tex_bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        ctx.queue.submit(Some(cmd.finish()));

        TextureHandle {
            width: w,
            height: h,
        }
    }

    /// Get a reference to the output texture view from the most recent decode_to_texture call.
    /// Returns None if no frame has been decoded yet.
    pub fn output_texture_view(&self) -> Option<std::cell::Ref<'_, wgpu::TextureView>> {
        let cached = self.cached.borrow();
        if cached.is_none() {
            return None;
        }
        Some(std::cell::Ref::map(cached, |c| {
            &c.as_ref().unwrap().output_texture_view
        }))
    }

    /// Submit GPU decode work without waiting for the result.
    /// Returns an opaque token that can be used with `finish_decode_u8` to get the result.
    /// This enables pipelined decode: submit frame N, do CPU work for frame N+1,
    /// then finish frame N's readback.
    pub fn submit_decode_u8(&self, ctx: &GpuContext, frame: &CompressedFrame) {
        let info = &frame.info;
        let w = info.width;
        let h = info.height;
        let total_f32s = w * h * 3;
        let packed_u32s = total_f32s.div_ceil(4);
        let packed_byte_size = (packed_u32s as u64) * 4;

        self.ensure_cached(
            ctx,
            info.padded_width(),
            info.padded_height(),
            w,
            h,
            info.tile_size,
        );

        self.prepare_frame_data(ctx, frame);

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let mut cmd = self.encode_gpu_work(ctx, frame, bufs);

        // GPU pack: f32 → packed u8 (using cached pack_params_buf)
        {
            let pack_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pack_bg"),
                layout: &self.pack_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bufs.pack_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bufs.cropped_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bufs.packed_u8_buf.as_entire_binding(),
                    },
                ],
            });

            let workgroups = packed_u32s.div_ceil(256);
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pack_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &pack_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy packed u8 to staging
        cmd.copy_buffer_to_buffer(
            &bufs.packed_u8_buf,
            0,
            &bufs.staging_u8,
            0,
            packed_byte_size,
        );

        ctx.queue.submit(Some(cmd.finish()));

        // Request map (non-blocking — will be ready after poll)
        let slice = bufs.staging_u8.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        drop(cached);

        // Store the receiver for later retrieval
        *self.pending_rx.borrow_mut() = Some(rx);
    }

    /// Finish a previously submitted decode_u8 operation.
    /// Blocks until the GPU work is complete and returns the u8 result.
    pub fn finish_decode_u8(&self, ctx: &GpuContext, width: u32, height: u32) -> Vec<u8> {
        let rx = self
            .pending_rx
            .borrow_mut()
            .take()
            .expect("finish_decode_u8 called without prior submit_decode_u8");

        let total_bytes = (width * height * 3) as usize;

        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let cached = self.cached.borrow();
        let bufs = cached.as_ref().unwrap();

        let slice = bufs.staging_u8.slice(..);
        let data = slice.get_mapped_range();
        let bytes: &[u8] = &data;
        let result = bytes[..total_bytes].to_vec();
        drop(data);
        bufs.staging_u8.unmap();
        drop(cached);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::pipeline::EncoderPipeline;
    use crate::CodecConfig;

    fn make_gradient_frame(w: u32, h: u32) -> Vec<f32> {
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = (x as f32 / w as f32 * 255.0).clamp(0.0, 255.0);
                let g = (y as f32 / h as f32 * 255.0).clamp(0.0, 255.0);
                let b = ((x + y) as f32 / (w + h) as f32 * 255.0).clamp(0.0, 255.0);
                data.push(r);
                data.push(g);
                data.push(b);
            }
        }
        data
    }

    #[test]
    fn test_decode_to_texture_dimensions() {
        let ctx = GpuContext::new();
        let mut enc = EncoderPipeline::new(&ctx);
        let dec = DecoderPipeline::new(&ctx);

        let w = 256;
        let h = 256;
        let frame_data = make_gradient_frame(w, h);

        let mut config = CodecConfig::default();
        config.tile_size = 256;
        config.keyframe_interval = 1;

        let compressed = enc.encode_sequence(&ctx, &[frame_data.as_slice()], w, h, &config);
        assert_eq!(compressed.len(), 1);

        let handle = dec.decode_to_texture(&ctx, &compressed[0]);
        assert_eq!(handle.width, w);
        assert_eq!(handle.height, h);

        // Verify the texture view is accessible
        let view = dec.output_texture_view();
        assert!(view.is_some());
    }

    #[test]
    fn test_decode_to_texture_non_square() {
        let ctx = GpuContext::new();
        let mut enc = EncoderPipeline::new(&ctx);
        let dec = DecoderPipeline::new(&ctx);

        let w = 320;
        let h = 192;
        let frame_data = make_gradient_frame(w, h);

        let mut config = CodecConfig::default();
        config.tile_size = 64;
        config.keyframe_interval = 1;

        let compressed = enc.encode_sequence(&ctx, &[frame_data.as_slice()], w, h, &config);
        let handle = dec.decode_to_texture(&ctx, &compressed[0]);
        assert_eq!(handle.width, w);
        assert_eq!(handle.height, h);
    }
}
