use bytemuck::{Pod, Zeroable};
use wgpu;

use crate::encoder::adaptive::AQ_LL_BLOCK_SIZE;
use crate::encoder::motion::ME_BLOCK_SIZE;
use crate::encoder::rans_gpu;
use crate::GpuContext;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct CropParams {
    pub(super) src_width: u32,
    pub(super) dst_width: u32,
    pub(super) dst_height: u32,
    pub(super) _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct PackParams {
    pub(super) total_f32s: u32,
    /// Max signal value: 255.0 for 8-bit, 1023.0 for 10-bit.
    pub(super) peak: f32,
    pub(super) _pad1: u32,
    pub(super) _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct TextureParams {
    pub(super) width: u32,
    pub(super) height: u32,
    /// Scale factor: 1.0 / max_val (1/255 for 8-bit, 1/1023 for 10-bit)
    pub(super) scale: f32,
    pub(super) _pad1: u32,
}

/// Cached GPU buffers for decode — allocated once, reused across frames.
/// Zero buffer allocations in steady-state decode: all buffers pre-allocated
/// on resolution init, data uploaded via queue.write_buffer().
pub(super) struct CachedBuffers {
    pub(super) padded_w: u32,
    pub(super) padded_h: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) tile_size: u32,
    pub(super) scratch_a: wgpu::Buffer,
    pub(super) scratch_b: wgpu::Buffer,
    pub(super) scratch_c: wgpu::Buffer,
    pub(super) plane_results: [wgpu::Buffer; 3],
    pub(super) ycocg_buf: wgpu::Buffer,
    pub(super) rgb_out_buf: wgpu::Buffer,
    pub(super) cropped_buf: wgpu::Buffer,
    pub(super) packed_u8_buf: wgpu::Buffer,
    pub(super) staging: wgpu::Buffer,
    pub(super) staging_u8: wgpu::Buffer,
    /// Dequantized Y wavelet coefficients for CfL prediction
    pub(super) y_ref_wavelet_buf: wgpu::Buffer,
    /// Upsampled Co plane for 4:2:2 / 4:2:0 decode path (luma-sized).
    pub(super) co_plane_up: wgpu::Buffer,
    /// Upsampled Cg plane for 4:2:2 / 4:2:0 decode path (luma-sized).
    pub(super) cg_plane_up: wgpu::Buffer,
    /// Reference planes from previous decoded frame (for temporal prediction)
    pub(super) reference_planes: [wgpu::Buffer; 3],

    // Per-plane entropy decode buffers (reused across frames)
    pub(super) entropy_params: [wgpu::Buffer; 3],
    pub(super) entropy_tile_info: [wgpu::Buffer; 3],
    pub(super) entropy_tile_info_cap: [u64; 3],
    pub(super) entropy_var_a: [wgpu::Buffer; 3],
    pub(super) entropy_var_b: [wgpu::Buffer; 3],
    pub(super) entropy_var_a_cap: [u64; 3],
    pub(super) entropy_var_b_cap: [u64; 3],

    // Fixed small params buffers (written once per resolution)
    pub(super) crop_params_buf: wgpu::Buffer,
    pub(super) pack_params_buf: wgpu::Buffer,

    // CfL buffers (reused when CfL enabled)
    pub(super) cfl_alpha_buf: wgpu::Buffer,
    pub(super) cfl_alpha_cap: u64,
    pub(super) plane_alpha_bufs: [wgpu::Buffer; 2],
    pub(super) plane_alpha_cap: u64,

    /// When true, entropy decoding was done on CPU (context-adaptive mode).
    /// `encode_gpu_work` should copy from `cpu_decoded_planes` instead of GPU decode.
    pub(super) ctx_adaptive_decode: bool,
    /// Per-plane CPU-decoded coefficient buffers (used for context-adaptive decode).
    pub(super) cpu_decoded_planes: [wgpu::Buffer; 3],

    // Conditional per-frame uploads
    pub(super) weight_map_buf: wgpu::Buffer,
    pub(super) weight_map_cap: u64,
    pub(super) mv_buf: wgpu::Buffer,
    pub(super) mv_cap: u64,

    // Intra prediction modes buffer (u32 per 8×8 block)
    pub(super) intra_modes_buf: wgpu::Buffer,
    pub(super) intra_modes_cap: u64,

    // B-frame buffers
    pub(super) bwd_mv_buf: wgpu::Buffer,
    pub(super) bwd_mv_cap: u64,
    pub(super) block_modes_buf: wgpu::Buffer,
    pub(super) block_modes_cap: u64,
    pub(super) bwd_reference_planes: [wgpu::Buffer; 3],

    // Motion compensation block size from MotionField (8 or 16)
    pub(super) mc_block_size: u32,

    // Output texture for zero-readback decode path
    #[allow(dead_code)]
    pub(super) output_texture: wgpu::Texture,
    pub(super) output_texture_view: wgpu::TextureView,
    pub(super) buf_to_tex_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    pub(super) buf_to_tex_params_buf: wgpu::Buffer,
}

impl CachedBuffers {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        ctx: &GpuContext,
        padded_w: u32,
        padded_h: u32,
        width: u32,
        height: u32,
        tile_size: u32,
        buf_to_tex_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
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

        // Upsample buffers for non-444 chroma: luma-sized, used when decoding 4:2:2 / 4:2:0.
        let co_plane_up = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_co_plane_up"),
            size: plane_size,
            usage: scratch_usage,
            mapped_at_creation: false,
        });
        let cg_plane_up = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_cg_plane_up"),
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
        let tile_info_size = (tiles_per_plane * rans_gpu::TILE_INFO_STRIDE * 4).max(4) as u64;
        let var_a_init_cap = (tiles_per_plane * 4096 * 4).max(4) as u64;
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
            // Default 8-bit; overwritten per-frame via write_buffer when bit_depth != 8
            peak: 255.0,
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

        // Weight map buffer: allocate for worst case (fewest wavelet levels).
        // Upper bound: padded_pixels / AQ_LL_BLOCK_SIZE^2 blocks.
        let max_wm_blocks = (padded_w * padded_h) / (AQ_LL_BLOCK_SIZE * AQ_LL_BLOCK_SIZE);
        let weight_map_cap = (max_wm_blocks.max(1) * 4) as u64;
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

        // Intra prediction modes buffer
        let intra_blocks = (padded_w / 8) * (padded_h / 8);
        let intra_modes_cap = (intra_blocks.max(1) * 4) as u64;
        let intra_modes_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_intra_modes"),
            size: intra_modes_cap,
            usage: storage_dst,
            mapped_at_creation: false,
        });

        // B-frame: backward motion vector buffer (small initial capacity)
        let bwd_mv_cap = 256u64;
        let bwd_mv_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_bwd_mv"),
            size: bwd_mv_cap,
            usage: storage_dst,
            mapped_at_creation: false,
        });

        // B-frame: block modes buffer (small initial capacity)
        let block_modes_cap = 256u64;
        let block_modes_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dec_block_modes"),
            size: block_modes_cap,
            usage: storage_dst,
            mapped_at_creation: false,
        });

        // B-frame: backward reference planes (same size/usage as forward reference planes)
        let bwd_reference_planes = std::array::from_fn(|p| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("dec_bwd_reference_{p}")),
                size: plane_size,
                usage: scratch_usage,
                mapped_at_creation: false,
            })
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
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_texture_view =
            output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let buf_to_tex_params = TextureParams {
            width,
            height,
            // Default 8-bit scale; overwritten per-frame via write_buffer when bit_depth != 8
            scale: 1.0 / 255.0,
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
            layout: buf_to_tex_bgl,
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

        Self {
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
            co_plane_up,
            cg_plane_up,
            reference_planes,
            entropy_params,
            entropy_tile_info,
            entropy_tile_info_cap: [tile_info_size; 3],
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
            ctx_adaptive_decode: false,
            cpu_decoded_planes: std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(["dec_cpu_plane0", "dec_cpu_plane1", "dec_cpu_plane2"][i]),
                    size: plane_size.max(4),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            }),
            weight_map_buf,
            weight_map_cap,
            mv_buf,
            mv_cap,
            intra_modes_buf,
            intra_modes_cap,
            bwd_mv_buf,
            bwd_mv_cap,
            block_modes_buf,
            block_modes_cap,
            bwd_reference_planes,
            mc_block_size: ME_BLOCK_SIZE, // default, overwritten per-frame in prepare_frame_data
            output_texture,
            output_texture_view,
            buf_to_tex_bind_group,
            buf_to_tex_params_buf,
        }
    }
}
