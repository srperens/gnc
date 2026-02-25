use wgpu;

use super::adaptive;
use crate::GpuContext;

/// Cached GPU buffers reused across encode() calls to avoid per-frame allocation.
pub(super) struct CachedEncodeBuffers {
    // Resolution these buffers were allocated for
    pub(super) padded_w: u32,
    pub(super) padded_h: u32,

    // 3-channel buffers (size = 3 * plane_size)
    pub(super) input_buf: wgpu::Buffer,
    pub(super) color_out: wgpu::Buffer,

    // Single-plane work buffers (size = plane_size each)
    pub(super) plane_a: wgpu::Buffer,
    pub(super) plane_b: wgpu::Buffer,
    pub(super) plane_c: wgpu::Buffer,
    pub(super) co_plane: wgpu::Buffer,
    pub(super) cg_plane: wgpu::Buffer,
    pub(super) recon_y: wgpu::Buffer,

    // Adaptive quantization (size = wm_buf_size, fixed for given resolution)
    pub(super) variance_buf: wgpu::Buffer,
    pub(super) wm_scratch: wgpu::Buffer,
    pub(super) weight_map_buf: wgpu::Buffer,

    // CfL alpha buffers (variable size, 2x growth, separate caps)
    pub(super) raw_alpha: wgpu::Buffer,
    pub(super) dq_alpha: wgpu::Buffer,
    pub(super) raw_alpha_cap: u64,
    pub(super) dq_alpha_cap: u64,

    // P-frame work buffers (size = plane_size each)
    pub(super) mc_out: wgpu::Buffer,
    pub(super) ref_upload: wgpu::Buffer,
    pub(super) recon_out: wgpu::Buffer,

    // Persistent GPU reference planes for sequence encoding (3 planes, size = plane_size each).
    // Eliminates CPU readback/re-upload of reference data between frames.
    pub(super) gpu_ref_planes: [wgpu::Buffer; 3],
}

impl CachedEncodeBuffers {
    /// Allocate all buffers for the given padded resolution.
    pub(super) fn new(ctx: &GpuContext, padded_w: u32, padded_h: u32) -> Self {
        let padded_pixels = (padded_w * padded_h) as usize;
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let buf_size_3 = (padded_pixels * 3 * std::mem::size_of::<f32>()) as u64;

        // Upper bound on weight map blocks: assumes worst case (fewest wavelet levels).
        // With tile_size=256, num_levels=3: 4x4=16 blocks per tile (common case).
        // With num_levels=1: up to 64x64=4096 blocks per tile (extreme).
        // Use padded_pixels / (AQ_LL_BLOCK_SIZE^2) as a safe ceiling.
        let max_wm_blocks =
            padded_pixels / (adaptive::AQ_LL_BLOCK_SIZE * adaptive::AQ_LL_BLOCK_SIZE) as usize;
        let wm_buf_size = (max_wm_blocks.max(1) * std::mem::size_of::<f32>()) as u64;

        // Initial alpha capacity: generous default
        let alpha_init_cap = 4096u64;

        let plane_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        Self {
            padded_w,
            padded_h,

            input_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_input"),
                size: buf_size_3,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            color_out: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_color_out"),
                size: buf_size_3,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),

            plane_a: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_plane_a"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),
            plane_b: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_plane_b"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),
            plane_c: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_plane_c"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),
            co_plane: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_co_plane"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),
            cg_plane: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_cg_plane"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),
            recon_y: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_recon_y"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),

            variance_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_variance"),
                size: wm_buf_size.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            wm_scratch: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_wm_scratch"),
                size: wm_buf_size.max(4),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            weight_map_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_weight_map"),
                size: wm_buf_size.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),

            raw_alpha: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_raw_alpha"),
                size: alpha_init_cap,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            dq_alpha: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_dq_alpha"),
                size: alpha_init_cap,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            raw_alpha_cap: alpha_init_cap,
            dq_alpha_cap: alpha_init_cap,

            mc_out: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_mc_out"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),
            ref_upload: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_ref_upload"),
                size: plane_size,
                usage: plane_usage, // needs COPY_SRC for CPU entropy encode readback
                mapped_at_creation: false,
            }),
            recon_out: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_recon_out"),
                size: plane_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),

            gpu_ref_planes: std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(["enc_gpu_ref_y", "enc_gpu_ref_co", "enc_gpu_ref_cg"][i]),
                    size: plane_size,
                    usage: plane_usage,
                    mapped_at_creation: false,
                })
            }),
        }
    }
}

/// Pad frame to tile-aligned dimensions by edge extension.
pub(super) fn pad_frame(data: &[f32], w: u32, h: u32, pw: u32, ph: u32) -> Vec<f32> {
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
