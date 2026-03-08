use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use super::adaptive;
use super::motion::{ME_BIDIR_PRED_FINE_RANGE, ME_BIDIR_SEARCH_RANGE, ME_BLOCK_SIZE, ME_PRED_FINE_RANGE, ME_SEARCH_RANGE, ME_SPLIT_BLOCK_SIZE};
use crate::GpuContext;

/// Cached GPU buffers reused across encode() calls to avoid per-frame allocation.
pub(super) struct CachedEncodeBuffers {
    // Resolution these buffers were allocated for
    pub(super) padded_w: u32,
    pub(super) padded_h: u32,
    pub(super) orig_w: u32,
    pub(super) orig_h: u32,

    // Raw (unpadded) input buffer for GPU padding (size = orig_w * orig_h * 3 * f32)
    pub(super) raw_input_buf: wgpu::Buffer,

    // 3-channel buffers (size = 3 * plane_size)
    pub(super) input_buf: wgpu::Buffer,
    pub(super) color_out: wgpu::Buffer,

    // Single-plane work buffers (size = plane_size each)
    pub(super) plane_a: wgpu::Buffer,
    pub(super) plane_b: wgpu::Buffer,
    pub(super) plane_c: wgpu::Buffer,
    pub(super) co_plane: wgpu::Buffer,
    pub(super) cg_plane: wgpu::Buffer,
    /// Downsampled Co plane for 4:2:2 / 4:2:0 encode path (sized at worst-case 4:2:0).
    pub(super) co_plane_ds: wgpu::Buffer,
    /// Downsampled Cg plane for 4:2:2 / 4:2:0 encode path (sized at worst-case 4:2:0).
    pub(super) cg_plane_ds: wgpu::Buffer,
    pub(super) recon_y: wgpu::Buffer,

    // Adaptive quantization (size = wm_buf_size, fixed for given resolution)
    pub(super) variance_buf: wgpu::Buffer,
    pub(super) wm_scratch: wgpu::Buffer,
    pub(super) weight_map_buf: wgpu::Buffer,

    // Weight map staging for deferred readback (avoid intermediate GPU poll)
    pub(super) weight_map_staging: wgpu::Buffer,

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
    // Backward (future) reference planes for B-frame encoding.
    pub(super) gpu_bwd_ref_planes: [wgpu::Buffer; 3],

    // Per-plane histogram buffers for fused quantize+histogram path.
    // Each buffer holds HIST_TILE_STRIDE * num_tiles u32s.
    // Only allocated when fused path is used (lazily via ensure_fused_hist_bufs).
    pub(super) fused_hist_bufs: Option<[wgpu::Buffer; 3]>,

    // Intra prediction buffers (Y plane only, allocated when intra_prediction enabled)
    pub(super) intra_residual: Option<wgpu::Buffer>,
    pub(super) intra_modes_buf: Option<wgpu::Buffer>,
    pub(super) intra_modes_staging: Option<wgpu::Buffer>,

    // Cached pad params buffer (constant for given resolution)
    pub(super) pad_params_buf: wgpu::Buffer,

    // --- Cached ME/MC buffers (avoid per-frame allocation) ---

    // ME params (constant for given resolution, 2 variants per ME type)
    pub(super) me_params_nopred: wgpu::Buffer,
    pub(super) me_params_pred: wgpu::Buffer,
    pub(super) bidir_params_nopred: wgpu::Buffer,
    pub(super) bidir_params_pred: wgpu::Buffer,

    // ME scratch (overwritten each frame, not returned to caller)
    pub(super) me_sad_buf: wgpu::Buffer,
    pub(super) me_dummy_pred: wgpu::Buffer,
    pub(super) bidir_sad_buf: wgpu::Buffer,
    pub(super) bidir_modes_scratch: wgpu::Buffer,

    // MC params (used by bidir + split-decision 8×8)
    pub(super) mc_bidir_fwd_params: wgpu::Buffer,

    // MC params for block_size=8 (split decision output)
    pub(super) mc_fwd_params_8: wgpu::Buffer,
    pub(super) mc_inv_params_8: wgpu::Buffer,

    // MC params for 4:2:0 chroma (block_size=4, chroma dims).
    // Used to run MC at chroma dims with scaled MVs, avoiding NN-upsample residual artifacts.
    pub(super) mc_fwd_params_chroma420: wgpu::Buffer,
    pub(super) mc_inv_params_chroma420: wgpu::Buffer,

    // Scaled chroma MV buffer (same block count as split_mv, values halved for 4:2:0).
    // Written by dispatch_mv_scale before chroma MC; reused across planes.
    pub(super) mv_chroma_buf: wgpu::Buffer,

    // Staging buffers for deferred MV/bidir readback (reused across frames)
    pub(super) mv_staging_size: u64,
    // Staging buffer for 8x8-resolution split MVs (4x larger than 16x16)
    pub(super) split_mv_staging_buf: wgpu::Buffer,
    pub(super) split_mv_staging_size: u64,
    pub(super) me_total_blocks: u32,
    pub(super) split_total_blocks: u32,
    pub(super) bidir_fwd_staging: wgpu::Buffer,
    pub(super) bidir_bwd_staging: wgpu::Buffer,
    pub(super) bidir_modes_staging: wgpu::Buffer,
}

/// Cached GPU buffers for temporal wavelet GOP encoding.
/// Reused across GOPs to avoid per-GOP allocation overhead (~22ms saved).
pub(super) struct CachedTemporalWaveletBuffers {
    /// Padded dimensions these buffers were allocated for.
    pub(super) padded_w: u32,
    pub(super) padded_h: u32,
    /// Number of frames in the group (must be power of two).
    pub(super) group_size: usize,
    /// Raw input size (unpadded, 3-channel f32).
    pub(super) raw_input_size: u64,
    /// Per-frame wavelet coefficient buffers: [frame][plane Y/Co/Cg].
    pub(super) frame_bufs: Vec<[wgpu::Buffer; 3]>,
    /// Snapshot buffers for multilevel Haar aliasing avoidance.
    pub(super) snapshot: Vec<wgpu::Buffer>,
    /// Per-frame raw input staging buffers.
    pub(super) per_frame_input: Vec<wgpu::Buffer>,
    /// Per-tile energy multiplier output buffers (one per high frame, num_tiles × f32).
    /// GPU-computed by tile_energy_reduce shader; used directly as quantize weight maps.
    pub(super) tile_muls_bufs: Vec<wgpu::Buffer>,
    /// Per-tile raw mean_abs energy (pre-mapping). Used by CPU to identify tiles that
    /// are too energetic to benefit from temporal wavelet coding (zeroed in highpass).
    pub(super) tile_energies_bufs: Vec<wgpu::Buffer>,
    /// Staging buffers for tile_energies readback (MAP_READ | COPY_DST, num_tiles × f32).
    pub(super) tile_energies_staging_bufs: Vec<wgpu::Buffer>,
    /// Per-frame global-max-abs output buffers (1 × u32 each, atomicMax bitcast f32).
    /// STORAGE | COPY_SRC so we can copy to staging for CPU readback.
    pub(super) max_abs_bufs: Vec<wgpu::Buffer>,
    /// Staging buffers for max_abs readback (MAP_READ | COPY_DST, 4 bytes each).
    pub(super) max_abs_staging_bufs: Vec<wgpu::Buffer>,
    /// Shared uniform params buffer for tile_energy_reduce dispatches (32 bytes).
    pub(super) ter_params_buf: wgpu::Buffer,
    /// True if the next GOP's raw frames have already been written to `per_frame_input`
    /// via async pre-upload at the end of the previous GOP's encode. When set, the next
    /// `encode_temporal_wavelet_gop_haar` call skips its write_buffer upload step (saving ~22ms).
    pub(super) next_gop_pre_uploaded: bool,
    /// True if spatial wavelet + temporal Haar have already been computed for this buffer set,
    /// overlapping with the previous GOP's high_enc. When set, the encode call skips those
    /// phases and goes directly to low_enc (saving ~72ms/GOP).
    pub(super) spatial_haar_precomputed: bool,
}

impl CachedTemporalWaveletBuffers {
    pub(super) fn new(
        ctx: &GpuContext,
        padded_w: u32,
        padded_h: u32,
        group_size: usize,
        raw_input_size: u64,
    ) -> Self {
        let padded_pixels = (padded_w * padded_h) as usize;
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let frame_bufs: Vec<[wgpu::Buffer; 3]> = (0..group_size)
            .map(|j| {
                std::array::from_fn(|p| {
                    ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("tw_frame_{}_{}", j, p)),
                        size: plane_size,
                        usage: storage_usage,
                        mapped_at_creation: false,
                    })
                })
            })
            .collect();

        let snapshot: Vec<wgpu::Buffer> = (0..group_size)
            .map(|s| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("tw_snap_{}", s)),
                    size: plane_size,
                    usage: storage_usage,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let per_frame_input: Vec<wgpu::Buffer> = (0..group_size)
            .map(|j| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("tw_raw_input_{}", j)),
                    size: raw_input_size,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        // Tile energy reduce buffers.
        // The number of high frames per GOP = group_size - 1 (all except the lowpass frame).
        // tile_size is always 256 at this codebase's defaults; we use the safe upper bound
        // ceil(padded_w/256) * ceil(padded_h/256) for the tile count.
        let tile_size: u32 = 256;
        let tiles_x = padded_w.div_ceil(tile_size);
        let tiles_y = padded_h.div_ceil(tile_size);
        let num_tiles = (tiles_x * tiles_y) as usize;
        let tile_muls_size = (num_tiles * std::mem::size_of::<f32>()) as u64;
        let num_high_frames = group_size.saturating_sub(1).max(1);

        let tile_muls_bufs: Vec<wgpu::Buffer> = (0..num_high_frames)
            .map(|j| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("tw_tile_muls_{}", j)),
                    size: tile_muls_size.max(4),
                    // STORAGE for shader write, COPY_SRC so we can read back for diagnostics
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let tile_energies_bufs: Vec<wgpu::Buffer> = (0..num_high_frames)
            .map(|j| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("tw_tile_energies_{}", j)),
                    size: tile_muls_size.max(4),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let tile_energies_staging_bufs: Vec<wgpu::Buffer> = (0..num_high_frames)
            .map(|j| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("tw_tile_energies_stg_{}", j)),
                    size: tile_muls_size.max(4),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let max_abs_bufs: Vec<wgpu::Buffer> = (0..num_high_frames)
            .map(|j| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("tw_max_abs_{}", j)),
                    size: 4,
                    // COPY_DST needed for write_buffer pre-clear before each GOP
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let max_abs_staging_bufs: Vec<wgpu::Buffer> = (0..num_high_frames)
            .map(|j| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("tw_max_abs_stg_{}", j)),
                    size: 4,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        // Shared uniform params buffer for tile_energy_reduce (32 bytes, constant across frames)
        let ter_params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tw_ter_params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            padded_w,
            padded_h,
            group_size,
            raw_input_size,
            frame_bufs,
            snapshot,
            per_frame_input,
            tile_muls_bufs,
            tile_energies_bufs,
            tile_energies_staging_bufs,
            max_abs_bufs,
            max_abs_staging_bufs,
            ter_params_buf,
            next_gop_pre_uploaded: false,
            spatial_haar_precomputed: false,
        }
    }

    /// Check if these cached buffers are compatible with the given parameters.
    pub(super) fn is_compatible(
        &self,
        padded_w: u32,
        padded_h: u32,
        group_size: usize,
        raw_input_size: u64,
    ) -> bool {
        self.padded_w == padded_w
            && self.padded_h == padded_h
            && self.group_size == group_size
            && self.raw_input_size == raw_input_size
    }
}

impl CachedEncodeBuffers {
    /// Allocate all buffers for the given padded resolution.
    pub(super) fn new(
        ctx: &GpuContext,
        padded_w: u32,
        padded_h: u32,
        orig_w: u32,
        orig_h: u32,
    ) -> Self {
        let padded_pixels = (padded_w * padded_h) as usize;
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let buf_size_3 = (padded_pixels * 3 * std::mem::size_of::<f32>()) as u64;
        let raw_pixels = (orig_w * orig_h) as usize;
        let raw_buf_size = (raw_pixels * 3 * std::mem::size_of::<f32>()) as u64;

        // Upper bound on weight map blocks: assumes worst case (fewest wavelet levels).
        // With tile_size=256, num_levels=3: 4x4=16 blocks per tile (common case).
        // With num_levels=1: up to 64x64=4096 blocks per tile (extreme).
        // Use padded_pixels / (AQ_LL_BLOCK_SIZE^2) as a safe ceiling.
        let max_wm_blocks =
            padded_pixels / (adaptive::AQ_LL_BLOCK_SIZE * adaptive::AQ_LL_BLOCK_SIZE) as usize;
        let wm_buf_size = (max_wm_blocks.max(1) * std::mem::size_of::<f32>()) as u64;

        // ME block grid dimensions
        let me_blocks_x = padded_w / ME_BLOCK_SIZE;
        let me_blocks_y = padded_h / ME_BLOCK_SIZE;
        let me_total_blocks = me_blocks_x * me_blocks_y;
        let mv_staging_size = (me_total_blocks as u64) * 2 * 4; // i32 pairs

        // Initial alpha capacity: generous default
        let alpha_init_cap = 4096u64;

        let plane_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        Self {
            padded_w,
            padded_h,
            orig_w,
            orig_h,

            raw_input_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_raw_input"),
                size: raw_buf_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),

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
            // Downsampled chroma buffers: sized at plane_size (padded_w * padded_h) to
            // handle worst-case padded layout. For 4:2:2 with small images, chroma_padded_w
            // can equal padded_w (e.g. 128×256 chroma padded to 256×256 for tile_size=256),
            // so the padded downsampled buffer can require up to plane_size bytes.
            co_plane_ds: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_co_plane_ds"),
                size: plane_size,
                usage: plane_usage,
                mapped_at_creation: false,
            }),
            cg_plane_ds: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_cg_plane_ds"),
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

            weight_map_staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_wm_staging"),
                size: wm_buf_size.max(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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
            gpu_bwd_ref_planes: std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(
                        [
                            "enc_gpu_bwd_ref_y",
                            "enc_gpu_bwd_ref_co",
                            "enc_gpu_bwd_ref_cg",
                        ][i],
                    ),
                    size: plane_size,
                    usage: plane_usage,
                    mapped_at_creation: false,
                })
            }),

            fused_hist_bufs: None,

            intra_residual: None,
            intra_modes_buf: None,
            intra_modes_staging: None,

            pad_params_buf: {
                #[repr(C)]
                #[derive(Copy, Clone, Pod, Zeroable)]
                struct PadParams {
                    width: u32,
                    height: u32,
                    padded_w: u32,
                    padded_h: u32,
                }
                let p = PadParams {
                    width: orig_w,
                    height: orig_h,
                    padded_w,
                    padded_h,
                };
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("enc_pad_params"),
                        contents: bytemuck::bytes_of(&p),
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
            },

            // --- Cached ME/MC buffers ---
            me_params_nopred: Self::make_block_match_params(ctx, padded_w, padded_h, ME_SEARCH_RANGE, false, 0),
            me_params_pred: Self::make_block_match_params(ctx, padded_w, padded_h, ME_SEARCH_RANGE, true, ME_PRED_FINE_RANGE),
            bidir_params_nopred: Self::make_block_match_params(ctx, padded_w, padded_h, ME_BIDIR_SEARCH_RANGE, false, 0),
            bidir_params_pred: Self::make_block_match_params(ctx, padded_w, padded_h, ME_BIDIR_SEARCH_RANGE, true, ME_BIDIR_PRED_FINE_RANGE),

            me_sad_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_me_sad"),
                size: (me_total_blocks as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            me_dummy_pred: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_me_dummy_pred"),
                size: 8,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            bidir_sad_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_bidir_sad"),
                size: (me_total_blocks as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            bidir_modes_scratch: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_bidir_modes_scratch"),
                size: (me_total_blocks as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),

            mc_bidir_fwd_params: Self::make_mc_params_bs(ctx, padded_w, padded_h, true, ME_BLOCK_SIZE),
            mc_fwd_params_8: Self::make_mc_params_bs(ctx, padded_w, padded_h, true, ME_SPLIT_BLOCK_SIZE),
            mc_inv_params_8: Self::make_mc_params_bs(ctx, padded_w, padded_h, false, ME_SPLIT_BLOCK_SIZE),

            // 4:2:0 chroma MC params: chroma dims = padded/2, block_size = ME_SPLIT_BLOCK_SIZE/2 = 4.
            // The block grid is identical to luma (same count), only the spatial dims differ.
            mc_fwd_params_chroma420: Self::make_mc_params_bs(ctx, padded_w / 2, padded_h / 2, true, ME_SPLIT_BLOCK_SIZE / 2),
            mc_inv_params_chroma420: Self::make_mc_params_bs(ctx, padded_w / 2, padded_h / 2, false, ME_SPLIT_BLOCK_SIZE / 2),

            mv_chroma_buf: {
                let split_blocks_x = padded_w / ME_SPLIT_BLOCK_SIZE;
                let split_blocks_y = padded_h / ME_SPLIT_BLOCK_SIZE;
                let mv_size = (split_blocks_x * split_blocks_y) as u64 * 2 * 4;
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("enc_mv_chroma"),
                    size: mv_size.max(8),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            },

            mv_staging_size,
            split_mv_staging_buf: {
                let split_blocks_x = padded_w / ME_SPLIT_BLOCK_SIZE;
                let split_blocks_y = padded_h / ME_SPLIT_BLOCK_SIZE;
                let split_total = split_blocks_x * split_blocks_y;
                let split_staging_size = (split_total as u64) * 2 * 4;
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("enc_split_mv_staging"),
                    size: split_staging_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            },
            split_mv_staging_size: {
                let split_blocks_x = padded_w / ME_SPLIT_BLOCK_SIZE;
                let split_blocks_y = padded_h / ME_SPLIT_BLOCK_SIZE;
                let split_total = split_blocks_x * split_blocks_y;
                (split_total as u64) * 2 * 4
            },
            me_total_blocks,
            split_total_blocks: {
                let split_blocks_x = padded_w / ME_SPLIT_BLOCK_SIZE;
                let split_blocks_y = padded_h / ME_SPLIT_BLOCK_SIZE;
                split_blocks_x * split_blocks_y
            },
            bidir_fwd_staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_bidir_fwd_staging"),
                size: mv_staging_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            bidir_bwd_staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_bidir_bwd_staging"),
                size: mv_staging_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            bidir_modes_staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("enc_bidir_modes_staging"),
                size: (me_total_blocks as u64) * 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    fn make_block_match_params(
        ctx: &GpuContext,
        padded_w: u32,
        padded_h: u32,
        search_range: u32,
        use_predictor: bool,
        pred_fine_range: u32,
    ) -> wgpu::Buffer {
        #[repr(C)]
        #[derive(Copy, Clone, Pod, Zeroable)]
        struct BlockMatchParams {
            width: u32,
            height: u32,
            block_size: u32,
            search_range: u32,
            blocks_x: u32,
            total_blocks: u32,
            use_predictor: u32,
            pred_fine_range: u32,
        }
        let blocks_x = padded_w / ME_BLOCK_SIZE;
        let blocks_y = padded_h / ME_BLOCK_SIZE;
        let total_blocks = blocks_x * blocks_y;
        let params = BlockMatchParams {
            width: padded_w,
            height: padded_h,
            block_size: ME_BLOCK_SIZE,
            search_range,
            blocks_x,
            total_blocks,
            use_predictor: u32::from(use_predictor),
            pred_fine_range: if use_predictor { pred_fine_range } else { 0 },
        };
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("enc_me_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    fn make_mc_params_bs(ctx: &GpuContext, padded_w: u32, padded_h: u32, forward: bool, block_size: u32) -> wgpu::Buffer {
        #[repr(C)]
        #[derive(Copy, Clone, Pod, Zeroable)]
        struct MotionCompensateParams {
            width: u32,
            height: u32,
            block_size: u32,
            mode: u32,
            blocks_x: u32,
            total_pixels: u32,
            _pad0: u32,
            _pad1: u32,
        }
        let params = MotionCompensateParams {
            width: padded_w,
            height: padded_h,
            block_size,
            mode: if forward { 0 } else { 1 },
            blocks_x: padded_w / block_size,
            total_pixels: padded_w * padded_h,
            _pad0: 0,
            _pad1: 0,
        };
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("enc_mc_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Ensure intra prediction buffers are allocated (Y plane only).
    /// Allocates lazily on first use.
    pub(super) fn ensure_intra_bufs(&mut self, ctx: &GpuContext) {
        if self.intra_residual.is_some() {
            return;
        }
        let plane_size = (self.padded_w as u64) * (self.padded_h as u64) * 4;
        let num_blocks = (self.padded_w / 8) * (self.padded_h / 8);
        let modes_size = (num_blocks as u64) * 4; // u32 per block

        let plane_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        self.intra_residual = Some(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_intra_residual"),
            size: plane_size,
            usage: plane_usage,
            mapped_at_creation: false,
        }));
        self.intra_modes_buf = Some(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_intra_modes"),
            size: modes_size,
            usage: plane_usage,
            mapped_at_creation: false,
        }));
        self.intra_modes_staging = Some(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("enc_intra_modes_staging"),
            size: modes_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }

    /// Ensure per-plane histogram buffers are allocated for the fused quantize+histogram path.
    /// Allocates lazily on first use.
    pub(super) fn ensure_fused_hist_bufs(&mut self, ctx: &GpuContext, num_tiles: u32) {
        if self.fused_hist_bufs.is_some() {
            return;
        }
        const HIST_TILE_STRIDE: u64 = 32793;
        let hist_size = (num_tiles as u64) * HIST_TILE_STRIDE * 4;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        self.fused_hist_bufs = Some(std::array::from_fn(|i| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(["enc_fused_hist_y", "enc_fused_hist_co", "enc_fused_hist_cg"][i]),
                size: hist_size.max(4),
                usage,
                mapped_at_creation: false,
            })
        }));
    }
}

/// Minimal intermediate GPU buffers needed to run the spatial wavelet pipeline
/// (pad → color_convert → deinterleave → transform) for temporal wavelet GOP pipelining.
///
/// A second instance (`sp_cached_b`) allows the next GOP's spatial wavelet to run
/// concurrently with the current GOP's high_enc (Rice encoding), overlapping ~64ms
/// of GPU compute per GOP.
pub(super) struct SpatialPrecomputeBuffers {
    pub(super) padded_w: u32,
    pub(super) padded_h: u32,
    /// Raw (unpadded) input staging for GPU pad — COPY_DST.
    pub(super) raw_input_buf: wgpu::Buffer,
    /// 3-channel padded input buffer — STORAGE | COPY_DST.
    pub(super) input_buf: wgpu::Buffer,
    /// Color-converted output buffer — STORAGE | COPY_DST.
    pub(super) color_out: wgpu::Buffer,
    /// Per-plane buffers: Y, Co, Cg after deinterleave; input to transform.
    pub(super) plane_a: wgpu::Buffer,
    pub(super) co_plane: wgpu::Buffer,
    pub(super) cg_plane: wgpu::Buffer,
    /// Transform scratch buffer.
    pub(super) plane_b: wgpu::Buffer,
    /// Transform output buffer — STORAGE | COPY_SRC (copied to tw_b.frame_bufs).
    pub(super) plane_c: wgpu::Buffer,
}

impl SpatialPrecomputeBuffers {
    pub(super) fn new(ctx: &GpuContext, padded_w: u32, padded_h: u32, orig_w: u32, orig_h: u32) -> Self {
        let padded_pixels = (padded_w * padded_h) as usize;
        let plane_size = (padded_pixels * std::mem::size_of::<f32>()) as u64;
        let buf_3ch = (padded_pixels * 3 * std::mem::size_of::<f32>()) as u64;
        let raw_size = (orig_w * orig_h * 3) as u64 * std::mem::size_of::<f32>() as u64;

        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        // raw_input_buf: pad shader reads it as STORAGE binding (not just COPY_DST)
        let raw_storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        Self {
            padded_w,
            padded_h,
            raw_input_buf: mk("sp_b_raw_input", raw_size, raw_storage),
            input_buf:     mk("sp_b_input",     buf_3ch,  storage),
            color_out:     mk("sp_b_color_out", buf_3ch,  storage),
            plane_a:       mk("sp_b_plane_a",   plane_size, storage),
            co_plane:      mk("sp_b_co_plane",  plane_size, storage),
            cg_plane:      mk("sp_b_cg_plane",  plane_size, storage),
            plane_b:       mk("sp_b_plane_b",   plane_size, storage),
            plane_c:       mk("sp_b_plane_c",   plane_size, storage),
        }
    }

    pub(super) fn is_compatible(&self, padded_w: u32, padded_h: u32) -> bool {
        self.padded_w == padded_w && self.padded_h == padded_h
    }
}
