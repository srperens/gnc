//! GPU rANS encoder — fused histogram → normalize → encode pipeline.
//!
//! Three-pass GPU pipeline in a single command encoder submission:
//!   1. rans_histogram.wgsl: build per-tile histograms (256 threads/workgroup)
//!   2. rans_normalize.wgsl: normalize frequencies + build cumfreq tables (256 threads/workgroup)
//!   3. rans_encode.wgsl: encode 32 interleaved rANS streams per tile (32 threads/workgroup)
//!
//! All intermediate data stays on GPU — only the final encoded streams and
//! frequency tables are read back to CPU in a single map operation.
//! Buffers are cached across calls to eliminate per-frame allocation overhead.

use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use super::rans::{InterleavedRansTile, SubbandGroupFreqs, SubbandRansTile, STREAMS_PER_TILE};
use crate::{FrameInfo, GpuContext};

const MAX_STREAM_BYTES: usize = 4096;
const HIST_TILE_STRIDE: usize = 32793; // 1 + MAX_GROUPS*(3+MAX_GROUP_ALPHABET)
const ENCODE_TILE_INFO_STRIDE: usize = 32;
const MAX_ALPHABET: usize = 4096;
const MAX_GROUP_ALPHABET: usize = 4096;
const MAX_GROUPS: usize = 8;

fn cumfreq_stride(per_subband: bool) -> usize {
    if per_subband {
        MAX_GROUPS * (MAX_GROUP_ALPHABET + 1)
    } else {
        MAX_ALPHABET + 1
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RansEncodeParams {
    num_tiles: u32,
    coefficients_per_tile: u32,
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    per_subband: u32,
    num_levels: u32,
    _pad: u32,
}

/// Normalized frequency data for one tile (single-table mode).
struct NormalizedTileFreqs {
    min_val: i32,
    alphabet_size: u32,
    zrun_base: i32,
    freqs: Vec<u32>,
    cumfreqs: Vec<u32>,
}

/// Normalized frequency data for one tile (per-subband mode).
struct NormalizedSubbandTileFreqs {
    num_groups: u32,
    groups: Vec<NormalizedGroupFreqs>,
}

struct NormalizedGroupFreqs {
    min_val: i32,
    alphabet_size: u32,
    zrun_base: i32,
    freqs: Vec<u32>,
    cumfreqs: Vec<u32>,
}

/// Unified normalized frequency data for one tile.
enum TileFreqs {
    Single(NormalizedTileFreqs),
    Subband(NormalizedSubbandTileFreqs),
}

/// Pre-allocated GPU buffers, reused across encode calls for the same configuration.
struct CachedEncodeBuffers {
    num_tiles: usize,
    per_subband: bool,
    // Intermediate GPU buffers (reused per plane within a command encoder)
    hist_buf: wgpu::Buffer,
    cumfreq_buf: wgpu::Buffer,
    tile_info_buf: wgpu::Buffer,
    stream_buf: wgpu::Buffer,
    meta_buf: wgpu::Buffer,
    // Per-plane staging buffers for CPU readback (3 sets for batched 3-plane encode)
    stream_staging: [wgpu::Buffer; 3],
    meta_staging: [wgpu::Buffer; 3],
    cumfreq_staging: [wgpu::Buffer; 3],
    tile_info_staging: [wgpu::Buffer; 3],
}

impl CachedEncodeBuffers {
    fn new(ctx: &GpuContext, num_tiles: usize, per_subband: bool) -> Self {
        let total_streams = num_tiles * STREAMS_PER_TILE;
        let cf_stride = cumfreq_stride(per_subband);

        let hist_size = (num_tiles * HIST_TILE_STRIDE * 4) as u64;
        let cumfreq_size = (num_tiles * cf_stride * 4) as u64;
        let tile_info_size = (num_tiles * ENCODE_TILE_INFO_STRIDE * 4) as u64;
        let stream_size = (total_streams * MAX_STREAM_BYTES) as u64;
        let meta_size = (total_streams * 2 * 4) as u64;

        let sc = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let mr = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        Self {
            num_tiles,
            per_subband,
            hist_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rans_hist"),
                size: hist_size.max(4),
                usage: sc,
                mapped_at_creation: false,
            }),
            cumfreq_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rans_cumfreq"),
                size: cumfreq_size.max(4),
                usage: sc,
                mapped_at_creation: false,
            }),
            tile_info_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rans_tile_info"),
                size: tile_info_size.max(4),
                usage: sc,
                mapped_at_creation: false,
            }),
            stream_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rans_stream"),
                size: stream_size.max(4),
                usage: sc | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            meta_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rans_meta"),
                size: meta_size.max(4),
                usage: sc,
                mapped_at_creation: false,
            }),
            stream_staging: std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(["rans_stream_stg0", "rans_stream_stg1", "rans_stream_stg2"][i]),
                    size: stream_size.max(4),
                    usage: mr,
                    mapped_at_creation: false,
                })
            }),
            meta_staging: std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(["rans_meta_stg0", "rans_meta_stg1", "rans_meta_stg2"][i]),
                    size: meta_size.max(4),
                    usage: mr,
                    mapped_at_creation: false,
                })
            }),
            cumfreq_staging: std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(["rans_cf_stg0", "rans_cf_stg1", "rans_cf_stg2"][i]),
                    size: cumfreq_size.max(4),
                    usage: mr,
                    mapped_at_creation: false,
                })
            }),
            tile_info_staging: std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(["rans_ti_stg0", "rans_ti_stg1", "rans_ti_stg2"][i]),
                    size: tile_info_size.max(4),
                    usage: mr,
                    mapped_at_creation: false,
                })
            }),
        }
    }

    fn matches(&self, num_tiles: usize, per_subband: bool) -> bool {
        self.num_tiles == num_tiles && self.per_subband == per_subband
    }
}

pub struct GpuRansEncoder {
    histogram_pipeline: wgpu::ComputePipeline,
    histogram_bgl: wgpu::BindGroupLayout,
    normalize_pipeline: wgpu::ComputePipeline,
    normalize_bgl: wgpu::BindGroupLayout,
    encode_pipeline: wgpu::ComputePipeline,
    encode_bgl: wgpu::BindGroupLayout,
    cached: Option<CachedEncodeBuffers>,
}

fn make_storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

impl GpuRansEncoder {
    pub fn new(ctx: &GpuContext) -> Self {
        // --- Histogram pipeline ---
        let hist_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rans_histogram"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/rans_histogram.wgsl").into(),
                ),
            });

        let histogram_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rans_histogram_bgl"),
                entries: &[
                    make_uniform_entry(0),
                    make_storage_entry(1, true),
                    make_storage_entry(2, false),
                ],
            });

        let hist_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rans_histogram_pl"),
                bind_group_layouts: &[&histogram_bgl],
                push_constant_ranges: &[],
            });

        let histogram_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("rans_histogram_pipeline"),
                    layout: Some(&hist_pl),
                    module: &hist_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // --- Normalize pipeline ---
        let norm_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rans_normalize"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/rans_normalize.wgsl").into(),
                ),
            });

        let normalize_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rans_normalize_bgl"),
                entries: &[
                    make_uniform_entry(0),        // params
                    make_storage_entry(1, true),  // hist_input
                    make_storage_entry(2, false), // cumfreq_out
                    make_storage_entry(3, false), // tile_info_out
                ],
            });

        let norm_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rans_normalize_pl"),
                bind_group_layouts: &[&normalize_bgl],
                push_constant_ranges: &[],
            });

        let normalize_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("rans_normalize_pipeline"),
                    layout: Some(&norm_pl),
                    module: &norm_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // --- Encode pipeline ---
        let enc_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rans_encode"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/rans_encode.wgsl").into(),
                ),
            });

        let encode_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rans_encode_bgl"),
                entries: &[
                    make_uniform_entry(0),
                    make_storage_entry(1, true),
                    make_storage_entry(2, true),
                    make_storage_entry(3, true),
                    make_storage_entry(4, false),
                    make_storage_entry(5, false),
                ],
            });

        let enc_pl = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("rans_encode_pl"),
                bind_group_layouts: &[&encode_bgl],
                push_constant_ranges: &[],
            });

        let encode_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("rans_encode_pipeline"),
                    layout: Some(&enc_pl),
                    module: &enc_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            histogram_pipeline,
            histogram_bgl,
            normalize_pipeline,
            normalize_bgl,
            encode_pipeline,
            encode_bgl,
            cached: None,
        }
    }

    /// Ensure cached buffers match the current tile configuration.
    fn ensure_buffers(&mut self, ctx: &GpuContext, num_tiles: usize, per_subband: bool) {
        if self
            .cached
            .as_ref()
            .is_none_or(|c| !c.matches(num_tiles, per_subband))
        {
            self.cached = Some(CachedEncodeBuffers::new(ctx, num_tiles, per_subband));
        }
    }

    /// Run the fused GPU encode pipeline and return packed tile structs.
    ///
    /// Single command encoder submission: histogram → normalize → encode → copy.
    /// Single device poll for all readbacks.
    #[allow(clippy::type_complexity)]
    pub fn encode_plane_to_tiles(
        &mut self,
        ctx: &GpuContext,
        quantized_buf: &wgpu::Buffer,
        info: &FrameInfo,
        per_subband: bool,
        num_levels: u32,
    ) -> (Vec<InterleavedRansTile>, Vec<SubbandRansTile>) {
        let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;
        let total_streams = num_tiles * STREAMS_PER_TILE;
        let cf_stride = cumfreq_stride(per_subband);

        self.ensure_buffers(ctx, num_tiles, per_subband);
        let bufs = self.cached.as_ref().unwrap();

        // Params buffer (small, changes per plane)
        let params = RansEncodeParams {
            num_tiles: num_tiles as u32,
            coefficients_per_tile: info.tile_size * info.tile_size,
            plane_width: info.padded_width(),
            tile_size: info.tile_size,
            tiles_x: info.tiles_x(),
            per_subband: u32::from(per_subband),
            num_levels,
            _pad: 0,
        };
        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rans_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // --- Bind groups ---
        let hist_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rans_hist_bg"),
            layout: &self.histogram_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: quantized_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs.hist_buf.as_entire_binding(),
                },
            ],
        });

        let norm_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rans_norm_bg"),
            layout: &self.normalize_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bufs.hist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs.cumfreq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bufs.tile_info_buf.as_entire_binding(),
                },
            ],
        });

        let encode_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rans_encode_bg"),
            layout: &self.encode_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: quantized_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs.cumfreq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bufs.tile_info_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bufs.stream_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bufs.meta_buf.as_entire_binding(),
                },
            ],
        });

        // --- Single command encoder: 3 compute passes + copies ---
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rans_fused_cmd"),
            });

        // Zero-initialize stream buffer (required by write_byte OR pattern)
        cmd.clear_buffer(&bufs.stream_buf, 0, None);

        // Pass 1: Histogram
        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rans_hist_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.histogram_pipeline);
            pass.set_bind_group(0, &hist_bg, &[]);
            pass.dispatch_workgroups(num_tiles as u32, 1, 1);
        }

        // Pass 2: Normalize (reads hist_buf, writes cumfreq_buf + tile_info_buf)
        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rans_norm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.normalize_pipeline);
            pass.set_bind_group(0, &norm_bg, &[]);
            pass.dispatch_workgroups(num_tiles as u32, 1, 1);
        }

        // Pass 3: Encode (reads cumfreq_buf + tile_info_buf, writes stream_buf + meta_buf)
        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rans_encode_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.encode_pipeline);
            pass.set_bind_group(0, &encode_bg, &[]);
            pass.dispatch_workgroups(num_tiles as u32, 1, 1);
        }

        // Copy all results to staging buffers (plane slot 0)
        let stream_size = (total_streams * MAX_STREAM_BYTES) as u64;
        let meta_size = (total_streams * 2 * 4) as u64;
        let cumfreq_size = (num_tiles * cf_stride * 4) as u64;
        let tile_info_size = (num_tiles * ENCODE_TILE_INFO_STRIDE * 4) as u64;

        cmd.copy_buffer_to_buffer(&bufs.stream_buf, 0, &bufs.stream_staging[0], 0, stream_size);
        cmd.copy_buffer_to_buffer(&bufs.meta_buf, 0, &bufs.meta_staging[0], 0, meta_size);
        cmd.copy_buffer_to_buffer(
            &bufs.cumfreq_buf,
            0,
            &bufs.cumfreq_staging[0],
            0,
            cumfreq_size,
        );
        cmd.copy_buffer_to_buffer(
            &bufs.tile_info_buf,
            0,
            &bufs.tile_info_staging[0],
            0,
            tile_info_size,
        );

        // Single submit
        ctx.queue.submit(Some(cmd.finish()));

        // Map all 4 staging buffers, single poll
        let (tx, rx) = std::sync::mpsc::channel();
        for staging in [
            &bufs.stream_staging[0],
            &bufs.meta_staging[0],
            &bufs.cumfreq_staging[0],
            &bufs.tile_info_staging[0],
        ] {
            let tx_clone = tx.clone();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx_clone.send(result).unwrap();
                });
        }
        drop(tx);
        ctx.device.poll(wgpu::Maintain::Wait);
        for _ in 0..4 {
            rx.recv().unwrap().unwrap();
        }

        // Read back all data
        let stream_data: Vec<u32> = {
            let view = bufs.stream_staging[0].slice(..).get_mapped_range();
            bytemuck::cast_slice(&view).to_vec()
        };
        let meta_data: Vec<u32> = {
            let view = bufs.meta_staging[0].slice(..).get_mapped_range();
            bytemuck::cast_slice(&view).to_vec()
        };
        let cumfreq_data: Vec<u32> = {
            let view = bufs.cumfreq_staging[0].slice(..).get_mapped_range();
            bytemuck::cast_slice(&view).to_vec()
        };
        let tile_info_data: Vec<u32> = {
            let view = bufs.tile_info_staging[0].slice(..).get_mapped_range();
            bytemuck::cast_slice(&view).to_vec()
        };

        bufs.stream_staging[0].unmap();
        bufs.meta_staging[0].unmap();
        bufs.cumfreq_staging[0].unmap();
        bufs.tile_info_staging[0].unmap();

        // Reconstruct TileFreqs from GPU-computed cumfreq + tile_info
        let tile_freqs =
            Self::reconstruct_tile_freqs(&cumfreq_data, &tile_info_data, num_tiles, per_subband);

        // Pack into tile structs
        Self::pack_tiles(
            &stream_data,
            &meta_data,
            &tile_freqs,
            num_tiles,
            info,
            per_subband,
            num_levels,
        )
    }

    /// Batched 3-plane GPU rANS encode: all 3 planes in one command encoder, single poll.
    /// Intermediate buffers (hist, cumfreq, etc.) are reused per plane since compute passes
    /// execute sequentially. Each plane's results are copied to separate staging buffers.
    pub fn encode_3planes_to_tiles(
        &mut self,
        ctx: &GpuContext,
        quantized_bufs: [&wgpu::Buffer; 3],
        info: &FrameInfo,
        per_subband: bool,
        num_levels: u32,
    ) -> (Vec<InterleavedRansTile>, Vec<SubbandRansTile>) {
        let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;
        let total_streams = num_tiles * STREAMS_PER_TILE;
        let cf_stride = cumfreq_stride(per_subband);

        self.ensure_buffers(ctx, num_tiles, per_subband);
        let bufs = self.cached.as_ref().unwrap();

        let stream_size = (total_streams * MAX_STREAM_BYTES) as u64;
        let meta_size = (total_streams * 2 * 4) as u64;
        let cumfreq_size = (num_tiles * cf_stride * 4) as u64;
        let tile_info_size = (num_tiles * ENCODE_TILE_INFO_STRIDE * 4) as u64;

        let params = RansEncodeParams {
            num_tiles: num_tiles as u32,
            coefficients_per_tile: info.tile_size * info.tile_size,
            plane_width: info.padded_width(),
            tile_size: info.tile_size,
            tiles_x: info.tiles_x(),
            per_subband: u32::from(per_subband),
            num_levels,
            _pad: 0,
        };
        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rans_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rans_3plane_cmd"),
            });

        // Dispatch all 3 planes sequentially in one command encoder
        for p in 0..3 {
            // Build bind groups for this plane's quantized buffer
            let hist_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rans_hist_bg"),
                layout: &self.histogram_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: quantized_bufs[p].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bufs.hist_buf.as_entire_binding(),
                    },
                ],
            });

            let norm_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rans_norm_bg"),
                layout: &self.normalize_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bufs.hist_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bufs.cumfreq_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bufs.tile_info_buf.as_entire_binding(),
                    },
                ],
            });

            let encode_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rans_encode_bg"),
                layout: &self.encode_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: quantized_bufs[p].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bufs.cumfreq_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bufs.tile_info_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: bufs.stream_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: bufs.meta_buf.as_entire_binding(),
                    },
                ],
            });

            // Zero-initialize stream buffer
            cmd.clear_buffer(&bufs.stream_buf, 0, None);

            // 3 compute passes
            {
                let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rans_hist_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.histogram_pipeline);
                pass.set_bind_group(0, &hist_bg, &[]);
                pass.dispatch_workgroups(num_tiles as u32, 1, 1);
            }
            {
                let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rans_norm_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.normalize_pipeline);
                pass.set_bind_group(0, &norm_bg, &[]);
                pass.dispatch_workgroups(num_tiles as u32, 1, 1);
            }
            {
                let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rans_encode_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.encode_pipeline);
                pass.set_bind_group(0, &encode_bg, &[]);
                pass.dispatch_workgroups(num_tiles as u32, 1, 1);
            }

            // Copy results to this plane's staging buffers
            cmd.copy_buffer_to_buffer(&bufs.stream_buf, 0, &bufs.stream_staging[p], 0, stream_size);
            cmd.copy_buffer_to_buffer(&bufs.meta_buf, 0, &bufs.meta_staging[p], 0, meta_size);
            cmd.copy_buffer_to_buffer(
                &bufs.cumfreq_buf,
                0,
                &bufs.cumfreq_staging[p],
                0,
                cumfreq_size,
            );
            cmd.copy_buffer_to_buffer(
                &bufs.tile_info_buf,
                0,
                &bufs.tile_info_staging[p],
                0,
                tile_info_size,
            );
        }

        // Single submit for all 3 planes
        ctx.queue.submit(Some(cmd.finish()));

        // Map all 12 staging buffers, single poll
        let (tx, rx) = std::sync::mpsc::channel();
        for p in 0..3 {
            for staging in [
                &bufs.stream_staging[p],
                &bufs.meta_staging[p],
                &bufs.cumfreq_staging[p],
                &bufs.tile_info_staging[p],
            ] {
                let tx_clone = tx.clone();
                staging
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        tx_clone.send(result).unwrap();
                    });
            }
        }
        drop(tx);
        ctx.device.poll(wgpu::Maintain::Wait);
        for _ in 0..12 {
            rx.recv().unwrap().unwrap();
        }

        // Read back and pack all 3 planes
        let mut all_rans_tiles = Vec::new();
        let mut all_subband_tiles = Vec::new();

        for p in 0..3 {
            let stream_data: Vec<u32> = {
                let view = bufs.stream_staging[p].slice(..).get_mapped_range();
                bytemuck::cast_slice(&view).to_vec()
            };
            let meta_data: Vec<u32> = {
                let view = bufs.meta_staging[p].slice(..).get_mapped_range();
                bytemuck::cast_slice(&view).to_vec()
            };
            let cumfreq_data: Vec<u32> = {
                let view = bufs.cumfreq_staging[p].slice(..).get_mapped_range();
                bytemuck::cast_slice(&view).to_vec()
            };
            let tile_info_data: Vec<u32> = {
                let view = bufs.tile_info_staging[p].slice(..).get_mapped_range();
                bytemuck::cast_slice(&view).to_vec()
            };

            bufs.stream_staging[p].unmap();
            bufs.meta_staging[p].unmap();
            bufs.cumfreq_staging[p].unmap();
            bufs.tile_info_staging[p].unmap();

            let tile_freqs = Self::reconstruct_tile_freqs(
                &cumfreq_data,
                &tile_info_data,
                num_tiles,
                per_subband,
            );

            let (mut rt, mut st) = Self::pack_tiles(
                &stream_data,
                &meta_data,
                &tile_freqs,
                num_tiles,
                info,
                per_subband,
                num_levels,
            );
            all_rans_tiles.append(&mut rt);
            all_subband_tiles.append(&mut st);
        }

        (all_rans_tiles, all_subband_tiles)
    }

    /// Reconstruct TileFreqs from GPU-computed cumfreq and tile_info readbacks.
    /// Frequencies are derived as freq[i] = cumfreq[i+1] - cumfreq[i].
    fn reconstruct_tile_freqs(
        cumfreq_data: &[u32],
        tile_info_data: &[u32],
        num_tiles: usize,
        per_subband: bool,
    ) -> Vec<TileFreqs> {
        let mut tile_freqs = Vec::with_capacity(num_tiles);

        for t in 0..num_tiles {
            let info_base = t * ENCODE_TILE_INFO_STRIDE;

            if per_subband {
                let num_groups = tile_info_data[info_base] as usize;
                let mut groups = Vec::with_capacity(num_groups);

                for g in 0..num_groups {
                    let gi = info_base + 1 + g * 4;
                    let min_val = tile_info_data[gi] as i32;
                    let alphabet_size = tile_info_data[gi + 1] as usize;
                    let cf_offset = tile_info_data[gi + 2] as usize;
                    let zrun_base = tile_info_data[gi + 3] as i32;

                    let cumfreqs: Vec<u32> =
                        cumfreq_data[cf_offset..cf_offset + alphabet_size + 1].to_vec();
                    let freqs: Vec<u32> = (0..alphabet_size)
                        .map(|i| cumfreqs[i + 1] - cumfreqs[i])
                        .collect();

                    groups.push(NormalizedGroupFreqs {
                        min_val,
                        alphabet_size: alphabet_size as u32,
                        zrun_base,
                        freqs,
                        cumfreqs,
                    });
                }

                tile_freqs.push(TileFreqs::Subband(NormalizedSubbandTileFreqs {
                    num_groups: num_groups as u32,
                    groups,
                }));
            } else {
                let min_val = tile_info_data[info_base] as i32;
                let alphabet_size = tile_info_data[info_base + 1] as usize;
                let cf_offset = tile_info_data[info_base + 2] as usize;
                let zrun_base = tile_info_data[info_base + 3] as i32;

                let cumfreqs: Vec<u32> =
                    cumfreq_data[cf_offset..cf_offset + alphabet_size + 1].to_vec();
                let freqs: Vec<u32> = (0..alphabet_size)
                    .map(|i| cumfreqs[i + 1] - cumfreqs[i])
                    .collect();

                tile_freqs.push(TileFreqs::Single(NormalizedTileFreqs {
                    min_val,
                    alphabet_size: alphabet_size as u32,
                    zrun_base,
                    freqs,
                    cumfreqs,
                }));
            }
        }

        tile_freqs
    }

    /// Read back encoded streams and pack into tile structs.
    #[allow(clippy::too_many_arguments)]
    fn pack_tiles(
        stream_data: &[u32],
        meta_data: &[u32],
        tile_freqs: &[TileFreqs],
        num_tiles: usize,
        info: &FrameInfo,
        per_subband: bool,
        num_levels: u32,
    ) -> (Vec<InterleavedRansTile>, Vec<SubbandRansTile>) {
        let mut rans_tiles = Vec::new();
        let mut subband_tiles = Vec::new();
        let coefficients_per_tile = (info.tile_size * info.tile_size) as usize;

        for t in 0..num_tiles {
            let mut per_stream_data: Vec<Vec<u8>> = Vec::with_capacity(STREAMS_PER_TILE);
            let mut per_stream_state: Vec<u32> = Vec::with_capacity(STREAMS_PER_TILE);

            for s in 0..STREAMS_PER_TILE {
                let stream_idx = t * STREAMS_PER_TILE + s;
                let meta_base = stream_idx * 2;
                let write_ptr = meta_data[meta_base] as usize;
                let final_state = meta_data[meta_base + 1];

                let stream_byte_base = stream_idx * MAX_STREAM_BYTES;
                let mut bytes = Vec::with_capacity(MAX_STREAM_BYTES - write_ptr);
                for byte_off in write_ptr..MAX_STREAM_BYTES {
                    let global_byte = stream_byte_base + byte_off;
                    let word_idx = global_byte / 4;
                    let byte_pos = global_byte % 4;
                    let byte_val = (stream_data[word_idx] >> (byte_pos as u32 * 8)) & 0xFF;
                    bytes.push(byte_val as u8);
                }

                per_stream_data.push(bytes);
                per_stream_state.push(final_state);
            }

            if per_subband {
                match &tile_freqs[t] {
                    TileFreqs::Subband(sb) => {
                        let groups: Vec<SubbandGroupFreqs> = sb
                            .groups
                            .iter()
                            .map(|g| SubbandGroupFreqs {
                                min_val: g.min_val,
                                alphabet_size: g.alphabet_size,
                                zrun_base: g.zrun_base,
                                freqs: g.freqs.clone(),
                                cumfreqs: g.cumfreqs.clone(),
                            })
                            .collect();

                        subband_tiles.push(SubbandRansTile {
                            num_coefficients: coefficients_per_tile as u32,
                            tile_size: info.tile_size,
                            num_levels,
                            num_groups: sb.num_groups,
                            groups,
                            stream_data: per_stream_data,
                            stream_initial_state: per_stream_state,
                        });
                    }
                    _ => unreachable!("Expected subband tile freqs"),
                }
            } else {
                match &tile_freqs[t] {
                    TileFreqs::Single(s) => {
                        rans_tiles.push(InterleavedRansTile {
                            min_val: s.min_val,
                            alphabet_size: s.alphabet_size,
                            num_coefficients: coefficients_per_tile as u32,
                            zrun_base: s.zrun_base,
                            freqs: s.freqs.clone(),
                            cumfreqs: s.cumfreqs.clone(),
                            stream_data: per_stream_data,
                            stream_initial_state: per_stream_state,
                        });
                    }
                    _ => unreachable!("Expected single tile freqs"),
                }
            }
        }

        (rans_tiles, subband_tiles)
    }
}
