//! GPU rANS encoder — dispatches histogram + encode compute shaders.
//!
//! Two-pass pipeline:
//!   1. rans_histogram.wgsl: build per-tile histograms on GPU (256 threads/workgroup)
//!   2. CPU: read back histograms, normalize frequencies, build cumfreq tables
//!   3. rans_encode.wgsl: encode 32 interleaved rANS streams per tile (32 threads/workgroup)
//!   4. CPU: read back encoded streams, pack into tile structs
//!
//! Eliminates the 30MB quantized-coefficient readback and 170ms of serial CPU encoding.

use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use super::rans::{
    normalize_histogram, InterleavedRansTile, SubbandGroupFreqs, SubbandRansTile, STREAMS_PER_TILE,
};
use crate::{FrameInfo, GpuContext};

const RANS_M: u32 = 1 << 12; // Must match RANS_PRECISION in shaders
const MAX_STREAM_BYTES: usize = 4096;
const HIST_TILE_STRIDE: usize = 2060;
const ENCODE_TILE_INFO_STRIDE: usize = 32;

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
pub struct NormalizedTileFreqs {
    pub min_val: i32,
    pub alphabet_size: u32,
    pub freqs: Vec<u32>,
    pub cumfreqs: Vec<u32>,
}

/// Normalized frequency data for one tile (per-subband mode).
pub struct NormalizedSubbandTileFreqs {
    pub num_groups: u32,
    pub groups: Vec<NormalizedGroupFreqs>,
}

pub struct NormalizedGroupFreqs {
    pub min_val: i32,
    pub alphabet_size: u32,
    pub freqs: Vec<u32>,
    pub cumfreqs: Vec<u32>,
}

/// Unified normalized frequency data for one tile.
pub enum TileFreqs {
    Single(NormalizedTileFreqs),
    Subband(NormalizedSubbandTileFreqs),
}

pub struct GpuRansEncoder {
    histogram_pipeline: wgpu::ComputePipeline,
    histogram_bgl: wgpu::BindGroupLayout,
    encode_pipeline: wgpu::ComputePipeline,
    encode_bgl: wgpu::BindGroupLayout,
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

        let histogram_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("rans_histogram_bgl"),
                    entries: &[
                        make_uniform_entry(0),        // params
                        make_storage_entry(1, true),   // input coefficients
                        make_storage_entry(2, false),  // histogram output
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

        // --- Encode pipeline ---
        let enc_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rans_encode"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/rans_encode.wgsl").into(),
                ),
            });

        let encode_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("rans_encode_bgl"),
                    entries: &[
                        make_uniform_entry(0),        // params
                        make_storage_entry(1, true),   // input coefficients
                        make_storage_entry(2, true),   // cumfreq_data
                        make_storage_entry(3, true),   // tile_info
                        make_storage_entry(4, false),  // stream_output
                        make_storage_entry(5, false),  // stream_metadata
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
            encode_pipeline,
            encode_bgl,
        }
    }

    /// Run the full GPU encode pipeline for one plane's quantized data.
    /// Returns encoded tile data ready for serialization.
    ///
    /// `quantized_buf` must contain quantized f32 coefficients in 2D plane layout.
    pub fn encode_plane(
        &self,
        ctx: &GpuContext,
        quantized_buf: &wgpu::Buffer,
        info: &FrameInfo,
        per_subband: bool,
        num_levels: u32,
    ) -> Vec<TileFreqs> {
        let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;

        // Step 1: GPU histogram
        let hist_buf = self.dispatch_histogram(ctx, quantized_buf, info, per_subband, num_levels);

        // Step 2: Read back and normalize
        let hist_data = read_buffer_u32(ctx, &hist_buf, num_tiles * HIST_TILE_STRIDE);
        Self::normalize_histograms(&hist_data, num_tiles, per_subband)
    }

    /// Run the full GPU encode pipeline and return packed tile structs.
    #[allow(clippy::type_complexity)]
    pub fn encode_plane_to_tiles(
        &self,
        ctx: &GpuContext,
        quantized_buf: &wgpu::Buffer,
        info: &FrameInfo,
        per_subband: bool,
        num_levels: u32,
    ) -> (Vec<InterleavedRansTile>, Vec<SubbandRansTile>) {
        let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;

        // Step 1: GPU histogram
        let hist_buf = self.dispatch_histogram(ctx, quantized_buf, info, per_subband, num_levels);

        // Step 2: Read back histograms and normalize on CPU
        let hist_data = read_buffer_u32(ctx, &hist_buf, num_tiles * HIST_TILE_STRIDE);
        let tile_freqs = Self::normalize_histograms(&hist_data, num_tiles, per_subband);

        // Step 3: GPU encode
        let (stream_buf, meta_buf) =
            self.dispatch_encode(ctx, quantized_buf, &tile_freqs, info, per_subband, num_levels);

        // Step 4: Read back encoded streams
        let total_streams = num_tiles * STREAMS_PER_TILE;
        let stream_u32s = total_streams * MAX_STREAM_BYTES / 4;
        let meta_u32s = total_streams * 2;

        let stream_data = read_buffer_u32(ctx, &stream_buf, stream_u32s);
        let meta_data = read_buffer_u32(ctx, &meta_buf, meta_u32s);

        // Step 5: Pack into tile structs
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

    /// Dispatch the histogram shader. Returns the histogram output buffer.
    fn dispatch_histogram(
        &self,
        ctx: &GpuContext,
        quantized_buf: &wgpu::Buffer,
        info: &FrameInfo,
        per_subband: bool,
        num_levels: u32,
    ) -> wgpu::Buffer {
        let num_tiles = info.tiles_x() * info.tiles_y();
        let tile_size = info.tile_size;
        let coefficients_per_tile = tile_size * tile_size;

        let params = RansEncodeParams {
            num_tiles,
            coefficients_per_tile,
            plane_width: info.padded_width(),
            tile_size,
            tiles_x: info.tiles_x(),
            per_subband: if per_subband { 1 } else { 0 },
            num_levels,
            _pad: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rans_hist_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let hist_buf_size =
            (num_tiles as usize * HIST_TILE_STRIDE * std::mem::size_of::<u32>()) as u64;
        let hist_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rans_hist_output"),
            size: hist_buf_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rans_histogram_bg"),
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
                    resource: hist_buf.as_entire_binding(),
                },
            ],
        });

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rans_histogram_cmd"),
            });

        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rans_histogram_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.histogram_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_tiles, 1, 1);
        }

        ctx.queue.submit(Some(cmd.finish()));

        hist_buf
    }

    /// Read back raw histograms and normalize on CPU.
    fn normalize_histograms(
        hist_data: &[u32],
        num_tiles: usize,
        per_subband: bool,
    ) -> Vec<TileFreqs> {
        let mut tile_freqs = Vec::with_capacity(num_tiles);

        for t in 0..num_tiles {
            let base = t * HIST_TILE_STRIDE;

            if per_subband {
                let num_groups = hist_data[base] as usize;
                let mut groups = Vec::with_capacity(num_groups);
                let mut off = 1usize;

                for _g in 0..num_groups {
                    let min_val = hist_data[base + off] as i32;
                    let alphabet_size = hist_data[base + off + 1] as usize;
                    off += 2;

                    let hist: Vec<u32> = hist_data[base + off..base + off + alphabet_size].to_vec();
                    off += alphabet_size;

                    let freqs = normalize_histogram(&hist, RANS_M);
                    let mut cumfreqs = vec![0u32; alphabet_size + 1];
                    for i in 0..alphabet_size {
                        cumfreqs[i + 1] = cumfreqs[i] + freqs[i];
                    }

                    groups.push(NormalizedGroupFreqs {
                        min_val,
                        alphabet_size: alphabet_size as u32,
                        freqs,
                        cumfreqs,
                    });
                }

                tile_freqs.push(TileFreqs::Subband(NormalizedSubbandTileFreqs {
                    num_groups: num_groups as u32,
                    groups,
                }));
            } else {
                let min_val = hist_data[base] as i32;
                let alphabet_size = hist_data[base + 1] as usize;
                let hist: Vec<u32> =
                    hist_data[base + 2..base + 2 + alphabet_size].to_vec();

                let freqs = normalize_histogram(&hist, RANS_M);
                let mut cumfreqs = vec![0u32; alphabet_size + 1];
                for i in 0..alphabet_size {
                    cumfreqs[i + 1] = cumfreqs[i] + freqs[i];
                }

                tile_freqs.push(TileFreqs::Single(NormalizedTileFreqs {
                    min_val,
                    alphabet_size: alphabet_size as u32,
                    freqs,
                    cumfreqs,
                }));
            }
        }

        tile_freqs
    }

    /// Upload cumfreqs and dispatch GPU rANS encode.
    /// Returns (stream_output_buf, stream_metadata_buf).
    fn dispatch_encode(
        &self,
        ctx: &GpuContext,
        quantized_buf: &wgpu::Buffer,
        tile_freqs: &[TileFreqs],
        info: &FrameInfo,
        per_subband: bool,
        num_levels: u32,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let num_tiles = tile_freqs.len();
        let tile_size = info.tile_size;
        let coefficients_per_tile = tile_size * tile_size;
        let total_streams = num_tiles * STREAMS_PER_TILE;

        let params = RansEncodeParams {
            num_tiles: num_tiles as u32,
            coefficients_per_tile,
            plane_width: info.padded_width(),
            tile_size,
            tiles_x: info.tiles_x(),
            per_subband: if per_subband { 1 } else { 0 },
            num_levels,
            _pad: 0,
        };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rans_encode_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Build tile_info and cumfreq_data buffers
        let mut tile_info_data = vec![0u32; num_tiles * ENCODE_TILE_INFO_STRIDE];
        let mut cumfreq_all: Vec<u32> = Vec::new();

        for (t, tf) in tile_freqs.iter().enumerate() {
            let base = t * ENCODE_TILE_INFO_STRIDE;
            match tf {
                TileFreqs::Single(s) => {
                    tile_info_data[base] = s.min_val as u32;
                    tile_info_data[base + 1] = s.alphabet_size;
                    tile_info_data[base + 2] = cumfreq_all.len() as u32;
                    cumfreq_all.extend_from_slice(&s.cumfreqs);
                }
                TileFreqs::Subband(sb) => {
                    tile_info_data[base] = sb.num_groups;
                    for (g, group) in sb.groups.iter().enumerate() {
                        let gi = base + 1 + g * 3;
                        tile_info_data[gi] = group.min_val as u32;
                        tile_info_data[gi + 1] = group.alphabet_size;
                        tile_info_data[gi + 2] = cumfreq_all.len() as u32;
                        cumfreq_all.extend_from_slice(&group.cumfreqs);
                    }
                }
            }
        }

        // Ensure non-empty buffers
        if cumfreq_all.is_empty() {
            cumfreq_all.push(0);
        }

        let tile_info_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rans_encode_tile_info"),
                contents: bytemuck::cast_slice(&tile_info_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let cumfreq_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rans_encode_cumfreq"),
                contents: bytemuck::cast_slice(&cumfreq_all),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Stream output buffer: zero-initialized (required by write_byte)
        let stream_buf_size = (total_streams * MAX_STREAM_BYTES) as u64;
        let stream_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rans_encode_stream_output"),
            size: stream_buf_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        // Zero-initialize by mapping then unmapping (mapped_at_creation gives zeroed memory)
        stream_buf.unmap();

        // Metadata buffer: [write_ptr, final_state] per stream
        let meta_buf_size = (total_streams * 2 * std::mem::size_of::<u32>()) as u64;
        let meta_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rans_encode_stream_meta"),
            size: meta_buf_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    resource: cumfreq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tile_info_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: stream_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: meta_buf.as_entire_binding(),
                },
            ],
        });

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rans_encode_cmd"),
            });

        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rans_encode_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.encode_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_tiles as u32, 1, 1);
        }

        ctx.queue.submit(Some(cmd.finish()));

        (stream_buf, meta_buf)
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
            // Extract per-stream data from the GPU output
            let mut per_stream_data: Vec<Vec<u8>> = Vec::with_capacity(STREAMS_PER_TILE);
            let mut per_stream_state: Vec<u32> = Vec::with_capacity(STREAMS_PER_TILE);

            for s in 0..STREAMS_PER_TILE {
                let stream_idx = t * STREAMS_PER_TILE + s;
                let meta_base = stream_idx * 2;
                let write_ptr = meta_data[meta_base] as usize;
                let final_state = meta_data[meta_base + 1];

                // Extract bytes from write_ptr..MAX_STREAM_BYTES
                let stream_byte_base = stream_idx * MAX_STREAM_BYTES;
                let mut bytes = Vec::new();
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
                            zrun_base: 0, // GPU encode does not use ZRL
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

/// Result of GPU-encoding one plane.
pub enum PlaneEncodedTiles {
    Rans(Vec<InterleavedRansTile>),
    SubbandRans(Vec<SubbandRansTile>),
}

impl GpuRansEncoder {
    /// High-level: GPU encode a full plane, returning packed tile structs.
    /// This is the main entry point used by the encoder pipeline.
    pub fn gpu_encode_plane(
        &self,
        ctx: &GpuContext,
        quantized_buf: &wgpu::Buffer,
        info: &FrameInfo,
        per_subband: bool,
        num_levels: u32,
    ) -> PlaneEncodedTiles {
        let (rans, subband) =
            self.encode_plane_to_tiles(ctx, quantized_buf, info, per_subband, num_levels);
        if per_subband {
            PlaneEncodedTiles::SubbandRans(subband)
        } else {
            PlaneEncodedTiles::Rans(rans)
        }
    }
}

/// Read a GPU buffer back to CPU as Vec<u32>.
fn read_buffer_u32(ctx: &GpuContext, buffer: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let size = (count * std::mem::size_of::<u32>()) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_read_u32"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut cmd = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_to_staging_u32"),
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
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}
