//! GPU Rice encoder/decoder — 256 interleaved streams per tile.
//!
//! Single-pass GPU pipeline: each workgroup (256 threads) encodes/decodes one tile.
//! Phase 1: Cooperatively compute optimal k per subband group.
//! Phase 2: Each thread encodes/decodes its stream independently (no state chain).
//!
//! Encode output is read back to CPU and packed into `RiceTile` structs.
//! Decode input is unpacked from `RiceTile` structs and uploaded to GPU.

use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use super::rice::{RiceTile, RICE_STREAMS_PER_TILE};
use crate::{FrameInfo, GpuContext};

const MAX_STREAM_BYTES: usize = 2048;
const MAX_GROUPS: usize = 8;
const K_STRIDE: usize = MAX_GROUPS * 2; // stride per tile in k_output (mag k + zrl k per group)

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RiceParams {
    num_tiles: u32,
    coefficients_per_tile: u32,
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    num_levels: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Pre-allocated GPU buffers for Rice encode, reused across calls.
struct CachedRiceEncodeBuffers {
    num_tiles: usize,
    stream_buf: wgpu::Buffer,
    lengths_buf: wgpu::Buffer,
    k_buf: wgpu::Buffer,
    // Per-plane staging for batched 3-plane encode
    stream_staging: [wgpu::Buffer; 3],
    lengths_staging: [wgpu::Buffer; 3],
    k_staging: [wgpu::Buffer; 3],
}

impl CachedRiceEncodeBuffers {
    fn new(ctx: &GpuContext, num_tiles: usize) -> Self {
        let total_streams = num_tiles * RICE_STREAMS_PER_TILE;
        let stream_size = (total_streams * MAX_STREAM_BYTES) as u64;
        let lengths_size = (total_streams * 4) as u64;
        let k_size = (num_tiles * K_STRIDE * 4) as u64;

        let sc = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let mr = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        Self {
            num_tiles,
            stream_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rice_stream"),
                size: stream_size.max(4),
                usage: sc | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            lengths_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rice_lengths"),
                size: lengths_size.max(4),
                usage: sc,
                mapped_at_creation: false,
            }),
            k_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rice_k"),
                size: k_size.max(4),
                usage: sc,
                mapped_at_creation: false,
            }),
            stream_staging: std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(
                        ["rice_stream_stg0", "rice_stream_stg1", "rice_stream_stg2"][i],
                    ),
                    size: stream_size.max(4),
                    usage: mr,
                    mapped_at_creation: false,
                })
            }),
            lengths_staging: std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(
                        ["rice_lengths_stg0", "rice_lengths_stg1", "rice_lengths_stg2"][i],
                    ),
                    size: lengths_size.max(4),
                    usage: mr,
                    mapped_at_creation: false,
                })
            }),
            k_staging: std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(["rice_k_stg0", "rice_k_stg1", "rice_k_stg2"][i]),
                    size: k_size.max(4),
                    usage: mr,
                    mapped_at_creation: false,
                })
            }),
        }
    }
}

/// GPU Rice encoder — encodes 3 planes of quantized coefficients into RiceTile structs.
pub struct GpuRiceEncoder {
    encode_pipeline: wgpu::ComputePipeline,
    encode_bgl: wgpu::BindGroupLayout,
    cached: Option<CachedRiceEncodeBuffers>,
}

impl GpuRiceEncoder {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rice_encode_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/rice_encode.wgsl").into(),
                ),
            });

        let encode_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("rice_encode_bgl"),
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
                        // binding 1: input coefficients (read-only)
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
                        // binding 2: stream_output (read-write)
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
                        // binding 3: stream_lengths (read-write)
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
                        // binding 4: k_output (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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

        let pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("rice_encode_layout"),
                    bind_group_layouts: &[&encode_bgl],
                    push_constant_ranges: &[],
                });

        let encode_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("rice_encode_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            encode_pipeline,
            encode_bgl,
            cached: None,
        }
    }

    fn ensure_buffers(&mut self, ctx: &GpuContext, num_tiles: usize) {
        let needs_realloc = self
            .cached
            .as_ref()
            .map_or(true, |c| c.num_tiles != num_tiles);
        if needs_realloc {
            self.cached = Some(CachedRiceEncodeBuffers::new(ctx, num_tiles));
        }
    }

    /// Encode 3 planes of quantized coefficients using GPU Rice.
    /// Returns a Vec<RiceTile> with tiles for all 3 planes concatenated.
    pub fn encode_3planes_to_tiles(
        &mut self,
        ctx: &GpuContext,
        quantized_bufs: [&wgpu::Buffer; 3],
        info: &FrameInfo,
        num_levels: u32,
    ) -> Vec<RiceTile> {
        let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;
        let total_streams = num_tiles * RICE_STREAMS_PER_TILE;

        self.ensure_buffers(ctx, num_tiles);
        let bufs = self.cached.as_ref().unwrap();

        let stream_size = (total_streams * MAX_STREAM_BYTES) as u64;
        let lengths_size = (total_streams * 4) as u64;
        let k_size = (num_tiles * K_STRIDE * 4) as u64;

        let params = RiceParams {
            num_tiles: num_tiles as u32,
            coefficients_per_tile: info.tile_size * info.tile_size,
            plane_width: info.padded_width(),
            tile_size: info.tile_size,
            tiles_x: info.tiles_x(),
            num_levels,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rice_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rice_3plane_cmd"),
            });

        for (p, quantized_buf) in quantized_bufs.iter().enumerate() {
            // Clear stream buffer (encode uses OR-packing within words)
            cmd.clear_buffer(&bufs.stream_buf, 0, None);

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rice_encode_bg"),
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
                        resource: bufs.stream_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bufs.lengths_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: bufs.k_buf.as_entire_binding(),
                    },
                ],
            });

            {
                let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rice_encode_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.encode_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(num_tiles as u32, 1, 1);
            }

            // Copy to staging
            cmd.copy_buffer_to_buffer(
                &bufs.stream_buf,
                0,
                &bufs.stream_staging[p],
                0,
                stream_size,
            );
            cmd.copy_buffer_to_buffer(
                &bufs.lengths_buf,
                0,
                &bufs.lengths_staging[p],
                0,
                lengths_size,
            );
            cmd.copy_buffer_to_buffer(&bufs.k_buf, 0, &bufs.k_staging[p], 0, k_size);
        }

        // Submit and wait
        ctx.queue.submit(Some(cmd.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        for p in 0..3 {
            for staging in [
                &bufs.stream_staging[p],
                &bufs.lengths_staging[p],
                &bufs.k_staging[p],
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
        for _ in 0..9 {
            rx.recv().unwrap().unwrap();
        }

        // Read back and pack into RiceTile structs
        let mut all_tiles = Vec::new();
        let num_groups = (num_levels * 2) as usize;

        for p in 0..3 {
            let stream_data: Vec<u8> = {
                let view = bufs.stream_staging[p].slice(..).get_mapped_range();
                view.to_vec()
            };
            let lengths_data: Vec<u32> = {
                let view = bufs.lengths_staging[p].slice(..).get_mapped_range();
                bytemuck::cast_slice(&view).to_vec()
            };
            let k_data: Vec<u32> = {
                let view = bufs.k_staging[p].slice(..).get_mapped_range();
                bytemuck::cast_slice(&view).to_vec()
            };

            bufs.stream_staging[p].unmap();
            bufs.lengths_staging[p].unmap();
            bufs.k_staging[p].unmap();

            for tile_idx in 0..num_tiles {
                let k_values: Vec<u8> = (0..num_groups)
                    .map(|g| k_data[tile_idx * K_STRIDE + g] as u8)
                    .collect();
                let k_zrl_values: Vec<u8> = (0..num_groups)
                    .map(|g| k_data[tile_idx * K_STRIDE + MAX_GROUPS + g] as u8)
                    .collect();

                let stream_lengths: Vec<u32> = (0..RICE_STREAMS_PER_TILE)
                    .map(|s| lengths_data[tile_idx * RICE_STREAMS_PER_TILE + s])
                    .collect();

                // Pack variable-length streams from fixed-size GPU slots
                let mut packed_data = Vec::new();
                for s in 0..RICE_STREAMS_PER_TILE {
                    let slot_byte_offset =
                        (tile_idx * RICE_STREAMS_PER_TILE + s) * MAX_STREAM_BYTES;
                    let len = stream_lengths[s] as usize;
                    packed_data
                        .extend_from_slice(&stream_data[slot_byte_offset..slot_byte_offset + len]);
                }

                all_tiles.push(RiceTile {
                    num_coefficients: info.tile_size * info.tile_size,
                    tile_size: info.tile_size,
                    num_levels,
                    num_groups: num_groups as u32,
                    k_values,
                    k_zrl_values,
                    stream_lengths,
                    stream_data: packed_data,
                });
            }
        }

        all_tiles
    }

    /// Dispatch Rice encode for 3 planes into an external command encoder.
    /// Call `finish_3planes_readback` after submit to read back the results.
    /// This is the split-phase API for batched P/B frame pipeline.
    pub fn dispatch_3planes_to_cmd(
        &mut self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        quantized_bufs: [&wgpu::Buffer; 3],
        info: &FrameInfo,
        num_levels: u32,
    ) {
        let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;
        let total_streams = num_tiles * RICE_STREAMS_PER_TILE;

        self.ensure_buffers(ctx, num_tiles);
        let bufs = self.cached.as_ref().unwrap();

        let stream_size = (total_streams * MAX_STREAM_BYTES) as u64;
        let lengths_size = (total_streams * 4) as u64;
        let k_size = (num_tiles * K_STRIDE * 4) as u64;

        let params = RiceParams {
            num_tiles: num_tiles as u32,
            coefficients_per_tile: info.tile_size * info.tile_size,
            plane_width: info.padded_width(),
            tile_size: info.tile_size,
            tiles_x: info.tiles_x(),
            num_levels,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rice_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        for (p, quantized_buf) in quantized_bufs.iter().enumerate() {
            // Clear stream buffer (encode uses OR-packing within words)
            cmd.clear_buffer(&bufs.stream_buf, 0, None);

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rice_encode_bg"),
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
                        resource: bufs.stream_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bufs.lengths_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: bufs.k_buf.as_entire_binding(),
                    },
                ],
            });

            {
                let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rice_encode_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.encode_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(num_tiles as u32, 1, 1);
            }

            // Copy to per-plane staging
            cmd.copy_buffer_to_buffer(
                &bufs.stream_buf,
                0,
                &bufs.stream_staging[p],
                0,
                stream_size,
            );
            cmd.copy_buffer_to_buffer(
                &bufs.lengths_buf,
                0,
                &bufs.lengths_staging[p],
                0,
                lengths_size,
            );
            cmd.copy_buffer_to_buffer(&bufs.k_buf, 0, &bufs.k_staging[p], 0, k_size);
        }
    }

    /// Read back Rice-encoded data after submit. Must be called after
    /// `dispatch_3planes_to_cmd` + `queue.submit`.
    pub fn finish_3planes_readback(
        &self,
        ctx: &GpuContext,
        info: &FrameInfo,
        num_levels: u32,
    ) -> Vec<RiceTile> {
        let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;
        let bufs = self.cached.as_ref().unwrap();
        let num_groups = (num_levels * 2) as usize;

        let (tx, rx) = std::sync::mpsc::channel();
        for p in 0..3 {
            for staging in [
                &bufs.stream_staging[p],
                &bufs.lengths_staging[p],
                &bufs.k_staging[p],
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
        for _ in 0..9 {
            rx.recv().unwrap().unwrap();
        }

        let mut all_tiles = Vec::new();

        for p in 0..3 {
            let stream_data: Vec<u8> = {
                let view = bufs.stream_staging[p].slice(..).get_mapped_range();
                view.to_vec()
            };
            let lengths_data: Vec<u32> = {
                let view = bufs.lengths_staging[p].slice(..).get_mapped_range();
                bytemuck::cast_slice(&view).to_vec()
            };
            let k_data: Vec<u32> = {
                let view = bufs.k_staging[p].slice(..).get_mapped_range();
                bytemuck::cast_slice(&view).to_vec()
            };

            bufs.stream_staging[p].unmap();
            bufs.lengths_staging[p].unmap();
            bufs.k_staging[p].unmap();

            for tile_idx in 0..num_tiles {
                let k_values: Vec<u8> = (0..num_groups)
                    .map(|g| k_data[tile_idx * K_STRIDE + g] as u8)
                    .collect();
                let k_zrl_values: Vec<u8> = (0..num_groups)
                    .map(|g| k_data[tile_idx * K_STRIDE + MAX_GROUPS + g] as u8)
                    .collect();

                let stream_lengths: Vec<u32> = (0..RICE_STREAMS_PER_TILE)
                    .map(|s| lengths_data[tile_idx * RICE_STREAMS_PER_TILE + s])
                    .collect();

                let mut packed_data = Vec::new();
                for s in 0..RICE_STREAMS_PER_TILE {
                    let slot_byte_offset =
                        (tile_idx * RICE_STREAMS_PER_TILE + s) * MAX_STREAM_BYTES;
                    let len = stream_lengths[s] as usize;
                    packed_data
                        .extend_from_slice(&stream_data[slot_byte_offset..slot_byte_offset + len]);
                }

                all_tiles.push(RiceTile {
                    num_coefficients: info.tile_size * info.tile_size,
                    tile_size: info.tile_size,
                    num_levels,
                    num_groups: num_groups as u32,
                    k_values,
                    k_zrl_values,
                    stream_lengths,
                    stream_data: packed_data,
                });
            }
        }

        all_tiles
    }
}

/// GPU Rice decoder — decodes Rice-coded tiles to f32 coefficient planes.
pub struct GpuRiceDecoder {
    decode_pipeline: wgpu::ComputePipeline,
    decode_bgl: wgpu::BindGroupLayout,
}

impl GpuRiceDecoder {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("rice_decode_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/rice_decode.wgsl").into(),
                ),
            });

        let decode_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("rice_decode_bgl"),
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
                        // binding 1: k_values (read-only)
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
                        // binding 2: stream_data (read-only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 3: stream_offsets (read-only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 4: output coefficients (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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

        let pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("rice_decode_layout"),
                    bind_group_layouts: &[&decode_bgl],
                    push_constant_ranges: &[],
                });

        let decode_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("rice_decode_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            decode_pipeline,
            decode_bgl,
        }
    }

    /// Pack RiceTile data for GPU decode. Returns (k_values, stream_data, stream_offsets) buffers.
    pub fn pack_decode_data(
        tiles: &[RiceTile],
        info: &FrameInfo,
    ) -> RiceDecodeData {
        let num_tiles = tiles.len();
        let total_streams = num_tiles * RICE_STREAMS_PER_TILE;

        // k values: flat array indexed by tile_id * K_STRIDE + group
        // k_zrl values stored at tile_id * K_STRIDE + MAX_GROUPS + group
        let mut k_values = vec![0u32; num_tiles * K_STRIDE];
        for (t, tile) in tiles.iter().enumerate() {
            for (g, &k) in tile.k_values.iter().enumerate() {
                k_values[t * K_STRIDE + g] = k as u32;
            }
            for (g, &k) in tile.k_zrl_values.iter().enumerate() {
                k_values[t * K_STRIDE + MAX_GROUPS + g] = k as u32;
            }
        }

        // Compute total stream data size and stream offsets
        let mut stream_offsets = vec![0u32; total_streams];
        let mut total_bytes = 0u32;
        for (t, tile) in tiles.iter().enumerate() {
            for s in 0..RICE_STREAMS_PER_TILE {
                stream_offsets[t * RICE_STREAMS_PER_TILE + s] = total_bytes;
                total_bytes += tile.stream_lengths[s];
            }
        }

        // Pack stream data (pad to u32 alignment)
        let padded_bytes = ((total_bytes as usize + 3) / 4) * 4;
        let mut stream_data = vec![0u8; padded_bytes];
        let mut write_pos = 0usize;
        for tile in tiles {
            stream_data[write_pos..write_pos + tile.stream_data.len()]
                .copy_from_slice(&tile.stream_data);
            write_pos += tile.stream_data.len();
        }

        RiceDecodeData {
            params: RiceParams {
                num_tiles: num_tiles as u32,
                coefficients_per_tile: info.tile_size * info.tile_size,
                plane_width: info.padded_width(),
                tile_size: info.tile_size,
                tiles_x: info.tiles_x(),
                num_levels: tiles.first().map_or(3, |t| t.num_levels),
                _pad0: 0,
                _pad1: 0,
            },
            k_values,
            stream_data: bytemuck::cast_slice(&stream_data).to_vec(),
            stream_offsets,
        }
    }

    /// Dispatch GPU Rice decode for one plane.
    pub fn dispatch_decode(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        params_buf: &wgpu::Buffer,
        k_buf: &wgpu::Buffer,
        stream_buf: &wgpu::Buffer,
        offsets_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_tiles: u32,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rice_decode_bg"),
            layout: &self.decode_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: stream_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rice_decode_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.decode_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(num_tiles, 1, 1);
    }
}

/// Packed data ready for GPU Rice decode upload.
pub struct RiceDecodeData {
    pub params: RiceParams,
    pub k_values: Vec<u32>,
    pub stream_data: Vec<u32>,
    pub stream_offsets: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::rice;

    /// Test GPU Rice decode against CPU decode for a known tile.
    #[test]
    fn gpu_decode_matches_cpu_decode() {
        let ctx = crate::GpuContext::new();
        let decoder = GpuRiceDecoder::new(&ctx);

        let tile_size = 64u32;
        let num_levels = 3u32;
        let coeffs_per_tile = (tile_size * tile_size) as usize;

        // Create a test pattern with a mix of zeros and non-zeros
        let mut coefficients = vec![0i32; coeffs_per_tile];
        for i in 0..coeffs_per_tile {
            let v = (i % 13) as i32 - 6; // values from -6 to 6
            coefficients[i] = v;
        }

        // CPU encode
        let tile = rice::rice_encode_tile(&coefficients, tile_size, num_levels);

        // CPU decode (known to be correct)
        let cpu_decoded = rice::rice_decode_tile(&tile);

        // Verify CPU roundtrip first
        assert_eq!(coefficients, cpu_decoded, "CPU roundtrip failed");

        // GPU decode: pack tile data for GPU
        let info = crate::FrameInfo {
            width: tile_size,
            height: tile_size,
            bit_depth: 8,
            tile_size,
        };
        let packed = GpuRiceDecoder::pack_decode_data(&[tile], &info);

        // Create GPU buffers
        let storage_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_params"),
            contents: bytemuck::bytes_of(&packed.params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let k_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_k"),
            contents: bytemuck::cast_slice(&packed.k_values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let stream_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_stream"),
            contents: bytemuck::cast_slice(&packed.stream_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let offsets_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_offsets"),
            contents: bytemuck::cast_slice(&packed.stream_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_size = (coeffs_per_tile * 4) as u64;
        let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Dispatch GPU decode
        let mut cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_decode"),
        });
        decoder.dispatch_decode(
            &ctx, &mut cmd, &params_buf, &k_buf, &stream_buf, &offsets_buf, &output_buf, 1,
        );

        // Copy output to staging
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        cmd.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, output_size);
        ctx.queue.submit(Some(cmd.finish()));

        // Read back
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let gpu_decoded: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        // Compare GPU decode with CPU decode
        let mut mismatches = 0;
        for i in 0..coeffs_per_tile {
            let cpu_val = cpu_decoded[i];
            let gpu_val = gpu_decoded[i] as i32;
            if cpu_val != gpu_val {
                if mismatches < 20 {
                    let row = i / tile_size as usize;
                    let col = i % tile_size as usize;
                    let stream_id = i % RICE_STREAMS_PER_TILE;
                    let s_pos = i / RICE_STREAMS_PER_TILE;
                    eprintln!(
                        "MISMATCH coeff[{i}] (row={row} col={col} stream={stream_id} s={s_pos}): CPU={cpu_val} GPU={gpu_val}"
                    );
                }
                mismatches += 1;
            }
        }
        eprintln!("Total mismatches: {mismatches} / {coeffs_per_tile}");
        assert_eq!(mismatches, 0, "GPU decode doesn't match CPU decode");
    }

    /// Test full encode→decode pipeline with Rice entropy at various resolutions.
    #[test]
    fn full_pipeline_rice_roundtrip() {
        let ctx = crate::GpuContext::new();
        let mut encoder = crate::encoder::pipeline::EncoderPipeline::new(&ctx);
        let decoder = crate::decoder::pipeline::DecoderPipeline::new(&ctx);

        // Test with synthetic gradient
        for &(w, h) in &[(256u32, 256u32), (1920, 1088)] {
            let mut rgb = Vec::with_capacity((w * h * 3) as usize);
            for y in 0..h {
                for x in 0..w {
                    rgb.push((x as f32 / w as f32 * 255.0).round());
                    rgb.push((y as f32 / h as f32 * 255.0).round());
                    rgb.push(((x + y) as f32 / (w + h) as f32 * 255.0).round());
                }
            }

            let mut config = crate::quality_preset(50);
            config.entropy_coder = crate::EntropyCoder::Rice;

            let compressed = encoder.encode(&ctx, &rgb, w, h, &config);
            let decoded = decoder.decode(&ctx, &compressed);

            let psnr = crate::bench::quality::psnr(&rgb, &decoded, 255.0);
            eprintln!("Rice pipeline {w}x{h} synthetic: PSNR={psnr:.2} dB");
            assert!(psnr > 30.0, "PSNR too low at {w}x{h}: {psnr:.2}");
        }

        // Test with real image if available
        let img_path = "test_material/frames/bbb_1080p.png";
        if std::path::Path::new(img_path).exists() {
            let (rgb, w, h) = crate::image_util::load_image_rgb_f32(img_path);
            let q = 25;
            let mut config = crate::quality_preset(q);
            config.entropy_coder = crate::EntropyCoder::Rice;

            let compressed = encoder.encode(&ctx, &rgb, w, h, &config);

            // Extract tiles and compare GPU decode vs CPU decode
            if let crate::EntropyData::Rice(ref tiles) = compressed.entropy {
                let info = &compressed.info;
                let tiles_per_plane = (info.tiles_x() * info.tiles_y()) as usize;
                let plane_tiles = &tiles[0..tiles_per_plane];
                let padded_w = info.padded_width() as usize;
                let padded_h = info.padded_height() as usize;
                let padded_pixels = padded_w * padded_h;
                let tile_sz = info.tile_size as usize;
                let tiles_x_count = info.tiles_x() as usize;

                // CPU decode all plane 0 tiles
                let mut cpu_plane = vec![0f32; padded_pixels];
                for (t, tile) in plane_tiles.iter().enumerate() {
                    let cpu_decoded = rice::rice_decode_tile(tile);
                    let tx_pos = t % tiles_x_count;
                    let ty_pos = t / tiles_x_count;
                    let origin_x = tx_pos * tile_sz;
                    let origin_y = ty_pos * tile_sz;
                    for ci in 0..tile_sz * tile_sz {
                        let row = ci / tile_sz;
                        let col = ci % tile_sz;
                        let pi = (origin_y + row) * padded_w + origin_x + col;
                        cpu_plane[pi] = cpu_decoded[ci] as f32;
                    }
                }

                // GPU decode same tiles
                let packed = GpuRiceDecoder::pack_decode_data(plane_tiles, info);
                let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("test_params"),
                    contents: bytemuck::bytes_of(&packed.params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let k_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("test_k"),
                    contents: bytemuck::cast_slice(&packed.k_values),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let stream_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("test_stream"),
                    contents: bytemuck::cast_slice(&packed.stream_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let offsets_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("test_offsets"),
                    contents: bytemuck::cast_slice(&packed.stream_offsets),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let output_size = (padded_pixels * 4) as u64;
                let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("test_output"),
                    size: output_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let dec = GpuRiceDecoder::new(&ctx);
                let mut cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("test_decode"),
                });
                // Clear output buffer to isolate uninitialized-memory issues
                cmd.clear_buffer(&output_buf, 0, None);
                dec.dispatch_decode(
                    &ctx, &mut cmd, &params_buf, &k_buf, &stream_buf, &offsets_buf, &output_buf,
                    tiles_per_plane as u32,
                );
                let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("test_staging"),
                    size: output_size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                cmd.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, output_size);
                ctx.queue.submit(Some(cmd.finish()));

                let slice = staging.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).unwrap();
                });
                ctx.device.poll(wgpu::Maintain::Wait);
                rx.recv().unwrap().unwrap();
                let data = slice.get_mapped_range();
                let gpu_plane: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                staging.unmap();

                // Compare GPU vs CPU decode
                let mut mismatches = 0;
                let mut first_mismatch_tile = None;
                for i in 0..padded_pixels {
                    if (cpu_plane[i] - gpu_plane[i]).abs() > 0.5 {
                        if mismatches < 10 {
                            let row = i / padded_w;
                            let col = i % padded_w;
                            let tile_x = col / tile_sz;
                            let tile_y = row / tile_sz;
                            let tile_id = tile_y * tiles_x_count + tile_x;
                            let local_row = row % tile_sz;
                            let local_col = col % tile_sz;
                            let coeff_idx = local_row * tile_sz + local_col;
                            eprintln!(
                                "MISMATCH [{i}] tile={tile_id} local=({local_row},{local_col}) ci={coeff_idx}: CPU={} GPU={}",
                                cpu_plane[i], gpu_plane[i]
                            );
                            if first_mismatch_tile.is_none() {
                                first_mismatch_tile = Some(tile_id);
                            }
                        }
                        mismatches += 1;
                    }
                }
                eprintln!("GPU vs CPU decode: {mismatches} mismatches / {padded_pixels} ({:.2}%)",
                    mismatches as f64 / padded_pixels as f64 * 100.0);
                if let Some(tid) = first_mismatch_tile {
                    eprintln!("First mismatch in tile {tid}: k_values={:?} k_zrl={:?}",
                        plane_tiles[tid].k_values, plane_tiles[tid].k_zrl_values);
                }
                assert_eq!(mismatches, 0, "GPU decode of real image differs from CPU decode");
            }
        }
    }

    /// Test GPU encode → CPU decode roundtrip.
    #[test]
    fn gpu_encode_cpu_decode_roundtrip() {
        let ctx = crate::GpuContext::new();
        let mut encoder = crate::encoder::pipeline::EncoderPipeline::new(&ctx);

        let (w, h) = (256, 256);
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                rgb.push((x as f32 / w as f32 * 255.0).round());
                rgb.push((y as f32 / h as f32 * 255.0).round());
                rgb.push(((x + y) as f32 / (w + h) as f32 * 255.0).round());
            }
        }

        let mut config = crate::quality_preset(50);
        config.entropy_coder = crate::EntropyCoder::Rice;

        let compressed = encoder.encode(&ctx, &rgb, w, h, &config);

        // Extract RiceTile data
        if let crate::EntropyData::Rice(ref tiles) = compressed.entropy {
            let info = &compressed.info;
            let tiles_per_plane = (info.tiles_x() * info.tiles_y()) as usize;

            // CPU decode each tile of plane 0 and pack into plane
            let plane_tiles = &tiles[0..tiles_per_plane];
            let padded_w = info.padded_width() as usize;
            let tile_sz = info.tile_size as usize;
            let tiles_x = info.tiles_x() as usize;

            // GPU decode the same data
            let packed = GpuRiceDecoder::pack_decode_data(plane_tiles, info);

            let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test_params"),
                contents: bytemuck::bytes_of(&packed.params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let k_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test_k"),
                contents: bytemuck::cast_slice(&packed.k_values),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let stream_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test_stream"),
                contents: bytemuck::cast_slice(&packed.stream_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let offsets_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test_offsets"),
                contents: bytemuck::cast_slice(&packed.stream_offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let padded_pixels = (info.padded_width() * info.padded_height()) as usize;
            let output_size = (padded_pixels * 4) as u64;
            let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test_output"),
                size: output_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let dec = GpuRiceDecoder::new(&ctx);
            let mut cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test_decode"),
            });
            dec.dispatch_decode(
                &ctx, &mut cmd, &params_buf, &k_buf, &stream_buf, &offsets_buf, &output_buf,
                tiles_per_plane as u32,
            );
            let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test_staging"),
                size: output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            cmd.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, output_size);
            ctx.queue.submit(Some(cmd.finish()));

            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            ctx.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let gpu_plane: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging.unmap();

            // CPU decode and place in plane for comparison
            let mut cpu_plane = vec![0f32; padded_pixels];
            for (t, tile) in plane_tiles.iter().enumerate() {
                let cpu_decoded = rice::rice_decode_tile(tile);
                let tx_pos = t % tiles_x;
                let ty_pos = t / tiles_x;
                let origin_x = tx_pos * tile_sz;
                let origin_y = ty_pos * tile_sz;
                for ci in 0..tile_sz * tile_sz {
                    let row = ci / tile_sz;
                    let col = ci % tile_sz;
                    let pi = (origin_y + row) * padded_w + origin_x + col;
                    cpu_plane[pi] = cpu_decoded[ci] as f32;
                }
            }

            // Compare
            let mut mismatches = 0;
            for i in 0..padded_pixels {
                if (cpu_plane[i] - gpu_plane[i]).abs() > 0.5 {
                    if mismatches < 20 {
                        let row = i / padded_w;
                        let col = i % padded_w;
                        eprintln!(
                            "MISMATCH [{i}] (row={row} col={col}): CPU={} GPU={}",
                            cpu_plane[i], gpu_plane[i]
                        );
                    }
                    mismatches += 1;
                }
            }
            eprintln!("GPU encode→GPU decode vs GPU encode→CPU decode: {mismatches} mismatches / {padded_pixels}");
            assert_eq!(mismatches, 0, "GPU decode of GPU-encoded data differs from CPU decode");
        } else {
            panic!("Expected Rice entropy data");
        }
    }

    /// Test GPU Rice decode with multiple tiles (2x2 grid).
    #[test]
    fn gpu_decode_multi_tile() {
        let ctx = crate::GpuContext::new();
        let decoder = GpuRiceDecoder::new(&ctx);

        let tile_size = 64u32;
        let num_levels = 3u32;
        let coeffs_per_tile = (tile_size * tile_size) as usize;
        let tiles_x = 2u32;
        let tiles_y = 2u32;
        let num_tiles = (tiles_x * tiles_y) as usize;
        let padded_w = tiles_x * tile_size;
        let padded_h = tiles_y * tile_size;
        let total_pixels = (padded_w * padded_h) as usize;

        // Create different patterns for each tile
        let mut all_tiles = Vec::new();
        for t in 0..num_tiles {
            let mut coefficients = vec![0i32; coeffs_per_tile];
            for i in 0..coeffs_per_tile {
                let v = ((i + t * 7) % 13) as i32 - 6;
                coefficients[i] = v;
            }
            let tile = rice::rice_encode_tile(&coefficients, tile_size, num_levels);
            all_tiles.push(tile);
        }

        // CPU decode all tiles and place in plane
        let mut cpu_plane = vec![0i32; total_pixels];
        for (t, tile) in all_tiles.iter().enumerate() {
            let decoded = rice::rice_decode_tile(tile);
            let tx = (t as u32) % tiles_x;
            let ty = (t as u32) / tiles_x;
            let origin_x = tx * tile_size;
            let origin_y = ty * tile_size;
            for ci in 0..coeffs_per_tile {
                let row = ci / tile_size as usize;
                let col = ci % tile_size as usize;
                let pi = (origin_y as usize + row) * padded_w as usize + origin_x as usize + col;
                cpu_plane[pi] = decoded[ci];
            }
        }

        // GPU decode
        let info = crate::FrameInfo {
            width: padded_w,
            height: padded_h,
            bit_depth: 8,
            tile_size,
        };
        let packed = GpuRiceDecoder::pack_decode_data(&all_tiles, &info);

        let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_params"),
            contents: bytemuck::bytes_of(&packed.params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let k_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_k"),
            contents: bytemuck::cast_slice(&packed.k_values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let stream_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_stream"),
            contents: bytemuck::cast_slice(&packed.stream_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let offsets_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_offsets"),
            contents: bytemuck::cast_slice(&packed.stream_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_size = (total_pixels * 4) as u64;
        let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut cmd = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_decode"),
        });
        decoder.dispatch_decode(
            &ctx, &mut cmd, &params_buf, &k_buf, &stream_buf, &offsets_buf, &output_buf,
            num_tiles as u32,
        );

        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        cmd.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, output_size);
        ctx.queue.submit(Some(cmd.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let gpu_plane: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        // Compare
        let mut mismatches = 0;
        for i in 0..total_pixels {
            let cpu_val = cpu_plane[i];
            let gpu_val = gpu_plane[i] as i32;
            if cpu_val != gpu_val {
                if mismatches < 20 {
                    let row = i / padded_w as usize;
                    let col = i % padded_w as usize;
                    let tile_x = col / tile_size as usize;
                    let tile_y = row / tile_size as usize;
                    let tile_id = tile_y * tiles_x as usize + tile_x;
                    eprintln!(
                        "MISMATCH pixel[{i}] (row={row} col={col} tile={tile_id}): CPU={cpu_val} GPU={gpu_val}"
                    );
                }
                mismatches += 1;
            }
        }
        eprintln!("Total mismatches: {mismatches} / {total_pixels}");
        assert_eq!(mismatches, 0, "GPU multi-tile decode doesn't match CPU decode");
    }
}
