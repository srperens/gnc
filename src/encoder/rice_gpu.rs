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
        let k_size = (num_tiles * MAX_GROUPS * 4) as u64;

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
        let k_size = (num_tiles * MAX_GROUPS * 4) as u64;

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
                    .map(|g| k_data[tile_idx * MAX_GROUPS + g] as u8)
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

        // k values: flat array indexed by tile_id * MAX_GROUPS + group
        let mut k_values = vec![0u32; num_tiles * MAX_GROUPS];
        for (t, tile) in tiles.iter().enumerate() {
            for (g, &k) in tile.k_values.iter().enumerate() {
                k_values[t * MAX_GROUPS + g] = k as u32;
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
