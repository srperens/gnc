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

use super::rice::{max_stream_bytes_for_tile, RiceTile, RICE_STREAMS_PER_TILE};
use crate::{FrameInfo, GpuContext};

const MAX_GROUPS: usize = 8;
const K_STRIDE: usize = MAX_GROUPS * 2 + 1; // stride per tile in k_output (mag k + zrl k per group + skip bitmap)

// Field order must stay in sync with shaders/rice_encode.wgsl Params struct (bytemuck::Pod).
// Fields in order: num_tiles, coefficients_per_tile, plane_width, tile_size, tiles_x,
//                  num_levels, max_stream_bytes, _pad0.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RiceParams {
    num_tiles: u32,
    coefficients_per_tile: u32,
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    num_levels: u32,
    /// Per-stream byte ceiling passed to the shader; set via max_stream_bytes_for_tile().
    max_stream_bytes: u32,
    _pad0: u32,
}

/// Pre-allocated GPU buffers for Rice encode, reused across calls.
struct CachedRiceEncodeBuffers {
    num_tiles: usize,
    /// Per-stream byte ceiling used when this cache was allocated.
    max_stream_bytes: usize,
    stream_buf: wgpu::Buffer,
    lengths_buf: wgpu::Buffer,
    k_buf: wgpu::Buffer,
    /// One u32 per tile; shader sets to 1 on stream overflow.
    overflow_buf: wgpu::Buffer,
    overflow_staging: wgpu::Buffer,
    /// Cached uniform params buffer — updated via write_buffer, avoids per-frame create_buffer_init.
    params_buf: wgpu::Buffer,
    // Per-plane staging for single-frame 3-plane encode (batch_size=1)
    stream_staging: [wgpu::Buffer; 3],
    lengths_staging: [wgpu::Buffer; 3],
    k_staging: [wgpu::Buffer; 3],
    // Per-frame-per-plane staging for multi-frame batched encode.
    // Indexed as [frame_slot][plane]. Empty when batch_size==1.
    stream_staging_batch: Vec<[wgpu::Buffer; 3]>,
    lengths_staging_batch: Vec<[wgpu::Buffer; 3]>,
    k_staging_batch: Vec<[wgpu::Buffer; 3]>,
}

impl CachedRiceEncodeBuffers {
    fn new(ctx: &GpuContext, num_tiles: usize, max_stream_bytes: usize) -> Self {
        let total_streams = num_tiles * RICE_STREAMS_PER_TILE;
        let stream_size = (total_streams * max_stream_bytes) as u64;
        let lengths_size = (total_streams * 4) as u64;
        let k_size = (num_tiles * K_STRIDE * 4) as u64;
        let overflow_size = (num_tiles * 4) as u64;

        let sc = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let mr = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        Self {
            num_tiles,
            max_stream_bytes,
            stream_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rice_stream"),
                size: stream_size.max(4),
                usage: sc,
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
            overflow_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rice_overflow"),
                size: overflow_size.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            overflow_staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rice_overflow_stg"),
                size: overflow_size.max(4),
                usage: mr,
                mapped_at_creation: false,
            }),
            params_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rice_params_cached"),
                size: std::mem::size_of::<RiceParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
            stream_staging_batch: Vec::new(),
            lengths_staging_batch: Vec::new(),
            k_staging_batch: Vec::new(),
        }
    }

    /// Ensure at least `batch_size` per-frame staging slots are allocated.
    /// Called before batching multiple frames into one command encoder.
    fn ensure_batch_staging(&mut self, ctx: &GpuContext, batch_size: usize) {
        let total_streams = self.num_tiles * RICE_STREAMS_PER_TILE;
        let stream_size = (total_streams * self.max_stream_bytes) as u64;
        let lengths_size = (total_streams * 4) as u64;
        let k_size = (self.num_tiles * K_STRIDE * 4) as u64;
        let mr = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;

        while self.stream_staging_batch.len() < batch_size {
            let slot = self.stream_staging_batch.len();
            self.stream_staging_batch.push(std::array::from_fn(|p| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("rice_stream_stgb_{}_{}", slot, p)),
                    size: stream_size.max(4),
                    usage: mr,
                    mapped_at_creation: false,
                })
            }));
            self.lengths_staging_batch.push(std::array::from_fn(|p| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("rice_lengths_stgb_{}_{}", slot, p)),
                    size: lengths_size.max(4),
                    usage: mr,
                    mapped_at_creation: false,
                })
            }));
            self.k_staging_batch.push(std::array::from_fn(|p| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("rice_k_stgb_{}_{}", slot, p)),
                    size: k_size.max(4),
                    usage: mr,
                    mapped_at_creation: false,
                })
            }));
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
                        // binding 5: overflow_flags (read-write, atomic u32 per tile)
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
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

    fn ensure_buffers(&mut self, ctx: &GpuContext, num_tiles: usize, tile_size: u32) {
        let msb = max_stream_bytes_for_tile(tile_size);
        let needs_realloc = self.cached.as_ref().is_none_or(|c| {
            c.num_tiles != num_tiles || c.max_stream_bytes != msb
        });
        if needs_realloc {
            self.cached = Some(CachedRiceEncodeBuffers::new(ctx, num_tiles, msb));
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
        let msb = max_stream_bytes_for_tile(info.tile_size);

        self.ensure_buffers(ctx, num_tiles, info.tile_size);
        let bufs = self.cached.as_ref().unwrap();

        let stream_size = (total_streams * msb) as u64;
        let lengths_size = (total_streams * 4) as u64;
        let k_size = (num_tiles * K_STRIDE * 4) as u64;
        let overflow_size = (num_tiles * 4) as u64;

        let params = RiceParams {
            num_tiles: num_tiles as u32,
            coefficients_per_tile: info.tile_size * info.tile_size,
            plane_width: info.padded_width(),
            tile_size: info.tile_size,
            tiles_x: info.tiles_x(),
            num_levels,
            max_stream_bytes: msb as u32,
            _pad0: 0,
        };
        ctx.queue
            .write_buffer(&bufs.params_buf, 0, bytemuck::bytes_of(&params));

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rice_3plane_cmd"),
            });

        // Clear overflow flags before dispatch (reuse across planes)
        cmd.clear_buffer(&bufs.overflow_buf, 0, Some(overflow_size));

        for (p, quantized_buf) in quantized_bufs.iter().enumerate() {

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rice_encode_bg"),
                layout: &self.encode_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bufs.params_buf.as_entire_binding(),
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
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: bufs.overflow_buf.as_entire_binding(),
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
        // Copy overflow flags once (shared across planes)
        cmd.copy_buffer_to_buffer(
            &bufs.overflow_buf,
            0,
            &bufs.overflow_staging,
            0,
            overflow_size,
        );

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
        {
            let tx_clone = tx.clone();
            bufs.overflow_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx_clone.send(result).unwrap();
                });
        }
        drop(tx);
        ctx.device.poll(wgpu::Maintain::Wait);
        // 9 data buffers + 1 overflow
        for _ in 0..10 {
            rx.recv().unwrap().unwrap();
        }

        // Check overflow flags
        {
            let ov_view = bufs.overflow_staging.slice(..).get_mapped_range();
            let ov_data: &[u32] = bytemuck::cast_slice(&ov_view);
            for (tile_idx, &flag) in ov_data.iter().enumerate() {
                if flag != 0 {
                    drop(ov_view);
                    bufs.overflow_staging.unmap();
                    panic!(
                        "Rice stream overflow in tile {tile_idx}: max_stream_bytes ({msb}) too small. \
                         Increase max_stream_bytes_for_tile() for tile_size={}.",
                        info.tile_size
                    );
                }
            }
            drop(ov_view);
            bufs.overflow_staging.unmap();
        }

        // Read back and pack into RiceTile structs — read directly from mapped views
        // to avoid 192MB of unnecessary to_vec() allocation+memcpy
        let mut all_tiles = Vec::with_capacity(num_tiles * 3);
        let num_groups = (num_levels * 2).max(1) as usize;

        for p in 0..3 {
            let stream_view = bufs.stream_staging[p].slice(..).get_mapped_range();
            let lengths_view = bufs.lengths_staging[p].slice(..).get_mapped_range();
            let k_view = bufs.k_staging[p].slice(..).get_mapped_range();
            let lengths_data: &[u32] = bytemuck::cast_slice(&lengths_view);
            let k_data: &[u32] = bytemuck::cast_slice(&k_view);

            for tile_idx in 0..num_tiles {
                let k_values: Vec<u8> = (0..num_groups)
                    .map(|g| k_data[tile_idx * K_STRIDE + g] as u8)
                    .collect();
                let k_zrl_values: Vec<u8> = (0..num_groups)
                    .map(|g| k_data[tile_idx * K_STRIDE + MAX_GROUPS + g] as u8)
                    .collect();
                let skip_bitmap = k_data[tile_idx * K_STRIDE + K_STRIDE - 1] as u8;

                let stream_lengths: Vec<u32> = (0..RICE_STREAMS_PER_TILE)
                    .map(|s| lengths_data[tile_idx * RICE_STREAMS_PER_TILE + s])
                    .collect();

                // Pre-compute total packed size for allocation
                let total_packed: usize = stream_lengths.iter().map(|&l| l as usize).sum();
                let mut packed_data = Vec::with_capacity(total_packed);
                for (s, &len) in stream_lengths.iter().enumerate() {
                    let slot_byte_offset =
                        (tile_idx * RICE_STREAMS_PER_TILE + s) * msb;
                    let len = len as usize;
                    packed_data
                        .extend_from_slice(&stream_view[slot_byte_offset..slot_byte_offset + len]);
                }

                all_tiles.push(RiceTile {
                    num_coefficients: info.tile_size * info.tile_size,
                    tile_size: info.tile_size,
                    num_levels,
                    num_groups: num_groups as u32,
                    k_values,
                    k_zrl_values,
                    skip_bitmap,
                    stream_lengths,
                    stream_data: packed_data,
                });
            }

            drop(stream_view);
            drop(lengths_view);
            drop(k_view);
            bufs.stream_staging[p].unmap();
            bufs.lengths_staging[p].unmap();
            bufs.k_staging[p].unmap();
        }

        all_tiles
    }

    /// Encode a single plane of quantized coefficients using GPU Rice.
    /// Returns a Vec<RiceTile> with tiles for that one plane.
    /// Uses staging slot 0 — do not mix with encode_3planes_to_tiles calls.
    pub fn encode_1plane_to_tiles(
        &mut self,
        ctx: &GpuContext,
        quantized_buf: &wgpu::Buffer,
        info: &FrameInfo,
        num_levels: u32,
    ) -> Vec<RiceTile> {
        let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;
        let total_streams = num_tiles * RICE_STREAMS_PER_TILE;
        let msb = max_stream_bytes_for_tile(info.tile_size);

        self.ensure_buffers(ctx, num_tiles, info.tile_size);
        let bufs = self.cached.as_ref().unwrap();

        let stream_size = (total_streams * msb) as u64;
        let lengths_size = (total_streams * 4) as u64;
        let k_size = (num_tiles * K_STRIDE * 4) as u64;
        let overflow_size = (num_tiles * 4) as u64;

        let params = RiceParams {
            num_tiles: num_tiles as u32,
            coefficients_per_tile: info.tile_size * info.tile_size,
            plane_width: info.padded_width(),
            tile_size: info.tile_size,
            tiles_x: info.tiles_x(),
            num_levels,
            max_stream_bytes: msb as u32,
            _pad0: 0,
        };
        ctx.queue
            .write_buffer(&bufs.params_buf, 0, bytemuck::bytes_of(&params));

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rice_1plane_cmd"),
            });

        cmd.clear_buffer(&bufs.overflow_buf, 0, Some(overflow_size));

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rice_encode_bg"),
            layout: &self.encode_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs.params_buf.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bufs.overflow_buf.as_entire_binding(),
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

        // Copy to staging slot 0
        cmd.copy_buffer_to_buffer(&bufs.stream_buf, 0, &bufs.stream_staging[0], 0, stream_size);
        cmd.copy_buffer_to_buffer(&bufs.lengths_buf, 0, &bufs.lengths_staging[0], 0, lengths_size);
        cmd.copy_buffer_to_buffer(&bufs.k_buf, 0, &bufs.k_staging[0], 0, k_size);
        cmd.copy_buffer_to_buffer(
            &bufs.overflow_buf,
            0,
            &bufs.overflow_staging,
            0,
            overflow_size,
        );

        ctx.queue.submit(Some(cmd.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        for staging in [
            &bufs.stream_staging[0],
            &bufs.lengths_staging[0],
            &bufs.k_staging[0],
        ] {
            let tx_clone = tx.clone();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx_clone.send(result).unwrap();
                });
        }
        {
            let tx_clone = tx.clone();
            bufs.overflow_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx_clone.send(result).unwrap();
                });
        }
        drop(tx);
        ctx.device.poll(wgpu::Maintain::Wait);
        // 3 data buffers + 1 overflow
        for _ in 0..4 {
            rx.recv().unwrap().unwrap();
        }

        // Check overflow flags
        {
            let ov_view = bufs.overflow_staging.slice(..).get_mapped_range();
            let ov_data: &[u32] = bytemuck::cast_slice(&ov_view);
            for (tile_idx, &flag) in ov_data.iter().enumerate() {
                if flag != 0 {
                    drop(ov_view);
                    bufs.overflow_staging.unmap();
                    panic!(
                        "Rice stream overflow in tile {tile_idx}: max_stream_bytes ({msb}) too small. \
                         Increase max_stream_bytes_for_tile() for tile_size={}.",
                        info.tile_size
                    );
                }
            }
            drop(ov_view);
            bufs.overflow_staging.unmap();
        }

        let stream_view = bufs.stream_staging[0].slice(..).get_mapped_range();
        let lengths_view = bufs.lengths_staging[0].slice(..).get_mapped_range();
        let k_view = bufs.k_staging[0].slice(..).get_mapped_range();
        let lengths_data: &[u32] = bytemuck::cast_slice(&lengths_view);
        let k_data: &[u32] = bytemuck::cast_slice(&k_view);
        let num_groups = (num_levels * 2).max(1) as usize;

        let mut tiles = Vec::with_capacity(num_tiles);
        for tile_idx in 0..num_tiles {
            let k_values: Vec<u8> = (0..num_groups)
                .map(|g| k_data[tile_idx * K_STRIDE + g] as u8)
                .collect();
            let k_zrl_values: Vec<u8> = (0..num_groups)
                .map(|g| k_data[tile_idx * K_STRIDE + MAX_GROUPS + g] as u8)
                .collect();
            let skip_bitmap = k_data[tile_idx * K_STRIDE + K_STRIDE - 1] as u8;
            let stream_lengths: Vec<u32> = (0..RICE_STREAMS_PER_TILE)
                .map(|s| lengths_data[tile_idx * RICE_STREAMS_PER_TILE + s])
                .collect();
            let total_packed: usize = stream_lengths.iter().map(|&l| l as usize).sum();
            let mut packed_data = Vec::with_capacity(total_packed);
            for (s, &len) in stream_lengths.iter().enumerate() {
                let slot_byte_offset = (tile_idx * RICE_STREAMS_PER_TILE + s) * msb;
                let len = len as usize;
                packed_data
                    .extend_from_slice(&stream_view[slot_byte_offset..slot_byte_offset + len]);
            }
            tiles.push(RiceTile {
                num_coefficients: info.tile_size * info.tile_size,
                tile_size: info.tile_size,
                num_levels,
                num_groups: num_groups as u32,
                k_values,
                k_zrl_values,
                skip_bitmap,
                stream_lengths,
                stream_data: packed_data,
            });
        }

        drop(stream_view);
        drop(lengths_view);
        drop(k_view);
        bufs.stream_staging[0].unmap();
        bufs.lengths_staging[0].unmap();
        bufs.k_staging[0].unmap();

        tiles
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
        let msb = max_stream_bytes_for_tile(info.tile_size);

        self.ensure_buffers(ctx, num_tiles, info.tile_size);
        let bufs = self.cached.as_ref().unwrap();

        let stream_size = (total_streams * msb) as u64;
        let lengths_size = (total_streams * 4) as u64;
        let k_size = (num_tiles * K_STRIDE * 4) as u64;
        let overflow_size = (num_tiles * 4) as u64;

        let params = RiceParams {
            num_tiles: num_tiles as u32,
            coefficients_per_tile: info.tile_size * info.tile_size,
            plane_width: info.padded_width(),
            tile_size: info.tile_size,
            tiles_x: info.tiles_x(),
            num_levels,
            max_stream_bytes: msb as u32,
            _pad0: 0,
        };
        ctx.queue
            .write_buffer(&bufs.params_buf, 0, bytemuck::bytes_of(&params));

        cmd.clear_buffer(&bufs.overflow_buf, 0, Some(overflow_size));

        for (p, quantized_buf) in quantized_bufs.iter().enumerate() {

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rice_encode_bg"),
                layout: &self.encode_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bufs.params_buf.as_entire_binding(),
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
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: bufs.overflow_buf.as_entire_binding(),
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
        // Copy overflow flags (shared across planes)
        cmd.copy_buffer_to_buffer(
            &bufs.overflow_buf,
            0,
            &bufs.overflow_staging,
            0,
            overflow_size,
        );
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
        let msb = bufs.max_stream_bytes;
        let num_groups = (num_levels * 2).max(1) as usize;
        let profile = std::env::var("GNC_PROFILE").is_ok();
        let _t0 = std::time::Instant::now();

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
        {
            let tx_clone = tx.clone();
            bufs.overflow_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx_clone.send(result).unwrap();
                });
        }
        drop(tx);
        ctx.device.poll(wgpu::Maintain::Wait);
        // 9 data buffers + 1 overflow
        for _ in 0..10 {
            rx.recv().unwrap().unwrap();
        }

        // Check overflow flags before unpacking
        {
            let ov_view = bufs.overflow_staging.slice(..).get_mapped_range();
            let ov_data: &[u32] = bytemuck::cast_slice(&ov_view);
            for (tile_idx, &flag) in ov_data.iter().enumerate() {
                if flag != 0 {
                    drop(ov_view);
                    bufs.overflow_staging.unmap();
                    panic!(
                        "Rice stream overflow in tile {tile_idx}: max_stream_bytes ({msb}) too small. \
                         Increase max_stream_bytes_for_tile() for tile_size={}.",
                        info.tile_size
                    );
                }
            }
            drop(ov_view);
            bufs.overflow_staging.unmap();
        }

        if profile {
            eprintln!("    Rice map+poll: {:.1}ms", _t0.elapsed().as_secs_f64() * 1000.0);
        }
        let _t1 = std::time::Instant::now();

        let mut all_tiles = Vec::with_capacity(num_tiles * 3);

        for p in 0..3 {
            let stream_view = bufs.stream_staging[p].slice(..).get_mapped_range();
            let lengths_view = bufs.lengths_staging[p].slice(..).get_mapped_range();
            let k_view = bufs.k_staging[p].slice(..).get_mapped_range();
            let lengths_data: &[u32] = bytemuck::cast_slice(&lengths_view);
            let k_data: &[u32] = bytemuck::cast_slice(&k_view);

            for tile_idx in 0..num_tiles {
                let k_values: Vec<u8> = (0..num_groups)
                    .map(|g| k_data[tile_idx * K_STRIDE + g] as u8)
                    .collect();
                let k_zrl_values: Vec<u8> = (0..num_groups)
                    .map(|g| k_data[tile_idx * K_STRIDE + MAX_GROUPS + g] as u8)
                    .collect();
                let skip_bitmap = k_data[tile_idx * K_STRIDE + K_STRIDE - 1] as u8;

                let stream_lengths: Vec<u32> = (0..RICE_STREAMS_PER_TILE)
                    .map(|s| lengths_data[tile_idx * RICE_STREAMS_PER_TILE + s])
                    .collect();

                let total_packed: usize = stream_lengths.iter().map(|&l| l as usize).sum();
                let mut packed_data = Vec::with_capacity(total_packed);
                for (s, &len) in stream_lengths.iter().enumerate() {
                    let slot_byte_offset =
                        (tile_idx * RICE_STREAMS_PER_TILE + s) * msb;
                    let len = len as usize;
                    packed_data
                        .extend_from_slice(&stream_view[slot_byte_offset..slot_byte_offset + len]);
                }

                all_tiles.push(RiceTile {
                    num_coefficients: info.tile_size * info.tile_size,
                    tile_size: info.tile_size,
                    num_levels,
                    num_groups: num_groups as u32,
                    k_values,
                    k_zrl_values,
                    skip_bitmap,
                    stream_lengths,
                    stream_data: packed_data,
                });
            }

            drop(stream_view);
            drop(lengths_view);
            drop(k_view);
            bufs.stream_staging[p].unmap();
            bufs.lengths_staging[p].unmap();
            bufs.k_staging[p].unmap();
        }

        if profile {
            let actual_bytes: usize = all_tiles.iter().map(|t| t.stream_data.len()).sum();
            let staging_bytes = num_tiles * RICE_STREAMS_PER_TILE * msb * 3;
            eprintln!("    Rice pack: {:.1}ms (actual {:.1}MB / staging {:.1}MB = {:.1}% utilization)",
                _t1.elapsed().as_secs_f64() * 1000.0,
                actual_bytes as f64 / 1_048_576.0,
                staging_bytes as f64 / 1_048_576.0,
                actual_bytes as f64 / staging_bytes as f64 * 100.0);
        }

        all_tiles
    }

    /// Ensure batch staging buffers are allocated for `batch_size` frames.
    /// Call before the first `dispatch_3planes_to_cmd_batch` in a GOP.
    pub fn prepare_batch_staging(
        &mut self,
        ctx: &GpuContext,
        num_tiles: usize,
        tile_size: u32,
        batch_size: usize,
    ) {
        self.ensure_buffers(ctx, num_tiles, tile_size);
        let bufs = self.cached.as_mut().unwrap();
        bufs.ensure_batch_staging(ctx, batch_size);
    }

    /// Write `params_buf` once before a batch of `dispatch_3planes_to_cmd_batch` calls.
    ///
    /// On Metal/wgpu, `queue.write_buffer` is staged: only the last write before
    /// `queue.submit` takes effect.  All high frames in a GOP share the same FrameInfo
    /// (identical tile dimensions and num_levels), so one write covers the whole batch.
    /// Call this after `prepare_batch_staging` and before building the command encoder.
    ///
    /// # Panics
    /// Panics in debug builds if the derived `num_tiles` disagrees with the cached buffer size.
    pub fn write_params_buf_for_batch(
        &self,
        ctx: &GpuContext,
        info: &FrameInfo,
        num_levels: u32,
    ) {
        let bufs = self.cached.as_ref().expect("prepare_batch_staging must be called first");
        let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;
        debug_assert_eq!(
            num_tiles, bufs.num_tiles,
            "FrameInfo tile dimensions differ from cached buffer size — all batch frames must share the same FrameInfo"
        );
        let msb = max_stream_bytes_for_tile(info.tile_size);
        let params = RiceParams {
            num_tiles: num_tiles as u32,
            coefficients_per_tile: info.tile_size * info.tile_size,
            plane_width: info.padded_width(),
            tile_size: info.tile_size,
            tiles_x: info.tiles_x(),
            num_levels,
            max_stream_bytes: msb as u32,
            _pad0: 0,
        };
        ctx.queue.write_buffer(&bufs.params_buf, 0, bytemuck::bytes_of(&params));
    }

    /// Dispatch Rice encode for 3 planes into an external command encoder,
    /// copying results to per-frame staging slot `frame_slot`.
    ///
    /// Must call `prepare_batch_staging` first to ensure slots exist.
    /// Call `finish_batch_readback` after submitting the command encoder.
    pub fn dispatch_3planes_to_cmd_batch(
        &mut self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        quantized_bufs: [&wgpu::Buffer; 3],
        info: &FrameInfo,
        frame_slot: usize,
    ) {
        let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;
        let total_streams = num_tiles * RICE_STREAMS_PER_TILE;
        let msb = self.cached.as_ref().unwrap().max_stream_bytes;
        let stream_size = (total_streams * msb) as u64;
        let lengths_size = (total_streams * 4) as u64;
        let k_size = (num_tiles * K_STRIDE * 4) as u64;
        let overflow_size = (num_tiles * 4) as u64;

        // ensure_buffers and write_params_buf_for_batch already called by the batch driver
        // (prepare_batch_staging + write_params_buf_for_batch, once before the loop).
        // DO NOT write params_buf here: on Metal/wgpu write_buffer is staged and only the
        // last write before queue.submit takes effect, so writing N times in a loop would
        // silently discard all but the last.  All high frames in a GOP share the same
        // FrameInfo (identical tile layout), so one pre-write is correct for the whole batch.
        let bufs = self.cached.as_ref().unwrap();

        // Clear overflow flags for this frame slot (reused per frame)
        cmd.clear_buffer(&bufs.overflow_buf, 0, Some(overflow_size));

        for (p, quantized_buf) in quantized_bufs.iter().enumerate() {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rice_encode_bg_batch"),
                layout: &self.encode_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bufs.params_buf.as_entire_binding(),
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
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: bufs.overflow_buf.as_entire_binding(),
                    },
                ],
            });

            {
                let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rice_encode_pass_batch"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.encode_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(num_tiles as u32, 1, 1);
            }

            // Copy to per-frame-slot staging buffers
            cmd.copy_buffer_to_buffer(
                &bufs.stream_buf,
                0,
                &bufs.stream_staging_batch[frame_slot][p],
                0,
                stream_size,
            );
            cmd.copy_buffer_to_buffer(
                &bufs.lengths_buf,
                0,
                &bufs.lengths_staging_batch[frame_slot][p],
                0,
                lengths_size,
            );
            cmd.copy_buffer_to_buffer(
                &bufs.k_buf,
                0,
                &bufs.k_staging_batch[frame_slot][p],
                0,
                k_size,
            );
        }
    }

    /// Read back Rice-encoded data for all `batch_size` frames after a single submit+poll.
    ///
    /// Returns a `Vec` of length `batch_size`, each element being a `Vec<RiceTile>` (3 planes).
    pub fn finish_batch_readback(
        &self,
        ctx: &GpuContext,
        info: &FrameInfo,
        num_levels: u32,
        batch_size: usize,
    ) -> Vec<Vec<RiceTile>> {
        let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;
        let bufs = self.cached.as_ref().unwrap();
        let msb = bufs.max_stream_bytes;
        let num_groups = (num_levels * 2).max(1) as usize;

        // Map all staging buffers for all frames at once, then single poll.
        // Also map overflow_staging to catch any overflow from the last dispatched frame.
        let (tx, rx) = std::sync::mpsc::channel();
        for slot in 0..batch_size {
            for p in 0..3 {
                for staging in [
                    &bufs.stream_staging_batch[slot][p],
                    &bufs.lengths_staging_batch[slot][p],
                    &bufs.k_staging_batch[slot][p],
                ] {
                    let tx_clone = tx.clone();
                    staging
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |result| {
                            tx_clone.send(result).unwrap();
                        });
                }
            }
        }
        {
            let tx_clone = tx.clone();
            bufs.overflow_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx_clone.send(result).unwrap();
                });
        }
        drop(tx);
        ctx.device.poll(wgpu::Maintain::Wait);
        // 9 data callbacks per frame + 1 overflow
        for _ in 0..batch_size * 9 + 1 {
            rx.recv().unwrap().unwrap();
        }

        // Check overflow flags (reflects the last frame in the batch; overflow is rare)
        {
            let ov_view = bufs.overflow_staging.slice(..).get_mapped_range();
            let ov_data: &[u32] = bytemuck::cast_slice(&ov_view);
            for (tile_idx, &flag) in ov_data.iter().enumerate() {
                if flag != 0 {
                    drop(ov_view);
                    bufs.overflow_staging.unmap();
                    panic!(
                        "Rice stream overflow in tile {tile_idx} (batch): max_stream_bytes ({msb}) too small. \
                         Increase max_stream_bytes_for_tile() for tile_size={}.",
                        info.tile_size
                    );
                }
            }
            drop(ov_view);
            bufs.overflow_staging.unmap();
        }

        let mut all_frames: Vec<Vec<RiceTile>> = Vec::with_capacity(batch_size);

        for slot in 0..batch_size {
            let mut frame_tiles = Vec::with_capacity(num_tiles * 3);
            for p in 0..3 {
                let stream_view = bufs.stream_staging_batch[slot][p].slice(..).get_mapped_range();
                let lengths_view = bufs.lengths_staging_batch[slot][p].slice(..).get_mapped_range();
                let k_view = bufs.k_staging_batch[slot][p].slice(..).get_mapped_range();
                let lengths_data: &[u32] = bytemuck::cast_slice(&lengths_view);
                let k_data: &[u32] = bytemuck::cast_slice(&k_view);

                for tile_idx in 0..num_tiles {
                    let k_values: Vec<u8> = (0..num_groups)
                        .map(|g| k_data[tile_idx * K_STRIDE + g] as u8)
                        .collect();
                    let k_zrl_values: Vec<u8> = (0..num_groups)
                        .map(|g| k_data[tile_idx * K_STRIDE + MAX_GROUPS + g] as u8)
                        .collect();
                    let skip_bitmap = k_data[tile_idx * K_STRIDE + K_STRIDE - 1] as u8;

                    let stream_lengths: Vec<u32> = (0..RICE_STREAMS_PER_TILE)
                        .map(|s| lengths_data[tile_idx * RICE_STREAMS_PER_TILE + s])
                        .collect();

                    let total_packed: usize = stream_lengths.iter().map(|&l| l as usize).sum();
                    let mut packed_data = Vec::with_capacity(total_packed);
                    for (s, &len) in stream_lengths.iter().enumerate() {
                        let slot_byte_offset =
                            (tile_idx * RICE_STREAMS_PER_TILE + s) * msb;
                        let len = len as usize;
                        packed_data.extend_from_slice(
                            &stream_view[slot_byte_offset..slot_byte_offset + len],
                        );
                    }

                    frame_tiles.push(RiceTile {
                        num_coefficients: info.tile_size * info.tile_size,
                        tile_size: info.tile_size,
                        num_levels,
                        num_groups: num_groups as u32,
                        k_values,
                        k_zrl_values,
                        skip_bitmap,
                        stream_lengths,
                        stream_data: packed_data,
                    });
                }

                drop(stream_view);
                drop(lengths_view);
                drop(k_view);
                bufs.stream_staging_batch[slot][p].unmap();
                bufs.lengths_staging_batch[slot][p].unmap();
                bufs.k_staging_batch[slot][p].unmap();
            }
            all_frames.push(frame_tiles);
        }

        all_frames
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
            k_values[t * K_STRIDE + K_STRIDE - 1] = tile.skip_bitmap as u32;
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

        // Pack stream data (allocate as u32 for alignment, then copy bytes in)
        let padded_words = (total_bytes as usize).div_ceil(4);
        let mut stream_data_u32 = vec![0u32; padded_words];
        {
            let stream_data_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut stream_data_u32);
            let mut write_pos = 0usize;
            for tile in tiles {
                stream_data_bytes[write_pos..write_pos + tile.stream_data.len()]
                    .copy_from_slice(&tile.stream_data);
                write_pos += tile.stream_data.len();
            }
        }

        RiceDecodeData {
            params: RiceParams {
                num_tiles: num_tiles as u32,
                coefficients_per_tile: info.tile_size * info.tile_size,
                plane_width: info.padded_width(),
                tile_size: info.tile_size,
                tiles_x: info.tiles_x(),
                num_levels: tiles.first().map_or(3, |t| t.num_levels),
                max_stream_bytes: max_stream_bytes_for_tile(info.tile_size) as u32,
                _pad0: 0,
            },
            k_values,
            stream_data: stream_data_u32,
            stream_offsets,
        }
    }

    /// Dispatch GPU Rice decode for one plane.
    #[allow(clippy::too_many_arguments)] // GPU dispatch requires separate buffer bindings
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
    use wgpu::util::DeviceExt;

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
            chroma_format: crate::ChromaFormat::Yuv444,
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
            chroma_format: crate::ChromaFormat::Yuv444,
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
