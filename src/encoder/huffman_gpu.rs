//! GPU Huffman encoder/decoder — 256 interleaved streams per tile.
//!
//! Two-pass encode: histogram (GPU) → codebook build (CPU) → encode (GPU).
//! Single-pass decode: 8-bit prefix table in shared memory.

use bytemuck::{Pod, Zeroable};
use wgpu;
use wgpu::util::DeviceExt;

use super::huffman::{
    self, build_decode_table, reconstruct_codes_from_lengths, HuffmanTile,
    HUFFMAN_ALPHABET_SIZE, HUFFMAN_STREAMS_PER_TILE,
};
use crate::{FrameInfo, GpuContext};

const MAX_STREAM_BYTES: usize = 512;
const _MAX_STREAM_WORDS: usize = MAX_STREAM_BYTES / 4;
const MAX_GROUPS: usize = 8;
const HIST_STRIDE: usize = MAX_GROUPS * HUFFMAN_ALPHABET_SIZE; // 512
const ZRL_STRIDE: usize = MAX_GROUPS * 2; // 16
const CB_STRIDE: usize = MAX_GROUPS * HUFFMAN_ALPHABET_SIZE; // 512
const DT_STRIDE: usize = MAX_GROUPS * 256; // 2048 (8-bit prefix table)

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct HuffmanParams {
    num_tiles: u32,
    coefficients_per_tile: u32,
    plane_width: u32,
    tile_size: u32,
    tiles_x: u32,
    num_levels: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Pre-allocated GPU buffers for Huffman encode (reused across calls).
struct CachedHuffmanEncodeBuffers {
    num_tiles: usize,
    // Histogram pass outputs
    hist_buf: wgpu::Buffer,
    hist_staging: wgpu::Buffer,
    zrl_buf: wgpu::Buffer,
    zrl_staging: wgpu::Buffer,
    // Encode pass inputs/outputs
    codebook_buf: wgpu::Buffer,
    k_zrl_buf: wgpu::Buffer,
    stream_buf: wgpu::Buffer,
    lengths_buf: wgpu::Buffer,
    // Per-plane staging for readback
    stream_staging: [wgpu::Buffer; 3],
    lengths_staging: [wgpu::Buffer; 3],
}

impl CachedHuffmanEncodeBuffers {
    fn new(ctx: &GpuContext, num_tiles: usize) -> Self {
        let total_streams = num_tiles * HUFFMAN_STREAMS_PER_TILE;
        let stream_size = (total_streams * MAX_STREAM_BYTES) as u64;
        let lengths_size = (total_streams * 4) as u64;
        let hist_size = (num_tiles * HIST_STRIDE * 4) as u64;
        let zrl_size = (num_tiles * ZRL_STRIDE * 4) as u64;
        let cb_size = (num_tiles * CB_STRIDE * 4) as u64;
        let k_zrl_size = (num_tiles * MAX_GROUPS * 4) as u64;

        let sc = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let scd = sc | wgpu::BufferUsages::COPY_DST;
        let mr = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let sd = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        Self {
            num_tiles,
            hist_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("huff_hist"),
                size: hist_size.max(4),
                usage: sc,
                mapped_at_creation: false,
            }),
            hist_staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("huff_hist_stg"),
                size: hist_size.max(4),
                usage: mr,
                mapped_at_creation: false,
            }),
            zrl_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("huff_zrl"),
                size: zrl_size.max(4),
                usage: sc,
                mapped_at_creation: false,
            }),
            zrl_staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("huff_zrl_stg"),
                size: zrl_size.max(4),
                usage: mr,
                mapped_at_creation: false,
            }),
            codebook_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("huff_codebook"),
                size: cb_size.max(4),
                usage: sd,
                mapped_at_creation: false,
            }),
            k_zrl_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("huff_k_zrl"),
                size: k_zrl_size.max(4),
                usage: sd,
                mapped_at_creation: false,
            }),
            stream_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("huff_stream"),
                size: stream_size.max(4),
                usage: scd,
                mapped_at_creation: false,
            }),
            lengths_buf: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("huff_lengths"),
                size: lengths_size.max(4),
                usage: sc,
                mapped_at_creation: false,
            }),
            stream_staging: std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(
                        ["huff_stream_stg0", "huff_stream_stg1", "huff_stream_stg2"][i],
                    ),
                    size: stream_size.max(4),
                    usage: mr,
                    mapped_at_creation: false,
                })
            }),
            lengths_staging: std::array::from_fn(|i| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(
                        ["huff_lengths_stg0", "huff_lengths_stg1", "huff_lengths_stg2"][i],
                    ),
                    size: lengths_size.max(4),
                    usage: mr,
                    mapped_at_creation: false,
                })
            }),
        }
    }
}

/// GPU Huffman encoder — 2-pass: histogram → codebook → encode.
pub struct GpuHuffmanEncoder {
    histogram_pipeline: wgpu::ComputePipeline,
    histogram_bgl: wgpu::BindGroupLayout,
    encode_pipeline: wgpu::ComputePipeline,
    encode_bgl: wgpu::BindGroupLayout,
    cached: Option<CachedHuffmanEncodeBuffers>,
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

impl GpuHuffmanEncoder {
    pub fn new(ctx: &GpuContext) -> Self {
        // Histogram pipeline
        let hist_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("huffman_histogram_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/huffman_histogram.wgsl").into(),
                ),
            });

        let histogram_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("huffman_hist_bgl"),
                    entries: &[
                        make_uniform_entry(0),       // params
                        make_storage_entry(1, true),  // input coefficients
                        make_storage_entry(2, false), // hist_output
                        make_storage_entry(3, false), // zrl_output
                    ],
                });

        let hist_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("huffman_hist_layout"),
                    bind_group_layouts: &[&histogram_bgl],
                    push_constant_ranges: &[],
                });

        let histogram_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("huffman_hist_pipeline"),
                    layout: Some(&hist_layout),
                    module: &hist_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Encode pipeline
        let enc_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("huffman_encode_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/huffman_encode.wgsl").into(),
                ),
            });

        let encode_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("huffman_enc_bgl"),
                    entries: &[
                        make_uniform_entry(0),       // params
                        make_storage_entry(1, true),  // input coefficients
                        make_storage_entry(2, true),  // codebook
                        make_storage_entry(3, true),  // k_zrl
                        make_storage_entry(4, false), // stream_output
                        make_storage_entry(5, false), // stream_lengths
                    ],
                });

        let enc_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("huffman_enc_layout"),
                    bind_group_layouts: &[&encode_bgl],
                    push_constant_ranges: &[],
                });

        let encode_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("huffman_enc_pipeline"),
                    layout: Some(&enc_layout),
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
            cached: None,
        }
    }

    fn ensure_buffers(&mut self, ctx: &GpuContext, num_tiles: usize) {
        let needs_realloc = self
            .cached
            .as_ref()
            .map_or(true, |c| c.num_tiles != num_tiles);
        if needs_realloc {
            self.cached = Some(CachedHuffmanEncodeBuffers::new(ctx, num_tiles));
        }
    }

    /// Encode 3 planes of quantized coefficients using GPU Huffman.
    /// Two-pass: histogram (GPU) → codebook build (CPU) → encode (GPU).
    pub fn encode_3planes_to_tiles(
        &mut self,
        ctx: &GpuContext,
        quantized_bufs: [&wgpu::Buffer; 3],
        info: &FrameInfo,
        num_levels: u32,
    ) -> Vec<HuffmanTile> {
        let num_tiles = (info.tiles_x() * info.tiles_y()) as usize;
        let total_streams = num_tiles * HUFFMAN_STREAMS_PER_TILE;
        let num_groups = (num_levels * 2) as usize;

        self.ensure_buffers(ctx, num_tiles);
        let bufs = self.cached.as_ref().unwrap();

        let stream_size = (total_streams * MAX_STREAM_BYTES) as u64;
        let lengths_size = (total_streams * 4) as u64;
        let hist_size = (num_tiles * HIST_STRIDE * 4) as u64;
        let zrl_size = (num_tiles * ZRL_STRIDE * 4) as u64;

        let params = HuffmanParams {
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
                    label: Some("huff_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let mut all_tiles = Vec::new();

        for (p, quantized_buf) in quantized_bufs.iter().enumerate() {
            // === Pass 1: GPU histogram ===
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("huff_hist_cmd"),
                });

            let hist_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("huff_hist_bg"),
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
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bufs.zrl_buf.as_entire_binding(),
                    },
                ],
            });

            {
                let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("huff_hist_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.histogram_pipeline);
                pass.set_bind_group(0, &hist_bg, &[]);
                pass.dispatch_workgroups(num_tiles as u32, 1, 1);
            }

            // Copy to staging
            cmd.copy_buffer_to_buffer(&bufs.hist_buf, 0, &bufs.hist_staging, 0, hist_size);
            cmd.copy_buffer_to_buffer(&bufs.zrl_buf, 0, &bufs.zrl_staging, 0, zrl_size);

            ctx.queue.submit(Some(cmd.finish()));

            // Map staging buffers
            {
                let (tx, rx) = std::sync::mpsc::channel();
                for staging in [&bufs.hist_staging, &bufs.zrl_staging] {
                    let tx_clone = tx.clone();
                    staging
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |result| {
                            tx_clone.send(result).unwrap();
                        });
                }
                drop(tx);
                ctx.device.poll(wgpu::Maintain::Wait);
                for _ in 0..2 {
                    rx.recv().unwrap().unwrap();
                }
            }

            // Read back histograms + ZRL stats
            let hist_data: Vec<u32> = {
                let view = bufs.hist_staging.slice(..).get_mapped_range();
                bytemuck::cast_slice(&view).to_vec()
            };
            let zrl_data: Vec<u32> = {
                let view = bufs.zrl_staging.slice(..).get_mapped_range();
                bytemuck::cast_slice(&view).to_vec()
            };
            bufs.hist_staging.unmap();
            bufs.zrl_staging.unmap();

            // === CPU: Build codebooks from histograms ===
            // codebook_data[tile * CB_STRIDE + group * ALPHABET_SIZE + sym] = (len << 16) | codeword
            let mut codebook_data = vec![0u32; num_tiles * CB_STRIDE];
            let mut k_zrl_data = vec![0u32; num_tiles * MAX_GROUPS];
            let mut tile_code_lengths: Vec<Vec<Vec<u8>>> = Vec::with_capacity(num_tiles);
            let mut tile_k_zrl: Vec<Vec<u8>> = Vec::with_capacity(num_tiles);

            for t in 0..num_tiles {
                let mut code_lengths_per_group = Vec::with_capacity(num_groups);
                for g in 0..num_groups {
                    let hist_base = t * HIST_STRIDE + g * HUFFMAN_ALPHABET_SIZE;
                    let freq: Vec<u32> = (0..HUFFMAN_ALPHABET_SIZE)
                        .map(|s| hist_data[hist_base + s])
                        .collect();

                    // Build codebook
                    let cb = huffman::build_canonical_codebook_from_freq(&freq);

                    // Pack into GPU format: (len << 16) | codeword
                    let cb_base = t * CB_STRIDE + g * HUFFMAN_ALPHABET_SIZE;
                    for s in 0..HUFFMAN_ALPHABET_SIZE {
                        if s < cb.0.len() && cb.0[s] > 0 {
                            codebook_data[cb_base + s] =
                                ((cb.0[s] as u32) << 16) | (cb.1[s] as u32);
                        }
                    }

                    code_lengths_per_group.push(cb.0);
                }

                // Compute k_zrl from ZRL stats
                let mut k_zrl_vec = Vec::with_capacity(num_groups);
                for g in 0..num_groups {
                    let sum = zrl_data[t * ZRL_STRIDE + g * 2] as u64;
                    let count = zrl_data[t * ZRL_STRIDE + g * 2 + 1] as u64;
                    let k = if count > 0 {
                        let mean = sum / count;
                        if mean == 0 {
                            0u32
                        } else {
                            (63 - mean.leading_zeros()).min(15)
                        }
                    } else {
                        0
                    };
                    k_zrl_data[t * MAX_GROUPS + g] = k;
                    k_zrl_vec.push(k as u8);
                }

                tile_code_lengths.push(code_lengths_per_group);
                tile_k_zrl.push(k_zrl_vec);
            }

            // === Upload codebooks + k_zrl ===
            ctx.queue.write_buffer(
                &bufs.codebook_buf,
                0,
                bytemuck::cast_slice(&codebook_data),
            );
            ctx.queue.write_buffer(
                &bufs.k_zrl_buf,
                0,
                bytemuck::cast_slice(&k_zrl_data),
            );

            // === Pass 2: GPU encode ===
            let mut cmd = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("huff_enc_cmd"),
                });

            // Clear stream buffer
            cmd.clear_buffer(&bufs.stream_buf, 0, None);

            let enc_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("huff_enc_bg"),
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
                        resource: bufs.codebook_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bufs.k_zrl_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: bufs.stream_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: bufs.lengths_buf.as_entire_binding(),
                    },
                ],
            });

            {
                let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("huff_enc_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.encode_pipeline);
                pass.set_bind_group(0, &enc_bg, &[]);
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

            ctx.queue.submit(Some(cmd.finish()));

            // Map staging
            {
                let (tx, rx) = std::sync::mpsc::channel();
                for staging in [&bufs.stream_staging[p], &bufs.lengths_staging[p]] {
                    let tx_clone = tx.clone();
                    staging
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |result| {
                            tx_clone.send(result).unwrap();
                        });
                }
                drop(tx);
                ctx.device.poll(wgpu::Maintain::Wait);
                for _ in 0..2 {
                    rx.recv().unwrap().unwrap();
                }
            }

            // Read back streams + lengths
            let stream_data: Vec<u8> = {
                let view = bufs.stream_staging[p].slice(..).get_mapped_range();
                view.to_vec()
            };
            let lengths_data: Vec<u32> = {
                let view = bufs.lengths_staging[p].slice(..).get_mapped_range();
                bytemuck::cast_slice(&view).to_vec()
            };
            bufs.stream_staging[p].unmap();
            bufs.lengths_staging[p].unmap();

            // Pack into HuffmanTile structs
            for t in 0..num_tiles {
                let stream_lengths: Vec<u32> = (0..HUFFMAN_STREAMS_PER_TILE)
                    .map(|s| lengths_data[t * HUFFMAN_STREAMS_PER_TILE + s])
                    .collect();

                let mut packed_data = Vec::new();
                for s in 0..HUFFMAN_STREAMS_PER_TILE {
                    let slot_offset =
                        (t * HUFFMAN_STREAMS_PER_TILE + s) * MAX_STREAM_BYTES;
                    let len = stream_lengths[s] as usize;
                    packed_data
                        .extend_from_slice(&stream_data[slot_offset..slot_offset + len]);
                }

                all_tiles.push(HuffmanTile {
                    num_coefficients: info.tile_size * info.tile_size,
                    tile_size: info.tile_size,
                    num_levels,
                    num_groups: num_groups as u32,
                    code_lengths: tile_code_lengths[t].clone(),
                    k_zrl_values: tile_k_zrl[t].clone(),
                    stream_lengths,
                    stream_data: packed_data,
                });
            }
        }

        all_tiles
    }
}

/// GPU Huffman decoder — 8-bit prefix table decode.
pub struct GpuHuffmanDecoder {
    decode_pipeline: wgpu::ComputePipeline,
    decode_bgl: wgpu::BindGroupLayout,
}

impl GpuHuffmanDecoder {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("huffman_decode_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/huffman_decode.wgsl").into(),
                ),
            });

        let decode_bgl =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("huffman_decode_bgl"),
                    entries: &[
                        make_uniform_entry(0),       // params
                        make_storage_entry(1, true),  // decode_table
                        make_storage_entry(2, true),  // k_zrl
                        make_storage_entry(3, true),  // stream_data
                        make_storage_entry(4, true),  // stream_offsets
                        make_storage_entry(5, false), // output
                    ],
                });

        let layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("huffman_decode_layout"),
                    bind_group_layouts: &[&decode_bgl],
                    push_constant_ranges: &[],
                });

        let decode_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("huffman_decode_pipeline"),
                    layout: Some(&layout),
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

    /// Pack HuffmanTile data for GPU decode.
    pub fn pack_decode_data(tiles: &[HuffmanTile], info: &FrameInfo) -> HuffmanDecodeData {
        let num_tiles = tiles.len();
        let total_streams = num_tiles * HUFFMAN_STREAMS_PER_TILE;
        let _num_groups = tiles.first().map_or(6, |t| t.num_groups as usize);

        // Build 8-bit prefix decode tables per tile per group
        let mut decode_table = vec![0u32; num_tiles * DT_STRIDE];
        let mut k_zrl_data = vec![0u32; num_tiles * MAX_GROUPS];

        for (t, tile) in tiles.iter().enumerate() {
            for (g, cl) in tile.code_lengths.iter().enumerate() {
                let codewords = reconstruct_codes_from_lengths(cl);
                let dt = build_decode_table(cl, &codewords);
                let base = t * DT_STRIDE + g * 256;
                decode_table[base..base + 256].copy_from_slice(&dt);
            }
            for (g, &k) in tile.k_zrl_values.iter().enumerate() {
                k_zrl_data[t * MAX_GROUPS + g] = k as u32;
            }
        }

        // Compute stream offsets and pack stream data
        let mut stream_offsets = vec![0u32; total_streams];
        let mut total_bytes = 0u32;
        for (t, tile) in tiles.iter().enumerate() {
            for s in 0..HUFFMAN_STREAMS_PER_TILE {
                stream_offsets[t * HUFFMAN_STREAMS_PER_TILE + s] = total_bytes;
                total_bytes += tile.stream_lengths[s];
            }
        }

        let padded_bytes = ((total_bytes as usize + 3) / 4) * 4;
        let mut stream_data = vec![0u8; padded_bytes];
        let mut write_pos = 0usize;
        for tile in tiles {
            stream_data[write_pos..write_pos + tile.stream_data.len()]
                .copy_from_slice(&tile.stream_data);
            write_pos += tile.stream_data.len();
        }

        HuffmanDecodeData {
            params: HuffmanParams {
                num_tiles: num_tiles as u32,
                coefficients_per_tile: info.tile_size * info.tile_size,
                plane_width: info.padded_width(),
                tile_size: info.tile_size,
                tiles_x: info.tiles_x(),
                num_levels: tiles.first().map_or(3, |t| t.num_levels),
                _pad0: 0,
                _pad1: 0,
            },
            decode_table,
            k_zrl_data,
            stream_data: bytemuck::cast_slice(&stream_data).to_vec(),
            stream_offsets,
        }
    }

    /// Dispatch GPU Huffman decode for one plane.
    pub fn dispatch_decode(
        &self,
        ctx: &GpuContext,
        cmd: &mut wgpu::CommandEncoder,
        params_buf: &wgpu::Buffer,
        decode_table_buf: &wgpu::Buffer,
        k_zrl_buf: &wgpu::Buffer,
        stream_buf: &wgpu::Buffer,
        offsets_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_tiles: u32,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("huffman_decode_bg"),
            layout: &self.decode_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: decode_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: k_zrl_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: stream_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("huffman_decode_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.decode_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(num_tiles, 1, 1);
    }
}

/// Packed data ready for GPU Huffman decode upload.
pub struct HuffmanDecodeData {
    pub params: HuffmanParams,
    pub decode_table: Vec<u32>,
    pub k_zrl_data: Vec<u32>,
    pub stream_data: Vec<u32>,
    pub stream_offsets: Vec<u32>,
}
