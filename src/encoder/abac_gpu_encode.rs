//! GPU encode for the adaptive binary code-block coder — the missing half of `abac_gpu`.
//!
//! abac has had a GPU decoder since it landed and a CPU encoder only, and that asymmetry was not
//! merely slow. It is the *cause* of the ARCH-3 / BUG-18 class of defect: selecting a coder with
//! no GPU encode shader cleared `gpu_entropy_encode`, and that flag also chose which whole-frame
//! P-frame pipeline ran, so asking for abac silently swapped the frame encoder. This module
//! removes what that routing was routing around.
//!
//! One thread per code-block, as on the decode side and for the same reason: the coder is serial
//! within a block and embarrassingly parallel across them, and a padded 1080p 4:4:4 frame holds
//! roughly 3000 blocks at cb=64.
//!
//! # The one structural problem: a block's coded size is not known before it is coded
//!
//! A decoder is told every block's length. An encoder has to discover it, and it has to place
//! ~3000 variable-length streams into one buffer without threads treading on each other. Both
//! documented answers are implemented here and [`Sizing`] selects between them, because which is
//! faster is a measurement and not a matter of taste:
//!
//! - [`Sizing::CountThenEmit`] — run the coder twice. The first pass writes nothing but each
//!   block's byte count; the host prefix-sums those into exact offsets; the second pass writes.
//!   Exact packing, no bound to get wrong, and the scratch buffer is the size of the output.
//!   Costs two coder passes.
//! - [`Sizing::BoundedSlots`] — compute a *provable* per-block ceiling (a cheap pass over the
//!   coefficients with no arithmetic coding in it), give every block a slot that size, code once,
//!   then compact the streams together on the GPU. One coder pass, at the price of a scratch
//!   buffer several times the output size and one more round trip.
//!
//! **The bound is derived, not guessed.** BUG-22 is the standing example of the alternative: a
//! Huffman stream slot fixed at 512 bytes from a rate estimate, with nothing checking the write
//! pointer, cost 7.8–10.9 dB at q=90 when a stream spilled into its neighbour's slot. The
//! ceiling here comes from the coders' own renormalisation invariants (`abac_encode.wgsl`,
//! "Provable per-block output ceiling"), and the shader *also* raises a flag if a slot is ever
//! exceeded, which this module turns into a panic. A bound that is only argued is a bound that
//! is wrong later.
//!
//! # Bit-exactness is the requirement, not decodability
//!
//! A GPU encoder that produced a different-but-valid stream would move every rate figure abac
//! has and would make CPU and GPU two coders rather than one. [`verify_against_cpu_encoder`]
//! asserts byte-identity per block against `abac.rs`, and `tests/abac_gpu_encode.rs` runs it over
//! the geometries the decoder is verified on plus the degenerate planes.

use bytemuck::{Pod, Zeroable};

use super::abac::Coder;
use super::abac_gpu::MAX_BLOCK_W;
use super::abac_tile::{code_blocks, AbacTile};
use crate::gpu_util::{ensure_var_buf, read_buffer_u32};
use crate::GpuContext;

/// How the encoder decides where each block's bytes go. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sizing {
    /// Two coder passes: count, prefix-sum, emit at exact offsets.
    #[default]
    CountThenEmit,
    /// One coder pass into provably-bounded slots, then a GPU compaction copy.
    BoundedSlots,
}

impl Sizing {
    /// `GNC_ABAC_GPU_SIZING=count|slots`. Defaults to [`Sizing::CountThenEmit`] — the exact one,
    /// which needs no bound at all. The default is a correctness preference, not a speed claim:
    /// the two are built to be measured together on an idle machine
    /// (`cargo test --release --test abac_bench -- --ignored --nocapture --test-threads=1`).
    pub fn from_env() -> Self {
        match std::env::var("GNC_ABAC_GPU_SIZING").as_deref() {
            Ok("slots") | Ok("bounded") => Sizing::BoundedSlots,
            _ => Sizing::CountThenEmit,
        }
    }
}

/// Per-block geometry and output slot. Layout must match `EncBlock` in `abac_encode.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct EncBlock {
    in_offset: u32,
    width: u32,
    height: u32,
    stride: u32,
    dst_byte: u32,
    cap_bytes: u32,
    index: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    num_blocks: u32,
    emit: u32,
    _pad: [u32; 2],
}

/// One block's stream move, for the compaction pass. Matches `CopyJob` in the shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CopyJob {
    src_byte: u32,
    dst_byte: u32,
    len_bytes: u32,
    _pad0: u32,
}

/// What the last plane encode actually did on the GPU.
///
/// This is the canary, and it is not a formality: a shader that silently fell back to the CPU
/// coder would satisfy bit-exactness, round-trip and rate *by construction*, so those three
/// checks cannot detect the one failure most likely to happen. CLAUDE.md, "no silent features".
#[derive(Debug, Clone, Copy, Default)]
pub struct EncodeStats {
    /// Blocks dispatched, one thread each.
    pub blocks: u32,
    /// Bytes of code-block stream the GPU produced.
    pub bytes: usize,
    /// Scratch the sizing mode needed to produce them. Equal to `bytes` rounded up per block for
    /// `CountThenEmit`; several times it for `BoundedSlots`.
    pub scratch_bytes: usize,
    /// Coder passes run over the coefficients. 2 for `CountThenEmit`, 1 for `BoundedSlots`.
    pub coder_passes: u32,
}

fn align4(v: u32) -> u32 {
    (v + 3) & !3
}

pub struct GpuAbacEncoder {
    /// One pipeline per coder variant, indexed by `Coder as usize`, exactly as the decoder does.
    coder: [wgpu::ComputePipeline; 2],
    bound: wgpu::ComputePipeline,
    compact: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    compact_bgl: wgpu::BindGroupLayout,

    params_buf: wgpu::Buffer,
    blocks_buf: wgpu::Buffer,
    blocks_cap: u64,
    slots_buf: wgpu::Buffer,
    slots_cap: u64,
    lengths_buf: wgpu::Buffer,
    lengths_cap: u64,
    jobs_buf: wgpu::Buffer,
    jobs_cap: u64,
    packed_buf: wgpu::Buffer,
    packed_cap: u64,
    flags_buf: wgpu::Buffer,
    /// A 4-byte stand-in bound to the output slot while counting, when nothing is written.
    dummy_out: wgpu::Buffer,

    stats: EncodeStats,
}

impl GpuAbacEncoder {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("abac_encode"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/abac_encode.wgsl").into(),
                ),
            });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("abac_encode_bgl"),
                entries: &[
                    entry(0, wgpu::BufferBindingType::Uniform),
                    entry(1, ro), // input coefficients
                    entry(2, ro), // block table
                    entry(3, rw), // output words
                    entry(4, rw), // per-block lengths
                    entry(5, rw), // flags
                ],
            });
        // The compaction entry point touches four of the bindings and none of the others, so it
        // gets a layout of its own rather than dummy buffers for the rest.
        let compact_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("abac_compact_bgl"),
                entries: &[
                    entry(0, wgpu::BufferBindingType::Uniform),
                    entry(3, rw), // slots, read
                    entry(6, ro), // copy jobs
                    entry(7, rw), // packed output
                ],
            });
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("abac_encode_layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let compact_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("abac_compact_layout"),
                bind_group_layouts: &[&compact_bgl],
                push_constant_ranges: &[],
            });
        let build = |name: &str, lay: &wgpu::PipelineLayout| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(lay),
                    module: &shader,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        // Index order must match `Coder`.
        let coder = [build("main", &layout), build("main_rc", &layout)];
        let bound = build("bound", &layout);
        let compact = build("compact", &compact_layout);

        let stor = wgpu::BufferUsages::STORAGE;
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            coder,
            bound,
            compact,
            bgl,
            compact_bgl,
            params_buf: mk(
                "abac_enc_params",
                std::mem::size_of::<Params>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            blocks_buf: mk("abac_enc_blocks", 4, stor | wgpu::BufferUsages::COPY_DST),
            blocks_cap: 4,
            slots_buf: mk("abac_enc_slots", 4, stor | wgpu::BufferUsages::COPY_SRC),
            slots_cap: 4,
            lengths_buf: mk("abac_enc_lengths", 4, stor | wgpu::BufferUsages::COPY_SRC),
            lengths_cap: 4,
            jobs_buf: mk("abac_enc_jobs", 4, stor | wgpu::BufferUsages::COPY_DST),
            jobs_cap: 4,
            packed_buf: mk("abac_enc_packed", 4, stor | wgpu::BufferUsages::COPY_SRC),
            packed_cap: 4,
            flags_buf: mk(
                "abac_enc_flags",
                8,
                stor | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            ),
            dummy_out: mk("abac_enc_dummy", 4, stor),
            stats: EncodeStats::default(),
        }
    }

    /// What the last `encode_plane_to_tiles` call did. See [`EncodeStats`].
    pub fn stats(&self) -> EncodeStats {
        self.stats
    }

    fn bind(&self, ctx: &GpuContext, input: &wgpu::Buffer, out: &wgpu::Buffer) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("abac_encode_bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.blocks_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.lengths_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.flags_buf.as_entire_binding() },
            ],
        })
    }

    /// Run one entry point over `num_blocks` blocks and wait for it.
    ///
    /// `workgroup` is the shader's `@workgroup_size`; getting it wrong silently under- or
    /// over-dispatches, which for `main` means blocks that never get coded.
    fn run(
        &self,
        ctx: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
        groups: u32,
        clear_flags: bool,
        label: &str,
    ) {
        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        if clear_flags {
            cmd.clear_buffer(&self.flags_buf, 0, Some(8));
        }
        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        ctx.queue.submit(Some(cmd.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
    }

    fn set_params(&self, ctx: &GpuContext, num_blocks: u32, emit: u32) {
        let p = Params { num_blocks, emit, _pad: [0; 2] };
        ctx.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
    }

    fn check_flags(&self, ctx: &GpuContext, sizing: Sizing) {
        let flags = read_buffer_u32(ctx, &self.flags_buf, 2);
        assert_eq!(
            flags[0], 0,
            "abac GPU encode: a code-block exceeded its output slot ({sizing:?}). For \
             CountThenEmit that means the emit pass disagreed with the count pass — the coder is \
             not deterministic, which it must be. For BoundedSlots it means the ceiling in \
             abac_encode.wgsl is wrong, which is BUG-22's failure mode and must be fixed rather \
             than padded."
        );
        assert_eq!(
            flags[1], 0,
            "abac GPU encode: a coefficient magnitude reached 2^29. The neighbourhood sum \
             saturates on the CPU (`saturating_add`) and wraps in both shaders, so above that \
             magnitude the GPU pair and the CPU reference can disagree on a context bucket. \
             Quantised wavelet coefficients cannot reach it; if this fires, the input is not what \
             this path assumes and the stream must not be trusted."
        );
    }

    /// Encode one quantised plane straight from its GPU buffer into `code_blocks`-ordered tiles.
    ///
    /// `quantized` holds `plane_width * tiles_y * tile_size` f32 coefficients — the same buffer
    /// the CPU path reads back and the same integral values, so `i32(round(v))` in the shader and
    /// `.round() as i32` on the host agree exactly.
    ///
    /// Tiles come out in the row-major order `entropy_encode_tiles` pushes them in, and blocks
    /// within a tile in `code_blocks` order, so the result is interchangeable with
    /// `abac_encode_tile`'s — which is what makes byte-identity testable.
    #[allow(clippy::too_many_arguments)] // plane geometry plus the coder and sizing choices
    pub fn encode_plane_to_tiles(
        &mut self,
        ctx: &GpuContext,
        quantized: &wgpu::Buffer,
        plane_width: usize,
        tiles_x: usize,
        tiles_y: usize,
        tile_size: u32,
        num_levels: u32,
        cb: u32,
        coder: Coder,
        sizing: Sizing,
    ) -> Vec<AbacTile> {
        let ts = tile_size as usize;
        let num_tiles = tiles_x * tiles_y;
        assert!(num_tiles > 0, "a plane with no tiles cannot be encoded");
        // Geometry is derived from (tile_size, num_levels, cb) by the same function the CPU
        // encoder, the CPU decoder and the GPU decoder call. Deriving it a second time here is
        // exactly how a coverage bug gets in, and a coverage bug shrinks the file while every
        // individual block still round-trips.
        let geom = code_blocks(ts, num_levels, cb as usize);
        assert!(
            geom.iter().all(|&(_, _, w, _)| w as u32 <= MAX_BLOCK_W),
            "code-block width must be <= {MAX_BLOCK_W}: the shader keeps two rows of neighbour \
             magnitudes per thread in workgroup memory, sized for it"
        );
        let per_tile = geom.len();
        let num_blocks = num_tiles * per_tile;

        let mut blocks: Vec<EncBlock> = Vec::with_capacity(num_blocks);
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                for &(bx, by, bw, bh) in &geom {
                    let index = blocks.len() as u32;
                    blocks.push(EncBlock {
                        in_offset: ((ty * ts + by) * plane_width + tx * ts + bx) as u32,
                        width: bw as u32,
                        height: bh as u32,
                        stride: plane_width as u32,
                        dst_byte: 0,
                        cap_bytes: 0,
                        index,
                        _pad0: 0,
                    });
                }
            }
        }
        // One thread codes one block and a Metal SIMD group runs at its slowest lane, so a group
        // holding an 8x8 block next to a 64x64 one wastes most of itself. The dispatch order is
        // therefore by descending area, and `index` — not position — says where a block's length
        // and bytes belong. Same argument, and same sort, as the decoder's `pack_decode_data`.
        blocks.sort_by_key(|b| std::cmp::Reverse(b.width * b.height));

        let groups = (num_blocks as u32).div_ceil(32);
        let bound_groups = (num_blocks as u32).div_ceil(64);
        let lengths_bytes = (num_blocks * 4) as u64;
        ensure_var_buf(
            ctx,
            &mut self.lengths_buf,
            &mut self.lengths_cap,
            lengths_bytes,
            "abac_enc_lengths",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let blocks_bytes = std::mem::size_of_val(&blocks[..]) as u64;
        ensure_var_buf(
            ctx,
            &mut self.blocks_buf,
            &mut self.blocks_cap,
            blocks_bytes,
            "abac_enc_blocks",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );

        // --- Pass 1: how big is every block? ---
        ctx.queue
            .write_buffer(&self.blocks_buf, 0, bytemuck::cast_slice(&blocks));
        self.set_params(ctx, num_blocks as u32, 0);
        let (slot_sizes, coder_passes) = match sizing {
            // The flags are read once, after the emit pass, and not here. Nothing the first pass
            // can raise is missed by that: `bound` touches neither flag, and the counting pass
            // cannot overflow a slot because it writes nothing. Checking twice would put an extra
            // copy-and-map round trip inside the region the bench times, and asymmetrically —
            // which is how a sizing mode gets chosen for the wrong reason.
            Sizing::CountThenEmit => {
                let bind = self.bind(ctx, quantized, &self.dummy_out);
                self.run(ctx, &self.coder[coder as usize], &bind, groups, true, "abac_count");
                let counts = read_buffer_u32(ctx, &self.lengths_buf, num_blocks);
                (counts.iter().map(|&c| align4(c)).collect::<Vec<u32>>(), 2)
            }
            Sizing::BoundedSlots => {
                let bind = self.bind(ctx, quantized, &self.dummy_out);
                self.run(ctx, &self.bound, &bind, bound_groups, true, "abac_bound");
                // Already word-aligned by the shader.
                (read_buffer_u32(ctx, &self.lengths_buf, num_blocks), 1)
            }
        };

        // --- Slot layout, in canonical order so the packed buffer reads back sequentially ---
        let mut slot_off = vec![0u32; num_blocks];
        let mut total: u64 = 0;
        for (off, &size) in slot_off.iter_mut().zip(slot_sizes.iter()) {
            *off = u32::try_from(total).expect("abac GPU encode: slot buffer exceeds 4 GiB");
            total += u64::from(size);
        }
        let limit = u64::from(ctx.device.limits().max_storage_buffer_binding_size);
        assert!(
            total <= limit,
            "abac GPU encode: {sizing:?} wants {total} bytes of scratch for one plane, over the \
             device's {limit}-byte storage binding limit. CountThenEmit needs only the output \
             size; BoundedSlots needs roughly 3 bytes per coefficient."
        );
        for b in &mut blocks {
            let i = b.index as usize;
            b.dst_byte = slot_off[i];
            b.cap_bytes = slot_sizes[i];
        }
        ensure_var_buf(
            ctx,
            &mut self.slots_buf,
            &mut self.slots_cap,
            total.max(4),
            "abac_enc_slots",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // --- Pass 2: code, writing into the slots ---
        ctx.queue
            .write_buffer(&self.blocks_buf, 0, bytemuck::cast_slice(&blocks));
        self.set_params(ctx, num_blocks as u32, 1);
        {
            let bind = self.bind(ctx, quantized, &self.slots_buf);
            self.run(ctx, &self.coder[coder as usize], &bind, groups, true, "abac_emit");
        }
        self.check_flags(ctx, sizing);
        let lengths = read_buffer_u32(ctx, &self.lengths_buf, num_blocks);
        if sizing == Sizing::CountThenEmit {
            // The two passes must agree: the same coder over the same coefficients. If they do
            // not, the slot offsets are wrong for every block after the first disagreement and
            // the tiles would be silently interleaved.
            for (i, (&len, &slot)) in lengths.iter().zip(slot_sizes.iter()).enumerate() {
                assert_eq!(
                    slot,
                    align4(len),
                    "abac GPU encode: block {i} counted {slot} bytes (word-aligned) in pass 1 \
                     and emitted {len} in pass 2"
                );
            }
        }

        // --- Gather the bytes ---
        let (bytes, src_off) = match sizing {
            Sizing::CountThenEmit => {
                // Slots are already the exact output, one block after another with at most three
                // bytes of word padding between them.
                let words = read_buffer_u32(ctx, &self.slots_buf, (total / 4) as usize);
                (words_to_bytes(&words, total as usize), slot_off.clone())
            }
            Sizing::BoundedSlots => {
                // Slots are several times the output size, so compact on the GPU rather than
                // reading the gaps back over the bus.
                let mut jobs = Vec::with_capacity(num_blocks);
                let mut packed_off = Vec::with_capacity(num_blocks);
                let mut acc: u64 = 0;
                for (&src, &len) in slot_off.iter().zip(lengths.iter()) {
                    let dst = u32::try_from(acc)
                        .expect("abac GPU encode: packed output exceeds 4 GiB");
                    packed_off.push(dst);
                    jobs.push(CopyJob {
                        src_byte: src,
                        dst_byte: dst,
                        len_bytes: len,
                        _pad0: 0,
                    });
                    acc += u64::from(align4(len));
                }
                let jobs_bytes = std::mem::size_of_val(&jobs[..]) as u64;
                ensure_var_buf(
                    ctx,
                    &mut self.jobs_buf,
                    &mut self.jobs_cap,
                    jobs_bytes,
                    "abac_enc_jobs",
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                );
                ensure_var_buf(
                    ctx,
                    &mut self.packed_buf,
                    &mut self.packed_cap,
                    acc.max(4),
                    "abac_enc_packed",
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                );
                ctx.queue
                    .write_buffer(&self.jobs_buf, 0, bytemuck::cast_slice(&jobs));
                let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("abac_compact_bind"),
                    layout: &self.compact_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.slots_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.jobs_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: self.packed_buf.as_entire_binding(),
                        },
                    ],
                });
                // One workgroup per block, threads striding over its words.
                self.run(
                    ctx,
                    &self.compact,
                    &bind,
                    num_blocks as u32,
                    false,
                    "abac_compact",
                );
                let words = read_buffer_u32(ctx, &self.packed_buf, (acc / 4) as usize);
                (words_to_bytes(&words, acc as usize), packed_off)
            }
        };

        // --- Cut into tiles ---
        let mut tiles = Vec::with_capacity(num_tiles);
        let mut total_bytes = 0usize;
        for t in 0..num_tiles {
            let mut block_lengths = Vec::with_capacity(per_tile);
            let mut block_data = Vec::new();
            for b in 0..per_tile {
                let i = t * per_tile + b;
                let len = lengths[i] as usize;
                let off = src_off[i] as usize;
                block_lengths.push(lengths[i]);
                block_data.extend_from_slice(&bytes[off..off + len]);
            }
            total_bytes += block_data.len();
            tiles.push(AbacTile {
                tile_size,
                num_levels,
                cb_size: cb,
                coder,
                block_lengths,
                block_data,
            });
        }

        self.stats = EncodeStats {
            blocks: num_blocks as u32,
            bytes: total_bytes,
            scratch_bytes: total as usize,
            coder_passes,
        };
        if super::diagnostics::enabled() {
            let empty = lengths.iter().filter(|&&l| l == 0).count();
            eprintln!(
                "  [abac-gpu] plane {tiles_x}x{tiles_y} tiles: abac_blocks={num_blocks} \
                 (empty={empty}) bytes={total_bytes} scratch={total} passes={coder_passes} \
                 coder={coder:?} cb={cb} sizing={sizing:?}"
            );
        }
        tiles
    }
}

/// Unpack little-endian bytes out of the words the shader wrote, exactly as the decoder's
/// `get_byte` reads them.
///
/// Note for whoever reads a throughput figure off this path: `read_buffer_u32` allocates a fresh
/// staging buffer per call, and this then copies its words into a `Vec<u8>` before the tiles are
/// cut. Both are inside what the bench times, deliberately — it is what the encoder pays — but if
/// the readback turns out to dominate, cached staging and a borrow instead of a copy are the two
/// obvious moves, and neither touches the shader or the bytes.
fn words_to_bytes(words: &[u32], len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out.truncate(len);
    out
}

/// Encode a plane on the GPU and assert every block's bytes are identical to `abac.rs`'s.
///
/// Returns the bytes the GPU produced. Byte-identity per block is the criterion, not
/// "decodes to the same picture": a valid-but-different stream would move every rate figure abac
/// has, and would mean the CPU and GPU encoders are two coders rather than one implementation.
#[allow(clippy::too_many_arguments)] // plane geometry, coder and sizing: all of it is the case under test
pub fn verify_against_cpu_encoder(
    ctx: &GpuContext,
    enc: &mut GpuAbacEncoder,
    plane: &[i32],
    plane_width: usize,
    tile_size: u32,
    tiles_x: usize,
    tiles_y: usize,
    num_levels: u32,
    cb: u32,
    coder: Coder,
    sizing: Sizing,
) -> usize {
    use wgpu::util::DeviceExt;
    let ts = tile_size as usize;
    assert_eq!(
        plane.len(),
        plane_width * tiles_y * ts,
        "plane must be exactly {tiles_x}x{tiles_y} tiles of {ts}px"
    );
    let floats: Vec<f32> = plane.iter().map(|&v| v as f32).collect();
    let input = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("abac_enc_verify_input"),
            contents: bytemuck::cast_slice(&floats),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let got = enc.encode_plane_to_tiles(
        ctx,
        &input,
        plane_width,
        tiles_x,
        tiles_y,
        tile_size,
        num_levels,
        cb,
        coder,
        sizing,
    );

    let mut bytes = 0usize;
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let t = ty * tiles_x + tx;
            let mut coeffs = Vec::with_capacity(ts * ts);
            for y in 0..ts {
                let row = (ty * ts + y) * plane_width + tx * ts;
                coeffs.extend_from_slice(&plane[row..row + ts]);
            }
            let want = super::abac_tile::abac_encode_tile(&coeffs, tile_size, num_levels, cb, coder);
            let g = &got[t];
            assert_eq!(
                g.block_lengths, want.block_lengths,
                "tile {t} ({coder:?}, {sizing:?}, cb={cb}, levels={num_levels}): GPU block \
                 lengths differ from the CPU encoder's"
            );
            if g.block_data != want.block_data {
                let at = g
                    .block_data
                    .iter()
                    .zip(want.block_data.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| want.block_data.len().min(g.block_data.len()));
                // Which block that byte fell in, so the failure names a block rather than an
                // offset into a concatenation.
                let mut acc = 0usize;
                let mut blk = 0usize;
                for (i, &l) in g.block_lengths.iter().enumerate() {
                    if at < acc + l as usize {
                        blk = i;
                        break;
                    }
                    acc += l as usize;
                }
                panic!(
                    "tile {t} block {blk} ({coder:?}, {sizing:?}): GPU encode diverged from \
                     abac.rs at byte {} of the block (0x{:02x} vs 0x{:02x}). The two encoders \
                     must be one implementation: a different-but-valid stream moves every rate \
                     figure abac has.",
                    at - acc,
                    g.block_data.get(at).copied().unwrap_or(0),
                    want.block_data.get(at).copied().unwrap_or(0),
                );
            }
            bytes += g.block_data.len();
        }
    }
    bytes
}
