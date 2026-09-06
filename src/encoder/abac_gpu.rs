//! GPU decode for the adaptive binary code-block coder.
//!
//! One thread per code-block. The coder is serial *within* a block — every symbol's interval
//! depends on the previous one and every context on already-decoded neighbours — so there is
//! nothing to parallelise inside a block. The parallelism is across blocks, and a padded 1080p
//! 4:4:4 frame holds roughly 3000 of them at 64px code-blocks, which is what makes an adaptive
//! arithmetic coder viable on a GPU at all.
//!
//! `abac_decode.wgsl` is a direct port of `abac.rs` and the two must agree bit-for-bit.
//! `verify_against_cpu` asserts exactly that on real coefficients, and it is not optional: an
//! adaptive arithmetic decoder that diverges from its encoder by one bit does not fail. It carries
//! on and produces plausible-looking garbage, because every later symbol is decoded from a
//! corrupted interval *and* a corrupted context. The first port had `32 - firstLeadingBit(nb)`
//! where the Rust has `32 - leading_zeros(nb)`, which differ by one; it decoded without error and
//! reconstructed a wrong image.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::abac::Coder;
use crate::GpuContext;

/// Widest code-block the decode shader can handle, set by its workgroup scratch. Must match
/// `MAX_BLOCK_W` in `abac_decode.wgsl`.
pub const MAX_BLOCK_W: u32 = 64;

/// Per-block geometry. Layout must match `BlockInfo` in `abac_decode.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct BlockInfo {
    /// Byte offset of this block's stream within the packed buffer.
    pub byte_offset: u32,
    /// Length of this block's stream in bytes. Explicit rather than derived from the next block's
    /// offset, so the host can order blocks freely — which it does, to put equal-sized blocks in
    /// the same SIMD group.
    pub byte_len: u32,
    /// Index of the block's top-left coefficient in the output plane.
    pub out_offset: u32,
    pub width: u32,
    pub height: u32,
    /// Row stride of the output plane, in coefficients.
    pub stride: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl BlockInfo {
    pub fn new(
        byte_offset: u32,
        byte_len: u32,
        out_offset: u32,
        width: u32,
        height: u32,
        stride: u32,
    ) -> Self {
        Self {
            byte_offset,
            byte_len,
            out_offset,
            width,
            height,
            stride,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Four scalars rather than a `vec3` tail: a `vec3<u32>` in WGSL forces 16-byte alignment, which
/// makes the uniform 32 bytes and stops it matching this struct.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    num_blocks: u32,
    _pad: [u32; 3],
}

pub struct GpuAbacDecoder {
    /// One pipeline per coder variant, indexed by `Coder as usize`. Specialised at compile time
    /// by entry point rather than branched at run time, so neither variant pays for the other.
    pipelines: [wgpu::ComputePipeline; 2],
    bgl: wgpu::BindGroupLayout,
}

impl GpuAbacDecoder {
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("abac_decode"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/abac_decode.wgsl").into(),
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
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("abac_decode_bgl"),
                entries: &[
                    entry(0, wgpu::BufferBindingType::Uniform),
                    entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("abac_decode_layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let build = |entry: &str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        // Index order must match `Coder`.
        let pipelines = [build("main"), build("main_rc")];
        Self { pipelines, bgl }
    }

    /// Decode every block into one output plane. Returns the plane and the submit-to-idle time.
    ///
    /// `packed` holds the blocks' streams concatenated in the same order as `infos`, each info's
    /// `byte_offset` pointing into it. The shader takes a block's stream end from the *next*
    /// block's offset, so that order is load-bearing.
    pub fn decode(
        &self,
        ctx: &GpuContext,
        packed: &[u8],
        infos: &[BlockInfo],
        out_len: usize,
        coder: Coder,
    ) -> (Vec<i32>, f64) {
        assert!(!infos.is_empty(), "nothing to decode");
        assert!(
            infos.iter().all(|i| i.width <= MAX_BLOCK_W),
            "code-block width must be <= {MAX_BLOCK_W}: the shader keeps two rows of neighbour \
             magnitudes per thread in workgroup memory, sized for it"
        );
        // Bytes are packed little-endian into u32 words; the shader unpacks them the same way.
        let mut words = vec![0u32; packed.len().div_ceil(4).max(1)];
        for (i, &b) in packed.iter().enumerate() {
            words[i >> 2] |= (b as u32) << ((i & 3) * 8);
        }

        let stream_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("abac_stream"),
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let info_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("abac_blocks"),
                contents: bytemuck::cast_slice(infos),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let params = Params {
            num_blocks: infos.len() as u32,
            _pad: [0; 3],
        };
        let param_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("abac_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("abac_out"),
            size: (out_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("abac_decode_bind"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: stream_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: info_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
            ],
        });

        let mut cmd = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("abac_decode_cmd"),
            });
        {
            let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("abac_decode_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines[coder as usize]);
            pass.set_bind_group(0, &bind, &[]);
            // workgroup_size(32) in the shader; one thread per block.
            pass.dispatch_workgroups((infos.len() as u32).div_ceil(32), 1, 1);
        }
        let t0 = std::time::Instant::now();
        ctx.queue.submit(Some(cmd.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
        let gpu_s = t0.elapsed().as_secs_f64();

        let raw = crate::gpu_util::read_buffer_u32(ctx, &out_buf, out_len);
        let out = raw.into_iter().map(|w| w as i32).collect();
        (out, gpu_s)
    }
}

/// Cut a plane into code-blocks, encode each on the CPU, decode them all on the GPU, and assert
/// the GPU reproduced the plane exactly. Returns `(packed_bytes, gpu_seconds)`.
#[allow(clippy::too_many_arguments)] // plane geometry plus the variant under test
pub fn verify_against_cpu(
    ctx: &GpuContext,
    decoder: &GpuAbacDecoder,
    plane: &[i32],
    stride: usize,
    height: usize,
    cb: usize,
    coder: Coder,
) -> (usize, f64) {
    let mut packed: Vec<u8> = Vec::new();
    let mut infos: Vec<BlockInfo> = Vec::new();

    let mut by = 0;
    while by < height {
        let bh = cb.min(height - by);
        let mut bx = 0;
        while bx < stride {
            let bw = cb.min(stride - bx);
            let mut blk = Vec::with_capacity(bw * bh);
            for y in 0..bh {
                let row = (by + y) * stride + bx;
                blk.extend_from_slice(&plane[row..row + bw]);
            }
            let bytes = coder.encode_block(&blk, bw);
            infos.push(BlockInfo::new(
                packed.len() as u32,
                bytes.len() as u32,
                (by * stride + bx) as u32,
                bw as u32,
                bh as u32,
                stride as u32,
            ));
            packed.extend_from_slice(&bytes);
            bx += cb;
        }
        by += cb;
    }

    // Group equal-sized blocks together. One thread decodes one block, and a Metal SIMD group
    // runs at its slowest lane, so mixing an 8x8 block with a 64x64 one in the same group wastes
    // most of the group. Sorting by area removes the geometric part of the divergence; what
    // remains is data-dependent and cannot be removed.
    infos.sort_by_key(|i| std::cmp::Reverse(i.width * i.height));

    let (got, gpu_s) = decoder.decode(ctx, &packed, &infos, plane.len(), coder);
    if got != plane {
        let at = got
            .iter()
            .zip(plane.iter())
            .position(|(a, b)| a != b)
            .unwrap();
        panic!(
            "GPU abac decode diverged from the CPU coder at coefficient {at} (row {}, col {}): \
             got {}, expected {}. The shader and abac.rs must agree bit-for-bit; a divergence \
             here decodes without error and reconstructs the wrong image. Coder: {coder:?}.",
            at / stride,
            at % stride,
            got[at],
            plane[at]
        );
    }
    (packed.len(), gpu_s)
}
