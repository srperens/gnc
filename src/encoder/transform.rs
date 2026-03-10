use bytemuck::{Pod, Zeroable};
use wgpu;

use crate::{FrameInfo, GpuContext, WaveletType};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TransformParams {
    width: u32,
    height: u32,
    tile_size: u32,
    direction: u32,
    pass_mode: u32,
    tiles_x: u32,
    region_size: u32,
    /// Pixels of overlap per side for the forward transform (encoder-only).
    /// At level 0: physical region = tile_size + 2*overlap. Higher levels: 0.
    /// The shader reads an extended region but writes only the central tile_size coefficients.
    overlap: u32,
}

/// Maximum wavelet decomposition levels supported
const MAX_LEVELS: usize = 8;
/// Maximum number of planes encoded in one command encoder (Y + Co + Cg = 3)
const MAX_PLANES: usize = 3;
/// Slots per plane: MAX_LEVELS * 2 (row+col pass per level)
const SLOTS_PER_PLANE: usize = MAX_LEVELS * 2;
/// Total slots: 2 directions (fwd+inv) × MAX_PLANES × SLOTS_PER_PLANE
const MAX_PARAM_SLOTS: usize = 2 * MAX_PLANES * SLOTS_PER_PLANE;
/// GPU uniform buffer offset alignment (256 bytes covers all backends)
const UBO_ALIGN: usize = 256;

pub struct WaveletTransform {
    pipeline_53: wgpu::ComputePipeline,
    pipeline_97: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Pre-allocated params buffer with dynamic offsets.
    /// Holds all param configurations for forward+inverse, avoiding per-dispatch allocation.
    dyn_params_buf: wgpu::Buffer,
}

impl WaveletTransform {
    pub fn new(ctx: &GpuContext) -> Self {
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("transform_bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: true,
                                min_binding_size: wgpu::BufferSize::new(
                                    std::mem::size_of::<TransformParams>() as u64,
                                ),
                            },
                            count: None,
                        },
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
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("transform_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // LeGall 5/3 pipeline
        let shader_53 = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("transform_53"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/transform_53.wgsl").into()),
            });

        let pipeline_53 = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("transform_pipeline_53"),
                layout: Some(&pipeline_layout),
                module: &shader_53,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // CDF 9/7 pipeline
        let shader_97 = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("transform_97"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/transform_97.wgsl").into(),
                ),
            });

        let pipeline_97 = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("transform_pipeline_97"),
                layout: Some(&pipeline_layout),
                module: &shader_97,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Pre-allocate dynamic params buffer
        let dyn_params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transform_dyn_params"),
            size: (MAX_PARAM_SLOTS * UBO_ALIGN) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline_53,
            pipeline_97,
            bind_group_layout,
            dyn_params_buf,
        }
    }

    /// Write params at a given slot in the dynamic params buffer.
    fn write_params_slot(&self, ctx: &GpuContext, slot: usize, params: &TransformParams) {
        let offset = (slot * UBO_ALIGN) as u64;
        ctx.queue
            .write_buffer(&self.dyn_params_buf, offset, bytemuck::bytes_of(params));
    }

    /// Create a bind group with dynamic offset support for the params buffer.
    fn create_bind_group(
        &self,
        ctx: &GpuContext,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transform_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.dyn_params_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(
                            std::mem::size_of::<TransformParams>() as u64,
                        ),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Run multi-level 2D forward wavelet transform.
    /// Uses dynamic uniform buffer to avoid per-dispatch buffer allocation.
    ///
    /// `plane_idx` (0, 1 or 2) selects an independent set of param slots so that
    /// Y, Co and Cg transforms can share the same command encoder without their
    /// params overwriting each other.
    ///
    /// `overlap` is the per-side pixel overlap for tile-boundary artifact reduction.
    /// When `overlap > 0`, level-0 reads `tile_size + 2*overlap` pixels per row/column
    /// but writes back only the central `tile_size` coefficients. Higher levels use overlap=0.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        temp_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        info: &FrameInfo,
        levels: u32,
        wavelet_type: WaveletType,
        plane_idx: usize,
        overlap: u32,
    ) {
        // Each plane gets its own SLOTS_PER_PLANE-wide range in the forward half.
        let plane_slot_base = plane_idx * SLOTS_PER_PLANE;
        // Pre-fill all param slots for this forward transform.
        // At level 0: region_size = tile_size + 2*overlap (physical), overlap = overlap.
        // At level 1+: region_size = tile_size>>level (standard), overlap = 0.
        let mut region = info.tile_size;
        for level in 0..levels as usize {
            let row_slot = plane_slot_base + level * 2;
            let col_slot = plane_slot_base + level * 2 + 1;
            let level_overlap = if level == 0 { overlap } else { 0 };
            // physical_region = standard region + 2*overlap; the shader reads this many
            // elements but writes back only the central `tile_size` (at level 0) or `region`
            // (at level 1+) coefficients.
            let physical_region = region + 2 * level_overlap;
            let params_row = TransformParams {
                width: info.padded_width(),
                height: info.padded_height(),
                tile_size: info.tile_size,
                direction: 0,
                pass_mode: 0,
                tiles_x: info.tiles_x(),
                region_size: physical_region,
                overlap: level_overlap,
            };
            let params_col = TransformParams {
                width: info.padded_width(),
                height: info.padded_height(),
                tile_size: info.tile_size,
                direction: 0,
                pass_mode: 1,
                tiles_x: info.tiles_x(),
                region_size: physical_region,
                overlap: level_overlap,
            };
            self.write_params_slot(ctx, row_slot, &params_row);
            self.write_params_slot(ctx, col_slot, &params_col);
            region /= 2;
        }

        // Create bind groups for the 3 unique (input, output) combos:
        // BG_A: (input_buf, temp_buf) — level 0 row pass only
        // BG_B: (temp_buf, output_buf) — all column passes
        // BG_C: (output_buf, temp_buf) — row passes for levels 1+
        let bg_a = self.create_bind_group(ctx, input_buf, temp_buf);
        let bg_b = self.create_bind_group(ctx, temp_buf, output_buf);
        let bg_c = self.create_bind_group(ctx, output_buf, temp_buf);

        let pipeline = match wavelet_type {
            WaveletType::LeGall53 => &self.pipeline_53,
            WaveletType::CDF97 => &self.pipeline_97,
        };

        let total_tiles = info.total_tiles();
        region = info.tile_size;

        for level in 0..levels as usize {
            let row_offset = ((plane_slot_base + level * 2) * UBO_ALIGN) as u32;
            let col_offset = ((plane_slot_base + level * 2 + 1) * UBO_ALIGN) as u32;

            let bg_row = if level == 0 { &bg_a } else { &bg_c };

            // Row pass
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("transform_fwd_row"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, bg_row, &[row_offset]);
                cpass.dispatch_workgroups(region, total_tiles, 1);
            }

            // Column pass
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("transform_fwd_col"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bg_b, &[col_offset]);
                cpass.dispatch_workgroups(region, total_tiles, 1);
            }

            region /= 2;
        }
    }

    /// Run multi-level 2D inverse wavelet transform.
    /// Uses dynamic uniform buffer to avoid per-dispatch buffer allocation.
    ///
    /// `plane_idx` (0, 1 or 2) selects an independent set of param slots so that
    /// Y, Co and Cg transforms can share the same command encoder without their
    /// params overwriting each other.
    #[allow(clippy::too_many_arguments)]
    pub fn inverse(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        temp_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        info: &FrameInfo,
        levels: u32,
        wavelet_type: WaveletType,
        plane_idx: usize,
    ) {
        let buf_size = (info.padded_width() * info.padded_height()) as u64 * 4;

        // Copy input_buf -> output_buf first so we have all subbands in output_buf.
        encoder.copy_buffer_to_buffer(input_buf, 0, output_buf, 0, buf_size);

        // Inverse slots occupy the second half of the param buffer (after forward slots).
        // Each plane gets its own SLOTS_PER_PLANE-wide range within the inverse half.
        let base_slot = MAX_PLANES * SLOTS_PER_PLANE + plane_idx * SLOTS_PER_PLANE;
        let min_region = info.tile_size >> (levels - 1);
        let mut region = min_region;
        for level in 0..levels as usize {
            let col_slot = base_slot + level * 2;
            let row_slot = base_slot + level * 2 + 1;
            let params_col = TransformParams {
                width: info.padded_width(),
                height: info.padded_height(),
                tile_size: info.tile_size,
                direction: 1,
                pass_mode: 1,
                tiles_x: info.tiles_x(),
                region_size: region,
                overlap: 0,
            };
            let params_row = TransformParams {
                width: info.padded_width(),
                height: info.padded_height(),
                tile_size: info.tile_size,
                direction: 1,
                pass_mode: 0,
                tiles_x: info.tiles_x(),
                region_size: region,
                overlap: 0,
            };
            self.write_params_slot(ctx, col_slot, &params_col);
            self.write_params_slot(ctx, row_slot, &params_row);
            region *= 2;
        }

        // Bind groups for inverse:
        // BG_D: (output_buf, temp_buf) — all inverse column passes
        // BG_E: (temp_buf, output_buf) — all inverse row passes
        let bg_d = self.create_bind_group(ctx, output_buf, temp_buf);
        let bg_e = self.create_bind_group(ctx, temp_buf, output_buf);

        let pipeline = match wavelet_type {
            WaveletType::LeGall53 => &self.pipeline_53,
            WaveletType::CDF97 => &self.pipeline_97,
        };

        let total_tiles = info.total_tiles();
        region = min_region;

        for level in 0..levels as usize {
            let col_offset = ((base_slot + level * 2) * UBO_ALIGN) as u32;
            let row_offset = ((base_slot + level * 2 + 1) * UBO_ALIGN) as u32;
            // Note: base_slot already includes the inverse-half offset and plane_idx offset.

            // Inverse column pass: output_buf -> temp_buf
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("transform_inv_col"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bg_d, &[col_offset]);
                cpass.dispatch_workgroups(region, total_tiles, 1);
            }

            // Inverse row pass: temp_buf -> output_buf
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("transform_inv_row"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bg_e, &[row_offset]);
                cpass.dispatch_workgroups(region, total_tiles, 1);
            }

            region *= 2;
        }
    }
}
