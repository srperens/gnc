#!/usr/bin/env python3
"""Apply temporal wavelet buffer caching changes atomically."""
import os

os.chdir('/Users/per/src/c2')

# 1. buffer_cache.rs
with open('src/encoder/buffer_cache.rs', 'r') as f:
    bc = f.read()

new_struct = r"""/// Cached GPU buffers for temporal wavelet GOP encoding.
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

        Self {
            padded_w,
            padded_h,
            group_size,
            raw_input_size,
            frame_bufs,
            snapshot,
            per_frame_input,
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

"""

marker = "impl CachedEncodeBuffers {\n    /// Allocate all buffers for the given padded resolution.\n    pub(super) fn new("
assert marker in bc, f"Could not find marker in buffer_cache.rs"
bc = bc.replace(marker, new_struct + marker)
with open('src/encoder/buffer_cache.rs', 'w') as f:
    f.write(bc)
print("buffer_cache.rs done")

# 2. pipeline.rs
with open('src/encoder/pipeline.rs', 'r') as f:
    pl = f.read()

pl = pl.replace(
    'use super::buffer_cache::CachedEncodeBuffers;',
    'use super::buffer_cache::{CachedEncodeBuffers, CachedTemporalWaveletBuffers};'
)

pl = pl.replace(
    '    pub(super) cached: Option<CachedEncodeBuffers>,\n}',
    '    pub(super) cached: Option<CachedEncodeBuffers>,\n    pub(super) tw_cached: Option<CachedTemporalWaveletBuffers>,\n}'
)

old_init = '            cached: None,\n        }\n    }\n\n    /// Ensure cached buffers exist and match the given resolution.'
new_init = r"""            cached: None,
            tw_cached: None,
        }
    }

    /// Ensure temporal wavelet buffers are cached and compatible.
    pub(super) fn ensure_tw_cached(
        &mut self,
        ctx: &GpuContext,
        padded_w: u32,
        padded_h: u32,
        group_size: usize,
        raw_input_size: u64,
    ) {
        let needs_alloc = match &self.tw_cached {
            Some(c) => !c.is_compatible(padded_w, padded_h, group_size, raw_input_size),
            None => true,
        };
        if needs_alloc {
            self.tw_cached = Some(CachedTemporalWaveletBuffers::new(
                ctx, padded_w, padded_h, group_size, raw_input_size,
            ));
        }
    }

    /// Ensure cached buffers exist and match the given resolution."""

assert old_init in pl, "Could not find old_init in pipeline.rs"
pl = pl.replace(old_init, new_init)
with open('src/encoder/pipeline.rs', 'w') as f:
    f.write(pl)
print("pipeline.rs done")

# 3. sequence.rs
with open('src/encoder/sequence.rs', 'r') as f:
    seq = f.read()

# Find the streaming path allocation block by searching for the exact pattern
# Use a simpler approach: find start and end markers
lines = seq.split('\n')

# Find the streaming path (encode_temporal_wavelet_gop_haar)
# Look for "// Allocate per-frame GPU buffers" after "ensure_cached" near line 1017
# Then replace up to "// Spatial wavelet per frame"

start_marker = "        // Allocate per-frame GPU buffers for wavelet coefficients: [frame][plane]"
end_marker = "        // Spatial wavelet per frame"

# Find which occurrence is in the streaming function (after encode_temporal_wavelet_gop_haar)
# The streaming version has 8-space indent, non-streaming has more
start_idx = None
end_idx = None
in_streaming = False
for i, line in enumerate(lines):
    if 'pub fn encode_temporal_wavelet_gop_haar(' in line:
        in_streaming = True
    if in_streaming and line.strip() == start_marker.strip() and line.startswith('        // Allocate'):
        # Check indent - streaming has exactly 8 spaces
        if line.startswith('        // Allocate per-frame') and not line.startswith('            '):
            start_idx = i
    if in_streaming and start_idx is not None and end_marker.strip() in line.strip() and line.startswith('        // Spatial') and not line.startswith('            '):
        end_idx = i
        break

assert start_idx is not None, f"Could not find start_marker in streaming path"
assert end_idx is not None, f"Could not find end_marker in streaming path"

new_block = [
    "        // Reuse cached temporal wavelet buffers across GOPs (avoids ~22ms per-GOP allocation)",
    "        let raw_input_size = std::mem::size_of_val(gop_frames[0]) as u64;",
    "        self.ensure_tw_cached(ctx, padded_w, padded_h, group_size, raw_input_size);",
    "        let tw = self.tw_cached.as_ref().unwrap();",
    "        let tw_frame_bufs = &tw.frame_bufs;",
    "        let tw_snapshot = &tw.snapshot;",
    "        let per_frame_input = &tw.per_frame_input;",
    "",
    "        // Upload all frames to per-frame GPU buffers (avoids write_buffer race)",
    "        for j in 0..group_size {",
    '            ctx.queue.write_buffer(&per_frame_input[j], 0, bytemuck::cast_slice(gop_frames[j]));',
    "        }",
    "",
]

lines = lines[:start_idx] + new_block + lines[end_idx:]
seq = '\n'.join(lines)

# Now remove timing instrumentation from the non-streaming path
# _tw_t0
seq = seq.replace('                    let _tw_t0 = std::time::Instant::now();\n                    // 1) Upload all GOP frames', '                    // 1) Upload all GOP frames')
seq = seq.replace('\n                    let _tw_t1_spatial = _tw_t0.elapsed();', '')
seq = seq.replace('\n                    let _tw_t2_temporal = _tw_t0.elapsed();', '')
seq = seq.replace('\n                    let _tw_t3_pre_encode = _tw_t0.elapsed();', '')
seq = seq.replace('\n                    let _tw_t4_low_done = _tw_t0.elapsed();', '')

# Remove the profiling block
old_prof = """                    if std::env::var("GNC_TW_PROFILE").is_ok() {
                        let total = _tw_t0.elapsed();
                        eprintln!(
                            "  TW_PROFILE: spatial={:.1}ms temporal={:.1}ms low_enc={:.1}ms high_enc={:.1}ms total={:.1}ms ({} frames)",
                            _tw_t1_spatial.as_secs_f64() * 1000.0,
                            (_tw_t2_temporal - _tw_t1_spatial).as_secs_f64() * 1000.0,
                            (_tw_t4_low_done - _tw_t3_pre_encode).as_secs_f64() * 1000.0,
                            (total - _tw_t4_low_done).as_secs_f64() * 1000.0,
                            total.as_secs_f64() * 1000.0,
                            group_size,
                        );
                    }
                    groups.push(TemporalGroup {"""
new_prof = """                    groups.push(TemporalGroup {"""

if old_prof in seq:
    seq = seq.replace(old_prof, new_prof)
    print("Removed non-streaming profiling block")
else:
    print("WARNING: Could not find non-streaming profiling block")

with open('src/encoder/sequence.rs', 'w') as f:
    f.write(seq)
print("sequence.rs done")
print("All files updated successfully")
