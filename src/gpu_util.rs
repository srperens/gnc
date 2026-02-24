use wgpu;

use crate::GpuContext;

/// Grow a variable-size cached buffer if the required size exceeds capacity (2× growth).
pub fn ensure_var_buf(
    ctx: &GpuContext,
    buf: &mut wgpu::Buffer,
    cap: &mut u64,
    required: u64,
    label: &str,
    usage: wgpu::BufferUsages,
) {
    if required > *cap {
        let new_cap = (required * 2).max(4);
        *buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: new_cap,
            usage,
            mapped_at_creation: false,
        });
        *cap = new_cap;
    }
}

/// Read a GPU buffer back to CPU as Vec<f32>, using a staging buffer with copy + map.
pub fn read_buffer_f32(ctx: &GpuContext, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let size = (count * std::mem::size_of::<f32>()) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_read"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut cmd = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_to_staging"),
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
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}
