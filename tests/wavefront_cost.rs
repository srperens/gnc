//! What does a serial dependency cost on this GPU?
//!
//! GNC has repeatedly declined tools that carry a per-pixel dependency — spatial prediction,
//! in-loop filtering, context-adaptive coding — partly on parallelism grounds, without ever
//! measuring the cost. LOSSLESS-1's compression gate passed at 10–26% but its decoder needs
//! MED prediction, which cannot produce pixel (x, y) before (x−1, y). That is the binding
//! question, and it generalises well beyond LOSSLESS-1.
//!
//! Two shaders, identical arithmetic, one difference: where the neighbours come from.
//!
//! * `independent` reads neighbours from a separate input buffer — no dependency, one thread
//!   per pixel, the shape GNC uses everywhere today.
//! * `wavefront` reads neighbours from its own output buffer, so pixel (x, y) genuinely waits
//!   for (x−1, y), (x, y−1) and (x−1, y−1). One workgroup per 256×256 tile, 256 threads,
//!   marching the 511 anti-diagonals with a storage barrier between each.
//!
//! Run: `cargo test --release --test wavefront_cost -- --ignored --nocapture`

use gnc::GpuContext;
use std::sync::OnceLock;
use std::time::Instant;

static GPU: OnceLock<GpuContext> = OnceLock::new();
fn gpu() -> &'static GpuContext {
    GPU.get_or_init(GpuContext::new)
}

/// 1080p 4:4:4 padded to whole 256px tiles: 8×5 tiles per plane, three planes.
const TILE: u32 = 256;
const TILES_X: u32 = 8;
const TILES_Y: u32 = 5;
const PLANES: u32 = 3;
const WIDTH: u32 = TILE * TILES_X;
const HEIGHT: u32 = TILE * TILES_Y;
const TILES_TOTAL: u32 = TILES_X * TILES_Y * PLANES;

const SHARED_MATH: &str = r#"
struct Params { width: u32, height: u32, tiles_x: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

// JPEG-LS / LOCO-I median predictor. Identical in both shaders.
fn med(a: f32, b: f32, c: f32) -> f32 {
    let mx = max(a, b);
    let mn = min(a, b);
    if (c >= mx) { return mn; }
    if (c <= mn) { return mx; }
    return a + b - c;
}
"#;

/// No dependency: neighbours come from `src`, so every pixel is independent.
fn independent_wgsl() -> String {
    format!(
        r#"{SHARED_MATH}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.width * params.height * 3u) {{ return; }}
    let plane_px = params.width * params.height;
    let local = idx % plane_px;
    let x = local % params.width;
    let y = local / params.width;
    var a = 0.0; var b = 0.0; var c = 0.0;
    if (x > 0u) {{ a = src[idx - 1u]; }}
    if (y > 0u) {{ b = src[idx - params.width]; }}
    if (x > 0u && y > 0u) {{ c = src[idx - params.width - 1u]; }}
    dst[idx] = src[idx] + med(a, b, c);
}}"#
    )
}

/// True dependency: neighbours come from `dst`, which this workgroup is still writing.
/// One workgroup per tile; 511 anti-diagonals, a barrier between each.
fn wavefront_wgsl() -> String {
    format!(
        r#"{SHARED_MATH}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let plane_px = params.width * params.height;
    let tiles_per_plane = params.tiles_x * (params.height / 256u);
    let plane = wg.x / tiles_per_plane;
    let t = wg.x % tiles_per_plane;
    let ox = (t % params.tiles_x) * 256u;
    let oy = (t / params.tiles_x) * 256u;
    let base = plane * plane_px;

    for (var d = 0u; d < 511u; d = d + 1u) {{
        var lo = 0u;
        if (d > 255u) {{ lo = d - 255u; }}
        let hi = min(d, 255u);
        var k = lid.x;
        loop {{
            if (k > hi - lo) {{ break; }}
            let lx = lo + k;
            let ly = d - lx;
            let idx = base + (oy + ly) * params.width + (ox + lx);
            var a = 0.0; var b = 0.0; var c = 0.0;
            // Neighbours read from dst — written by earlier diagonals of this same workgroup.
            if (lx > 0u) {{ a = dst[idx - 1u]; }}
            if (ly > 0u) {{ b = dst[idx - params.width]; }}
            if (lx > 0u && ly > 0u) {{ c = dst[idx - params.width - 1u]; }}
            dst[idx] = src[idx] + med(a, b, c);
            k = k + 256u;
        }}
        // storageBarrier(), not workgroupBarrier(): the dependency is through a storage
        // buffer, and workgroupBarrier only orders workgroup memory. Measuring with the weaker
        // barrier understates the cost.
        storageBarrier();
    }}
}}"#
    )
}

fn run(label: &str, wgsl: &str, workgroups: u32, iters: u32) -> f64 {
    let ctx = gpu();
    let d = &ctx.device;
    let px = (WIDTH * HEIGHT * PLANES) as u64;
    let bytes = px * 4;

    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let params: [u32; 4] = [WIDTH, HEIGHT, TILES_X, 0];
    let ubuf = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&ubuf, 0, bytemuck::cast_slice(&params));

    let mk = || {
        d.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let (src, dst) = (mk(), mk());
    ctx.queue.write_buffer(&src, 0, &vec![0u8; bytes as usize]);

    let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dst.as_entire_binding() },
        ],
    });

    // Warm up: first submit pays shader compilation and allocation.
    for _ in 0..2 {
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(workgroups, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
    }
    let _ = d.poll(wgpu::Maintain::Wait);

    let t0 = Instant::now();
    for _ in 0..iters {
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups(workgroups, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
    }
    let _ = d.poll(wgpu::Maintain::Wait);
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

#[test]
#[ignore = "benchmark; run explicitly with --ignored --nocapture"]
fn serial_dependency_cost() {
    let px = WIDTH * HEIGHT * PLANES;
    let iters = 20;

    let indep_wgs = px.div_ceil(256);
    let ms_indep = run("independent", &independent_wgsl(), indep_wgs, iters);
    let ms_wave = run("wavefront", &wavefront_wgsl(), TILES_TOTAL, iters);

    println!("\n  Serial-dependency cost, {WIDTH}x{HEIGHT} x{PLANES} planes ({px} px), {iters} iterations\n");
    println!("    independent (1 thread/px, no dependency) : {ms_indep:7.3} ms   {:6.1} fps", 1000.0 / ms_indep);
    println!("    wavefront   (511 diagonals, 1 wg/tile)   : {ms_wave:7.3} ms   {:6.1} fps", 1000.0 / ms_wave);
    println!("\n    wavefront costs {:.1}x the independent pass", ms_wave / ms_indep);
    println!("    per-frame budget at 50 fps is 20 ms; this pass alone uses {:.1}%\n",
             ms_wave / 20.0 * 100.0);
}
