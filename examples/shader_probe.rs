//! One shader, one process: create a device and a single compute pipeline.
//! A driver crash therefore kills only the run that caused it, which turns
//! "which shader kills Vulkan" into a bisect instead of a guess.
//!
//! `--trusted` compiles the module with `ShaderRuntimeChecks { bounds_checks: false }`, which is
//! the *only* thing it changes: wgpu then hands naga all four bounds-check policies as
//! `Unchecked` (`wgpu-hal/src/vulkan/device.rs:1831`) instead of the adapter-derived pair. That
//! makes this the decisive test for BUG-33, and it runs the WGSL through wgpu exactly as the
//! encoder does rather than through a reconstruction:
//!
//!   * `--trusted` stops the crash → the shipped module **does** carry naga's clamp, so wgpu is
//!     choosing `buffer: Restrict` on an adapter that reports `robustBufferAccess2`, against its
//!     own rule. The question in BUG-33 is real and the fix is this call or an upstream one.
//!   * `--trusted` still crashes → the clamp was never the cause. Then `docs/bug25/minimal_repro.spvasm`
//!     and "`buffer: Unchecked` is the only proven fix" are both about some other defect.
//!
//! Usage: `shader_probe <file.wgsl> [--trusted]`.

fn entry_point_of(src: &str) -> String {
    // The fn named right after the first @compute attribute.
    let after = match src.split("@compute").nth(1) {
        Some(s) => s,
        None => return "main".to_string(),
    };
    let after_fn = match after.split("fn ").nth(1) {
        Some(s) => s,
        None => return "main".to_string(),
    };
    let name: String = after_fn
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        "main".to_string()
    } else {
        name
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let trusted = args.iter().any(|a| a == "--trusted");
    let path = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .expect("usage: shader_probe <file.wgsl> [--trusted]");
    let src = std::fs::read_to_string(&path).expect("read shader");
    let entry = entry_point_of(&src);

    pollster::block_on(async move {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all()),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("no adapter");
        eprintln!("adapter: {}", adapter.get_info().name);

        let (device, _queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("probe"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 10,
                        ..wgpu::Limits::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("no device");

        let desc = || wgpu::ShaderModuleDescriptor {
            label: Some("probe_module"),
            source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
        };
        let module = if trusted {
            // The canary: a run has to say which path it took, or the two runs are indistinguishable
            // in the log (CLAUDE.md, "No silent features").
            eprintln!("bounds_checks: false (ShaderRuntimeChecks)");
            // SAFETY: dropping the injected bounds checks means an out-of-bounds index in this
            // shader would be undefined behaviour rather than a clamp. Nothing is dispatched here
            // — the pipeline is created and the process exits — and on Vulkan the hardware clamps
            // anyway wherever `robustBufferAccess2` is enabled, which is why wgpu itself omits the
            // software checks on that path.
            unsafe {
                device.create_shader_module_trusted(
                    desc(),
                    wgpu::ShaderRuntimeChecks {
                        bounds_checks: false,
                        force_loop_bounding: true,
                    },
                )
            }
        } else {
            eprintln!("bounds_checks: default (adapter-derived)");
            device.create_shader_module(desc())
        };
        let _pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe_pipeline"),
            layout: None,
            module: &module,
            entry_point: Some(&entry),
            compilation_options: Default::default(),
            cache: None,
        });
        println!("OK {path} entry={entry} trusted={trusted}");
    });
}
