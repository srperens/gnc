//! One shader, one process: create a device and a single compute pipeline.
//! A driver crash therefore kills only the run that caused it, which turns
//! "which shader kills Vulkan" into a bisect instead of a guess.

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
    let path = std::env::args()
        .nth(1)
        .expect("usage: shader_probe <file.wgsl>");
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

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe_module"),
            source: wgpu::ShaderSource::Wgsl(src.as_str().into()),
        });
        let _pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe_pipeline"),
            layout: None,
            module: &module,
            entry_point: Some(&entry),
            compilation_options: Default::default(),
            cache: None,
        });
        println!("OK {path} entry={entry}");
    });
}
