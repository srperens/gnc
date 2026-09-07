//! BUG-25: create one compute pipeline from a raw **SPIR-V** module, in its own process.
//!
//! `shader_probe` does this from WGSL, which is enough to say *which shader* kills the driver but
//! not *which part of it*. A WGSL-level bisect can only delete whole statements, and four of its
//! truncated variants were rejected by naga and counted as passes — so it answers a question about
//! WGSL when the evidence (two unrelated drivers, `spirv-val` clean) points at naga's output.
//!
//! Taking SPIR-V directly makes the module the unit of reduction, which is what
//! `spirv-reduce` needs: it wants an *interestingness test* — a program that exits 0 when the
//! input still exhibits the bug. Here "interesting" means the driver dies, so the exit codes are
//! inverted relative to a normal probe:
//!
//!   * **exit 0** — the module still crashes the driver (interesting, keep reducing)
//!   * **exit 1** — the pipeline was created successfully (not interesting)
//!   * **exit 2** — wgpu rejected the module or the run failed for some other reason
//!
//! A crash kills the process with a signal, so the wrapper script, not this binary, converts
//! "died" into exit 0. See `scripts/bug25_interesting.sh`.
//!
//! Usage: `spirv_pipeline_probe <module.spv> [entry_point]` (entry point defaults to `main`).
//!
//! Build with `--features bug25-spirv`; without it this is a stub, so the default build, the wasm
//! target and the clippy gate are all unaffected by a diagnostic.
//!
//! The `unsafe` is unavoidable rather than convenient (CLAUDE.md, "No `unsafe`"): handing the
//! driver bytes wgpu did not generate is the entire purpose, and wgpu marks that unsafe precisely
//! because it can crash the driver — which here is the observation, not the accident.

#[cfg(not(feature = "bug25-spirv"))]
fn main() {
    eprintln!(
        "spirv_pipeline_probe needs the SPIR-V passthrough path:\n  \
         cargo run --release --features bug25-spirv --example spirv_pipeline_probe -- <module.spv>"
    );
    std::process::exit(2);
}

#[cfg(feature = "bug25-spirv")]
fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: spirv_pipeline_probe <module.spv> [entry_point]");
            std::process::exit(2);
        }
    };
    let entry = args.next().unwrap_or_else(|| "main".to_string());

    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {path}: {e}");
            std::process::exit(2);
        }
    };
    if bytes.len() % 4 != 0 || bytes.len() < 20 {
        eprintln!("{path}: not a SPIR-V module ({} bytes)", bytes.len());
        std::process::exit(2);
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if words[0] != 0x0723_0203 {
        eprintln!("{path}: bad SPIR-V magic {:#x}", words[0]);
        std::process::exit(2);
    }

    pollster::block_on(async move {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all()),
            ..Default::default()
        });
        let Some(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("no adapter");
            std::process::exit(2);
        };
        eprintln!("adapter: {}", adapter.get_info().name);

        // Same limits the real encoder asks for, so a module that fits there fits here.
        let Ok((device, _queue)) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("spirv_probe"),
                    required_features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 10,
                        ..wgpu::Limits::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
        else {
            eprintln!("no device");
            std::process::exit(2);
        };

        // Errors must not be swallowed: without this, a rejected module looks like a pass and
        // the reducer happily reduces to something that never reached the driver at all.
        device.on_uncaptured_error(Box::new(|e| {
            eprintln!("wgpu error: {e}");
            std::process::exit(2);
        }));

        // SAFETY: the module is exactly what naga produced for a shader in this repository and
        // `spirv-val` passes it. wgpu cannot check it, which is the point — the defect is that a
        // valid module kills two drivers, so anything that pre-validated it would hide the bug.
        let module = unsafe {
            device.create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
                label: Some("spirv_probe_module"),
                source: std::borrow::Cow::Owned(words),
            })
        };
        let _pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("spirv_probe_pipeline"),
            layout: None,
            module: &module,
            entry_point: Some(&entry),
            compilation_options: Default::default(),
            cache: None,
        });
        device.poll(wgpu::Maintain::Wait);
        println!("OK {path} entry={entry}");
        // Reached the end without the driver dying: NOT interesting to the reducer.
        std::process::exit(1);
    });
}
