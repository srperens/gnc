//! Every field of the `Limits` GNC requests must equal `Limits::default()`, except overrides
//! that are named here (BUG-34).
//!
//! `GpuContext` used to ask for `max_storage_buffers_per_shader_stage: 10` as a silent extra
//! on top of the defaults. The adapter here offers 31, so `request_device` always succeeded
//! and CLAUDE.md's "we ask for wgpu's defaults, so the same shaders run under WebGPU" was
//! false for that row. A conformant implementation held to 8 would fail device creation.
//!
//! The workgroup-storage test (BUG-31) already follows `Limits::default()` for one field.
//! This file is the same check for the whole struct: PartialEq against an explicitly
//! constructed expected value, so a new override cannot land as a one-line struct update.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn shader_paths() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("src/shaders is readable")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
        .collect();
    v.sort();
    assert!(!v.is_empty(), "found no shaders in {}", dir.display());
    v
}

fn storage_buffers_per_entry_point(src: &str, path: &Path) -> BTreeMap<String, u32> {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("{}: WGSL parse failed: {e:?}", path.display()));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}: WGSL validation failed: {e:?}", path.display()));

    let mut out = BTreeMap::new();
    for (i, ep) in module.entry_points.iter().enumerate() {
        if ep.stage != naga::ShaderStage::Compute {
            continue;
        }
        let ep_info = info.get_entry_point(i);
        let mut n = 0u32;
        for (handle, gv) in module.global_variables.iter() {
            if !matches!(gv.space, naga::AddressSpace::Storage { .. }) {
                continue;
            }
            if ep_info[handle].is_empty() {
                continue;
            }
            n += 1;
        }
        out.insert(ep.name.clone(), n);
    }
    out
}

#[test]
fn wgpu_default_storage_buffers_is_still_eight() {
    // If wgpu or the spec raises the default to 9, the override is free to delete.
    assert_eq!(
        wgpu::Limits::default().max_storage_buffers_per_shader_stage,
        8,
        "wgpu::Limits::default() is no longer 8 storage buffers/stage — the BUG-34 override \
         may be unnecessary; delete it from required_limits() and this expected value"
    );
}

#[test]
fn required_limits_only_overrides_the_named_storage_buffer_row() {
    let expected = wgpu::Limits {
        max_storage_buffers_per_shader_stage: gnc::MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        ..Default::default()
    };
    assert_eq!(
        gnc::MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        9,
        "the named override moved; update docs/decisions/0047 and CLAUDE.md's portability table"
    );
    assert_eq!(
        gnc::required_limits(),
        expected,
        "required_limits() differs from Limits::default() by more than the named storage-buffer \
         override. Raising a request is a decision record, not a commit (BUG-34). Name the new \
         field here and in required_limits()."
    );
}

#[test]
fn no_compute_entry_point_exceeds_requested_storage_buffers() {
    let budget = gnc::required_limits().max_storage_buffers_per_shader_stage;
    let mut problems: Vec<String> = Vec::new();
    let mut checked = 0usize;
    let mut worst = (0u32, String::new());
    let mut notable: Vec<(String, String, u32)> = Vec::new();

    for path in shader_paths() {
        let src = std::fs::read_to_string(&path).expect("read shader");
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        for (ep, n) in storage_buffers_per_entry_point(&src, &path) {
            checked += 1;
            if n > worst.0 {
                worst = (n, format!("{name}:{ep}"));
            }
            if n >= 7 {
                notable.push((name.clone(), ep.clone(), n));
            }
            if n > budget {
                problems.push(format!(
                    "  {name}:{ep} uses {n} storage buffers, request is {budget}"
                ));
            }
        }
    }

    assert!(
        checked >= 60,
        "only {checked} compute entry points inspected — the sweep is not running"
    );
    println!(
        "[bug34] {checked} compute entry points; max {1} at {0} storage buffers; request {budget}",
        worst.0, worst.1
    );
    println!("entry points using >= 7 storage buffers:");
    for (name, ep, n) in &notable {
        println!("  {n}  {name}:{ep}");
    }

    assert_eq!(
        worst.0, 9,
        "the heaviest shader is {} at {} storage buffers; BUG-34 recorded block_match_bidir at 9, \
         which is why the request is 9 not 8. If this dropped to <=8, delete the override. If it \
         rose, the request is now wrong.",
        worst.1, worst.0
    );
    assert!(
        worst.1.starts_with("block_match_bidir.wgsl"),
        "the 9-buffer shader is {} — BUG-34's reason for the override was block_match_bidir.wgsl",
        worst.1
    );
    assert!(
        problems.is_empty(),
        "storage-buffer budget: {} problem(s).\n{}",
        problems.len(),
        problems.join("\n")
    );
}
