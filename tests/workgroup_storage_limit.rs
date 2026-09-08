//! Every shader's workgroup memory must fit the limit the device is created with (BUG-31).
//!
//! This is the check the stack does not perform. `max_compute_workgroup_storage_size` appears
//! nowhere in `wgpu-core`'s validation or device code: the limit is negotiated with the adapter
//! and reported back, never compared against a shader at pipeline creation. Native Metal
//! therefore honours the 32 KB the hardware has, and the 16 KB GNC asked for is a number nobody
//! enforces — so a shader can sit 2 KB over budget for months and every test passes.
//!
//! A conformant WebGPU implementation *does* enforce it. 16384 is the spec's guaranteed minimum,
//! not an accident of wgpu's defaults, and validation against it is a spec rule. So the failure
//! this test describes is a browser failure, and the reason to assert it here rather than in a
//! browser is that a browser which happens not to validate would not make the shader conformant
//! — it would only hide the defect behind one implementation's leniency.
//!
//! The budget is read from `wgpu::Limits::default()` rather than written as 16384, because that
//! is literally what `GpuContext` requests (`src/lib.rs`, which overrides only
//! `max_storage_buffers_per_shader_stage`). If someone raises the request, this test follows.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Sum of `var<workgroup>` sizes for the globals a given entry point actually uses.
///
/// Per entry point, not per module: two entry points in one file may use different arrays, and
/// charging a pipeline for memory its entry point never touches would report defects that are
/// not there.
fn workgroup_bytes_per_entry_point(src: &str, path: &Path) -> BTreeMap<String, u32> {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("{}: WGSL parse failed: {e:?}", path.display()));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}: WGSL validation failed: {e:?}", path.display()));

    let ctx = module.to_ctx();
    let mut out = BTreeMap::new();
    for (i, ep) in module.entry_points.iter().enumerate() {
        if ep.stage != naga::ShaderStage::Compute {
            continue;
        }
        let ep_info = info.get_entry_point(i);
        let mut bytes = 0u32;
        for (handle, gv) in module.global_variables.iter() {
            if gv.space != naga::AddressSpace::WorkGroup {
                continue;
            }
            // An entry point that never reads or writes the global does not allocate it.
            if ep_info[handle].is_empty() {
                continue;
            }
            bytes += module.types[gv.ty].inner.size(ctx);
        }
        out.insert(ep.name.clone(), bytes);
    }
    out
}

/// Entry points that are still over budget, with their exact size. Filed as **BUG-35**.
///
/// Exact sizes rather than a list of names, so this record cannot rot quietly: an offender that
/// grows fails, one that shrinks fails, and one that is *fixed* fails until its row is deleted.
/// Closing BUG-35 means this array is empty and the `known` machinery below can go with it.
///
/// `quantize_histogram_fused.wgsl` is the one that is not opt-in — `EncoderPipeline::new`
/// constructs it unconditionally (`src/encoder/pipeline.rs:727`), so it is the default encode
/// path, not a parked backend. The four `rans_*` entry points belong to a backend GOALS §5b parks
/// as never-default, which is why they are less urgent and not less real.
const KNOWN_OVER_BUDGET: &[(&str, &str, u32)] = &[
    ("quantize_histogram_fused.wgsl", "main", 23800),
    ("rans_encode.wgsl", "main", 16388),
    ("rans_histogram.wgsl", "main", 23752),
    ("rans_normalize.wgsl", "main", 18460),
    ("rans_normalize_encode_fused.wgsl", "main", 33816),
];

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

#[test]
fn every_compute_entry_point_fits_the_requested_workgroup_budget() {
    let budget = wgpu::Limits::default().max_compute_workgroup_storage_size;
    let mut problems: Vec<String> = Vec::new();
    let mut seen_known: Vec<(&str, &str, u32)> = Vec::new();
    let mut checked = 0usize;
    let mut worst = (0u32, String::new());
    let mut notable: Vec<(String, String, u32)> = Vec::new();

    for path in shader_paths() {
        let src = std::fs::read_to_string(&path).expect("read shader");
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        for (ep, bytes) in workgroup_bytes_per_entry_point(&src, &path) {
            checked += 1;
            if bytes > worst.0 {
                worst = (bytes, format!("{name}:{ep}"));
            }
            if bytes >= 1024 {
                notable.push((name.clone(), ep.clone(), bytes));
            }
            let known = KNOWN_OVER_BUDGET
                .iter()
                .find(|(f, e, _)| *f == name && *e == ep);
            match (bytes > budget, known) {
                // Clean, and not on the BUG-35 list. The intended state for every shader.
                (false, None) => {}
                // Over budget and nobody recorded it: this is the case the test exists for.
                (true, None) => problems.push(format!(
                    "  NEW: {name}:{ep} declares {bytes} B, budget is {budget} B (over by {} B)",
                    bytes - budget
                )),
                // Recorded, but the number moved. Either direction is worth a failure: the record
                // is evidence in BUG-35 and a stale number is worse than no number.
                (true, Some((_, _, was))) if *was != bytes => problems.push(format!(
                    "  CHANGED: {name}:{ep} is {bytes} B, BUG-35 records {was} B"
                )),
                (true, Some(k)) => seen_known.push(*k),
                // Fixed. Delete the row; if it was the last one, close BUG-35.
                (false, Some((_, _, was))) => problems.push(format!(
                    "  FIXED: {name}:{ep} is {bytes} B and now fits (BUG-35 recorded {was} B) — \
                     remove it from KNOWN_OVER_BUDGET"
                )),
            }
        }
    }

    // A run that inspected nothing would otherwise pass silently.
    assert!(
        checked >= 60,
        "only {checked} compute entry points inspected — the sweep is not running"
    );
    for k in KNOWN_OVER_BUDGET {
        assert!(
            seen_known.contains(k),
            "BUG-35 records {}:{} but the sweep never saw it — renamed or deleted?",
            k.0,
            k.1
        );
    }
    println!(
        "{checked} compute entry points checked against a {budget} B budget; \
         {} still over (BUG-35), largest overall {} at {} B",
        KNOWN_OVER_BUDGET.len(),
        worst.1,
        worst.0
    );
    // The shaders that actually use workgroup memory, so a change in any of them is visible in
    // the test log rather than only when it crosses the budget.
    println!("entry points using >= 1 KiB:");
    for (name, ep, bytes) in &notable {
        println!("  {bytes:>6} B  {name}:{ep}");
    }

    assert!(
        problems.is_empty(),
        "workgroup budget: {} problem(s). Native Metal does not enforce this limit; a conformant \
         WebGPU implementation does, and pipelines are constructed unconditionally, so a \
         validation failure fails every decode of that kind rather than only the feature that \
         needs the shader.\n{}",
        problems.len(),
        problems.join("\n")
    );
}
