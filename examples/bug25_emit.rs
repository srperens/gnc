//! BUG-25: emit SPIR-V for a WGSL shader under **wgpu's** backend options, one knob at a time.
//!
//! The bug's shape changed when the raw SPIR-V from `naga` (the CLI) turned out **not** to crash
//! the driver, while the same shader compiled through wgpu does. Entropy coding aside, that means
//! the defect is not in the WGSL and not in "naga's output" in general — it is in the module wgpu
//! asks naga for, which differs from the CLI's defaults in several ways at once
//! (`wgpu-hal/src/vulkan/adapter.rs`, `spv::Options`):
//!
//!   * `lang_version` — (1, 0) unless subgroup features are on, not the CLI's default
//!   * `bounds_check_policies.index` — `Restrict`
//!   * `bounds_check_policies.buffer` — `Unchecked` with `robustBufferAccess2`, else `Restrict`
//!   * `zero_initialize_workgroup_memory` — `Native` where
//!     `VK_KHR_zero_initialize_workgroup_memory` is present, otherwise `Polyfill`
//!   * `WriterFlags::ADJUST_COORDINATE_SPACE | LABEL_VARYINGS | FORCE_POINT_SIZE`
//!
//! This emits one module per named configuration so each knob can be tested on the box
//! separately. That turns "which of five differences is it" into five runs instead of a guess.
//!
//! **Which configuration is "wgpu's" depends on the adapter, and getting that wrong is what cost
//! BUG-25 a day.** `wgpu_native` assumes the adapter does *not* report `robustBufferAccess2`, so it
//! carries the `buffer: Restrict` clamp; `caps_index_restrict` assumes it *does*, and carries none.
//! The bench box reports it on both of its Vulkan implementations, which makes
//! `caps_index_restrict` the faithful one there — see BUG-33. `--wgpu-only` therefore emits **both**
//! rather than picking for you.
//!
//! Usage: `bug25_emit <shader.wgsl> <out_dir>` — writes `<out_dir>/<config>.spv` for every
//! configuration below and prints the list.

use naga::back::spv;
use naga::proc::{BoundsCheckPolicies, BoundsCheckPolicy};

/// The base capability set wgpu 24 hands naga on Vulkan (`wgpu-hal/src/vulkan/adapter.rs`),
/// before the feature-conditional additions, none of which GNC requests.
fn wgpu_caps() -> naga::FastHashSet<spv::Capability> {
    [
        spv::Capability::Shader,
        spv::Capability::Matrix,
        spv::Capability::Sampled1D,
        spv::Capability::Image1D,
        spv::Capability::ImageQuery,
        spv::Capability::DerivativeControl,
        spv::Capability::StorageImageExtendedFormats,
    ]
    .into_iter()
    .collect()
}

fn configs() -> Vec<(&'static str, spv::Options<'static>)> {
    // wgpu starts from `WriterFlags::empty()` and sets only these two for a non-Qualcomm adapter
    // with InstanceFlags::DEBUG off (`wgpu-hal/src/vulkan/adapter.rs`). It does **not** set
    // ADJUST_COORDINATE_SPACE, which an earlier version of this file wrongly included.
    let base_flags = spv::WriterFlags::LABEL_VARYINGS | spv::WriterFlags::FORCE_POINT_SIZE;

    let restrict = BoundsCheckPolicies {
        index: BoundsCheckPolicy::Restrict,
        buffer: BoundsCheckPolicy::Restrict,
        image_load: BoundsCheckPolicy::Restrict,
        binding_array: BoundsCheckPolicy::Unchecked,
    };
    let unchecked = BoundsCheckPolicies {
        index: BoundsCheckPolicy::Unchecked,
        buffer: BoundsCheckPolicy::Unchecked,
        image_load: BoundsCheckPolicy::Unchecked,
        binding_array: BoundsCheckPolicy::Unchecked,
    };

    use spv::ZeroInitializeWorkgroupMemoryMode as Z;
    let default_flags = spv::Options::default().flags;
    let default_bounds = spv::Options::default().bounds_check_policies;
    let default_zero = spv::Options::default().zero_initialize_workgroup_memory;
    let default_lang = spv::Options::default().lang_version;

    let mk = |lang, flags, bounds, zero| spv::Options {
        lang_version: lang,
        flags,
        capabilities: None,
        bounds_check_policies: bounds,
        zero_initialize_workgroup_memory: zero,
        binding_map: Default::default(),
        debug_info: None,
    };

    vec![
        // The control: naga's own defaults, which is what the CLI emits and what passed
        // `spirv-val` and did NOT crash the driver.
        ("naga_default", mk(default_lang, default_flags, default_bounds, default_zero)),
        // One knob at a time away from that control.
        ("zero_native", mk(default_lang, default_flags, default_bounds, Z::Native)),
        ("zero_none", mk(default_lang, default_flags, default_bounds, Z::None)),
        ("zero_polyfill", mk(default_lang, default_flags, default_bounds, Z::Polyfill)),
        ("bounds_restrict", mk(default_lang, default_flags, restrict, default_zero)),
        ("bounds_unchecked", mk(default_lang, default_flags, unchecked, default_zero)),
        ("lang_1_0", mk((1, 0), default_flags, default_bounds, default_zero)),
        ("lang_1_3", mk((1, 3), default_flags, default_bounds, default_zero)),
        ("flags_wgpu", mk(default_lang, base_flags, default_bounds, default_zero)),
        // wgpu's Vulkan combination, as reconstructed from wgpu-hal's adapter.rs.
        ("wgpu_native", mk((1, 0), base_flags, restrict, Z::Native)),
        ("wgpu_polyfill", mk((1, 0), base_flags, restrict, Z::Polyfill)),
        // `Options::default()` sets WriterFlags::DEBUG only under `debug_assertions`, so a
        // release build of naga emits a *different module* from a debug build. That makes the
        // presence of debug names a variable in its own right, and it is the only difference left
        // between this emitter and the naga CLI whose output was valid.
        ("debug_on", mk(default_lang, default_flags | spv::WriterFlags::DEBUG, default_bounds, default_zero)),
        ("debug_off", mk(default_lang, default_flags - spv::WriterFlags::DEBUG, default_bounds, default_zero)),
        ("debug_on_restrict", mk(default_lang, default_flags | spv::WriterFlags::DEBUG, restrict, default_zero)),
        ("wgpu_native_debug", mk((1, 0), base_flags | spv::WriterFlags::DEBUG, restrict, Z::Native)),
        // `Restrict` split by policy. The reduced reproducer is `OpArrayLength` + `OpISub`, which
        // is the *buffer* check (`min(i, arrayLength(buf) - 1)`), not the index check — so these
        // two separate a driver bug about storage-buffer bounds from one about array indexing.
        ("index_restrict_only", mk((1, 0), base_flags, BoundsCheckPolicies {
            index: BoundsCheckPolicy::Restrict,
            buffer: BoundsCheckPolicy::Unchecked,
            image_load: BoundsCheckPolicy::Unchecked,
            binding_array: BoundsCheckPolicy::Unchecked,
        }, Z::Native)),
        ("buffer_restrict_only", mk((1, 0), base_flags, BoundsCheckPolicies {
            index: BoundsCheckPolicy::Unchecked,
            buffer: BoundsCheckPolicy::Restrict,
            image_load: BoundsCheckPolicy::Unchecked,
            binding_array: BoundsCheckPolicy::Unchecked,
        }, Z::Native)),
        // The last untested difference from wgpu: it passes `capabilities: Some([...])`, which
        // constrains what the writer may emit and can change lowering. Everything above passes
        // `None`. This adapter reports `robustBufferAccess2 = true`, so wgpu should be asking for
        // `buffer: Unchecked` — and that configuration does *not* crash — yet the real WGSL path
        // does. One of those two things is wrong, and this is the remaining candidate.
        ("caps_index_restrict", spv::Options {
            lang_version: (1, 0),
            flags: base_flags,
            capabilities: Some(wgpu_caps()),
            bounds_check_policies: BoundsCheckPolicies {
                index: BoundsCheckPolicy::Restrict,
                buffer: BoundsCheckPolicy::Unchecked,
                image_load: BoundsCheckPolicy::Unchecked,
                binding_array: BoundsCheckPolicy::Unchecked,
            },
            zero_initialize_workgroup_memory: Z::Native,
            binding_map: Default::default(),
            debug_info: None,
        }),
        ("caps_wgpu_native", spv::Options {
            lang_version: (1, 0),
            flags: base_flags,
            capabilities: Some(wgpu_caps()),
            bounds_check_policies: restrict,
            zero_initialize_workgroup_memory: Z::Native,
            binding_map: Default::default(),
            debug_info: None,
        }),
    ]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let shader = args.next().expect("usage: bug25_emit <shader.wgsl> <out_dir> [--wgpu-only]");
    let out_dir = args.next().expect("usage: bug25_emit <shader.wgsl> <out_dir> [--wgpu-only]");
    // `--wgpu-only` emits wgpu's two possible Vulkan configurations, named after the shader, so the
    // whole tree can be swept in one pass instead of one directory of variants per shader. Both,
    // not one: which of them wgpu ships depends on whether the adapter reports
    // `robustBufferAccess2`, and assuming the answer is how BUG-25's cause came to be misattributed.
    let wgpu_only = args.any(|a| a == "--wgpu-only");
    std::fs::create_dir_all(&out_dir).expect("create out_dir");

    let stem = std::path::Path::new(&shader)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "shader".to_string());

    let src = std::fs::read_to_string(&shader).expect("read shader");
    let module = naga::front::wgsl::parse_str(&src).expect("wgsl parse");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("naga validate");

    let selected: Vec<(String, spv::Options<'static>)> = if wgpu_only {
        // `<stem>` is the no-robustness2 module (with the clamp); `<stem>.robust2` is the one wgpu
        // ships on an adapter that reports the feature (without it).
        configs()
            .into_iter()
            .filter_map(|(n, o)| match n {
                "wgpu_native" => Some((stem.clone(), o)),
                "caps_index_restrict" => Some((format!("{stem}.robust2"), o)),
                _ => None,
            })
            .collect()
    } else {
        configs()
            .into_iter()
            .map(|(n, o)| (n.to_string(), o))
            .collect()
    };

    for (name, opts) in selected {
        match spv::write_vec(&module, &info, &opts, None) {
            Ok(words) => {
                let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                let path = format!("{out_dir}/{name}.spv");
                std::fs::write(&path, &bytes).expect("write spv");
                println!("{name:<18} {:>7} bytes  {path}", bytes.len());
            }
            // A configuration naga refuses is a result, not a failure: it means wgpu could not
            // have asked for that combination either.
            Err(e) => println!("{name:<18} REFUSED BY NAGA: {e}"),
        }
    }
}
