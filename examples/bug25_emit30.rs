//! BUG-25: emit the same shader through **naga 30** under **wgpu 24's** SPIR-V options.
//!
//! The decisive experiment for "would upgrading wgpu fix this". The naga-30 module that built a
//! pipeline earlier came from the CLI, which uses its own bounds-check defaults — and the crash is
//! caused by `BoundsCheckPolicy::Restrict` on *buffers*, which wgpu requests and the CLI does not.
//! So that run compared two things at once and settled nothing. This holds the options fixed and
//! moves only the compiler version.
//!
//! Usage: `bug25_emit30 <shader.wgsl> <out.spv>`

use naga::back::spv;
use naga::proc::{BoundsCheckPolicies, BoundsCheckPolicy};
use naga30 as naga;

fn main() {
    let mut args = std::env::args().skip(1);
    let shader = args
        .next()
        .expect("usage: bug25_emit30 <shader.wgsl> <out.spv>");
    let out = args
        .next()
        .expect("usage: bug25_emit30 <shader.wgsl> <out.spv>");

    let src = std::fs::read_to_string(&shader).expect("read shader");
    let module = naga::front::wgsl::parse_str(&src).expect("wgsl parse");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("naga validate");

    // wgpu 24's Vulkan options, verbatim from wgpu-hal/src/vulkan/adapter.rs: the combination that
    // crashes under naga 24. `buffer: Restrict` is the one that matters.
    let opts = spv::Options {
        lang_version: (1, 0),
        flags: spv::WriterFlags::ADJUST_COORDINATE_SPACE
            | spv::WriterFlags::LABEL_VARYINGS
            | spv::WriterFlags::FORCE_POINT_SIZE,
        capabilities: None,
        bounds_check_policies: BoundsCheckPolicies {
            index: BoundsCheckPolicy::Restrict,
            buffer: BoundsCheckPolicy::Restrict,
            image_load: BoundsCheckPolicy::Restrict,
            binding_array: BoundsCheckPolicy::Unchecked,
        },
        zero_initialize_workgroup_memory: spv::ZeroInitializeWorkgroupMemoryMode::Native,
        debug_info: None,
        ..Default::default()
    };

    let words = spv::write_vec(&module, &info, &opts, None).expect("spv write");
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&out, &bytes).expect("write spv");
    println!("naga30 {} -> {} ({} bytes)", shader, out, bytes.len());
}
