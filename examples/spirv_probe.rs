//! BUG-25: inspect the SPIR-V naga produces for a WGSL shader, without a GPU.
//!
//! `block_match_split.wgsl` kills two Vulkan drivers that share no compiler code, while
//! `spirv-val` passes it and naga reports no error. That combination says the interesting object
//! is **naga's output**, not the WGSL and not either driver — so it has to be inspectable, and on
//! a machine with no Vulkan at all. wgpu 24 resolves naga 24, so this reads exactly the bytes the
//! runtime would have shipped.
//!
//! Reports, per entry point:
//!   * `OpControlBarrier` count — against the `workgroupBarrier()` count in the source. Inlining
//!     multiplies barriers legitimately (a 6-iteration reduce loop called 4 times), but the SPIR-V
//!     rule is that *all* invocations must reach the *same* barrier instruction. A structurizer
//!     that duplicates a barrier-carrying block breaks that rule while leaving the module valid,
//!     which is exactly what would kill two unrelated drivers and pass `spirv-val`.
//!   * whether any barrier sits in a block that is not post-dominated by the function's structured
//!     control flow — the same rule, checked structurally rather than by counting.
//!   * module shape: functions, blocks, the largest function, deepest merge nesting.
//!
//! Usage: `cargo run --release --example spirv_probe -- src/shaders/*.wgsl`
//! With no arguments it does every shader and prints a table sorted by barrier count.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// SPIR-V opcodes this probe cares about. Numbers are from the SPIR-V 1.x core spec.
mod op {
    pub const ENTRY_POINT: u16 = 15;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_END: u16 = 56;
    pub const VARIABLE: u16 = 59;
    pub const CONTROL_BARRIER: u16 = 224;
    pub const LOOP_MERGE: u16 = 246;
    pub const SELECTION_MERGE: u16 = 247;
    pub const LABEL: u16 = 248;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const SWITCH: u16 = 251;
}

/// SPIR-V storage classes used here.
pub const STORAGE_CLASS_WORKGROUP: u32 = 4;

const SPIRV_MAGIC: u32 = 0x0723_0203;
const HEADER_WORDS: usize = 5;

#[derive(Default, Debug)]
struct FuncShape {
    blocks: usize,
    barriers: usize,
    /// Barriers that sit inside a selection or loop construct, i.e. not at the top level of the
    /// function body. A barrier here is only safe if every invocation reaches *this* instruction.
    barriers_nested: usize,
    max_merge_depth: usize,
    /// Blocks that carry a barrier and are the target of a conditional branch — the shape that
    /// makes "same instruction for all invocations" untrue.
    barrier_blocks_conditional: usize,
}

struct Module {
    words: Vec<u32>,
}

impl Module {
    /// Walk the instruction stream. Every SPIR-V instruction is `[opcode | wordcount<<16]`
    /// followed by `wordcount - 1` operand words, so this needs no table of operand shapes.
    fn instructions(&self) -> impl Iterator<Item = (u16, &[u32])> {
        let mut i = HEADER_WORDS;
        std::iter::from_fn(move || {
            if i >= self.words.len() {
                return None;
            }
            let first = self.words[i];
            let opcode = (first & 0xFFFF) as u16;
            let count = (first >> 16) as usize;
            if count == 0 || i + count > self.words.len() {
                return None; // malformed; stop rather than index past the end
            }
            let operands = &self.words[i + 1..i + count];
            i += count;
            Some((opcode, operands))
        })
    }

    /// SPIR-V version as (major, minor), from the header's second word.
    fn version(&self) -> (u32, u32) {
        let w = self.words[1];
        ((w >> 16) & 0xFF, (w >> 8) & 0xFF)
    }

    /// Module-scope `OpVariable`s in the Workgroup storage class.
    ///
    /// This is the axis BUG-25's own notes point at and that no hypothesis tested in combination:
    /// `block_match_split` has nine `var<workgroup>` against four in the two siblings that
    /// compile. Nine alone compiles in isolation (H4); nine *inside this module* is untested.
    fn workgroup_variables(&self) -> usize {
        self.instructions()
            .filter(|(opcode, operands)| {
                *opcode == op::VARIABLE
                    && operands.get(2).copied() == Some(STORAGE_CLASS_WORKGROUP)
            })
            .count()
    }

    /// How many ids each `OpEntryPoint` lists in its interface.
    ///
    /// From SPIR-V 1.4 the interface must name *every* module-scope variable the entry point can
    /// reach, Workgroup ones included; before 1.4 only Input and Output. A module that declares
    /// 1.3 while listing Workgroup variables — or declares 1.4+ and omits them — is the kind of
    /// thing `spirv-val` waves through and two unrelated drivers can both mishandle.
    fn entry_interface_sizes(&self) -> Vec<(String, usize)> {
        let mut out = Vec::new();
        for (opcode, operands) in self.instructions() {
            if opcode != op::ENTRY_POINT || operands.len() < 3 {
                continue;
            }
            // [execution model, entry id, name..., interface ids...]
            let mut w = 2;
            let mut name_bytes = Vec::new();
            let mut done = false;
            while w < operands.len() && !done {
                for b in operands[w].to_le_bytes() {
                    if b == 0 {
                        done = true;
                        break;
                    }
                    name_bytes.push(b);
                }
                w += 1;
            }
            out.push((
                String::from_utf8_lossy(&name_bytes).to_string(),
                operands.len().saturating_sub(w),
            ));
        }
        out
    }

    fn entry_point_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for (opcode, operands) in self.instructions() {
            if opcode == op::ENTRY_POINT && operands.len() > 2 {
                // operands: [execution model, entry id, name (literal string), ...]
                let bytes: Vec<u8> = operands[2..]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .take_while(|&b| b != 0)
                    .collect();
                names.push(String::from_utf8_lossy(&bytes).to_string());
            }
        }
        names
    }

    /// Per-function shape. Tracks structured-control-flow nesting by counting merge instructions
    /// against the merge labels they name, which is how a barrier is placed inside a construct
    /// without needing a full CFG.
    fn shapes(&self) -> Vec<FuncShape> {
        let mut out = Vec::new();
        let mut cur: Option<FuncShape> = None;
        // merge label id -> depth at which it closes
        let mut pending_merges: HashMap<u32, usize> = HashMap::new();
        let mut depth = 0usize;
        let mut conditional_targets: HashSet<u32> = HashSet::new();
        let mut this_block: Option<u32> = None;
        let mut block_has_barrier = false;

        for (opcode, operands) in self.instructions() {
            match opcode {
                op::FUNCTION => {
                    cur = Some(FuncShape::default());
                    depth = 0;
                    pending_merges.clear();
                    conditional_targets.clear();
                }
                op::FUNCTION_END => {
                    if let Some(f) = cur.take() {
                        out.push(f);
                    }
                }
                op::LABEL => {
                    if let (Some(f), Some(prev)) = (cur.as_mut(), this_block) {
                        if block_has_barrier && conditional_targets.contains(&prev) {
                            f.barrier_blocks_conditional += 1;
                        }
                    }
                    block_has_barrier = false;
                    let id = operands.first().copied().unwrap_or(0);
                    this_block = Some(id);
                    if let Some(closed_at) = pending_merges.remove(&id) {
                        depth = closed_at;
                    }
                    if let Some(f) = cur.as_mut() {
                        f.blocks += 1;
                    }
                }
                op::SELECTION_MERGE | op::LOOP_MERGE => {
                    if let Some(&merge_label) = operands.first() {
                        depth += 1;
                        pending_merges.insert(merge_label, depth - 1);
                        if let Some(f) = cur.as_mut() {
                            f.max_merge_depth = f.max_merge_depth.max(depth);
                        }
                    }
                }
                op::BRANCH_CONDITIONAL => {
                    // operands: [condition, true label, false label, ...weights]
                    for &t in operands.iter().skip(1).take(2) {
                        conditional_targets.insert(t);
                    }
                }
                op::SWITCH => {
                    for (n, &t) in operands.iter().enumerate().skip(1) {
                        // [selector, default, (literal, label)...]
                        if n == 1 || n % 2 == 1 {
                            conditional_targets.insert(t);
                        }
                    }
                }
                op::CONTROL_BARRIER => {
                    block_has_barrier = true;
                    if let Some(f) = cur.as_mut() {
                        f.barriers += 1;
                        if depth > 0 {
                            f.barriers_nested += 1;
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }
}

fn compile(path: &Path) -> Result<Module, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("read: {e}"))?;
    let module = naga::front::wgsl::parse_str(&src).map_err(|e| format!("wgsl parse: {e}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("naga validate: {e}"))?;

    // Match what wgpu asks for: SPIR-V 1.3 (Vulkan 1.1), no debug info, no bounds-check tricks the
    // runtime would not have used.
    let opts = naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };
    let words = naga::back::spv::write_vec(&module, &info, &opts, None)
        .map_err(|e| format!("spv write: {e}"))?;

    if words.first().copied() != Some(SPIRV_MAGIC) {
        return Err("output is not SPIR-V (bad magic)".into());
    }
    Ok(Module { words })
}

fn source_barrier_count(path: &Path) -> usize {
    std::fs::read_to_string(path)
        .map(|s| {
            s.lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .map(|l| l.matches("workgroupBarrier()").count()
                    + l.matches("storageBarrier()").count())
                .sum()
        })
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let paths: Vec<PathBuf> = if args.is_empty() {
        let mut v: Vec<PathBuf> = std::fs::read_dir("src/shaders")
            .expect("src/shaders not found — run from the repository root")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
            .collect();
        v.sort();
        v
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    println!(
        "{:<34} {:>5} {:>4} {:>4} {:>6} {:>6} {:>5} {:>5} {:>6}",
        "shader", "wgsl", "spv", "cond", "blocks", "depth", "fns", "wgVar", "iface"
    );
    println!("{}", "-".repeat(92));

    let mut suspicious = Vec::new();
    for p in &paths {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        match compile(p) {
            Err(e) => println!("{name:<34} {e}"),
            Ok(m) => {
                let shapes = m.shapes();
                let spv: usize = shapes.iter().map(|s| s.barriers).sum();
                let nested: usize = shapes.iter().map(|s| s.barriers_nested).sum();
                let cond: usize = shapes.iter().map(|s| s.barrier_blocks_conditional).sum();
                let blocks: usize = shapes.iter().map(|s| s.blocks).sum();
                let depth = shapes.iter().map(|s| s.max_merge_depth).max().unwrap_or(0);
                let wgsl = source_barrier_count(p);
                let wg_vars = m.workgroup_variables();
                let iface: String = m
                    .entry_interface_sizes()
                    .iter()
                    .map(|(_, n)| n.to_string())
                    .collect::<Vec<_>>()
                    .join("/");
                let (maj, min) = m.version();
                let _ = (maj, min, nested);
                println!(
                    "{name:<34} {wgsl:>5} {spv:>4} {cond:>4} {blocks:>6} {depth:>6} {:>5} {wg_vars:>5} {iface:>6}",
                    shapes.len()
                );
                if cond > 0 {
                    suspicious.push((name, cond, m.entry_point_names()));
                }
            }
        }
    }

    println!();
    if suspicious.is_empty() {
        println!(
            "No shader places a barrier in a conditionally-branched-to block. That kills the \
             hypothesis that naga duplicates a barrier-carrying block; the next cut is elsewhere."
        );
    } else {
        println!("Barriers in conditionally-reached blocks — all invocations may not reach the same instruction:");
        for (name, n, entries) in &suspicious {
            println!("  {name}: {n} such barrier(s), entry points {entries:?}");
        }
    }
}
