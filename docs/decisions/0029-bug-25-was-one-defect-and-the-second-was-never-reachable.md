# 0029 — BUG-25 was one defect, and the second one was never reachable from GNC

**Date:** 2026-09-08
**Status:** accepted
**Supersedes:** the "defect B" half of BUG-25, and the whole of BUG-33

## Decision

**BUG-25 is fixed, and it was fixed on 2026-09-08 by `51a9ac6` — the defect-A commit.** GNC now
encodes and decodes inter frames on Vulkan. The second defect that session went on to characterise
is a real driver bug, but it lives in a bounds-check configuration wgpu never asks for on either of
the adapters it was measured against, so GNC does not reach it. **BUG-33, which existed only to
explain why wgpu would ask for that configuration, is closed: it does not.**

What is reversed, explicitly, because it was written down as established:

* ~~"The valid module still segfaults. Defect B is `BoundsCheckPolicy::Restrict` on buffers."~~
* ~~"`buffer: Unchecked` is the only proven fix."~~
* ~~"`block_match_split.wgsl` crashes three independent drivers, so P/B coding is unreachable on
  any of them."~~

## Why

`51a9ac6` fixed naga 24 emitting an undeclared function-local temporary for four
`let … = array<i32, 8>` tables indexed by a loop counter — **upstream `gfx-rs/wgpu#7048`, closed by
PR #7239**, found here independently and worked around by rewriting the 8-point diamond as a
`switch`. The module GNC shipped before that commit was **invalid SPIR-V**.

Every driver crash on record was measured on that invalid module:

| where | when | built from | contains `51a9ac6`? |
|---|---|---|---|
| NVIDIA RTX 4000 Ada, Linux (SIGSEGV) | 2026-09-07 | `07c01b1` | no |
| Mesa lavapipe, Linux (`Parent device is lost`) | 2026-09-07 | `07c01b1` | no |
| NVIDIA RTX 2000 Ada, Windows (`0xC0000005`) | 2026-09-08 | `f17bf1b` | **no** |
| Intel Arc Pro, Windows (`0xC0000409`) | 2026-09-08 | `f17bf1b` | **no** |

The last two are the ones that mattered to check, because they were *committed after* the fix and
read as independent confirmation. They are not: `f17bf1b` predates `51a9ac6`, and the shader differs
between them by 66 deleted lines. And lavapipe's symptom — `Parent device is lost` — is verbatim
what `#7198` reports for the *same* upstream defect on llvmpipe. **Four crashes, one cause.**

**Nobody re-ran the real path after the fix.** What was run instead was `bug25_emit`, which
reconstructs wgpu's options and emits a module per configuration, plus `spirv_pipeline_probe` on
those modules. The reconstruction hard-coded `buffer: Restrict`. wgpu-hal 24.0.4 asks for
`buffer: robust_buffer_access2 ? Unchecked : Restrict` (`vulkan/adapter.rs:1899`), reading the cap
from a **queried** `VK_EXT_robustness2` (`:1595`, `:1372`), and **both** Vulkan implementations on
the bench box report `robustBufferAccess2 = true`. So the crashing modules were never the shipped
ones, and the elimination argument that concluded otherwise rested on one stale datum: "the real
WGSL path crashes" — true when written, one commit out of date by the time it was used.

Measured on 2026-09-08, NVIDIA RTX 4000 Ada + Mesa lavapipe (LLVM 20.1.2), driver 580.173.02, at
`766196a`:

| test | result |
|---|---|
| `shader_probe` sweep, all shaders, WGSL through wgpu | **62 of 63 compile**; the one failure is `blit.wgsl`, which has no `@compute` entry point and so cannot make a compute pipeline — a harness limit, not a driver one |
| `shader_probe block_match_split.wgsl` | **OK** (was the crash) |
| `shader_probe block_match_split.wgsl --trusted` | **OK** — bounds checks off changes nothing, because there was no clamp to remove |
| `gnc encode-sequence`, 3 frames, 1I + 2P, NVIDIA | **OK** — 102244 / 30301 / 27563 bytes, 180.1 ms |
| `gnc decode-sequence` of that container, NVIDIA | **OK** — 3 frames, 28.6 ms |
| same encode on **lavapipe** | **OK** — byte-identical frame sizes, 1553.8 ms |
| `spirv_pipeline_probe` on an emitted `buffer: Restrict` module | **still SIGSEGV** — the driver bug is real |
| `spirv_pipeline_probe` on the emitted `buffer: Unchecked` module | **OK** |

Two Vulkan implementations sharing no compiler code produce byte-identical output, which is a
stronger correctness statement than "it did not crash".

## What was not chosen, and what it would have cost

* **A `[patch.crates-io]` pin on wgpu-hal, or a fork.** BUG-33 had this as one of its two possible
  outcomes. Cost avoided: a standing maintenance commitment to a patched dependency, for a bug GNC
  does not have.
* **`create_shader_module_trusted(…, ShaderRuntimeChecks { bounds_checks: false, … })` in the
  encoder.** This *is* wgpu's own supported knob and it needs no fork, so it was the cheap local
  fix — but it buys nothing here, and it would have cost an `unsafe` call inside a library that is
  `#![forbid(unsafe_code)]` (`src/lib.rs:1`). The flag survives where it belongs: as
  `shader_probe --trusted`, an example, which is what proved the clamp was absent.
* **Rewriting the shader's control flow.** Tried twice before this session and measured dead
  (`b5a909c`); the early `return` in the reduced module was an artefact of `spirv-reduce`.
* **Upgrading wgpu/naga.** Also measured dead for the crash (`b5a909c`) — but note it would have
  fixed defect A, since #7048 is closed upstream. Our hand-written `switch` duplicates that fix.
  Worth knowing the next time "upgrade wgpu" is priced: it was the right instinct for the defect
  that actually mattered.

## Consequences

* GOALS rule 4 and the README's *Portability, as measured* table both change: **Vulkan now runs
  intra *and* inter, on two independent implementations.** Vulkan is no longer a "headline defect".
* **Intel Arc Pro and Windows NVIDIA are not re-measured**, and this record does not claim they
  work. Their crash was on the invalid module, so the expectation is that they are fine — an
  expectation, not a measurement, and the README says so.
* `docs/bug25/` keeps its reproducer. A valid module that segfaults a driver's compiler is worth
  reporting whether or not we hit it, and `OpArrayLength` sourced from a StorageBuffer variable has
  segfaulted Intel's compiler before (Mesa release notes) — three vendors, one instruction. It is
  now labelled as an upstream report rather than as GNC's blocker.

## The rule this leaves behind

**When a fix lands, re-run the failing path before characterising what is left.** Two sessions
spent a day on a defect that a single `encode-sequence` would have retired, because the
investigation moved from the real path to a reconstruction at exactly the moment the real path
started working. `spirv-reduce`, `spirv-val`, four dead hypotheses and a 42-line reproducer are all
downstream of one unrepeated measurement.

The related trap, and this one is older than the item: **"reproduced on an independent driver" is
only independent if the builds are.** Both Windows crashes were read as confirmation from new
hardware when they were the same commit's invalid SPIR-V on new hardware. Recording the commit is
what made that recoverable — `f17bf1b` was in the log, and one `git merge-base --is-ancestor`
settled it.
