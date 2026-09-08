# BUG-25 — the minimal reproducer

> **Standing, 2026-09-08 — final: this is an upstream driver bug report, not GNC's blocker.**
> **BUG-25 is fixed** (`51a9ac6`, the defect-A commit) and GNC now codes inter frames on Vulkan on
> both an RTX 4000 Ada and Mesa lavapipe, with byte-identical output. This file still crashes the
> driver, and that is still worth reporting — a valid module should be compiled or rejected, never
> segfault the compiler — but the shape it isolates is naga's `buffer: Restrict` clamp, and
> `wgpu-hal`'s rule (`adapter.rs:1899`) gives `buffer: Unchecked` on any adapter reporting
> `robustBufferAccess2`, which both of these adapters do. **GNC never emits it.** The binding-number
> caveat below is dead — `minimal_repro_binding0.spvasm` still segfaults, so the file does isolate
> the clamp. See `docs/decisions/0029`.

`minimal_repro.spvasm` is 42 lines of SPIR-V that **segfaults NVIDIA's Vulkan driver at pipeline
creation** and passes `spirv-val`. It was produced by `spirv-reduce` from
`src/shaders/block_match_split.wgsl` as naga 24 compiles it under wgpu's options — 41 068 bytes
reduced to 524.

```bash
spirv-as docs/bug25/minimal_repro.spvasm -o /tmp/repro.spv
spirv-val /tmp/repro.spv                       # passes
cargo build --release --features bug25-spirv --example spirv_pipeline_probe
WGPU_BACKEND=vulkan ./target/release/examples/spirv_pipeline_probe /tmp/repro.spv main
# NVIDIA RTX 4000 Ada, driver 580.173.02: SIGSEGV (139)
```

## What it isolates

```spirv
        %398 = OpLabel
               OpSelectionMerge %1638 None
               OpBranchConditional %1897 %1553 %1554
       %1553 = OpLabel
       %1596 = OpArrayLength %uint %32 0     ; length of a runtime-sized storage array
       %1597 = OpISub %uint %1596 %uint_1    ; length - 1
               OpBranch %1639
       %1639 = OpLabel
               OpReturn                      ; returns instead of reaching the merge block
       %1638 = OpLabel                       ; %1638 is the merge, and nothing branches to it
               OpReturn
```

`OpArrayLength` + `OpISub %uint_1` is exactly naga's **`BoundsCheckPolicy::Restrict` check on a
storage buffer** — `min(i, arrayLength(buf) - 1)`. It sits in a branch of a selection construct
whose merge block that branch never reaches, because the branch returns.

## Which policy, measured

Emitting the same shader with one policy changed at a time (`examples/bug25_emit.rs`):

| policy | driver |
|---|---|
| `buffer: Restrict`, index Unchecked | **CRASH** |
| `index: Restrict`, buffer Unchecked | pipeline OK |
| everything Unchecked | pipeline OK |

So within this set of emitted modules it is the **buffer** policy, not array indexing.

**Corrected 2026-09-08.** This section used to continue "wgpu requests `buffer: Restrict` whenever
the adapter does not report `robustBufferAccess2`, which is why every GNC build hits it". The rule
is right and the conclusion is backwards: this adapter *does* report `robustBufferAccess2`
(`vulkaninfo`), so wgpu's own source says it requests `buffer: **Unchecked**` here, and no GNC build
should be hitting the clamp at all. The configuration wgpu should actually be shipping,
`caps_index_restrict`, is byte-identical to `index_restrict_only`
(`sha256 537e7329…`) — measured **pipeline OK** — and carries **0** `OpArrayLength` against **48**
in every configuration that crashed.

## What it is not

* **Not invalid SPIR-V.** `spirv-val` passes. A separate defect — naga 24 emitting an undeclared
  function-local temporary for a dynamically indexed `let`-array — was found and fixed in the same
  session, and fixing it did **not** stop the crash.
* **Not driver stack exhaustion.** Reproduces at `ulimit -s` 8 MB, 64 MB and unlimited.
* **Not NVIDIA-only, before reduction.** The full-size valid module also segfaults Mesa lavapipe.
  The reduced module does not, because the reduction's interestingness test only ran the default
  adapter — so this file is minimal *for NVIDIA*. Reduce again against lavapipe if a
  two-implementation reproducer is wanted.
* **Possibly not even GNC's crash.** `spirv-reduce` deleted bindings 0-3, leaving `Binding 4` as
  the module's only binding, and `spirv_pipeline_probe` passes `layout: None` so wgpu derives the
  descriptor set layout from the module — first binding 4. An NVIDIA report from August 2026 has
  `vkCreateComputePipeline` segfaulting inside the driver, with no validation output, **when the
  descriptor set layout's first binding is not 0**. Renumbering this file's `Binding 4` to `0` and
  re-probing tests whether the binding number matters. **Run 2026-09-08: it does not** —
  `minimal_repro_binding0.spvasm` (identical but for that one decoration) still segfaults, exit 139.
  So the descriptor-layout hypothesis is dead and this file does isolate the clamp.
* **Not fixed.** GNC still cannot run inter coding on Vulkan. `split_pipeline` is built lazily so
  the shader is only compiled when inter runs, which is what keeps intra, decode and CANARY-1
  alive.

## Where to take it

Both original candidates are now **measured dead** (`b5a909c`):

1. ~~Upgrade wgpu/naga~~ — **DEAD.** naga 30 under wgpu's identical options crashes NVIDIA *and*
   lavapipe. The earlier naga-30 module that built a pipeline came from the CLI with its own bounds
   defaults and moved two variables at once.
2. ~~Change the shader's control flow~~ — **DEAD.** Deleting the early `return` outright still
   crashes, so the returning branch in this file is an artefact of the reduction.

All three of the steps that were left here were run on 2026-09-08 and the item is closed; kept for
what each one settled:

1. **Probe `minimal_repro_binding0.spvasm`** — **exit 139**, still crashes:

   ```bash
   spirv-as docs/bug25/minimal_repro_binding0.spvasm -o /tmp/repro0.spv && spirv-val /tmp/repro0.spv
   WGPU_BACKEND=vulkan ./target/release/examples/spirv_pipeline_probe /tmp/repro0.spv main
   ```

   The binding number is irrelevant, so the file does isolate the clamp and the NVIDIA
   descriptor-layout report is not what we are hitting.
2. **Probe `caps_index_restrict.spv`** (the `buffer: Unchecked` module, `537e7329…`) — **exit 1,
   `OK`**, as predicted from it being byte-identical to `index_restrict_only`.
3. **The dump was never needed.** Running the real WGSL through wgpu answered it directly:
   `shader_probe block_match_split.wgsl` is **OK**, with and without `--trusted`, and
   `gnc encode-sequence` codes 1I + 2P on Vulkan. So the shipped module never carried the clamp,
   and the crash on record predated `51a9ac6`. **BUG-25 fixed, BUG-33 closed** — `docs/decisions/0029`.

For the upstream report, which is all this file is now for: defect A was
`gfx-rs/wgpu#7048` (closed by PR #7239) and is what actually broke GNC; `#6329` is the same failure
mode on AMD with valid SPIR-V — where adding `OpLine` debug instructions makes it disappear, a
discriminator `debug_on_restrict` and `wgpu_native_debug` are emitted for and nobody has probed —
and `OpArrayLength` sourced from a StorageBuffer variable has segfaulted Intel's compiler before
(Mesa release notes). Three vendors, one instruction.

It is also a legitimate driver bug report: a valid module should be rejected or compiled, never
segfault the compiler. Two independent implementations crashing says the shape is unusual, not
that either driver is uniquely broken.
