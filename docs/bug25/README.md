# BUG-25 — the minimal reproducer

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

So it is the **buffer** policy, not array indexing. wgpu requests `buffer: Restrict` whenever the
adapter does not report `robustBufferAccess2` (`wgpu-hal/src/vulkan/adapter.rs`), which is why
every GNC build hits it and no configuration of GNC's own avoids it.

## What it is not

* **Not invalid SPIR-V.** `spirv-val` passes. A separate defect — naga 24 emitting an undeclared
  function-local temporary for a dynamically indexed `let`-array — was found and fixed in the same
  session, and fixing it did **not** stop the crash.
* **Not driver stack exhaustion.** Reproduces at `ulimit -s` 8 MB, 64 MB and unlimited.
* **Not NVIDIA-only, before reduction.** The full-size valid module also segfaults Mesa lavapipe.
  The reduced module does not, because the reduction's interestingness test only ran the default
  adapter — so this file is minimal *for NVIDIA*. Reduce again against lavapipe if a
  two-implementation reproducer is wanted.
* **Not fixed.** GNC still cannot run inter coding on Vulkan. `split_pipeline` is built lazily so
  the shader is only compiled when inter runs, which is what keeps intra, decode and CANARY-1
  alive.

## Where to take it

Two candidate fixes, neither tried:

1. **Upgrade wgpu/naga.** naga 30 compiles this shader to SPIR-V that builds a pipeline fine,
   though not under a controlled bounds policy, so this is suggestive rather than measured.
2. **Change the shader's control flow** so the bounds check does not land in a branch that
   returns. All three `block_match*` shaders have exactly one early `return`, and only this one
   crashes, so the early return is necessary and not sufficient — the interaction is unidentified.

It is also a legitimate driver bug report: a valid module should be rejected or compiled, never
segfault the compiler. Two independent implementations crashing says the shape is unusual, not
that either driver is uniquely broken.
