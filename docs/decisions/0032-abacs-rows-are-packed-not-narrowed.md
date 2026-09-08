# 0032 — abac's row buffer is packed, not the workgroup narrowed

**Date:** 2026-09-08
**Item:** BUG-31
**Status:** accepted

## The problem

`abac_decode.wgsl` and `abac_encode.wgsl` each declared 18688 B of workgroup memory — 2304 B of
context probabilities plus 16384 B of neighbour magnitudes — against a device created with
`max_compute_workgroup_storage_size: 16384`. **2304 B over, on both sides of the codec.**

Nothing enforced it. That limit appears nowhere in `wgpu-core`'s validation or device code: it is
negotiated with the adapter and reported back, never compared against a shader at pipeline
creation. Native Metal honours the 32 KB the hardware has, so the 16 KB GNC *asked for* was a
number nobody checked. A conformant WebGPU implementation does check it — 16384 is the spec's
guaranteed minimum, not an accident of wgpu's defaults.

`DecoderPipeline::new` constructs `GpuAbacDecoder::new(ctx)` unconditionally
(`src/decoder/pipeline.rs:296`) and `wasm::decode_gnc` builds a `DecoderPipeline` per call, so the
failure mode was **every WASM decode, Rice files included** — not only abac ones.

## What was chosen

Store one clamped byte per magnitude instead of one u32, four to a word. `rows` goes from 4096
words to 1024; the shader goes from **18688 B to 6400 B**.

**The clamp is exact, not an approximation.** `bucket` saturates: any `nb >= 1 << (NUM_BUCKETS - 2)`
— 16, with `NUM_BUCKETS = 6` — returns `NUM_BUCKETS - 1`. So with `ROW_CLAMP = 16`:

- a contributor below 16 is stored exactly, and a sum of such contributors is exact;
- a contributor at or above 16 stores 16, and the clamped *and* true sums are then both `>= 16`,
  so both bucket to `NUM_BUCKETS - 1`.

Either way `bucket(nb)` is identical, the coder sees the same context sequence, and the bytes are
the same. The clamp is written as `1u << (NUM_BUCKETS - 2u)` so it cannot drift from `bucket`.

Thread-interleaved indexing survives: magnitude `i` for thread `t` lives in word
`(i >> 2) * WG + t`, so lane `t` is still bank `t` — the property the original comment says is
"the whole performance story". Each thread owns its words exclusively, so the read-modify-write in
`row_set` needs no barrier.

## What was rejected, and what it would have cost

**Narrow the workgroup, `WG` 32 → 28.** 28 x (18 + 128) x 4 = 16352 B, which fits. Four lines,
and byte-identical by construction with no argument needed — genuinely the simpler change, and
GOALS §5 says prefer the simpler one when results are similar. **The results are not similar.**
It idles 4 of every 32 SIMD lanes, ~12.5% of the threads, on the codec's slowest stage (abac is
~1.69x frame decode), and at 16352 B it *still* fits only one workgroup per core against the 16 KB
budget. It buys conformance and pays throughput for nothing. Packing reaches 6400 B, which is
inside the two-workgroups-per-core point CLAUDE.md names as full occupancy at 16 KB — the old
layout could not reach it at all.

**Reduce `MAX_BLOCK_W` 64 → 48.** 14592 B, fits. Rejected: the code-block width is a bitstream
parameter, and every abac rate figure on record (−16.6% to −18.8% intra, −13.4% lossless, −12.0%
to −22.9% inter) is at cb=64. This would invalidate all of them to fix a limits bug.

**Raise the request to 32768.** One line, and the adapter here offers it. Rejected on GOALS §2
rule 4: 16384 is the WebGPU guaranteed minimum, so this trades the portability axis the project
claims to *win* on for the convenience of not touching a shader. If it is ever wanted it needs its
own decision record, and this is not it.

**Move `probs` back to function scope.** Would leave `rows` at exactly 16384 B, which fits. The
shader's own comment records that this *was* the first version and that a dynamically indexed
function array spills to device memory on Metal — the change from it was "the difference between
'a GPU port exists' and 'a GPU port is worth shipping'". Rejected as a measured regression.

## The item's stated first step was not made the gate

BUG-31 said: run the WASM decode in a browser first, and "if it passes, this is P3
documentation." **That inference does not hold, which is why the browser run was not the gate.** A
browser that happens not to validate would not make 18688 B against a 16384 B device conformant;
it would only hide the defect behind one implementation's leniency, and the spec rule would still
be there for the next one. The decisive test is the one that is deterministic and permanent:
`tests/workgroup_storage_limit.rs` computes declared workgroup storage per compute entry point
from naga and asserts it against `wgpu::Limits::default()` — the budget read from the same place
the device request reads it, so raising the request moves the test with it.

That test found the class rather than the instance: **nine** entry points were over budget, not
four. The five that are not abac are filed as **BUG-35**, and one of them —
`quantize_histogram_fused.wgsl` at 23800 B — is on the **default encode path**, not an opt-in
backend. BUG-31 as filed understated its own finding by more than half.

## Evidence

- **98 of 98 whole-file byte comparisons identical** (`scripts/ent5_gpu_encode_gate.sh`): both
  arithmetic engines, both output-sizing modes, 4:4:4/4:2:2/4:2:0, lossy and bit-exact lossless,
  four stills and two sequences. The CPU encoder is untouched, so GPU == CPU after the change and
  GPU == CPU before it means the emitted bytes did not move.
- **Decoded output hashed before and after**, on `bbb_1080p` at q=50/90/100 and
  `kristensara_720p` q=90 4:2:0: all four SHA-256s identical, and the four `.gnc` bitstreams
  compare equal.
- `gpu_decode_matches_cpu_coder`, `gpu_encode_matches_cpu_encoder_byte_for_byte` and the rest of
  the suite: **234 tests pass, 0 failures.**
- Native clippy clean. `--target wasm32-unknown-unknown --lib` clean and builds; the `bin` target
  still fails there, which is **BUG-24** and predates this change (a `.wgsl` diff cannot affect
  it).

## Not measured

**Throughput.** The occupancy argument above is structural, not a measurement: another session was
active, and this repository has paid for wall-clock numbers taken on a loaded machine more than
once. The 6400 B figure is what the shader declares; whether it converts into decode time is owed,
on an idle machine, with `cargo test --release --test abac_bench -- --ignored`. Note BUG-32 before
choosing an instrument: `benchmark-sequence`'s wall clock is 86% CPU quality metrics.

So this record claims **conformance and byte-identity, both verified**, and *does not* claim a
speedup.
