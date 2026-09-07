# 0024 — The GPU abac encoder counts before it writes, and abac is not folded into `gpu_entropy_encode`'s other job

Date: 2026-09-07
Item: ENT-5
Status: accepted (one criterion outstanding — see "What this does not discharge")

## Context

abac has had a GPU decoder since it landed and a CPU encoder only. That asymmetry was the third of
the three reasons decision [0017](0017-abac-ships-as-an-opt-in-coder-not-a-default.md) keeps abac
opt-in — **129 ms of CPU encode per 1080p frame against Rice's 23 ms** — and it was also the
*cause* of the ARCH-3 / BUG-18 class of defect: selecting a coder with no GPU encode shader
cleared `gpu_entropy_encode`, and that flag also chose which whole-frame P-frame pipeline ran, so
asking for abac silently swapped the frame encoder for the defective one.

ENT-5 supplies the shader. Two things in it were judgement calls rather than transcription.

## Decision 1 — the encoder runs the coder twice by default, rather than guessing a slot once

A decoder is told every block's length; an encoder has to discover it, and it has to place ~3000
variable-length streams into one buffer without threads treading on each other. Both documented
answers are implemented and selectable with `GNC_ABAC_GPU_SIZING`, and they produce **identical
bytes** (98 of 98 whole-file comparisons), so the choice is purely one of scratch memory against
coder passes:

| mode | coder passes | scratch, padded 1080p luma plane | round trips |
|---|---|---|---|
| `CountThenEmit` (**default**) | 2 | 411 364 B — the output size, +3 B/block of word padding | 2 |
| `BoundedSlots` | 1 | 16 836 768 B — **28.6× the output** | 3 |

`CountThenEmit` is the default because it needs **no bound at all**. The first pass runs the whole
coder and writes nothing but each block's byte count; the host prefix-sums those into exact
offsets; the second pass writes into them. There is no ceiling to derive, so there is no ceiling
to get wrong, and the scratch buffer is the size of the compressed plane.

`BoundedSlots` halves the coder work and costs 28× the scratch. Its ceiling is derived from the
coders' own renormalisation invariants rather than estimated: the interval coder keeps its
interval above `QUARTER = 2^14` and clamps a context probability into `[1, 4095]`, so one
context-coded decision emits at most 13 bits; the range coder keeps its interval above `2^24` and
may start from `2^32`, and a clamped probability can drop it to `2^13` — three renormalisation
bytes, 24 bits. A bypass decision halves the interval exactly. So 24 bits per context decision, 8
per bypass, and a 256-bit tail for `finish` bounds both engines with slack. **It is loose by
design** — a zero coefficient is charged 3 bytes where it actually costs about a fiftieth of one —
and that looseness is the 28×.

**What was rejected, and why.** A *heuristic* slot with a panic behind it, which is what Rice and
Huffman do: `max_stream_bytes_for_tile` fits a bpp model and asserts on overflow. That is BUG-22
exactly — a Huffman stream slot fixed at 512 bytes from a rate estimate, with nothing checking the
write pointer, spilled into its neighbour's slot and cost **7.8–10.9 dB at q=90 on all four
stills**. An estimate plus an assertion is not a bound; it is a bound-shaped thing that fails
later. Both modes here still raise an overflow flag the host turns into a panic, but in
`CountThenEmit` that flag can only fire if the two passes disagree — i.e. if the coder is not
deterministic — and in `BoundedSlots` only if the derivation above is wrong.

Also rejected: a single pass into an atomic bump allocator handing out fixed chunks, which would
be memory-optimal *and* one coder pass. It needs a per-block chunk list to reassemble the stream,
and the length of that list is bounded by the same worst case the slot mode has — so it
reintroduces the unbounded structure it was meant to avoid, with a chained writer on top.

**Which of the two is faster is not settled here, and that is deliberate.** Four Claude sessions
were working this M1 at load 10.2 while this landed, and COORDINATION's rule is that a wall-clock
figure taken under load is worth nothing — the same abac input has timed 25.2, 31.1 and 37.5 ms
across three runs on this machine. `tests/abac_bench.rs::abac_encode_throughput_grid` times both
modes, both engines and the CPU arm paired in one process, best of 24 repeats with `med/best`
printed beside it; one idle-machine run settles it and can flip the default without touching a
line of coder code, because the bytes are identical either way.

## Decision 2 — abac routes on `gpu_entropy_encode` and is *not* added to `use_gpu_encode`

The obvious wiring was to delete `&& config.entropy_coder != EntropyCoder::Abac` from the three
`use_gpu_encode` expressions. That was not done, and the exclusion is left in place with the
comment corrected.

`use_gpu_encode` carries three unrelated jobs: which entropy encoder runs, whether the fused
quantize+histogram shader runs (`use_fused_qh = … && use_gpu_encode && !use_cfl`), and — in
`sequence.rs` — which whole-frame P-frame pipeline runs. ARCH-3 exists to separate the third.
The second matters here for a measurement reason: **BUG-16 records that Rice's two encode paths
disagree on the picture at q=25**, 35.51 dB on the fused path against 35.63 dB off it, and the
suspect is the quantiser rather than the coder. Folding abac into `use_gpu_encode` would therefore
have moved abac's *pixels* in the same commit that moved its encoder, and the byte-identity check
that makes this work verifiable would have been impossible to run.

So the abac branch tests `config.gpu_entropy_encode` directly, one condition inside
`encode_entropy`. Nothing else moves: for abac `use_gpu_encode` is false in both arms, quantisation
is untouched, and `--cpu-encode` against the default is a clean controlled comparison in which the
only difference is where the entropy coding runs. It also means this change is orthogonal to
ARCH-3 rather than racing it, and that after ARCH-3 lands the abac video path picks up the correct
frame pipeline without anything here changing.

## Two porting facts worth keeping

- **`low` is u64 in `abac.rs` and WGSL has no u64.** It is emulated as (low 32 bits, carry count)
  with every use in the Rust mapped term for term, and `carry` is a *count* rather than a flag. The
  tempting argument that at most one carry can be pending needs `shift_low` to run between two
  adds, and the renormalisation loop does not run while `range` stays above `RC_TOP`. Counting
  costs one instruction and makes the emulation exact whether the argument holds or not.
- **The neighbourhood sum saturates on the CPU and wraps in both shaders.** `neighbour_sum` uses
  `saturating_add`; `abac_decode.wgsl` has always used a plain add, and the new encoder matches it
  so the GPU pair agrees with itself. They diverge from the CPU reference only if four neighbour
  magnitudes sum past 2^32, which needs a quantised coefficient near 2^30 — unreachable in this
  codec. The encode shader raises a flag at 2^29 and the host panics on it, so the argument is
  *checked on every encode* rather than merely believed. Fixing the decoder to saturate would cost
  four extra ops per coefficient in the hottest loop of a measured 1.69× decode figure, for a case
  that cannot occur; that trade was declined.

## What this discharges

Decision 0017's reason 2 — "encode is CPU-side and single-threaded … it is also fixable, the work
is parallel across ~3000 code-blocks, but it is not fixed today" — is no longer true as stated.
The encoder exists, runs one thread per code-block, and is bit-exact.

## What this does not discharge

- **The reason 2 *figure*.** The encode-time-per-frame criterion ENT-5 set for itself is
  unmeasured, because the machine was not idle. Until that run happens, "abac has a GPU encoder"
  is a fact and "abac's encode is fast enough to be a default" is not a claim.
- **Reason 1**, the 1.69× frame decode, is untouched: nothing here changes the decoder.
- **Reason 3**, abac on inter, was discharged by ARCH-3 rather than here: it landed while this was
  in flight, replaced the two P/B encoders with one, and measured abac's inter rate at −12.0% to
  −22.9% on three sequences at bit-identical pixels. What this item contributes there is only that
  the GPU encoder is byte-exact on P-frame residuals too — it verifies the *coder*, not the
  pipeline around it.

So abac stays opt-in and Rice stays the default. This item removed one of three obstacles and one
class of defect; it did not make the decision that 0017 declined to make.
