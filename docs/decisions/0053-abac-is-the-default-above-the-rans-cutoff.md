# 0053 — abac is the default entropy coder above the rANS cutoff

**Date:** 2026-09-14
**Item:** ENT-10 (closes it) + ENT-14 (the change that unblocked it)
**Status:** accepted — **supersedes `0052`**, which refused this same flip earlier the same day
**Machine:** Apple M1 Pro, 16 cores, 16 GB, Metal — `gnc gpu-info`. One of two Macs (COORDINATION).
**Binary:** `codec-fingerprint v1 fb2fe82a` → **`027e58bc`**

## Context

`0052` measured abac's rate case, found it overwhelming on every real picture, made the flip — and
withdrew it within the hour because `regression_gradient_q50` failed at **+41% bpp**. abac was
**+234.5%** against Rice on a sparse synthetic gradient, and the cause was measured rather than
guessed: `abac_blocks=280 (empty=0)` per plane, **840 code-blocks and not one recognised as
empty**, ~11.5 bytes of floor each. On sparse content the file was nothing but floor.

`0052` filed that as ENT-14 and said the flip lands when it closes. It has closed.

## ENT-14, and why it needed no format change

**An all-zero code-block now codes to no bytes at all — length 0 — instead of a terminated
arithmetic interval.**

The part that makes this cheap is a property that was *tested rather than assumed*: **a zero-length
stream already decodes to zeros on both engines.** The decoders read past the end as zero bytes,
and the initial contexts then resolve every significance decision to "not significant".
`empty_stream_decodes_to_zeros` pins it. So:

- **No bitstream generation moves.** The format already expressed "this block is nothing"; the
  encoder simply never said it.
- **Existing decoders read the new streams correctly**, and files written before today are
  unaffected.
- The change is three places that must agree — the CPU reference in `abac_tile.rs`, and
  `block_is_empty` in `abac_encode.wgsl` used by both coder entry points *and* by `bound` — and the
  existing byte-identity gate (`gpu_encode_matches_cpu_encoder_byte_for_byte`) is what proves they do.

One degenerate case became reachable: a plane whose every block is empty now codes to **zero total
bytes**, and `read_buffer_u32` panics on a zero-sized staging buffer. Handled explicitly rather
than by rounding a size up.

**The canary fires:** `empty=264` of 280 blocks on the gradient, and **`empty=420..476` of 2800 on
a photograph** — 15-17% of blocks are empty even on busy content, which is why this helped
everywhere and not only in the corner it was filed for.

## What it did to the numbers

| | before ENT-14 | after |
|---|---|---|
| gradient q=25 | **+234.5%** | **−8.7%** |
| gradient q=90 | +64.7% | −27.9% |
| stills, q=25..85 | −10.1% to −15.0% | **−12.9% to −20.7%** |
| stills, q=90..99 | −11.8% to −15.4% | −11.9% to −15.4% |
| sequences, q=90 | −12.3% to −17.4% | −12.7% to **−17.8%** |

**And the case nobody had measured.** `0052` said the sequence rows were all busy content and that
a static shot was the real risk. One real frame held for nine frames, 1920x1080 4:2:0:

| | Rice | abac | |
|---|---|---|---|
| q=90 | 1 657 892 B | 1 380 721 B | **−16.7%** |
| q=99 | 3 818 184 B | 3 172 079 B | **−16.9%** |

The predicted bad case is abac's **best** case. All five of ENT-14's success criteria, written
before the work, are met; two are exceeded (the gradient target was ±5% of Rice and it came in at
−8.7% to −27.9%).

## Decision

**`quality_preset` selects `EntropyCoder::Abac` for q > 20.** rANS keeps q ≤ 20.

**The cutoff did not move, and that is a finding rather than a convenience.** abac loses to rANS
below it and wins above it, on all three images, at the same q where rANS already stopped being the
default — measured *before* ENT-14, so these are its floor: q=15 read +5.8% / +0.3% / −0.1%, q=20
read +1.6% / −2.8% / −3.4%, q=25 read −6.0% / −7.2% / −7.9%. Whatever makes rANS strong at four
decomposition levels makes abac weak there too. **The three constants move together if any moves.**

**Cost, stated in the same breath as the win** (GOALS §5 consequence 2): **1.7x to 5.4x encode and
1.0x to 2.3x decode.** The ratio is worst where Rice is fastest, because abac's encode is nearly
content-independent — ~73-80 ms per 1080p frame at cb=32 where Rice moves 16-91 ms with the
content. ENT-13 is the item that attacks it and `0051` already took 2.3x out.

## What was rejected

**Signalling empty blocks with a bitmap or a run-length instead of a zero varint.** A zero varint
is already one byte, so a bitmap saves at most 7/8 of 840 bytes on the worst frame measured — real,
and far below the noise of the change that just landed. A run-length over consecutive empties is
better motivated (deep subbands are empty in runs) and is worth its own measurement, not a guess
bundled into this one.

**Keeping Rice as a per-content fallback.** There is no longer a measured content class where Rice
wins: stills, sequences, the static shot and the sparse gradient all go abac's way. A fallback with
no case to serve is a branch that rots.

**Flipping `Sizing::BoundedSlots` at the same time.** See below — it is now the larger remaining
lever, and it is not this decision's to take.

## The sizing measurement, since it landed the same hour

Another session's ENT-5 handoff asked for `abac_encode_throughput_grid` on an idle Mac, with a
Windows prior (RTX 2000 Ada) of **1.54x** for one coder pass against two. Run here, M1 Pro, Metal,
best of 24, `med/best` 1.02-1.03 so the absolutes are quotable — **and with ENT-14 in the build,
which the Windows prior did not have**:

| path | bytes | plane ms | frame ms |
|---|---|---|---|
| GPU Range / CountThenEmit (2 passes) | 395 501 | 51.97 | **155.92** |
| GPU Range / BoundedSlots (1 pass) | 395 501 | 32.49 | **97.46** |
| CPU Range, one thread | 395 501 | 26.57 | **79.71** |

**The Windows ratio reproduces on Metal: 1.60x against their 1.54x**, and bytes are identical
across the pair as required. That also **discharges `0017` reason 2 / ENT-5 criterion 3**: 97.46
ms/frame against the 129 ms that record quotes.

**Two things in that table deserve their own items rather than a footnote.** First, the
single-threaded **CPU** coder is *faster than the GPU one* (79.71 against 97.46 ms/frame) — on a
16-core GPU, for work that is 2800 independent blocks. Second, a whole-frame `gnc benchmark` moves
only **1.10x-1.27x** between the two sizing modes where the coder stage moves 1.60x, which says
**the coder stage is a minority of abac's frame encode time.** ENT-13 was filed on the assumption
that double-coding is the main cost; it is not, and that item should be re-read in this light
before anyone starts it.

## Consequences

- **ENT-10 closes.** Open since 2026-09-08, blocked on cost for all of it.
- **The headline rate figure moves.** BASELINE, GOALS §1 and POSITIONING quote **+89.2% BD-rate**
  against x264 with Rice; MEAS-11 recorded **+61.0%** with abac. That re-take is now a re-take of
  the *default*, not of an option, and it should be done at cb=32 post-ENT-14 — every `--abac` row
  in BASELINE predates both.
- **`0052` is superseded**, not withdrawn: its refusal was correct on the evidence it had, and the
  record of *why* a 41% regression test outranked a −17% average is worth keeping.
- **`0045`'s inter decay is superseded** for the third time — −3.7% at q=99 now reads −11.8% to
  −17.8%.
- `--rice` remains reachable and is still the default below the cutoff via rANS; nothing was
  removed.
