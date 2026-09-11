# 0083 — The sequence encoder takes the source's colour space, and only at q=100

**Date:** 2026-09-11
**Item:** LOSSLESS-5
**Status:** accepted

## Context

LOSSLESS-4 made `gnc encode` code a Y4M still's own Y'CbCr planes at **every** quality. LOSSLESS-5
does the same for `benchmark-sequence`, which is the only sequence command that reads Y4M, and has
to decide whether to copy that "every quality" default.

A Y4M sequence reaches the encoder as interleaved triples — `encode_sequence_streaming` takes
`Vec<f32>` per frame — so the native arm is the file's Y'CbCr interleaved, with chroma
nearest-neighbour replicated to luma resolution and taken back down by the encoder's own box
filter. That round trip is exact, because the average of four copies of one integer is that
integer, and `chroma_4_2_0_survives_the_interleaved_round_trip` asserts it rather than assuming it.

## Decision

1. `benchmark-sequence` codes a Y4M in the file's own colour space **and** its own chroma format,
   which overrides `--chroma-format` — at that point the flag would be a request to resample the
   file. `GNC_RGB_PATH=1` is the other arm, the same switch LOSSLESS-4 gave the still path.
2. **Only when the request is lossless.** Below q=100 the RGB path stays the default.

## Why the gate, when the still path has none

At q=100 the two arms decode the same picture — the native one exactly — so the comparison is
bytes alone and there is nothing for CLAUDE.md's metric table to arbitrate. Measured, four clips,
both arms at the source's 4:2:0: **−12.4%** at ki=8 and **−12.9%** at ki=2, which lands on the
corrected still-path figure of −13.2%.

Below it, the same change is a rate/quality trade in a different colour space:

- the subband weights and the CfL range were tuned on YCoCg-R and are simply applied to Y'CbCr;
- VMAF scores luma only, so it cannot see the chroma half of what moves (CHROMA-1: a 6% rate move
  read 97.08 before and 97.08 after);
- and a luma PSNR computed in two different spaces is not a comparison at all.

That measurement has not been made. Shipping the change anyway would move every lossy Y4M sequence
figure in BASELINE with no number behind it, which is the shape of result this repository keeps
retracting. **LOSSLESS-6** is the item; the gate is one boolean and reverses in one line.

## What was not chosen

**Threading `EncodeInput` through `sequence.rs`,** which is what the backlog entry proposed as step
2. It is unnecessary: the interleaved door is exact for every chroma format (above), so the planar
plumbing would buy a shorter code path and a smaller upload, not a different result. `sequence.rs`
is 7 590 lines in one impl block (ARCH-4) and the change would touch every entry point. If it is
worth doing it is worth doing for the upload cost, measured, not for correctness.

**Making `read_frame_rgb` return Y'CbCr.** Every caller would have silently changed what it
compares against, including the VMAF writers. Instead the reader carries a `native` flag and
`read_frame_interleaved` dispatches, and `Y4mWriter` carries the same flag — applying BT.601 to
both the reference and the distorted stream would be symmetric and still wrong, because VMAF would
be scoring a pair of doubly-converted pictures.

## Consequences

- `encode-sequence` is unaffected: it reads PNG patterns only and has no Y4M path at all. A
  native-coded GNV1 is produced with `benchmark-sequence -o`.
- The canary prints on every Y4M input and names which of the three arms ran — native, refused for
  being lossy, or refused by `GNC_RGB_PATH` — so a silent default is not one of the outcomes.
