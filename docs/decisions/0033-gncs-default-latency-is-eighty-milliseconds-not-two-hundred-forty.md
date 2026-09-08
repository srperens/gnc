# 0033 — GNC's default latency is ~80 ms, not ~240 ms

**Date:** 2026-09-08
**Item:** MEAS-6
**Status:** accepted; reverses a claim carried by POSITIONING, the README and MEAS-6's own entry

## What changed

Nothing in the codec. This record exists because a **recorded conclusion was reversed by a
configuration change made two days earlier, and no document noticed.**

On 2026-09-06, MEAS-6's first pass measured the hierarchical B-pyramid at 8 frames of lookahead —
160 ms of structural delay at 50 fps — and concluded:

> GNC's default configuration sits in the low-latency-HEVC band, not the JPEG XS band.

Later the same day, on that finding plus BUG-5's rate measurement, `quality_preset()` was changed
to veto the pyramid:

```rust
// src/lib.rs:1021
b_pyramid: std::env::var("GNC_B_PYRAMID").map(|v| v == "1").unwrap_or(false),
```

The conclusion was correct when written and false a few hours later. It stayed in three documents
until 2026-09-08: `docs/POSITIONING.md` (§3, "current default"), `README.md` (the latency bullet)
and `BACKLOG.md` (MEAS-6's own entry). **The change that invalidated it was made by the item that
produced it.**

## The corrected position

Verified on the current build at the default `ki=9`, from the encoder's own output rather than
from the source:

| configuration | frame mix | canary | reordering delay |
|---|---|---|---|
| `benchmark-sequence -q 75` (default) | `2I+16P+0B` | `B-pyramid suppressed … zero reordering latency` | **0 frames** |
| `GNC_B_PYRAMID=1` | `2I+2P+14B` | silent | **8 frames**, 160 ms at 50 fps |

| | latency |
|---|---|
| JPEG XS | 1–32 lines; EBU measured under one frame |
| NDI High Bandwidth | under 16 ms |
| **GNC, default (P-only)** | **~80 ms** |
| low-latency HEVC | 120–3060 ms (EBU, real vendors) |
| GNC, `GNC_B_PYRAMID=1` (opt-in) | ~240 ms |

At ~80 ms the default is **below** the low-latency-HEVC band's 120 ms floor, between NDI High
Bandwidth and low-latency HEVC. It remains two orders of magnitude off JPEG XS, which is
line-based by construction and not a gap a GOP change can close.

## What was *not* chosen

**Re-measuring the ~80 ms instead of pinning the structural half.** This is the more attractive
option and it was rejected on the measurement rules, not on effort. BASELINE records that an fps
run taken during another session's `cargo test` reads 20% slow; this machine sat at load 21–39 for
the entire session with two other sessions running the suite. A number taken there is not a
number, and the 240→80 ms correction does not depend on it: the part that moved is the reordering
delay, which is 0 or 8 frames and is unmovable by load. **Re-taking the coding time on an idle
machine is still owed and is recorded as MEAS-6's cheapest remaining step.**

**Quoting the two halves as one figure.** They have different standing and the old text hid it.
The reordering delay is exact and structural. The ~80 ms is a 2026-09-06 wall-clock reading on a
non-idle machine, labelled M1 when the box was in fact the M5 Pro (BUG-29). "~80 ms" is a bound,
not a measurement, and it is now written that way.

**Turning the pyramid back on, or removing it.** Neither was considered here — MEAS-6 measures,
it does not set the default. BUG-5 owns that choice and the veto's own comment records the trade:
the pyramid *wins* 34–39% on animation and costs 7–31% on camera content, so it is a content bet
kept as an opt-in rather than a mistake to delete.

## Why it went unnoticed for two days

The veto is loud at run time — it prints `B-pyramid suppressed …` on every affected encode — and
silent at review time, because no document is checked against it. The generalisable point:
**a config change that invalidates a written conclusion has no way to reach the document that
carries it.** The canary caught nothing here because nobody was looking; what found it was reading
`sequence.rs` for an unrelated reason.

The cheap countermeasure is the one used above — quote the *canary string and the frame mix*
rather than the prose claim, so a document states something a command can contradict. Every table
in this record is reproducible with one invocation.

## Consequences

- POSITIONING §3, README and MEAS-6's BACKLOG entry corrected; each says what it used to claim.
- Anything quoting "~240 ms" as GNC's latency is quoting an opt-in configuration.
- `docs/decisions/0025`'s inter figure (−12.0% to −22.9%, 18 frames, ki=9) **does not state
  whether the pyramid was on**. 0025:90's "B-pyramid on and off" covers its 54-configuration
  byte-identity sweep, which is a different claim from its nine-point rate table. The frame mix at
  ki=9 is either `2I+16P+0B` or `2I+2P+14B` — materially different residual statistics — so the
  table should say which. Flagged to the ENT-3 session, whose item it is; not changed here.
- **BUG-37** filed: `benchmark-sequence`'s `-q` is an `Option` with no default, so without it the
  command still codes the pyramid. No recorded measurement is affected — every harness in
  `scripts/` passes `-q` — but the flag named *quality* silently selects the *GOP structure*.
