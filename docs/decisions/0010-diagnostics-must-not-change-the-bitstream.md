# 0010 — Diagnostics must not change the bitstream

**Date:** 2026-09-06
**Status:** Accepted

## What happened

`GNC_DIAGNOSTICS=1` made the encoder produce a **32% larger file** on the same input
(blue_sky, 8 frames, q=50: 2,808,848 bytes quiet, 3,703,862 loud).

The temporal-wavelet diagnostic captures original-signal wavelet coefficients so that a P-frame's
residual can be compared against them. To do that it runs a second, full wavelet transform — and
it ran that transform through the encoder's *shared* GPU buffer pool, which is where the
motion-compensation reference lives. Every P-frame from the third onward was then motion-compensated
against clobbered data. The residual came out as large as the raw frame difference, meaning motion
compensation was contributing nothing at all.

## Why it took so long to find

Two reasons, both worth internalising.

**The symptom looked like a codec result, not a bug.** "P-frames cost as much as I-frames" is a
plausible thing for a wavelet codec to do — plausible enough that it was written into the backlog
as a finding and used to steer the research programme, rather than investigated as an anomaly. The
tell was sitting right next to it: one frame in every sequence came out at *2%* of an I-frame. Two
percent is not a plausible codec result. An implausible number adjacent to a plausible one means
neither can be trusted, and the plausible one is the more dangerous of the two precisely because it
does not provoke a second look.

**It was perfectly reproducible, and that read as evidence it was real.** It reproduced because the
corruption is deterministic — same shader, same buffers, same order. Reproducibility distinguishes
a bug from noise. It does not distinguish a codec property from an instrumentation artefact, and
those are the two hypotheses that mattered here.

The check that does distinguish them is trivial and was never run: encode twice, once observed and
once not, and compare the bytes.

## What it invalidates

- `ratio_vs_iframe` and every "temporal prediction may not be effective" warning ever emitted. The
  real ratios on blue_sky at q=50 are 0.55–0.61; the warning was firing on damage the diagnostic
  caused.
- Residual statistics from the third frame of any sequence onward.
- **MEAS-4**, which is reopened. Its residual dumps used `GNC_DUMP_RESIDUAL` together with
  `GNC_DIAGNOSTICS=1`. MEAS-4 concluded that the inter gap lies in prediction quality rather than
  the coding model, and four subsequent experiments — multi-reference P-frames, the sub-pel
  interpolation filter, motion-search quality, MCTF — were all chosen because of that conclusion.
  All four came back negative. A run of negatives against a hypothesis is weak evidence that the
  hypothesis was wrong; there is now a concrete reason to think it was never properly tested.

MEAS-1's 5–7x video figure is unaffected: `meas1_vs_h264.py` encodes without diagnostics.

## The rule

**A diagnostic may not touch encoder state.** On a CPU codec "read-only" is nearly free to
guarantee; on a GPU pipeline it is not, because even a read-only computation needs scratch memory,
and scratch memory came from the working set. Any diagnostic that needs a transform needs its own
buffers.

Enforced by `tests/diagnostics_neutral.rs`: six synthetic frames encoded twice, with and without
diagnostics, asserted byte-identical. It was verified to fail when the offending diagnostic is
re-enabled (+72.1%), which is the only property that makes such a test worth having.

Two details of that test are load-bearing and easy to get wrong. It needs
`keyframe_interval = 9`, because the default quality preset is all-intra and an all-intra sequence
cannot exercise a P-frame bug. And it needs at least three P-frames, because the corruption only
appears from the third frame on — a shorter sequence passes whether or not the bug is present.

## Status of the diagnostic itself

Gated behind `GNC_DIAG_TWAV=1` rather than deleted: the measurement it makes is genuinely useful,
it is just implemented through the wrong buffers. Anyone who sets that flag is opting into a
corrupted encode, and the flag's documentation says so. Giving it private buffers would make it
safe to re-enable by default; that is not done yet.
