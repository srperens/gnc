# 0002 — Disable CfL for Temporal Highpass Frames

- **Date:** 2026-03-06
- **Status:** Accepted

## Context

CfL (Chroma-from-Luma) predicts chroma wavelet coefficients as `chroma ≈ alpha * recon_luma`
in the wavelet domain. Alpha is estimated per-tile via GPU regression and stored in the
bitstream. The mechanism is effective because spatial content has correlated luma-chroma
channels at the same spatial frequency — the same texture produces related energy in both
channels.

In temporal Haar decomposition, highpass frames hold `H[j] = (frame[2j+1] - frame[2j]) / 2`:
temporal differences of spatial wavelet coefficients. They represent motion energy, not
spatial texture.

Two problems arose when CfL was applied to these frames:

**Quality.** The luma-chroma correlation that CfL exploits is a property of spatial content.
In temporal highpass frames, luma and chroma *differences* decorrelate — especially for
moving objects where luma and chroma change at different rates. Alpha estimation on
low-energy temporal residuals fits noise rather than structure, producing unreliable alpha
values that can hurt compression rather than help it.

**Performance.** Every CfL-enabled encode requires a `device.poll(Maintain::Wait)` to read
alpha values back from the GPU. With 7 highpass frames per GOP-8, this added 7 blocking
CPU-GPU synchronisation points per GOP, costing approximately 21–35 ms per GOP.

## Decision

Disable CfL (`cfl_enabled = false`) for all temporal highpass frames in both the Haar and
LeGall 5/3 temporal paths (Haar: H frames; LeGall 5/3: d0, d1 frames).

CfL remains enabled for:

- The temporal lowpass frame (a full spatial frame — CfL applies normally).
- All non-temporal frames (I, P, B).
- Still-image encoding.

The decision was made on theoretical grounds: the signal model that justifies CfL does not
hold for temporal residuals, and the performance cost is concrete and measurable.

## Consequences

- Approximately 21–35 ms per GOP saved in temporal Haar mode (7 fewer blocking GPU readbacks).
- No measurable quality impact expected; alpha estimation on residuals was unreliable.
- The lowpass frame and all spatial frames continue to benefit from CfL.
- Code paths are simpler: highpass encode configs unconditionally set `cfl_enabled = false`.

## Open Questions

- No formal PSNR A/B test was run comparing CfL-on vs CfL-off for temporal highpass frames.
  The decision rests on theoretical reasoning. To validate: temporarily set
  `high_cfg.cfl_enabled = true` in the Haar/LeGall temporal encode paths and compare
  bpp/PSNR on `crowd_run` at q=75.
- If a future temporal transform produces highpass frames with measurably stronger
  luma-chroma correlation (e.g. a transform that preserves DC structure in residuals),
  this decision should be revisited.
