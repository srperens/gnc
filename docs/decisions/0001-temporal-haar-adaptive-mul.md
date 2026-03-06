# 0001 — Temporal Haar Per-Tile Adaptive Multiplier Recalibration

- **Date:** 2026-03-06
- **Status:** Accepted

## Context

The temporal Haar encoder applies a per-tile qstep multiplier to highpass frames.
The intent: tiles with low temporal energy (static regions) get coarser quantization
(higher mul), tiles with high energy (motion) get finer quantization (mul near 1.0).

Two bugs made the feature completely non-functional:

1. **Threshold miscalibration.** The original `map_energy_to_mul` used thresholds
   calibrated for synthetic or normalized data. Real temporal highpass energy on 1080p
   content at q=75 is typically in the range 0–20. The old thresholds placed the
   interpolation range far outside this band, so virtually every tile clamped to the
   floor value — the curve never engaged.

2. **Floor was 0.8 — directionally backwards.** The minimum multiplier was less than
   1.0, meaning static (low-energy) tiles could receive *finer* quantization than the
   lowpass. This is wrong: highpass should never be quantized more finely than lowpass.
   A mul < 1.0 wastes bits on noise in regions that contribute little to quality.

Additionally, `compute_temporal_tile_muls` returned only `Vec<f32>`, forcing a
separate GPU readback to detect all-zero highpass frames. Zero frames (entire highpass
quantizes to dead-zone) are common in low-motion content and a wasted encode round-trip.

## Decision

**Recalibrate `map_energy_to_mul` to range [1.0, max_mul] with log-linear
interpolation between thresholds 0.5 and 10.0.**

- `energy <= 0.5` → `max_mul` (static tile, quantize coarsely)
- `energy >= 10.0` → `1.0` (high-motion tile, preserve detail)
- Between: log-linear in energy, interpolating from `max_mul` down to `1.0`

The floor is now exactly 1.0, enforcing the invariant that highpass is never finer
than lowpass.

**Change `compute_temporal_tile_muls` return type to `(Vec<f32>, f32)`**, where the
second value is `global_max_abs` — the maximum absolute Y coefficient across all tiles,
computed during the same CPU readback that produces the per-tile means. The call site
compares `global_max_abs` against `dead_zone * qstep`; if below threshold, it skips
all GPU Rice encode work and returns `make_zero_compressed_frame()` directly.

## Consequences

Results on benchmark sequences at q=75:

| Sequence   | bpp change | Notes                                      |
|------------|------------|--------------------------------------------|
| rush_hour  | -37%       | Low-motion; static tiles now quantized aggressively |
| crowd_run  | -4%        | Mixed motion; marginal gain               |
| stockholm  | +7%        | High-motion content; highpass can't compress well regardless |

The stockholm regression is expected: for high-motion content, most tiles have energy
above the high threshold, so mul stays at 1.0 everywhere — the adaptive path is inert.
A per-tile temporal mode (Backlog #2) would let those tiles fall back to All-I instead
of wasting bits on uncompressible highpass.

The zero-frame skip eliminates all GPU Rice encode dispatches for all-zero highpass
frames, which are common in near-static sequences. No measurable quality impact; pure
throughput improvement.

## Open Questions

- PSNR is not measured in temporal wavelet streaming mode. Quality validation relies on
  bpp and visual inspection only. A streaming PSNR path is needed before adaptive mul
  parameters (thresholds, max_mul) can be tuned rigorously.
- The high_thresh of 10.0 is calibrated on bbb/crowd_run/rush_hour at 1080p q=75.
  Different resolutions or quality settings may need different thresholds.
