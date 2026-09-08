# 0048 — the fused histogram arena cannot shrink; the missing guard was the defect

**Date:** 2026-09-08
**Item:** BUG-35 (the rANS half; the default path remains `0035`)
**Status:** accepted

## Context

`0035` took the default encode path off `quantize_histogram_fused.wgsl:main` (23800 B against a
16384 B device) by adding `main_quantize_only` and creating `main` lazily. What it left was
explicit: guard the 5120-entry `shared_hist` arena, measure how large it actually gets, then
size it. Shrinking to ≤3266 entries is what would put `main` under budget, and it was refused
as the *first* move because nothing compared `total_hist_entries` to 5120. naga clamps
out-of-range atomics, so an overflow silently corrupted frequencies.

The neighbouring rANS encode shader already had the shape of the guard, on the host:
`check_cumfreq_capacity` refuses a tile whose group tables sum past 4097.

## Decision

**Add the guard. Do not shrink the arena.**

`check_hist_arena_capacity` runs before `check_cumfreq_capacity` and before any stream is
kept. It sums per-group `alphabet_size` — not `alphabet_size + 1`, which is the encode
table — and panics with the count and the cap when the sum exceeds 5120. The two histogram
shaders (`quantize_histogram_fused.wgsl`, `rans_histogram.wgsl`) skip `atomicAdd` /
`atomicLoad` past the end so the dispatch that is about to be thrown away is defined.

The canary is `[rans] hist_arena_max=N/5120 (tile T)` under `GNC_PROFILE` or diagnostics.
A Rice encode never prints it; `GNC_PROFILE` still shows `with_histogram=0` on the default
path.

## What the measurement showed

Four stills, `--rans`, 4:4:4, not idle. Worst tile per frame:

| image | q=15 | q=70 | q=85 | q=90 |
|---|---|---|---|---|
| bbb_1080p | 313 | 3428 | 5322 | 6648 |
| blue_sky_1080p | 320 | — | — | 6843 |
| touchdown_1080p | 335 | — | — | 6575 |
| kristensara_720p | — | — | — | 7004 |

q=15 is the preset's rANS range and uses **6% of the arena**. q=70 `--rans` was in `0035`'s
identity gate and still fits. q≥85 `--rans` overflows 5120 on every image tried. That
configuration was producing a file whose frequencies were the naga clamp of the real
histogram; the check turns it into a named refusal.

## What was not chosen

**Shrink 5120 → ≤3266.** That is the number that would fit `main` in 16384 B
(23800 − 4×(5120−3266) ≈ 16384). It would refuse bbb at q=70 (3428 > 3266), which is a
shipped `--rans` point. The theoretical max is 12×4096 = 49152; the measured max at
contribution quality is ~7000. Neither fits a 3266-entry arena, and growing to 7004 takes
`main` to ~31 KB — raising the device request, which `0032` refused.

**Clamping in the shader as the guard.** That is what naga already did. A `min(idx, 5119)`
helper without the host check would have been the defect with a comment on it.

**Returning a `Result` instead of `assert!`.** Matched `check_cumfreq_capacity`. A quiet
`Err` on a path whose only previous behaviour was a wrong file would have been easy to
ignore in a harness; a panic is the same shape as the limit that already binds first on
kristensara at q=76 (BUG-9).

## Evidence

- Four unit tests on the sum and the refusal, no GPU.
- `GNC_PROFILE=1` Rice q=90: nine `main_quantize_only` dispatches, `with_histogram=0`, no
  `hist_arena_max` line.
- `GNC_PROFILE=1` `--rans` q=15 on three 1080p stills: 313 / 320 / 335, file produced.
- `GNC_PROFILE=1` `--rans` q=90 on four stills: 6648 / 6843 / 6575 / 7004, panic at the
  check, no file.

## Still open

The five over-budget rANS entry points. This record does not close BUG-35. It closes the
"guard, measure, size" order the item wrote: the first two are done, the third is refused,
and packing / a storage-buffer histogram / parking the shaders are the remaining answers.
