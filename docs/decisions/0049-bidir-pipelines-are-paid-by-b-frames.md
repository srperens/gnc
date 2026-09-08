# 0049 — Bidirectional pipelines are paid by B-frames, not by everything else

**Date:** 2026-09-08
**Status:** accepted
**Items:** BUG-40 (P2)
**Follows:** the laziness rule in `0029` / BUG-25's `split_pipeline`

## The decision

`match_bidir_pipeline`, `compensate_bidir_pipeline` and `compensate_bidir_chroma_pipeline` are
created on first dispatch, not in `MotionEstimator::new`. The pattern, the layout and the
comment already existed for `split_pipeline`. This is that rule applied to the other feature
that is off by default.

B-frames have been off since BUG-5 and the pyramid has been suppressed since 2026-09-06, so the
default path never dispatches these shaders. Creating them eagerly is what made every DX12
encode — including intra — die at pipeline creation on FXC `X3695` in `block_match_bidir.wgsl`,
on both Intel Arc Pro and NVIDIA RTX 2000 Ada. Vulkan and Metal compile the same WGSL.

## What was not chosen

- **Fixing the HLSL race FXC reports.** Unknown whether it is a real groupshared race or an
  over-conservative check, and it needs a Windows machine. Step 1 converts "DX12 does not run
  GNC" into a measurement of whatever the *used* shaders do. Step 2 is a different item, on a
  different machine, and only for a B-frame dispatch.
- **Making the shader module lazy too.** `split_pipeline` still builds the module and the
  layout in `new` and only defers `create_compute_pipeline`. The DX12 crash was recorded at
  pipeline creation, which is where FXC runs. Matching the existing pattern is the whole
  point of this change; a second laziness shape would be a second thing to keep true.
- **Leaving `compensate_bidir_chroma_pipeline` eager.** It is the same feature (4:2:0 B-frame
  chroma). The rule is per feature, not per named crash site.

## Evidence, Metal, this machine

Not idle. Bitstream identity does not care.

| artefact | before | after |
|---|---|---|
| still q=75 bbb_1080p | `0b5cc743…d063c`, 1 116 667 B, 4.31 bpp | identical |
| 9-frame ki=9 q=75 bbb_extended (1I+8P) | `d75c72ee…c7702`, 6 776 008 B, 2.90 bpp | identical |
| still q=90 bbb_1080p | (BASELINE 7.21 bpp) | 7.21 bpp, 1 869 133 B |
| still q=75 blue_sky / touchdown | (BASELINE 3.54 / 3.84 bpp) | 3.54 / 3.84 bpp |

Canary: `GNC_PROFILE=1` on an intra encode prints no `[bug40]`; the unit test
`bidir_pipelines_are_lazy_until_dispatched` asserts the three `OnceLock`s are empty after
`MotionEstimator::new` and after a P-frame `estimate`, and become `Some` on first
`match_bidir_pipeline()` call. `GNC_B_PYRAMID=1` on the 9-frame clip prints
`[bug40] match_bidir_pipeline=1` and `[bug40] compensate_bidir_pipeline=1`.

DX12 intra is not re-run here. That is the measurement the laptop round owes.

## Invalidation

None. Default output is byte-identical on the artefacts above. No published DX12 throughput
figure exists to retract; what this removes is the instrument failure that made every DX12
row a shader-compile crash.
