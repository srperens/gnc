# 0038 — one quantiser beats a wash: the sparse dead zone is off by default

**Date:** 2026-09-08
**Item:** BUG-16
**Status:** accepted

## What was wrong

`quantize_histogram_fused.wgsl` had a Phase 1.5 "content-adaptive dead zone": for non-LL subband
groups where at least 95% of coefficients were already zero, it re-quantised the remaining ±1
values to zero, up to a 1.25× dead-zone expansion. **No other quantise path had it** — not
`quantize.wgsl`, not the CPU quantiser.

So the same `CodecConfig` produced different coefficients depending on which quantiser ran, and
which one ran depended on `use_fused_qh = use_fused_quantize_histogram && use_gpu_encode &&
!use_cfl`. That is the whole of BUG-16:

| bbb, q=25, 4:4:4 | recorded in BUG-16 | now |
|---|---|---|
| GPU encode | 35.51 dB, 415 544 B | **35.63 dB, 425 944 B** |
| CPU encode | 35.63 dB, 610 264 B | 35.63 dB, 610 264 B |
| 4:2:2 q=75, GPU vs CPU pixels | max abs diff 1.69 | **max abs diff 0** |

All three rows are explained by this one feature. The old GPU figures reproduce exactly with
`GNC_SPARSE_DZ=1`, so nothing about the original measurement was wrong.

**The remaining size difference is expected and is not a defect.** The CPU reference Rice coder
lacks per-stream *k* and the checkerboard *k*-context, so it is simply a worse coder — which is
why the two arms agree on the *picture* to three decimals and still differ by 43% in size.

## The measurement that decided the direction

The obvious framing — "the GPU path emits a smaller *and* worse file, so it is discarding
something" — cannot be settled against the CPU arm, because that arm is independently worse. It
needs the same entropy coder on both sides, which is what the new `flags` bit 1 provides.

Three stills, q=15/25/30, GPU both arms:

| | rate | PSNR |
|---|---|---|
| bbb_1080p | −2.17% to −3.27% | −0.101 to −0.152 dB |
| blue_sky_1080p | −2.18% to −4.12% | −0.094 to −0.214 dB |
| kristensara_720p | −2.47% to −3.97% | −0.123 to −0.156 dB |

**BD-rate of the feature against no feature: −0.35%, +4.20%, −0.79% — mean +1.02%.** It costs
about a percent on average and does not agree with itself on the sign. Three points per arm is a
thin ladder and `blue_sky` dominates the mean, so the honest reading is *neutral*, not *harmful*.

Either way it does not pay for a divergence.

## What was chosen

**Off by default, kept behind `flags` bit 1 (`GNC_SPARSE_DZ=1`).** All three quantise paths now
agree, which is what BUG-16 asked for.

**Rejected: delete it.** Tempting on simplicity grounds, and GOALS §5 does say prefer the simpler
option when results are similar. But three quality points on three stills is thin evidence to
destroy a deliberate feature with documented reasoning, and the flag costs one bit and one `if`.
Whoever re-prices it on a real ladder should not have to re-implement it first.

**Rejected: port it to the other two quantisers.** This is what "make them agree" would normally
mean, and it is the right move for a feature that *pays*. At BD-rate +1.02% it does not, so
porting would spread a wash across two more code paths.

**Rejected: leave it and document the divergence.** It makes every GPU-arm-versus-CPU-arm
comparison invalid below q=30, which is exactly the trap BUG-16 was filed for, and the repository
already has a list of retracted results that came from harness asymmetries rather than from code.

## Scope, measured rather than assumed

**Only q ≤ 30 is affected.** Output is byte-identical either way at q=40, 50, 75, 85, 90 and 100 —
the dead zone above q≈30 is too narrow for a subband to reach 95% zeros. Two consequences: GNC's
stated home range (q > 85, contribution) is untouched, and BASELINE's q=50/75/90 rows did not need
re-measuring.

**BASELINE's q=25 row moved** from 35.51 dB / 1.60 bpp / VMAF 90.25 to **35.63 dB / 1.64 bpp /
VMAF 90.31**. VMAF is the leading metric at this operating point (CLAUDE.md), and it moved **+0.06
— an improvement**, far inside the 0.5-point tolerance. The `GOALS.md` copy of that table is
updated too.

## Evidence

- GPU and CPU arms agree at q=25: **35.631 dB both**, where they differed by 0.118 dB.
- 4:2:2 at q=75, GPU versus CPU decoded pixels: **max abs difference 0, zero differing pixels of
  6 220 800**, where BUG-16 recorded 1.69. That configuration has fusion active because CfL
  requires 4:4:4, which is why it was the entry's third row.
- `GNC_SPARSE_DZ=1` reproduces the old row to the byte: 415 544 B, 35.51 dB, VMAF 90.25.
- Byte-identity at q ≥ 40 on six quality points.
