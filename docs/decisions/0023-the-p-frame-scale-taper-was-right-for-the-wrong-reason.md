# 0023 — The P-frame quantiser taper was right for the wrong reason, and its evidence was a bug

**Date:** 2026-09-07
**Status:** Accepted. Closes BACKLOG **INTER-1**. Changes **no default**. Corrects the
justification for TUNE-6's taper, corrects the figures in [0056](0056-the-inter-paths-saving-was-an-equal-setting-figure.md),
and records **BUG-27** as the reason both needed correcting. Files **INTER-2**.

> **On BUG-27's number and its two discoverers.** Filed as BUG-25 in this worktree and renumbered
> per COORDINATION's resolution of that id being used twice — the Vulkan BUG-25 was pushed first,
> and BUG-26 had gone to abac/Rice subsampled chroma by the time the renumbering landed. The
> `arch3` session found the same defect independently while reading both P-frame implementations
> for ARCH-3 ([0025](0025-the-entropy-stage-is-not-a-frame-encoder.md), `a312d6f`), which is worth
> recording for what it says about method: the same fault was reached by *reading two
> implementations against each other* and by *a measurement that refused to make physical sense*,
> and neither route had fixed it. It is also **not** BUG-18 cause 2 — that was the forward
> quantise on the implementation ARCH-3 deleted; this is the dequantise that rebuilds the
> encoder's reference, present in both, so deleting one did not remove it.

## The decision, in three parts

**1. The keyframe interval stays at 9.** Measured at the contribution operating point (q=85–99,
three sequences, 24 frames, 4:4:4), BD-rate against all-intra improves monotonically with GOP
length, so ki=9 is the best of the four intervals tested on *both* mean and worst-frame PSNR:

| sequence | ki=2 | ki=4 | ki=9 |
|---|---|---|---|
| crowd_run | +0.9% / +2.1% | +1.4% / +2.9% | +1.7% / +3.2% |
| old_town_cross | +2.9% / +4.1% | +4.5% / +5.9% | +5.3% / +6.8% |
| bbb_extended | −7.7% / −4.7% | −11.1% / −8.7% | **−12.7% / −10.6%** |
| **mean** | −1.30% / +0.50% | −1.73% / +0.03% | **−1.90% / −0.20%** |

INTER-1 asked this because "if the win only appears at short GOPs, the default is wrong rather
than the feature." The win appears at *long* GOPs. The sign is set by content, not by GOP length —
the camera sequences lose at every interval, the animation wins at every interval — so no default
fixes both classes and moving it only trades one for the other.

**2. The P-frame quantiser scale taper stays exactly as it is** — 1.25 at quantiser step ≥ 4.6,
tapering to 1.0 at ≤ 2.8 — but for reasons that are now measured rather than inherited. The two
ends are right for *different* reasons, which is why the shape is a taper and not a constant:

| | scale 1.00 | taper | scale 1.25 | scale 1.50 |
|---|---|---|---|---|
| q=25–70, mean PSNR | −0.8% | **−4.5%** | −4.5% | −5.3% |
| q=25–70, worst-frame | +18.0% | **+13.7%** | +13.8% | +18.7% |
| q=85–99, mean PSNR | −3.0% | −3.0% | −3.4% | −3.6% |
| q=85–99, worst-frame | −1.2% | −1.2% | **−2.3%** | −1.5% |

- **Below q=85, 1.25 wins on efficiency** — 3.7 points of mean BD-rate and 4.3 of worst-frame over
  a flat 1.0, with 1.50 overshooting on the worst frame.
- **Above q=85, 1.0 wins on the ceiling, and efficiency is deliberately declined.** 1.25 is 1.1
  points better on worst-frame BD-rate up here and is still not taken, because at scale 1.0 the
  worst frame in a GOP equals what all-intra reaches at the same q, and above 1.0 it does not:

  | scale | crowd_run q=99 worst-frame | vs all-intra (59.49 dB) |
  |---|---|---|
  | 0.90 | 59.50 | +0.01 |
  | **1.00** | **59.50** | **+0.01** |
  | 1.25 | 57.05 | −2.45 |
  | 1.50 | 54.99 | −4.50 |

  Identical to two decimals on old_town_cross and bbb_extended. The mechanism is simple: at 1.0
  the P-frames sit at or above the I-frame, so the I-frame is the floor and the floor is the
  all-intra result; any coarser scale pushes the P-frames below it. For a codec whose output is
  re-encoded downstream (GOALS §1), *never being worse than your own all-intra mode* is worth more
  than 1.1 points of BD-rate.

**3. Inter stays a default and is not demoted to opt-in.** See "What was not chosen".

## Why the old justification has to go

TUNE-6's comment in `sequence.rs` justified the taper with measurements like "old_town q=99:
**−3.8 dB avg, −14.2 dB worst** for scale 1.25". **Those were measuring BUG-27, not the trade.**

`encode_pframe` quantises P residuals at `res_qstep = quantization_step * p_qp_scale` and records
that for the decoder; its six *local-decode dequantise* dispatches read `config.quantization_step`
— both encode paths, all three of luma / 4:2:0 chroma / 4:2:2 chroma. The encoder therefore
rebuilt its reference with the intra step while the bitstream carried the residual at `res_qstep`,
so its reference differed from the decoder's by `quantization_step / res_qstep` and every P
predicting from another P inherited a picture no decoder holds. Fixed in `2224c50`.

The tell was a monotone PSNR ramp down each GOP resetting at every I-frame, with the **first** P
correct in both directions (crowd_run q=90 ki=9 scale 1.25: 47.95 / 41.10 / 38.27 / 36.46 / 35.63
/ 35.12 / 34.61 / 34.14, then 49.24 at the next I). Only a P predicting from a P can inherit a
wrong reference. After the fix that GOP is flat at 47.95.

**The defect was invisible at the operating point this project cares about, and that is why it
lasted.** The taper returns exactly 1.0 for every quantiser step at or below 2.8 — which is q=85
and up — so the wrong value and the right one coincide there and the whole contribution range was
correct by coincidence. Verified, not assumed: **27/27 byte-identical** across three sequences ×
ki=2/4/9 × q=85/92/99 before and after the fix, against a pinned `07c01b1` build. Below q=85 it
was live on the shipped default, and worth up to **+1.82 dB mean / +3.62 dB worst-frame for +2.0%
bytes** (crowd_run q=70).

## What this corrects elsewhere

**Decision 0056 and MEAS-3.** Their ladder is q=25–95, so most of it ran through the defect.
Re-run on the same harness, same 18 frames, same sequences:

| sequence | mean: 0056 → now | worst-frame: 0056 → now |
|---|---|---|
| crowd_run | +15.9% → **+6.5%** | +32.4% → **+12.0%** |
| old_town_cross | +22.2% → **+19.3%** | +35.4% → **+28.7%** |
| bbb_extended | −24.2% → **−26.8%** | −10.5% → **−16.5%** |
| **mean** | **+4.6% → −0.3%** | **+19.1% → +8.0%** |

0056's *shape* survives — inter still costs +8% on the worst frame, nearly all of it
old_town_cross — but its magnitude does not, and **INTER-1's own title ("the inter path is a loss
at contribution quality") does not survive at all**: at q=85–99 the shipped configuration is
−1.9% mean / −0.2% worst-frame. 0056's stated reasons for the *old* figure being wrong remain
correct and untouched; it simply also had a bug underneath it.

**TUNE-5.** Its flat-1.25 "−3.3% BD-rate at ki=9" was measured at q=15–50, entirely inside the
live range. Re-measured here as −4.5% mean against a flat 1.0's −0.8%, so **TUNE-5's conclusion
survives its own invalidation** on corrected code.

**Not affected:** anything at q ≥ 85 (byte-identical), every still, QUAL-1's +90.5% (its ladder is
q=85/92/96/99), and INTER-1's own ki sweep. BASELINE's q=75 sequence table *is* inside the range,
but it was already marked stale for an unrelated reason (BUG-5's B-pyramid veto) and still needs
re-measuring.

## What was not chosen

- **Making inter opt-in, or all-intra the default.** Rejected on the corrected measurement, not on
  principle. The case for it was 0056's +4.6%/+19.1%; about 60% of the worst-frame penalty was
  BUG-27, and at the contribution operating point the shipped configuration is a wash tilted
  slightly *toward* inter. It also wins −12.7% on animation. Separately, GOALS §1 rejects going
  all-intra as a positioning matter and LOOP.md reserves that for escalation — but the measurement
  does not ask for the change, so no escalation was needed.
- **A flat 1.25 everywhere.** Simpler — one constant instead of a taper — and better on mean PSNR
  at q≥85 (−3.4% against −3.0%). Rejected: it caps the top rung 2.45 dB below all-intra on every
  sequence tested, and the top rung is what a contribution codec is bought for.
- **A flat 1.0 everywhere.** Also simpler, and it is what INTER-1 predicted would be right.
  Rejected: it gives up 3.7 points of mean BD-rate and 4.3 of worst-frame below q=70.
- **Chasing the inter dead zone.** `inter_dz_mul=1.0` at q=85 restores the worst frame to exactly
  all-intra's 47.48 dB for 12.7% more bytes than the shipped 2.0. That is a large lever and a
  *point* measurement, which cannot rank the options (COORDINATION rule 4). Filed as **INTER-2**
  for a proper BD-rate rather than settled here; a claim taken for one item does not cover what
  you trip over inside it.
- **Changing the taper's breakpoints (4.6 / 2.8).** Not measured. The sweep tested scale values at
  two ends of the ladder, not where the transition should sit, and the ceiling argument only says
  the scale must be 1.0 by the time the codec is asked for near-lossless output — it does not say
  q=85 is the right place to arrive there. Left open deliberately rather than tuned by eye.

## The uncomfortable part

This is the third headline inter figure to be withdrawn in as many days, and the second time the
*instrument* was the problem rather than the codec — after the `chroma_weight` sign reversal, the
inflated `byte_size()`, the 8-bit-truncated 10-bit run. The new wrinkle is where it hid: **a knob
that is a no-op at the operating point you test cannot be validated by testing that operating
point.** `p_qp_scale` is 1.0 for all q ≥ 85, so every contribution-quality measurement in this
repository was structurally blind to a defect that exists only where the knob acts, and the range
where it acts was measured twice (TUNE-5, TUNE-6) with the defect present both times.

Two habits follow, both cheap. **Bound the direction you expect to lose.** Forcing the scale
*below* 1.0 was run only to rule out a direction the mechanism argued against, and it is what
turned a measurement into a bug hunt: it spent 4% more bits for 5 dB less quality, and
more-bits-for-worse-quality has no reading as a bad trade. Had only 1.25 and 1.50 been run, the
divergence would have been published as a rate/quality curve — coarser-and-worse is exactly what
one expects to see. **And report the top rung beside any common-interval BD-rate.** Normalising
four arms of unequal reach onto one interval is the right fix for comparing them, and it silently
deletes the evidence about reach — which is the whole of decision 2 above.
