# 0061 — The B-pyramid's leaf quantiser scale is right below q≈80 and wrong above it

Date: 2026-09-08
Item: BACKLOG MEAS-2 (P3) — the last two of its five toggles
Status: accepted
**Changes a default.** `GNC_PYRAMID_L3_QP_SCALE` was a constant 1.5 and is now a taper on the
quantiser step: 1.5 at step ≥ 4.0 (q ≤ 75), 1.0 at step ≤ 2.8 (q ≥ 85), linear between. The env
var still overrides, and the canary now says which of the two the encoder read. `GNC_PYRAMID_L2_QP_SCALE`
stays 1.0 — measured, and off is the right answer. **Nothing changes unless `GNC_B_PYRAMID=1`**:
the pyramid has been off by default since BUG-5, so no shipped default output moves.

**Tree:** `7b58071` (main `d44bc39` merged). Every figure at q ≥ 85 was re-taken here after
LOSSLESS-3 (`0073`) landed; the q ≤ 75 figures were taken on `a73e0a2` and **verified
byte-identical on this tree** at four points, so they carry over rather than being re-run.

## What was open

MEAS-2 measured five toggles and left two, both behind BUG-5's pyramid-off default:

- `GNC_PYRAMID_L3_QP_SCALE`, default 1.5 — the leaf B-frames (B₁,B₃,B₅,B₇), which nothing
  references. The comment justified it as "matches H.264 QP+4 practice for inner B-frames", which
  is true and is not a measurement.
- `GNC_PYRAMID_L2_QP_SCALE`, default 1.0, commented "off until validated". It never was validated,
  and **it could not have been**: `[pyramid_b] layer=2` printed the reference indices and not the
  quantiser, so there was no way to observe whether the encoder had read the variable at all. The
  layer-3 line printed both. That asymmetry is the whole reason one knob has a week-old default
  nobody could check.

## The measurement

`scripts/meas2_pyramid_qp.py`. Six arms — the pyramid suppressed, plus five l2:l3 pairs — on
crowd_run, old_town_cross and bbb_extended, 24 frames, 4:4:4, **ki=17 in every arm**.

Two choices in that sentence matter. **The reference is the pyramid off at the same keyframe
interval**, not P-only at ki=8: BUG-5 compared ki=17-with-B against ki=8-P-only, which moves GOP
length and the B-pyramid together — the conflation TUNE-1 was warned about — so no figure here is
comparable with BUG-5's. And **every arm is integrated over one common quality interval per
sequence**, because a coarser leaf lowers the top of the arm's ladder and QUAL-1 measured what
ranking arms over different intervals costs: 47.5 points of movement on average.

### Below the fine end, 1.5× is the best of the three — on *both* metrics

q=30/50/65/75, BD-rate against the pyramid off, negative = fewer bits:

| | l3=1.0 | **l3=1.5** | l3=2.0 |
|---|---|---|---|
| VMAF | −3.9% | **−12.1%** | −11.5% |
| worst-frame PSNR | +10.0% | **+1.8%** | +13.4% |
| mean PSNR | −7.9% | −10.3% | **−11.1%** |

VMAF leads here (q ≤ 85) and it agrees with the worst frame, which is the unusual and convincing
part: 1.5× is not buying the mean at the tail's expense down there, it is better on the tail than
either neighbour. crowd_run's VMAF was discarded by the harness's saturation guard (overlap
99.53–99.87); old_town_cross and bbb_extended carry the figure (−5.1% and −19.0%).

### At contribution quality the same 1.5× is a pure move along the RD curve

q=85/88/90/92/94 on this tree, BD-rate against the pyramid off:

| | l3=1.0 | l3=1.5 | l3=2.0 | l2=1.25 l3=1.5 | l2=1.5 l3=1.5 |
|---|---|---|---|---|---|
| mean PSNR | −4.6% | −5.0% | −4.9% | −4.9% | −4.8% |
| worst-frame PSNR, bbb_extended | **−14.4%** | −5.6% | +7.3% | −7.4% | −5.7% |
| worst-frame PSNR, camera clips | — | — | — | — | — |

Mean PSNR cannot separate the five arms: 0.4 points of spread over a 4.6–5.0% range. The worst
frame separates them by 22 points, and **the two camera clips have no overlap to integrate at
all** — that is the em-dash row, and it is a stronger statement than a BD-rate. On crowd_run the
pyramid-off arm's *worst* rung reaches 47.48 dB on its worst frame; the l3=1.5 arm's *best* rung
over the whole q=85–94 ladder reaches 46.71. The tail penalty exceeds the entire ladder's quality
span, so no amount of extra rate inside the range buys it back.

At matched q the same thing without any fitting, crowd_run q=85:

| arm | bytes | mean PSNR | worst frame |
|---|---|---|---|
| pyramid off | 74 997 760 | 47.91 | **47.48** |
| l3=1.0 | 74 809 828 | 47.91 | **47.48** |
| l3=1.5 (was the default) | 70 692 750 | 46.82 | 44.65 |
| l3=2.0 | 67 875 419 | 46.03 | 42.29 |

−5.7% of the bits for **−2.83 dB on the worst frame**. For a contribution codec the worst frame is
what survives the next re-encode (GOALS §1), so that is the wrong side of the trade.

### Why the direction flips, and why the taper is keyed on the step

At q=30 the leaf's quantiser step is 13.9 and its error is dominated by what bi-prediction failed
to predict; coarsening the quantiser adds little to an error that large, so the rate saving is
nearly free. At q=90 the step is 2.24 and the leaf's error *is* its own quantiser, so a 1.5×
scale converts directly into 2.8 dB on that frame. Same knob, opposite regimes — the same shape
as AQ (off below q=30) and Rice-vs-rANS (boundary at q=20), and MEAS-2's own recurring finding
that a toggle measured at one operating point is not measured.

Keyed on the quantiser step rather than on `q`, for `p_qp_scale`'s two reasons: the step is the
physically relevant quantity, and `q` is not available at that site at all under `--qstep` or rate
control. The breakpoints are the measured ones — step 4.0 is q=75 and 2.8 is q=85 — and the ramp
between them is interpolation, since **nothing was measured in that gap**. The step read is the
one actually being scaled, so under rate control the taper keys on `estimate_qstep()` rather than
on the config the sequence started with.

## What was not chosen

- **Flipping the default to a constant 1.0.** That was the draft conclusion after the q ≥ 85 half
  and it is wrong: it costs 8.2 points of VMAF BD-rate and 8.2 points of worst-frame BD-rate in
  the lossy range. Measuring one operating point would have shipped it.
- **Leaving 1.5 and documenting the boundary.** Defensible — the knob only reaches users who set
  `GNC_B_PYRAMID=1`. Rejected because the boundary sits inside GNC's own home range, so the
  documented advice would be "the default is wrong for what this codec is for", and the fix is
  twelve lines mirroring a taper that already exists two hundred lines below.
- **l3=2.0 anywhere.** Best on mean PSNR in both ranges (−11.1% and −4.9%) and worst on the tail
  in both (+13.4% and +7.3%). It is the clearest single illustration that mean PSNR cannot referee
  this knob.
- **Any l2 scale.** At contribution quality 1.25 and 1.5 move mean PSNR by 0.1–0.2 points and cost
  the tail (bbb_extended −7.4% and −5.7% against l3=1.0's −14.4%). Validated as off; **the lossy
  half of l2 is not measured** and is what is left of MEAS-2.
- **Reversing BUG-5's pyramid-off default.** Not on this evidence, and the item was not asked to.
  Two things are worth recording for whoever does: at matched ki and with the leaf scale off, the
  pyramid is cheaper than P-only on all three clips on both metrics (bbb_extended −12.2% mean and
  −14.4% worst-frame), so part of what BUG-5 measured was GOP length rather than B-frames. And it
  changes nothing today, because the pyramid also costs 8 frames of reordering latency (`0033`)
  and because on this content inter is *itself* losing: on crowd_run at q=85 the all-intra arm
  costs 69 919 536 bytes against I+P's 74 997 760, so every arm in this record sits inside a
  configuration that is 7.3% behind not coding inter at all.

## Verification

Four points, each byte-exact against the arm it should reproduce:

| what | bytes | equals |
|---|---|---|
| default, pyramid off, q=90 | 82 057 506 | unchanged from before the change |
| `GNC_B_PYRAMID=1`, q=50, taper | 26 688 057 | the l3=1.5 arm — nothing moves below the fine end |
| `GNC_B_PYRAMID=1`, q=90, taper | 81 869 091 | the l3=1.0 arm exactly |
| `GNC_B_PYRAMID=1`, q=90, env=1.5 | 77 632 449 | the old default, so the override still reaches it |

Canary: `[pyramid_b] … layer=3 … qstep=2.24 (taper, l3_scale=1.00x, taper default 1.00)`, and
`(env, l3_scale=1.50x, taper default 1.00)` when the variable is set — INTER-1's rule that "the
knob did nothing" and "the knob was never read" must not be confusable. Layer 2 now prints its
quantiser and scale too, which is what made its default checkable at all.

## A trap this run walked into, and the guard it left behind

The first q ≥ 85 table was taken at q=85/90/92/95/99 on `a73e0a2`. LOSSLESS-3 (`0073`) then landed
on `main` and emits a q=95..99 4:4:4 camera sequence **bit-exact** when the lossy encode is
larger — so on the current tree those two rungs return `psnr = inf`, and a Bjontegaard fit over an
infinity is a silent non-number. Worse than INTRA-1's version of the same trap: a bit-exact encode
is all-intra, so every arm's rung is *the same bytes* and the toggle under test is not in the
output being compared. `finite_or_die` in the harness now drops and reports a non-finite rung, and
the re-taken ladder stops at q=94. The re-take also confirmed the merge changed nothing in range:
all 30 q ≤ 94 rungs are byte-identical across it.
