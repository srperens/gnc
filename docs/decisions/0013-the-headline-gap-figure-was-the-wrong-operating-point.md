# 0013 — The headline compression figure is +90.5%, and 5–7x was the wrong operating point

**Date:** 2026-09-06
**Status:** Accepted (supersedes MEAS-1's figure as the current number; MEAS-1 stands as history)

## The decision

The project's headline compression claim changes from **"GNC needs 5–7x the bitrate of H.264"** to
**"+90.5% BD-rate on PSNR at contribution quality — about 1.9x."** Nothing in the codec changed
between the two measurements. The old figure was taken at distribution bitrates (crf 18–38), which
GOALS §1 says is not what this codec is for, and with the quality ladder above q=92 dead.

Corrected in GOALS, BASELINE, README, POSITIONING and BACKLOG. MEAS-1's entry keeps its numbers
with a superseded note rather than being edited, because the figure was correct for the range it
measured.

## Why this is a decision and not just a number

Because of what the old number was used to justify. "The gap is multiples, not percentages. Work
targeting single-digit-percent improvements is not addressing it" was written into the backlog on
the strength of it, and POSITIONING carried a matching dead-end entry: *any expectation that a
transform change closes a 5–7x gap*, since no published transform result on motion-compensated
residuals exceeds ~15%.

That reasoning is arithmetically fine and was applied to the wrong denominator. **Against 1.9x, an
accumulation of single-digit wins reaches the target; against 5–7x it provably could not.** Every
"too small to bother with" judgement in this repository predating today was made against a figure
that was off by 3x, which is a larger correction than most of the effects those judgements
dismissed.

## The measurement

`scripts/meas1_vs_h264.py`, one normalised reference through PNG for both codecs, the same `vmaf`
binary and arguments for every score, rate from the actual coded bitstream. 1920x1080, 17 frames,
ki=9, 4:2:0, 8-bit, x264 at defaults — MEAS-1's parameters exactly, only the operating point
changed. q=85,92,96,99 against crf=1,2,4,8.

| sequence | BD-rate PSNR-Y | overlap |
|---|---|---|
| bbb_extended | +129.0% | 49.4–55.9 dB |
| old_town_cross | +71.9% | 50.0–56.2 dB |
| crowd_run | +70.6% | 49.9–56.3 dB |
| **mean** | **+90.5%** | |

A canary ran first, because the whole point was that the high-q ladder had been dead: q=85/92/96/99
give 49.48 / 51.69 / 55.19 / 59.77 dB, monotone, rate climbing 7.05 → 14.28 bpp. Without that
check the re-run would have measured the same picture four times, which is what the original did.

## Why PSNR leads, with a magnitude attached

The first pass used q=92,96,99 against crf=4,8,12 and left only **1.8 dB** of curve overlap. Both
ladders were extended for 6.5 dB. What the two metrics did when the window widened:

| sequence | VMAF: narrow → wide | shift | PSNR: narrow → wide | shift |
|---|---|---|---|---|
| bbb_extended | 126.5% → 122.6% | −3.9 | 131.1% → 129.0% | −2.1 |
| old_town_cross | 81.1% → **191.4%** | **+110.3** | 72.0% → 71.9% | −0.1 |
| crowd_run | 85.4% → 113.6% | +28.2 | 71.4% → 70.6% | −0.8 |

**Mean absolute shift: VMAF 47.5 points, PSNR 1.0 point.** On old_town VMAF reads 99.62–99.68
across a 6 dB PSNR spread — there is no signal left to integrate, so the BD-rate fits noise.
CLAUDE.md's blanket "VMAF is the primary quality metric" was written for the lossy range and is
wrong above it; it is now a table keyed on operating point, and COORDINATION's rule 3 no longer has
to contradict it.

## Colour, and why one number will not do

At rate matched to 1%, CIEDE2000 on decoded RGB:

| sequence | GNC dE00 mean / p95 | x264 dE00 mean / p95 |
|---|---|---|
| bbb_extended | **0.611** / 1.304 | 0.684 / 1.503 |
| old_town_cross | **0.911** / 1.943 | 0.949 / 2.196 |
| crowd_run | 0.837 / 1.844 | 0.913 / 2.195 |

GNC is ahead on colour on all three, with fewer pixels past the JND, **while losing luma by
7.4–8.8 dB at those same points.** The two codecs allocate rate differently between luma and
chroma. So a single luma BD-rate overstates the gap for a colour-critical use case and understates
the luma deficit — quote both, or the comparison misleads in whichever direction suits the arguer.
(crowd_run's pair is 1.08x on rate, so its colour win is partly bought; the two matched pairs give
+4.1% and +10.7%.)

That asymmetry was the obvious cheap explanation for the luma gap, and it was tested and
eliminated — see 0014.

## A correction to MEAS-1's record

MEAS-1 states 17 frames on bbb / touchdown / old_town. `bbb.y4m` in the tree has **8 frames**, and
there is no `touchdown` sequence at all; at those parameters the GNC arm dies on a missing PNG,
which is what happened on the first attempt here. Either it ran against sources no longer present
or against fewer frames than recorded, so **its absolute figures are not reproducible as written.**
Sources used here are stated explicitly and confirmed distinct by first-frame MD5.

## Note on what did not change

The +306% to +617% distribution-bitrate figures are unaffected and remain correct for that range.
GNC is worse at distribution bitrates than at contribution quality, which is the expected shape for
a codec with no context modelling and 256 independent streams per tile — and it is the operating
point the project has explicitly chosen not to serve.
