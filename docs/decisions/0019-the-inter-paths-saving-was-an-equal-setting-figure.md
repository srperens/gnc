# 0019 — The inter path's saving was an equal-setting figure, not a saving

**Date:** 2026-09-07
**Status:** Accepted, and **its numbers are superseded by [0023](0023-the-p-frame-scale-taper-was-right-for-the-wrong-reason.md)
(2026-09-07).** Corrects GOALS §4's "saves 17–27% vs all-I"; the default is **not** changed
here — what to do about it was BACKLOG **INTER-1**, now done.

> **Superseded figures.** The table below was measured over q=25–95 with **BUG-27** live: the
> encoder's P-frame reference was dequantised with the *intra* quantiser step, so it disagreed
> with the decoder's at every q ≤ 80. Re-run after the fix, on the same harness, frames and
> sequences: crowd_run **+6.5% / +12.0%**, old_town_cross **+19.3% / +28.7%**, bbb_extended
> **−26.8% / −16.5%**, mean **−0.3% / +8.0%** (was +4.6% / +19.1%). About 60% of the worst-frame
> penalty was the defect. **Everything else in this record stands** — the three reasons the *old*
> 17–27% figure read the other way are unaffected, and so is the rule that both PSNR columns must
> be quoted. What changes is the magnitude of the verdict, and with it the answer to the question
> this record handed to INTER-1: inter stays a default. See 0023.

## The decision

GOALS §4 records the I/P/B inter path as saving **17–27% against all-intra**, from the 2026-03
ablation sweep ("crowd_run all-I 7.32 bpp against current 5.34 bpp, −27.0%"). That figure is
**withdrawn as a saving and kept only as what it is** — a rate comparison at equal settings.

Measured as BD-rate against all-intra on three sequences, 18 frames each, q=25–95, 4:4:4:

| sequence | mean PSNR | worst-frame PSNR |
|---|---|---|
| crowd_run | **+15.9%** | **+32.4%** |
| old_town_cross | **+22.2%** | **+35.4%** |
| bbb_extended | **−24.2%** | **−10.5%** |
| **mean** | **+4.6%** | **+19.1%** |

Positive means the inter arm needs *more* bits for the same quality. So the shipped configuration
is a wash on the mean, a clear loss on the frame that matters for contribution, and a real win on
one sequence — low-motion animation.

## Why the old figure read the other way

Three reasons, and each is a failure mode this repository has already written down.

1. **It compared rates at the same q, not at the same quality.** At q=70 on crowd_run the inter arm
   spends 4.74 bpp against all-intra's 7.96 — 40% cheaper — **while sitting 7.6 dB lower** (34.50
   against 42.07). That is a different operating point, not a saving. COORDINATION rule 4.
2. **Its quality evidence was a saturated metric.** The ablation offered VMAF 99.09 against 99.10
   as proof the quality matched. QUAL-1 later measured what VMAF does up there: a BD-rate moved
   47.5 points on average, 110 at worst, when the ladder widened. COORDINATION rule 3.
3. **The inter arm is coarser by construction and nothing accounted for it.** TUNE-6 scales the P
   frame quantiser 1.25× at step ≥4.6. That is a deliberate quality-for-rate trade inside the arm
   being credited with a rate win, so any equal-q comparison double-counts it.

## What was not chosen

- **Changing the keyframe-interval default, or turning inter off.** Rejected on evidence, not on
  principle: three sequences at 18 frames and a single ki cannot carry a default change, and the
  one sequence where prediction is easy shows −24.2%. MEAS-4 located the inter gap in *prediction
  quality*, which is fixable; an architecture verdict from this data would be overreach. INTER-1
  says what to measure first — a ki sweep at q=85–99, and TUNE-6's scale re-priced at 1.0 for the
  contribution range.
- **Reporting the mean-PSNR BD-rate alone.** Rejected: +4.6% reads as "roughly neutral" and hides
  that the inter arm gets there by making some frames much worse. On crowd_run at q=70 it reads
  34.50 dB mean against a 32.08 dB worst frame; all-intra reads 42.07 / 42.06, flat. A contribution
  codec's output is re-encoded downstream, so the worst frame sets what survives. Both columns are
  quoted, always.
- **Quoting crowd_run's VMAF BD-rate.** It came out **+132.4%**, the most dramatic number in the
  run, over an overlap of 99.55–99.84 — a saturated range with nothing to integrate. Discarded, and
  the harness now prints the overlap beside every VMAF BD-rate and refuses one whose floor exceeds
  99, because MEAS-3's own q≤85 cap was *not* sufficient: the all-intra arm is already saturated at
  q=25.
- **Editing GOALS' targets.** Only the measured claim is annotated. What the project is *for* is
  not a measurement's call (LOOP.md's escalation rule), and the "H.264 saves 60–70%" half of that
  sentence was not re-measured here.

## The uncomfortable part

This is the second headline figure in two days to survive only as an equal-setting comparison — the
first was the Rice-vs-rANS pair withdrawn in 0015 and re-measured in 0018. Both were quoted for
months. The common shape: a number taken at fixed settings, blessed by a metric that had no
resolution left at that operating point, then carried forward as though it were a rate-quality
result. **Any figure in this repository of the form "X% smaller" should be read as suspect until it
names either a BD-rate or a matched-quality point.**
