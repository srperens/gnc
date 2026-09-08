# 0043 — The inter dead zone is the intra dead zone, not double it

> **Renumbered from 0041 to 0043 on 2026-09-08, and the measurement is untouched.** `dr-0041` was
> reserved through `scripts/claim` at 13:38:33Z by the INTRA-2 session before it wrote its file;
> this record took the number without reserving it. Same asymmetry as the `0024` and `0027` pairs
> COORDINATION records, and the resolution it prescribes — the unreserved record moves. Fixed here
> rather than left for BUG-19 because there were five inbound references and three of them were
> being edited anyway for the INTRA-2 merge.

**Date:** 2026-09-08
**Item:** INTER-2
**Status:** accepted — default changed, `GNC_INTER_DZ_MUL=2.0` restores the old behaviour

## Decision

`inter_dz_mul` goes from **2.0 to 1.0**. Motion-compensated residuals now take the same dead zone
as intra coefficients instead of twice it.

## Why the old value existed, and why it is still not 0.0

The rationale for doubling is in the code and it is sound: on a pure pan the residual is largely
the reference's own quantisation noise, and coding it finely makes the P-frame *better* than the
I-frame it predicts from — bits spent on nothing a viewer asked for. That argument justifies
having an inter dead zone. It does not justify the factor, which was never measured.

**Removing it entirely is worse than shipping**, which is the result that keeps the feature: at
`mul=0.0` bbb_extended measures **+12.48%** BD-rate. So the dead zone earns its place; 2.0 simply
overshot it.

## Evidence

4-rung ladder (q=70/75/80/85), three sequences, 24 frames, ki=9, 4:4:4, Rice. BD-rate on PSNR
against the shipped 2.0, each sequence integrated over the **intersection** of all four arms'
quality ranges — a per-arm interval would compare figures taken over different ranges, which
QUAL-1 measured as worth 47.5 BD-rate points on average:

| sequence | mul=1.5 | **mul=1.0** | mul=0.0 |
|---|---|---|---|
| bbb_extended (animation) | −2.70% | −2.13% | **+12.48%** |
| crowd_run (high motion) | −3.01% | **−6.04%** | −2.40% |
| old_town_cross (camera) | −2.76% | **−6.14%** | −3.34% |
| **mean** | −2.82% | **−4.77%** | +2.25% |

**Worst-frame PSNR — the metric a contribution codec is judged on (QUAL-1) — improves at 12 of 12
points**, by +2.44 to +5.23 dB. There is no point, on any sequence, at any rung, where 1.0 is
worse than 2.0 on either mean or worst-frame.

**The optimum is bracketed, not assumed.** 1.0 beats 1.5 above it and 0.0 below it, on the mean
and on worst-frame at every point.

## What was not chosen

**1.5, the runner-up.** It is better than shipped (−2.82%) and better than 1.0 on animation alone
(−2.70% against −2.13%). It loses on the mean, loses on both camera sequences by more than 3
points, and loses worst-frame at all 12 points. Camera content is the contribution use case
(POSITIONING), and BUG-5 already established the animation-versus-camera split as a content bet
rather than a quality one.

**0.0 — deleting the inter dead zone.** +12.48% on animation, +2.25% mean. The mechanism is real
and worth keeping.

**A tuned constant between 1.0 and 1.5.** The measured curve would support fitting one, and it was
rejected on simplicity: at exactly 1.0 the inter dead zone *is* the intra dead zone, so the special
case disappears rather than becoming a second magic number to justify and re-tune whenever the
intra anchors move. The remaining value between 1.0 and 1.5 is under a point of BD-rate on one
content type, against a permanent maintenance cost.

**VMAF as the deciding metric, despite q ≤ 85 being nominally its range.** It is saturated here
and says so: crowd_run's four rungs span **99.86 to 99.88**, a 0.02-point interval across a
**5.5 dB** PSNR spread, and a BD-rate integrated over it returns +35.41% — not a weak number, not
a number. CLAUDE.md's q≤85 rule was written for stills; GNC's inter ladder at 4:4:4 runs at
4.9–12.0 bpp, where VMAF is saturated well below q=85. On old_town_cross, the one sequence where
VMAF is *not* saturated at the bottom (96.45 at q=70), it reads −6.82% and agrees with PSNR.
**PSNR leads here, and the reason is measured rather than asserted.**

## Scope, measured rather than argued

A dead zone of 0.5 or less is arithmetically a no-op: GNC quantises as `floor(|v|/step + 0.5)`
after a `|v| < dz*step` test, so below 0.5 the test only zeroes what the rounding already zeroes
(`tests/cli_shipped_config.rs::a_dead_zone_of_half_a_step_changes_nothing` asserts this over 4000
values). The preset anchors put `dead_zone` at 0.5 by q=85 and 0.05 by q=92, so the old 2.0 was
active up to q≈88 and the new 1.0 is active only below q=85.

Confirmed by encoding both ways, crowd_run, 12 frames, ki=9:

| q | 84 | 85 | 86 | 87 | 88 | **89** | **90** |
|---|---|---|---|---|---|---|---|
| | differs | differs | differs | differs | differs | **identical** | **identical** |

**q ≥ 89 is byte-identical, and q=100 lossless is byte-identical** (15710346 B either way, since
`dead_zone` is 0.0 there and the factor multiplies zero). The still-image path is untouched by
construction — this factor only ever applied to inter residuals.

## Also fixed: three copies of the default

The factor was written out at **three** sites in `sequence.rs` — the P-frame path and two B-frame
paths — each with its own `std::env::var("GNC_INTER_DZ_MUL") ... unwrap_or(2.0)`. Changing the
default at two of three would have produced a frame-type-dependent quantiser difference, which is
exactly BUG-16's failure mode and BUG-37's shape. It is now `gnc::inter_dead_zone_mul()`, and
`tests/cli_shipped_config.rs` fails if `sequence.rs` reads the variable again.

## What this invalidates

Anything that measured **inter frames at q ≤ 88**. Above that, nothing moves.

- **BASELINE's sequence table (q=75, ki=9)** — already stale twice over (the B-pyramid veto and
  BUG-27's P-frame dequant); this is a third reason. Noted there.
- **BASELINE's headline `+90.5%` BD-rate against x264** uses rungs q=85/92/96/99, so **one rung of
  four moves** and the other three do not. It needs re-measuring and the direction should favour
  GNC, since this change improves the inter RD curve — but that is a prediction, not a result, and
  the number stands as-is until someone runs `meas1_vs_h264.py` again. Flagged in BASELINE.
- **`docs/decisions/0025`'s abac-versus-Rice inter table** — its q=50 and q=75 columns are in
  range; q=90 is not. Both arms move together so the *ratio* may survive, which is precisely the
  kind of thing that has to be measured rather than assumed.

## Noticed while verifying, not caused here

q=100 sequences decode to a worst-frame **7.91 dB** on crowd_run. That is **BUG-39**, already
reserved by the RATE-3 session, and it is byte-identical under both multipliers, so this change
neither causes nor fixes it.
