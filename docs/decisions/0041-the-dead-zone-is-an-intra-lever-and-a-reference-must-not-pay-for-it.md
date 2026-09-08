# 0041 — the dead zone is an intra lever, and a referenced I-frame must not pay for it

**Date:** 2026-09-08
**Item:** INTRA-2
**Status:** accepted

## What shipped

`dead_zone` is floored at **0.6 over q=85..95**, and a new `dead_zone_referenced` carries the
ladder's own value for the two places that must not see the raised one: **an I-frame inside a
P-chain, and the inter residual path.**

| | value at q=90 | who uses it |
|---|---|---|
| `dead_zone` | 0.600 | stills, and I-frames at ki=1 |
| `dead_zone_referenced` | 0.179 | an I-frame other frames predict from, and `res_dead_zone` |

## Why the lever exists at all

GNC quantises as `floor(|v|/step + 0.5)` after a `|v| < dead_zone * step` test, so **any dead zone
at or below 0.5 changes nothing.** The shipped ladder interpolated 0.5 at q=85 down to 0.0 at
q>=96. So GNC had *no dead zone at all* in its own operating range, and INTRA-1 step 2c priced
turning one on at ~3 points of the JPEG 2000 gap.

## Why 0028 could not ship it, and what that measurement was actually measuring

0028 set `GNC_DEAD_ZONE=0.6` and found worst-frame PSNR negative on 9 of 9 sequence points, by up
to 1.93 dB. It attributed that to the dead zone hitting sparse motion-compensated residuals.

**That is right about the mechanism and wrong about the magnitude, because the knob moved two
things.** `res_dead_zone = config.dead_zone * inter_dz_mul` with `inter_dz_mul` defaulting to 2.0,
so `GNC_DEAD_ZONE=0.6` took the *inter* dead zone from the ladder's 1.0 / 0.357 / 0.025 at
q=85/90/95 to **1.2 at all three** — which at q=90 and q=95 turned a dormant no-op into a strongly
active one. Three arms, three sequences, 16 frames, ki=9, worst-frame PSNR against production:

| arm | q=85 | q=90 | q=95 |
|---|---|---|---|
| `GNC_DEAD_ZONE=0.6` (0028's) | −1.10 to −1.60 | −0.85 to −1.62 | −1.53 to −1.93 |
| the same, inter dead zone **held at the ladder's value** | **0.00 / 0.00 / +0.03** | −0.16 to −0.19 | −0.12 to −0.16 |

So roughly 90% of what blocked this item was the inter dead zone, not the intra lever.

## And the residue is still a bad trade, which is what decided the design

Even with inter held, a referenced I-frame costs **−0.19 dB of worst-frame for −0.33% of rate** at
q=90. That is not a tuning accident, it is arithmetic: at ki=9 the I-frame is **one frame in
sixteen**, so its rate saving is diluted 16:1 across the sequence, while worst-frame PSNR — the
metric a contribution codec is judged on (QUAL-1) — is *fully* exposed to it, because **the worst
frame is the I-frame** (49.23 dB against the P-frames' 49.69 on crowd_run at q=90). The cost scales
with the dead zone (−0.01 dB at 0.52, −0.05 at 0.55, −0.19 at 0.60) and is 0.00 dB at q=85, where
the step from today's 0.5 is small.

A frame others predict from has to be **better** than its own rate suggests, not worse, because its
error propagates instead of staying in one picture. That is why encoders give I-frames a QP bonus,
and it is the same conclusion PAD-1 reached from a different lever one commit earlier — its
`pad_fill_decay = false` sits on the line above this one, for the same reason and with the same
ki=1 exemption (`0039`).

## What was rejected

**Ship it globally** — 0028 already refused this, and the residue above says it stays refused.

**Gate on frame type alone (`I` gets it, `P`/`B` do not).** This is what the item proposed, and it
is not enough: it leaves the raised value on an I-frame that a P-chain predicts from, which is
exactly the −0.19 dB case. The gate has to be on *being referenced*, not on being intra. `ki > 1`
is the available proxy and it is exact for the shipped GOP structures.

**Lower the floor to 0.52, where the sequence cost is −0.01 dB.** It also gives up most of the
stills win, and it would still be a dilution trade on a P-chain. Better to take the full lever
where it pays and none of it where it does not.

**Truncation plus a reconstruction offset at `(|q| + r) * step`**, which the item scoped in as "the
principled version of this lever" and which J2K actually uses. Still the right long-term answer,
still needs a decoder change, and now separable from this: the naive wide zero bin is worth −5% on
its own where nothing predicts from the frame.

## Evidence against the item's own success criterion

> ≥2% rate at matched RGB PSNR on four stills at q>=85, with no worst-frame regression on any of
> the three sequences and dE00 no worse.

**1. Stills — BD-rate over q=85..95, five points, against production:**

| bbb_1080p | blue_sky_1080p | kristensara_720p | touchdown_1080p | mean |
|---|---|---|---|---|
| −3.31% | −4.67% | −6.96% | −5.09% | **−5.01%** |

**2. Sequences — no regression is possible, because they are byte-identical.** crowd_run at
q=85/90/95, old_town_cross and bbb_extended at q=90, all ki=9, 8 frames: **identical to the byte**
against a build from before the change. So is a q=75 still (below the floor's range) and a q=100
encode, which is also verified bit-exact (max error 0) — BUG-30's guard holds.

**3. dE00 at matched rate, twelve points, old arm interpolated onto the new arm's rate:**

dE00 mean **−5.2%** (−0.1% to −8.1%), dE00 p95 better on 11 of 12, and YCoCg-R Y-PSNR **+0.39 dB**
mean (+0.04 to +0.63). Better on every axis at every point.

**4. The lever is active where nothing predicts:** an all-intra (ki=1) crowd_run sequence is
**−2.51%**. That content gets less than any of the four stills, which is worth knowing — the lever
is content-dependent and high-motion detail benefits least.

**5. VMAF at q=90 on bbb: 97.06 against 97.07** — a move of −0.01 against a 0.5-point block
threshold.

**Canary:** `GNC_DIAGNOSTICS=1` prints, once per sequence,
`INTRA-2: I-frame is a reference (ki=9), dead zone 0.600 -> 0.179` or
`INTRA-2: all-intra (ki=1), nothing predicts from this frame, dead zone 0.600 kept`. It is printed
*before* the assignment on purpose: a first version printed it after, which showed two identical
numbers and proved nothing.

## Two harness errors worth carrying forward

**The floor must belong to the preset, not to the override.** Applying `.max(0.6)` after
`GNC_DEAD_ZONE` made the knob unable to set anything *below* 0.6 in the range — silently disarming
the one control BUG-30 exists to keep honest. Caught because two arms that should have differed
produced identical files.

**`np.interp` clamps, and clamping flipped a sign.** The new arm's file at q=85 is smaller than
the old arm's *smallest* file, so interpolating the old arm onto that rate returned its endpoint
and compared two different rates. That read as **+3% dE00 and −0.25 dB** — a loss — where
extending the old ladder to q=78..82 and skipping out-of-range rows gives **−3% and +0.26 dB**.
Rows outside the reference ladder are now skipped and labelled, not clamped.

## Interaction with INTER-2, which is in flight

INTER-2 is measuring `inter_dz_mul` and its finding — "the inter dead zone is the intra one, not
double it" — was taken on code where `config.dead_zone` *was* the ladder value. In this record's
terms that is `dead_zone_referenced`, which is what `res_dead_zone` now multiplies. So the two
compose: after both, `res_dead_zone = dead_zone_referenced * 1.0`. Stated explicitly because
"the intra one" means something different once the intra value is floored at 0.6.
