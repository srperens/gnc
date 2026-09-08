# 0045 — abac's inter saving decays with quality, and its context headroom is not inter-specific

**Date:** 2026-09-08
**Item:** ENT-3
**Status:** accepted — no default moves; two figures added, one option rejected, one published table corrected

## Decision

ENT-3's three open questions are closed:

1. **The frame mix behind `0025`'s nine-point table is `2I+16P+0B`.** Read out of the runs, not
   reasoned about.
2. **abac against Rice on inter is measured through the contribution range.** The saving is real at
   every point and it **decays monotonically with quality** — P-frame bytes go from −20.6%/−12.2%/−11.9%
   at q=90 to **−14.5%/−4.3%/−3.7% at q=99**.
3. **Retuning abac's contexts for residual statistics is rejected as a separate opportunity.** The
   headroom above a richer-neighbourhood bound is the *same* on inter as on intra — slightly
   smaller, in fact — so there is nothing inter-specific to collect.

And one correction that fell out of the reproduction: **`0025`'s q=50 and q=75 columns do not
reproduce on today's `main`, and its q=90 column reproduces exactly.**

## Why the mix had to be read rather than derived

At ki=9 the codec can code either `2I+16P+0B` or `2I+2P+14B`: `quality_preset()` sets
`b_pyramid: false` unless `GNC_B_PYRAMID=1` (`src/lib.rs:1021`) while `CodecConfig::default()`
sets it `true` (`src/lib.rs:571`). Those are very different residual statistics and `0025` records
neither.

Three independent readings, all `2I+16P+0B`:

- `benchmark-sequence -q 90 -n 18 -k 9` prints `2I+16P+0B` on stdout, and emits
  `GNC: B-pyramid suppressed (ki=9 would allow it)` on stderr — the veto firing, not inferred.
- `encode-sequence` prints `Encoded 18 frames (2I + 16P)`, on all 36 encodes of the sweep below.
- `encode-sequence`'s `-q` **defaults to 75**, so that path cannot reach
  `CodecConfig::default()` at all; the pyramid is unreachable there without the environment
  variable. (`benchmark-sequence`'s `-q` is `Option<u32>` with no default — BUG-37 — which is the
  trap this check had to avoid, and did, by passing `-q`.)

So whichever of the two subcommands produced `0025`'s table, it was `2I+16P+0B`. The q=90
reproduction below confirms it by evidence rather than by elimination.

## The measurement

`scripts/ent3_abac_inter.py`, binary pinned and hash-recorded (`364e6aaf…` at `f3f7254`), 18
frames, ki=9, 4:4:4, `.gnv` container bytes. abac against Rice, negative meaning abac is smaller.

**Domain declaration.** Both arms quantise identically and differ only in the entropy stage, so
what is compared is the coded size of the *same* quantised wavelet coefficients — of the
motion-compensated residual on a P frame, of the frame itself on an I frame. Since ARCH-3 the
entropy choice cannot reach the pixels (`0025`), and that premise is **checked, not assumed**:
every point decodes both arms and hashes all 18 frames. 18 of 18 frames identical at all 18
points, so no BD-rate is needed and none is quoted.

**The container ratio is not an inter figure, and `0025`'s was one.** At ki=9 the file is 2 I-frames
and 16 P-frames, so a whole-file ratio mixes abac's already-known intra saving into the answer.
`encode-sequence` tags every frame's byte count `[I]` or `[P]`, so the two halves are summed
separately here. The **P column is ENT-3's actual question.**

### P-frames only

| sequence | q=50 | q=75 | q=90 | q=95 | q=97 | q=99 |
|---|---|---|---|---|---|---|
| bbb_extended | −16.1% | −21.1% | −20.6% | −18.2% | −16.6% | −14.5% |
| crowd_run | −18.3% | −14.3% | −12.2% | −9.6% | −7.5% | −4.3% |
| old_town_cross | −21.6% | −14.7% | −11.9% | −9.5% | −7.2% | −3.7% |

### I-frames only, same runs

| sequence | q=50 | q=75 | q=90 | q=95 | q=97 | q=99 |
|---|---|---|---|---|---|---|
| bbb_extended | −17.8% | −14.7% | −14.2% | −12.3% | −11.1% | −9.5% |
| crowd_run | −16.0% | −12.7% | −11.4% | −8.6% | −6.4% | −3.3% |
| old_town_cross | −20.2% | −14.5% | −12.7% | −10.2% | −8.0% | −4.7% |

### Whole container, which is the column comparable to `0025`

| sequence | q=50 | q=75 | q=90 | q=95 | q=97 | q=99 |
|---|---|---|---|---|---|---|
| bbb_extended | −16.6% | −20.0% | −19.7% | −17.5% | −15.9% | −13.9% |
| crowd_run | −18.0% | −14.1% | −12.1% | −9.5% | −7.3% | −4.2% |
| old_town_cross | −21.5% | −14.6% | −12.0% | −9.6% | −7.3% | −3.8% |

**Re-taken after RATE-3, and the answer survives (2026-09-08, same session).** RATE-3 landed
between the measurement and the merge, lifting the lossless-fallback gate for sequences at
q=95..=99. Every table above is pinned to `f3f7254` and stays correct for it; re-run on a pinned
post-RATE-3 binary (`ba9e1c6e…` at `56c7b6c`):

- **The control passes exactly.** At q=90, I-frame *and* P-frame byte counts are **equal integers**
  for both coders on all three sequences, so RATE-3 does not reach q=90 and the re-run is
  comparable rather than merely similar.
- **The P column — this item's answer — is unchanged.** Largest move 0.25 points (bbb_extended
  q=99, −14.49% → −14.24%); the other eight q≥95 points move by ≤0.03. The decay stands: q=99 is
  −14.2% / −4.3% / −3.7%.
- **The I and container columns move, and RATE-3's own mechanism is visible in the bytes.**
  I-frames shrink 15.3–32.3% at q≥95 while P-frames grow 0.68–1.56% — a bit-exact reference
  carries detail a lossy one had quantised away, so the residual against it is bigger, exactly as
  `0040`/RATE-3 describe.

| I-frames only, post-RATE-3 | q=90 | q=95 | q=97 | q=99 |
|---|---|---|---|---|
| bbb_extended | −14.2% | −12.3% | −11.1% | −14.3% |
| crowd_run | −11.4% | **−9.9%** | **−9.9%** | **−9.9%** |
| old_town_cross | −12.7% | **−11.2%** | **−11.2%** | **−11.2%** |

**That constant column is a finding, not a rounding artefact.** crowd_run's Rice I-frame is
6 492 092 bytes at q=95, q=97 *and* q=99 — the same integer — and old_town_cross's is 6 352 476 at
all three. Above the RATE-2 crossover the kept I-frame is the **bit-exact lossless** candidate, so
its size stops being a function of q, and the entropy-coder ratio measured on sequence I-frames
there is a comparison of the **lossless** path rather than the lossy one. Anyone measuring a coder
on sequence I-frames at q≥95 after RATE-3 is measuring something different from what they measured
before it, and the container column inherits that.

## What the numbers say

**1. ENT-3's own prediction is falsified, and by the cleanest possible comparison.** The entry
said to "expect a smaller number than intra's −17%, and treat that as an answer": after motion
compensation the residual is noise-like, so a context-adaptive coder should find less to exploit.
Measured inside the *same run*, on the same content, at the same q, **inter is the stronger half on
two of three sequences** — bbb_extended −20.6% against −14.2% at q=90, crowd_run −12.2% against
−11.4% — and behind by 0.8 points on the third. Averaged over all 18 points, P −11.3% against I
−9.4%. So the case for context-adaptive coding does not weaken on inter, and **the standing
assumption that the inter gap against H.264 lives in the coder is not refuted by this half of the
argument.** ENT-3's own second branch — "then the inter gap lives in the motion model" — is not
supported by these numbers and should not be read out of them.

**2. The saving decays monotonically with quality, and that is the new finding.** Every sequence,
both frame types: q=90 → q=99 costs roughly two thirds of the saving on crowd_run (−12.2% →
−4.3%) and old_town_cross (−11.9% → −3.7%), and a third on bbb_extended (−20.6% → −14.5%). GNC is
a contribution codec (GOALS §1) and q=95-99 is its home range, so **the figure that matters for
positioning is the smallest one, not the −16.6%-to-−22.9% band `0025` published.** At q=99 abac
buys under 4.5% on two of three sequences.

The mechanism is not mysterious: as the quantiser fines, significance density rises, the
neighbourhood context saturates towards "everything is significant", and more of the file moves
into the bypassed Exp-Golomb suffix and sign bits that abac does not context-code at all. Rice
codes those two populations well. Nothing here is a defect.

**3. There is no inter-specific context headroom.** `GNC_COEF_ENTROPY_INTER=1` prices the shipped
abac tiles of the **first P frame** the way `GNC_COEF_ENTROPY=1` already prices a still's: `Hctx`
is abac's own binarisation under abac's own 6-bucket context, pooled per plane and subband; `Hnb`
and `Hbig` widen the causal neighbourhood to 50 and 200 contexts and code the whole magnitude as
one symbol. `shipped` above `Hnb` is what a *better context model over these coefficients* could
recover. crowd_run, all three planes, `TOTAL` rows:

| | q=95 intra | q=95 inter | q=99 intra | q=99 inter |
|---|---|---|---|---|
| shipped over `Hnb` | +7.7% | **+6.7%** | +12.8% | **+12.5%** |
| shipped over `Hbig` | +8.3% | +7.0% | +13.1% | +12.7% |
| shipped over `Hctx` (adaptation loss) | +0.05% | +0.33% | +0.56% | +0.70% |

Inter's headroom is **smaller than intra's at both quality points**. Whatever abac's 6 magnitude
buckets are leaving on the table, they leave the same amount on a residual as on an intra subband —
so retuning them *for residual statistics* has nothing to collect that a single retune for both
populations would not, and a single retune is INTRA-1/ENT-6 territory, not ENT-3's. **Rejected.**

Note the second row of that table against the first: adaptation loss — cold-started per-block
probabilities plus the length field — is under 0.7% everywhere, consistent with ENT-6's 1.3%. The
6-7% (12-13% at q=99) is the *template*, not the adaptation. That is the number a context
experiment should be aimed at, and it is not inter-specific.

## `0025`'s q=50 and q=75 columns do not reproduce, and its q=90 column does exactly

Running the same comparison on today's `main`:

| sequence | `0025` q=50 | here | `0025` q=75 | here | `0025` q=90 | here |
|---|---|---|---|---|---|---|
| bbb_extended | −16.3% | −16.6% | −22.9% | −20.0% | −19.7% | **−19.7%** |
| crowd_run | −20.7% | −18.0% | −18.4% | −14.1% | −12.1% | **−12.1%** |
| old_town_cross | −22.7% | −21.5% | −21.8% | −14.6% | −12.0% | **−12.0%** |

Three of three exact at q=90, and up to 7.2 points adrift at q=75. **Attributed by measurement,
not by fit.** A detached worktree pinned at `a312d6f` — ARCH-3, the commit that landed `0025` —
built and hash-recorded (`fd793751…`), reproduces `0025`'s column **9 of 9 exactly**. That settles
two things at once: this harness *is* `0025`'s measurement, and the mix behind it was `2I+16P+0B`.
So the difference is in the codec, not the instrument.

Raw byte counts across the two commits say where. I-frame bytes are **equal integers** at all nine
points; P-frame bytes are **equal integers at q=90** and +51.9% to +217.1% (Rice) / +50.6% to
+225.5% (abac) at q=50, +52.4% to +59.9% / +62.3% to +77.7% at q=75. Inter-only, no-op at q=90 —
and far too large for BUG-27, whose own table moved P bytes by +0.4% to +2.0%. **The dominant
cause is INTER-2 (`0043`), which halved `inter_dz_mul` from 2.0 to 1.0.** The boundaries coincide
for an independent reason: the ladder's dead zone is 0.75 at q=50 *and* q=75, so the inter dead
zone went 1.5 → 0.75 and stopped zeroing a large population of small residual coefficients, while
at q=90 it interpolates to ≈0.18 between the q=85 (0.5) and q=92 (0.05) anchors — so the inter
value went ≈0.36 → ≈0.18 and **both are no-ops**, since the quantiser is
`floor(|v|/step + 0.5)` after the dead-zone test and anything at or below 0.5 changes nothing.
BUG-27 (`p_qp_scale` exactly 1.0 for every step ≤ 2.8, i.e. q ≥ 85) and INTRA-2's
`dead_zone_referenced` split sit in the same window and are inter-affecting below q=85 as well;
this was not bisected between the three, and the magnitude says the dead zone carries it.

**So `0025`'s q=50 and q=75 columns should be read as superseded by the table above**, and the
sentence "−12.0% to −22.9%" that BACKLOG, CLAUDE.md and `0025` all carry describes a range that no
longer exists at those two operating points. COORDINATION's BUG-27 note already says "every inter
figure taken at q ≤ 80" is invalidated; this is one of them, and nobody had gone back for it.

## What was not chosen

**A BD-rate.** The two arms decode to bit-identical pixels at every point — 18 frames × 18 points
hashed — so the quality delta is exactly zero, not small, and a byte ratio is the whole answer.
Computing a BD-rate here would add a ladder's worth of interpolation error to a number that has
none.

**Making abac the default.** Untouched, and `0045` moves it no closer. Decision `0017` rests on
three reasons; ENT-5 discharged the GPU-encoder one, this record prices the inter one *down* at
the operating point that matters, and the 1.69× frame decode is unchanged. If anything, a saving
that reads −3.7% at q=99 weakens the case in GNC's home range rather than strengthening it.

**Retuning the contexts on inter anyway, to see.** The bound says the ceiling on that work is
6.7% at q=95 and it is *shared with intra*, so the experiment would be mis-scoped as an inter
item. Left where it belongs: a context-template question over both populations.

**Extending the inter diagnostic to B frames.** `encode_bframe` has the same tail and the hook
would be six lines, but the shipped mix is `2I+16P+0B` — a diagnostic that never fires by default
is an untested one (CLAUDE.md, "no silent features"), so it is P-frames only and says so.

## Cost

One read-only env-gated diagnostic (`GNC_COEF_ENTROPY_INTER`, 38 lines in `encode_pframe`) and one
harness. Both halves of "no silent features" are checked rather than asserted:

- **The canary fires on both outcomes.** With `--abac` it prints
  `[coef-entropy-inter] first P frame, 120 tiles, intra qstep=… res_qstep=…`; with `--rice` it
  prints `[coef-entropy-inter] no abac tiles on this P frame — run with --abac`. A success-only
  canary would have made "wrong coder" indistinguishable from "no headroom".
- **The variable does not move the bitstream.** Same input, `-q 95 -n 3 -k 9 --abac`, with
  `GNC_COEF_ENTROPY_INTER=1` and without: both `.gnv` files hash `756c0cbd…`. With the variable
  unset the diagnostic prints nothing at all (grep count 0).
