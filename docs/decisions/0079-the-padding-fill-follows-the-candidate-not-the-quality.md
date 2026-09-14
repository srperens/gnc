# 0079 — The padding fill follows the candidate that is kept, not the quality number

**Date:** 2026-09-08
**Status:** accepted; shipped
**Item:** BUG-48 (P3), filed by LOSSLESS-3
**Supersedes** BUG-47's `pad_fill_decay` inheritance in `lossless_sibling` (`0072`), with the same
guarantee reached a stronger way

> **Corrected on landing, by a peer's measurement, and the correction is the useful part.** This
> record first put the switch on `is_lossless_intent()`. A session working the same item in parallel
> measured the case that rule assumed away: with `GNC_MED=0` a q=100 encode is a lossless **wavelet**
> encode, and there decay is still worth **−4.64%** on the same four stills — PAD-1's figure
> reproduced at the top of the ladder. Reproduced here at **+5.39%** on bbb when replicate is
> forced. So `is_lossless_intent()` would have handed that arm a ~5% regression, and the switch
> belongs where they put it: inside the **MED** branch of `quality_preset`.
>
> `lossless_sibling` therefore does **not** force `false` either — it takes `&=`, so decay survives
> only where the preset asks for it *and* the caller allows it. That keeps all three cases right:
> MED replicate, a lossless-wavelet still keeping its −4.64%, and a sequence keyframe replicate in
> both candidates, which is what `0072`'s source-built reference needs. Everything below about
> *why* the fill follows the candidate stands; the mechanism for detecting "bit-exact" was wrong,
> and it was wrong in the direction of a regression nobody would have seen on the default path.

## The decision

Two lines, one rule: **a bit-exact picture is padded by replication, never by PAD-1's fade.**

- `quality_preset` clears `pad_fill_decay` when the preset reaches `is_lossless_intent()`, applied
  after the anchors so it covers the lossless *wavelet* path (`GNC_MED=0` at q=100) as well as MED.
- `lossless_sibling` sets it to `false` outright instead of inheriting it.

The reason is not a quality threshold. **At a bit-exact setting the padding is coded exactly**, so a
fill that fades to flat has to be spent on, where edge replication is what MED predicts for free.
PAD-1 (`0039`) measured the fade at −4.63% RGB over q=80..94 and the preset applied it everywhere,
including where it inverts.

## What it is worth, and where it does nothing

`scripts/meas_bug48_pad_fill.py`, the four stills `0039` used, shipped against forced `decay`:

| | bbb | blue_sky | kristensara | touchdown | mean |
|---|---|---|---|---|---|
| q=95 | 0.00% | 0.00% | 0.00% | 0.00% | **0.00%** |
| q=97 | 0.00% | −0.64% | −0.39% | −0.64% | **−0.42%** |
| q=99 | −0.66% | −0.64% | −0.39% | −0.64% | **−0.58%** |
| q=100 | −0.66% | −0.64% | −0.39% | −0.64% | **−0.58%** |

**No point is worse anywhere**, and q=95 is untouched because all four stills keep the lossy
candidate there — where the fade is worth +4.86% and PAD-1 is simply right.

**The lever reverses with the candidate, not with `q`, and q=97 shows both signs in one column.**
bbb still keeps the lossy candidate at q=97 and pays **+6.26%** if forced to replicate, while the
other three have already switched to bit-exact and gain 0.4–0.6%. A `q` threshold would have to be
fitted per image — the same trap RATE-2 refused when it chose to code both ways rather than guess
the boundary, and the reason this keys on `is_lossless_intent()`.

**Sequences do not move: 8 of 8 byte-identical**, measured against the actual pre-fix binary at
q ∈ {95, 99, 100} × ki ∈ {2, 9} on bbb and crowd_run. The sequence encoder already clears the flag
unconditionally for keyframes, so there was nothing there to fix.

## Why this supersedes BUG-47's inheritance rather than sitting beside it

`0072` made the sibling *inherit* `pad_fill_decay`, because `quality_preset(100)` set it to `true`
while the sequence encoder had cleared it — so the two candidates left differently padded sources
in `input_buf` and 10 of 24 sequence points moved. Inheriting fixed that by making the sibling
agree with its frame.

Forcing `false` gives the same guarantee and a better one: it **cannot disagree with anything**,
because the sequence path already clears the flag for every keyframe, so both candidates are
replicate there either way. The two rules differ only for a still — which has no reference to
disagree with, and where the measurement above says replicate. The RATE-4 byte-identity gate still
passes 24 of 24 with the route firing 78 times.

## What was rejected

- **`quality_preset(100).pad_fill_decay = false` alone**, which is how the item proposed it. It
  fixes a q=100 still and leaves every bit-exact *candidate* RATE-2 codes at q=95..99 paying the
  fade — which is the case the item itself said compounds.
- **A `q` threshold.** See q=97 above: it would need one number per image.
- **Reading `GNC_PAD_FILL=decay` as the before-arm for a sequence**, which the first version of the
  harness did. That knob also overrides the sequence encoder's deliberate keyframe clear, so it
  reported this change as a **−7.21% regression on bbb** that the pre-fix binary does not show. The
  harness now compares sequences against `replicate` and says in as many words why. **A knob that
  does more than the change under test is not a control arm** — and this one reads as a plausible
  regression rather than an error, which is the expensive kind.

## What this does not claim

- **Stills only.** 4:2:2 and 4:2:0 stills are unmeasured; the fill is format-independent, so they
  should move the same way.
- **No quality figure, and none is needed:** the q≥97 arms are bit-exact both ways (the pixels are
  the source), and at q=95 nothing changed.
- **No encode-time claim.** The fill is one dispatch either way.
