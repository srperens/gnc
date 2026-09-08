# 0077 — BUG-48: the padding fill is a wavelet lever, not a quality one

**Date:** 2026-09-08
**Status:** accepted
**Item:** BACKLOG BUG-48 (P3) — fixed
**Bitstream-visible:** no. `pad.wgsl` runs only in the encoder and writes pixels outside the
visible area; the decoder reconstructs whatever was coded (`pad_fill_mode`'s own note).
**Corrects the fix proposed in:** BUG-48's filing, which put the gate on `q == 100`

## What changed

`quality_preset(100)` cleared `pad_fill_decay` **inside the MED branch**, so the fade-to-flat
padding fill from `0039` is off exactly when the transform is MED and on everywhere else.

## Why the fill reverses sign

`0039` fades the padding flat so that its *detail subbands go to zero* — worth **−4.63% RGB** on
four stills at q=80..94. That reasoning is about a wavelet. MED has no subbands: it predicts each
pixel from its left and upper neighbours, so a fade is a gradient the residual has to **code**
across, where plain edge replication predicts exactly and costs nothing.

Four stills, one binary, both arms via `GNC_PAD_FILL`, `q=100`:

| still | decay (was shipped) | replicate | |
|---|---|---|---|
| bbb_1080p | 3 257 157 | 3 235 737 | **−0.658%** |
| blue_sky_1080p | 2 166 911 | 2 153 118 | **−0.637%** |
| kristensara_720p | 931 263 | 927 600 | **−0.393%** |
| touchdown_1080p | 2 627 186 | 2 610 478 | **−0.636%** |

## The gate goes on the transform, and the item's proposed one-liner would have cost 4.6%

BUG-48 was filed as "one line in `quality_preset`", meaning `pad_fill_decay: q != 100`. That reads
naturally — the loss was found at q=100 — and it is wrong, because **q=100 is not always MED**.
With `GNC_MED=0` the same preset is a lossless *wavelet* encode, and there the fill is worth what
`0039` measured, at the very top of the ladder:

| still | decay | replicate | |
|---|---|---|---|
| bbb_1080p | 3 260 563 | 3 436 337 | +5.391% |
| blue_sky_1080p | 2 744 873 | 2 866 708 | +4.439% |
| kristensara_720p | 1 169 457 | 1 178 520 | +0.775% |
| touchdown_1080p | 2 967 518 | 3 131 749 | +5.534% |
| **total** | | | **+4.643%** |

**−4.64% against `0039`'s −4.63%, on the same four images, three years of ladder apart from where
it was taken.** Keying the fix on the quality would have handed that arm a 4.6% regression to buy
0.6% on the other one. It is the same shape as the mistake `0039` itself avoided: the fill is a
property of *what codes the padding*, not of how hard the picture is being squeezed.

## Gate

`scripts/gate_bug48.py` — before/after binaries, 40 encodes. **Only the four q=100 MED cells move.**
Byte-identical: q=85, 90, 95, 97, 99 on all four stills; all four stills at q=100 under
`GNC_MED=0`; and 8 sequence points (crowd_run and bbb at q=99/100, ki=2 and 9), which were never
at risk because both sequence paths clear the flag themselves — `0039` for referenced I-frames and
LOSSLESS-3's all-intra arm, which already cited BUG-48 by name while clearing it.

**Canary:** `pad_fill_mode` prints the mode and the path default under `GNC_DIAGNOSTICS`. After
the change, `q=100` reads `pad fill = replicate (path default replicate/sequence)`, `q=100
GNC_MED=0` reads `decay (path default decay/intra)`, and `q=90` reads `decay`. Three paths, three
answers, one run each.

## What was not chosen

- **`pad_fill_decay: q != 100`** — the filing's own suggestion. Costs 4.64% on the `GNC_MED=0`
  arm, above.
- **Keying it on `is_lossless_intent()`** — same defect in a nicer wrapper: the lossless *wavelet*
  configuration satisfies it and wants the fill on.
- **Clearing it in `lossless_sibling` too.** That is the still path's bit-exact candidate at
  q=95..99, which *is* a MED encode and *is* decay-filled — 3 of 4 stills emit exactly that file
  at q=97 and q=99, so it is shipped output and the same 0.4–0.7% is on the table. It is not this
  item's line: `0072` deliberately made the sibling inherit the caller's fill so that the two
  candidates leave the *same* padded source behind, and unpicking that needs `0072`'s invariant
  re-measured rather than a second opinion about padding. Filed as **BUG-50** with these numbers.

## Caveat

`--dct` at q=100 is not reachable through the preset and is not measured. Same argument would
apply — a block DCT has no wavelet subbands either — but that is a prediction, not a number.
