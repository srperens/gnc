# 0034 — The last of the intra gap is padding nobody looks at

**Date:** 2026-09-08
**Item:** INTRA-1 step 3
**Status:** accepted — the accounting closes; the fix is filed as PAD-1, not taken here

## Context

INTRA-1 asked where GNC's **+27.1% RGB / +48.3% Y** intra gap to JPEG 2000 9/7 goes, and required
"a number per cause that sums to roughly the measured gap". Four instalments had named most of it:

| cause | RGB points | where |
|---|---|---|
| chroma allocation against an RGB metric — a deliberate perceptual trade | 8.5 | `0026` |
| entropy-coder headroom above the coefficients' own entropy | ≤7.5 | `0024` |
| dead zone / quantiser rounding rule | ~3 | `0028` (INTRA-2, not shipped) |
| tiling, realisable in GNC | 0.6 | `0026` |
| lifting normalisation | ~0 | `0026` |
| cross-tile rate allocation | 0.95, rejected | `0027` |
| tile-boundary extension | 0, already correct | step 2c |
| **remainder** | **~6** | nothing accounted for it |

COORDINATION's own handover said it plainly: *"~6 of the 27.1 points remain and the item's own
candidate list is exhausted, so the next step needs a new hypothesis rather than another sweep."*

## Decision

**The remainder is tile-alignment padding, and it is ~6.6 points of the 27.1.**

GNC pads every plane up to a whole multiple of `tile_size` with edge replication
(`src/shaders/pad.wgsl`) and codes the padded plane. **A 1920x1080 frame is coded as 2048x1280 —
26.4% more coefficients, 20.9% of the coded samples outside the picture.** The decoder crops them
and nothing ever looks at them. OpenJPEG in whole-picture mode codes 1920x1080 exactly, and the
harness divides *both* codecs' bytes by the visible pixel count, so the GNC arm has been paying a
tax the J2K arm does not — in every figure quoted since MEAS-9.

Two measurements, by different methods, on the four ENT-4 images:

| | method | mean over 4 images |
|---|---|---|
| projected from a content-controlled crop pair | GNC only, content fixed | **+6.60%** |
| drop in the cross-codec gap, native against padding-free | GNC vs J2K 9/7, one ladder | **+6.70 points** (26.54 → 19.83) |

**The means agree to 0.10 points; the per-image figures scatter by ±2.2 in both directions.** That
scatter is the content change between a frame and its centre crop, which the cross-codec arm
carries and the content-controlled arm does not — so the cross-codec arm corroborates the mean and
is not a per-image second opinion. **The content-controlled measurement is the one this decision
rests on**, and its own control is now measured rather than argued: JPEG 2000 reads **−0.13% to
+0.02%** between the same two crops where GNC reads +11.6% to +25.2%.

**With this the accounting closes.** 8.5 chroma + ≤7.5 entropy + ~3 dead zone + 0.6 tiling + **6.6
padding** is 26.2 of 27.1, with the causes being roughly independent rather than strictly additive
— the entropy headroom was measured as a fraction of a file that is itself padding-inflated, so
the two compound rather than add.
INTRA-1's success criterion is met.

**Nothing here is a proposal to change a default.** The fix is filed as **PAD-1** and the reason it
is filed rather than taken is in "What was not chosen" below.

## Measurements

### The canary, first

The whole result rests on one claim — that GNC codes the edge-replicated padded plane — so it is
checked rather than assumed. Rebuild the padding in Python, encode the result as a picture in its
own right, and compare byte counts with the production encode of the unpadded original:

```
bbb_1080p:        crop 1920x1080 -> 2048x1280   1 793 794 B  vs pre-padded 1 793 794 B   IDENTICAL
kristensara_720p: crop 1280x720  -> 1280x768      554 346 B  vs pre-padded   554 346 B   IDENTICAL
```

Byte-identical, and 1 793 794 B is the figure BACKLOG already records for bbb at q=90 with
`--abac`. So the pre-padded plane *is* the production encode, which is what lets Part 3 below
substitute a different fill without touching the codec.

### Part 1 — content-controlled

Largest tile-aligned centred crop `A`; `B` one pixel wider, `C` one pixel taller, `D` both. Each of
those adds ~0.15% of picture and a whole tile row or column of padding. `--abac`, q=80/85/90/95/98,
BD-rate against `A`:

| image | variant | coded/visible | RGB BD-rate | Y BD-rate | ΔRGB at q=90 |
|---|---|---|---|---|---|
| bbb_1080p | B (+1 col) | 1.1422 | +3.68% | +3.73% | +0.001 dB |
| bbb_1080p | C (+1 row) | 1.2488 | **+8.32%** | +8.20% | +0.002 dB |
| bbb_1080p | D (both) | 1.4264 | +12.31% | +11.99% | +0.004 dB |
| blue_sky_1080p | C | 1.2488 | **+8.52%** | +8.59% | +0.002 dB |
| kristensara_720p | C | 1.4971 | +18.04% | +19.35% | +0.005 dB |
| touchdown_1080p | C | 1.2488 | **+8.41%** | +8.32% | +0.003 dB |

The quality column is the check that matters: **0.001–0.005 dB** across every pair, so the two
ladders sit at the same operating point and the rate difference is padding and nothing else.

**The cost is separable**, which is what licenses carrying it to another frame size: `B + C` lands
within 0.3–1.1 points of `D` on all four images (bbb +3.68 + +8.32 = +12.00 against +12.31).
Projected to native geometry — 200 padding rows and 128 padding columns at 1080p, 48 rows at 720p:

| image | native | padrows | padcols | projected tax |
|---|---|---|---|---|
| bbb_1080p | 1920x1080 | 200 | 128 | **+7.90%** |
| blue_sky_1080p | 1920x1080 | 200 | 128 | **+7.54%** |
| kristensara_720p | 1280x720 | 48 | 0 | **+2.65%** |
| touchdown_1080p | 1920x1080 | 200 | 128 | **+8.30%** |
| | | | mean | **+6.60%** |

### Part 1's control — the same crop pair through a codec that does not pad

Part 1's attribution rests on the two crops differing by ~0.15% of picture. That was an argument;
JPEG 2000 makes it a measurement, since it codes both crops exactly as given, so its own
`A` -> `D` BD-rate is the content term on its own.

| image | J2K 9/7, A -> D, RGB | Y | GNC on the same pair, RGB |
|---|---|---|---|
| bbb_1080p | **−0.13%** | −0.02% | +12.31% |
| blue_sky_1080p | **+0.02%** | −0.02% | +11.63% |
| kristensara_720p | **−0.06%** | +0.04% | +25.21% |
| touchdown_1080p | **+0.02%** | +0.03% | +13.00% |

**At most 0.13 points of content against 11.6–25.2 in GNC — under 1% of the effect.**

### Part 2 — cross-codec, native against padding-free, on one ladder

**The first version of this part was wrong in a way worth recording.** It read the padding-free gap
against ENT-4's published **+27.1%**, taken on a *different* GNC ladder (q=60–99 there, q=80–98
here). A BD-rate is integrated over the **overlapping** quality range, so two ladders give two
different figures on identical content: bbb's native gap reads **+19.46%** here against ENT-4's
+17.3%, and blue_sky **+30.66%** against +32.5%. **Per image the artefact is up to 2.2 points, the
same size as the effect being measured**, even though the four-image means nearly coincide (26.76
on that first run, 26.54 on today's, against ENT-4's 27.1). It gave 6.98 points and "the two methods agree to 0.4" — the right answer for the
wrong reason. So the native arm is re-run here rather than cited, same ladder, same rates, same
metric path, only the picture changing.

| image | native gap | padding-free gap | drop | Part 1 projection |
|---|---|---|---|---|
| bbb_1080p | +18.41% | +8.60% | 9.81 | 7.90 |
| blue_sky_1080p | +30.63% | +23.83% | 6.80 | 7.54 |
| kristensara_720p | +28.87% | +28.13% | 0.74 | 2.65 |
| touchdown_1080p | +28.24% | +18.77% | 9.47 | 8.30 |
| **mean** | **+26.54%** | **+19.83%** | **+6.70** | **+6.60** |

Y-PSNR moves the same way: **+51.17% → +40.66%, a drop of 10.51 points.** This arm carries the
content change from a frame to its centre crop and Part 1 does not, which is why the means agree
and the per-image figures do not.

### Part 3 — how much of the tax is a fill choice

The padded samples are **don't-care**, so edge replication is a *choice*. Because the canary shows
the pre-padded plane is the production encode, alternative fills can be priced with no code change
at all: build the padded plane in Python, encode it as a tile-aligned picture, score quality on the
visible region only. The `replicate` arm reproduces production byte for byte (1 793 794 B on bbb at
q=90), which is what ties the oracle to the shipped path.

BD-rate against the shipped edge-replicate fill, negative = cheaper at the same visible quality:

| fill | bbb | blue_sky | kristensara | touchdown | mean RGB | mean Y |
|---|---|---|---|---|---|---|
| `decay8` — replicate, then fade to one scalar over 8 px | −5.67% | −4.74% | −1.62% | −5.88% | **−4.48%** | −4.52% |
| `flat` — one scalar everywhere outside the picture | −5.41% | −4.70% | −1.29% | −5.97% | −4.34% | −4.42% |
| `decay32` — the same fade over 32 px | −5.33% | −4.52% | −1.24% | −5.38% | −4.12% | −4.14% |
| `mirror` — whole-point symmetric extension of the picture | +12.10% | +15.78% | +3.00% | +14.74% | **+11.41%** | +11.28% |

**A fill change alone recovers 4.48 of the 6.60 points — 61–72% per image.** Visible-pixel quality
does not move: bbb q=90 reads 50.060 dB under `flat` against 50.061 dB under `replicate`, so the
step discontinuity a flat fill puts at the picture edge costs less than the detail it saves. Note
also that **replication was already the better of the two textbook extensions** — mirroring the
picture into the padding copies real detail there and costs 11.5 points more.

### Reproduction, and the pin

Re-measured on `c84fbd5` after RATE-2 (`a7273ab`) landed, because RATE-2 codes q=95..99 both ways
and keeps the smaller and this harness's first ladder was q=80..98. **One rung of eight moved**: the
padded crop at q=98 came back 5.78% smaller, the aligned crop not at all. Worse than a shifted byte
count — the padded arms come back **bit-exact lossless**, `psnr()` returns `inf`, and a Bjontegaard
fit over an infinity is a silent non-number. `bd()` now refuses a non-finite rung and the default
ladder is **q=80/85/90/94**, clear of the dual-path range.

On that ladder the `C` rows read +8.07 / +8.45 / +18.06 / +8.29 against the original
+8.32 / +8.52 / +18.04 / +8.41, **within 0.25 points**, and the projection is **identical at
+6.60%** — it reads the q=90 rung, which is byte-identical across RATE-2. Every part was
re-measured, not only Part 1: Part 2's drop goes 6.58 -> **6.70** and Part 3's `decay8` fill goes
−4.62% -> **−4.48%**, the latter in the direction predicted (RATE-2's own reclaim leaves the
ladder). **No figure moves by more than 0.15 points and the decision does not move.**

Two things follow. **RATE-2 already reclaims part of the padding tax for free above q=95**, and it
reaches the padded arm first, which concentrates PAD-1's value below q=95 — recorded there. And
**any still figure in this repository taken on a ladder reaching q>=95 before `a7273ab` is pinned
to that code**, including INTRA-1's own step 1 and step 2; nothing is retracted, but they will not
reproduce byte for byte today.

## What was not chosen, and what it would have cost

- **Changing `pad.wgsl`'s fill in this item.** It is a ~20-line shader change worth ~4.6% of intra
  rate, which by this repository's standards is a large win for the effort — and it is still the
  wrong thing to do here, for a reason the intra measurement cannot see. **The decoder keeps the
  padded region in the reference buffer, and motion compensation reads it for blocks at the frame
  edge.** Edge replication is the standard choice there precisely because it extends the picture
  plausibly; a flat or faded fill would change inter prediction for every edge block, and none of
  the numbers above say anything about that. Filed as **PAD-1** with the intra evidence attached
  and the inter gate named. There is a variant that sidesteps the conflict — let the decoder
  re-replicate the edge into the padding after reconstruction, so the encoder may write a cheap
  fill while the reference stays MC-friendly — but that changes the decoding process and needs a
  bitstream version, which is a design decision and not a shader tweak.
- **Not padding at all — partial border tiles, the way JPEG 2000 handles them.** This is the fix
  that would recover the whole 6.6 points rather than 70% of it, and it is a large change: tile
  origins, the tile grid, every shader that derives a position from `tile_size`, and the per-tile
  CRC and seek structures. Worth knowing the ceiling is 6.6 and not 27; not worth guessing at the
  cost here.
- **Choosing a tile size that divides the frame.** No multiple of `2^levels` in
  `[MIN_TILE_SIZE, MAX_TILE_SIZE]` divides 1080, so at 1080p this is not available at five levels,
  and dropping levels to reach one costs more than the padding does.
- **Re-deriving +27.1% before starting.** Not needed and not skipped either: the canary reproduces
  BACKLOG's own recorded 1 793 794 B for bbb at q=90 with `--abac`, and Part 2's arms are the same
  `meas9_contribution.py` arms that reproduce ENT-4 exactly.
- **Reading the remainder as one more coding deficiency.** It is not one. Like the 8.5 points of
  chroma allocation in `0026`, this is a **quoting** correction as much as a defect: "+27.1% intra
  coding gap" contains 6.6 points of samples GNC codes and JPEG 2000 does not, on top of the 8.5
  points of deliberate chroma allocation. **The honest form of the intra coding gap on these four
  images is closer to +12% than to +27%**, and unlike the chroma half, this one is also 4.5 points
  of *shipped* rate that a fill change would return.

## Also settled on the way

BACKLOG's INTRA-1 entry still listed "the wavelet's tile-boundary handling (`transform_97.wgsl`
replicates the edge sample where J2K uses symmetric extension)" as an untested candidate. It is not
one, and re-deriving it from the shader agrees with what step 2c already found and recorded in
COORDINATION: the lifting steps substitute `low[half-1]` for `low[half]` and `high[0]` for
`high[-1]`, which is exactly whole-point symmetric extension (`x[N] = x[N-2]`, `x[-1] = x[1]`), and
the inverse pass substitutes the same values, so the pair is an exact inverse. **0 points, and the
BACKLOG text is now corrected** — the "boundary replication" comment at `transform_97.wgsl:78`
describes the image-edge clamp in the overlap load path, which does nothing on the default path
because `overlap_pixels` is 0 and the plane is already a whole number of tiles.

The candidate pointed at the right phenomenon in the wrong place. GNC does replicate its picture
edge, and it costs 6.6 points — but it happens in `pad.wgsl`, on pixels, before the transform ever
runs.

## Consequences

- `scripts/meas_intra1_padding.py` — the harness, with the canary as a first-class mode
  (`--canary`) and a projection mode that needs no GPU (`--project-from`).
- **PAD-1** filed: change the fill, gated on inter.
- INTRA-1's accounting closes; the item's remaining open work is PAD-1 and INTRA-2, both filed.
- Any figure quoting the intra gap should now say whether it includes the padding tax. README's
  MEAS-9 table and BACKLOG's headline both do, and both now say so.
