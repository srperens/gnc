# 0039 — The padding fill is a still-image lever, because the padding is a reference

**Date:** 2026-09-08
**Item:** PAD-1
**Status:** accepted — shipped as the default for stills, refused on anything a frame predicts from

## Context

INTRA-1 step 3 (`0034`) measured that GNC pads every plane up to a whole multiple of `tile_size`
with edge replication and codes the padded plane, so **a 1920x1080 frame is coded as 2048x1280 and
20.9% of the coded samples lie outside the picture.** The decoder crops all of them. That made
their content a free choice worth **6.6 of the 27.1-point intra gap to JPEG 2000**, of which an
oracle said ~4.6 points were recoverable by choosing a better fill.

`0034` filed PAD-1 rather than shipping the change, on one stated reason: the decoder keeps the
padded plane in the reference buffer and motion compensation reads it for blocks at the frame edge.
That reason was asserted, not verified. **Verified here:** MC is handed the *padded* dimensions
(`p_padded_w/h`, `src/decoder/gpu_work.rs:478`) and clamps its reads to them, so an edge block whose
motion vector points outward genuinely predicts from the fill.

## Decision

**Fade the padding flat on the still-image path. Refuse it on every frame something predicts from.**

`pad.wgsl` now replicates the edge and then fades over 8 px to one value per channel, so the
padding is flat in *both* axes and its detail bands go to zero. Measured on the **shipped encoder**,
four stills, q=80..94, `--abac`, BD-rate against the previous replication at identical visible
quality:

| image | RGB | Y | ΔRGB at q=90 | Δbytes at q=90 |
|---|---|---|---|---|
| bbb_1080p | −5.86% | −5.72% | −0.001 dB | −5.82% |
| blue_sky_1080p | −4.99% | −4.90% | −0.001 dB | −4.95% |
| kristensara_720p | −1.75% | −1.79% | −0.002 dB | −1.81% |
| touchdown_1080p | −5.90% | −6.01% | +0.000 dB | −6.06% |
| **mean** | **−4.63%** | **−4.60%** | | |

The offline oracle predicted **−4.63% / −4.60%**. Two decimals, both metrics, independent
implementations — that agreement is the main reason to believe either number.

**And it is refused wherever the padding is a reference**, because there it is a large loss.
`benchmark-sequence`, ki=9, 17 frames, forcing the fill on:

As first run, before INTER-2 landed — superseded by the table in the next paragraph, and kept only
because the decision was taken on it:

| sequence | chroma | q | rate | worst-frame PSNR |
|---|---|---|---|---|
| crowd_run | 444 / 420 | 85, 92 | −6.9% to −9.2% | **+0.000 dB** |
| old_town_cross | 444 / 420 | 85, 92 | −7.8% to −10.1% | **+0.000 dB** |
| bbb_extended | 444 | 85 | −7.42% | **−1.300 dB** |
| bbb_extended | 444 | 92 | −9.14% | **−4.030 dB** |
| bbb_extended | 420 | 92 | −10.24% | **−0.350 dB** |

**Re-measured on `main` with INTER-2 (`a069e55`, merged after this gate first ran), and the hazard
got *worse*.** INTER-2 lowered the inter dead zone (`inter_dz_mul` 2.0 -> 1.0), which changes inter
output for q <= 88 — so the q=85 rows above are from the older path and the table below is the
current one. Regressing points go from **3 of 12 to 6 of 12**:

| sequence | chroma | q | rate | dWORST before | **dWORST now** |
|---|---|---|---|---|---|
| crowd_run | 444 / 420 | 85 | −6.9% / −9.0% | +0.000 | **+0.000** |
| old_town_cross | 444 | 85 | −7.81% | +0.000 | **−0.900** |
| old_town_cross | 420 | 85 | −10.20% | +0.000 | **−0.490** |
| bbb_extended | 444 | 85 | −8.70% | −1.300 | **−2.470** |
| bbb_extended | 420 | 85 | −9.80% | −0.250 | **−0.360** |

The mechanism is coherent: a larger dead zone was quantising away part of the prediction error the
fill causes, so **the old figures understated the hazard.** Refusing the fill on inter is more
clearly right after INTER-2, not less.

**The q=92 rows are unchanged to the last digit**, including the −4.030 dB the verdict rests on,
and that is a property rather than a coincidence — verified independently by the INTER-2 session in
this exact configuration (bbb_extended, 24 frames, ki=9, q=92, fill forced on, byte-identical at
both multipliers). The reason is arithmetic and content-independent: `res_dead_zone = dead_zone *
mul`, the anchors put `dead_zone` at 0.05 by q=92, so both 0.05 and 0.10 sit below the **0.5 no-op
threshold** — GNC quantises as `floor(|v|/step + 0.5)` after a `|v| < dz*step` test, so under 0.5
the test only zeroes what the rounding already zeroed. Asserted over 4000 values in
`tests/cli_shipped_config.rs::a_dead_zone_of_half_a_step_changes_nothing`.

**The shipped default is unaffected in every row: +0.000 dB and 0.0000% of rate at all 12 points.**

**Pre-declared criterion was 0.3 dB of worst-frame PSNR. It failed at 4.03 dB on one of three
sequences** — the same shape as INTRA-2's dead zone, and the reason worst-frame rather than mean is
the figure: an error in a reference propagates until the next keyframe. bbb_extended's mean moved
−2.28 dB while its worst frame moved −4.03 dB.

The control that turns this from a correlation into a cause: **the same clip at ki=1 loses nothing
at all** — −5.65% of rate, PSNR identical to 0.01 dB. Same content, same q, same fill; only the
references removed.

So the policy is per-path, and it is fail-safe in the direction that matters:

* `quality_preset` sets `pad_fill_decay: true` — it serves the still path.
* `CodecConfig::default()` sets it **false**, because the sequence path builds from it.
* All four `main.rs` funnels that already refuse RATE-2's `lossless_fallback` refuse this too.
* Both sites where the sequence encoder codes an I-frame through `EncoderPipeline::encode` clear
  it, because an I-frame in a chain is a reference.
* Every other padding dispatch in the sequence encoder goes through `dispatch_gpu_pad_cached`,
  which asserts replication before each dispatch.

**Sequence output is byte-identical to the pre-PAD-1 encoder** on every figure
`benchmark-sequence` prints.

## What was measured before implementing, and what it saved

The fade needs a target value, and the obvious one — the picture's mean — needs a full reduction
over the plane, i.e. a bandwidth pass on the encode path of every frame. Measured first, on the
four stills:

| fade target | mean RGB | cost to compute |
|---|---|---|
| **8 strided samples of the edge line** | **−4.63%** | **nothing** |
| that line's exact mean | −4.63% | 1-D reduction |
| the picture's mean | −4.48% | full 2-D reduction |
| a hardcoded mid-grey | −4.44% | nothing |
| one sample of the edge line | −4.38% | nothing |

**The value barely matters; flatness does.** A constant that reads nothing from the picture gets
96% of the best result. Eight strided samples of the edge line reproduce that line's exact mean to
two decimals, and their positions are a pure function of the plane dimensions — so the shipped
version needs **no reduction, no per-frame uniform and no host-side pass**, and every thread in a
strip reads the same few addresses.

That mattered concretely: the pad uniform is created once with `UNIFORM` usage, so a per-frame
target would have meant making it writable and threading a value through all five sites that fill
the raw input buffer. Five chances of a silent per-path bug, to buy 0.19 points.

## What was not chosen

- **Shipping it as the default everywhere.** It fails the inter gate by 4.03 dB. This is the whole
  reason PAD-1 existed as an item rather than a patch.
- **Making it opt-in only (`GNC_PAD_FILL=decay`), the way `--abac` shipped.** That would leave
  every still encode paying a tax the codec knows how to avoid, for no benefit — the still path
  provably has no reference, so there is nothing for the user to weigh.
- **Enabling it for I-frames inside a chain.** An I-frame is exactly what the following P-frames
  predict from, so this is the failing configuration rather than a compromise.
- **Having the decoder re-replicate the picture edge after reconstruction**, which would let the
  encoder write a cheap fill while the reference stays MC-friendly, collecting the inter half too.
  It changes the decoding process and needs a bitstream version — a real design decision, and left
  to PAD-2 with this measurement as its input. Note `0034` claimed this was the *only* shape
  needing versioning; that was right about this variant and wrong to imply the still-path fill
  needed it, since **`pad.wgsl` is compiled only in `src/encoder/pipeline.rs`** and the decoder
  reconstructs whatever was coded.
- **A tile size that divides the frame exactly, so there is no padding to fill.** The obvious
  question, and the answer is a measurement rather than the arithmetic `0034` stopped at. Such a
  size does exist for each common resolution — it is `gcd(W, H)`, constrained to
  `[MIN_TILE_SIZE, MAX_TILE_SIZE]`:

  | resolution | divides both exactly | clean halvings, so max levels |
  |---|---|---|
  | 1920x1080 | **120** | 3 |
  | 1280x720 | **80** | 4 |
  | 3840x2160 | 80, 120, **240** | 4 |

  **But none of them can carry five levels, and no tile size divisible by 32 divides 1080, 720 or
  2160 at all.** The root cause is not GNC: broadcast heights are not power-of-two friendly.
  1080 = 8 x 135, 720 = 16 x 45, 2160 = 16 x 135 — the factors of two run out after three or four
  and an odd factor is left. So the choice is genuinely between a deep wavelet with padding and a
  shallow one without.

  Measured on bbb_1080p at q=90 with `--abac`, which nobody had done:

  | | padding | levels | bytes | RGB PSNR |
  |---|---|---|---|---|
  | tile 256 (default) | 20.9% | 5 | **1 689 447** | 50.062 dB |
  | tile 120 (zero padding) | **0%** | 3 | 3 056 603 | 50.036 dB |

  **Zero padding costs +81% of rate.** Against padding's 6.6 points that is about twelve times the
  wrong direction, so the existing choice is right by a wide margin — and now for a stated reason.
  Two caveats kept because the figure is indicative rather than a BD-rate: it is one image at one
  q, and it mixes two effects, since tile 120 also means 144 tiles instead of 40 and therefore
  more per-tile overhead and more of ENT-6's code-block cold start. The margin is far too large
  for either to change the conclusion.
- **Recovering the whole 6.6 points.** Most of it turns out to be cheap, which was not known when
  this record was first written: **padding to a multiple of `2^levels` instead of `tile_size`**
  takes the tax from a 14.1% mean across common formats to 1.8% (1080p 20.9% -> 0.7%) with no
  odd-length arithmetic, because every tile extent stays a multiple of 32. That is what VC-2 and
  JPEG XR do. The rest needs partial border tiles the way JPEG 2000 has
  them, touching tile origins, the tile grid, every shader deriving a position from `tile_size`,
  and the per-tile CRC and seek structures. The ceiling above a fill change is 6.6, not 27.
  **This is what the tile-size row above actually argues for**: partial border tiles decouple tile
  size from frame size, so the deep wavelet and zero padding stop being alternatives. Having to
  choose between them is the defect; the fill shipped here only makes the choice cheaper. Filed as
  **TILE-1**, with the +81% as its justification for being scoped at all.
- **A 32 px ramp instead of 8.** −4.08% against −4.63%: a longer fade leaves more of the edge
  line's detail in the padding.
- **Mirroring the picture into the padding**, the textbook alternative: **+11.4%**, i.e. 16 points
  worse than replication. Smooth at the seam is not what matters; flat in the direction of
  extension is.

## Two defects found by verifying rather than assuming

Both would have shipped the configuration that loses 4 dB.

1. **The pad uniform is shared and persistent, and `benchmark-sequence` calls the still path
   several times per run.** So a sequence encode inherited `decay` from a previous still. Fixed by
   having `dispatch_gpu_pad_cached` write the fill mode before every dispatch, which makes
   cross-path leakage impossible rather than unlikely.
2. **Sequence configs built from `quality_preset` inherited the flag.** Found by the
   `GNC_DIAGNOSTICS` fill canary as a single stray `decay` line among five `replicate` — in a run
   whose byte counts happened to be identical, so nothing else would have shown it.

**The canary is not optional for this feature.** A fill writes pixels nobody ever looks at, so it
is silent by construction: `GNC_DIAGNOSTICS=1` now prints which fill each path took and why, and
`scripts/meas_intra1_padding.py --canary` checks the shader byte-for-byte against an independent
Python reimplementation — passing in both modes, which is exact rather than approximate because
every term is a multiple of 1/64 for 8-bit input.

`scripts/meas_pad1_inter.py` keeps three arms on purpose: forced off, forced on, and the default.
The forced-on arm is **expected to regress** and is retained as the guard on this decision — if it
ever stops regressing, the inter half of PAD-1 is worth re-opening.

## Consequences

- ~4.6% of intra rate returned on stills, at unchanged visible quality; nothing changes for video.
- `GNC_PAD_FILL=decay|replicate` forces either mode on every path, for the two harnesses.
- Any recorded still figure taken before this commit is on the old padding. `BASELINE.md` and
  README's MEAS-9 table are updated; the GNC-vs-J2K intra gap narrows by about 4.6 points on the
  arms that are stills.
- **PAD-2** filed for the decoder-side re-replication that would collect the inter half.
