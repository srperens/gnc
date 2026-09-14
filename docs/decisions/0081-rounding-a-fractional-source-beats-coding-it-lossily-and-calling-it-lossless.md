# 0081 — Rounding a fractional source beats coding it lossily and calling it lossless

Date: 2026-09-10
Item: BACKLOG BUG-45 (filed P3; it was P1)
Status: accepted
Reverses: the "warn, do not normalise" half of `0078` / BUG-45's original filing

## Context

`q=100` on any Y4M source was not lossless. Measured on `bbb.y4m`,
`benchmark-sequence -n 8 -k 8 -q 100`:

```
PSNR:  avg 32.89 dB  min 32.48  max 33.16  stddev 0.22
BPP:   avg 8.4193
```

**32.89 dB, from the mode whose entire purpose is bit-exactness**, and the per-frame variation is
not noise — it is what a viewer sees as a tile-shaped flicker that drifts sideways, because
LOSSLESS-2 re-codes every frame as an I-frame at this operating point and each one drifts
independently.

The encoder knew. It printed BUG-45's warning **on every frame**:

> `lossless settings (q=100 / qstep<=1) with non-integer input samples — the step-1 quantiser
> rounds, so this frame is NOT bit-exact (BUG-45)`

Nobody reads encoder stderr through a demo script. It was found by a person opening the web player
and saying the picture flickered.

## Why the samples are fractional, and it is not what BUG-45 assumed

BUG-45 was filed as a caller's problem — *"only an API caller can reach it"*, because PNG and Y4M
input is integral. **The Y4M half of that is wrong.** `Y4mReader::read_frame_rgb` converts with the
BT.601 limited-range matrix:

```
yy = 1.164 * (Y - 16);  r = yy + 1.596*Pr;  g = yy - 0.392*Pb - 0.813*Pr;  b = yy + 2.017*Pb
```

Those coefficients are not integers, so **every Y4M frame arrives fractional — including `C444`**.
It is the matrix, not the chroma upsample. Counted by the new canary on `bbb.y4m`: **6 150 134 of
6 220 800 samples per frame, 98.9%**.

The amplifier is `0080`'s, one producer over. The step-1 quantiser rounds the *residual*;
`med_predict.wgsl` is open-loop by design — the encoder predicts from `src`, the decoder from its
own reconstruction — so the rounding accumulates along each tile's diagonal scan rather than
staying a half-LSB.

## The decision

**Round the source to integers when the configuration is lossless**, in `encode_once`, replacing
the warning that stood there.

`0078` and BUG-45 both argued the opposite: *"the samples cannot be normalised without changing the
picture, so the honest options are to say so or to reject the frame."* **That reasoning weighs
rounding against keeping the picture, and keeping the picture was never on the table.** The
alternative to a half-LSB round on the input is what shipped: 33 dB, a drifting picture, and a
guarantee that reads true in every `is_lossless()` in the codebase. Rounding is strictly less
destructive than the thing it replaces, by about two orders of magnitude.

**What was not chosen:**

- **Refuse the frame.** Rejecting `-q 100` on Y4M rejects the most common video input to a
  contribution codec. `0078` already refused to refuse for a weaker reason.
- **Keep warning.** Tried, for the whole life of the feature. A warning that fires on 100% of
  frames of a supported input is indistinguishable from noise, and was.
- **Round in the Y4M reader.** Fixes the CLI and leaves the library — and the WASM and API
  callers — with the defect. The encoder is the one place every caller passes through.
- **Round on the GPU in the colour shader.** Cheaper than a CPU pass over 6.2M floats, and it
  hides the count. The pass only runs when the source is fractional *and* the config is lossless,
  so the normal path pays nothing; the canary is worth more than the microseconds.

**No bitstream change, no generation bump.** The decoder is untouched. Files written before today
still decode to exactly what they always decoded to — they were simply not the picture that was
put in.

## Result

| sequence (Y4M, 8 frames, ki=8, q=100) | PSNR before | **after** |
|---|---|---|
| bbb | 32.89 dB | **inf** |
| blue_sky | — | **inf** |
| crowd_run | — | **inf** |
| old_town_cross | — | **inf** |

SSIM 1.0000, max PSNR drop 0.00 dB. Four of four.

**Rate rises 8.42 → 10.42 bpp on bbb, +23.7%.** That is not a regression: the smaller file was
smaller because it was discarding the picture. Comparing a lossy file's rate to a lossless one's
was never a valid comparison, and any q=100 rate figure taken from a Y4M source before today is
superseded.

**Lossy output does not move.** Re-encoded against demo artefacts built by the previous binary,
byte-identical at q=25 (1 153 801), q=50 (2 352 009) and q=75 (6 190 783) — the new work is behind
`if config.is_lossless()`.

Guard: `tests/bug45_fractional_source_lossless.rs`, which asserts the invariant (*a lossless
encode of a fractional source reproduces `round(source)` exactly*) rather than a number.
Mutation-tested: dropping the rounding gives **max abs error 134** on a 0–255 scale, while the
integral-source control still passes.

## What this does not settle

`q=100` from Y4M is now bit-exact **with respect to the RGB the reader produces**, not with respect
to the file's original Y'CbCr samples. The BT.601 matrix is not invertible in integers, so a
Y4M → RGB → YCoCg-R → RGB → Y'CbCr round trip cannot return the input whatever the codec does.
"Lossless" therefore means something narrower for Y4M input than for PNG input, and nothing in the
CLI says so. **LOSSLESS-4.**
