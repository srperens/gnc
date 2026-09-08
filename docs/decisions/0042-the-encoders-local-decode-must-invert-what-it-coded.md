# 0042 — BUG-39: the encoder's local decode must invert what it coded, and the frame header must say what it is

**Date:** 2026-09-08
**Status:** accepted
**Item:** BACKLOG BUG-39 (P1) — **two of three causes fixed and proven.** `q=100` video goes from
12.45 dB to 26.30 dB and stops drifting; it is not yet bit-exact.
**Bitstream-visible:** every P-frame's `transform_type` byte now says `Wavelet`, which is what the
encoder always used. Unchanged for wavelet configs, corrected for `q=100` and `--dct` sequences.

## What was broken

A `q=100` sequence coded bit-exact I-frames and then P-frames that decoded at **12.45 dB**. The
lossless *stills* path was never wrong — every lossless figure GNC publishes is a still — and the
sentence that implied otherwise carried no number, so two documentation sweeps had passed over it.

## The measurement that found it, after two that did not

RATE-3 spent three attempts on mechanism hypotheses and refuted two of them (`0040`). What
settled it was the diff the encoder and decoder can both already produce: `read_reference_planes`
exists on **both** pipelines, and `test_pframe_reference_matches_decoder` has been comparing them
since before this — **with `CodecConfig::default()`, which is qstep 4.0 and the wavelet.** The
lossless case had never been run through it.

Running it, encoder reference against decoder reference for the same I-frame:

| | max \|enc − dec\| | pixels differing |
|---|---|---|
| q=99, wavelet (control) | **0.0000** on all three planes | 0 / 65 536 |
| q=100, MED — Y | **33.0000** | 63 029 / 65 536 |
| q=100, MED — Co | 0.0000 | 0 |
| q=100, MED — Cg | **64.0000** | 64 266 / 65 536 |

**Two hours of mechanism-guessing against ten minutes of diffing the two things that must be
equal.** The lesson is not "we lacked a tool" — the tool was already there, with a test using it —
it is that the test's *configuration* was the untested axis.

## Cause 1 — the encoder's local decode had no MED inverse

`local_decode_iframe_gpu` dequantised and called `transform.inverse` unconditionally, so a MED
frame's reference was built by running the inverse **wavelet** over a MED residual. The decoder has
always done this correctly (`decoder/gpu_work.rs:350`, `:373`) and `med.inverse` has always
existed; only the encoder's copy of the reconstruction was missing the branch.

**Fixed.** The table above now reads 0.0000 on every plane at q=100 as well as q=99, and
`lossless_iframe_reference_matches_the_decoders` asserts it with the q=99 wavelet case as its
control.

## Cause 2 — a P-frame advertised a transform it had not used

`encode_pframe` codes its residual with `transform.forward` **always** — it never calls
`med.forward` — but it cloned the sequence config into the frame it emitted, so a `q=100` P-frame
carried `transform_type = MedPredict`. The decoder branches on that byte for P-frames as well as
I-frames, so it obeyed and inverted a MED prediction over a wavelet residual. The error then
compounded down the GOP.

**Fixed** by setting `res_config.transform_type = Wavelet` where the residual config is built — the
label now describes what the code does. Measured on crowd_run, 10 frames, `q=100`:

| | ki=2 P-frames | ki=9 P-frames |
|---|---|---|
| before both fixes | 21.35 – 21.48 dB | **9.06 – 21.37 dB** |
| after cause 1 | 21.35 – 21.48 | 9.06 – 21.37 |
| after both | **26.30 – 26.76** | **21.63 – 26.51** |

The ki=9 span collapsing from 12.3 dB to 4.9 dB is the drift disappearing: each P-frame's reference
is now the one the decoder has, so the error stops accumulating.

**This also corrects `--dct` sequences**, whose P-frames were mislabelled the same way and are also
wavelet-coded. **Not measured** — flagged rather than claimed.

## Cause 3 — not fixed, and it is a design question

P-frames at `q=100` are still **26 dB, not bit-exact.** They are lossy by construction: the
residual is quantised at the P-frame taper (up to 1.25× the intra step) with a dead zone, and at
`q=100` `wavelet_levels` is 0. So "bit-exact lossless at q=100" is true of a still and false of a
sequence for a reason that has nothing to do with the two bugs above: **nothing in the P-frame path
asks to be lossless when the sequence is.**

That is a decision, not a patch — it means the P-scale taper and the dead zone must be suppressed
when the configuration is lossless, and it needs its own before/after on rate as well as quality.
BUG-39 stays open on it.

## What was not chosen

- **Stopping at cause 1.** It moved q=100 from 12.45 to 21.37 dB with the drift intact, which is
  the kind of partial number that reads as progress and hides a second cause.
- **Labelling the P-frame in `encode_from_wavelet_coeffs` instead.** Tried first, and it changed
  **nothing** — the P-frame's config comes from `res_config` in `encode_pframe`, not from those
  emitters. Reverted rather than left in with a confident comment attached to an inert change.
- **Teaching `encode_pframe` to use MED on residuals.** MED is a spatial predictor and a
  motion-compensated residual is not a spatial signal. The label was wrong, not the transform.
- **Making P-frames lossless here.** Cause 3, above.

## Caveats

- **Only the reference planes are proven equal, at 256×256 on a gradient.** The test asserts the
  thing that must hold; it does not sweep content or geometry.
- **`--dct` video is corrected and unmeasured.**
- **q=100 sequences are still not what the label promises.** 26 dB is not lossless. The README
  correction RATE-3 landed stays accurate until cause 3 is fixed, and its 12.45 dB figure should be
  updated to 26.30 dB rather than deleted.
