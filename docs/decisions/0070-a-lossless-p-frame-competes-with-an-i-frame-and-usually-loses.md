# 0070 — A lossless P-frame competes with an I-frame of the same picture, and on camera content it loses

Date: 2026-09-08
Item: BACKLOG LOSSLESS-2 (P2)
Status: accepted
**Changes a default.** At `q=100`, a P-frame that serialises larger than the previous I-frame is
re-coded as an I-frame. Nothing below `q=100` is affected — verified byte-identical.

## The defect

BUG-39 (`0064`) made `q=100` video bit-exact. That turned an unanswerable comparison into an exact
one, and the answer was bad: coding P-frames at `q=100` costs **+38% to +69% of the whole
sequence** on camera content.

8 frames, `q=100`, 4:4:4, container payload, `GNC_LOSSLESS_INTRA_RECODE=0` against `-k 1`:

| sequence | ki | I+P | all-intra | I+P costs |
|---|---|---|---|---|
| crowd_run | 2 | 35 712 641 | 25 855 950 | **+38.1%** |
| crowd_run | 9 | 43 003 751 | 25 855 950 | **+66.3%** |
| old_town_cross | 2 | 35 209 443 | 25 246 827 | **+39.5%** |
| old_town_cross | 9 | 42 778 003 | 25 246 827 | **+69.4%** |
| blue_sky | 2 | 23 911 136 | 17 294 529 | **+38.3%** |
| blue_sky | 9 | 28 942 166 | 17 294 529 | **+67.4%** |
| bbb (animation) | 2 | 25 484 805 | 25 899 452 | −1.6% |
| bbb (animation) | 9 | 25 183 274 | 25 899 452 | −2.8% |

With no quantiser to discard anything, a motion-compensated residual is noise-like and costs more
to code than the MED-predicted frame it replaces. This is INTER-1 / `0023` (inter is a wash at
q=85–99) carried past the wash into a loss.

## Why this one is decidable on bytes alone

Every frame at `q=100` decodes bit-exact, so **the two candidates for a frame are the same
picture.** There is no rate/quality trade, no BD-rate, and nothing for CLAUDE.md's metric table to
arbitrate — the only axis is bytes. It is also a **local** choice: the reference every later frame
predicts from is the source frame whichever way this one was coded, so a frame's cost does not
depend on what the frames before it chose. That is what makes the per-frame minimum *achievable*
rather than an oracle bound, and it is the property this decision rests on.

At `q < 100` the same comparison is not free — the candidates differ in quality — which is why the
gate is `is_lossless()` and not a quality threshold. LOSSLESS-3 is that other question, and it now
has numbers.

## The decision

**Code the P-frame; if it is larger than the previous I-frame, re-code the frame as an I-frame and
keep that.** One extra frame encode, only on the frames that lose. `GNC_LOSSLESS_INTRA_RECODE=0`
restores the old behaviour for measurement.

Result, same eight points: **the shipped encoder equals the per-frame minimum to the byte on 8 of
8** — −27.6% to −41.0% on camera content, **±0 on animation** (bbb keeps every P-frame). All 8
frames of every point still decode bit-exact: 32 of 32 verified against the source PNGs through
the real container (`encode-sequence` → `.gnv` → `decode-sequence`, ffmpeg rawvideo md5), on both
the re-coded and the kept-P paths.

## What was not chosen

- **"`q=100` video is all-intra by construction", which is what FFV1 does.** Simplest, and it
  throws away two things. Animation: bbb loses its −2.8%. Static content: measured in
  `tests/lossless2_intra_recode.rs`, a lossless P-frame over a static picture is **156 B against
  the I-frame's 61 414 B — 394× cheaper.** A contribution codec sees locked-off cameras and
  graphics; refusing P-frames outright would inflate exactly that content by orders of magnitude.
- **Decide once per sequence (probe the first P-frame, latch).** Captures 100% of the gain on all
  eight homogeneous points above — the sign never varies inside a shot — and fails on mixed
  content. Measured on a synthetic shot cut (4 frames bbb + 4 frames crowd_run, ki=9): per-frame
  **25 562 037 B**, against **32 938 051 B** for the latch that keeps P (what a first-P probe
  decides here, −22.4%) and **25 879 801 B** for all-intra (−1.2%). **Per-frame beats the better
  of the two per-sequence answers**, because it keeps the animation shot's P-frames and refuses the
  camera shot's.
- **Code both ways per frame and keep the smaller (RATE-2's shape, `0036`).** The rigorous version,
  and it buys **0.00%** over the previous-I-frame estimate on all eight points: I-frame sizes vary
  by **±0.4%** inside a shot, and at a shot change the scene-cut detector inserts a keyframe, which
  refreshes the estimate. It would double the encode of every P-frame to buy nothing measurable.
  The estimate is the cheap half of the same idea.
- **Ship it for B-frames too.** Out of scope: `quality_preset` vetoes the B-pyramid (BUG-5), so
  `q=100` runs the P-only path, and the two other `encode_pframe` call sites (the B anchor and the
  temporal-wavelet path) are untouched. If B-frames ever come back at `q=100` they need the same
  comparison; noted here rather than built blind.

## Consequences

- `docs/BITSTREAM_SPEC.md` is unchanged. This produces *more* I-frames, which every existing
  decoder already handles — and more keyframes means better seeking, not worse.
- A `q=100` camera sequence now codes as `8I+0P`, so the full-pel MV rounding BUG-39 shipped
  (`0064`, +1.83% mean) no longer applies to that content: there are no inter frames to round for.
  It still pays on animation and static content, where P-frames survive. `0064`'s trade is
  unchanged where it applies.
- Encode time on the losing path is one wasted P-frame encode per frame. Not measured on an idle
  machine (eight sessions, one GPU), so no wall-clock figure is quoted. If it matters, the fix is a
  latch that stops probing after N consecutive losses and re-arms at each keyframe — cheap, but it
  is a tuning rule and nothing measured today needs it.
