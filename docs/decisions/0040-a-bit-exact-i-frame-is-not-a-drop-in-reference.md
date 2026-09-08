# 0040 — RATE-3: a bit-exact I-frame is not a drop-in reference, and the source-copy shortcut is rejected

**Date:** 2026-09-08
**Status:** accepted
**Item:** BACKLOG RATE-3 (P1) — **not fixed.** Redirected with a root-cause direction and one
decisive measurement named.
**Changes no default.** The tree is back to exactly what `0036` shipped, verified byte-for-byte on
stills and on P-frame PSNR for sequences. One thing is kept: the *order* of `encode`'s two
candidate encodes, which was a latent bug.

## What this had to decide

`0036` shipped RATE-2's lossless fallback for stills only, because letting it reach sequence
I-frames made the P-frames referencing them decode at 9.80 dB against 60.69 dB. RATE-3 asked
whether that is fixable, on the reasoning that a bit-exact reference ought to be the *best*
reference there is — no drift, no propagated error — so the inter half of RATE-2's win should be
larger than the intra half rather than zero.

**It is not fixable by the obvious route, and the obvious route is now measured rather than
assumed.**

## Established, with numbers

1. **`q=100` video is broken on `main` today, and RATE-2 has nothing to do with it.** P-frames
   referencing a bit-exact MED I-frame decode at **12.45 dB** (crowd_run, 4 frames, ki=2). Filed
   as **BUG-39**. Nothing in the repository recorded this; every lossless claim in README, GOALS
   and BASELINE is about stills, and README's "bit-exact lossless at q=100" sat in a sentence
   about the video pipeline. Corrected.
2. **With lossless *wavelet* I-frames instead of MED** (`GNC_MED=0`), the same case reads
   **44.18 dB**. So the defect is partly MED-specific and partly not.
3. **RATE-2's double encode clobbers a GPU side channel, and that was a real latent bug.**
   `local_decode_iframe_gpu` builds an I-frame's reference from the quantised planes `encode()`
   leaves behind (`Y → mc_out, Co → ref_upload, Cg → plane_b`), so whichever candidate runs
   *last* decides the reference. With the sibling second, bbb q=95 — where the lossy file is the
   smaller one and is correctly kept — gave P-frames at **9.83 dB** and **+40.55%** bytes.
   **Fixed by running the sibling first**, which is the one change kept from this item.
4. **Taking the reference from the source instead is not sufficient.** A bit-exact frame decodes
   to its input, so the reference can be the colour-converted source, which both forward
   transforms only read. Implemented and measured: q=100/MED improved from 12.45 to **21.37 dB**
   — broken either way — and in RATE-2's own window it gave P-frames at **51.8–52.2 dB** against
   **60.6** for the ordinary lossy reference. **Reverted.**
5. **The reference's quality is not the limit, and this is the finding that matters.** With a
   perfect reference P-frames read **52.17 dB**; with a deliberately broken 34.30 dB reference
   they read **34.17 dB**; with the ordinary 59.5 dB lossy reference they read **60.62 dB**. So
   P-frames track a *poor* reference faithfully but are capped near 52 dB when the reference is
   perfect. **A better reference producing a worse P-frame cannot be a quantisation ceiling — it
   means the encoder and the decoder disagree about what the reference is.**

## Two hypotheses refuted, so nobody spends the afternoon again

- **"The MED inverse is missing from `local_decode_iframe_gpu`."** True, and necessary, and not
  sufficient — point 4.
- **"The colour transform's rounding mode desynchronises the reference from the current frame."**
  `color_convert.wgsl` switches its lifting between `floor(x/2)` and `x*0.5` on
  `config.is_lossless()`, which predicted a ~0.5 LSB chroma error and therefore ~54 dB — close
  enough to the observed 52 to be worth testing. **Refuted twice:** forcing the fractional lifting
  for a lossless config drops the I-frames to 34.30 dB and the P-frames follow at 34.17, so it
  does not lift the cap; and a MED sibling and a lossless-*wavelet* sibling give **identical**
  P-frame PSNR (52.17 / 52.04) although only one of them uses MED at all.
- **"The padded geometry differs when `wavelet_levels = 0`."** Refuted by reading:
  `padded_width = tiles_x * tile_size` (`lib.rs:86`), independent of the level count, and
  `lossless_sibling` carries the caller's tile size.

## The decision

**Keep `0036`'s gate. Do not lift it on the strength of a plausible mechanism.** Three attempts at
this item produced two refuted hypotheses and one real but insufficient fix, which is the pattern
CLAUDE.md's "if the same bug resurfaces after two fix attempts, stop and diagnose the root cause
properly" is written for.

**The one measurement that settles it, and it is not one of the ones already run:** read back the
encoder's `gpu_ref_planes` after a lossless I-frame and diff them against the decoder's own
reference for that same frame. Point 5 says they differ; nothing measured says *how*, and every
remaining hypothesis is downstream of that diff. It needs readback plumbing on both sides, which is
why it was not done here — but it is a bounded piece of work with an unambiguous answer, unlike
another round of mechanism-guessing.

## What was not chosen

- **Shipping point 4's improvement anyway** (12.45 → 21.37 dB on q=100 video). Both numbers are
  broken, there is no criterion under which 21 dB is a result, and it would have changed shipped
  output for no defensible reason.
- **Teaching `local_decode_iframe_gpu` the MED inverse properly.** Point 5 says the reference is
  not merely reconstructed by the wrong transform, so this would be work aimed at the wrong
  target. Worth doing *after* the diff, if the diff points at it.
- **Removing the reference side channel** — having `encode()` publish state that another function
  reads is what made point 3 possible, and an explicit reference API would prevent that whole
  class. It is a refactor of the hottest function in the encoder and it needs its own item and its
  own byte-exactness gate.

## Caveat

**BUG-39 is the more urgent half of what this item found.** RATE-3 is about lifting a gate to win
rate; BUG-39 is a shipped codec producing 12.45 dB video at its highest quality setting. The two
share a cause and should probably be taken together, but if only one is taken it should be BUG-39.
