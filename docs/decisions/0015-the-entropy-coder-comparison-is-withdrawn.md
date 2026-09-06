# 0015 — The Rice-vs-rANS compression comparison is withdrawn, not updated

**Date:** 2026-09-06
**Status:** Accepted (ENT-2 opened to re-measure; the default is unaffected)

## The decision

The README's Entropy Coders table carried a Compression column — **Rice 4.01 bpp @ q=75 against
rANS 4.22 bpp @ q=75** — and the `benchmark --rans` example three paragraphs above it was commented
"better compression", which said the opposite. The column is **removed** rather than corrected, and
the comment now points at the table instead of making a claim.

The alternative was to keep the column with Rice's current 4.53 bpp against rANS's 4.22. That would
have been worse than the contradiction it replaced, because it reads as a measurement and is not
one.

## Why the column could not simply be updated

Three changes sit between the two figures, and only the first is shared by both coders.

1. **The operating point moved.** q=75 went from 42.17 dB / 3.83 bpp (perceptual weights, #64) to
   44.84 dB / 4.53 bpp (uniform weights, 5 levels at q≥25 after BUG-6). Different quality, so
   different rate; 4.53 and 4.22 are not points on one curve.
2. **GP17 shrank Rice's headers at bit-identical output** — Rice-coded stream-length tables. It
   moved Rice's bpp and left rANS's untouched by construction.
3. **The 4.01 figure predates both** and does not match any row in BASELINE's history, including
   the perceptual-weights row it should correspond to. Its provenance is unrecoverable.

So the two numbers differ by an operating point, a Rice-only improvement, and an unknown. **The
ordering they imply is not supported by anything currently in the repository**, and BASELINE.md's
single-frame table is Rice-only, so there is no rANS figure to fall back on.

## What is not being claimed

That rANS might now be better. That is possible arithmetically — 4.53 against 4.22 — but the
comparison is invalid in the direction that would show it too. The honest statement is that the
ordering is unmeasured.

**The prediction, recorded so ENT-2 can falsify it:** Rice still wins, and by more than 4.01 vs
4.22 suggested, because header overhead scales with stream count and Rice runs 256 streams per tile
to rANS's 32 — which is exactly the overhead GP17 attacked. If ENT-2 comes back the other way, this
record was wrong and the default deserves re-examination.

## What this does not change

**The default stays Rice, and no bpp figure would change that.** Rice is the default because it
eliminates the sequential state chain that limits rANS — 256 fully independent streams, no shared
state, < 1 KB of shared memory against rANS's 16 KB of frequency tables. That is a GPU-parallelism
argument, which is the project's whole premise; a compression column is a tiebreak it does not need.

## The wider cleanup this came out of

The README had three separate copies of the same 2026-02-27 figures, and the main table had already
been corrected without the copies following:

| block | said | BASELINE says |
|---|---|---|
| Quality Spectrum | q=75 → 42 dB, q=50 → 37, q=25 → 33 | 44.84, 40.30, 35.51 |
| Entropy Coders | Rice 4.01 bpp @ q=75 | 4.53 |
| `benchmark --rans` comment | rANS = better compression | table said worse |

All three now either point at [BASELINE.md](../../BASELINE.md) or carry no number. The rule the
README states for itself — BASELINE is the single source, and nothing else in the file quotes a
figure — is the only thing that stops this recurring, since it has now drifted twice.

Two further errors found in the same pass, both recorded here because they are the same failure
mode:

- **CLAUDE.md had the default entropy coder backwards** — "rANS (default)", "Rice+ZRL (fastest)" —
  since before `--rans` became an opt-in flag. Four other sessions read that file as reference.
- **The shader count was wrong in two files at once**: README said 42, CLAUDE.md said 32,
  `src/shaders/` holds 62. The README's count is now gone rather than corrected, because a count in
  prose rots on the next shader added.
