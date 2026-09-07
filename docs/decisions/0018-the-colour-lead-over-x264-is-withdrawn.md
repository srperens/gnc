# 0018 — The colour lead over x264 is withdrawn

**Date:** 2026-09-07
**Status:** accepted
**Supersedes the colour half of:** [0013](0013-the-headline-gap-figure-was-the-wrong-operating-point.md)
**Item:** CHROMA-2

## Decision

**GNC does not lead x264 on colour, and every claim that it does comes out of the documentation.**
The luma BD-rate from 0013 (+90.5%) is unaffected and stands.

## What was recorded

0013 measured, at rate matched to within 1%, GNC ahead of x264 on mean CIEDE2000 on all three
sequences (0.611 vs 0.684 on bbb_extended) while trailing 7.4–8.8 dB on luma, and read it as the
two codecs allocating rate differently between luma and chroma. That reading was careful and the
README said so explicitly rather than claiming the stronger thing. It also named the control:
give x264 the same allocation via `--chroma-qp-offset` and re-measure at the same total rate.

GOALS.md went further than the evidence and listed "keep the colour lead" as a **target**.

## What the control shows

`scripts/meas_chroma2.py`, on the three QUAL-1 sequences at 4:2:0 and 4:4:4, q=85, 24 frames,
ki=9, x264's crf bisected to land within 1% of GNC's byte count for each offset in
{0, −2, −4, −6, −8}:

**x264 wins colour on 6 runs out of 6.** On five of them it needs no offset at all — it is ahead
on dE00 at offset 0 *while simultaneously leading luma by 4.1 to 7.4 dB*. The control was designed
to see whether x264 could buy the colour back by paying luma for it; on this material it does not
have to trade.

## Why the old number was wrong

Three causes, all already documented in the repo, none of them noticed at the time:

1. **0013's colour table predates CHROMA-1 by an hour** (QUAL-1 18:25, CHROMA-1 19:16) and
   CHROMA-1 changes default output at **q ≥ 85 only** — precisely that table's operating point.
   COORDINATION already carried the rule "any q ≥ 85 file size measured before this is stale".
   Nobody applied it to the table it invalidated.
2. **The frame count is recorded two ways** — 24 in the QUAL-1 log, 17 in BASELINE.md.
3. **The source sequences were not in the tree.** bbb_extended, old_town_cross and crowd_run have
   never been in `fetch_test_frames.sh`; they were absent from the machine entirely until
   2026-09-07, so nothing had been re-checked against them.

The old figures do not reproduce: the same nominal configuration now gives 32% more bytes *and*
worse dE00 on the same content, which cannot both be true of one measurement.

## What was not chosen, and what it would have cost

- **Subtracting the colour-conversion floor** and quoting a "coding-only" dE00. The x264 arm goes
  RGB → yuv → encode → yuv → RGB and pays a conversion cost measured here at dE00 **1.057** on bbb
  at 4:2:0 — about 90% of the total — while GNC pays nothing, since it takes the reference PNGs
  directly and YCoCg-R is integer-reversible. Subtracting it would have produced a cleaner-looking
  number resting on the assumption that conversion and coding error add independently, which they
  do not. Left in, the floor makes the comparison *conservative*: the arm carrying the handicap
  wins anyway. `meas_chroma2.py` prints the floor and flags any arm that lands on it.
- **Reporting only 4:4:4**, where the floor is 3x smaller. It would have been the stronger-looking
  result (x264 ahead by 15–140%), but 4:2:0 is what the original claim was measured in and
  dropping it would have been choosing the flattering half of the evidence in reverse.
- **Keeping the row with a stronger caveat.** Rejected: the caveat was already there, correctly
  worded, and it did not stop the number reaching GOALS.md as a target to defend. A claim that
  needs a paragraph to stay honest is better withdrawn.

## Consequences

- README, GOALS, BACKLOG, BASELINE, POSITIONING and CLAUDE.md lose the colour lead. GOALS loses
  "keep the colour lead" as a target.
- **GNC has no measured advantage over x264 on any axis at the contribution operating point.** The
  +90.5% is the whole picture, not one side of a trade. Prioritisation that treated colour as a
  banked win has to be redone; intra luma is the only thing on the board.
- The `chroma_weight` frontier from 0014 is untouched and still steep. What is gone is the claim
  that GNC's position on it beats x264's.
- **Any two-codec colour comparison in this repo must state its conversion floor.** Three of the
  six runs had x264's dE00 equal to the floor to three decimals — its coded colour error was nil
  and the metric was reading the harness.
