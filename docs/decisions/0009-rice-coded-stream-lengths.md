# 0009 — Golomb-Rice stream-length tables (GP17)

**Date:** 2026-09-06
**Status:** Accepted

## Why this, and not another prediction experiment

GNC needs 5–7× H.264's bitrate for the same VMAF on video (MEAS-1). MEAS-4 located the gap in
prediction quality rather than the coding model, and the four follow-ups that took that seriously —
multi-reference P-frames, the sub-pel interpolation filter, motion-search quality, and the earlier
transform/entropy sweep — all came back negative. Continuing to guess at prediction was not
producing anything.

So this looked at the bit budget instead: not "what could predict better" but "what is a P-frame
actually made of". The answer was surprising enough to be worth acting on immediately.

## What was found

Tile headers are 5% of an I-frame and **17–31% of a P-frame**. Almost all of it is one field.

GNC's architecture gives every tile 256 fully independent entropy streams — that is what makes the
decode embarrassingly parallel, and it is a deliberate, load-bearing choice. The consequence is
that each tile must carry a 256-entry table of stream lengths. As byte-aligned varints that is
~256 bytes per tile and ~30 KB per 1080p frame, **whether the tile holds 200 KB of coefficients or
400 bytes**. It is the fixed cost of the parallelism, and it had never been measured.

That fixed cost is why the share grows as quality drops: the coefficients shrink, the table does
not. Which is precisely the operating point GNC is positioned for (contribution, low bitrate).

## Why Rice, measured rather than assumed

Three encodings were priced on the real tables before any of them was implemented — varints (the
existing scheme), Exp-Golomb order 0 with a zero bitmap (what GP16 already does for motion
vectors), and Golomb-Rice with a per-tile `k` chosen by exhaustive search.

Rice wins at every point, by 20–61%. Exp-Golomb *loses* to varints on high-quality I-frames (+4%
at q=50, +24% at q=75), where lengths cluster near the 4096-byte stream cap and its unary prefix
gets long. So the obvious reuse of the existing GP16 machinery would have been the wrong choice,
and only measuring the alternatives showed that.

A per-tile best-of-three with two signalling bits was also priced. It never beat plain Rice by
more than rounding, so there is no mode flag — the encoder still compares against varints and can
fall back, but that is a guard, not a feature.

The reason Rice fits is the shape of the data: within a tile the 256 streams have a characteristic
size with a long tail. That is the distribution Rice exists for, and it is the same argument the
codec already makes for coefficient magnitudes. The length table simply never got the same
treatment.

## Result

Headers only — the coefficients are untouched, the decoded image is bit-for-bit identical, and the
entire size reduction is gain rather than an RD tradeoff.

| | low rate | mid | high |
|---|---|---|---|
| stills (4 images) | −2.8% to −7.6% | −1.8% to −4.5% | −0.5% to −2.7% |
| video (2 sequences, 8 frames) | −6.9% to −7.6% | −3.7% | −1.4% to −1.5% |

No shader change: the length table is parsed on the host before the GPU sees the streams.

## The near miss

The first decoder capped the Rice quotient at 64, as a guard against corrupt input spinning
forever. That silently truncated any length above `64 << k` — over 2048 bytes at a typical k=5,
which occurs on high-quality I-frames. Files would have encoded and decoded without complaint,
just wrong.

The lesson is about where the termination guarantee actually comes from. `BitReader::get_bit`
returns 0 past the end of the buffer, so a truncated or all-ones stream always terminates on its
own; the cap was never load-bearing. And an *optimal* per-tile `k` deliberately leaves single
outlier streams with a long unary run rather than raising `k` for all 256 entries — so any cap
near the typical quotient is guaranteed to hit real data. The cap is now 65536, a sanity bound
well clear of the longest legitimate run, and the unit test includes a 4095 outlier among small
values specifically to keep it honest.

## Generation handling

GP17 was necessary because a GP16 decoder has no tile flag 0x08 and would misparse. While bumping
it, the eight `is_gpXX` booleans ORed into a dozen chains became one `gen: u32` with `gen >= N`
tests — `is_gp15` and `is_gp16` were already ORed twice into the same assert, harmlessly but as a
clear sign of where that pattern was heading. A future generation is now one table entry.

Both directions verified: a GP16 file decodes correctly in the GP17 binary, and a GP17 file in the
GP16 binary is refused with "invalid magic" rather than read as garbage.

## What this does not do

It does not touch the 5–7× video gap. That gap is not in the headers, and this decision record
should not be read as progress against it. What it does say is that the bit budget was worth
reading before designing another experiment — the largest single win available today was sitting
in a field nobody had priced.
