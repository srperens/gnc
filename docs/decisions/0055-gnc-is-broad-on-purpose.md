# 0055 — GNC is broad on purpose, and "pick one segment" is rejected

**Date:** 2026-09-07
**Status:** adopted — project owner's decision, stated explicitly
**Scope:** what GNC is for. Sets the targets in GOALS §1, supersedes the central recommendation of
docs/POSITIONING.md §2, and changes the priority order in BACKLOG. No effect on the bitstream or on
any measurement already taken.

> **Renumbered 0055 from 0018 by BUG-19 on 2026-09-08.** `0018` was taken by ENT-2's
> [the entropy coders are level](0018-the-entropy-coders-are-level-and-0015s-prediction-was-wrong.md),
> committed one minute earlier, so two files carried `0018` on `main` for a day. Nothing in the
> record below changed. Commit messages, and any citation older than this date, call it `0018`.

## The decision

GNC is to be **good at many things rather than excellent at one**. All of the following are in
scope simultaneously, and none of them is the "primary" one:

| axis | target |
|---|---|
| intra | strong — H.264-intra-class quality per bit |
| inter | strong — a static studio shot should cost almost nothing |
| quality range | heavy compression, through visually lossless, to bit-exact lossless |
| chroma | 4:2:0, 4:2:2, 4:4:4, at 8 and 10 bits |
| uses | contribution, mezzanine, **archival**, low-latency preview, browser playback |
| parallelism | massively parallel; portable across every GPU with WebGPU/Vulkan/Metal/DX12 |
| compression | roughly **H.264-class, across that whole range** |

And explicitly: **several internal strategies, selected by quality and bitrate, is a legitimate
design.** If one mechanism cannot cover the range, the codec selects per operating point.

## What was rejected, and what it would have cost

**Rejected: narrowing to one segment.** docs/POSITIONING.md §2 concluded *"commit to cloud
mezzanine, proxy and preview. Stop claiming both,"* on the reasoning that live contribution and
cloud mezzanine "share almost no buyers, sales channels, or requirements" and that the requirements
"diverge by years of work". That reasoning is sound about *channels* and was allowed to decide the
*codec*.

The cost of having followed it, stated plainly because it was already being paid:

- **It set the operating point everything was measured at.** The README said so in as many words.
  Work was steered to q ≥ 85 and the low end went unmeasured for weeks — which is the mirror image
  of the TUNE-5 error (measured q=15-50, shipped, then found to cost 10.3 dB at q=99).
- **It made "single-digit improvements are pointless" sound like strategy.** Against a headline gap
  believed to be 5-7x they were arguably pointless; the gap is ~1.9x (QUAL-1) and every such
  judgement predating 2026-09-06 was made against a denominator 3x wrong.
- **It devalued the levers that pay everywhere.** Entropy coding is the largest single gap against
  H.264 and the one mechanism that improves intra, inter, lossless and every chroma format at once.
  Under a one-segment framing it competes with segment features; under this decision it leads.
- **It would have thrown away the codec's actual distinguishing property.** There are already dozens
  of good codecs for any single niche. A patent-free, GPU-portable, massively parallel codec that
  spans heavy compression to bit-exact lossless is not one of many.

**Not rejected: everything factual in POSITIONING.** The market requirements, the sourced external
claims (EBU R 153, TR 091, TR 092, the NVENC session limits, the JPEG XS licence terms) and every
measurement stand. Its §2 table is kept, reframed to separate two entry costs that the "pick one"
framing had conflated:

- **Format requirements are engineering and in scope:** 10-bit, 4:2:2, constant bitrate, sub-frame
  latency, multi-generation robustness. Each is a measurable target.
- **Channel requirements are not codec work and are not being pursued:** SDI or ST 2110 I/O, an
  NMOS IS-04/IS-05 control plane, JT-NM Tested certification. These gate entry into EBU TR 091's
  evaluation. That is a reason not to claim a live-contribution *product* — not a reason to narrow
  the codec.

**Also considered and not adopted: leaving the documents alone and treating the breadth as
implicit.** GOALS §1 already said "GNC is still both intra and inter", so the intra/inter half was
never lost. But the same section opened by declaring GNC a contribution codec and listing
"consequences that follow from this positioning", and that is what got quoted in practice. A
principle that contradicts the first paragraph of the document it lives in does not survive contact
with a new session reading top to bottom. Rewriting was cheaper than repeating the correction.

## What follows for the work

1. **Entropy coding leads.** It is the largest gap against H.264, and the only lever that pays on
   every row of the table above. MEAS-9 (measured the same day) sharpens this: JPEG 2000 in
   irreversible 9/7 mode uses **the same transform family as GNC — 9/7 wavelet, five levels — and
   still needs 38-60% fewer bits** at matched quality. When the transform is the same, the gap is
   not the transform. J2K's difference is EBCOT.
2. **The low end has to be measured again.** Not instead of the high end — as well as it. Any
   result that names one end only is incomplete under this decision.
3. **Inter is not a second-class axis.** The 2026-09-06 finding that inter breaks even at
   contribution quality is correct *at that operating point* and was read as "stop spending inter
   effort". Under this decision it means the opposite: inter has to work across the range, and the
   entropy gap on inter residuals is unmeasured.

## Cross-references

- [GOALS.md](../../GOALS.md) §1 — the authority on scope; §6 now lists narrowing as a non-goal
- [docs/POSITIONING.md](../POSITIONING.md) §1-2 — reframed, with the rejection recorded inline
- [RESEARCH_LOG.md](../../RESEARCH_LOG.md), 2026-09-07 — MEAS-9, the measurement that makes the
  entropy-coding priority quantitative
- [0013](0013-the-headline-gap-figure-was-the-wrong-operating-point.md) — the operating-point error
  this decision generalises
