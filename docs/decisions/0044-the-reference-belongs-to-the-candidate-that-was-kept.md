# 0044 — RATE-3: the sequence gate is lifted, and the reference belongs to the candidate that was kept

**Date:** 2026-09-08
**Status:** accepted
**Item:** BACKLOG RATE-3 (P1) — **fixed and shipped.**
**Changes a default:** RATE-2's lossless fallback now runs inside the I/P sequence path at
q = 95..=99, where `0036` refused it. Stills are untouched. Base commit `f3f7254`
(post-INTRA-2, so the q=95 rung already carries `0041`'s dead-zone floor and its
`dead_zone_referenced` split).

## What this had to decide

`0036` shipped RATE-2's lossless fallback for stills only. Letting it reach sequence I-frames made
the P-frames referencing them decode at 9.80 dB, and `0040` spent three attempts on why, refuted
two mechanisms, and stopped with a named measurement it did not run. This item asks the same
question `0040` did — is a bit-exact I-frame usable as a reference? — with BUG-39's cause 1 fixed
underneath it.

**It is, and the gate is now unnecessary. The mean is −4.28% of sequence bytes at unchanged
quality, up to −13.16%.**

## The defect the gate was hiding, which is not the one `0040` found

`encode` codes two candidates and returns the smaller file. `encode_once` *also* leaves the
quantised planes on the GPU (`Y → mc_out`, `Co → ref_upload`, `Cg → plane_b`), and that side
channel is how `local_decode_iframe_gpu` builds a reference without a CPU entropy decode. Only the
**last** encode's planes survive.

`0040` found half of this: with the sibling running *second*, a kept **lossy** frame got the
sibling's planes. It fixed that by running the sibling first, and that fix is on `main`.

The other half is the mirror image and was unreachable while the gate stood: with the sibling
running first, a kept **bit-exact** frame gets the *lossy* candidate's planes. Since BUG-39 cause 1
(`0042`) taught that function to branch on `transform_type`, the failure is now `med.inverse` run
over wavelet coefficients. Measured on crowd_run q=99 ki=2 with the gate lifted and nothing else
changed: I-frames bit-exact at −32.3%, **P-frames 5.93 dB** and 4.99 → 8.89 MB, sequence +24.9%.

So the ordering is not fixable by ordering. Whichever candidate runs last, the other one can win.

## The fix

`encode_as_reference` (`pipeline.rs`), used by the sequence path in place of `encode`: when the
bit-exact sibling wins, re-run it so the side channel is its own. The re-encode is deterministic,
so the frame returned is the same bytes `encode` chose — `debug_assert` checks that rather than
assuming it.

**It costs a third encode of that frame**, and only on frames where the bit-exact candidate wins.
That is why it is a separate entry point rather than behaviour inside `encode`: a still has no
reference to build and must not pay for one. The still path is untouched and its output is
unchanged.

**Canary:** `GNC: RATE-3 reference repair — …` prints on exactly the frames that pay the third
encode, so "the repair ran" is distinguishable from "the sibling never won". The existing RATE-2
canary still prints both candidate sizes on every frame that codes twice.

## Measured

`scripts/meas_rate3.py`, three sequences, q ∈ {95, 99}, ki ∈ {2, 9}, 8–10 frames, 4:4:4. Both arms
are the same binary and the same command; only `GNC_LOSSLESS_FALLBACK` differs, so these are exact
differences and not BD-rate estimates. The off arm was verified byte-identical to the shipped
binary at `f3f7254` on crowd_run q=99 ki=2 before anything was changed.

| sequence | q | ki | bytes off | bytes on | Δbytes | worst P off | worst P on | ΔP | I bit-exact |
|---|---|---|---|---|---|---|---|---|---|
| bbb | 95 | 2 | 19 110 162 | 19 110 162 | +0.00% | 53.12 | 53.12 | +0.00 | no |
| bbb | 95 | 9 | 17 896 443 | 17 896 443 | +0.00% | 53.00 | 53.00 | +0.00 | no |
| bbb | 99 | 2 | 26 165 418 | 26 316 852 | **+0.58%** | 60.68 | 60.68 | +0.00 | yes |
| bbb | 99 | 9 | 24 610 862 | 24 708 364 | **+0.40%** | 60.67 | 60.67 | +0.00 | yes |
| crowd_run | 95 | 2 | 39 627 016 | 37 399 616 | −5.62% | 52.84 | 52.84 | +0.00 | yes |
| crowd_run | 95 | 9 | 40 034 340 | 38 966 554 | −2.67% | 52.84 | 52.84 | +0.00 | yes |
| crowd_run | 99 | 2 | 48 799 611 | 42 377 516 | **−13.16%** | 60.62 | 60.62 | +0.00 | yes |
| crowd_run | 99 | 9 | 49 328 550 | 46 535 419 | −5.66% | 60.61 | 60.61 | +0.00 | yes |
| old_town_cross | 95 | 2 | 38 866 054 | 37 010 354 | −4.77% | 52.84 | 52.84 | +0.00 | yes |
| old_town_cross | 95 | 9 | 39 839 530 | 38 898 505 | −2.36% | 52.83 | 52.84 | +0.01 | yes |
| old_town_cross | 99 | 2 | 48 088 416 | 42 012 499 | **−12.63%** | 60.61 | 60.61 | +0.00 | yes |
| old_town_cross | 99 | 9 | 49 173 140 | 46 466 028 | −5.51% | 60.61 | 60.60 | −0.01 | yes |
| **mean** | | | | | **−4.28%** | | | **+0.00** | |

**Metric.** PSNR leads at q > 85 (CLAUDE.md) and VMAF is saturated here — the I-frames are
*bit-exact*, so it would read a constant. There is no rate/quality trade to arbitrate and therefore
no chroma question either: the I-frame improves on every plane at once or is not kept, and the
worst P-frame moves by at most 0.01 dB, which is the run-to-run floor.

**Verified outside the harness**, the standard `0036` set: `encode-sequence` → `decode-sequence` on
crowd_run q=99 ki=2, and the decoded PNGs md5-compared against the sources as raw RGB. Frames 0 and
2 (the I-frames) are **bit-exact**; frames 1 and 3 (P) are not, which is BUG-39 cause 3 and not this
item.

**Regression test:** `fallback_iframe_reference_matches_the_decoders` diffs the encoder's reference
against the decoder's for the two-candidate case, and asserts the bit-exact candidate wins at least
once so it cannot go green by never exercising the path.

## What was not chosen

- **Keeping the gate.** It is 4.28% of sequence rate on average and 13% at the best point, for no
  quality cost, and the defect behind it is now understood rather than avoided.
- **A source-copy reference instead of the third encode.** A bit-exact frame's reference *is* its
  colour-converted source, which both forward transforms only read — so this would be free where the
  repair costs an encode. `0040` implemented it, measured 21.37 dB and reverted it, **but that
  measurement is confounded**: BUG-39 cause 2 was live at the time, so the P-frames were decoding a
  wavelet residual as a MED prediction regardless of what the reference held. The refutation does
  not survive its own cause being fixed. Not re-tested here — the repair was already measured and
  correct, and re-opening it is a follow-up with a free win in it, not a blocker.
- **Refusing the bit-exact candidate when it would grow the sequence** (the bbb q=99 rows). The
  candidate is chosen on the I-frame's own size, and the cost it imposes is downstream: a bit-exact
  reference carries detail a lossy one had already quantised away, so the P-residual against it is
  larger. On bbb that downstream cost exceeds the I-frame saving; on the two camera sequences it is
  a third of it. Fixing this properly means choosing on *sequence* bytes rather than frame bytes,
  which needs a second pass. A margin constant would fit three sequences and RATE-2's own comment
  says why that fails: the boundary is content-dependent. Filed as **RATE-4**.
- **Lifting the refusal on the temporal-wavelet path** (`sequence.rs`'s tail, and the three
  `config_tw` sites in `main.rs`). That mode is off by default and was not measured here; one
  unmeasured default is enough.

## Caveat — encode time is not measured

The frames where the bit-exact candidate wins now cost **three** encodes instead of two. The
observed drop on crowd_run q=99 ki=2 was 34.6 → 6.6 fps, but seven other sessions were live on this
machine and COORDINATION rule 1 forbids reading a throughput number under load, so **that figure is
an order of magnitude, not a measurement.** The structural cost is the honest statement: 3× intra
encode on the frames that take the repair, at q = 95..=99 only, decode unchanged.
