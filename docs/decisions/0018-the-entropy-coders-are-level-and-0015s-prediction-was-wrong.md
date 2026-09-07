# 0018 — The entropy coders are level above q=25, and 0015's prediction was wrong

**Date:** 2026-09-07
**Status:** Accepted. Supersedes the prediction in
[0015](0015-the-entropy-coder-comparison-is-withdrawn.md); no default changes.

## The decision

The README's Entropy Coders table gets a **rate column back**, measured on one commit, and the
speed column is **removed** rather than corrected. Both defaults stand: Rice above q=20, rANS at
or below it.

0015 withdrew the old compression column (Rice 4.01 bpp against rANS 4.22 bpp @ q=75) because the
two figures sat on opposite sides of three changes, and it recorded a prediction so that ENT-2
could falsify it:

> Rice still wins, and by more than 4.01 vs 4.22 suggested, because header overhead scales with
> stream count and Rice runs 256 streams per tile to rANS's 32 — which is exactly the overhead
> GP17 attacked. If ENT-2 comes back the other way, this record was wrong and the default deserves
> re-examination.

**It came back the other way.** Measured on four stills at one commit, mean across images, negative
meaning rANS is smaller: **−6.4% at q=10, −7.1% at q=15, −6.7% at q=20, then +0.4% at q=25, +0.1%
at q=40 and q=55, −0.6% at q=70.** Above q=25 the coders are level; below q=20 rANS is ahead by
6–7%. Rice does not win, and it does not win by more.

## Why the default still does not move

0015 said the default rests on GPU parallelism, not on a bpp figure, and that survives the
correction intact: Rice's 256 streams carry no sequential state chain and under 1 KB of shared
memory, against rANS's 32 streams and 16 KB of frequency tables. What changes is that the rate
figure no longer *argues against* the default — it is level where Rice is selected.

Three further reasons the re-examination 0015 called for ends here rather than in a change:

1. **The mean hides a content bet.** At q=25 the same setting runs from −3.5% (touchdown) to +8.2%
   (kristensara). Choosing the coder by the mean picks a loser on some content at every quality
   point measured.
2. **rANS cannot reach the operating point at all.** It overflows a fixed 4 KB per-stream buffer
   above q≈76 — q=75 encodes on all four images, q=77 on none — and GNC is a contribution codec
   (GOALS §1), so the range above that is the range that matters.
3. **The throughput half is unmeasured today and points the same way.** TUNE-3's ~8% encode /
   ~15% decode penalty for rANS is the only figure in the repository, and it was not re-measured
   here because up to five sessions share this machine (COORDINATION rule 1).

## What was *not* chosen, and what it would have cost

- **Making rANS the default above q=20.** Rejected: it measures level, not better, and it is a
  content bet against a coder that has no sequential state chain. The cost of being wrong is paid
  on every stream at once.
- **Threading a runtime buffer size through the rANS encode shaders** so it could reach q>76 —
  three encode shaders and six allocation sites, the way Rice already sizes its buffer from qstep.
  Rejected on the same ground BUG-9 rejected it, now with the rate number attached: it buys access
  to a range where the coder measures level at best. The cheap half — a guard that names the
  overflow instead of wrapping a pointer — is already in and is the right amount of fixing.
- **Moving the q=20 cutoff.** Rejected, and ENT-2 supplies a reason better than the original one:
  the cutoff sits exactly where `quality_preset` goes from 4 decomposition levels to 5, and every
  image's Rice-vs-rANS delta jumps in the same direction across that boundary. rANS pays a
  frequency table per subband group while Rice adapts its k per subband nearly for free, so rANS's
  advantage is an advantage *at 4 levels*. **The two constants are coupled and must move
  together** — which also means a future change to the level rule silently changes the right
  coder cutoff.
- **Keeping a speed column with a corrected number.** Rejected: any number here would have to come
  from a loaded machine. An absent column is honest; "1.5–2× faster" was not, and it contradicted
  TUNE-3's own 15% by a factor of three to six.

## What this cost to find out, and the one instrument bug

The harness reads which coder produced each file **out of the bitstream** (`entropy_type` in the
GP17 frame header) rather than trusting the flag that requested it, because BUG-9's entry records
`--rans` as "a no-op flag kept for backward compatibility". Had that been true, every row would
have been Rice against Rice. It is not true — 40 of 40 points carry the requested coder — and only
the flag's own help text was stale. **That is the second time this quarter a documented claim about
a flag turned out to be about the flag's documentation.**

The instrument bug: the first run reported all eight q≥80 failures as
`note: run with RUST_BACKTRACE=1`, because it took the last non-empty stderr line and a Rust panic
ends with the backtrace note. The reason is the line *after* the `panicked at` header. The run had
already found the real threshold and was discarding the evidence — LOOP.md's "suspect the
measurement before the codec" applies to the error path too.
