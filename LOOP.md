# Autonomous work loop

Standing instructions for an unattended session. Invoke with:

```
/loop improve the codec per LOOP.md
```

Omit an interval and the model paces itself. Everything here is subordinate to
[GOALS.md](GOALS.md) and [CLAUDE.md](CLAUDE.md); this file says *how to keep moving*, not what the
project is for.

## The loop

1. Read [BACKLOG.md](BACKLOG.md). Pick the item with the best value-to-effort ratio that is not
   blocked — not necessarily the highest-numbered priority.
2. **Measure the current state before changing anything.** A change with no before-number is not
   an improvement, it is a hope.
3. Make the change.
4. Measure again, on **≥3 sequences or images** and at **≥2 quality points**. One data point is
   an anecdote.
5. Run `cargo test --release`, `cargo clippy --release`, and
   `cargo clippy --release --target wasm32-unknown-unknown --lib`. All three clean, every time.
6. Write the numbers into [RESEARCH_LOG.md](RESEARCH_LOG.md) — including the failures. Update
   [BACKLOG.md](BACKLOG.md) and [BASELINE.md](BASELINE.md) if the picture changed.
7. Commit with the numbers in the message. Push.
8. Go to 1. **Do not stop to ask whether to continue.**

## Escalate only for

- A change to what the project is *for* (positioning, dropping a whole area).
- Something irreversible that measurement cannot settle.

Everything else is yours to decide. A bitstream change is fine — GOALS rule 10 says there are no
users and no compatibility burden — as long as it is measured and the format marker is bumped.

## What this session learned the hard way

**Suspect the measurement before the codec.** Roughly half of this session's dramatic findings
were bugs in the measuring, not the measured:

- VMAF scores **luma only**. It cannot validate a chroma decision. A `chroma_weight` sweep looked
  like a free 15% and reversed sign once measured with CIEDE2000.
- Pillow **silently truncates 16-bit PNGs to 8 bits on open**. The first 10-bit measurement showed
  no benefit and looked like a codec failure. Use `scripts/png16.py`.
- Comparing bitrates at equal *qstep* rather than equal *distortion* is meaningless — the two
  transforms land at different quality.
- An unnormalised lifting DWT loses to an orthonormal DCT on scaling alone. That one correction
  moved a result from "41% better" to "4% better".
- Reported sizes were inflated 27–58% because `byte_size()` counted raw motion vectors while the
  bitstream delta-codes them.

So: when a new measurement disagrees with an existing trusted number, **check the new harness
first**. Cross-check it against something already believed before drawing a conclusion.

**Offline models understate the real coder.** Simulated with ideal entropy, going from 3 to 4
wavelet levels was worth 1.2%; in the codec it was 6%, because Rice adapts its parameter per
subband and the model could not see that. Prefer in-codec measurement; use offline models to
decide what is worth building, not what it is worth.

**Negative results are the main product.** Twenty-odd ideas have been measured and rejected. Each
one is recorded in BACKLOG under "closed by measurement — do not re-test" so it is not paid for
twice. Write them up as carefully as the wins.

## Where things stand

Read the tail of [RESEARCH_LOG.md](RESEARCH_LOG.md) — it is chronological and the last entries are
the current picture. In brief: intra is roughly 1.4x behind H.264 and has room; inter is about 4x
behind and the gap is architectural (ARCH-2, closed — fine-grained skip is unreachable with a
tile-wide wavelet); 10-bit now works end to end and is worth 2.1–2.4x on colour accuracy at
matched rate.
