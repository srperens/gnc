# Autonomous work loop

Standing instructions for an unattended session. Invoke with:

```
/loop improve the codec per LOOP.md
```

Omit an interval and the model paces itself. Everything here is subordinate to
[GOALS.md](GOALS.md) and [CLAUDE.md](CLAUDE.md); this file says *how to keep moving*, not what the
project is for.

## The loop

0. **Move into your own worktree before anything else.** Up to five sessions run at once; the
   shared checkout is for reading and merging, not for working. See
   [COORDINATION.md](COORDINATION.md) rule 0 for the exact commands — pin it to a commit, give it
   its own `target/`, symlink `test_material/frames`, and remove it when you are done. Then claim
   your area in COORDINATION.md.
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

**Timing needs an idle machine (added 2026-09-06).** Two agents share this Mac. The same fps run
measured 20% slower during another session's `cargo test` than after it — larger than most effects
this project chases. Compression figures (bpp, VMAF, dE00) are deterministic and unaffected;
**every fps, throughput and latency number is not.** Check with `uptime` and `ps` before timing,
and say in the write-up whether the machine was idle.

**Say which fps you mean.** Three quantities have been called "encode fps" and they differ by
2.4x: the GPU encode phase (`benchmark-sequence`, Y4M in), the encoder loop (what
`encode-sequence` prints), and end-to-end wall clock. See [BASELINE.md](BASELINE.md).

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
behind; 10-bit now works end to end and is worth 2.1–2.4x on colour accuracy at matched rate.

Also read **[docs/POSITIONING.md](docs/POSITIONING.md)** — what GNC is *for* (a contribution
codec, not a distribution one), what that market requires, and which of the project's claims are
real. It sets the operating point every target should be measured at, and several historical
numbers in this repo were measured at the wrong one.

**On the inter gap, be precise (reconciled 2026-09-06 — see the RESEARCH_LOG entry of that date).**
"Architectural" is right about the *coupling* and wrong if read as a ceiling:

- Right: 256 independent streams per tile → ~290 B fixed per-tile header → smaller tiles cost
  +70% → the smallest region that can decline to be coded is 256x256 → almost nothing skips
  (0–3% of tiles at q=75). The design choice that makes decode parallel is the one that blocks
  fine-grained skip. ARCH-2 closed all three routes to finer granularity, correctly.
- Wrong: Dirac shipped **this exact architecture** — closed-loop hybrid, OBMC, wavelet on the
  motion-compensated residual, RDO — and landed at roughly H.264-class. The pipeline shape is not
  the cap.
- Also settled: adding rate-distortion decisions is **not** the answer either. Coefficient RDOQ
  measured +0.1%, per-tile allocation 0.00 dB, tile skip dominated by simply raising q. A
  POSITIONING.md draft prescribed RD decisions from published magnitudes; this repo's own numbers
  refute it, and the document has been corrected.

- **MCTF is now measured locally, not just cited (2026-09-06).** `src/temporal.rs` has no motion
  compensation, so warping first and then filtering temporally was genuinely untested. Two offline
  gates: the open loop is worth **0.98-1.01x** on camera content (nothing — real motion dominates
  reference noise 4-5x) and 1.34-1.37x on animation; the multi-frame temporal transform is
  **1.04-1.14x worse** than a P-chain on *every* sequence. Rejected. Do not rebuild it.

So the honest state is **open and unexplained after exhausting the local levers** — do not fill
it with a guess. The one untested lever with a mechanism specific to a wavelet codec is OBMC.
Expect single digits from it.
