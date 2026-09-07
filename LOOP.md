# Autonomous work loop

Standing instructions for an unattended session. Invoke with:

```
/loop improve the codec per LOOP.md
```

Omit an interval and the model paces itself. Everything here is subordinate to
[GOALS.md](GOALS.md) and [CLAUDE.md](CLAUDE.md); this file says *how to keep moving*, not what the
project is for.

## The loop

0. **Start the session: four commands, in order.** See [COORDINATION.md](COORDINATION.md),
   "Start of session" — your own worktree, the test-material symlink, `scripts/claim worktree`,
   `scripts/claim next`. That section is the single copy of the procedure; do not reconstruct it
   from memory. Eight sessions run at once, and the shared checkout is for reading and merging,
   never for working. `scripts/claim` enforces both: it refuses to hand you work from the shared
   checkout, and refuses to hand you work in a worktree you do not hold.

0b. **Sync with `main` before you pick, never while you measure.** All the worktrees share one
   `.git`, so another session's commit to `main` is visible to you *instantly* — `git log main`
   is always current, no fetch needed — but your own files do not move until you
   `git rebase main`. Visibility is automatic; adoption is not.

   So at the top of each iteration: `git log --oneline main` to see what landed, read the
   "Landed today, and what each one invalidates" section of
   [COORDINATION.md](COORDINATION.md), then `git rebase main` and **re-run the gates** — a rebase
   breaks things silently. After that, do not rebase again until the item is finished and
   committed: a before-number and an after-number taken across a rebase are numbers from two
   different codebases, and rule 1 says neither is valid. If something that landed invalidates the
   item you were about to do, that is the cheapest possible time to find out.

1. **Take your item with `scripts/claim next "why, briefly"`.** It walks the startable BACKLOG
   items in priority order and takes the first one nobody holds — the pick *is* the
   compare-and-swap, so eight sessions running it at the same instant get eight different items.

   **Do not read BACKLOG, decide, and then claim.** That is three steps, and every session
   applying the same rule to the same file gets the same answer inside the same two minutes: on
   2026-09-07 it sent several sessions at MEAS-9 at once. Being deterministic is precisely what
   makes the collision reliable rather than unlikely, and an atomic claim alone does not fix it —
   it just means seven sessions lose a round and go back to the same list.

   Then read the item's BACKLOG entry and **question whether it is still the right item**;
   something that landed this morning may have invalidated it. If it has, `scripts/claim drop` it,
   record why in its entry, and run `next` again. Use `scripts/claim take <ITEM>` only when you
   want one specific item for a reason you can state — a bug you just found, a follow-up the last
   item obliges you to do.

   `scripts/claim touch <ITEM>` as you go: a claim with no heartbeat for an hour shows as `STALE`,
   and a held item is invisible to `next`, so an abandoned claim takes work away from seven other
   sessions.

   **Record the claim nowhere else.** BACKLOG's `(in progress)` markers and COORDINATION's
   worktree table are documentation, not locks; a second copy of "who holds what" only goes stale
   and sends the next session at an item that is already taken. Write the BACKLOG status when the
   item is *finished*, and add a worktree row for the prose a claim note cannot carry.

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
8. `scripts/claim drop <ITEM>`, then go to 0b. **Do not stop to ask whether to continue.**

## Escalate only for

- A change to what the project is *for* (positioning, dropping a whole area).
- Something irreversible that measurement cannot settle.

Everything else is yours to decide. A bitstream change is fine — GOALS rule 10 says there are no
users and no compatibility burden — as long as it is measured and the format marker is bumped.

## What this session learned the hard way

**Suspect the measurement before the codec.** Roughly half of this session's dramatic findings
were bugs in the measuring, not the measured:

- VMAF scores **luma only**, and it cannot validate a chroma decision. Twice proven: CHROMA-1
  shipped a 6% rate cut on which VMAF read 97.08 before and 97.08 after — no change at all, while
  the real cost sat entirely in dE00. And earlier, a `chroma_weight` sweep looked
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
