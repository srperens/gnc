# Coordination between concurrent Claude sessions

**Eight Claude sessions work on this repository at the same time.** They share one `.git`, one
GPU and one test-material directory, and nothing else. Every rule below exists because it has
already been broken and cost a retracted result or lost work.

**There is exactly one lock, and it is `scripts/claim`.** Every other record of who is doing
what — the worktree table further down, BACKLOG's `(in progress)` markers, a sentence in
RESEARCH_LOG — is documentation written *after* the claim succeeded. None of them excludes
anyone. If you are deciding what to work on by reading prose, you are about to collide with
another session: reading, deciding and writing your row are three separate steps, and eight
sessions that start together all read "free" before any of them writes.

## Start of session — four commands, in this order, no exceptions

```bash
# 1. Your own worktree. The shared checkout is for reading, coordination and merging, never
#    for working: two sessions in one directory is how one committed the other's uncommitted
#    work on 2026-09-07, and how the abac.rs and gpu_util.rs edits were lost on 2026-09-06.
#    Derive the paths, never type them (CLAUDE.md, "Never commit secrets or local
#    infrastructure detail"). Name the AREA for the *work*, not for the session.
REPO=$(git rev-parse --show-toplevel)
AREA=<area>
git -C "$REPO" worktree add -b "$AREA" "$REPO-$AREA" main
cd "$REPO-$AREA"

# 2. Link the test material in. NEVER run this in the shared checkout, and never with -f:
#    `frames` there is itself a symlink to the real ~31 GB directory, and `ln -sfn` pointed it
#    at itself on 2026-09-06, breaking every session's measurements at once.
[ "$PWD" != "$REPO" ] || { echo "refusing: you are in the shared checkout"; return 1 2>/dev/null || exit 1; }
ln -s "$(readlink "$REPO/test_material/frames" || echo "$REPO/test_material/frames")" \
      test_material/frames

# 3. Claim the worktree. Refuses if a live session already holds it; takes over automatically
#    if that session is gone, which is the normal case, not an exception.
scripts/claim worktree

# 4. Claim your work. This picks the top free BACKLOG item AND claims it in one atomic step.
scripts/claim next "why this item, briefly"
```

That is the whole start-of-session procedure. It is also enforced: `scripts/claim` refuses to
hand you work from the shared checkout, and refuses to hand you work in a worktree you do not
hold. You cannot get to step 4 by skipping steps 1 and 3.

## Why `next`, and not "read the backlog, pick, then claim"

The MEAS-9 collision on 2026-09-07 was not bad luck. Every session applies the same rule — *the
best value-to-effort open item* — to the same file, so every session gets the same answer, inside
the same two minutes, before any of them can write a row. **Being deterministic is exactly what
makes the collision reliable rather than unlikely.** An atomic claim on its own does not fix
that: it just means seven sessions lose a round and go back to the same list.

`scripts/claim next` closes the window by making the pick *be* the compare-and-swap. It walks
BACKLOG's startable items in priority order and takes the first one nobody holds, so N sessions
running it at the same instant get N *different* items, by construction. `scripts/claim selftest`
asserts both halves: 16 processes racing for one item produce exactly one winner, and 6 processes
racing on one queue produce 6 distinct claims.

Use `scripts/claim take <ITEM> "<why>"` only when you have a specific reason to want *that* item
— a bug you just found, a follow-up the previous item obliges you to do. For "what should I work
on", use `next`.

## The claim commands

```bash
scripts/claim worktree                    # step 3 above; the precondition for everything else
scripts/claim next "why, briefly"         # pick AND claim the top free item, atomically
scripts/claim items                       # the startable queue, and which entries are free
scripts/claim list                        # what is held, by whom, for how long
scripts/claim take <ITEM> "why, briefly"  # claim one named item; non-zero means you lost the race
scripts/claim touch <ITEM>                # heartbeat, so the age in `list` stays honest
scripts/claim drop <ITEM>                 # when you are done — do this, or the item stays parked
scripts/claim steal <ITEM> "why"          # take over from a session that is gone
scripts/claim selftest                    # proves the exclusion rather than asserting it
```

**What makes an item startable.** `next` and `items` read the committed `main:BACKLOG.md` — not
your working copy, and not anyone else's — and treat a heading as a queue entry when it looks
like `### NAME-<n> — title (todo, P<n>)`. That is the convention BACKLOG already uses. Finished
items carry `**DONE**` / `**CLOSED**` / `**FIXED**` / `**REJECTED**` and no priority, so they
drop out by construction; items in flight drop out because they are claimed. Two consequences
worth knowing:

- **To put an idea into rotation**, give it an ID of the form `NAME-<n>` and a `(todo, P<n>)`
  marker in its heading. An idea with no ID and no priority is invisible to `next` — that is why
  `ARCH-1`, `EBCOT` part 2 and the numbered legacy entries are not offered.
- **To take an item out of rotation without holding it as a session**, park it:
  `scripts/claim take --as blocked-<reason> <ITEM> "why"`. `CANARY-1` and `MEAS-5` are parked
  this way — both need a second GPU — so `next` skips them instead of handing out work nobody
  can do.

**Identity is `<worktree>@<branch>#s<pid>`**, where the pid is the session's own `claude`
process. It has to be the session, not the directory: identity was `<worktree>@<branch>` for the
first hour and two sessions in one worktree were therefore *the same principal*, so one's `touch`
succeeded against the other's claim — the lock excluded every session except the one it needed
to. The pid also makes an abandoned claim detectable rather than merely old: `list` marks it
`SESSION GONE, safe to steal`, and separately marks a live holder that has not touched in an hour
`STALE, no heartbeat`. Those need different actions — steal the first, ask about the second.

**Why this is atomic and needs no new infrastructure:** every worktree shares one `.git`, so
`refs/claims/*` is visible to all of them the instant it is written — no fetch, no daemon, no
lock file in the working tree — and `git update-ref <ref> <new> <old>` is a compare-and-swap
under git's own ref lock, where an empty `<old>` means "must not exist". Claims are metadata
blobs, so `git cat-file -p refs/claims/MEAS-9` shows the owner, the time and the commit it was
taken against. Nothing is pushed; the contention is local to this machine. See
[docs/decisions/0016-claiming-an-item-is-a-ref-cas-not-a-table-row.md](docs/decisions/0016-claiming-an-item-is-a-ref-cas-not-a-table-row.md).

**Do not leave a claim held for hours without a `touch`, and drop it when you are done.** A
parked item is invisible to `next`, so an abandoned claim removes work from seven other sessions.

**Decision-record numbers are a shared resource too, and they collided the same way.** `main`
currently carries **two** files numbered `0018` — `0018-gnc-is-broad-on-purpose.md` and
`0018-the-entropy-coders-are-level-and-0015s-prediction-was-wrong.md` — plus an 0020 that had to
be renumbered from 0018 by hand. Same mechanism as the MEAS-9 collision: two sessions read
`ls docs/decisions/`, both computed "next is 0018", and neither could see the other. The lock
already handles this — **reserve the number before you write the file**:

```bash
scripts/claim take dr-0019 "the pick is the lock"   # non-zero: someone has it, use the next one
```

Reserving costs one command and is the difference between picking a number and being *given*
one. Drop it once the record is merged.

## Builds queue on one lock, and that looks like a hang

Each worktree has its own `target/`, so builds no longer block on each other's **target** lock —
that alone is worth the disk. **They do still block on the package-cache lock**, and that
surprised the `abacship` session on 2026-09-07: `cargo test --release` sat at `Blocking waiting for file lock on package cache` for
minutes while other sessions compiled, with 35 `rustc` processes on the machine. The lock is in
`~/.cargo`, which every worktree shares; a separate `target/` does not help. So **a build queued
behind seven other sessions is normal, not a hang** — start long test runs in the background rather
than waiting on them, and before assuming something is stuck, ask *who holds the lock*:

```bash
lsof ~/.cargo/.package-cache                     # the shared one: another session is building
lsof "$PWD/target/release/.cargo-lock"           # your own: see the trap below
```

`lsof -p <pid> -a -d cwd` then says which worktree the holder is in, which is usually enough to
tell "another session is working" from "something of mine is wedged".

**The trap, hit on 2026-09-07: killing a backgrounded `cargo` by its shell leaves `cargo` alive
holding your build lock.** A background Bash call is `zsh -c '… cargo test …'`, so `pkill -f` on
the command text matches the *wrapper*, not the cargo underneath it. The wrapper dies, cargo keeps
running with its parent reparented to launchd, and every later build in that worktree blocks on
`file lock on build directory` — with no other session at fault and nothing in `ps` that looks
wrong unless you go looking for the lock holder. Kill the pid `lsof` names, not the pattern.

## End of session

```bash
cargo test --release && cargo clippy --release   # the usual gates, in your worktree
git fetch origin && git rebase origin/main       # rebase before merging, so conflicts land here
cargo test --release                             # a rebase can break things silently
git push -u origin "$AREA"                       # or merge to main if you own it
scripts/claim drop <ITEM>                        # release the item
git -C "$REPO" worktree remove "$REPO-$AREA"     # only when the area is finished
```

Removing the worktree does not release its claim; the next session to stand in a directory of
that name reclaims it automatically once your session is gone.

**The shared checkout stays on `main` and is for merging and reading, not for editing.**

## Worktrees currently out

**This table is not the lock and never was — `scripts/claim list` is the live answer to "who
holds what".** It carries the prose a claim note cannot: what the item is really about, what it
touches, what was found on the way in. Write the row *after* the claim succeeds, and write the
worktree **relative to the shared checkout** — no absolute paths, no session ids. Remove your
row when you are done.

If this table and `scripts/claim list` disagree, the table is wrong.

| worktree | branch | area |
|---|---|---|
| `../gnc-abac`, `.claude/worktrees/abac` (`abac-gate`) | `abac` | **released — question answered, see BACKLOG Part 6.** The idle-machine bench is run. Range at cb=64 costs **1.69× frame decode for −16.7% rate** at q=90; Interval costs 3.99×. Rice's own entropy stage is 47% of frame decode, which caps any entropy work at 1.9×. What remains is a positioning call, not an engineering one. |
| `../gnc-abac` | `abac` | same worktree, now on **BUG-8** — the encoder's local decode diverges from the real decoder down a GOP. |
| `../gnc-nearlossless` | `nearlossless` | **INTRA at contribution quality (priority 1).** Gating whether MED prediction *instead of* the wavelet — LOSSLESS-1's mechanism, −14.9% at q=100 — survives into the lossy near-lossless range q=88–99, closed-loop with a quantised residual (JPEG-LS near-lossless). Offline gate first, no code change yet. |
| `../gnc-abacship` | `abacship` | **ABAC-SHIP merged to main 2026-09-07** (`60bed17`, `378a0c7`). abac is a real entropy coder now: `--abac`, entropy type 5, **GP18**. −16.6% to −18.8% of rate against Rice at *identical pixels*, −13.4% at lossless (FFV1 gap +23.9% → +7.3%). Rice stays the default — `docs/decisions/0017`. **Every frame this encoder writes now says GP18**; a GP18 Rice frame is a GP17 payload with a new label, proved by relabelling and decoding, and GP17 still reads. Worktree kept: next on this row is a real inter measurement (correctness is tested, rate is not). |
| `../gnc-rate1` | `rate1` | **DONE, merged, worktree removed.** RATE-1 answered **no** — a bit-depth-aware rate rule recovers **0.0% on all four photographic stills** (89.4% on the synthetic gradient, which is the trap). The sweep found **RATE-2 instead, filed P1**: above q≈95-98 the lossy ladder costs more bytes than bit-exact lossless on every real image, mean **+28.9% at q=99** (blue_sky +40.6%). LOSSLESS-1 made lossless cheap enough to undercut the top of the lossy ladder and nothing noticed. |
| `../gnc-coord` | `coord` | **COORD-1 — the claim mechanism enforces the rules instead of restating them.** Docs and `scripts/claim` only; no codec change, invalidates no measurement. `scripts/claim next` makes the *pick* atomic, the shared-checkout and worktree preconditions are now refusals rather than prose, and the session identity bug that made `SESSION GONE` undetectable is fixed. See `docs/decisions/0019`. Claimed 2026-09-07. |
| `../gnc-meas9` | `meas9` | **MEAS-9 merged (`ed62de7`) — now on ENT-3.** MEAS-9's result: J2K in irreversible 9/7 mode uses **GNC's own transform at GNC's own depth** and needs **54.2% fewer bits on RGB PSNR / 79.7% on Y-PSNR**, so the intra gap is the entropy coder, not the transform. GNC is −10.2%/+29.4% against JPEG XS 4:4:4 and +20.2%/+29.3% against ProRes 4444. **ENT-3 is the inter half**: abac against Rice on P-frame residuals, which ABAC-SHIP explicitly left out of scope. Touches nothing another row owns — measurement first. |

## The test material was missing entirely, and was refetched (2026-09-07) — RESOLVED

**`test_material/frames` did not exist anywhere on disk** at the start of 2026-09-07 — not in the
shared checkout, not under any worktree. The 2026-09-06 `ln -sfn` accident is the likely cause: the
link was restored that day, but the ~31 GB directory it pointed at is gone. Nothing on disk
survived it.

**DONE — the material is back (2026-09-07 ~19:30), and it reproduces BASELINE.** All 22
artefacts are present and verified, and `q=75` on `bbb_1080p` reads **44.84 dB / 4.53 bpp**,
exactly the committed figure. Symlink your worktree at it per "Start of session" and carry on.

**Two things the refetch turned up, both now fixed in `fetch_test_frames.sh`:**

- **The script exited 0 with the blue_sky sequence missing.** ffmpeg 9 removed `-vsync` (renamed
  to `-fps_mode` back in 5.1), the extraction died on the unrecognised option, and `|| true`
  swallowed it. The script now probes which option the installed ffmpeg takes — it runs on more
  than one machine, so pinning either one just moves the breakage — and ends with an explicit
  per-file check of all 22 artefacts that exits non-zero if any is missing.
- **Do not edit a bash script while it is running.** Bash reads the file incrementally, so
  rewriting it mid-run made the next read land at a shifted offset and the run died with
  `syntax error near unexpected token '50-57'` — after the download, before the verification.
  Cost a whole extra pass. Edit, then run.

**Who fetched: the `chroma2` session** (worktree since removed), into the **shared checkout**
(`test_material/frames`), which is where the directory belongs and what every worktree's symlink
resolves to.

**If you ever need to refetch: check `ps` for a running fetch first, and do not start a second
one.** Two concurrent runs write the same paths, and the script's `[skip] already exists` guard is
checked before the write, so simultaneous starts all miss it and a half-finished download looks
complete to whoever checks next. That happened here — three runs at once — and nothing was
corrupted only by luck. There is no session left to ask, so verify with the script's own
per-file check rather than by asking anyone.

**Checked 2026-09-07 19:10 by the `abacship` session: there are TWO fetch runs live, not one.**
`ps` shows `bash ./fetch_test_frames.sh` (started 19:08:58, launched from the shared checkout's
`test_material/` — this is the chroma2 session) *and* `bash test_material/fetch_test_frames.sh`
(started 19:08:21, now orphaned, parent is `launchd`). A third exited at ~19:09. Because every
worktree's `test_material/frames` is a symlink to the same shared directory, all of them write the
same paths: two `ffmpeg` processes were writing `frames/blue_sky_1080p.png` **simultaneously**.
No damage this time — `bbb_1080p`, `blue_sky_1080p` and `kristensara_720p` are all complete, valid
PNGs at the right dimensions (checked header + `IEND`) — but they will collide again on
`touchdown_1080p` and on the two 8-frame sequences. **Before using any frame written after 19:08,
verify it**: `python3 -c "import sys;d=open(sys.argv[1],'rb').read();print(d[:8]==b'\x89PNG\r\n\x1a\n' and d[-8:-4]==b'IEND')" <file>`.
Whoever owns the orphan should kill it; whoever owns the survivor should re-run the script once at
the end, since its `[skip] already exists` guard will then leave the good files alone and refill
only what is missing.

**What the script does and does not restore.** It fetches four single frames (`bbb_1080p`,
`blue_sky_1080p`, `kristensara_720p`, `touchdown_1080p`) and two 8-frame sequences (`bbb`,
`blue_sky`, each with a generated `.y4m`). It does **not** fetch `old_town`, `aerial`, `crowd_run`,
`rush_hour`, `stockholm` or anything else quoted in RESEARCH_LOG — those were multi-GB and were
never in the script. It does not restore `frames/hdr/` either; regenerate that with
`scripts/png16.py` if a 10-bit measurement is needed.

**So any measurement in this repo naming a sequence outside that list cannot currently be
reproduced.** That is not a retraction of those numbers, but treat "re-measure X on old_town" as
blocked until someone re-fetches it, and say which sequences a new result actually used.

## Test material was missing from the machine entirely (2026-09-07)

`test_material/frames` did not exist in the shared checkout at the start of the day, so **every
worktree's symlink was dangling and no session could measure anything.** Refetched with
`test_material/fetch_test_frames.sh`; the four stills (`bbb_1080p`, `blue_sky_1080p`,
`kristensara_720p`, `touchdown_1080p`) and `sequences/bbb/` are back and every PNG verified to
decode. **Updated later that day: `sequences/blue_sky/` is back too** — the `chroma2` session fixed
the script's silent-failure path and refetched, and all 22 artefacts now verify. `frames/hdr/` is
still absent; regenerate it with `scripts/png16.py` if a 10-bit measurement is needed.

Two things to carry from how that went:

- **Three sessions ran the fetch script at the same time**, all writing the same output paths with
  `ffmpeg -y` and `curl -o`. The script's `[ -f "$out" ]` guard is not a lock: it is checked before
  the write, so simultaneous starts all miss it, and a half-written PNG looks complete to whoever
  checks next. Nothing was corrupted this time (verified by decoding all 12 files), but that was
  luck. **Check `ps` for a running fetch before starting one.**
- **Copy the images you measure on into your own worktree and record their hashes.** Another
  session's fetch overwrote `touchdown_1080p.png` while a measurement was being set up against it.
  Rule 1 says a number is only valid against a commit; it is equally only valid against a known
  input.

**`python3` on this machine has neither `numpy` nor `pillow`**, so `scripts/lossless_gate.py`,
`ypsnr_de00.py` and `chroma_metric.py` do not run out of the box, and the system python is
externally managed so `pip install` refuses outright.

**There is now one shared venv in the shared checkout — use it, do not make your own.**

```bash
REPO=$(git rev-parse --show-toplevel)          # resolves to your worktree
SHARED=$(git -C "$REPO" worktree list --porcelain | head -1 | cut -d' ' -f2)
"$SHARED/.venv/bin/python" scripts/chroma_metric.py ...
```

It has numpy 2.5.3 and pillow 12.3.0, and it is gitignored. `chroma_metric.py --selftest` passes
**16/16 Sharma reference pairs**, so the dE00 implementation itself is validated — run it once
before quoting a colour number, it costs a second.

## JPEG XS *is* measurable here — SVT-JPEG-XS now builds on arm64 (2026-09-07, chroma2)

**MEAS-9 is recorded in two places as blocked on this, with VC-2 standing in. It is not blocked.**
`scripts/build_jpegxs_arm64.sh` clones, patches and builds SVT-JPEG-XS on this M1, and verifies
the result: a 1920x1080 yuv422p frame at `--bpp 3` round-trips to **PSNR y 44.484343 dB**,
reproduced exactly from a clean clone.

Only the *build system* assumes x86. The C sources already carry `#else /* ARCH_X86_64 */`
fallbacks for every dispatch, so the scalar path was written and simply never selected;
`scripts/svt-jpegxs-arm64.patch` gates the nasm discovery, `-DARCH_X86_64` and the nine ASM object
libraries on a detected arch, gives non-x86 a `get_cpu_flags()` returning 0 (EncHandle/DecHandle
call it unconditionally), and guards the SIMD headers that pull in `<immintrin.h>`.

**Rate and quality from this build are exact — that is what MEAS-9 needs.** Throughput is not:
every SIMD kernel is off, so the 9.17 fps encode it reports says nothing about JPEG XS and must
never be quoted or compared against GNC's fps. The latency row in the README stands on MEAS-6, not
on this.

## Three more sequences are back: bbb_extended, old_town_cross, crowd_run (2026-09-07, chroma2)

QUAL-1, MEAS-1 and BASELINE's BD-rate table are all measured on **bbb_extended, old_town_cross and
crowd_run**, and none of the three was in the tree — the fetch script has never fetched them.
`test_material/frames/sequences/` now has **24 frames of each** (1920x1080, PNG). Streaming the
first 24 frames of a 1.55 GB Xiph y4m costs about 80 MB, so this is cheap to redo; taking them
from the *start* of the file is what makes it cheap.

**They are 24 frames, not 200.** QUAL-1 used 200 frames of old_town_cross and an unstated count of
crowd_run, so a run against these reproduces the *content* but not the length, and BASELINE says
17 frames where the QUAL-1 log says 24. Say which you used.

## `cargo test --release` is flaky on `main` right now — abac_bitstream races itself (2026-09-07)

**`abac_handles_subsampled_chroma` fails under the documented gate and passes with
`--test-threads=1`.** 7/7 green serially, 6/7 in parallel, on `main` at 2c9e4d4, on a machine at
load 57. It is not content-dependent and not a rebase artefact — the CHROMA-2 branch that surfaced
it touches no `.rs`, `.wgsl` or `.toml` at all.

This file already says benches need `--test-threads=1` because cargo runs `#[test]` bodies
concurrently and two GPU tests contend for the device. That rule now applies to a **correctness**
test in `tests/abac_bitstream.rs`, not just a bench, which is worse: the gate CLAUDE.md and LOOP
both require (`cargo test --release`) can now go red on code that is fine, and — the real risk —
green on code that is not, depending on scheduling.

**So do not read a single red `abac_bitstream` run as your regression.** Re-run with
`--test-threads=1` before you believe it. And if it is yours, this wants a real fix (serialise the
device, or a lock around the GPU context) rather than everyone learning the workaround: a gate that
is sometimes wrong stops being a gate. Flagged by the chroma2 session; the abac/abacship sessions
own that file.

## Timing: an idle machine is necessary and NOT sufficient (added 2026-09-06, after an idle run still lied)

**The GPU ramps its clocks, and a repeat count chosen for a CPU benchmark will not outlast the
ramp.** On a genuinely idle machine (load 3.8, GPU free), three consecutive *processes* on
identical input read **66.5, 45.3 and 34.9 ms** — a 1.9× spread, monotonically decreasing. A
freshly-idle M1 starts in a low power state and needs on the order of a second of sustained work
to boost. `abac_bench` had a `spread` column that blamed a busy machine for exactly this, so the
instrument was reporting the ramp *and* misattributing it.

Two rules that follow, and they cost nothing:

- **Take the best of ~24 repeats, not the median of ~7.** For a deterministic kernel on fixed
  input every error source — clock ramp, another session, scheduler noise — only makes a reading
  *slower*. The minimum is therefore the least-contaminated estimate, and `median / best` is a
  free settled-or-not diagnostic: near 1.0 means quotable, well above means keep only the ratios.
- **Run benches with `--test-threads=1`.** Cargo runs `#[test]` functions concurrently by default,
  so two GPU benches in one file contend for the device and interleave their output. This was
  happening and nothing warned about it.

**Isolating one GPU stage without timestamp queries:** dispatch it k times behind an env var and
difference. Rice's entropy decode is idempotent, so `GNC_RICE_DISPATCH_REPEAT=k` gives frame times
of 29.50 / 85.16 / 141.55 ms at k = 1 / 5 / 9 — slope 14.0 ms, two independent estimates agreeing
to 0.6%. That is how the abac question stopped being a bracket and became a number, and the same
trick works for any idempotent stage.

## Timing: the machine is shared, so throughput numbers are not measurable during a session

Eight sessions compile and run GPU work on this one M1 at the same time. That makes every
wall-clock figure unreliable while anyone else is working — on 2026-09-06 the same abac decode
input timed **25.2, 31.1 and 37.5 ms across three runs**, a 48% spread on identical work, and
three targeted shader optimisations against three different suspected bottlenecks all returned
exactly nothing. Three plausible fixes measuring null is far likelier to be a broken instrument
than three wrong hypotheses.

So:

- **Compression figures (bpp, PSNR, VMAF, dE00) are safe.** They are deterministic and unaffected
  by load. Measure and quote them freely.
- **Throughput figures are not.** Do not tune against them, and do not record one without saying
  the machine was loaded. An optimisation validated against noise looks justified and is not.
- **Build the alternatives behind switches and measure them later, together.** When something has
  several plausible implementations, implement them all, make them selectable, and add a bench
  that runs the whole set in one command. Then one idle-machine run settles it. `abac` does this:
  `GNC_ABAC_CODER` selects the entropy coder variant and
  `cargo test --release --test abac_bench -- --ignored --nocapture` times every combination.

## The four rules that have actually bitten us

**1. A number is only valid against a commit.** The tree can change under you with no signal. A
BD-rate figure moved from −3.7% to −2.3% purely because the coefficient path changed between the
sweep and the commit, and it had to be corrected in three documents after publication. Working in
your own worktree fixes most of this; still, measure a committed state, or a worktree pinned to a
hash (`git worktree add <dir> <sha>`).

**2. Measure the range the project cares about, not the range that is convenient.** GNC is a
*contribution* codec (GOALS §1). TUNE-5 was measured at q=15-50, shipped, and then found to cost
10.3 dB at q=99 — the operating point that actually matters. A sweep that stops at q=50 has not
tested this codec.

**3. Above about q=80, VMAF is saturated and PSNR must lead.** On old_town at q≥85 VMAF reads
identical for two settings whose worst-frame PSNR differs by 4.8 dB. State which metric led, and
why. CLAUDE.md now carries this as a table rather than a blanket "VMAF primary" rule, and QUAL-1
put a magnitude on it: widening a BD-rate ladder moved the VMAF figure by **47.5 points on
average, 110 at worst**, while the PSNR figure moved **1.0**.

**4. A point measurement at fixed q cannot judge a rate/quality trade**, and it always flatters
the option that spends more bits. Use BD-rate, or compare at matched rate. At least four separate
wrong conclusions have come from this one error.

## Landed today, and what each one invalidates

- **Correction to a diagnosis I committed: the `abac_bitstream` flake was NOT GPU contention.**
  I read "a different test fails each run, all pass with `--test-threads=1`" and concluded two GPU
  tests were contending for the device; that is in commit `4133f54`'s message and it is wrong. The
  real mechanism, found by the ABAC-SHIP owner: a test called `std::env::set_var("GNC_ABAC_CODER")`,
  the environment is process-global, and cargo runs `#[test]` functions as threads — so it changed
  which arithmetic engine a *concurrently running* test encoded with. The concurrency half was
  right, the shared resource was not. It also masked a real decoder bug: `CachedBuffers` held one
  `abac_coder` for all three planes, so planes 0 and 1 decoded with whatever plane 2 used. Fixed at
  the source by moving the coder and code-block size out of the environment into `CodecConfig`, so
  `cargo test --release` stays the gate unqualified and no serialising mutex was needed. Worth
  carrying: a symptom can fit a mechanism exactly and still have a different cause, and "it is
  contention" is a comfortable answer that stops the search early.

- **RATE-2 filed — above q≈95-98 GNC is dominated by its own lossless path.** No code change, so
  no output moves, but it **invalidates how any BD-rate whose ladder reaches q=95-99 should be
  read**: those rungs spend more bytes than bit-exact lossless for a worse picture (mean +28.9% at
  q=99; blue_sky 3026470 bytes at 60.14 dB against 2153118 bit-exact, +40.6%). Crossover per image
  q=98/95/96/96. Not universal — smoothramp, flat and noise are not dominated, because MED
  prediction is poor there. Also from the same sweep: **RATE-1 is answered no** (0.0% recoverable
  on photographic content), and **the ladder is not monotonic in rate** (flat512 costs 0.0450 bpp
  at q=86 and 0.0370 at q=90), so anything interpolating GNC by rate should flag a rung whose rate
  falls while q rises.

- **MEAS-3 — the inter path's rate saving does not survive matched quality. Invalidates a GOALS
figure; changes no default.** BD-rate of the shipped ki=9 configuration against all-intra, three
sequences, 18 frames, q=25-95, 4:4:4: **+15.9% (crowd_run), +22.2% (old_town_cross), −24.2%
(bbb_extended), mean +4.6% on mean PSNR — and mean +19.1% on worst-frame PSNR.** Positive means
inter needs *more* bits. Above q≈85 the saving is gone; at q=95 inter costs more than all-intra on
two of three sequences. Decision record 0019; the follow-up is BACKLOG **INTER-1**, unclaimed.

Three things to carry:

- **GOALS §4's "the inter path saves only 17-27% vs all-I" is annotated, not deleted** — it is an
  equal-setting rate figure. At q=70 on crowd_run the inter arm spends 4.74 bpp against intra's
  7.96 *while sitting 7.6 dB lower*, and the 2026-03 run judged that quality equal on VMAF 99.09
  vs 99.10. **Any inter-vs-intra rate claim in this repo predating today is suspect** unless it
  names a BD-rate or a matched-quality point.
- **Quote worst-frame PSNR next to the mean for anything touching the inter path.** The mean says
  +4.6%, which reads as neutral; the worst frame says +19.1%, and for a contribution codec the
  worst frame is what survives downstream re-encoding. crowd_run q=70: inter 34.50 mean / 32.08
  worst, all-intra 42.07 / 42.06.
- **A q≤85 cap does not make a VMAF BD-rate safe.** crowd_run's overlap came out 99.55-99.84
  because the *all-intra* arm is already saturated at q=25, and the arithmetic produced +132.4% —
  the most dramatic number in the run and not a number at all. `meas3_sequence_rd.py` now prints
  the overlap beside every VMAF BD-rate and discards one whose floor exceeds 99. Worth copying
  into any harness that reports a VMAF BD-rate.
- **`git checkout --theirs .` after resolving a stash-pop conflict silently threw away three other
  sessions' work** — mine, in this worktree, on 2026-09-07. The two files were resolved correctly
  in the working tree but still marked `UU`, so that command replaced them with the stash side
  alone: RESEARCH_LOG lost the ABAC-SHIP, ENT-2 and MEAS-9 entries and COORDINATION reverted to a
  base predating three merges. Nothing was pushed — it was caught by counting the other sessions'
  headings before committing, and both files were rebuilt from `origin/main` plus the new block.
  **After resolving a conflict, `git add` the file; never run a bulk `checkout --ours/--theirs` to
  "clean up".** And when a conflict resolution touches a shared document, grep for the other
  sessions' entries as a matter of course — a rebuilt file that compiles and reads fine can still
  be missing 400 lines of someone else's day.

**GPU selection from the environment, and a tier/density harness — invalidates nothing.**
  `GNC_GPU_ADAPTER` (name substring), `GNC_GPU_BACKEND`, `GNC_GPU_POWER` and `GNC_GPU_INFO` choose
  the device at context creation; `gnc gpu-info` lists what wgpu can see. Encoder output is
  unchanged and verified so — q=75 on bbb_1080p is still 44.84 dB / 4.53 bpp, matching BASELINE.
  A `GNC_GPU_ADAPTER` that matches nothing is a hard error rather than a silent fallback, so a
  tier comparison cannot quietly be two runs on the same card. **Nothing is measured yet**:
  `scripts/gpu_tier_bench.py` and `docs/GPU_TIER_TEST.md` are waiting on a machine with more than
  one GPU. Do not quote a throughput number from this repo until that has run.

- **CHROMA-1 — `chroma_weight` now stays at 1.2 above q=85 instead of dropping to 1.0.**
  Changes default output at **q ≥ 85 only**: BASELINE's q=90 row moves 50.41 → 50.06 dB and
  8.58 → 8.07 bpp, which is −5.2% luma BD-rate for +1.2% on colour. q=25/50/75 are byte-identical
  and q=100 stays bit-exact (the quantiser is bypassed there). **Any q ≥ 85 file size measured
  before this is stale**; ratios within one build are fine.
  Two things worth carrying: **VMAF read 97.08 before and after, on 6% fewer bits** — luma-only and
  saturated, so a VMAF-only verdict here is worthless, which is the same illusion that made the
  2026-09-05 sweep look like a free 15%. And **luma measured as BT.709 Y from decoded RGB is
  contaminated by chroma error** — it overstated the loss 3.7x. Use YCoCg-R Y (`ypsnr_de00.py`).
  Also settled by elimination: the +90.5% video gap is **not** an allocation artefact. The knob is
  intra-only (−20.8% all-intra against −2.9% on a ki=9 P-chain), so intra really is the only route.

- **QUAL-1 — the headline gap figure is corrected.** At the contribution operating point GNC needs
  **+90.5% BD-rate on PSNR** (about 1.9x), not the **+456.7% to +672.1%** recorded from MEAS-1.
  Nothing in the coder changed; the old figure was measured at distribution bitrates.
  **Invalidates every use of the 5-7x figure**, including in POSITIONING and any argument that
  single-digit improvements are pointless — against 1.9x they are not.
  Two further things to carry: **never quote a VMAF BD-rate above about q=85** (widening the ladder
  moved it 47.5 points on average, 110 on old_town, while PSNR moved 1.0); and **at matched rate
  GNC beats x264 on dE00** while losing 7.4-8.8 dB of luma, so quote luma and colour together or
  the number misleads in whichever direction suits.
  Also: **MEAS-1's stated sources are not reproducible** — `bbb.y4m` has 8 frames, not 17, and no
  `touchdown` sequence exists in the tree.

- **The test material was unreachable for about ten minutes (18:04, restored).** `ln -sfn` run
  inside the shared checkout pointed `test_material/frames` at itself, so every worktree's link
  resolved to a loop and no session could read a test image. Restored to the real directory and
  the start-of-session command is now guarded against being run in the shared checkout. **Lost in the process:
  the `frames/hdr/` 10-bit material**, which was generated rather than fetched — regenerate it with
  `scripts/png16.py` if a 10-bit measurement is needed. Nothing else was lost, and no committed
  result depended on it.

- **BUG-11 + BUG-12** — Rice's stream mapping is now tile-width-aware (column-major, cut into 256
  contiguous segments) and the wavelet-level ceiling now follows the tile size actually in use.
  **Tile 256 output is byte-identical at all 12 measured points**, so nothing in BASELINE.md moves
  and no measurement taken at the default tile size is affected. What *is* invalidated: **every
  tile-size result in this repo, #47 and this morning's sweep included** — all of them scored the
  larger-tile arm through a coder that penalised it by 13–19%. Corrected figure: 512 over 256 is
  **−0.91% BD-rate**, so the geometry is worth ~1% and the default stays at 256. Also closed:
  6 wavelet levels at tile 512 is worth −0.1%, so deeper decomposition is not a lever either.
  **Use `CodecConfig::set_tile_size()`, never assign `tile_size` directly** — the level ceiling
  depends on it.

- **BUG-13 — FIXED 2026-09-06, and this bullet said "filed" for a day after it was closed.**
  BACKLOG is the current record: the reconstruction bug is fixed, block intra prediction was then
  measured, and it **costs 4-8% at lossless** because it predicts *and then* transforms, handing
  the wavelet a harder signal. That refutes prediction *before* a wavelet and says nothing about
  prediction *instead of* one, which is what INTRA-NEARLOSSLESS is gating. Original text follows,
  for the diagnosis in it: `GNC_INTRA_PRED=1` produced corrupt output at every quality: max
  error 197-255 from q=50 to q=100, and at q=100 it loses 62 dB against a bit-exact baseline.
  **The historical "-11.76 dB / +29%" measurement that set `intra_prediction: false` was measuring
  this bug**, not the idea. Error accumulates toward the bottom-right of every 32x32 block, which
  is an encoder/decoder predictor mismatch. Nothing else is invalidated -- the feature has always
  been off by default -- but the *conclusion* recorded against it is.
- **q=100 verified bit-exact lossless** on all three entropy coders. GOALS' "no true lossless with
  Rice" was stale and is corrected. GNC beats JPEG 2000 lossless by 10.8% and PNG by 7.8%; loses to
  FFV1 by 27% and x264 `-qp 0` by 43%, both of which predict against the neighbour.
- **BUG-17 and BUG-18 were assigned within minutes of each other on 2026-09-07, and it is
  resolved.** The `coord` session filed BUG-17 for the `abac_bitstream` parallel flake; the
  `abacship` session filed a different BUG-17 for an encode-path divergence on the inter path.
  **BUG-17 keeps the flake** (it was on main first) and is now **FIXED** — the `abacship` commit
  root-causes it as an `std::env::set_var` race that was masking a real decoder bug, which is the
  second of the two possibilities BUG-17's own entry asked someone to distinguish. The inter-path
  divergence is **BUG-18** (todo, P1). This is the second double-assignment in two days; read this
  file *and* run `scripts/claim list` before taking a number.

- **BUG-16 is taken** (2026-09-07, filed by `abacship`): Rice's GPU and CPU encode paths disagree
on the coefficients at q≤30 and at subsampled chroma. Unclaimed, on the default path, P2.

**Note on numbering:** BUG-11 was assigned twice on 2026-09-06 (Rice tile width, and intra
  prediction). The intra one has been renumbered **BUG-13**. Check this file before taking a number.

**ENT-2 — the entropy coders are measured against each other on one commit, and the answer is
"level". Invalidates no measurement; corrects four documents.** rANS against Rice, mean of four
stills, negative meaning rANS is smaller: **−6.4% at q=10, −7.1% at q=15, −6.7% at q=20, then
+0.4% at q=25 and +0.1% at q=40/55**. Above q=25 they are level and *which* coder wins is
content-dependent (−5.9% touchdown to +8.2% kristensara at the same setting). **No default moves**;
decision record 0018 records why, and 0015's prediction that "Rice still wins, and by more" is
falsified.

Four things to carry:

- **`--rans` is not a no-op**, contrary to BUG-9's entry — only the flag's `--help` text was stale,
  and that text (in five subcommands) called rANS the default and Rice "~30% worse compression".
  Verified at bitstream level: the harness parses `entropy_type` out of the GP17 header rather than
  trusting the flag, 40/40 points correct. **If a measurement of yours picked a coder with that
  flag, it did take effect.**
- **rANS's ceiling is q=75, not q=78.** q=75 encodes on all four images, q=77 on none, q=76 splits
  by content. The panic names stream **32** and a 3712-byte overrun, not stream 320. Handed to the
  BUG-9 owner; they report their fix is byte-identical on output (36/36 md5) and changes only the
  failure mode, so **no ENT-2 rate moves** — the q≥77 rows become a clean refusal rather than a
  crash, and rANS still stops below the operating point by design.
- **The q=20 coder cutoff and the 4→5 wavelet-level rule are coupled.** Every image's delta jumps
  in the same direction across q=25, because rANS pays a frequency table per subband group. Change
  one constant and the other is wrong.
- **The README's "1.5–2× faster" for Rice is gone, not corrected.** It contradicted TUNE-3's own
  ~15% by 3–6x, and re-timing needs an idle machine. That run is the one piece of ENT-2 left open.

Newest first. If you have measurements taken before one of these, they are suspect.

- **For the abac track, a result you did not ask for.** The `GNC_ABAC_COMPARE=1` harness was gated
  on `TransformType::Wavelet`, so it could not see the new lossless path. Gate widened (one
  `matches!`); `subbands()` at `num_levels = 0` already yields the right single region. **abac on
  MED residuals measures −14.3% mean on real coefficients** (bbb −14.9, blue_sky −15.6,
  kristensara −15.5, touchdown −11.2). With LOSSLESS-1 that is −27% against this morning's
  lossless and **+7.7% behind FFV1, from +48.4%**. Rate only. Your GPU decode throughput gate is
  now the only thing between that and a shipped result. Not implemented here — it is your code.

- **LOSSLESS-1** — q=100 now codes MED prediction residuals instead of wavelet coefficients
  (`TransformType::MedPredict`, bitstream `transform_type = 2`, `GNC_MED=0` to revert). Files are
  **14.9% smaller, still bit-exact**. **Invalidates every lossless figure in this repo**, including
  the "beats JPEG 2000 by 10.8%, loses to FFV1 by 27%" line — the FFV1 gap is now +25.8% measured
  as +48.4% before the change, so that comparison was against a different baseline than it reads.
  Also note `is_lossless()` now decides per transform type; anything keying on it should be
  re-read.

- **ENT-1** — subband-rANS frequency tables are now Exp-Golomb packed (bit 31 of the tile's
  `num_groups` word). Files shrink 11.7–26.6% on that coder at bit-identical quality; the preset
  path (q ≤ 20) shrinks 4.5%. **Invalidates every absolute byte or bpp figure for the rANS coder**,
  and it invalidates the *reason* for the Rice/rANS crossover at q=20 — re-sweep before quoting it.
  Landed inside commit `0e8987c`, which swept up another session's uncommitted work; the change was
  verified and measured only afterwards.

- **Tile-size sweep (BUG-11 / ENT-1 / BUG-12)** — measurement only, no code change. Result: 256 px
  is a local optimum and both 128 and 512 are worse *through Rice*, but the sign reverses on the
  rANS path (512 gains 14–20%). Rice hardcodes tile width 256 in its `i % 256` stream mapping.
  **Invalidates the conclusion of every past tile-size experiment, #47 included** — they were all
  scored through the coder that penalises the larger-tile arm. Also: `--tile-size` never reached
  its own level ceiling (BUG-12), so those runs were capped at 5 levels too.
- **TUNE-6** — the P-frame quantiser scale now follows the quantiser step (1.25× at step ≥ 4.6,
  tapering to 1.0 at step ≤ 2.8). Fixes BUG-10. **Invalidates any inter measurement taken between
  TUNE-5 and now at q > 80.**
- **abac (EBCOT part 3)** — new standalone module plus a `GNC_ABAC_COMPARE=1` diagnostic. No
  bitstream change, nothing wired into the pipeline. Invalidates nothing.
- **BUG-7** — `GNC_DIAGNOSTICS=1` was inflating encoded files by 32% by clobbering the
  motion-compensation reference. **Invalidates every sequence measurement ever taken with that
  flag**, including MEAS-4's residual dumps. `tests/diagnostics_neutral.rs` now guards it.
- **MEAS-2** — AQ off below q=30 (its gradient was inverted), reference deblocking off by default
  (measured neutral-to-negative). **Invalidates AQ and deblock measurements from before today.**
- **FMT-2 / GP17** — Rice-coded stream-length tables. Changes all file sizes; invalidates absolute
  byte figures, not ratios measured within one build.
- **BUG-6** — 5 wavelet levels at q ≥ 25. Changes the LL subband AQ measures variance on, and the
  per-group table count rANS pays for. **Any AQ or entropy-coder tuning from before this is
  stale** — that is how the rANS/Rice crossover came to sit exactly at q=25.

## Conventions worth keeping

- **Every new code path needs a canary** — a logged count or value proving it ran on real data.
  `GNC_DIAGNOSTICS=1` prints `groups=N deep_skipped=M`, `rice_tiles=N/N`, and so on. A feature
  without one is not done (CLAUDE.md).
- **A result that beats its own theoretical ceiling is a bug.** That is how the abac comparison's
  bad baseline was caught (−35% against a ceiling of −13.7%).
- **A toggle measured on the wrong metric reads as dead weight.** CfL looks like a loss on VMAF
  and is a clear win on CIEDE2000, because VMAF is luma-only. Anything touching chroma needs
  `scripts/chroma_metric.py`.
- **Point measurements at fixed q cannot judge a rate/quality trade**, and they always flatter the
  option that spends more bits. Use BD-rate, or compare at matched rate. Three separate wrong
  conclusions today came from this one error.
