# 0016 — Claiming an item is a ref compare-and-swap, not a table row

**Date:** 2026-09-07
**Status:** adopted; the "does not fix" section is partly superseded by
[0019](0019-the-pick-is-the-lock.md), which makes the *pick* atomic and turns rules 0 and 0b into
refusals
**Scope:** coordination between concurrent sessions. No effect on the codec, the bitstream or any
measurement.

## The problem

COORDINATION.md has carried a "Worktrees currently out" table since concurrent sessions started.
It is documentation, and it was being used as a lock. It cannot be one: reading the table, deciding
what to work on, and writing your row are three separate steps with no interlock between them, so
every session that starts inside that window reads "free".

On 2026-09-07 this stopped being theoretical. Up to eight instances were running against one
checkout. Several picked MEAS-9 at the same time, two wrote a `meas9` row within about 90 seconds
of each other (`cf08d0d`, then `c0dd27f` to clean up the duplicate), and a third was found running
`meas9_contribution.py` **with its cwd in another session's worktree**, both sessions editing the
same file. That last one is the failure mode COORDINATION rule 1 was written for after two
sessions' edits to `abac.rs` and `gpu_util.rs` overwrote each other on 2026-09-06.

Worth being precise about the cause, because it is not carelessness. LOOP step 1 is a
*deterministic* rule — "pick the best value-to-effort item that is not blocked" — applied by every
session to one shared list. There were exactly three `(todo, P1)` items: CANARY-1 needs a second
GPU this machine does not have, CHROMA-2 was taken, so MEAS-9 was the only startable P1. Identical
inputs and an identical rule produce identical output. **Determinism is what makes the collision
reliable rather than unlikely**, and no amount of care in following the instructions avoids it.

## What was chosen

`scripts/claim`, a small wrapper over `git update-ref`:

```bash
scripts/claim take MEAS-9 "why, briefly"   # atomic; non-zero means someone else holds it
scripts/claim list / owner / touch / drop / steal / selftest
```

Two properties of this repository make it the right primitive, and neither needed building:

1. **Every worktree shares one `.git`.** A ref written in one worktree is visible in all of them
   the same instant — no fetch, no push, no daemon, and nothing in any working tree to conflict on.
   `refs/claims/*` is repository-wide (only `HEAD`, `refs/bisect/*` and `refs/worktree/*` are
   per-worktree), which is exactly the sharing we want.
2. **`git update-ref <ref> <new> <old>` is a compare-and-swap** under git's own ref lock, and an
   empty `<old>` means "must not exist" — precisely create-if-absent. `drop` and `steal` pass the
   value they read as `<old>`, so neither can clobber an owner that changed underneath them.

Claims are metadata blobs, so `git cat-file -p refs/claims/MEAS-9` prints the owner, the time and
the commit it was taken against. Identity is worktree-basename@branch: no username, no hostname, no
absolute path (CLAUDE.md's infrastructure-detail rule). Nothing is pushed — the contention is local
to this machine, so the claims should be too.

`scripts/claim selftest` races 16 processes for one claim and asserts exactly one wins *and* that
the losers are told who holds it. That is the canary CLAUDE.md requires: the exclusion is measured,
not asserted. It passes.

## What was not chosen, and what it would have cost

- **A lock file committed to the tree** (`CLAIMS.json`, or a file per claim). This is the obvious
  move and it is strictly worse than the table it replaces: two sessions can both create the file,
  and now you have a merge conflict in a file whose entire purpose is to prevent conflict. Git's
  index is not a mutex.
- **`flock`** on a shared file. Genuinely atomic, but a lock lives only as long as the process
  holding it, and a session here is not a process — every Bash call is a fresh shell. A claim has
  to outlive the shell that took it, so the lock would have to be released by an explicit call
  anyway, i.e. it would be state, not a lock. That is what a ref already is.
- **`mkdir` / `ln` / `open(O_CREAT|O_EXCL)`** in a scratch directory. All three are atomic and any
  of them would have worked. Rejected only because they add a second place where coordination state
  lives, one that no existing tool inspects, that is invisible to `git log`, and that has to be
  cleaned up out of band. The ref costs nothing extra and `git for-each-ref` already reads it.
- **A daemon or a coordination service.** Disproportionate. Five to eight processes on one laptop
  do not need a service to agree on six strings.
- **Doing nothing, and writing a stronger warning in COORDINATION.md.** This was in effect the
  status quo, and the file already carried "we have the scars" plus four numbered rules. The
  sessions that collided today had all read it. A rule that requires N deterministic agents to
  behave non-deterministically is not a rule.

## What this does not fix

- **Anyone who does not run it.** The mechanism is cooperative; `--as` will record whatever owner
  you claim to be. It excludes racing sessions, not sessions that ignore it. Mitigation is that
  LOOP step 1 and COORDINATION rule 0b now both put the `take` before the work, and the seeded
  claims mean `scripts/claim list` is informative from the first read.
- **Stale claims, partly.** A session that dies holding an *item* leaves it held; `list` now marks
  it `SESSION GONE, safe to steal`, but nothing frees it on its own. Deliberately manual for items
  — automatic expiry would eventually expire a live one, and an item's claim carries a
  half-finished train of thought. Worktrees are the exception and reclaim automatically, because
  there the holder's death is unambiguous and the cost of waiting is a directory nobody can use.
- ~~**Two sessions in one worktree**~~ — **fixed within the hour, because it bit immediately.**
  Two things were wrong in the first version. Identity was `<worktree>@<branch>`, which makes two
  sessions in one worktree *the same principal*: the other session's `touch` succeeded against my
  claim, so the mechanism excluded every session except the one it needed to. And a claim on an
  item does not protect a directory at all. Identity is now `<worktree>@<branch>#s<pid>` where the
  pid is the session's own `claude` process — stable across calls, and the same number the peer
  sockets already use as an address — and `scripts/claim worktree` claims the directory under the
  same CAS. It refuses when a live other session holds it, takes over automatically when the
  holder's session is gone (a finished session's worktree is free; that is the normal case), and
  refuses to claim the shared checkout, which is shared on purpose. The pid also turns "stale
  claim" from a guess into a check: `list` marks one `SESSION GONE, safe to steal` when `kill -0`
  on the holder fails.

  Worth recording that the first version's own bug was to lock the shared checkout — the one
  directory every session legitimately meets in.

## Also corrected while here

BACKLOG's MEAS-9 heading read `(todo, P1)` for the whole time it was claimed in COORDINATION.md.
The claim lived in one file and the pick was made from another, so the file sessions choose *from*
was advertising a taken item. The `(in progress <date>)` marker already existed in BACKLOG and was
used once in 90 KB; it was set for MEAS-9, and LOOP step 1 was made to require it.

**That remedy is withdrawn — see [0019](0019-the-pick-is-the-lock.md).** It created a third copy
of "who holds what", and a second copy of state cannot be kept in sync by instruction. `scripts/claim
next` now reads the queue and the claims together, so a claimed item is never offered whatever its
heading says. BACKLOG's status is written when the item finishes, not when it is claimed.

Recorded at the same time, so the next session does not re-derive it: **JPEG XS is not measurable
on this machine.** ffmpeg knows the codec id and has no implementation (`..VILS jpegxs`), neither
`libjxs` nor `svt-jpeg-xs` is a Homebrew formula, and SVT-JPEG-XS ships Linux/Windows build trees
with x86 assembly. MEAS-9 is therefore delivered against ProRes 4444/422, VC-2 and JPEG 2000, with
VC-2 as the nearest available relative to JPEG XS; the JPEG XS row stays open.
