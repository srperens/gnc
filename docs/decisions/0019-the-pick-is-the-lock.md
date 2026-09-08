# 0019 — The pick is the lock, and the preconditions are refusals

**Date:** 2026-09-07
**Status:** adopted
**Supersedes parts of:** [0016](0016-claiming-an-item-is-a-ref-cas-not-a-table-row.md)
**Scope:** coordination between concurrent sessions. No effect on the codec, the bitstream or any
measurement.

> **This number collided.** A second `0019` — the inter path's saving was an equal-setting figure
> — sat on `main` beside this one from 2026-09-07 to 2026-09-08 and is now
> [0056](0056-the-inter-paths-saving-was-an-equal-setting-figure.md) (BUG-19). A citation of
> `0019` written on either of those days may mean that record; this one is COORD-1.

## The problem 0016 left open

0016 made `take` atomic and was right about the cause: LOOP step 1 is a *deterministic* rule
applied by every session to one shared list, so identical inputs and an identical rule produce
identical output, and **determinism is what makes the collision reliable rather than unlikely.**

It then fixed the wrong half. An atomic `take` decides who wins a race it does not prevent. Eight
sessions still read the same BACKLOG, still choose the same item, and seven still lose — and the
instruction they are following at that point is "pick again", which sends them back to the same
list with the same rule. The collision rate falls only because the winner's claim removes one item
per round.

Three further gaps were found while measuring this, and each one had already fired:

1. **Identity recorded the wrong process.** `session_pid` walked up the tree looking for the
   session's `claude` process and then printed that process's *parent*. So identity was the
   terminal claude was launched from, not the session: two sessions started from one shell were
   again the same principal — the exact defect 0016 fixed at the worktree level and reintroduced
   one level up. Worse, `kill -0` was testing a login shell that outlives the session, so
   `SESSION GONE` could not fire and no claim was ever detectable as abandoned. Measured on the
   live tree: this session's claim recorded `s56066`, the login zsh; the session was `s69710`.
2. **Rule 0 was prose.** Nothing stopped `take` from the shared checkout, and two claims
   (`CHROMA-2`, `RATE-1`) were live from `gnc@main` when this was written. Rule 0 has existed
   since concurrent sessions started, is stated in three files, and was being broken anyway.
3. **Unknown liveness authorised a takeover.** `session_alive` returns three states — alive, gone,
   cannot-say — and `cmd_worktree` tested it with `if`, so cannot-say fell through to the
   *take over* branch. A worktree whose owner could not be identified was treated as abandoned.

## What was chosen

**`scripts/claim next "<why>"` — the pick and the claim are one compare-and-swap.** It walks the
startable BACKLOG headings in priority order and takes the first one nobody holds. N sessions
running it at the same instant get N different items, by construction; there is no window between
deciding and claiming because there is no deciding step that the claim does not perform.

The queue is BACKLOG.md's own headings — `### NAME-<n> — title (todo, P<n>)` — read from
committed `main`, not from anyone's working tree. That last part matters: a queue that depends on
whose editor is dirty is a queue that gives two sessions different answers. Finished items lose
their priority marker when they gain `**DONE**`, so they leave the queue by construction rather
than by anyone remembering to remove them.

**The two preconditions are enforced.** `claim` refuses to hand out work from the shared checkout,
and refuses to hand out work in a worktree the caller does not hold. Holding the worktree is now a
precondition for holding an item, which chains the four start-of-session commands together: you
cannot reach step 4 by skipping steps 1 and 3.

**`list` distinguishes three states** instead of two, because they need different actions:
`SESSION GONE, safe to steal`; `STALE, no heartbeat` — alive but silent for an hour, ask before
taking; and `no session recorded, verify before trusting` — a seeded or `--as` claim, whose
liveness nothing can check. The last state was previously indistinguishable from healthy.

**`selftest` asserts both properties**, since they fail differently: 16 processes racing for one
item produce exactly one winner and the losers are told who holds it; six processes racing on one
six-entry queue produce six distinct claims. That is the canary CLAUDE.md requires — the exclusion
is measured, not asserted.

## The three registries are now one

Before this, "who is working on what" was recorded in three places: `refs/claims/*`,
COORDINATION's worktree table, and BACKLOG's `(in progress)` heading markers — and LOOP step 1
instructed sessions to write the third. 0016 itself added that instruction, reasoning that the
file sessions pick *from* should not advertise a taken item.

That reasoning is right and the remedy was wrong. A second copy of the state cannot be kept in
sync by instruction; it can only be made unnecessary. Now `next` reads the queue and the claims
together, so a taken item is not offered no matter what its heading says, and the instruction to
mark it is gone. BACKLOG's status is written when the item *finishes*. The worktree table keeps
the prose a claim note cannot carry and says, at the top, that `scripts/claim list` wins if they
disagree.

## What was not chosen, and what it would have cost

- **A separate machine-readable queue file** (`BACKLOG.ready`, or a JSON list). Cleaner to parse
  and it is a fourth registry: it can disagree with BACKLOG, and keeping it current is another
  instruction nobody executes under load. Parsing the headings costs one `awk` and cannot drift
  from the file it describes.
- **Randomising the pick** instead of ordering it, so sessions collide only by chance. Cheaper to
  implement and it throws away the priority order the backlog exists to express — a P4 bug and a
  P0 measurement would be equally likely. It also only reduces the collision probability;
  `next` removes it.
- **Making `--as` an authorisation boundary.** It is the seeding, handover and park mechanism, and
  it deliberately skips the worktree guard. Anyone who wants to bypass the lock can; the mechanism
  excludes racing sessions, not sessions that ignore it. Guarding it would break the parking of
  `CANARY-1` and `MEAS-5` and buy nothing against an actor the design does not model.
- **Automatic expiry of item claims.** Still rejected, for 0016's reason: an expiry short enough
  to be useful will eventually expire a live claim, and an item's claim carries a half-finished
  train of thought. `STALE` surfaces the same information and leaves the decision to a session
  that can check. Worktrees remain the exception and reclaim automatically, because there the
  holder's death is unambiguous.
- **Rewriting the identity of existing claims.** The pid fix changes what `me()` computes, so
  every claim taken before it would have been refused by its own owner's `touch` — generating
  precisely the manual cleanup this mechanism exists to prevent. Instead the owner check accepts
  an owner whose worktree and branch match and whose recorded pid is one of the caller's
  ancestors: that is the old identity and nothing else, and it decays as those claims are dropped.

## What this still does not fix

- **A session that does not run `claim` at all.** Unchanged from 0016, and unchangeable without a
  hook. The mitigation is that the start-of-session procedure is now four commands in one block in
  one file, and the last of them is the claim.
- **Seeded claims.** Four are held by an owner with no session part, so nothing can say whether
  they are still real. They are now labelled rather than shown as healthy, but somebody has to
  look.
