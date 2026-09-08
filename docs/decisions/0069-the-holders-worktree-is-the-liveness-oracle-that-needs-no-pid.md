# 0069 — The holder's worktree is the liveness oracle that needs no pid

**Date:** 2026-09-08
**Item:** COORD-5
**Status:** accepted — `claim list` reports the worktree; the pid walk is left alone

## Context

COORD-1 made a claim's identity `<worktree>@<branch>#s<pid>` and said why the pid is there:

> The pid also makes an abandoned claim detectable rather than merely old: `list` marks it
> `SESSION GONE, safe to steal`, and separately marks a live holder that has not touched in an
> hour `STALE, no heartbeat`. Those need different actions — steal the first, ask about the
> second.

On 2026-09-08 at 18:58 that property was **not holding for 4 of the 15 live item claims**, and
`scripts/claim next` had just reported every startable item claimed, so those four were the whole
difference between a working queue and an empty one:

| item | owner | held | worktree |
|---|---|---|---|
| BUG-35 | `gnc-bug35rans@bug35rans#s?` | 74m | 1 file uncommitted, last edit 67m ago |
| PAD-2 | `gnc-g41232@g41232#s?` | 75m | 7 files uncommitted, last edit 67m ago |
| TILE-1 | `gnc-tile1@tile1#s?` | 75m | 13 files uncommitted, last edit 68m ago |
| PERF-2 | `gnc-next2@next2#g01a08196` | 69m | 8 files uncommitted, last edit 67m ago |

Three recorded `s?` — `session_pid` walks up to twelve ancestors looking for a process whose
`comm` is `claude` and prints `s?` when it finds none. The fourth recorded `g01a08196`, which
**this script cannot produce at all**: `me()` prints `s<pid>` or `s?`, so that identity came from
somewhere else. `session_alive` returns "cannot say" for all four, and `list` printed
`OWNER UNIDENTIFIABLE` with nothing after it.

**That is the MEAS-5 shape exactly** — a session gone with 145 uncommitted lines nobody looked at
for eleven hours — and the pid was introduced to make it visible. Here it made it invisible in a
new way: not "healthy", but "unknown", which a session cannot act on either. Stealing risks
duplicating or destroying up to 13 files of someone's work; not stealing leaves four items and
that work idle.

## Decision

**Report the holder's worktree whenever liveness cannot be established, instead of telling the
next session to go and look.** COORDINATION already names the check — *"When you see `SESSION
GONE`, look in the worktree before you steal the item. `git -C "$REPO-<area>" status --short` is
one command and it is the difference between inheriting the work and repeating it"* — so this is
that advice, mechanised. The identity already contains the worktree name; `git worktree list`
resolves it to a path; `git status --porcelain` and the mtimes of the dirty files answer the
question.

`claim list` now prints one of:

```
SESSION GONE, safe to steal, worktree clean
OWNER UNIDENTIFIABLE, 13 file(s) uncommitted, newest edit 69m ago
no session recorded, verify before trusting            # a parked owner: no worktree to look at
```

The first is safe to take. The second must not be stolen without reading the diff. The third is
`--as blocked-…`, deliberate, and names a reason rather than a directory — so nothing is looked
up and the generic warning is all there is to say.

**Verified by mutation, not by reading.** `claim selftest` gained a case that claims an item under
an unidentifiable owner naming a real worktree and asserts the evidence appears, plus a parked
owner and asserts it does *not* get described as a directory. Breaking `worktree_evidence` to
return nothing makes it fail:

```
FAIL: an unidentifiable owner naming a real worktree reported no evidence
```

That check exists because `0062` had just found two runtime assertions over compile-time constants
in this repository — a test that cannot fail is worse than no test, and a new assertion should be
shown to fail before it is trusted.

## What was not chosen

- **Fixing `session_pid` so the walk always finds the session.** This is the real repair and it is
  deliberately not attempted here, because **the cause is not diagnosable from the claims.** The
  walk works in this session — the chain is `zsh → claude(8815) → zsh → login → ghostty` — and the
  three `s?` claims were written over an hour ago by sessions whose process trees no longer exist
  to inspect. Guessing at a fix (match `node`, widen the hop limit, read an env var) and shipping
  it would be a change with no before-number, which is what this project's protocol exists to
  refuse. The `g01a08196` identity says something else is also writing claims, and that wants
  finding before the walk is touched.
- **Making `s?` fail closed — treat "cannot say" as `SESSION GONE`.** Simple, and it would have
  freed the four items. It would also have handed a session an item with 13 uncommitted files in
  another worktree and called it safe. The whole reason COORD-1 separated `GONE` from `STALE` is
  that they need different actions; collapsing "unknown" into "gone" undoes that in the direction
  that loses work.
- **Making it fail open — hide the row, treat the claim as healthy.** That is the pre-COORD-1
  behaviour the pid was added to end.
- **A liveness heartbeat file per session.** More moving parts, and it answers a question the
  worktree already answers better: "is the process alive" is not actually what a session wants to
  know before stealing — "is there work in there" is.
- **Stealing the four items as part of this item.** Out of scope and not this session's call: the
  evidence now says all four hold uncommitted work, so the right next step is a human or a session
  reading those diffs, not an automatic takeover. The item that fixes the queue is not the item
  that fixes the diagnostic.

## What this leaves open

The four claims are still unverifiable and the queue is still fully claimed — this changes what
`list` *says*, not who holds what. What it buys is that the next session sees
`13 file(s) uncommitted, newest edit 69m ago` instead of a shrug, which is the difference between
a takeover that inherits the work and one that repeats it.

Two follow-ups worth their own items: why `session_pid` returns `s?` for some sessions and not
others, and what writes a `#g…` identity.
