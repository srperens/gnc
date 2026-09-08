# 0076 — The orphan count is arithmetic, and `pgrep` is the wrong oracle

**Date:** 2026-09-08
**Item:** COORD-8 (COORD-5's open half)
**Status:** accepted — `claim list` prints its half of the arithmetic; the session count stays out of the repo

## Context

`0069` made an untestable claim *actionable* by reporting the holder's worktree, and left the
question it could not answer: **is the holder still there?** `0071` then removed one cause
(`CLAUDE_PID`) and instrumented the rest. What remained was a listing that says
`OWNER UNIDENTIFIABLE … verify before trusting` **without saying how to verify** — and a peer
session pointed out that the obvious how is wrong.

## What was found

**`pgrep -x claude` is unsound for this, and it fails in the direction that loses work.** It
omitted a live session whose own `ps -o comm= -p <pid>` reads `claude` — this session, pid 8815,
missing from `pgrep -x claude` output taken minutes earlier. A peer session hit the same thing and
nearly concluded four holders were gone from a listing that had missed *itself*. `kill -0` plus
`ps -o comm=` is sound, and is what `session_alive` already does.

**The individually unanswerable question is answerable in aggregate.** An untestable identity
cannot be resolved one row at a time, but the *count* falls out of arithmetic:

| | |
|---|---|
| worktree claims | 17 |
| held by a live, testable session | **8 distinct sessions** |
| identity cannot be tested | 6 |
| live sessions in this repo (`ListAgents`, socket registry) | **9** |

9 live − 8 accounted = **1 live session unaccounted for**, so **at least 5 of the 6 untestable
claims are orphaned** and at most one could still be real. Claim age narrows *which*: those six
were taken over two hours ago, so a session started an hour ago cannot own one.

**That bound is weaker than "they are all gone", and the difference matters.** The peer's read was
"almost certainly gone" for four of them; the arithmetic says at least five of six, not six of
six. Stealing on the stronger reading is four steals on an inference, and this project has been
punished for that twice today.

## Decision

**`claim list` prints its own half of the arithmetic and names the other half rather than
computing it.** The summary reads:

```
17 worktree claim(s): 8 distinct live session(s) hold one or more, 6 with an identity that cannot be tested.
Count the sessions actually running and subtract: anything above 8 live session(s) is
the most that could still own those 6 rows. Do NOT use `pgrep -x claude` (COORD-8).
```

**Distinct sessions, not rows.** One session can hold several worktrees — pid 19376 held two while
this was being written — and counting rows overstates how many holders are accounted for, which
is precisely the number the summary exists to get right. The first version of this counted rows
and reported 9 where the truth was 8; it was caught by *using* it, one paragraph after writing it.
`claim selftest` now asserts the delta — two worktrees taken by one new session must raise the
count by exactly one — and the assertion was mutation-tested: removing the dedupe makes it read
`9 -> 11`.

## What was not chosen

- **Reading the agent harness's socket registry from `scripts/claim`.** It works — there is a
  socket per live session and its set matched `ListAgents` exactly, 16 of 16, with every pid alive.
  Rejected because **it is not the repository's business how the sessions are supervised.** A lock
  that hardcodes `/tmp/cc-socks/<pid>.sock` couples this project's coordination to one harness's
  private layout, and it would fail silently the day that layout changes — silently in the
  direction of declaring live sessions dead. The half the repository owns is its own claims; the
  session count belongs to whoever runs the sessions.
- **Failing closed: treating "cannot be tested" as gone.** Already rejected in `0069` and it is
  more tempting now that the arithmetic almost closes. It still hands a session someone else's
  13 uncommitted files and calls it safe.
- **Stealing the orphans as part of this item.** At least five of six are dead, and the work in
  them — 16 uncommitted files across `bug35rans`, `g41232`, `next2`, plus `bug32`, `coord2`,
  `rebaseline` — is unowned. But "at least five of six" does not say *which* five, and a takeover
  is a decision about someone's work rather than about a diagnostic. It wants a person, or a
  session that has read the diff, which is what `0069` said and still says.
- **Preserving the orphaned work behind a ref** (`git stash create` into `refs/wip/<area>`, which
  touches no worktree, index or branch). This was attempted and **refused by the permission
  layer**, so it is not done and is not smuggled in another way. It is recorded here because it is
  the cheapest known way to make 16 files of unowned work survive, and it needs a decision from
  the project owner rather than from a session.

## What this leaves open

One live session is unaccounted for and one of the six untestable claims may be real. The
instrument for closing that is already in place from `0071`: the next claim written with an
untestable identity carries the ancestor chain that produced it. Until one appears, there is
nothing to read.
