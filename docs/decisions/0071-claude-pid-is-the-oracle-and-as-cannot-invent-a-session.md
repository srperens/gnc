# 0071 — `CLAUDE_PID` is the oracle, and `--as` cannot invent a session

**Date:** 2026-09-08
**Item:** COORD-7
**Status:** accepted — hole closed, oracle improved, remaining unknown instrumented

## Context

`0069` fixed the *diagnostic* for a claim whose holder cannot be tested for liveness and
deliberately left the *cause* alone, because there was nothing left to measure: the walk worked in
the session that fixed it, and the three `s?` claims had been written an hour earlier by process
trees that no longer existed. COORD-7 was filed with an instrument rather than a hypothesis, and
with a second question attached — what writes an identity like `gnc-next2@next2#g01a08196`, which
`me()` cannot produce.

Both were investigated before anything was changed.

## What was found

**1. No version of `scripts/claim` has ever emitted a `g` prefix.** Checked mechanically over
every commit that touched the file: the only match is `0069`'s own commit, which quotes the string
in a comment. `me()` prints `<tree>@<branch>#s<pid>` or `…#s?`, and the single other route into
that field is `CLAIM_AS`, which `--as` sets. So the identity came through `--as`.

**And `--as` accepted it, which is the actual defect.** It took any string. The cost is precise:
`claim_state` treats an owner containing `#` as a session identity, `session_alive` cannot parse
`g01a08196`, so **three claims — `PERF-2`, `dr-0051` and `worktree.gnc-next2` — became
permanently untestable** rather than merely unheld. An owner with no `#` (`blocked-idle-machine`)
was always handled correctly; it is the session-*shaped* value that slips through.

`01a08196` is eight hex characters, the shape of a truncated session UUID, but **no session
directory for this project starts with it**, so its provenance is unresolved and is left that way.
It cannot recur, which is the part that mattered.

**2. `CLAUDE_PID` is set in the environment of the shells a session runs, and it is exactly what
the walk is looking for.** Measured in `gnc-loopa`: `CLAUDE_PID=8815`, the twelve-hop walk
independently arrived at 8815, and `ps -o comm= -p 8815` prints `claude`. Two independent methods,
one answer.

**3. A latent bug in the walk, found by reading its own output.** The instrument printed
`30357:` with an empty name for a login shell, because `ps -o comm=` reports `-/bin/zsh` and
`basename` reads the leading `-` as an option. Four call sites, all now `basename --`. It was not
the cause of any `s?` — the comparison it feeds is against `claude`, which is never a login shell
— but it was making the new diagnostic lossy in the exact place it is meant to be read.

## Decision

Three changes, all in `scripts/claim`:

- **Prefer `CLAUDE_PID`, and only if it still names a live `claude`.** The walk stays as the
  fallback, because it is not known whether the sessions that recorded `s?` set the variable at
  all. Trusting a stale exported value would make a dead session look alive, which is the one
  direction that loses work, so the value is checked against `ps` before it is used.
- **`--as` refuses a session part the liveness test cannot read.** An `--as` value may name no
  session (`blocked-<reason>` — a reason, not a directory) or name one that can be evaluated
  (`s<pid>`, or `s?` when genuinely unknown). Handing an area over to a real session still works;
  inventing a session does not.
- **An `s?` claim records the chain that was walked**, as a `walk:` field in the claim blob:
  `walk: claude-pid=unset chain: 86265:zsh`. So the next `s?` is a reading. Identifiable claims
  record nothing.

**The instrument's first implementation was wrong and the mutation test is why that is known.**
It set a global inside `session_pid`, which `me()` calls in a command substitution — a subshell —
so `blob_for` would have written nothing, forever, while the code read as if it worked. The
diagnostic now recomputes the walk in its own function. This is the third time in one session that
a check had to be shown to fail before it could be trusted (`0062`'s two constant assertions,
`0069`'s worktree evidence, this), which is starting to look less like coincidence than like the
cost of writing checks in a language with no test framework.

Both new behaviours are asserted in `claim selftest` and both assertions were mutation-tested:
disabling `valid_as` produces `FAIL: --as accepted a session part it cannot evaluate` and
`FAIL: the refused --as still took the claim`; removing the `walk:` line produces
`FAIL: an s? claim recorded no walk diagnostic`.

## What was not chosen

- **Replacing the walk with `CLAUDE_PID` outright.** It is one measurement in one session. If some
  sessions do not export it — which is the live hypothesis for the three `s?` claims — removing
  the fallback would turn a working identity into `s?` for exactly those sessions.
- **Rewriting the three existing `#g…` claims.** They are held by a session that may be alive, and
  `0069` already makes them actionable: `list` reports their worktree. Rewriting another session's
  claim to make a diagnostic tidier is the kind of thing this lock exists to prevent.
- **Refusing `#` in `--as` altogether.** Simpler, and it would break handover, which COORDINATION
  names as one of `--as`'s three purposes. Validating the session part keeps all three.
- **Chasing `01a08196` further.** It matches no session directory for this project. The remaining
  routes are other projects' session ids or an agent id, and none of them changes what to do: the
  hole is closed either way, and a provenance answer would only be interesting, not actionable.

## What this leaves open

**Why a session's ancestry sometimes contains no `claude`.** Still unknown, and now the only open
half. It is no longer open-ended: any future `s?` carries `claude-pid=…` and the chain it walked,
which distinguishes the three candidates — no `CLAUDE_PID` and a reparented shell, a `claude`
running under a different `comm`, or a chain longer than twelve hops — without guessing. If
`CLAUDE_PID` turns out to be universally set, the walk becomes dead code and can go.
