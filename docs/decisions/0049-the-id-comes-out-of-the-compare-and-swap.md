# 0049 — The id comes *out* of the compare-and-swap

**Date:** 2026-09-08
**Item:** COORD-2
**Status:** accepted

## Context

`scripts/claim take dr-NNNN` excludes another *taker of that string*. It never reads the
namespace. Three failures followed, all the same shape:

1. Two sessions compute "next is 0018" and both write (the original race).
2. A session follows the reserve-first rule and still gets a number that has been on `main`
   for hours (`dr-0029` vs `fcc3d33`), because the CAS is on `refs/claims/dr-0029` and
   nothing looks at `docs/decisions/`.
3. A session reserves from a stale worktree whose `ls docs/decisions/` stops at 0028.

`claim next` already closed (1) for *work* items by making the pick be the CAS. Ids did not
get that.

## Decision

`scripts/claim bug "<why>"` and `scripts/claim dr "<why>"`. Each walks committed `main`
(BACKLOG `BUG-N` for bugs, `git ls-tree main docs/decisions/` for records) **and** live
`refs/claims/*`, then `update-ref` the first gap. A lost race retries. The number is a
return value, not an argument.

`take dr-NNNN` stays, for parking and for a number you already hold. It is no longer how
you *get* a number.

`claim selftest` now races 8 processes on `claim bug` and 8 on `claim dr` and asserts N
distinct ids. Measured: 8 claimed, 8 distinct, both kinds. This record itself was allocated
by `claim dr`.

## What was not chosen

- **Keep reserve-first plus `git ls-tree main`.** Those two oracles cover a committed
  number and a held-but-unwritten number. They do not cover two sessions reading "0031 is
  free" in the same minute, which is why this item existed.
- **Scan other worktrees' uncommitted files.** Invisible to git refs; the habit that covers
  it is still "commit the stub with the reservation".
- **Renumber the live 0018/0019/0024/0027 pairs.** That is BUG-19. This item stops new
  collisions; it does not rewrite history.

## Canary

`scripts/claim selftest` prints `claim bug, 8 racing processes: 8 claimed, 8 distinct` and
the same line for `claim dr`, then `PASS`.
