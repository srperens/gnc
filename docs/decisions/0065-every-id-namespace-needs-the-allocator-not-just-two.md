# 0065 — Every id namespace needs the allocator, not just the two we noticed

Date: 2026-09-08
Item: COORD-3
Status: accepted
Supersedes nothing; extends `0050` (COORD-2)

## Context

`0050` established the rule: **an id must come *out* of the compare-and-swap, not be checked
against it.** It shipped `scripts/claim bug` and `scripts/claim dr`, because the two collisions on
record at the time were a `BUG-N` and the `0018` / `0024` decision-record pairs.

Item ids were left to be picked by hand. On 2026-09-08 that produced the fifth instance of the same
mechanism and the first one inside BACKLOG: the ENT-3 session filed `### ENT-9 — abac context-codes
three decisions ...` at 18:12, DOC-3 filed `### ENT-9 — should abac be the default?` at 18:29, and
both reached `main`. Neither session could see the other's heading, which is exactly what `0050`
says makes the collision reliable rather than unlikely.

An item id collision is worse than a decision-record one. `refs/claims/ENT-9` is *the lock*: it can
hold one of the two headings, and whoever takes it makes the other item **invisible to
`claim next`** — silently, until someone renumbers. `claim items` printed `ENT-9` twice for half an
hour and it read as a display quirk.

## Decision

**`scripts/claim id <PREFIX> "<why>"` allocates the first free `PREFIX-N` as the reservation, for
every prefix**, and `claim bug` becomes a shorthand for `claim id BUG` rather than a second copy of
the logic. `used_prefix_numbers` unions every `PREFIX-<n>` mentioned in committed `main:BACKLOG.md`
with live `refs/claims/*` and CASes the first gap.

**A mention counts as taken, not only a heading.** Every collision in this repository's history was
"someone filed it and I could not see it", so the conservative direction is to treat an id that is
spoken for anywhere as spent. Ids are free; a collision costs a renumber and, this time, a
retraction.

**`claim items` and `claim next` warn on two startable headings sharing an id — they do not
refuse.** Nothing can be double-granted (there is exactly one ref), so the failure mode is
unavailability, and refusing would remove *two* items from the queue to punish one filing mistake.
The warning names the prefix to allocate from.

## Why not the alternatives

- **Leave it to convention and reading `grep '^### '`.** That is what was in place. It has now
  failed five times, three of them on the same afternoon, and it fails *because* every session
  applies the same correct rule to the same file inside the same two minutes.
- **A monotonic counter file (`.claim-next-id`) instead of scanning.** One more piece of state to
  keep consistent with the thing it describes, and it cannot be reconstructed from `main` if it
  drifts. Scanning committed BACKLOG plus live refs has no state at all: both inputs are already
  authoritative and already shared through one `.git`.
- **Refuse to hand out a duplicated id from `next`.** Rejected above: the cost falls on the wrong
  people. Warn, and let the renumber be cheap.
- **Renumber the *first* ENT-9 (theirs) instead of DOC-3's.** Refused: theirs was filed 16 minutes
  earlier *and* is held by a live session, so renumbering it would break a claim someone is
  currently working under. The rule that follows — **a held id does not move; an unclaimed one
  does** — is what COORDINATION now says, and it is the opposite of the `0024` precedent only in
  appearance: there, "the unreserved record moves" picked out the same side.
- **Renumber both to leave the collision unambiguous in history.** Two edits, one of them to
  another session's in-flight entry, for no gain over one.

## Consequence

`scripts/claim selftest` now asserts four properties instead of two: one item has exactly one
winner, N pickers get N distinct items, N `bug`/`dr` allocators get N distinct ids, and N `id`
allocators get N distinct ids **none of which is a number the backlog merely mentions in prose**.
Plus a fixture pair for the duplicate warning — present when there are two, silent when there is
one.

What is still open is the cleanup, not the mechanism: the `0018` and `0024` decision-record pairs
are BUG-19's, held by another session. `claim dr` has prevented new ones since COORD-2, and
`claim id` now does the same one level up.
