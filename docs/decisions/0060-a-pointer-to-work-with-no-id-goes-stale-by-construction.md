# 0060 — A forward pointer to work with no ID goes stale by construction

Date: 2026-09-08
Item: DOC-3
Status: accepted

## Context

BACKLOG's **priority order** is the section a session reads to answer LOOP.md step 2 — "is this
still the right item?". On 2026-09-08 it named six things and five of them were closed, some for
two days: PAD-1 and INTRA-2 (both shipped), LOSSLESS-1 ("buildable now", built 2026-09-06),
CANARY-1 ("never been measured", DONE at 34x), BUG-14 (DONE), and abac's inter figure (superseded
by `0045` the same morning).

Every one of those lines was true when written. They went stale because *someone else finished the
work*, which is the normal case with eight concurrent sessions, not an accident.

The interesting one is item 1: **"the next largest known intra lever is still unbuilt — see EBCOT
Part 7's open items."** Part 7 has no open items left. And the lever is not unbuilt: it is
`--abac`, shipped 2026-09-07, worth −16.6% to −18.8% of intra rate at *identical pixels*, sitting
behind a flag. Making it the default had been named three times — ENT-5's "not in scope" note,
decision `0017`, and this pointer — and never once as a `### NAME-<n> ... (todo, P<n>)` heading.

## Decision

**When a document points forward at work, the work gets an ID first; the pointer cites the ID.**

Concretely, for DOC-3: the five stale lines are corrected with their evidence, and the destination
of the sixth is filed as **ENT-9 — should abac be the default?**, parked `blocked-idle-machine`
because deciding it needs two wall-clock figures that cannot be taken on a machine shared by eight
sessions.

## Why

A prose pointer has nothing that maintains it. `scripts/claim` maintains headings: an item's status
is visible to every session, `next` stops offering it when it is held, and it drops out of the queue
when it is marked DONE. An ID is the only part of this repository that another session can see
without reading prose — which is exactly COORD-2's finding (`0050`) one level up: there, an id had
to come *out* of the compare-and-swap rather than be checked against a table; here, a priority has
to point *at* an id rather than describe the work.

The cost of not doing it is not cosmetic. For two days the queue could not offer the largest built
compression lever in the codec, because it had no heading, while sessions picking "the top free
item" were handed P3 documentation work.

## What was not chosen

- **Reword item 1 without filing ENT-9.** Cheapest, and it is what produced the defect: the
  sentence would go stale again the moment the abac question moved, and `next` would still be unable
  to offer it.
- **File ENT-9 as open at P1.** Its value argues for it — larger than everything shipped to date
  put together. Refused because it is gated on `abac_bench` on an idle GPU and on a re-take of the
  1.69x decode debt (the same abac decode has read 25.2 / 31.1 / 37.5 ms across three runs under
  load). An open P1 nobody can execute removes a slot from seven other sessions, which is what
  parking exists to prevent — cf. ENT-8 step 2, MEAS-6, MEAS-5's Claim B.
- **Delete the priority order and let `scripts/claim items` be the only ranking.** Tempting, and
  wrong: the queue ranks by P-number, and the P-numbers are *set* by the reasoning in that section.
  Deleting it would leave the ordering unexplained rather than current.
- **A decision record per stale line.** Five corrections with no judgement in them; the judgement
  is the rule above, and it is one record.

## Consequence

DOC-1 (README), DOC-2 (GOALS), DOC-3 (BACKLOG) have now each found the same class of defect in a
different file, and all three were found by reading, not by measuring. The rule above is the first
one of the three that is preventive rather than corrective.
