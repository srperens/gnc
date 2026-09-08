# 0059 — When two records share a number, the half that moves is the half that did not reserve it

**Date:** 2026-09-08
**Status:** accepted
**Item:** BUG-19 (P3)
**Scope:** `docs/decisions/` numbering and the citations into it. No effect on the codec, the
bitstream, or any measurement — this record and the four renames it describes touch prose and one
doc comment.

## The decision

Four pairs of decision records shared a number on `main`. The **later-written** half of each pair
was renumbered and the **earlier** half kept the number:

| collided | kept by | moved to |
|---|---|---|
| `0018` | ENT-2 — the entropy coders are level above q=25 | **`0055`** — GNC is broad on purpose |
| `0019` | COORD-1 — the pick is the lock | **`0056`** — the inter path's saving was an equal-setting figure |
| `0024` | INTRA-1 step 1 — the J2K gap is upstream of the coder | **`0057`** — the GPU abac encoder counts before it writes |
| `0027` | INTRA-1 step 2b — cross-tile allocation is worth one point | **`0058`** — a simple perf fix is one whose win is a count |

The new numbers came out of `scripts/claim dr` (`docs/decisions/0050`), reserved before a single
file was touched, so this fix could not itself collide with the seven other live sessions.

## Why *that* rule, and what the alternatives would have cost

Two candidate rules were available and **they agree on every pair here**, which is what makes the
choice cheap:

- **"The half that did not reserve the number moves."** COORDINATION.md already argued for this on
  the `0024` and `0027` pairs: the INTRA-1 session held `dr-0024` and `dr-0027` through
  `scripts/claim` before writing, and the other session did not. It is the rule with an incentive
  attached — reserving buys you the number.
- **"The later add-commit moves."** Decidable for all four, including `0018` and `0019` where
  *nobody* reserved: 19:59 against 20:00, and 20:23 against 20:35. It is the rule that always has
  an answer.

So the shipped rule is the first where a reservation exists and the second where none does. **The
reservation rule alone is not sufficient** — half the pairs predate the reserve-first convention —
and the timestamp rule alone would have thrown away the only lever the project has for making
sessions reserve.

**Rejected: renumber by citation count** — move whichever record is cited less, so fewer
references need editing. It is the cheapest edit and the worst rule: it makes the number depend on
how much prose an item happened to generate, it changes answer as citations accumulate, and it
rewards the session that skipped the lock whenever its record is the busier one. It would in fact
have moved the *same* four files here, since the keepers are the more-cited half in every pair —
which is precisely why it is not worth adopting: it buys nothing and costs the incentive.

**Rejected: leave the numbers and disambiguate by slug** — cite `0024-the-jpeg-2000-gap…` in full
everywhere. It is not wrong, and it is what the two 0018s survived on for a day; the reason not to
is that citations in this repository are overwhelmingly bare — `` `0024` ``, "decision 0019",
"decision record 0018" — and a convention nobody follows in the file it is written in is not a
convention. The four colliding numbers appear about 170 times across the tree; the handful written
as a full filename are the exception.

## What this fix cannot reach, and what stands in for it

**Commit messages cite the old numbers and are not being rewritten.** `192267c` announces the ENT-5
record as 0024; four merges say the same. Rewriting history to fix a citation is a far larger risk
than the ambiguity it removes, and CLAUDE.md's own rule about git history being the expensive place
points the same way.

So **both files in every pair now carry a note naming the other**, with the dates the collision was
live. That is what makes a citation written on 2026-09-07 or 2026-09-08 resolvable: a reader
following `0024` lands on INTRA-1's record and is told, in the header, that ENT-5's is now `0057`.
The renamed files carry the mirror note. This is the part of the fix that outlives it, and it is the
part the BACKLOG item did not ask for.

## The evidence that ambiguity costs more than renaming

The item asked for `0020` to be checked for stale citations, because `0020` had been renumbered from
`0018` by hand a day earlier. Two were found, both in RESEARCH_LOG's MEAS-9 entry, and both are
**content** errors rather than renumbering ones:

> This lands the same day as decision 0020 (GNC is broad on purpose) …
> … the obvious successor to this item, especially under decision 0020.

`0020` is the withdrawal of GNC's colour lead over x264. "GNC is broad on purpose" was `0018` when
that entry was written and is `0055` now. The MEAS-9 session read a directory that was being
renumbered underneath it and cited the number that had just been vacated. **Nobody noticed for a
day**, because a citation to a wrong-but-existing record reads exactly like a correct one — which
is the whole argument for `scripts/claim dr` and for the notes above.

## What this does not change

- **No measurement moves.** Four renames, 31 repointed citation sites, one doc comment in
  `src/encoder/entropy_helpers.rs`, and two corrected mis-citations. `cargo test --release` and both
  clippy targets were run to confirm exactly that.
- **The second half of BUG-19 was already closed.** It asked whether `claim` should hand out `dr-`
  numbers; COORD-2 built it (`0050`) and this item is the first to use it in anger.
- **`0022`, `0043`, `0050` and the collision narratives in COORDINATION.md and BACKLOG.md keep
  their old numbers in prose**, because they are describing the collision as it happened. Where they
  asserted the pairs were *unfixed*, they now say what moved where.
