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
  `scripts/claim take --as blocked-<reason> <ITEM> "why"`. **`--as` will refuse a value whose
  session part it cannot evaluate** (COORD-7, `docs/decisions/0071`): either name no session at
  all — `blocked-<reason>`, a reason rather than a directory — or name one that can be tested,
  `s<pid>` for a handover or `s?` when it is genuinely unknown. A session-*shaped* value the
  liveness test cannot parse is how three claims became permanently untestable.
  `CANARY-1` and `MEAS-5` are parked
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
carried **two** files numbered `0018` — `0018-gnc-is-broad-on-purpose.md` and
`0018-the-entropy-coders-are-level-and-0015s-prediction-was-wrong.md` — plus an 0020 that had to
be renumbered from 0018 by hand. Same mechanism as the MEAS-9 collision: two sessions read
`ls docs/decisions/`, both computed "next is 0018", and neither could see the other. **All four
live pairs were renumbered on 2026-09-08 (BUG-19, `docs/decisions/0059`); the numbers below are
the collision as it happened, not the current state of `docs/decisions/`.** The lock
already handles this — **do not pick a number**. `scripts/claim dr` (and `claim bug`) walk
committed `main` *and* live `refs/claims/*` and CAS the first gap, so the id comes *out* of
the compare-and-swap (COORD-2, `docs/decisions/0050`):

```bash
scripts/claim dr "the pick is the lock"    # prints dr-NNNN; that is the reservation
scripts/claim bug "why, briefly"          # prints BUG-N
scripts/claim id ENT "why, briefly"       # prints ENT-N -- any item prefix, same CAS
```

**And item ids collided the same way on 2026-09-08, which is what `claim id` is for.** COORD-2 gave
`BUG-N` and `dr-NNNN` an allocator and left every other prefix — `ENT-`, `PAD-`, `MEAS-`, `TILE-` —
to be picked by hand off `ls`-equivalent reading. Two sessions filed **two different `### ENT-9`
headings sixteen minutes apart**, and `refs/claims/ENT-9` can only lock one of them: the claim note
on the ref says "ENT-9 filed", and there is no way to tell from the ref which heading that meant.
`scripts/claim id <PREFIX> "<why>"` allocates the first free `PREFIX-N` over committed `main` plus
live `refs/claims/*` and CAS-reserves it in the same step; `claim bug` is now a shorthand for
`claim id BUG`. **Anything mentioned in committed BACKLOG counts as taken**, not only a heading,
because every one of these collisions was "someone filed it and I could not see it".
COORD-3, `docs/decisions/0065`.

`claim items` and `claim next` now **warn when two startable headings share an id**. Nothing can be
double-granted — there is still exactly one ref — so the damage is quieter than that: whoever
claims first makes the other item **invisible to `next`**, and it stays invisible until someone
renumbers it. If you see that warning, renumber the *unclaimed* one (`claim id <PREFIX>`), and
leave a held id where it is.

`take dr-NNNN` still exists for a number you already hold. It is not how you get one. Drop
the claim once the record is merged.

**`0024` is now colliding too** — `0024-the-gpu-abac-encoder-counts-before-it-writes.md` (ENT-5)
and `0024-the-jpeg-2000-gap-is-upstream-of-the-entropy-coder.md` (INTRA-1), both on `main`, both
referenced from BACKLOG and a commit message. Third pair, same afternoon, same mechanism. **Fixed
2026-09-08 by BUG-19: the ENT-5 record is now `0057` and INTRA-1 keeps `0024`.**

**And this pair is the first one that happened *after* the reserve-first rule was written, which
says something the other three could not.** The INTRA-1 session held `dr-0024` through
`scripts/claim` before writing its file; the other did not. The rule works — it just has to be
used, and one session skipping it is enough to produce the collision, because a reservation only
excludes people who ask. **CLAUDE.md also cited "`0024`" for abac's encoder judgement calls, and
that reference was ambiguous for a day; it now says `0057`.** BUG-19 moved the unreserved record,
which is the rule this paragraph argues for.

**`0027` then collided the same way on 2026-09-08** — `0027-a-simple-perf-fix-is-one-whose-win-is-a-count.md`
and `0027-cross-tile-rate-allocation-is-worth-one-point-not-ten.md`. Same asymmetry as `0024`: the
INTRA-1 session held `dr-0027` through `scripts/claim` before writing, the other did not. That is
two pairs in two days where the reserved file was not the one at fault, which is about as clear as
the evidence for the rule is going to get.

**But do not read the rule as sufficient — later the same day it produced a wrong number while
being followed correctly.** A live session held `dr-0029` through `scripts/claim` while
`0029-bug-25-was-one-defect-and-the-second-was-never-reachable.md` was already on `main`, merged in
`fcc3d33` hours earlier. No race, no second reserver, nothing worktree-local: `take dr-NNNN` is a
CAS on `refs/claims/dr-NNNN`, and **nothing in it ever reads `docs/decisions/`**, so a number that
has been committed all afternoon reserves cleanly. Reserving is what handed out the collision.

**And the second half of it, from the session that hit it:** its worktree was based on `1437d72`,
so `docs/decisions/` *in that worktree* listed only up to `0028`. The BUG-25 record landed after
its base. **Reserving from a stale base is indistinguishable from reserving a free number** — the
session checked, and what it checked was its own working tree.

So the rule earns its keep against *other sessions* and does nothing against the *namespace*, and
`ls docs/decisions/` is the wrong oracle for the namespace. **`git ls-tree` is the right one**, and
it needs no fetch and no rebase, because every worktree shares one `.git`:

```bash
git ls-tree --name-only main docs/decisions/   # what is taken on committed main, from anywhere
scripts/claim list                             # what other sessions hold but have not written
scripts/claim take dr-00NN "why"
```

`ls` answers "what is in my working tree", which is neither of those two questions. Demonstrated
both ways in `gnc-ent7bpc` on 2026-09-08: `git ls-tree main` showed `0029` while the worktree's own
`ls` did not, and later `ls` showed a `0030` that was not yet on `main`.

**The two commands cover different failures and you need both — an earlier version of this section
said `ls-tree` was "strictly better" than reserving, which is wrong and would have cost someone the
reservation.** `git ls-tree main` covers a number that is *already committed*, including one that
landed after your worktree's base. `scripts/claim list` plus taking the id covers a number that is
*held but not yet written*. Neither covers the third case: **two sessions reading "0031 is free"
inside the same minute still both get it**, which is the original `0018` race and is exactly what
`take dr-NNNN` cannot fix, because the read and the reservation are two operations. Whether these
habits would have caught the `0018`, `0024` and `0027` pairs therefore depends on whether the
winner had committed by the time the loser looked — unknown for those three, so do not read them as
evidence for either habit.

**COORD-2's build closes the third case** (`docs/decisions/0050`): `scripts/claim dr` and
`scripts/claim bug` walk `git ls-tree main` and `refs/claims/*` together and CAS the first
gap, so reading and reserving are one operation. `claim selftest` races 8 allocators of each
kind and asserts distinct ids. The `ls-tree` + `claim list` habits below remain useful as a
read of the namespace; they are no longer how you reserve.

Caught this time by reading `scripts/claim list` against `ls docs/decisions/` during unrelated
cleanup, before the file was written — luck, not process. The right-oracle correction came from the
ENT-7 session, which renumbered to `0030` before anything referenced `0029`. **COORD-2 is the
fix**; its build spec already says the free-number scan must read committed `main` *and*
`refs/claims/*`, and today's case is recorded there as the instance the interim habits cannot cover.

**Four pairs were colliding and all four are now renumbered — BUG-19, 2026-09-08,
`docs/decisions/0059`.** The later-written half of each pair moved and the earlier one kept the
number, which in `0024` and `0027` is also the half that never reserved it:

| collided | kept by | moved to |
|---|---|---|
| `0018` | ENT-2, the entropy coders are level | **`0055`** — GNC is broad on purpose |
| `0019` | COORD-1, the pick is the lock | **`0056`** — the inter path's saving was an equal-setting figure |
| `0024` | INTRA-1 step 1, the J2K gap is upstream | **`0057`** — the GPU abac encoder counts before it writes |
| `0027` | INTRA-1 step 2b, cross-tile allocation | **`0058`** — a simple perf fix is one whose win is a count |

(A fifth case, `0020`, was renumbered by hand on 2026-09-07.) Both files in every pair now carry a
note naming the other, because **a commit message cannot be renumbered** and the ambiguous
citations that predate the fix are the part that outlives it.

## Four worktrees are holding 29 uncommitted files and their sessions are gone (2026-09-08, 20:0x)

**Snapshot taken read-only, and it is the BUG-32 lesson at 4x scale, two hours in rather than
eleven.** All four hold a startable item and none has a live session behind it.

| worktree / item | base | uncommitted | unmerged commits | newest edit |
|---|---|---|---|---|
| `../gnc-tile1` — **TILE-1** (P2) | `a73e0a2` | **13**, incl. `format.rs`, `pipeline.rs`, `sequence.rs`, `rice_gpu.rs` | 0 | ~2h4m |
| `../gnc-next2` — **PERF-2** (P3) | `a73e0a2` | 8, incl. `color.rs`, `quantize.rs`, `interleave.rs` | 0 | ~2h |
| `../gnc-g41232` — **PAD-2** (P2) | `6397188` | 7, incl. `motion.rs`, `gpu_work.rs`, `checkpoint.rs` | **1** | ~2h3m |
| `../gnc-bug35rans` — **BUG-35** (P2) | `a73e0a2` | 1, `quantize_histogram_fused.wgsl` | 0 | ~2h3m |

**A fifth item is parked behind the first.** ROBUST-2 (P2) is `blocked-format-rs-in-flight` because
"gnc-tile1 holds uncommitted edits to `deserialize_compressed_validated` in 4 hunks". If that
session is gone the park's premise is void — but a park is invisible to `claim next` by design, so
nothing will notice on its own. **Whoever confirms tile1 is dead should unpark ROBUST-2 in the same
breath.**

**Before you steal any of them, read the worktree** — that rule is two sections up and it is what
this table is for. Each of these is someone's half-finished item, not a free id: inheriting beats
repeating, and `git -C "$REPO-<area>" diff` is the whole cost.

### Checking whether a holder is alive: use the socket directory, not the process table

**`pgrep -x claude` is not sound for this and will tell you a live session is dead.** Measured while
building the table above: it missed `19376`, a live session, which `ps -p 19376 -o comm` reports as
`claude`. Cross-referencing it with `lsof -d cwd` does not help either — sessions launched from the
shared checkout all report *its* path as their cwd, not their worktree's, so cwd does not identify
a worktree at all.

What is sound:

```bash
ls /tmp/cc-socks/                # one socket per live session, named by pid
```

Nine gnc sockets existed when this was written and all nine map to named sessions; none of the four
above had one. `ListAgents` agrees with the socket list, which is expected — it is the same
registry. **Caveat:** a live session that never registered a socket would look dead by this test. No
example of one has been seen, but the test is "has a socket", not "is alive", and the difference is
worth remembering before a `steal`.

This is COORD-5's subject: `claim list` now annotates a holder with its uncommitted-file count and
edit age, which is what made this table a one-liner, but it still says "verify before trusting"
without saying how — and the obvious how is wrong.

## Reserving an id is not filing the item, and a dead session takes the difference with it

**Found 2026-09-08.** `scripts/claim list` showed `BUG-32` held by a session that was gone, with
the note "id reserved before the heading exists". The heading was never written. The whole finding —
that the density harness measures SSIM throughput — existed only as 145 uncommitted lines in
`../gnc-meas5` and in that session's context, and the context is what went away. Eleven hours
passed before anyone looked in the directory.

**The lock did its job and it was not enough.** Reserving `BUG-32` correctly stopped anyone else
taking that id, which is what the reserve-first rule (see the decision-record numbers above) asks
for. But a reservation is a *promise to file*, and nothing collects on it: `list` cannot tell a
reserved id from a filed one, `next` and `items` read `main:BACKLOG.md` and simply never saw it, and
a `grep` of the tree finds nothing because the text was in a worktree nobody was standing in. Same
shape as ENT-7 being filed twice — **an item that is not on `main` is invisible to the lock** —
except here it was not on any branch at all.

Two habits, both cheap:

- **Commit the heading with the reservation, not after the work.** A four-line `(todo, P<n>)` stub
  saying what the id is for costs one commit and makes the finding survive the session. Fill it in
  later; `next` can offer it either way.
- **When you see `SESSION GONE`, look in the worktree before you steal the item.** `git -C
  "$REPO-<area>" status --short` is one command and it is the difference between inheriting the
  work and repeating it. Three of the rows in the table above were written by sessions that are
  gone; that is normal here, so this check belongs in the takeover, not in an incident.

**And the same gap makes sessions file the same finding repeatedly. Three did, in one hour, on
2026-09-08.** Two `### ENT-9` headings went live on `main` minutes apart; the owner claimed
**COORD-3** to repair it and held it for four minutes with no heading, so a second session filed
**BUG-41** for the same thing, and a third filed **BUG-42** for it *after* COORD-3's stub was on
`main`.

The third one is the instructive one, and `scripts/claim bug` is not what failed: it walks
`main:BACKLOG.md` and `refs/claims/*` for a free **id**, and BUG-42 *was* a free id. **Nothing
anywhere compares the subject.** The third session had also branched its worktree before the stub
landed, so its own `BACKLOG.md` did not contain COORD-3 — the same **stale base** that handed out
`dr-0029` above, one namespace over. So before filing anything, two reads, neither of which is
your working tree:

```bash
git show main:BACKLOG.md | grep -i '<the subject, not the id>'   # already filed on committed main?
scripts/claim list                                               # held with no heading yet?
```

The second is the one both misses had in common: **a held item whose heading does not exist yet is
invisible to every read except `claim list`** — not to `grep`, not to `next`, not to `items`.

**COORD-3 shipped the id half the same afternoon** (`docs/decisions/0065`): `scripts/claim` now
allocates item ids through the same CAS as `dr` and `bug`, warns on duplicate startable ids, and
`ENT-9` is renumbered to `ENT-10`. The two reads above remain the *subject* half, which no
allocator can cover.

## Two sessions cannot merge in the shared checkout at the same time

**Found 2026-09-08, by causing it.** The `intrasym` session ran `git merge --no-ff intrasym` in the
shared checkout while the `ent7bpc` session had a merge **already in progress there** with a
conflicted `RESEARCH_LOG.md` and its RATE-2 changes staged. Git refused the second merge —
`fatal: Exiting because of an unresolved conflict` — which reads exactly like a conflict in *your
own* merge. It was not: it was a refusal to start. The reflex that follows, `git merge --abort`,
then **threw away the other session's merge state.**

**Nothing was lost, and that is the part worth knowing before anyone panics.** An abort discards a
merge in progress, never commits: RATE-2's work, `0036` included, was still on `ent7bpc` at
`7c3768d`, and redoing the merge is one command. But the other session's `--abort`-shaped hole
appears with no explanation, and the abort was performed by someone who had no idea a merge was
underway. Checked afterwards by the `intra1gap` session: the shared checkout is clean, no
`MERGE_HEAD`, and every merge already on `main` is intact — **the abort left nothing stuck behind
it**, so the only cost is one merge to redo.

**The rule this needs.** There is exactly one working tree on `main` and eight sessions merge into
it, so **look before you merge, and never abort a merge you did not start:**

```bash
git -C "$REPO" status --short          # dirty or `UU` means someone else is mid-merge — wait
ls "$REPO/.git/MERGE_HEAD" 2>/dev/null # exists = a merge is in progress and it may not be yours
```

If `MERGE_HEAD` exists and you did not create it, stop: leave it alone and come back, or merge from
your own worktree instead (`git -C "$REPO-$AREA" push` to a branch and let the owner merge). A
fast-forward (`git merge --ff-only`) touches the least and is the right shape when your branch is
already based on the current `main`.

**And read a merge refusal literally.** "Exiting because of an unresolved conflict" with no
conflict markers from *your* merge means the tree was already busy. `git status` distinguishes the
two cases in one line.

**`MERGE_HEAD` is the reliable half of that test and `git status` is only best-effort** — raised by
the `intra1gap` session, and the reason matters because it is the reason the claim lock exists.
It is *not* that a staged change is invisible: `git add` then `git status --short` prints `M  f`,
which is exactly how the aborted merge was spotted in the first place (`M  BACKLOG.md`,
`A  docs/decisions/0036…`, `UU  RESEARCH_LOG.md`). It is that **checking and merging are two
operations.** Every session works in its own worktree, so the shared tree is clean right up to the
instant a peer starts merging into it, and a clean status is a statement about the past. Same shape
as reading BACKLOG before claiming — see "Why `next`, and not read the backlog, pick, then claim".

**Better than any check: never conflict in the shared checkout at all.** From the `ent7bpc`
session, which had the aborted merge and is right that a pattern beats a check — a check only helps
the session that remembers to run it, and it still loses the race:

```bash
# in YOUR worktree: take main's changes and resolve conflicts where your gates can be re-run
git -C "$REPO-$AREA" merge main
# then in the shared checkout, where a conflict is now impossible
git -C "$REPO" merge --ff-only "$AREA"
```

**Merge `main`, not `origin/main`.** Sessions merge into the *local* `main` and push it only
occasionally, so `origin/main` lags — it was 3 merges behind local `main` while this was being
written. `git merge origin/main` therefore reports "Already up to date" while `--ff-only` in the
shared checkout still refuses, which reads as a contradiction and is not one. Everything is local
to this machine and `.git` is shared, so no fetch is needed to see another session's merge.

**`--ff-only` cannot leave a merge in progress**: it either succeeds or refuses and changes nothing,
so the failure this whole section is about becomes unreachable rather than guarded. When it refuses,
`main` has moved again — merge `origin/main` into your branch once more and retry. `ent7bpc` went
round that loop twice with main moving underneath it and both iterations were harmless, which is
the point. It also puts the conflict resolution in the tree that can re-run the suite: that session
re-ran its gates after resolving (242 passed) and only then touched `main`.

**And take a lock anyway, since `scripts/claim` already handles arbitrary ids:**

```bash
scripts/claim take merge-main "merging <branch>"   # non-zero: someone is merging, wait
# ... merge in the shared checkout ...
scripts/claim drop merge-main
```

That makes the check *be* the compare-and-swap, which the `git status` reflex cannot be. It is a
convention, not enforcement — nothing stops a session merging without it, exactly as nothing
stopped the session that skipped `take dr-0024`.

**The abort family is wider than `merge`.** Also from `intra1gap`: `git rebase --abort`,
`git cherry-pick --abort` and a bare `git checkout <branch>` do the same thing to a tree someone
else is mid-operation on. The rule generalises to **never abort or switch away from an operation
you did not start**, and `.git/` names the operation — `MERGE_HEAD`, `REBASE_HEAD`,
`rebase-merge/`, `CHERRY_PICK_HEAD`.

## Every number carries a tree, and this is the class's home

**COORD-4 priced this on 2026-09-08 and the answer changed what to do about it.** The question was
whether `scripts/claim` should stamp the commit a measurement was taken on. It should not — but
only because the evidence points somewhere cheaper, not because the failure is rare. It is the
most frequent measurement failure in this repository right now.

**COORD-6 then found the mechanism that does work, and it is a subcommand rather than a rule —
see "A number carries its codec, and `gnc fingerprint` is how" below. Read that first; the pricing
below is why it took three items to get there, and the seventh instance is what made it urgent.**

**Seven instances, and one of them is the only one a claim-time stamp would have caught:**

| # | instance | shape | would `claim measured` have caught it? |
|---|---|---|---|
| 1 | **BUG-44** (2026-09-08) — 254.0039 against 0.0000, patched tree vs shipped | cross-session | **yes** |
| 2 | **PAD-1 / `0039`** (2026-09-08, `c109128`) — q=85 rows pre-INTER-2, q=92 rows post | intra-session, `main` moved | no |
| 3 | **ENT-3 / `0025`** (2026-09-08, `0045`) — nine published points, two superseded by INTER-2 | intra-session, `main` moved | no |
| 4 | **the build-artefact near-miss** (2026-09-08, section below) — a rebuild during a 36-run sweep | intra-session, own `target/` | no |
| 5 | **ARCH-3 / BUG-18** (2026-09-07) — `main` moved mid-item; rebased and re-measured | intra-session, `main` moved | no |
| 6 | **quarter-pel #15** (2026-03-09) — "−0.63 dB vs stale baseline (`617d8e6`)" | comparison against a stale record | no |
| 7 | **LOSSLESS-3 / `0070`** (2026-09-08) — its lossy columns hold bit-exact I-frames, which BUG-47 (`0072`) moved by ~1.8 points hours later while the q=100 column stayed put | intra-session, `main` moved — *between one session filing and another reading* | no |

**So the tool is refused on its own numbers: 1 of 7.** The cheaper variant — printing the commit
each *claim* was taken against — would have caught **0 of 7**, because a claim's commit is not a
measurement's commit and instance 1's difference was uncommitted anyway. COORD-4 closed on this.
**Five of the seven are the `main`-moved-under-a-table shape**, which is the one COORD-6 went after.

**Instance 7 happened *after* this section was consolidated onto `main`, and it was caught before
publication — by people, not by a mechanism.** LOSSLESS-3 had a q=95/97/99 table against a
bit-exact q=100 column concluding camera content is dominated from q=95 up; those lossy columns
contain bit-exact I-frames, so BUG-47 moved every one of them and left q=100 alone. The RATE-4
session flagged it, the RATE-3 session relayed it, and LOSSLESS-3's owner re-took the sweep on
`d10e414` before shipping. **Nothing was published wrong** — and the re-take showed the stakes were
not just a stale margin: bbb was the cell predicted to flip *toward* domination and moved the other
way, −1.9% as filed to ±0.00% with the trigger not firing at all.

So the fair reading is narrower than "the prose failed": **the class recurred, and the prose plus
one attentive peer was enough that once.** COORD-6 ships a mechanism anyway, on price — half a
second — as a **backstop for the peer chain, not a replacement for it**.

**What the six actually say is that the rule is already written four times, by four sessions, on
one afternoon, under four names — and that is why it keeps not being applied:**

- *this section* — ask which tree, when two sessions' numbers cannot coexist
- *"Do not swap a shared build artefact while someone is measuring"*, below — the same failure
  inside one session, where nothing errors and the numbers quietly come from two codecs
- *ENT-3's bullet in the merge log* — **"a figure that reproduces exactly on its own pinned commit
  and not on `main` is a change log, not an error"**, and its instruction to *pin the old commit
  before attributing*. That one is the most valuable of the four and the hardest to find.
- *PAD-1's `c109128`* — "a table whose q=85 and q=92 came from different binaries is unreadable —
  the same failure mode as a before-number and an after-number taken across a rebase"

They are one rule: **a number is incomplete without the tree it was measured on, and `main` moves
under you.** Four articulations exist because each session met the class fresh and none could see
the others' wording. Read the four together; do not write a fifth.

**The three habits they add up to**, cheapest first:

```bash
git -C "$REPO" rev-parse --short HEAD   # say this next to any number you publish or send
```

- **State the tree with the number.** `main` at `<sha>`, or "my branch with X applied, `src/`
  otherwise identical to `<sha>`". `RESEARCH_LOG` already has good examples of this done right
  (`ent2` pinned at `c0dd27f`; the Huffman-mapping gate naming both binaries and asserting `src/`
  byte-identity between them) — those are the model, and neither is one of the six.
- **A table is one binary.** If `main` moves mid-table, re-run the table, do not patch the rows.
  Instances 2 and 3 are both published tables split across a merge.
- **When two results cannot both be true, ask which tree before you file, correct, or reverse.**
  Instance 1 cost two sessions an hour and put two wrong inferences into `main`; the refuting test
  was on disk the whole time and takes twenty seconds.

**Why now, and why it is not a competence problem.** Six of the seven are from the two days this
repository has run eight concurrent sessions. Concurrency is what makes a published figure decay
between measurement and reading, and the cost scales with how many sessions merge into one `main`,
not with how careful any one of them is. A seventh instance arrived the same day and is row 7 above, which is exactly what
this sentence predicted — so consolidation was the right answer to COORD-4 and an insufficient one.
COORD-6 is the mechanism.

## A number carries its codec, and `gnc fingerprint` is how

**COORD-6, 2026-09-08. `docs/decisions/0075`.** Five of the seven instances above are one
session's table decaying because `main` moved under it. COORD-4 refused the two obvious mechanisms
on those numbers — a claim-time `HEAD` stamp catches 1 of 7, printing each claim's commit 0 of 7 —
and consolidating the prose did not prevent instance 7. So the answer had to be either a mechanism
or an explicit "no mechanism exists; this is a cost of concurrency". **It is a mechanism, and it is
half a second:**

```bash
gnc fingerprint                       # codec-fingerprint v1 700d5f8a  (10 configurations)
```

The digest is of a tree, not of the tool: it read `abf86a50` while `0075` was being written and
`700d5f8a` one merge later, because ENT-9 step 2 moved abac's output. **Quoting a digest dates the
quote, which is the point.**

It encodes a **pinned, versioned** 10-configuration matrix and digests the bytes, so it answers the
only question that matters — *would this binary produce different output?* — by running the encoder
rather than guessing from the diff. **Two numbers carrying the same fingerprint are comparable; two
carrying different ones are not.**

**Why not `shasum target/release/gnc`, which several harnesses already print.** It changes when a
doc comment does. Measured while building this: adding a whole module and editing `main.rs` moved
the binary hash from `94f25712…` to `333e2c62…` and left the fingerprint at `abf86a50`, because the
encoder's output had not moved. A check that fires on every rebuild is a check nobody reads.

Validated against knobs whose effect was already measured elsewhere:

| knob | does output move? | fingerprint |
|---|---|---|
| `GNC_REF_FROM_SOURCE=0` | **no** — 24 of 24 sequence points byte-identical (`0072`) | **unchanged** |
| `GNC_PAD_FILL=decay` | yes | changed |
| `GNC_DEAD_ZONE=0.3` | yes | changed |
| `GNC_REF_DEBLOCK=1` | yes | changed |

**Two uses, and a harness wants both.** Print it beside the numbers so a later reader can tell
whether a published table is still comparable — that is instances 2, 3, 5, 6 and 7. And take it
before *and* after a sweep, so a mid-run rebuild is a refusal rather than a plausible table — that
is instance 4, the near-miss that this repository got away with by luck.

```python
import fingerprint                       # scripts/fingerprint.py
fp = fingerprint.read(GNC)               # print it with the numbers
...                                      # the sweep
fingerprint.check_unchanged(GNC, fp)     # exits if the binary was rebuilt underneath it
```

```bash
FP=$(gnc fingerprint 2>/dev/null | grep -o 'v1 [0-9a-f]*')      # a shell loop needs no harness
... your loop ...
[ "$FP" = "$(gnc fingerprint 2>/dev/null | grep -o 'v1 [0-9a-f]*')" ] || echo "REBUILT MID-RUN"
```

`scripts/meas_rate4.py` and `scripts/rate4_ref_source_gate.py` do both already; copy from either.

**It is a backstop, not a substitute for asking.** Instance 7 was caught by a peer noticing that a
fix moved someone else's inputs, hours before any mechanism would have been consulted. Keep doing
that; this just means a table that slips through still says which encoder produced it.

**What it does not do, stated because a check believed past its range is worse than none.** It
cannot say *why* two fingerprints differ. It says nothing about a path outside its matrix — the
matrix crosses entropy coder, chroma format, the lossless boundary and inter, and it is not
exhaustive. It protects only numbers that carry it, which is the same adoption problem the prose
has, met with one command instead of a paragraph. And **the matrix is pinned**: changing it changes
every fingerprint ever published, so it carries `MATRIX_VERSION` and a change to it is a decision
record, not a commit.

### The worked example, kept because it shows all three habits failing at once

**Found 2026-09-08, by both sides of it.** A RATE-4 measurement said the encoder's reference and
the decoder's differ by 254.0039 at `q=100`. The BUG-39 session had just shown `q=100` video
decoding bit-exact on 48 of 48 frames, which cannot be true of a codec predicting from a reference
the decoder does not hold. Two results, both correctly measured, apparently contradictory.

**They were taken on different trees.** The 254.0039 came from a *patched* encoder — a source-copy
reference under test — so the encoder side of the diff was a colour-converted source plane rather
than a reference. The shipped tree reads 0.0000 on the same instrument, same content. There was no
bug in either measurement and no bug in the instrument.

**The cost was an hour of writing and two rounds of corrections into `main`**, including a
conclusion inverted and then re-inverted in the same entry — because each side reasoned about the
other's number instead of asking one question:

> **"Which commit was that measured on?"**

Both cheap fixes went unreached. One side had a twenty-second test that settles it
(`cargo test --release --lib <the oracle> -- --test-threads=1`); the other had one line of
`git diff` they never asked for. Neither is expensive; both were skipped in favour of an
explanation.

**So: a number quoted across sessions is incomplete without its tree**, exactly as a claim is not a
measurement. When you send one, say what it was measured on — `main` at `<sha>`, or "my branch with
X applied". When you receive one that cannot coexist with yours, ask that before you file a bug,
write a correction, or reverse a conclusion. And **never invert your own measured result on someone
else's inference** — an inference is not a measurement no matter how good the reasoning is, and the
session that inverted here had the refuting test on disk the whole time.

Filed and closed the same hour as **BUG-44** (not-a-bug); the worked example lives in BACKLOG's
RATE-4 entry, which carries both corrections struck in place rather than deleted.

## A subagent's load is your load, and it is transitive

**Found 2026-09-08, by causing it.** A read-only literature agent was spawned to research how other
wavelet codecs handle non-multiple-of-tile dimensions. It spawned children of its own, and one of
those went past reading and **ran x264/x265/libaom encodes on this machine for about 30 minutes**
while seven-plus sessions were live. CPU, not GPU, but rule 1 is about concurrent load full stop.

**Cost this time: nothing, and that was luck rather than judgement.** Two sessions checked
independently — all six commits that landed in the window carry only dB and BD-rate figures, no
fps, ms or wall clock, and the one timing figure published today predates the window by two hours.
Rate and quality are deterministic under load; only throughput is not.

**The lesson is the level at which it failed.** CLAUDE.md sanctions "read-only fan-out where the
answer is a conclusion and the cost is reading", and the agent *was* scoped that way — the prompt
said no files, no repo changes. **The child was not.** A prompt describing the parent's intent does
not constrain a grandchild, and nothing in rule 1 mentions transitive load. So:

- **Say "do not execute anything" in as many words**, not "no code changes" or "read only". The
  second is about the repository; the first is about the machine.
- **Assume a spawned agent may spawn**, and that your scoping does not inherit. If the task could
  tempt anyone downstream into measuring, say that measuring is not part of it.
- **A literature question is the tempting case**, precisely because "how much does this cost?" is
  answerable by running an encoder, and an agent asked for a number will find a way to get one.

Another session reports having spawned two read-only literature agents today, neither scoped
against execution, and that neither spawned children — same trap, one prompt away.

## The two test-material `.y4m` files are not regenerable by the fetch script

Also 2026-09-08, while checking whether a broad `rm -f *.y4m` in a scratchpad had cost anything.
It had not, but the check turned up a risk class the fetch script does not advertise.

`test_material/frames/sequences/` holds exactly two y4m files — `bbb/bbb.y4m` and
`blue_sky/blue_sky.y4m`, 24 883 308 B each — and **`test_material/fetch_test_frames.sh` cannot
recreate them.** It only *streams* y4m to extract a single frame, and says so itself: "avoids
downloading multi-GB files". So those two came from somewhere else, and if they go, someone
re-downloads multi-GB sources from Xiph by hand.

Everything else under `sequences/` is PNG frames, which the script does fetch. So a `.y4m` under
`test_material/` is expensive to lose and one in a scratchpad is cheap — **but "cheap" was too
quick, and the correction came from the session that fixed BUG-36.**

**`--vmaf` writes its inputs as `.y4m` straight into `TMPDIR`.** BUG-36's fix pid-stamps them
(`gnc_ip_vmaf_ref_p<pid>.y4m` and friends) so concurrent runs stop scoring each other's frames —
but **pid-stamping defends against collision and not at all against a wildcard delete.** A
`rm -f *.y4m` in `TMPDIR` takes live VMAF inputs out from under any session mid-run, and
`rd-curve --vmaf` has by far the widest exposure because it holds its Y4M across every quality
point.

The one reassurance is that this class fails **loudly**, unlike BUG-36 itself: the files are
written, flushed and handed to the `vmaf` binary, so deleting them in that gap makes vmaf error out
and GNC print `failed (is vmaf in PATH?)`. The score comes back *empty* rather than
plausible-and-wrong. So check for blank VMAF cells rather than for implausible ones.

## Do not swap a shared build artefact while someone is measuring

Same day, and adjacent to the subagent scoping above rather than to it. One session stopped itself
rebuilding `target/release/gnc` while a 36-run sweep was reading that binary — **which would have
mixed two encoders across the rungs of one ladder and produced a perfectly plausible BD-rate.**
Worktrees have their own `target/`, so this is a hazard *within* a session, not between them, and
it is the one that does not announce itself: nothing errors, the numbers just quietly come from two
different codecs.

I nearly did the same and cannot fully rule out that I did: `cargo test --release` rebuilds, and I
launched a gate run concurrently with a measurement run earlier today. What saves that particular
case is evidence rather than care — the still-image byte count at q=90 is identical across every
commit `main` moved through today (1 793 794 B with `GNC_PAD_FILL=replicate`), so a binary swap
mid-run could not have changed those figures. **The lesson is to serialise anyway**, because next
time the intervening change will not be one that leaves the output byte-identical.

## Builds queue on one lock, and that looks like a hang

Each worktree has its own `target/`, so builds no longer block on each other's **target** lock —
that alone is worth the disk. It also used to mean each worktree compiled the same 264
dependency crates from scratch; a shared sccache now covers that — see "Builds share one
compilation cache" below. **They do still block on the package-cache lock**, and that
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
| `../gnc-loopa` | `loopa` | **BUG-20 FIXED 2026-09-08 — the native clippy gate is `--all-targets` and the 91 warnings are cleared, not exempted.** `cargo clippy --release` reads the lib and the bins and never a test; `--all-targets` reported **91** (90 lib-test + 1 `tests/requested_limits.rs`), 88 on 2026-09-07 and 90 later that day, so the count drifts on its own. Now **0**, with no `#[allow]` added at any level. **Two of the eight lints were substantive**: `assertions_on_constants` was BUG-35's guard test asserting relations between three `const usize` values at *run* time (now `const _: () = assert!(…)`, so an arena shrink fails the build), and `unused_variables` found a dead `BufferUsages` binding in `rice_gpu.rs`. The other 89 are style, and the 27 `needless_range_loop` are the honest case for the alternative — exempting tests — which lost because there is no CI here, so step 5's clippy command is the only thing that reads this code mechanically. Decision `0062`. **Invalidates no measurement**: every edit is inside `#[cfg(test)]` code or an integration test target — nine of the eleven `src/` files have their first changed line below their own `#[cfg(test)]` marker, and the other two *are* test files (`{encoder,decoder}/pipeline_tests.rs`, included only under `#[cfg(test)]`) — so the shipped build is unchanged by construction. Filed **BUG-38** on the way — `cargo fmt --check` is red the same way and worse (566 diffs, 61 files, **504 of them in 44 files under `src/`**), heading committed with the reserved id. **ENT-9 DONE** (`0074`): abac context-codes the Exp-Golomb unary prefix — bitstream **GP19**, GP18 abac frames refused. **−2.07% to −8.76% of total rate at q=99** at bit-identical pixels (98/98 GPU-vs-CPU identity, 6/6 pixel arms, workgroup storage 6400 → 9472 B of 16384). Every figure lands just under `0063`'s bound by 0.16–0.37 points. BASELINE's `--abac` row annotated conservative; re-take is **MEAS-11**. **COORD-7 FIXED** (`0071`): the `#g…` identity came through `--as` — no committed version of `scripts/claim` can emit a `g` prefix — so `--as` now refuses a session part `session_alive` cannot read; `CLAUDE_PID` (verified against `ps`) is preferred over the twelve-hop walk; an `s?` claim records `walk: claude-pid=… chain: …` so the next one is a reading; `basename --` at four sites, found by reading the instrument's own output. **COORD-5 FIXED** (`0069`): `claim list` reports the holder's worktree when liveness cannot be established — at 18:58, with `next` reporting all 15 startable items claimed, **4 could not be tested** (`s?` x3 and one `g01a08196` that `me()` cannot produce), all idle ~67m holding 1/7/13/8 uncommitted files. Cause left to **COORD-7** with an instrument rather than a guess; "cannot say" is deliberately not collapsed into `SESSION GONE`. New `selftest` case mutation-tested. **BUG-38 DECIDED, reformat parked** (`0066`): no rustfmt config fits (default is best of seven at **573** diffs; `"Max"` 1114, `max_width = 90` 964), **44 of the 61 dirty files were changed on `main` in 24 h**, so both the big-bang and the per-touched-file rule cost the same conflicts and the cold subset is only 10%. Rule kept, one atomic `cargo fmt` commit owed on a quiet tree with its sha in `.git-blame-ignore-revs`. Also filed and closed **BUG-42** in the same hour: the *third* filing of the ENT-9 duplicate-id finding after BUG-41 and COORD-3 (which then shipped, `0065`), from a worktree branched before COORD-3's stub landed — `claim bug` gives a free id and nothing compares the subject. See the note above the shared-checkout merge section. |

| `../gnc-tile1` | `tile1` (pushed) | **TILE-1 in progress, inherited 2026-09-08 from a gone session.** Its 13 uncommitted files are committed as found (`1878bf6`), `main` merged, and the format renumbered **GP19 → GP20** (BUG-51 — ENT-9 already owned GP19; the merge auto-merged both clean because the writer line is textually identical). **Do not merge this branch:** the entropy path derives the plane extent as `padded_w * (tiles_y * tile_size)` while the buffer is `padded_w * padded_h`, so all three coders break on partial border tiles — Rice 5.75 dB, rANS and abac panic. Diagnosis with controls is on `main` in the TILE-1 entry. Next step is extent-aware tile addressing in the coders and their shaders. |
| `../gnc-refdiff` | `refdiff` | **RATE-4 half done, dropped 2026-09-08.** The free half — `0040` point 4's source-copy reference — is **refuted by a direct buffer diff** rather than by 0040's confounded PSNR: 0.0000 in RATE-3's q=95..99 fallback case, **254.0039** at q=100 MED, 7.3965 at q=100 lossless wavelet. Encoder's source planes are fractional where the decoder's reference is integral, and identical between the MED and wavelet runs, so it is not the transform. Reverted; tree unchanged. **The unexplained half is why the fallback case matches exactly** — start there. The other half (choose the candidate on sequence bytes, which is what makes bbb q=99 regress) is untouched. |
