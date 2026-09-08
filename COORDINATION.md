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
  `scripts/claim take --as blocked-<reason> <ITEM> "why"`. `CANARY-1` and `MEAS-5` are parked
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
currently carries **two** files numbered `0018` — `0018-gnc-is-broad-on-purpose.md` and
`0018-the-entropy-coders-are-level-and-0015s-prediction-was-wrong.md` — plus an 0020 that had to
be renumbered from 0018 by hand. Same mechanism as the MEAS-9 collision: two sessions read
`ls docs/decisions/`, both computed "next is 0018", and neither could see the other. The lock
already handles this — **do not pick a number**. `scripts/claim dr` (and `claim bug`) walk
committed `main` *and* live `refs/claims/*` and CAS the first gap, so the id comes *out* of
the compare-and-swap (COORD-2, `docs/decisions/0050`):

```bash
scripts/claim dr "the pick is the lock"    # prints dr-NNNN; that is the reservation
scripts/claim bug "why, briefly"          # prints BUG-N
```

`take dr-NNNN` still exists for a number you already hold. It is not how you get one. Drop
the claim once the record is merged.

**`0024` is now colliding too** — `0024-the-gpu-abac-encoder-counts-before-it-writes.md` (ENT-5)
and `0024-the-jpeg-2000-gap-is-upstream-of-the-entropy-coder.md` (INTRA-1), both on `main`, both
referenced from BACKLOG and a commit message. Third pair, same afternoon, same mechanism. Noted
rather than fixed: renumbering is BUG-19's job and the inbound references are where it goes wrong.

**And this pair is the first one that happened *after* the reserve-first rule was written, which
says something the other three could not.** The INTRA-1 session held `dr-0024` through
`scripts/claim` before writing its file; the other did not. The rule works — it just has to be
used, and one session skipping it is enough to produce the collision, because a reservation only
excludes people who ask. **CLAUDE.md also cites "`0024`" for abac's encoder judgement calls, and
that reference is now ambiguous.** When BUG-19 gets picked up, the unreserved record is the one that
should move.

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

**Four more records are colliding and nobody has renumbered them: `0018` twice and `0019`
twice** (a fifth case, `0020`, was renumbered by hand). Filed as **BUG-19** — the fix is
mechanical but the inbound references are where it goes wrong, so it is a claimable item rather
than something to do in passing.

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
| `../gnc-refdiff` | `refdiff` | **RATE-4 half done, dropped 2026-09-08.** The free half — `0040` point 4's source-copy reference — is **refuted by a direct buffer diff** rather than by 0040's confounded PSNR: 0.0000 in RATE-3's q=95..99 fallback case, **254.0039** at q=100 MED, 7.3965 at q=100 lossless wavelet. Encoder's source planes are fractional where the decoder's reference is integral, and identical between the MED and wavelet runs, so it is not the transform. Reverted; tree unchanged. **The unexplained half is why the fallback case matches exactly** — start there. The other half (choose the candidate on sequence bytes, which is what makes bbb q=99 regress) is untouched. |
| `../gnc-refdiff` | `refdiff` | **RATE-3 DONE 2026-09-08.** `0036`'s sequence gate lifted; mean **−4.28%** of sequence bytes (3 sequences × q ∈ {95,99} × ki ∈ {2,9}), best −13.16%, worst ΔP −0.01 dB, I-frames bit-exact through a real `encode-sequence` → `decode-sequence` md5 round trip. The gate was hiding the *mirror image* of `0040`'s bug: `encode_once` leaves only the **last** candidate's quantised planes in the side channel `local_decode_iframe_gpu` reads, so a kept *bit-exact* frame got the lossy candidate's — P-frames at 5.93 dB. `encode_as_reference` re-runs whichever was kept, at a third encode on those frames. Stills byte-identical. bbb q=99 regresses +0.4/+0.58% → **RATE-4** (with `0040` point 4's source-copy reference, whose refutation is confounded by BUG-39 cause 2). Decision `0044`. |
