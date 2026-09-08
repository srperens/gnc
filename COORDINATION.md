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
already handles this — **reserve the number before you write the file**:

```bash
scripts/claim take dr-0019 "the pick is the lock"   # non-zero: someone has it, use the next one
```

Reserving costs one command and is the difference between picking a number and being *given*
one. Drop it once the record is merged.

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

**Only COORD-2's build closes the third case**, and its title is the design: the number must come
*out* of the compare-and-swap rather than be checked against it — walk `git ls-tree main
docs/decisions/` and `refs/claims/dr-*` together and CAS the first gap, so reading and reserving
are one operation. Until then these two commands are mitigation, not a fix.

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
git -C "$REPO-$AREA" fetch origin && git -C "$REPO-$AREA" merge origin/main
# then in the shared checkout, where a conflict is now impossible
git -C "$REPO" merge --ff-only "$AREA"
```

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
| `../gnc-abac`, `.claude/worktrees/abac` (`abac-gate`) | `abac` | **released — question answered, see BACKLOG Part 6.** The idle-machine bench is run. Range at cb=64 costs **1.69× frame decode for −16.7% rate** at q=90; Interval costs 3.99×. Rice's own entropy stage is 47% of frame decode, which caps any entropy work at 1.9×. What remains is a positioning call, not an engineering one. |
| `../gnc-abac` | `abac` | same worktree, now on **BUG-8** — the encoder's local decode diverges from the real decoder down a GOP. |
| `../gnc-nearlossless` | `nearlossless` | **done and merged 2026-09-07** — BUG-15 fixed, INTRA-NEARLOSSLESS closed by measurement, RATE-2 confirmed independently. Worktree removed. |
| _(removed)_ | `abacship`, `bug18` | **ABAC-SHIP merged, both worktrees gone 2026-09-07** (`60bed17`, `378a0c7`, `436680e`, `6513bb6`). abac is a real entropy coder: `--abac`, entropy type 5, **GP18**, stills *and* sequences. **Standing figures: intra −16.6% to −18.8% at identical pixels, lossless −13.4%** (FFV1 gap +23.9% → +7.3%). ~~inter −14.4%~~ **retracted** — see the retraction section above. Rice stays the default (`docs/decisions/0017`). **Every frame this encoder writes now says GP18**; a GP18 Rice frame is a GP17 payload with a new label, proved by relabelling and decoding, and GP17 still reads. **Left open and unclaimed: BUG-18 (P1)** — the CPU-entropy P-frame path encodes every P-frame wrong; cause 1 fixed, cause 2 open, `tests/bug18_locate.rs` reproduces it in one command. |
| `../gnc-rate1` | `rate1` | **DONE, merged, worktree removed.** RATE-1 answered **no** — a bit-depth-aware rate rule recovers **0.0% on all four photographic stills** (89.4% on the synthetic gradient, which is the trap). The sweep found **RATE-2 instead, filed P1**: above q≈95-98 the lossy ladder costs more bytes than bit-exact lossless on every real image, mean **+28.9% at q=99** (blue_sky +40.6%). LOSSLESS-1 made lossless cheap enough to undercut the top of the lossy ladder and nothing noticed. |
| `../gnc-intrasym` | `intrasym` | **INTRA-1 step 3 — DONE 2026-09-08. The accounting closes: the remaining ~6 points are tile-alignment padding.** GNC pads every plane to whole tiles with edge replication (`pad.wgsl`) and codes the padded plane, so a 1920x1080 frame is coded as **2048x1280 — 20.9% of the coded samples outside the picture** and the decoder crops them; J2K in whole-picture mode codes none, and both arms are divided by the visible pixel count. **Two methods, means agreeing to 0.02 points: +6.60% projected from a content-controlled crop pair (quality held to 0.005 dB), +6.58 points as the drop in the cross-codec gap from native to padding-free content on one ladder (26.76 -> 20.17 RGB, 51.09 -> 40.54 Y).** Per-image the two scatter by ±2.2 points, which is the content change from a frame to its centre crop — the cross-codec arm carries it and the content-controlled arm does not, so the former corroborates the mean and is not a per-image second opinion. **A trap found while fixing my own first version, and it is general: do not difference a BD-rate against one taken on a different q ladder.** The first Part 2 read this harness's q=80-98 crop gap against ENT-4's q=60-99 native +27.1% and called the difference padding. On this ladder bbb's native gap is +19.46% where ENT-4 reads +17.3% — **2.2 points of pure ladder artefact per image**, the size of the effect, even though the four-image means nearly coincide (26.76 vs 27.1). It gave 6.98 and "agree to 0.4": the right answer for the wrong reason, withdrawn and re-measured. **Part 1's control is measured, not argued: JPEG 2000 reads −0.13% to +0.02% between the same two crops where GNC reads +11.6% to +25.2%**, so the extra row and column of content are under 1% of the effect. **Canary: the Python-rebuilt padded plane encodes byte-identically to production** (bbb q=90 `--abac`, 1 793 794 B both ways, which is the figure BACKLOG already records). **Two thirds is a fill choice** — replicate-then-fade is −4.62% RGB at unchanged visible quality, a mirror of the picture into the padding is +11.54%, so replication was already the better textbook extension. **Not shipped: the reference buffer keeps the padding and MC reads it for edge blocks**, so the gate is an inter measurement — filed as **PAD-1 (P1)** with three shapes and a worst-frame criterion. Also re-derived, agreeing with step 2c: `transform_97.wgsl` is **already whole-point symmetric extension**, and BACKLOG's stale "replicates the edge sample" candidate is now corrected in place. **INTRA-1 is closed as a queue entry** — its success criterion ("a number per cause that sums to roughly the measured gap") is met at 26.2 of 27.1, so it loses its priority marker rather than sending the next session after 0.9 points of residue; the work that is left is PAD-1 (P1) and INTRA-2 (P1), both filed with numbers and gates. Decision `0034`; harness `scripts/meas_intra1_padding.py`. **Re-measured on `c84fbd5` after RATE-2 landed, because RATE-2 codes q=95..99 both ways and this harness's first ladder was q=80..98.** One rung of eight moved and it moved badly: the padded arms come back **bit-exact lossless**, `psnr()` returns `inf`, and a Bjontegaard fit over an infinity is a silent non-number — `bd()` now refuses a non-finite rung and the default ladder is q=80/85/90/94. On that ladder the figures move by at most 0.25 points and **the projection is identical at +6.60%** (it reads q=90, byte-identical across RATE-2), so the result does not move. **Two things for others:** RATE-2 already reclaims part of the padding tax for free above q=95 and reaches the *padded* arm first (bbb q=98: padded −5.78%, aligned unchanged), which concentrates PAD-1's value below q=95; and **any still figure here taken on a ladder reaching q>=95 before `a7273ab` is pinned to that code**, INTRA-1's own steps 1 and 2 included — not retracted, just not reproducible byte for byte today. **Invalidates no measurement** — no Rust changed; `cargo test --release`, `clippy --release` and `clippy --target wasm32-unknown-unknown --lib` (BUG-24's new form) all clean. |
| `../gnc-intra1` | `intra1` | **RELEASED 2026-09-08 — everything merged (`f1b6b68`), all claims dropped, INTRA-1 is free to take.** It is *not* finished: **~6 of the 27.1 points remain and the item's own candidate list is exhausted**, so the next step needs a new hypothesis rather than another sweep. Two things left ready to build, both fully measured: **INTRA-2 (P1)** — the dead zone on I-frames only, ~3 points — and **ENT-6 (P2)** — abac's short code-blocks in the deep subbands, ~4% of the file. Step 2c (2026-09-08): every candidate the item listed is now measured; ~6 points remain.** Tile-boundary handling is **already correct symmetric extension** (0 points — and it corrects a wrong claim I put in `0027`; the replication is in the polyphase split, which *is* WSS, verified bit-identical over 1000 signals). The **dead zone** is worth **~3 points on stills** (+0.24 to +0.38 dB RGB *and* −2.9 to −4.5% dE00 at matched rate, VMAF −0.01) but is a **worst-frame regression on 9 of 9 sequence points, up to −1.93 dB** — filed as **INTRA-2 (P1)**, not shipped, because the dead zone hits P-frame residuals and the error propagates. **BUG-30 fixed**: `GNC_DEAD_ZONE` silently defeated bit-exact lossless at q=100 (BUG-15's hole, one knob over). Decision `0028`. Step 2b (2026-09-07): cross-tile rate allocation REJECTED — a clairvoyant, free per-tile allocator saves **0.95%**, about one point of the remaining ten, and on kristensara the oracle picks the *same* q for all 15 tiles at q>=92. Uniform q is not near-optimal there, it is optimal. Decision `0027`. **BUG-26 fixed**: it was two silent defects — tile > 512 (shader workgroup array) and tile not divisible by `2^levels` (`max_wavelet_levels` uses integer division, so 260 gets 5 levels and decodes to 20.1 dB). Both now refused, not clamped; 256/512 output byte-identical. **~9 points of the gap remain and the obvious candidates are spent.** Step 2 first instalment landed 2026-09-07: the gap decomposes. 8.5 of the 27.1 RGB points is chroma allocation against an RGB metric (YCoCg-R synthesis norms 1.73/0.71/0.87, spread 2.45, against J2K's ICT at 1.15 — **not** a coding deficiency, and it also explains why the Y gap was +48.3%: it falls to +13.5% when corrected). Lifting normalisation is clean. Tiling costs *J2K* 12.4 points but GNC realises 0.6%. **~9.8 points left; next test is cross-tile rate allocation.** Decision `0026`. **Two traps recorded: a full-frame tile-size comparison is confounded by padding (+6.1% that is really −0.6%), and `--tile-size 1024` silently destroys the image (BUG-26, P1).** Step 1 answered 2026-09-07: `GNC_COEF_ENTROPY=1` prices the *shipped* abac tiles against the entropy of the coefficients they carry, and GNC spends within **7.5% of it at q >= 85**. So entropy coding can account for at most ~7.5 of the 27.1 points against JPEG 2000 9/7 and **~72% is upstream of the coder** — the item is now its step-2 branch (tiling, quantiser shape, lifting normalisation). Decision `0024`. Turned up **ENT-6** on the way: the deep subbands are one short code-block each and cost ~4% of the file. Adds one diagnostic module and two `pub(crate)` exports; **invalidates no measurement** — the encoder path is untouched and the diagnostic is env-gated and read-only. |
| _(removed)_ | `arch3` | **ARCH-3 + BUG-18 — merged 2026-09-07** (`a312d6f`, `e35d3b2`, `66c9a2e`), worktree removed. One P/B frame encoder; `gpu_entropy_encode` now picks the entropy step and nothing else. Default output **byte-identical, 54/54** against a baseline pinned at `07c01b1`, so no measurement is invalidated. `tests/bug18_locate.rs` reads 0.000 everywhere. **abac's inter figure is un-retracted: −12.0% to −22.9% against Rice at bit-identical pixels.** Also fixed, unasked: `--huffman` video, which shipped writing empty P-frame tile vectors. `docs/decisions/0025`. |
| `../gnc-inter1` | `inter1` | **INTER-1 done and merged 2026-09-08 (`ee286c2`); worktree removed.** No default changed — ki stays 9 (best of four at q=85-99, BD-rate monotone in GOP length), the P-scale taper stays with both endpoints re-justified, and inter stays a default. **Found and fixed BUG-27** (P-frame local-decode dequant used the intra qstep): live at all q <= 80, byte-identical at q >= 85 (27/27 against two pinned baselines), +1.82/+3.62 dB at q=70. Corrected MEAS-3 and decision 0019 (mean +4.6% -> −0.3%, worst-frame +19.1% -> +8.0%), and TUNE-5/TUNE-6's justifications. Filed **INTER-2** (the inter dead zone is a large unpriced lever at q=85) and **COORD-2** (ids must come *from* the compare-and-swap). Decision `0023`. |
| `../gnc-perffix` | `perffix` | **PERF-1 — DONE 2026-09-08, merged.** Verified `docs/SIMPLE_PERF_FIXES.md` and landed 5 of its 12 items. **Invalidates no measurement: 24 artefacts (4:4:4 and 4:2:0, Rice and abac, stills and sequences, encode and decode) hash identical to `origin/main` after the rebase**, which is the whole point of the item — every win is a count, not a time. Counts: frame loads per display index 2.62 → 1.00, `poll(Wait)` per I-frame 3 → 1, `queue.submit` per `encode()` 2 → 1, two dummy CfL buffers gone, 1.22 MB/frame of decode pack allocation reused. Three permanent canaries under `GNC_PROFILE`. The retracted 31.7 fps is gone from GOALS, README and POSITIONING. **Item 4 verified and deliberately not landed — PERF-2 says why** (the scan's cached-UBO fix collides with Metal's `write_buffer` staging at every site whose params vary within a submit). Rest is PERF-3. Decision `0027`. |
| `../gnc-meas5` | `meas5` | **BUG-32 filed and merged 2026-09-08 (`b9e3d9b`) — a rescue, not a measurement.** The MEAS-5 session was gone with **145 lines uncommitted in this worktree, nothing in RESEARCH_LOG, and `BUG-32` reserved through `scripts/claim` against a heading it never wrote**; this session took both claims over, reviewed the diff and filed it. The finding: `benchmark-sequence` spends **86% of its wall clock on CPU quality metrics**, so `gpu_tier_bench.py --density`'s `frames / wall` measures **how well N SSIM computations share the CPU**. It had never been run, so **nothing is retracted** — what it removes is the instrument MEAS-5 was going to use. Worked around with `--density-still` plus GPU power sampling (`utilization.gpu` reads 100% at 43 W of 130 W). Second harness defect also fixed: the `--hwenc` arm passed no GOP length, so NVENC used its own 250-frame default against whatever `-k` GNC got. **POSITIONING's M-series density table is flagged as a candidate, not retracted** — wrong date to be this code path. **No Rust, no WGSL, no bitstream; GPU suite deliberately not run** (DOC-1/ENT-7 precedent). **MEAS-5 itself is dropped back to free and still unmeasured** — it is runnable now via `--density-still` on the bench box, and the proper fix (a metrics-off flag on `benchmark-sequence`) is left in BUG-32. |
| `../gnc-intra1gap` | `intra1gap` | **MEAS-6 second pass + BUG-36 FIXED, merged 2026-09-08.** Worktree name predates the claim (`next` handed me MEAS-5, then MEAS-6, never INTRA-1). **MEAS-6: GNC's default latency is ~80 ms, not ~240 ms.** `quality_preset()` has vetoed the B-pyramid since **2026-09-06** (`src/lib.rs:1021`) — the item's own 2026-09-06 conclusion was invalidated by the change that conclusion caused, and POSITIONING, README and MEAS-6's entry carried it for two days. Default `ki=9` codes `2I+16P+0B` with zero reordering delay; `GNC_B_PYRAMID=1` codes `2I+2P+14B`, 8 frames, 160 ms. So ~80 ms sits **below** the low-latency-HEVC band (120 ms floor), not inside it, and ~240 ms describes an opt-in config. Decision `0033`. **The ~80 ms itself was NOT re-taken** — load 21-39 all session, BASELINE's 20% rule — and that is MEAS-6's cheapest remaining step; the correction does not depend on it, because the half that moved is structural. **BUG-36 FIXED — the one to know about: concurrent `--vmaf` runs scored each other's frames.** Every VMAF site wrote a *fixed* `TMPDIR` filename; eight sessions share one `TMPDIR`. Serial 97.39/95.91 bit-stable; concurrent gave **97.19 (+1.28)** and **96.37 (−1.02)**, one of the pair wrong in every run, 2-2.5x the >0.5 BLOCK threshold, silent and plausible. Now pid-stamped via `gnc::session_temp_path()`; **6/6 concurrent runs correct after, 3/6 before**; `tests/temp_path_collision.rs` blocks reintroduction. **Nothing retracted** — no run records whether another session was in its VMAF window. Rate figures immune; `rd-curve --vmaf` was the highest-risk caller. Filed **BUG-37** (`benchmark-sequence`'s `-q` is an `Option` with no default, so without it the command still codes the pyramid — all five `scripts/` harnesses pass `-q`, so nothing recorded is contaminated). **MEAS-5 corrected and parked** as `blocked-idle-machine-or-nvenc`: its "fix this first" blocker was resolved 2026-09-06, and what remains needs a discrete NVIDIA card with driver 610+/nvenc 13.1 or an idle Mac — it was being handed to every fresh session as the top P0. Flagged to ENT-3 that `0025`'s inter table does not say which frame mix produced it. |
| `../gnc-meas6` | `meas6` | **BUG-25 FIXED and BUG-33 CLOSED 2026-09-08 — GNC runs inter on Vulkan.** It was fixed by `51a9ac6` (the defect-A commit) hours before this session started, and nobody re-ran the failing path. Measured at `766196a` on the bench box: `encode-sequence` codes **1I + 2P on NVIDIA/Vulkan**, `decode-sequence` round-trips it, and **lavapipe gives byte-identical frame sizes**; `shader_probe` compiles **62 of 63** shaders (the miss is `blit.wgsl`, no `@compute` entry point). **All four recorded driver crashes were the invalid module** — the two Windows ones that read as independent confirmation were built from `f17bf1b`, which *predates* the fix by 66 deleted lines of that shader. Defect B is real in a `buffer: Restrict` module but wgpu never asks for one here (both adapters report `robustBufferAccess2`), so GNC does not reach it; `docs/bug25/` is relabelled an upstream report. Also: **defect A is upstream `gfx-rs/wgpu#7048`, closed by PR #7239**. Docs, one probe flag, one `.spvasm`, one code comment; **no codec code, no shader, no bitstream — invalidates no measurement.** Decision `0029`. **Owed: an inter throughput figure on Vulkan, now measurable for the first time; Intel Arc not re-run since the fix.** Older text follows. |
| _(older)_ | `meas6` | **BUG-25 — defect B's cause WITHDRAWN 2026-09-08; the measurements stand, the attribution does not.** Taken over from a session that was gone. `wgpu-hal` 24.0.4 picks `buffer: robust_buffer_access2 ? Unchecked : Restrict` from a *queried* `VK_EXT_robustness2` (`adapter.rs:1595`, `:1372`, `:1899`), and **both** Vulkan implementations on the bench box report the feature — the RTX per `vulkaninfo`, lavapipe since Mesa 22.2. Measured here with no GPU: the faithful reconstruction (`caps_index_restrict`) is **byte-identical** to `index_restrict_only`, already measured **pipeline OK**, and carries **0** `OpArrayLength` against **48** in every crashing config — so `docs/bug25/minimal_repro.spvasm`, which is nothing but that clamp, is probably **not a reduction of GNC's crash**. Also killed locally: `capabilities: Some([…])`, the entry's last untested candidate (byte-identical). **Defect A is upstream `gfx-rs/wgpu#7048`, closed by PR #7239** — our `switch` rewrite duplicates an existing fix. Docs, one example and one new `.spvasm` only; **no codec code, no shader, no bitstream, no measurement moved.** BUG-33 rewritten and held; the one run that settles it is a `sha256` of the module wgpu hands `vkCreateShaderModule`. Older text follows. |
| _(older)_ | `meas6` | **BUG-25 — cause found, one defect fixed, and the expensive fix ruled out.** Four candidates tested: upgrading wgpu/naga is **DEAD** (naga 30 crashes identically under wgpu's own options), removing the `let`-array construct is dead for the crash, removing the early `return` is dead, and **`buffer: Unchecked` works** — the only proven fix. So the shader does not need rewriting and the remaining question is about our dependency, filed as **BUG-33**: wgpu asks for `buffer: Restrict` on an adapter that reports `robustBufferAccess2 = true`, which its own source says should give `Unchecked`. That answer decides upstream fix vs `[patch.crates-io]` pin. **Correction to my own earlier row:** I first attributed the policy to the adapter *not* reporting robustness2; it does report it, and the conclusion now rests on elimination, not on reading wgpu-hal. Older text follows. |
| _(superseded)_ | `meas6` | **BUG-25 — cause found, one of two defects fixed** (`332aab4`, `a27e082`, `51a9ac6`, `bc742a3`). Worktree name predates the claim; `scripts/claim next` handed me BUG-25, not MEAS-6. **The item's premise was wrong**: "spirv-val passes all 62" was measured with the `naga` CLI at **30.0.1** while GNC ships **naga 24.0.0** — the validated module is not the shipped one. **Defect A, fixed:** naga 24 emitted an undeclared function-local temporary for four `let … = array<i32,8>` tables indexed dynamically; 1 of 63 shaders invalid before, 0 after, Metal output byte-identical. **Defect B, open:** the valid module still segfaults NVIDIA *and* lavapipe; reduced to **42 lines** (`docs/bug25/`) and isolated to `BoundsCheckPolicy::Restrict` on **buffers** (`OpArrayLength`/`ISub` in a branch that returns instead of reaching its merge). Not fixed — inter on Vulkan still hangs. |
| `../gnc-coord` | `coord` | **COORD-1 — the claim mechanism enforces the rules instead of restating them.** Docs and `scripts/claim` only; no codec change, invalidates no measurement. `scripts/claim next` makes the *pick* atomic, the shared-checkout and worktree preconditions are now refusals rather than prose, and the session identity bug that made `SESSION GONE` undetectable is fixed. See `docs/decisions/0019`. Claimed 2026-09-07. |
| _(removed)_ | `docfix` | **DOC-1 — DONE and merged 2026-09-08 (`6d99907`).** Five stale prose claims in the public README, four of them wrong. **Documentation only; no code, no shader, no bitstream, no measurement moved.** The numbers were already current — PERF-1 had retired 31.7 fps and BUG-29 had replaced every labelled "M1" — and all five defects were sentences carrying no figure, which is why both sweeps missed them. The one that mattered: README stated "Runs on Metal, Vulkan, DX12 and WebGPU" as fact while **Vulkan is intra-only (BUG-25 open), DX12 has never been run at all, and WASM has never been verified in a browser (BUG-31 open)**. Replaced with a **Portability, as measured** section giving the four rows and their evidence. Also: the "eight-core integrated GPU / real time" line BUG-29 missed, "three" entropy coders against the same file's "five", the "remaining 27%" that INTRA-1 had since decomposed to ~6, and `BITSTREAM_SPEC` §2.6's list of `code_blocks` callers, which ENT-5 made incomplete — checked, and the single-definition property survives because the GPU encoder calls it on the host (`abac_gpu_encode.rs:381`). **The GPU suite was deliberately not run**: a two-file `.md` diff does not justify taking the one GPU from seven sessions. |
| `../gnc-abacgpu` | `abacgpu` | **ENT-5 — abac now encodes on the GPU, one thread per code-block, and it is the CPU encoder's exact bytes** (98 of 98 whole-file comparisons; `scripts/ent5_gpu_encode_gate.sh`). **Invalidates no measurement and moves no bitstream** — byte identity is what is asserted, so ENT-4's −16.0% and the −13.4% lossless figure are unchanged by construction. Deliberately does *not* touch `use_gpu_encode` or `sequence.rs`'s frame-pipeline selection: abac routes on `gpu_entropy_encode` directly inside `encode_entropy`, so this is orthogonal to **ARCH-3** rather than racing it, and it keeps abac off the fused quantiser that BUG-16 says moves Rice's pixels. **One criterion outstanding: encode time per 1080p frame is NOT measured** — the machine had three other sessions on it, one of them running `encode-sequence`. Instrument: `cargo test --release --test abac_bench -- --ignored --nocapture --test-threads=1`. Until it runs, decision 0017's reason 2 has lost its mechanism and kept its 129 ms. See `docs/decisions/0024`. **Merged 2026-09-07 (`a2405d7`), claim dropped, worktree removed.** Rebased onto ARCH-3 after it landed — one markdown conflict, no code change, gates re-run (224 tests, identity gate 98/98). Two notes for whoever picks up the leftovers: **abac's 4:2:2 and 4:2:0 output is byte-identical between the GPU and CPU encoders**, so whatever BUG-28 is (filed as BUG-26 and renumbered — see below), it is not in abac's encoder; and ARCH-3's `inter_gpu_entropy_available()` returning `false` for abac now means "not in the batched dispatch", not "on the CPU" — its comment says so. |
| `../gnc-ent7bpc` | `ent7bpc` | **RATE-2 FIXED 2026-09-08 — the top of the lossy ladder codes both ways and keeps the smaller. `docs/decisions/0036`.** **This one DOES change output**, and only in one place: a still at q = 95..=99 may now come back **bit-exact**. Mean rate against the file the same command produced before, four stills: **−1.33% / −4.67% / −10.39% / −15.65% / −21.66% at q = 95/96/97/98/99**, with **12 of 20 points becoming bit-exact** from 52.5–60.1 dB. Below q=95, at q=100, with `--dct`, and **inside any sequence**, output is byte-identical. **No format change and no GP version** — `transform_type` is a header byte independent of the quality byte, so a q=97 file with `transform_type = 2` decodes on every existing build (verified through the shipped `gnc decode` + raw RGB md5). **Three things a later session needs.** (1) **RATE-3, filed P1:** letting the fallback reach sequence I-frames makes the P-frames referencing them decode at **9.80 dB against 60.69 dB** and *grows* the sequence 13.08 → 15.28 MB, because a MED I-frame has `wavelet_levels = 0` and the P-frame reference cannot reconstruct from it. The flag is cleared in `build_ip_config`, the three temporal-wavelet config sites in `main.rs`, **and** inside `encode_sequence` — three places rather than one because the GPU warm-up encodes call `encode` directly with the sequence config. (2) **Do not compare a BD-rate across this commit:** ladders reaching q≥95 now flatten at the top (correctly — it is the convex hull) so `bd_rate` drops those points, and QUAL-1 measured that changing a ladder's extent moves the figure 1.0 PSNR points by itself. (3) **BASELINE's 1.9x-against-H.264 caveat is updated, not lifted** — that ladder is video, this fix is intra, so it reproduces exactly; the rung to re-run it against is RATE-3. Earlier text for this worktree follows. |
| `../gnc-ent7bpc` | `ent7bpc` | **ENT-8 step 1 — the lockstep scan is affordable on abac's template, and the stripe width should be 4, not BPC-PaCo's 2.** **Invalidates no measurement** — `Scan` added to the existing read-only diagnostic, no encoder path touched, `0024`'s columns still reproduce byte-for-byte. abac does **not** inherit BPC-PaCo's free-parallelism result: their eight-neighbour template keeps 4 of 8 coded neighbours under the stripe schedule, abac's four are all causal, so the first column of each stripe loses its *left* neighbour. Measured as % of total rate against raster, mean of four stills: **k=2 +0.74%/+0.65% (q=85/90) for 32 threads per block, k=4 +0.41%/+0.37% for 16, k=8 +0.23%/+0.20% for 8.** The 1% gate is hit exactly by k=2 on 1 of 8 points, so the last doubling of threads costs as much as the first four together. Cost falls 0.55 per doubling against the 0.50 that "one column per stripe pays" predicts, which is the mechanism's own canary; the worst rows are the **LH** bands, where the left neighbour is the informative one. **Step 2 is untouched and must not start without an idle box** — it is a CPU coder variant, both shaders and an ENT-5-scale byte-exactness gate whose whole payoff is a throughput number that cannot be taken here. BUG-31 left ~9 984 B of workgroup headroom so the exchange buffer is free, but `tests/workgroup_storage_limit.rs` asserts exact sizes: add a number, never a tolerance. No decision record — nothing shipped; the width becomes a decision when step 2 does. Earlier text for this worktree follows. |
| `../gnc-wgmem35` | `wgmem35` | **BUG-35, default path done 2026-09-08 — the fused histogram was dead work.** `quantize_histogram_fused.wgsl:main` is 23800 B against a 16384 B device and, unlike BUG-31's shaders, **not opt-in**. Its histogram has one consumer (the rANS batch encoder) and the entropy branch tests Rice first, so on the default path it was filling 20 KB of workgroup atomics, re-reading the tile to do it, and nobody read the result. New `main_quantize_only` entry point: **3264 B measured** (I predicted 3320 by subtraction and was wrong). **A runtime flag could not have worked** — declared workgroup memory is charged per entry point, so a guarded array is still allocated — and **the histogram pipeline is now lazy**, because it is pipeline *creation* WebGPU refuses, not dispatch. **Invalidates no measurement: 10 of 10 encodes byte-identical**, Rice and rANS and abac, stills and sequences at ki=1 and ki=9; the Rice arm being identical is itself the proof the histogram was dead. Permanent canary asserts zero histogram dispatches on Rice *and* that the quantise path ran, so it cannot pass vacuously. Decision `0035`. **Still open (BUG-35 stays P2): `main` is unchanged at 23800 B, and shrinking `shared_hist` needs a guard first — `total_hist_entries` can reach 49152 against a 5120 arena and nothing checks it**, so an overflow silently corrupts frequencies. Also found: `rans_decode.wgsl` and `rans_encode_lean.wgsl` sit at *exactly* 16384 B, zero headroom. **Owed: throughput on an idle machine.** |
| `../gnc-ent7bpc` | `ent7bpc` | **ENT-6 — CLOSED by measurement 2026-09-08. abac's cold start is worth 1.3% of rate, not the ~4% it was filed on, and the 4% was a bound artefact.** `docs/decisions/0031`. **Invalidates no measurement** — a second env-gated read-only diagnostic (`abac_init_diag.rs`), no encoder path touched, `0024`'s six columns still reproduce byte-for-byte in the same run. The reason the old figure could not be right: `coef_entropy_diag`'s columns pool their probabilities over a whole plane's worth of a subband, so on a short block "shipped vs bound" mixes the cold start with the gap between a per-block model and a plane-wide oracle — **the pooling is the thing being priced, so a bound cannot price a change to initialisation.** Simulating the coder instead (shipped `Prob::update`, only the initialisation changing between arms) gives **−1.48% / −1.25% / −0.93% / −0.61% at q=85/90/95/99**, mean of four stills, against a ≥2% ship bar. Three findings worth carrying: the effect **shrinks with quality** while the bound ratio grows, which is the artefact itself; the item's own per-tile signalling design is **+0.4% to +1.2% — larger files** — against 864 B per frame for the same win, a factor of 40 nothing in the item disambiguated; and candidate 2 (drop the band-aligned cut below `cb`) is a **substitute, not a complement** — the two together buy no more than candidate 1 alone. Candidate 2 survives as a −0.40% win *and* a simplification whose only objection is parallelism, which **ENT-8 would remove**; cross-linked there. One instrument bug, caught by its own canary: the first version compared against `encode_block` (the **Interval** engine) while the shipped tiles are `Coder::Range` — 5 bytes per block, which looked like a broken model. The test now runs both engines and its per-block band (81.3 bits worst) is measured rather than chosen. Earlier text for this worktree follows. |
| `../gnc-wgmem31` | `wgmem31` | **BUG-31 FIXED 2026-09-08, and it was nine entry points, not four.** abac's two shaders were 18688 B against a device created with 16384 B; now **6400 B**, by packing `rows` to one clamped byte per magnitude rather than narrowing the workgroup. **Invalidates no measurement: 98 of 98 whole-file byte comparisons identical, and four decoded outputs plus their four bitstreams hash equal before and after** — the clamp is exact because `bucket` saturates at `nb >= 16`, so it cannot move a context. Rejected `WG` 32->28 (fits, but idles 12.5% of the lanes and still one workgroup per core), `MAX_BLOCK_W` 64->48 (invalidates every abac rate figure) and raising the request to 32768 (GOALS rule 4). Decision `0032`. **New: `tests/workgroup_storage_limit.rs`** computes declared workgroup storage per compute entry point from naga and asserts it against `wgpu::Limits::default()` — it found **five more offenders, filed as BUG-35, one of them (`quantize_histogram_fused.wgsl`, 23800 B) on the default encode path**. Its exception list holds exact sizes, so a fix cannot land silently and the record cannot rot. **The item's "run it in a browser first, and if it passes this is P3" was deliberately not made the gate** — a browser that does not validate would not make the shader conformant. **Owed: throughput, on an idle machine** (`abac_bench --ignored`); 6400 B reaches the two-workgroups-per-core occupancy point but no time was measured and none is claimed. |
| `../gnc-ent7bpc` | `ent7bpc` | **ENT-7 steps 2 and 3 — BPC-PaCo is rejected as a sixth coder, and two of its mechanisms are extracted.** `docs/decisions/0030`. **Invalidates no measurement**: one new env-gated read-only diagnostic module plus a print block, `BinCount` made `pub(crate)`, no encoder path touched, and decision `0024`'s six columns reproduce byte-for-byte in the same run (bbb q=90: 1791863 / 2080743 / 1790424 / 1696020 / 1669242). Priced on the shipped coefficients, four stills, q=85 and 90: BPC-PaCo's model is **−7.8%** of abac, then stationarity costs **5.7 pts** and its 32 codeword streams per block **4.0 pts**, landing at **+2.81% (q=85) / +1.85% (q=90)** — outside the item's own +2% gate at one point and +6.21%/+4.07% on bbb. The 7.9% is abac's cold start seen from the other side (`Y LL` −50.4%), so it goes to **ENT-6**, whose candidate 3 is now sized rather than speculated. Filed **ENT-8** (the two-column lockstep scan gives 32 threads per code-block at an *unchanged* average of 4 coded neighbours — abac has 1 thread today) and **BUG-34** (GNC requests 10 storage buffers against `Limits::default()`'s 8, so CLAUDE.md's rule-4 claim was false; prose corrected). **Part 1 was deliberately not touched — it is BUG-31 and stays free.** Two traps for the next reader: I first modelled BPC-PaCo as freezing the significance state per bitplane and got +10.2%, which is **withdrawn** — the lockstep scan sees current-plane neighbours and the papers' own ablation puts the whole penalty on the codeword streams; and **subgroups cannot save the throughput half**, because the ballot *is* the bitstream ordering rule and WGSL guarantees no relation between lane ids and `local_invocation_index`, so a subgroup port has a device-dependent bitstream. The reference CUDA has no licence file at all. |
| _(removed)_ | `ent7file` | **ENT-7 filed at the project owner's request 2026-09-08 — and it was already filed, which is the news.** Another session had committed an ENT-7 for BPC-PaCo hours earlier on **`claude/wgsl-bpc-paco-encoder-tvws1m`, pushed to origin and never merged**, so `scripts/claim` (which reads `main:BACKLOG.md`) could not offer it and a grep of the tree found nothing. **The lock does not cover ideas that live only on an unmerged branch** — that is the `0018` collision in a new place, and the cheap habit that would have caught it is `git fetch && git log --all --oneline` before filing, not a grep. Resolved by cherry-picking the original commit (authorship preserved) and merging both filings into **one** ENT-7 rather than leaving two: the original's BUG-31 framing, order of work, criteria and canary stand, and the second filing adds `0024`'s ≤7.5% ceiling (abac is +4.1% of the bound on the 82% of rate in full 64x64 blocks, so stationary models should be expected to **lose** rate there), the **1.9x** cap Rice's 47%-of-decode entropy stage puts on the throughput half, the two things it must not be bought for (PCRD at 0.00 dB; ~72% of the J2K gap is upstream), and two cheap gated steps that can settle it with no GPU — a literature read and a seventh model column in the existing `GNC_COEF_ENTROPY=1` harness. **Markdown only; no code, no shader, no measurement moved, GPU suite deliberately not run.** ENT-7 is left **unclaimed** so `next` can hand it out. If the original branch is merged later, expect a duplicate-content conflict in that section — keep the merged version. |

## `main` and `origin/main` diverged for about an hour — RESOLVED 2026-09-08, and the lesson is not about pushing

Noticed while filing ENT-7, resolved within the hour by another session, which rebased the local
commits onto `origin/main` and pushed (DOC-1 and ENT-7 now read `a8ac8c5`, `290920a`, `7dc9302`,
`bae4b83`, `0918d30` — the hashes in older rows and messages predate that rebase). What diverged:
local `main` had DOC-1 unpushed while `origin/main` had `6d2ecce` (BUG-33) and `0b9ff78` (DX12 does
not run GNC) unmerged. Nothing was lost and nothing was forced.

**Keep the part that outlives it.** `scripts/claim next` and `items` read **`main:BACKLOG.md`** —
the local branch in the shared checkout — so *an item that exists only on a branch, pushed or not,
is invisible to the lock*. That is how ENT-7 came to be filed twice in one day: the first filing
sat on `claude/wgsl-bpc-paco-encoder-tvws1m`, and the second session's grep of the tree found
nothing because a grep does not read unmerged branches. The habit that catches it costs one
command — **`git fetch && git log --all --oneline --grep=<idea>` before filing, not a grep** — and
it is the same class of miss as the `0018` decision-record collision.

## The test material was missing entirely, and was refetched (2026-09-07) — RESOLVED

**`test_material/frames` did not exist anywhere on disk** at the start of 2026-09-07 — not in the
shared checkout, not under any worktree. The 2026-09-06 `ln -sfn` accident is the likely cause: the
link was restored that day, but the ~31 GB directory it pointed at is gone. Nothing on disk
survived it.

**DONE — the material is back (2026-09-07 ~19:30), and it reproduces BASELINE.** All 22
artefacts are present and verified, and `q=75` on `bbb_1080p` reads **44.84 dB / 4.53 bpp**,
exactly the committed figure. Symlink your worktree at it per "Start of session" and carry on.

**Two things the refetch turned up, both now fixed in `fetch_test_frames.sh`:**

- **The script exited 0 with the blue_sky sequence missing.** ffmpeg 9 removed `-vsync` (renamed
  to `-fps_mode` back in 5.1), the extraction died on the unrecognised option, and `|| true`
  swallowed it. The script now probes which option the installed ffmpeg takes — it runs on more
  than one machine, so pinning either one just moves the breakage — and ends with an explicit
  per-file check of all 22 artefacts that exits non-zero if any is missing.
- **Do not edit a bash script while it is running.** Bash reads the file incrementally, so
  rewriting it mid-run made the next read land at a shifted offset and the run died with
  `syntax error near unexpected token '50-57'` — after the download, before the verification.
  Cost a whole extra pass. Edit, then run.

**Who fetched: the `chroma2` session** (worktree since removed), into the **shared checkout**
(`test_material/frames`), which is where the directory belongs and what every worktree's symlink
resolves to.

**If you ever need to refetch: check `ps` for a running fetch first, and do not start a second
one.** Two concurrent runs write the same paths, and the script's `[skip] already exists` guard is
checked before the write, so simultaneous starts all miss it and a half-finished download looks
complete to whoever checks next. That happened here — three runs at once — and nothing was
corrupted only by luck. There is no session left to ask, so verify with the script's own
per-file check rather than by asking anyone.

**Checked 2026-09-07 19:10 by the `abacship` session: there are TWO fetch runs live, not one.**
`ps` shows `bash ./fetch_test_frames.sh` (started 19:08:58, launched from the shared checkout's
`test_material/` — this is the chroma2 session) *and* `bash test_material/fetch_test_frames.sh`
(started 19:08:21, now orphaned, parent is `launchd`). A third exited at ~19:09. Because every
worktree's `test_material/frames` is a symlink to the same shared directory, all of them write the
same paths: two `ffmpeg` processes were writing `frames/blue_sky_1080p.png` **simultaneously**.
No damage this time — `bbb_1080p`, `blue_sky_1080p` and `kristensara_720p` are all complete, valid
PNGs at the right dimensions (checked header + `IEND`) — but they will collide again on
`touchdown_1080p` and on the two 8-frame sequences. **Before using any frame written after 19:08,
verify it**: `python3 -c "import sys;d=open(sys.argv[1],'rb').read();print(d[:8]==b'\x89PNG\r\n\x1a\n' and d[-8:-4]==b'IEND')" <file>`.
Whoever owns the orphan should kill it; whoever owns the survivor should re-run the script once at
the end, since its `[skip] already exists` guard will then leave the good files alone and refill
only what is missing.

**What the script does and does not restore.** It fetches four single frames (`bbb_1080p`,
`blue_sky_1080p`, `kristensara_720p`, `touchdown_1080p`) and two 8-frame sequences (`bbb`,
`blue_sky`, each with a generated `.y4m`). It does **not** fetch `old_town`, `aerial`, `crowd_run`,
`rush_hour`, `stockholm` or anything else quoted in RESEARCH_LOG — those were multi-GB and were
never in the script. It does not restore `frames/hdr/` either; regenerate that with
`scripts/png16.py` if a 10-bit measurement is needed.

**So any measurement in this repo naming a sequence outside that list cannot currently be
reproduced.** That is not a retraction of those numbers, but treat "re-measure X on old_town" as
blocked until someone re-fetches it, and say which sequences a new result actually used.

## Test material was missing from the machine entirely (2026-09-07)

`test_material/frames` did not exist in the shared checkout at the start of the day, so **every
worktree's symlink was dangling and no session could measure anything.** Refetched with
`test_material/fetch_test_frames.sh`; the four stills (`bbb_1080p`, `blue_sky_1080p`,
`kristensara_720p`, `touchdown_1080p`) and `sequences/bbb/` are back and every PNG verified to
decode. **Updated later that day: `sequences/blue_sky/` is back too** — the `chroma2` session fixed
the script's silent-failure path and refetched, and all 22 artefacts now verify. `frames/hdr/` is
still absent; regenerate it with `scripts/png16.py` if a 10-bit measurement is needed.

Two things to carry from how that went:

- **Three sessions ran the fetch script at the same time**, all writing the same output paths with
  `ffmpeg -y` and `curl -o`. The script's `[ -f "$out" ]` guard is not a lock: it is checked before
  the write, so simultaneous starts all miss it, and a half-written PNG looks complete to whoever
  checks next. Nothing was corrupted this time (verified by decoding all 12 files), but that was
  luck. **Check `ps` for a running fetch before starting one.**
- **Copy the images you measure on into your own worktree and record their hashes.** Another
  session's fetch overwrote `touchdown_1080p.png` while a measurement was being set up against it.
  Rule 1 says a number is only valid against a commit; it is equally only valid against a known
  input.

**`python3` on this machine has neither `numpy` nor `pillow`**, so `scripts/lossless_gate.py`,
`ypsnr_de00.py` and `chroma_metric.py` do not run out of the box, and the system python is
externally managed so `pip install` refuses outright.

**There is now one shared venv in the shared checkout — use it, do not make your own.**

```bash
REPO=$(git rev-parse --show-toplevel)          # resolves to your worktree
SHARED=$(git -C "$REPO" worktree list --porcelain | head -1 | cut -d' ' -f2)
"$SHARED/.venv/bin/python" scripts/chroma_metric.py ...
```

It has numpy 2.5.3 and pillow 12.3.0, and it is gitignored. `chroma_metric.py --selftest` passes
**16/16 Sharma reference pairs**, so the dE00 implementation itself is validated — run it once
before quoting a colour number, it costs a second.

## JPEG XS *is* measurable here — SVT-JPEG-XS now builds on arm64 (2026-09-07, chroma2)

**MEAS-9 is recorded in two places as blocked on this, with VC-2 standing in. It is not blocked.**
`scripts/build_jpegxs_arm64.sh` clones, patches and builds SVT-JPEG-XS on this Mac, and verifies
the result: a 1920x1080 yuv422p frame at `--bpp 3` round-trips to **PSNR y 44.484343 dB**,
reproduced exactly from a clean clone.

Only the *build system* assumes x86. The C sources already carry `#else /* ARCH_X86_64 */`
fallbacks for every dispatch, so the scalar path was written and simply never selected;
`scripts/svt-jpegxs-arm64.patch` gates the nasm discovery, `-DARCH_X86_64` and the nine ASM object
libraries on a detected arch, gives non-x86 a `get_cpu_flags()` returning 0 (EncHandle/DecHandle
call it unconditionally), and guards the SIMD headers that pull in `<immintrin.h>`.

**Rate and quality from this build are exact — that is what MEAS-9 needs.** Throughput is not:
every SIMD kernel is off, so the 9.17 fps encode it reports says nothing about JPEG XS and must
never be quoted or compared against GNC's fps. The latency row in the README stands on MEAS-6, not
on this.

## Three more sequences are back: bbb_extended, old_town_cross, crowd_run (2026-09-07, chroma2)

QUAL-1, MEAS-1 and BASELINE's BD-rate table are all measured on **bbb_extended, old_town_cross and
crowd_run**, and none of the three was in the tree — the fetch script has never fetched them.
`test_material/frames/sequences/` now has **24 frames of each** (1920x1080, PNG). Streaming the
first 24 frames of a 1.55 GB Xiph y4m costs about 80 MB, so this is cheap to redo; taking them
from the *start* of the file is what makes it cheap.

**They are 24 frames, not 200.** QUAL-1 used 200 frames of old_town_cross and an unstated count of
crowd_run, so a run against these reproduces the *content* but not the length, and BASELINE says
17 frames where the QUAL-1 log says 24. Say which you used.

## The wasm clippy gate is red on `main`, and it is the *binary*, not the library (2026-09-07, intra1)

CLAUDE.md names `cargo clippy --release --target wasm32-unknown-unknown` as a gate that must be
clean. **It does not compile on `main` at `07c01b1`** — 11 errors, all `src/main.rs` calling
`GpuContext::new()`, which does not exist on that target. Verified pre-existing by stashing a
branch's changes and re-running, so do not spend time believing it is yours.

The **library** target is clean: `cargo clippy --release --target wasm32-unknown-unknown --lib`.
That is the target that actually matters for the WASM story — `src/main.rs` is the CLI and is not
part of a WASM build — so the useful gate is the `--lib` form until someone either gates the CLI's
GPU paths behind `#[cfg(not(target_arch = "wasm32"))]` or drops the binary from that target.
Not claimed by anyone; it is a small mechanical fix and wants an ID before it is worked on.

## `cargo test --release` is flaky on `main` right now — abac_bitstream races itself (2026-09-07)

**`abac_handles_subsampled_chroma` fails under the documented gate and passes with
`--test-threads=1`.** 7/7 green serially, 6/7 in parallel, on `main` at 2c9e4d4, on a machine at
load 57. It is not content-dependent and not a rebase artefact — the CHROMA-2 branch that surfaced
it touches no `.rs`, `.wgsl` or `.toml` at all.

This file already says benches need `--test-threads=1` because cargo runs `#[test]` bodies
concurrently and two GPU tests contend for the device. That rule now applies to a **correctness**
test in `tests/abac_bitstream.rs`, not just a bench, which is worse: the gate CLAUDE.md and LOOP
both require (`cargo test --release`) can now go red on code that is fine, and — the real risk —
green on code that is not, depending on scheduling.

**So do not read a single red `abac_bitstream` run as your regression.** Re-run with
`--test-threads=1` before you believe it. And if it is yours, this wants a real fix (serialise the
device, or a lock around the GPU context) rather than everyone learning the workaround: a gate that
is sometimes wrong stops being a gate. Flagged by the chroma2 session; the abac/abacship sessions
own that file.

## Two retractions worth knowing about, both 2026-09-07 (from `abacship` / `bug18`)

**abac's inter rate figure (−14.4% at q=90) is withdrawn.** abac has no GPU encode path, so every
abac video encode runs the *non-batched* P-frame path — and BUG-18 shows that path encodes every
P-frame wrong: the first P after an I already diverges from the batched path by 28.8 (q=50) / 4.2
(q=90) and costs 2.1–2.8× the bytes. The measurement put abac on a broken arm and Rice on a working
one. **abac's intra and lossless figures are unaffected** and stand: −16.6% to −18.8% at identical
pixels, −13.4% bit-exact lossless.

**BOTH RETRACTIONS' CAUSE IS FIXED (2026-09-07, `arch3`). The inter figure is re-measured: abac is
−12.0% to −22.9% against Rice at bit-identical pixels**, nine of nine points across bbb_extended /
crowd_run / old_town_cross at q=50/75/90, 18 frames, ki=9, 4:4:4. Pixel identity established by
hashing decoded PNGs, not inferred from PSNR. The retraction below stands as a description of what
went wrong; the number it withdrew now has a replacement. `docs/decisions/0025`.

**The mechanism was a design defect, filed as ARCH-3 (P1), now fixed.** `gpu_entropy_encode` reads as
"entropy-encode on the GPU"; in `sequence.rs` it also selects which whole-frame P pipeline runs.
abac and bitplane are GPU-*decoded* and have no GPU *encoder* shader, so choosing either silently
swaps the frame encoder for the defective one. Nothing about those coders is broken. **This will
bite the next coder that lands decode-first, which is the natural order here** — decode is the side
the product is judged on. Separating the concerns is the smallest fix and removes the class.

**Anything encoded with `gpu_entropy_encode = false` on video before 2026-09-07 is suspect**, which
is abac *and* bitplane. Intra is fine — single-frame encodes go through `pipeline.rs` and were
verified pixel-identical between coders. **After ARCH-3 there is no such second path**: both arms
run the same frame encoder and decode to bit-identical pixels, asserted by
`tests/arch3_entropy_stage.rs`.

**And the general lesson, which cost three wrong published explanations in one afternoon: a matched
aggregate is not evidence that two arms are comparable.** On crowd_run at q=85 the two encode paths
agree on avg, min, max *and* stddev PSNR while the pixels differ by 8. This file already carries
the mirror-image rule everywhere — aggregates *hiding* a real difference, the whole VMAF-saturation
thread. This is the same failure with the roles swapped. If a comparison's premise is "these two
are equivalent", compare the pixels, not the summary.

## Timing: an idle machine is necessary and NOT sufficient (added 2026-09-06, after an idle run still lied)

**The GPU ramps its clocks, and a repeat count chosen for a CPU benchmark will not outlast the
ramp.** On a genuinely idle machine (load 3.8, GPU free), three consecutive *processes* on
identical input read **66.5, 45.3 and 34.9 ms** — a 1.9× spread, monotonically decreasing. A
freshly-idle Mac starts in a low power state and needs on the order of a second of sustained work
to boost. `abac_bench` had a `spread` column that blamed a busy machine for exactly this, so the
instrument was reporting the ramp *and* misattributing it.

Two rules that follow, and they cost nothing:

- **Take the best of ~24 repeats, not the median of ~7.** For a deterministic kernel on fixed
  input every error source — clock ramp, another session, scheduler noise — only makes a reading
  *slower*. The minimum is therefore the least-contaminated estimate, and `median / best` is a
  free settled-or-not diagnostic: near 1.0 means quotable, well above means keep only the ratios.
- **Run benches with `--test-threads=1`.** Cargo runs `#[test]` functions concurrently by default,
  so two GPU benches in one file contend for the device and interleave their output. This was
  happening and nothing warned about it.

**Isolating one GPU stage without timestamp queries:** dispatch it k times behind an env var and
difference. Rice's entropy decode is idempotent, so `GNC_RICE_DISPATCH_REPEAT=k` gives frame times
of 29.50 / 85.16 / 141.55 ms at k = 1 / 5 / 9 — slope 14.0 ms, two independent estimates agreeing
to 0.6%. That is how the abac question stopped being a bracket and became a number, and the same
trick works for any idempotent stage.

## Timing: the machine is shared, so throughput numbers are not measurable during a session

Eight sessions compile and run GPU work on this one Mac at the same time. That makes every
wall-clock figure unreliable while anyone else is working — on 2026-09-06 the same abac decode
input timed **25.2, 31.1 and 37.5 ms across three runs**, a 48% spread on identical work, and
three targeted shader optimisations against three different suspected bottlenecks all returned
exactly nothing. Three plausible fixes measuring null is far likelier to be a broken instrument
than three wrong hypotheses.

So:

- **Compression figures (bpp, PSNR, VMAF, dE00) are safe.** They are deterministic and unaffected
  by load. Measure and quote them freely.
- **Throughput figures are not.** Do not tune against them, and do not record one without saying
  the machine was loaded. An optimisation validated against noise looks justified and is not.
- **Build the alternatives behind switches and measure them later, together.** When something has
  several plausible implementations, implement them all, make them selectable, and add a bench
  that runs the whole set in one command. Then one idle-machine run settles it. `abac` does this:
  `GNC_ABAC_CODER` selects the entropy coder variant and
  `cargo test --release --test abac_bench -- --ignored --nocapture` times every combination.

## Builds share one compilation cache, and cargo is capped at 6 jobs (2026-09-07, buildperf)

**Two settings now live in `~/.cargo/config.toml`. Nothing is committed — they describe this
machine, not the project — so a fresh clone elsewhere is unaffected.** Decision record
[0021](docs/decisions/0021-builds-share-one-compilation-cache.md) has the full reasoning.

```toml
[build]
rustc-wrapper = "sccache"   # shared artefacts across all ten worktrees
jobs = 6                    # three concurrent builds fit 18 cores exactly
```

Why, in two numbers measured at 20:14 with five sessions in their gates: **load average 47.0 on 18
cores**, and **no `.cargo/config.toml` anywhere on the machine**. Every cargo therefore took the
default `-j` = 18, so five concurrent gate runs asked for 90 parallel rustc jobs. Separately, ten
`target/` directories held **~10.6 GB** of the same 264 dependency crates compiled ten times.

**What this changes for you:**

- **A lone session now gets 6 of 18 cores** and builds slower than it used to. If you are genuinely
  alone — rare at eight instances — use `cargo build -j18` or `CARGO_BUILD_JOBS=18`. Do not raise
  the value in the config file; the next session to build will be sharing with you.
- **Your existing warm `target/` is untouched** and you will notice nothing until a clean build or
  a rebase that invalidates dependencies. A *new* worktree is where the cache pays: measured
  **80.67 % hit rate** on an empty worktree at the same commit, a hit costing 0.077 s against
  2.725 s to compile.
- **Proc-macros and binaries are never cached** (`crate-type`), so a floor of per-worktree
  compilation remains. 118 such calls in a full build.
- **A shader edit is safe.** This was the one thing that could have made the cache dangerous: 61
  `include_str!` sites pull the WGSL files into the Rust crates, and a cache that ignored them
  would serve a stale object after a shader edit — you would measure a codec you had not written.
  Tested before enabling: editing the `.wgsl` produces a cache **miss**, and the artefact carries
  the new shader. `sccache --show-stats` is the canary if you ever doubt it.
- **The cache is worth more the busier the machine is**, which is unintuitive and useful. Measured
  at load ~36 and again at load ~70: `Average compiler` went 2.725 s -> **13.481 s** while
  `Average cache read hit` went 0.077 s -> **0.086 s**. Compiling got 4.9x more expensive under
  contention; reading the cache did not. So the cap and the cache are not substitutes — the cap
  lowers contention, the cache makes what remains cheap — and the benefit peaks exactly when five
  sessions hit their gates together.
- **When a test goes red just after this landed, it will look like the cache.** It probably is not.
  The first gate run here went red on `abac_handles_subsampled_chroma`, which turned out to be
  BUG-17 on the old base, not a stale object. A serial re-run and `git log` settle it faster than
  reasoning about the cache does.
- **`sccache --show-stats` is machine-wide and cumulative**, so it mixes all sessions together.
  `sccache --zero-stats` before a build you want to read in isolation.

**`--offline` does not avoid the package-cache lock — do not re-test it.** It was the obvious
candidate, since `Cargo.lock` is committed and every crate is already in the registry. Holding the
lock from another process and running a no-op build both ways: plain `cargo build --release`
blocked past 15 s and printed `Blocking waiting`, and `--offline` blocked past 15 s and printed it
too, while both finished in 1–2 s with the lock free. Cargo takes the lock whether or not anything
needs fetching. The paragraph above about queued builds stands unchanged.

**If the load average still sits above ~30 with the cap in place**, the next step is a build
semaphore — a `scripts/build` wrapper holding one of N slots, `flock` via `python3` since macOS has
no `flock(1)`. It is deliberately not built yet: it binds only if every session calls it, and that
is a convention rather than a lock. Rule 0b exists because this repository already learned what
conventions are worth here.

## The four rules that have actually bitten us

**1. A number is only valid against a commit.** The tree can change under you with no signal. A
BD-rate figure moved from −3.7% to −2.3% purely because the coefficient path changed between the
sweep and the commit, and it had to be corrected in three documents after publication. Working in
your own worktree fixes most of this; still, measure a committed state, or a worktree pinned to a
hash (`git worktree add <dir> <sha>`).

**2. Measure the range the project cares about, not the range that is convenient.** GNC is a
*contribution* codec (GOALS §1). TUNE-5 was measured at q=15-50, shipped, and then found to cost
10.3 dB at q=99 — the operating point that actually matters. A sweep that stops at q=50 has not
tested this codec.

**3. Above about q=80, VMAF is saturated and PSNR must lead.** On old_town at q≥85 VMAF reads
identical for two settings whose worst-frame PSNR differs by 4.8 dB. State which metric led, and
why. CLAUDE.md now carries this as a table rather than a blanket "VMAF primary" rule, and QUAL-1
put a magnitude on it: widening a BD-rate ladder moved the VMAF figure by **47.5 points on
average, 110 at worst**, while the PSNR figure moved **1.0**.

**4. A point measurement at fixed q cannot judge a rate/quality trade**, and it always flatters
the option that spends more bits. Use BD-rate, or compare at matched rate. At least four separate
wrong conclusions have come from this one error.

## Landed today, and what each one invalidates

- **RATE-2 — a still at q = 95..=99 may now come back bit-exact, and it is smaller.**
  `docs/decisions/0036`. **This one changes output**: mean **−21.66% at q=99** on four stills, 12
  of 20 points bit-exact. **Unchanged**: everything below q=95, q=100, `--dct`, and every
  sequence — byte-identical. Three things worth carrying:

  - **"A bit-exact reference must be a better reference" is a reasonable thought and it is
    wrong here.** Letting the fallback reach sequence I-frames made the P-frames referencing them
    decode at **9.80 dB against 60.69 dB**, and the sequence *grew*. Filed as **RATE-3 (P1)**. The
    fix was tested on the sequence path before being believed, which is the only reason this is a
    filed item rather than a shipped regression.
  - **A gate in one place was not a gate.** `encode_sequence` clears the flag internally, and the
    canary still fired during `benchmark-sequence` — because the **GPU warm-up encodes call
    `encode` directly** with the sequence config. A canary that prints on *both* outcomes is what
    made that visible; one that printed only on success would have hidden it.
  - **BD-rate ladders that reach q≥95 are not comparable across this commit.** The top of the
    ladder now flattens onto the lossless point, which is the correct convex hull, and
    `bd_rate` drops non-finite PSNR — so the figure is computed over fewer points than before.
    BASELINE's 1.9x-vs-H.264 caveat is **updated rather than lifted**: that ladder is video and
    this fix is intra, so it reproduces exactly, and the rung it waits on is RATE-3.

- **ENT-6 — closed by measurement: abac's cold start is 1.3% of rate, not 4%.**
  `docs/decisions/0031`. **Invalidates nothing** — one more env-gated read-only diagnostic. One
  thing worth carrying, and it applies to every bound in `coef_entropy_diag`:

  - **A bound whose statistics are pooled cannot price a change to initialisation, because
    pooling is what the change is trying to buy.** `0024` measured the short code-blocks at +25.9%
    over a plane-pooled bound and the cold start was recorded as "worth about 4% of the file".
    Simulating the actual coder and changing *only* the initialisation gives 1.25% at q=90. The 4%
    was a correct reading of a bound ratio and a wrong reading of a collectable rate — the two
    diverge with quality, in opposite directions.
  - **Two arithmetic engines sharing a binarisation make "compare against the real thing"
    ambiguous.** This diagnostic's canary compared against `encode_block` while the tiles it was
    pricing were `Coder::Range` → `encode_block_rc`; the two differ by about 5 bytes per block of
    flush, which presented as a broken model rather than as a wrong baseline. `AbacTile` records
    the engine per tile precisely so this cannot happen, and the canary was not reading it.

- **ENT-7 steps 2–3 — BPC-PaCo is not the sixth entropy coder; ENT-6 gets its mechanism.**
  `docs/decisions/0030`. **Invalidates nothing** — the only code is an env-gated read-only
  diagnostic, and `0024`'s columns reproduce exactly in the same run. Three things worth carrying
  that are not about this item:

  - **A mechanism attributed to a paper nobody had read yet was a guess, and it flattered the
    prior.** ENT-7's own text reasoned that a coefficient-parallel coder cannot see its neighbours
    in the current bitplane; the first implementation followed that and measured BPC-PaCo at
    **+10.2% of abac's rate**, a number that went to the project owner before the papers came
    back. BPC-PaCo actually gets an average of **4 coded neighbours — identical to a raster
    scan** — by scheduling two-column stripes in lockstep. The wrong version was kept as a column
    (`Hbpcn`), where it prices something we do need: dropping the cross-lane exchange costs
    **8.6%**.
  - **`Limits::default()` was not what GNC asks for, and `gnc gpu-info` had been printing the
    proof for months.** `max_storage_buffers_per_shader_stage: 10` against a default of 8
    (BUG-34). Found by checking someone else's claim about the WebGPU default, not by looking. The
    static-limits assertion BUG-31 already needs should cover **every** field of `Limits`, not
    just workgroup storage.
  - **An external critique of BPC-PaCo's headline speedup applies to us.** Rossinelli et al. note
    that its 25x over Kakadu confounds the algorithm with the CPU→GPU move. Any GNC throughput
    figure comparing coders across substrates has the same hole; abac's are stated against Rice on
    the same GPU, and that is not a detail to relax.

- **ARCH-3 + BUG-18 — one P/B frame encoder, and `gpu_entropy_encode` stopped choosing one.**
  `docs/decisions/0025`. **Invalidates nothing**: the default configuration is byte-identical on
  **54 of 54** encodes against a binary pinned at `07c01b1` (bbb 8 frames and bbb_extended 18,
  ki=2 and 9, q=50/75/90, 4:4:4 / 4:2:2 / 4:2:0, B-pyramid on and off). What it *changes* is abac,
  bitplane and Huffman on video, all of which were wrong before.

  Four things worth carrying that are not about this item:

  - **The unit test said the fix was complete and it was not.** The invariant — the entropy choice
    cannot reach the pixels — held on 256x256 synthetic content and failed on **5 of 9 points on
    1080p**. A second gate keyed on the coder (`dispatch_zero_skip_tiles_by_map`, Rice-only, while
    the MV-zeroing beside it ran for every coder) was still making Rice and abac code different
    coefficients for the same frame. **A full-frame pan has no static tiles, so skip mode never
    fires and the mechanism under test is never exercised.** Content chosen for convenience
    certifies this class of fix as complete. The test now runs a half-frozen frame too.
  - **A measurement script reported "identical" for nine points it had not measured.** `cd
    "$(dirname $0)"` moved it out of the worktree, every encode failed, and two empty directories
    hashed equal. `set -e` and an explicit output-count check are cheap; a comparison whose inputs
    are missing is a broken instrument, not a null result.
  - **`--huffman` video was broken on `main` and no test encoded Huffman video.** Empty P-frame
    tile vectors; decoding panics in `frame_data.rs:335`. Verified on the pinned baseline, so it
    shipped. The class is "a coder that is not the default is not exercised end to end", and
    bitplane was in the same state.
  - **A dead parameter survived because only the unused path read it.** `encode_pframe` took a
    temporal MV predictor; only the deleted implementation used it, so GNC has never done
    temporal MV prediction on the shipped path. Deleting the unused branch is what made the
    compiler say so.

  **The local-decode dequant defect: found here too, already owned, and I named the wrong item.**
  The encoder's local decode dequantises P residuals with `config.quantization_step` while the
  forward pass quantises with `res_qstep = quantization_step × p_qp_scale` (TUNE-6), so the
  encoder's reference is reconstructed at a different step from the decoder's wherever the scale
  exceeds 1.0. Both P-frame implementations did it and the surviving one still does — this change
  neither causes nor hides it. **`a312d6f`'s commit message and decision `0025` attribute it to
  BUG-8; that is wrong** — BUG-8 is closed and was a metric bug. It is the defect `gnc-inter1`
  holds under the second `BUG-25`, and this is an independent confirmation of it from the other
  side. **It now needs id BUG-27**: I filed a BUG-26 (below, since renumbered to BUG-28) while
  that renumbering was still in a worktree, so the "next free id" moved while nobody could see it. Same failure the
  section on the BUG-25 collision describes, one round later — an id is still not allocated by the
  lock.

  **New item filed on the way out, BUG-28 — filed as BUG-26, renumbered (P2):** abac and Rice decode to *different pixels* at
  4:2:2 and 4:2:0 on the **intra** path — max |diff| 12-13 over ~4% of samples at q=50, 5 over
  ~1% at q=75, identical at q=90. **Reproduces on `main` at `1d67d29`, so it is not this change.**
  No published figure is wrong (every abac measurement in the repo is 4:4:4), but abac's "at
  identical pixels" and decision `0018`'s "every chroma format at once" are 4:4:4 claims, and
  nothing said so. Small differences over a large area is the shape a PSNR average hides.

  **For whoever holds ENT-5 (abac GPU encoder):** rebase onto this. `inter_gpu_entropy_available()`
  in `entropy_helpers.rs` is now the single place that says which coders have a GPU entropy
  encoder; adding abac's shader means adding it there plus a dispatch arm beside Rice and rANS,
  and no longer means touching a frame encoder. abac video is also correct now, so a new encoder
  can be checked bit-exact against the CPU coder on inter as well as intra.

- **BUG-27 — the encoder's P-frame reference was dequantised with the *intra* quantiser step, so
  every inter measurement at q <= 80 in this repository is wrong.** From INTER-1. **Filed as
  BUG-25 in the worktree and renumbered to 27 per the collision section below** — the Vulkan
  BUG-25 was pushed first and BUG-26 then went to abac/Rice subsampled chroma. `arch3` found
  this defect independently while reading both P-frame implementations for ARCH-3; **neither of
  us had fixed it, and this fixes it.** Six sites when found, three after `a312d6f` deleted the
  second implementation — both had it, so it was an artefact of neither.
  `encode_pframe` quantises P residuals at `res_qstep = quantization_step * p_qp_scale` (TUNE-6's
  taper) and records that for the decoder; its six **local-decode dequantise** dispatches read
  `config.quantization_step` — both encode paths, all three of luma / 4:2:0 / 4:2:2 chroma. The
  encoder's reference therefore differed from the decoder's by `quantization_step / res_qstep`,
  and every P predicting from another P inherited a picture no decoder holds.

  **Whether it bit you depends entirely on q, and the boundary is sharp.** The taper is keyed on
  the quantiser step and returns exactly 1.0 for every step at or below 2.8 — which is q=85 and
  above. There the wrong value and the right one coincide, and output is **byte-identical** before
  and after, verified against a pinned `07c01b1` build (crowd_run ki=9: q=85/90/99 give
  27588358 / 34128220 / 49328550 bytes both ways). Below q=85 the taper leaves 1.0 and the defect
  is live on the **default path**. crowd_run, 10 frames, ki=9, 4:4:4, mean/worst-frame PSNR:

  | q | before | after | delta |
  |---|---|---|---|
  | 25 | 3233137 B, 29.18/27.78 dB | 3247396 B, 29.47/28.19 dB | +0.4% B, +0.29/+0.41 dB |
  | 50 | 7201466 B, 32.49/30.37 dB | 7316803 B, 33.39/32.15 dB | +1.6% B, +0.90/+1.78 dB |
  | 70 | 13161434 B, 35.20/32.08 dB | 13423148 B, 37.02/35.70 dB | +2.0% B, **+1.82/+3.62 dB** |
  | 80 | 21595433 B, 40.77/38.91 dB | 21734608 B, 41.57/40.53 dB | +0.6% B, +0.80/+1.62 dB |

  **What it invalidates.** Every inter figure taken at q <= 80, which is most of them:
  **MEAS-3's +4.6% mean / +19.1% worst-frame** (ladder q=25-95) and **decision 0019**, which rests
  on it — re-running now, and the worst-frame column is where the fix lands hardest, so expect it
  to move most. **TUNE-6's own justification too**: its recorded "old_town q=99: -3.8 dB avg,
  **-14.2 dB worst**" for scale 1.25 was measuring this defect and not the trade, so the taper's
  shape is unjustified until re-measured, and "1.25 is bad at high q" is not currently a
  supported claim. **Not invalidated:** anything at q >= 85 (byte-identical), every still, and
  INTER-1's own ki sweep, which ran entirely at q=85-99.

  Three things worth carrying beyond this bug:

  - **A knob that is a no-op at the operating point you test hides bugs in itself.** `p_qp_scale`
    is 1.0 for all q >= 85, so the contribution range was correct *by coincidence*. Nothing was
    wrong with the taper; what was wrong was only reachable where the taper does something.
  - **More bits for worse quality is not a trade, it is a broken arm** — and it was the tell here.
    Forcing scale 0.90 spent 4% *more* bits and lost 5 dB. Coarser-and-worse (1.25) looked like a
    plausible bad trade and nearly got written up as one; the sub-1.0 direction had no such
    reading available, which is the argument for bounding a direction you expect to lose rather
    than assuming it.
  - **`cargo test` rewrites `target/release/gnc` and will do it underneath a running sweep.** Mine
    was replaced 3.5 minutes into a 25-minute run by a comment-only edit, so the output was
    certainly identical — and the run was still discarded and restarted against a **copied,
    hash-recorded binary**, because "certainly identical" is not a measurement. If you background
    a sweep, copy the binary first and point the harness at the copy.

  Filed and fixed in one session; the number was reserved with `scripts/claim take BUG-27` before
  being written down, per the bug-number collisions above. Regression test
  `tests/pframe_reference_drift.rs` gates on drift magnitude down a GOP (4.04 dB with the defect,
  0.86 dB from the inter dead zone alone, threshold 1.5) and uses **no environment variable** — it
  reaches the taper through q=50, since a `set_var` in a `#[test]` is the race that masked a real
  decoder bug in `abac_bitstream`.

  **For the ARCH-3 / BUG-18 owner specifically:** this is in `encode_pframe`, the same function
  you are splitting, and it is the same *class* as BUG-18 cause 1 — encoder and decoder disagreeing
  about which quantiser step a P residual used — with the sides swapped: BUG-18 was the quantise
  call, this is the dequantise that builds the reference. Both clusters of six are fixed
  symmetrically, so a rebase onto `2224c50` should be mechanical, but expect a conflict.

- **BUG-9 — the recorded cause was backwards, and a figure I gave another session needs its
  baseline attached.** `328e76a` + `2b120b1`. rANS's ceiling is not the 4 KB per-stream slot: it
  is the cumfreq table, where every subband group's table for a tile shares one workgroup array
  of 4097 entries and the *sum* over the groups has to fit. With `--rans` at the default step,
  worst tile, Y plane — kristensara 4020 at q=75 and 4165 at q=76 (refused), bbb 4052 at q=76 and
  4197 at q=77 — and **no stream overflowed its slot at any point that completed.** That is
  ENT-2's q=75/76/77 content split measured a second way, with the mechanism: the Y-plane alphabet
  crosses 4097 at different qualities per image. The slot bug was real too (an unguarded
  `write_ptr` decrement wrapped into the *previous* stream's slot and ORed bits into correct
  data), but it was the symptom. **Invalidates nothing measured**: byte-identical to the parent on
  64/64 points. Corrects BUG-9's "this is not the symbol alphabet", its q=80 ceiling, and its
  "--rans is a no-op" — the last two were handed over by ENT-2. Decision record 0022.

  Two things to carry that are not about rANS:

  - **A sibling worktree's build is not a baseline.** I gave the MEAS-9 session "byte-identical
    36/36" without naming the *before* binary. It was `gnc-meas9/target/release/gnc`, verified
    src-identical to my branch point at the time. That session rebased and rebuilt, and a re-run
    returned **0/36 — every point differing, including lossless q=100, which does not touch
    rANS.** Nothing in my branch had changed. Rule 1 says a number is only valid against a commit;
    the sharper form is that **the baseline binary lives in a worktree you own, pinned to a hash**
    (`git worktree add --detach <dir> <sha>`, then build). If RESEARCH_LOG still carries that
    36/36 as "reported, not verified here", the useful edit is to name the baseline: the figure
    is now 64/64 against a pinned 436680e.
  - **Out-of-bounds workgroup access is not deterministic, so old results near a limit are
    suspect in a way a reproducible bug is not.** Two builds of near-identical source disagreed
    about whether the same input overflowed.

  And a note for whoever owns conventions: **decision-record numbers collided a third time
  today.** I wrote 0021; main took 0021 for the shared-compilation-cache record while I was
  pushing, so mine is 0022 and `2b120b1`'s commit message is wrong on that one detail. Two 0018s
  and two 0019s already exist. It is the same read/decide/write race `scripts/claim` removed for
  backlog items, on the same afternoon that script was written.


- **BUG-15 — the wavelet lossless arm was not lossless, and it is fixed.** `GNC_MED=0` at q=100,
  and any `--qstep 1 --wavelet 53` config below it, returned **53–56 dB with dE00 0.5–0.9** instead
  of bit-exact output; `GNC_PHYSICAL_WEIGHTS=1` returned 6 255 bytes at 49.2 dB on a gradient.
  `is_lossless()` never checked the **subband weights**, and `pack_weights_chroma()` scales the
  quantiser step by `chroma_weight`, which CHROMA-1 raised to 1.2 for every q >= 60 — q=100
  included. Invisible because LOSSLESS-1 had routed q=100 to MED the day before, so the only
  configuration CHROMA-1 broke was the one no test built.
  **What it invalidates:** any lossless figure taken with `GNC_MED=0` between `cbfa17f` and
  `d399a99` is a 53–56 dB file mislabelled lossless. COORDINATION's own line "q=100 verified
  bit-exact lossless on all three entropy coders" was false for the wavelet arm in that window.
  **What it does not:** the default q=100 path (MED) was and is bit-exact — output is
  byte-identical before and after the fix — so **LOSSLESS-1's −14.9%, the +25.8% FFV1 gap and the
  −14.3% abac follow-up all stand**, and so does CHROMA-1's −5.2% (the lossy side is untouched and
  asserted so). Fix is `CodecConfig::normalized_for_lossless()`, called from `quality_preset` and
  again at the encoder entry because `--qstep`/`--wavelet` land after the preset.
  Two conventions worth carrying: **a feature that stops being reachable by default stops being
  tested even when its tests still run**, and **assert bit-exactness, not a PSNR threshold** — a
  `psnr > 45.0` assertion reads 55 dB as a pass, which is how this survived a day.

- **INTRA-NEARLOSSLESS — closed by measurement; MED instead of the wavelet does not survive into
  the lossy range.** Gate failed on criteria set beforehand: BD-rate luma is **sign-varying**
  (bbb +14.1%, blue_sky −27.2%, kristensara −26.4%, touchdown −22.5%) and colour is worse
  everywhere (+31.6% to +106.3%). The mechanism is DPCM's, not the implementation's — quantisation
  error feeds back through the predictor, so rate falls far more slowly than quality and **the
  usable ladder is delta=1 or delta=2 with nothing between** (bit-exact or ~51 dB, no way to ask
  for 55 dB). Scoped narrowly on purpose, after the abac near-miss: what is closed is *replacing
  the wavelet with a quantised closed-loop MED predictor in the lossy range*. `scripts/nearlossless_gate.py`.
  Two modelling artefacts recorded there, both caught by checking monotonicity: **a fractional
  quantiser step does not divide the integer pixel lattice** (modelled rate *rose* as the step
  coarsened) and **the calibration point must actually be lossless**. Four runs void before that
  was found. A coarser quantiser producing more bits is a broken instrument, not a finding.

- **RATE-2 was found twice within the hour, by two sessions, from opposite directions.** The
  RATE-1 sweep filed it and that entry stands; this session's numbers fold in as an independent
  confirmation on different inputs (crops, +1.6% to +33.2% against their +9.3% to +40.6%) plus the
  boundary in **qstep** terms. Also rules out one framing: sub-unit qstep is not wasted precision —
  qstep 0.75 buys 3.9 dB over 1.0 for 13% more bits, a normal RD slope, so the rungs are mispriced
  only against lossless. **A claim taken when you pick an item does not cover what you trip over
  inside it**, which is why `scripts/claim` did not prevent this one.

- **Correction to a diagnosis I committed: the `abac_bitstream` flake was NOT GPU contention.**
  I read "a different test fails each run, all pass with `--test-threads=1`" and concluded two GPU
  tests were contending for the device; that is in commit `4133f54`'s message and it is wrong. The
  real mechanism, found by the ABAC-SHIP owner: a test called `std::env::set_var("GNC_ABAC_CODER")`,
  the environment is process-global, and cargo runs `#[test]` functions as threads — so it changed
  which arithmetic engine a *concurrently running* test encoded with. The concurrency half was
  right, the shared resource was not. It also masked a real decoder bug: `CachedBuffers` held one
  `abac_coder` for all three planes, so planes 0 and 1 decoded with whatever plane 2 used. Fixed at
  the source by moving the coder and code-block size out of the environment into `CodecConfig`, so
  `cargo test --release` stays the gate unqualified and no serialising mutex was needed. Worth
  carrying: a symptom can fit a mechanism exactly and still have a different cause, and "it is
  contention" is a comfortable answer that stops the search early.

- **RATE-2 filed — above q≈95-98 GNC is dominated by its own lossless path.** No code change, so
  no output moves, but it **invalidates how any BD-rate whose ladder reaches q=95-99 should be
  read**: those rungs spend more bytes than bit-exact lossless for a worse picture (mean +28.9% at
  q=99; blue_sky 3026470 bytes at 60.14 dB against 2153118 bit-exact, +40.6%). Crossover per image
  q=98/95/96/96. Not universal — smoothramp, flat and noise are not dominated, because MED
  prediction is poor there. Also from the same sweep: **RATE-1 is answered no** (0.0% recoverable
  on photographic content), and **the ladder is not monotonic in rate** (flat512 costs 0.0450 bpp
  at q=86 and 0.0370 at q=90), so anything interpolating GNC by rate should flag a rung whose rate
  falls while q rises.

- **MEAS-3 — the inter path's rate saving does not survive matched quality. Invalidates a GOALS
figure; changes no default.** BD-rate of the shipped ki=9 configuration against all-intra, three
sequences, 18 frames, q=25-95, 4:4:4: **+15.9% (crowd_run), +22.2% (old_town_cross), −24.2%
(bbb_extended), mean +4.6% on mean PSNR — and mean +19.1% on worst-frame PSNR.** Positive means
inter needs *more* bits. Above q≈85 the saving is gone; at q=95 inter costs more than all-intra on
two of three sequences. Decision record 0019; the follow-up is BACKLOG **INTER-1**, unclaimed.

Three things to carry:

- **GOALS §4's "the inter path saves only 17-27% vs all-I" is annotated, not deleted** — it is an
  equal-setting rate figure. At q=70 on crowd_run the inter arm spends 4.74 bpp against intra's
  7.96 *while sitting 7.6 dB lower*, and the 2026-03 run judged that quality equal on VMAF 99.09
  vs 99.10. **Any inter-vs-intra rate claim in this repo predating today is suspect** unless it
  names a BD-rate or a matched-quality point.
- **Quote worst-frame PSNR next to the mean for anything touching the inter path.** The mean says
  +4.6%, which reads as neutral; the worst frame says +19.1%, and for a contribution codec the
  worst frame is what survives downstream re-encoding. crowd_run q=70: inter 34.50 mean / 32.08
  worst, all-intra 42.07 / 42.06.
- **A q≤85 cap does not make a VMAF BD-rate safe.** crowd_run's overlap came out 99.55-99.84
  because the *all-intra* arm is already saturated at q=25, and the arithmetic produced +132.4% —
  the most dramatic number in the run and not a number at all. `meas3_sequence_rd.py` now prints
  the overlap beside every VMAF BD-rate and discards one whose floor exceeds 99. Worth copying
  into any harness that reports a VMAF BD-rate.
- **`git checkout --theirs .` after resolving a stash-pop conflict silently threw away three other
  sessions' work** — mine, in this worktree, on 2026-09-07. The two files were resolved correctly
  in the working tree but still marked `UU`, so that command replaced them with the stash side
  alone: RESEARCH_LOG lost the ABAC-SHIP, ENT-2 and MEAS-9 entries and COORDINATION reverted to a
  base predating three merges. Nothing was pushed — it was caught by counting the other sessions'
  headings before committing, and both files were rebuilt from `origin/main` plus the new block.
  **After resolving a conflict, `git add` the file; never run a bulk `checkout --ours/--theirs` to
  "clean up".** And when a conflict resolution touches a shared document, grep for the other
  sessions' entries as a matter of course — a rebuilt file that compiles and reads fine can still
  be missing 400 lines of someone else's day.

**GPU selection from the environment, and a tier/density harness — invalidates nothing.**
- **`scripts/claim` locks a session out of its own worktree if it registers during a rebase, and
  its `--force` only works in one position.** Both cost me time on 2026-09-07 and will cost the
  next session the same, because everyone rebases. `claim worktree` run from a detached HEAD — which
  is where a rebase leaves you — records the holder as `<worktree>@detached#sNNNNN`. Finish the
  rebase and you are `<worktree>@<branch>#sNNNNN`, a different identity, so `take`, `touch` and
  `drop` all refuse with *"held by ...@detached..., not by ...@branch..."*, including on your own
  worktree. The way out is **`claim drop --force <target>`, flag before target**:
  `claim drop <target> --force` prints *"Use --force if you mean it"* and then ignores the flag,
  and `claim --force drop <target>` is `unknown subcommand`. Two cheap fixes for whoever owns
  COORD-1: accept the flag in either position, and leave the branch out of the holder identity.

- **Bug numbers collided four times on one branch, so put them in the claim registry.** BUG-14's
  branch filed 16-19, then 17-20, then 19-22, and landed on **21-24** — each rebase found the
  numbers taken underneath it (`abacship` took 16, `coord` and `abacship` took 17/18 minutes apart,
  `coord` took 19/20 while this branch was rebasing). Reserving them with `claim take BUG-NN` is
  what finally held, and it works today with no code change. **A `claim bug` that allocates the
  next free number atomically would end this class of churn**; reading COORDINATION first will not,
  because read-decide-write is three steps and everyone is doing it at once.

- **BUG-14 — Huffman's stream mapping is fixed, and Huffman was broken three further ways.
  Invalidates nothing that ships, and one thing that does not.** The mapping fix is byte-identical
  at the default 256 px tile on all 8 measured points and through both encoders, so no preset, no
  BASELINE row and no committed figure moves. What it *does* invalidate: **any future tile-size
  experiment run through Huffman before today** would have scored the larger-tile arm through a
  13–23% penalty that has nothing to do with geometry — the same error BUG-11 corrected for Rice.
  Worth −2.9% at tile 128 and −20.1% at tile 512 (q=75, four stills, PSNR identical at every
  point). 256 is still Huffman's best tile: after the fix 512 is still larger *and* lower in PSNR.

  Three defects the gate found before it found anything about the mapping, all pre-existing, none
  of them the mapping, all in a coder nothing measures through — **so if you are about to route
  anything through `--huffman`, read this first:**

  - **BUG-21 (fixed)** — `num_groups = num_levels * 2` was zero on the MED lossless path, so
    `--huffman -q 100` panicked on the host encoder and, on the GPU encoder, built no codebook and
    wrote **4.1–9.7 dB output with max error 255 in a file smaller than the same image at q=90**.
    `.max(1)`, as `rice.rs` has always had. q=100 Huffman is now bit-exact; it is +9.4% behind
    Rice, which is not a reason to un-park it.
  - **BUG-22 (guarded, not fixed)** — the GPU encoder writes each stream into a fixed 512-byte slot
    with no bound check, so a longer stream silently overwrites its neighbour's. That is the whole
    of the tile-512 corruption (7.8–10.9 dB at q=90 on all four stills). It now refuses with the
    tile, the stream and the byte count. **Every Huffman tile-512 figure ever taken here is
    suspect**, which in practice means none, because nobody took one.
  - **BUG-23 (bounded, not fixed)** — `clamp_code_lengths` did not terminate. `-q 100 -t 512` on
    bbb spun at 79% CPU for 8 minutes before it was killed; it now fails in 0.148 s. It was
    unreachable only because BUG-21 meant no codebook was ever built.

- **BUG-24 filed — `cargo clippy --release --target wasm32-unknown-unknown` already fails on
  `main`, and has for a while.** 11 × `no associated function or constant named 'new' found for
  struct GpuContext`, all in the **bin** target: `GpuContext::new` is `#[cfg(not(target_arch =
  "wasm32"))]` and `main.rs` calls it unconditionally. Reproduced on a clean tree at `bc851c7`, so
  it predates the BUG-14 branch; most likely arrived with `fcac02f`. **`--lib` is clean with no
  warnings**, and the library is what WASM actually ships. So CLAUDE.md's "both clippy targets must
  be clean" currently cannot be satisfied as written, and every session is hitting it on a CLI
  binary that is never built for WASM. Whoever owns `fcac02f` is best placed to pick the fix.

- **A second bug-number collision in one week.** This branch filed 16–19; ABAC-SHIP landed its own
  **BUG-16** while it was in flight, and the clash only surfaced at rebase. Renumbered to 17–20.
  Read/decide/write is three steps in a markdown table and the table cannot exclude anyone —
  `scripts/claim` can, and it now holds the areas. **Bug numbers are not in it.** Until they are,
  take the next number *and rebase before you write it down*.

- **GPU selection from the environment, and a tier/density harness — invalidates nothing.**
  `GNC_GPU_ADAPTER` (name substring), `GNC_GPU_BACKEND`, `GNC_GPU_POWER` and `GNC_GPU_INFO` choose
  the device at context creation; `gnc gpu-info` lists what wgpu can see. Encoder output is
  unchanged and verified so — q=75 on bbb_1080p is still 44.84 dB / 4.53 bpp, matching BASELINE.
  A `GNC_GPU_ADAPTER` that matches nothing is a hard error rather than a silent fallback, so a
  tier comparison cannot quietly be two runs on the same card. **Nothing is measured yet**:
  `scripts/gpu_tier_bench.py` and `docs/GPU_TIER_TEST.md` are waiting on a machine with more than
  one GPU. Do not quote a throughput number from this repo until that has run.

- **CHROMA-1 — `chroma_weight` now stays at 1.2 above q=85 instead of dropping to 1.0.**
  Changes default output at **q ≥ 85 only**: BASELINE's q=90 row moves 50.41 → 50.06 dB and
  8.58 → 8.07 bpp, which is −5.2% luma BD-rate for +1.2% on colour. q=25/50/75 are byte-identical
  and q=100 stays bit-exact (the quantiser is bypassed there). **Any q ≥ 85 file size measured
  before this is stale**; ratios within one build are fine.
  Two things worth carrying: **VMAF read 97.08 before and after, on 6% fewer bits** — luma-only and
  saturated, so a VMAF-only verdict here is worthless, which is the same illusion that made the
  2026-09-05 sweep look like a free 15%. And **luma measured as BT.709 Y from decoded RGB is
  contaminated by chroma error** — it overstated the loss 3.7x. Use YCoCg-R Y (`ypsnr_de00.py`).
  Also settled by elimination: the +90.5% video gap is **not** an allocation artefact. The knob is
  intra-only (−20.8% all-intra against −2.9% on a ki=9 P-chain), so intra really is the only route.

- **QUAL-1 — the headline gap figure is corrected.** At the contribution operating point GNC needs
  **+90.5% BD-rate on PSNR** (about 1.9x), not the **+456.7% to +672.1%** recorded from MEAS-1.
  Nothing in the coder changed; the old figure was measured at distribution bitrates.
  **Invalidates every use of the 5-7x figure**, including in POSITIONING and any argument that
  single-digit improvements are pointless — against 1.9x they are not.
  Two further things to carry: **never quote a VMAF BD-rate above about q=85** (widening the ladder
  moved it 47.5 points on average, 110 on old_town, while PSNR moved 1.0); and **at matched rate
  GNC beats x264 on dE00** while losing 7.4-8.8 dB of luma, so quote luma and colour together or
  the number misleads in whichever direction suits.
  Also: **MEAS-1's stated sources are not reproducible** — `bbb.y4m` has 8 frames, not 17, and no
  `touchdown` sequence exists in the tree.

- **The test material was unreachable for about ten minutes (18:04, restored).** `ln -sfn` run
  inside the shared checkout pointed `test_material/frames` at itself, so every worktree's link
  resolved to a loop and no session could read a test image. Restored to the real directory and
  the start-of-session command is now guarded against being run in the shared checkout. **Lost in the process:
  the `frames/hdr/` 10-bit material**, which was generated rather than fetched — regenerate it with
  `scripts/png16.py` if a 10-bit measurement is needed. Nothing else was lost, and no committed
  result depended on it.

- **BUG-11 + BUG-12** — Rice's stream mapping is now tile-width-aware (column-major, cut into 256
  contiguous segments) and the wavelet-level ceiling now follows the tile size actually in use.
  **Tile 256 output is byte-identical at all 12 measured points**, so nothing in BASELINE.md moves
  and no measurement taken at the default tile size is affected. What *is* invalidated: **every
  tile-size result in this repo, #47 and this morning's sweep included** — all of them scored the
  larger-tile arm through a coder that penalised it by 13–19%. Corrected figure: 512 over 256 is
  **−0.91% BD-rate**, so the geometry is worth ~1% and the default stays at 256. Also closed:
  6 wavelet levels at tile 512 is worth −0.1%, so deeper decomposition is not a lever either.
  **Use `CodecConfig::set_tile_size()`, never assign `tile_size` directly** — the level ceiling
  depends on it.

- **BUG-13 — FIXED 2026-09-06, and this bullet said "filed" for a day after it was closed.**
  BACKLOG is the current record: the reconstruction bug is fixed, block intra prediction was then
  measured, and it **costs 4-8% at lossless** because it predicts *and then* transforms, handing
  the wavelet a harder signal. That refutes prediction *before* a wavelet and says nothing about
  prediction *instead of* one, which is what INTRA-NEARLOSSLESS is gating. Original text follows,
  for the diagnosis in it: `GNC_INTRA_PRED=1` produced corrupt output at every quality: max
  error 197-255 from q=50 to q=100, and at q=100 it loses 62 dB against a bit-exact baseline.
  **The historical "-11.76 dB / +29%" measurement that set `intra_prediction: false` was measuring
  this bug**, not the idea. Error accumulates toward the bottom-right of every 32x32 block, which
  is an encoder/decoder predictor mismatch. Nothing else is invalidated -- the feature has always
  been off by default -- but the *conclusion* recorded against it is.
- **q=100 verified bit-exact lossless** on all three entropy coders. GOALS' "no true lossless with
  Rice" was stale and is corrected. GNC beats JPEG 2000 lossless by 10.8% and PNG by 7.8%; loses to
  FFV1 by 27% and x264 `-qp 0` by 43%, both of which predict against the neighbour.
- **BUG-21 and BUG-22 were assigned within minutes of each other on 2026-09-07, and it is
  resolved.** The `coord` session filed BUG-21 for the `abac_bitstream` parallel flake; the
  `abacship` session filed a different BUG-21 for an encode-path divergence on the inter path.
  **BUG-21 keeps the flake** (it was on main first) and is now **FIXED** — the `abacship` commit
  root-causes it as an `std::env::set_var` race that was masking a real decoder bug, which is the
  second of the two possibilities BUG-21's own entry asked someone to distinguish. The inter-path
  divergence is **BUG-22** (todo, P1). This is the second double-assignment in two days; read this
  file *and* run `scripts/claim list` before taking a number.

- **BUG-16 is taken** (2026-09-07, filed by `abacship`): Rice's GPU and CPU encode paths disagree
on the coefficients at q≤30 and at subsampled chroma. Unclaimed, on the default path, P2.

**Note on numbering:** BUG-11 was assigned twice on 2026-09-06 (Rice tile width, and intra
  prediction). The intra one has been renumbered **BUG-13**. Check this file before taking a number.

**ENT-2 — the entropy coders are measured against each other on one commit, and the answer is
"level". Invalidates no measurement; corrects four documents.** rANS against Rice, mean of four
stills, negative meaning rANS is smaller: **−6.4% at q=10, −7.1% at q=15, −6.7% at q=20, then
+0.4% at q=25 and +0.1% at q=40/55**. Above q=25 they are level and *which* coder wins is
content-dependent (−5.9% touchdown to +8.2% kristensara at the same setting). **No default moves**;
decision record 0018 records why, and 0015's prediction that "Rice still wins, and by more" is
falsified.

Four things to carry:

- **`--rans` is not a no-op**, contrary to BUG-9's entry — only the flag's `--help` text was stale,
  and that text (in five subcommands) called rANS the default and Rice "~30% worse compression".
  Verified at bitstream level: the harness parses `entropy_type` out of the GP17 header rather than
  trusting the flag, 40/40 points correct. **If a measurement of yours picked a coder with that
  flag, it did take effect.**
- **rANS's ceiling is q=75, not q=78.** q=75 encodes on all four images, q=77 on none, q=76 splits
  by content. The panic names stream **32** and a 3712-byte overrun, not stream 320. Handed to the
  BUG-9 owner; they report their fix is byte-identical on output (36/36 md5) and changes only the
  failure mode, so **no ENT-2 rate moves** — the q≥77 rows become a clean refusal rather than a
  crash, and rANS still stops below the operating point by design.
- **The q=20 coder cutoff and the 4→5 wavelet-level rule are coupled.** Every image's delta jumps
  in the same direction across q=25, because rANS pays a frequency table per subband group. Change
  one constant and the other is wrong.
- **The README's "1.5–2× faster" for Rice is gone, not corrected.** It contradicted TUNE-3's own
  ~15% by 3–6x, and re-timing needs an idle machine. That run is the one piece of ENT-2 left open.

Newest first. If you have measurements taken before one of these, they are suspect.

- **For the abac track, a result you did not ask for.** The `GNC_ABAC_COMPARE=1` harness was gated
  on `TransformType::Wavelet`, so it could not see the new lossless path. Gate widened (one
  `matches!`); `subbands()` at `num_levels = 0` already yields the right single region. **abac on
  MED residuals measures −14.3% mean on real coefficients** (bbb −14.9, blue_sky −15.6,
  kristensara −15.5, touchdown −11.2). With LOSSLESS-1 that is −27% against this morning's
  lossless and **+7.7% behind FFV1, from +48.4%**. Rate only. Your GPU decode throughput gate is
  now the only thing between that and a shipped result. Not implemented here — it is your code.

- **LOSSLESS-1** — q=100 now codes MED prediction residuals instead of wavelet coefficients
  (`TransformType::MedPredict`, bitstream `transform_type = 2`, `GNC_MED=0` to revert). Files are
  **14.9% smaller, still bit-exact**. **Invalidates every lossless figure in this repo**, including
  the "beats JPEG 2000 by 10.8%, loses to FFV1 by 27%" line — the FFV1 gap is now +25.8% measured
  as +48.4% before the change, so that comparison was against a different baseline than it reads.
  Also note `is_lossless()` now decides per transform type; anything keying on it should be
  re-read.

- **ENT-1** — subband-rANS frequency tables are now Exp-Golomb packed (bit 31 of the tile's
  `num_groups` word). Files shrink 11.7–26.6% on that coder at bit-identical quality; the preset
  path (q ≤ 20) shrinks 4.5%. **Invalidates every absolute byte or bpp figure for the rANS coder**,
  and it invalidates the *reason* for the Rice/rANS crossover at q=20 — re-sweep before quoting it.
  Landed inside commit `0e8987c`, which swept up another session's uncommitted work; the change was
  verified and measured only afterwards.

- **Tile-size sweep (BUG-11 / ENT-1 / BUG-12)** — measurement only, no code change. Result: 256 px
  is a local optimum and both 128 and 512 are worse *through Rice*, but the sign reverses on the
  rANS path (512 gains 14–20%). Rice hardcodes tile width 256 in its `i % 256` stream mapping.
  **Invalidates the conclusion of every past tile-size experiment, #47 included** — they were all
  scored through the coder that penalises the larger-tile arm. Also: `--tile-size` never reached
  its own level ceiling (BUG-12), so those runs were capped at 5 levels too.
- **TUNE-6** — the P-frame quantiser scale now follows the quantiser step (1.25× at step ≥ 4.6,
  tapering to 1.0 at step ≤ 2.8). Fixes BUG-10. **Invalidates any inter measurement taken between
  TUNE-5 and now at q > 80.**
- **abac (EBCOT part 3)** — new standalone module plus a `GNC_ABAC_COMPARE=1` diagnostic. No
  bitstream change, nothing wired into the pipeline. Invalidates nothing.
- **BUG-7** — `GNC_DIAGNOSTICS=1` was inflating encoded files by 32% by clobbering the
  motion-compensation reference. **Invalidates every sequence measurement ever taken with that
  flag**, including MEAS-4's residual dumps. `tests/diagnostics_neutral.rs` now guards it.
- **MEAS-2** — AQ off below q=30 (its gradient was inverted), reference deblocking off by default
  (measured neutral-to-negative). **Invalidates AQ and deblock measurements from before today.**
- **FMT-2 / GP17** — Rice-coded stream-length tables. Changes all file sizes; invalidates absolute
  byte figures, not ratios measured within one build.
- **BUG-6** — 5 wavelet levels at q ≥ 25. Changes the LL subband AQ measures variance on, and the
  per-group table count rANS pays for. **Any AQ or entropy-coder tuning from before this is
  stale** — that is how the rANS/Rice crossover came to sit exactly at q=25.

## Conventions worth keeping

- **Every new code path needs a canary** — a logged count or value proving it ran on real data.
  `GNC_DIAGNOSTICS=1` prints `groups=N deep_skipped=M`, `rice_tiles=N/N`, and so on. A feature
  without one is not done (CLAUDE.md).
- **A result that beats its own theoretical ceiling is a bug.** That is how the abac comparison's
  bad baseline was caught (−35% against a ceiling of −13.7%).
- **A toggle measured on the wrong metric reads as dead weight.** CfL looks like a loss on VMAF
  and is a clear win on CIEDE2000, because VMAF is luma-only. Anything touching chroma needs
  `scripts/chroma_metric.py`.
- **Point measurements at fixed q cannot judge a rate/quality trade**, and they always flatter the
  option that spends more bits. Use BD-rate, or compare at matched rate. Three separate wrong
  conclusions today came from this one error.

## Validate the artefact you ship, with the compiler you ship (2026-09-08)

BUG-25 spent five hypotheses and a session on "two unrelated drivers both die on SPIR-V that
`spirv-val` passes". The validation was real and the module was valid — **it was just not the
module that ships**. `naga` on the box's `PATH` is **30.0.1**; GNC depends on **naga 24.0.0**
through wgpu 24. Two compilers, one investigation, and no reason to suspect a gap.

**A tool on `PATH` is not the tool in `Cargo.lock`.** Before a shader-level or codegen-level claim,
check the version of whatever produced the artefact, and prefer emitting through the crate the
project actually links (`examples/bug25_emit.rs` does this — it links the same naga wgpu resolves
and reconstructs wgpu's own `spv::Options`). The naga-24 module for `block_match_split` is invalid;
the naga-30 one is not, and only one of them ever reaches a driver.

Same shape as the M5-vs-M1 finding (BUG-29) four hours earlier: an environment fact nobody wrote
down, silently invalidating a measurement that looked clean. Neither was carelessness; both were
invisible until something forced a comparison.

## Every session commits as the same git user, so authorship attributes nothing (2026-09-07)

Found by the session holding `gnc-bug25@bug25#s52348`, after I credited two of its findings to the
wrong sessions in one message — the `0024` decision-record collision (it is ENT-5's and INTRA-1's,
not theirs) and abac's 4:2:2/4:2:0 GPU-vs-CPU byte identity (ENT-5's).

**`git log --author` cannot separate us, and neither can a decision record's byline.** Eight
sessions, one git identity, one machine. The only reliable attribution is **the claim ref**
(`gnc-abacgpu@abacgpu#s53810`, `gnc-bug25@bug25#s52348`) or **the item id in the commit subject**.
Use one of those when you attribute a finding, and never a session's chat name — names are assigned
per connection and do not survive.

This is the id-collision problem in a third namespace: a field that looks like it identifies
someone and does not. The cost is not credit, it is that the next reader asks the wrong session to
expand on a measurement it never took, and starts their trail in the wrong file.

**And the practical habit that fell out of it: write the artefact first, then write the message
from it.** The two misattributions above were wrong in a chat message and right in the committed
file, from the same session in the same minute — because the file was written while looking at
`ls docs/decisions/` and the message was written from memory. Prose composed from recall is where
this fails; prose composed from the artefact is not.

## `BUG-25` was used twice, by two sessions, for two different defects (2026-09-07)

The claim mechanism excludes sessions from an *item*; it does not stop two sessions inventing the
same *id*. Both read `main:BACKLOG.md`, saw BUG-24 as the highest, and numbered the next one 25.
This is BUG-19's problem (colliding decision-record numbers) in a second namespace, and it will
recur until an id is allocated by the same compare-and-swap that hands out work.

| id | defect | state |
|---|---|---|
| **BUG-25 (on `main`, `92f4d3a`)** | GNC does not run on Vulkan; `block_match_split.wgsl` kills the NVIDIA driver and lavapipe | committed to `main`, referenced in RESEARCH_LOG and two commit messages |
| **BUG-25 (worktree `gnc-inter1`)** | P-frame local-decode dequant uses the intra qstep, not `res_qstep`, so the encoder's reference diverges from the decoder whenever `p_qp_scale != 1.0` | **renumbered to BUG-27 and pushed 2026-09-07 — resolved, and fixed** |

**Resolution: the one on `main` keeps the number; the dequant defect takes the next free id.** Not
because it is more important — the dequant defect looks like the more valuable find, and it is
adjacent to BUG-18's open cause 2 — but because renumbering text that is already public, and
cross-referenced from a log and two commit messages, costs more than renumbering text that is still
in a worktree. First-pushed wins, on the same reasoning as the RATE-2 reconciliation.

**You will meet this as a merge conflict in `BACKLOG.md` on the `### BUG-25` heading**, which is the
notification working correctly: it arrives when you push, not when someone guesses which terminal
you are. Take both bodies, renumber the dequant one, and check whether it is BUG-18 cause 2 seen
from the other side before filing it as separate.

**Done, 2026-09-07, by `gnc-inter1`: the dequant defect is `BUG-27`, and it is fixed.** The
conflict arrived exactly as predicted — on this section, in `COORDINATION.md` and `BACKLOG.md`, at
rebase time — and the recipe worked: both bodies kept, mine renumbered. It is **not** BUG-18 cause
2: BUG-18 was the *forward quantise* on the path ARCH-3 has since deleted, this is the
*dequantise* that rebuilds the encoder's reference, and it was present in **both** P-frame
implementations, so deleting one did not remove it. `a312d6f` and this fix are the same defect
found from two directions — by reading two implementations against each other, and by a
measurement that refused to make physical sense — and neither of us had fixed it until now.

**And note what the claim then says.** `BUG-25` is held by `gnc-inter1` for the dequant defect, so
the Vulkan work proceeded under `worktree.gnc-bug25` alone. That is a real gap in the exclusion, not
a licence: two sessions holding one id for two defects means the lock protected neither.

**Update 2026-09-07, `arch3`: the number has moved again, and the same way.** "The next free id"
was 26 when this was written; a `BUG-26` went to `main` (abac vs Rice pixels on subsampled chroma),
filed by a session that had reserved `BUG-26` with `scripts/claim take` and could not see a
renumbering that lived only in a worktree. **The dequant defect takes `BUG-27`.**

**And then 26 collided again, within the hour.** `intra1` filed its own `BUG-26` (`--tile-size
1024` silently destroys the image, P1) and claimed it, at which point one `scripts/claim take
BUG-26` covered *two* headings on `main` and neither defect could be claimed on its own — the
lock stopped excluding, which is worse than the duplicate heading. **Resolved by renumbering the
abac/Rice one to `BUG-28`**, on a different tiebreak from the one above: both were public, so
"first-pushed wins" no longer separates them, and what did separate them is that intra1's was
*held and being worked on*. Renumber the idle one; never renumber inside a live worktree. Reserving the id
was the right move and it was not enough: the claim excludes another *taker* of that id, not a
pending rename of an id nobody has taken yet. The fix is the one this section already names — the
id has to come *from* the compare-and-swap rather than be checked against it — and until it does,
the cheap habit is to **`scripts/claim take` the id you intend to use before you write the
heading, and to push a filing quickly rather than holding it in a worktree**, because an id that
exists only locally is invisible to exactly the mechanism that would protect it.

**Independent confirmation of the dequant defect, from `arch3`:** the local decode dequantises P
residuals with `config.quantization_step` while the forward pass uses `res_qstep`. Found while
reading both P-frame implementations for ARCH-3 — *both* did it, so it is not an artefact of
either. `a312d6f`'s commit message misattributes it to BUG-8, which is closed and was a metric
bug; the docs are corrected, the commit message cannot be.
