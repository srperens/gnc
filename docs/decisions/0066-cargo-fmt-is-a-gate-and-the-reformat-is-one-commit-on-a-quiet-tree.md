# 0066 — `cargo fmt` stays a rule, and the reformat is one commit on a quiet tree

**Date:** 2026-09-08
**Item:** BUG-38
**Status:** accepted — rule kept, mechanical half parked on a checkable precondition

## Context

GOALS §9 requires code to pass `cargo fmt` and `cargo clippy` with zero warnings. BUG-20 settled
the clippy half (`0062`). The `cargo fmt` half has been **false for an unknown length of time**:
`cargo fmt --check` reports **573 diffs in 61 of the 90 `.rs` files**, 504 of them under `src/`.
Neither CLAUDE.md nor LOOP.md ever named it as a gate, so nothing ran it.

BUG-38's entry offered two options — run `cargo fmt` and gate on it, or drop the `cargo fmt` half
of the rule. Both were priced, and so were two others that only appeared once there were numbers.

## What was measured

**No rustfmt configuration makes this cheap.** The first hypothesis was that the tree is written
in a consistent wider style and that a `rustfmt.toml` matching it would collapse the diffs into a
one-file config change with no conflicts. **Falsified, and in the opposite direction** — every
deviation from rustfmt's default is worse:

| config | diffs |
|---|---|
| **none (rustfmt default)** | **573** |
| `fn_call_width = 80` | 646 |
| `max_width = 90` | 964 |
| `max_width = 100` (= the default) | 573 |
| `use_small_heuristics = "Max"` | 1114 |
| `use_small_heuristics = "Max"`, `max_width = 110` | 1358 |
| `use_small_heuristics = "Max"`, `max_width = 120` | 1518 |
| `use_small_heuristics = "Off"` | 1053 |

So the tree is *closest* to plain rustfmt default and the 573 is genuine drift, not a house style.
There is no config to adopt and `rustfmt.toml` should not be added.

**The drift is ongoing and measurable.** 566 diffs when BUG-38 was filed; **573** seventy-five
minutes later, with `main` having moved five times in between. Nothing regressed — that is just
what an unread gate does, the same mechanism as clippy's 88 → 90 → 91 in `0062`.

**The dirty files are the hot files, which is what kills the cheap options.** Of the 61 fmt-dirty
files, **44 were changed on `main` in the last 24 hours** — a 72% overlap — and 19 `.rs` files are
uncommitted in some worktree right now. The two dirtiest files in the tree, `src/main.rs` (55) and
`src/decoder/pipeline.rs` (52), sit alongside `src/encoder/pipeline_tests.rs` (37) and
`src/encoder/rice.rs` (22), both of which two *other* sessions committed to during this item.

**A genuinely cold subset exists and is not worth taking.** Intersecting out everything changed on
`main` in 24 h and everything dirty in any worktree leaves **16 files carrying 59 of the 573 diffs
— 10%**. Formatting those would leave `cargo fmt --check` red, so the gate still could not be
adopted, and it would add a third state to the tree for no stated invariant.

## Decision

**Keep the rule. Do the reformat as one atomic commit, on a quiet tree, and only then add the
gate.** The value here is binary: `cargo fmt --check` is either clean, in which case it can go
into LOOP.md step 5 and cost about a second per run with no compilation, or it is not, in which
case the rule stays decorative. Ten per cent of the way is worth nothing.

BUG-38 is therefore **parked** rather than done, with a precondition that is checkable rather than
a feeling:

```bash
scripts/claim list | grep -v '^  worktree\.'   # no item held but this one
git worktree list | wc -l                      # and every worktree clean:
for wt in $(git worktree list --porcelain | awk '/^worktree /{print $2}'); do
  git -C "$wt" status --porcelain; done         # must print nothing
```

Then, in one commit that changes nothing else:

```bash
cargo fmt                                      # no rustfmt.toml — the default is the best fit
cargo test --release -- --test-threads=1        # rustfmt cannot change semantics, but prove it
cargo clippy --release --all-targets
git commit -am "fmt: one pass over the tree, no behaviour"
git rev-parse HEAD >> .git-blame-ignore-revs    # so `git blame` still reaches the real authors
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

The `.git-blame-ignore-revs` line is not optional. A 573-diff commit across 61 files otherwise
becomes the blame answer for a quarter of the codebase, and this project reads history constantly
— five of the corrections recorded this week were found by asking when a figure was written.

## What was not chosen

- **Dropping the `cargo fmt` half of GOALS §9.** This is the cheapest option and it was close.
  Against it: **rustfmt cannot find a defect, but an unformatted tree can hide one.** Any
  contributor or session with format-on-save silently reformats the files it touches, mixing
  incidental whitespace into a semantic commit — and this project's whole method is reading what
  changed. `0062`'s argument does not transfer (clippy found two real things; rustfmt finds none
  by construction), so the case for the rule is diff legibility, not defect detection. That is
  enough here, where the diff *is* the evidence.
- **A `rustfmt.toml` fitted to the tree.** Measured above: there is nothing to fit. This was the
  hypothesis that would have made the item free, and it is dead.
- **Reformatting now.** 573 diffs across 61 files, 44 of which `main` touched today, against seven
  live sessions. It is one merge conflict per session in the files they are holding, for zero
  behaviour. The same commit on a quiet tree costs nobody anything, so the only thing paying now
  buys is finishing an item sooner.
- **Making the rule per-touched-file** — "the files you change must be `cargo fmt` clean". This
  looked like the best answer before the overlap was measured: incremental, no big-bang, converges.
  The 72% dirty∩hot overlap is what refutes it. It does not avoid the conflicts, it **distributes**
  them over the same files, and it does so by mixing a reformat into every semantic commit that
  touches a dirty file — which is precisely the hazard the rule exists to prevent. It would also
  have hit BUG-20 itself: of the twelve files that item touched, `rice.rs`, `rice_gpu.rs` and
  `pipeline_tests.rs` are all fmt-dirty and all three were committed to by other sessions the same
  afternoon.
- **Formatting the 16 cold files.** 10% of the diffs, gate still red. Priced and declined above.

## What this costs while parked

`cargo fmt --check` stays red and GOALS §9 stays half-false, so the annotation added to it in
BUG-20's commit — naming the number and pointing here — is load-bearing and should not be tidied
away. Drift continues at roughly the rate above, which makes the eventual commit slightly larger
and changes nothing else: rustfmt is idempotent and order-independent, so a later reformat is not
harder for having waited.
