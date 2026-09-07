# 0022 — rANS's ceiling is a table, not a slot, and it stays where it is

**Date:** 2026-09-07
**Status:** accepted
**Item:** BUG-9

## Context

BUG-9 recorded rANS as crashing at fine quantiser steps because streams overran a fixed 4 KB
output slot, and explicitly ruled out the symbol alphabet as the cause. It asked for the cheap
half of a fix — turn the wrapped-pointer crash into a sentence — and argued against the expensive
half, runtime buffer sizing, on the grounds that rANS measures no better than Rice in the range it
would unlock.

Implementing the cheap half showed the diagnosis was inverted. The binding limit is the cumfreq
table: every subband group's table for a tile shares one workgroup array, and it is the sum over
the tile's groups that has to fit. Measured with `--rans` at the default step, no stream overflows
its slot at any point that completes — the tables give out first, every time, on every content
tried. The slot overflows the entry recorded were streams coded with frequencies read from outside
that array.

## Decision

Three things, and one deliberate omission.

1. **Refuse a frame whose tables do not fit, by name, on the host.** Not only clamp in the shader.
2. **Bound the shader's writes** so `write_ptr` cannot wrap into the previous stream's slot and
   the cumfreq array cannot be indexed past its end.
3. **Size the shared array `MAX_ALPHABET + 1`,** which is what a table of n symbols actually needs.
4. **Do not add runtime buffer sizing.** rANS still cannot encode above q=75.

## Why a host refusal and not just a shader clamp

Because the two failures are not equally visible. A slot overflow announces itself: the host reads
a wrapped `write_ptr` and cannot miss it. A table overrun need not — a tile can overrun its tables
and still emit streams that fit their slots, and then nothing downstream notices and the file is
quietly wrong. Clamping alone would have converted undefined behaviour into *defined* wrong
output, which is worse than the crash it replaces. The host check is what makes the failure
impossible to ship.

The same reasoning is why the shader clamp is not enough on its own even though it removes the
undefined behaviour: correct-by-clamping still codes with the wrong frequencies.

## What was rejected, and what it would have cost

**Runtime buffer sizing** — threading a computed stream-buffer size through three encode shaders
and six allocation sites, the way Rice already sizes its own from qstep. This is the fix that
would let rANS reach the contribution operating point. Rejected, and the rejection is now measured
from both sides: BUG-9 argued it from Rice-vs-rANS rate, and ENT-2 measured the coders level at
q=25–70 with rANS 6–7% smaller only at q<=20, where qstep is 32 or coarser and the worst tile asks
for 361 of 4097 entries. The range this unlocks is one where the coder measures level at best.
Cost if someone reverses this: real, and the entry point is `check_cumfreq_capacity`.

**A qstep-keyed guard or fallback** — rejected before implementation, and the measurements say
why. The ceiling is not a qstep threshold: at the default step it sits at q=76 for kristensara and
blue_sky and q=77 for bbb and touchdown, because what crosses is the Y-plane alphabet and that
depends on content. Any static threshold either refuses working configurations or misses broken
ones.

**Returning a `Result` instead of panicking** — the backlog's other suggested route. The encoder
has no error type and five public `encode_*` methods return plain values; introducing `Result`
through them for a P3 guard is a larger API change than the defect warrants. The panic carries the
tile, the count, the capacity and the remedy, which is what "a sentence instead of a crash" asked
for. Revisit if the encoder ever grows an error type for other reasons.

## Consequences

- Nothing in any working configuration moves: byte-identical to the parent commit on 64/64 points,
  including `--rans` at q=75, the tightest passing point.
- `--rans` above q=75 changes from a wrapped-pointer panic to a named refusal. It never produced a
  usable file there.
- `--no-per-subband` at a saturated alphabet now works instead of overrunning by one entry.
- Two canaries per plane under `GNC_DIAGNOSTICS=1`, so the margin is observable rather than
  assumed.
- BUG-9's "this is not the symbol alphabet" is corrected in place, with the original reasoning
  kept — most of it was right, and the part that was wrong is instructive.

## Note on numbering

Decision record numbers collided twice on main today (two 0018s, two 0019s) because the number is
picked by reading the directory, which is the same read/decide/write race `scripts/claim` was
built to remove for backlog items. This record was written as 0021, which was free at the time;
by the time it was ready to push, main had taken 0021 for the shared-compilation-cache record.
Renumbered to 0022 — the third collision today, on the same afternoon someone built a compare-and-
swap to stop exactly this for backlog items. The convention needs the same treatment; the commit
message of the parent commit still says 0021 and is wrong on that one detail.
