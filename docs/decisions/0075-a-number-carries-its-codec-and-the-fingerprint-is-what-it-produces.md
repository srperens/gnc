# 0075 — A number carries its codec, and the fingerprint is what the encoder *produces*

**Date:** 2026-09-08
**Status:** accepted; `gnc fingerprint` ships, COORD-6 closes with a mechanism
**Item:** COORD-6 (P4)
**Answers a question COORD-4 left open**, and refuses the same two tools it refused

## The question COORD-6 asked

COORD-4 counted the most frequent measurement failure in this repository — a number read against
the wrong tree — and **refused the tool it was filed to consider**: a claim-time `HEAD` stamp would
have caught 1 of 6 instances, printing each claim's commit 0 of 6. It shipped consolidated prose
instead. COORD-6 asked one thing: **is there a cheap mechanism, or should this class be accepted as
a cost of eight-session concurrency?** Both answers close the item. A third round of prose does
not.

**A seventh instance arrived while COORD-6 sat in the queue, and it arrived *after* the prose was
consolidated onto `main`.** LOSSLESS-3 (`0070`) published a q=95/97/99 sequence table against a
bit-exact q=100 column and concluded that camera content is dominated from q=95 up. Those lossy
columns contain bit-exact I-frames — RATE-3 put them there — so BUG-47 (`0072`) moved every one of
them by roughly 1.8 points hours later while the q=100 column stayed put, because there is no
sibling at q=100. Its P1 conclusion survives; every margin in the published table is overstated.

That puts the tally at **5 of 7 for one shape** — a correct measurement decaying because `main`
moved under it — against 1 of 7 for the cross-session shape COORD-4 refused a tool for. And it
demonstrates that the prose does not prevent the class, which cuts both ways: it is the strongest
argument for a mechanism *if* a cheap one exists, and the strongest argument for closing
answered-no if none does.

## The decision

**`gnc fingerprint` ships.** It encodes a pinned, versioned 10-configuration matrix and digests the
resulting bytes:

```
$ gnc fingerprint
codec-fingerprint v1 abf86a50  (10 configurations)
```

**Two numbers carrying the same fingerprint are comparable; two carrying different ones are not.**
It answers the only question that matters — *would this binary produce different output?* — by
running the encoder, which is the honest test the item said was too expensive. It is not: **0.52 s**
for the whole matrix, against the minutes a real sweep costs.

Two uses, and a harness wants both. **Print it beside the numbers**, so a later reader can tell
whether a published table is still comparable — instances 2, 3, 5, 6, 7. **Take it before *and*
after a sweep**, so a mid-run rebuild is a refusal rather than a plausible table — instance 4, the
near-miss this repository got away with by luck. `scripts/fingerprint.py` is the two-line harness
helper; a bare shell loop needs no helper, and COORDINATION shows both.

## Why this is not the tool COORD-4 refused

COORD-4's candidates were both *proxies for* an output change: a commit id, or a dirty bit. This is
the output change. The distinction shows up as a false-positive rate, and the false-positive rate
is why checks get ignored.

**`shasum target/release/gnc`, which several harnesses already print, is the proxy to beat.** It
changes when a doc comment does. Measured while building this: adding an entire module and editing
`main.rs` moved the binary hash from `94f25712…` to `333e2c62…` and left the fingerprint at
`abf86a50`, because the encoder's output had not moved. A check that fires on every rebuild trains
its reader to skip it.

Validated in the other direction too, against knobs whose effect was already measured elsewhere:

| knob | does output move? | fingerprint |
|---|---|---|
| `GNC_REF_FROM_SOURCE=0` | **no** — 24 of 24 sequence points byte-identical (`0072`) | **unchanged** |
| `GNC_PAD_FILL=decay` | yes (`0039`) | changed |
| `GNC_DEAD_ZONE=0.3` | yes (`0028`) | changed |
| `GNC_REF_DEBLOCK=1` | yes | changed |

So it is silent on the one knob known to be output-neutral and it fires on all three known to move
output. That is the property the `src/`-touched proxy the item suggested cannot have: `0045`'s
diagnostic-only change is byte-identical with its env var unset, and a `src/` proxy warns on it.

**Would it have caught the seven?** Instances 2, 3, 5, 6 and 7 are published tables whose rows came
from different encoders — caught, *provided the rows carry the fingerprint*, which is the adoption
condition and is why the harnesses were changed rather than only the docs. Instance 4 is a rebuild
mid-sweep — caught by the before/after check, mechanically, with no reader involved. Instance 1 is
the cross-session patched-tree case — caught, since a patched encoder has a different fingerprint.
**That is a better rate than either refused tool, on the same instance list.** It is also the
optimistic reading, and the pessimistic one is worth stating: none of it fires unless a number
carries the token.

## What was rejected

- **A claim-time `HEAD` stamp in `scripts/claim`** — COORD-4's own numbers, 1 of 7. A claim is
  taken when an item is picked up; a measurement happens somewhere else entirely, often hours
  later and several merges on.
- **"Did the merge touch `src/` or `src/shaders/`?"** — the cheap proxy COORD-6 itself suggested.
  It over-warns on diagnostic-only changes (`0045`) and, worse, it answers a question about the
  *diff* when the question is about the *output*. The fingerprint costs 0.52 s and is exact on what
  it covers.
- **Hashing the binary** — see above. Over-warns to the point of being ignored, and this session
  produced the demonstrating pair while building the alternative.
- **Closing answered-no.** That was the live option and instance 7 is its best argument: prose had
  just been consolidated and did not prevent it. It is refused because the mechanism turned out to
  cost half a second, which is below the threshold at which "accepted cost of concurrency" is an
  honest description.
- **Making the check mandatory** — a hook, or a `claim` refusal. Nothing enforces it and nothing
  should yet: the matrix's coverage is unproven outside the four knobs above, and a mandatory check
  believed past its range is worse than an optional one read with judgement.

## What it does not do

- **It cannot say why two fingerprints differ.** It says they do, and that a comparison across them
  is not one.
- **It says nothing about a path outside its matrix.** The matrix crosses entropy coder (Rice, rANS,
  abac), chroma format (4:4:4, 4:2:0), the lossless boundary (q=50/90/99/100) and inter (3-frame
  sequences at ki=2), and it is not exhaustive. A change that moves output only elsewhere will not
  show. The output says so, every time it prints.
- **It protects only numbers that carry it** — the same adoption problem prose has, met with one
  command instead of a paragraph, and with two harnesses already doing it so the next one can copy.
- **The matrix is pinned.** Changing it changes every fingerprint ever published, so it carries
  `MATRIX_VERSION` and a change to it is a decision record, not a commit.
- **It is encoder-side.** The class is about encoder output; a decoder fingerprint is a different
  and mostly redundant instrument, since a moved decoder that still decodes the same bytes to the
  same pixels is not what makes two tables incomparable.

## Two things the implementation had to get right, recorded because both were wrong first

- **Determinism is the whole product.** Asserted, not assumed:
  `fingerprint_is_deterministic_and_every_row_is_a_distinct_sample` runs the matrix twice in one
  process and compares every row.
- **Every row must be a distinct sample.** The first input generator was hash noise, on which the
  bit-exact candidate wins every frame — so the q=99 and q=100 sequence rows coded to *identical
  bytes* and one of the ten configurations was measuring what another already had. The content is
  now smooth-plus-texture, and the test fails if any two rows collide. It is also integer-valued,
  because BUG-45: a fractional source makes a lossless configuration quietly lossy, and a
  fingerprint whose lossless rows are secretly lossy would be measuring that instead.
