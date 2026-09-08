# 0067 — The decoder should reject malformed input, and the mechanism is a checked cursor

*2026-09-08. ROBUST-1. Supersedes nothing; it decides a question that had never been asked.*

## The question

`deserialize_compressed_validated` opens with `assert!(data.len() >= 37, "File too small")` and
then indexes with `data[pos..pos + 4].try_into().unwrap()` for the rest of its length. **That is
a contract, written in code: a GNC decoder panics on input it cannot parse.** Nobody chose it —
it is what writing a parser with slice indexing produces — and nothing in the project says
whether it is intended.

It matters because of what GNC is for. GOALS §1 positions this as a *contribution* codec: its
decoder's input arrives from somewhere else, over a network, from a device, from another
vendor's encoder. A library whose documented behaviour on a corrupt frame is "the process dies"
pushes the problem to every embedder, and pushes it somewhere they cannot fix it.

And **the per-tile CRC-32 does not cover this, though its name suggests it might.** The CRC is
verified *after* the header is parsed. It is error resilience for bit rot in a stream that is
otherwise well-formed; it is not input validation. Anyone reading GOALS' "error resilience
(per-tile CRC-32)" as "malformed input is handled" is reading it wrong, which is a good reason to
settle this explicitly.

## What was already fixed, so the decision is not about these

Three defects found in the ROBUST-1 audit and fixed before this record, because each was a bug
under *any* contract:

- **BUG-43** — `k` came off the wire as a raw byte and was used as a shift distance
  (`1u32 << k`, `read_bits(k)`, `1u << shared_k[g]`). A WGSL shift of ≥32 is undefined. Clamped
  to `RICE_MAX_K = 15`, which is already the most the format can express.
- **`read_tile_varint`** shifted by up to 77 on a run of `0x80` bytes — a panic with overflow
  checks on, a silent wrong answer with them off, and an unbounded advance of the parse position
  either way. Bounded to 3 bytes, which is what a `u16` varint can occupy.
- **Four wire counts reached `Vec::with_capacity` unbounded**, so a packet-sized file claiming
  `u32::MAX` records aborted the process on allocation. Bounded by `wire_count` against what the
  remaining buffer could hold. `alpha_count` also overflowed `u32` before being compared to
  anything.

**So the amplification is gone and only the panic remains.** A malformed frame now runs off the
end of its buffer and panics, instead of aborting the process on a huge allocation. That is
strictly better and it is not enough.

## The decision

**Reject, don't panic — via a checked cursor and an additive API.** Concretely:

1. A `Cursor<'a> { data, pos }` with `u8() / u32() / f32() / bytes(n) -> Result<_, DecodeError>`,
   each checking against `data.len()` once, in one place.
2. `try_deserialize_compressed(data) -> Result<CompressedFrame, DecodeError>` as the real parser,
   mechanically rewritten onto the cursor.
3. `deserialize_compressed` **stays**, as `try_deserialize_compressed(data).expect(...)`.

## Why this shape, and what the alternatives cost

**Additive, not a signature change — and that is a measured claim, not a hope.**
`deserialize_compressed` has **33 call sites** (`src/lib.rs` ×10, `src/main.rs` ×2, and the rest
in tests). Changing its return type touches all 33 and every one of them would have to decide
what to do with an error, in the same diff that rewrites the parser. Keeping it as a panicking
wrapper leaves all 33 compiling untouched, so the parser rewrite can be verified on its own —
and the interesting property is that **the wrapper's behaviour is exactly today's contract**, so
the change is provably behaviour-preserving for existing callers.

**Rejected: a validating pre-pass** (ROBUST-1's original option (b), and it looked cheapest).
A pass that walks the header bounding every length before the parser runs is **a second parser
that has to agree with the first.** Two parsers that must agree is a new class of bug, not the
absence of one — and the failure mode is the worst kind, where the validator accepts what the
parser then misreads. What survived from (b) is the part that needs no second parse: `wire_count`
bounds each count *at the point it is read*, inside the one parser. That is already landed.

**Rejected: `catch_unwind` at the boundary.** It converts a panic to an error in about ten lines,
and it is wrong twice. It cannot distinguish "this input is malformed" from "this decoder has a
bug", so it would silently turn our own defects into `Err`, which is precisely the class of thing
the project's rules exist to surface. And it does nothing under `panic = "abort"`, which an
embedder may well set.

**Rejected: document panic-on-malformed as the contract** (option (c)). It is honest and it is
free, and for a contribution codec it is the wrong answer: it makes every embedder sandbox the
decoder or validate the bitstream themselves, which means reimplementing the parser outside the
library. The one thing worth keeping from (c) is its documentation: until (a) lands, the contract
**should be written down**, because right now it is only implied by an `assert!`.

## Cost, stated so it can be argued with

The parser is ~400 lines of a single function, and the rewrite is mechanical but not small. It
also touches `format.rs`, which several sessions edit. **That is why this record exists without
the implementation:** the decision is cheap and the diff is not, and separating them means the
next session inherits a specified job rather than an open question. Filed as ROBUST-2.

## What would change this decision

Evidence that no shipped configuration can hand the decoder untrusted bytes. Given GOALS §1 and
a container format with keyframe seeking and error resilience, that evidence does not exist.
