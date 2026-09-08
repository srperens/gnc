# 0078 — The candidate must be honest about its format, and the fallback must then refuse what it cannot compare

Date: 2026-09-08
Item: BACKLOG BUG-46 (P3)
Status: accepted
**No output moves.** Byte-identical to `main` on every checked point. What changes is that a
refusal which used to happen by accident now happens on purpose, and says why.

## Context

`lossless_sibling` built the bit-exact candidate from `quality_preset(100)` — which is 4:4:4 — and
carried `entropy_coder`, `gpu_entropy_encode`, the abac knobs, `tile_size` and (since BUG-47)
`pad_fill_decay`, but **not `chroma_format`**. The RATE-2 canary on bbb at q=97 therefore reported
the same candidate for every request:

```
--chroma-format 444:  lossy 2 843 371 B vs bit-exact 3 257 157 B (+14.55%), keeping the lossy one
--chroma-format 420:  lossy 1 660 019 B vs bit-exact 3 257 157 B (+96.21%), keeping the lossy one
```

So on subsampled input RATE-2 compared a 4:2:0 wavelet encode against a **4:4:4** lossless one,
three times the chroma samples, and the fallback could essentially never fire off 4:4:4. Every
figure RATE-2, RATE-3 and LOSSLESS-3 report is, in consequence, a 4:4:4 figure — not "unmeasured
elsewhere", but *unreachable* elsewhere.

## What the one-line fix turned on, and why it could not ship

`out.chroma_format = cfg.chroma_format` makes the candidate honest, and the fallback then fires on
subsampled input: bbb at 4:2:0 q=99 takes the bit-exact file, 1 847 304 B against 1 898 635 B,
**−2.70% of rate**.

**And −3.9 dB of picture.** RATE-2's whole justification is that the bit-exact candidate wins on
*both* axes when it wins at all. On subsampled chroma it does not, because `q=100` there is **not
lossless — not even in luma**, which subsampling does not touch. Per plane against the source:

| still | format | q | y | u | v |
|---|---|---|---|---|---|
| blue_sky | 4:2:0 | 100 | **51.16** | 43.23 | 44.74 |
| blue_sky | 4:2:0 | 95 | 53.09 | 56.73 | 57.88 |
| kristensara | 4:2:2 | 100 | **51.04** | 43.63 | 43.54 |
| bbb | 4:2:2 | 100 | **47.34** | 37.10 | 38.71 |

4:4:4 at q=100 is exact (`inf`). In whole-image RGB, q=100 reads **8.5–13.1 dB below q=95** at the
same format on every image measured, and 4:2:2 comes out worse than 4:2:0 on three of four — an
inversion, since 4:2:2 keeps twice the chroma. That is **BUG-49**, filed P1 with the places to look.

## The decision

**Fix the sibling and refuse the comparison.** Two separate things, and both are needed:

1. `lossless_sibling` carries `chroma_format`. The candidate is what it claims to be, in every
   format, and would carry the caller's format into the file if it ever won.
2. `EncoderPipeline::encode` refuses the fallback when `chroma_format != Yuv444`, with a canary
   naming the format and BUG-49. Not a silent skip: the refusal is the interesting path.

**Byte-identical to `main`** on bbb at 4:4:4 / 4:2:2 / 4:2:0, q=97 and q=99. The refusal restores
exactly the behaviour the missing field produced by accident.

## Why refuse rather than take the rate

Because −2.70% for −3.9 dB is a rate/quality trade, and RATE-2 (`0036`) exists precisely because
its own trade needs no metric: *"when the lossless candidate wins it wins on both axes at once,
which is why this needs no rate/quality trade-off rule."* Shipping the fix without the refusal
would smuggle a trade into a mechanism whose licence to act without a metric comes from there not
being one. CLAUDE.md's tolerance for a PSNR regression is 0.3 dB "flag and investigate"; this is 13
times that.

The same reasoning is why LOSSLESS-3 (`0073`) is gated to 4:4:4. That gate cited BUG-46; it now
cites BUG-49, which is the actual obstacle.

## What was not chosen

- **Ship the fix as-is and record the trade.** Refused above. It would also make `--chroma-format
  420` silently change what "q=99" means for quality, which is the one thing a contribution codec
  must not do quietly.
- **Leave `lossless_sibling` alone.** Cheapest, and it keeps a defect that is invisible until
  someone reads the canary twice. The accident and the intention happen to agree today; when
  BUG-49 is fixed they stop agreeing, and the accident would keep the win locked away.
- **Fix BUG-49 here.** Out of scope for a P3 that started as a one-line field copy: the luma
  damage is format-dependent and points at plane geometry in the MED path (BUG-11 / BUG-14's
  class), which needs its own measurement and its own success criterion. Filed with both.
- **Refuse in `lossless_sibling` instead** (e.g. return the config unchanged for subsampled
  input). Wrong layer: the sibling's job is to describe a candidate, and a caller that wants one
  for a legitimate reason — a diagnostic, BUG-49's own fix — should get it.

## Consequence

When BUG-49 is fixed, the refusal in `encode` comes off and the sweep in BUG-46 is re-run; the
numbers above are what it has to beat. Until then, `RATE-2 lossless fallback refused` on stderr is
the honest statement of what the encoder is doing on subsampled input, and `tests/lossless_intra_
fallback.rs` asserts both halves — that the sibling carries the format, and that the fallback
refuses to use it.
