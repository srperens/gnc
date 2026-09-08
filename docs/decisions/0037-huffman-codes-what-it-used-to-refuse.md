# 0037 — Huffman codes what it used to refuse: length limiting by frequency scaling, and a slot sized from the work

**Date:** 2026-09-08
**Items:** BUG-23, BUG-22
**Status:** accepted

## Why the two are one record

They are the same configuration failing twice in a row. `--huffman -q 100 -t 512` on `bbb_1080p`
hit BUG-23 first: the codebook builder could not limit its code lengths, so it spun (79% CPU, 8
minutes) and, after 2026-09-07's guard, refused. Fixing that produced a codebook and the encode
then hit BUG-22 immediately — `stream 0 overflowed its 512-byte output slot (753 bytes)`. Neither
fix is observable without the other, so they were measured and are recorded together.

Huffman is a parked coder (GOALS §5b: Rice is the default, always), which is why both were filed
as P3 and left guarded rather than fixed. What changed is only that "guarded" means *this
configuration does not work*, and both guards were reached by an ordinary command line.

## BUG-23 — length limiting

**Was:** build the natural Huffman tree, then redistribute any length above 8 bits by moving one
symbol from length *j* to two at *j*+1, only from lengths below the maximum. That donor pool is
finite — on the order of a hundred donations for a 64-symbol alphabet — and the excess is not
bounded by it. A geometric histogram over 32 symbols exhausts the pool after 247 donations with 52
bits still to place.

**Now:** build a real tree on scaled frequencies. While the tree is deeper than the limit, halve
every non-zero frequency and rebuild.

Two properties carry it, and both are checked rather than trusted:

- **It terminates.** 32 halvings take any `u32` to 1, and a uniform 64-symbol alphabet has depth 6,
  inside the 8-bit limit. An assert names that bound if it is ever exceeded.
- **The result satisfies Kraft by construction**, because it is an actual Huffman tree rather than
  a patched length histogram. This is the part that matters most: the old code could return with
  excess still unplaced, and then `assign_canonical_codes` would hand out codewords that are not a
  prefix code — a tile decoding to noise. The guard added in 2026-09-07 was there to stop exactly
  that, by refusing.

Rounding up on the halving (`div_ceil(2)`, floored at 1) is not cosmetic: a live symbol that scaled
to zero would drop out of the active set and have no code at all.

**Rejected: package-merge.** It is the textbook length-limited construction and it is optimal.
Frequency scaling is not optimal — it discards histogram resolution — but it is about fifteen lines
against a few hundred, it terminates for a reason a reader can check in one line, and it is
standard practice (zlib and JPEG implementations do it). For a parked coder whose job is to exist
and be correct, optimality of the codebook is worth less than a construction nobody has to audit.
If Huffman is ever unparked and measured, package-merge is the upgrade and it changes only the
codebook.

**This changes Huffman's bitstream wherever clamping used to occur** — permitted by GOALS rule 10,
and in practice the affected configurations previously either hung or refused, so there is nothing
to be compatible with.

## BUG-22 — the output slot

**Was:** `MAX_STREAM_WORDS = 128u`, a fixed 512-byte slot per stream, with nothing in `emit_byte`
checking `p_word_pos` against it. A stream needing more wrote into its neighbour's slot and the
host packed those bytes back out as data. That is the whole of the tile-512 corruption: **7.8–10.9
dB at q=90 on all four stills.**

**Now:** the host computes the slot from the work — `STREAMS_PER_TILE` threads split a tile's
coefficients evenly, and the worst case per symbol is four bytes (significance bit, sign bit, an
8-bit code, an exp-Golomb escape), so four bytes per symbol is an upper bound and not an estimate.
It arrives in the params as `max_stream_words`, and **both** `stream_output` writes in the shader
are bounded by it.

The bound in the shader is deliberate belt-and-braces. Correct sizing is what makes the
configuration work; the bound is what makes a *wrong* size truncate one stream instead of
corrupting the next one. The host assert stays for the same reason and its message now says so: if
it fires, the four-bytes-per-symbol bound is wrong, not the configuration.

Cost: 4 KiB per stream at tile 512, about 37 MB of scratch for 1080p 4:4:4. The slot size is now
part of the buffer cache's key, so a tile-size change at the same tile count reallocates instead of
reusing a slot of the wrong size.

**Rejected: keep refusing.** BUG-22's own entry proposed exactly this fix and declined it because
"BUG-14's tile-512 arm is the only thing that wants it". That was true when the alternative was a
guard; it stops being true once the guard is the only thing between an ordinary command line and a
crash.

## Evidence

`--huffman -q 100 -t 512` on `bbb_1080p`, the configuration that hung for 8 minutes:

| | before | after |
|---|---|---|
| encode | hang, then refuse, then slot overflow | **0.63 s, 3412338 B** |
| decode | — | **bit-exact lossless: max error 0, 0 wrong pixels** over 1920×1080×3 |

q=90 at tile 512, the arm BUG-22 measured at 7.8–10.9 dB:

| image | PSNR now | max error |
|---|---|---|
| bbb_1080p | 50.08 dB | 4 |
| blue_sky_1080p | 49.95 dB | 4 |
| kristensara_720p | 49.66 dB | 4 |
| touchdown_1080p | 49.57 dB | 4 |

BASELINE puts q=90 at 50.06 dB, so these are the operating point rather than merely "better" — a
recovery of about 40 dB on all four.

- **239 tests pass, 0 failures**, including two new ones: the steeply skewed histogram is now
  length-limited and Kraft-complete rather than refused, and a second test asserts the
  unconstrained tree for that histogram really is deeper than the limit, so the first cannot pass
  for the wrong reason.
- `cargo clippy --release` and `--target wasm32-unknown-unknown --lib` clean.
- **The default path did not move: Rice at q=90 is byte-identical** to the baseline taken before
  this session's earlier work. Only Huffman files changed.

## Not measured

**Huffman's rate and speed against the other coders.** It is parked and unmeasured against the
defaults, and this record does not change that — it makes two configurations work that previously
could not run. Whether Huffman is worth unparking is a separate question with no number behind it
yet.
