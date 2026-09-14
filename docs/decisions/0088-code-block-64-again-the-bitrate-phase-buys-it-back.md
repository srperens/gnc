# 0088 — the code-block default is 64 again: the bitrate phase buys the rate back

**Date:** 2026-09-15
**Item:** RATE-6
**Status:** accepted
**Machine:** Apple M5 Pro [Metal, IntegratedGpu] — `gnc gpu-info`. One of two Macs (COORDINATION).
**Binary:** `codec-fingerprint v1 1ea00f81` before, **`99a2e8ec`** after — the encoder writes
different bytes and the fingerprint says so.
**Reverses:** `0051` — *abac's code-block default is 32, not 64* — 2026-09-14, one day old.

*(The doc comment on `DEFAULT_CB` cited `0086` for that change. `0086` is ENT-15, a different
record about abac's GPU encode losing to the CPU; the citation was wrong and is corrected here.)*

## What changed

`DEFAULT_CB` in `src/encoder/abac_tile.rs`: **32 → 64**. One constant. `cb` is written per tile, so
this is an encoder default and not a format change — files written at either size keep decoding,
and `GNC_ABAC_CB` still overrides it for a measurement without a rebuild.

## Why, and it is not that `0051` was wrong

`0051` halved the code-block edge because abac's GPU encode and decode both run **one thread per
code-block**: at cb=64 a 1080p plane has 1000 blocks, so 1000 threads each serially coding 4096
coefficients. Halving the edge quadruples the parallelism. It priced the trade honestly —
**~2.5 to 3.5 points of rate for 2.3x off encode and ~2x off decode**, 6 of 6 points — and the
owner had asked for exactly that: *"abac är för mycket latency nu."*

**What was missing was the rate side measured the way the project keeps score.** `0051` measured
three stills at q ∈ {90,99}. The contribution ladder against x264, three sequences, 17 frames,
ki=9, 4:2:0, is what Current Focus tracks, and on it the same knob costs:

| sequence | cb=32 (`d7dc8d8`) | cb=64 (this change) | delta |
|---|---|---|---|
| bbb_extended | +93.9% | **+88.8%** | −5.1 |
| old_town_cross | +50.8% | **+47.3%** | −3.5 |
| crowd_run | +49.7% | **+46.5%** | −3.2 |
| **mean BD-rate vs x264** | **+64.8%** | **+60.9%** | **−3.9** |

Pixels do not move — PSNR-Y is identical at every rung, the change is bytes only. In bpp at q=85:
bbb 3.0627 → 2.9550 (−3.52%), old_town 5.5817 → 5.4670 (−2.06%), crowd_run 5.7291 → 5.6109
(−2.06%).

**The project is in its bitrate phase** (`8ebfd9e`, 2026-09-14): *"A rate win is worth taking even
when it costs encode or decode time."* 3.9 points of BD-rate is a large fraction of what the whole
entropy-coding effort has bought — ENT-9's context-coded prefix was worth ~5 points on this ladder
— so spending it on encode latency, in the phase where latency is explicitly deferred, is spending
it in the wrong currency. The owner's decision, in their words:

> *"jag vill att vi går tillbaka till 64 just nu. notera att priset för 32 är mycket bitrate just
> nu."*

## What this costs, stated plainly

Everything `0051` bought is given back: **encode goes from ~2.2–5.1x Rice to ~5.0–12.0x, frame
decode from ~1.3–2.1x to ~2.6–4.8x** (`0051`'s own table, the measurement is not retracted). In
absolute terms abac's encode was ~73–80 ms per 1080p frame at cb=32 and ~127–199 ms at cb=64, on an
M1 Pro — and abac's encode is nearly content-independent, so those are flat costs rather than worst
cases. **Anyone reading this for a latency answer should read `0051`, not this record.**

## What was considered and not chosen

- **Keeping 32 and re-taking the scoreboard at 32.** Honest, and it was the alternative on the
  table. Rejected because it accepts the 3.9 points as spent rather than deciding to spend them,
  in the phase where bitrate is the only thing counted.
- **cb=16.** `0051` measured it: ~8% more encode speed for **+17% rate** against cb=64's +3.8%.
  The lever is spent below 32; this is a knee, not a slide.
- **cb above 64.** Refused by the decoder, not by judgement: `abac_decode.wgsl` keeps two rows of
  neighbour magnitudes per thread in workgroup memory and is sized for 64 (`MAX_BLOCK_W`), and
  `abac_cb_from_env` asserts it rather than letting a larger value encode and then fail to decode.
- **Making the default depend on quality or on a latency flag.** GOALS §1 blesses per-operating-point
  strategy, and this is a real candidate — but it needs the encode/decode half re-measured on an
  idle machine (`docs/QUIET_HOUR.md`) before a second default is worth defining. Filed, not taken.

## When to revisit

**When the phase changes.** Phase 2 in Current Focus is concurrent sessions, performance, low
latency and robustness; on the day that leads, 32 is the knee to come back to and this record is
the price list. The deeper question this raises — whether a per-symbol-serial coder is the right
tool for a GPU codec at all, or whether abac's ~14% over Rice lives in its *context modelling*
rather than its arithmetic engine, in which case the modelling could move to a structure that
parallelises without paying a per-block floor — is filed as **ENT-18**, and ENT-15 is its starting
point.
