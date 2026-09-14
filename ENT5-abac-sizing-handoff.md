# ENT-5 handoff — settle the abac GPU-encode sizing mode on an idle Mac

**Status:** ENT-5 shipped the GPU encoder (DONE 2026-09-07); its **criterion 3** is still open —
encode time per 1080p frame on an idle Mac, reported as one of BASELINE's three named fps
quantities, to discharge `docs/decisions/0017` reason 2 (129 ms/frame CPU encode vs Rice's 23 ms).
Coupled to it is the **sizing-mode default**: `CountThenEmit` (2 coder passes, "dubbelkodning",
exactly-sized scratch) vs `BoundedSlots` (1 coder pass, 22–29× scratch). `docs/decisions/0057`.

## The one command

```bash
cargo test --release --test abac_bench abac_encode_throughput_grid \
  -- --ignored --nocapture --test-threads=1
```

Run it on an **idle** Mac (name the machine — M5 Pro or M1 Pro; they are not comparable, CLAUDE.md
Platform Notes / BUG-29). The bench warms up, does best-of-24 paired back-to-back, and prints
`med/best` — only quote absolutes when `med/best` is near 1.0 on the row (the GPU clock ramp
inflated one row until it settled; see the module comment).

## What to check

1. **Absolute `frame ms` for the winning row against 129 ms** — this is criterion 3. Report it as a
   BASELINE fps quantity, saying which Mac.
2. **The sizing ratio `CountThenEmit / BoundedSlots`** — decides whether the default flips off the
   double-coding.
3. **`bytes` column identical across every row** — the GPU encoder is bit-exact; any difference is a
   bug, not a trade. (Held here: Range 397005, Interval 392310 on both sizing modes.)

## Prior from a non-canonical Windows run (RTX 2000 Ada, DX12/Vulkan — NOT BASELINE, NOT M1)

Absolute ms below are throwaway (wrong machine, wrong backend). **Only the ratio is portable**, and
even it shifts with the GPU's compute-vs-readback mix. Two settled runs, variance < 2 %:

| coder | CountThenEmit (2 pass) | BoundedSlots (1 pass) | dubbelkodningens kostnad |
|---|---|---|---|
| Range    | 14.2–14.5 ms/plane | 9.29 ms/plane    | **+54 %** (1.54×) |
| Interval | 34.6–34.8 ms/plane | 21.5–21.7 ms/plane | **+60 %** (1.60×) |

**Prediction to verify on the Mac:** single-pass `BoundedSlots` is faster than `CountThenEmit` on
both engines, by roughly **1.5–1.6×** (i.e. dropping the double-coding saves ~35–38 % of GPU encode
time). Range is ~2.4× faster than Interval on GPU. If the Mac ratio is materially different (say
< 1.2× or > 2×), investigate before flipping the default — the readback share differs by GPU.

**Cost of `BoundedSlots`:** 22–29× scratch; confirm it stays inside the 256 MiB max-buffer request
(`gnc::required_limits()`) at 1080p 4:4:4. It fit on the Windows box.

## Scope / hygiene

- This is criterion 3 + the sizing default only. **Not** the abac-as-default decision — that is
  ENT-10 step 2 (P0) and also needs the 1.69× frame *decode* re-priced.
- Nothing was committed for this; no RESEARCH_LOG/BASELINE edit. A Windows relative figure is not an
  ENT-5 closure. Do the Mac run through the normal claim/worktree flow.
