# 0074 — abac context-codes the Exp-Golomb prefix, and it is worth 2.1–8.8% of total rate

**Date:** 2026-09-08
**Item:** ENT-9 step 2 (the bitstream half)
**Status:** accepted — shipped, bitstream generation GP18 → GP19, abac still opt-in

## Decision

**abac's Exp-Golomb unary prefix is context-coded on (position, bucket) instead of bypassed at
p = 1/2.** 24 contexts on top of the existing 18. The mantissa and the sign stay bypassed. This is
candidate A from `0063`, which priced it before anything was built and named this as the only
candidate above ENT-9's gate.

**ENT-9's gate — ≥2% of total rate at q=99 on ≥3 sequences at bit-identical pixels — is met on
three of three, and the pixels are verified identical rather than argued:**

| sequence (q=99) | `0063`'s bound, coder bits | **measured, total rate** | gap to bound |
|---|---|---|---|
| crowd_run | −8.20% | **−8.04%** | 0.16 |
| bbb_extended | −2.44% | **−2.07%** | 0.37 |
| old_town_cross | −9.07% | **−8.76%** | 0.31 |

18 frames, ki=9, 4:4:4, whole container including headers and motion vectors abac does not code.
**Every measured figure lands just under its bound, by 0.16 to 0.37 points**, which is the one
direction that makes sense: the bound was ideal-adaptive with no signalling charged, and `0063`
said in as many words to "expect the realisable figure lower". A figure *above* the bound would
have meant the bound was wrong.

It is not only a q=99 effect:

| q | total rate, three sequences | mean |
|---|---|---|
| 99 | −2.07% … −8.76% | −6.29% |
| 95 | −1.26% … −4.56% | −3.43% |
| 90 | −0.85% … −2.75% | −2.06% |

I-frame bytes move too (−2.74% to −6.23%), consistent with `0063`'s finding that intra bypasses
the same three quarters. So this is not an inter result.

## Verification

**Bit-identical pixels, measured on the shipped path.** 3 sequences × q ∈ {99, 90} × 18 frames,
encoded and decoded through GPU encode → GPU decode on both arms, comparing per-frame PNG hashes:
**identical on all 6 arms, 108 frames**. The change is entropy-only by construction — same
quantised coefficients in and out — and this is the check that says so rather than assuming it.

**Byte-exact three ways, which is what makes a bitstream change in this codec safe.**
`scripts/ent5_gpu_encode_gate.sh`: **98 of 98 identical**, GPU encoder against the CPU reference
across both arithmetic engines, q = 90/99/100, 4:4:4 / 4:2:2 / 4:2:0, cb = 16/32/64, and the
sequence path at ki = 1 and 9. Plus `gpu_encode_matches_cpu_encoder_byte_for_byte`,
`gpu_decode_matches_cpu_coder` and `gpu_encode_round_trips_through_gpu_decode`.

**Workgroup storage was the constraint to check first, and it fits.** `probs` is `WG *
NUM_CONTEXTS` in both shaders, so 18 → 42 contexts takes 576 u32 to 1344: measured
**6400 B → 9472 B** per entry point against the 16384 B requested budget, on all four
(`abac_encode.wgsl:main`, `:main_rc`, `abac_decode.wgsl:main`, `:main_rc`). Checked before any
code was written, because a design that did not fit would have needed a different context layout
rather than a bug fix.

## The bitstream generation is GP19, and GP18 abac frames are refused

Only entropy type 5 changed, so a GP19 frame using any other coder is byte-identical to the GP18
one apart from four bytes — asserted directly by
`gp19_rice_frames_are_gp18_payloads_with_a_new_label`, which relabels a Rice frame and requires
the identical decode.

**GP18 abac frames are rejected, not decoded.** The old and new binarisations differ only in *how*
bits are modelled, so a GP18 abac frame read as GP19 would come back as a **plausible wrong
image** rather than an error — the failure `abac_tile.rs` warns about in its own header. The
entropy-type-5 gate therefore moved from `gen >= 18` to `gen >= 19` with a message naming the
cause. GP18 files using Rice, rANS, Huffman or bitplane still decode, because nothing they use
moved.

The GNV1/GNV2 sequence containers are **not** bumped: their layout is unchanged and the frames
inside them carry their own generation, so the marker that matters is already there. GNV's own
`version` field is read and never validated, which is worth knowing but is not this item's to fix.

## What was not chosen

- **Candidate B, contexting the sign** on the left and up neighbours' signs. `0063` priced it at
  −0.57% to −1.29% of the coder's bits at q=99, below ENT-9's gate on three of three. **Not spent,
  and now cheaper to re-price than before**: A has moved the denominator, so B's share of the
  remaining bits is larger than it was. Worth a re-price, not worth building blind.
- **Contexting the mantissa.** It is the low bits of a magnitude and carries no causal information
  about it, so a context would buy nothing and cost adaptation on 24 more slots.
- **More than four prefix positions.** Position saturates at 3. Deep positions are rare — they
  need magnitudes past 2^4 — so distinguishing them would cold-start contexts that never see the
  symbols to pay for themselves. This is the same reasoning that keeps `bucket` at 6.
- **Backward compatibility for GP18 abac files.** Both binarisations in one decoder means a
  runtime flag through the CPU coder and both shaders, for files that exist only on this machine.
  GOALS rule 10 says there are no users and no compatibility burden; refusing them cleanly is the
  honest version of that.
- **Re-taking BASELINE's `--abac` BD-rate row in this item.** It is now conservative — the ladder
  is q = 85/92/96/99, squarely in the range this change helps, so "+66.0% mean, 1.66x against
  H.264" will improve. It is *not* re-taken here, because today's `main` also carries RATE-3,
  BUG-39, INTER-2 and more, and a ladder taken now would attribute all of it to ENT-9 — which is
  exactly the failure COORD-6 was filed for. Filed as **MEAS-11**, to be run on a pinned commit.

## What this leaves

abac remains **opt-in**. `0045` weakened `0017`'s case for it at the top of the range, and this
strengthens it again by 2.1–8.8 points of total rate at q=99 — but the two figures `0017` rests on
are an intra rate saving and a 1.69× decode debt, and **only the rate half has moved**. Whether
abac should be the default is ENT-10, and it is parked on an idle machine because it needs the
throughput half re-taken. This item deliberately does not touch that question.
