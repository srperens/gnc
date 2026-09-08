# 0073 — Below q=100 the lossless swap is a sequence choice, because the frames are coupled

Date: 2026-09-08
Item: BACKLOG LOSSLESS-3 (P1)
Status: accepted
**Changes a default.** At q = 95..=99, 4:4:4, with no bitrate target, a sequence whose lossy
encode is larger than a bit-exact all-intra encode of the same frames is emitted bit-exact
instead. Nothing at q ≤ 94, nothing at q=100, nothing on subsampled chroma, and no still is
affected — all verified byte-identical.

**Tree:** every figure below was taken on `d10e414` plus this change, which is **after BUG-47
(`0072`)**. That matters: BUG-47 made `lossless_sibling` carry `pad_fill_decay`, which is exactly
the bytes in the q=95..99 columns here. The first take of this table predated it, and the numbers
moved by up to 3 points — the conclusion did not. COORDINATION, "Every number carries a tree".

## The defect

RATE-2 (`0036`) found that above q≈95 a **still** costs more as a wavelet encode than as a
bit-exact MED encode, and fixed it by coding both and keeping the smaller. RATE-3 lifted that into
sequences **for I-frames only**. Nothing compared a *P-frame* against a bit-exact alternative, and
that is where the rate is.

8 frames, ki=9, 4:4:4, container bytes, before LOSSLESS-3:

| sequence | q=95 | q=97 | q=99 | bit-exact | q=99 costs |
|---|---|---|---|---|---|
| crowd_run | 31 427 614 | 34 374 105 | 37 984 009 | **25 856 146** | **+46.9%** |
| old_town_cross | 31 440 634 | 34 391 448 | 38 010 958 | **25 247 023** | **+50.5%** |
| blue_sky | 19 231 299 | 21 659 527 | 24 812 142 | **17 294 725** | **+43.5%** |
| bbb (animation) | 17 896 639 | 20 924 647 | 24 290 268 | 25 885 896 (est.) | **−6.2%** |

On camera content every rung from q=95 up is dominated: more bytes than bit-exact, for pixels that
are not exact. LOSSLESS-2 (`0070`) is what exposed it — it took `q=100` on crowd_run from 43.0 MB
to 25.9 MB while the ladder above stood still.

## The finding that shaped the fix: a per-frame rule is a ratchet

The per-frame version was built first, because it is what LOSSLESS-2 does and it had just worked.
Measured over 24 points (4 sequences × q=95/97/99 × ki=2/9): **22 improved by 7.9% to 33.6%, and
bbb at q=99 ki=9 got 5.51% larger.** The pre-stated success criterion — *the container never
larger than today's* — failed, so it was reverted rather than tuned around.

**Why.** A P-frame costs *more* when it predicts from a bit-exact reference than from a lossy
P-frame reconstruction. Mean over frames 2–7 at q=99 ki=9, P-frame bytes against a lossy reference
versus the same frames against an exact one:

| sequence | lossy reference | exact reference | |
|---|---|---|---|
| crowd_run | 4 968 556 | 5 210 086 | **+4.86%** |
| old_town_cross | 4 986 250 | 5 210 913 | **+4.51%** |
| blue_sky | 3 238 448 | 3 403 731 | **+5.10%** |
| bbb | 3 021 205 | 3 318 982 | **+9.86%** |

Four of four, so it is a mechanism and not a quirk: a quantised reference carries error the next
frame's own quantiser lands on, and an exact one does not. **Replacing frame *k* therefore inflates
frame *k+1*'s candidate**, so each greedy step makes the next one likelier: every step looks like a
win and the whole is worse. That is why LOSSLESS-2 is safe at `q=100` and this is not — there,
every reference is exact by construction and the coupling cannot exist, which is exactly the
locality argument `0070` rests on.

## The decision

**Compare whole arms, decided on measured totals.** After the sequence is coded, if the lossy total
exceeds a bit-exact all-intra encode of the same frames, emit that instead. The two arms are each
internally consistent, so there is no coupling to be wrong about, and no estimate in the decision.

`last_lossless_candidate_bytes` — the bit-exact candidate `encode` already codes for RATE-2's own
comparison — is used **only as a trigger**, `n` times a bit-exact I-frame less 1%, so the second
arm is not coded when it obviously loses. The 1% is the padding decay fill an all-intra arm may use
and a referenced I-frame may not (`0039`), measured at 0.77%. A wrong trigger costs a wasted pass
or a missed sub-1% win; it cannot pick the wrong arm.

**Result, all 24 points: never larger, no frame worse, no PSNR regression anywhere.**

| sequence | ki=2 (q=95/97/99) | ki=9 (q=95/97/99) |
|---|---|---|
| crowd_run | −10.81% / −15.68% / −20.96% | −17.73% / −24.78% / **−31.93%** |
| old_town_cross | −12.04% / −16.90% / −22.15% | −19.70% / −26.59% / **−33.58%** |
| blue_sky | −5.95% / −12.67% / −19.97% | −10.07% / −20.15% / **−30.30%** |
| bbb (animation) | ±0.00% on all three | ±0.00% on all three |

Every point that switched decodes **bit-exact, 8 of 8 frames md5-identical to the source PNGs**
through the container. And the switched output is *the same file* a `q=100` encode produces —
byte-identical on crowd_run, old_town_cross and blue_sky at ki=9 — which is the cleanest statement
of what the fallback does: above q≈95 on camera content, GNC now emits its lossless encode.

**Animation is never dominated after BUG-47, and the second arm is never even coded there.** All
six bbb points are ±0.00% because the *trigger* does not fire: 8 × 3 235 737 B for the bit-exact
candidate is 25 885 896, and the lossy totals are 24 943 006 (ki=2) and 24 290 268 (ki=9) — under
the threshold, so no second pass runs and bbb pays nothing for this feature. The first take of this
table, before `0072`, had bbb winning −1.59% at q=99 ki=2; the sibling fix made the lossy arm
cheaper and took that away. So the content split inherited from LOSSLESS-2 and `0023` is **sharper**
now, not weaker: camera content is dominated at every rung from q=95, animation at none.

The point that regressed 5.51% under the per-frame rule is +0.00% under this one.

## What was not chosen

- **The per-frame rule.** Priced above: 1 regression in 24 and a mechanism that guarantees more of
  them on other content. Reverted, not tuned: a margin large enough to protect bbb (its P-frames
  exceed the bit-exact I by only 1.68–2.08%) would be a constant fitted to one clip, and RATE-2's
  own conclusion was that no preset constant finds this boundary.
- **A quality-based rule ("switch at the q where lossless wins").** That q is content-dependent —
  q≈95 on crowd_run, never on bbb — which is RATE-2's finding restated. Coding both arms is what
  removes the constant.
- **Applying it with a bitrate target set.** Refused: the bit-exact arm ignores the target, and
  silently blowing a CBR/VBR budget to save bytes on average is not the same trade. `rate_ctrl`
  must be `None`.
- **Applying it on subsampled chroma.** Gated off, and the reason is a bug rather than a trade:
  `lossless_sibling` does not carry the caller's chroma format, so the trigger would read a
  **4:4:4** size (3 257 157 B on bbb whether the request is 4:4:4 or 4:2:0) and never fire. Filed
  as **BUG-46**; the gate is explicit so the behaviour is deliberate rather than incidental, and it
  comes off with the fix.
- **Re-entering `encode_sequence_streaming` recursively for the second arm.** Not needed: the
  bit-exact arm is all-intra, so it is a plain loop over `encode` with no references, no rate
  control and no local decode — and no risk of a recursive fallback.

## Consequences

- At q = 95..=99 a camera sequence now emits **8I+0P and bit-exact pixels**. More keyframes means
  better seeking, and `docs/BITSTREAM_SPEC.md` is unchanged: `transform_type` is a per-frame header
  byte (`0036`).
- The encode cost is a second all-intra pass, and only on sequences where the trigger fires. Not
  measured on an idle machine (eight sessions, one GPU), so no wall-clock figure is quoted.
- **This is the third time the same shape has paid** — RATE-2 on stills, LOSSLESS-2 at q=100,
  LOSSLESS-3 on the lossy sequence ladder. In all three the bit-exact candidate wins on rate *and*
  quality when it wins at all, which is why none of them needs a metric. Worth checking wherever
  else two candidates differ only in coded size.
