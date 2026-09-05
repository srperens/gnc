# 0006 — Pyramid B-frame reference handling must not depend on chroma format

**Date:** 2026-09-05. **Status:** accepted. **Closes:** BUG-2.

## Context

Two reference-buffer defects were identified by code reading during the BUG-1 diagnosis and
deliberately left unfixed there, because neither was implicated in the chroma collapse that
item addressed. Both were measured before fixing.

## Defect 1 — B₇'s backward reference was stale

In `layer3_order`, the leaf B-frames are encoded B₁, B₃, B₅, B₇. Each loads its own references
from pyramid slots. B₇ requests `bwd_idx = 1` (the future P), which hit a no-op arm whose
comment asserted the future P was "already in gpu_bwd_ref_planes (restored for B₆)".

That was true when B₁ ran and false by the time B₇ ran: B₁, B₃ and B₅ each overwrite the
backward reference on their way through the same loop, and B₅ leaves it holding B₆. So the
encoder predicted B₇ from B₆ while the decoder loaded P₈ — a plain encoder/decoder mismatch,
present in every chroma format.

**Fix:** load slot 4 explicitly, like every other arm in the match.

## Defect 2 — the end-of-group reference restore was gated on 4:4:4

After a pyramid group finishes, the decoded anchor P has to be moved from slot 4 back into
`gpu_ref_planes` so the *next* group's P/I encode uses it as the forward reference. That restore
existed but was written `if pyramid_enabled && chroma_format == Yuv444`. In 4:2:0 the `else`
branch ran `swap_ref_planes()`, which left `gpu_ref_planes` holding B₆.

The gate has no justification: the restore exists because the backward buffer gets clobbered
during the pyramid, and that happens in every chroma format.

**Fix:** drop the format condition.

## Why no test caught this

Every existing sequence test uses `keyframe_interval <= 9`. With ki=9 the group is 8 frames, so
the frame after the group is an I-frame and the restored reference is never read. The defect is
only observable with **ki > group_size**, where a second group's anchor P actually consumes it.
The new regression test uses ki=17 for exactly this reason.

## Measured effect

1080p BBB, q=75, 17 frames, `GNC_REF_DEBLOCK=0`.

**Defect 1** (4:4:4, ki=9, B₇ is the affected frame):

| | PSNR | bytes |
|---|---|---|
| before | 39.21 dB | 309 908 |
| after | **40.17 dB** | **240 163** |

B₇ now sits in line with the other leaf frames (B₁ 41.19, B₃ 40.43, B₅ 40.30, B₇ 40.17) instead
of 1–2 dB below them, at 22% fewer bits.

**Defect 2** (4:2:0, ki=17, the second group's anchor P₁₆):

| | P₈ | P₁₆ | sequence VMAF | VMAF min | bpp |
|---|---|---|---|---|---|
| before | 40.54 dB | **30.39 dB** | 84.10 | 69.74 | 1.35 |
| after | 40.54 dB | **40.35 dB** | **95.68** | **94.72** | **1.31** |

A 10 dB cliff on the second group's anchor, and everything predicted from it, in the codec's
default chroma format at any GOP longer than 9 frames. 4:4:4 at the same settings was unaffected
before the fix (P₁₆ 40.57 dB), confirming the format gate as the cause.

Both fixes raise quality while lowering rate, which is the signature of a prediction fix rather
than a masking change.
