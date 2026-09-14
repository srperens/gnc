#!/usr/bin/env python3
"""0024/INTRA-1 follow-up: does a cross-subband *parent* context find rate the spatial models missed?

Decision 0024 bounded the entropy coder's headroom with *spatial* neighbourhood models only and
concluded ~72% of the JPEG 2000 gap is upstream of the coder. It named one untested model class as
the thing to try if the spatial line came back empty:

  "Testing a parent / cross-subband context. SPIHT and EZW condition on the coefficient in the
   coarser band; this measurement covers only causal spatial neighbourhoods. It is the one model
   class that could still find something ... If step 2 comes back empty, this is the thing to try
   before concluding."

The spatial line is now empty on every axis measured, including direction (ENT-12, 2026-09-14). So
this runs the pre-registered test. A coefficient at (y,x) in a detail band has a parent at
(y//2, x//2) in the same-orientation band one level coarser, which a decoder has already fully
decoded (coarse-to-fine). Condition on its **quantised** magnitude — what the decoder actually has.

Clean isolation, parent term and nothing else:
  - the coder: significant? / |v|>1 / |v|>2 as context-coded binary decisions, then a bypassed
    Exp-Golomb remainder and sign (common-mode, cancel in the delta), exactly as meas_ent12_pattern.
  - arm A (spatial): context = (subband, bucket(magL + magU + magD))  -- abac's model.
  - arm B (spatial+parent): context = (subband, spatial bucket, bucket(parent magnitude)).
  - LL and the coarsest level have no parent; both arms code them spatial-only, so they are
    common-mode and the total delta is exactly what the parent term buys where a parent exists.
  - both arms charged the same 0.5*log2(n)-per-context KT cost. +KT is the number to believe: arm B
    has B x more contexts, which mechanically lowers raw conditional entropy, and KT is what makes
    the extra table pay for itself.

Run:  python3 scripts/meas_parent_context.py <png> [<png> ...] [--qstep 4 8 16]

Criterion (from 0024's framing): the remaining coder headroom is single digits; a parent term has
to move the coded size by more than a few percent to be the missing lever rather than a footnote.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meas4_oracle import dwt2, quantize, subband_gains  # noqa: E402
from meas_ebcot_context import cond_entropy_bits  # noqa: E402

B = 4  # magnitude buckets: 0, 1, 2-3, >=4


def bucket(m):
    return np.where(m == 0, 0, np.minimum(np.log2(np.maximum(m, 1)).astype(np.int64) + 1, B - 1))


def spatial_bucket(a):
    """abac's context: bucket of the causal-neighbour magnitude sum (left, up, up-left+up-right)."""
    p = np.pad(a, 1)
    s = p[1:-1, :-2] + p[:-2, 1:-1] + p[:-2, :-2] + p[:-2, 2:]
    return bucket(s)


def parent_up(parent_absq, shape):
    """Parent quantised magnitude broadcast to the child's grid: child (y,x) <- parent (y//2,x//2)."""
    up = np.repeat(np.repeat(parent_absq, 2, axis=0), 2, axis=1)
    return up[:shape[0], :shape[1]]


def code_band(a, ctx):
    """Bits to code magnitudes `a` (>=0) under integer context `ctx`. Returns (raw, +KT)."""
    v = a.ravel().astype(np.int64)
    c = ctx.ravel()
    raw = pen = 0.0
    for sel, dec in (
        (np.ones(v.shape, dtype=bool), v > 0),
        (v > 0, v > 1),
        (v > 1, v > 2),
    ):
        if sel.any():
            t, p = cond_entropy_bits(c[sel], dec[sel].astype(np.int64))
            raw += t
            pen += p
    m3 = v > 2
    if m3.any():
        rem = v[m3] - 3
        bp = float((2 * np.floor(np.log2(rem + 1)) + 1).sum())
        raw += bp
        pen += bp
    sgn = float((v > 0).sum())
    return raw + sgn, pen + sgn


def measure(img, levels, qstep):
    ll, bands = dwt2(img, levels)
    g = subband_gains(img.shape, levels)
    # Quantise every band once; the parent context reads the parent's quantised magnitude.
    q = {"LL": quantize(ll * g["LL"], qstep, 0.75)}
    for lv, (lh, hl, hh) in enumerate(bands):
        for oi, (b, o) in enumerate(zip((lh, hl, hh), ("LH", "HL", "HH"))):
            q[(lv, oi)] = quantize(b * g[f"{o}{lv+1}"], qstep, 0.75)

    a_raw = a_pen = b_raw = b_pen = 0.0
    bid = 0

    # LL: spatial-only, identical in both arms.
    a = np.abs(q["LL"]).astype(np.int64)
    ctx = bid * B + spatial_bucket(a)
    r, p = code_band(a, ctx)
    a_raw += r; a_pen += p; b_raw += r; b_pen += p
    bid += 1

    for lv, (lh, hl, hh) in enumerate(bands):
        for oi in range(3):
            a = np.abs(q[(lv, oi)]).astype(np.int64)
            sb = spatial_bucket(a)
            # arm A: spatial only
            r, p = code_band(a, bid * B + sb)
            a_raw += r; a_pen += p
            # arm B: spatial + parent (if this band has a parent one level coarser)
            has_parent = lv < levels - 1
            if has_parent:
                pm = bucket(parent_up(np.abs(q[(lv + 1, oi)]).astype(np.int64), a.shape))
                ctxb = bid * (B * B) + sb * B + pm
            else:
                ctxb = bid * (B * B) + sb  # no parent: same information as arm A
            r, p = code_band(a, ctxb)
            b_raw += r; b_pen += p
            bid += 1
    return a_raw, a_pen, b_raw, b_pen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png", nargs="+")
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--qstep", type=float, nargs="*", default=[4.0, 8.0, 16.0])
    ap.add_argument("--crop", type=int, nargs=2, default=[1792, 1024])
    args = ap.parse_args()
    from PIL import Image

    print("\n=== 0024 parent context: spatial vs spatial+parent, conditional entropy (bits/px) ===")
    print("  +KT is the number to believe (arm B has more contexts; KT charges the bigger table).")
    print("  'parent vs' < 0 means the cross-subband parent term helps. Bar: a few %.\n")
    print(f"  {'image':<22} {'qstep':>6} {'spatial':>9} {'+parent':>9} {'parent vs':>10} "
          f"{'sp+KT':>9} {'par+KT':>9} {'+KT vs':>9}")
    for path in args.png:
        im = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
        m = 1 << args.levels
        ch = min(args.crop[1], im.shape[0]) // m * m
        cw = min(args.crop[0], im.shape[1]) // m * m
        y0 = (im.shape[0] - ch) // 2
        x0 = (im.shape[1] - cw) // 2
        img = im[y0:y0 + ch, x0:x0 + cw]
        px = img.size
        for qq in args.qstep:
            ar, ap_, br, bp = measure(img, args.levels, qq)
            print(f"  {os.path.basename(path):<22} {qq:>6.1f} {ar/px:>9.4f} {br/px:>9.4f} "
                  f"{(br/ar-1)*100:>+9.1f}% {ap_/px:>9.4f} {bp/px:>9.4f} {(bp/ap_-1)*100:>+8.1f}%")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
