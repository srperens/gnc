#!/usr/bin/env python3
"""ENT-12: does keying the context on the neighbour *pattern* (direction) beat abac's *sum*?

abac buckets a neighbourhood magnitude sum into a context. A sum discards direction: a horizontal
and a vertical edge with the same neighbour-magnitude total land in the same context, where EBCOT's
orientation-separated zero-coding contexts keep them apart. This measures whether that distinction
is worth anything, in the one way that falsifies it cheaply: conditional entropy of the *same*
coder under two context definitions that differ *only* in sum-vs-direction.

Held identical between the two arms, so the delta is direction and nothing else:
  - the coder: significant? / |v|>1 / |v|>2 as context-coded binary decisions, then an Exp-Golomb
    remainder and a sign bit bypassed at one bit each (common-mode; they cancel in the delta but
    are included so the % is of the full coefficient cost).
  - the neighbours: the four *causal* ones a raster/parallel-safe decode already has — left (h),
    up (v), up-left + up-right (d). No non-causal neighbour is used by either arm.
  - the table: static and per-subband. Each subband gets its own context space (so orientation is
    separated for free and the table matches ENT-12's "(subband, magnitude bucket, pattern)").
    Both arms are charged the same 0.5*log2(n)-per-context KT learning cost, which is what punishes
    the pattern arm for having more contexts to fill — the real price of direction.

  sum   context = (subband, bucket(magL + magU + magUL + magUR))          -- abac's model
  dir   context = (subband, bucket(magL), bucket(magU), bucket(magD))     -- direction kept

Run:  python3 scripts/meas_ent12_pattern.py <png> [<png> ...] [--qstep 4 8 16]

Pre-registered criterion (BACKLOG ENT-12): abac closed exactly half of the +54% RGB gap to
JPEG 2000, so ~27 points are unaccounted for. If direction does not move the coded size by more
than a few percent, the hypothesis is wrong and the gap is somewhere else.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meas4_oracle import dwt2, quantize, subband_gains  # noqa: E402
from meas_ebcot_context import cond_entropy_bits  # noqa: E402

B = 4  # magnitude buckets per direction: 0, 1, 2-3, >=4


def bucket(m):
    """0 -> 0, 1 -> 1, 2-3 -> 2, >=4 -> 3. Same log-ish spread abac uses."""
    return np.where(m == 0, 0, np.minimum(np.log2(np.maximum(m, 1)).astype(np.int64) + 1, B - 1))


def causal_mags(a):
    """Magnitudes of the four causal neighbours at every position (left, up, diag-sum)."""
    p = np.pad(a, 1)
    left = p[1:-1, :-2]
    up = p[:-2, 1:-1]
    diag = p[:-2, :-2] + p[:-2, 2:]
    return left, up, diag


def coded_bits(coef, band_id, mode):
    """Bits to code one subband under `mode` in {'sum','dir'}. Returns (raw, +KT)."""
    a = np.abs(coef).astype(np.int64)
    left, up, diag = causal_mags(a)
    if mode == "sum":
        ctx = band_id * (B * 3) + bucket(left + up + diag)
    else:  # dir: keep the three directions apart
        ctx = band_id * (B * B * B) + bucket(left) * B * B + bucket(up) * B + bucket(diag)
    v = a.ravel()
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
    # Exp-Golomb remainder of (|v|-3) and one sign bit, bypassed — identical in both arms.
    m3 = v > 2
    if m3.any():
        rem = v[m3] - 3
        bypass = float((2 * np.floor(np.log2(rem + 1)) + 1).sum())
        raw += bypass
        pen += bypass
    sgn = float((v > 0).sum())
    return raw + sgn, pen + sgn


def measure(img, levels, qstep):
    ll, bands = dwt2(img, levels)
    g = subband_gains(img.shape, levels)
    named = [("LL", ll, "LL")]
    for lv, (lh, hl, hh) in enumerate(bands):
        named += [(f"LH{lv+1}", lh, "LH"), (f"HL{lv+1}", hl, "HL"), (f"HH{lv+1}", hh, "HH")]
    s_raw = s_pen = d_raw = d_pen = 0.0
    for bid, (nm, band, _orient) in enumerate(named):
        qq = quantize(band * g[nm], qstep, 0.75).astype(np.int64)
        a, b = coded_bits(qq, bid, "sum")
        s_raw += a
        s_pen += b
        a, b = coded_bits(qq, bid, "dir")
        d_raw += a
        d_pen += b
    return s_raw, s_pen, d_raw, d_pen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png", nargs="+")
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--qstep", type=float, nargs="*", default=[4.0, 8.0, 16.0])
    ap.add_argument("--crop", type=int, nargs=2, default=[1792, 1024])
    args = ap.parse_args()
    from PIL import Image

    print("\n=== ENT-12: sum-context vs direction-context, conditional entropy (bits/px) ===")
    print("  raw = coder ceiling; +KT charges the learning cost that punishes more contexts.")
    print("  'dir vs sum' < 0 means direction helps. Criterion: a few % is the bar to clear.\n")
    print(f"  {'image':<22} {'qstep':>6} {'sum':>9} {'dir':>9} {'dir vs sum':>11} "
          f"{'sum+KT':>9} {'dir+KT':>9} {'+KT vs':>9}")
    for path in args.png:
        im = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
        # Snap the crop down to a multiple of 2**levels so the 5/3 DWT divides cleanly, and never
        # ask for more than the image has (720p is smaller than the default 1792x1024 window).
        m = 1 << args.levels
        ch = min(args.crop[1], im.shape[0]) // m * m
        cw = min(args.crop[0], im.shape[1]) // m * m
        y0 = (im.shape[0] - ch) // 2
        x0 = (im.shape[1] - cw) // 2
        img = im[y0:y0 + ch, x0:x0 + cw]
        px = img.size
        for q in args.qstep:
            s_raw, s_pen, d_raw, d_pen = measure(img, args.levels, q)
            print(f"  {os.path.basename(path):<22} {q:>6.1f} {s_raw/px:>9.4f} {d_raw/px:>9.4f} "
                  f"{(d_raw/s_raw-1)*100:>+10.1f}% {s_pen/px:>9.4f} {d_pen/px:>9.4f} "
                  f"{(d_pen/s_pen-1)*100:>+8.1f}%")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
