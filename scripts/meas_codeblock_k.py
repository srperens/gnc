#!/usr/bin/env python3
"""What is per-code-block parameter adaptation worth to GNC?

GNC codes each subband of a tile as one group with a single Rice k. JPEG 2000 instead partitions
every subband into code-blocks (typically 64x64) and adapts its coder state within each. The
wavelet-depth result (RESEARCH_LOG 2026-09-05) showed this axis matters more than an ideal-entropy
model predicts: going 3 -> 4 levels doubled the number of subbands and therefore the number of k
parameters, and the real codec gained 6% where an ideal model predicted 1.2%.

So this measures the same axis directly: hold the transform fixed and vary only how finely the
Rice parameter adapts.

  whole-subband : one k per subband per tile (what GNC does)
  code-block    : one k per NxN block within each subband (what JPEG 2000's granularity implies)

Rate is computed as the actual Golomb-Rice code length, not Shannon entropy, because the question
is specifically about a parameterised coder's ability to track local statistics — an entropy
figure would assume perfect adaptation and hide the whole effect.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meas4_oracle import dwt2, quantize, subband_gains  # noqa: E402


def rice_bits(mags, k):
    """Golomb-Rice code length for non-negative magnitudes at parameter k."""
    if mags.size == 0:
        return 0.0
    q = mags >> k
    return float((q + 1 + k).sum())


def best_k_bits(mags, kmax=15):
    """Bits under the best single Rice parameter for this set."""
    if mags.size == 0:
        return 0.0, 0
    best = None
    bk = 0
    for k in range(kmax + 1):
        b = rice_bits(mags, k)
        if best is None or b < best:
            best, bk = b, k
    return best, bk


def code_subband(coef, qstep, dead_zone, block=None, k_side_bits=4):
    """Bits to code one subband, either as a whole or per `block`x`block` code-block."""
    q = quantize(coef, qstep, dead_zone)
    # Sign bit for every non-zero, magnitude via Rice — the shape of GNC's magnitude coder.
    mags = np.abs(q).astype(np.int64)
    signs = int((q != 0).sum())

    if block is None:
        bits, _ = best_k_bits(mags.reshape(-1))
        return bits + signs + k_side_bits

    h, w = mags.shape
    total = 0.0
    n = 0
    for y in range(0, h, block):
        for x in range(0, w, block):
            sub = mags[y : y + block, x : x + block].reshape(-1)
            b, _ = best_k_bits(sub)
            total += b
            n += 1
    return total + signs + k_side_bits * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--crop", type=int, nargs=2, default=[1792, 1024])
    args = ap.parse_args()

    from PIL import Image

    im = np.asarray(Image.open(args.png).convert("L"), dtype=np.float64)
    cw, ch = args.crop
    y0 = max(0, (im.shape[0] - ch) // 2)
    x0 = max(0, (im.shape[1] - cw) // 2)
    img = im[y0 : y0 + ch, x0 : x0 + cw]
    t = args.tile
    ty, tx = img.shape[0] // t, img.shape[1] // t
    px = ty * tx * t * t

    print(f"\n=== Rice parameter granularity: {os.path.basename(args.png)} "
          f"({tx}x{ty} tiles of {t}px, {args.levels} levels) ===")
    print(f"  {'qstep':>6} {'whole subband':>14} {'cb 64':>12} {'cb 32':>12} {'cb 16':>12}")

    for qstep in (2.0, 4.0, 8.0, 16.0):
        totals = {None: 0.0, 64: 0.0, 32: 0.0, 16: 0.0}
        for j in range(ty):
            for i in range(tx):
                tile = img[j * t : (j + 1) * t, i * t : (i + 1) * t]
                ll, bands = dwt2(tile, args.levels)
                g = subband_gains(tile.shape, args.levels)
                pieces = [(ll, "LL")]
                for lv, (lh, hl, hh) in enumerate(bands):
                    for b, nm in zip((lh, hl, hh), (f"LH{lv+1}", f"HL{lv+1}", f"HH{lv+1}")):
                        pieces.append((b, nm))
                for blk in totals:
                    for coef, nm in pieces:
                        totals[blk] += code_subband(coef * g[nm], qstep, 0.75, block=blk)
        base = totals[None]
        print(f"  {qstep:6.1f} {base / px:14.4f} " + " ".join(
            f"{totals[b] / px:7.4f} ({(1 - totals[b] / base) * 100:+4.1f}%)" for b in (64, 32, 16)
        ))
    print()


if __name__ == "__main__":
    main()
