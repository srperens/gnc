#!/usr/bin/env python3
"""What is per-tile rate-distortion bit allocation worth to GNC?

GNC quantises every tile with the same step (modulated only by per-subband AQ weights) and does
no global rate-distortion pass. JPEG 2000's PCRD instead truncates each code-block's embedded
stream so that every block ends at the same RD slope, which is provably the optimal allocation
for a given total rate. This repo's own analysis attributed ~10-15% of the remaining gap to
exactly that, and noted PCRD is inaccessible with Rice because Rice is not truncatable.

But the *allocation* idea does not need a truncatable code — choosing a different quantiser step
per tile achieves the same thing, and only needs the step signalled per tile (a byte). This
measures the ceiling of that: each tile's RD curve is computed independently, then

  uniform : every tile at the same step (what GNC does)
  equal-slope : per-tile steps chosen so all tiles sit at the same RD slope (the PCRD optimum)

are compared at matched total rate.

Coefficient-level RDOQ was measured separately and gives +0.1% — GNC's quantiser is already on
its own RD curve. This asks the different question of whether the *bits are in the right tiles*.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meas4_oracle import dwt2, idwt2, quantize, dequantize, subband_gains  # noqa: E402


def tile_rd(tile, qsteps, levels=3, dead_zone=0.75):
    """(bits, sum-squared-error) for one tile at each quantiser step."""
    out = []
    ll, bands = dwt2(tile, levels)
    g = subband_gains(tile.shape, levels)
    for q in qsteps:
        bits = 0.0

        def code(coef, name):
            nonlocal bits
            qq = quantize(coef * g[name], q, dead_zone)
            vals, counts = np.unique(qq, return_counts=True)
            p = counts / counts.sum()
            bits += float(-(p * np.log2(p)).sum() * qq.size)
            return dequantize(qq, q, dead_zone) / g[name]

        rec_ll = code(ll, "LL")
        rec_bands = []
        for lv, (lh, hl, hh) in enumerate(bands):
            names = (f"LH{lv+1}", f"HL{lv+1}", f"HH{lv+1}")
            rec_bands.append(tuple(code(b, nm) for b, nm in zip((lh, hl, hh), names)))
        rec = idwt2(rec_ll, rec_bands)
        out.append((bits, float(((tile - rec) ** 2).sum())))
    return out


def allocate_equal_slope(curves, target_bits):
    """Pick a step per tile so all tiles sit at the same RD slope, hitting `target_bits`.

    Lagrangian sweep: for a given lambda every tile independently picks the point minimising
    D + lambda*R, which is the equal-slope condition. Bisect lambda to hit the rate target.
    """
    def total_for(lam):
        bits = dist = 0.0
        choice = []
        for pts in curves:
            costs = [d + lam * b for b, d in pts]
            k = int(np.argmin(costs))
            choice.append(k)
            bits += pts[k][0]
            dist += pts[k][1]
        return bits, dist, choice

    lo, hi = 1e-4, 1e6
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        bits, _, _ = total_for(mid)
        if bits > target_bits:
            lo = mid
        else:
            hi = mid
    return total_for((lo * hi) ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png")
    ap.add_argument("--tile", type=int, default=256)
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

    qsteps = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]
    curves = []
    for j in range(ty):
        for i in range(tx):
            curves.append(tile_rd(img[j * t : (j + 1) * t, i * t : (i + 1) * t], qsteps))

    px = ty * tx * t * t
    print(f"\n=== per-tile RD allocation: {os.path.basename(args.png)} "
          f"({tx}x{ty} tiles of {t}px) ===")
    print(f"  {'uniform qstep':>14} {'bpp':>9} {'PSNR':>8}   "
          f"{'equal-slope PSNR':>17} {'gain':>8}")

    for qi, q in enumerate(qsteps):
        u_bits = sum(c[qi][0] for c in curves)
        u_dist = sum(c[qi][1] for c in curves)
        u_psnr = 10 * np.log10(255.0 ** 2 / (u_dist / px))
        a_bits, a_dist, _ = allocate_equal_slope(curves, u_bits)
        a_psnr = 10 * np.log10(255.0 ** 2 / (a_dist / px))
        print(f"  {q:14.1f} {u_bits / px:9.4f} {u_psnr:8.2f}   "
              f"{a_psnr:17.2f} {a_psnr - u_psnr:+7.2f} dB")
    print()


if __name__ == "__main__":
    main()
