#!/usr/bin/env python3
"""What is rate-distortion optimised quantisation worth on GNC's wavelet coefficients?

GNC quantises with a uniform quantiser plus a dead zone and codes whatever comes out. It makes
no rate-distortion decision anywhere: a coefficient that would cost 6 bits to raise the
reconstruction by a hair is coded regardless. x264's trellis and JPEG 2000's PCRD both spend
their gains here, and this repo's own JPEG 2000 analysis attributed ~10-15% of the remaining
spatial gap to PCRD bit allocation.

Unlike PCRD, coefficient-level RDOQ needs no truncatable code — it only changes which values are
quantised to what, so it works with Rice or rANS unchanged and needs no bitstream change.

Method: transform an image with the same 3-level dyadic wavelet the codec uses, then compare
  baseline: uniform quantiser with dead zone (what GNC does)
  rdoq:     for each coefficient, consider the baseline level and the levels below it (including
            zero) and pick the one minimising D + lambda*R, with R the actual code length under
            the empirical per-subband distribution.
Reports rate and distortion for both, at several quantiser steps.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meas4_oracle import dwt2, idwt2, quantize, dequantize, subband_gains  # noqa: E402


def code_lengths(levels):
    """Empirical code length per symbol value, from its frequency in this subband."""
    vals, counts = np.unique(levels, return_counts=True)
    p = counts / counts.sum()
    return dict(zip(vals, -np.log2(p)))


def rdoq_subband(coef, qstep, dead_zone, lam, candidates=3):
    """Choose each coefficient's level by D + lambda*R rather than by rounding alone.

    Two passes: quantise normally to learn the symbol distribution, then re-decide each
    coefficient against that distribution. A single refinement pass captures most of the gain and
    keeps this an estimate of the ceiling rather than a full trellis.
    """
    base = quantize(coef, qstep, dead_zone)
    lut = code_lengths(base.reshape(-1))
    default_cost = max(lut.values()) + 2.0 if lut else 8.0

    def bits(v):
        return lut.get(v, default_cost)

    best = base.copy()
    best_cost = None
    for drop in range(candidates):
        cand = np.sign(base) * np.maximum(np.abs(base) - drop, 0)
        cand = cand.astype(np.int32)
        rec = dequantize(cand, qstep, dead_zone)
        d = (coef - rec) ** 2
        r = np.vectorize(bits)(cand)
        cost = d + lam * r
        if best_cost is None:
            best_cost = cost
            best = cand
        else:
            take = cost < best_cost
            best = np.where(take, cand, best)
            best_cost = np.minimum(cost, best_cost)
    return best


def run(img, qstep, levels=3, dead_zone=0.75, rdoq=False, lam_scale=0.85):
    ll, bands = dwt2(img, levels)
    g = subband_gains(img.shape, levels)
    lam = lam_scale * qstep * qstep
    bits = 0.0

    def code(coef, name):
        nonlocal bits
        scaled = coef * g[name]
        q = rdoq_subband(scaled, qstep, dead_zone, lam) if rdoq else quantize(scaled, qstep, dead_zone)
        vals, counts = np.unique(q, return_counts=True)
        p = counts / counts.sum()
        bits += float(-(p * np.log2(p)).sum() * q.size)
        return dequantize(q, qstep, dead_zone) / g[name]

    rec_ll = code(ll, "LL")
    rec_bands = []
    for lv, (lh, hl, hh) in enumerate(bands):
        names = (f"LH{lv+1}", f"HL{lv+1}", f"HH{lv+1}")
        rec_bands.append(tuple(code(b, nm) for b, nm in zip((lh, hl, hh), names)))
    rec = idwt2(rec_ll, rec_bands)
    mse = float(((img - rec) ** 2).mean())
    return bits / img.size, 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png")
    ap.add_argument("--crop", type=int, nargs=2, default=[1024, 1024])
    args = ap.parse_args()

    from PIL import Image

    im = np.asarray(Image.open(args.png).convert("L"), dtype=np.float64)
    cw, ch = args.crop
    y0 = max(0, (im.shape[0] - ch) // 2)
    x0 = max(0, (im.shape[1] - cw) // 2)
    img = im[y0 : y0 + ch, x0 : x0 + cw]

    print(f"\n=== RDOQ on wavelet coefficients: {os.path.basename(args.png)} "
          f"({img.shape[1]}x{img.shape[0]} luma crop) ===")
    print(f"  {'qstep':>6} {'baseline':>20}   {'RDOQ':>20}   {'rate saved':>11}")
    print(f"  {'':6} {'bpp':>9} {'PSNR':>10}   {'bpp':>9} {'PSNR':>10}")
    for q in (2.0, 4.0, 8.0, 16.0):
        b_bpp, b_psnr = run(img, q)
        r_bpp, r_psnr = run(img, q, rdoq=True)
        print(f"  {q:6.1f} {b_bpp:9.4f} {b_psnr:10.2f}   {r_bpp:9.4f} {r_psnr:10.2f}   "
              f"{(1 - r_bpp / b_bpp) * 100:9.1f}%  (dPSNR {r_psnr - b_psnr:+.2f})")
    print()


if __name__ == "__main__":
    main()
