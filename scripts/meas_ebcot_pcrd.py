#!/usr/bin/env python3
"""What could EBCOT's PCRD-opt buy GNC? Measured at code-block granularity.

This repo already measured the PCRD *idea* at tile granularity and got 0% (`meas_pcrd.py`), and
concluded "a uniform quantiser step already equalises the RD slope across tiles". That result does
not bound EBCOT, and the reason is the partition:

  - a 256x256 **tile** contains every subband and every kind of content in that image region, so
    its RD slope is an average over all of them — and averages across tiles look alike.
  - a 64x64 **code-block** is a piece of *one* subband, so it is homogeneous: all high-frequency
    or all low, all busy or all flat. Its RD slope is not averaged with anything.

If the slope variance lives below the tile level, tile-granular allocation cannot see it and
code-block allocation can. That is the question EBCOT's PCRD answers and this measures.

Method, per image:
  1. DWT once, with the same subband gain normalisation the codec applies.
  2. Cut every subband into code-blocks of `--cb` pixels.
  3. Give each code-block its own RD curve over a quantiser ladder: bits from the zeroth-order
     entropy of *its own* quantised values (so each block pays for its own statistics, as an
     EBCOT code-block does), distortion from squared coefficient error.
  4. Compare, at matched total rate:
       uniform     — every block at the same step (what GNC does)
       equal-slope — Lagrangian allocation, the PCRD optimum
  Reported as PSNR at matched bits, which is the only fair form.

**Distortion is reported from a real reconstruction, not from the coefficient domain.** The
per-block RD curves have to use coefficient-domain error — that is unavoidable and it is what
JPEG 2000's own PCRD does, since a code-block cannot be inverse-transformed alone. But the
gain-normalised CDF 9/7 is only *approximately* orthonormal: measured here it flatters
coefficient-domain PSNR by 0.24 dB at qstep 4 and 0.52 dB at qstep 16. So the curves are used only
to *choose* the allocation, and both the uniform and the allocated result are then reconstructed
through a real inverse DWT and scored on pixel error. Reporting the coefficient-domain number
would inherit that bias.

Run:  python3 scripts/meas_ebcot_pcrd.py <png> [--cb 64] [--levels 5]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meas4_oracle import dwt2, idwt2, quantize, dequantize, subband_gains  # noqa: E402

LADDER = [1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0, 32.0, 45.0, 64.0, 90.0, 128.0]


def block_rd(coef, qsteps, dead_zone):
    """RD curve for one code-block: [(bits, sum-squared coefficient error), ...]."""
    out = []
    for q in qsteps:
        qq = quantize(coef, q, dead_zone)
        _, counts = np.unique(qq, return_counts=True)
        p = counts / counts.sum()
        # Zeroth-order entropy of this block's own symbols, plus a small per-block header for the
        # coding state an independent code-block cannot share.
        bits = float(-(p * np.log2(p)).sum() * qq.size) + 16.0
        rec = dequantize(qq, q, dead_zone)
        out.append((bits, float(((coef - rec) ** 2).sum())))
    return out


def cut_blocks(band, cb):
    h, w = band.shape
    for y in range(0, h, cb):
        for x in range(0, w, cb):
            b = band[y : y + cb, x : x + cb]
            if b.size:
                yield b


def allocate_equal_slope(curves, target_bits):
    """Lagrangian sweep: at a given lambda each block independently minimises D + lambda*R,
    which is the equal-slope condition. Bisect lambda to hit the rate target."""

    def total_for(lam):
        bits = dist = 0.0
        choice = []
        for pts in curves:
            k = int(np.argmin([d + lam * b for b, d in pts]))
            choice.append(k)
            bits += pts[k][0]
            dist += pts[k][1]
        return bits, dist, choice

    lo, hi = 1e-6, 1e8
    for _ in range(70):
        mid = (lo * hi) ** 0.5
        if total_for(mid)[0] > target_bits:
            lo = mid
        else:
            hi = mid
    return total_for((lo * hi) ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png")
    ap.add_argument("--cb", type=int, default=64, help="code-block size")
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--dead-zone", type=float, default=0.75)
    ap.add_argument("--crop", type=int, nargs=2, default=[1792, 1024])
    args = ap.parse_args()

    from PIL import Image

    im = np.asarray(Image.open(args.png).convert("L"), dtype=np.float64)
    cw, ch = args.crop
    y0 = max(0, (im.shape[0] - ch) // 2)
    x0 = max(0, (im.shape[1] - cw) // 2)
    img = im[y0 : y0 + ch, x0 : x0 + cw]
    px = img.size

    ll, bands = dwt2(img, args.levels)
    g = subband_gains(img.shape, args.levels)

    named = [("LL", ll)]
    for lv, (lh, hl, hh) in enumerate(bands):
        named += [(f"LH{lv+1}", lh), (f"HL{lv+1}", hl), (f"HH{lv+1}", hh)]

    # --- per-code-block RD curves on gain-scaled coefficients (used only to allocate) ---
    # Keep the block geometry so a chosen allocation can be reassembled into full bands.
    layout = []  # (band_name, y, x, curve_index)
    curves = []
    for nm, band in named:
        sc = band * g[nm]
        h, w = sc.shape
        for y in range(0, h, args.cb):
            for x in range(0, w, args.cb):
                blk = sc[y : y + args.cb, x : x + args.cb]
                if not blk.size:
                    continue
                layout.append((nm, y, x, len(curves)))
                curves.append(block_rd(blk, LADDER, args.dead_zone))

    scaled = {nm: band * g[nm] for nm, band in named}

    def reconstruct(choice):
        """Quantise every block at its chosen step, then inverse-DWT for exact pixel error."""
        out = {nm: np.empty_like(v) for nm, v in scaled.items()}
        for nm, y, x, ci in layout:
            blk = scaled[nm][y : y + args.cb, x : x + args.cb]
            q = LADDER[choice[ci]]
            out[nm][y : y + args.cb, x : x + args.cb] = dequantize(
                quantize(blk, q, args.dead_zone), q, args.dead_zone
            )
        rec_ll = out["LL"] / g["LL"]
        rec_bands = []
        for lv in range(args.levels):
            rec_bands.append(
                tuple(out[f"{d}{lv+1}"] / g[f"{d}{lv+1}"] for d in ("LH", "HL", "HH"))
            )
        rec = idwt2(rec_ll, rec_bands)
        return 10 * np.log10(255.0 ** 2 / (float(((img - rec) ** 2).sum()) / px))

    print(f"\n=== EBCOT PCRD ceiling: {os.path.basename(args.png)} "
          f"({len(curves)} code-blocks of {args.cb}px, {args.levels} levels) ===")
    print("  PSNR from a real inverse DWT; coefficient-domain slopes used only to allocate.")
    print(f"  {'uniform qstep':>13} {'bpp':>8} {'uniform PSNR':>13} "
          f"{'equal-slope':>12} {'gain':>10}")
    gains = []
    for qi, q in enumerate(LADDER):
        u_bits = sum(c[qi][0] for c in curves)
        if u_bits / px < 0.05 or u_bits / px > 8.0:
            continue
        _, _, choice = allocate_equal_slope(curves, u_bits)
        u_psnr = reconstruct([qi] * len(curves))
        a_psnr = reconstruct(choice)
        gains.append(a_psnr - u_psnr)
        print(f"  {q:>13.1f} {u_bits/px:>8.4f} {u_psnr:>13.2f} "
              f"{a_psnr:>12.2f} {a_psnr-u_psnr:>+8.2f} dB")
    if gains:
        print(f"\n  mean gain from code-block RD allocation: {np.mean(gains):+.2f} dB")
        print("  (the same allocation at tile granularity measured 0.00 dB — meas_pcrd.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
