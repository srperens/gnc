#!/usr/bin/env python3
"""Chroma-aware quality metric for GNC (MEAS-7).

VMAF scores the luma plane only, so it cannot validate any decision about a chroma parameter —
chroma weighting, the CfL range, chroma-format trade-offs. A 2026-09-05 sweep of `chroma_weight`
looked like a free 15% rate saving on VMAF and collapsed to +0.3 dB once measured with a metric
that includes chroma.

This provides the missing half: **CIEDE2000**, the CIE's standard perceptual colour-difference
formula, computed on decoded RGB. Reported alongside VMAF it gives two numbers that answer
different questions — VMAF for luma structure, mean/95th-percentile dE00 for colour accuracy — so
a chroma parameter can be tuned by holding luma quality fixed and minimising colour error.

CIEDE2000 rather than a weighted YUV-PSNR because the weights in the latter would be the very
thing under dispute; dE00 is a published standard calibrated against human colour judgements, and
a dE00 of about 1 is the nominal just-noticeable difference.

Implemented directly (no skimage/colour dependency, neither of which is installed here) and
validated against all 16 critical pairs of the Sharma et al. CIEDE2000 test data — the ones that
exercise the RT rotation term, the blue region, the achromatic case and hue wrap-around — to
within 1e-3. `python3 scripts/chroma_metric.py --selftest` re-runs that check.

Usage:
  python3 scripts/chroma_metric.py <reference.png> <distorted.png> [more pairs...]
  python3 scripts/chroma_metric.py --dir <ref_dir> <dist_dir>
"""

import argparse
import glob
import os
import sys

import numpy as np

# sRGB D65 -> XYZ
_M = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE = np.array([0.95047, 1.00000, 1.08883])


def srgb_to_lab(rgb):
    """rgb in [0,255] float -> CIE L*a*b* under D65."""
    c = rgb / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def ciede2000(lab1, lab2):
    """CIEDE2000 colour difference, per CIE 142-2001. kL = kC = kH = 1."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    Cbar7 = Cbar ** 7
    G = 0.5 * (1.0 - np.sqrt(Cbar7 / (Cbar7 + 25.0 ** 7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    both = (C1p * C2p) != 0
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)
    dHp = np.where(both, dHp, 0.0)

    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)

    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    hbarp = np.where(
        ~both, hsum,
        np.where(hdiff <= 180.0, 0.5 * hsum,
                 np.where(hsum < 360.0, 0.5 * (hsum + 360.0), 0.5 * (hsum - 360.0))),
    )

    T = (1.0
         - 0.17 * np.cos(np.radians(hbarp - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hbarp))
         + 0.32 * np.cos(np.radians(3.0 * hbarp + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hbarp - 63.0)))

    dtheta = 30.0 * np.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    Cbarp7 = Cbarp ** 7
    RC = 2.0 * np.sqrt(Cbarp7 / (Cbarp7 + 25.0 ** 7))
    Lm50 = (Lbarp - 50.0) ** 2
    SL = 1.0 + 0.015 * Lm50 / np.sqrt(20.0 + Lm50)
    SC = 1.0 + 0.045 * Cbarp
    SH = 1.0 + 0.015 * Cbarp * T
    RT = -np.sin(np.radians(2.0 * dtheta)) * RC

    return np.sqrt(
        (dLp / SL) ** 2
        + (dCp / SC) ** 2
        + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )


def load_rgb(path):
    """Load an 8- or 16-bit RGB PNG and return values on a 0-255 scale.

    16-bit files must not go through `convert("RGB")` — that truncates to 8 bits and silently
    destroys exactly the precision a 10-bit encode exists to preserve.
    """
    from PIL import Image

    im = Image.open(path)
    a = np.asarray(im)
    if a.dtype == np.uint16:
        # GNC writes 10-bit samples in the high bits of 16-bit channels.
        return a.astype(np.float64) / 65535.0 * 255.0
    if im.mode != "RGB":
        im = im.convert("RGB")
        a = np.asarray(im)
    return a.astype(np.float64)


def compare(ref_path, dist_path):
    a = load_rgb(ref_path)
    b = load_rgb(dist_path)
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    de = ciede2000(srgb_to_lab(a), srgb_to_lab(b))
    return de


SHARMA_CASES = [
    ((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
    ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
    ((50, 2.8361, -74.0200), (50, 0, -82.7485), 3.4412),
    ((50, -1.3802, -84.2814), (50, 0, -82.7485), 1.0000),
    ((50, -1.1848, -84.8006), (50, 0, -82.7485), 1.0000),
    ((50, -0.9009, -85.5211), (50, 0, -82.7485), 1.0000),
    ((50, 0, 0), (50, -1, 2), 2.3669),
    ((50, -1, 2), (50, 0, 0), 2.3669),
    ((50, 2.4900, -0.0010), (50, -2.4900, 0.0009), 7.1792),
    ((50, 2.4900, -0.0010), (50, -2.4900, 0.0010), 7.1792),
    ((50, 2.5000, 0), (50, 0, -2.5000), 4.3065),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5382),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
]


def selftest():
    bad = 0
    for l1, l2, expected in SHARMA_CASES:
        got = float(ciede2000(np.array([l1], dtype=float), np.array([l2], dtype=float))[0])
        if abs(got - expected) >= 1e-3:
            bad += 1
            print(f"  MISMATCH expected {expected:7.4f}, got {got:7.4f}")
    n = len(SHARMA_CASES)
    print(f"CIEDE2000: {n - bad}/{n} Sharma reference pairs match to 1e-3 -> "
          f"{'VALIDATED' if bad == 0 else 'CHECK'}")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--selftest", action="store_true",
                    help="check the implementation against the CIE reference pairs")
    ap.add_argument("--dir", action="store_true", help="treat the two paths as directories")
    ap.add_argument("--quiet", action="store_true", help="print only the summary line")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.paths:
        ap.error("give image pairs, two directories with --dir, or --selftest")

    if args.dir:
        if len(args.paths) != 2:
            sys.exit("--dir takes exactly two directories")
        refs = sorted(glob.glob(os.path.join(args.paths[0], "*.png")))
        dists = sorted(glob.glob(os.path.join(args.paths[1], "*.png")))
        pairs = list(zip(refs, dists))
    else:
        if len(args.paths) % 2 != 0:
            sys.exit("expected reference/distorted pairs")
        pairs = list(zip(args.paths[0::2], args.paths[1::2]))
    if not pairs:
        sys.exit("no image pairs found")

    all_de = []
    for r, d in pairs:
        de = compare(r, d)
        all_de.append(de.reshape(-1))
        if not args.quiet:
            print(f"  {os.path.basename(d):28} dE00 mean {de.mean():6.3f}  "
                  f"p95 {np.percentile(de, 95):6.3f}  max {de.max():7.3f}")
    cat = np.concatenate(all_de)
    print(f"dE00 mean {cat.mean():.4f}  p95 {np.percentile(cat, 95):.4f}  "
          f"over-1 {float((cat > 1.0).mean()) * 100:.1f}%  ({len(pairs)} frames)")


if __name__ == "__main__":
    main()
