#!/usr/bin/env python3
"""INTRA-1 step 2: what would per-tile rate allocation be worth to GNC?

**The question.** Untiled JPEG 2000 runs PCRD across the whole picture, so it can spend more on a
hard region and less on an easy one. GNC gives every tile the same quantiser step above q=80 —
adaptive quantisation is enabled only for q in 30..=80 — and step 2 measured that GNC gains almost
nothing (0.6%) from a bigger tile while J2K loses 3.8% for the same step. Global rate allocation is
the leading explanation for that asymmetry, and this measures its ceiling before anyone builds it.

**Method — an oracle, not an implementation.** Encode the image at every q on a ladder. For each
encode take the per-tile rate (`GNC_TILE_RATE=1`, summed over the three planes) and the per-tile
squared error in RGB (decoded against original, on the *visible* part of each tile — the right
column and bottom row of tiles are mostly padding and must not be scored). Then for a Lagrangian
lambda, give every tile independently the q that minimises D_t + lambda * R_t, and sum. Sweeping
lambda traces the oracle's RD curve; BD-rate against the fixed-q ladder is the ceiling of the lever.

The oracle is generous on purpose — it sees the future, and a real encoder would have to search or
model. It is also charged nothing for signalling the per-tile q, which is about a byte per tile
(0.02% here). If a *free, clairvoyant* per-tile allocator cannot find the points, no built one will.

Rate is the sum of the serialized entropy tiles, not the file, so the frame header is excluded from
both arms equally.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meas1_vs_h264 import bd_rate  # noqa: E402


def sh(cmd, env=None):
    e = None
    if env:
        e = dict(os.environ)
        e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(cmd, capture_output=True, text=True, env=e)


def encode_point(gnc, src, q, tmp, tile):
    """One ladder rung: per-tile bytes, per-tile visible SSE, and the totals."""
    bs, png = tmp / f"ct_q{q}.gnc", tmp / f"ct_q{q}.png"
    r = sh([str(gnc), "encode", "-i", str(src), "-o", str(bs), "-q", str(q), "--abac",
            "-t", str(tile)], env={"GNC_TILE_RATE": "1"})
    if r.returncode != 0:
        raise RuntimeError(f"encode q={q} failed: {r.stderr.strip().splitlines()[-1:]}")
    rates = {}
    for m in re.finditer(r"\[tile_rate\] plane=(\d+) tx=(\d+) ty=(\d+) bytes=(\d+)", r.stderr):
        _, tx, ty, b = (int(x) for x in m.groups())
        rates[(tx, ty)] = rates.get((tx, ty), 0) + b
    if not rates:
        raise RuntimeError(f"q={q}: no [tile_rate] lines — is this an abac encode?")
    if sh([str(gnc), "decode", "-i", str(bs), "-o", str(png)]).returncode != 0:
        raise RuntimeError(f"decode q={q} failed")
    return rates, png


def tile_sse(orig, dec, tile):
    """Squared error per tile, over the visible pixels only.

    The tile grid is laid out on the *padded* plane, so the last column and row extend past the
    picture. Scoring the padding would credit those tiles with error they do not carry and make
    them look free to coarsen — the one way this measurement could favour the oracle by accident.
    """
    h, w = orig.shape[:2]
    d = (orig.astype(np.float64) - dec.astype(np.float64)) ** 2
    out = {}
    for ty in range(0, (h + tile - 1) // tile):
        for tx in range(0, (w + tile - 1) // tile):
            y0, x0 = ty * tile, tx * tile
            out[(tx, ty)] = float(d[y0:y0 + tile, x0:x0 + tile].sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--gnc-binary", default="target/release/gnc")
    ap.add_argument("--qualities", default="60,65,70,75,80,85,88,90,92,94,96,98")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()

    qs = [int(q) for q in args.qualities.split(",")]
    tmp = Path(args.tmp or os.environ.get("TMPDIR", "/tmp")) / "gnc_cross_tile"
    tmp.mkdir(parents=True, exist_ok=True)

    print("INTRA-1 step 2 — the ceiling on per-tile rate allocation")
    print(f"  ladder q={qs}, tile {args.tile}, --abac, rate = serialized entropy tiles\n")

    all_bd_rgb = []
    for img in args.images:
        orig = np.array(Image.open(img).convert("RGB"))
        h, w = orig.shape[:2]
        pts = []
        for q in qs:
            rates, png = encode_point(args.gnc_binary, img, q, tmp, args.tile)
            sse = tile_sse(orig, np.array(Image.open(png).convert("RGB")), args.tile)
            keys = sorted(set(rates) & set(sse))
            missing = set(sse) - set(rates)
            if missing:
                raise RuntimeError(f"q={q}: {len(missing)} visible tiles have no rate line")
            pts.append((q, {k: (rates[k], sse[k]) for k in keys}))

        tiles = sorted(pts[0][1])
        npix = w * h * 3

        def curve_fixed():
            out = []
            for q, d in pts:
                R = sum(d[k][0] for k in tiles)
                D = sum(d[k][1] for k in tiles)
                out.append((R * 8 / (w * h), 10 * np.log10(255.0 ** 2 / (D / npix))))
            return out

        def curve_oracle():
            """Per tile, the q minimising D + lambda*R. Lambda swept to trace the frontier."""
            out = []
            for lam in np.logspace(-2, 6, 240):
                R = D = 0.0
                for k in tiles:
                    best = min(pts, key=lambda p: p[1][k][1] + lam * p[1][k][0])
                    R += best[1][k][0]
                    D += best[1][k][1]
                out.append((R * 8 / (w * h), 10 * np.log10(255.0 ** 2 / (D / npix))))
            # keep the upper convex staircase: one point per distinct rate, best quality
            out.sort()
            keep, top = [], -1e9
            for r, p in out:
                if p > top:
                    keep.append((r, p))
                    top = p
            return keep

        # Compare at *matched distortion*, exactly, rather than by fitting two curves and
        # integrating between them. The first version of this script did the latter and reported
        # the oracle as +0.24% — worse than fixed q, which a clairvoyant allocator cannot be. That
        # was an interpolation artefact, and it is exactly the kind of number that would have been
        # published as "the lever is worth nothing". An oracle that does not dominate is a broken
        # instrument, so dominance is asserted below rather than assumed.
        def oracle_at(lam):
            R = D = 0.0
            for k in tiles:
                best = min(pts, key=lambda p: p[1][k][1] + lam * p[1][k][0])
                R += best[1][k][0]
                D += best[1][k][1]
            return R, D

        # Matching a distortion target exactly needs care, and two earlier constructions here
        # were wrong in ways that showed up as the oracle *losing* — which a clairvoyant allocator
        # cannot do. The q ladder is discrete, so no single lambda lands on an arbitrary target,
        # and a convex hull built from a fixed lambda grid interpolates along chords that can sit
        # a hair above the true envelope (it overshot by 0.02%).
        #
        # The exact construction: bisect lambda until two *adjacent* Lagrangian allocations
        # bracket the target distortion, then time-share between them. Both are optimal for their
        # own lambda, so the fixed-q allocation — feasible, therefore on or above both supporting
        # lines — is on or above the chord joining them. Dominance then holds by construction and
        # is asserted below rather than hoped for.
        def oracle_rate_at_distortion(target_d):
            lo, hi = 1e-6, 1e12                  # D increases with lambda
            r_lo, d_lo = oracle_at(lo)
            r_hi, d_hi = oracle_at(hi)
            if d_lo > target_d:
                return None                      # finer than the oracle's finest allocation
            if d_hi <= target_d:
                return r_hi
            for _ in range(200):
                mid = (lo * hi) ** 0.5
                r_m, d_m = oracle_at(mid)
                if d_m <= target_d:
                    lo, r_lo, d_lo = mid, r_m, d_m
                else:
                    hi, r_hi, d_hi = mid, r_m, d_m
                if hi / lo < 1.0 + 1e-12:
                    break
            if d_hi <= d_lo:
                return r_lo
            t = (target_d - d_lo) / (d_hi - d_lo)
            return r_lo + t * (r_hi - r_lo)

        # The oracle can save rate two different ways, and they must not be reported as one.
        #
        #   (a) spatially — different tiles get different q, which is the lever under test;
        #   (b) along the ladder — the whole frame would be better served by a quality *between*
        #       two rungs, and the oracle reaches it by mixing. That is a property of the q preset
        #       ladder (see RATE-2, where rate is not even monotonic in q), not of tiling.
        #
        # bbb at q=90 reads -7.9% against its own rung while its neighbours read -1%: that rung
        # simply sits off its own ladder's convex hull. Comparing against the hull of the fixed-q
        # points removes (b) and leaves the spatial part alone.
        fixed_pts = [(sum(d[k][0] for k in tiles), sum(d[k][1] for k in tiles)) for _, d in pts]

        def fixed_hull_rate_at_distortion(target_d):
            """Least rate a *uniform* q reaches `target_d` at, time-sharing between rungs."""
            front, best_r = [], float("inf")
            for dd, rr in sorted((d_, r_) for r_, d_ in fixed_pts):
                if rr < best_r:
                    front.append((dd, rr))
                    best_r = rr
            # Convex envelope of that frontier. Without this step the interpolation walks the
            # Pareto staircase itself, so a rung sitting *above* its own ladder's chord is treated
            # as reachable and its inefficiency is credited to the spatial lever. bbb's q=90 rung
            # is 4.7% above the chord between q=88 and q=92 — that is a mispriced rung (RATE-2),
            # not a cross-tile allocation gain.
            pareto = []
            for pt in front:
                while len(pareto) >= 2:
                    (d0, r0), (d1, r1) = pareto[-2], pareto[-1]
                    if (d1 - d0) * (pt[1] - r0) - (r1 - r0) * (pt[0] - d0) <= 0:
                        pareto.pop()
                    else:
                        break
                pareto.append(pt)
            if target_d < pareto[0][0] or len(pareto) < 2:
                return None
            if target_d >= pareto[-1][0]:
                return pareto[-1][1]
            prev = pareto[0]
            for cur in pareto[1:]:
                if cur[0] > target_d:
                    t = (target_d - prev[0]) / (cur[0] - prev[0])
                    return prev[1] + t * (cur[1] - prev[1])
                prev = cur
            return pareto[-1][1]

        name = Path(img).name
        print(f"  {name}  {w}x{h}, {len(tiles)} tiles")
        print(f"    {'q':>4} {'fixed bpp':>10} {'oracle bpp':>11} {'vs rung':>8} "
              f"{'vs ladder hull':>15} {'PSNR':>7} {'q spread':>16}")
        savings, spatial = [], []
        for q, d in pts:
            R = sum(d[k][0] for k in tiles)
            D = sum(d[k][1] for k in tiles)
            ro = oracle_rate_at_distortion(D)
            if ro is None:
                continue
            assert ro <= R * 1.0001, (
                f"oracle spent {ro} against fixed {R} at the same distortion — the oracle must "
                f"dominate by construction (all-tiles-same-q is in its own search space), so this "
                f"is an instrument fault, not a result"
            )
            # Which q's the oracle uses near this operating point — the lambda whose allocation
            # lands closest to the target distortion.
            lam_star = min(np.logspace(-4, 10, 300), key=lambda l: abs(oracle_at(l)[1] - D))
            chosen = sorted({min(pts, key=lambda p: p[1][k][1] + lam_star * p[1][k][0])[0]
                             for k in tiles})
            sav = (ro / R - 1.0) * 100.0
            savings.append((q, sav))
            rh = fixed_hull_rate_at_distortion(D)
            sp = (ro / rh - 1.0) * 100.0 if rh else None
            if sp is not None:
                spatial.append((q, sp))
            print(f"    {q:>4} {R*8/(w*h):>10.3f} {ro*8/(w*h):>11.3f} {sav:>+7.2f}% "
                  f"{(f'{sp:+.2f}%' if sp is not None else '-'):>15} "
                  f"{10*np.log10(255.0**2/(D/npix)):>7.2f} "
                  f"{str(chosen[0])+'-'+str(chosen[-1]):>16}")
        hi_q = [s for q, s in savings if q >= 85]
        hi_sp = [s for q, s in spatial if q >= 85]
        if savings:
            m_all = sum(s for _, s in savings) / len(savings)
            print(f"    mean vs rung: {m_all:+.2f}% over the ladder"
                  + (f", {sum(hi_q)/len(hi_q):+.2f}% at q>=85" if hi_q else ""))
            if hi_sp:
                print(f"    mean vs ladder hull (the purely spatial part): "
                      f"{sum(s for _, s in spatial)/len(spatial):+.2f}% over the ladder, "
                      f"{sum(hi_sp)/len(hi_sp):+.2f}% at q>=85")
            all_bd_rgb.append(sum(hi_sp) / len(hi_sp) if hi_sp else 0.0)
        print()

    if all_bd_rgb:
        print(f"  MEAN over {len(all_bd_rgb)} images, q>=85, spatial only: "
              f"{sum(all_bd_rgb)/len(all_bd_rgb):+.2f}% — this is the ceiling on cross-tile rate "
              f"allocation, with the q-ladder's own coarseness removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
