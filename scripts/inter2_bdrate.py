#!/usr/bin/env python3
"""INTER-2 — BD-rate the inter dead-zone arms against the shipped one, over a common interval.

Pure Python on purpose: this machine has no numpy, and the arithmetic here is a cubic fit through
four points, which is exact interpolation and needs no least-squares package.

The interval matters more than the fit. Each `inter_dz_mul` reaches a different quality at the
same q -- that is the whole effect being measured -- so integrating each arm over its own overlap
would compare figures taken over different ranges. QUAL-1 measured what that does: widening a
ladder moved a BD-rate by 47.5 points on average. So every arm for a given sequence and metric is
integrated over the *intersection* of all arms' ranges, following meas_inter1_pscale.py.
"""
import csv, sys
from collections import defaultdict


def solve(a, b):
    """Gaussian elimination with partial pivoting. a is n x n, b length n."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-15:
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r == c:
                continue
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def polyfit(xs, ys, deg):
    """Least-squares fit via normal equations; exact interpolation when len(xs) == deg+1."""
    n = deg + 1
    A = [[sum(x ** (i + j) for x in xs) for j in range(n)] for i in range(n)]
    b = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(n)]
    c = solve(A, b)
    return c  # ascending powers


def polyint_eval(c, x):
    return sum(ci / (i + 1) * x ** (i + 1) for i, ci in enumerate(c))


def bd_rate_over(rate_a, q_a, rate_b, q_b, lo, hi):
    """BD-rate of B against A over a given [lo, hi]. Negative = B needs fewer bits."""
    import math
    if hi <= lo:
        return None
    pa = polyfit(q_a, [math.log10(r) for r in rate_a], min(3, len(q_a) - 1))
    pb = polyfit(q_b, [math.log10(r) for r in rate_b], min(3, len(q_b) - 1))
    if pa is None or pb is None:
        return None
    ia = polyint_eval(pa, hi) - polyint_eval(pa, lo)
    ib = polyint_eval(pb, hi) - polyint_eval(pb, lo)
    return (10 ** ((ib - ia) / (hi - lo)) - 1) * 100


def main(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("bytes") or r["sequence"] == "control":
                continue
            rows[(r["sequence"], r["inter_dz_mul"])].append(
                {"q": int(r["q"]), "bytes": float(r["bytes"]),
                 "psnr": float(r["psnr_mean"]), "worst": float(r["psnr_worst"]),
                 "vmaf": float(r["vmaf"]) if r.get("vmaf") else None})
    for k in rows:
        rows[k].sort(key=lambda d: d["q"])

    seqs = sorted({s for s, _ in rows})
    arms = sorted({m for _, m in rows}, key=float, reverse=True)
    base = "2.0"

    for metric in ("psnr", "vmaf"):
        print(f"\n=== BD-rate on {metric.upper()} vs shipped inter_dz_mul={base} "
              f"(negative = arm needs fewer bits) ===")
        for seq in seqs:
            series = {m: rows[(seq, m)] for m in arms if (seq, m) in rows}
            vals = {m: [d[metric] for d in s if d[metric] is not None] for m, s in series.items()}
            if any(len(v) < 2 for v in vals.values()):
                print(f"  {seq}: insufficient {metric} data")
                continue
            lo = max(min(v) for v in vals.values())
            hi = min(max(v) for v in vals.values())
            out = []
            for m in arms:
                if m == base:
                    continue
                bd = bd_rate_over([d["bytes"] for d in series[base]], vals[base],
                                  [d["bytes"] for d in series[m]], vals[m], lo, hi)
                out.append(f"mul={m}: {bd:+.2f}%" if bd is not None else f"mul={m}: n/a")
            print(f"  {seq:16s} interval [{lo:.2f}, {hi:.2f}]   " + "   ".join(out))

    print("\n=== worst-frame PSNR (the contribution gate) ===")
    for seq in seqs:
        print(f"  {seq}")
        for q in sorted({d["q"] for m in arms if (seq, m) in rows for d in rows[(seq, m)]}):
            cells = []
            for m in arms:
                d = next((x for x in rows.get((seq, m), []) if x["q"] == q), None)
                cells.append(f"mul={m}: {d['worst']:.2f} dB @ {d['bytes']/1e6:.2f} MB" if d else f"mul={m}: -")
            print(f"    q={q}  " + "   ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "inter2.csv")
