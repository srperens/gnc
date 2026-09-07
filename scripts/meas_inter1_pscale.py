#!/usr/bin/env python3
"""INTER-1 step 2 — price TUNE-6's P-frame quantiser scale at contribution quality, on one ladder.

The per-scale BD-rates that `meas_inter1_ki.py` prints cannot be compared with each other, and
that is not a detail. A coarser P-frame scale lowers the top of the inter arm's quality ladder —
at scale 1.0 the ki=9 arm reaches 59.5 dB, at 1.25 it reaches 57.4 and at 1.50 only 55.6 — so
each scale's BD-rate is integrated over a *different* quality interval. QUAL-1 measured what that
does: widening a ladder moved a BD-rate figure by 47.5 points on average. Ranking four scales by
four numbers taken over four different intervals would be that error with the sign hidden.

So this recomputes every scale over **one common interval**: the intersection of the overlaps of
all scales for a given sequence and metric. The arithmetic is the repo's own `bd_rate` (cubic fit
of log10(rate) against quality, integrated and divided by the interval) with `lo`/`hi` forced
rather than inferred, so a figure here is comparable to one from `bd_rate` over the same range and
to the other scales in the table.

Reads the CSVs written by `meas_inter1_ki.py`; runs no encodes of its own.

Usage:
    scripts/meas_inter1_pscale.py --baseline step1_ki_sweep.csv --scaled fixed_pscale_125.csv,...
"""

import argparse
import csv as csvmod
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def bd_rate_over(rate_a, q_a, rate_b, q_b, lo, hi):
    """BD-rate of B against A integrated over a *given* [lo, hi]. Negative = B more efficient.

    Same cubic-fit Bjontegaard as `meas1_vs_h264.bd_rate`; the only change is that the interval
    is supplied instead of derived from the data, which is what makes several arms comparable.
    """
    la, lb = np.log10(rate_a), np.log10(rate_b)
    pa = np.polyfit(q_a, la, min(3, len(q_a) - 1))
    pb = np.polyfit(q_b, lb, min(3, len(q_b) - 1))
    if hi <= lo:
        return None
    ia = np.polyval(np.polyint(pa), hi) - np.polyval(np.polyint(pa), lo)
    ib = np.polyval(np.polyint(pb), hi) - np.polyval(np.polyint(pb), lo)
    return (10 ** ((ib - ia) / (hi - lo)) - 1) * 100


def load(path):
    """{(sequence, ki, scale): [rows sorted by q]} from a meas_inter1_ki.py CSV."""
    out = {}
    with open(path) as f:
        for r in csvmod.DictReader(f):
            if r.get("failed"):
                continue
            key = (r["sequence"], int(r["ki"]), r["p_qp_scale"])
            out.setdefault(key, []).append(
                {"q": int(r["q"]), "bpp": float(r["bpp"]),
                 "psnr_avg": float(r["psnr_avg"]), "psnr_min": float(r["psnr_min"])}
            )
    for v in out.values():
        v.sort(key=lambda r: r["q"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True,
                    help="one or more meas_inter1_ki.py CSVs; the union is analysed")
    ap.add_argument("--ref-ki", type=int, default=1)
    ap.add_argument("--arm-ki", type=int, default=9)
    args = ap.parse_args()

    data = {}
    for p in args.csv:
        data.update(load(p))

    sequences = sorted({k[0] for k in data})
    scales = sorted({k[2] for k in data if k[1] == args.arm_ki},
                    key=lambda s: float(s) if s != "taper" else 1.0)

    for metric, label in (("psnr_avg", "mean PSNR"), ("psnr_min", "worst-frame PSNR")):
        print(f"\n=== BD-rate of ki={args.arm_ki} against all-intra, on {label}, "
              f"over one common interval per sequence ===")
        table = {}
        for seq in sequences:
            ref = data.get((seq, args.ref_ki, next(iter(scales))))
            # The all-intra reference is identical for every scale (no P-frames to scale), so
            # take whichever copy exists; assert they agree rather than trusting it.
            refs = [v for k, v in data.items() if k[0] == seq and k[1] == args.ref_ki]
            ref = refs[0]
            for other in refs[1:]:
                same = all(abs(a["bpp"] - b["bpp"]) < 1e-9 and abs(a["psnr_avg"] - b["psnr_avg"]) < 1e-9
                           for a, b in zip(ref, other))
                if not same:
                    sys.exit(f"{seq}: the all-intra arm differs between runs — the P-frame scale "
                             f"must not touch intra, so this is a harness or build error")

            arms = {s: data.get((seq, args.arm_ki, s)) for s in scales}
            arms = {s: v for s, v in arms.items() if v and len(v) >= 4}
            if not arms or len(ref) < 4:
                continue
            # One interval for every scale: intersect each arm's own overlap with the reference.
            lo = max([min(r[metric] for r in ref)] + [min(r[metric] for r in v) for v in arms.values()])
            hi = min([max(r[metric] for r in ref)] + [max(r[metric] for r in v) for v in arms.values()])
            table[seq] = (lo, hi, {})
            for s, v in arms.items():
                bd = bd_rate_over([r["bpp"] for r in ref], [r[metric] for r in ref],
                                  [r["bpp"] for r in v], [r[metric] for r in v], lo, hi)
                table[seq][2][s] = bd

        hdr = f"{'sequence':<16} {'interval (dB)':>15}  " + "  ".join(f"{s:>8}" for s in scales)
        print(hdr)
        print("-" * len(hdr))
        for seq, (lo, hi, vals) in table.items():
            row = f"{seq:<16} {lo:>6.2f}-{hi:<8.2f}  " + "  ".join(
                (f"{vals[s]:>+7.1f}%" if vals.get(s) is not None else f"{'—':>8}") for s in scales)
            print(row)
        means = {}
        for s in scales:
            vs = [v[2][s] for v in table.values() if v[2].get(s) is not None]
            if len(vs) == len(table) and vs:
                means[s] = sum(vs) / len(vs)
        if means:
            print("-" * len(hdr))
            print(f"{'MEAN':<16} {'':>15}  " + "  ".join(
                (f"{means[s]:>+7.1f}%" if s in means else f"{'—':>8}") for s in scales))
            best = min(means, key=means.get)
            print(f"\n  best on {label}: scale {best} at {means[best]:+.1f}%")


if __name__ == "__main__":
    main()
