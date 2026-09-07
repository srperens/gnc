#!/usr/bin/env python3
"""MEAS-3 — the RD curve on *sequences*, and what inter coding is worth across it.

Every RD curve in this repository is a single still. The video path has been measured at points
(q=75 here, ki=9 there) and compared at equal settings, which cannot judge a rate/quality trade —
COORDINATION rule 4, and at least four wrong conclusions have come from that error. So this walks
a quality ladder on three sequences and reports BD-rate.

Two things about the shape of the run:

  * **Two arms, not one.** `--ki 9` is the shipped inter configuration; `--ki 1` is all-intra. A
    bare curve says how many bits a quality costs; the pair says **what the whole inter machinery
    is worth at each end of the ladder**, which is the open question — RESEARCH_LOG puts GNC's own
    inter crossover "between q=75 and q=92" from equal-setting figures, i.e. from exactly the kind
    of point comparison that cannot settle it.
  * **The metric that leads changes partway up the ladder** (CLAUDE.md's table). VMAF leads at
    q<=85 and is *saturated above it*: QUAL-1 measured a VMAF BD-rate moving 47.5 points on
    average, 110 at worst, when the ladder widened, while PSNR moved 1.0. So this reports a PSNR
    BD-rate over the full ladder and a VMAF BD-rate **only over the q<=85 points**, labelled as
    such. A VMAF number spanning q=99 is not a weak number, it is not a number.

MEAS-3 as filed says "rd-curve lacks --chroma-format and --vmaf on sequences". That is stale:
`benchmark-sequence` has both, so no encoder change is needed and this is a harness over the
existing binary. It also asks for park_joy, which is not in the test material (COORDINATION: the
refetch restored four stills plus bbb/blue_sky/bbb_extended/crowd_run/old_town_cross only), so
old_town_cross stands in — the QUAL-1 sequence set, which keeps this comparable to the +90.5%
figure.

Usage:
    scripts/meas3_sequence_rd.py [--sequences crowd_run,old_town_cross,bbb_extended]
                                 [--qualities 25,40,55,70,85,92,99] [--frames 24] [--csv out.csv]
"""

import argparse
import csv as csvmod
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meas1_vs_h264 import bd_rate  # noqa: E402  (one BD-rate implementation for the whole repo)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def repo_root():
    out = sh(["git", "rev-parse", "--show-toplevel"])
    return Path(out.stdout.strip()) if out.returncode == 0 else Path.cwd()


def run_point(gnc, pattern, frames, ki, q, chroma, tmp):
    """One (sequence, q, ki) point. Returns None if the encode failed."""
    out_csv = tmp / f"seq_{Path(pattern).parent.name}_q{q}_ki{ki}.csv"
    cmd = [str(gnc), "benchmark-sequence", "-i", str(pattern), "-n", str(frames),
           "-k", str(ki), "-q", str(q), "--vmaf", "--chroma-format", chroma,
           "--csv", str(out_csv)]
    r = sh(cmd)
    if r.returncode != 0 or not os.path.exists(out_csv):
        lines = [ln.strip() for ln in (r.stderr + "\n" + r.stdout).splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            # The panic header carries the location; the reason is the line after it.
            if re.match(r"thread '[^']*'.*panicked at ", ln):
                return {"failed": lines[i + 1] if i + 1 < len(lines) else ln}
        return {"failed": lines[-1] if lines else "no output"}

    # Per-frame rows, then a summary row whose frame_idx is the literal "summary".
    rows = [r_ for r_ in csvmod.DictReader(open(out_csv)) if r_["frame_idx"] != "summary"]
    bpp = [float(r_["bpp"]) for r_ in rows]
    psnr = [float(r_["psnr"]) for r_ in rows]
    nbytes = sum(int(r_["encoded_bytes"]) for r_ in rows)

    # VMAF is printed, not written to the CSV.
    vmaf_mean = vmaf_min = None
    # "  VMAF: computing... mean=95.12  min=93.00  max=96.50"
    m = re.search(r"VMAF:[^\n]*?mean=([0-9.]+)\s+min=([0-9.]+)", r.stdout)
    if not m and "VMAF" in r.stdout:
        # Reported but unparsed: say so rather than silently returning None for the whole arm,
        # since a missing VMAF column silently switches the BD-rate to PSNR-only.
        print(f"    warning: VMAF line not parsed: "
              f"{[ln.strip() for ln in r.stdout.splitlines() if 'VMAF' in ln]}")
    if m:
        vmaf_mean, vmaf_min = float(m.group(1)), float(m.group(2))

    return {
        "frames": len(rows),
        "bytes": nbytes,
        "bpp": float(np.mean(bpp)),
        "psnr_avg": float(np.mean(psnr)),
        "psnr_min": float(np.min(psnr)),
        "vmaf_mean": vmaf_mean,
        "vmaf_min": vmaf_min,
        "stdout": r.stdout,
    }


def print_arm(name, pts):
    print(f"  {name}")
    print(f"    {'q':>4} {'bpp':>8} {'PSNR avg':>9} {'PSNR min':>9} {'VMAF':>8} {'VMAF min':>9}")
    for p in pts:
        v = f"{p['vmaf_mean']:.2f}" if p.get("vmaf_mean") is not None else "—"
        vm = f"{p['vmaf_min']:.2f}" if p.get("vmaf_min") is not None else "—"
        print(f"    {p['q']:>4} {p['bpp']:>8.4f} {p['psnr_avg']:>9.2f} {p['psnr_min']:>9.2f} "
              f"{v:>8} {vm:>9}")


def main():
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", default="crowd_run,old_town_cross,bbb_extended")
    ap.add_argument("--dir", default="test_material/frames/sequences")
    ap.add_argument("--qualities", default="25,40,55,70,85,92,99")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--chroma-format", default="444")
    ap.add_argument("--arms", default="9,1", help="keyframe intervals to run (9 = shipped inter, "
                                                  "1 = all-intra)")
    ap.add_argument("--gnc-binary", default=str(root / "target/release/gnc"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.gnc_binary):
        sys.exit(f"GNC binary not found: {args.gnc_binary} (cargo build --release)")
    qualities = [int(q) for q in args.qualities.split(",")]
    kis = [int(k) for k in args.arms.split(",")]

    all_rows = []
    with tempfile.TemporaryDirectory(prefix="meas3_") as td:
        for seq in args.sequences.split(","):
            seq_dir = Path(args.dir) / seq
            pattern = seq_dir / "frame_%04d.png"
            n_avail = len(list(seq_dir.glob("frame_*.png")))
            if n_avail == 0:
                print(f"\n=== {seq}: no frames at {seq_dir}, skipped ===")
                continue
            frames = min(args.frames, n_avail)
            print(f"\n=== {seq}  ({frames} of {n_avail} frames, ki={'/'.join(map(str, kis))}, "
                  f"{args.chroma_format}) ===")

            arms = {}
            for ki in kis:
                pts = []
                for q in qualities:
                    res = run_point(args.gnc_binary, pattern, frames, ki, q,
                                    args.chroma_format, Path(td))
                    if "failed" in res:
                        print(f"    ki={ki} q={q} FAILED: {res['failed']}")
                        all_rows.append({"sequence": seq, "ki": ki, "q": q,
                                         "failed": res["failed"]})
                        continue
                    res.pop("stdout", None)
                    row = {"sequence": seq, "ki": ki, "q": q, **res}
                    pts.append(row)
                    all_rows.append(row)
                arms[ki] = pts
                print_arm(f"ki={ki}" + (" (shipped inter)" if ki == 9 else " (all-intra)"), pts)

            # BD-rate of the inter arm against all-intra, where both arms exist.
            if 9 in arms and 1 in arms and len(arms[9]) >= 4 and len(arms[1]) >= 4:
                for metric, label, cap in (("psnr_avg", "PSNR", None),
                                           ("vmaf_mean", "VMAF", 85)):
                    a = [p for p in arms[1] if p.get(metric) is not None
                         and (cap is None or p["q"] <= cap)]
                    b = [p for p in arms[9] if p.get(metric) is not None
                         and (cap is None or p["q"] <= cap)]
                    if len(a) < 4 or len(b) < 4:
                        note = f" (only {min(len(a), len(b))} points" + (
                            f" at q<={cap})" if cap else ")")
                        print(f"    BD-rate on {label}: not computed{note}")
                        continue
                    bd, (lo, hi) = bd_rate([p["bpp"] for p in a], [p[metric] for p in a],
                                           [p["bpp"] for p in b], [p[metric] for p in b])
                    scope = f"q<={cap}" if cap else f"q={qualities[0]}-{qualities[-1]}"
                    if bd is None:
                        print(f"    BD-rate on {label} ({scope}): no overlap")
                        continue
                    # A BD-rate is an integral over the *overlapping* quality range, so the range
                    # itself decides whether the number means anything. VMAF above ~99 has no
                    # signal left to integrate: on crowd_run the overlap came out 99.55-99.84 and
                    # the BD-rate read +132.4%, which is a saturation artefact and not a result.
                    # The q<=85 cap is not enough on its own — an all-intra arm can already be
                    # saturated at q=25 — so the floor of the overlap is checked, and printed
                    # either way.
                    if label == "VMAF" and lo > 99.0:
                        print(f"    BD-rate on VMAF ({scope}): DISCARDED — overlap "
                              f"{lo:.2f}-{hi:.2f} is saturated, nothing to integrate "
                              f"(the arithmetic said {bd:+.1f}%)")
                        continue
                    verdict = "fewer" if bd < 0 else "more"
                    suspect = "  [suspect: overlap floor >97]" if (
                        label == "VMAF" and lo > 97.0) else ""
                    print(f"    BD-rate on {label} ({scope}): **{bd:+.1f}%** — inter needs "
                          f"{abs(bd):.1f}% {verdict} bits than all-intra "
                          f"(overlap {lo:.2f}-{hi:.2f}){suspect}")

    if args.csv and all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with open(args.csv, "w", newline="") as f:
            wtr = csvmod.DictWriter(f, fieldnames=keys)
            wtr.writeheader()
            wtr.writerows(all_rows)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
