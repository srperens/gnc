#!/usr/bin/env python3
"""MEAS-2, remaining toggles — the B-pyramid and its two layer quantiser scales.

MEAS-2 measured five toggles and left two: `GNC_PYRAMID_L2_QP_SCALE` (default 1.0, the comment
says "off until validated" and it never was) and `GNC_PYRAMID_L3_QP_SCALE` (default 1.5, borrowed
from H.264's QP+4 practice for inner B-frames, never measured here). Both sit behind
`GNC_B_PYRAMID`, which BUG-5 turned off.

Three things about the shape of the run:

  * **Both arms use the same keyframe interval.** BUG-5 priced the pyramid as ki=17-with-B against
    ki=8-P-only, which moves GOP length and the B-pyramid at the same time — the conflation TUNE-1
    was warned about ("its -24% for longer GOPs was read as GNC uses too few B-frames"). Here the
    reference is ki=17 with the pyramid *suppressed*, so the only difference between arms is the
    toggle under test. A figure from this script is therefore not comparable with BUG-5's.
  * **One common quality interval per sequence and metric.** A coarser layer scale lowers the top
    of the arm's ladder, so each arm's own overlap with the reference is a different interval, and
    QUAL-1 measured what ranking arms over different intervals does: 47.5 points of movement on
    average. `bd_rate_over` from `meas_inter1_pscale` is reused for exactly that reason.
  * **PSNR leads, and worst-frame PSNR is a headline.** This runs at q >= 85 where VMAF is
    saturated (CLAUDE.md's metric table); the VMAF BD-rate is computed and discarded when its
    overlap floor is above 99. Worst-frame PSNR gets its own column because a pyramid trades tail
    quality for mean by construction — the leaf B-frames are the coarsest frames in the GOP.

**Canary.** Every arm is proved to have run: a pyramid arm must print `[pyramid_b] ... layer=2
... (l2_scale=Nx)` and `layer=3 ... (l3_scale=Nx)` with the scales this harness asked for, and the
reference arm must print the `B-pyramid suppressed` line. The layer-2 half of that canary did not
exist before this item — the diagnostic printed the references and not the quantiser, so
`GNC_PYRAMID_L2_QP_SCALE` had no way to be observed at all, which is how a knob stays
"unvalidated" for a week.

Usage:
    scripts/meas2_pyramid_qp.py                                   # the l3 sweep at l2=1.0
    scripts/meas2_pyramid_qp.py --grid 1.25:1.5,1.5:1.5           # then the l2 sweep
"""

import argparse
import csv as csvmod
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meas3_sequence_rd import repo_root, run_point  # noqa: E402
from meas_inter1_pscale import bd_rate_over  # noqa: E402

# (key, label, saturation floor above which the figure is discarded)
METRICS = (
    ("psnr_avg", "mean PSNR", None),
    ("psnr_min", "worst-frame PSNR", None),
    ("vmaf_mean", "VMAF", 99.0),
)

OFF = "off"  # the reference arm's label: ki as given, pyramid suppressed


def arm_env(arm):
    """The environment for one arm. `OFF` leaves GNC_B_PYRAMID unset (the shipped default)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("GNC_B_PYRAMID", "GNC_PYRAMID_L2_QP_SCALE", "GNC_PYRAMID_L3_QP_SCALE")}
    if arm == OFF:
        return env
    l2, l3 = arm
    env["GNC_B_PYRAMID"] = "1"
    env["GNC_PYRAMID_L2_QP_SCALE"] = f"{l2}"
    env["GNC_PYRAMID_L3_QP_SCALE"] = f"{l3}"
    return env


def arm_label(arm):
    return "pyramid off (P-only)" if arm == OFF else f"pyramid l2={arm[0]} l3={arm[1]}"


def canary(gnc, pattern, arm, ki, q, chroma):
    """Prove the arm's code path ran and read the scales this harness set.

    9 frames is the smallest run that contains a whole pyramid group (I0 + 7 B + P8).
    """
    env = dict(arm_env(arm), GNC_BFRAME_PYRAMID="1")
    r = subprocess.run(
        [str(gnc), "benchmark-sequence", "-i", str(pattern), "-n", "9", "-k", str(ki),
         "-q", str(q), "--chroma-format", chroma],
        capture_output=True, text=True, env=env,
    )
    out = r.stdout + r.stderr
    if arm == OFF:
        if "B-pyramid suppressed" not in out:
            sys.exit("canary: the reference arm did not print 'B-pyramid suppressed' — it may "
                     "have coded B-frames, and then both arms are the same feature")
        return "suppressed"
    seen = {}
    for layer in (2, 3):
        # The layer-3 line also reports whether the scale came from the env var or the taper
        # (`0061`), so the source token is optional here rather than assumed away.
        m = re.search(
            rf"layer={layer}\b[^\n]*qstep=([0-9.]+) \((?:[a-z]+, )?l{layer}_scale=([0-9.]+)x",
            out,
        )
        if not m:
            sys.exit(f"canary: no layer={layer} line with a scale under GNC_BFRAME_PYRAMID=1 — "
                     f"cannot prove GNC_PYRAMID_L{layer}_QP_SCALE was read")
        qstep, scale = float(m.group(1)), float(m.group(2))
        want = float(arm[layer - 2])
        if abs(scale - want) > 1e-6:
            sys.exit(f"canary: asked for l{layer}={want} and the encoder read {scale} — "
                     f"the sweep would measure the wrong thing")
        seen[layer] = (qstep, scale)
    return " ".join(f"l{k}: qstep={v[0]:.2f} ({v[1]:.2f}x)" for k, v in sorted(seen.items()))


def finite_or_die(seq, arm, pts, key):
    """Drop and report a rung whose metric is not finite.

    LOSSLESS-3 (`0073`) emits a q=95..99 4:4:4 camera sequence bit-exact, so `psnr()` returns
    `inf` up there and a Bjontegaard fit over an infinity is a silent non-number — the trap
    INTRA-1 hit and guarded in its own harness. Worse here than there: a bit-exact encode is
    all-intra, so every arm's rung is the *same bytes*, and the toggle under test is not even in
    the output being compared.
    """
    ok = [p for p in pts if p.get(key) is not None and math.isfinite(p[key])]
    for p in pts:
        if p.get(key) is not None and not math.isfinite(p[key]):
            print(f"    !! {seq} {arm} q={p['q']}: {key} is not finite — bit-exact output "
                  f"(LOSSLESS-3), rung dropped; a BD-rate spanning it is not a number")
    return ok


def monotonic_flags(pts):
    """Rungs whose rate falls while q rises (RATE-2)."""
    return [(a["q"], b["q"], a["bpp"], b["bpp"]) for a, b in zip(pts, pts[1:]) if b["bpp"] < a["bpp"]]


def print_arm(label, pts):
    print(f"  {label}")
    print(f"    {'q':>4} {'bpp':>9} {'PSNR avg':>9} {'PSNR min':>9} {'VMAF':>7} {'bytes':>12}")
    for p in pts:
        v = f"{p['vmaf_mean']:.2f}" if p.get("vmaf_mean") is not None else "—"
        print(f"    {p['q']:>4} {p['bpp']:>9.4f} {p['psnr_avg']:>9.2f} {p['psnr_min']:>9.2f} "
              f"{v:>7} {p['bytes']:>12}")
    for q_lo, q_hi, b_lo, b_hi in monotonic_flags(pts):
        print(f"    !! NON-MONOTONIC: q={q_lo} costs {b_lo:.4f} bpp, q={q_hi} costs {b_hi:.4f} "
              f"— RATE-2; a BD-rate spanning this rung is not a number")


def main():
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", default="crowd_run,old_town_cross,bbb_extended")
    ap.add_argument("--dir", default="test_material/frames/sequences")
    ap.add_argument("--qualities", default="85,90,92,95,99")
    ap.add_argument("--grid", default="1.0:1.0,1.0:1.5,1.0:2.0",
                    help="l2:l3 pairs to measure against the pyramid-off reference")
    ap.add_argument("--ki", type=int, default=17,
                    help="the same keyframe interval for every arm; >=9 or the pyramid cannot run")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--chroma-format", default="444")
    ap.add_argument("--gnc-binary", default=str(root / "target/release/gnc"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.gnc_binary):
        sys.exit(f"GNC binary not found: {args.gnc_binary} (cargo build --release)")
    if args.ki < 9:
        sys.exit(f"ki={args.ki} suppresses B-frames by itself (needs ki >= 9) — both arms would "
                 f"be P-only and the toggle would measure 0.0% for the wrong reason")
    qualities = [int(q) for q in args.qualities.split(",")]
    arms = [OFF] + [tuple(p.split(":")) for p in args.grid.split(",")]

    seq0 = Path(args.dir) / args.sequences.split(",")[0] / "frame_%04d.png"
    print(f"Canary — what each arm's quantiser resolves to (ki={args.ki}, q={qualities[0]}):")
    for arm in arms:
        print(f"    {arm_label(arm):<28} {canary(args.gnc_binary, seq0, arm, args.ki, qualities[0], args.chroma_format)}")

    all_rows, data = [], {}
    with tempfile.TemporaryDirectory(prefix="meas2_pyr_") as td:
        for seq in args.sequences.split(","):
            seq_dir = Path(args.dir) / seq
            pattern = seq_dir / "frame_%04d.png"
            n_avail = len(list(seq_dir.glob("frame_*.png")))
            if n_avail == 0:
                print(f"\n=== {seq}: no frames at {seq_dir}, skipped ===")
                continue
            frames = min(args.frames, n_avail)
            print(f"\n=== {seq}  ({frames} of {n_avail} frames, ki={args.ki}, "
                  f"{args.chroma_format}, q={args.qualities}) ===")
            for arm in arms:
                # Build the arm's environment *before* clearing, or `arm_env` copies an empty
                # one: the encode then runs without PATH and VMAF silently reports
                # "failed (is vmaf in PATH?)" for every point.
                env_arm, env_saved = arm_env(arm), dict(os.environ)
                os.environ.clear()
                os.environ.update(env_arm)
                pts = []
                # One directory per arm: `run_point` names its CSV from (sequence, q, ki), all
                # three of which are equal across arms here, so a shared directory would let a
                # failed encode read the previous arm's file back as its own result.
                arm_dir = Path(td) / re.sub(r"[^0-9a-z]+", "_", arm_label(arm).lower())
                arm_dir.mkdir(exist_ok=True)
                for q in qualities:
                    res = run_point(args.gnc_binary, pattern, frames, args.ki, q,
                                    args.chroma_format, arm_dir)
                    row = {"sequence": seq, "arm": arm_label(arm), "ki": args.ki, "q": q}
                    if "failed" in res:
                        print(f"    {arm_label(arm)} q={q} FAILED: {res['failed']}")
                        all_rows.append({**row, "failed": res["failed"]})
                        continue
                    res.pop("stdout", None)
                    row.update(res)
                    pts.append(row)
                    all_rows.append(row)
                os.environ.clear()
                os.environ.update(env_saved)
                data[(seq, arm)] = pts
                print_arm(arm_label(arm), pts)

    sequences = [s for s in args.sequences.split(",") if (s, OFF) in data]
    for key, label, sat_floor in METRICS:
        print(f"\n=== BD-rate of each pyramid arm against pyramid-off, on {label}, "
              f"over one common interval per sequence ===")
        table = {}
        for seq in sequences:
            ref = finite_or_die(seq, arm_label(OFF), data[(seq, OFF)], key)
            got = {a: finite_or_die(seq, arm_label(a), data.get((seq, a), []), key)
                   for a in arms if a != OFF}
            got = {a: v for a, v in got.items() if len(v) >= 4}
            if len(ref) < 4 or not got:
                continue
            lo = max([min(p[key] for p in ref)] + [min(p[key] for p in v) for v in got.values()])
            hi = min([max(p[key] for p in ref)] + [max(p[key] for p in v) for v in got.values()])
            if sat_floor is not None and lo > sat_floor:
                print(f"  {seq}: DISCARDED — overlap {lo:.2f}-{hi:.2f} is saturated, "
                      f"nothing to integrate")
                continue
            vals = {a: bd_rate_over([p["bpp"] for p in ref], [p[key] for p in ref],
                                    [p["bpp"] for p in v], [p[key] for p in v], lo, hi)
                    for a, v in got.items()}
            table[seq] = (lo, hi, vals)
        if not table:
            continue
        cols = [a for a in arms if a != OFF]
        hdr = f"{'sequence':<18} {'interval':>16}  " + "  ".join(
            f"{'l2=' + a[0] + ' l3=' + a[1]:>15}" for a in cols)
        print(hdr)
        print("-" * len(hdr))
        for seq, (lo, hi, vals) in table.items():
            print(f"{seq:<18} {f'{lo:.2f}-{hi:.2f}':>16}  " + "  ".join(
                (f"{vals[a]:>+14.1f}%" if vals.get(a) is not None else f"{'—':>15}") for a in cols))
        means = {}
        for a in cols:
            vs = [t[2][a] for t in table.values() if t[2].get(a) is not None]
            if len(vs) == len(table) and vs:
                means[a] = sum(vs) / len(vs)
        if means:
            print("-" * len(hdr))
            print(f"{'MEAN':<18} {'':>16}  " + "  ".join(
                (f"{means[a]:>+14.1f}%" if a in means else f"{'—':>15}") for a in cols))
            best = min(means, key=means.get)
            print(f"\n  best on {label}: l2={best[0]} l3={best[1]} at {means[best]:+.1f}% "
                  f"(negative = cheaper than pyramid off)")

    if args.csv and all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with open(args.csv, "w", newline="") as f:
            wtr = csvmod.DictWriter(f, fieldnames=keys)
            wtr.writeheader()
            wtr.writerows(all_rows)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
