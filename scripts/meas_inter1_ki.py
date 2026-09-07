#!/usr/bin/env python3
"""INTER-1 — is the inter path's loss at contribution quality a *default* problem or a *feature*
problem, and what is TUNE-6's P-frame quantiser scale worth up there?

MEAS-3 measured one inter configuration (ki=9) against all-intra over q=25-99 and found +4.6% on
mean PSNR, +19.1% on worst-frame PSNR. That is a verdict on *the shipped default*, not on inter
coding: a GOP of 9 and a GOP of 2 are different features priced by the same machinery. INTER-1
step 1 is therefore a ki sweep at the operating point the codec is for, and this is that sweep.

Three things about the shape of the run, each of them a rule this repo has already paid for:

  * **Every arm is BD-rated against all-intra, and worst-frame PSNR gets its own BD-rate.**
    COORDINATION: "quote worst-frame PSNR next to the mean for anything touching the inter path" —
    MEAS-3's mean said +4.6% (reads neutral) while its worst frame said +19.1%. For a contribution
    codec the worst frame is what survives downstream re-encoding, so it is a headline number here
    and not a footnote.
  * **PSNR leads and VMAF is only reported where it has signal.** This runs at q >= 85, which is
    squarely in the saturated range (CLAUDE.md's table, and QUAL-1's 47.5-point ladder
    sensitivity). The VMAF BD-rate is computed, printed with its overlap, and *discarded* when the
    overlap floor exceeds 99 — the guard `meas3_sequence_rd.py` already carries.
  * **The ladder is checked for monotonicity in rate before anything is integrated.** RATE-2: GNC's
    own ladder is not monotonic up here (flat512 costs 0.0450 bpp at q=86 and 0.0370 at q=90) and
    above q~95-98 the lossy rungs cost more than bit-exact lossless. A BD-rate over a ladder that
    doubles back is not a number, so a non-monotonic arm is flagged loudly and the restricted
    ladder is reported beside the full one.

`run_point` and `bd_rate` are imported rather than reimplemented — one point runner and one
BD-rate for the whole repo, which is what let MEAS-3's VMAF-saturation guard be written once.

Phase 2 (`--p-qp-scale`) re-runs the same sweep with `GNC_P_QP_SCALE` pinned, so TUNE-6's taper can
be priced against a fixed ladder. The encoder prints a `p_qp_scale=... (env|taper, default ...)`
canary under `GNC_DIAGNOSTICS=1`; this harness asserts it, because at q >= 85 the taper *already*
returns 1.0 and an override to 1.0 is correctly a no-op — "the knob did nothing" and "the knob was
never read" are the two outcomes that must not be confused.

Usage:
    scripts/meas_inter1_ki.py                                  # step 1: the ki sweep
    scripts/meas_inter1_ki.py --p-qp-scale 1.25 --arms 9       # step 2: price the scale
"""

import argparse
import csv as csvmod
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meas1_vs_h264 import bd_rate  # noqa: E402
from meas3_sequence_rd import repo_root, run_point  # noqa: E402

# The metrics a BD-rate is taken on, and whether they saturate. `cap` restricts the ladder;
# `sat_floor` discards the result when the overlap floor is above it (nothing left to integrate).
METRICS = (
    ("psnr_avg", "PSNR mean", None, None),
    ("psnr_min", "PSNR worst-frame", None, None),
    ("vmaf_mean", "VMAF", None, 99.0),
)


def canary_p_qp_scale(gnc, pattern, q, chroma, expect_env):
    """Prove the P-scale knob was read, and report what it resolved to.

    Returns (scale, source). Runs 3 frames at ki=3 — enough to encode a P-frame, which is the
    only frame type that reaches the taper at all.
    """
    env = dict(os.environ, GNC_DIAGNOSTICS="1")
    r = subprocess.run(
        [str(gnc), "benchmark-sequence", "-i", str(pattern), "-n", "3", "-k", "3",
         "-q", str(q), "--chroma-format", chroma],
        capture_output=True, text=True, env=env,
    )
    for ln in r.stdout.splitlines():
        if "p_qp_scale=" in ln:
            body = ln.strip()
            scale = float(body.split("p_qp_scale=")[1].split()[0])
            source = body.split("(")[1].split(",")[0]
            if expect_env and source != "env":
                sys.exit(f"canary: GNC_P_QP_SCALE was set but the encoder read the {source} "
                         f"({body}) — the sweep would measure nothing")
            return scale, source
    sys.exit("canary: no p_qp_scale line under GNC_DIAGNOSTICS=1 — cannot prove the path ran")


def monotonic_flags(pts):
    """Rungs whose rate falls while q rises (RATE-2). Returns a list of (q_lo, q_hi, bpp_lo, bpp_hi)."""
    bad = []
    for a, b in zip(pts, pts[1:]):
        if b["bpp"] < a["bpp"]:
            bad.append((a["q"], b["q"], a["bpp"], b["bpp"]))
    return bad


def print_arm(label, pts):
    print(f"  {label}")
    print(f"    {'q':>4} {'bpp':>9} {'PSNR avg':>9} {'PSNR min':>9} {'VMAF':>7} {'VMAF min':>9}"
          f" {'bytes':>12}")
    for p in pts:
        v = f"{p['vmaf_mean']:.2f}" if p.get("vmaf_mean") is not None else "—"
        vm = f"{p['vmaf_min']:.2f}" if p.get("vmaf_min") is not None else "—"
        print(f"    {p['q']:>4} {p['bpp']:>9.4f} {p['psnr_avg']:>9.2f} {p['psnr_min']:>9.2f} "
              f"{v:>7} {vm:>9} {p['bytes']:>12}")
    for q_lo, q_hi, b_lo, b_hi in monotonic_flags(pts):
        print(f"    !! NON-MONOTONIC: q={q_lo} costs {b_lo:.4f} bpp, q={q_hi} costs {b_hi:.4f} "
              f"— RATE-2; a BD-rate spanning this rung is not a number")


def report_bd(ref_pts, arm_pts, ref_label, arm_label, qualities):
    """BD-rate of arm against ref on every metric. Negative = the arm needs fewer bits."""
    out = {}
    for key, label, cap, sat_floor in METRICS:
        a = [p for p in ref_pts if p.get(key) is not None and (cap is None or p["q"] <= cap)]
        b = [p for p in arm_pts if p.get(key) is not None and (cap is None or p["q"] <= cap)]
        if len(a) < 4 or len(b) < 4:
            print(f"    BD-rate on {label}: not computed (only {min(len(a), len(b))} points)")
            continue
        bd, (lo, hi) = bd_rate([p["bpp"] for p in a], [p[key] for p in a],
                               [p["bpp"] for p in b], [p[key] for p in b])
        scope = f"q={qualities[0]}-{qualities[-1]}"
        if bd is None:
            print(f"    BD-rate on {label} ({scope}): no overlap")
            continue
        if sat_floor is not None and lo > sat_floor:
            print(f"    BD-rate on {label} ({scope}): DISCARDED — overlap {lo:.2f}-{hi:.2f} is "
                  f"saturated, nothing to integrate (the arithmetic said {bd:+.1f}%)")
            continue
        verdict = "fewer" if bd < 0 else "more"
        print(f"    BD-rate on {label} ({scope}): **{bd:+.1f}%** — {arm_label} needs "
              f"{abs(bd):.1f}% {verdict} bits than {ref_label} (overlap {lo:.2f}-{hi:.2f})")
        out[key] = bd
    return out


def main():
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", default="crowd_run,old_town_cross,bbb_extended")
    ap.add_argument("--dir", default="test_material/frames/sequences")
    ap.add_argument("--qualities", default="85,90,92,95,99",
                    help="the contribution ladder; INTER-1 asks for q=85-99")
    ap.add_argument("--arms", default="1,2,4,9",
                    help="keyframe intervals; 1 (all-intra) is the reference and is required")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--chroma-format", default="444")
    ap.add_argument("--p-qp-scale", default=None,
                    help="pin GNC_P_QP_SCALE for the whole run (INTER-1 step 2)")
    ap.add_argument("--gnc-binary", default=str(root / "target/release/gnc"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.gnc_binary):
        sys.exit(f"GNC binary not found: {args.gnc_binary} (cargo build --release)")
    qualities = [int(q) for q in args.qualities.split(",")]
    kis = [int(k) for k in args.arms.split(",")]
    ref_ki = 1
    if ref_ki not in kis:
        sys.exit("--arms must include 1 (all-intra), which is the reference every arm is priced against")

    if args.p_qp_scale is not None:
        os.environ["GNC_P_QP_SCALE"] = args.p_qp_scale
    seq0 = Path(args.dir) / args.sequences.split(",")[0] / "frame_%04d.png"
    print("Canary — what the P-frame quantiser taper resolves to at each rung "
          f"({'GNC_P_QP_SCALE=' + args.p_qp_scale if args.p_qp_scale else 'shipped taper'}):")
    for q in qualities:
        scale, source = canary_p_qp_scale(args.gnc_binary, seq0, q, args.chroma_format,
                                          expect_env=args.p_qp_scale is not None)
        print(f"    q={q:>3}  p_qp_scale={scale:.4f}  ({source})")

    all_rows = []
    with tempfile.TemporaryDirectory(prefix="inter1_") as td:
        for seq in args.sequences.split(","):
            seq_dir = Path(args.dir) / seq
            pattern = seq_dir / "frame_%04d.png"
            n_avail = len(list(seq_dir.glob("frame_*.png")))
            if n_avail == 0:
                print(f"\n=== {seq}: no frames at {seq_dir}, skipped ===")
                continue
            frames = min(args.frames, n_avail)
            print(f"\n=== {seq}  ({frames} of {n_avail} frames, ki={'/'.join(map(str, kis))}, "
                  f"{args.chroma_format}, q={args.qualities}) ===")

            arms = {}
            for ki in kis:
                pts = []
                for q in qualities:
                    res = run_point(args.gnc_binary, pattern, frames, ki, q,
                                    args.chroma_format, Path(td))
                    if "failed" in res:
                        print(f"    ki={ki} q={q} FAILED: {res['failed']}")
                        all_rows.append({"sequence": seq, "ki": ki, "q": q,
                                         "p_qp_scale": args.p_qp_scale or "taper",
                                         "failed": res["failed"]})
                        continue
                    res.pop("stdout", None)
                    row = {"sequence": seq, "ki": ki, "q": q,
                           "p_qp_scale": args.p_qp_scale or "taper", **res}
                    pts.append(row)
                    all_rows.append(row)
                arms[ki] = pts
                tag = " (all-intra, reference)" if ki == ref_ki else (
                    " (shipped inter)" if ki == 9 else "")
                print_arm(f"ki={ki}{tag}", pts)

            ref = arms.get(ref_ki, [])
            if len(ref) < 4:
                print("    reference arm has <4 points, no BD-rate")
                continue
            for ki in kis:
                if ki == ref_ki or len(arms.get(ki, [])) < 4:
                    continue
                print(f"  --- ki={ki} against all-intra ---")
                report_bd(ref, arms[ki], "all-intra", f"ki={ki}", qualities)

    if args.csv and all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with open(args.csv, "w", newline="") as f:
            wtr = csvmod.DictWriter(f, fieldnames=keys)
            wtr.writeheader()
            wtr.writerows(all_rows)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
