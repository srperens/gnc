#!/usr/bin/env python3
"""ENT-7 step 3: price BPC-PaCo's stationary, coefficient-parallel model on shipped coefficients.

Runs the `GNC_COEF_ENTROPY=1` diagnostic (`src/encoder/bpc_paco_diag.rs`) over the four stills of
decision 0024 at the contribution quality points, and reports the three numbers ENT-7's success
criterion needs:

  Hbpc    BPC-PaCo's own two-column lockstep scan and its 14 contexts per bitplane, with the
          probabilities pooled per subband *of the image being coded*. An oracle table; generous
          on purpose, the same convention Hctx and Hnb already use.
  Hbpcn   the same model with the neighbourhood frozen at each plane boundary — a WGSL port with
          no cross-lane exchange, since core WebGPU has no subgroup operations.
  Hbpcf   Hbpc's model priced against a table trained on the *other three* images at the same
          quality point — leave-one-image-out, which is what a stationary coder really ships.
  flw     the excess bits in the final codeword of each of a block's stripe coders, which TIP
          2016 §IV identifies as the whole of BPC-PaCo's rate penalty against JPEG 2000.

Two passes are needed because the third column requires the other images' statistics: pass 1
dumps counts per (image, q), pass 2 prices each image against the concatenation of the rest.

Usage:
    python3 scripts/meas_ent7_bpc.py [-q 85 90] [--images bbb_1080p ...] [--csv out.csv]
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile

REPO = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
).stdout.strip()
GNC = os.path.join(REPO, "target", "release", "gnc")
FRAMES = os.path.join(REPO, "test_material", "frames")

# The four stills decision 0024 and ENT-4 used. Same set, or it is not a comparison.
IMAGES = ["bbb_1080p", "blue_sky_1080p", "kristensara_720p", "touchdown_1080p"]

TOTAL_BPC = re.compile(
    r"^\s+TOTAL\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(NaN|\d+)\s+(\d+)\s+([-+][\d.]+)%\s+"
    r"([-+][\d.]+)%\s+([\d.]+)%"
)
TOTAL_BOUNDS = re.compile(r"^\s+TOTAL\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s")
BAND_ROW = re.compile(
    r"^\s+(Y|Co|Cg)\s+(LL|[HL]{2}\d|ALL)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(NaN|\d+)\s"
)


def run(image, q, env_extra):
    """One encode with the diagnostic on. Returns (stdout+stderr, parsed totals, band rows)."""
    env = dict(os.environ, GNC_COEF_ENTROPY="1", **env_extra)
    cmd = [GNC, "benchmark", "-i", os.path.join(FRAMES, image + ".png"), "-q", str(q),
           "--abac", "-n", "1"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.exit(f"gnc failed on {image} q={q}:\n{p.stderr[-4000:]}")
    text = p.stdout + p.stderr
    bounds, bpc, bands = None, None, []
    for line in text.splitlines():
        m = TOTAL_BOUNDS.match(line)
        if m and bounds is None:
            bounds = [float(x) for x in m.groups()]
            continue
        m = TOTAL_BPC.match(line)
        if m:
            g = m.groups()
            bpc = dict(
                shipped=float(g[0]), hctx=float(g[1]), hbpc=float(g[2]), hbpcn=float(g[3]),
                hbpcf=float("nan") if g[4] == "NaN" else float(g[4]), flw=float(g[5]),
                miss=float(g[8]),
            )
            continue
        m = BAND_ROW.match(line)
        if m and bpc is None:
            g = m.groups()
            bands.append((g[0], g[1], *[float(x) for x in g[2:6]],
                          float("nan") if g[6] == "NaN" else float(g[6])))
    if bounds is None or bpc is None:
        sys.exit(f"could not parse the diagnostic output for {image} q={q} — "
                 "is GNC_COEF_ENTROPY still printing both TOTAL rows?")
    return bpc, bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--quality", type=int, nargs="+", default=[85, 90])
    ap.add_argument("--images", nargs="+", default=IMAGES)
    ap.add_argument("--csv")
    args = ap.parse_args()

    if not os.path.exists(GNC):
        sys.exit(f"{GNC} not built — run cargo build --release")

    tmp = tempfile.mkdtemp(prefix="ent7bpc-")
    rows = []
    for q in args.quality:
        # Pass 1: dump each image's context counts at this q.
        dumps = {}
        for img in args.images:
            path = os.path.join(tmp, f"{img}-q{q}.tsv")
            run(img, q, {"GNC_BPC_DUMP": path})
            dumps[img] = path
            print(f"  dumped {img} q={q}")

        # Pass 2: price each image against the other three, concatenated.
        for img in args.images:
            others = [dumps[o] for o in args.images if o != img]
            table = os.path.join(tmp, f"train-not-{img}-q{q}.tsv")
            with open(table, "w") as out:
                for o in others:
                    with open(o) as f:
                        out.write(f.read())
            bpc, bands = run(img, q, {"GNC_BPC_TABLE": table})
            bpc.update(image=img, q=q)
            rows.append(bpc)
            print(
                f"{img:>18} q={q}  shipped {bpc['shipped']:>9.0f}  "
                f"Hbpc {bpc['hbpc']/bpc['shipped']-1:+7.2%}  "
                f"+flw {(bpc['hbpc']+bpc['flw'])/bpc['shipped']-1:+7.2%}  "
                f"Hbpcf+flw {(bpc['hbpcf']+bpc['flw'])/bpc['shipped']-1:+7.2%}  "
                f"no-exchange {bpc['hbpcn']/bpc['hbpc']-1:+7.2%}  miss {bpc['miss']:.2f}%"
            )

    print("\n=== ENT-7 step 3 summary: everything as a percentage of what abac shipped ===")
    print(
        f"{'image':>18} {'q':>4} {'Hbpc':>9} {'Hbpc+flw':>9} {'Hbpcf':>9} {'Hbpcf+flw':>10} "
        f"{'no-exch':>9}"
    )
    for r in rows:
        print(
            f"{r['image']:>18} {r['q']:>4} {r['hbpc']/r['shipped']-1:>+8.2%} "
            f"{(r['hbpc']+r['flw'])/r['shipped']-1:>+8.2%} "
            f"{r['hbpcf']/r['shipped']-1:>+8.2%} "
            f"{(r['hbpcf']+r['flw'])/r['shipped']-1:>+9.2%} "
            f"{r['hbpcn']/r['hbpc']-1:>+8.2%}"
        )
    for q in args.quality:
        sel = [r for r in rows if r["q"] == q]
        if not sel:
            continue
        n = len(sel)
        print(
            f"{'MEAN':>18} {q:>4} "
            f"{sum(r['hbpc']/r['shipped']-1 for r in sel)/n:>+8.2%} "
            f"{sum((r['hbpc']+r['flw'])/r['shipped']-1 for r in sel)/n:>+8.2%} "
            f"{sum(r['hbpcf']/r['shipped']-1 for r in sel)/n:>+8.2%} "
            f"{sum((r['hbpcf']+r['flw'])/r['shipped']-1 for r in sel)/n:>+9.2%} "
            f"{sum(r['hbpcn']/r['hbpc']-1 for r in sel)/n:>+8.2%}"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")
    print(f"dumps and tables kept in {tmp}")


if __name__ == "__main__":
    main()
