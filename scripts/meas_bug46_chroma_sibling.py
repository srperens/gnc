#!/usr/bin/env python3
"""BUG-46 — RATE-2's bit-exact candidate must be coded in the format the caller asked for.

`lossless_sibling` built from `quality_preset(100)`, which is 4:4:4, and did not carry
`chroma_format`. So on subsampled input RATE-2 compared a 4:2:0 wavelet encode against a **4:4:4**
lossless one — three times the chroma samples — and reported the *same* candidate size whatever
the request (3 257 157 B on bbb at q=97 for 4:4:4 and for 4:2:0). The fallback could essentially
never fire off 4:4:4.

**What "bit-exact" means at 4:2:0, and why the comparison is still two-axis.** The candidate is
exact in the domain the caller chose to code in: the chroma subsampling is the caller's decision,
and both arms pay it. Against the *lossy* arm at the same request the bit-exact one has no
quantiser error at all, so it is better on quality and, when it is smaller, on rate too. This
script measures both arms at each request and reports bytes **and** PSNR against the source, so
"better on both axes" is checked rather than asserted.

Arms, one binary:
  * lossy      — `GNC_LOSSLESS_FALLBACK=0`, the wavelet ladder alone
  * shipped    — default: codes both and keeps the smaller
  * bit-exact  — `-q 100` at the same chroma format, which is what the candidate is

Usage:
    scripts/meas_bug46_chroma_sibling.py --bin <pinned gnc> [--qualities 95,97,99]
"""

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile

STILLS = ["bbb_1080p", "blue_sky_1080p", "kristensara_720p", "touchdown_1080p"]
FORMATS = ["444", "422", "420"]
BYTES_RE = re.compile(r"Compressed:\s+(\d+)\s+bytes")
PSNR_RE = re.compile(r"average:([0-9.]+|inf)")


def repo_root():
    return subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=True).stdout.strip()


def run(cmd, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, capture_output=True, text=True, env=e)
    if p.returncode != 0:
        sys.stderr.write(f"FAILED: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}\n")
        raise SystemExit(1)
    return p.stdout, p.stderr


def encode(gnc, src, out, q, fmt, env=None):
    stdout, _ = run([gnc, "encode", "-i", src, "-o", out, "-q", str(q),
                     "--chroma-format", fmt], env)
    m = BYTES_RE.search(stdout)
    if not m:
        raise SystemExit(f"no byte count in:\n{stdout}")
    return int(m.group(1))


def decode(gnc, comp, png):
    run([gnc, "decode", "-i", comp, "-o", png])
    return png


def psnr(a, b):
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", a, "-i", b, "-lavfi", "psnr",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = PSNR_RE.search(p.stderr)
    if not m:
        raise SystemExit(f"no psnr for {a} vs {b}:\n{p.stderr[-400:]}")
    return float("inf") if m.group(1) == "inf" else float(m.group(1))


def pct(new, old):
    return 100.0 * (new - old) / old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--stills", default=",".join(STILLS))
    ap.add_argument("--formats", default=",".join(FORMATS))
    ap.add_argument("--qualities", default="95,97,99")
    ap.add_argument("--csv")
    ap.add_argument("--workdir")
    args = ap.parse_args()

    gnc = os.path.abspath(args.bin)
    root = repo_root()
    print(f"binary: {gnc}")
    print(f"sha256: {hashlib.sha256(open(gnc, 'rb').read()).hexdigest()}\n")
    work = args.workdir or tempfile.mkdtemp(prefix="bug46_")
    os.makedirs(work, exist_ok=True)
    rows = []

    for still in args.stills.split(","):
        src = os.path.join(root, "test_material/frames", f"{still}.png")
        if not os.path.exists(src):
            print(f"{still}: missing, skipped")
            continue
        for fmt in args.formats.split(","):
            # The bit-exact arm for this request: `-q 100` at the same chroma format.
            bx_path = os.path.join(work, f"{still}_{fmt}_q100.gnc")
            bx_bytes = encode(gnc, src, bx_path, 100, fmt)
            bx_png = decode(gnc, bx_path, os.path.join(work, f"{still}_{fmt}_q100.png"))
            bx_psnr = psnr(src, bx_png)
            print(f"{still}  {fmt}  bit-exact (q=100): {bx_bytes:>10,} B  "
                  f"PSNR {bx_psnr if bx_psnr == float('inf') else round(bx_psnr, 2)}")
            for q in [int(x) for x in args.qualities.split(",")]:
                lossy_path = os.path.join(work, f"{still}_{fmt}_q{q}_lossy.gnc")
                ship_path = os.path.join(work, f"{still}_{fmt}_q{q}_ship.gnc")
                lossy_bytes = encode(gnc, src, lossy_path, q, fmt,
                                     env={"GNC_LOSSLESS_FALLBACK": "0"})
                ship_bytes = encode(gnc, src, ship_path, q, fmt)
                lossy_psnr = psnr(src, decode(gnc, lossy_path,
                                              os.path.join(work, f"{still}_{fmt}_q{q}_l.png")))
                ship_psnr = psnr(src, decode(gnc, ship_path,
                                             os.path.join(work, f"{still}_{fmt}_q{q}_s.png")))
                kept = "bit-exact" if ship_bytes == bx_bytes else "lossy"
                better_both = ship_bytes <= lossy_bytes and ship_psnr >= lossy_psnr - 1e-9
                print(f"    q={q}: lossy {lossy_bytes:>10,} B / {lossy_psnr:.2f} dB   "
                      f"shipped {ship_bytes:>10,} B / "
                      f"{'inf' if ship_psnr == float('inf') else format(ship_psnr, '.2f')} dB   "
                      f"({pct(ship_bytes, lossy_bytes):+.2f}%)  kept the {kept}"
                      + ("" if better_both else "   ** NOT better on both axes"))
                rows.append(dict(still=still, chroma=fmt, q=q,
                                 lossy_bytes=lossy_bytes, shipped_bytes=ship_bytes,
                                 bit_exact_bytes=bx_bytes,
                                 delta_pct=round(pct(ship_bytes, lossy_bytes), 2),
                                 lossy_psnr=round(lossy_psnr, 2),
                                 shipped_psnr=("inf" if ship_psnr == float("inf")
                                               else round(ship_psnr, 2)),
                                 kept=kept, better_on_both=better_both))
            print()

    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
