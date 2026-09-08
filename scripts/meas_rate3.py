#!/usr/bin/env python3
"""RATE-3: does a bit-exact I-frame reference pay on inter, and does it cost quality?

RATE-2 shipped the lossless fallback for stills only, because letting it reach sequence I-frames
made the P-frames referencing them decode at 9.80 dB against 60.69 dB: `local_decode_iframe_gpu`
inverted a MED frame with the wavelet it was not coded with. RATE-3 takes a bit-exact I-frame's
reference from the source instead, and this measures both halves of the result — the rate the
sequence saves, and whether the P-frames it feeds are as good as before.

Both arms come from the same binary and the same command; only `GNC_LOSSLESS_FALLBACK` differs,
so the comparison is exact rather than a BD-rate estimate. Only the **I+P+B** arm is read;
`benchmark-sequence` also prints an all-intra baseline, which is RATE-2's intra win applied to
every frame and a different question.

Usage:  python3 scripts/meas_rate3.py [-q 95 99] [-k 2 9]
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(
    subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                   check=True).stdout.strip()
)
GNC = REPO / "target" / "release" / "gnc"
SEQ_DIR = REPO / "test_material" / "frames" / "sequences"

# (name, frames). bbb ships 8 PNGs; asking for more fails silently under 2>/dev/null, which is
# how the first version of this sweep lost a whole sequence.
SEQUENCES = [("bbb", 8), ("crowd_run", 10), ("old_town_cross", 10)]

TOTAL = re.compile(r"^\s+Total:\s+(\d+) bytes")
PFRAME = re.compile(r"^\s+Frame\s+\d+ \[P\]:.*PSNR ([\d.]+) dB")
IFRAME = re.compile(r"^\s+Frame\s+\d+ \[I\]:.*PSNR (inf|[\d.]+) dB")


def run(seq, frames, q, k, fallback):
    env = dict(os.environ)
    if not fallback:
        env["GNC_LOSSLESS_FALLBACK"] = "0"
    cmd = [
        str(GNC), "benchmark-sequence",
        "-i", str(SEQ_DIR / seq / "frame_%04d.png"),
        "-q", str(q), "-n", str(frames), "-k", str(k),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.exit(f"gnc failed on {seq} q={q} ki={k}:\n{p.stderr[-3000:]}")
    # The I+P+B arm only; the all-intra baseline that follows is a different question.
    text = p.stdout
    start = text.index("=== I+P+B")
    end = text.index("=== All I-frames", start)
    arm = text[start:end]

    total = next((int(m.group(1)) for m in map(TOTAL.match, arm.splitlines()) if m), None)
    pf = [float(m.group(1)) for m in map(PFRAME.match, arm.splitlines()) if m]
    isf = [m.group(1) for m in map(IFRAME.match, arm.splitlines()) if m]
    if total is None or not pf or not isf:
        sys.exit(f"could not parse the I+P arm for {seq} q={q} ki={k}; output shape changed")
    return total, min(pf), sum(pf) / len(pf), all(v == "inf" for v in isf), len(isf), len(pf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--quality", type=int, nargs="+", default=[95, 99])
    ap.add_argument("-k", "--keyframe-interval", type=int, nargs="+", default=[2, 9])
    args = ap.parse_args()
    if not GNC.exists():
        sys.exit(f"{GNC} not built — run cargo build --release")

    print(f"{'sequence':>15} {'q':>4} {'ki':>3} {'bytes off':>10} {'bytes on':>10} {'Δbytes':>8} "
          f"{'worst P off':>12} {'worst P on':>11} {'ΔP':>7} {'I bit-exact':>12}")
    rows = []
    for seq, frames in SEQUENCES:
        for q in args.quality:
            for k in args.keyframe_interval:
                off = run(seq, frames, q, k, fallback=False)
                on = run(seq, frames, q, k, fallback=True)
                d_bytes = on[0] / off[0] - 1
                d_p = on[1] - off[1]
                rows.append((seq, q, k, d_bytes, d_p, on[3]))
                print(f"{seq:>15} {q:>4} {k:>3} {off[0]:>10} {on[0]:>10} {d_bytes:>+7.2%} "
                      f"{off[1]:>12.2f} {on[1]:>11.2f} {d_p:>+7.2f} {str(on[3]):>12}")

    n = len(rows)
    print(f"\n{'MEAN':>15} {'':>4} {'':>3} {'':>10} {'':>10} "
          f"{sum(r[3] for r in rows)/n:>+7.2%} {'':>12} {'':>11} "
          f"{sum(r[4] for r in rows)/n:>+7.2f}")
    worst_bytes = max(rows, key=lambda r: r[3])
    worst_p = min(rows, key=lambda r: r[4])
    print(f"\nworst byte point: {worst_bytes[0]} q={worst_bytes[1]} ki={worst_bytes[2]} "
          f"{worst_bytes[3]:+.2%}  (criterion: not larger)")
    print(f"worst P point:    {worst_p[0]} q={worst_p[1]} ki={worst_p[2]} "
          f"{worst_p[4]:+.2f} dB  (criterion: within 0.1 dB)")
    print(f"I-frames bit-exact on all {n} points: {all(r[5] for r in rows)}")


if __name__ == "__main__":
    main()
