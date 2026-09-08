#!/usr/bin/env python3
"""LOSSLESS-2 — is a lossless P-frame ever worth coding, and can the choice be made per frame?

**The question.** At `q=100` the whole-sequence figures say inter loses badly on camera content
(+38.1% crowd_run, +39.5% old_town_cross) and wins slightly on animation (−1.6% bbb). BACKLOG
asks whether a lossless configuration should code P-frames at all, and whether the choice belongs
to the sequence or to the frame.

**Why the per-frame minimum is not an oracle here.** Since BUG-39 (`0064`) every frame at `q=100`
decodes bit-exact, so an encoded frame's reconstruction *is* its source frame whichever way it was
coded. A P-frame therefore predicts from the same pixels no matter what the frames before it
chose, and its byte count does not depend on those choices. That makes `sum_i min(I_i, P_i)`
**exactly achievable by a real encoder**, not an upper bound — which is the property this script
measures rather than assumes: the I-frame byte counts of the all-intra arm are compared frame by
frame against the I-frames the I+P arm codes at the same indices, and any mismatch voids the run.

**Domain declaration.** The comparison is whole coded frames — container payload per frame as the
encoder reports it, at identical (bit-exact) pixels. Not a BD-rate, not a quality trade: both arms
decode to the source, so bytes are the only axis.

Usage:
    scripts/meas_lossless2_inter.py --bin <pinned gnc> [--sequences a,b] [--ki 2,9] [--csv out.csv]
"""

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile

SEQUENCES = ["crowd_run", "old_town_cross", "bbb", "blue_sky"]
FRAME_RE = re.compile(r"^\s*Frame\s+(\d+)\s+\[([IPB])\]:\s+(\d+)\s+bytes")
MIX_RE = re.compile(r"Encoded\s+\d+\s+frames\s+\(([^)]*)\)")


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


def encode(gnc, pattern, out, frames, ki, chroma, env=None):
    """One arm. Returns ([(idx, type, bytes)], container_size, mix)."""
    stdout, _ = run([gnc, "encode-sequence", "-i", pattern, "-o", out,
                     "-q", "100", "-n", str(frames),
                     "--keyframe-interval", str(ki),
                     "--chroma-format", chroma], env)
    rows = []
    for line in stdout.splitlines():
        m = FRAME_RE.match(line)
        if m:
            rows.append((int(m.group(1)), m.group(2), int(m.group(3))))
    mix = MIX_RE.search(stdout)
    if len(rows) != frames or mix is None:
        raise SystemExit(f"parsed {len(rows)} of {frames} frame lines, mix={mix}")
    return rows, os.path.getsize(out), mix.group(1).replace(" ", "")


def decode_hashes(gnc, container, outdir):
    os.makedirs(outdir, exist_ok=True)
    run([gnc, "decode-sequence", "-i", container, "-o", os.path.join(outdir, "f_%04d.png")])
    names = sorted(n for n in os.listdir(outdir) if n.endswith(".png"))
    return [hashlib.sha256(open(os.path.join(outdir, n), "rb").read()).hexdigest()
            for n in names]


def pct(new, old):
    return 100.0 * (new - old) / old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="pinned gnc binary (a copy, not target/)")
    ap.add_argument("--sequences", default=",".join(SEQUENCES))
    ap.add_argument("--ki", default="2,9")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--chroma", default="444")
    ap.add_argument("--csv")
    ap.add_argument("--workdir")
    ap.add_argument("--verify-decode", action="store_true",
                    help="decode both arms and hash every frame (slow; BUG-39 did 48/48)")
    args = ap.parse_args()

    gnc = os.path.abspath(args.bin)
    root = repo_root()
    print(f"binary: {gnc}")
    print(f"sha256: {hashlib.sha256(open(gnc, 'rb').read()).hexdigest()}")
    print(f"{args.frames} frames, q=100, chroma {args.chroma}\n")

    work = args.workdir or tempfile.mkdtemp(prefix="lossless2_")
    os.makedirs(work, exist_ok=True)
    rows_out = []

    for seq in args.sequences.split(","):
        pattern = os.path.join(root, "test_material/frames/sequences", seq, "frame_%04d.png")
        if not os.path.exists(pattern % 0):
            print(f"{seq}: missing input, skipped")
            continue
        # All-intra arm: ki=1. The same encode for every ki, so coded once per sequence.
        intra_rows, intra_total, intra_mix = encode(
            gnc, pattern, os.path.join(work, f"{seq}_i.gnv"), args.frames, 1, args.chroma)
        intra_by_idx = {i: b for i, _t, b in intra_rows}
        assert all(t == "I" for _i, t, _b in intra_rows), f"{seq}: ki=1 coded a non-I frame"

        for ki in [int(k) for k in args.ki.split(",")]:
            # The "before" arm is the shipped encoder with the re-code switched off, so both
            # arms come from one binary and differ only in the decision under test.
            ip_rows, ip_total, ip_mix = encode(
                gnc, pattern, os.path.join(work, f"{seq}_p{ki}.gnv"), args.frames, ki,
                args.chroma, env={"GNC_LOSSLESS_INTRA_RECODE": "0"})
            if any(t == "B" for _i, t, _b in ip_rows):
                raise SystemExit(f"{seq} ki={ki}: B-frames in the mix ({ip_mix}); not this item")

            # Control: an I-frame at index i must cost the same in both arms, or the two arms
            # are not coding the same thing and no per-frame minimum can be taken across them.
            for i, t, b in ip_rows:
                if t == "I" and intra_by_idx[i] != b:
                    raise SystemExit(
                        f"{seq} ki={ki} frame {i}: I-frame is {b} B in the I+P arm and "
                        f"{intra_by_idx[i]} B in the all-intra arm -- arms not comparable")

            ip_sum = sum(b for _i, _t, b in ip_rows)
            intra_sum = sum(intra_by_idx[i] for i, _t, _b in ip_rows)
            best_sum = sum(min(b, intra_by_idx[i]) for i, _t, b in ip_rows)
            p_frames = [(i, b) for i, t, b in ip_rows if t == "P"]
            p_wins = [i for i, b in p_frames if b < intra_by_idx[i]]

            # The shipped arm: same command, no env override.
            new_rows, new_total, new_mix = encode(
                gnc, pattern, os.path.join(work, f"{seq}_n{ki}.gnv"), args.frames, ki, args.chroma)
            new_sum = sum(b for _i, _t, b in new_rows)

            print(f"{seq}  ki={ki}")
            print(f"  before (P kept)   {ip_sum:>12,} B   mix {ip_mix}")
            print(f"  all-intra (ki=1)  {intra_sum:>12,} B   ({pct(ip_sum, intra_sum):+.2f}% for I+P)")
            print(f"  per-frame min     {best_sum:>12,} B   "
                  f"({pct(best_sum, intra_sum):+.2f}% vs all-intra, "
                  f"{pct(best_sum, ip_sum):+.2f}% vs I+P)")
            print(f"  SHIPPED           {new_sum:>12,} B   mix {new_mix}   "
                  f"({pct(new_sum, ip_sum):+.2f}% vs before)")
            if new_sum != best_sum:
                print(f"  ** shipped is {new_sum - best_sum:+,} B off the per-frame minimum")
            print(f"  P-frames that beat their own I-frame: {len(p_wins)} of {len(p_frames)}"
                  + (f"  {p_wins}" if p_wins else ""))
            for i, b in p_frames:
                print(f"    frame {i:>2}  P {b:>10,}   I {intra_by_idx[i]:>10,}   "
                      f"{pct(b, intra_by_idx[i]):+7.2f}%  -> {'P' if b < intra_by_idx[i] else 'I'}")
            print()

            rows_out.append(dict(sequence=seq, ki=ki, frames=args.frames,
                                 before_mix=ip_mix, shipped_mix=new_mix,
                                 before_bytes=ip_sum, intra_bytes=intra_sum,
                                 perframe_min_bytes=best_sum, shipped_bytes=new_sum,
                                 p_wins=len(p_wins), p_total=len(p_frames)))

            if args.verify_decode:
                h_new = decode_hashes(gnc, os.path.join(work, f"{seq}_n{ki}.gnv"),
                                      os.path.join(work, f"{seq}_n{ki}_dec"))
                h_i = decode_hashes(gnc, os.path.join(work, f"{seq}_i.gnv"),
                                    os.path.join(work, f"{seq}_i_dec"))
                same = sum(1 for a, b in zip(h_new, h_i) if a == b)
                print(f"  shipped decodes identical to the all-intra arm: {same} of {len(h_i)}\n")
                rows_out[-1]["decode_identical"] = f"{same}/{len(h_i)}"

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
