#!/usr/bin/env python3
"""LOSSLESS-3 — is the top of the lossy *sequence* ladder dominated by bit-exact lossless?

RATE-2 (`0036`) found that above q≈95 a **still** costs more as a wavelet encode than as a
bit-exact MED encode, and fixed it by coding both and keeping the smaller. RATE-3 lifted that into
sequences for I-frames only. This measures the half that was left: a **P-frame** at q=95..99
against a bit-exact I-frame of the same picture.

**Why no BD-rate and no metric arbitration.** The bit-exact candidate is better on *both* axes
when it is smaller — fewer bytes and exact pixels — so there is nothing to trade. What must still
be checked is that quality does not fall anywhere, because replacing a frame changes what the
frames after it predict from. This script therefore reports per-frame PSNR against the source for
the shipped arm, and flags any frame below the same frame in the before arm.

Both arms come from one binary: `GNC_LOSSLESS_SEQUENCE_FALLBACK=0` is the before arm.

Usage:
    scripts/meas_lossless3_ladder.py --bin <pinned gnc> [--qualities 95,97,99] [--ki 2,9]
"""

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
import tempfile

SEQUENCES = ["crowd_run", "old_town_cross", "blue_sky", "bbb"]
FRAME_RE = re.compile(r"^\s*Frame\s+(\d+)\s+\[([IPB])\]:\s+(\d+)\s+bytes")
MIX_RE = re.compile(r"Encoded\s+\d+\s+frames\s+\(([^)]*)\)")
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


def encode(gnc, pattern, out, q, frames, ki, chroma, env=None):
    stdout, _ = run([gnc, "encode-sequence", "-i", pattern, "-o", out,
                     "-q", str(q), "-n", str(frames),
                     "--keyframe-interval", str(ki),
                     "--chroma-format", chroma], env)
    rows = [(int(m.group(1)), m.group(2), int(m.group(3)))
            for m in (FRAME_RE.match(l) for l in stdout.splitlines()) if m]
    mix = MIX_RE.search(stdout)
    if len(rows) != frames or mix is None:
        raise SystemExit(f"parsed {len(rows)} of {frames} frame lines, mix={mix}")
    return rows, os.path.getsize(out), mix.group(1).replace(" ", "")


def decode(gnc, container, outdir):
    os.makedirs(outdir, exist_ok=True)
    run([gnc, "decode-sequence", "-i", container, "-o", os.path.join(outdir, "f_%04d.png")])
    return sorted(os.path.join(outdir, n) for n in os.listdir(outdir) if n.endswith(".png"))


def raw_md5(png):
    """md5 of the decoded RGB samples, so PNG container differences cannot mask a mismatch."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", png, "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True)
    if p.returncode != 0:
        raise SystemExit(f"ffmpeg failed on {png}: {p.stderr.decode()[:200]}")
    return hashlib.md5(p.stdout).hexdigest()


def psnr(a, b):
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", a, "-i", b,
                        "-lavfi", "psnr", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = PSNR_RE.search(p.stderr)
    if not m:
        raise SystemExit(f"no psnr in ffmpeg output for {a} vs {b}:\n{p.stderr[-400:]}")
    return float("inf") if m.group(1) == "inf" else float(m.group(1))


def pct(new, old):
    return 100.0 * (new - old) / old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--sequences", default=",".join(SEQUENCES))
    ap.add_argument("--qualities", default="95,97,99")
    ap.add_argument("--ki", default="2,9")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--chroma", default="444")
    ap.add_argument("--csv")
    ap.add_argument("--workdir")
    args = ap.parse_args()

    gnc = os.path.abspath(args.bin)
    root = repo_root()
    print(f"binary: {gnc}")
    print(f"sha256: {hashlib.sha256(open(gnc, 'rb').read()).hexdigest()}")
    print(f"{args.frames} frames, chroma {args.chroma}\n")
    work = args.workdir or tempfile.mkdtemp(prefix="lossless3_")
    os.makedirs(work, exist_ok=True)
    out_rows = []

    for seq in args.sequences.split(","):
        pattern = os.path.join(root, "test_material/frames/sequences", seq, "frame_%04d.png")
        if not os.path.exists(pattern % 0):
            print(f"{seq}: missing input, skipped")
            continue
        sources = [pattern % i for i in range(args.frames)]
        for q in [int(x) for x in args.qualities.split(",")]:
            for ki in [int(x) for x in args.ki.split(",")]:
                tag = f"{seq}_q{q}_k{ki}"
                b_rows, b_bytes, b_mix = encode(
                    gnc, pattern, os.path.join(work, f"{tag}_before.gnv"), q, args.frames, ki,
                    args.chroma, env={"GNC_LOSSLESS_SEQUENCE_FALLBACK": "0"})
                n_rows, n_bytes, n_mix = encode(
                    gnc, pattern, os.path.join(work, f"{tag}_new.gnv"), q, args.frames, ki,
                    args.chroma)

                b_png = decode(gnc, os.path.join(work, f"{tag}_before.gnv"),
                               os.path.join(work, f"{tag}_before_d"))
                n_png = decode(gnc, os.path.join(work, f"{tag}_new.gnv"),
                               os.path.join(work, f"{tag}_new_d"))
                b_psnr = [psnr(s, d) for s, d in zip(sources, b_png)]
                n_psnr = [psnr(s, d) for s, d in zip(sources, n_png)]
                exact = sum(1 for s, d in zip(sources, n_png) if raw_md5(s) == raw_md5(d))
                worse = [(i, b, n) for i, (b, n) in enumerate(zip(b_psnr, n_psnr)) if n < b - 1e-9]

                def summary(vals):
                    """Worst and mean over the *lossy* frames, plus how many are bit-exact.
                    Averaging in an `inf` would hide every real number in the row."""
                    fin = [v for v in vals if v != float("inf")]
                    n_inf = len(vals) - len(fin)
                    if not fin:
                        return f"all {n_inf} frames bit-exact"
                    return (f"worst {min(fin):.2f} dB  mean {sum(fin)/len(fin):.2f} dB"
                            + (f"  (+{n_inf} bit-exact)" if n_inf else ""))

                print(f"{seq}  q={q}  ki={ki}")
                print(f"  before  {b_bytes:>12,} B  {b_mix:>8}  {summary(b_psnr)}")
                print(f"  shipped {n_bytes:>12,} B  {n_mix:>8}  {summary(n_psnr)}  "
                      f"({pct(n_bytes, b_bytes):+.2f}%)  md5-exact {exact}/{args.frames}")
                if worse:
                    print(f"  ** PSNR fell on {len(worse)} frame(s): "
                          + ", ".join(f"{i}: {b:.2f}->{n:.2f}" for i, b, n in worse))
                else:
                    print("  no frame is worse than before")
                print()
                out_rows.append(dict(
                    sequence=seq, q=q, ki=ki, frames=args.frames,
                    before_bytes=b_bytes, shipped_bytes=n_bytes,
                    delta_pct=round(pct(n_bytes, b_bytes), 2),
                    before_mix=b_mix, shipped_mix=n_mix,
                    bit_exact_frames=f"{exact}/{args.frames}",
                    frames_worse=len(worse)))

    if args.csv and out_rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
