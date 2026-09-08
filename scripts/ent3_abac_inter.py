#!/usr/bin/env python3
"""ENT-3 — abac against Rice on inter, with the intra half held separate.

Why this exists. Decision `0025` measured abac against Rice over whole `.gnv` containers at
q=50/75/90 and got −12.0% to −22.9% at bit-identical pixels. Two things are missing from that
table and both are answerable with the encoder's own stdout:

  * **It is a container ratio, so it is not an inter figure.** At ki=9 over 18 frames the file
    holds 2 I-frames and 16 P-frames, and abac's intra saving (−16.6% to −18.8%, ABAC-SHIP) is
    already known. `encode-sequence` prints `Frame N [I]:` / `Frame N [P]:` byte counts, so the
    two halves can be summed separately and the P-only column is the number ENT-3 actually asks
    for. Reported here as three columns: container, I-frames only, P-frames only.
  * **It stops at q=90.** GNC is a contribution codec (GOALS §1) and q=95-99 is its home range.

**Domain declaration.** Both arms quantise identically and differ only in the entropy stage, so
what is being compared is the coded size of the *same* quantised wavelet coefficients — for a P
frame, the coefficients of the motion-compensated residual; for an I frame, of the frame itself.
Since ARCH-3 the entropy choice cannot reach the pixels (`0025`), which is why no BD-rate is
needed: the premise is checked rather than assumed, by decoding both arms and hashing every frame.
A per-frame hash mismatch voids that point instead of averaging it in.

**Why the frame mix is printed rather than assumed.** At ki=9 the codec can code either
`2I+16P+0B` or `2I+2P+14B`: `quality_preset()` sets `b_pyramid: false` unless `GNC_B_PYRAMID=1`
while `CodecConfig::default()` sets it `true`, and `0025` does not record which it measured.
`encode-sequence` names the mix on its last line, so it is read out of the run. Note that
`benchmark-sequence`'s `-q` is `Option<u32>` with no default (BUG-37) — this harness uses
`encode-sequence`, whose `-q` defaults to 75 and therefore always goes through `quality_preset`.

**Why no VMAF.** Nothing here is a perceptual question — at equal q the two arms decode to the
same bytes, so quality is a control and it is checked exactly (hash), not statistically (PSNR).

Usage:
    scripts/ent3_abac_inter.py --bin <pinned gnc> [--qualities 90,95,97,99] [--csv out.csv]
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

SEQUENCES = ["bbb_extended", "crowd_run", "old_town_cross"]
FRAME_RE = re.compile(r"^\s*Frame\s+(\d+)\s+\[([IPB])\]:\s+(\d+)\s+bytes")
MIX_RE = re.compile(r"Encoded\s+\d+\s+frames\s+\(([^)]*)\)")


def repo_root():
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(f"FAILED: {' '.join(cmd)}\n{p.stdout}\n{p.stderr}\n")
        raise SystemExit(1)
    return p.stdout, p.stderr


def encode(gnc, pattern, out, quality, frames, ki, chroma, coder):
    """Encode one arm. Returns (container_bytes, {frame_type: bytes}, mix_string)."""
    stdout, _ = run([
        gnc, "encode-sequence",
        "-i", pattern, "-o", out,
        "-q", str(quality), "-n", str(frames),
        "--keyframe-interval", str(ki),
        "--chroma-format", chroma,
        f"--{coder}",
    ])
    per_type = {}
    n_frames = 0
    for line in stdout.splitlines():
        m = FRAME_RE.match(line)
        if m:
            per_type[m.group(2)] = per_type.get(m.group(2), 0) + int(m.group(3))
            n_frames += 1
    mix = MIX_RE.search(stdout)
    if n_frames != frames or mix is None:
        raise SystemExit(
            f"{coder} q={quality}: parsed {n_frames} of {frames} frame lines, mix={mix}"
        )
    return os.path.getsize(out), per_type, mix.group(1).replace(" ", "")


def decode_hashes(gnc, container, outdir, frames):
    """Decode a container and return one hash per frame, in order."""
    os.makedirs(outdir, exist_ok=True)
    run([gnc, "decode-sequence", "-i", container, "-o", os.path.join(outdir, "f_%04d.png")])
    names = sorted(n for n in os.listdir(outdir) if n.endswith(".png"))
    if len(names) != frames:
        raise SystemExit(f"{container}: decoded {len(names)} frames, expected {frames}")
    out = []
    for n in names:
        with open(os.path.join(outdir, n), "rb") as f:
            out.append(hashlib.sha256(f.read()).hexdigest())
    return out


def pct(new, old):
    return 100.0 * (new - old) / old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="pinned gnc binary (a copy, not target/)")
    ap.add_argument("--sequences", default=",".join(SEQUENCES))
    ap.add_argument("--qualities", default="90,95,97,99")
    ap.add_argument("--frames", type=int, default=18)
    ap.add_argument("--ki", type=int, default=9)
    ap.add_argument("--chroma", default="444")
    ap.add_argument("--csv")
    ap.add_argument("--workdir", help="keep intermediates here instead of a temp dir")
    args = ap.parse_args()

    gnc = os.path.abspath(args.bin)
    root = repo_root()
    seqs = args.sequences.split(",")
    qualities = [int(q) for q in args.qualities.split(",")]

    print(f"binary: {gnc}")
    print(f"sha256: {hashlib.sha256(open(gnc, 'rb').read()).hexdigest()}")
    print(f"{args.frames} frames, ki={args.ki}, chroma {args.chroma}\n")

    work = args.workdir or tempfile.mkdtemp(prefix="ent3_")
    os.makedirs(work, exist_ok=True)
    rows = []
    try:
        for seq in seqs:
            pattern = os.path.join(
                root, "test_material/frames/sequences", seq, "frame_%04d.png"
            )
            if not os.path.exists(pattern % 0):
                raise SystemExit(f"missing input: {pattern % 0}")
            for q in qualities:
                arms = {}
                for coder in ("rice", "abac"):
                    container = os.path.join(work, f"{seq}_q{q}_{coder}.gnv")
                    total, per_type, mix = encode(
                        gnc, pattern, container, q, args.frames,
                        args.ki, args.chroma, coder,
                    )
                    hashes = decode_hashes(
                        gnc, container, os.path.join(work, f"{seq}_q{q}_{coder}_png"),
                        args.frames,
                    )
                    arms[coder] = (total, per_type, mix, hashes)
                    os.remove(container)
                    shutil.rmtree(os.path.join(work, f"{seq}_q{q}_{coder}_png"))

                r_total, r_types, r_mix, r_hash = arms["rice"]
                a_total, a_types, a_mix, a_hash = arms["abac"]

                if r_mix != a_mix:
                    raise SystemExit(f"{seq} q={q}: mix differs, {r_mix} vs {a_mix}")
                bad = [i for i, (x, y) in enumerate(zip(r_hash, a_hash)) if x != y]
                if bad:
                    print(f"  {seq} q={q}: PIXELS DIFFER on frames {bad} — point VOID")
                    rows.append({
                        "sequence": seq, "q": q, "mix": r_mix, "identical": 0,
                        "rice_total": r_total, "abac_total": a_total,
                    })
                    continue

                row = {
                    "sequence": seq, "q": q, "mix": r_mix, "identical": 1,
                    "rice_total": r_total, "abac_total": a_total,
                    "delta_total_pct": pct(a_total, r_total),
                }
                for t in ("I", "P", "B"):
                    if t in r_types:
                        row[f"rice_{t}"] = r_types[t]
                        row[f"abac_{t}"] = a_types[t]
                        row[f"delta_{t}_pct"] = pct(a_types[t], r_types[t])
                rows.append(row)
                p = f"{row.get('delta_P_pct', float('nan')):+.1f}%"
                print(
                    f"  {seq:16s} q={q:3d} {r_mix:10s} "
                    f"container {row['delta_total_pct']:+.1f}%  "
                    f"I {row.get('delta_I_pct', float('nan')):+.1f}%  P {p}"
                )
    finally:
        if not args.workdir:
            shutil.rmtree(work, ignore_errors=True)

    print("\n=== abac against Rice, negative means abac is smaller ===")
    for label, key in (
        ("container (2I+16P)", "delta_total_pct"),
        ("I-frames only", "delta_I_pct"),
        ("P-frames only", "delta_P_pct"),
    ):
        print(f"\n{label}")
        header = "| sequence | " + " | ".join(f"q={q}" for q in qualities) + " |"
        print(header)
        print("|---" * (len(qualities) + 1) + "|")
        for seq in seqs:
            cells = []
            for q in qualities:
                hit = [r for r in rows if r["sequence"] == seq and r["q"] == q]
                if hit and hit[0]["identical"] and key in hit[0]:
                    cells.append(f"{hit[0][key]:+.1f}%")
                else:
                    cells.append("VOID" if hit and not hit[0]["identical"] else "—")
            print(f"| {seq} | " + " | ".join(cells) + " |")
        vals = [r[key] for r in rows if r["identical"] and key in r]
        if vals:
            print(f"mean {sum(vals) / len(vals):+.1f}%  "
                  f"range {min(vals):+.1f}% .. {max(vals):+.1f}%")

    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
