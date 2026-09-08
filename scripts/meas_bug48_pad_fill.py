#!/usr/bin/env python3
"""BUG-48: PAD-1's decay padding fill reverses sign at the top of the quality range.

`0039` measured the decay fill at **−4.63% RGB on stills over q=80..94** and `quality_preset`
applies it at every q. At q=100 the transform is MED and the padding is coded *exactly*, so a fill
that fades to flat has to be coded rather than predicted — plain edge replication is cheaper. The
lever was never measured at the top of its own range.

LOSSLESS-3 saw it on two stills (−0.78% and −0.66%). BUG-48 asks for the four PAD-1 used, at q=100
**and** at q=95..99 where RATE-2's bit-exact candidate is coded and can win, plus the sequence
path, which already clears the flag for referenced I-frames and should therefore not move.

Arms differ only in `GNC_PAD_FILL`. The comparison is **shipped against forced `decay`**, i.e.
against the behaviour before the fix, so the table stays informative once the fix has landed —
comparing against `replicate` reads 0.00% everywhere the moment the preset agrees with it, which
says the fix is active and nothing about what it was worth. Bytes only: the bit-exact arm's pixels are
the source by definition, and at q=95..99 the arms are compared at *identical* settings, so a
quality column would read the same twice.

Usage:  python3 scripts/meas_bug48_pad_fill.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fingerprint  # noqa: E402  (sibling script, not a package)

REPO = Path(
    subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                   check=True).stdout.strip()
)
GNC = REPO / "target" / "release" / "gnc"
FRAMES = REPO / "test_material" / "frames"
# The four PAD-1 used. Same set, or it is not a comparison with `0039`.
STILLS = ["bbb_1080p", "blue_sky_1080p", "kristensara_720p", "touchdown_1080p"]
SEQ_DIR = FRAMES / "sequences"

SIZE = re.compile(r"Size: (\d+) bytes")
TOTAL = re.compile(r"^\s+Total:\s+(\d+) bytes")


def still(name, q, fill):
    env = dict(os.environ)
    if fill:
        env["GNC_PAD_FILL"] = fill
    p = subprocess.run(
        [str(GNC), "benchmark", "-i", str(FRAMES / f"{name}.png"), "-q", str(q)],
        capture_output=True, text=True, env=env,
    )
    m = SIZE.search(p.stdout)
    if p.returncode != 0 or not m:
        sys.exit(f"gnc failed on {name} q={q} fill={fill}:\n{(p.stderr or p.stdout)[-2000:]}")
    return int(m.group(1))


def sequence(name, frames, q, ki, fill):
    env = dict(os.environ)
    if fill:
        env["GNC_PAD_FILL"] = fill
    p = subprocess.run(
        [str(GNC), "benchmark-sequence", "-i", str(SEQ_DIR / name / "frame_%04d.png"),
         "-q", str(q), "-n", str(frames), "-k", str(ki), "--throughput"],
        capture_output=True, text=True, env=env,
    )
    if p.returncode != 0:
        sys.exit(f"gnc failed on {name} q={q} ki={ki} fill={fill}:\n{p.stderr[-2000:]}")
    total = next((int(m.group(1)) for m in map(TOTAL.match, p.stdout.splitlines()) if m), None)
    if total is None:
        sys.exit(f"could not parse a total for {name} q={q} ki={ki}")
    return total


def main():
    if not GNC.exists():
        sys.exit(f"no binary at {GNC} — cargo build --release first")
    fp = fingerprint.read(GNC)
    print(f"codec-fingerprint {fp} — COORD-6: two numbers carrying the same one are comparable")
    print()

    hdr = f"{'still':<20}{'q':>5}{'shipped':>18}{'forced decay':>14}{'delta':>9}"
    print(hdr)
    print("-" * len(hdr))
    per_q = {}
    for q in (95, 97, 99, 100):
        deltas = []
        for name in STILLS:
            a = still(name, q, None)
            b = still(name, q, "decay")
            d = (b / a - 1) * 100
            deltas.append(d)
            print(f"{name:<20}{q:>5}{a:>18}{b:>12}{d:>8.2f}%")
        per_q[q] = sum(deltas) / len(deltas)
        print(f"{'  mean':<20}{q:>5}{'':>18}{'':>12}{per_q[q]:>8.2f}%")
        print()

    # **This arm is `replicate`, not `decay`, and the reason is a trap worth naming.**
    # `GNC_PAD_FILL=decay` is not "before the fix" for a sequence: it also overrides the sequence
    # encoder's unconditional clear for referenced I-frames, which PAD-1 chose on worst-frame
    # PSNR (up to 4.03 dB on bbb_extended). Read as a before-arm it says this fix costs bbb 7.2%;
    # measured against the actual pre-fix binary, all eight sequence points are **byte-identical**.
    # So this arm asks the only question the preset change can affect here — is the shipped
    # sequence path already replicate — and 0.00% is the expected, correct answer.
    print("Sequences. The preset change cannot reach these: the encoder clears the fill")
    print("unconditionally for keyframes, so they are already replicate, and 0.00% is the pass.")
    print("Do NOT use GNC_PAD_FILL=decay as a before-arm here — it overrides that clear too and")
    print("reads as a 7.2% regression on bbb that the actual pre-fix binary does not show.")
    hdr = f"{'sequence':<18}{'q':>5}{'ki':>4}{'shipped':>18}{'replicate'   :>14}{'delta':>9}"
    print(hdr)
    print("-" * len(hdr))
    seq_moved = 0
    for name, n in (("bbb", 8), ("crowd_run", 10)):
        for q, ki in ((95, 2), (99, 2), (100, 2), (100, 9)):
            a = sequence(name, n, q, ki, None)
            b = sequence(name, n, q, ki, "replicate")
            d = (b / a - 1) * 100
            if a != b:
                seq_moved += 1
            print(f"{name:<18}{q:>5}{ki:>4}{a:>18}{b:>12}{d:>8.2f}%")

    fingerprint.check_unchanged(GNC, fp)
    print()
    for q in (95, 97, 99, 100):
        print(f"q={q:<4} mean over four stills: {per_q[q]:+.2f}%")
    print(f"sequence points that moved: {seq_moved} of 8 "
          f"({'none — the shipped sequence path is already replicate, which is the pass' if seq_moved == 0 else 'INVESTIGATE: the keyframe clear is not reaching every frame'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
