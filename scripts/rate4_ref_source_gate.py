#!/usr/bin/env python3
"""RATE-4: the source-built I-frame reference must not move a single byte.

`local_decode_iframe_gpu` builds a bit-exact I-frame's reference by colour-converting the source
instead of dequantising and inverting what was coded, and `encode_as_reference` therefore stops
paying RATE-3's third encode on those frames. The two paths are supposed to produce the *same
picture* — measured directly by
`a_bit_exact_frames_reference_is_its_colour_converted_source` — so every P-frame that predicts
from it must code to the identical bytes.

**Byte identity is the whole gate.** It is a stronger statement than a rate comparison and it is
the one this change is allowed to make: the win is a count (three intra encodes per fallback
I-frame become two), never a number, because encode time cannot be measured on a machine with
seven other sessions on it (COORDINATION rule 1).

Arms differ only in `GNC_REF_FROM_SOURCE`: unset (the new default) against `0` (forced
reconstruct path). Cases:

* q = 95, 99 with the lossless fallback — the RATE-3 case, where the route fires only on the
  frames whose bit-exact candidate wins;
* q = 100 — every I-frame is bit-exact, so the route fires on all of them;
* 4:2:0 at q = 99 and q = 100 — the route must **refuse**, because the reference holds
  nearest-neighbour-upsampled chroma there. This arm is the canary for the refusal: identical
  bytes here prove nothing about the route and everything about the gate not having leaked.

Usage:  python3 scripts/rate4_ref_source_gate.py
"""

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
SEQUENCES = [("bbb", 8), ("crowd_run", 10), ("old_town_cross", 10)]

FRAME = re.compile(r"^\s+Frame\s+(\d+) \[([IPB])\]:\s+(\d+) bytes")
ROUTE = re.compile(r"RATE-4: reference built from the colour-converted source")


def run(seq, frames, q, k, chroma, from_source):
    env = dict(os.environ)
    if not from_source:
        env["GNC_REF_FROM_SOURCE"] = "0"
    cmd = [str(GNC), "benchmark-sequence",
           "-i", str(SEQ_DIR / seq / "frame_%04d.png"),
           "-q", str(q), "-n", str(frames), "-k", str(k),
           "--chroma-format", chroma, "--throughput", "--diagnostics"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.exit(f"gnc failed on {seq} q={q} ki={k} {chroma} from_source={from_source}:\n"
                 f"{p.stderr[-3000:]}")
    rows = [(int(m.group(1)), m.group(2), int(m.group(3)))
            for m in map(FRAME.match, p.stdout.splitlines()) if m]
    if not rows:
        sys.exit(f"no per-frame lines parsed for {seq} q={q} ki={k} {chroma}")
    took = len(ROUTE.findall(p.stdout)) + len(ROUTE.findall(p.stderr))
    return rows, took


def main():
    if not GNC.exists():
        sys.exit(f"no binary at {GNC} — cargo build --release first")
    print(subprocess.run(["shasum", "-a", "256", str(GNC)], capture_output=True,
                         text=True).stdout.split()[0], "(binary, hash-recorded)")
    print()
    cases = ([(q, 444, k) for q in (95, 99, 100) for k in (2, 9)]
             + [(99, 420, 2), (100, 420, 2)])
    hdr = f"{'sequence':<16}{'q':>5}{'chroma':>8}{'ki':>4}{'frames':>8}{'route fired':>13}{'bytes':>10}"
    print(hdr)
    print("-" * len(hdr))
    bad = 0
    fired_total = 0
    for seq, n in SEQUENCES:
        for q, chroma, k in cases:
            a, took_a = run(seq, n, q, k, str(chroma), True)
            b, took_b = run(seq, n, q, k, str(chroma), False)
            same = a == b
            if not same:
                bad += 1
            fired_total += took_a
            if took_b:
                bad += 1
                note = f"LEAK: forced-off arm still took the route {took_b}x"
            elif chroma == 420 and took_a:
                bad += 1
                note = f"LEAK: 4:2:0 took the route {took_a}x"
            elif chroma == 444 and q >= 100 and took_a == 0:
                bad += 1
                note = "VACUOUS: q=100 4:4:4 never took the route"
            else:
                note = "identical" if same else "BYTES MOVED"
            print(f"{seq:<16}{q:>5}{chroma:>8}{k:>4}{len(a):>8}{took_a:>13}"
                  f"{sum(x[2] for x in a):>10}  {note}")
    print()
    print(f"route fired on {fired_total} I-frame(s) across the sweep")
    if bad:
        print(f"FAIL: {bad} problem(s) above")
        return 1
    print("PASS: every arm byte-identical, the route fired where it should and nowhere else")
    return 0


if __name__ == "__main__":
    sys.exit(main())
