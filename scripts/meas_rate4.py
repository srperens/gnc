#!/usr/bin/env python3
"""RATE-4: is the candidate choice wrong per *frame*, and would a per-GOP ledger fix it?

RATE-3 shipped the lossless fallback on sequence I-frames at q = 95..=99 and measured its own
cost: the choice is made on the I-frame's own bytes, but a bit-exact I-frame carries detail a
lossy one quantised away, so the P-frames predicted from it are larger. Two of RATE-3's twelve
points regress for that reason (bbb q=99, +0.58% at ki=2 and +0.40% at ki=9, `0044`).

This measures the ledger, not a fix. A GOP is closed -- every P references back only within its
own GOP -- so an I-frame's downstream cost is confined to its GOP and **GOP bytes** is the
correct unit. Both arms come from the same binary and the same command with only
`GNC_LOSSLESS_FALLBACK` differing, so per-GOP totals are exact and not a model.

**Do not pass `--bitrate` to either arm.** `rate_ctrl` is `None` unless a target bitrate is given
(`sequence.rs`), and with it running its state carries across the boundary, which is the one input
that breaks GOP independence. Everything else was checked by the RATE-3 session and does not: no
B-frames at ki=2 or ki=9, `pending_me` reset at every keyframe, and `gpu_ref_planes` overwritten
rather than accumulated at each I-frame.

  arm ON  = today's per-frame ledger      (bit-exact I kept wherever its own bytes are smaller)
  arm OFF = the control                   (lossy I always)
  oracle  = sum over GOPs of min(ON, OFF) -- what a per-GOP ledger would have produced

The oracle cannot be worse than the control at any point, because the control is one of its two
arms. That is the property RATE-4's success criterion asks for, and it is why this needs no
margin constant (which the item forbids: three sequences cannot fit one, and RATE-2 already
found the boundary is content-dependent).

**Built-in canary for the GOP-independence premise.** Where both arms keep the same I-frame, the
GOP is byte-identical between them by construction. Any GOP whose I bytes agree while its P
bytes differ means something crosses the boundary -- rate-control state, or a B-frame reaching
past its own I -- and the premise is wrong. The script reports that count and refuses to print an
oracle if it is non-zero.

Usage:  python3 scripts/meas_rate4.py [-q 95 99] [-k 2 9] [--sequences bbb crowd_run ...]
"""

import argparse
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
SEQ_DIR = REPO / "test_material" / "frames" / "sequences"

# RATE-3's set and frame counts, unchanged -- a different n is a different measurement.
SEQUENCES = {"bbb": 8, "crowd_run": 10, "old_town_cross": 10}

FRAME = re.compile(r"^\s+Frame\s+(\d+) \[([IPB])\]:\s+(\d+) bytes")
TOTAL = re.compile(r"^\s+Total:\s+(\d+) bytes")


def run(seq, frames, q, k, fallback):
    """-> (total_bytes, [(idx, kind, bytes), ...]) for the I+P+B arm only."""
    env = dict(os.environ)
    if not fallback:
        env["GNC_LOSSLESS_FALLBACK"] = "0"
    cmd = [str(GNC), "benchmark-sequence",
           "-i", str(SEQ_DIR / seq / "frame_%04d.png"),
           "-q", str(q), "-n", str(frames), "-k", str(k)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        sys.exit(f"gnc failed on {seq} q={q} ki={k} fallback={fallback}:\n{p.stderr[-3000:]}")
    # The I+P+B arm only. The all-I baseline that follows is RATE-2's intra win on every frame,
    # a different question, and reading it by accident is how a sweep loses its meaning.
    text = p.stdout
    arm = text[text.index("=== I+P+B"):text.index("=== All I-frames")]
    rows = [(int(m.group(1)), m.group(2), int(m.group(3)))
            for m in map(FRAME.match, arm.splitlines()) if m]
    total = next((int(m.group(1)) for m in map(TOTAL.match, arm.splitlines()) if m), None)
    if total is None or not rows:
        sys.exit(f"could not parse the I+P+B arm for {seq} q={q} ki={k}")
    if sum(b for _, _, b in rows) != total:
        sys.exit(f"per-frame bytes do not sum to Total on {seq} q={q} ki={k}: "
                 f"{sum(b for _, _, b in rows)} vs {total}")
    return total, rows


def gops(rows):
    """Split display-ordered frames into GOPs. A GOP starts at each I-frame."""
    out = []
    for idx, kind, nbytes in rows:
        if kind == "I" or not out:
            out.append([])
        out[-1].append((idx, kind, nbytes))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", type=int, nargs="+", default=[95, 99])
    ap.add_argument("-k", type=int, nargs="+", default=[2, 9])
    ap.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    a = ap.parse_args()

    if not GNC.exists():
        sys.exit(f"no binary at {GNC} -- cargo build --release first")

    print(f"binary {GNC}")
    print(subprocess.run(["shasum", "-a", "256", str(GNC)], capture_output=True,
                         text=True).stdout.split()[0], "(hash-recorded before the sweep)")
    fp = fingerprint.read(GNC)
    print(f"codec-fingerprint {fp} — COORD-6: two numbers carrying the same one are comparable")
    print()
    hdr = (f"{'sequence':<16}{'q':>4}{'ki':>4}{'  ON (today)':>14}{'  OFF (control)':>16}"
           f"{'  per-GOP oracle':>17}{'  ON vs OFF':>12}{'  oracle vs ON':>15}{'  flips':>8}")
    print(hdr)
    print("-" * len(hdr))

    unsound = 0
    rows_out = []
    ledgers = []
    decay = []
    for seq in a.sequences:
        n = SEQUENCES[seq]
        for q in a.q:
            for k in a.k:
                on_total, on_rows = run(seq, n, q, k, True)
                off_total, off_rows = run(seq, n, q, k, False)
                # Assert the two arms partitioned the sequence identically before differencing
                # them. Raised by the RATE-3 session: scene-cut detection runs before the
                # keyframe decision, and if it ever reads reconstructed pixels the arms could
                # disagree about where a GOP starts -- at which point they are not comparing the
                # same partition and every number below is meaningless. On RATE-3's sequences no
                # cut fires and the I positions are exactly `display_idx % ki == 0`, so this is
                # cheap insurance rather than an expected failure.
                if [(i, kd) for i, kd, _ in on_rows] != [(i, kd) for i, kd, _ in off_rows]:
                    sys.exit(f"the two arms disagree about frame types on {seq} q={q} ki={k}:\n"
                             f"  ON  {[(i, kd) for i, kd, _ in on_rows]}\n"
                             f"  OFF {[(i, kd) for i, kd, _ in off_rows]}")
                on_g, off_g = gops(on_rows), gops(off_rows)
                if len(on_g) != len(off_g):
                    sys.exit(f"GOP structure differs between arms on {seq} q={q} ki={k}")
                oracle = 0
                flips = 0
                for g_on, g_off in zip(on_g, off_g):
                    s_on = sum(b for _, _, b in g_on)
                    s_off = sum(b for _, _, b in g_off)
                    # GOP-independence canary: same I bytes must mean the same whole GOP.
                    if g_on[0][2] == g_off[0][2] and s_on != s_off:
                        unsound += 1
                    if s_off < s_on:
                        flips += 1
                    oracle += min(s_on, s_off)
                # How far ahead does the ledger actually have to look? Three ledgers on the
                # same GOPs: the I-frame's own bytes (today), I + the first P (a one-frame
                # lookahead, which is exact at ki=2 and a 1-of-8 sample at ki=9), and the whole
                # GOP. If the one-frame ledger never disagrees with the whole-GOP one, the
                # criterion is reachable at one extra P-frame encode per GOP instead of the
                # losing arm's whole GOP -- an eightfold difference at ki=9.
                for g_on, g_off in zip(on_g, off_g):
                    if g_on[0][2] == g_off[0][2]:
                        continue          # both arms kept the same I: no choice to make
                    full = sum(b for _, _, b in g_on) - sum(b for _, _, b in g_off)
                    ip1 = (g_on[0][2] + (g_on[1][2] if len(g_on) > 1 else 0)) \
                        - (g_off[0][2] + (g_off[1][2] if len(g_off) > 1 else 0))
                    ionly = g_on[0][2] - g_off[0][2]
                    ledgers.append((seq, q, k, ionly, ip1, full))
                    # Where in the GOP the penalty is actually paid. Per-frame deltas inside one
                    # GOP, bit-exact arm minus lossy arm: if the extra detail a bit-exact
                    # reference carries propagates, these decay slowly; if it is re-coded once
                    # and gone, P2 onwards are ~0 and the ledger only ever needs one frame.
                    if k > 2 and len(g_on) > 2:
                        decay.append((seq, q, [b_on - b_off
                                               for (_, _, b_on), (_, _, b_off)
                                               in zip(g_on, g_off)]))
                rows_out.append((seq, q, k, on_total, off_total, oracle, flips, len(on_g)))
                print(f"{seq:<16}{q:>4}{k:>4}{on_total:>14}{off_total:>16}{oracle:>17}"
                      f"{(on_total / off_total - 1) * 100:>11.2f}%"
                      f"{(oracle / on_total - 1) * 100:>14.2f}%"
                      f"{flips:>5}/{len(on_g)}")

    print()
    if unsound:
        print(f"REFUSING to summarise: {unsound} GOP(s) have identical I bytes across the arms "
              f"and different totals.\nGOPs are not independent here, so per-GOP is the wrong "
              f"unit and the oracle above is meaningless.")
        return 1

    print(f"GOP-independence canary: 0 of {sum(r[7] for r in rows_out)} GOPs disagree "
          f"(same I bytes => identical GOP), so the closed-GOP premise holds on this data.")
    n = len(rows_out)
    mean_on = sum((r[3] / r[4] - 1) * 100 for r in rows_out) / n
    mean_or = sum((r[5] / r[4] - 1) * 100 for r in rows_out) / n
    worst_on = max((r[3] / r[4] - 1) * 100 for r in rows_out)
    worst_or = max((r[5] / r[4] - 1) * 100 for r in rows_out)
    print(f"against the control, mean of {n} points: today {mean_on:+.2f}%, "
          f"per-GOP oracle {mean_or:+.2f}%")
    print(f"worst single point:                     today {worst_on:+.2f}%, "
          f"per-GOP oracle {worst_or:+.2f}%")
    regress = [r for r in rows_out if r[3] > r[4]]
    print(f"points worse than the control: today {len(regress)} of {n} "
          f"({', '.join(f'{r[0]} q={r[1]} ki={r[2]}' for r in regress) or 'none'}), "
          f"per-GOP oracle {sum(1 for r in rows_out if r[5] > r[4])} of {n} (0 by construction)")
    fingerprint.check_unchanged(GNC, fp)
    print(f"GOPs where the per-GOP choice differs from the per-frame one: "
          f"{sum(r[6] for r in rows_out)} of {sum(r[7] for r in rows_out)}")

    # How far the ledger has to look. Sign convention: negative means keeping the bit-exact
    # I-frame is the smaller of the two, which is the choice today's ledger makes on `ionly`.
    print()
    print("How far ahead the ledger has to look, over every GOP where the arms chose differently")
    print("(bytes, negative = keeping the bit-exact I-frame is smaller under that ledger):")
    print(f"  {'sequence':<16}{'q':>4}{'ki':>4}{'I only (today)':>16}{'I + first P':>14}"
          f"{'whole GOP':>12}{'  1-frame == GOP?':>18}")
    agree = 0
    for seq, q, k, ionly, ip1, full in ledgers:
        same = (ip1 < 0) == (full < 0)
        agree += same
        print(f"  {seq:<16}{q:>4}{k:>4}{ionly:>16}{ip1:>14}{full:>12}"
              f"{'  yes' if same else '  NO':>18}")
    if ledgers:
        print(f"  the one-frame lookahead reaches the whole-GOP decision on "
              f"{agree} of {len(ledgers)} GOPs")

    if decay:
        print()
        print("Where in the GOP the penalty is paid — per-frame bytes, bit-exact arm minus lossy")
        print("arm, inside each ki=9 GOP whose arms chose differently (frame 0 is the I-frame):")
        for seq, q, ds in decay:
            cells = "  ".join(f"{d:+9d}" for d in ds)
            print(f"  {seq:<16}q={q}  {cells}")
        print("  If P2.. are small against P1, the cost is re-coded once and does not propagate,")
        print("  and a one-frame ledger is not an approximation of the GOP — it is nearly all of it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
