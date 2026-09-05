#!/usr/bin/env python3
"""Gate for BACKLOG #25 (multi-reference P-frames).

Question: would a P-frame that could choose between the previous decoded frame and the one
before it actually pick the older reference often enough, and gain enough, to be worth building?

This runs entirely offline on the source Y4M — no encoder changes — so the gate can be decided
before any implementation, as the backlog item requires.

Method: for every 16x16 luma block of frame n, full-search motion estimation (integer pel,
+/-search range) against frame n-1 and against frame n-2 independently, using SAD. Report:

  * the fraction of blocks where n-2 wins by a meaningful margin, and
  * the total SAD reduction from taking the per-block best of the two.

The second figure matters more than the first: blocks can prefer the older reference by a hair
without that being worth a reference-index bit. The margin threshold makes "meaningful" explicit.

Note this is an OPTIMISTIC bound in one respect and a PESSIMISTIC one in another. Optimistic:
it uses source frames rather than decoded references, so it ignores the extra quantization noise
a real second reference carries (which is worse for n-2 than n-1, since n-2 is older). Pessimistic:
it charges nothing for the reference-index bit, but also gives the encoder no rate-distortion
search, only SAD. Treat it as an indicator of headroom, not as a bpp prediction.

Run:  python3 scripts/meas_multiref_gate.py <seq.y4m> [--frames N] [--range R] [--margin M]
"""

import argparse
import sys

import numpy as np


def read_y4m_luma(path, max_frames):
    """Yield successive luma planes of a Y4M as float arrays. 8-bit 420/444 only."""
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"\n"):
            c = f.read(1)
            if not c:
                sys.exit(f"{path}: truncated header")
            header += c
        tags = header.decode("ascii", "replace").split()
        if tags[0] != "YUV4MPEG2":
            sys.exit(f"{path}: not a Y4M file")
        w = h = None
        cs = "420"
        for t in tags[1:]:
            if t.startswith("W"):
                w = int(t[1:])
            elif t.startswith("H"):
                h = int(t[1:])
            elif t.startswith("C"):
                cs = t[1:]
        if w is None or h is None:
            sys.exit(f"{path}: missing dimensions")
        if cs.startswith("420"):
            chroma_bytes = 2 * (w // 2) * (h // 2)
        elif cs.startswith("444"):
            chroma_bytes = 2 * w * h
        elif cs.startswith("422"):
            chroma_bytes = 2 * (w // 2) * h
        else:
            sys.exit(f"{path}: unsupported colourspace {cs}")

        frames = []
        while len(frames) < max_frames:
            line = b""
            while not line.endswith(b"\n"):
                c = f.read(1)
                if not c:
                    return w, h, frames
                line += c
            if not line.startswith(b"FRAME"):
                return w, h, frames
            y = f.read(w * h)
            if len(y) < w * h:
                return w, h, frames
            f.read(chroma_bytes)
            frames.append(np.frombuffer(y, dtype=np.uint8).reshape(h, w).astype(np.int32))
        return w, h, frames


def block_sad_field(cur, ref, bs, rng):
    """Best integer-pel SAD per block over a +/-rng full search.

    Vectorised over blocks: for each candidate displacement the whole frame is shifted once and
    all block SADs are computed together, which is what makes a full search tractable in numpy.
    """
    h, w = cur.shape
    by, bx = h // bs, w // bs
    ch, cw = by * bs, bx * bs
    cur_c = cur[:ch, :cw]

    best = np.full((by, bx), np.inf)
    best_dx = np.zeros((by, bx), dtype=np.int32)
    best_dy = np.zeros((by, bx), dtype=np.int32)

    ref_pad = np.pad(ref, rng, mode="edge")
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            y0 = rng + dy
            x0 = rng + dx
            shifted = ref_pad[y0 : y0 + h, x0 : x0 + w][:ch, :cw]
            diff = np.abs(cur_c - shifted)
            sad = (
                diff.reshape(by, bs, bx, bs)
                .sum(axis=(1, 3))
                .astype(np.float64)
            )
            better = sad < best
            best = np.where(better, sad, best)
            best_dx = np.where(better, dx, best_dx)
            best_dy = np.where(better, dy, best_dy)
    return best, best_dx, best_dy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("y4m")
    ap.add_argument("--frames", type=int, default=8, help="frames to analyse (needs >= 3)")
    ap.add_argument("--range", type=int, default=16, help="+/- integer-pel search range")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--margin", type=float, default=0.05,
                    help="n-2 must beat n-1 by this fraction of SAD to count as a real win")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    w, h, frames = read_y4m_luma(args.y4m, args.frames)
    if len(frames) < 3:
        sys.exit("need at least 3 frames")

    name = args.label or args.y4m.split("/")[-1]
    print(f"\n=== #25 multi-reference gate: {name} ===")
    print(f"{len(frames)} frames, {w}x{h}, {args.block}x{args.block} blocks, "
          f"+/-{args.range} full search, margin {args.margin:.0%}")

    tot_blocks = 0
    tot_win = 0
    sad1_sum = 0.0
    sadbest_sum = 0.0
    per_frame = []

    for n in range(2, len(frames)):
        cur = frames[n]
        sad1, _, _ = block_sad_field(cur, frames[n - 1], args.block, args.range)
        sad2, _, _ = block_sad_field(cur, frames[n - 2], args.block, args.range)

        wins = sad2 < sad1 * (1.0 - args.margin)
        best = np.minimum(sad1, sad2)

        tot_blocks += sad1.size
        tot_win += int(wins.sum())
        sad1_sum += float(sad1.sum())
        sadbest_sum += float(best.sum())
        per_frame.append((n, wins.mean() * 100, (1 - best.sum() / sad1.sum()) * 100))

    print(f"\n   {'frame':>5}  {'blocks preferring n-2':>22}  {'SAD reduction':>14}")
    for n, wpct, red in per_frame:
        print(f"   {n:5d}  {wpct:21.1f}%  {red:13.2f}%")

    win_pct = tot_win / tot_blocks * 100
    sad_red = (1 - sadbest_sum / sad1_sum) * 100
    print(f"\n   overall blocks preferring n-2: {win_pct:.1f}%")
    print(f"   overall SAD reduction from best-of-2: {sad_red:.2f}%")
    verdict = "PASS" if win_pct > 15.0 else "FAIL"
    print(f"   GATE (>15% non-adjacent references): {verdict}")
    print()


if __name__ == "__main__":
    main()
