#!/usr/bin/env python3
"""Does filtering the reference frame improve prediction?

GNC has no in-loop filter. Every mature codec does, and the reason is not cosmetic: the
reference a P-frame predicts from is a *decoded* frame, so it carries quantisation noise that
is uncorrelated with the source. Predicting from that noise costs bits in every frame that
follows. A deblocking or de-ringing filter removes some of it before the reference is used.

This measures the ceiling offline, before anything is implemented. It block-matches the source
frame against the decoded reference (what GNC does) and against several filtered versions of the
same reference, and reports SATD — the same criterion the encoder's own search uses, because SAD
rewards a blurred predictor and would make any low-pass filter look good for free.

Reads dumps from `GNC_DUMP_RESIDUAL=<dir>` (verify the dump is bitstream-neutral first; see
BUG-7). Uses resid_Pcur_* (source) and resid_Pref_* (decoded reference luma).

**Known limits of this proxy, read before quoting it.** The search range defaults small, and on
panning content GNC's own vectors run well past it, so both arms are matched badly and real
differences get compressed. And do NOT extend this to "predict from the previous frame's clean
source to find the ceiling" — that comparison is confounded, because a decoded reference is not
the source plus noise, it is the source *low-pass filtered by the quantiser*. Those two effects
have opposite signs, and the confound is large enough to flip the answer: measured that way,
bbb17 said −31.6% and old_town said +12.3%. Only the filter variants below are meaningful.

Run:  python3 scripts/meas_ref_filter.py <dump_dir> [--block 16] [--range 24]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def load_plane(path):
    meta = json.load(open(os.path.splitext(path)[0] + ".json"))
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(meta["height"], meta["width"]).astype(np.float64)


def best_match_satd(cur, ref, block, rng):
    """Full integer search per block, scored by SATD. Returns total SATD."""
    h, w = cur.shape
    by, bx = h // block, w // block
    best = np.full((by, bx), np.inf)
    step = block // 8
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            sh = np.roll(np.roll(ref, dy, axis=0), dx, axis=1)
            d = (cur - sh)[: by * block, : bx * block]
            t = d.reshape(by * step, 8, bx * step, 8).transpose(0, 2, 1, 3)
            H = _H8
            satd8 = np.abs(H @ t @ H.T).sum(axis=(2, 3))
            cell = satd8.reshape(by, step, bx, step).sum(axis=(1, 3))
            np.minimum(best, cell, out=best)
    return best.sum()


_H8 = np.array(
    [[1, 1, 1, 1, 1, 1, 1, 1],
     [1, -1, 1, -1, 1, -1, 1, -1],
     [1, 1, -1, -1, 1, 1, -1, -1],
     [1, -1, -1, 1, 1, -1, -1, 1],
     [1, 1, 1, 1, -1, -1, -1, -1],
     [1, -1, 1, -1, -1, 1, -1, 1],
     [1, 1, -1, -1, -1, -1, 1, 1],
     [1, -1, -1, 1, -1, 1, 1, -1]],
    dtype=np.float64,
)


def blur3(a, w_center):
    """Symmetric 3x3 low-pass with the given centre weight, edges replicated."""
    k = np.array([1.0, w_center, 1.0])
    k = k / k.sum()
    p = np.pad(a, 1, mode="edge")
    tmp = k[0] * p[:, :-2] + k[1] * p[:, 1:-1] + k[2] * p[:, 2:]
    out = k[0] * tmp[:-2, :] + k[1] * tmp[1:-1, :] + k[2] * tmp[2:, :]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--range", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=3)
    args = ap.parse_args()

    curs = sorted(glob.glob(os.path.join(args.dump_dir, "resid_Pcur_*.f32")))
    refs = sorted(glob.glob(os.path.join(args.dump_dir, "resid_Pref_*.f32")))
    if not curs or len(curs) != len(refs):
        print(f"need matching resid_Pcur_* and resid_Pref_* in {args.dump_dir}", file=sys.stderr)
        return 1
    curs, refs = curs[: args.max_frames], refs[: args.max_frames]

    # Centre weights: 1e9 is effectively identity (no filtering).
    variants = [("unfiltered", None), ("blur w=12", 12.0), ("blur w=8", 8.0), ("blur w=4", 4.0)]
    totals = {name: 0.0 for name, _ in variants}
    for cp, rp in zip(curs, refs):
        cur, ref = load_plane(cp), load_plane(rp)
        for name, wc in variants:
            r = ref if wc is None else blur3(ref, wc)
            totals[name] += best_match_satd(cur, r, args.block, args.range)

    print(f"=== reference filtering vs prediction quality ===")
    print(f"{len(curs)} P-frames, {args.block}x{args.block} blocks, +/-{args.range} integer search, SATD")
    base = totals["unfiltered"]
    for name, _ in variants:
        d = (totals[name] / base - 1.0) * 100.0
        print(f"   {name:14s} SATD {totals[name]:15.0f}   {d:+6.2f}%")
    print()
    print("   Negative = filtering the reference predicts better. A real in-loop filter is")
    print("   edge-adaptive, so a flat 3x3 low-pass is a lower bound on what one could do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
